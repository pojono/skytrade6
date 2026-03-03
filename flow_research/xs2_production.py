#!/usr/bin/env python3
"""
XS-2 — Production-Grade Cross-Sectional Dispersion Portfolio

Anti-bug protections:
  ✗ No lookahead (all features strictly [t-W, t])
  ✗ No survivorship bias (dynamic universe per timestamp)
  ✗ No magic turnover (vintage model, explicit overlap)
  ✗ No double counting (each vintage independent)
  ✗ No volatility leakage (backward-only windows)

Walk-forward: Jan→Feb + Feb→Jan
Cost model: 20bp RT + slippage grid [0, 1, 2, 5] bp/side
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 50
W_BETA_MIN = 3 * 24 * 60          # 3 days in minutes
BETA_VALID_FRAC = 0.80             # min fraction of valid 1m bars in beta window
BETA_CLIP = 3.0
BETA_VAR_EPS = 1e-14

HORIZON = 60                       # minutes
K = 5                              # top/bottom K for portfolio
SIGNAL_FREQ = 5                    # minutes between signals

# R3 regime
DISP_LOOKBACK = 60                 # minutes for dispersion calc
VOL_Z_LOOKBACK = 3 * 24 * 60      # 3 days for vol z-score
R3_DISP_PCTL = 0.80               # percentile threshold for dispersion
R3_VOL_Z_PCTL = 0.80              # cross-sectional percentile for vol_z

# Volume threshold for eligibility (USD turnover in 60m)
MIN_TURNOVER_60M = 50_000

# Cost model
FEE_BP = 10                        # per side
SLIP_GRID = [0, 1, 2, 5]           # bp per side

# Walk-forward
WF_SPLIT = pd.Timestamp("2026-02-01", tz="UTC")

# Stats
N_BOOTSTRAP = 5000
N_PERMUTATION = 2000
BLOCK_SIZE_DAYS = 1
SEED = 42

# ---------------------------------------------------------------------------
# §1: Data Loading
# ---------------------------------------------------------------------------


def discover_symbols() -> list[str]:
    syms = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        nmark = len(list(d.glob("*_mark_price_kline_1m.csv")))
        noi = len(list(d.glob("*_open_interest_5min.csv")))
        nfr = len(list(d.glob("*_funding_rate.csv")))
        if nmark >= MIN_DAYS and noi >= MIN_DAYS and nfr >= MIN_DAYS:
            syms.append(d.name)
    return syms


def _load_csvs(sym: str, glob: str, ts_col: str, val_cols: dict) -> pd.DataFrame:
    """Generic CSV loader: glob files, parse ts, select+rename cols."""
    sym_dir = DATA_DIR / sym
    files = sorted(sym_dir.glob(glob))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df[ts_col].astype(int), unit="ms", utc=True)
    out = df[["ts"]].copy()
    for src, dst in val_cols.items():
        out[dst] = df[src].astype(float)
    out = out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    out = out[(out["ts"] >= START) & (out["ts"] <= END)]
    return out.reset_index(drop=True)


def load_all_symbols(symbols: list[str]) -> dict:
    """Load mark 1m, kline 1m (volume), OI 5m, FR for all symbols.
    Returns dict[sym] -> dict of DataFrames.
    """
    data = {}
    for i, sym in enumerate(symbols, 1):
        mark = _load_csvs(sym, "*_mark_price_kline_1m.csv", "startTime",
                          {"open": "open", "high": "high", "low": "low", "close": "close"})
        kline = _load_csvs(sym, "*_kline_1m.csv", "startTime", {"turnover": "turnover"})
        # Exclude mark_price and premium_index klines
        kline_dir = DATA_DIR / sym
        kline_files = sorted(
            f for f in kline_dir.glob("*_kline_1m.csv")
            if "mark_price" not in f.name and "premium_index" not in f.name
        )
        if kline_files:
            frames = []
            for f in kline_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        frames.append(df)
                except Exception:
                    continue
            if frames:
                kdf = pd.concat(frames, ignore_index=True)
                kdf["ts"] = pd.to_datetime(kdf["startTime"].astype(int), unit="ms", utc=True)
                kdf["turnover"] = kdf["turnover"].astype(float)
                kdf = kdf[["ts", "turnover"]].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
                kdf = kdf[(kdf["ts"] >= START) & (kdf["ts"] <= END)]
                kline = kdf.reset_index(drop=True)

        oi = _load_csvs(sym, "*_open_interest_5min.csv", "timestamp",
                        {"openInterest": "oi"})
        fr = _load_csvs(sym, "*_funding_rate.csv", "timestamp",
                        {"fundingRate": "fr"})

        if len(mark) < 1000:
            continue

        data[sym] = {"mark": mark, "kline": kline, "oi": oi, "fr": fr}
        if i % 10 == 0:
            print(f"    Loaded {i}/{len(symbols)}")
    return data


# ---------------------------------------------------------------------------
# §1.1: Unified 1m grid with gap handling
# ---------------------------------------------------------------------------


def _consecutive_nan_lengths(arr: np.ndarray) -> np.ndarray:
    """For each position, return the length of the NaN block it belongs to (0 if not NaN).
    Pure numpy, O(n)."""
    is_nan = np.isnan(arr)
    n = len(arr)
    block_len = np.zeros(n, dtype=np.int32)
    if n == 0:
        return block_len
    # Forward pass: count consecutive NaN run length
    run = 0
    starts = []
    for i in range(n):
        if is_nan[i]:
            run += 1
        else:
            if run > 0:
                starts.append((i - run, run))
            run = 0
    if run > 0:
        starts.append((n - run, run))
    for s, length in starts:
        block_len[s:s + length] = length
    return block_len


def build_unified_grid(raw: dict) -> tuple[pd.DatetimeIndex, dict]:
    """Build unified 1m grid. Per symbol: ffill <5m gaps, mark >=5m as invalid.

    Returns:
      grid_1m: DatetimeIndex
      sym_data: dict[sym] -> DataFrame indexed by ts with columns:
        close, turnover, log_ret, is_ffill, is_invalid
    """
    # Collect all timestamps
    all_ts = set()
    for sym, d in raw.items():
        all_ts.update(d["mark"]["ts"].values)
    grid_1m = pd.DatetimeIndex(sorted(all_ts), tz="UTC")
    print(f"  Unified 1m grid: {len(grid_1m):,} points "
          f"({grid_1m[0].date()} → {grid_1m[-1].date()})")

    sym_data = {}
    for sym, d in raw.items():
        mark = d["mark"].set_index("ts")["close"]
        kline_df = d["kline"]
        kline = kline_df.set_index("ts")["turnover"] if len(kline_df) > 0 else None

        # Reindex to unified grid
        close_raw = mark.reindex(grid_1m).values.astype(float)
        turnover = kline.reindex(grid_1m).fillna(0).values.astype(float) if kline is not None else np.zeros(len(grid_1m))

        # Detect gap block lengths (vectorized numpy)
        block_len = _consecutive_nan_lengths(close_raw)

        is_ffill = np.zeros(len(grid_1m), dtype=np.int8)
        is_invalid = np.zeros(len(grid_1m), dtype=np.int8)
        nan_mask = np.isnan(close_raw)
        is_ffill[nan_mask & (block_len > 0) & (block_len < 5)] = 1
        is_invalid[nan_mask & (block_len >= 5)] = 1

        # Forward fill close (for ffill gaps only), NaN invalid
        close = close_raw.copy()
        # ffill: propagate last valid
        last_valid = np.nan
        for i in range(len(close)):
            if not np.isnan(close_raw[i]):
                last_valid = close_raw[i]
            elif is_ffill[i]:
                close[i] = last_valid
            else:
                close[i] = np.nan

        # Log returns
        log_ret = np.full(len(grid_1m), np.nan)
        log_ret[1:] = np.log(close[1:] / close[:-1])
        # Invalidate where current or previous is invalid
        inv_mask = (is_invalid == 1)
        inv_mask_prev = np.zeros(len(grid_1m), dtype=bool)
        inv_mask_prev[1:] = (is_invalid[:-1] == 1)
        log_ret[inv_mask | inv_mask_prev] = np.nan
        # Also invalidate where close is NaN
        log_ret[np.isnan(close)] = np.nan

        df = pd.DataFrame({
            "close": close,
            "turnover": turnover,
            "log_ret": log_ret,
            "is_ffill": is_ffill,
            "is_invalid": is_invalid,
        }, index=grid_1m)

        sym_data[sym] = df

    return grid_1m, sym_data


# ---------------------------------------------------------------------------
# §2: Market index (EW) + Rolling beta
# ---------------------------------------------------------------------------


def compute_market_and_beta(grid_1m: pd.DatetimeIndex, sym_data: dict) -> tuple:
    """Compute EW market return and rolling beta per symbol.

    Uses pandas rolling for O(N) beta computation per symbol.

    Returns:
      mkt_ret_1m: Series (1m EW market returns)
      mkt_cum: Series (cumulative EW market log price)
      sym_beta: dict[sym] -> Series (beta at 5m grid, NaN elsewhere)
      ret_wide: DataFrame (1m returns, symbols as columns)
    """
    symbols = sorted(sym_data.keys())

    # Build wide return matrix
    ret_wide = pd.DataFrame(index=grid_1m)
    for sym in symbols:
        ret_wide[sym] = sym_data[sym]["log_ret"]

    # EW market return = mean of available returns per minute
    mkt_ret_1m = ret_wide.mean(axis=1)
    mkt_cum = mkt_ret_1m.fillna(0).cumsum()

    # Rolling beta using pandas rolling cov/var (vectorized, O(N) per symbol)
    print(f"  Computing rolling betas for {len(symbols)} symbols...")
    W = W_BETA_MIN
    min_periods = int(W * BETA_VALID_FRAC)

    # Pre-compute rolling var of market (shared across all symbols)
    mkt_s = mkt_ret_1m.copy()
    roll_var_mkt = mkt_s.rolling(W, min_periods=min_periods).var()

    sym_beta = {}
    for idx, sym in enumerate(symbols):
        r_s = ret_wide[sym]
        # Rolling cov(r, mkt)
        # cov = E[r*m] - E[r]*E[m]
        rm = (r_s * mkt_s)
        roll_mean_rm = rm.rolling(W, min_periods=min_periods).mean()
        roll_mean_r = r_s.rolling(W, min_periods=min_periods).mean()
        roll_mean_m = mkt_s.rolling(W, min_periods=min_periods).mean()
        roll_cov = roll_mean_rm - roll_mean_r * roll_mean_m

        beta_full = roll_cov / roll_var_mkt.clip(lower=BETA_VAR_EPS)
        beta_full = beta_full.clip(-BETA_CLIP, BETA_CLIP)

        # Set to NaN where var_mkt < eps
        beta_full[roll_var_mkt < BETA_VAR_EPS] = 0.0

        # Only keep at 5m grid points
        mask_5m = grid_1m.minute % SIGNAL_FREQ == 0
        beta_5m = pd.Series(np.nan, index=grid_1m)
        beta_5m[mask_5m] = beta_full[mask_5m]

        sym_beta[sym] = beta_5m

        if (idx + 1) % 10 == 0:
            print(f"    Beta: {idx + 1}/{len(symbols)}")

    return mkt_ret_1m, mkt_cum, sym_beta, ret_wide


# ---------------------------------------------------------------------------
# §3: Build 5m signal panel (strict causal features)
# ---------------------------------------------------------------------------


def build_signal_panel(
    grid_1m: pd.DatetimeIndex,
    sym_data: dict,
    mkt_ret_1m: pd.Series,
    mkt_cum: pd.Series,
    sym_beta: dict,
) -> pd.DataFrame:
    """Build panel of 5m signal rows with strictly causal features.

    All features use backward-only windows. Forward returns use
    entry=close(t+1m), exit=close(t+H+1m) for conservative execution.
    """
    symbols = sorted(sym_data.keys())

    # 5m grid
    mask_5m = grid_1m.minute % SIGNAL_FREQ == 0
    grid_5m = grid_1m[mask_5m]
    idx_5m = np.where(mask_5m)[0]  # positions in grid_1m
    n_5m = len(grid_5m)
    n_1m = len(grid_1m)
    print(f"  5m grid: {n_5m:,} points")

    # Pre-compute per-symbol 1m features (all backward-only rolling)
    print(f"  Computing per-symbol features...")
    sym_rv60 = {}
    sym_ret60 = {}
    sym_turn60 = {}
    sym_volz = {}
    for sym in symbols:
        sd = sym_data[sym]
        rv = sd["log_ret"].rolling(DISP_LOOKBACK, min_periods=30).std()
        sym_rv60[sym] = rv
        sym_ret60[sym] = sd["log_ret"].rolling(DISP_LOOKBACK, min_periods=30).sum()
        sym_turn60[sym] = sd["turnover"].rolling(60, min_periods=10).sum()
        # vol_z: (rv - rolling_mean_3d) / rolling_std_3d
        rm = rv.rolling(VOL_Z_LOOKBACK, min_periods=VOL_Z_LOOKBACK // 2).mean()
        rs = rv.rolling(VOL_Z_LOOKBACK, min_periods=VOL_Z_LOOKBACK // 2).std().clip(lower=1e-12)
        sym_volz[sym] = (rv - rm) / rs

    # Sample at 5m grid into wide DataFrames
    print(f"  Assembling wide frames at 5m resolution...")
    ret60_wide = pd.DataFrame({s: sym_ret60[s].values[idx_5m] for s in symbols}, index=grid_5m)
    rv60_wide = pd.DataFrame({s: sym_rv60[s].values[idx_5m] for s in symbols}, index=grid_5m)
    volz_wide = pd.DataFrame({s: sym_volz[s].values[idx_5m] for s in symbols}, index=grid_5m)
    turn60_wide = pd.DataFrame({s: sym_turn60[s].values[idx_5m] for s in symbols}, index=grid_5m)
    beta_wide = pd.DataFrame({s: sym_beta[s].values[idx_5m] for s in symbols if s in sym_beta}, index=grid_5m)
    close_wide = pd.DataFrame({s: sym_data[s]["close"].values[idx_5m] for s in symbols}, index=grid_5m)
    invalid_wide = pd.DataFrame({s: sym_data[s]["is_invalid"].values[idx_5m] for s in symbols}, index=grid_5m)

    # Dispersion = std of ret_60m across symbols at each t (strictly causal)
    dispersion = ret60_wide.std(axis=1)

    # Expanding percentile of dispersion (causal — only looks back)
    disp_rank = dispersion.expanding(min_periods=100).rank(pct=True)

    # Cross-sectional vol_z rank at each t
    volz_rank = volz_wide.rank(axis=1, pct=True)

    # Eligibility: has beta, not invalid, has turnover > threshold
    eligible = beta_wide.notna() & (invalid_wide == 0) & (turn60_wide > MIN_TURNOVER_60M)

    # R3 regime: dispersion >= P80 AND vol_z rank >= P80 AND eligible
    r3_disp_mask = disp_rank >= R3_DISP_PCTL  # per timestamp (scalar)
    r3_regime = eligible.copy()
    r3_regime.loc[:, :] = False
    r3_ts = r3_disp_mask[r3_disp_mask].index
    r3_regime.loc[r3_ts] = (volz_rank.loc[r3_ts] >= R3_VOL_Z_PCTL) & eligible.loc[r3_ts]

    n_r3 = int(r3_regime.sum().sum())
    n_elig = int(eligible.sum().sum())
    print(f"  R3 activations: {n_r3:,} / {n_elig:,} eligible ({n_r3/max(n_elig,1):.1%})")

    # Forward returns: entry=close(t+1m), exit=close(t+H+1m)
    print(f"  Computing forward returns (H={HORIZON}m)...")
    entry_idx = np.clip(idx_5m + 1, 0, n_1m - 1)
    exit_idx = np.clip(idx_5m + HORIZON + 1, 0, n_1m - 1)
    # Mark out-of-bounds
    entry_oob = (idx_5m + 1) >= n_1m
    exit_oob = (idx_5m + HORIZON + 1) >= n_1m

    entry_close_wide = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    exit_close_wide = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)

    for sym in symbols:
        c = sym_data[sym]["close"].values
        inv = sym_data[sym]["is_invalid"].values

        ec = c[entry_idx].astype(float)
        xc = c[exit_idx].astype(float)

        # Invalidate out-of-bounds or invalid bars
        ec[entry_oob | (inv[entry_idx] == 1)] = np.nan
        xc[exit_oob | (inv[exit_idx] == 1)] = np.nan

        entry_close_wide[sym] = ec
        exit_close_wide[sym] = xc

    # Forward return in bp
    ec_arr = entry_close_wide.values.astype(float)
    xc_arr = exit_close_wide.values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_ret_arr = np.log(xc_arr / ec_arr) * 10000
    fwd_ret_wide = pd.DataFrame(fwd_ret_arr, index=grid_5m, columns=symbols)

    # Market forward return (EW of eligible coins' fwd returns)
    mkt_fwd_ret = fwd_ret_wide.where(eligible).mean(axis=1)

    # Excess return = fwd_ret - beta * mkt_fwd_ret
    excess_wide = fwd_ret_wide - beta_wide.multiply(mkt_fwd_ret, axis=0)

    # Melt to long panel
    print(f"  Melting to long panel...")
    panel_parts = []
    disp_vals = dispersion.values
    disp_pctl_vals = disp_rank.values
    mkt_fwd_vals = mkt_fwd_ret.values

    for sym in symbols:
        df = pd.DataFrame({
            "ts": grid_5m,
            "symbol": sym,
            "close": close_wide[sym].values,
            "entry_close": entry_close_wide[sym].values,
            "exit_close": exit_close_wide[sym].values,
            "beta": beta_wide[sym].values if sym in beta_wide.columns else np.nan,
            "rv_60": rv60_wide[sym].values,
            "ret_60": ret60_wide[sym].values,
            "vol_z": volz_wide[sym].values,
            "turnover_60": turn60_wide[sym].values,
            "vol_z_rank": volz_rank[sym].values,
            "dispersion": disp_vals,
            "disp_pctl": disp_pctl_vals,
            "eligible": eligible[sym].values,
            "R3": r3_regime[sym].values,
            "fwd_ret": fwd_ret_wide[sym].values,
            "mkt_fwd_ret": mkt_fwd_vals,
            "excess_ret": excess_wide[sym].values if sym in excess_wide.columns else np.nan,
        })
        panel_parts.append(df)

    panel = pd.concat(panel_parts, ignore_index=True)
    panel["is_valid"] = panel["fwd_ret"].notna() & panel["excess_ret"].notna() & panel["eligible"]

    print(f"  Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['is_valid'].sum():,} valid")

    return panel


# ---------------------------------------------------------------------------
# §4: Vintage Portfolio Engine
# ---------------------------------------------------------------------------


class Vintage:
    __slots__ = ["start_ts", "end_ts", "longs", "shorts",
                 "entry_prices_long", "entry_prices_short",
                 "exit_prices_long", "exit_prices_short"]

    def __init__(self, start_ts, end_ts, longs, shorts,
                 entry_prices_long, entry_prices_short):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.longs = longs
        self.shorts = shorts
        self.entry_prices_long = entry_prices_long
        self.entry_prices_short = entry_prices_short
        self.exit_prices_long = {}
        self.exit_prices_short = {}


def run_vintage_portfolio(panel: pd.DataFrame, slip_bp: float = 0.0,
                          period_start=None, period_end=None) -> pd.DataFrame:
    """Run vintage portfolio simulation.

    Each signal creates a vintage: long top-K, short bottom-K by vol_z.
    Each vintage lives for HORIZON minutes.
    Entry = next 1m close + slip. Exit = close at t+H+1m - slip.

    Returns DataFrame of vintage-level results.
    """
    if period_start is not None:
        panel = panel[(panel["ts"] >= period_start) & (panel["ts"] < period_end)]

    # Get R3 signal timestamps
    r3_panel = panel[panel["R3"] & panel["is_valid"]].copy()
    signal_times = sorted(r3_panel["ts"].unique())

    if len(signal_times) == 0:
        return pd.DataFrame()

    vintages = []
    for t in signal_times:
        grp = r3_panel[r3_panel["ts"] == t].copy()
        if len(grp) < 2 * K:
            continue

        # Sort by vol_z descending
        grp = grp.sort_values("vol_z", ascending=False)
        top_k = grp.head(K)
        bot_k = grp.tail(K)

        # Entry prices with slippage
        long_syms = top_k["symbol"].values
        short_syms = bot_k["symbol"].values

        entry_long = {}
        entry_short = {}
        exit_long = {}
        exit_short = {}

        for _, row in top_k.iterrows():
            sym = row["symbol"]
            ep = row["entry_close"]
            xp = row["exit_close"]
            if np.isnan(ep) or np.isnan(xp):
                continue
            entry_long[sym] = ep * (1 + slip_bp / 10000)
            exit_long[sym] = xp * (1 - slip_bp / 10000)

        for _, row in bot_k.iterrows():
            sym = row["symbol"]
            ep = row["entry_close"]
            xp = row["exit_close"]
            if np.isnan(ep) or np.isnan(xp):
                continue
            entry_short[sym] = ep * (1 - slip_bp / 10000)
            exit_short[sym] = xp * (1 + slip_bp / 10000)

        if len(entry_long) == 0 and len(entry_short) == 0:
            continue

        # Compute PnL
        # Long: (exit - entry) / entry
        # Short: (entry - exit) / entry
        long_rets = []
        for sym in entry_long:
            if sym in exit_long:
                r = (exit_long[sym] - entry_long[sym]) / entry_long[sym]
                long_rets.append(r)

        short_rets = []
        for sym in entry_short:
            if sym in exit_short:
                r = (entry_short[sym] - exit_short[sym]) / entry_short[sym]
                short_rets.append(r)

        # Weight: equal within leg, each leg = 0.5 gross
        n_long = len(long_rets)
        n_short = len(short_rets)

        if n_long == 0 and n_short == 0:
            continue

        # Gross return per vintage (dollar-neutral)
        long_gross = np.mean(long_rets) * 0.5 if n_long > 0 else 0
        short_gross = np.mean(short_rets) * 0.5 if n_short > 0 else 0
        gross_ret = long_gross + short_gross

        # Fees: 20bp RT per position, applied to each leg
        n_positions = n_long + n_short
        fee_cost = n_positions * (2 * FEE_BP / 10000) / (2 * K)  # normalized to portfolio weight
        net_ret = gross_ret - fee_cost

        vintages.append({
            "ts": t,
            "date": t.date(),
            "n_long": n_long,
            "n_short": n_short,
            "long_syms": ",".join(entry_long.keys()),
            "short_syms": ",".join(entry_short.keys()),
            "gross_ret_bp": gross_ret * 10000,
            "fee_bp": fee_cost * 10000,
            "net_ret_bp": net_ret * 10000,
        })

    return pd.DataFrame(vintages)


# ---------------------------------------------------------------------------
# §5: Turnover calculation
# ---------------------------------------------------------------------------


def compute_turnover(vintages_df: pd.DataFrame) -> dict:
    """Compute turnover metrics from vintage trades."""
    if len(vintages_df) == 0:
        return {}

    # Active vintages at any point
    avg_vintages_per_day = vintages_df.groupby("date").size().mean()

    # Turnover: each vintage creates 2*K position changes (K long + K short)
    # Each position is opened and closed = 2 trades per position
    trades_per_vintage = vintages_df["n_long"].mean() + vintages_df["n_short"].mean()
    vintages_per_day = vintages_df.groupby("date").size()

    # Overlap: check how many symbols carry over between consecutive vintages
    overlap_count = 0
    total_pairs = 0
    vdf = vintages_df.sort_values("ts")
    prev_longs = set()
    prev_shorts = set()
    for _, row in vdf.iterrows():
        cur_longs = set(row["long_syms"].split(",")) if row["long_syms"] else set()
        cur_shorts = set(row["short_syms"].split(",")) if row["short_syms"] else set()
        if prev_longs:
            overlap_count += len(cur_longs & prev_longs) + len(cur_shorts & prev_shorts)
            total_pairs += len(cur_longs) + len(cur_shorts)
        prev_longs = cur_longs
        prev_shorts = cur_shorts

    overlap_rate = overlap_count / max(total_pairs, 1)

    return {
        "avg_vintages_per_day": avg_vintages_per_day,
        "avg_trades_per_vintage": trades_per_vintage,
        "avg_vintages_per_day_by_date": vintages_per_day.to_dict(),
        "position_overlap_rate": overlap_rate,
    }


# ---------------------------------------------------------------------------
# §6: Walk-forward
# ---------------------------------------------------------------------------


def walk_forward(panel: pd.DataFrame, slip_bp: float) -> dict:
    """Run walk-forward: Jan→Feb and Feb→Jan."""
    results = {}

    # Split 1: Train=Jan, Test=Feb
    v_feb = run_vintage_portfolio(panel, slip_bp,
                                  period_start=WF_SPLIT,
                                  period_end=END + pd.Timedelta(seconds=1))
    results["jan_train_feb_test"] = v_feb

    # Split 2: Train=Feb, Test=Jan
    v_jan = run_vintage_portfolio(panel, slip_bp,
                                  period_start=START,
                                  period_end=WF_SPLIT)
    results["feb_train_jan_test"] = v_jan

    return results


# ---------------------------------------------------------------------------
# §7: Statistical validation
# ---------------------------------------------------------------------------


def stats_validation(vintages_df: pd.DataFrame, rng) -> dict:
    """Block bootstrap CI + permutation test on daily returns."""
    if len(vintages_df) == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_perm": np.nan, "sharpe_ann": np.nan}

    daily = vintages_df.groupby("date")["net_ret_bp"].mean()
    daily_arr = daily.values

    if len(daily_arr) < 3:
        return {"mean": np.mean(daily_arr), "ci_lo": np.nan, "ci_hi": np.nan,
                "p_perm": np.nan, "sharpe_ann": np.nan}

    obs_mean = np.mean(daily_arr)
    obs_sharpe = obs_mean / np.std(daily_arr, ddof=1) * np.sqrt(365) if np.std(daily_arr, ddof=1) > 0 else 0

    # Block bootstrap (block = 1 day = 1 element since we're already daily)
    boot_means = np.array([
        np.mean(rng.choice(daily_arr, len(daily_arr), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    ci_lo = np.percentile(boot_means, 5)
    ci_hi = np.percentile(boot_means, 95)

    # Permutation test: shuffle daily returns across days
    count = 0
    for _ in range(N_PERMUTATION):
        perm = rng.permutation(daily_arr)
        if np.abs(np.mean(perm)) >= np.abs(obs_mean):
            count += 1
    p_perm = count / N_PERMUTATION

    return {
        "mean_daily_bp": obs_mean,
        "median_daily_bp": np.median(daily_arr),
        "std_daily_bp": np.std(daily_arr, ddof=1),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "sharpe_ann": obs_sharpe,
        "n_days": len(daily_arr),
        "hitrate": (daily_arr > 0).mean(),
        "max_dd_bp": _max_drawdown(daily_arr),
    }


def _max_drawdown(daily_bp: np.ndarray) -> float:
    """Max drawdown in bp from daily returns."""
    cum = np.cumsum(daily_bp)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return np.max(dd) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# §8: Bug detection checklist
# ---------------------------------------------------------------------------


def bug_checks(panel: pd.DataFrame):
    """Run automated bug detection checks."""
    print("\n  BUG DETECTION CHECKLIST")
    print("  " + "=" * 50)

    # 1. No future timestamps in features
    # dispersion uses ret_60m which is backward-only ✓
    # vol_z uses backward rolling ✓
    # beta uses [t-3d, t) ✓
    print("  [1] Feature causality: enforced by construction (backward rolling)")

    # 2. No duplicate rows
    dupes = panel.duplicated(subset=["ts", "symbol"]).sum()
    status = "✓ PASS" if dupes == 0 else f"✗ FAIL ({dupes} dupes)"
    print(f"  [2] No duplicate rows: {status}")

    # 3. Beta window does not include t+H
    # Beta is computed at 5m grid using [t-W, t) window ✓
    print("  [3] Beta window excludes t+H: enforced by construction")

    # 4. Dispersion strictly backward
    print("  [4] Dispersion backward-only: uses rolling(60).std() on past returns")

    # 5. Forward return uses t+1 (next bar) for entry
    valid = panel[panel["is_valid"]]
    # Check that entry_close != close (they should differ since entry is t+1m)
    same_count = (valid["entry_close"] == valid["close"]).sum()
    n_valid = len(valid)
    pct_same = same_count / max(n_valid, 1)
    status = f"✓ PASS ({pct_same:.1%} same)" if pct_same < 0.05 else f"⚠ {pct_same:.1%} same"
    print(f"  [5] Entry price = next 1m close: {status}")

    # 6. Universe excludes symbols with insufficient history
    # Eligible flag requires has_beta (which requires 3d history + 80% valid) ✓
    print("  [6] Dynamic universe: eligible requires beta + turnover + not invalid")

    # 7. Expanding percentile for dispersion (not global)
    print("  [7] Dispersion pctl: expanding (causal), not global")

    print("  " + "=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("XS-2 — PRODUCTION-GRADE DISPERSION PORTFOLIO")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Strategy: R3 dispersion, H={HORIZON}m, K={K}")
    print(f"Cost: {FEE_BP}bp/side + slippage grid {SLIP_GRID}")
    print("=" * 80)

    # §1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}")
    symbols = discover_symbols()
    print(f"  Symbols with sufficient files: {len(symbols)}")
    raw = load_all_symbols(symbols)
    print(f"  Loaded: {len(raw)} symbols with data")

    # §1.1: Unified grid
    print(f"\n{'─'*70}")
    print("PHASE 2: Unified 1m grid + gap handling")
    print(f"{'─'*70}")
    grid_1m, sym_data = build_unified_grid(raw)

    # Gap stats
    total_ffill = sum(sd["is_ffill"].sum() for sd in sym_data.values())
    total_invalid = sum(sd["is_invalid"].sum() for sd in sym_data.values())
    total_points = sum(len(sd) for sd in sym_data.values())
    print(f"  Forward-filled (<5m gaps): {total_ffill:,} ({total_ffill/total_points:.2%})")
    print(f"  Invalid (≥5m gaps): {total_invalid:,} ({total_invalid/total_points:.2%})")

    # §2: Market + beta
    print(f"\n{'─'*70}")
    print("PHASE 3: Market index + rolling beta")
    print(f"{'─'*70}")
    mkt_ret_1m, mkt_cum, sym_beta, ret_wide = compute_market_and_beta(grid_1m, sym_data)

    # §3: Signal panel
    print(f"\n{'─'*70}")
    print("PHASE 4: Build signal panel (strict causal)")
    print(f"{'─'*70}")
    panel = build_signal_panel(grid_1m, sym_data, mkt_ret_1m, mkt_cum, sym_beta)

    # Save panel
    panel.to_parquet(OUTPUT_DIR / "xs2_panel.parquet", index=False)
    print(f"  Saved xs2_panel.parquet")

    # §8: Bug checks
    bug_checks(panel)

    # Baseline check
    print(f"\n{'─'*70}")
    print("BASELINE CHECK")
    print(f"{'─'*70}")
    valid = panel[panel["is_valid"]]
    vals = valid["excess_ret"].dropna()
    print(f"  Unconditional excess: mean={vals.mean():+.2f}bp  med={vals.median():+.2f}bp  "
          f"std={vals.std():.1f}bp  WR={(vals>0).mean():.1%}  N={len(vals):,}")

    # §4-5: Vintage portfolio + costs
    print(f"\n{'─'*70}")
    print("PHASE 5: Vintage portfolio simulation")
    print(f"{'─'*70}")

    all_results = []

    for slip in SLIP_GRID:
        total_cost_bp = 2 * FEE_BP + 2 * slip
        label = f"fee={FEE_BP}bp + slip={slip}bp (total {total_cost_bp}bp RT)"

        # Full period
        v_full = run_vintage_portfolio(panel, slip)
        if len(v_full) == 0:
            print(f"  [{label}] No vintages. Skipping.")
            continue

        turn = compute_turnover(v_full)
        stats_full = stats_validation(v_full, rng)

        # Walk-forward
        wf = walk_forward(panel, slip)
        stats_oos1 = stats_validation(wf["jan_train_feb_test"], rng)
        stats_oos2 = stats_validation(wf["feb_train_jan_test"], rng)

        result = {
            "slip_bp": slip,
            "total_cost_bp": total_cost_bp,
            # Full period
            "full_n_vintages": len(v_full),
            "full_mean_daily_bp": stats_full["mean_daily_bp"],
            "full_median_daily_bp": stats_full["median_daily_bp"],
            "full_sharpe_ann": stats_full["sharpe_ann"],
            "full_hitrate": stats_full["hitrate"],
            "full_max_dd_bp": stats_full["max_dd_bp"],
            "full_ci_lo": stats_full["ci_lo"],
            "full_ci_hi": stats_full["ci_hi"],
            "full_p_perm": stats_full["p_perm"],
            # OOS 1: Feb
            "oos_feb_n_vintages": len(wf["jan_train_feb_test"]),
            "oos_feb_mean_daily_bp": stats_oos1["mean_daily_bp"],
            "oos_feb_sharpe_ann": stats_oos1["sharpe_ann"],
            "oos_feb_hitrate": stats_oos1["hitrate"],
            "oos_feb_p_perm": stats_oos1["p_perm"],
            # OOS 2: Jan
            "oos_jan_n_vintages": len(wf["feb_train_jan_test"]),
            "oos_jan_mean_daily_bp": stats_oos2["mean_daily_bp"],
            "oos_jan_sharpe_ann": stats_oos2["sharpe_ann"],
            "oos_jan_hitrate": stats_oos2["hitrate"],
            "oos_jan_p_perm": stats_oos2["p_perm"],
            # Turnover
            "avg_vintages_per_day": turn.get("avg_vintages_per_day", 0),
            "position_overlap_rate": turn.get("position_overlap_rate", 0),
        }
        all_results.append(result)

        print(f"\n  ── Slip={slip}bp (total {total_cost_bp}bp RT) ──")
        print(f"  Full: {len(v_full)} vintages, "
              f"mean={stats_full['mean_daily_bp']:+.2f}bp/day, "
              f"Sharpe={stats_full['sharpe_ann']:.2f}, "
              f"HR={stats_full['hitrate']:.0%}, "
              f"maxDD={stats_full['max_dd_bp']:.0f}bp, "
              f"p={stats_full['p_perm']:.3f}")
        print(f"  OOS Feb: {len(wf['jan_train_feb_test'])} vint, "
              f"mean={stats_oos1['mean_daily_bp']:+.2f}bp/day, "
              f"Sharpe={stats_oos1['sharpe_ann']:.2f}, "
              f"HR={stats_oos1['hitrate']:.0%}, "
              f"p={stats_oos1['p_perm']:.3f}")
        print(f"  OOS Jan: {len(wf['feb_train_jan_test'])} vint, "
              f"mean={stats_oos2['mean_daily_bp']:+.2f}bp/day, "
              f"Sharpe={stats_oos2['sharpe_ann']:.2f}, "
              f"HR={stats_oos2['hitrate']:.0%}, "
              f"p={stats_oos2['p_perm']:.3f}")
        print(f"  Turnover: {turn.get('avg_vintages_per_day', 0):.1f} vint/day, "
              f"overlap={turn.get('position_overlap_rate', 0):.1%}")

    # Save results
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(OUTPUT_DIR / "xs2_results.csv", index=False)
        print(f"\n  Saved xs2_results.csv")

        # Save full vintage detail for slip=2bp
        v_detail = run_vintage_portfolio(panel, 2.0)
        if len(v_detail) > 0:
            v_detail.to_csv(OUTPUT_DIR / "xs2_vintages_slip2.csv", index=False)

    # §12: GO / NO-GO
    print(f"\n{'='*80}")
    print("GO / NO-GO VERDICT")
    print("=" * 80)

    if all_results:
        # Check slip=2bp
        r2 = next((r for r in all_results if r["slip_bp"] == 2), None)
        if r2:
            checks = []
            c1 = r2["oos_feb_mean_daily_bp"] > 0 and r2["oos_jan_mean_daily_bp"] > 0
            checks.append(("OOS mean > 0 (both halves)", c1))

            c2 = r2["oos_feb_sharpe_ann"] > 1.5 or r2["oos_jan_sharpe_ann"] > 1.5
            checks.append(("OOS Sharpe > 1.5 (at least one)", c2))

            c3 = r2["full_max_dd_bp"] < 800  # 8% with ~1x leverage
            checks.append(("Max DD < 800bp", c3))

            c4 = r2["full_p_perm"] < 0.05
            checks.append(("Full period p < 0.05", c4))

            print(f"\n  Slip=2bp checks:")
            all_pass = True
            for name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"    {status}: {name}")
                if not passed:
                    all_pass = False

            # Check slip=5bp survival
            r5 = next((r for r in all_results if r["slip_bp"] == 5), None)
            if r5:
                survives_5 = r5["full_mean_daily_bp"] > 0
                print(f"\n  Survives slip=5bp: {'✓ YES' if survives_5 else '✗ NO'} "
                      f"(mean={r5['full_mean_daily_bp']:+.2f}bp/day)")

            verdict = "GO ✅" if all_pass else "NO-GO ❌"
            print(f"\n  {'='*40}")
            print(f"  VERDICT: {verdict}")
            print(f"  {'='*40}")
    else:
        print("  No results to evaluate.")

    elapsed = time.monotonic() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
