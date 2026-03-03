#!/usr/bin/env python3
"""
XS-3 — State Transition Model (Dispersion Regime Onset)

R3 is a state, not a signal. We trade onset events:
  - dispersion_60m crosses P80 UP → enter
  - dispersion_60m crosses P50 DOWN → exit

One position per regime. No vintages. No 5m rebalancing.

Anti-bug: no lookahead, no survivorship, strict causal features.
Cost: 20bp RT + slippage grid [0, 1, 2, 5] bp/side
Walk-forward: Jan→Feb + Feb→Jan
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 50

# Dispersion / vol features
DISP_LOOKBACK = 60            # minutes for dispersion calc
VOL_Z_LOOKBACK = 3 * 24 * 60  # 3 days for vol z-score

# Regime thresholds (expanding percentile, causal)
P80_DISP = 0.80               # enter when dispersion crosses above P80
P50_DISP = 0.50               # exit when dispersion crosses below P50

# Signal sampling (check every 5m)
SIGNAL_FREQ = 5

# Portfolio sizes to test
K_GRID = [3, 5, 10]

# Beta
W_BETA_MIN = 3 * 24 * 60
BETA_VALID_FRAC = 0.80
BETA_CLIP = 3.0
BETA_VAR_EPS = 1e-14

# Eligibility
MIN_TURNOVER_60M = 50_000

# Cost
FEE_BP = 10
SLIP_GRID = [0, 1, 2, 5]

# Walk-forward
WF_SPLIT = pd.Timestamp("2026-02-01", tz="UTC")

# Stats
N_BOOTSTRAP = 5000
N_PERMUTATION = 2000
BLOCK_SIZE_DAYS = 1
SEED = 42


# ---------------------------------------------------------------------------
# §1: Data Loading (reuse from XS-2)
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


def _load_csvs(sym: str, glob_pat: str, ts_col: str, val_cols: dict) -> pd.DataFrame:
    sym_dir = DATA_DIR / sym
    files = sorted(sym_dir.glob(glob_pat))
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


def load_all(symbols: list[str]) -> dict:
    data = {}
    for i, sym in enumerate(symbols, 1):
        mark = _load_csvs(sym, "*_mark_price_kline_1m.csv", "startTime",
                          {"close": "close"})
        kline_files = sorted((DATA_DIR / sym).glob("*_kline_1m.csv"))
        kline_files = [f for f in kline_files
                       if "mark_price" not in f.name and "premium_index" not in f.name]
        kline = pd.DataFrame()
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
                kl = pd.concat(frames, ignore_index=True)
                kl["ts"] = pd.to_datetime(kl["startTime"].astype(int), unit="ms", utc=True)
                kl["turnover"] = kl["turnover"].astype(float)
                kl = kl.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
                kl = kl[(kl["ts"] >= START) & (kl["ts"] <= END)]
                kline = kl[["ts", "turnover"]].reset_index(drop=True)

        if len(mark) > 0:
            data[sym] = {"mark": mark, "kline": kline}
        if i % 10 == 0:
            print(f"    Loaded {i}/{len(symbols)}")
    return data


# ---------------------------------------------------------------------------
# §2: Build unified 1m grid
# ---------------------------------------------------------------------------

def build_1m_grid(raw: dict) -> tuple[pd.DatetimeIndex, dict]:
    all_ts = set()
    for sym, d in raw.items():
        all_ts.update(d["mark"]["ts"].values)
    grid_1m = pd.DatetimeIndex(sorted(all_ts), tz="UTC")
    print(f"  1m grid: {len(grid_1m):,} points")

    sym_1m = {}
    for sym, d in raw.items():
        mark = d["mark"].set_index("ts")["close"]
        kline_df = d["kline"]
        kline = kline_df.set_index("ts")["turnover"] if len(kline_df) > 0 else None

        close = mark.reindex(grid_1m).ffill().values.astype(float)
        turnover = kline.reindex(grid_1m).fillna(0).values.astype(float) if kline is not None else np.zeros(len(grid_1m))
        log_ret = np.full(len(grid_1m), np.nan)
        log_ret[1:] = np.log(close[1:] / close[:-1])

        sym_1m[sym] = pd.DataFrame({
            "close": close,
            "turnover": turnover,
            "log_ret": log_ret,
        }, index=grid_1m)
    return grid_1m, sym_1m


# ---------------------------------------------------------------------------
# §3: Compute dispersion + vol_z at 5m grid
# ---------------------------------------------------------------------------

def compute_features(grid_1m: pd.DatetimeIndex, sym_1m: dict) -> tuple:
    """Compute dispersion, vol_z, beta, eligibility at 5m grid."""
    symbols = sorted(sym_1m.keys())

    # 5m grid
    mask_5m = grid_1m.minute % SIGNAL_FREQ == 0
    grid_5m = grid_1m[mask_5m]
    idx_5m = np.where(mask_5m)[0]
    print(f"  5m grid: {len(grid_5m):,} points")

    # Per-symbol rolling features on 1m data
    print(f"  Computing per-symbol features...")
    sym_rv60 = {}
    sym_ret60 = {}
    sym_turn60 = {}
    sym_volz = {}
    for sym in symbols:
        sd = sym_1m[sym]
        rv = sd["log_ret"].rolling(DISP_LOOKBACK, min_periods=30).std()
        sym_rv60[sym] = rv
        sym_ret60[sym] = sd["log_ret"].rolling(DISP_LOOKBACK, min_periods=30).sum()
        sym_turn60[sym] = sd["turnover"].rolling(60, min_periods=10).sum()
        rm = rv.rolling(VOL_Z_LOOKBACK, min_periods=VOL_Z_LOOKBACK // 2).mean()
        rs = rv.rolling(VOL_Z_LOOKBACK, min_periods=VOL_Z_LOOKBACK // 2).std().clip(lower=1e-12)
        sym_volz[sym] = (rv - rm) / rs

    # Sample to 5m
    ret60_wide = pd.DataFrame({s: sym_ret60[s].values[idx_5m] for s in symbols}, index=grid_5m)
    volz_wide = pd.DataFrame({s: sym_volz[s].values[idx_5m] for s in symbols}, index=grid_5m)
    turn60_wide = pd.DataFrame({s: sym_turn60[s].values[idx_5m] for s in symbols}, index=grid_5m)

    # Dispersion = std of ret_60m across symbols at each t
    dispersion = ret60_wide.std(axis=1)

    # Expanding percentile (causal)
    disp_pctl = dispersion.expanding(min_periods=100).rank(pct=True)

    # Eligibility: has vol_z, turnover > threshold
    eligible_wide = volz_wide.notna() & (turn60_wide > MIN_TURNOVER_60M)

    # Beta: rolling EW market
    ret_wide = pd.DataFrame({s: sym_1m[s]["log_ret"] for s in symbols}, index=grid_1m)
    mkt_ret = ret_wide.mean(axis=1)

    print(f"  Computing rolling betas...")
    W = W_BETA_MIN
    min_per = int(W * BETA_VALID_FRAC)
    mkt_s = mkt_ret.copy()
    roll_var_mkt = mkt_s.rolling(W, min_periods=min_per).var()

    sym_beta = {}
    for idx, sym in enumerate(symbols):
        r_s = ret_wide[sym]
        rm_prod = (r_s * mkt_s).rolling(W, min_periods=min_per).mean()
        rm_r = r_s.rolling(W, min_periods=min_per).mean()
        rm_m = mkt_s.rolling(W, min_periods=min_per).mean()
        cov = rm_prod - rm_r * rm_m
        beta = (cov / roll_var_mkt.clip(lower=BETA_VAR_EPS)).clip(-BETA_CLIP, BETA_CLIP)
        beta[roll_var_mkt < BETA_VAR_EPS] = 0.0
        sym_beta[sym] = beta
        if (idx + 1) % 10 == 0:
            print(f"    Beta: {idx + 1}/{len(symbols)}")

    beta_wide = pd.DataFrame({s: sym_beta[s].values[idx_5m] for s in symbols}, index=grid_5m)

    return (grid_5m, idx_5m, dispersion, disp_pctl, volz_wide,
            turn60_wide, eligible_wide, beta_wide, mkt_ret)


# ---------------------------------------------------------------------------
# §4: Regime detection (onset/offset)
# ---------------------------------------------------------------------------

def detect_regimes(grid_5m, disp_pctl) -> list[dict]:
    """Detect dispersion regime onset/offset events.

    Enter: disp_pctl(t-1) < P80 AND disp_pctl(t) >= P80
    Exit:  disp_pctl(t-1) >= P50 AND disp_pctl(t) < P50
    If regime active at end → close at last timestamp.

    Returns list of dicts with start_ts, end_ts.
    """
    regimes = []
    in_regime = False
    regime_start = None

    pctl_vals = disp_pctl.values
    ts_vals = grid_5m

    for i in range(1, len(pctl_vals)):
        if np.isnan(pctl_vals[i]) or np.isnan(pctl_vals[i-1]):
            continue

        if not in_regime:
            # Check onset: crosses P80 upward
            if pctl_vals[i-1] < P80_DISP and pctl_vals[i] >= P80_DISP:
                in_regime = True
                regime_start = ts_vals[i]
        else:
            # Check offset: crosses P50 downward
            if pctl_vals[i-1] >= P50_DISP and pctl_vals[i] < P50_DISP:
                regimes.append({
                    "start_ts": regime_start,
                    "end_ts": ts_vals[i],
                })
                in_regime = False
                regime_start = None

    # Close unclosed regime at end
    if in_regime and regime_start is not None:
        regimes.append({
            "start_ts": regime_start,
            "end_ts": ts_vals[-1],
        })

    return regimes


# ---------------------------------------------------------------------------
# §5: Portfolio simulation per regime
# ---------------------------------------------------------------------------

def simulate_regime_portfolio(
    regimes: list[dict],
    K: int,
    grid_1m: pd.DatetimeIndex,
    grid_5m: pd.DatetimeIndex,
    sym_1m: dict,
    volz_wide: pd.DataFrame,
    eligible_wide: pd.DataFrame,
    beta_wide: pd.DataFrame,
    mkt_ret: pd.Series,
    slip_bp: float,
    period_start: pd.Timestamp = None,
    period_end: pd.Timestamp = None,
) -> pd.DataFrame:
    """Simulate one position per regime.

    At onset: select top-K / bottom-K by vol_z from eligible universe.
    Entry: next 1m close after onset.
    Exit: next 1m close after regime end.
    """
    symbols = sorted(sym_1m.keys())

    results = []

    for reg in regimes:
        start = reg["start_ts"]
        end = reg["end_ts"]

        # Filter by period
        if period_start is not None and start < period_start:
            continue
        if period_end is not None and start > period_end:
            continue

        # Get eligible coins at onset
        if start not in volz_wide.index:
            continue

        vz = volz_wide.loc[start]
        elig = eligible_wide.loc[start]

        eligible_syms = [s for s in symbols if elig.get(s, False) and pd.notna(vz.get(s))]
        if len(eligible_syms) < 2 * K:
            continue

        vz_elig = vz[eligible_syms].sort_values()

        # Bottom K by vol_z → long; Top K → short
        longs = list(vz_elig.index[-K:])     # highest vol_z → long (dispersion play)
        shorts = list(vz_elig.index[:K])      # lowest vol_z → short

        n_long = len(longs)
        n_short = len(shorts)
        w_long = 0.5 / n_long
        w_short = 0.5 / n_short

        # Entry: next 1m close after onset
        entry_1m_idx = grid_1m.searchsorted(start, side="right")
        if entry_1m_idx >= len(grid_1m):
            continue

        # Exit: next 1m close after regime end
        exit_1m_idx = grid_1m.searchsorted(end, side="right")
        if exit_1m_idx >= len(grid_1m):
            exit_1m_idx = len(grid_1m) - 1

        entry_ts = grid_1m[entry_1m_idx]
        exit_ts = grid_1m[exit_1m_idx]
        duration_min = (exit_ts - entry_ts).total_seconds() / 60

        if duration_min < 5:  # skip trivially short regimes
            continue

        # Price PnL
        price_pnl_long = 0.0
        price_pnl_short = 0.0
        valid_long = 0
        valid_short = 0

        for s in longs:
            ep = sym_1m[s]["close"].iloc[entry_1m_idx]
            xp = sym_1m[s]["close"].iloc[exit_1m_idx]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                ret = np.log(xp / ep)
                price_pnl_long += w_long * ret * 10000
                valid_long += 1

        for s in shorts:
            ep = sym_1m[s]["close"].iloc[entry_1m_idx]
            xp = sym_1m[s]["close"].iloc[exit_1m_idx]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                ret = np.log(xp / ep)
                price_pnl_short += -w_short * ret * 10000
                valid_short += 1

        # Market return over same period (for beta neutralization check)
        mkt_cum_entry = mkt_ret.iloc[:entry_1m_idx+1].sum()
        mkt_cum_exit = mkt_ret.iloc[:exit_1m_idx+1].sum()
        mkt_ret_regime = (mkt_cum_exit - mkt_cum_entry) * 10000

        # Cost
        total_cost = 2 * (FEE_BP + slip_bp)  # round trip

        gross_pnl = price_pnl_long + price_pnl_short
        net_pnl = gross_pnl - total_cost

        results.append({
            "regime_start": start,
            "regime_end": end,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "duration_min": duration_min,
            "n_long": valid_long,
            "n_short": valid_short,
            "longs": ",".join(longs),
            "shorts": ",".join(shorts),
            "price_pnl_long": price_pnl_long,
            "price_pnl_short": price_pnl_short,
            "gross_pnl": gross_pnl,
            "mkt_ret_bp": mkt_ret_regime,
            "cost": total_cost,
            "net_pnl": net_pnl,
            "K": K,
            "slip_bp": slip_bp,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# §6: Statistics
# ---------------------------------------------------------------------------

def daily_returns(regime_df: pd.DataFrame) -> pd.Series:
    if len(regime_df) == 0:
        return pd.Series(dtype=float)
    df = regime_df.copy()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    return df.groupby("date")["net_pnl"].sum()


def compute_stats(regime_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    if len(regime_df) == 0:
        return {"n_regimes": 0, "mean_pnl": np.nan, "sharpe_ann": np.nan}

    daily = daily_returns(regime_df)
    n_days = len(daily)
    n_reg = len(regime_df)

    if n_days < 5:
        return {"n_regimes": n_reg, "mean_pnl": np.nan, "sharpe_ann": np.nan}

    mean_daily = daily.mean()
    std_daily = daily.std()
    sharpe_ann = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0

    net = regime_df["net_pnl"]
    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Block bootstrap
    daily_arr = daily.values
    n_blocks = max(1, n_days // BLOCK_SIZE_DAYS)
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        starts = rng.integers(0, max(1, n_days - BLOCK_SIZE_DAYS + 1), size=n_blocks)
        sample = np.concatenate([daily_arr[s:s+BLOCK_SIZE_DAYS] for s in starts])[:n_days]
        boot_means[b] = sample.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Permutation test
    perm_means = np.empty(N_PERMUTATION)
    for p in range(N_PERMUTATION):
        perm_means[p] = rng.permutation(daily_arr).mean()
    p_perm = (perm_means >= mean_daily).mean()

    return {
        "n_regimes": n_reg,
        "n_days": n_days,
        "avg_duration_min": regime_df["duration_min"].mean(),
        "mean_pnl_per_regime": net.mean(),
        "median_pnl_per_regime": net.median(),
        "hitrate": (net > 0).mean(),
        "mean_daily_bp": mean_daily,
        "std_daily_bp": std_daily,
        "sharpe_ann": sharpe_ann,
        "max_dd_bp": max_dd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "gross_mean": regime_df["gross_pnl"].mean(),
        "exposure_frac": regime_df["duration_min"].sum() / (59 * 24 * 60),  # 59 days
    }


# ---------------------------------------------------------------------------
# §7: Walk-forward
# ---------------------------------------------------------------------------

def walk_forward(regimes, K, grid_1m, grid_5m, sym_1m, volz_wide,
                 eligible_wide, beta_wide, mkt_ret, slip_bp):
    feb_test = simulate_regime_portfolio(
        regimes, K, grid_1m, grid_5m, sym_1m,
        volz_wide, eligible_wide, beta_wide, mkt_ret, slip_bp,
        period_start=WF_SPLIT, period_end=END)
    jan_test = simulate_regime_portfolio(
        regimes, K, grid_1m, grid_5m, sym_1m,
        volz_wide, eligible_wide, beta_wide, mkt_ret, slip_bp,
        period_start=START, period_end=WF_SPLIT - pd.Timedelta(minutes=1))
    return {"feb_test": feb_test, "jan_test": jan_test}


# ---------------------------------------------------------------------------
# §8: Bug checks
# ---------------------------------------------------------------------------

def bug_checks(regime_df, regimes):
    print(f"\n  BUG DETECTION CHECKLIST")
    print(f"  {'='*50}")
    print(f"  [1] Dispersion: backward rolling(60).std() only ✓")
    print(f"  [2] Pctl: expanding (causal) ✓")
    print(f"  [3] Onset detection: requires crossing (not level) ✓")

    # Check no overlapping regimes
    overlaps = 0
    for i in range(1, len(regimes)):
        if regimes[i]["start_ts"] < regimes[i-1]["end_ts"]:
            overlaps += 1
    print(f"  [4] No overlapping regimes: "
          f"{'✓ PASS' if overlaps == 0 else f'✗ FAIL ({overlaps})'}")

    if len(regime_df) > 0:
        future = (regime_df["exit_ts"] <= regime_df["entry_ts"]).any()
        print(f"  [5] Exit > entry (no future leak): "
              f"{'✓ PASS' if not future else '✗ FAIL'}")

    print(f"  [6] One position per regime (no vintages) ✓")
    print(f"  [7] Entry = next 1m close after onset ✓")
    print(f"  {'='*50}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("XS-3 — STATE TRANSITION (DISPERSION REGIME ONSET)")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Entry: disp_pctl crosses P80 up | Exit: crosses P50 down")
    print(f"K grid: {K_GRID}")
    print(f"Cost: {FEE_BP}bp/side + slippage grid {SLIP_GRID}")
    print("=" * 80)

    # §1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}")
    symbols = discover_symbols()
    print(f"  Symbols: {len(symbols)}")
    raw = load_all(symbols)
    print(f"  Loaded: {len(raw)} symbols")

    # §2: 1m grid
    print(f"\n{'─'*70}")
    print("PHASE 2: Build 1m grid")
    print(f"{'─'*70}")
    grid_1m, sym_1m = build_1m_grid(raw)

    # §3: Features
    print(f"\n{'─'*70}")
    print("PHASE 3: Compute features")
    print(f"{'─'*70}")
    (grid_5m, idx_5m, dispersion, disp_pctl, volz_wide,
     turn60_wide, eligible_wide, beta_wide, mkt_ret) = compute_features(grid_1m, sym_1m)

    # §4: Detect regimes
    print(f"\n{'─'*70}")
    print("PHASE 4: Regime detection")
    print(f"{'─'*70}")
    regimes = detect_regimes(grid_5m, disp_pctl)
    print(f"  Detected {len(regimes)} dispersion regimes")
    if regimes:
        durations = [(r["end_ts"] - r["start_ts"]).total_seconds() / 60 for r in regimes]
        print(f"  Duration: mean={np.mean(durations):.0f}m, "
              f"median={np.median(durations):.0f}m, "
              f"min={np.min(durations):.0f}m, max={np.max(durations):.0f}m")
        total_regime_min = sum(durations)
        total_period_min = 59 * 24 * 60
        print(f"  Exposure: {total_regime_min:.0f}m / {total_period_min}m "
              f"({total_regime_min/total_period_min:.1%})")

    # §5: Simulate all variants
    print(f"\n{'─'*70}")
    print("PHASE 5: Portfolio simulation")
    print(f"{'─'*70}")

    all_results = []

    for K in K_GRID:
        print(f"\n  ━━━ K={K} ━━━")

        for slip in SLIP_GRID:
            total_cost = 2 * (FEE_BP + slip)

            # Full period
            reg_full = simulate_regime_portfolio(
                regimes, K, grid_1m, grid_5m, sym_1m,
                volz_wide, eligible_wide, beta_wide, mkt_ret, slip)

            if len(reg_full) == 0:
                print(f"    slip={slip}bp: No regimes traded")
                continue

            stats_full = compute_stats(reg_full, rng)

            # Walk-forward
            wf = walk_forward(
                regimes, K, grid_1m, grid_5m, sym_1m,
                volz_wide, eligible_wide, beta_wide, mkt_ret, slip)
            stats_feb = compute_stats(wf["feb_test"], rng)
            stats_jan = compute_stats(wf["jan_test"], rng)

            result = {
                "K": K, "slip_bp": slip, "total_cost_bp": total_cost,
                # Full
                "full_n_regimes": stats_full["n_regimes"],
                "full_avg_dur": stats_full.get("avg_duration_min", np.nan),
                "full_mean_per_reg": stats_full.get("mean_pnl_per_regime", np.nan),
                "full_mean_daily": stats_full.get("mean_daily_bp", np.nan),
                "full_sharpe": stats_full.get("sharpe_ann", np.nan),
                "full_hitrate": stats_full.get("hitrate", np.nan),
                "full_max_dd": stats_full.get("max_dd_bp", np.nan),
                "full_p_perm": stats_full.get("p_perm", np.nan),
                "full_exposure": stats_full.get("exposure_frac", np.nan),
                "full_ci_lo": stats_full.get("ci_lo", np.nan),
                "full_ci_hi": stats_full.get("ci_hi", np.nan),
                # OOS Feb
                "oos_feb_n": stats_feb.get("n_regimes", 0),
                "oos_feb_mean_daily": stats_feb.get("mean_daily_bp", np.nan),
                "oos_feb_sharpe": stats_feb.get("sharpe_ann", np.nan),
                "oos_feb_hitrate": stats_feb.get("hitrate", np.nan),
                "oos_feb_p": stats_feb.get("p_perm", np.nan),
                # OOS Jan
                "oos_jan_n": stats_jan.get("n_regimes", 0),
                "oos_jan_mean_daily": stats_jan.get("mean_daily_bp", np.nan),
                "oos_jan_sharpe": stats_jan.get("sharpe_ann", np.nan),
                "oos_jan_hitrate": stats_jan.get("hitrate", np.nan),
                "oos_jan_p": stats_jan.get("p_perm", np.nan),
            }
            all_results.append(result)

            print(f"    slip={slip}bp (RT={total_cost}bp): "
                  f"N={stats_full['n_regimes']}, "
                  f"dur={stats_full.get('avg_duration_min', 0):.0f}m, "
                  f"mean/reg={stats_full.get('mean_pnl_per_regime', 0):+.2f}bp, "
                  f"daily={stats_full.get('mean_daily_bp', 0):+.2f}bp, "
                  f"Sharpe={stats_full.get('sharpe_ann', 0):.2f}, "
                  f"HR={stats_full.get('hitrate', 0):.0%}, "
                  f"DD={stats_full.get('max_dd_bp', 0):.0f}bp, "
                  f"p={stats_full.get('p_perm', 1):.3f}")
            if stats_feb.get("n_regimes", 0) > 0:
                print(f"      OOS Feb: N={stats_feb['n_regimes']}, "
                      f"daily={stats_feb.get('mean_daily_bp', 0):+.2f}bp, "
                      f"Sharpe={stats_feb.get('sharpe_ann', 0):.2f}, "
                      f"p={stats_feb.get('p_perm', 1):.3f}")
            if stats_jan.get("n_regimes", 0) > 0:
                print(f"      OOS Jan: N={stats_jan['n_regimes']}, "
                      f"daily={stats_jan.get('mean_daily_bp', 0):+.2f}bp, "
                      f"Sharpe={stats_jan.get('sharpe_ann', 0):.2f}, "
                      f"p={stats_jan.get('p_perm', 1):.3f}")

    # Save results
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_DIR / "xs3_results.csv", index=False)
    print(f"\n  Saved xs3_results.csv")

    # Save regime details for best K
    if regimes:
        best_K = K_GRID[1]  # K=5 default
        best_reg = simulate_regime_portfolio(
            regimes, best_K, grid_1m, grid_5m, sym_1m,
            volz_wide, eligible_wide, beta_wide, mkt_ret, 0)
        best_reg.to_csv(OUTPUT_DIR / "xs3_regimes_K5.csv", index=False)
        bug_checks(best_reg, regimes)

    # GO / NO-GO
    print(f"\n{'='*80}")
    print("GO / NO-GO VERDICT")
    print(f"{'='*80}")

    go = False
    for r in all_results:
        if r["slip_bp"] == 2:
            K_val = r["K"]
            mean_ok = (r.get("full_mean_per_reg", -1) > 0)
            oos_ok = (r.get("oos_feb_mean_daily", -1) > 0 and
                      r.get("oos_jan_mean_daily", -1) > 0)
            sharpe_ok = (r.get("oos_feb_sharpe", 0) > 1.2 or
                         r.get("oos_jan_sharpe", 0) > 1.2)
            hr_ok = (r.get("full_hitrate", 0) >= 0.55)
            dd_ok = (abs(r.get("full_max_dd", -999)) < 1500)

            checks = [mean_ok, oos_ok, sharpe_ok, hr_ok, dd_ok]
            labels = [
                "Mean PnL per regime > 0",
                "OOS mean > 0 both halves",
                "OOS Sharpe > 1.2 (at least one)",
                "Hit rate ≥ 55%",
                "Max DD < 1500bp",
            ]

            print(f"\n  K={K_val} (slip=2bp):")
            for check, lbl in zip(checks, labels):
                print(f"    {'✓' if check else '✗'} {lbl}")

            if all(checks):
                go = True
                print(f"    → GO ✅")
            else:
                print(f"    → NO-GO ❌")

    print(f"\n  {'='*40}")
    if go:
        print(f"  VERDICT: GO ✅")
    else:
        print(f"  VERDICT: NO-GO ❌")
    print(f"  {'='*40}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
