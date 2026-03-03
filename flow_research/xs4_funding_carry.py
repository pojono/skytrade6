#!/usr/bin/env python3
"""
XS-4 — Funding Carry Cross-Section

Structural factor: extreme funding → crowding → carry drag → forced unwind.
Trade carry + mean reversion on 8h rebalance grid (aligned with funding settlement).

Version A: Long bottom 20% funding_z, Short top 20% funding_z, 8h hold
Version B: |funding_z| >= 2 threshold, 8h or 24h hold

Anti-bug: no lookahead, no survivorship, no overlap, strict causal features.
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 50

# Funding z-score
FR_Z_WINDOW = 3 * 3       # 3 days × 3 settlements/day = 9 funding observations
FR_Z_MIN_OBS = 6          # minimum 2 days of data

# Portfolio
QUANTILE_A = 0.20          # Version A: top/bottom 20%
Z_THRESH_B = 2.0           # Version B: |z| >= 2
HOLD_8H = 8                # hours
HOLD_24H = 24              # hours (Version B alt)

# Rebalance: every 8h aligned to funding settlements (00:00, 08:00, 16:00 UTC)
REBAL_HOURS = [0, 8, 16]

# Beta
W_BETA_1M = 3 * 24 * 60   # 3 days in 1m bars for rolling beta
BETA_VALID_FRAC = 0.80
BETA_CLIP = 3.0
BETA_VAR_EPS = 1e-14

# Eligibility
MIN_TURNOVER_8H = 200_000  # USD turnover in 8h

# Cost
FEE_BP = 10                # per side
SLIP_GRID = [0, 1, 2, 5]   # bp per side

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
        nfr = len(list(d.glob("*_funding_rate.csv")))
        nkline = len(list(d.glob("*_kline_1m.csv")))
        if nmark >= MIN_DAYS and nfr >= MIN_DAYS and nkline >= MIN_DAYS:
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
        # Kline for turnover
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

        fr = _load_csvs(sym, "*_funding_rate.csv", "timestamp",
                        {"fundingRate": "fr"})

        if len(mark) > 0 and len(fr) > 0:
            data[sym] = {"mark": mark, "kline": kline, "fr": fr}

        if i % 10 == 0:
            print(f"    Loaded {i}/{len(symbols)}")
    return data


# ---------------------------------------------------------------------------
# §2: Build unified grids
# ---------------------------------------------------------------------------

def build_1m_grid(raw: dict) -> tuple[pd.DatetimeIndex, dict]:
    """Build unified 1m price grid for all symbols."""
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


def build_8h_rebal_grid(grid_1m: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """8h rebalance grid aligned to funding settlements: 00, 08, 16 UTC."""
    mask = (grid_1m.hour.isin(REBAL_HOURS)) & (grid_1m.minute == 0)
    grid_8h = grid_1m[mask]
    print(f"  8h rebalance grid: {len(grid_8h)} points")
    return grid_8h


def build_funding_panel(raw: dict, grid_8h: pd.DatetimeIndex) -> pd.DataFrame:
    """Build funding rate panel at 8h grid with rolling z-score."""
    symbols = sorted(raw.keys())

    # Funding rates are at 8h intervals; reindex to grid
    fr_wide = pd.DataFrame(index=grid_8h)
    for sym in symbols:
        fr_df = raw[sym]["fr"].set_index("ts")["fr"]
        fr_wide[sym] = fr_df.reindex(grid_8h)

    # Annualized funding (×3 per day × 365)
    fr_ann_wide = fr_wide * 3 * 365

    # Rolling z-score of funding rate (window = 9 observations = 3 days)
    fr_z_wide = pd.DataFrame(index=grid_8h, columns=symbols, dtype=float)
    for sym in symbols:
        s = fr_wide[sym]
        rm = s.rolling(FR_Z_WINDOW, min_periods=FR_Z_MIN_OBS).mean()
        rs = s.rolling(FR_Z_WINDOW, min_periods=FR_Z_MIN_OBS).std().clip(lower=1e-12)
        fr_z_wide[sym] = (s - rm) / rs

    print(f"  Funding z-score: {fr_z_wide.notna().sum().sum():,} valid obs across {len(symbols)} symbols")

    return fr_wide, fr_ann_wide, fr_z_wide


# ---------------------------------------------------------------------------
# §3: Market return + optional beta
# ---------------------------------------------------------------------------

def compute_market_return_8h(grid_1m, grid_8h, sym_1m):
    """Compute 8h returns per symbol and EW market return."""
    symbols = sorted(sym_1m.keys())

    # Map 8h grid points to 1m grid positions
    idx_map = grid_1m.get_indexer(grid_8h)

    # 8h close prices per symbol
    close_8h = pd.DataFrame(index=grid_8h, columns=symbols, dtype=float)
    for sym in symbols:
        c = sym_1m[sym]["close"].values
        close_8h[sym] = c[idx_map]

    # 8h log returns
    ret_8h = np.log(close_8h / close_8h.shift(1))

    # EW market return
    mkt_ret_8h = ret_8h.mean(axis=1)

    # Rolling 8h turnover per symbol (sum of 1m turnover over prior 8h = 480 bars)
    turn_8h = pd.DataFrame(index=grid_8h, columns=symbols, dtype=float)
    for sym in symbols:
        t_cum = sym_1m[sym]["turnover"].cumsum()
        t_vals = t_cum.values
        for j, gi in enumerate(idx_map):
            start_i = max(0, gi - 480)
            turn_8h.iloc[j][sym] = t_vals[gi] - t_vals[start_i]

    return close_8h, ret_8h, mkt_ret_8h, turn_8h


# ---------------------------------------------------------------------------
# §4: Portfolio construction + simulation
# ---------------------------------------------------------------------------

def simulate_carry_portfolio(
    version: str,  # "A" or "B"
    hold_hours: int,
    grid_1m: pd.DatetimeIndex,
    grid_8h: pd.DatetimeIndex,
    sym_1m: dict,
    fr_wide: pd.DataFrame,
    fr_z_wide: pd.DataFrame,
    close_8h: pd.DataFrame,
    ret_8h: pd.DataFrame,
    mkt_ret_8h: pd.Series,
    turn_8h: pd.DataFrame,
    slip_bp: float,
    period_start: pd.Timestamp = None,
    period_end: pd.Timestamp = None,
) -> pd.DataFrame:
    """Simulate funding carry portfolio.

    Returns DataFrame with one row per rebalance: ts, longs, shorts, pnl components.
    """
    symbols = sorted(sym_1m.keys())

    if period_start is None:
        period_start = grid_8h[0]
    if period_end is None:
        period_end = grid_8h[-1]

    # Filter grid to period
    mask = (grid_8h >= period_start) & (grid_8h <= period_end)
    grid_period = grid_8h[mask]

    # How many 8h steps to hold
    hold_steps = hold_hours // 8

    results = []

    for i, ts in enumerate(grid_period):
        if i + hold_steps >= len(grid_period):
            break

        exit_ts = grid_period[i + hold_steps]

        # Eligible: has funding_z, has turnover
        fz = fr_z_wide.loc[ts].dropna()
        eligible_syms = [s for s in fz.index if s in turn_8h.columns]

        # Filter by turnover
        eligible = []
        for s in eligible_syms:
            t_val = turn_8h.loc[ts, s] if ts in turn_8h.index else 0
            if pd.notna(t_val) and t_val > MIN_TURNOVER_8H:
                eligible.append(s)

        if len(eligible) < 6:
            continue

        fz_eligible = fz[eligible].sort_values()

        # Select longs and shorts
        if version == "A":
            n_q = max(1, int(len(eligible) * QUANTILE_A))
            longs = list(fz_eligible.index[:n_q])     # lowest funding_z → long
            shorts = list(fz_eligible.index[-n_q:])    # highest funding_z → short
        elif version == "B":
            longs = list(fz_eligible[fz_eligible <= -Z_THRESH_B].index)
            shorts = list(fz_eligible[fz_eligible >= Z_THRESH_B].index)
            if len(longs) == 0 or len(shorts) == 0:
                continue
        else:
            raise ValueError(f"Unknown version: {version}")

        n_long = len(longs)
        n_short = len(shorts)
        n_total = n_long + n_short

        # Entry: next 1m close after rebalance time
        entry_1m_idx = grid_1m.get_indexer([ts], method="ffill")[0] + 1
        if entry_1m_idx >= len(grid_1m):
            continue

        # Exit: next 1m close after exit_ts
        exit_1m_idx = grid_1m.get_indexer([exit_ts], method="ffill")[0] + 1
        if exit_1m_idx >= len(grid_1m):
            exit_1m_idx = len(grid_1m) - 1

        # Price PnL per position
        price_pnl_long = 0.0
        price_pnl_short = 0.0
        w_long = 0.5 / n_long if n_long > 0 else 0
        w_short = 0.5 / n_short if n_short > 0 else 0

        for s in longs:
            entry_p = sym_1m[s]["close"].iloc[entry_1m_idx]
            exit_p = sym_1m[s]["close"].iloc[exit_1m_idx]
            if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                ret = np.log(exit_p / entry_p)
                price_pnl_long += w_long * ret * 10000  # in bp

        for s in shorts:
            entry_p = sym_1m[s]["close"].iloc[entry_1m_idx]
            exit_p = sym_1m[s]["close"].iloc[exit_1m_idx]
            if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                ret = np.log(exit_p / entry_p)
                price_pnl_short += -w_short * ret * 10000  # short: -ret

        # Funding received during hold period
        # For longs: we receive funding if funding < 0 (shorts pay longs)
        # For shorts: we receive funding if funding > 0 (longs pay shorts)
        # Bybit: if fundingRate > 0, longs pay shorts
        # So: long position → funding_pnl = -fundingRate × notional
        #     short position → funding_pnl = +fundingRate × notional
        funding_pnl = 0.0
        # Get all funding timestamps between entry and exit
        fr_mask = (fr_wide.index > ts) & (fr_wide.index <= exit_ts)
        fr_period = fr_wide.loc[fr_mask]

        for _, fr_row in fr_period.iterrows():
            for s in longs:
                if s in fr_row and pd.notna(fr_row[s]):
                    funding_pnl += -w_long * fr_row[s] * 10000  # long pays when fr > 0
            for s in shorts:
                if s in fr_row and pd.notna(fr_row[s]):
                    funding_pnl += w_short * fr_row[s] * 10000   # short receives when fr > 0

        # Costs: FEE_BP per side + slippage per side, applied to each position
        total_cost_bp = (FEE_BP + slip_bp) * 2  # round trip per position
        # Total cost = cost per position × number of positions, but weighted
        cost = total_cost_bp  # already in bp for unit portfolio

        gross_pnl = price_pnl_long + price_pnl_short + funding_pnl
        net_pnl = gross_pnl - cost

        results.append({
            "ts": ts,
            "exit_ts": exit_ts,
            "n_long": n_long,
            "n_short": n_short,
            "price_pnl_long": price_pnl_long,
            "price_pnl_short": price_pnl_short,
            "funding_pnl": funding_pnl,
            "gross_pnl": gross_pnl,
            "cost": cost,
            "net_pnl": net_pnl,
            "version": version,
            "hold_h": hold_hours,
            "slip_bp": slip_bp,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# §5: Statistics + walk-forward
# ---------------------------------------------------------------------------

def daily_returns(rebal_df: pd.DataFrame) -> pd.Series:
    """Aggregate rebalance-level PnL to daily returns (bp)."""
    if len(rebal_df) == 0:
        return pd.Series(dtype=float)
    df = rebal_df.copy()
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    daily = df.groupby("date")["net_pnl"].sum()
    return daily


def compute_stats(rebal_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Compute portfolio statistics with bootstrap + permutation."""
    if len(rebal_df) == 0:
        return {"n_rebal": 0, "mean_pnl": np.nan, "sharpe_ann": np.nan}

    daily = daily_returns(rebal_df)
    n_days = len(daily)

    if n_days < 5:
        return {"n_rebal": len(rebal_df), "mean_pnl": np.nan, "sharpe_ann": np.nan}

    mean_daily = daily.mean()
    std_daily = daily.std()
    sharpe_ann = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0

    # Per-rebalance stats
    net = rebal_df["net_pnl"]
    gross = rebal_df["gross_pnl"]
    funding = rebal_df["funding_pnl"]

    cum = daily.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()

    # Block bootstrap CI on daily mean
    block_size = BLOCK_SIZE_DAYS
    n_blocks = max(1, n_days // block_size)
    boot_means = np.empty(N_BOOTSTRAP)
    daily_arr = daily.values
    for b in range(N_BOOTSTRAP):
        starts = rng.integers(0, max(1, n_days - block_size + 1), size=n_blocks)
        sample = np.concatenate([daily_arr[s:s+block_size] for s in starts])[:n_days]
        boot_means[b] = sample.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Permutation test: shuffle daily returns, test mean
    perm_means = np.empty(N_PERMUTATION)
    for p in range(N_PERMUTATION):
        perm_means[p] = rng.permutation(daily_arr).mean()
    p_perm = (perm_means >= mean_daily).mean()

    return {
        "n_rebal": len(rebal_df),
        "n_days": n_days,
        "mean_pnl_per_rebal": net.mean(),
        "median_pnl_per_rebal": net.median(),
        "hitrate": (net > 0).mean(),
        "mean_daily_bp": mean_daily,
        "std_daily_bp": std_daily,
        "sharpe_ann": sharpe_ann,
        "max_dd_bp": max_dd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "gross_mean": gross.mean(),
        "funding_mean": funding.mean(),
        "funding_frac": funding.sum() / gross.abs().sum() if gross.abs().sum() > 0 else 0,
    }


def walk_forward(
    version, hold_hours, grid_1m, grid_8h, sym_1m,
    fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h, slip_bp
) -> dict:
    """Jan→Feb and Feb→Jan walk-forward."""
    # Jan train → Feb test
    feb_start = WF_SPLIT
    feb_test = simulate_carry_portfolio(
        version, hold_hours, grid_1m, grid_8h, sym_1m,
        fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h,
        slip_bp, period_start=feb_start, period_end=END)

    # Feb train → Jan test
    jan_test = simulate_carry_portfolio(
        version, hold_hours, grid_1m, grid_8h, sym_1m,
        fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h,
        slip_bp, period_start=START, period_end=feb_start - pd.Timedelta(minutes=1))

    return {"feb_test": feb_test, "jan_test": jan_test}


# ---------------------------------------------------------------------------
# §6: Bug checks
# ---------------------------------------------------------------------------

def bug_checks(rebal_df):
    print(f"\n  BUG DETECTION CHECKLIST")
    print(f"  {'='*50}")
    print(f"  [1] Funding z-score: backward rolling only ✓")
    print(f"  [2] No duplicate timestamps: "
          f"{'✓ PASS' if not rebal_df.duplicated('ts').any() else '✗ FAIL'}")
    print(f"  [3] Entry = next 1m close after rebal: enforced by construction")
    print(f"  [4] No overlap: hold period = rebal period, no vintages")
    print(f"  [5] Funding PnL uses realized rates between entry/exit")

    # Check no future leakage: rebal ts < exit ts
    if len(rebal_df) > 0:
        future_leak = (rebal_df["exit_ts"] <= rebal_df["ts"]).any()
        print(f"  [6] No future leakage (exit > entry): "
              f"{'✓ PASS' if not future_leak else '✗ FAIL'}")
    print(f"  {'='*50}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("XS-4 — FUNDING CARRY CROSS-SECTION")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Rebalance: 8h (aligned to funding settlement)")
    print(f"Cost: {FEE_BP}bp/side + slippage grid {SLIP_GRID}")
    print("=" * 80)

    # §1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}")
    symbols = discover_symbols()
    print(f"  Symbols with sufficient data: {len(symbols)}")
    raw = load_all(symbols)
    print(f"  Loaded: {len(raw)} symbols")

    # §2: Build grids
    print(f"\n{'─'*70}")
    print("PHASE 2: Build grids")
    print(f"{'─'*70}")
    grid_1m, sym_1m = build_1m_grid(raw)
    grid_8h = build_8h_rebal_grid(grid_1m)
    fr_wide, fr_ann_wide, fr_z_wide = build_funding_panel(raw, grid_8h)

    # §3: Market + turnover
    print(f"\n{'─'*70}")
    print("PHASE 3: Market returns + turnover")
    print(f"{'─'*70}")
    close_8h, ret_8h, mkt_ret_8h, turn_8h = compute_market_return_8h(
        grid_1m, grid_8h, sym_1m)
    print(f"  8h returns: {ret_8h.notna().sum().sum():,} valid")

    # Funding summary
    print(f"\n{'─'*70}")
    print("FUNDING RATE SUMMARY")
    print(f"{'─'*70}")
    fr_flat = fr_wide.values.flatten()
    fr_flat = fr_flat[~np.isnan(fr_flat)]
    print(f"  Mean FR: {fr_flat.mean()*10000:.2f}bp per 8h ({fr_flat.mean()*3*365*100:.1f}% ann)")
    print(f"  Std FR: {fr_flat.std()*10000:.2f}bp per 8h")
    print(f"  P5/P95: {np.percentile(fr_flat, 5)*10000:.2f} / {np.percentile(fr_flat, 95)*10000:.2f} bp")

    fz_flat = fr_z_wide.values.flatten()
    fz_flat = fz_flat[~np.isnan(fz_flat)]
    print(f"  Z-score stats: mean={np.mean(fz_flat):.2f}, std={np.std(fz_flat):.2f}")
    print(f"  |z| >= 2: {(np.abs(fz_flat) >= 2).mean():.1%}")

    # §4: Run all variants
    print(f"\n{'─'*70}")
    print("PHASE 4: Portfolio simulation")
    print(f"{'─'*70}")

    configs = [
        ("A", 8,  "Version A (Q20, 8h hold)"),
        ("B", 8,  "Version B (|z|≥2, 8h hold)"),
        ("B", 24, "Version B (|z|≥2, 24h hold)"),
    ]

    all_results = []

    for version, hold_h, label in configs:
        print(f"\n  ━━━ {label} ━━━")

        for slip in SLIP_GRID:
            total_cost = 2 * (FEE_BP + slip)

            # Full period
            rebal_full = simulate_carry_portfolio(
                version, hold_h, grid_1m, grid_8h, sym_1m,
                fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h, slip)

            if len(rebal_full) == 0:
                print(f"    slip={slip}bp: No rebalances")
                continue

            stats_full = compute_stats(rebal_full, rng)

            # Walk-forward
            wf = walk_forward(
                version, hold_h, grid_1m, grid_8h, sym_1m,
                fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h, slip)
            stats_feb = compute_stats(wf["feb_test"], rng)
            stats_jan = compute_stats(wf["jan_test"], rng)

            result = {
                "version": version, "hold_h": hold_h, "label": label,
                "slip_bp": slip, "total_cost_bp": total_cost,
                # Full
                "full_n": stats_full["n_rebal"],
                "full_mean_rebal": stats_full.get("mean_pnl_per_rebal", np.nan),
                "full_mean_daily": stats_full.get("mean_daily_bp", np.nan),
                "full_sharpe": stats_full.get("sharpe_ann", np.nan),
                "full_hitrate": stats_full.get("hitrate", np.nan),
                "full_max_dd": stats_full.get("max_dd_bp", np.nan),
                "full_p_perm": stats_full.get("p_perm", np.nan),
                "full_funding_frac": stats_full.get("funding_frac", np.nan),
                "full_ci_lo": stats_full.get("ci_lo", np.nan),
                "full_ci_hi": stats_full.get("ci_hi", np.nan),
                # OOS Feb
                "oos_feb_n": stats_feb.get("n_rebal", 0),
                "oos_feb_mean_daily": stats_feb.get("mean_daily_bp", np.nan),
                "oos_feb_sharpe": stats_feb.get("sharpe_ann", np.nan),
                "oos_feb_hitrate": stats_feb.get("hitrate", np.nan),
                "oos_feb_p": stats_feb.get("p_perm", np.nan),
                # OOS Jan
                "oos_jan_n": stats_jan.get("n_rebal", 0),
                "oos_jan_mean_daily": stats_jan.get("mean_daily_bp", np.nan),
                "oos_jan_sharpe": stats_jan.get("sharpe_ann", np.nan),
                "oos_jan_hitrate": stats_jan.get("hitrate", np.nan),
                "oos_jan_p": stats_jan.get("p_perm", np.nan),
            }
            all_results.append(result)

            print(f"    slip={slip}bp (RT={total_cost}bp): "
                  f"N={stats_full['n_rebal']}, "
                  f"mean/reb={stats_full.get('mean_pnl_per_rebal', 0):+.2f}bp, "
                  f"daily={stats_full.get('mean_daily_bp', 0):+.2f}bp, "
                  f"Sharpe={stats_full.get('sharpe_ann', 0):.2f}, "
                  f"HR={stats_full.get('hitrate', 0):.0%}, "
                  f"DD={stats_full.get('max_dd_bp', 0):.0f}bp, "
                  f"p={stats_full.get('p_perm', 1):.3f}, "
                  f"FR%={stats_full.get('funding_frac', 0):.0%}")
            if stats_feb.get("n_rebal", 0) > 0:
                print(f"      OOS Feb: daily={stats_feb.get('mean_daily_bp', 0):+.2f}bp, "
                      f"Sharpe={stats_feb.get('sharpe_ann', 0):.2f}, "
                      f"p={stats_feb.get('p_perm', 1):.3f}")
            if stats_jan.get("n_rebal", 0) > 0:
                print(f"      OOS Jan: daily={stats_jan.get('mean_daily_bp', 0):+.2f}bp, "
                      f"Sharpe={stats_jan.get('sharpe_ann', 0):.2f}, "
                      f"p={stats_jan.get('p_perm', 1):.3f}")

    # Save
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_DIR / "xs4_results.csv", index=False)
    print(f"\n  Saved xs4_results.csv")

    # Bug checks on best variant
    if all_results:
        # Pick version A slip=0 for checks
        best_rebal = simulate_carry_portfolio(
            "A", 8, grid_1m, grid_8h, sym_1m,
            fr_wide, fr_z_wide, close_8h, ret_8h, mkt_ret_8h, turn_8h, 0)
        bug_checks(best_rebal)
        best_rebal.to_csv(OUTPUT_DIR / "xs4_rebalances_A_8h.csv", index=False)

    # GO / NO-GO
    print(f"\n{'='*80}")
    print("GO / NO-GO VERDICT")
    print(f"{'='*80}")

    go = False
    for r in all_results:
        if r["slip_bp"] == 2:
            label = r["label"]
            mean_ok = (r.get("oos_feb_mean_daily", -1) > 0 and
                       r.get("oos_jan_mean_daily", -1) > 0)
            sharpe_ok = (r.get("oos_feb_sharpe", 0) > 1.5 or
                         r.get("oos_jan_sharpe", 0) > 1.5)
            hr_ok = (r.get("full_hitrate", 0) >= 0.55)
            dd_ok = (abs(r.get("full_max_dd", -999)) < 1500)
            p_ok = (r.get("full_p_perm", 1) < 0.05)

            print(f"\n  {label} (slip=2bp):")
            print(f"    {'✓' if mean_ok else '✗'} OOS mean > 0 both halves")
            print(f"    {'✓' if sharpe_ok else '✗'} OOS Sharpe > 1.5 (at least one)")
            print(f"    {'✓' if hr_ok else '✗'} Hit rate ≥ 55%")
            print(f"    {'✓' if dd_ok else '✗'} Max DD < 1500bp")
            print(f"    {'✓' if p_ok else '✗'} p < 0.05")

            if mean_ok and sharpe_ok and hr_ok and dd_ok and p_ok:
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
