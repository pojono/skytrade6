#!/usr/bin/env python3
"""
XS-6 — Extreme Move Probability Model (Big Move Uplift)

Goal: For each coin and universe-wide, estimate:
  P(big_move in next H | state S)
  uplift = P(big_move | S) / P(big_move)

Find states where uplift is robustly > 1.5-2.5x OOS.

Data: 50+ Bybit perps, 2026-01-01 → 2026-02-28
Grid: 5-minute signal intervals on unified 1m grid
Targets: |ret| over 12h/24h, ATR-normalized (Def A) and raw bp (Def B)
States: 10 interpretable states (S1-S10) from funding/OI/vol/trend
Protection: volatility-matched baseline, permutation tests, BH FDR, purged OOS split
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 50  # minimum data files per type

# Signal grid: every 5 minutes
SIGNAL_STEP_MIN = 5

# Target horizons (in minutes)
HORIZONS = {
    "12h": 12 * 60,
    "24h": 24 * 60,
}

# Big move definitions
# Def A: ATR-normalized  |ret| >= k * ATR_1h
ATR_K_VALUES = [3.0, 4.0]
# Def B: Raw bp thresholds
RAW_BP_THRESHOLDS = {
    "12h": 300,  # 300bp = 3%
    "24h": 500,  # 500bp = 5%
}

# Feature windows (in 1m bars)
FR_Z_WINDOW = 7 * 24 * 60      # 7 days rolling for z-score
OI_Z_WINDOW = 7 * 24 * 60      # 7 days rolling
OI_CHG_1H = 60
RV_2H = 120                    # 2h realized vol
RV_6H = 360                    # 6h realized vol
RET_30M = 30
TREND_2H = 120
ATR_14H = 14 * 60              # 14 1h bars for ATR (= 840 1m bars)
VOL_2H = 120                   # 2h volume window
VOL_1H = 60                    # 1h volume window

# Walk-forward split with purge
TRAIN_END = pd.Timestamp("2026-01-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-02-01", tz="UTC")
PURGE_HOURS = 24  # ±24h around boundary

# Permutation / bootstrap
N_PERMUTATION = 2000
SEED = 42

# Volatility-matched control
N_MATCHED_CONTROLS = 10

# PASS criteria
MIN_RATE_S = 0.003       # state frequency >= 0.3%
MIN_NS_TEST = 30         # minimum state events in OOS
MIN_UPLIFT_MATCHED = 1.5
MIN_DELTA_ABS = 0.01     # +1% absolute probability lift
MAX_Q_FDR = 0.10
MIN_WEEKLY_STABLE = 6    # uplift>1 in at least 6/9 weeks


# ---------------------------------------------------------------------------
# §1: Data loading (reuse xs5 patterns)
# ---------------------------------------------------------------------------

def discover_symbols() -> list[str]:
    syms = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        nmark = len(list(d.glob("*_mark_price_kline_1m.csv")))
        nkline = len([f for f in d.glob("*_kline_1m.csv")
                       if "mark_price" not in f.name and "premium_index" not in f.name])
        noi = len(list(d.glob("*_open_interest_5min.csv")))
        nfr = len(list(d.glob("*_funding_rate.csv")))
        if nmark >= MIN_DAYS and nkline >= MIN_DAYS and noi >= MIN_DAYS and nfr >= MIN_DAYS:
            syms.append(d.name)
    return syms


def _load_glob(sym: str, pattern: str, ts_col: str, val_cols: dict) -> pd.DataFrame:
    sym_dir = DATA_DIR / sym
    files = sorted(sym_dir.glob(pattern))
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
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out = out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    out = out[(out["ts"] >= START) & (out["ts"] <= END)]
    return out.reset_index(drop=True)


def load_symbol(sym: str) -> dict:
    mark = _load_glob(sym, "*_mark_price_kline_1m.csv", "startTime",
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
            for c in ["open", "high", "low", "close", "volume", "turnover"]:
                kl[c] = pd.to_numeric(kl[c], errors="coerce")
            kl = kl.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            kl = kl[(kl["ts"] >= START) & (kl["ts"] <= END)]
            kline = kl[["ts", "open", "high", "low", "close", "volume", "turnover"]].reset_index(drop=True)

    oi = _load_glob(sym, "*_open_interest_5min.csv", "timestamp",
                    {"openInterest": "oi"})
    fr = _load_glob(sym, "*_funding_rate.csv", "timestamp",
                    {"fundingRate": "fr"})

    return {"mark": mark, "kline": kline, "oi": oi, "fr": fr}


# ---------------------------------------------------------------------------
# §2: Build unified 1m grid per symbol
# ---------------------------------------------------------------------------

def build_sym_1m(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    mark = raw["mark"].set_index("ts")["close"] if len(raw["mark"]) > 0 else pd.Series(dtype=float)
    kl = raw["kline"]
    oi_df = raw["oi"]
    fr_df = raw["fr"]

    n = len(grid_1m)
    df = pd.DataFrame(index=grid_1m)

    # Close from mark price
    close_raw = mark.reindex(grid_1m)
    is_nan = close_raw.isna()
    nan_arr = is_nan.values
    block_len = np.zeros(n, dtype=np.int32)
    run = 0
    starts = []
    for i in range(n):
        if nan_arr[i]:
            run += 1
        else:
            if run > 0:
                starts.append((i - run, run))
            run = 0
    if run > 0:
        starts.append((n - run, run))
    for s, length in starts:
        block_len[s:s + length] = length

    is_ffill = np.zeros(n, dtype=np.int8)
    is_invalid = np.zeros(n, dtype=np.int8)
    is_ffill[nan_arr & (block_len > 0) & (block_len < 5)] = 1
    is_invalid[nan_arr & (block_len >= 5)] = 1

    close = close_raw.ffill()
    close[is_invalid == 1] = np.nan

    df["close"] = close
    df["is_ffill"] = is_ffill
    df["is_invalid"] = is_invalid

    # High/Low/Volume from kline
    if len(kl) > 0:
        kl_idx = kl.set_index("ts")
        df["high"] = kl_idx["high"].reindex(grid_1m).ffill()
        df["low"] = kl_idx["low"].reindex(grid_1m).ffill()
        df["volume"] = kl_idx["volume"].reindex(grid_1m).fillna(0)
        df["turnover"] = kl_idx["turnover"].reindex(grid_1m).fillna(0)
    else:
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0.0
        df["turnover"] = 0.0

    # OI: 5m → ffill to 1m, shifted +5min for causal alignment
    if len(oi_df) > 0:
        oi_shifted = oi_df.copy()
        oi_shifted["ts"] = oi_shifted["ts"] + pd.Timedelta(minutes=5)
        oi_s = oi_shifted.set_index("ts")["oi"]
        df["oi"] = oi_s.reindex(grid_1m).ffill()
    else:
        df["oi"] = np.nan

    # Funding: ffill to 1m, shifted +1min for causal alignment
    if len(fr_df) > 0:
        fr_shifted = fr_df.copy()
        fr_shifted["ts"] = fr_shifted["ts"] + pd.Timedelta(minutes=1)
        fr_s = fr_shifted.set_index("ts")["fr"]
        df["fr"] = fr_s.reindex(grid_1m).ffill()
    else:
        df["fr"] = np.nan

    # Log return
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.loc[df["is_invalid"] == 1, "log_ret"] = np.nan
    df.loc[df["is_invalid"].shift(1) == 1, "log_ret"] = np.nan

    return df


# ---------------------------------------------------------------------------
# §3: Feature computation (strictly causal)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for one symbol on 1m grid. All windows are causal."""

    # --- Funding z-score (7d rolling) ---
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["funding_z"] = (fr - fr_rm) / fr_rs

    # --- OI features ---
    oi = df["oi"]
    oi_lag_1h = oi.shift(OI_CHG_1H)
    df["oi_chg_1h"] = (oi - oi_lag_1h) / oi_lag_1h.clip(lower=1)
    # OI change z-score (7d rolling on oi_chg_1h)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["oi_z"] = (oi_chg - oi_rm) / oi_rs

    # --- Realized vol ---
    lr = df["log_ret"]
    df["rv_2h"] = lr.rolling(RV_2H, min_periods=RV_2H // 2).std()
    df["rv_6h"] = lr.rolling(RV_6H, min_periods=RV_6H // 2).std()

    # --- Volume aggregates ---
    df["volume_2h"] = df["turnover"].rolling(VOL_2H, min_periods=10).sum()
    df["volume_1h"] = df["turnover"].rolling(VOL_1H, min_periods=10).sum()

    # --- Trend strength 2h ---
    close = df["close"]
    ret_2h = np.log(close / close.shift(TREND_2H))
    rv_scaled = df["rv_2h"] * np.sqrt(TREND_2H)
    df["trend_strength_2h"] = ret_2h.abs() / rv_scaled.clip(lower=1e-12)

    # --- |ret_30m| ---
    df["abs_ret_30m"] = np.log(close / close.shift(RET_30M)).abs()

    # --- ATR 1h (14 periods = 14h) ---
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    # Aggregate to 1h bars first, then 14-period ATR
    tr_1h = tr.rolling(60, min_periods=30).sum()  # sum of 1m TRs in 1h
    df["atr_1h"] = tr_1h.rolling(14, min_periods=7).mean()  # rolling mean of 14 such sums
    # Simpler: just use rolling 14*60 = 840 bars average of 1m TR
    df["atr_1h_raw"] = tr.rolling(ATR_14H, min_periods=ATR_14H // 2).mean() * 60  # scale to hourly

    # --- Causal percentiles (rolling 30d = 43200 1m bars) ---
    PCTL_WINDOW = 30 * 24 * 60  # 30 days
    min_p = PCTL_WINDOW // 4

    # We compute percentile ranks for each feature causally
    # Using rolling rank / count for efficiency
    for col in ["rv_2h", "rv_6h", "volume_2h", "volume_1h", "trend_strength_2h", "abs_ret_30m"]:
        sr = df[col]
        rm = sr.rolling(PCTL_WINDOW, min_periods=min_p)
        df[f"{col}_pctl"] = rm.rank(pct=True)

    # ATR quintile (for vol-matched baseline)
    df["atr_quintile"] = pd.cut(
        df["atr_1h_raw"].rolling(PCTL_WINDOW, min_periods=min_p).rank(pct=True),
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )

    # Hour-of-day bucket (4h buckets: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23)
    df["hod_bucket"] = df.index.hour // 4

    return df


# ---------------------------------------------------------------------------
# §4: Target computation (big_move definitions)
# ---------------------------------------------------------------------------

def compute_targets(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute big_move targets at 5m grid using 1m close prices.

    For each 5m signal time t, compute forward return over (t, t+H].
    """
    close_1m = df_1m["close"]
    atr_1h = df_1m["atr_1h_raw"]
    invalid_1m = df_1m["is_invalid"]

    results = {}
    for h_label, h_min in HORIZONS.items():
        # Forward return: close at t+H / close at t - 1
        fwd_close = close_1m.shift(-h_min)
        fwd_ret = (fwd_close / close_1m) - 1.0  # simple return in ratio

        # Check no invalid bars in forward window
        fwd_invalid = invalid_1m.rolling(h_min, min_periods=1).sum().shift(-h_min)
        fwd_ret[fwd_invalid > 0] = np.nan

        # Align to 5m grid
        fwd_ret_5m = fwd_ret.reindex(df_5m.index)
        atr_5m = atr_1h.reindex(df_5m.index)

        results[f"fwd_ret_{h_label}"] = fwd_ret_5m

        # Def A: ATR-normalized
        for k in ATR_K_VALUES:
            threshold = k * atr_5m / df_5m["close"].clip(lower=1e-8)  # convert ATR to return
            col = f"big_A_k{k}_{h_label}"
            results[col] = (fwd_ret_5m.abs() >= threshold).astype(float)
            results[col][fwd_ret_5m.isna()] = np.nan

        # Def B: Raw bp
        bp_thresh = RAW_BP_THRESHOLDS[h_label] / 10000.0
        col = f"big_B_{h_label}"
        results[col] = (fwd_ret_5m.abs() >= bp_thresh).astype(float)
        results[col][fwd_ret_5m.isna()] = np.nan

    for col, vals in results.items():
        df_5m[col] = vals.values if hasattr(vals, 'values') else vals

    return df_5m


# ---------------------------------------------------------------------------
# §5: State definitions (S1-S10)
# ---------------------------------------------------------------------------

STATE_DEFS = {
    "S01_fund_high":       lambda d: d["funding_z"] >= 2.0,
    "S02_fund_low":        lambda d: d["funding_z"] <= -2.0,
    "S03_oi_surge":        lambda d: d["oi_z"] >= 2.0,
    "S04_fund_hi_oi_hi":   lambda d: (d["funding_z"] >= 2.0) & (d["oi_z"] >= 2.0),
    "S05_fund_lo_oi_hi":   lambda d: (d["funding_z"] <= -2.0) & (d["oi_z"] >= 2.0),
    "S06_compress_vol":    lambda d: (d["rv_2h_pctl"] <= 0.20) & (d["volume_2h_pctl"] >= 0.60),
    "S07_compress_oi":     lambda d: (d["rv_6h_pctl"] <= 0.20) & (d["oi_z"] >= 1.5),
    "S08_stall_oi":        lambda d: (d["trend_strength_2h_pctl"] <= 0.20) & (d["oi_z"] >= 2.0),
    "S09_stall_fund":      lambda d: (d["trend_strength_2h_pctl"] <= 0.20) & (d["funding_z"].abs() >= 2.0),
    "S10_thin_move":       lambda d: (d["volume_1h_pctl"] <= 0.20) & (d["abs_ret_30m_pctl"] >= 0.80),
}


def compute_states(df_5m: pd.DataFrame) -> pd.DataFrame:
    for name, fn in STATE_DEFS.items():
        try:
            df_5m[name] = fn(df_5m).astype(float)
        except Exception:
            df_5m[name] = 0.0
    return df_5m


# ---------------------------------------------------------------------------
# §6: Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0, centre - spread), min(1, centre + spread)


# ---------------------------------------------------------------------------
# §7: Permutation test (shuffle state labels within day)
# ---------------------------------------------------------------------------

def permutation_pvalue(big_move: np.ndarray, state: np.ndarray,
                       day_labels: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Compute permutation p-value by shuffling state labels within each day."""
    mask = ~np.isnan(big_move) & ~np.isnan(state)
    bm = big_move[mask].astype(bool)
    st = state[mask].astype(bool)
    days = day_labels[mask]

    ns = st.sum()
    if ns < 5:
        return 1.0

    observed = bm[st].mean()

    unique_days = np.unique(days)
    # Build day-indexed arrays for fast shuffling
    day_indices = {d: np.where(days == d)[0] for d in unique_days}
    day_ns = {d: st[day_indices[d]].sum() for d in unique_days}

    count_ge = 0
    for _ in range(n_perm):
        perm_mean_sum = 0.0
        perm_count = 0
        for d in unique_days:
            nd = day_ns[d]
            if nd == 0:
                continue
            idx = day_indices[d]
            perm_idx = rng.choice(idx, size=nd, replace=False)
            perm_mean_sum += bm[perm_idx].sum()
            perm_count += nd
        if perm_count > 0:
            perm_rate = perm_mean_sum / perm_count
            if perm_rate >= observed:
                count_ge += 1

    return (count_ge + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# §8: Volatility-matched baseline
# ---------------------------------------------------------------------------

def vol_matched_baseline(df_5m: pd.DataFrame, state_col: str, target_col: str,
                         n_controls: int, rng: np.random.Generator) -> float:
    """For each t where state=1, find n_controls matched times with same ATR quintile
    and hour-of-day bucket, then compute mean(big_move) at those controls."""
    mask_valid = ~df_5m[target_col].isna()
    st = df_5m[state_col].values == 1.0
    bm = df_5m[target_col].values
    aq = df_5m["atr_quintile"].values.astype(float)
    hod = df_5m["hod_bucket"].values

    # Build lookup: (atr_quintile, hod_bucket) -> list of valid non-state indices
    non_state_valid = mask_valid.values & ~st
    lookup = {}
    for i in range(len(df_5m)):
        if non_state_valid[i] and not np.isnan(aq[i]):
            key = (int(aq[i]), int(hod[i]))
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(i)

    control_hits = []
    state_indices = np.where(st & mask_valid.values & ~np.isnan(aq))[0]

    for idx in state_indices:
        key = (int(aq[idx]), int(hod[idx]))
        pool = lookup.get(key, [])
        if len(pool) < 3:
            continue
        chosen = rng.choice(pool, size=min(n_controls, len(pool)), replace=False)
        control_hits.extend(bm[chosen].tolist())

    if not control_hits:
        return np.nan
    return np.nanmean(control_hits)


# ---------------------------------------------------------------------------
# §9: Main analysis loop
# ---------------------------------------------------------------------------

def build_target_columns():
    """Return list of (target_col, horizon_label, definition_label)."""
    cols = []
    for h_label in HORIZONS:
        for k in ATR_K_VALUES:
            cols.append((f"big_A_k{k}_{h_label}", h_label, f"A_k{k}"))
        cols.append((f"big_B_{h_label}", h_label, f"B"))
    return cols


def analyze_symbol(sym: str, raw: dict, grid_1m: pd.DatetimeIndex):
    """Run full XS-6 analysis for one symbol. Returns (results, df_5m) or ([], None)."""
    t0 = time.monotonic()

    # Build 1m grid
    df_1m = build_sym_1m(sym, raw, grid_1m)

    # Check we have enough valid data
    valid_pct = 1 - df_1m["is_invalid"].mean()
    if valid_pct < 0.5:
        print(f"  {sym}: only {valid_pct:.0%} valid — skipping")
        return [], None

    # Compute features on 1m
    df_1m = compute_features(df_1m)

    # Sample to 5m grid
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]
    df_5m = df_1m.loc[grid_5m].copy()

    # Compute targets
    df_5m = compute_targets(df_5m, df_1m)

    # Compute states
    df_5m = compute_states(df_5m)

    # Day labels for permutation (integer day since epoch)
    df_5m["day_label"] = (df_5m.index - pd.Timestamp("2020-01-01", tz="UTC")).days

    # Week labels for stability (compute once)
    df_5m["_week"] = df_5m.index.isocalendar().week.values
    weeks_arr = df_5m["_week"].unique()

    # Split train/test with purge
    purge_start = TRAIN_END - pd.Timedelta(hours=PURGE_HOURS)
    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
    mask_train = df_5m.index <= purge_start
    mask_test = df_5m.index >= purge_end

    # Pre-build test subset (once)
    df_test = df_5m[mask_test].copy()

    target_cols = build_target_columns()
    state_names = list(STATE_DEFS.keys())
    rng = np.random.default_rng(SEED)

    results = []
    for state_name in state_names:
        for target_col, h_label, def_label in target_cols:
            # Full sample
            valid = ~df_5m[target_col].isna()
            bm = df_5m[target_col].values
            st = df_5m[state_name].values

            n_total = valid.sum()
            ns_full = int((st[valid] == 1.0).sum())
            if ns_full < 5:
                continue

            p0 = np.nanmean(bm[valid])
            ps_full = np.nanmean(bm[valid & (st == 1.0)])
            uplift_full = ps_full / max(p0, 1e-8)
            delta_full = ps_full - p0
            ci_lo, ci_hi = wilson_ci(int(bm[valid & (st == 1.0)].sum()), ns_full)

            # Permutation test (full sample)
            p_perm = permutation_pvalue(bm, st, df_5m["day_label"].values,
                                        N_PERMUTATION, rng)

            # Volatility-matched baseline (full sample)
            p_control = vol_matched_baseline(df_5m, state_name, target_col,
                                             N_MATCHED_CONTROLS, rng)
            uplift_matched = ps_full / max(p_control, 1e-8) if not np.isnan(p_control) else np.nan

            # Train split
            train_valid = valid & mask_train
            ns_train = int((st[train_valid] == 1.0).sum())
            if ns_train >= 3:
                p0_train = np.nanmean(bm[train_valid])
                ps_train = np.nanmean(bm[train_valid & (st == 1.0)])
                uplift_train = ps_train / max(p0_train, 1e-8)
            else:
                ps_train = np.nan
                uplift_train = np.nan

            # Test split
            test_valid = valid & mask_test
            ns_test = int((st[test_valid] == 1.0).sum())
            p0_test = np.nan
            ps_test = np.nan
            uplift_test = np.nan
            delta_test = np.nan
            p_control_test = np.nan
            uplift_matched_test = np.nan
            if ns_test >= 3:
                p0_test = np.nanmean(bm[test_valid])
                ps_test = np.nanmean(bm[test_valid & (st == 1.0)])
                uplift_test = ps_test / max(p0_test, 1e-8)
                delta_test = ps_test - p0_test

                # Vol-matched baseline on test only
                p_control_test = vol_matched_baseline(df_test, state_name, target_col,
                                                       N_MATCHED_CONTROLS, rng)
                uplift_matched_test = ps_test / max(p_control_test, 1e-8) if not np.isnan(p_control_test) else np.nan

            # Weekly stability (count weeks where state uplift > 1)
            weeks_positive = 0
            weeks_total = 0
            for w in weeks_arr:
                wm = df_5m["_week"].values == w
                wv = valid.values & wm
                wns = int((st[wv] == 1.0).sum())
                if wns < 2:
                    continue
                weeks_total += 1
                wp0 = np.nanmean(bm[wv])
                wps = np.nanmean(bm[wv & (st == 1.0)])
                if wp0 > 0 and wps / wp0 > 1.0:
                    weeks_positive += 1

            rate_s = ns_full / max(n_total, 1)

            results.append({
                "symbol": sym,
                "state": state_name,
                "horizon": h_label,
                "definition": def_label,
                "n_total": int(n_total),
                "nS": ns_full,
                "rateS": rate_s,
                "p0": p0,
                "pS": ps_full,
                "p_control": p_control,
                "uplift": uplift_full,
                "uplift_matched": uplift_matched,
                "delta": delta_full,
                "CI_low": ci_lo,
                "CI_high": ci_hi,
                "p_perm": p_perm,
                "nS_train": ns_train,
                "pS_train": ps_train,
                "train_uplift": uplift_train,
                "nS_test": ns_test,
                "pS_test": ps_test,
                "p0_test": p0_test,
                "test_uplift": uplift_test,
                "delta_test": delta_test,
                "p_control_test": p_control_test,
                "uplift_matched_test": uplift_matched_test,
                "weeks_positive": weeks_positive,
                "weeks_total": weeks_total,
            })

    elapsed = time.monotonic() - t0
    n_pass = sum(1 for r in results if r["nS_test"] >= MIN_NS_TEST and
                 r.get("uplift_matched_test", 0) >= MIN_UPLIFT_MATCHED)
    print(f"  {sym}: {len(results)} combos, {n_pass} preliminary PASS, {elapsed:.1f}s")
    return results, df_5m


# ---------------------------------------------------------------------------
# §10: FDR correction
# ---------------------------------------------------------------------------

def apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    pvals = df["p_perm"].values
    valid = ~np.isnan(pvals)
    q = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        _, q_vals, _, _ = multipletests(pvals[valid], alpha=0.10, method="fdr_bh")
        q[valid] = q_vals
    df["q_fdr"] = q
    return df


# ---------------------------------------------------------------------------
# §11: PASS flag
# ---------------------------------------------------------------------------

def compute_pass_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["flag_PASS"] = (
        (df["rateS"] >= MIN_RATE_S) &
        (df["nS_test"] >= MIN_NS_TEST) &
        (df["uplift_matched_test"] >= MIN_UPLIFT_MATCHED) &
        (df["delta_test"] >= MIN_DELTA_ABS) &
        (df["q_fdr"] <= MAX_Q_FDR)
    ).astype(int)

    # Soft: weekly stability
    df["weekly_stable"] = (df["weeks_positive"] >= MIN_WEEKLY_STABLE).astype(int)

    return df


# ---------------------------------------------------------------------------
# §12: Universe-level analysis (pool all symbols)
# ---------------------------------------------------------------------------

def universe_analysis(all_5m: dict, grid_1m: pd.DatetimeIndex) -> list[dict]:
    """Run uplift analysis on pooled universe data."""
    print("\n--- Universe-level analysis ---")
    target_cols = build_target_columns()
    state_names = list(STATE_DEFS.keys())
    rng = np.random.default_rng(SEED + 999)

    # Stack all symbol 5m frames
    frames = []
    for sym, df in all_5m.items():
        d = df.copy()
        d["symbol"] = sym
        frames.append(d)

    if not frames:
        return []

    pooled = pd.concat(frames)
    print(f"  Pooled universe: {len(pooled)} rows, {len(all_5m)} symbols")

    purge_start = TRAIN_END - pd.Timedelta(hours=PURGE_HOURS)
    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
    mask_test = pooled.index >= purge_end

    results = []
    for state_name in state_names:
        for target_col, h_label, def_label in target_cols:
            valid = ~pooled[target_col].isna()
            bm = pooled[target_col].values
            st = pooled[state_name].values

            n_total = valid.sum()
            ns = int((st[valid] == 1.0).sum())
            if ns < 10:
                continue

            p0 = np.nanmean(bm[valid])
            ps = np.nanmean(bm[valid & (st == 1.0)])
            uplift = ps / max(p0, 1e-8)
            delta = ps - p0

            # Test split
            test_valid = valid & mask_test
            ns_test = int((st[test_valid] == 1.0).sum())
            if ns_test >= 10:
                p0_test = np.nanmean(bm[test_valid])
                ps_test = np.nanmean(bm[test_valid & (st == 1.0)])
                uplift_test = ps_test / max(p0_test, 1e-8)
                delta_test = ps_test - p0_test
            else:
                ps_test = uplift_test = delta_test = p0_test = np.nan

            # Count unique symbols contributing
            sym_mask = valid & (st == 1.0)
            n_syms = pooled.loc[sym_mask, "symbol"].nunique() if sym_mask.sum() > 0 else 0

            results.append({
                "symbol": "_UNIVERSE_",
                "state": state_name,
                "horizon": h_label,
                "definition": def_label,
                "n_total": int(n_total),
                "nS": ns,
                "rateS": ns / max(n_total, 1),
                "p0": p0,
                "pS": ps,
                "p_control": np.nan,
                "uplift": uplift,
                "uplift_matched": np.nan,
                "delta": delta,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "p_perm": np.nan,
                "nS_train": np.nan,
                "pS_train": np.nan,
                "train_uplift": np.nan,
                "nS_test": ns_test,
                "pS_test": ps_test,
                "p0_test": p0_test,
                "test_uplift": uplift_test,
                "delta_test": delta_test,
                "p_control_test": np.nan,
                "uplift_matched_test": np.nan,
                "weeks_positive": np.nan,
                "weeks_total": np.nan,
                "n_syms_contributing": n_syms,
            })

    print(f"  Universe: {len(results)} combos computed")
    return results


# ---------------------------------------------------------------------------
# §13: Generate findings markdown
# ---------------------------------------------------------------------------

def generate_findings(df: pd.DataFrame, output_dir: Path):
    """Write xs6_top_states.md with top candidates."""
    passed = df[df["flag_PASS"] == 1].sort_values("uplift_matched_test", ascending=False)

    lines = [
        "# XS-6 — Top States with Big Move Uplift",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"## Summary",
        "",
        f"- Total combos tested: {len(df)}",
        f"- Combos passing all hard filters: {len(passed)}",
        f"- Unique states passing: {passed['state'].nunique() if len(passed) > 0 else 0}",
        f"- Unique symbols: {passed['symbol'].nunique() if len(passed) > 0 else 0}",
        "",
    ]

    if len(passed) == 0:
        lines.append("**No candidates passed all hard filters.**")
        lines.append("")
        # Show best near-misses
        near = df.sort_values("uplift_matched_test", ascending=False).head(20)
        lines.append("## Near-misses (top 20 by uplift_matched_test)")
        lines.append("")
        lines.append("| symbol | state | horizon | def | nS_test | uplift_m_test | delta_test | q_fdr | weekly |")
        lines.append("|--------|-------|---------|-----|---------|---------------|------------|-------|--------|")
        for _, r in near.iterrows():
            lines.append(
                f"| {r['symbol']} | {r['state']} | {r['horizon']} | {r['definition']} | "
                f"{r.get('nS_test', 0):.0f} | {r.get('uplift_matched_test', 0):.2f} | "
                f"{r.get('delta_test', 0):.3f} | {r.get('q_fdr', 1):.3f} | "
                f"{r.get('weeks_positive', 0):.0f}/{r.get('weeks_total', 0):.0f} |"
            )
    else:
        top = passed.head(30)
        lines.append("## Top 30 PASS candidates (by uplift_matched_test)")
        lines.append("")
        lines.append("| symbol | state | horizon | def | nS_test | pS_test | p0_test | uplift_m_test | delta_test | q_fdr | weekly |")
        lines.append("|--------|-------|---------|-----|---------|---------|---------|---------------|------------|-------|--------|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['symbol']} | {r['state']} | {r['horizon']} | {r['definition']} | "
                f"{r['nS_test']:.0f} | {r['pS_test']:.3f} | {r['p0_test']:.3f} | "
                f"{r['uplift_matched_test']:.2f} | {r['delta_test']:.3f} | "
                f"{r['q_fdr']:.3f} | {r['weeks_positive']:.0f}/{r['weeks_total']:.0f} |"
            )

    lines.append("")

    # State frequency summary
    lines.append("## State Frequency Summary (full sample)")
    lines.append("")
    sym_level = df[df["symbol"] != "_UNIVERSE_"]
    if len(sym_level) > 0:
        state_freq = sym_level.groupby("state")["rateS"].mean()
        lines.append("| state | mean rateS | interpretation |")
        lines.append("|-------|-----------|----------------|")
        interp = {
            "S01_fund_high": "Funding z >= +2",
            "S02_fund_low": "Funding z <= -2",
            "S03_oi_surge": "OI change z >= +2",
            "S04_fund_hi_oi_hi": "High funding + OI surge",
            "S05_fund_lo_oi_hi": "Low funding + OI surge",
            "S06_compress_vol": "Low vol + high volume",
            "S07_compress_oi": "Low vol + OI build",
            "S08_stall_oi": "Trend stall + OI surge",
            "S09_stall_fund": "Trend stall + extreme funding",
            "S10_thin_move": "Thin liquidity + sharp move",
        }
        for st in sorted(state_freq.index):
            lines.append(f"| {st} | {state_freq[st]:.3f} | {interp.get(st, '')} |")

    lines.append("")
    lines.append("## PASS Criteria Used")
    lines.append("")
    lines.append(f"- rateS >= {MIN_RATE_S}")
    lines.append(f"- nS_test >= {MIN_NS_TEST}")
    lines.append(f"- uplift_matched_test >= {MIN_UPLIFT_MATCHED}")
    lines.append(f"- delta_test >= {MIN_DELTA_ABS}")
    lines.append(f"- q_fdr < {MAX_Q_FDR}")
    lines.append(f"- Weekly stability: uplift>1 in >= {MIN_WEEKLY_STABLE}/N weeks (soft)")
    lines.append("")

    md_path = output_dir / "xs6_top_states.md"
    md_path.write_text("\n".join(lines))
    print(f"\nFindings written to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.monotonic()
    print("=" * 70)
    print("XS-6 — Extreme Move Probability Model (Big Move Uplift)")
    print("=" * 70)
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Train: ≤ {TRAIN_END.date()}, Test: ≥ {TEST_START.date()}, Purge: ±{PURGE_HOURS}h")
    print(f"Horizons: {list(HORIZONS.keys())}")
    print(f"ATR k: {ATR_K_VALUES}, Raw BP: {RAW_BP_THRESHOLDS}")
    print(f"States: {len(STATE_DEFS)}")
    print(f"Permutations: {N_PERMUTATION}")
    print()

    # Discover symbols
    symbols = discover_symbols()
    print(f"Discovered {len(symbols)} symbols with sufficient data")

    # Build 1m grid
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"1m grid: {len(grid_1m)} bars")
    print()

    # Process each symbol
    all_results = []
    all_5m = {}  # for universe analysis
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Loading {sym}...")
        raw = load_symbol(sym)
        sym_results, df_5m = analyze_symbol(sym, raw, grid_1m)
        all_results.extend(sym_results)
        if df_5m is not None:
            all_5m[sym] = df_5m

    print(f"\n--- Symbol-level: {len(all_results)} total combos ---")

    # Universe-level
    uni_results = universe_analysis(all_5m, grid_1m)
    all_results.extend(uni_results)

    # Build results DataFrame
    df_results = pd.DataFrame(all_results)
    if len(df_results) == 0:
        print("No results! Check data.")
        return

    # Apply FDR correction
    print("\nApplying BH FDR correction...")
    df_results = apply_fdr(df_results)

    # Compute PASS flag
    df_results = compute_pass_flag(df_results)

    n_pass = df_results["flag_PASS"].sum()
    print(f"Total PASS: {n_pass} / {len(df_results)}")

    # Save CSV
    csv_path = OUTPUT_DIR / "xs6_uplift.csv"
    df_results.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Results saved to {csv_path}")

    # Generate findings
    generate_findings(df_results, OUTPUT_DIR)

    elapsed = time.monotonic() - t_start
    print(f"\nXS-6 done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    sym_level = df_results[df_results["symbol"] != "_UNIVERSE_"]
    if len(sym_level) > 0:
        for st in sorted(STATE_DEFS.keys()):
            sub = sym_level[sym_level["state"] == st]
            if len(sub) > 0:
                best = sub.sort_values("uplift_matched_test", ascending=False).iloc[0]
                print(f"  {st}: best uplift_m_test={best.get('uplift_matched_test', 0):.2f}, "
                      f"delta_test={best.get('delta_test', 0):.3f}, "
                      f"nS_test={best.get('nS_test', 0):.0f}, "
                      f"sym={best['symbol']}, h={best['horizon']}, def={best['definition']}")


if __name__ == "__main__":
    main()
