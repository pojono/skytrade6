#!/usr/bin/env python3
"""
Regime-Only Instability Research (No Shock)

Predicts next 60-minute outcome (direction and/or range) from OI/funding/price/vol regimes.
Uses per-symbol quantile-based regime definitions — no threshold optimization.

Implements full spec: §0–§12.

Output:
  flow_research/output/regime/regime_dataset.parquet
  flow_research/output/regime/report_baseline.csv
  flow_research/output/regime/report_regimes.csv
  flow_research/output/regime/report_weekly.csv
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "1000BONKUSDT",
    "ARBUSDT",
    "APTUSDT",
    "ATOMUSDT",
    "AIXBTUSDT",
    "1000RATSUSDT",
    "ARCUSDT",
    "1000TURBOUSDT",
]

# §1: Period
START_TS = pd.Timestamp("2026-01-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# §4: Target thresholds (bp)
DIR_THRESHOLDS_BP = [20, 30, 50]

# §8: Statistical params
N_BOOTSTRAP = 2000
PERMUTATION_N = 2000
FDR_ALPHA = 0.10

# §9: Success criteria
MIN_DIRECTIONAL_MEDIAN_BP = 20
MIN_WR_UPLIFT = 0.05
MIN_RANGE_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# §2: Data loading (1-minute resolution)
# ---------------------------------------------------------------------------


def load_symbol_1m(symbol: str) -> dict[str, pd.DataFrame]:
    """Load all per-day CSV files for a symbol into DataFrames."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        print(f"  SKIP {symbol}: no data directory")
        return {}

    dfs = {}

    # --- Kline 1m (price + volume) ---
    kline_files = sorted(
        f for f in sym_dir.glob("*_kline_1m.csv")
        if "mark_price" not in f.name and "premium_index" not in f.name
    )
    if kline_files:
        frames = [pd.read_csv(f) for f in kline_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            kline = pd.concat(frames, ignore_index=True)
            kline["ts"] = pd.to_datetime(kline["startTime"].astype(int), unit="ms", utc=True)
            kline = kline.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                if col in kline.columns:
                    kline[col] = kline[col].astype(float)
            dfs["kline"] = kline

    # --- Mark Price Kline 1m ---
    mark_files = sorted(sym_dir.glob("*_mark_price_kline_1m.csv"))
    if mark_files:
        frames = [pd.read_csv(f) for f in mark_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            mark = pd.concat(frames, ignore_index=True)
            mark["ts"] = pd.to_datetime(mark["startTime"].astype(int), unit="ms", utc=True)
            mark = mark.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            mark["mark_close"] = mark["close"].astype(float)
            dfs["mark"] = mark[["ts", "mark_close"]]

    # --- Open Interest 5min ---
    oi_files = sorted(sym_dir.glob("*_open_interest_5min.csv"))
    if oi_files:
        frames = [pd.read_csv(f) for f in oi_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            oi = pd.concat(frames, ignore_index=True)
            oi["ts"] = pd.to_datetime(oi["timestamp"].astype(int), unit="ms", utc=True)
            oi["openInterest"] = oi["openInterest"].astype(float)
            oi = oi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["oi"] = oi[["ts", "openInterest"]]

    # --- Funding Rate ---
    fr_files = sorted(sym_dir.glob("*_funding_rate.csv"))
    if fr_files:
        frames = [pd.read_csv(f) for f in fr_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            fr = pd.concat(frames, ignore_index=True)
            fr["ts"] = pd.to_datetime(fr["timestamp"].astype(int), unit="ms", utc=True)
            fr["fundingRate"] = fr["fundingRate"].astype(float)
            fr = fr.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["fr"] = fr[["ts", "fundingRate"]]

    # --- Long/Short Ratio 5min ---
    ls_files = sorted(sym_dir.glob("*_long_short_ratio_5min.csv"))
    if ls_files:
        frames = [pd.read_csv(f) for f in ls_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            ls = pd.concat(frames, ignore_index=True)
            ls["ts"] = pd.to_datetime(ls["timestamp"].astype(int), unit="ms", utc=True)
            ls["buyRatio"] = ls["buyRatio"].astype(float)
            ls["sellRatio"] = ls["sellRatio"].astype(float)
            ls = ls.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["ls"] = ls[["ts", "buyRatio", "sellRatio"]]

    # --- Premium Index Kline 1m ---
    pi_files = sorted(sym_dir.glob("*_premium_index_kline_1m.csv"))
    if pi_files:
        frames = [pd.read_csv(f) for f in pi_files]
        frames = [f for f in frames if len(f) > 0]
        if frames:
            pi = pd.concat(frames, ignore_index=True)
            pi["ts"] = pd.to_datetime(pi["startTime"].astype(int), unit="ms", utc=True)
            pi = pi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            pi["index_close"] = pi["close"].astype(float)
            dfs["premium"] = pi[["ts", "index_close"]]

    return dfs


# ---------------------------------------------------------------------------
# §3: Build 5-minute grid from 1m data
# ---------------------------------------------------------------------------


def build_1m_grid(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a 1-minute price grid (needed for feature computation windows)."""
    kline = data.get("kline")
    if kline is None or len(kline) == 0:
        return pd.DataFrame()

    df = kline[["ts", "open", "high", "low", "close", "volume", "turnover"]].copy()
    df = df.set_index("ts").sort_index()

    # Merge mark price
    if "mark" in data:
        mark = data["mark"].set_index("ts")["mark_close"]
        df = df.join(mark, how="left")
        df["mark_close"] = df["mark_close"].ffill()
    else:
        df["mark_close"] = df["close"]

    # Merge OI (5min → ffill to 1m)
    if "oi" in data:
        oi = data["oi"].set_index("ts")["openInterest"]
        df = df.join(oi, how="left")
        df["openInterest"] = df["openInterest"].ffill()

    # Merge FR (sparse → piecewise constant ffill)
    if "fr" in data:
        fr = data["fr"].set_index("ts")["fundingRate"]
        df = df.join(fr, how="left")
        df["fundingRate"] = df["fundingRate"].ffill()

    # Merge LS ratio (5min → ffill to 1m)
    if "ls" in data:
        ls = data["ls"].set_index("ts")[["buyRatio", "sellRatio"]]
        df = df.join(ls, how="left")
        df["buyRatio"] = df["buyRatio"].ffill()
        df["sellRatio"] = df["sellRatio"].ffill()

    # Merge premium/index price (1m)
    if "premium" in data:
        idx = data["premium"].set_index("ts")["index_close"]
        df = df.join(idx, how="left")
        df["index_close"] = df["index_close"].ffill()

    df = df.reset_index()
    # Filter to study period
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)].reset_index(drop=True)
    return df


def sample_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """§3: Sample every 5 minutes (minute % 5 == 0). Features computed on 1m data."""
    mask = df_1m["ts"].dt.minute % 5 == 0
    return df_1m[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# §4: Target construction
# ---------------------------------------------------------------------------


def add_targets(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute forward 60-minute targets from 1m data. Strictly causal: uses (t, t+60m]."""
    df = df_5m.copy()

    # Build lookup arrays from 1m data (int64 ns timestamps for fast searchsorted)
    ts_1m_ns = df_1m["ts"].values.astype("int64")
    close_1m = df_1m["close"].values.astype(np.float64)
    high_1m = df_1m["high"].values.astype(np.float64)
    low_1m = df_1m["low"].values.astype(np.float64)
    logret_1m = np.diff(np.log(close_1m)) * 10000  # 1m log returns in bp

    n = len(df)
    ret_60m = np.full(n, np.nan)
    range_60m = np.full(n, np.nan)
    rv_60m = np.full(n, np.nan)
    high_60m_arr = np.full(n, np.nan)
    low_60m_arr = np.full(n, np.nan)
    price_t60 = np.full(n, np.nan)

    ts_5m_ns = df["ts"].values.astype("int64")
    delta_60m_ns = int(60 * 60 * 1e9)

    print("    Computing 60m forward targets (vectorized)...")
    # Use searchsorted for O(N log M) instead of O(N*M)
    # For each 5m point t, find 1m bars in (t, t+60m]
    idx_start = np.searchsorted(ts_1m_ns, ts_5m_ns, side="right")  # first bar > t
    idx_end = np.searchsorted(ts_1m_ns, ts_5m_ns + delta_60m_ns, side="right")  # first bar > t+60m

    for i in range(n):
        i_s = idx_start[i]
        i_e = idx_end[i]
        n_bars = i_e - i_s

        if n_bars < 30:
            continue

        # Price at t: last close <= t
        p_idx = i_s - 1
        if p_idx < 0:
            continue
        p_now = close_1m[p_idx]
        if p_now <= 0 or np.isnan(p_now):
            continue

        sl = slice(i_s, i_e)
        p_end = close_1m[i_e - 1]
        h = np.max(high_1m[sl])
        lo = np.min(low_1m[sl])

        ret_60m[i] = np.log(p_end / p_now) * 10000
        high_60m_arr[i] = h
        low_60m_arr[i] = lo
        price_t60[i] = p_end

        if h > 0 and lo > 0:
            range_60m[i] = np.log(h / lo) * 10000

        # RV: std of 1m returns within forward window
        ret_s = max(i_s - 1, 0)
        ret_e = min(i_e - 1, len(logret_1m))
        if ret_e - ret_s >= 10:
            rv_60m[i] = np.std(logret_1m[ret_s:ret_e])

        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{n} targets computed")

    df["price_t"] = df["close"].values
    df["price_t60"] = price_t60
    df["high_60m"] = high_60m_arr
    df["low_60m"] = low_60m_arr
    df["ret_60m_bp"] = ret_60m
    df["range_60m_bp"] = range_60m
    df["rv_60m_bp"] = rv_60m

    # Directional targets
    df["y_dir"] = np.sign(df["ret_60m_bp"])
    for thr in DIR_THRESHOLDS_BP:
        df[f"y_up{thr}"] = (df["ret_60m_bp"] > thr).astype(float)
        df[f"y_dn{thr}"] = (df["ret_60m_bp"] < -thr).astype(float)
        # Set to NaN where ret_60m is NaN
        df.loc[df["ret_60m_bp"].isna(), f"y_up{thr}"] = np.nan
        df.loc[df["ret_60m_bp"].isna(), f"y_dn{thr}"] = np.nan

    df.loc[df["ret_60m_bp"].isna(), "y_dir"] = np.nan

    return df


# ---------------------------------------------------------------------------
# §5: Feature engineering (strictly causal)
# ---------------------------------------------------------------------------


def engineer_features(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from §5. All use data ≤ t only."""
    df = df_5m.copy()

    price = df["close"].values
    n = len(df)

    # -- §5.1: OI features --
    if "openInterest" in df.columns:
        oi = df["openInterest"]
        for w, lbl in [(3, "15"), (6, "30"), (12, "60"), (48, "240")]:
            df[f"oi_chg_{lbl}"] = (oi - oi.shift(w)) / oi.shift(w).replace(0, np.nan)
        # z-score of oi_chg_60 over rolling 7d (7*24*12 = 2016 five-min bars)
        roll_7d = 7 * 24 * 12
        oi_chg_60 = df["oi_chg_60"]
        df["oi_z_7d"] = (oi_chg_60 - oi_chg_60.rolling(roll_7d, min_periods=288).mean()) / \
                         oi_chg_60.rolling(roll_7d, min_periods=288).std().replace(0, np.nan)

    # -- §5.2: Funding features --
    if "fundingRate" in df.columns:
        fr = df["fundingRate"] * 10000  # to bps
        df["funding_1m_bp"] = fr  # piecewise constant
        roll_7d = 7 * 24 * 12
        df["funding_z_7d"] = (fr - fr.rolling(roll_7d, min_periods=288).mean()) / \
                              fr.rolling(roll_7d, min_periods=288).std().replace(0, np.nan)
        df["funding_sign"] = np.sign(fr)
        df["funding_extreme_1s"] = (df["funding_z_7d"].abs() > 1).astype(int)
        df["funding_extreme_2s"] = (df["funding_z_7d"].abs() > 2).astype(int)
        df["funding_extreme_3s"] = (df["funding_z_7d"].abs() > 3).astype(int)

    # -- §5.3: Price extension / trend state --
    # Using 5-min bars: 12 bars = 60m, 48 bars = 240m
    df["ret_past_60"] = np.log(df["close"] / df["close"].shift(12)) * 10000
    df["ret_past_240"] = np.log(df["close"] / df["close"].shift(48)) * 10000

    # -- §5.4: Volatility regime --
    # Vectorized: use searchsorted on int64 ns timestamps
    ts_1m_ns = df_1m["ts"].values.astype("int64")
    close_1m = df_1m["close"].values.astype(np.float64)
    logret_1m = np.diff(np.log(close_1m)) * 10000
    ts_ret_1m_ns = ts_1m_ns[1:]
    high_1m = df_1m["high"].values.astype(np.float64)
    low_1m = df_1m["low"].values.astype(np.float64)

    rv_past_60 = np.full(n, np.nan)
    rv_past_240 = np.full(n, np.nan)
    range_past_60 = np.full(n, np.nan)

    ts_5m_ns = df["ts"].values.astype("int64")
    delta_60m_ns = int(60 * 60 * 1e9)
    delta_240m_ns = int(240 * 60 * 1e9)

    print("    Computing backward volatility features (vectorized)...")
    # Precompute searchsorted boundaries for all 5m points
    # For returns: ts_ret_1m_ns, window [t-Xm, t]
    end_idx_ret = np.searchsorted(ts_ret_1m_ns, ts_5m_ns, side="right")  # first > t
    start_idx_ret_60 = np.searchsorted(ts_ret_1m_ns, ts_5m_ns - delta_60m_ns, side="left")
    start_idx_ret_240 = np.searchsorted(ts_ret_1m_ns, ts_5m_ns - delta_240m_ns, side="left")
    # For bars: ts_1m_ns, window [t-60m, t]
    end_idx_bar = np.searchsorted(ts_1m_ns, ts_5m_ns, side="right")
    start_idx_bar_60 = np.searchsorted(ts_1m_ns, ts_5m_ns - delta_60m_ns, side="left")

    for i in range(n):
        # rv_past_60
        s60 = start_idx_ret_60[i]
        e = end_idx_ret[i]
        if e - s60 >= 30:
            rv_past_60[i] = np.std(logret_1m[s60:e])

        # rv_past_240
        s240 = start_idx_ret_240[i]
        if e - s240 >= 120:
            rv_past_240[i] = np.std(logret_1m[s240:e])

        # range_past_60
        bs = start_idx_bar_60[i]
        be = end_idx_bar[i]
        if be - bs >= 30:
            h = np.max(high_1m[bs:be])
            lo = np.min(low_1m[bs:be])
            if h > 0 and lo > 0:
                range_past_60[i] = np.log(h / lo) * 10000

        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{n} vol features computed")

    df["rv_past_60"] = rv_past_60
    df["rv_past_240"] = rv_past_240
    df["rv_ratio"] = np.where(rv_past_240 > 0, rv_past_60 / rv_past_240, np.nan)
    df["range_past_60"] = range_past_60

    # Trend strength = |ret_past_240| / rv_past_240
    df["trend_strength"] = np.where(
        df["rv_past_240"] > 0,
        df["ret_past_240"].abs() / df["rv_past_240"],
        np.nan,
    )

    # VWAP distance (from 1m volume)
    # Simple: rolling 60m VWAP = sum(close*volume) / sum(volume) on 5min bars
    if "turnover" in df.columns and "volume" in df.columns:
        df["vwap_60"] = df["turnover"].rolling(12, min_periods=6).sum() / \
                         df["volume"].rolling(12, min_periods=6).sum().replace(0, np.nan)
        df["dist_vwap_60_bp"] = (df["close"] / df["vwap_60"] - 1) * 10000

    # Compression flag: per-symbol quantile (computed later in regime section)

    # -- §5.5: Volume / activity --
    if "turnover" in df.columns:
        df["vol_notional_60"] = df["turnover"].rolling(12, min_periods=6).sum()
        roll_7d = 7 * 24 * 12
        vol_60 = df["vol_notional_60"]
        df["vol_z_7d"] = (vol_60 - vol_60.rolling(roll_7d, min_periods=288).mean()) / \
                          vol_60.rolling(roll_7d, min_periods=288).std().replace(0, np.nan)
        df["vol_spike"] = (df["vol_z_7d"] > 2).astype(int)

    # -- §5.6: Basis --
    if "mark_close" in df.columns and "index_close" in df.columns:
        df["basis_bp"] = (df["mark_close"] / df["index_close"].replace(0, np.nan) - 1) * 10000
        roll_7d = 7 * 24 * 12
        basis = df["basis_bp"]
        df["basis_z_7d"] = (basis - basis.rolling(roll_7d, min_periods=288).mean()) / \
                            basis.rolling(roll_7d, min_periods=288).std().replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# §6: Regime definitions (per-symbol quantiles)
# ---------------------------------------------------------------------------


def define_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Define regime flags using per-symbol quantiles. No threshold optimization."""
    df = df.copy()

    # §6.1: REG_OI_FUND — Instability (leveraged buildup)
    has_oi = "oi_chg_60" in df.columns
    has_fund = "funding_z_7d" in df.columns
    has_ret = "ret_past_60" in df.columns

    if has_oi and has_fund and has_ret:
        oi_p90 = df["oi_chg_60"].quantile(0.90)
        ret_p70 = df["ret_past_60"].abs().quantile(0.70)

        cond_oi = df["oi_chg_60"] >= oi_p90
        cond_fund = df["funding_z_7d"].abs() >= 1.0
        cond_ret = df["ret_past_60"].abs() >= ret_p70

        df["REG_OI_FUND"] = (cond_oi & cond_fund & cond_ret).astype(int)

        # Signed version for symmetry test
        df["REG_OI_FUND_LONG"] = (df["REG_OI_FUND"] == 1) & (df["funding_sign"] > 0)
        df["REG_OI_FUND_SHORT"] = (df["REG_OI_FUND"] == 1) & (df["funding_sign"] < 0)
        df["REG_OI_FUND_LONG"] = df["REG_OI_FUND_LONG"].astype(int)
        df["REG_OI_FUND_SHORT"] = df["REG_OI_FUND_SHORT"].astype(int)
    else:
        df["REG_OI_FUND"] = 0
        df["REG_OI_FUND_LONG"] = 0
        df["REG_OI_FUND_SHORT"] = 0

    # §6.2: REG_COMPRESSION — Compression → Expansion candidate
    has_rv = "rv_ratio" in df.columns

    if has_rv:
        rv_p20 = df["rv_ratio"].quantile(0.20)
        cond_comp = df["rv_ratio"] <= rv_p20

        if has_oi:
            oi_p70 = df["oi_chg_60"].quantile(0.70)
            cond_oi_comp = df["oi_chg_60"] >= oi_p70
            df["REG_COMPRESSION"] = (cond_comp & cond_oi_comp).astype(int)
        else:
            df["REG_COMPRESSION"] = cond_comp.astype(int)
    else:
        df["REG_COMPRESSION"] = 0

    # §6.3: REG_EXHAUST — Trend exhaustion candidate
    has_trend = "trend_strength" in df.columns
    has_fund_ext = "funding_extreme_2s" in df.columns

    if has_trend and has_fund_ext:
        ts_p80 = df["trend_strength"].quantile(0.80)
        cond_trend = df["trend_strength"] >= ts_p80
        cond_fext = df["funding_extreme_2s"] == 1
        df["REG_EXHAUST"] = (cond_trend & cond_fext).astype(int)
    else:
        df["REG_EXHAUST"] = 0

    return df


# ---------------------------------------------------------------------------
# §7: Analysis — baselines, conditional effects, symmetry
# ---------------------------------------------------------------------------


def compute_baseline(df: pd.DataFrame, symbol: str) -> dict:
    """§7.1: Unconditional distribution of targets."""
    valid = df.dropna(subset=["ret_60m_bp", "range_60m_bp"])
    n = len(valid)
    if n == 0:
        return {}

    return {
        "symbol": symbol,
        "n": n,
        "median_ret_60m": valid["ret_60m_bp"].median(),
        "mean_ret_60m": valid["ret_60m_bp"].mean(),
        "std_ret_60m": valid["ret_60m_bp"].std(),
        "WR": (valid["ret_60m_bp"] > 0).mean(),
        "median_range_60m": valid["range_60m_bp"].median(),
        "mean_range_60m": valid["range_60m_bp"].mean(),
        "median_rv_60m": valid["rv_60m_bp"].median() if "rv_60m_bp" in valid.columns else np.nan,
        "p25_ret": valid["ret_60m_bp"].quantile(0.25),
        "p75_ret": valid["ret_60m_bp"].quantile(0.75),
    }


def bootstrap_ci(arr, n_boot=N_BOOTSTRAP, func=np.nanmedian, ci=(0.05, 0.95)):
    """§8.1: Bootstrap confidence interval."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return np.nan, np.nan
    rng = np.random.RandomState(42)
    boot_stats = np.array([func(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    return np.percentile(boot_stats, ci[0] * 100), np.percentile(boot_stats, ci[1] * 100)


def permutation_pvalue(regime_vals, all_vals, func=np.nanmedian, n_perm=PERMUTATION_N):
    """§8.2: Permutation test — compare observed statistic to null distribution."""
    regime_vals = regime_vals[~np.isnan(regime_vals)]
    all_vals = all_vals[~np.isnan(all_vals)]
    n_regime = len(regime_vals)
    if n_regime < 5 or len(all_vals) < 20:
        return np.nan

    observed = func(regime_vals)
    rng = np.random.RandomState(42)

    count_extreme = 0
    for _ in range(n_perm):
        perm_sample = rng.choice(all_vals, size=n_regime, replace=False)
        if abs(func(perm_sample)) >= abs(observed):
            count_extreme += 1

    return (count_extreme + 1) / (n_perm + 1)  # +1 to avoid p=0


def analyze_regime(df: pd.DataFrame, regime_col: str, symbol: str, baseline: dict) -> dict:
    """§7.2: Conditional effect for a single regime."""
    valid = df.dropna(subset=["ret_60m_bp", "range_60m_bp"])
    regime_mask = valid[regime_col] == 1
    reg = valid[regime_mask]
    n = len(reg)

    if n < 10:
        return None

    all_ret = valid["ret_60m_bp"].values
    all_range = valid["range_60m_bp"].values
    reg_ret = reg["ret_60m_bp"].values
    reg_range = reg["range_60m_bp"].values
    reg_rv = reg["rv_60m_bp"].values if "rv_60m_bp" in reg.columns else np.array([])

    # Metrics
    med_ret = np.nanmedian(reg_ret)
    mean_ret = np.nanmean(reg_ret)
    wr = (reg_ret > 0).mean()
    med_range = np.nanmedian(reg_range)
    med_rv = np.nanmedian(reg_rv) if len(reg_rv) > 0 else np.nan

    # Uplifts
    unc_wr = baseline.get("WR", 0.5)
    unc_med_range = baseline.get("median_range_60m", 1)

    uplift_wr = wr - unc_wr
    uplift_range = med_range / unc_med_range if unc_med_range > 0 else np.nan

    # Bootstrap CI on median ret
    ci_lo, ci_hi = bootstrap_ci(reg_ret)

    # Permutation p-value on median ret
    p_perm_ret = permutation_pvalue(reg_ret, all_ret)
    p_perm_range = permutation_pvalue(reg_range, all_range, func=np.nanmedian)

    return {
        "symbol": symbol,
        "regime": regime_col,
        "n": n,
        "pct_of_total": n / len(valid) * 100,
        "median_ret_60m": med_ret,
        "mean_ret_60m": mean_ret,
        "WR": wr,
        "uplift_WR": uplift_wr,
        "median_range_60m": med_range,
        "uplift_range": uplift_range,
        "median_rv_60m": med_rv,
        "CI_lo_ret": ci_lo,
        "CI_hi_ret": ci_hi,
        "p_perm_ret": p_perm_ret,
        "p_perm_range": p_perm_range,
    }


def weekly_stability(df: pd.DataFrame, regime_col: str, symbol: str) -> list[dict]:
    """§10: Weekly breakdown of regime performance."""
    valid = df.dropna(subset=["ret_60m_bp", "range_60m_bp"]).copy()
    valid["week"] = valid["ts"].dt.isocalendar().week.astype(int)
    valid["year"] = valid["ts"].dt.year

    rows = []
    for (yr, wk), grp in valid.groupby(["year", "week"]):
        reg = grp[grp[regime_col] == 1]
        n = len(reg)
        if n < 3:
            continue
        rows.append({
            "symbol": symbol,
            "regime": regime_col,
            "year": yr,
            "week": wk,
            "n": n,
            "median_ret_60m": reg["ret_60m_bp"].median(),
            "mean_ret_60m": reg["ret_60m_bp"].mean(),
            "WR": (reg["ret_60m_bp"] > 0).mean(),
            "median_range_60m": reg["range_60m_bp"].median(),
        })

    return rows


# ---------------------------------------------------------------------------
# §8.3: FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------


def apply_fdr(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """Return BH-adjusted q-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])

    valid_mask = ~np.isnan(pvals)
    valid_p = pvals[valid_mask]
    m = len(valid_p)
    if m == 0:
        return np.full(n, np.nan)

    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    q = np.zeros(m)
    q[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        rank = i + 1
        q[i] = min(q[i + 1], sorted_p[i] * m / rank)

    # Unsort
    q_unsorted = np.zeros(m)
    q_unsorted[sorted_idx] = q

    result = np.full(n, np.nan)
    result[valid_mask] = q_unsorted
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("REGIME-ONLY INSTABILITY RESEARCH (No Shock)")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Period: {START_TS.date()} → {END_TS.date()}")
    print("=" * 80)

    all_datasets = []
    all_baselines = []
    all_regime_results = []
    all_weekly = []

    regime_cols = [
        "REG_OI_FUND", "REG_OI_FUND_LONG", "REG_OI_FUND_SHORT",
        "REG_COMPRESSION", "REG_EXHAUST",
    ]

    for sym_i, sym in enumerate(SYMBOLS, 1):
        sym_t0 = time.monotonic()
        print(f"\n{'─'*70}")
        print(f"[{sym_i}/{len(SYMBOLS)}] {sym}")
        print(f"{'─'*70}")

        # Load data
        data = load_symbol_1m(sym)
        if not data or "kline" not in data:
            print(f"  SKIP: no kline data")
            continue

        for k, v in data.items():
            print(f"  {k}: {len(v):,} rows")

        # Build 1m grid
        df_1m = build_1m_grid(data)
        if len(df_1m) == 0:
            print(f"  SKIP: empty 1m grid")
            continue
        print(f"  1m grid: {len(df_1m):,} rows ({df_1m['ts'].min()} → {df_1m['ts'].max()})")

        # Sample 5min
        df_5m = sample_5min(df_1m)
        print(f"  5m samples: {len(df_5m):,}")

        # §5: Features
        print("  Engineering features...")
        df_5m = engineer_features(df_5m, df_1m)

        # §4: Targets
        print("  Computing targets...")
        df_5m = add_targets(df_5m, df_1m)
        valid_count = df_5m["ret_60m_bp"].notna().sum()
        print(f"  Valid target rows: {valid_count:,} / {len(df_5m):,}")

        # §6: Regimes
        print("  Defining regimes...")
        df_5m = define_regimes(df_5m)
        for rc in regime_cols:
            n_active = (df_5m[rc] == 1).sum()
            pct = n_active / len(df_5m) * 100 if len(df_5m) > 0 else 0
            print(f"    {rc}: {n_active} ({pct:.1f}%)")

        df_5m["symbol"] = sym

        # §7.1: Baseline
        baseline = compute_baseline(df_5m, sym)
        if baseline:
            all_baselines.append(baseline)
            print(f"  Baseline: WR={baseline['WR']:.3f}, med_ret={baseline['median_ret_60m']:.1f}bp, "
                  f"med_range={baseline['median_range_60m']:.1f}bp")

        # §7.2 + §8: Conditional analysis with statistics
        for rc in regime_cols:
            result = analyze_regime(df_5m, rc, sym, baseline)
            if result:
                all_regime_results.append(result)

            # §10: Weekly stability
            weekly = weekly_stability(df_5m, rc, sym)
            all_weekly.extend(weekly)

        all_datasets.append(df_5m)

        sym_elapsed = time.monotonic() - sym_t0
        print(f"  {sym} done in {sym_elapsed:.1f}s")

    # ===================================================================
    # Cross-symbol results
    # ===================================================================
    if not all_baselines:
        print("\nNo data loaded. Exiting.")
        return

    print(f"\n{'='*80}")
    print("RESULTS")
    print("=" * 80)

    # §11.1: Dataset
    dataset = pd.concat(all_datasets, ignore_index=True)
    dataset.to_parquet(OUTPUT_DIR / "regime_dataset.parquet", index=False)
    print(f"\nDataset: {len(dataset):,} rows → {OUTPUT_DIR / 'regime_dataset.parquet'}")

    # §11.2.1: Baseline report
    baseline_df = pd.DataFrame(all_baselines)
    baseline_df.to_csv(OUTPUT_DIR / "report_baseline.csv", index=False)

    print(f"\n{'─'*70}")
    print("BASELINES (unconditional)")
    print(f"{'─'*70}")
    print(f"{'Symbol':<18} {'N':>7} {'WR':>6} {'Med_ret':>9} {'Mean_ret':>9} {'Med_range':>10} {'P25_ret':>8} {'P75_ret':>8}")
    for _, r in baseline_df.iterrows():
        print(f"{r['symbol']:<18} {r['n']:>7.0f} {r['WR']:>6.3f} {r['median_ret_60m']:>+9.1f} "
              f"{r['mean_ret_60m']:>+9.1f} {r['median_range_60m']:>10.1f} {r['p25_ret']:>+8.1f} {r['p75_ret']:>+8.1f}")

    # §11.2.2: Regime report
    if all_regime_results:
        regime_df = pd.DataFrame(all_regime_results)

        # §8.3: FDR correction on p_perm_ret
        regime_df["q_fdr_ret"] = apply_fdr(regime_df["p_perm_ret"].values)
        regime_df["q_fdr_range"] = apply_fdr(regime_df["p_perm_range"].values)

        regime_df.to_csv(OUTPUT_DIR / "report_regimes.csv", index=False)

        print(f"\n{'─'*70}")
        print("REGIME EFFECTS")
        print(f"{'─'*70}")
        print(f"{'Symbol':<16} {'Regime':<22} {'N':>5} {'%Tot':>5} {'Med_ret':>8} {'WR':>6} {'Upl_WR':>7} "
              f"{'Med_rng':>8} {'Upl_rng':>8} {'CI_lo':>7} {'CI_hi':>7} {'p_ret':>7} {'q_ret':>7}")
        for _, r in regime_df.sort_values(["symbol", "regime"]).iterrows():
            sig = ""
            if r["q_fdr_ret"] < 0.05:
                sig = " ***"
            elif r["q_fdr_ret"] < 0.10:
                sig = " **"
            elif r["p_perm_ret"] < 0.05:
                sig = " *"
            print(f"{r['symbol']:<16} {r['regime']:<22} {r['n']:>5.0f} {r['pct_of_total']:>5.1f} "
                  f"{r['median_ret_60m']:>+8.1f} {r['WR']:>6.3f} {r['uplift_WR']:>+7.3f} "
                  f"{r['median_range_60m']:>8.1f} {r['uplift_range']:>8.2f}x "
                  f"{r['CI_lo_ret']:>7.1f} {r['CI_hi_ret']:>7.1f} "
                  f"{r['p_perm_ret']:>7.3f} {r['q_fdr_ret']:>7.3f}{sig}")

        # §9: Success criteria check
        print(f"\n{'─'*70}")
        print("LIVE REGIME CANDIDATES (§9 criteria)")
        print(f"{'─'*70}")

        # Directional candidates
        dir_cands = regime_df[
            (regime_df["median_ret_60m"].abs() >= MIN_DIRECTIONAL_MEDIAN_BP) &
            (regime_df["uplift_WR"].abs() >= MIN_WR_UPLIFT) &
            (regime_df["q_fdr_ret"] < FDR_ALPHA)
        ]
        if len(dir_cands) > 0:
            print("\nDirectional candidates (|med_ret| ≥ 20bp, |uplift_WR| ≥ 5%, q_fdr < 0.10):")
            for _, r in dir_cands.iterrows():
                print(f"  {r['symbol']} / {r['regime']}: med={r['median_ret_60m']:+.1f}bp, "
                      f"WR={r['WR']:.3f} ({r['uplift_WR']:+.3f}), N={r['n']:.0f}, q={r['q_fdr_ret']:.3f}")
        else:
            print("\nNo directional candidates pass all criteria.")

        # Range candidates
        range_cands = regime_df[
            (regime_df["uplift_range"] >= MIN_RANGE_MULTIPLIER) &
            (regime_df["q_fdr_range"] < FDR_ALPHA)
        ]
        if len(range_cands) > 0:
            print(f"\nRange/vol candidates (uplift ≥ {MIN_RANGE_MULTIPLIER}x, q_fdr < 0.10):")
            for _, r in range_cands.iterrows():
                print(f"  {r['symbol']} / {r['regime']}: med_range={r['median_range_60m']:.1f}bp "
                      f"({r['uplift_range']:.2f}x), N={r['n']:.0f}, q={r['q_fdr_range']:.3f}")
        else:
            print("\nNo range candidates pass all criteria.")

        # §7.3: Symmetry test
        print(f"\n{'─'*70}")
        print("SYMMETRY TEST (REG_OI_FUND by funding sign)")
        print(f"{'─'*70}")
        long_results = regime_df[regime_df["regime"] == "REG_OI_FUND_LONG"]
        short_results = regime_df[regime_df["regime"] == "REG_OI_FUND_SHORT"]

        if len(long_results) > 0 or len(short_results) > 0:
            print(f"\n{'Symbol':<16} {'Leg':<8} {'N':>5} {'Med_ret':>9} {'WR':>6} {'Med_rng':>9}")
            for _, r in long_results.iterrows():
                print(f"{r['symbol']:<16} {'FR>0':<8} {r['n']:>5.0f} {r['median_ret_60m']:>+9.1f} "
                      f"{r['WR']:>6.3f} {r['median_range_60m']:>9.1f}")
            for _, r in short_results.iterrows():
                print(f"{r['symbol']:<16} {'FR<0':<8} {r['n']:>5.0f} {r['median_ret_60m']:>+9.1f} "
                      f"{r['WR']:>6.3f} {r['median_range_60m']:>9.1f}")
        else:
            print("\nNo OI_FUND regime observations with signed funding split.")

    # §11.2.3: Weekly report
    if all_weekly:
        weekly_df = pd.DataFrame(all_weekly)
        weekly_df.to_csv(OUTPUT_DIR / "report_weekly.csv", index=False)

        print(f"\n{'─'*70}")
        print("WEEKLY STABILITY (selected regimes)")
        print(f"{'─'*70}")

        for rc in ["REG_OI_FUND", "REG_COMPRESSION", "REG_EXHAUST"]:
            rc_weekly = weekly_df[weekly_df["regime"] == rc]
            if len(rc_weekly) == 0:
                continue
            print(f"\n  {rc}:")
            for sym in rc_weekly["symbol"].unique():
                sym_w = rc_weekly[rc_weekly["symbol"] == sym].sort_values(["year", "week"])
                n_weeks = len(sym_w)
                pos_weeks = (sym_w["median_ret_60m"] > 0).sum()
                neg_weeks = (sym_w["median_ret_60m"] < 0).sum()
                total_n = sym_w["n"].sum()
                print(f"    {sym}: {n_weeks} weeks, {total_n} obs, "
                      f"positive={pos_weeks} negative={neg_weeks}")

    elapsed = time.monotonic() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Outputs in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
