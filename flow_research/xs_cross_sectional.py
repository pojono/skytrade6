#!/usr/bin/env python3
"""
Cross-Sectional Market Structure Research — 8 Specifications

Implements all 8 cross-sectional analyses at the MARKET level (not coin level):
  1. Market State Engine (Systemic vs Idiosyncratic)
  2. Leadership / Lead-Lag Matrix
  3. Market Clustering (Unsupervised Regimes)
  4. Cross-Sectional Entropy / Concentration
  5. Clustered OI Build-up
  6. Extreme Co-Movement Percentiles
  7. Market Compression Index
  8. Correlation Network Structure

Data: 50 Bybit perps, 2025-07-01 → 2026-03-02
Grid: 5-minute cross-sectional snapshots on unified 1m base
Train: Jul 2025 - Dec 2025 (6 months)
Test OOS: Jan-Feb 2026 (2 months)
Protection: expanding percentiles, shuffle tests, OOS validation
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs_cross"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-03-02 23:59:59", tz="UTC")

MIN_DAYS = 100  # minimum kline days to include a coin

# Train / Test split
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
PURGE_HOURS = 24

# Cross-sectional snapshot interval
SNAP_MINUTES = 5

# Rolling windows
RV_6H_WINDOW = 360       # 6h in 1m bars
RV_2H_WINDOW = 120
OI_Z_WINDOW = 7 * 24 * 60  # 7 days
FR_Z_WINDOW = 7 * 24 * 60
ATR_14H = 14 * 60

# Target: big move definitions (same as xs6 for compatibility)
ATR_K = 3.0
HORIZONS_MIN = {"12h": 720, "24h": 1440}

# S07 definition (from xs6 — the validated signal)
S07_RV6H_PCTL = 0.20
S07_OI_Z_THRESH = 1.5

SEED = 42
N_SHUFFLE = 500

# ---------------------------------------------------------------------------
# §1: Data loading (reuse xs6 patterns)
# ---------------------------------------------------------------------------

def discover_symbols() -> list[str]:
    """Find symbols with sufficient data coverage."""
    syms = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        nkline = len([f for f in d.glob("*_kline_1m.csv")
                      if "mark_price" not in f.name and "premium_index" not in f.name])
        noi = len(list(d.glob("*_open_interest_5min.csv")))
        nfr = len(list(d.glob("*_funding_rate.csv")))
        nmark = len(list(d.glob("*_mark_price_kline_1m.csv")))
        if nkline >= MIN_DAYS and noi >= MIN_DAYS and nfr >= MIN_DAYS and nmark >= MIN_DAYS:
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


def build_sym_1m(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    """Build unified 1m grid for one symbol."""
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

    is_invalid = np.zeros(n, dtype=np.int8)
    is_invalid[nan_arr & (block_len >= 5)] = 1

    close = close_raw.ffill()
    close[is_invalid == 1] = np.nan
    df["close"] = close
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


def compute_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-coin features on 1m grid. All windows are causal."""
    # Funding z-score (7d rolling)
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["funding_z"] = (fr - fr_rm) / fr_rs

    # OI features
    oi = df["oi"]
    oi_lag_1h = oi.shift(60)
    df["oi_chg_1h"] = (oi - oi_lag_1h) / oi_lag_1h.clip(lower=1)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["oi_z"] = (oi_chg - oi_rm) / oi_rs

    # Realized vol
    lr = df["log_ret"]
    df["rv_2h"] = lr.rolling(RV_2H_WINDOW, min_periods=RV_2H_WINDOW // 2).std()
    df["rv_6h"] = lr.rolling(RV_6H_WINDOW, min_periods=RV_6H_WINDOW // 2).std()

    # Volume aggregates
    df["turnover_1h"] = df["turnover"].rolling(60, min_periods=10).sum()

    # ATR 1h (14-period)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_1h"] = tr.rolling(ATR_14H, min_periods=ATR_14H // 2).mean() * 60

    # 1h return
    df["ret_1h"] = np.log(df["close"] / df["close"].shift(60))
    df.loc[df["is_invalid"] == 1, "ret_1h"] = np.nan

    # 5m return
    df["ret_5m"] = np.log(df["close"] / df["close"].shift(5))
    df.loc[df["is_invalid"] == 1, "ret_5m"] = np.nan

    # Causal percentiles (expanding from 30d minimum for rv_6h)
    PCTL_WINDOW = 30 * 24 * 60
    min_p = PCTL_WINDOW // 4
    df["rv_6h_pctl"] = df["rv_6h"].rolling(PCTL_WINDOW, min_periods=min_p).rank(pct=True)

    return df


def compute_big_move_targets(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute big move targets on 1m grid. Returns df with target columns."""
    close = df_1m["close"]
    atr = df_1m["atr_1h"]
    invalid = df_1m["is_invalid"]

    for h_label, h_min in HORIZONS_MIN.items():
        fwd_close = close.shift(-h_min)
        fwd_ret = (fwd_close / close) - 1.0

        fwd_invalid = invalid.rolling(h_min, min_periods=1).sum().shift(-h_min)
        fwd_ret[fwd_invalid > 0] = np.nan

        df_1m[f"fwd_ret_{h_label}"] = fwd_ret

        # Def A: ATR-normalized
        threshold = ATR_K * atr / close.clip(lower=1e-8)
        col = f"big_A_{h_label}"
        df_1m[col] = (fwd_ret.abs() >= threshold).astype(float)
        df_1m.loc[fwd_ret.isna(), col] = np.nan

    # S07 state (the validated signal from xs6)
    df_1m["S07"] = ((df_1m["rv_6h_pctl"] <= S07_RV6H_PCTL) &
                     (df_1m["oi_z"] >= S07_OI_Z_THRESH)).astype(float)

    return df_1m


# ---------------------------------------------------------------------------
# §2: Build universe-wide 5m cross-sectional panel
# ---------------------------------------------------------------------------

def build_cross_section_panel(symbols: list[str]) -> tuple:
    """Load all symbols, build 1m grids, compute features, return 5m panel.

    Returns:
        ret_5m_panel: DataFrame (timestamps × symbols) of 5m log returns
        ret_1h_panel: DataFrame (timestamps × symbols) of 1h log returns
        rv_6h_panel:  DataFrame (timestamps × symbols) of rv_6h
        oi_z_panel:   DataFrame (timestamps × symbols) of oi_z
        fund_z_panel: DataFrame (timestamps × symbols) of funding_z
        big_move_panel: dict of DataFrames per target
        s07_panel:    DataFrame (timestamps × symbols) of S07 state
        sym_1m_dict:  dict of 1m DataFrames per symbol
    """
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    grid_5m = pd.date_range(START, END, freq="5min", tz="UTC")

    # Panels to fill
    ret_5m = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    ret_1h = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    rv_6h = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    rv_2h = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    oi_z = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    fund_z = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    s07 = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    big_move = {}
    for h_label in HORIZONS_MIN:
        big_move[f"big_A_{h_label}"] = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    fwd_ret_panels = {}
    for h_label in HORIZONS_MIN:
        fwd_ret_panels[f"fwd_ret_{h_label}"] = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)

    t0 = time.monotonic()
    for i, sym in enumerate(symbols, 1):
        t1 = time.monotonic()
        raw = load_symbol(sym)
        df = build_sym_1m(sym, raw, grid_1m)
        df = compute_coin_features(df)
        df = compute_big_move_targets(df)

        # Sample to 5m grid
        df_5m = df.reindex(grid_5m)

        ret_5m[sym] = df_5m["ret_5m"].values
        ret_1h[sym] = df_5m["ret_1h"].values
        rv_6h[sym] = df_5m["rv_6h"].values
        rv_2h[sym] = df_5m["rv_2h"].values
        oi_z[sym] = df_5m["oi_z"].values
        fund_z[sym] = df_5m["funding_z"].values
        s07[sym] = df_5m["S07"].values

        for h_label in HORIZONS_MIN:
            big_move[f"big_A_{h_label}"][sym] = df_5m[f"big_A_{h_label}"].values
            fwd_ret_panels[f"fwd_ret_{h_label}"][sym] = df_5m[f"fwd_ret_{h_label}"].values

        elapsed = time.monotonic() - t0
        dt = time.monotonic() - t1
        eta = (len(symbols) - i) * elapsed / i
        print(f"  [{i}/{len(symbols)}] {sym:<20s} {dt:.1f}s  (total {elapsed:.0f}s, ETA {eta:.0f}s)")

    panels = {
        "ret_5m": ret_5m,
        "ret_1h": ret_1h,
        "rv_6h": rv_6h,
        "rv_2h": rv_2h,
        "oi_z": oi_z,
        "fund_z": fund_z,
        "s07": s07,
        "big_move": big_move,
        "fwd_ret": fwd_ret_panels,
    }
    return panels


# ---------------------------------------------------------------------------
# §3: Cross-sectional feature computation (market-level, every 5m)
# ---------------------------------------------------------------------------

def compute_market_features(panels: dict) -> pd.DataFrame:
    """Compute market-level cross-sectional features at each 5m snapshot.

    All features are causal (use only information available at that time).
    """
    ret_1h = panels["ret_1h"]
    ret_5m = panels["ret_5m"]
    rv_6h = panels["rv_6h"]
    rv_2h = panels["rv_2h"]
    oi_z = panels["oi_z"]
    fund_z = panels["fund_z"]

    idx = ret_1h.index
    mf = pd.DataFrame(index=idx)

    n_valid = ret_1h.notna().sum(axis=1)
    mf["n_coins"] = n_valid

    print("\nComputing market features...")
    t0 = time.monotonic()

    # --- Spec 1 & 3: Market state features ---
    # dispersion_1h: std of cross-sectional 1h returns
    mf["dispersion_1h"] = ret_1h.std(axis=1)

    # breadth_up: % of coins with ret_1h > 0
    mf["breadth_up"] = (ret_1h > 0).sum(axis=1) / n_valid.clip(lower=1)

    # breadth_extreme: % of coins with |ret_1h| > 1.5 * median |ret_1h|
    abs_ret_1h = ret_1h.abs()
    median_abs_ret = abs_ret_1h.median(axis=1)
    mf["breadth_extreme"] = (abs_ret_1h.gt(1.5 * median_abs_ret, axis=0)).sum(axis=1) / n_valid.clip(lower=1)

    # median_rv_6h
    mf["median_rv_6h"] = rv_6h.median(axis=1)

    # median_rv_2h
    mf["median_rv_2h"] = rv_2h.median(axis=1)

    # mean_oi_z
    mf["mean_oi_z"] = oi_z.mean(axis=1)

    # % funding_z > 2
    mf["pct_fund_z_gt2"] = (fund_z.abs() > 2).sum(axis=1) / n_valid.clip(lower=1)

    # mean funding_z
    mf["mean_fund_z"] = fund_z.mean(axis=1)

    # cross-sectional skew of 1h returns
    mf["xs_skew"] = ret_1h.skew(axis=1)

    print(f"  Basic features done ({time.monotonic()-t0:.1f}s)")

    # --- Spec 4: Entropy / Concentration ---
    abs_ret_5m = ret_5m.abs()
    E_abs = abs_ret_5m.sum(axis=1)
    # Avoid log(0): add tiny epsilon, compute only where E_abs > 0
    p_i = abs_ret_5m.div(E_abs.clip(lower=1e-20), axis=0)
    # Shannon entropy: -sum(p_i * log(p_i)), handle 0*log(0) = 0
    log_p = np.log(p_i.clip(lower=1e-20))
    mf["entropy_5m"] = -(p_i * log_p).sum(axis=1)
    mf["entropy_5m"][E_abs < 1e-15] = np.nan

    # Also do entropy on 1h returns
    abs_ret_1h_safe = abs_ret_1h.copy()
    E_abs_1h = abs_ret_1h_safe.sum(axis=1)
    p_i_1h = abs_ret_1h_safe.div(E_abs_1h.clip(lower=1e-20), axis=0)
    log_p_1h = np.log(p_i_1h.clip(lower=1e-20))
    mf["entropy_1h"] = -(p_i_1h * log_p_1h).sum(axis=1)
    mf["entropy_1h"][E_abs_1h < 1e-15] = np.nan

    # Max entropy for N coins = log(N)
    mf["max_entropy"] = np.log(n_valid.clip(lower=1))
    mf["norm_entropy_1h"] = mf["entropy_1h"] / mf["max_entropy"].clip(lower=1e-10)

    print(f"  Entropy done ({time.monotonic()-t0:.1f}s)")

    # --- Spec 5: Clustered OI Build-up ---
    mf["pct_oi_z_gt1p5"] = (oi_z > 1.5).sum(axis=1) / n_valid.clip(lower=1)
    mf["median_oi_z"] = oi_z.median(axis=1)
    # Cluster OI index = mean of top 10 oi_z values (vectorized)
    oi_arr = oi_z.values.copy()
    oi_arr[np.isnan(oi_arr)] = -np.inf  # push NaN to bottom
    top10_sorted = np.sort(oi_arr, axis=1)[:, -10:]  # top 10 per row
    top10_sorted[top10_sorted == -np.inf] = np.nan
    n_valid_top10 = np.sum(~np.isnan(top10_sorted), axis=1)
    mf["cluster_oi_top10"] = np.where(
        n_valid_top10 >= 5, np.nanmean(top10_sorted, axis=1), np.nan
    )

    print(f"  OI features done ({time.monotonic()-t0:.1f}s)")

    # --- Spec 7: Market Compression Index ---
    # market_rv = median rv_6h across coins
    # State: market_rv <= P20 (expanding percentile)
    PCTL_WIN = 30 * 24 * 12  # 30d in 5m bars
    min_p = PCTL_WIN // 4
    mf["market_rv_6h_pctl"] = mf["median_rv_6h"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)
    mf["market_compressed"] = (mf["market_rv_6h_pctl"] <= 0.20).astype(float)
    mf["market_compressed"][mf["market_rv_6h_pctl"].isna()] = np.nan

    print(f"  Market compression done ({time.monotonic()-t0:.1f}s)")

    # --- Causal percentiles for cross-sectional features ---
    for col in ["dispersion_1h", "entropy_1h", "norm_entropy_1h", "pct_oi_z_gt1p5",
                "cluster_oi_top10", "median_rv_6h"]:
        mf[f"{col}_pctl"] = mf[col].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    # Entropy quintiles
    mf["entropy_quintile"] = pd.cut(
        mf["norm_entropy_1h_pctl"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        include_lowest=True,
    )

    print(f"  All market features done ({time.monotonic()-t0:.1f}s)")
    return mf


# ---------------------------------------------------------------------------
# §4: Spec 4 — Cross-Sectional Entropy / Concentration
# ---------------------------------------------------------------------------

def run_spec4_entropy(mf: pd.DataFrame, panels: dict) -> dict:
    """Test: low entropy → concentrated moves → more big moves?"""
    print("\n" + "=" * 70)
    print("SPEC 4: Cross-Sectional Entropy / Concentration")
    print("=" * 70)

    results = {}
    rng = np.random.default_rng(SEED)

    for target_key, bm_panel in panels["big_move"].items():
        # Universe-wide big move rate at each 5m snapshot
        bm_rate = bm_panel.mean(axis=1)  # fraction of coins with big move
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)  # any coin has big move
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        # Join with market features
        joined = mf[["norm_entropy_1h", "entropy_quintile", "market_rv_6h_pctl",
                      "dispersion_1h_pctl"]].copy()
        joined["bm_rate"] = bm_rate
        joined["bm_any"] = bm_any

        # Split train/test
        train = joined[joined.index <= TRAIN_END]
        test = joined[joined.index >= TEST_START]

        for period_name, data in [("train", train), ("test", test)]:
            data_valid = data.dropna(subset=["entropy_quintile", "bm_any"])
            if len(data_valid) < 100:
                continue

            baseline_rate = data_valid["bm_any"].mean()

            print(f"\n  {target_key} — {period_name} (n={len(data_valid)}, baseline={baseline_rate:.4f})")
            print(f"  {'Quintile':<12} {'N':>7} {'BM Rate':>9} {'Uplift':>8} {'p-value':>9}")
            print(f"  {'-'*50}")

            for q in ["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]:
                mask = data_valid["entropy_quintile"] == q
                n_q = mask.sum()
                if n_q < 20:
                    continue
                rate_q = data_valid.loc[mask, "bm_any"].mean()
                uplift = rate_q / baseline_rate if baseline_rate > 0 else np.nan

                # Permutation test: shuffle entropy quintile labels within days
                day_labels = data_valid.index.date
                bm_arr = data_valid["bm_any"].values
                q_arr = (data_valid["entropy_quintile"] == q).values
                observed = bm_arr[q_arr].mean() if q_arr.sum() > 0 else 0

                count_ge = 0
                unique_days = np.unique(day_labels)
                day_idx = {d: np.where(day_labels == d)[0] for d in unique_days}
                for _ in range(N_SHUFFLE):
                    perm_sum = 0.0
                    perm_n = 0
                    for d in unique_days:
                        idx = day_idx[d]
                        nd = q_arr[idx].sum()
                        if nd == 0:
                            continue
                        perm_idx = rng.choice(idx, size=nd, replace=False)
                        perm_sum += bm_arr[perm_idx].sum()
                        perm_n += nd
                    if perm_n > 0 and perm_sum / perm_n >= observed:
                        count_ge += 1
                pval = (count_ge + 1) / (N_SHUFFLE + 1)

                print(f"  {q:<12} {n_q:>7} {rate_q:>9.4f} {uplift:>8.2f}x {pval:>9.4f}")

                results[f"{target_key}_{period_name}_{q}"] = {
                    "n": n_q, "rate": rate_q, "uplift": uplift, "pval": pval,
                }

    return results


# ---------------------------------------------------------------------------
# §5: Spec 2 — Leadership / Lead-Lag Matrix
# ---------------------------------------------------------------------------

def run_spec2_leadership(panels: dict) -> dict:
    """Compute lead-lag correlation matrix and test if leaders predict big moves."""
    print("\n" + "=" * 70)
    print("SPEC 2: Leadership / Lead-Lag Matrix")
    print("=" * 70)

    ret_5m = panels["ret_5m"]
    ret_1h = panels["ret_1h"]

    # Use training period only for computing lead scores
    train_mask = ret_5m.index <= TRAIN_END
    ret_5m_train = ret_5m[train_mask].dropna(axis=1, how="all")
    symbols = ret_5m_train.columns.tolist()
    n_syms = len(symbols)

    print(f"  Computing lead-lag matrix for {n_syms} symbols (vectorized)...")
    t0 = time.monotonic()

    # Lead-lag at different lags (in 5m bars) — vectorized via correlation matrix
    lags = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}
    lead_scores = pd.DataFrame(0.0, index=symbols, columns=lags.keys())

    # Fill NaN with 0 for correlation (after demeaning)
    ret_arr = ret_5m_train[symbols].values  # (T, N)

    for lag_name, lag_bars in lags.items():
        t1 = time.monotonic()
        # corr(ret_i(t), ret_j(t+lag)) = i leads j
        # = corr between ret_arr[:-lag] and ret_arr[lag:]
        A = ret_arr[:-lag_bars]  # leader returns at time t
        B = ret_arr[lag_bars:]   # follower returns at time t+lag

        # Mask where both are valid
        valid = ~np.isnan(A) & ~np.isnan(B)

        # Compute pairwise correlations: corr(A_i, B_j) for all i,j
        # Use column-wise demeaning and normalization
        A_clean = np.where(valid, A, 0.0)
        B_clean = np.where(valid, B, 0.0)

        # Per-column means (only where both valid)
        n_valid_pairs = valid.astype(float)  # per-element
        # For each (i,j) pair, valid count differs. Use simplified approach:
        # Compute correlation matrix via pandas (handles NaN properly)
        df_a = pd.DataFrame(A, columns=symbols)
        df_b = pd.DataFrame(B, columns=symbols)

        # corr_matrix[i,j] = corr(A_col_i, B_col_j)
        # This is cross-correlation, not standard corr matrix
        # Efficient: demean, normalize, dot product
        a_vals = df_a.values
        b_vals = df_b.values
        n_t = len(a_vals)

        # Replace NaN with 0, compute valid counts per pair
        a_nan = np.isnan(a_vals)
        b_nan = np.isnan(b_vals)
        a_clean = np.nan_to_num(a_vals, 0.0)
        b_clean = np.nan_to_num(b_vals, 0.0)

        # Column means (excluding NaN)
        a_count = (~a_nan).sum(axis=0).clip(min=1)
        b_count = (~b_nan).sum(axis=0).clip(min=1)
        a_mean = np.nansum(a_vals, axis=0) / a_count
        b_mean = np.nansum(b_vals, axis=0) / b_count

        # Demean (set NaN positions to 0 after demeaning)
        a_dm = np.where(a_nan, 0.0, a_vals - a_mean)
        b_dm = np.where(b_nan, 0.0, b_vals - b_mean)

        # Cross-covariance matrix: (a_dm.T @ b_dm) / n_t
        cross_cov = a_dm.T @ b_dm / n_t

        # Standard deviations
        a_std = np.sqrt((a_dm ** 2).sum(axis=0) / n_t).clip(min=1e-12)
        b_std = np.sqrt((b_dm ** 2).sum(axis=0) / n_t).clip(min=1e-12)

        # Correlation matrix
        corr_matrix = cross_cov / np.outer(a_std, b_std)
        np.fill_diagonal(corr_matrix, 0.0)  # remove self-correlation

        # Lead score for symbol i = mean of corr(i leads j) across all j
        lead_scores[lag_name] = corr_matrix.sum(axis=1) / (n_syms - 1)
        print(f"  Lag {lag_name}: done ({time.monotonic()-t1:.1f}s)")

    # Aggregate lead score
    lead_scores["total"] = lead_scores.sum(axis=1)
    lead_scores = lead_scores.sort_values("total", ascending=False)

    print(f"\n  Top 10 Leaders (train period):")
    print(f"  {'Symbol':<20} {'5m':>7} {'15m':>7} {'30m':>7} {'60m':>7} {'Total':>8}")
    for sym in lead_scores.head(10).index:
        row = lead_scores.loc[sym]
        print(f"  {sym:<20} {row['5m']:>7.4f} {row['15m']:>7.4f} {row['30m']:>7.4f} {row['60m']:>7.4f} {row['total']:>8.4f}")

    # Save lead scores
    lead_scores.to_csv(OUTPUT_DIR / "lead_lag_scores.csv")

    # Test: when a top-5 leader makes a large move, is P(big move in others) elevated?
    print("\n  Testing: top-5 leader large move → big move in others?")
    top5 = lead_scores.head(5).index.tolist()

    # Use test period
    test_mask = ret_1h.index >= TEST_START
    ret_1h_test = ret_1h[test_mask]

    for target_key, bm_panel in panels["big_move"].items():
        bm_test = bm_panel[test_mask]
        # "Others" = all non-leader coins
        others = [s for s in symbols if s not in top5]
        bm_others_rate = bm_test[others].mean(axis=1)
        bm_others_any = (bm_test[others].sum(axis=1) > 0).astype(float)

        baseline = bm_others_any.dropna().mean()

        # Leader large move: |ret_1h| > P90 of that leader
        for sym in top5:
            ret_sym = ret_1h_test[sym].dropna()
            if len(ret_sym) < 50:
                continue
            thresh = ret_sym.abs().quantile(0.90)
            leader_active = ret_sym.abs() > thresh

            # Conditional big move rate in others in SAME and NEXT hour
            common = leader_active.index.intersection(bm_others_any.dropna().index)
            if len(common) < 20:
                continue

            rate_conditional = bm_others_any.loc[common][leader_active.loc[common]].mean()
            n_events = leader_active.loc[common].sum()
            uplift = rate_conditional / baseline if baseline > 0 else np.nan

            print(f"  {target_key} | Leader {sym}: {n_events:.0f} events, "
                  f"cond rate={rate_conditional:.4f}, baseline={baseline:.4f}, "
                  f"uplift={uplift:.2f}x")

    print(f"\n  Lead-lag analysis done ({time.monotonic()-t0:.0f}s)")
    return {"lead_scores": lead_scores}


# ---------------------------------------------------------------------------
# §6: Spec 7 — Market Compression Index
# ---------------------------------------------------------------------------

def run_spec7_compression(mf: pd.DataFrame, panels: dict) -> dict:
    """Test: market compression (low median rv_6h) → more big moves next 24h?"""
    print("\n" + "=" * 70)
    print("SPEC 7: Market Compression Index")
    print("=" * 70)

    results = {}

    for target_key, bm_panel in panels["big_move"].items():
        bm_rate = bm_panel.mean(axis=1)
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        joined = mf[["market_compressed", "market_rv_6h_pctl", "median_rv_6h"]].copy()
        joined["bm_any"] = bm_any
        joined["bm_rate"] = bm_rate

        # Also test interaction: S07 ∩ market_compressed
        s07_any = (panels["s07"].sum(axis=1) > 0).astype(float)
        joined["s07_any"] = s07_any
        joined["s07_and_compress"] = ((s07_any == 1) & (mf["market_compressed"] == 1)).astype(float)

        for period_name, mask in [("train", joined.index <= TRAIN_END),
                                   ("test", joined.index >= TEST_START)]:
            data = joined[mask].dropna(subset=["market_compressed", "bm_any"])
            if len(data) < 100:
                continue

            baseline = data["bm_any"].mean()

            # Market compressed
            comp_mask = data["market_compressed"] == 1
            n_comp = comp_mask.sum()
            rate_comp = data.loc[comp_mask, "bm_any"].mean() if n_comp > 20 else np.nan
            uplift_comp = rate_comp / baseline if baseline > 0 and not np.isnan(rate_comp) else np.nan

            # S07 AND compressed
            s07c_mask = data["s07_and_compress"] == 1
            n_s07c = s07c_mask.sum()
            rate_s07c = data.loc[s07c_mask, "bm_any"].mean() if n_s07c > 10 else np.nan
            uplift_s07c = rate_s07c / baseline if baseline > 0 and not np.isnan(rate_s07c) else np.nan

            # S07 alone (for comparison)
            s07_mask = data["s07_any"] == 1
            n_s07 = s07_mask.sum()
            rate_s07 = data.loc[s07_mask, "bm_any"].mean() if n_s07 > 10 else np.nan
            uplift_s07 = rate_s07 / baseline if baseline > 0 and not np.isnan(rate_s07) else np.nan

            print(f"\n  {target_key} — {period_name} (n={len(data)}, baseline={baseline:.4f})")
            print(f"  {'Condition':<30} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
            print(f"  {'-'*58}")
            print(f"  {'Market compressed (P20)':<30} {n_comp:>7} {rate_comp:>9.4f} {uplift_comp:>8.2f}x")
            print(f"  {'S07 alone':<30} {n_s07:>7} {rate_s07:>9.4f} {uplift_s07:>8.2f}x")
            print(f"  {'S07 ∩ compressed':<30} {n_s07c:>7} {rate_s07c:>9.4f} {uplift_s07c:>8.2f}x")

            results[f"{target_key}_{period_name}"] = {
                "baseline": baseline,
                "compress_uplift": uplift_comp,
                "s07_uplift": uplift_s07,
                "s07_compress_uplift": uplift_s07c,
                "n_compress": n_comp,
                "n_s07": n_s07,
                "n_s07_compress": n_s07c,
            }

    return results


# ---------------------------------------------------------------------------
# §7: Spec 3 — Market Clustering (Unsupervised Regimes)
# ---------------------------------------------------------------------------

def run_spec3_clustering(mf: pd.DataFrame, panels: dict) -> dict:
    """PCA + KMeans on market features → regime-dependent big move rates."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    print("\n" + "=" * 70)
    print("SPEC 3: Market Clustering (Unsupervised Regimes)")
    print("=" * 70)

    feature_cols = ["dispersion_1h", "median_rv_6h", "median_oi_z",
                    "mean_fund_z", "breadth_extreme", "xs_skew"]

    # Prepare data
    train_mask = mf.index <= TRAIN_END
    test_mask = mf.index >= TEST_START

    data_all = mf[feature_cols].copy()
    data_train = data_all[train_mask].dropna()
    data_test = data_all[test_mask].dropna()

    if len(data_train) < 500:
        print("  Not enough training data for clustering")
        return {}

    # Standardize on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data_train)
    X_test = scaler.transform(data_test.reindex(data_test.index))

    # PCA (3 components)
    pca = PCA(n_components=3, random_state=SEED)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

    results = {}
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels_train = km.fit_predict(Z_train)
        labels_test = km.predict(Z_test)

        # Assign labels back
        mf_labeled = mf.copy()
        mf_labeled[f"cluster_k{k}"] = np.nan
        mf_labeled.loc[data_train.index, f"cluster_k{k}"] = labels_train
        mf_labeled.loc[data_test.index, f"cluster_k{k}"] = labels_test

        print(f"\n  k={k} clustering:")

        for target_key, bm_panel in panels["big_move"].items():
            bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
            bm_any[bm_panel.isna().all(axis=1)] = np.nan

            for period_name, pmask in [("train", train_mask), ("test", test_mask)]:
                joined = pd.DataFrame({
                    "cluster": mf_labeled.loc[pmask, f"cluster_k{k}"],
                    "bm_any": bm_any[pmask],
                })
                joined = joined.dropna()
                if len(joined) < 100:
                    continue

                baseline = joined["bm_any"].mean()
                print(f"\n    {target_key} — {period_name} (baseline={baseline:.4f})")
                print(f"    {'Cluster':>8} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
                print(f"    {'-'*40}")

                for c in range(k):
                    cm = joined["cluster"] == c
                    n_c = cm.sum()
                    if n_c < 20:
                        continue
                    rate_c = joined.loc[cm, "bm_any"].mean()
                    uplift_c = rate_c / baseline if baseline > 0 else np.nan
                    print(f"    {c:>8} {n_c:>7} {rate_c:>9.4f} {uplift_c:>8.2f}x")
                    results[f"k{k}_{target_key}_{period_name}_c{c}"] = {
                        "n": n_c, "rate": rate_c, "uplift": uplift_c,
                    }

    return results


# ---------------------------------------------------------------------------
# §8: Spec 1 — Market State Engine (Systemic vs Idiosyncratic)
# ---------------------------------------------------------------------------

def run_spec1_market_state(mf: pd.DataFrame, panels: dict) -> dict:
    """Discretize market features into states, test big move rates."""
    print("\n" + "=" * 70)
    print("SPEC 1: Market State Engine (Systemic vs Idiosyncratic)")
    print("=" * 70)

    results = {}

    # Define market states
    states = {
        "high_dispersion": mf["dispersion_1h_pctl"] >= 0.80,
        "low_dispersion": mf["dispersion_1h_pctl"] <= 0.20,
        "high_breadth_ext": mf["breadth_extreme"] >= 0.30,
        "low_entropy": mf["norm_entropy_1h_pctl"] <= 0.20,
        "high_entropy": mf["norm_entropy_1h_pctl"] >= 0.80,
        "high_mean_oi_z": mf["mean_oi_z"] >= 1.0,
        "high_pct_fund_ext": mf["pct_fund_z_gt2"] >= 0.10,
        "market_compressed": mf["market_compressed"] == 1,
    }

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        # Also get S07 signal
        s07_any = (panels["s07"].sum(axis=1) > 0).astype(float)

        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            joined = pd.DataFrame({"bm_any": bm_any[pmask], "s07": s07_any[pmask]})
            for sn, sv in states.items():
                joined[sn] = sv[pmask].astype(float)

            joined = joined.dropna(subset=["bm_any"])
            if len(joined) < 100:
                continue

            baseline = joined["bm_any"].mean()
            print(f"\n  {target_key} — {period_name} (n={len(joined)}, baseline={baseline:.4f})")
            print(f"  {'State':<25} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
            print(f"  {'-'*53}")

            for sn in states:
                sm = joined[sn] == 1
                n_s = sm.sum()
                if n_s < 20:
                    continue
                rate_s = joined.loc[sm, "bm_any"].mean()
                uplift_s = rate_s / baseline if baseline > 0 else np.nan
                print(f"  {sn:<25} {n_s:>7} {rate_s:>9.4f} {uplift_s:>8.2f}x")

                # Interaction: state AND S07
                s07_int = sm & (joined["s07"] == 1)
                n_int = s07_int.sum()
                if n_int >= 10:
                    rate_int = joined.loc[s07_int, "bm_any"].mean()
                    uplift_int = rate_int / baseline if baseline > 0 else np.nan
                    print(f"    + S07 interaction: n={n_int}, rate={rate_int:.4f}, uplift={uplift_int:.2f}x")

                results[f"{target_key}_{period_name}_{sn}"] = {
                    "n": n_s, "rate": rate_s, "uplift": uplift_s,
                }

    return results


# ---------------------------------------------------------------------------
# §9: Spec 5 — Clustered OI Build-up
# ---------------------------------------------------------------------------

def run_spec5_clustered_oi(mf: pd.DataFrame, panels: dict) -> dict:
    """Test: systemic OI build-up → more big moves?"""
    print("\n" + "=" * 70)
    print("SPEC 5: Clustered OI Build-up")
    print("=" * 70)

    results = {}
    rng = np.random.default_rng(SEED + 5)

    # OI metrics thresholds
    oi_conditions = {
        "pct_oi_z>1.5 >= 20%": mf["pct_oi_z_gt1p5"] >= 0.20,
        "pct_oi_z>1.5 >= 30%": mf["pct_oi_z_gt1p5"] >= 0.30,
        "median_oi_z >= 1.0": mf["median_oi_z"] >= 1.0,
        "cluster_top10 >= 2.0": mf["cluster_oi_top10"] >= 2.0,
    }

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        s07_any = (panels["s07"].sum(axis=1) > 0).astype(float)

        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            data = pd.DataFrame({"bm_any": bm_any[pmask], "s07": s07_any[pmask]})
            for cn, cv in oi_conditions.items():
                data[cn] = cv[pmask].astype(float)
            data = data.dropna(subset=["bm_any"])
            if len(data) < 100:
                continue

            baseline = data["bm_any"].mean()
            print(f"\n  {target_key} — {period_name} (n={len(data)}, baseline={baseline:.4f})")
            print(f"  {'Condition':<30} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
            print(f"  {'-'*58}")

            for cn in oi_conditions:
                cm = data[cn] == 1
                n_c = cm.sum()
                if n_c < 20:
                    continue
                rate_c = data.loc[cm, "bm_any"].mean()
                uplift_c = rate_c / baseline if baseline > 0 else np.nan
                print(f"  {cn:<30} {n_c:>7} {rate_c:>9.4f} {uplift_c:>8.2f}x")

                # S07 interaction
                s07_int = cm & (data["s07"] == 1)
                n_int = s07_int.sum()
                if n_int >= 10:
                    rate_int = data.loc[s07_int, "bm_any"].mean()
                    uplift_int = rate_int / baseline if baseline > 0 else np.nan
                    print(f"    + S07: n={n_int}, rate={rate_int:.4f}, uplift={uplift_int:.2f}x")

                results[f"{target_key}_{period_name}_{cn}"] = {
                    "n": n_c, "rate": rate_c, "uplift": uplift_c,
                }

    return results


# ---------------------------------------------------------------------------
# §10: Spec 6 — Extreme Co-Movement Percentiles
# ---------------------------------------------------------------------------

def run_spec6_comovement(panels: dict) -> dict:
    """For each big move: is it isolated or part of a wave?"""
    print("\n" + "=" * 70)
    print("SPEC 6: Extreme Co-Movement Percentiles")
    print("=" * 70)

    ret_1h = panels["ret_1h"]
    results = {}

    for target_key, bm_panel in panels["big_move"].items():
        for period_name, pmask in [("train", ret_1h.index <= TRAIN_END),
                                    ("test", ret_1h.index >= TEST_START)]:
            ret_p = ret_1h[pmask]
            bm_p = bm_panel[pmask]

            if len(ret_p) < 100:
                continue

            # For each timestamp, count how many coins are in P95 of |ret_1h|
            abs_ret = ret_p.abs()
            # Per-coin P95 threshold (expanding with 30d min)
            n_valid = abs_ret.notna().sum(axis=1)

            # Cross-sectional P95 at each timestamp
            p95_xs = abs_ret.quantile(0.95, axis=1)
            n_extreme = (abs_ret.gt(p95_xs, axis=0)).sum(axis=1)
            pct_extreme = n_extreme / n_valid.clip(lower=1)

            # Timestamps with any big move
            bm_any = bm_p.sum(axis=1) > 0
            bm_count = bm_p.sum(axis=1)

            # Compare pct_extreme during big move hours vs random hours
            bm_hours = pct_extreme[bm_any].dropna()
            random_hours = pct_extreme[~bm_any].dropna()

            if len(bm_hours) < 10 or len(random_hours) < 100:
                continue

            mean_bm = bm_hours.mean()
            mean_random = random_hours.mean()
            ratio = mean_bm / mean_random if mean_random > 0 else np.nan

            # Test: are big moves clustered (wave) or isolated?
            # Multi-coin big move: 2+ coins have big move simultaneously
            multi_bm = bm_count >= 2
            n_multi = multi_bm.sum()
            n_single = ((bm_count == 1)).sum()

            # Continuation: after a multi-coin wave, is there another wave in next 12 bars (1h)?
            if n_multi > 10:
                multi_idx = np.where(multi_bm.values)[0]
                continuation_count = 0
                for mi in multi_idx:
                    # Check next 12 bars
                    for offset in range(1, 13):
                        if mi + offset < len(multi_bm) and multi_bm.iloc[mi + offset]:
                            continuation_count += 1
                            break
                cont_rate = continuation_count / len(multi_idx) if len(multi_idx) > 0 else 0

                # Random baseline for continuation
                random_cont = n_multi / len(multi_bm) * 12  # expected under independence
            else:
                cont_rate = np.nan
                random_cont = np.nan

            print(f"\n  {target_key} — {period_name}")
            print(f"    Mean %extreme during big move hours: {mean_bm:.4f}")
            print(f"    Mean %extreme during random hours:   {mean_random:.4f}")
            print(f"    Ratio: {ratio:.2f}x")
            print(f"    Multi-coin big moves: {n_multi}, Single: {n_single}")
            if not np.isnan(cont_rate):
                print(f"    Wave continuation rate: {cont_rate:.3f} (random baseline: {random_cont:.3f})")

            results[f"{target_key}_{period_name}"] = {
                "mean_extreme_bm": mean_bm,
                "mean_extreme_random": mean_random,
                "ratio": ratio,
                "n_multi": n_multi,
                "n_single": n_single,
                "continuation_rate": cont_rate,
            }

    return results


# ---------------------------------------------------------------------------
# §11: Spec 8 — Correlation Network Structure
# ---------------------------------------------------------------------------

def run_spec8_network(panels: dict) -> dict:
    """Rolling correlation network → graph metrics before big moves."""
    print("\n" + "=" * 70)
    print("SPEC 8: Correlation Network Structure")
    print("=" * 70)

    ret_1h = panels["ret_1h"]
    symbols = ret_1h.columns.tolist()
    n_syms = len(symbols)

    # Rolling 6h correlation matrix (6h = 72 5m bars)
    # We compute this every 1h (12 5m bars) to save time
    step_bars = 12  # 1h step
    window_bars = 72  # 6h window

    idx_all = ret_1h.index
    n_total = len(idx_all)

    timestamps = []
    densities = []
    avg_clusterings = []
    n_components_list = []
    eigvec_disp = []

    print(f"  Computing rolling 6h correlation network ({n_syms} symbols)...")
    t0 = time.monotonic()

    # Threshold for correlation edge
    CORR_THRESH = 0.3

    compute_indices = list(range(window_bars, n_total, step_bars))
    for ci, end_idx in enumerate(compute_indices):
        start_idx = end_idx - window_bars
        window_data = ret_1h.iloc[start_idx:end_idx]

        # Skip if too many NaN
        valid_cols = window_data.dropna(axis=1, thresh=window_bars // 2).columns
        if len(valid_cols) < 10:
            continue

        corr_mat = window_data[valid_cols].corr()
        n = len(valid_cols)

        # Adjacency matrix (threshold)
        adj = (corr_mat.abs() > CORR_THRESH).values.astype(int)
        np.fill_diagonal(adj, 0)

        # Graph density
        n_edges = adj.sum() / 2
        max_edges = n * (n - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        # Average clustering coefficient (local)
        # For each node: fraction of neighbor pairs that are also connected
        clustering_coeffs = []
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            # Count edges among neighbors
            sub = adj[np.ix_(neighbors, neighbors)]
            n_neighbor_edges = sub.sum() / 2
            max_neighbor_edges = k * (k - 1) / 2
            clustering_coeffs.append(n_neighbor_edges / max_neighbor_edges)
        avg_clustering = np.mean(clustering_coeffs)

        # Connected components via BFS
        visited = set()
        n_comp = 0
        for i in range(n):
            if i in visited:
                continue
            n_comp += 1
            queue = [i]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                for j in np.where(adj[node] > 0)[0]:
                    if j not in visited:
                        queue.append(j)

        # Eigenvector centrality dispersion
        try:
            eigenvalues = np.linalg.eigvalsh(adj.astype(float))
            eigvec_d = np.std(eigenvalues) / (np.abs(np.mean(eigenvalues)) + 1e-10)
        except Exception:
            eigvec_d = np.nan

        timestamps.append(idx_all[end_idx - 1])
        densities.append(density)
        avg_clusterings.append(avg_clustering)
        n_components_list.append(n_comp)
        eigvec_disp.append(eigvec_d)

        if (ci + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            eta = (len(compute_indices) - ci - 1) * elapsed / (ci + 1)
            print(f"    [{ci+1}/{len(compute_indices)}] ({elapsed:.0f}s, ETA {eta:.0f}s)")

    net_df = pd.DataFrame({
        "ts": timestamps,
        "density": densities,
        "avg_clustering": avg_clusterings,
        "n_components": n_components_list,
        "eigvec_dispersion": eigvec_disp,
    }).set_index("ts")

    print(f"  Network metrics computed: {len(net_df)} snapshots ({time.monotonic()-t0:.0f}s)")

    # Save network metrics
    net_df.to_csv(OUTPUT_DIR / "network_metrics.csv")

    # Test: compare network metrics before big moves vs baseline
    results = {}
    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        # Align network metrics with big moves
        common_idx = net_df.index.intersection(bm_any.dropna().index)
        if len(common_idx) < 100:
            continue

        net_aligned = net_df.loc[common_idx]
        bm_aligned = bm_any.loc[common_idx]

        # Look at network metrics 1h BEFORE big moves
        # Shift bm_any forward by 12 bars (1h) to get "big move happening in next 1h"
        bm_next_1h = bm_aligned.shift(-12)

        bm_mask = bm_next_1h == 1
        no_bm_mask = bm_next_1h == 0

        for period_name, pmask in [("all", common_idx >= START),
                                    ("test", common_idx >= TEST_START)]:
            pidx = common_idx[pmask]
            if len(pidx) < 50:
                continue

            bm_m = bm_mask.loc[pidx].dropna()
            no_bm_m = no_bm_mask.loc[pidx].dropna()

            print(f"\n  {target_key} — {period_name}")
            print(f"  {'Metric':<25} {'Pre-BM':>10} {'Baseline':>10} {'Diff':>10}")
            print(f"  {'-'*58}")

            for metric in ["density", "avg_clustering", "n_components", "eigvec_dispersion"]:
                vals_bm = net_aligned.loc[bm_m[bm_m].index, metric].dropna()
                vals_no = net_aligned.loc[no_bm_m[no_bm_m].index, metric].dropna()

                if len(vals_bm) < 10 or len(vals_no) < 50:
                    continue

                mean_bm = vals_bm.mean()
                mean_no = vals_no.mean()
                diff = mean_bm - mean_no

                # Significance via Mann-Whitney U
                try:
                    stat, pval = sp_stats.mannwhitneyu(vals_bm, vals_no, alternative="two-sided")
                except Exception:
                    pval = 1.0

                print(f"  {metric:<25} {mean_bm:>10.4f} {mean_no:>10.4f} {diff:>+10.4f}  (p={pval:.4f})")

                results[f"{target_key}_{period_name}_{metric}"] = {
                    "mean_before_bm": mean_bm,
                    "mean_baseline": mean_no,
                    "diff": diff,
                    "pval": pval,
                }

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Cross-Sectional Market Structure Research")
    print("=" * 70)
    print(f"Data range:  {START} → {END}")
    print(f"Train:       {START} → {TRAIN_END}")
    print(f"Test OOS:    {TEST_START} → {END}")
    print(f"Min days:    {MIN_DAYS}")
    print()

    # §1: Discover and load
    symbols = discover_symbols()
    print(f"Discovered {len(symbols)} symbols with >= {MIN_DAYS} days data")
    if len(symbols) < 20:
        print("ERROR: Need at least 20 symbols for cross-sectional analysis")
        sys.exit(1)
    print(f"  Symbols: {symbols[:10]}... ({len(symbols)} total)")

    # §2: Build cross-sectional panel
    print(f"\nBuilding cross-sectional panel ({len(symbols)} symbols)...")
    t0 = time.monotonic()
    panels = build_cross_section_panel(symbols)
    print(f"Panel built in {time.monotonic()-t0:.0f}s")

    # §3: Compute market features
    mf = compute_market_features(panels)
    mf.to_csv(OUTPUT_DIR / "market_features.csv")
    print(f"Market features saved ({len(mf)} rows, {len(mf.columns)} cols)")

    # Run all 8 specs
    all_results = {}

    # Spec 4: Entropy (fastest)
    all_results["spec4_entropy"] = run_spec4_entropy(mf, panels)

    # Spec 2: Leadership
    all_results["spec2_leadership"] = run_spec2_leadership(panels)

    # Spec 7: Market Compression
    all_results["spec7_compression"] = run_spec7_compression(mf, panels)

    # Spec 3: Clustering
    all_results["spec3_clustering"] = run_spec3_clustering(mf, panels)

    # Spec 1: Market State Engine
    all_results["spec1_market_state"] = run_spec1_market_state(mf, panels)

    # Spec 5: Clustered OI Build-up
    all_results["spec5_clustered_oi"] = run_spec5_clustered_oi(mf, panels)

    # Spec 6: Extreme Co-Movement
    all_results["spec6_comovement"] = run_spec6_comovement(panels)

    # Spec 8: Network Structure
    all_results["spec8_network"] = run_spec8_network(panels)

    print("\n" + "=" * 70)
    print("ALL 8 SPECS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Cross-Sectional Market Structure Research Summary\n")
        f.write(f"Symbols: {len(symbols)}\n")
        f.write(f"Date range: {START} → {END}\n")
        f.write(f"Train/Test split: {TRAIN_END.date()}\n\n")
        for spec_name, spec_results in all_results.items():
            f.write(f"\n{'='*50}\n{spec_name}\n{'='*50}\n")
            if isinstance(spec_results, dict):
                for k, v in spec_results.items():
                    if isinstance(v, dict):
                        f.write(f"  {k}: {v}\n")
                    elif isinstance(v, pd.DataFrame):
                        f.write(f"  {k}: DataFrame ({v.shape})\n")
                    else:
                        f.write(f"  {k}: {v}\n")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
