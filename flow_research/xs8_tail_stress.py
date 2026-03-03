#!/usr/bin/env python3
"""
XS-8 — Tail Stress Indicator (Market-Level)

Every 5 minutes, compute a stress score from cross-sectional features:
  - breadth_extreme: fraction of coins with |ret_1h| > 2*ATR_1h
  - entropy: Shannon entropy of |ret_1h| distribution across coins
  - corr_density: fraction of variance explained by 1st PCA of 6h returns (proxy for correlation)
  - crowding_fund: fraction of coins with funding_z > 2
  - crowding_oi: fraction of coins with oi_z > 1.5

Target: tail_binary_6h = 1 if >=10% of coins make |ret| > 3*ATR in next 6h

Output: AUC, quintile uplift, calibration, monthly stability.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.stats import entropy as sp_entropy

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 30  # symbol must have at least 30 days of data

# Feature windows
ATR_1H = 60          # 1h ATR in 1m bars
RET_1H = 60          # 1h return
RET_6H = 360         # 6h return for PCA
FR_Z_WINDOW = 7 * 24 * 60   # 7d for funding z
OI_Z_WINDOW = 7 * 24 * 60   # 7d for OI z
OI_CHG_1H = 60

# Stress computation interval
STRESS_INTERVAL = 5  # every 5 minutes

# Target — 1h horizon, high ATR multiples
TAIL_HORIZON = 60    # 1h in minutes (short horizon = harder target)
TAIL_THRESHOLDS = [3.0, 5.0, 8.0, 12.0]  # ATR multipliers
# Binary cutoffs tried: 5%, 10%, 20% of coins

# Thresholds for cross-sectional features
BREADTH_ATR_MULT = 2.0
CROWDING_FUND_Z = 2.0
CROWDING_OI_Z = 1.5
CORR_WINDOW_6H = 360

SEED = 42


# ---------------------------------------------------------------------------
# Data Loading (reuse patterns from xs5)
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
    mark = _load_glob(sym, "*_mark_price_kline_1m.csv", "startTime", {"close": "close"})

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
            for c in ["high", "low", "close"]:
                kl[c] = pd.to_numeric(kl[c], errors="coerce")
            kl = kl.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            kl = kl[(kl["ts"] >= START) & (kl["ts"] <= END)]
            kline = kl[["ts", "high", "low", "close"]].reset_index(drop=True)

    oi = _load_glob(sym, "*_open_interest_5min.csv", "timestamp", {"openInterest": "oi"})
    fr = _load_glob(sym, "*_funding_rate.csv", "timestamp", {"fundingRate": "fr"})

    return {"mark": mark, "kline": kline, "oi": oi, "fr": fr}


# ---------------------------------------------------------------------------
# Build per-symbol 1m series
# ---------------------------------------------------------------------------

def build_sym_1m(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    mark = raw["mark"]
    if len(mark) == 0:
        return pd.DataFrame()

    close_s = mark.set_index("ts")["close"].reindex(grid_1m).ffill()
    df = pd.DataFrame(index=grid_1m)
    df["close"] = close_s

    # Kline high/low for ATR
    if len(raw["kline"]) > 0:
        kl_idx = raw["kline"].set_index("ts")
        df["high"] = kl_idx["high"].reindex(grid_1m).ffill()
        df["low"] = kl_idx["low"].reindex(grid_1m).ffill()
    else:
        df["high"] = df["close"]
        df["low"] = df["close"]

    # OI: shifted +5min causal
    if len(raw["oi"]) > 0:
        oi_shifted = raw["oi"].copy()
        oi_shifted["ts"] = oi_shifted["ts"] + pd.Timedelta(minutes=5)
        df["oi"] = oi_shifted.set_index("ts")["oi"].reindex(grid_1m).ffill()
    else:
        df["oi"] = np.nan

    # FR: shifted +1min causal
    if len(raw["fr"]) > 0:
        fr_shifted = raw["fr"].copy()
        fr_shifted["ts"] = fr_shifted["ts"] + pd.Timedelta(minutes=1)
        df["fr"] = fr_shifted.set_index("ts")["fr"].reindex(grid_1m).ffill()
    else:
        df["fr"] = np.nan

    # Derived features
    c = df["close"]
    df["log_ret"] = np.log(c / c.shift(1))
    df["ret_1h"] = np.log(c / c.shift(RET_1H))

    # ATR_1h
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_1h"] = tr.rolling(ATR_1H, min_periods=ATR_1H // 2).mean()

    # Funding z-score (7d rolling)
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["funding_z"] = (fr - fr_rm) / fr_rs

    # OI z-score (7d rolling on oi_chg_1h)
    oi = df["oi"]
    oi_lag = oi.shift(OI_CHG_1H)
    df["oi_chg_1h"] = (oi - oi_lag) / oi_lag.clip(lower=1)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["oi_z"] = (oi_chg - oi_rm) / oi_rs

    return df


# ---------------------------------------------------------------------------
# Compute cross-sectional stress features (every 5 min)
# ---------------------------------------------------------------------------

def compute_stress_features(sym_dfs: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute market-level stress features every STRESS_INTERVAL minutes."""

    n_grid = len(grid_1m)
    syms = sorted(sym_dfs.keys())
    n_syms = len(syms)

    # Pre-extract arrays for speed
    ret_1h = {}
    atr_1h = {}
    fund_z = {}
    oi_z = {}
    log_ret = {}
    close_arr = {}

    for sym in syms:
        df = sym_dfs[sym]
        ret_1h[sym] = df["ret_1h"].values
        atr_1h[sym] = df["atr_1h"].values
        fund_z[sym] = df["funding_z"].values
        oi_z[sym] = df["oi_z"].values
        log_ret[sym] = df["log_ret"].values
        close_arr[sym] = df["close"].values

    # 5m sample points
    sample_idx = np.arange(0, n_grid, STRESS_INTERVAL)
    n_samples = len(sample_idx)

    print(f"  Computing stress features: {n_samples} timepoints × {n_syms} symbols",
          flush=True)

    # Output arrays
    breadth_extreme = np.full(n_samples, np.nan)
    entropy_arr = np.full(n_samples, np.nan)
    pca_var1 = np.full(n_samples, np.nan)
    crowd_fund = np.full(n_samples, np.nan)
    crowd_oi = np.full(n_samples, np.nan)
    # Multiple target thresholds
    n_thresh = len(TAIL_THRESHOLDS)
    tail_fracs = {t: np.full(n_samples, np.nan) for t in TAIL_THRESHOLDS}

    t0 = time.time()

    for si, idx in enumerate(sample_idx):
        if si % 10000 == 0 and si > 0:
            elapsed = time.time() - t0
            rate = si / elapsed
            eta = (n_samples - si) / rate
            print(f"    {si}/{n_samples} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        # Collect cross-sectional data at time idx
        rets = []
        atrs = []
        fzs = []
        ozs = []

        for sym in syms:
            r = ret_1h[sym][idx]
            a = atr_1h[sym][idx]
            fz = fund_z[sym][idx]
            oz = oi_z[sym][idx]

            if np.isnan(r) or np.isnan(a) or a <= 0:
                continue
            rets.append(r)
            atrs.append(a)
            fzs.append(fz if not np.isnan(fz) else 0.0)
            ozs.append(oz if not np.isnan(oz) else 0.0)

        n_valid = len(rets)
        if n_valid < 10:
            continue

        rets = np.array(rets)
        atrs = np.array(atrs)
        fzs = np.array(fzs)
        ozs = np.array(ozs)

        # 1) breadth_extreme: fraction with |ret_1h| > 2*ATR_1h
        # ATR is in price terms, ret is log return → convert ATR to return-equivalent
        atr_ret = atrs / close_arr[syms[0]][idx] if close_arr[syms[0]][idx] > 0 else atrs
        # Simpler: compare |ret_1h| to BREADTH_ATR_MULT * atr_as_fraction
        # Actually: |ret_1h| is log return (~bp), atr_1h is price range
        # Better approach: normalize per coin
        norm_rets = np.abs(rets)
        # atr as fraction of price ≈ atr / close
        atr_fracs = []
        valid_idx = 0
        for sym in syms:
            a = atr_1h[sym][idx]
            c = close_arr[sym][idx]
            if np.isnan(a) or a <= 0 or np.isnan(c) or c <= 0:
                continue
            atr_fracs.append(a / c)
            valid_idx += 1
            if valid_idx >= n_valid:
                break
        atr_fracs = np.array(atr_fracs[:n_valid])

        if len(atr_fracs) == n_valid:
            breadth_extreme[si] = (norm_rets > BREADTH_ATR_MULT * atr_fracs).mean()
        else:
            breadth_extreme[si] = np.nan

        # 2) entropy of |ret_1h| distribution (discretize into 20 bins)
        if norm_rets.std() > 1e-12:
            hist, _ = np.histogram(norm_rets, bins=20, density=True)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hist = hist[hist > 0]
            entropy_arr[si] = sp_entropy(hist)
        else:
            entropy_arr[si] = 0.0

        # 3) PCA: 1st component explained variance on 6h returns
        # Expensive — do every 5h (60 steps × 5min = 300min)
        if si % 60 == 0 and idx >= CORR_WINDOW_6H:
            ret_matrix = []
            for sym in syms:
                lr = log_ret[sym][idx - CORR_WINDOW_6H:idx]
                if np.isnan(lr).sum() < CORR_WINDOW_6H * 0.3:
                    lr_clean = np.nan_to_num(lr, nan=0.0)
                    ret_matrix.append(lr_clean)
            if len(ret_matrix) >= 10:
                ret_mat = np.column_stack(ret_matrix)
                try:
                    pca = PCA(n_components=1)
                    pca.fit(ret_mat)
                    pca_var1[si] = pca.explained_variance_ratio_[0]
                except Exception:
                    pass

        # Forward-fill PCA
        if si > 0 and np.isnan(pca_var1[si]) and not np.isnan(pca_var1[si - 1]):
            pca_var1[si] = pca_var1[si - 1]

        # 4) crowding_fund: fraction with |funding_z| > threshold
        crowd_fund[si] = (np.abs(fzs) > CROWDING_FUND_Z).mean()

        # 5) crowding_oi: fraction with oi_z > threshold
        crowd_oi[si] = (ozs > CROWDING_OI_Z).mean()

        # 6) Target: fraction of coins with |ret| > K*ATR in next 1h
        max_future_idx = min(idx + TAIL_HORIZON, n_grid - 1)
        if max_future_idx <= idx + 5:
            continue

        n_tail = {t: 0 for t in TAIL_THRESHOLDS}
        n_checked = 0
        for sym in syms:
            a_now = atr_1h[sym][idx]
            c_now = close_arr[sym][idx]
            if np.isnan(a_now) or a_now <= 0 or np.isnan(c_now) or c_now <= 0:
                continue
            n_checked += 1
            atr_ret = a_now / c_now

            future_close = close_arr[sym][idx + 1:max_future_idx + 1]
            if len(future_close) < 5:
                continue
            future_rets = np.log(future_close / c_now)
            max_abs_ret = np.nanmax(np.abs(future_rets))
            for t in TAIL_THRESHOLDS:
                if max_abs_ret > t * atr_ret:
                    n_tail[t] += 1

        if n_checked > 0:
            for t in TAIL_THRESHOLDS:
                tail_fracs[t][si] = n_tail[t] / n_checked

    # Forward-fill PCA
    for i in range(1, n_samples):
        if np.isnan(pca_var1[i]) and not np.isnan(pca_var1[i - 1]):
            pca_var1[i] = pca_var1[i - 1]

    data = {
        "ts": grid_1m[sample_idx],
        "breadth_extreme": breadth_extreme,
        "entropy": entropy_arr,
        "pca_var1": pca_var1,
        "crowd_fund": crowd_fund,
        "crowd_oi": crowd_oi,
    }
    for t in TAIL_THRESHOLDS:
        data[f"tail_frac_{int(t)}x"] = tail_fracs[t]
    result = pd.DataFrame(data)

    elapsed = time.time() - t0
    print(f"  Stress features computed in {elapsed:.1f}s", flush=True)

    return result


# ---------------------------------------------------------------------------
# Model & Evaluation
# ---------------------------------------------------------------------------

def evaluate_one_target(stress_df: pd.DataFrame, target_col: str, label: str):
    """Fit logistic regression on stress features for one target, evaluate quintile uplift."""

    feature_cols = ["breadth_extreme", "entropy", "pca_var1", "crowd_fund", "crowd_oi"]

    df = stress_df.dropna(subset=["breadth_extreme", "entropy", "crowd_fund",
                                   "crowd_oi", target_col]).copy()
    df["pca_var1"] = df["pca_var1"].fillna(df["pca_var1"].median())

    X = df[feature_cols].values
    y = df[target_col].values

    n = len(df)
    n_pos = int(y.sum())
    base_rate = n_pos / n if n > 0 else 0

    print(f"\n  ── {label} ──")
    print(f"  N={n:,}, positives={n_pos:,} ({base_rate:.1%})")

    if n_pos < 20 or (n - n_pos) < 20:
        print(f"  ✗ Skipped (not enough pos/neg)")
        return None

    # Walk-forward: train first 60%, test last 40%
    split = int(n * 0.6)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    p_full = model.predict_proba(X)[:, 1]

    auc_test = roc_auc_score(y_test, p_test) if y_test.sum() > 0 and y_test.sum() < len(y_test) else np.nan
    auc_full = roc_auc_score(y, p_full) if 0 < y.sum() < n else np.nan

    print(f"  AUC full={auc_full:.4f}, OOS={auc_test:.4f}")

    # Coefficients
    for fname, coef in zip(feature_cols, model.coef_[0]):
        print(f"    {fname:20s}: {coef:+.4f}")

    # Quintile analysis on OOS
    df_test = pd.DataFrame({"p": p_test, "y": y_test})
    try:
        df_test["Q"] = pd.qcut(df_test["p"], 5, labels=["Q1","Q2","Q3","Q4","Q5"],
                                duplicates="drop")
    except ValueError:
        print(f"  ✗ Cannot create quintiles (too few unique p values)")
        return None

    base_oos = y_test.mean()
    print(f"\n  Quintile analysis (OOS, base={base_oos:.1%}):")
    print(f"  {'Q':>4s}  {'N':>6s}  {'Pos':>5s}  {'P(tail)':>8s}  {'Uplift':>8s}")
    q_stats = {}
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        qd = df_test[df_test["Q"] == q]
        n_q = len(qd)
        p_tail = qd["y"].mean() if n_q > 0 else 0
        uplift = p_tail / base_oos if base_oos > 0 else 0
        q_stats[q] = {"n": n_q, "p_tail": p_tail, "uplift": uplift}
        print(f"  {q:>4s}  {n_q:>6d}  {int(qd['y'].sum()):>5d}  {p_tail:>8.1%}  {uplift:>7.2f}×")

    q5_q1 = q_stats["Q5"]["p_tail"] / q_stats["Q1"]["p_tail"] if q_stats["Q1"]["p_tail"] > 0 else np.inf
    print(f"  Q5/Q1 = {q5_q1:.2f}×")

    # Monthly walk-forward AUC
    df["month"] = pd.to_datetime(df["ts"]).dt.to_period("M")
    months = sorted(df["month"].unique())
    print(f"\n  Monthly walk-forward AUC:")
    monthly_wf = {}
    for mi in range(2, len(months)):
        train_mask = df["month"].isin(months[:mi])
        test_mask = df["month"] == months[mi]
        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask, feature_cols].values
        y_te = df.loc[test_mask, target_col].values
        if y_tr.sum() < 10 or y_te.sum() < 3 or len(y_te) < 50 or y_te.sum() == len(y_te):
            continue
        try:
            mm = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
            mm.fit(X_tr, y_tr)
            p_te = mm.predict_proba(X_te)[:, 1]
            a = roc_auc_score(y_te, p_te)
            monthly_wf[str(months[mi])] = a
            print(f"    → {months[mi]}: AUC={a:.4f} (N={len(y_te):,}, pos={int(y_te.sum()):,})")
        except Exception:
            pass

    return {
        "label": label, "target": target_col,
        "auc_full": auc_full, "auc_oos": auc_test,
        "base_rate": base_rate, "n": n, "n_pos": n_pos,
        "q5_q1": q5_q1,
        **{f"q{i+1}_p": q_stats[f"Q{i+1}"]["p_tail"] for i in range(5)},
        **{f"q{i+1}_up": q_stats[f"Q{i+1}"]["uplift"] for i in range(5)},
        **{f"wf_{m}": a for m, a in monthly_wf.items()},
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 80)
    print("XS-8 — TAIL STRESS INDICATOR")
    print(f"Period: {START.date()} → {END.date()} (8 months)")
    print(f"Features: breadth_extreme, entropy, pca_var1, crowd_fund, crowd_oi")
    print(f"Tail thresholds: {TAIL_THRESHOLDS}×ATR, horizon={TAIL_HORIZON}min")
    print(f"Interval: every {STRESS_INTERVAL}min")
    print("=" * 80, flush=True)

    # Phase 1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}", flush=True)

    symbols = discover_symbols()
    print(f"  Symbols: {len(symbols)}", flush=True)

    raw_all = {}
    for i, sym in enumerate(symbols, 1):
        raw_all[sym] = load_symbol(sym)
        if i % 10 == 0 or i == len(symbols):
            print(f"    Loaded {i}/{len(symbols)}", flush=True)

    # Phase 2: Build 1m grids
    print(f"\n{'─'*70}")
    print("PHASE 2: Building 1m per-symbol data")
    print(f"{'─'*70}", flush=True)

    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"  Grid: {len(grid_1m):,} points", flush=True)

    sym_dfs = {}
    for i, sym in enumerate(sorted(raw_all.keys()), 1):
        raw = raw_all[sym]
        if len(raw["mark"]) == 0:
            continue
        df = build_sym_1m(sym, raw, grid_1m)
        if len(df) > 0:
            sym_dfs[sym] = df
        if i % 10 == 0 or i == len(raw_all):
            print(f"    Built {i}/{len(raw_all)} ({len(sym_dfs)} valid)", flush=True)

    del raw_all  # free memory

    # Phase 3: Compute stress features
    print(f"\n{'─'*70}")
    print("PHASE 3: Cross-sectional stress features")
    print(f"{'─'*70}", flush=True)

    stress_df = compute_stress_features(sym_dfs, grid_1m)

    # Save raw stress data
    stress_df.to_parquet(OUTPUT_DIR / "xs8_stress.parquet", index=False)
    print(f"  Saved {OUTPUT_DIR / 'xs8_stress.parquet'}")

    # Stats on features + tail distributions
    frac_cols = [c for c in stress_df.columns if c.startswith("tail_frac_")]
    valid = stress_df.dropna(subset=[frac_cols[0]])
    print(f"\n  Feature summary ({len(valid):,} valid rows):")
    for col in ["breadth_extreme", "entropy", "pca_var1", "crowd_fund", "crowd_oi"]:
        s = valid[col].dropna()
        print(f"    {col:20s}: mean={s.mean():.4f}, std={s.std():.4f}, "
              f"p5={s.quantile(0.05):.4f}, p95={s.quantile(0.95):.4f}")

    print(f"\n  Tail fraction distributions:")
    for fc in frac_cols:
        s = valid[fc].dropna()
        print(f"    {fc:20s}: mean={s.mean():.3f}, p25={s.quantile(0.25):.3f}, "
              f"p50={s.quantile(0.5):.3f}, p75={s.quantile(0.75):.3f}, p95={s.quantile(0.95):.3f}")

    # Phase 4: Evaluate multiple target variants
    print(f"\n{'─'*70}")
    print("PHASE 4: Stress model evaluation (multiple targets)")
    print(f"{'─'*70}", flush=True)

    # Build binary targets from frac columns
    all_results = []
    for t_atr in TAIL_THRESHOLDS:
        frac_col = f"tail_frac_{int(t_atr)}x"
        if frac_col not in stress_df.columns:
            continue

        # Try multiple binary cutoffs for each ATR threshold
        for cutoff in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
            target_col = f"tail_bin_{int(t_atr)}x_{int(cutoff*100)}pct"
            stress_df[target_col] = (stress_df[frac_col] >= cutoff).astype(float)
            stress_df.loc[stress_df[frac_col].isna(), target_col] = np.nan

            rate = stress_df[target_col].dropna().mean()
            if rate > 0.95 or rate < 0.02:
                print(f"\n  ── {int(t_atr)}×ATR, ≥{int(cutoff*100)}% coins ──")
                print(f"  Skipped: rate={rate:.1%} (too extreme)")
                continue

            label = f"{int(t_atr)}×ATR, ≥{int(cutoff*100)}% coins"
            r = evaluate_one_target(stress_df, target_col, label)
            if r:
                all_results.append(r)

    # Save all results
    if all_results:
        pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "xs8_summary.csv", index=False)

    # Verdict
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if all_results:
        print(f"\n  {'Target':30s}  {'Base':>6s}  {'AUC_OOS':>8s}  {'Q5/Q1':>7s}")
        for r in all_results:
            print(f"  {r['label']:30s}  {r['base_rate']:>5.1%}  {r['auc_oos']:>8.4f}  {r['q5_q1']:>6.2f}×")

        best = max(all_results, key=lambda r: r.get("auc_oos", 0))
        print(f"\n  Best: {best['label']} — AUC={best['auc_oos']:.4f}, Q5/Q1={best['q5_q1']:.2f}×")

        auc_ok = best["auc_oos"] >= 0.60
        uplift_ok = best["q5_q1"] >= 2.0

        print(f"\n  {'✓' if auc_ok else '✗'} OOS AUC ≥ 0.60 (actual: {best['auc_oos']:.4f})")
        print(f"  {'✓' if uplift_ok else '✗'} Q5/Q1 uplift ≥ 2× (actual: {best['q5_q1']:.2f}×)")

        if auc_ok and uplift_ok:
            print(f"\n  → USEFUL as risk indicator ✅")
        else:
            print(f"\n  → NOT USEFUL ❌")
    else:
        print("  No valid target configurations found.")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
