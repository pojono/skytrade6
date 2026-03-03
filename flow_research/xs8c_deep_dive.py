#!/usr/bin/env python3
"""
XS-8c — Deep Dive: Fragility Indicator

Four research angles from XS-8 crowd_oi discovery:

  A) Directional asymmetry — up-tail vs down-tail separately
  B) BTC-only tail prediction from crowd_oi
  C) Conditional interaction — crowd_oi × pca_var1 matrix
  D) Multi-horizon comparison — 30m, 1h, 2h, 4h

Reloads raw data to compute signed (directional) targets.
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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs8c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 30

ATR_1H = 60
RET_1H = 60
RET_6H = 360
FR_Z_WINDOW = 7 * 24 * 60
OI_Z_WINDOW = 7 * 24 * 60
OI_CHG_1H = 60

STRESS_INTERVAL = 5
BREADTH_ATR_MULT = 2.0
CROWDING_FUND_Z = 2.0
CROWDING_OI_Z = 1.5
CORR_WINDOW_6H = 360

# Multi-horizon targets (in minutes)
HORIZONS = [30, 60, 120, 240]
ATR_MULT = 12.0       # focus on 12×ATR (best from XS-8)
FRAC_CUTOFF = 0.10    # ≥10% of coins

SEED = 42
FEATURE_COLS = ["breadth_extreme", "entropy", "pca_var1", "crowd_fund", "crowd_oi"]

# ---------------------------------------------------------------------------
# Data Loading (from xs8_tail_stress.py)
# ---------------------------------------------------------------------------

def discover_symbols():
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


def _load_glob(sym, pattern, ts_col, val_cols):
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


def load_symbol(sym):
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


def build_sym_1m(sym, raw, grid_1m):
    mark = raw["mark"]
    if len(mark) == 0:
        return pd.DataFrame()
    close_s = mark.set_index("ts")["close"].reindex(grid_1m).ffill()
    df = pd.DataFrame(index=grid_1m)
    df["close"] = close_s
    if len(raw["kline"]) > 0:
        kl_idx = raw["kline"].set_index("ts")
        df["high"] = kl_idx["high"].reindex(grid_1m).ffill()
        df["low"] = kl_idx["low"].reindex(grid_1m).ffill()
    else:
        df["high"] = df["close"]
        df["low"] = df["close"]
    if len(raw["oi"]) > 0:
        oi_shifted = raw["oi"].copy()
        oi_shifted["ts"] = oi_shifted["ts"] + pd.Timedelta(minutes=5)
        df["oi"] = oi_shifted.set_index("ts")["oi"].reindex(grid_1m).ffill()
    else:
        df["oi"] = np.nan
    if len(raw["fr"]) > 0:
        fr_shifted = raw["fr"].copy()
        fr_shifted["ts"] = fr_shifted["ts"] + pd.Timedelta(minutes=1)
        df["fr"] = fr_shifted.set_index("ts")["fr"].reindex(grid_1m).ffill()
    else:
        df["fr"] = np.nan
    c = df["close"]
    df["log_ret"] = np.log(c / c.shift(1))
    df["ret_1h"] = np.log(c / c.shift(RET_1H))
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_1h"] = tr.rolling(ATR_1H, min_periods=ATR_1H // 2).mean()
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW, min_periods=FR_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["funding_z"] = (fr - fr_rm) / fr_rs
    oi = df["oi"]
    oi_lag = oi.shift(OI_CHG_1H)
    df["oi_chg_1h"] = (oi - oi_lag) / oi_lag.clip(lower=1)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW, min_periods=OI_Z_WINDOW // 4).std().clip(lower=1e-12)
    df["oi_z"] = (oi_chg - oi_rm) / oi_rs
    return df


# ---------------------------------------------------------------------------
# Extended stress features with directional + multi-horizon targets
# ---------------------------------------------------------------------------

def compute_extended_features(sym_dfs, grid_1m):
    """Compute features + directional targets at multiple horizons."""

    n_grid = len(grid_1m)
    syms = sorted(sym_dfs.keys())
    n_syms = len(syms)

    # Pre-extract arrays
    ret_1h = {}; atr_1h_arr = {}; fund_z = {}; oi_z_arr = {}
    log_ret = {}; close_arr = {}

    for sym in syms:
        df = sym_dfs[sym]
        ret_1h[sym] = df["ret_1h"].values
        atr_1h_arr[sym] = df["atr_1h"].values
        fund_z[sym] = df["funding_z"].values
        oi_z_arr[sym] = df["oi_z"].values
        log_ret[sym] = df["log_ret"].values
        close_arr[sym] = df["close"].values

    sample_idx = np.arange(0, n_grid, STRESS_INTERVAL)
    n_samples = len(sample_idx)

    print(f"  Computing extended features: {n_samples} timepoints × {n_syms} symbols",
          flush=True)

    # Feature arrays
    breadth_extreme = np.full(n_samples, np.nan)
    entropy_arr = np.full(n_samples, np.nan)
    pca_var1 = np.full(n_samples, np.nan)
    crowd_fund = np.full(n_samples, np.nan)
    crowd_oi = np.full(n_samples, np.nan)

    # Target arrays: per horizon × direction
    # Directions: any (|ret|), up (ret>0), down (ret<0)
    targets = {}
    for h in HORIZONS:
        for direction in ["any", "up", "down"]:
            key = f"tail_{direction}_{h}m"
            targets[key] = np.full(n_samples, np.nan)

    # BTC-specific targets
    btc_targets = {}
    btc_sym = None
    for sym in syms:
        if "BTC" in sym and "USDT" in sym:
            btc_sym = sym
            break
    if btc_sym:
        for h in HORIZONS:
            for direction in ["any", "up", "down"]:
                key = f"btc_{direction}_{h}m"
                btc_targets[key] = np.full(n_samples, np.nan)
        print(f"  BTC symbol found: {btc_sym}", flush=True)
    else:
        print(f"  WARNING: No BTC symbol found!", flush=True)

    t0 = time.time()

    for si, idx in enumerate(sample_idx):
        if si % 5000 == 0 and si > 0:
            elapsed = time.time() - t0
            rate = si / elapsed
            eta = (n_samples - si) / rate
            print(f"    {si}/{n_samples} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        # Collect cross-sectional data
        rets = []; atr_fracs = []; fzs = []; ozs = []

        for sym in syms:
            r = ret_1h[sym][idx]
            a = atr_1h_arr[sym][idx]
            c = close_arr[sym][idx]
            fz = fund_z[sym][idx]
            oz = oi_z_arr[sym][idx]

            if np.isnan(r) or np.isnan(a) or a <= 0 or np.isnan(c) or c <= 0:
                continue
            rets.append(r)
            atr_fracs.append(a / c)
            fzs.append(fz if not np.isnan(fz) else 0.0)
            ozs.append(oz if not np.isnan(oz) else 0.0)

        n_valid = len(rets)
        if n_valid < 10:
            continue

        rets = np.array(rets)
        atr_fracs_np = np.array(atr_fracs)
        fzs = np.array(fzs)
        ozs = np.array(ozs)

        # 1) breadth_extreme
        breadth_extreme[si] = (np.abs(rets) > BREADTH_ATR_MULT * atr_fracs_np).mean()

        # 2) entropy
        norm_rets = np.abs(rets)
        if norm_rets.std() > 1e-12:
            hist, _ = np.histogram(norm_rets, bins=20, density=True)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hist = hist[hist > 0]
            entropy_arr[si] = sp_entropy(hist)
        else:
            entropy_arr[si] = 0.0

        # 3) PCA (every 5h)
        if si % 60 == 0 and idx >= CORR_WINDOW_6H:
            ret_matrix = []
            for sym in syms:
                lr = log_ret[sym][idx - CORR_WINDOW_6H:idx]
                if np.isnan(lr).sum() < CORR_WINDOW_6H * 0.3:
                    ret_matrix.append(np.nan_to_num(lr, nan=0.0))
            if len(ret_matrix) >= 10:
                try:
                    pca = PCA(n_components=1)
                    pca.fit(np.column_stack(ret_matrix))
                    pca_var1[si] = pca.explained_variance_ratio_[0]
                except Exception:
                    pass

        if si > 0 and np.isnan(pca_var1[si]) and not np.isnan(pca_var1[si - 1]):
            pca_var1[si] = pca_var1[si - 1]

        # 4) crowding
        crowd_fund[si] = (np.abs(fzs) > CROWDING_FUND_Z).mean()
        crowd_oi[si] = (ozs > CROWDING_OI_Z).mean()

        # 5) Multi-horizon directional targets
        for h in HORIZONS:
            max_fi = min(idx + h, n_grid - 1)
            if max_fi <= idx + 5:
                continue

            n_tail_any = 0; n_tail_up = 0; n_tail_down = 0; n_checked = 0

            for sym in syms:
                a_now = atr_1h_arr[sym][idx]
                c_now = close_arr[sym][idx]
                if np.isnan(a_now) or a_now <= 0 or np.isnan(c_now) or c_now <= 0:
                    continue
                n_checked += 1
                atr_ret = a_now / c_now

                future_close = close_arr[sym][idx + 1:max_fi + 1]
                if len(future_close) < 5:
                    continue
                future_rets = np.log(future_close / c_now)
                valid_mask = ~np.isnan(future_rets)
                if valid_mask.sum() < 3:
                    continue
                fr_valid = future_rets[valid_mask]

                max_abs = np.max(np.abs(fr_valid))
                max_up = np.max(fr_valid)
                min_down = np.min(fr_valid)

                threshold = ATR_MULT * atr_ret

                if max_abs > threshold:
                    n_tail_any += 1
                if max_up > threshold:
                    n_tail_up += 1
                if -min_down > threshold:
                    n_tail_down += 1

            if n_checked > 0:
                targets[f"tail_any_{h}m"][si] = n_tail_any / n_checked
                targets[f"tail_up_{h}m"][si] = n_tail_up / n_checked
                targets[f"tail_down_{h}m"][si] = n_tail_down / n_checked

        # 6) BTC-specific targets
        if btc_sym:
            a_btc = atr_1h_arr[btc_sym][idx]
            c_btc = close_arr[btc_sym][idx]
            if not (np.isnan(a_btc) or a_btc <= 0 or np.isnan(c_btc) or c_btc <= 0):
                atr_ret_btc = a_btc / c_btc
                for h in HORIZONS:
                    max_fi = min(idx + h, n_grid - 1)
                    if max_fi <= idx + 5:
                        continue
                    fc = close_arr[btc_sym][idx + 1:max_fi + 1]
                    if len(fc) < 5:
                        continue
                    fr_btc = np.log(fc / c_btc)
                    valid = ~np.isnan(fr_btc)
                    if valid.sum() < 3:
                        continue
                    fr_v = fr_btc[valid]
                    # BTC tail: use lower multiplier (2× for BTC since it's less volatile)
                    for mult, label in [(2.0, "2x"), (3.0, "3x"), (5.0, "5x")]:
                        thr = mult * atr_ret_btc
                        btc_targets.setdefault(f"btc_any_{h}m_{label}", np.full(n_samples, np.nan))
                        btc_targets.setdefault(f"btc_up_{h}m_{label}", np.full(n_samples, np.nan))
                        btc_targets.setdefault(f"btc_down_{h}m_{label}", np.full(n_samples, np.nan))
                        btc_targets[f"btc_any_{h}m_{label}"][si] = float(np.max(np.abs(fr_v)) > thr)
                        btc_targets[f"btc_up_{h}m_{label}"][si] = float(np.max(fr_v) > thr)
                        btc_targets[f"btc_down_{h}m_{label}"][si] = float(-np.min(fr_v) > thr)

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
    data.update(targets)
    data.update(btc_targets)

    result = pd.DataFrame(data)
    elapsed = time.time() - t0
    print(f"  Extended features computed in {elapsed:.1f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_auc_quintile(X_train, y_train, X_test, y_test, label=""):
    """Quick LogReg eval: AUC + Q5/Q1."""
    if y_train.sum() < 5 or y_test.sum() < 3 or y_test.sum() == len(y_test):
        return None
    try:
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        lr.fit(X_train, y_train)
        p = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
    except Exception:
        return None

    df_t = pd.DataFrame({"p": p, "y": y_test})
    try:
        df_t["Q"] = pd.qcut(df_t["p"], 5, labels=["Q1","Q2","Q3","Q4","Q5"],
                             duplicates="drop")
    except ValueError:
        return {"label": label, "auc": auc, "q5_q1": np.nan, "q1": np.nan, "q5": np.nan,
                "base": y_test.mean(), "n": len(y_test), "coefs": lr.coef_[0]}

    qs = {}
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        qd = df_t[df_t["Q"] == q]
        qs[q] = qd["y"].mean() if len(qd) > 0 else 0.0

    q5q1 = qs["Q5"] / qs["Q1"] if qs["Q1"] > 0 else np.inf

    return {"label": label, "auc": auc, "q5_q1": q5q1,
            "q1": qs["Q1"], "q5": qs["Q5"],
            "base": y_test.mean(), "n": len(y_test),
            "coefs": lr.coef_[0]}


def monthly_wf(df, fcols, tcol):
    """Monthly expanding walk-forward AUCs."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["ts"]).dt.to_period("M")
    months = sorted(df["month"].unique())
    aucs = []
    for mi in range(2, len(months)):
        train = df[df["month"].isin(months[:mi])]
        test = df[df["month"] == months[mi]]
        Xtr = train[fcols].values; ytr = train[tcol].values
        Xte = test[fcols].values; yte = test[tcol].values
        if ytr.sum() < 10 or yte.sum() < 3 or len(yte) < 50 or yte.sum() == len(yte):
            continue
        try:
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
            lr.fit(Xtr, ytr)
            p = lr.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass
    return aucs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0_all = time.time()

    print("=" * 80)
    print("XS-8c — FRAGILITY DEEP DIVE")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Angles: directional asymmetry, BTC-only, conditional, multi-horizon")
    print("=" * 80, flush=True)

    # ── Check for cached extended features ──
    cache_path = OUTPUT_DIR / "xs8c_extended.parquet"
    if cache_path.exists():
        print(f"\nLoading cached features from {cache_path}...")
        stress_df = pd.read_parquet(cache_path)
        print(f"  {len(stress_df):,} rows, {len(stress_df.columns)} columns")
    else:
        # Phase 1: Load data
        print(f"\n{'─'*70}")
        print("PHASE 0: Data loading + feature computation")
        print(f"{'─'*70}", flush=True)

        symbols = discover_symbols()
        print(f"  Symbols: {len(symbols)}", flush=True)

        raw_all = {}
        for i, sym in enumerate(symbols, 1):
            raw_all[sym] = load_symbol(sym)
            if i % 10 == 0 or i == len(symbols):
                print(f"    Loaded {i}/{len(symbols)}", flush=True)

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

        del raw_all

        stress_df = compute_extended_features(sym_dfs, grid_1m)
        stress_df.to_parquet(cache_path, index=False)
        print(f"  Saved {cache_path}")

    # ── Prepare data ──
    df = stress_df.dropna(subset=FEATURE_COLS).copy()
    df["pca_var1"] = df["pca_var1"].fillna(df["pca_var1"].median())
    n = len(df)
    split = int(n * 0.6)
    print(f"\n  Valid rows: {n:,}, train: {split:,}, test: {n-split:,}")

    X_train = df[FEATURE_COLS].values[:split]
    X_test = df[FEATURE_COLS].values[split:]

    # ══════════════════════════════════════════════════════════════════════════
    # ANGLE A: DIRECTIONAL ASYMMETRY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("ANGLE A: DIRECTIONAL ASYMMETRY (up-tail vs down-tail)")
    print(f"{'='*80}", flush=True)

    print(f"\n  12×ATR, ≥10% of coins, LogReg:")
    print(f"  {'Target':30s}  {'Base':>6s}  {'AUC':>7s}  {'Q5/Q1':>7s}  {'Q1':>7s}  {'Q5':>7s}")

    dir_results = []
    for h in HORIZONS:
        for direction in ["any", "up", "down"]:
            frac_col = f"tail_{direction}_{h}m"
            if frac_col not in df.columns:
                continue
            tcol = f"bin_{frac_col}"
            df[tcol] = (df[frac_col] >= FRAC_CUTOFF).astype(float)
            df.loc[df[frac_col].isna(), tcol] = np.nan

            valid = df[tcol].dropna()
            rate = valid.mean()
            if rate > 0.95 or rate < 0.02:
                print(f"  {frac_col:30s}  {rate:>5.1%}  SKIP (extreme rate)")
                continue

            y_tr = df[tcol].values[:split]
            y_te = df[tcol].values[split:]
            mask_tr = ~np.isnan(y_tr); mask_te = ~np.isnan(y_te)

            r = eval_auc_quintile(
                X_train[mask_tr], y_tr[mask_tr],
                X_test[mask_te], y_te[mask_te],
                label=frac_col)

            if r:
                print(f"  {frac_col:30s}  {r['base']:>5.1%}  {r['auc']:>7.4f}  "
                      f"{r['q5_q1']:>6.2f}×  {r['q1']:>6.1%}  {r['q5']:>6.1%}")
                r["horizon"] = h
                r["direction"] = direction
                dir_results.append(r)

    # Summary
    if dir_results:
        print(f"\n  ── Directional asymmetry summary (1h horizon) ──")
        for d in ["any", "up", "down"]:
            matches = [r for r in dir_results if r["direction"] == d and r["horizon"] == 60]
            if matches:
                r = matches[0]
                print(f"    {d:6s}: AUC={r['auc']:.4f}, Q5/Q1={r['q5_q1']:.2f}×, "
                      f"base={r['base']:.1%}")
                print(f"           crowd_oi coef = {r['coefs'][4]:+.3f}")

        # Check if down-tail has stronger signal
        down_1h = [r for r in dir_results if r["direction"] == "down" and r["horizon"] == 60]
        up_1h = [r for r in dir_results if r["direction"] == "up" and r["horizon"] == 60]
        if down_1h and up_1h:
            d_auc = down_1h[0]["auc"]; u_auc = up_1h[0]["auc"]
            print(f"\n  Down vs Up AUC gap: {d_auc - u_auc:+.4f}")
            if d_auc > u_auc + 0.01:
                print(f"  → Fragility is STRONGER on downside ✅")
            elif u_auc > d_auc + 0.01:
                print(f"  → Fragility is STRONGER on upside ⚠️")
            else:
                print(f"  → Fragility is SYMMETRIC (no directional bias)")

    # ══════════════════════════════════════════════════════════════════════════
    # ANGLE B: BTC-ONLY TAIL PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("ANGLE B: BTC-ONLY TAIL PREDICTION FROM CROSS-SECTIONAL FEATURES")
    print(f"{'='*80}", flush=True)

    btc_cols = [c for c in df.columns if c.startswith("btc_")]
    if btc_cols:
        print(f"\n  BTC tail columns: {len(btc_cols)}")
        print(f"\n  {'Target':35s}  {'Base':>6s}  {'AUC':>7s}  {'Q5/Q1':>7s}  "
              f"{'crowd_oi':>10s}")

        btc_results = []
        for col in sorted(btc_cols):
            valid = df[col].dropna()
            if len(valid) < 1000:
                continue
            rate = valid.mean()
            if rate > 0.95 or rate < 0.02:
                continue

            y_tr = df[col].values[:split]
            y_te = df[col].values[split:]
            mask_tr = ~np.isnan(y_tr); mask_te = ~np.isnan(y_te)

            if mask_tr.sum() < 100 or mask_te.sum() < 50:
                continue

            r = eval_auc_quintile(
                X_train[mask_tr], y_tr[mask_tr],
                X_test[mask_te], y_te[mask_te],
                label=col)

            if r:
                oi_coef = r["coefs"][4]
                print(f"  {col:35s}  {r['base']:>5.1%}  {r['auc']:>7.4f}  "
                      f"{r['q5_q1']:>6.2f}×  {oi_coef:>+10.3f}")
                btc_results.append(r)

        if btc_results:
            best_btc = max(btc_results, key=lambda r: r["auc"])
            print(f"\n  Best BTC target: {best_btc['label']} "
                  f"— AUC={best_btc['auc']:.4f}, Q5/Q1={best_btc['q5_q1']:.2f}×")

            # Monthly stability for best
            if best_btc["auc"] > 0.55:
                aucs = monthly_wf(df.dropna(subset=[best_btc["label"]]),
                                  FEATURE_COLS, best_btc["label"])
                if aucs:
                    print(f"  Monthly WF AUCs: {[f'{a:.3f}' for a in aucs]}")
                    print(f"  Mean={np.mean(aucs):.4f}, Std={np.std(aucs):.4f}")
    else:
        print("  No BTC target columns found.")

    # ══════════════════════════════════════════════════════════════════════════
    # ANGLE C: CONDITIONAL INTERACTION (crowd_oi × pca_var1)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("ANGLE C: CONDITIONAL INTERACTION MATRIX")
    print(f"{'='*80}", flush=True)

    # Focus on 1h any-tail, 12×ATR ≥10%
    tcol_main = "bin_tail_any_60m"
    if tcol_main in df.columns:
        valid_df = df.dropna(subset=[tcol_main, "crowd_oi", "pca_var1"]).copy()

        # Terciles of crowd_oi and pca_var1
        valid_df["oi_terc"] = pd.qcut(valid_df["crowd_oi"], 3,
                                       labels=["OI_low", "OI_mid", "OI_high"],
                                       duplicates="drop")
        valid_df["pca_terc"] = pd.qcut(valid_df["pca_var1"], 3,
                                        labels=["PCA_low", "PCA_mid", "PCA_high"],
                                        duplicates="drop")

        print(f"\n  Tail probability by crowd_oi × pca_var1 tercile:")
        print(f"  (12×ATR, ≥10% coins, 1h horizon)\n")

        pivot = valid_df.groupby(["oi_terc", "pca_terc"])[tcol_main].agg(
            ["mean", "count"]).unstack("pca_terc")

        print(f"  {'':12s}  {'PCA_low':>10s}  {'PCA_mid':>10s}  {'PCA_high':>10s}")
        for oi_level in ["OI_low", "OI_mid", "OI_high"]:
            vals = []
            for pca_level in ["PCA_low", "PCA_mid", "PCA_high"]:
                try:
                    rate = pivot.loc[oi_level, ("mean", pca_level)]
                    n_obs = pivot.loc[oi_level, ("count", pca_level)]
                    vals.append(f"{rate:.1%} ({int(n_obs):,})")
                except Exception:
                    vals.append("  N/A")
            print(f"  {oi_level:12s}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")

        # Interaction strength: is OI_low + PCA_low different from individual effects?
        try:
            corner_ll = valid_df[(valid_df["oi_terc"] == "OI_low") &
                                  (valid_df["pca_terc"] == "PCA_low")][tcol_main].mean()
            corner_hh = valid_df[(valid_df["oi_terc"] == "OI_high") &
                                  (valid_df["pca_terc"] == "PCA_high")][tcol_main].mean()
            marginal_oi_low = valid_df[valid_df["oi_terc"] == "OI_low"][tcol_main].mean()
            marginal_pca_low = valid_df[valid_df["pca_terc"] == "PCA_low"][tcol_main].mean()
            base_rate = valid_df[tcol_main].mean()

            print(f"\n  Interaction analysis:")
            print(f"    Base rate:                    {base_rate:.1%}")
            print(f"    OI_low marginal:              {marginal_oi_low:.1%}")
            print(f"    PCA_low marginal:             {marginal_pca_low:.1%}")
            print(f"    OI_low & PCA_low combined:    {corner_ll:.1%}")
            print(f"    OI_high & PCA_high combined:  {corner_hh:.1%}")

            # Additive prediction
            additive = (marginal_oi_low - base_rate) + (marginal_pca_low - base_rate) + base_rate
            print(f"    Additive prediction:          {additive:.1%}")
            synergy = corner_ll - additive
            print(f"    Synergy (actual - additive):  {synergy:+.1%}")
            if abs(synergy) > 0.03:
                print(f"    → MEANINGFUL interaction ({'amplifying' if synergy > 0 else 'dampening'}) ✅")
            else:
                print(f"    → Interaction is weak / additive ⚠️")
        except Exception as e:
            print(f"  Interaction analysis failed: {e}")

        # Also check breadth_extreme × crowd_oi
        print(f"\n  ── Tail probability by crowd_oi × breadth_extreme tercile ──\n")
        valid_df["bx_terc"] = pd.qcut(valid_df["breadth_extreme"], 3,
                                        labels=["BX_low", "BX_mid", "BX_high"],
                                        duplicates="drop")

        print(f"  {'':12s}  {'BX_low':>10s}  {'BX_mid':>10s}  {'BX_high':>10s}")
        for oi_level in ["OI_low", "OI_mid", "OI_high"]:
            vals = []
            for bx_level in ["BX_low", "BX_mid", "BX_high"]:
                try:
                    sub = valid_df[(valid_df["oi_terc"] == oi_level) &
                                   (valid_df["bx_terc"] == bx_level)]
                    rate = sub[tcol_main].mean()
                    vals.append(f"{rate:.1%} ({len(sub):,})")
                except Exception:
                    vals.append("  N/A")
            print(f"  {oi_level:12s}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")

    # ══════════════════════════════════════════════════════════════════════════
    # ANGLE D: MULTI-HORIZON COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("ANGLE D: MULTI-HORIZON COMPARISON (30m / 1h / 2h / 4h)")
    print(f"{'='*80}", flush=True)

    print(f"\n  12×ATR, ≥10% of coins, LogReg, any-direction:")
    print(f"  {'Horizon':>8s}  {'Base':>6s}  {'AUC':>7s}  {'Q5/Q1':>7s}  "
          f"{'Q1':>7s}  {'Q5':>7s}  {'crowd_oi_coef':>14s}")

    horizon_results = []
    for h in HORIZONS:
        frac_col = f"tail_any_{h}m"
        if frac_col not in df.columns:
            continue
        tcol = f"bin_{frac_col}"
        if tcol not in df.columns:
            df[tcol] = (df[frac_col] >= FRAC_CUTOFF).astype(float)
            df.loc[df[frac_col].isna(), tcol] = np.nan

        y_tr = df[tcol].values[:split]
        y_te = df[tcol].values[split:]
        mask_tr = ~np.isnan(y_tr); mask_te = ~np.isnan(y_te)

        rate = df[tcol].dropna().mean()
        if rate > 0.95 or rate < 0.02:
            print(f"  {h:>6d}m  {rate:>5.1%}  SKIP")
            continue

        r = eval_auc_quintile(
            X_train[mask_tr], y_tr[mask_tr],
            X_test[mask_te], y_te[mask_te],
            label=f"{h}m")

        if r:
            print(f"  {h:>6d}m  {r['base']:>5.1%}  {r['auc']:>7.4f}  "
                  f"{r['q5_q1']:>6.2f}×  {r['q1']:>6.1%}  {r['q5']:>6.1%}  "
                  f"{r['coefs'][4]:>+14.3f}")
            r["horizon"] = h
            horizon_results.append(r)

    if horizon_results:
        best_h = max(horizon_results, key=lambda r: r["auc"])
        print(f"\n  Best horizon: {best_h['label']} — AUC={best_h['auc']:.4f}")

        # Monthly WF for each horizon
        print(f"\n  Monthly walk-forward stability by horizon:")
        for r in horizon_results:
            h = r["horizon"]
            tcol = f"bin_tail_any_{h}m"
            aucs = monthly_wf(df.dropna(subset=[tcol]), FEATURE_COLS, tcol)
            if aucs:
                print(f"    {h:>4d}m: mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}, "
                      f"range=[{np.min(aucs):.4f}, {np.max(aucs):.4f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SYNTHESIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SYNTHESIS")
    print(f"{'='*80}", flush=True)

    # Collect key numbers
    if dir_results:
        down_60 = [r for r in dir_results if r["direction"] == "down" and r["horizon"] == 60]
        up_60 = [r for r in dir_results if r["direction"] == "up" and r["horizon"] == 60]
        any_60 = [r for r in dir_results if r["direction"] == "any" and r["horizon"] == 60]

        if down_60 and up_60 and any_60:
            print(f"\n  1h tail prediction (12×ATR, ≥10%):")
            print(f"    Any direction:  AUC={any_60[0]['auc']:.4f}, Q5/Q1={any_60[0]['q5_q1']:.2f}×")
            print(f"    Down only:      AUC={down_60[0]['auc']:.4f}, Q5/Q1={down_60[0]['q5_q1']:.2f}×")
            print(f"    Up only:        AUC={up_60[0]['auc']:.4f}, Q5/Q1={up_60[0]['q5_q1']:.2f}×")

    if btc_cols and 'btc_results' in dir(main) or True:
        try:
            if btc_results:
                best_b = max(btc_results, key=lambda r: r["auc"])
                print(f"\n  BTC prediction: best={best_b['label']}, "
                      f"AUC={best_b['auc']:.4f}, Q5/Q1={best_b['q5_q1']:.2f}×")
        except NameError:
            pass

    elapsed = time.time() - t0_all
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
