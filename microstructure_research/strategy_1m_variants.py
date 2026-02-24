#!/usr/bin/env python3
"""
Variant testing for 1m directional strategy.

Tests multiple configurations to find if any survive WFO:
  V1: Short-only, threshold=0.65 (forced)
  V2: Long-only, threshold=0.65 (forced)
  V3: Both sides, threshold=0.65 (forced)
  V4: Both sides, hold=5 bars, threshold=0.65
  V5: Both sides, hold=15 bars, threshold=0.65
  V6: Both sides, hold=30 bars, threshold=0.60
  V7: Short-only, hold=30 bars, threshold=0.60

All use same anti-lookahead discipline as main strategy.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = Path("./parquet/SOLUSDT/1m_cache")
SYMBOL = "SOLUSDT"
START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

BARS_PER_DAY = 1440
MIN_TRAIN_DAYS = 60
TRADE_DAYS = 14
PURGE_BARS = 15
FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0
SLIPPAGE_BARS = 1
INTERVAL_1M_US = 60_000_000

MODEL_PARAMS = dict(
    objective="binary", metric="auc", verbosity=-1,
    n_estimators=150, max_depth=4, learning_rate=0.05,
    num_leaves=15, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
)

# Variants to test
VARIANTS = [
    {"name": "V1_short_only_h10_t65",  "hold": 10, "threshold": 0.65, "sides": ["SHORT"]},
    {"name": "V2_long_only_h10_t65",   "hold": 10, "threshold": 0.65, "sides": ["LONG"]},
    {"name": "V3_both_h10_t65",        "hold": 10, "threshold": 0.65, "sides": ["LONG", "SHORT"]},
    {"name": "V4_both_h5_t65",         "hold": 5,  "threshold": 0.65, "sides": ["LONG", "SHORT"]},
    {"name": "V5_both_h15_t65",        "hold": 15, "threshold": 0.65, "sides": ["LONG", "SHORT"]},
    {"name": "V6_both_h30_t60",        "hold": 30, "threshold": 0.60, "sides": ["LONG", "SHORT"]},
    {"name": "V7_short_h30_t60",       "hold": 30, "threshold": 0.60, "sides": ["SHORT"]},
    {"name": "V8_both_h10_t60",        "hold": 10, "threshold": 0.60, "sides": ["LONG", "SHORT"]},
    {"name": "V9_both_h10_t70",        "hold": 10, "threshold": 0.70, "sides": ["LONG", "SHORT"]},
]


# ============================================================
# DATA LOADING
# ============================================================
def load_1m_bars():
    dates = pd.date_range(START_DATE, END_DATE)
    all_bars = []
    for date in dates:
        ds = date.strftime("%Y-%m-%d")
        cache_path = CACHE_DIR / f"{ds}.parquet"
        if cache_path.exists():
            all_bars.append(pd.read_parquet(cache_path))

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.set_index("datetime", inplace=True)
    df["returns"] = df["close"].pct_change()
    print(f"  Loaded {len(df):,} bars", flush=True)
    return df


def add_features(df):
    bph = 60
    df["rvol_5m"] = df["returns"].rolling(5).std()
    df["rvol_15m"] = df["returns"].rolling(15).std()
    df["rvol_1h"] = df["returns"].rolling(bph).std()
    df["rvol_4h"] = df["returns"].rolling(4 * bph).std()
    df["vol_ratio_5m_1h"] = df["rvol_5m"] / df["rvol_1h"].clip(lower=1e-10)

    vol_roll = df["volume"].rolling(bph)
    df["vol_zscore_1h"] = (df["volume"] - vol_roll.mean()) / vol_roll.std().clip(lower=1e-10)

    rate_roll = df["arrival_rate"].rolling(bph)
    df["rate_zscore_1h"] = (df["arrival_rate"] - rate_roll.mean()) / rate_roll.std().clip(lower=1e-10)

    df["mom_5m"] = df["close"].pct_change(5)
    df["mom_15m"] = df["close"].pct_change(15)
    df["mom_1h"] = df["close"].pct_change(bph)

    for w, label in [(bph, "1h"), (4 * bph, "4h")]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"price_zscore_{label}"] = (df["close"] - ma) / std.clip(lower=1e-10)

    rng_roll = df["price_range"].rolling(bph)
    df["range_zscore_1h"] = (df["price_range"] - rng_roll.mean()) / rng_roll.std().clip(lower=1e-10)

    df["cum_imbalance_5m"] = df["vol_imbalance"].rolling(5).sum()
    df["cum_imbalance_15m"] = df["vol_imbalance"].rolling(15).sum()
    df["cum_imbalance_1h"] = df["vol_imbalance"].rolling(bph).sum()

    vwap_roll = df["close_vs_vwap"].rolling(bph)
    df["vwap_zscore_1h"] = (df["close_vs_vwap"] - vwap_roll.mean()) / vwap_roll.std().clip(lower=1e-10)

    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    df["parkvol_1h"] = np.sqrt((log_hl**2).rolling(bph).mean()) / np.sqrt(4 * np.log(2))

    for h, label in [(5, "5m"), (15, "15m"), (bph, "1h")]:
        net_move = (df["close"] - df["close"].shift(h)).abs()
        sum_moves = df["returns"].abs().rolling(h).sum() * df["close"]
        df[f"efficiency_{label}"] = net_move / sum_moves.clip(lower=1e-10)

    df["kyle_lambda_5m"] = df["kyle_lambda"].rolling(5).mean()
    df["kyle_lambda_15m"] = df["kyle_lambda"].rolling(15).mean()
    df["large_imb_5m"] = df["large_imbalance"].rolling(5).mean()
    df["large_imb_15m"] = df["large_imbalance"].rolling(15).mean()

    return df


def add_targets_for_hold(df, hold_bars):
    """Add targets for a specific hold period."""
    c = df["close"].values
    n = len(df)
    entry_offset = SLIPPAGE_BARS
    exit_offset = SLIPPAGE_BARS + hold_bars

    fwd_ret = np.full(n, np.nan)
    for i in range(n - exit_offset):
        fwd_ret[i] = c[i + exit_offset] / c[i + entry_offset] - 1.0

    df["fwd_ret"] = fwd_ret
    df["tgt_long"] = np.where(np.isnan(fwd_ret), np.nan, (fwd_ret > FEE_FRAC).astype(float))
    df["tgt_short"] = np.where(np.isnan(fwd_ret), np.nan, (fwd_ret < -FEE_FRAC).astype(float))

    return df


def select_features(df_train, target_col, feat_cols, top_n=30):
    y = df_train[target_col].values
    valid_y = np.isfinite(y)
    corrs = []
    for f in feat_cols:
        x = df_train[f].values
        mask = valid_y & np.isfinite(x)
        if mask.sum() < 500:
            corrs.append(0.0)
            continue
        try:
            c_val, _ = spearmanr(x[mask], y[mask])
            corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
        except:
            corrs.append(0.0)
    top_idx = np.argsort(corrs)[::-1][:top_n]
    return [feat_cols[j] for j in top_idx if corrs[j] > 0.005]


# ============================================================
# WFO ENGINE
# ============================================================
def run_variant(df_base, feat_cols, variant):
    """Run a single variant through WFO."""
    name = variant["name"]
    hold = variant["hold"]
    threshold = variant["threshold"]
    sides = variant["sides"]

    # Add targets for this hold period
    df = df_base.copy()
    df = add_targets_for_hold(df, hold)

    n = len(df)
    min_train_bars = MIN_TRAIN_DAYS * BARS_PER_DAY
    trade_bars = TRADE_DAYS * BARS_PER_DAY

    folds = []
    test_start = min_train_bars + PURGE_BARS
    while test_start + trade_bars <= n:
        train_end = test_start - PURGE_BARS
        test_end = test_start + trade_bars
        folds.append((0, train_end, test_start, test_end))
        test_start = test_end

    all_trades = []
    fold_rets = []

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        df_train = df.iloc[tr_start:tr_end]
        df_test = df.iloc[te_start:te_end]

        fold_trades = []

        for direction in sides:
            tgt_col = "tgt_long" if direction == "LONG" else "tgt_short"

            selected = select_features(df_train, tgt_col, feat_cols)
            if len(selected) < 5:
                continue

            y_tr = df_train[tgt_col].values
            valid_tr = np.isfinite(y_tr)
            X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
            y_tr_c = y_tr[valid_tr].astype(int)

            if len(np.unique(y_tr_c)) < 2 or len(y_tr_c) < 1000:
                continue

            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(X_tr, y_tr_c)

            y_te = df_test[tgt_col].values
            valid_te = np.isfinite(y_te)
            X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
            pred = model.predict_proba(X_te)[:, 1]

            fwd_ret = df_test["fwd_ret"].values[valid_te]
            test_idx = np.where(valid_te)[0]

            last_exit_bar = -1
            for j in range(len(pred)):
                if pred[j] < threshold:
                    continue
                bar_idx = test_idx[j]
                if bar_idx <= last_exit_bar:
                    continue
                if not np.isfinite(fwd_ret[j]):
                    continue

                pnl = (fwd_ret[j] if direction == "LONG" else -fwd_ret[j]) - FEE_FRAC

                fold_trades.append({
                    "fold": fold_idx + 1,
                    "direction": direction,
                    "confidence": pred[j],
                    "pnl": pnl,
                })
                last_exit_bar = bar_idx + hold

        all_trades.extend(fold_trades)
        fold_pnl = sum(t["pnl"] for t in fold_trades)
        fold_rets.append(fold_pnl)

    # Summarize
    if not all_trades:
        return {"name": name, "n_trades": 0}

    pnls = np.array([t["pnl"] for t in all_trades])
    n_trades = len(pnls)
    total_ret = pnls.sum()
    avg_ret = pnls.mean()
    win_rate = (pnls > 0).mean()
    n_long = sum(1 for t in all_trades if t["direction"] == "LONG")
    n_short = sum(1 for t in all_trades if t["direction"] == "SHORT")

    trades_per_year = 252 * 24 * 60 / hold
    sharpe = avg_ret / pnls.std() * np.sqrt(trades_per_year) if pnls.std() > 0 else 0

    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-10)

    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = (cum_pnl - running_max).min()

    positive_folds = sum(1 for r in fold_rets if r > 0)
    total_folds = len(fold_rets)

    return {
        "name": name,
        "hold": hold,
        "threshold": threshold,
        "sides": "/".join(sides),
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": win_rate,
        "avg_bps": avg_ret * 10000,
        "total_pct": total_ret * 100,
        "sharpe": sharpe,
        "pf": pf,
        "max_dd_pct": max_dd * 100,
        "pos_folds": positive_folds,
        "tot_folds": total_folds,
        "fold_rets": [r * 100 for r in fold_rets],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  1-MINUTE STRATEGY VARIANT TESTING")
    print(f"  {SYMBOL}, {START_DATE} to {END_DATE}")
    print(f"  {len(VARIANTS)} variants, WFO with {TRADE_DAYS}d folds")
    print("=" * 80)

    df = load_1m_bars()
    print("  Adding features...", flush=True)
    df = add_features(df)

    warmup = 4 * 60
    df = df.iloc[warmup:].copy()
    print(f"  After warmup: {len(df):,} bars")

    feat_cols = sorted([c for c in df.columns
                        if not c.startswith("tgt_")
                        and c not in ("open", "high", "low", "close", "volume",
                                      "timestamp_us", "returns", "fwd_ret")])
    print(f"  Features: {len(feat_cols)}\n")

    results = []
    for i, variant in enumerate(VARIANTS, 1):
        vt0 = time.time()
        print(f"  [{i}/{len(VARIANTS)}] {variant['name']}...", end="", flush=True)
        r = run_variant(df, feat_cols, variant)
        elapsed = time.time() - vt0
        results.append(r)

        if r["n_trades"] > 0:
            print(f" {r['n_trades']:4d} trades, WR={r['win_rate']:.1%}, "
                  f"avg={r['avg_bps']:+.1f}bp, total={r['total_pct']:+.1f}%, "
                  f"Sharpe={r['sharpe']:+.1f}, PF={r['pf']:.2f}, "
                  f"DD={r['max_dd_pct']:.1f}%, "
                  f"folds={r['pos_folds']}/{r['tot_folds']} "
                  f"[{elapsed:.0f}s]", flush=True)
        else:
            print(f" 0 trades [{elapsed:.0f}s]", flush=True)

    # Summary table
    print(f"\n{'=' * 80}")
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"\n  {'Variant':<28} {'Hold':>4} {'Thr':>5} {'Side':<6} {'N':>5} "
          f"{'WR':>6} {'Avg':>7} {'Tot%':>7} {'Shp':>6} {'PF':>5} {'DD%':>6} {'F+':>5}")
    print(f"  {'-'*28} {'-'*4} {'-'*5} {'-'*6} {'-'*5} "
          f"{'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*5}")

    for r in results:
        if r["n_trades"] == 0:
            print(f"  {r['name']:<28} — no trades")
            continue
        print(f"  {r['name']:<28} {r['hold']:>4} {r['threshold']:>5.2f} {r['sides']:<6} "
              f"{r['n_trades']:>5} {r['win_rate']:>6.1%} {r['avg_bps']:>+7.1f} "
              f"{r['total_pct']:>+7.1f} {r['sharpe']:>+6.1f} {r['pf']:>5.2f} "
              f"{r['max_dd_pct']:>6.1f} {r['pos_folds']}/{r['tot_folds']}")

    # Show fold-by-fold for best variant
    best = max([r for r in results if r["n_trades"] > 0],
               key=lambda x: x["total_pct"], default=None)
    if best:
        print(f"\n  Best variant: {best['name']}")
        print(f"  Fold returns: {['%.1f%%' % f for f in best['fold_rets']]}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
