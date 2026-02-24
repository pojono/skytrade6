#!/usr/bin/env python3
"""
Quick 1m predictability test.

Aggregates tick data into 1m bars with microstructure features,
adds targets, and runs LightGBM predictability scan.

Uses 6 months of SOL data (Jul-Dec 2025) to keep it fast.
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
import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================
PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
SYMBOL = "SOLUSDT"
CACHE_DIR = Path("./parquet/SOLUSDT/1m_cache")

# 6 months of data
START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

INTERVAL_1M_US = 60_000_000
FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0


# ============================================================
# PHASE 0: Aggregate ticks → 1m bars (day-by-day, cached)
# ============================================================
def aggregate_ticks_to_1m(trades):
    """Aggregate tick trades into 1m OHLCV + microstructure bars."""
    bucket = (trades["timestamp_us"].values // INTERVAL_1M_US) * INTERVAL_1M_US
    trades = trades.copy()
    trades["bucket"] = bucket

    features = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        t = grp["timestamp_us"].values
        n = len(grp)
        if n < 2:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()
        buy_quote = qq[buy_mask].sum()
        sell_quote = qq[sell_mask].sum()

        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)

        # Large trades
        q90 = np.percentile(q, 90) if n >= 10 else q.max()
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        # Arrival rate
        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        arrival_rate = n / duration_s

        # Inter-trade time CV
        if n > 2:
            iti = np.diff(t).astype(np.float64)
            iti_cv = iti.std() / max(iti.mean(), 1)
        else:
            iti_cv = 0.0

        # Trade acceleration
        mid_t = (t[0] + t[-1]) / 2
        first_half = int((t < mid_t).sum())
        trade_acceleration = (n - first_half - first_half) / max(n, 1)

        # Price features
        vwap = qq.sum() / max(total_vol, 1e-10)
        price_range = (p.max() - p.min()) / max(vwap, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        # Kyle's lambda (price impact)
        if n > 10:
            signed_vol = q * s
            price_changes = np.diff(p)
            if len(price_changes) > 1 and signed_vol[1:].std() > 0:
                kyle_lambda = float(np.corrcoef(signed_vol[1:], price_changes)[0, 1])
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0

        # Candle shape
        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
            body_pct = abs(close_p - open_p) / full_range
        else:
            upper_wick = 0.0; lower_wick = 0.0; body_pct = 0.0

        # Size imbalance
        avg_buy_size = buy_vol / max(buy_count, 1)
        avg_sell_size = sell_vol / max(sell_count, 1)
        size_imbalance = (avg_buy_size - avg_sell_size) / max(avg_buy_size + avg_sell_size, 1e-10)

        features.append({
            "timestamp_us": bkt,
            "open": open_p, "high": high_p, "low": low_p, "close": close_p,
            "volume": total_vol, "quote_volume": buy_quote + sell_quote,
            "trade_count": n,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "count_imbalance": count_imbalance,
            "arrival_rate": arrival_rate,
            "iti_cv": iti_cv,
            "trade_acceleration": trade_acceleration,
            "price_range": price_range,
            "close_vs_vwap": close_vs_vwap,
            "kyle_lambda": kyle_lambda,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body_pct,
            "size_imbalance": size_imbalance,
        })

    return pd.DataFrame(features)


def load_1m_bars():
    """Load 1m bars from cache or build from ticks."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START_DATE, END_DATE)
    all_bars = []
    t0 = time.time()
    new_count = 0
    cache_count = 0

    print(f"  Loading 1m bars for {SYMBOL} ({len(dates)} days)...", flush=True)

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = CACHE_DIR / f"{ds}.parquet"

        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            all_bars.append(bars)
            cache_count += 1
        else:
            tick_path = PARQUET_DIR / SYMBOL / "trades" / SOURCE / f"{ds}.parquet"
            if not tick_path.exists():
                continue
            trades = pd.read_parquet(tick_path)
            bars = aggregate_ticks_to_1m(trades)
            del trades
            gc.collect()
            if not bars.empty:
                bars.to_parquet(cache_path, index=False, compression="snappy")
                all_bars.append(bars)
            new_count += 1

        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.1)
        eta = (len(dates) - i) / max(rate, 0.01)

        if i % 20 == 0 or i == len(dates) or i == 1:
            print(f"    [{i:3d}/{len(dates)}] {ds}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"new={new_count} cached={cache_count}", flush=True)

    if not all_bars:
        print("ERROR: No bars loaded!")
        sys.exit(1)

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.set_index("datetime", inplace=True)
    df["returns"] = df["close"].pct_change()

    print(f"  Loaded {len(df):,} 1m bars ({cache_count} cached, {new_count} new)", flush=True)
    return df


# ============================================================
# PHASE 1: Add derived features (rolling)
# ============================================================
def add_derived_features(df):
    """Add rolling features at 1m resolution."""
    bph = 60  # bars per hour

    # Rolling volatility
    df["rvol_5m"] = df["returns"].rolling(5).std()
    df["rvol_15m"] = df["returns"].rolling(15).std()
    df["rvol_1h"] = df["returns"].rolling(bph).std()
    df["rvol_4h"] = df["returns"].rolling(4 * bph).std()
    df["vol_ratio_5m_1h"] = df["rvol_5m"] / df["rvol_1h"].clip(lower=1e-10)

    # Volume z-score
    w1h = bph
    df["vol_zscore_1h"] = (df["volume"] - df["volume"].rolling(w1h).mean()) / \
                          df["volume"].rolling(w1h).std().clip(lower=1e-10)

    # Arrival rate z-score
    df["rate_zscore_1h"] = (df["arrival_rate"] - df["arrival_rate"].rolling(w1h).mean()) / \
                           df["arrival_rate"].rolling(w1h).std().clip(lower=1e-10)

    # Price momentum
    df["mom_5m"] = df["close"].pct_change(5)
    df["mom_15m"] = df["close"].pct_change(15)
    df["mom_1h"] = df["close"].pct_change(bph)

    # Mean-reversion z-scores
    df["price_zscore_1h"] = (df["close"] - df["close"].rolling(bph).mean()) / \
                            df["close"].rolling(bph).std().clip(lower=1e-10)
    df["price_zscore_4h"] = (df["close"] - df["close"].rolling(4 * bph).mean()) / \
                            df["close"].rolling(4 * bph).std().clip(lower=1e-10)

    # Range expansion
    df["range_zscore_1h"] = (df["price_range"] - df["price_range"].rolling(w1h).mean()) / \
                            df["price_range"].rolling(w1h).std().clip(lower=1e-10)

    # Cumulative imbalance
    df["cum_imbalance_5m"] = df["vol_imbalance"].rolling(5).sum()
    df["cum_imbalance_15m"] = df["vol_imbalance"].rolling(15).sum()
    df["cum_imbalance_1h"] = df["vol_imbalance"].rolling(bph).sum()

    # VWAP deviation z-score
    df["vwap_zscore_1h"] = (df["close_vs_vwap"] - df["close_vs_vwap"].rolling(w1h).mean()) / \
                           df["close_vs_vwap"].rolling(w1h).std().clip(lower=1e-10)

    # Parkinson volatility
    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    df["parkvol_1h"] = (log_hl**2).rolling(bph).mean().apply(np.sqrt) / (4 * np.log(2))**0.5

    # Efficiency ratio
    for h, label in [(5, "5m"), (15, "15m"), (bph, "1h")]:
        net_move = (df["close"] - df["close"].shift(h)).abs()
        sum_moves = df["returns"].abs().rolling(h).sum() * df["close"]
        df[f"efficiency_{label}"] = net_move / sum_moves.clip(lower=1e-10)

    # Kyle lambda rolling
    df["kyle_lambda_5m"] = df["kyle_lambda"].rolling(5).mean()
    df["kyle_lambda_15m"] = df["kyle_lambda"].rolling(15).mean()

    # Large trade imbalance rolling
    df["large_imb_5m"] = df["large_imbalance"].rolling(5).mean()
    df["large_imb_15m"] = df["large_imbalance"].rolling(15).mean()

    print(f"  Added derived features. Total columns: {len(df.columns)}", flush=True)
    return df


# ============================================================
# PHASE 2: Add targets (forward-looking)
# ============================================================
def add_targets(df):
    """Add forward-looking targets at various horizons."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    ret = df["returns"].values
    n = len(df)

    # --- Return targets ---
    for p in [1, 3, 5, 10, 15, 30, 60]:
        # Forward return
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-p] = c[p:] / c[:n-p] - 1.0
        df[f"tgt_ret_{p}"] = fwd_ret

        # Return magnitude
        df[f"tgt_ret_mag_{p}"] = np.abs(fwd_ret)

    # --- Volatility targets ---
    for p in [5, 15, 30, 60]:
        fwd_vol = np.full(n, np.nan)
        for i in range(n - p):
            fwd_vol[i] = ret[i+1:i+1+p].std()
        df[f"tgt_vol_{p}"] = fwd_vol

    # --- Directional targets ---
    for p in [3, 5, 10, 15]:
        # Profitable long
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-p] = c[p:] / c[:n-p] - 1.0
        df[f"tgt_profitable_long_{p}"] = (fwd_ret > FEE_FRAC).astype(float)
        df.loc[np.isnan(fwd_ret), f"tgt_profitable_long_{p}"] = np.nan

        # Profitable short
        df[f"tgt_profitable_short_{p}"] = (fwd_ret < -FEE_FRAC).astype(float)
        df.loc[np.isnan(fwd_ret), f"tgt_profitable_short_{p}"] = np.nan

    # --- Breakout targets ---
    for p in [3, 5, 10]:
        bu = np.full(n, np.nan)
        bd = np.full(n, np.nan)
        for i in range(n - p):
            bu[i] = 1.0 if (h[i+1:i+1+p] > h[i]).any() else 0.0
            bd[i] = 1.0 if (l[i+1:i+1+p] < l[i]).any() else 0.0
        df[f"tgt_breakout_up_{p}"] = bu
        df[f"tgt_breakout_down_{p}"] = bd

    # --- Range / consolidation ---
    for p in [5, 15, 30]:
        fwd_range = np.full(n, np.nan)
        for i in range(n - p):
            fwd_h = h[i+1:i+1+p].max()
            fwd_l = l[i+1:i+1+p].min()
            fwd_range[i] = (fwd_h - fwd_l) / c[i] * 10000  # in bps
        df[f"tgt_range_bps_{p}"] = fwd_range

    # --- Vol expansion ---
    vol = df["rvol_1h"].values
    for p in [5, 15, 30]:
        fwd_vol = np.full(n, np.nan)
        for i in range(n - p):
            fwd_vol[i] = ret[i+1:i+1+p].std()
        df[f"tgt_vol_expansion_{p}"] = (fwd_vol > vol).astype(float)
        df.loc[np.isnan(fwd_vol) | np.isnan(vol), f"tgt_vol_expansion_{p}"] = np.nan

    # --- Mean reversion target ---
    # Does price revert to 1h mean within next N bars?
    ma_1h = df["close"].rolling(60).mean().values
    for p in [5, 15, 30]:
        revert = np.full(n, np.nan)
        for i in range(n - p):
            if np.isnan(ma_1h[i]) or c[i] == 0:
                continue
            dist_now = abs(c[i] - ma_1h[i]) / c[i]
            min_dist = min(abs(c[j] - ma_1h[i]) / c[i] for j in range(i+1, i+1+p))
            revert[i] = 1.0 if min_dist < dist_now * 0.5 else 0.0  # reverts >50%
        df[f"tgt_mean_revert_{p}"] = revert

    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    print(f"  Added {len(tgt_cols)} targets", flush=True)
    return df


# ============================================================
# PHASE 3: Predictability scan
# ============================================================
def run_predictability_scan(df):
    """Run LightGBM predictability scan on all targets."""
    tgt_cols = sorted([c for c in df.columns if c.startswith("tgt_")])
    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_")
                 and c not in ("open", "high", "low", "close", "volume",
                               "timestamp_us", "returns")]

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Targets:  {len(tgt_cols)}")
    print(f"  Bars:     {len(df):,}")

    # Use last 25% as test
    n = len(df)
    split = int(n * 0.75)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    print(f"  Train: {len(df_train):,} bars, Test: {len(df_test):,} bars")
    print(f"\n  {'Target':<35} {'Type':<8} {'Score':>8} {'BaseRate':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10}")

    results = []

    for i, tgt in enumerate(tgt_cols, 1):
        y_train = df_train[tgt].values
        y_test = df_test[tgt].values

        valid_tr = np.isfinite(y_train)
        valid_te = np.isfinite(y_test)

        if valid_tr.sum() < 1000 or valid_te.sum() < 500:
            continue

        # Is it binary?
        unique_vals = np.unique(y_train[valid_tr])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0})

        # Select top 30 features by correlation on train
        corrs = []
        for f in feat_cols:
            x = df_train[f].values
            mask = valid_tr & np.isfinite(x)
            if mask.sum() < 500:
                corrs.append(0.0)
                continue
            try:
                c_val, _ = spearmanr(x[mask], y_train[mask])
                corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
            except:
                corrs.append(0.0)

        top_idx = np.argsort(corrs)[::-1][:30]
        selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]

        if len(selected) < 5:
            continue

        X_tr = df_train[selected].values[valid_tr]
        y_tr = y_train[valid_tr]
        X_te = df_test[selected].values[valid_te]
        y_te = y_test[valid_te]

        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

        try:
            if is_binary:
                y_tr_int = y_tr.astype(int)
                if len(np.unique(y_tr_int)) < 2:
                    continue
                model = lgb.LGBMClassifier(
                    objective="binary", metric="auc", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr_int)
                pred = model.predict_proba(X_te)[:, 1]
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_te.astype(int), pred) - 0.5
                base_rate = y_te.mean()
                ttype = "binary"
            else:
                model = lgb.LGBMRegressor(
                    objective="regression", metric="rmse", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                score_val, _ = spearmanr(pred, y_te)
                score = score_val if np.isfinite(score_val) else 0.0
                base_rate = np.nan
                ttype = "cont"

            results.append({
                "target": tgt,
                "type": ttype,
                "score": score,
                "base_rate": base_rate,
                "n_train": len(y_tr),
                "n_test": len(y_te),
            })

            marker = "***" if score > 0.05 else "  *" if score > 0.02 else "   "
            br_str = f"{base_rate:.2%}" if not np.isnan(base_rate) else "N/A"
            print(f"  {tgt:<35} {ttype:<8} {score:>+.4f} {br_str:>10} {marker}")

        except Exception as e:
            pass

        if i % 10 == 0:
            print(f"  ... {i}/{len(tgt_cols)} targets processed", flush=True)

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  1-MINUTE PREDICTABILITY TEST")
    print(f"  {SYMBOL}, {START_DATE} to {END_DATE}")
    print("=" * 80)

    # Load/build 1m bars
    df = load_1m_bars()

    # Add derived features
    print("\n  Adding derived features...", flush=True)
    df = add_derived_features(df)

    # Add targets
    print("  Adding targets...", flush=True)
    df = add_targets(df)

    # Drop warmup period (first 4h = 240 bars)
    df = df.iloc[240:].copy()
    print(f"  After warmup drop: {len(df):,} bars")

    # Run scan
    print("\n  Running predictability scan...", flush=True)
    results = run_predictability_scan(df)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — 1m Predictability ({SYMBOL})")
    print(f"{'=' * 80}")

    if results:
        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"\n  Top 15 most predictable targets:")
        print(f"  {'Rank':<6} {'Target':<35} {'Type':<8} {'Score':>8}")
        print(f"  {'-'*6} {'-'*35} {'-'*8} {'-'*8}")
        for i, r in enumerate(results[:15], 1):
            print(f"  {i:<6} {r['target']:<35} {r['type']:<8} {r['score']:>+.4f}")

        # Compare with 4h scores
        print(f"\n  Score distribution:")
        scores = [r["score"] for r in results]
        print(f"    Max:    {max(scores):+.4f}")
        print(f"    Median: {np.median(scores):+.4f}")
        print(f"    >0.05:  {sum(1 for s in scores if s > 0.05)}")
        print(f"    >0.02:  {sum(1 for s in scores if s > 0.02)}")
        print(f"    >0.00:  {sum(1 for s in scores if s > 0.00)}")

    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
