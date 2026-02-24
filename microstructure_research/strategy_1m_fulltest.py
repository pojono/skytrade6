#!/usr/bin/env python3
"""
Full-year, multi-coin test of 1m directional strategy.

Tests the two best variants from initial testing:
  V5: hold=15m, threshold=0.65, both sides
  V6: hold=30m, threshold=0.60, both sides

On 3 coins: SOLUSDT, XRPUSDT, DOGEUSDT
Full year: 2025-01-01 to 2025-12-31

Same anti-lookahead discipline:
  - Expanding-window WFO, 14-day folds, 60-day min training, 15-bar purge
  - Feature selection on training fold only
  - Fixed threshold (no calibration to avoid overfit)
  - 1-bar slippage, 4bps fees, no overlapping positions
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
PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"

SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT"]
START_DATE = "2025-01-01"
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

VARIANTS = [
    {"name": "V5_h15_t65", "hold": 15, "threshold": 0.65, "sides": ["LONG", "SHORT"]},
    {"name": "V6_h30_t60", "hold": 30, "threshold": 0.60, "sides": ["LONG", "SHORT"]},
]


# ============================================================
# TICK AGGREGATION
# ============================================================
def aggregate_ticks_to_1m(trades):
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

        q90 = np.percentile(q, 90) if n >= 10 else q.max()
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        arrival_rate = n / duration_s

        if n > 2:
            iti = np.diff(t).astype(np.float64)
            iti_cv = iti.std() / max(iti.mean(), 1)
        else:
            iti_cv = 0.0

        mid_t = (t[0] + t[-1]) / 2
        first_half = int((t < mid_t).sum())
        trade_acceleration = (n - first_half - first_half) / max(n, 1)

        vwap = qq.sum() / max(total_vol, 1e-10)
        price_range = (p.max() - p.min()) / max(vwap, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        if n > 10:
            signed_vol = q * s
            price_changes = np.diff(p)
            if len(price_changes) > 1 and signed_vol[1:].std() > 0:
                kyle_lambda = float(np.corrcoef(signed_vol[1:], price_changes)[0, 1])
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
            body_pct = abs(close_p - open_p) / full_range
        else:
            upper_wick = 0.0; lower_wick = 0.0; body_pct = 0.0

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


# ============================================================
# DATA LOADING
# ============================================================
def load_1m_bars(symbol):
    cache_dir = PARQUET_DIR / symbol / "1m_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START_DATE, END_DATE)
    all_bars = []
    t0 = time.time()
    new_count = 0
    cache_count = 0

    print(f"  Loading 1m bars for {symbol} ({len(dates)} days)...", flush=True)

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = cache_dir / f"{ds}.parquet"

        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            all_bars.append(bars)
            cache_count += 1
        else:
            tick_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
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

        if i % 50 == 0 or i == len(dates) or i == 1:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(dates) - i) / max(rate, 0.01)
            print(f"    [{i:3d}/{len(dates)}] {ds}  "
                  f"new={new_count} cached={cache_count} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    if not all_bars:
        print(f"  ERROR: No bars for {symbol}!")
        return None

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.set_index("datetime", inplace=True)
    df["returns"] = df["close"].pct_change()

    print(f"  Loaded {len(df):,} bars ({cache_count} cached, {new_count} new)", flush=True)
    return df


# ============================================================
# FEATURES
# ============================================================
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


def add_targets(df, hold_bars):
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
def run_wfo(df_base, feat_cols, variant, symbol):
    hold = variant["hold"]
    threshold = variant["threshold"]
    sides = variant["sides"]

    df = df_base.copy()
    df = add_targets(df, hold)

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
    fold_results = []

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        fold_t0 = time.time()
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
        n_trades = len(fold_trades)
        n_long = sum(1 for t in fold_trades if t["direction"] == "LONG")
        n_short = sum(1 for t in fold_trades if t["direction"] == "SHORT")
        wr = np.mean([t["pnl"] > 0 for t in fold_trades]) if fold_trades else 0

        fold_results.append({
            "fold": fold_idx + 1,
            "n_trades": n_trades,
            "n_long": n_long,
            "n_short": n_short,
            "pnl": fold_pnl,
            "wr": wr,
        })

        elapsed = time.time() - fold_t0
        if n_trades > 0:
            print(f"    F{fold_idx+1:2d}: {n_trades:4d} (L={n_long:3d} S={n_short:3d}) "
                  f"WR={wr:.0%} ret={fold_pnl*100:+.2f}% [{elapsed:.1f}s]", flush=True)
        else:
            print(f"    F{fold_idx+1:2d}: 0 trades [{elapsed:.1f}s]", flush=True)

    return all_trades, fold_results


def summarize(all_trades, fold_results, variant_name, symbol, hold):
    if not all_trades:
        return {"name": variant_name, "symbol": symbol, "n_trades": 0}

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

    active_folds = [f for f in fold_results if f["n_trades"] > 0]
    pos_folds = sum(1 for f in active_folds if f["pnl"] > 0)
    tot_folds = len(fold_results)

    # Long/short breakdown
    long_pnls = np.array([t["pnl"] for t in all_trades if t["direction"] == "LONG"])
    short_pnls = np.array([t["pnl"] for t in all_trades if t["direction"] == "SHORT"])

    return {
        "name": variant_name,
        "symbol": symbol,
        "hold": hold,
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": win_rate,
        "avg_bps": avg_ret * 10000,
        "total_pct": total_ret * 100,
        "sharpe": sharpe,
        "pf": pf,
        "max_dd_pct": max_dd * 100,
        "pos_folds": pos_folds,
        "tot_folds": tot_folds,
        "long_total": long_pnls.sum() * 100 if len(long_pnls) > 0 else 0,
        "long_wr": (long_pnls > 0).mean() if len(long_pnls) > 0 else 0,
        "short_total": short_pnls.sum() * 100 if len(short_pnls) > 0 else 0,
        "short_wr": (short_pnls > 0).mean() if len(short_pnls) > 0 else 0,
        "fold_rets": [f["pnl"] * 100 for f in fold_results],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0_global = time.time()

    print("=" * 90)
    print("  1-MINUTE DIRECTIONAL STRATEGY — FULL YEAR, MULTI-COIN TEST")
    print(f"  Coins: {', '.join(SYMBOLS)}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Variants: {len(VARIANTS)}")
    print(f"  WFO: {MIN_TRAIN_DAYS}d min train, {TRADE_DAYS}d folds, {PURGE_BARS}-bar purge")
    print(f"  Fees: {FEE_BPS}bps RT, {SLIPPAGE_BARS}-bar slippage")
    print("=" * 90)

    all_results = []

    for symbol in SYMBOLS:
        sym_t0 = time.time()
        print(f"\n{'=' * 90}")
        print(f"  {symbol}")
        print(f"{'=' * 90}")

        df = load_1m_bars(symbol)
        if df is None:
            continue

        print("  Adding features...", flush=True)
        df = add_features(df)

        warmup = 4 * 60
        df = df.iloc[warmup:].copy()

        feat_cols = sorted([c for c in df.columns
                            if not c.startswith("tgt_")
                            and c not in ("open", "high", "low", "close", "volume",
                                          "timestamp_us", "returns", "fwd_ret")])
        print(f"  {len(df):,} bars, {len(feat_cols)} features\n")

        for variant in VARIANTS:
            print(f"  --- {variant['name']} (hold={variant['hold']}m, thresh={variant['threshold']}) ---",
                  flush=True)
            trades, fold_results = run_wfo(df, feat_cols, variant, symbol)
            result = summarize(trades, fold_results, variant["name"], symbol, variant["hold"])
            all_results.append(result)

            if result["n_trades"] > 0:
                print(f"  => {result['n_trades']} trades, WR={result['win_rate']:.1%}, "
                      f"avg={result['avg_bps']:+.1f}bp, total={result['total_pct']:+.1f}%, "
                      f"Sharpe={result['sharpe']:+.1f}, PF={result['pf']:.2f}, "
                      f"DD={result['max_dd_pct']:.1f}%, "
                      f"folds+={result['pos_folds']}/{result['tot_folds']}\n", flush=True)
            else:
                print(f"  => 0 trades\n", flush=True)

        sym_elapsed = time.time() - sym_t0
        print(f"  {symbol} done in {sym_elapsed:.0f}s", flush=True)

        del df
        gc.collect()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_elapsed = time.time() - t0_global

    print(f"\n{'=' * 90}")
    print("  FINAL SUMMARY — ALL COINS, ALL VARIANTS")
    print(f"{'=' * 90}")

    print(f"\n  {'Variant':<16} {'Symbol':<10} {'N':>5} {'L':>4} {'S':>4} "
          f"{'WR':>6} {'Avg':>7} {'Tot%':>7} {'Shp':>6} {'PF':>5} {'DD%':>6} {'F+':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*5} {'-'*4} {'-'*4} "
          f"{'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")

    for r in all_results:
        if r["n_trades"] == 0:
            print(f"  {r['name']:<16} {r['symbol']:<10} — no trades")
            continue
        print(f"  {r['name']:<16} {r['symbol']:<10} {r['n_trades']:>5} "
              f"{r['n_long']:>4} {r['n_short']:>4} "
              f"{r['win_rate']:>6.1%} {r['avg_bps']:>+7.1f} "
              f"{r['total_pct']:>+7.1f} {r['sharpe']:>+6.1f} {r['pf']:>5.2f} "
              f"{r['max_dd_pct']:>6.1f} "
              f"{r['pos_folds']}/{r['tot_folds']}")

    # Cross-coin aggregation per variant
    print(f"\n  Cross-coin aggregation:")
    for vname in [v["name"] for v in VARIANTS]:
        vr = [r for r in all_results if r["name"] == vname and r["n_trades"] > 0]
        if not vr:
            continue
        total_trades = sum(r["n_trades"] for r in vr)
        avg_wr = np.mean([r["win_rate"] for r in vr])
        avg_avg_bps = np.mean([r["avg_bps"] for r in vr])
        total_total = sum(r["total_pct"] for r in vr)
        avg_sharpe = np.mean([r["sharpe"] for r in vr])
        avg_pf = np.mean([r["pf"] for r in vr])
        coins_positive = sum(1 for r in vr if r["total_pct"] > 0)

        print(f"\n  {vname}:")
        print(f"    Total trades:    {total_trades}")
        print(f"    Avg win rate:    {avg_wr:.1%}")
        print(f"    Avg trade (bps): {avg_avg_bps:+.1f}")
        print(f"    Combined return: {total_total:+.1f}%")
        print(f"    Avg Sharpe:      {avg_sharpe:+.1f}")
        print(f"    Avg PF:          {avg_pf:.2f}")
        print(f"    Coins positive:  {coins_positive}/{len(vr)}")

    # Long vs Short breakdown
    print(f"\n  Long vs Short breakdown:")
    for r in all_results:
        if r["n_trades"] == 0:
            continue
        print(f"    {r['name']:<16} {r['symbol']:<10} "
              f"L: {r['n_long']:>4} trades, WR={r['long_wr']:.1%}, ret={r['long_total']:+.1f}% | "
              f"S: {r['n_short']:>4} trades, WR={r['short_wr']:.1%}, ret={r['short_total']:+.1f}%")

    print(f"\n  Total time: {total_elapsed:.0f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
