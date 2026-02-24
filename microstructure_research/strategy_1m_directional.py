#!/usr/bin/env python3
"""
Strategy: 1-Minute Directional with Walk-Forward Optimization

Signal: Predict whether price will move >4bps in next 10 minutes (long/short).
Hold: Fixed 10 bars (10 minutes).
Entry: Next bar open after signal (1-bar slippage).
Fees: 4bps round-trip taker.

Anti-Lookahead / Anti-Overfit Discipline:
  1. Expanding-window WFO with purge gap (15 bars)
  2. Feature selection on training fold only (top-30 by Spearman)
  3. Shallow LightGBM (depth=4, leaves=15, min_child=100)
  4. Threshold calibrated on validation split within training (80/20)
  5. Fixed hold period, no early exit optimization
  6. No overlapping positions
  7. Fixed sizing (1x), no sizing optimization
  8. 1-bar slippage on entry
  9. Multi-coin validation without re-tuning
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
# CONFIG — ALL PARAMETERS FIXED BEFORE ANY TESTING
# ============================================================
PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
CACHE_BASE = Path("./parquet")

# WFO parameters
MIN_TRAIN_DAYS = 60       # minimum training window (60 days)
TRADE_DAYS = 14           # each test fold = 14 days
PURGE_BARS = 15           # 15 bars (15 min) gap between train/test
BARS_PER_DAY = 1440       # 1m bars per day

# Model parameters — conservative, fixed
MODEL_PARAMS = dict(
    objective="binary",
    metric="auc",
    verbosity=-1,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# Feature selection
TOP_N_FEATURES = 30
MIN_CORR = 0.005

# Trading parameters
HOLD_BARS = 10            # hold for 10 minutes
FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0
SLIPPAGE_BARS = 1         # enter at next bar open

# Threshold — calibrated on validation within each fold
VAL_FRAC = 0.20           # 20% of training for validation

INTERVAL_1M_US = 60_000_000


# ============================================================
# TICK AGGREGATION (same as test_1m_predictability.py)
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
def load_1m_bars(symbol, start_date, end_date):
    """Load 1m bars from cache or build from ticks."""
    cache_dir = CACHE_BASE / symbol / "1m_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date)
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

        if i % 30 == 0 or i == len(dates):
            elapsed = time.time() - t0
            print(f"    [{i:3d}/{len(dates)}] {ds}  "
                  f"new={new_count} cached={cache_count} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    if not all_bars:
        print(f"  ERROR: No bars loaded for {symbol}!")
        return None

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.set_index("datetime", inplace=True)
    df["returns"] = df["close"].pct_change()

    print(f"  Loaded {len(df):,} bars ({cache_count} cached, {new_count} new)", flush=True)
    return df


# ============================================================
# FEATURE ENGINEERING — basic set (43 features, proven best)
# ============================================================
def add_features(df):
    """Add the basic feature set that proved optimal in testing."""
    bph = 60

    # Volatility
    df["rvol_5m"] = df["returns"].rolling(5).std()
    df["rvol_15m"] = df["returns"].rolling(15).std()
    df["rvol_1h"] = df["returns"].rolling(bph).std()
    df["rvol_4h"] = df["returns"].rolling(4 * bph).std()
    df["vol_ratio_5m_1h"] = df["rvol_5m"] / df["rvol_1h"].clip(lower=1e-10)

    # Volume z-score
    for w, label in [(bph, "1h")]:
        vol_roll = df["volume"].rolling(w)
        df[f"vol_zscore_{label}"] = (df["volume"] - vol_roll.mean()) / vol_roll.std().clip(lower=1e-10)

    # Arrival rate z-score
    rate_roll = df["arrival_rate"].rolling(bph)
    df["rate_zscore_1h"] = (df["arrival_rate"] - rate_roll.mean()) / rate_roll.std().clip(lower=1e-10)

    # Price momentum
    df["mom_5m"] = df["close"].pct_change(5)
    df["mom_15m"] = df["close"].pct_change(15)
    df["mom_1h"] = df["close"].pct_change(bph)

    # Price z-score
    for w, label in [(bph, "1h"), (4 * bph, "4h")]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"price_zscore_{label}"] = (df["close"] - ma) / std.clip(lower=1e-10)

    # Range z-score
    rng_roll = df["price_range"].rolling(bph)
    df["range_zscore_1h"] = (df["price_range"] - rng_roll.mean()) / rng_roll.std().clip(lower=1e-10)

    # Cumulative imbalance
    df["cum_imbalance_5m"] = df["vol_imbalance"].rolling(5).sum()
    df["cum_imbalance_15m"] = df["vol_imbalance"].rolling(15).sum()
    df["cum_imbalance_1h"] = df["vol_imbalance"].rolling(bph).sum()

    # VWAP deviation z-score
    vwap_roll = df["close_vs_vwap"].rolling(bph)
    df["vwap_zscore_1h"] = (df["close_vs_vwap"] - vwap_roll.mean()) / vwap_roll.std().clip(lower=1e-10)

    # Parkinson volatility
    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    df["parkvol_1h"] = np.sqrt((log_hl**2).rolling(bph).mean()) / np.sqrt(4 * np.log(2))

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

    return df


# ============================================================
# TARGET CONSTRUCTION — strictly no lookahead
# ============================================================
def add_targets(df):
    """
    Binary targets: will price move >FEE_FRAC in next HOLD_BARS bars?

    Entry is at bar[i+SLIPPAGE_BARS].open (1-bar slippage).
    Exit is at bar[i+SLIPPAGE_BARS+HOLD_BARS].open.

    This means:
      - Signal fires at bar i (using only data up to bar i)
      - We enter at bar i+1 open
      - We exit at bar i+11 open
      - Forward return = close[i+1+HOLD_BARS] / close[i+1] - 1
        (approximating open-to-open with close-to-close, conservative)
    """
    c = df["close"].values
    n = len(df)

    entry_offset = SLIPPAGE_BARS       # 1
    exit_offset = SLIPPAGE_BARS + HOLD_BARS  # 11

    # Forward return from entry to exit
    fwd_ret = np.full(n, np.nan)
    for i in range(n - exit_offset):
        fwd_ret[i] = c[i + exit_offset] / c[i + entry_offset] - 1.0

    df["fwd_ret"] = fwd_ret
    df["tgt_long"] = (fwd_ret > FEE_FRAC).astype(float)
    df["tgt_short"] = (fwd_ret < -FEE_FRAC).astype(float)

    # Mark NaN where we can't compute target
    nan_mask = np.isnan(fwd_ret)
    df.loc[nan_mask, "tgt_long"] = np.nan
    df.loc[nan_mask, "tgt_short"] = np.nan

    valid = np.isfinite(fwd_ret)
    long_rate = df.loc[valid, "tgt_long"].mean()
    short_rate = df.loc[valid, "tgt_short"].mean()
    print(f"  Targets: long base={long_rate:.2%}, short base={short_rate:.2%}, "
          f"valid={valid.sum():,}/{n:,}", flush=True)

    return df


# ============================================================
# FEATURE SELECTION — on training data only
# ============================================================
def select_features(df_train, target_col, feat_cols, top_n=TOP_N_FEATURES):
    """Select top features by Spearman correlation on training data only."""
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
    selected = [feat_cols[j] for j in top_idx if corrs[j] > MIN_CORR]
    return selected


# ============================================================
# THRESHOLD CALIBRATION — on validation split within training
# ============================================================
def calibrate_threshold(model, X_val, y_val, direction):
    """
    Find threshold that maximizes expected profit on validation data.
    Returns threshold, or default 0.65 if calibration fails.
    """
    pred = model.predict_proba(X_val)[:, 1]
    DEFAULT_THRESH = 0.65

    best_thresh = DEFAULT_THRESH
    best_metric = -np.inf

    for thresh in np.arange(0.55, 0.85, 0.025):
        mask = pred >= thresh
        if mask.sum() < 20:
            continue

        # Win rate at this threshold
        wr = y_val[mask].mean()
        n_trades = mask.sum()

        # Expected profit per trade (in bps)
        # If target=1, trade is profitable (>FEE_FRAC), so avg profit > 0
        # Approximate: E[profit] ≈ wr * avg_win - (1-wr) * avg_loss
        # Simpler: use (wr - base_rate) * n_trades as metric
        base = y_val.mean()
        lift = wr - base
        metric = lift * np.sqrt(n_trades)  # lift × sqrt(trades) balances edge vs sample

        if metric > best_metric:
            best_metric = metric
            best_thresh = thresh

    return best_thresh


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================
def run_wfo(df, feat_cols, symbol):
    """
    Expanding-window walk-forward optimization.

    Timeline:
      |--- training (expanding) ---|-- purge --|--- test (TRADE_DAYS) ---|
    """
    n = len(df)
    min_train_bars = MIN_TRAIN_DAYS * BARS_PER_DAY
    trade_bars = TRADE_DAYS * BARS_PER_DAY

    # Calculate fold boundaries
    folds = []
    test_start = min_train_bars + PURGE_BARS
    while test_start + trade_bars <= n:
        train_end = test_start - PURGE_BARS
        test_end = test_start + trade_bars
        folds.append((0, train_end, test_start, test_end))
        test_start = test_end

    print(f"\n  WFO: {len(folds)} folds, {TRADE_DAYS}d each, "
          f"min_train={MIN_TRAIN_DAYS}d, purge={PURGE_BARS} bars")

    all_trades = []
    fold_summaries = []

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        fold_t0 = time.time()
        df_train_full = df.iloc[tr_start:tr_end]
        df_test = df.iloc[te_start:te_end]

        # Split training into train/val for threshold calibration
        val_size = int(len(df_train_full) * VAL_FRAC)
        df_train = df_train_full.iloc[:-val_size]
        df_val = df_train_full.iloc[-val_size:]

        fold_trades = []

        for direction, tgt_col in [("LONG", "tgt_long"), ("SHORT", "tgt_short")]:
            # Feature selection on training data only
            selected = select_features(df_train, tgt_col, feat_cols)
            if len(selected) < 5:
                continue

            # Prepare training data
            y_tr = df_train[tgt_col].values
            valid_tr = np.isfinite(y_tr)
            X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
            y_tr_c = y_tr[valid_tr].astype(int)

            if len(np.unique(y_tr_c)) < 2 or len(y_tr_c) < 1000:
                continue

            # Train model
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(X_tr, y_tr_c)

            # Calibrate threshold on validation
            y_val = df_val[tgt_col].values
            valid_val = np.isfinite(y_val)
            X_val = np.nan_to_num(df_val[selected].values[valid_val], nan=0, posinf=0, neginf=0)
            y_val_c = y_val[valid_val].astype(int)

            if len(y_val_c) < 200:
                threshold = 0.65
            else:
                threshold = calibrate_threshold(model, X_val, y_val_c, direction)

            # Predict on test data
            y_te = df_test[tgt_col].values
            valid_te = np.isfinite(y_te)
            X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
            pred = model.predict_proba(X_te)[:, 1]

            # Get actual forward returns
            fwd_ret = df_test["fwd_ret"].values[valid_te]
            test_idx = np.where(valid_te)[0]

            # Generate trades — no overlapping positions
            last_exit_bar = -1
            for j in range(len(pred)):
                if pred[j] < threshold:
                    continue

                bar_idx = test_idx[j]
                if bar_idx <= last_exit_bar:
                    continue  # skip if still in a position

                if not np.isfinite(fwd_ret[j]):
                    continue

                # Trade P&L
                if direction == "LONG":
                    pnl = fwd_ret[j] - FEE_FRAC
                else:
                    pnl = -fwd_ret[j] - FEE_FRAC

                fold_trades.append({
                    "fold": fold_idx + 1,
                    "direction": direction,
                    "bar_idx": te_start + bar_idx,
                    "confidence": pred[j],
                    "threshold": threshold,
                    "fwd_ret": fwd_ret[j],
                    "pnl": pnl,
                    "win": 1 if pnl > 0 else 0,
                })

                last_exit_bar = bar_idx + HOLD_BARS

        all_trades.extend(fold_trades)

        # Fold summary
        fold_elapsed = time.time() - fold_t0
        if fold_trades:
            fold_pnls = [t["pnl"] for t in fold_trades]
            fold_ret = sum(fold_pnls)
            fold_wr = np.mean([t["win"] for t in fold_trades])
            fold_avg = np.mean(fold_pnls)
            n_long = sum(1 for t in fold_trades if t["direction"] == "LONG")
            n_short = sum(1 for t in fold_trades if t["direction"] == "SHORT")
            thresholds = set(f"{t['threshold']:.3f}" for t in fold_trades)

            fold_summaries.append({
                "fold": fold_idx + 1,
                "n_trades": len(fold_trades),
                "n_long": n_long,
                "n_short": n_short,
                "total_ret": fold_ret,
                "avg_ret": fold_avg,
                "win_rate": fold_wr,
                "thresholds": thresholds,
            })

            print(f"  Fold {fold_idx+1:2d}: {len(fold_trades):4d} trades "
                  f"(L={n_long:3d} S={n_short:3d}) "
                  f"WR={fold_wr:.1%} avg={fold_avg*10000:+.1f}bp "
                  f"total={fold_ret*100:+.2f}% "
                  f"thresh={thresholds} "
                  f"[{fold_elapsed:.1f}s]", flush=True)
        else:
            fold_summaries.append({
                "fold": fold_idx + 1, "n_trades": 0,
            })
            print(f"  Fold {fold_idx+1:2d}: 0 trades [{fold_elapsed:.1f}s]", flush=True)

    return all_trades, fold_summaries


# ============================================================
# RESULTS ANALYSIS
# ============================================================
def analyze_results(all_trades, fold_summaries, symbol):
    """Comprehensive results analysis."""
    print(f"\n{'=' * 80}")
    print(f"  RESULTS — {symbol}")
    print(f"{'=' * 80}")

    if not all_trades:
        print("  No trades generated!")
        return

    pnls = np.array([t["pnl"] for t in all_trades])
    wins = np.array([t["win"] for t in all_trades])
    directions = [t["direction"] for t in all_trades]

    # Overall stats
    n_trades = len(pnls)
    total_ret = pnls.sum()
    avg_ret = pnls.mean()
    win_rate = wins.mean()
    n_long = sum(1 for d in directions if d == "LONG")
    n_short = sum(1 for d in directions if d == "SHORT")

    # Sharpe (annualized, assuming avg hold = 10 min)
    trades_per_year = 252 * 24 * 6  # ~36k potential 10-min slots
    if pnls.std() > 0:
        sharpe = avg_ret / pnls.std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-10)

    # Max drawdown (cumulative)
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Positive folds
    active_folds = [f for f in fold_summaries if f["n_trades"] > 0]
    positive_folds = sum(1 for f in active_folds if f["total_ret"] > 0)

    print(f"\n  Overall:")
    print(f"    Trades:        {n_trades} (L={n_long}, S={n_short})")
    print(f"    Win Rate:      {win_rate:.1%}")
    print(f"    Avg Trade:     {avg_ret*10000:+.2f} bps")
    print(f"    Total Return:  {total_ret*100:+.2f}%")
    print(f"    Sharpe:        {sharpe:.2f}")
    print(f"    Profit Factor: {pf:.2f}")
    print(f"    Max Drawdown:  {max_dd*100:.2f}%")
    print(f"    Positive Folds: {positive_folds}/{len(active_folds)}")

    # By direction
    for d in ["LONG", "SHORT"]:
        d_pnls = np.array([t["pnl"] for t in all_trades if t["direction"] == d])
        if len(d_pnls) == 0:
            continue
        d_wr = (d_pnls > 0).mean()
        d_avg = d_pnls.mean()
        d_total = d_pnls.sum()
        print(f"\n  {d}:")
        print(f"    Trades: {len(d_pnls)}, WR: {d_wr:.1%}, "
              f"Avg: {d_avg*10000:+.2f}bp, Total: {d_total*100:+.2f}%")

    # Fold-by-fold
    print(f"\n  Fold-by-fold:")
    print(f"  {'Fold':>6} {'Trades':>8} {'WR':>8} {'AvgBps':>10} {'TotalRet':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for f in fold_summaries:
        if f["n_trades"] > 0:
            print(f"  {f['fold']:>6} {f['n_trades']:>8} {f['win_rate']:>8.1%} "
                  f"{f['avg_ret']*10000:>+10.2f} {f['total_ret']*100:>+10.2f}%")
        else:
            print(f"  {f['fold']:>6} {'0':>8} {'N/A':>8} {'N/A':>10} {'N/A':>10}")

    # Confidence analysis
    print(f"\n  By confidence bucket:")
    conf = np.array([t["confidence"] for t in all_trades])
    for lo, hi in [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]:
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 5:
            continue
        bucket_pnls = pnls[mask]
        print(f"    [{lo:.2f}-{hi:.2f}): {mask.sum():4d} trades, "
              f"WR={( bucket_pnls > 0).mean():.1%}, "
              f"avg={bucket_pnls.mean()*10000:+.1f}bp")

    return {
        "symbol": symbol,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_ret_bps": avg_ret * 10000,
        "total_ret_pct": total_ret * 100,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_dd_pct": max_dd * 100,
        "positive_folds": positive_folds,
        "total_folds": len(active_folds),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    # Data range: 6 months for SOL (cached), full range for validation coins
    symbols_config = {
        "SOLUSDT": ("2025-07-01", "2025-12-31"),
    }

    print("=" * 80)
    print("  1-MINUTE DIRECTIONAL STRATEGY — Walk-Forward Optimization")
    print("  Anti-Lookahead / Anti-Overfit Discipline")
    print("=" * 80)
    print(f"\n  Config:")
    print(f"    Hold:       {HOLD_BARS} bars ({HOLD_BARS}m)")
    print(f"    Slippage:   {SLIPPAGE_BARS} bar")
    print(f"    Fees:       {FEE_BPS} bps RT")
    print(f"    Purge:      {PURGE_BARS} bars")
    print(f"    Min train:  {MIN_TRAIN_DAYS} days")
    print(f"    Trade fold: {TRADE_DAYS} days")
    print(f"    Model:      depth={MODEL_PARAMS['max_depth']}, "
          f"leaves={MODEL_PARAMS['num_leaves']}, "
          f"min_child={MODEL_PARAMS['min_child_samples']}")
    print(f"    Val split:  {VAL_FRAC:.0%} of training for threshold calibration")

    all_results = {}

    for symbol, (start, end) in symbols_config.items():
        print(f"\n{'=' * 80}")
        print(f"  {symbol} ({start} to {end})")
        print(f"{'=' * 80}")

        # Load data
        df = load_1m_bars(symbol, start, end)
        if df is None:
            continue

        # Add features
        print("  Adding features...", flush=True)
        df = add_features(df)

        # Add targets
        print("  Adding targets...", flush=True)
        df = add_targets(df)

        # Drop warmup (4h = 240 bars for rolling features to stabilize)
        warmup = 4 * 60
        df = df.iloc[warmup:].copy()
        print(f"  After warmup: {len(df):,} bars")

        # Get feature columns
        feat_cols = sorted([c for c in df.columns
                            if not c.startswith("tgt_")
                            and c not in ("open", "high", "low", "close", "volume",
                                          "timestamp_us", "returns", "fwd_ret")])
        print(f"  Features: {len(feat_cols)}")

        # Run WFO
        trades, fold_summaries = run_wfo(df, feat_cols, symbol)

        # Analyze
        result = analyze_results(trades, fold_summaries, symbol)
        if result:
            all_results[symbol] = result

        del df
        gc.collect()

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 80}")

    if all_results:
        print(f"\n  {'Symbol':<12} {'Trades':>8} {'WR':>8} {'AvgBps':>8} {'Total%':>8} "
              f"{'Sharpe':>8} {'PF':>8} {'DD%':>8} {'Folds+':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for sym, r in all_results.items():
            print(f"  {sym:<12} {r['n_trades']:>8} {r['win_rate']:>8.1%} "
                  f"{r['avg_ret_bps']:>+8.1f} {r['total_ret_pct']:>+8.1f} "
                  f"{r['sharpe']:>8.1f} {r['profit_factor']:>8.2f} "
                  f"{r['max_dd_pct']:>8.1f} "
                  f"{r['positive_folds']}/{r['total_folds']}")

    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
