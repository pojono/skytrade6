#!/usr/bin/env python3
"""
Extended test on Bybit BTCUSDT futures (30 days).
Same approach as 7-day test, but longer period for validation.
"""

import sys
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config (same as 7d test)
# ---------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
SOURCE = "bybit_futures"
INTERVAL_US = 300_000_000  # 5 min
START_DATE = "2025-12-01"
END_DATE = "2025-12-30"  # 30 days

# Bybit VIP0 fees (bps)
TAKER_FEE_BPS = 5.0
MAKER_FEE_BPS = 2.0
ROUND_TRIP_FEE_BPS = TAKER_FEE_BPS + MAKER_FEE_BPS  # 7 bps

PARQUET_DIR = Path("./parquet")


# ---------------------------------------------------------------------------
# Feature computation (reuse from 7d test)
# ---------------------------------------------------------------------------

def compute_features(trades: pd.DataFrame, interval_us: int) -> pd.DataFrame:
    """Compute microstructure features from raw tick trades per interval bucket."""
    bucket = (trades["timestamp_us"].values // interval_us) * interval_us
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

        # --- Aggression ---
        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)

        q90 = np.percentile(q, 90)
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)
        large_vol_pct = q[large_mask].sum() / max(total_vol, 1e-10)

        # --- Flow ---
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        arrival_rate = n / duration_s

        if n > 2:
            iti = np.diff(t).astype(np.float64)
            iti_cv = iti.std() / max(iti.mean(), 1)
            sub_buckets = np.linspace(t[0], t[-1], 6)
            sub_counts = np.histogram(t, bins=sub_buckets)[0]
            burstiness = float(sub_counts.max()) / max(n, 1)
        else:
            iti_cv = 0.0
            burstiness = 1.0

        mid_t = (t[0] + t[-1]) / 2
        first_half = int((t < mid_t).sum())
        trade_acceleration = (n - first_half - first_half) / max(n, 1)

        # --- Price impact ---
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

        ret = (p[-1] - p[0]) / max(p[0], 1e-10)
        amihud = abs(ret) / max(total_vol, 1e-10)

        # --- Volume profile ---
        price_mid = (p.max() + p.min()) / 2
        vol_above = q[p >= price_mid].sum()
        vol_below = q[p < price_mid].sum()
        vol_profile_skew = (vol_above - vol_below) / max(total_vol, 1e-10)

        # --- Candle shape ---
        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
        else:
            upper_wick = 0.0
            lower_wick = 0.0

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "large_vol_pct": large_vol_pct,
            "count_imbalance": count_imbalance,
            "arrival_rate": arrival_rate,
            "iti_cv": iti_cv,
            "burstiness": burstiness,
            "trade_acceleration": trade_acceleration,
            "price_range": price_range,
            "close_vs_vwap": close_vs_vwap,
            "kyle_lambda": kyle_lambda,
            "amihud": amihud,
            "vol_profile_skew": vol_profile_skew,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "open": open_p,
            "close": close_p,
            "high": high_p,
            "low": low_p,
            "volume": total_vol,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "quote_volume": buy_quote + sell_quote,
            "trade_count": n,
        })

    return pd.DataFrame(features)


def build_composite_signal(df):
    """Build composite signal from top features (rank-based)."""
    SIGNAL_FEATURES = [
        "vol_imbalance", "dollar_imbalance", "large_imbalance",
        "count_imbalance", "close_vs_vwap", "vol_profile_skew",
    ]
    
    RANK_WINDOW = 288 * 3  # 3 days
    
    for feat in SIGNAL_FEATURES:
        df[f"{feat}_rank"] = df[feat].rolling(RANK_WINDOW, min_periods=288).rank(pct=True)
    
    rank_cols = [f"{f}_rank" for f in SIGNAL_FEATURES]
    df["composite"] = df[rank_cols].mean(axis=1)
    
    df["signal"] = (df["composite"] - df["composite"].rolling(RANK_WINDOW, min_periods=288).mean()) / \
                   df["composite"].rolling(RANK_WINDOW, min_periods=288).std().clip(lower=1e-10)
    
    return df


def backtest_fixed_holding(df, entry_threshold=1.5, holding_bars=6, fee_bps=7.0):
    """Simple contrarian backtest with fixed holding period."""
    signals = df["signal"].values
    closes = df["close"].values
    timestamps = df["timestamp_us"].values
    n = len(df)
    
    trades = []
    in_trade = False
    entry_idx = 0
    direction = 0
    
    for i in range(n - holding_bars):
        if in_trade:
            if i - entry_idx >= holding_bars:
                exit_price = closes[i]
                entry_price = closes[entry_idx]
                raw_return_bps = (exit_price / entry_price - 1) * 10000 * direction
                net_return_bps = raw_return_bps - fee_bps
                
                trades.append({
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[i],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "signal_value": signals[entry_idx],
                    "net_return_bps": net_return_bps,
                })
                in_trade = False
        
        if not in_trade:
            if signals[i] > entry_threshold:
                in_trade = True
                entry_idx = i
                direction = -1  # SHORT on buying pressure
            elif signals[i] < -entry_threshold:
                in_trade = True
                entry_idx = i
                direction = 1   # LONG on selling pressure
    
    if not trades:
        return pd.DataFrame(), pd.Series(dtype=float)
    
    trades_df = pd.DataFrame(trades)
    trades_df["cum_pnl_bps"] = trades_df["net_return_bps"].cumsum()
    return trades_df, trades_df["cum_pnl_bps"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"EXTENDED TEST: {SYMBOL} {SOURCE} microstructure features")
    print(f"Period: {START_DATE} → {END_DATE} (30 days)")
    print(f"Fees: {ROUND_TRIP_FEE_BPS} bps round-trip (Bybit VIP0)")
    print("=" * 70)
    
    dates = pd.date_range(START_DATE, END_DATE)
    all_features = []
    
    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        file_path = PARQUET_DIR / SYMBOL / "trades" / SOURCE / f"{date_str}.parquet"
        
        if not file_path.exists():
            print(f"[{i:2d}/{len(dates)}] ⊘ {date_str}: no file")
            continue
        
        mem_gb = psutil.virtual_memory().used / (1024**3)
        print(f"[{i:2d}/{len(dates)}] ⏳ {date_str}: loading... (RAM: {mem_gb:.1f}GB)")
        
        trades = pd.read_parquet(file_path)
        print(f"[{i:2d}/{len(dates)}] 📊 {date_str}: {len(trades):,} trades")
        
        t0 = time.time()
        features = compute_features(trades, INTERVAL_US)
        del trades
        print(f"[{i:2d}/{len(dates)}] ✅ {date_str}: {len(features)} bars in {time.time()-t0:.1f}s")
        
        all_features.append(features)
        
        mem_gb = psutil.virtual_memory().used / (1024**3)
        print(f"[{i:2d}/{len(dates)}] 💾 {date_str}: RAM {mem_gb:.1f}GB")
    
    if not all_features:
        print("No data processed!")
        return
    
    print("\n🔗 Combining features...")
    df = pd.concat(all_features, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df["returns"] = df["close"].pct_change()
    
    print(f"📈 Total: {len(df):,} bars, {len(df.columns)} columns")
    print(f"📅 Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    
    print("\n🎯 Building composite signal...")
    df = build_composite_signal(df)
    
    for bars, label in [(3, "15m"), (6, "30m"), (12, "1h"), (24, "2h")]:
        df[f"fwd_{label}"] = df["close"].pct_change(bars).shift(-bars)
        clean = df[["signal", f"fwd_{label}"]].dropna()
        ic, pval = stats.spearmanr(clean["signal"], clean[f"fwd_{label}"])
        print(f"  IC vs {label:>3s}: {ic:+.4f} (p={pval:.2e})")
    
    # Focus on best config from 7d test: thresh=1.5, hold=2h
    print("\n🏆 Testing best config from 7d test (thresh=1.5, hold=2h)...")
    trades_df, equity = backtest_fixed_holding(
        df, entry_threshold=1.5, holding_bars=24, fee_bps=ROUND_TRIP_FEE_BPS)
    
    if trades_df.empty:
        print("No trades generated!")
        return
    
    n_trades = len(trades_df)
    total_pnl = trades_df["net_return_bps"].sum()
    avg_pnl = trades_df["net_return_bps"].mean()
    win_rate = (trades_df["net_return_bps"] > 0).mean()
    sharpe = trades_df["net_return_bps"].mean() / trades_df["net_return_bps"].std() * np.sqrt(252 * 288 / 24) if trades_df["net_return_bps"].std() > 0 else 0
    max_dd = (trades_df["cum_pnl_bps"] - trades_df["cum_pnl_bps"].cummax()).min()
    
    print(f"\n📊 30-DAY RESULTS:")
    print(f"   Trades: {n_trades}")
    print(f"   Avg PnL: {avg_pnl:+.2f} bps")
    print(f"   Total PnL: {total_pnl:+.1f} bps")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Sharpe: {sharpe:.2f}")
    print(f"   Max DD: {max_dd:+.1f} bps")
    
    # Annualized estimate
    days = (df['datetime'].max() - df['datetime'].min()).days
    trades_per_day = n_trades / days
    daily_pnl = total_pnl / days
    annual_pnl = daily_pnl * 365
    annual_trades = trades_per_day * 365
    
    print(f"\n📈 ANNUALIZED (extrapolated):")
    print(f"   Trades/year: {annual_trades:.0f}")
    print(f"   PnL/year: {annual_pnl:+.0f} bps")
    print(f"   Assuming $100k BTC per trade: ~${annual_pnl/10000 * 100000:.0f} USD/year")
    
    if avg_pnl > 0:
        print("\n✅ POSITIVE edge holds on 30 days!")
    else:
        print("\n⚠️  Edge weakened on longer period")
    
    print("\n✅ Extended test complete!")


if __name__ == "__main__":
    main()
