#!/usr/bin/env python3
"""
Quick variations test on Bybit BTCUSDT (30 days).
Test different thresholds and holding periods to see if any config works.
"""

import sys
import time
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
SOURCE = "bybit_futures"
INTERVAL_US = 300_000_000  # 5 min
START_DATE = "2025-12-01"
END_DATE = "2025-12-30"  # 30 days
ROUND_TRIP_FEE_BPS = 7.0  # Bybit VIP0
PARQUET_DIR = Path("./parquet")


# ---------------------------------------------------------------------------
# Feature computation (reuse)
# ---------------------------------------------------------------------------

def compute_features(trades: pd.DataFrame, interval_us: int) -> pd.DataFrame:
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

        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)

        q90 = np.percentile(q, 90)
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)
        large_vol_pct = q[large_mask].sum() / max(total_vol, 1e-10)

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
            iti_cv = 0.0; burstiness = 1.0

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

        price_mid = (p.max() + p.min()) / 2
        vol_above = q[p >= price_mid].sum()
        vol_below = q[p < price_mid].sum()
        vol_profile_skew = (vol_above - vol_below) / max(total_vol, 1e-10)

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
        else:
            upper_wick = 0.0; lower_wick = 0.0

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "count_imbalance": count_imbalance,
            "close_vs_vwap": close_vs_vwap,
            "vol_profile_skew": vol_profile_skew,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "buy_volume": buy_vol, "sell_volume": sell_vol,
            "trade_count": n,
        })

    return pd.DataFrame(features)


def build_signal(df, features=None):
    """Build composite signal from specified features."""
    if features is None:
        features = ["vol_imbalance", "dollar_imbalance", "large_imbalance",
                   "count_imbalance", "close_vs_vwap", "vol_profile_skew"]
    
    RANK_WINDOW = 288 * 3
    for feat in features:
        df[f"{feat}_rank"] = df[feat].rolling(RANK_WINDOW, min_periods=288).rank(pct=True)
    
    rank_cols = [f"{f}_rank" for f in features]
    df["composite"] = df[rank_cols].mean(axis=1)
    df["signal"] = (df["composite"] - df["composite"].rolling(RANK_WINDOW, min_periods=288).mean()) / \
                   df["composite"].rolling(RANK_WINDOW, min_periods=288).std().clip(lower=1e-10)
    return df


def backtest(df, entry_threshold=1.5, holding_bars=24, fee_bps=7.0):
    """Simple contrarian backtest."""
    signals = df["signal"].values
    closes = df["close"].values
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
                net_return_bps = (exit_price / entry_price - 1) * 10000 * direction - fee_bps
                trades.append(net_return_bps)
                in_trade = False
        
        if not in_trade:
            if signals[i] > entry_threshold:
                in_trade = True; entry_idx = i; direction = -1
            elif signals[i] < -entry_threshold:
                in_trade = True; entry_idx = i; direction = 1
    
    if not trades:
        return 0, 0, 0
    
    trades = np.array(trades)
    return len(trades), trades.mean(), trades.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"VARIATIONS TEST: {SYMBOL} {SOURCE}")
    print(f"Period: {START_DATE} → {END_DATE} (30 days)")
    print("=" * 70)
    
    # Load data (reuse from previous run)
    print("📊 Loading features...")
    dates = pd.date_range(START_DATE, END_DATE)
    all_features = []
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        file_path = PARQUET_DIR / SYMBOL / "trades" / SOURCE / f"{date_str}.parquet"
        if not file_path.exists():
            continue
        
        trades = pd.read_parquet(file_path)
        features = compute_features(trades, INTERVAL_US)
        del trades
        all_features.append(features)
    
    df = pd.concat(all_features, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df["returns"] = df["close"].pct_change()
    print(f"📈 {len(df):,} bars loaded")
    
    # Test variations
    variations = [
        # (features, thresholds, holding_periods, description)
        (["vol_imbalance", "count_imbalance"], [1.0, 1.5, 2.0], [12, 24, 48], "Minimal features"),
        (["vol_imbalance", "dollar_imbalance", "count_imbalance"], [1.0, 1.5, 2.0], [12, 24, 48], "Core features"),
        (["vol_imbalance", "dollar_imbalance", "large_imbalance", "count_imbalance"], [1.0, 1.5, 2.0], [12, 24, 48], "Add large trades"),
        (["vol_imbalance", "dollar_imbalance", "large_imbalance", "count_imbalance", "close_vs_vwap"], [1.0, 1.5, 2.0], [12, 24, 48], "Add VWAP"),
        (["vol_imbalance", "dollar_imbalance", "large_imbalance", "count_imbalance", "vol_profile_skew"], [1.0, 1.5, 2.0], [12, 24, 48], "Add vol profile"),
    ]
    
    print(f"\n🔍 Testing {len(variations)} variations...")
    print("=" * 70)
    
    best_total = -999999
    best_config = None
    
    for features, thresholds, holding_periods, desc in variations:
        print(f"\n📋 {desc}")
        print(f"   Features: {', '.join(features)}")
        
        df_test = build_signal(df.copy(), features)
        
        print(f"   {'Thresh':>7s} {'Hold':>6s} {'Trades':>7s} {'Avg PnL':>9s} {'Total PnL':>11s}")
        print("   " + "-" * 55)
        
        for thresh in thresholds:
            for hp_bars in holding_periods:
                n_trades, avg_pnl, total_pnl = backtest(df_test, entry_threshold=thresh, holding_bars=hp_bars)
                hp_label = f"{hp_bars//12}h" if hp_bars >= 12 else f"{hp_bars*5}m"
                print(f"   {thresh:>7.1f} {hp_label:>6s} {n_trades:>7d} {avg_pnl:>+9.2f} {total_pnl:>+11.1f}")
                
                if total_pnl > best_total and n_trades > 30:
                    best_total = total_pnl
                    best_config = (desc, features, thresh, hp_bars, n_trades, avg_pnl, total_pnl)
    
    print("\n" + "=" * 70)
    if best_config:
        desc, features, thresh, hp_bars, n_trades, avg_pnl, total_pnl = best_config
        print(f"🏆 BEST CONFIGURATION:")
        print(f"   Description: {desc}")
        print(f"   Features: {', '.join(features)}")
        print(f"   Threshold: {thresh}, Holding: {hp_bars} bars ({hp_bars//12}h)")
        print(f"   Trades: {n_trades}, Avg PnL: {avg_pnl:+.2f} bps, Total PnL: {total_pnl:+.1f} bps")
        
        if avg_pnl > 0:
            print("   ✅ POSITIVE edge found!")
        else:
            print("   ⚠️  Best config still negative")
    else:
        print("❌ No configuration with >30 trades found")
    
    print("\n✅ Variations test complete!")


if __name__ == "__main__":
    main()
