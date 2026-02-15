#!/usr/bin/env python3
"""
Test the BTCUSDT contrarian edge on ETHUSDT and SOLUSDT (Bybit futures).
Same signal: composite of vol/dollar/large/count imbalance + close_vs_vwap.
Same params: threshold=1.0, holding=4h, Bybit VIP0 fees (7 bps RT).

Runs 7-day test first, then 30-day if 7-day looks ok.
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

SYMBOLS = ["ETHUSDT", "SOLUSDT"]
SOURCE = "bybit_futures"
INTERVAL_US = 300_000_000  # 5 min
ROUND_TRIP_FEE_BPS = 7.0  # Bybit VIP0
PARQUET_DIR = Path("./parquet")

SIGNAL_FEATURES = [
    "vol_imbalance", "dollar_imbalance", "large_imbalance",
    "count_imbalance", "close_vs_vwap",
]
RANK_WINDOW = 288 * 3  # 3 days


# ---------------------------------------------------------------------------
# Feature computation (same as BTC test)
# ---------------------------------------------------------------------------

def compute_features(trades, interval_us):
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

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        vwap = qq.sum() / max(total_vol, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        price_mid = (p.max() + p.min()) / 2
        vol_above = q[p >= price_mid].sum()
        vol_below = q[p < price_mid].sum()
        vol_profile_skew = (vol_above - vol_below) / max(total_vol, 1e-10)

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "count_imbalance": count_imbalance,
            "close_vs_vwap": close_vs_vwap,
            "vol_profile_skew": vol_profile_skew,
            "open": p[0], "close": p[-1], "high": p.max(), "low": p.min(),
            "volume": total_vol, "trade_count": n,
        })

    return pd.DataFrame(features)


def load_and_compute(symbol, start_date, end_date):
    """Load trades day-by-day, compute features, return DataFrame."""
    dates = pd.date_range(start_date, end_date)
    all_features = []
    t0 = time.time()

    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{date_str}.parquet"

        if not path.exists():
            continue

        mem_gb = psutil.virtual_memory().used / (1024**3)
        trades = pd.read_parquet(path)
        feat = compute_features(trades, INTERVAL_US)
        del trades
        all_features.append(feat)

        elapsed = time.time() - t0
        eta = (len(dates) - i) / (i / elapsed) if i > 0 else 0
        print(f"  [{i:2d}/{len(dates)}] {date_str}: {len(feat)} bars  "
              f"RAM={mem_gb:.1f}GB  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

    if not all_features:
        return pd.DataFrame()

    df = pd.concat(all_features, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df["returns"] = df["close"].pct_change()
    return df


def build_signal(df):
    """Build composite contrarian signal."""
    for feat in SIGNAL_FEATURES:
        df[f"{feat}_rank"] = df[feat].rolling(RANK_WINDOW, min_periods=288).rank(pct=True)
    rank_cols = [f"{f}_rank" for f in SIGNAL_FEATURES]
    df["composite"] = df[rank_cols].mean(axis=1)
    df["signal"] = (df["composite"] - df["composite"].rolling(RANK_WINDOW, min_periods=288).mean()) / \
                   df["composite"].rolling(RANK_WINDOW, min_periods=288).std().clip(lower=1e-10)
    return df


def backtest(df, entry_threshold=1.0, holding_bars=48, fee_bps=7.0):
    """Contrarian backtest: short on buy pressure, long on sell pressure."""
    data = df.dropna(subset=["signal"]).copy()
    signals = data["signal"].values
    closes = data["close"].values
    n = len(data)

    pnls = []
    in_trade = False
    entry_idx = 0
    direction = 0

    for i in range(n - holding_bars):
        if in_trade and i - entry_idx >= holding_bars:
            raw = (closes[i] / closes[entry_idx] - 1) * 10000 * direction
            pnls.append(raw - fee_bps)
            in_trade = False

        if not in_trade:
            if signals[i] > entry_threshold:
                in_trade = True; entry_idx = i; direction = -1
            elif signals[i] < -entry_threshold:
                in_trade = True; entry_idx = i; direction = 1

    return np.array(pnls) if pnls else np.array([])


def run_test(symbol, start_date, end_date, label):
    """Run full test for one symbol and date range."""
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    print(f"\n{'='*70}")
    print(f"  {symbol} {SOURCE} — {label} ({days} days: {start_date} → {end_date})")
    print(f"{'='*70}")

    print(f"  Loading & computing features...", flush=True)
    df = load_and_compute(symbol, start_date, end_date)
    if df.empty:
        print(f"  ❌ No data!")
        return None

    print(f"\n  Total: {len(df):,} bars")
    print(f"  Price: {df['close'].min():.2f} – {df['close'].max():.2f}")

    print(f"  Building signal...", flush=True)
    df = build_signal(df)

    # IC check
    print(f"\n  Information Coefficient (Spearman):")
    for bars, label_h in [(3, "15m"), (6, "30m"), (12, "1h"), (24, "2h"), (48, "4h")]:
        fwd = df["close"].pct_change(bars).shift(-bars)
        clean = pd.DataFrame({"s": df["signal"], "f": fwd}).dropna()
        if len(clean) < 50:
            continue
        ic, pval = stats.spearmanr(clean["s"], clean["f"])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else ""
        print(f"    vs {label_h:>3s}: IC={ic:+.4f}  p={pval:.2e}  {sig}")

    # Parameter sweep
    print(f"\n  Parameter sweep (Bybit VIP0: {ROUND_TRIP_FEE_BPS} bps RT):")
    print(f"  {'Thresh':>7s} {'Hold':>6s} {'Trades':>7s} {'Avg PnL':>9s} {'Total':>10s} {'WinRate':>8s}")
    print(f"  {'-'*55}")

    best_total = -999999
    best_cfg = None

    for thresh in [1.0, 1.5, 2.0]:
        for hp_bars, hp_label in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, entry_threshold=thresh, holding_bars=hp_bars)
            if len(pnls) == 0:
                continue
            avg = pnls.mean()
            total = pnls.sum()
            wr = (pnls > 0).mean()
            marker = " ★" if avg > 0 and len(pnls) > 20 else ""
            print(f"  {thresh:>7.1f} {hp_label:>6s} {len(pnls):>7d} {avg:>+9.2f} {total:>+10.1f} {wr:>8.1%}{marker}")

            if total > best_total and len(pnls) > 20:
                best_total = total
                best_cfg = (thresh, hp_label, hp_bars, len(pnls), avg, wr)

    if best_cfg:
        thresh, hp_label, hp_bars, n_trades, avg, wr = best_cfg
        print(f"\n  🏆 BEST: thresh={thresh}, hold={hp_label}")
        print(f"     Trades={n_trades}, Avg={avg:+.2f} bps, Total={best_total:+.1f} bps, WR={wr:.1%}")
        if avg > 0:
            print(f"     ✅ POSITIVE edge!")
        else:
            print(f"     ⚠️  Negative edge")
    else:
        print(f"\n  ❌ No viable configuration found")

    return best_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CROSS-ASSET EDGE VALIDATION")
    print(f"  Exchange: Bybit Futures (VIP0: {ROUND_TRIP_FEE_BPS} bps RT)")
    print(f"  Signal: contrarian composite ({', '.join(SIGNAL_FEATURES)})")
    print("=" * 70)

    results = {}

    for symbol in SYMBOLS:
        # 7-day test first
        cfg_7d = run_test(symbol, "2025-12-01", "2025-12-07", "7-day test")

        # 30-day test
        cfg_30d = run_test(symbol, "2025-12-01", "2025-12-30", "30-day test")

        results[symbol] = {"7d": cfg_7d, "30d": cfg_30d}

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: Cross-Asset Edge Validation")
    print(f"{'='*70}")
    print(f"  {'Symbol':>10s} {'Period':>8s} {'Thresh':>7s} {'Hold':>6s} {'Trades':>7s} {'Avg PnL':>9s} {'Total':>10s} {'WR':>6s}")
    print(f"  {'-'*65}")

    # Include BTC results for comparison
    print(f"  {'BTCUSDT':>10s} {'30d':>8s} {'1.0':>7s} {'4h':>6s} {'161':>7s} {'+13.68':>9s} {'+2202.2':>10s} {'n/a':>6s}  (reference)")

    for symbol in SYMBOLS:
        for period, label in [("7d", "7d"), ("30d", "30d")]:
            cfg = results[symbol][period]
            if cfg:
                thresh, hp_label, _, n_trades, avg, wr = cfg
                total = avg * n_trades
                print(f"  {symbol:>10s} {label:>8s} {thresh:>7.1f} {hp_label:>6s} {n_trades:>7d} {avg:>+9.2f} {total:>+10.1f} {wr:>6.1%}")
            else:
                print(f"  {symbol:>10s} {label:>8s}  — no viable config —")

    print(f"\n✅ Cross-asset validation complete!")


if __name__ == "__main__":
    main()
