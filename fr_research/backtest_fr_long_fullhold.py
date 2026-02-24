#!/usr/bin/env python3
# ruff: noqa: E501
"""
Backtest: LONG futures after negative FR — hold to NEXT settlement with REAL price moves.

Strategy:
  1. After extreme negative FR settlement, go LONG futures at market
  2. Hold through to NEXT settlement, collect FR payment
  3. Exit at NEXT settlement with market order
  4. Test various SL levels during the hold period

Uses Bybit 1m kline API to get actual prices at entry and exit settlements,
plus intermediate prices for SL simulation.

For each trade we download:
  - Entry price: close at T+1min after settlement
  - Exit price: close at next_settlement_time (1-8h later)
  - Intermediate highs/lows: hourly klines for SL checking
"""
import sys
import os
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

import builtins
_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True)
    _print(*a, **k)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"

NOTIONAL = 10_000
ENTRY_FEE_BPS = 5.5
TP_FEE_BPS = 2.0
SL_FEE_BPS = 5.5
MIN_FR_BPS = 20.0

def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except:
            if attempt == retries - 1:
                return None
            time.sleep(0.3 * (attempt + 1))

def get_klines(symbol, start_ms, end_ms, interval="1"):
    """Get klines from Bybit API."""
    url = (f"https://api.bybit.com/v5/market/kline"
           f"?category=linear&symbol={symbol}&interval={interval}"
           f"&start={start_ms}&end={end_ms}&limit=200")
    data = api_get(url)
    if data is None or data.get("retCode") != 0:
        return None
    rows = []
    for c in data.get("result", {}).get("list", []):
        rows.append({
            "ts_ms": int(c[0]),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
        })
    return rows

def main():
    t0 = time.time()

    print("=" * 100)
    print("BACKTEST: LONG after neg FR — FULL HOLD to next settlement (real prices)")
    print("=" * 100)

    # Load FR history
    print("\n1. Loading Bybit FR history...", flush=True)
    bb_fr = pd.read_parquet(DATA / "historical_fr" / "bybit_fr_history.parquet")
    bb_fr["fr_bps"] = bb_fr["fundingRate"] * 10000
    bb_fr["settle_ts_ms"] = bb_fr["fundingTime"].astype(np.int64) // 10**6
    bb_fr = bb_fr.sort_values(["symbol", "settle_ts_ms"])

    # Build next-settlement lookup
    bb_fr["next_fr_bps"] = bb_fr.groupby("symbol")["fr_bps"].shift(-1)
    bb_fr["next_settle_ms"] = bb_fr.groupby("symbol")["settle_ts_ms"].shift(-1)
    bb_fr = bb_fr.dropna(subset=["next_fr_bps", "next_settle_ms"])
    bb_fr["next_settle_ms"] = bb_fr["next_settle_ms"].astype(int)
    bb_fr["hold_hours"] = (bb_fr["next_settle_ms"] - bb_fr["settle_ts_ms"]) / 3600000

    # Filter to extreme negative FR
    neg = bb_fr[bb_fr["fr_bps"] <= -MIN_FR_BPS].copy()
    days = (bb_fr["settle_ts_ms"].max() - bb_fr["settle_ts_ms"].min()) / 86400000

    print(f"   {len(neg):,} neg FR settlements, {neg['symbol'].nunique()} symbols, {days:.0f} days")
    print(f"   Hold hours: {neg['hold_hours'].value_counts().to_dict()}")

    # 2. Download entry + exit + intermediate prices
    # For each trade: get 1m klines from entry to exit
    cache_path = DATA / "historical_fr" / "fullhold_klines.parquet"

    if cache_path.exists():
        print(f"\n2. Loading cached klines from {cache_path}...", flush=True)
        all_klines = pd.read_parquet(cache_path)
        cached_keys = set(zip(all_klines["symbol"], all_klines["settle_ts_ms"]))
        new_neg = neg[~neg.apply(lambda r: (r["symbol"], r["settle_ts_ms"]) in cached_keys, axis=1)]
        print(f"   Cached: {len(cached_keys):,} windows, New: {len(new_neg):,}")
    else:
        all_klines = pd.DataFrame()
        new_neg = neg
        print(f"\n2. Need to download {len(new_neg):,} windows")

    if len(new_neg) > 0:
        print(f"   Downloading {len(new_neg):,} hold windows (20 threads)...", flush=True)

        def download_one(row):
            sym = row.symbol
            start = int(row.settle_ts_ms)
            end = int(row.next_settle_ms) + 60000  # +1min after exit settle
            hold_h = row.hold_hours
            interval = "1" if hold_h <= 2 else "5"

            kl = get_klines(sym, start, end, interval)
            if kl is None:
                return []
            for r in kl:
                r["symbol"] = sym
                r["settle_ts_ms"] = start
            return kl

        new_rows = []
        done = 0
        jobs = list(new_neg.itertuples(index=False))

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(download_one, j): j for j in jobs}
            for f in as_completed(futures):
                done += 1
                result = f.result()
                new_rows.extend(result)
                if done % 500 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print(f"   [{done:,}/{len(jobs):,}] {len(new_rows):,} candles, "
                          f"{elapsed:.0f}s, ~{eta:.0f}s ETA")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            all_klines = pd.concat([all_klines, new_df], ignore_index=True) if len(all_klines) > 0 else new_df
            all_klines = all_klines.drop_duplicates(subset=["symbol", "settle_ts_ms", "ts_ms"])
            all_klines.to_parquet(cache_path, index=False)
            print(f"   Saved {len(all_klines):,} total candles to cache")

    # 3. Build trades with real prices
    print(f"\n3. Building trades with real entry/exit prices...", flush=True)
    t3 = time.time()

    # Group klines
    kl_groups = {}
    for (sym, sts), g in all_klines.groupby(["symbol", "settle_ts_ms"]):
        g = g.sort_values("ts_ms")
        kl_groups[(sym, int(sts))] = {
            "ts": g["ts_ms"].values,
            "open": g["open"].values,
            "high": g["high"].values,
            "low": g["low"].values,
            "close": g["close"].values,
        }

    print(f"   {len(kl_groups):,} kline groups [{time.time()-t3:.1f}s]")

    # SL configs to test
    sl_levels = [0, 25, 50, 75, 100, 150, 200, 300, 500, 9999]  # 0 = no SL

    results = {sl: [] for sl in sl_levels}

    for _, row in neg.iterrows():
        sym = row["symbol"]
        sts = int(row["settle_ts_ms"])
        next_sts = int(row["next_settle_ms"])
        fr_bps = abs(row["fr_bps"])
        next_fr = row["next_fr_bps"]
        fr_income = -next_fr  # neg FR → longs receive

        key = (sym, sts)
        if key not in kl_groups:
            continue

        kl = kl_groups[key]
        if len(kl["close"]) < 2:
            continue

        # Entry: first candle close after settlement
        entry_price = kl["close"][0]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # Exit: last candle close (at/near next settlement)
        exit_price = kl["close"][-1]
        if exit_price <= 0 or np.isnan(exit_price):
            continue

        price_pnl_full = (exit_price - entry_price) / entry_price * 10000

        for sl_bps in sl_levels:
            if sl_bps == 0 or sl_bps >= 9999:
                # No SL — hold to exit
                px_pnl = price_pnl_full
                exit_type = "hold"
                fee = ENTRY_FEE_BPS + SL_FEE_BPS  # market both sides
            else:
                # Check if SL triggered during hold
                sl_price = entry_price * (1 - sl_bps / 10000)
                triggered = False

                for i in range(1, len(kl["low"])):
                    if kl["low"][i] <= sl_price:
                        triggered = True
                        px_pnl = -sl_bps  # stopped at SL level
                        exit_type = "SL"
                        fee = ENTRY_FEE_BPS + SL_FEE_BPS
                        # No FR collected if stopped before next settlement
                        fr_income_adj = 0
                        break

                if not triggered:
                    px_pnl = price_pnl_full
                    exit_type = "hold"
                    fee = ENTRY_FEE_BPS + SL_FEE_BPS
                    fr_income_adj = fr_income

                if triggered:
                    net = px_pnl + fr_income_adj - fee
                    results[sl_bps].append({
                        "symbol": sym, "settle_ts_ms": sts, "fr_bps": fr_bps,
                        "next_fr_bps": next_fr, "fr_income_bps": 0,
                        "price_pnl_bps": px_pnl, "fee_bps": fee,
                        "net_bps": net, "net_usd": net / 10000 * NOTIONAL,
                        "exit_type": exit_type, "hold_hours": row["hold_hours"],
                    })
                    continue

            net = px_pnl + fr_income - fee
            results[sl_bps].append({
                "symbol": sym, "settle_ts_ms": sts, "fr_bps": fr_bps,
                "next_fr_bps": next_fr, "fr_income_bps": fr_income,
                "price_pnl_bps": px_pnl, "fee_bps": fee,
                "net_bps": net, "net_usd": net / 10000 * NOTIONAL,
                "exit_type": exit_type, "hold_hours": row["hold_hours"],
            })

    # 4. Results
    print(f"\n{'='*100}")
    print("RESULTS: LONG after neg FR — hold to NEXT settlement")
    print("=" * 100)
    print(f"Period: {days:.0f} days | Min |FR|: {MIN_FR_BPS} bps")
    print()

    print(f"  {'SL':>6} {'N':>6} {'WR':>5} {'AvgNet':>8} {'Daily':>8} {'ROI/yr':>8}  "
          f"{'SL%':>5} {'Hold%':>5} {'AvgFR':>7} {'AvgPx':>8} {'AvgFee':>7}")
    print("  " + "─" * 105)

    for sl_bps in sl_levels:
        trades = results[sl_bps]
        if not trades:
            continue
        df = pd.DataFrame(trades)
        n = len(df)
        wr = (df["net_usd"] > 0).mean() * 100
        avg = df["net_bps"].mean()
        daily = df["net_usd"].sum() / days
        roi = daily / NOTIONAL * 365 * 100
        sl_pct = (df["exit_type"] == "SL").mean() * 100
        hold_pct = (df["exit_type"] == "hold").mean() * 100
        avg_fr = df["fr_income_bps"].mean()
        avg_px = df["price_pnl_bps"].mean()
        avg_fee = df["fee_bps"].mean()
        label = "NoSL" if sl_bps >= 9999 else f"{sl_bps}bp"
        marker = " <<<" if daily > 0 else ""
        print(f"  {label:>6} {n:>6} {wr:>4.0f}% {avg:>+7.1f} ${daily:>+7,.0f} {roi:>7.0f}%  "
              f"{sl_pct:>4.0f}% {hold_pct:>4.0f}% {avg_fr:>+6.1f} {avg_px:>+7.1f} {avg_fee:>6.1f}{marker}")

    # Detailed on best configs
    print(f"\n{'='*100}")
    print("DETAILED ANALYSIS")
    print("=" * 100)

    for sl_bps in [0, 100, 200, 9999]:
        trades = results[sl_bps]
        if not trades:
            continue
        df = pd.DataFrame(trades)
        daily = df["net_usd"].sum() / days
        label = "NoSL" if sl_bps >= 9999 else f"SL{sl_bps}bp" if sl_bps > 0 else "NoSL(0)"

        print(f"\n  --- {label} ({len(df):,} trades, ${daily:+,.0f}/day) ---")
        print(f"  P&L distribution (bps): "
              f"5th={df['net_bps'].quantile(0.05):+.0f}, "
              f"25th={df['net_bps'].quantile(0.25):+.0f}, "
              f"median={df['net_bps'].median():+.0f}, "
              f"75th={df['net_bps'].quantile(0.75):+.0f}, "
              f"95th={df['net_bps'].quantile(0.95):+.0f}")

        # Max drawdown per trade
        print(f"  Worst trade: {df['net_bps'].min():+.0f} bps (${df['net_usd'].min():+,.0f})")
        print(f"  Best trade: {df['net_bps'].max():+.0f} bps (${df['net_usd'].max():+,.0f})")

        # By FR magnitude
        print(f"  By FR magnitude:")
        for lo, hi in [(20,30), (30,50), (50,100), (100,500)]:
            b = df[(df["fr_bps"] >= lo) & (df["fr_bps"] < hi)]
            if len(b) == 0: continue
            wr = (b["net_usd"] > 0).mean() * 100
            print(f"    FR {lo:>3}-{hi:>3}: {len(b):>5} trades, {wr:.0f}% WR, "
                  f"net {b['net_bps'].mean():+.1f}, price {b['price_pnl_bps'].mean():+.1f}, FR {b['fr_income_bps'].mean():+.1f}")

        # Monthly
        df["month"] = pd.to_datetime(df["settle_ts_ms"], unit="ms").dt.to_period("M")
        monthly = df.groupby("month").agg(
            n=("net_usd", "count"),
            total=("net_usd", "sum"),
            wr=("net_usd", lambda x: (x > 0).mean() * 100),
            avg_px=("price_pnl_bps", "mean"),
        )
        print(f"  Monthly:")
        for m, r in monthly.iterrows():
            print(f"    {str(m):>10}: {int(r['n']):>4} trades, ${r['total']:>+8,.0f} "
                  f"({r['wr']:.0f}% WR, price {r['avg_px']:+.1f} bps)")

        # By hold hours
        print(f"  By hold duration:")
        for h in sorted(df["hold_hours"].unique()):
            hdf = df[df["hold_hours"] == h]
            if len(hdf) < 10: continue
            wr = (hdf["net_usd"] > 0).mean() * 100
            print(f"    {h:.0f}h hold: {len(hdf):>5} trades, {wr:.0f}% WR, "
                  f"net {hdf['net_bps'].mean():+.1f}, price {hdf['price_pnl_bps'].mean():+.1f}")

    # Final comparison
    print(f"\n{'='*100}")
    print(f"STRATEGY COMPARISON ({days:.0f} days)")
    print("=" * 100)
    print(f"\n  {'Strategy':>40} {'Daily':>8} {'Capital':>8} {'ROI/yr':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8}")

    best_sl = None
    best_daily = -1e18
    for sl_bps in sl_levels:
        trades = results[sl_bps]
        if trades:
            d = sum(t["net_usd"] for t in trades) / days
            if d > best_daily:
                best_daily = d
                best_sl = sl_bps

    if best_sl is not None:
        label = "NoSL" if best_sl >= 9999 else f"SL{best_sl}bp"
        print(f"  {'LONG+FR (best SL) ' + label:>40} ${best_daily:>+7,.0f} $    10k {best_daily/10000*365*100:>7.0f}%")

    # Also show a few others
    for sl_bps in [100, 200, 9999]:
        if sl_bps == best_sl: continue
        trades = results[sl_bps]
        if trades:
            d = sum(t["net_usd"] for t in trades) / days
            label = "NoSL" if sl_bps >= 9999 else f"SL{sl_bps}bp"
            print(f"  {'LONG+FR ' + label:>40} ${d:>+7,.0f} $    10k {d/10000*365*100:>7.0f}%")

    print(f"  {'Delta-neutral Bybit 1h (audit)':>40} $   +273 $    20k     498%")
    print(f"  {'Delta-neutral 4-pool (audit)':>40} $   +879 $    80k     401%")

    print(f"\n[{time.time()-t0:.0f}s total]")

if __name__ == "__main__":
    main()
