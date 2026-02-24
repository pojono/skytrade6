#!/usr/bin/env python3
"""Download Bitget historical funding rates (1h + 4h coins) and run analysis."""
import json, time, os, sys
import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
import builtins
_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True)
    _print(*a, **k)

NOTIONAL = 10000
START_MS = int(datetime(2025, 11, 10, tzinfo=timezone.utc).timestamp() * 1000)

def api_get(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def download_symbol(sym):
    records = []
    end_time = int(time.time() * 1000)
    for _ in range(30):
        url = (f"https://api.bitget.com/api/v2/mix/market/history-fund-rate"
               f"?symbol={sym}&productType=usdt-futures&pageSize=100&endTime={end_time}")
        try:
            data = api_get(url)
            if data.get("code") != "00000": break
            batch = data.get("data", [])
            if not batch: break
            for r in batch:
                ft = int(r.get("fundingTime", 0))
                if ft >= START_MS:
                    records.append((sym, float(r["fundingRate"]), ft))
            oldest = min(int(r["fundingTime"]) for r in batch)
            if oldest <= START_MS: break
            end_time = oldest - 1
        except:
            break
    return records

def run_hold(df, coins, entry_th=20, exit_th=8, max_pos=1, rt_bps=50.0):
    sub = df[df["symbol"].isin(coins)].copy()
    if len(sub) == 0: return pd.DataFrame()
    sub = sub.sort_values(["symbol", "fundingTime"])
    sub["settle_hour"] = sub["fundingTime"].dt.floor("h")
    positions = {}; closed = []
    for st in sorted(sub["settle_hour"].unique()):
        hour_data = sub[sub["settle_hour"] == st]
        to_close = []
        for s, pos in positions.items():
            match = hour_data[hour_data["symbol"] == s]
            if len(match) > 0:
                fr = float(match.iloc[0]["fr_signed_bps"])
                collected = fr if pos["direction"] == "short_fut" else -fr
                pos["total_fr_bps"] += collected; pos["n_settle"] += 1
                if abs(fr) < exit_th: to_close.append(s)
        for s in to_close: closed.append({**positions.pop(s), "exit_time": st})
        if len(positions) < max_pos:
            cands = hour_data[(hour_data["fr_bps"] >= entry_th) & (~hour_data["symbol"].isin(positions.keys()))]
            best = cands.nlargest(max_pos - len(positions), "fr_bps")
            for _, row in best.iterrows():
                fr = float(row["fr_signed_bps"])
                positions[row["symbol"]] = {
                    "symbol": row["symbol"], "entry_time": st,
                    "total_fr_bps": 0, "n_settle": 0,
                    "direction": "short_fut" if fr > 0 else "long_fut"}
    for s, pos in positions.items():
        closed.append({**pos, "exit_time": sorted(sub["settle_hour"].unique())[-1]})
    if not closed: return pd.DataFrame()
    cdf = pd.DataFrame(closed)
    cdf["net_usd"] = cdf["total_fr_bps"] / 10000 * NOTIONAL - rt_bps / 10000 * NOTIONAL
    cdf["profitable"] = cdf["net_usd"] > 0
    return cdf

def main():
    t0 = time.time()

    # 1. Metadata
    print("1. Fetching Bitget metadata...")
    current = api_get("https://api.bitget.com/api/v2/mix/market/current-fund-rate?productType=usdt-futures")
    sym_intervals = {}
    for item in current.get("data", []):
        sym_intervals[item["symbol"]] = int(item.get("fundingRateInterval", "8"))

    spot_data = api_get("https://api.bitget.com/api/v2/spot/public/symbols")
    spot_syms = {item["symbol"] for item in spot_data.get("data", [])
                 if item.get("quoteCoin") == "USDT" and item.get("status") == "online"}

    target = sorted(s for s, iv in sym_intervals.items() if iv in (1, 4))
    n1h = sum(1 for s in target if sym_intervals[s] == 1)
    n4h = len(target) - n1h
    print(f"   {len(sym_intervals)} perps total, downloading {len(target)} (1h:{n1h}, 4h:{n4h}), {len(spot_syms)} spot")

    # 2. Parallel download
    print(f"2. Downloading FR history (20 threads)...")
    all_records = []
    done = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download_symbol, sym): sym for sym in target}
        for future in as_completed(futures):
            done += 1
            all_records.extend(future.result())
            if done % 50 == 0 or done == len(target):
                print(f"   {done}/{len(target)} done, {len(all_records):,} records, {time.time()-t0:.0f}s")

    # 3. Build DataFrame
    print("3. Processing...")
    df = pd.DataFrame(all_records, columns=["symbol", "fundingRate", "fundingTime"])
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.tz_localize("UTC")
    df = df.drop_duplicates(subset=["symbol", "fundingTime"]).sort_values(["symbol", "fundingTime"])
    df["interval_h"] = df["symbol"].map(lambda s: sym_intervals.get(s, 8))
    df["fr_bps"] = df["fundingRate"].abs() * 10000
    df["fr_signed_bps"] = df["fundingRate"] * 10000
    df.to_parquet("data_all/historical_fr/bitget_fr_history.parquet", index=False)
    days = (df["fundingTime"].max() - df["fundingTime"].min()).total_seconds() / 86400
    print(f"   {len(df):,} records, {df['symbol'].nunique()} symbols, {days:.0f} days")
    print(f"   {df['fundingTime'].min().date()} to {df['fundingTime'].max().date()}")

    # 4. Tradeability
    bg_1h = set(df[df["interval_h"] == 1]["symbol"].unique())
    bg_4h = set(df[df["interval_h"] == 4]["symbol"].unique())
    bg_1h_spot = bg_1h & spot_syms
    bg_4h_spot = bg_4h & spot_syms
    print(f"\n4. TRADEABILITY:")
    print(f"   1h: {len(bg_1h)} total, {len(bg_1h_spot)} with spot ({len(bg_1h_spot)/max(len(bg_1h),1)*100:.0f}%)")
    print(f"   4h: {len(bg_4h)} total, {len(bg_4h_spot)} with spot ({len(bg_4h_spot)/max(len(bg_4h),1)*100:.0f}%)")

    # 5. Backtest
    print(f"\n5. BACKTEST (50 bps RT, $10k notional, {days:.0f} days)")
    bg_1h_sub = df[df["interval_h"] == 1]
    bg_4h_sub = df[df["interval_h"] == 4]

    print(f"\n   === BITGET 1h, SPOT-AVAILABLE ({len(bg_1h_spot)} coins) ===")
    print(f"   {'Pos':>4} {'Trades':>7} {'WR':>5} {'Daily':>8} {'Capital':>10} {'ROI/yr':>8} {'Marginal':>10}")
    prev = 0
    for mp in [1, 2, 3, 4, 5]:
        cdf = run_hold(bg_1h_sub, bg_1h_spot, 20, 8, mp, 50)
        if len(cdf) == 0:
            print(f"   {mp:>4}       0    — $      0 ${mp*20000:>8,}       — $      +0"); continue
        d = cdf["net_usd"].sum() / days; wr = cdf["profitable"].mean() * 100
        print(f"   {mp:>4} {len(cdf):>7} {wr:>4.0f}% ${d:>+7,.0f} ${mp*20000:>8,} {d/(mp*20000)*365*100:>7.0f}% ${d-prev:>+8,.0f}")
        prev = d

    print(f"\n   Threshold sensitivity (1 pos, 1h, spot-available):")
    print(f"   {'Entry':>6} {'Exit':>5} {'Trades':>7} {'WR':>5} {'Daily':>8} {'ROI/yr':>8}")
    for entry, exit_th in [(10,5), (15,5), (20,5), (20,8), (25,5), (30,5), (50,8)]:
        cdf = run_hold(bg_1h_sub, bg_1h_spot, entry, exit_th, 1, 50)
        if len(cdf) == 0:
            print(f"   {entry:>5}bp {exit_th:>4}bp       0    — $      0       —"); continue
        d = cdf["net_usd"].sum() / days; wr = cdf["profitable"].mean() * 100
        print(f"   {entry:>5}bp {exit_th:>4}bp {len(cdf):>7} {wr:>4.0f}% ${d:>+7,.0f} {d/20000*365*100:>7.0f}%")

    print(f"\n   === BITGET 4h, SPOT-AVAILABLE ({len(bg_4h_spot)} coins) ===")
    for entry, exit_th in [(20,5), (30,5), (30,8)]:
        cdf = run_hold(bg_4h_sub, bg_4h_spot, entry, exit_th, 1, 50)
        if len(cdf) == 0:
            print(f"   e={entry} x={exit_th}: 0 trades"); continue
        d = cdf["net_usd"].sum() / days; wr = cdf["profitable"].mean() * 100
        print(f"   e={entry} x={exit_th}: ${d:>+,.0f}/day, {wr:.0f}% WR, {len(cdf)} trades, {d/20000*365*100:.0f}% ROI")

    # Monthly
    print(f"\n   Monthly (1 pos, 1h spot, e=20 x=5):")
    tmp = bg_1h_sub[bg_1h_sub["symbol"].isin(bg_1h_spot)].copy()
    tmp["month"] = tmp["fundingTime"].dt.to_period("M")
    for m in sorted(tmp["month"].unique()):
        mdf = tmp[tmp["month"] == m]
        mdays = (mdf["fundingTime"].max() - mdf["fundingTime"].min()).total_seconds() / 86400
        if mdays < 5: continue
        cdf = run_hold(mdf, bg_1h_spot, 20, 5, 1, 50)
        if len(cdf) == 0: print(f"   {str(m):>10}: $0, 0 trades"); continue
        d = cdf["net_usd"].sum() / mdays; wr = cdf["profitable"].mean() * 100
        print(f"   {str(m):>10}: ${d:>+,.0f}/day, {wr:.0f}% WR, {len(cdf)} trades")

    # Combined
    cdf1 = run_hold(bg_1h_sub, bg_1h_spot, 20, 5, 1, 50)
    cdf4 = run_hold(bg_4h_sub, bg_4h_spot, 30, 5, 1, 50)
    d1 = cdf1["net_usd"].sum() / days if len(cdf1) > 0 else 0
    d4 = cdf4["net_usd"].sum() / days if len(cdf4) > 0 else 0

    print(f"\n{'='*90}")
    print(f"FULL COMBINED: All exchanges, 1 position per pool, 50 bps RT")
    print(f"{'='*90}")
    print(f"\n  {'Pool':>15} {'Coins':>6} {'Daily':>8} {'WR':>5} {'Capital':>8} {'ROI/yr':>8}")
    pools = [
        ("Bybit 1h",   32, 273, "59%"),
        ("Binance 1h", 13, 218, "81%"),
        ("Binance 4h", 231, 177, "76%"),
    ]
    if len(cdf1) > 0:
        pools.append(("Bitget 1h", len(bg_1h_spot), round(d1), f"{cdf1['profitable'].mean()*100:.0f}%"))
    if len(cdf4) > 0:
        pools.append(("Bitget 4h", len(bg_4h_spot), round(d4), f"{cdf4['profitable'].mean()*100:.0f}%"))

    total_d = 0
    for name, coins, daily, wr in pools:
        total_d += daily
        print(f"  {name:>15} {coins:>6} ${daily:>+7,} {wr:>5} ${'20k':>7} {daily/20000*365*100:>7.0f}%")
    n = len(pools)
    print(f"  {'─'*15} {'─'*6} {'─'*8} {'─'*5} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL':>15} {'':>6} ${total_d:>+7,} {'':>5} ${n*20:>6}k {total_d/(n*20000)*365*100:>7.0f}%")
    print(f"\n[{time.time()-t0:.0f}s total]")

if __name__ == "__main__":
    main()
