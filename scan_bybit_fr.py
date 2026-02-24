#!/usr/bin/env python3
"""
Bybit Funding Rate Scanner — shows ranked coin opportunities RIGHT NOW.

Fetches live data from Bybit public API (no auth needed):
  - All USDT perpetual tickers (funding rate, next settlement, spreads)
  - Spot tickers (for spread comparison)

Usage:
  python3 scan_bybit_fr.py              # default: show top 20, all intervals
  python3 scan_bybit_fr.py --top 50     # show top 50
  python3 scan_bybit_fr.py --1h         # only 1h settlement coins
  python3 scan_bybit_fr.py --min-fr 30  # minimum |FR| in bps
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

# ─── Config ──────────────────────────────────────────────────────────
BYBIT_BASE = "https://api.bybit.com"
NOTIONAL = 10_000  # USD per position
RT_COST_BPS = 50   # realistic round-trip cost (fees + spread + borrow)


def api_get(path, params=None):
    """GET request to Bybit public API."""
    url = f"{BYBIT_BASE}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get("retCode") != 0:
            print(f"  API error: {data.get('retMsg')}", file=sys.stderr)
            return None
        return data.get("result", {})
    except URLError as e:
        print(f"  Request failed: {e}", file=sys.stderr)
        return None


def fetch_futures_tickers():
    """Fetch all USDT linear perpetual tickers."""
    result = api_get("/v5/market/tickers", {"category": "linear"})
    if not result:
        return []
    tickers = []
    for item in result.get("list", []):
        sym = item.get("symbol", "")
        # Only USDT perpetuals (skip USDC, inverse, delivery)
        if not sym.endswith("USDT"):
            continue
        # Skip delivery contracts
        if item.get("deliveryTime", "0") != "0":
            continue
        try:
            fr = float(item.get("fundingRate", "0"))
            next_funding_ms = int(item.get("nextFundingTime", "0"))
            interval_h = int(item.get("fundingIntervalHour", "8") or "8")
            last_price = float(item.get("lastPrice", "0"))
            bid1 = float(item.get("bid1Price", "0"))
            ask1 = float(item.get("ask1Price", "0"))
            volume_24h = float(item.get("turnover24h", "0"))
            oi_value = float(item.get("openInterestValue", "0"))

            if last_price <= 0:
                continue

            fut_spread_bps = (ask1 - bid1) / last_price * 10000 if bid1 > 0 and ask1 > 0 else 999

            tickers.append({
                "symbol": sym,
                "fr": fr,
                "fr_bps": abs(fr) * 10000,
                "fr_signed_bps": fr * 10000,
                "interval_h": interval_h,
                "next_funding_ts": next_funding_ms / 1000,
                "last_price": last_price,
                "bid1": bid1,
                "ask1": ask1,
                "fut_spread_bps": fut_spread_bps,
                "volume_24h_usd": volume_24h,
                "oi_usd": oi_value,
            })
        except (ValueError, TypeError):
            continue
    return tickers


def fetch_spot_tickers():
    """Fetch all spot tickers for spread comparison."""
    result = api_get("/v5/market/tickers", {"category": "spot"})
    if not result:
        return {}
    spot = {}
    for item in result.get("list", []):
        sym = item.get("symbol", "")
        try:
            bid1 = float(item.get("bid1Price", "0"))
            ask1 = float(item.get("ask1Price", "0"))
            last = float(item.get("lastPrice", "0"))
            if last > 0 and bid1 > 0 and ask1 > 0:
                spot[sym] = {
                    "bid1": bid1,
                    "ask1": ask1,
                    "last": last,
                    "spread_bps": (ask1 - bid1) / last * 10000,
                }
        except (ValueError, TypeError):
            continue
    return spot


def main():
    parser = argparse.ArgumentParser(description="Bybit FR Scanner")
    parser.add_argument("--top", type=int, default=20, help="Show top N coins")
    parser.add_argument("--1h", dest="only_1h", action="store_true", help="Only 1h settlement coins")
    parser.add_argument("--min-fr", type=float, default=0, help="Minimum |FR| in bps")
    parser.add_argument("--min-volume", type=float, default=0, help="Minimum 24h volume in USD")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"\n{'='*100}")
    print(f"  BYBIT FUNDING RATE SCANNER — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*100}")

    # Fetch data
    print("  Fetching futures tickers...", end="", flush=True)
    futures = fetch_futures_tickers()
    print(f" {len(futures)} symbols")

    print("  Fetching spot tickers...", end="", flush=True)
    spot = fetch_spot_tickers()
    print(f" {len(spot)} symbols")

    if not futures:
        print("  ERROR: No futures data received.")
        sys.exit(1)

    # Enrich with spot data
    for t in futures:
        sym = t["symbol"]
        if sym in spot:
            t["spot_spread_bps"] = spot[sym]["spread_bps"]
            t["total_spread_bps"] = t["fut_spread_bps"] + spot[sym]["spread_bps"]
            t["has_spot"] = True
        else:
            t["spot_spread_bps"] = None
            t["total_spread_bps"] = None
            t["has_spot"] = False

    # Filter
    filtered = futures
    if args.only_1h:
        filtered = [t for t in filtered if t["interval_h"] == 1]
    if args.min_fr > 0:
        filtered = [t for t in filtered if t["fr_bps"] >= args.min_fr]
    if args.min_volume > 0:
        filtered = [t for t in filtered if t["volume_24h_usd"] >= args.min_volume]

    # Sort by |FR| descending
    filtered.sort(key=lambda x: x["fr_bps"], reverse=True)
    top = filtered[:args.top]

    if not top:
        print("\n  No coins match the filters.")
        sys.exit(0)

    # Time to next funding
    def time_to_next(ts):
        diff = ts - now.timestamp()
        if diff < 0:
            return "SETTLED"
        m = int(diff // 60)
        s = int(diff % 60)
        return f"{m}m{s:02d}s"

    # Direction label
    def direction(fr):
        if fr < 0:
            return "SHORT spot + LONG fut"
        elif fr > 0:
            return "LONG spot  + SHORT fut"
        return "—"

    # Estimated income per settlement
    def est_income(fr_bps):
        return fr_bps / 10000 * NOTIONAL

    # ─── Summary header ─────────────────────────────────────────────
    n_1h = sum(1 for t in futures if t["interval_h"] == 1)
    n_4h = sum(1 for t in futures if t["interval_h"] == 4)
    n_8h = sum(1 for t in futures if t["interval_h"] == 8)
    above_20 = sum(1 for t in futures if t["fr_bps"] >= 20)
    above_50 = sum(1 for t in futures if t["fr_bps"] >= 50)

    print(f"\n  Symbols: {len(futures)} total ({n_1h} × 1h, {n_4h} × 4h, {n_8h} × 8h)")
    print(f"  |FR| ≥ 20 bps: {above_20} coins | |FR| ≥ 50 bps: {above_50} coins")
    print(f"  RT cost assumption: {RT_COST_BPS} bps | Notional: ${NOTIONAL:,}")

    # ─── Table ───────────────────────────────────────────────────────
    print(f"\n  {'#':>3} {'Symbol':>14} {'Intv':>4} {'FR bps':>8} {'Sign':>5} {'Next':>8} "
          f"{'$/settle':>9} {'FutSpr':>7} {'SpotSpr':>8} {'Vol24h':>10} {'OI':>10} "
          f"{'Direction':>24}")
    print(f"  {'—'*3} {'—'*14} {'—'*4} {'—'*8} {'—'*5} {'—'*8} "
          f"{'—'*9} {'—'*7} {'—'*8} {'—'*10} {'—'*10} "
          f"{'—'*24}")

    for i, t in enumerate(top, 1):
        sign = "+" if t["fr"] >= 0 else "−"
        spot_spr = f"{t['spot_spread_bps']:.1f}" if t["spot_spread_bps"] is not None else "N/A"
        vol_str = f"${t['volume_24h_usd']/1e6:.1f}M" if t['volume_24h_usd'] >= 1e6 else f"${t['volume_24h_usd']/1e3:.0f}K"
        oi_str = f"${t['oi_usd']/1e6:.1f}M" if t['oi_usd'] >= 1e6 else f"${t['oi_usd']/1e3:.0f}K"
        income = est_income(t["fr_bps"])

        # Profitable after RT cost AND has spot market for hedging?
        profitable = t["fr_bps"] > RT_COST_BPS and t["has_spot"]
        if not t["has_spot"]:
            marker = "⊘"  # no spot market
        elif profitable:
            marker = "✓"
        else:
            marker = " "

        print(f"  {i:>3} {t['symbol']:>14} {t['interval_h']:>3}h {t['fr_bps']:>7.1f} {sign:>5} "
              f"{time_to_next(t['next_funding_ts']):>8} "
              f"${income:>7.1f} {t['fut_spread_bps']:>6.1f} {spot_spr:>8} "
              f"{vol_str:>10} {oi_str:>10} "
              f"{direction(t['fr']):>24} {marker}")

    # ─── Action summary ──────────────────────────────────────────────
    actionable = [t for t in top if t["fr_bps"] >= 20 and t["has_spot"] 
                   and t.get("total_spread_bps", 999) < 80]
    if actionable:
        print(f"\n  {'='*100}")
        print(f"  ACTION: Top {min(3, len(actionable))} coins to trade NOW (|FR| ≥ 20 bps, spot available)")
        print(f"  {'='*100}")
        for i, t in enumerate(actionable[:3], 1):
            total_spr = t.get("total_spread_bps", 0) or 0
            net_per_settle = est_income(t["fr_bps"]) - RT_COST_BPS / 10000 * NOTIONAL / max(t.get("fr_bps", 1) / 20, 1)
            settles_to_breakeven = RT_COST_BPS / t["fr_bps"] if t["fr_bps"] > 0 else 999

            print(f"\n  #{i} {t['symbol']} — {t['fr_bps']:.1f} bps ({t['interval_h']}h settlement)")
            print(f"      Direction:       {direction(t['fr'])}")
            print(f"      FR per settle:   ${est_income(t['fr_bps']):.2f} on ${NOTIONAL:,}")
            print(f"      RT cost:         ${RT_COST_BPS/10000*NOTIONAL:.2f} (fees + spread + borrow)")
            print(f"      Break-even:      {settles_to_breakeven:.1f} settlements ({settles_to_breakeven * t['interval_h']:.1f}h)")
            print(f"      Next settle in:  {time_to_next(t['next_funding_ts'])}")
            print(f"      Spread: fut {t['fut_spread_bps']:.1f}bp + spot {t.get('spot_spread_bps', 0) or 0:.1f}bp = {total_spr:.1f}bp")
            print(f"      24h volume:      ${t['volume_24h_usd']:,.0f}")
            print(f"      Open interest:   ${t['oi_usd']:,.0f}")
    else:
        print(f"\n  ⚠ No coins with |FR| ≥ 20 bps and spot market available right now.")
        print(f"    Check back at the next settlement hour.")

    print()


if __name__ == "__main__":
    main()
