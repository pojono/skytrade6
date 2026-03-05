#!/usr/bin/env python3
"""Print comma-separated USDT perpetual symbols common to Bybit, Binance, and OKX.

Sorted by Bybit 24h turnover descending. Use as argument placeholder in download commands.

Usage:
  python3 list_symbols.py                              # all common symbols
  python3 list_symbols.py --limit 50                    # top 50 by turnover
  python3 list_symbols.py --limit 15 --offset 5         # symbols ranked 6th-20th
  python3 list_symbols.py --limit 20 -v                 # top 20 with detailed table

  # Pipe into downloaders:
  python3 download_bybit_data.py $(python3 list_symbols.py --limit 50) 2024-01-01 2026-03-04 -t MetricsLinear
  python3 download_binance_data.py $(python3 list_symbols.py --limit 50) 2024-01-01 2026-03-04
  python3 download_okx_data.py $(python3 list_symbols.py --limit 50) 2024-01-01 2026-03-04
"""

import argparse
import json
import sys
import urllib.request


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "datalake/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_bybit_symbols() -> dict[str, float]:
    data = fetch_json("https://api.bybit.com/v5/market/tickers?category=linear")
    symbols = {}
    for t in data["result"]["list"]:
        sym = t["symbol"]
        if not sym.endswith("USDT") or "-" in sym:
            continue
        symbols[sym] = float(t.get("turnover24h", "0"))
    return symbols


def get_binance_symbols() -> dict[str, float]:
    data = fetch_json("https://fapi.binance.com/fapi/v1/ticker/24hr")
    return {t["symbol"]: float(t.get("quoteVolume", "0"))
            for t in data if t["symbol"].endswith("USDT") and "_" not in t["symbol"]}


def get_okx_symbols() -> dict[str, float]:
    data = fetch_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
    symbols = {}
    for t in data.get("data", []):
        inst_id = t["instId"]
        if inst_id.endswith("-USDT-SWAP"):
            sym = inst_id.replace("-USDT-SWAP", "") + "USDT"
            symbols[sym] = float(t.get("volCcy24h", "0"))
    return symbols


def fmt_vol(val: float) -> str:
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.0f}M"
    if val >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:.0f}"


def main():
    parser = argparse.ArgumentParser(
        description="Print comma-separated USDT perp symbols common to Bybit, Binance, and OKX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output is a single line: BTCUSDT,ETHUSDT,SOLUSDT,...\n"
               "Sorted by Bybit 24h turnover. Use --limit and --offset to paginate.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max number of symbols to return (default: all)",
    )
    parser.add_argument(
        "--offset", "-o",
        type=int,
        default=0,
        help="Skip first N symbols (default: 0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print a detailed table (to stderr) alongside the symbol list",
    )
    args = parser.parse_args()

    sys.stderr.write("Fetching tickers from Bybit, Binance, OKX...\n")
    bybit = get_bybit_symbols()
    binance = get_binance_symbols()
    okx = get_okx_symbols()

    common = set(bybit.keys()) & set(binance.keys()) & set(okx.keys())
    common_sorted = sorted(common, key=lambda s: -bybit[s])

    total = len(common_sorted)
    common_sorted = common_sorted[args.offset:]
    if args.limit is not None:
        common_sorted = common_sorted[: args.limit]

    sys.stderr.write(f"{len(common_sorted)} of {total} symbols (offset={args.offset})\n")

    if args.verbose:
        w = sys.stderr.write
        w(f"\n{'#':>4}  {'Symbol':<18} {'Turnover 24h':>12}  {'Bybit':^5} {'Binance':^7} {'OKX':^5}\n")
        w(f"{'─'*4}  {'─'*18} {'─'*12}  {'─'*5} {'─'*7} {'─'*5}\n")
        for i, sym in enumerate(common_sorted, args.offset + 1):
            vol = fmt_vol(bybit.get(sym, 0))
            bb = '  ✓  ' if sym in bybit else '  -  '
            bn = '   ✓   ' if sym in binance else '   -   '
            ox = '  ✓  ' if sym in okx else '  -  '
            w(f"{i:>4}  {sym:<18} {vol:>12}  {bb}{bn}{ox}\n")
        w("\n")

    print(",".join(common_sorted))


if __name__ == "__main__":
    main()
