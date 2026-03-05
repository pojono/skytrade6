#!/usr/bin/env python3
"""Print comma-separated USDT perpetual symbols common to Bybit, Binance, and OKX.

Sorted by Bybit 24h turnover descending. Use as argument placeholder in download commands.

Usage:
  python3 list_symbols.py                          # all common symbols
  python3 list_symbols.py -n 50                    # top 50 by turnover

  # Pipe into downloaders:
  python3 download_bybit_data.py $(python3 list_symbols.py -n 50) 2024-01-01 2026-03-04 -t MetricsLinear
  python3 download_binance_data.py $(python3 list_symbols.py -n 50) 2024-01-01 2026-03-04
  python3 download_okx_data.py $(python3 list_symbols.py -n 50) 2024-01-01 2026-03-04
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


def get_binance_symbols() -> set[str]:
    data = fetch_json("https://fapi.binance.com/fapi/v1/ticker/24hr")
    return {t["symbol"] for t in data if t["symbol"].endswith("USDT") and "_" not in t["symbol"]}


def get_okx_symbols() -> set[str]:
    data = fetch_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
    symbols = set()
    for t in data.get("data", []):
        inst_id = t["instId"]
        if inst_id.endswith("-USDT-SWAP"):
            symbols.add(inst_id.replace("-USDT-SWAP", "") + "USDT")
    return symbols


def main():
    parser = argparse.ArgumentParser(
        description="Print comma-separated USDT perp symbols common to Bybit, Binance, and OKX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output is a single line: BTCUSDT,ETHUSDT,SOLUSDT,...\n"
               "Sorted by Bybit 24h turnover. Use -n to limit.",
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=None,
        help="Only output top N symbols (by Bybit 24h turnover)",
    )
    args = parser.parse_args()

    sys.stderr.write("Fetching tickers from Bybit, Binance, OKX...\n")
    bybit = get_bybit_symbols()
    binance = get_binance_symbols()
    okx = get_okx_symbols()

    common = set(bybit.keys()) & binance & okx
    common_sorted = sorted(common, key=lambda s: -bybit[s])

    if args.top:
        common_sorted = common_sorted[: args.top]

    sys.stderr.write(f"{len(common_sorted)} symbols\n")
    print(",".join(common_sorted))


if __name__ == "__main__":
    main()
