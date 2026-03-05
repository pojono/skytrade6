#!/usr/bin/env python3

import argparse
import json
import requests


def fetch_binance_symbols():
    """Fetch USDT perpetual futures symbols from Binance."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    symbols = [s['symbol'] for s in data['symbols'] if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT']
    return symbols

def fetch_bybit_symbols():
    """Fetch USDT perpetual futures symbols from Bybit."""
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    symbols = [s['symbol'] for s in data['result']['list'] if s['contractType'] == 'LinearPerpetual' and s['quoteCoin'] == 'USDT']
    return symbols

def fetch_okx_symbols():
    """Fetch USDT perpetual futures symbols from OKX."""
    url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print("OKX API response code:", data.get('code', 'N/A'))
    print("OKX API response message:", data.get('msg', 'N/A'))
    print("OKX data length:", len(data.get('data', [])))
    data_list = data.get('data', [])
    symbols = []
    for s in data_list:
        if s.get('instType') == 'SWAP':
            inst_id = s.get('instId', '')
            settle_ccy = s.get('settleCcy', '')
            if inst_id.endswith('-USDT-SWAP') and settle_ccy == 'USDT':
                symbol = inst_id.replace('-USDT-SWAP', 'USDT')
                symbols.append(symbol)
            else:
                print(f"Skipping {inst_id}: settleCcy={settle_ccy}")
        else:
            print(f"Skipping non-SWAP: {s.get('instId', 'Unknown')}")
    print(f"Extracted {len(symbols)} USDT perpetual symbols from OKX")
    return symbols

def main():
    parser = argparse.ArgumentParser(description="List USDT perpetual futures symbols across exchanges.")
    parser.add_argument("--top", type=int, help="Limit to top N symbols by volume (not implemented yet)")
    parser.add_argument("--common-only", action="store_true", help="Only show symbols available on all exchanges")
    args = parser.parse_args()

    print("Fetching symbols from Binance...")
    binance_symbols = set(fetch_binance_symbols())
    print(f"Binance: {len(binance_symbols)} USDT perpetual symbols")

    print("Fetching symbols from Bybit...")
    bybit_symbols = set(fetch_bybit_symbols())
    print(f"Bybit: {len(bybit_symbols)} USDT perpetual symbols")

    print("Fetching symbols from OKX...")
    okx_symbols = set(fetch_okx_symbols())
    print(f"OKX: {len(okx_symbols)} USDT perpetual symbols")

    if args.common_only:
        common_symbols = binance_symbols & bybit_symbols & okx_symbols
        print(f"\nCommon symbols across all 3 exchanges: {len(common_symbols)}")
        for symbol in sorted(common_symbols):
            print(symbol)
    else:
        print("\nBinance symbols:")
        for symbol in sorted(binance_symbols)[:10]:
            print(symbol)
        if len(binance_symbols) > 10:
            print(f"... and {len(binance_symbols) - 10} more")

        print("\nBybit symbols:")
        for symbol in sorted(bybit_symbols)[:10]:
            print(symbol)
        if len(bybit_symbols) > 10:
            print(f"... and {len(bybit_symbols) - 10} more")

        print("\nOKX symbols:")
        for symbol in sorted(okx_symbols)[:10]:
            print(symbol)
        if len(okx_symbols) > 10:
            print(f"... and {len(okx_symbols) - 10} more")

        common_symbols = binance_symbols & bybit_symbols & okx_symbols
        print(f"\nCommon symbols across all 3 exchanges: {len(common_symbols)}")
        for symbol in sorted(common_symbols)[:10]:
            print(symbol)
        if len(common_symbols) > 10:
            print(f"... and {len(common_symbols) - 10} more")

if __name__ == "__main__":
    main()
