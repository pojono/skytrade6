#!/usr/bin/env python3
"""List and download USDT perpetual symbols available on Bybit, Binance, and OKX.

Shows:
  1. Symbols present on ALL exchanges (sorted by Bybit 24h turnover)
  2. Symbols exclusive to each exchange

Sources:
  - Bybit:   https://api.bybit.com/v5/market/tickers?category=linear
  - Binance: https://fapi.binance.com/fapi/v1/ticker/24hr
  - OKX:     https://www.okx.com/api/v5/market/tickers?instType=SWAP

Usage:
  python3 list_symbols.py                                    # show all
  python3 list_symbols.py -n 100                             # top 100 common symbols
  python3 list_symbols.py --common-only -n 100               # comma-separated for piping

  # Download top 100 common symbols from all exchanges (futures/swap):
  python3 list_symbols.py --download all -n 100 --start 2025-07-01 --end 2026-03-04

  # Download spot data from all exchanges:
  python3 list_symbols.py --download all -n 100 --start 2025-07-01 --end 2026-03-04 --market spot

  # Download from one exchange only:
  python3 list_symbols.py --download bybit -n 50 --start 2025-07-01 --end 2026-03-04
  python3 list_symbols.py --download binance -n 50 --start 2025-07-01 --end 2026-03-04 -c 10
  python3 list_symbols.py --download okx -n 50 --start 2025-07-01 --end 2026-03-04

  # Download with specific data types:
  python3 list_symbols.py --download all -n 100 --start 2025-07-01 --end 2026-03-04 -t klines,fundingRate
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def fetch_json(url: str) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "datalake/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_bybit_symbols() -> dict[str, float]:
    """Return {symbol: turnover24h} for Bybit USDT linear perps."""
    data = fetch_json("https://api.bybit.com/v5/market/tickers?category=linear")
    symbols = {}
    for t in data["result"]["list"]:
        sym = t["symbol"]
        if not sym.endswith("USDT") or "-" in sym:
            continue
        symbols[sym] = float(t.get("turnover24h", "0"))
    return symbols


def get_binance_symbols() -> dict[str, float]:
    """Return {symbol: quoteVolume (24h turnover)} for Binance USDT-M perps."""
    data = fetch_json("https://fapi.binance.com/fapi/v1/ticker/24hr")
    symbols = {}
    for t in data:
        sym = t["symbol"]
        if not sym.endswith("USDT") or "_" in sym:
            continue
        symbols[sym] = float(t.get("quoteVolume", "0"))
    return symbols


def get_okx_symbols() -> dict[str, float]:
    """Return {symbol: volCcy24h (24h turnover in quote)} for OKX USDT SWAP perps."""
    data = fetch_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
    symbols = {}
    for t in data.get("data", []):
        inst_id = t["instId"]
        # Only USDT swaps, e.g. BTC-USDT-SWAP -> BTCUSDT
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        base = inst_id.replace("-USDT-SWAP", "")
        sym = base + "USDT"
        symbols[sym] = float(t.get("volCcy24h", "0"))
    return symbols


def fmt_turnover(val: float) -> str:
    """Format turnover as human-readable string."""
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.0f}M"
    if val >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:.0f}"


def print_report(common_sorted, bybit, binance, okx, bybit_only, binance_only, okx_only, top):
    """Print the full symbol comparison report."""
    print(f"\n{'='*92}")
    print(f"  COMMON SYMBOLS: {len(common_sorted)}"
          + (f" (top {top})" if top else "")
          + f"  |  Bybit-only: {len(bybit_only)}  |  Binance-only: {len(binance_only)}"
          + f"  |  OKX-only: {len(okx_only)}")
    print(f"{'='*92}")

    # --- Common ---
    print(f"\n{'─'*92}")
    print(f"  COMMON ({len(common_sorted)} symbols) — sorted by Bybit 24h turnover")
    print(f"{'─'*92}")
    print(f"  {'#':>4}  {'Symbol':<20} {'Bybit 24h':>12} {'Binance 24h':>12} {'OKX 24h':>12}")
    print(f"  {'─'*4}  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    for i, sym in enumerate(common_sorted, 1):
        bb = fmt_turnover(bybit.get(sym, 0))
        bn = fmt_turnover(binance.get(sym, 0))
        ox = fmt_turnover(okx.get(sym, 0))
        print(f"  {i:>4}  {sym:<20} {bb:>12} {bn:>12} {ox:>12}")

    # --- Bybit-only ---
    bybit_only_sorted = sorted(bybit_only, key=lambda s: -bybit[s])
    print(f"\n{'─'*92}")
    print(f"  BYBIT-ONLY ({len(bybit_only_sorted)} symbols)")
    print(f"{'─'*92}")
    if bybit_only_sorted:
        print(f"  {'#':>4}  {'Symbol':<20} {'Bybit 24h':>12}")
        print(f"  {'─'*4}  {'─'*20} {'─'*12}")
        for i, sym in enumerate(bybit_only_sorted, 1):
            bb = fmt_turnover(bybit[sym])
            print(f"  {i:>4}  {sym:<20} {bb:>12}")
    else:
        print("  (none)")

    # --- Binance-only ---
    binance_only_sorted = sorted(binance_only, key=lambda s: -binance[s])
    print(f"\n{'─'*92}")
    print(f"  BINANCE-ONLY ({len(binance_only_sorted)} symbols)")
    print(f"{'─'*92}")
    if binance_only_sorted:
        print(f"  {'#':>4}  {'Symbol':<20} {'Binance 24h':>12}")
        print(f"  {'─'*4}  {'─'*20} {'─'*12}")
        for i, sym in enumerate(binance_only_sorted, 1):
            bn = fmt_turnover(binance[sym])
            print(f"  {i:>4}  {sym:<20} {bn:>12}")
    else:
        print("  (none)")

    # --- OKX-only ---
    okx_only_sorted = sorted(okx_only, key=lambda s: -okx[s])
    print(f"\n{'─'*92}")
    print(f"  OKX-ONLY ({len(okx_only_sorted)} symbols)")
    print(f"{'─'*92}")
    if okx_only_sorted:
        print(f"  {'#':>4}  {'Symbol':<20} {'OKX 24h':>12}")
        print(f"  {'─'*4}  {'─'*20} {'─'*12}")
        for i, sym in enumerate(okx_only_sorted, 1):
            ox = fmt_turnover(okx[sym])
            print(f"  {i:>4}  {sym:<20} {ox:>12}")
    else:
        print("  (none)")

    print()


def run_downloader(exchange: str, symbols: list[str], start: str, end: str,
                   concurrency: int, types: str | None, market: str | None = None):
    """Run a download script as a subprocess."""
    if exchange == "bybit":
        script = SCRIPT_DIR / "download_bybit_data.py"
    elif exchange == "binance":
        script = SCRIPT_DIR / "download_binance_data.py"
    else:
        script = SCRIPT_DIR / "download_okx_data.py"

    cmd = [sys.executable, str(script), ",".join(symbols), start, end, "-c", str(concurrency)]
    if market:
        cmd.extend(["-m", market])
    if types:
        cmd.extend(["-t", types])

    print(f"\n{'='*80}")
    print(f"  DOWNLOADING: {exchange.upper()} — {len(symbols)} symbols")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    return subprocess.Popen(cmd, cwd=str(SCRIPT_DIR))


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="List and download USDT perpetual symbols on Bybit, Binance, and OKX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 list_symbols.py -n 100\n"
               "  python3 list_symbols.py --common-only -n 100\n"
               "  python3 list_symbols.py --download all -n 100 --start 2025-07-01\n"
               "  python3 list_symbols.py --download bybit -n 50 --start 2025-07-01 -t klines,fundingRate\n"
               "  python3 list_symbols.py --download okx -n 50 --start 2025-07-01\n",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=None,
        help="Only use top N common symbols (by Bybit 24h turnover)",
    )
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Only print common symbols comma-separated (for piping)",
    )
    parser.add_argument(
        "--download",
        choices=["bybit", "binance", "okx", "all"],
        default=None,
        help="Download data for common symbols using the specified exchange downloader(s)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date inclusive (YYYY-MM-DD). Required with --download.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=today,
        help=f"End date inclusive (YYYY-MM-DD). Default: today ({today})",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Max concurrent downloads passed to downloader scripts (default: 5)",
    )
    parser.add_argument(
        "--market", "-m",
        type=str,
        default=None,
        choices=["linear", "futures", "swap", "spot"],
        help="Market type passed to downloader scripts. "
             "Bybit: linear/spot, Binance: futures/spot, OKX: swap/spot. "
             "Use 'spot' for all. Omit for each script's default (futures/swap).",
    )
    parser.add_argument(
        "--types", "-t",
        type=str,
        default=None,
        help="Comma-separated data types passed to downloader scripts (default: each script's default)",
    )
    args = parser.parse_args()

    # Fetch data
    sys.stderr.write("Fetching Bybit tickers...\n")
    bybit = get_bybit_symbols()
    sys.stderr.write(f"  {len(bybit)} USDT linear perps\n")

    sys.stderr.write("Fetching Binance tickers...\n")
    binance = get_binance_symbols()
    sys.stderr.write(f"  {len(binance)} USDT-M perps\n")

    sys.stderr.write("Fetching OKX tickers...\n")
    okx = get_okx_symbols()
    sys.stderr.write(f"  {len(okx)} USDT SWAP perps\n\n")

    # Sets
    bybit_set = set(bybit.keys())
    binance_set = set(binance.keys())
    okx_set = set(okx.keys())
    common = bybit_set & binance_set & okx_set
    bybit_only = bybit_set - binance_set - okx_set
    binance_only = binance_set - bybit_set - okx_set
    okx_only = okx_set - bybit_set - binance_set

    # Common symbols sorted by Bybit turnover descending
    common_sorted = sorted(common, key=lambda s: -bybit[s])
    if args.top:
        common_sorted = common_sorted[: args.top]

    sys.stderr.write(f"Common: {len(common_sorted)}"
                     + (f" (top {args.top})" if args.top else "")
                     + f"  |  Bybit-only: {len(bybit_only)}"
                     + f"  |  Binance-only: {len(binance_only)}"
                     + f"  |  OKX-only: {len(okx_only)}\n")

    # --common-only: comma-separated for piping
    if args.common_only:
        print(",".join(common_sorted))
        return

    # --download: run downloader scripts
    if args.download:
        if not args.start:
            print("Error: --start is required with --download", file=sys.stderr)
            sys.exit(1)

        # Validate dates
        try:
            s = datetime.strptime(args.start, "%Y-%m-%d")
            e = datetime.strptime(args.end, "%Y-%m-%d")
            if s > e:
                print("Error: --start must be <= --end", file=sys.stderr)
                sys.exit(1)
        except ValueError as exc:
            print(f"Error: invalid date format: {exc}", file=sys.stderr)
            sys.exit(1)

        processes = []

        # Map generic market names to exchange-specific ones
        def exchange_market(exchange: str, market: str | None) -> str | None:
            if market is None:
                return None
            if market == "spot":
                return "spot"
            # Non-spot: use each exchange's default futures name
            mkt_map = {"bybit": "linear", "binance": "futures", "okx": "swap"}
            return mkt_map.get(exchange, market)

        if args.download in ("bybit", "all"):
            p = run_downloader("bybit", common_sorted, args.start, args.end,
                               args.concurrency, args.types,
                               exchange_market("bybit", args.market))
            processes.append(("bybit", p))

        if args.download in ("binance", "all"):
            p = run_downloader("binance", common_sorted, args.start, args.end,
                               args.concurrency, args.types,
                               exchange_market("binance", args.market))
            processes.append(("binance", p))

        if args.download in ("okx", "all"):
            p = run_downloader("okx", common_sorted, args.start, args.end,
                               args.concurrency, args.types,
                               exchange_market("okx", args.market))
            processes.append(("okx", p))

        # Wait for all downloaders
        failed = False
        for name, p in processes:
            rc = p.wait()
            if rc != 0:
                print(f"\n{name} downloader exited with code {rc}", file=sys.stderr)
                failed = True

        if failed:
            sys.exit(1)
        return

    # Default: full report
    print_report(common_sorted, bybit, binance, okx, bybit_only, binance_only, okx_only, args.top)


if __name__ == "__main__":
    main()
