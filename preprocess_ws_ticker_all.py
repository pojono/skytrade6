#!/usr/bin/env python3
"""Preprocess WS ticker files into fast CSV for all symbols that have bybit/ticker dir."""
import sys, json, gzip, time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT',
           'ADAUSDT', 'BCHUSDT', 'LTCUSDT', 'NEARUSDT', 'POLUSDT', 'TONUSDT', 'XLMUSDT']

for symbol in SYMBOLS:
    out_path = Path(f"data/{symbol}/ticker_prices.csv.gz")
    ws_dir = Path(f"data/{symbol}/bybit/ticker")
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"{symbol}: already exists ({out_path.stat().st_size:,} bytes), skipping")
        continue
    if not ws_dir.exists():
        print(f"{symbol}: no WS ticker dir, skipping")
        continue
    files = sorted(ws_dir.glob("ticker_*.jsonl.gz"))
    print(f"{symbol}: processing {len(files)} files...", end='', flush=True)
    t0 = time.time()
    n = 0
    with gzip.open(out_path, 'wt') as out:
        out.write("ts,price\n")
        for i, f in enumerate(files, 1):
            if i % 500 == 0:
                print(f" {i}", end='', flush=True)
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        r = d['result'].get('data', {})
                        if 'lastPrice' in r:
                            out.write(f"{d['ts']},{r['lastPrice']}\n")
                            n += 1
                    except Exception:
                        continue
    elapsed = time.time() - t0
    size = out_path.stat().st_size
    print(f" done ({n:,} records, {size/1024/1024:.1f}MB, {elapsed:.0f}s)")

print("All done!")
