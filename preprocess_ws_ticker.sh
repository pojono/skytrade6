#!/bin/bash
# Preprocess WS ticker files into fast CSV format for new symbols
# Extracts timestamp + lastPrice from snapshot/delta messages
# Output: data/{SYMBOL}/bybit/ticker_prices.csv.gz

set -euo pipefail

SYMBOLS=("ADAUSDT" "BCHUSDT" "LTCUSDT" "NEARUSDT" "POLUSDT" "TONUSDT" "XLMUSDT")

for SYMBOL in "${SYMBOLS[@]}"; do
    DIR="data/$SYMBOL/bybit/ticker"
    OUT="data/$SYMBOL/ticker_prices.csv.gz"
    
    if [[ -f "$OUT" ]]; then
        echo "  $SYMBOL: already exists, skipping"
        continue
    fi
    
    if [[ ! -d "$DIR" ]]; then
        echo "  $SYMBOL: no ticker dir, skipping"
        continue
    fi
    
    N_FILES=$(find "$DIR" -name "ticker_*.jsonl.gz" | wc -l)
    echo -n "  $SYMBOL: processing $N_FILES files..."
    T_START=$SECONDS
    
    # Extract ts and lastPrice from all files, pipe to gzipped CSV
    # Use grep to filter only lines with lastPrice (skip delta messages without it)
    # Use python for fast JSON extraction
    (echo "ts,price"
     find "$DIR" -name "ticker_*.jsonl.gz" -type f | sort | \
     xargs -I{} zcat {} | \
     grep '"lastPrice"' | \
     python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        r = d['result']
        if 'data' in r and 'lastPrice' in r['data']:
            print(f\"{d['ts']},{r['data']['lastPrice']}\")
        elif 'list' in r:
            print(f\"{d['ts']},{r['list'][0]['lastPrice']}\")
    except:
        pass
") | gzip > "$OUT"
    
    ELAPSED=$((SECONDS - T_START))
    SIZE=$(du -sh "$OUT" | cut -f1)
    LINES=$(zcat "$OUT" | wc -l)
    echo " done (${LINES} records, ${SIZE}, ${ELAPSED}s)"
done

echo "All done!"
