#!/bin/bash
# Simple reliable download using scp with explicit date/hour loops

REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"

SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT")

# 2025 dates from May 11 to Aug 10
DATES=(
    2025-05-11 2025-05-12 2025-05-13 2025-05-14 2025-05-15 2025-05-16 2025-05-17 2025-05-18 2025-05-19 2025-05-20
    2025-05-21 2025-05-22 2025-05-23 2025-05-24 2025-05-25 2025-05-26 2025-05-27 2025-05-28 2025-05-29 2025-05-30
    2025-05-31 2025-06-01 2025-06-02 2025-06-03 2025-06-04 2025-06-05 2025-06-06 2025-06-07 2025-06-08 2025-06-09
    2025-06-10 2025-06-11 2025-06-12 2025-06-13 2025-06-14 2025-06-15 2025-06-16 2025-06-17 2025-06-18 2025-06-19
    2025-06-20 2025-06-21 2025-06-22 2025-06-23 2025-06-24 2025-06-25 2025-06-26 2025-06-27 2025-06-28 2025-06-29
    2025-06-30 2025-07-01 2025-07-02 2025-07-03 2025-07-04 2025-07-05 2025-07-06 2025-07-07 2025-07-08 2025-07-09
    2025-07-10 2025-07-11 2025-07-12 2025-07-13 2025-07-14 2025-07-15 2025-07-16 2025-07-17 2025-07-18 2025-07-19
    2025-07-20 2025-07-21 2025-07-22 2025-07-23 2025-07-24 2025-07-25 2025-07-26 2025-07-27 2025-07-28 2025-07-29
    2025-07-30 2025-07-31 2025-08-01 2025-08-02 2025-08-03 2025-08-04 2025-08-05 2025-08-06 2025-08-07 2025-08-08
    2025-08-09 2025-08-10
)

echo "======================================================================"
echo "TICKER DATA DOWNLOAD - 2025 Data (May 11 - Aug 10)"
echo "======================================================================"
echo "Symbols: ${SYMBOLS[@]}"
echo "Total dates: ${#DATES[@]}"
echo "======================================================================"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Downloading $SYMBOL"
    echo "======================================================================"
    
    mkdir -p "$LOCAL_BASE/$SYMBOL"
    
    downloaded=0
    skipped=0
    failed=0
    
    for DATE in "${DATES[@]}"; do
        echo -n "[$DATE] "
        date_downloaded=0
        
        for HOUR in {00..23}; do
            LOCAL_FILE="$LOCAL_BASE/$SYMBOL/ticker_${DATE}_hr${HOUR}.jsonl.gz"
            
            # Skip if already exists
            if [ -f "$LOCAL_FILE" ]; then
                ((skipped++))
                continue
            fi
            
            REMOTE_PATH="$REMOTE_BASE/dt=$DATE/hr=$HOUR/exchange=bybit/source=rest/market=linear/stream=ticker/symbol=$SYMBOL/data.jsonl.gz"
            
            # Try to download
            scp -q -i "$SSH_KEY" "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_FILE" 2>/dev/null
            
            if [ $? -eq 0 ]; then
                ((downloaded++))
                ((date_downloaded++))
            else
                ((failed++))
            fi
        done
        
        echo "$date_downloaded files"
    done
    
    total_files=$(ls "$LOCAL_BASE/$SYMBOL"/ticker_*.jsonl.gz 2>/dev/null | wc -l)
    total_size=$(du -sh "$LOCAL_BASE/$SYMBOL" 2>/dev/null | cut -f1)
    
    echo ""
    echo "Summary for $SYMBOL:"
    echo "  Downloaded: $downloaded"
    echo "  Skipped (existing): $skipped"
    echo "  Failed/missing: $failed"
    echo "  Total ticker files: $total_files"
    echo "  Total size: $total_size"
    echo "======================================================================"
done

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================"
for SYMBOL in "${SYMBOLS[@]}"; do
    file_count=$(ls "$LOCAL_BASE/$SYMBOL"/ticker_*.jsonl.gz 2>/dev/null | wc -l)
    total_size=$(du -sh "$LOCAL_BASE/$SYMBOL" 2>/dev/null | cut -f1)
    echo "  $SYMBOL: $file_count ticker files, $total_size total"
done
echo "======================================================================"
