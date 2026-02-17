#!/bin/bash
# Download ticker data by creating tar archives on remote server first

REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"

SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "DOGEUSDT" "XRPUSDT")

echo "======================================================================"
echo "TICKER DATA DOWNLOAD - Efficient tar method"
echo "======================================================================"
echo "Period: 2026-02-09 to 2026-02-17"
echo "Symbols: ${SYMBOLS[@]}"
echo "======================================================================"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Downloading $SYMBOL"
    echo "======================================================================"
    
    mkdir -p "$LOCAL_BASE/$SYMBOL"
    
    # Create tar archive on remote server
    echo "Creating tar archive on remote server..."
    REMOTE_TAR="/tmp/ticker_${SYMBOL}_2025.tar.gz"
    
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "cd $REMOTE_BASE && \
        find dt=2026-*/hr=*/exchange=bybit/source=rest/market=linear/stream=ticker/symbol=$SYMBOL/ \
        -name 'data.jsonl.gz' -print0 | \
        tar -czf $REMOTE_TAR --null -T - 2>/dev/null && \
        echo 'Archive created' && \
        ls -lh $REMOTE_TAR"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to create archive on remote server"
        continue
    fi
    
    # Download the tar archive
    echo "Downloading archive..."
    LOCAL_TAR="$LOCAL_BASE/${SYMBOL}_temp.tar.gz"
    scp -i "$SSH_KEY" "$REMOTE_HOST:$REMOTE_TAR" "$LOCAL_TAR"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to download archive"
        continue
    fi
    
    # Extract and reorganize
    echo "Extracting and organizing files..."
    cd "$LOCAL_BASE/$SYMBOL"
    tar -xzf "../${SYMBOL}_temp.tar.gz"
    
    # Reorganize files with proper names
    find . -name "data.jsonl.gz" | while read file; do
        # Extract date and hour from path
        date=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
        hour=$(echo "$file" | grep -oP 'hr=\K[0-9]+')
        
        if [ -n "$date" ] && [ -n "$hour" ]; then
            newname="ticker_${date}_hr${hour}.jsonl.gz"
            mv "$file" "$newname" 2>/dev/null
        fi
    done
    
    # Clean up directory structure
    rm -rf dt=*
    rm -f "../${SYMBOL}_temp.tar.gz"
    cd - > /dev/null
    
    # Clean up remote tar
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "rm -f $REMOTE_TAR"
    
    # Summary
    file_count=$(ls "$LOCAL_BASE/$SYMBOL"/ticker_*.jsonl.gz 2>/dev/null | wc -l)
    total_size=$(du -sh "$LOCAL_BASE/$SYMBOL" 2>/dev/null | cut -f1)
    
    echo ""
    echo "Summary for $SYMBOL:"
    echo "  Ticker files: $file_count"
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
    echo "  $SYMBOL: $file_count ticker files, $total_size"
done
echo "======================================================================"
