#!/bin/bash
# Bulk download ticker data using rsync for efficiency

REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"

SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT")

echo "======================================================================"
echo "BULK TICKER DATA DOWNLOAD (rsync)"
echo "======================================================================"
echo "Remote: $REMOTE_HOST"
echo "Period: 2025-05-11 to 2025-08-10"
echo "Symbols: ${SYMBOLS[@]}"
echo "======================================================================"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Downloading $SYMBOL"
    echo "======================================================================"
    
    mkdir -p "$LOCAL_BASE/$SYMBOL"
    
    # Use rsync to download all ticker files for this symbol
    # This will be much faster than individual scp calls
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY" \
        --include='*/' \
        --include='data.jsonl.gz' \
        --exclude='*' \
        "$REMOTE_HOST:$REMOTE_BASE/dt=2025-*/hr=*/exchange=bybit/source=rest/market=linear/stream=ticker/symbol=$SYMBOL/" \
        "$LOCAL_BASE/$SYMBOL/temp_rsync/"
    
    # Reorganize files into flat structure with meaningful names
    echo "Reorganizing files..."
    find "$LOCAL_BASE/$SYMBOL/temp_rsync" -name "data.jsonl.gz" | while read file; do
        # Extract date and hour from path
        # Path format: data/SYMBOL/temp_rsync/dt=2025-05-11/hr=12/.../data.jsonl.gz
        date=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
        hour=$(echo "$file" | grep -oP 'hr=\K[0-9]+')
        
        if [ -n "$date" ] && [ -n "$hour" ]; then
            newname="ticker_${date}_hr${hour}.jsonl.gz"
            mv "$file" "$LOCAL_BASE/$SYMBOL/$newname"
            echo "  ✓ $newname"
        fi
    done
    
    # Clean up temp directory
    rm -rf "$LOCAL_BASE/$SYMBOL/temp_rsync"
    
    # Count files
    file_count=$(ls "$LOCAL_BASE/$SYMBOL" | wc -l)
    total_size=$(du -sh "$LOCAL_BASE/$SYMBOL" | cut -f1)
    echo ""
    echo "Summary for $SYMBOL:"
    echo "  Files downloaded: $file_count"
    echo "  Total size: $total_size"
    echo "======================================================================"
done

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================"
echo "Data location: $LOCAL_BASE/"
for SYMBOL in "${SYMBOLS[@]}"; do
    file_count=$(ls "$LOCAL_BASE/$SYMBOL" 2>/dev/null | wc -l)
    total_size=$(du -sh "$LOCAL_BASE/$SYMBOL" 2>/dev/null | cut -f1)
    echo "  $SYMBOL: $file_count files, $total_size"
done
echo "======================================================================"
