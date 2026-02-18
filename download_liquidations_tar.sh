#!/bin/bash
# Download liquidations data using efficient tar method

REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"

SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "DOGEUSDT" "XRPUSDT")

echo "======================================================================"
echo "LIQUIDATIONS DATA DOWNLOAD - Efficient tar method"
echo "======================================================================"
echo "Period: 2026-02-09 to 2026-02-17"
echo "Symbols: ${SYMBOLS[@]}"
echo "======================================================================"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Processing $SYMBOL"
    echo "======================================================================"
    
    # Create local directory
    mkdir -p "$LOCAL_BASE/$SYMBOL"
    
    # Create tar archive on remote server and download
    REMOTE_TAR="/tmp/liquidation_${SYMBOL}_2026.tar.gz"
    
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "cd $REMOTE_BASE && \
        find dt=2026-*/hr=*/exchange=bybit/source=ws/market=linear/stream=liquidation/symbol=$SYMBOL/ \
        -name 'data.jsonl.gz' -print0 | \
        tar -czf $REMOTE_TAR --null -T - 2>/dev/null && \
        echo 'Archive created' && \
        ls -lh $REMOTE_TAR"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to create archive for $SYMBOL"
        continue
    fi
    
    # Download the tar file
    echo "  Downloading archive..."
    scp -i "$SSH_KEY" "$REMOTE_HOST:$REMOTE_TAR" "/tmp/liquidation_${SYMBOL}_2026.tar.gz"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to download archive for $SYMBOL"
        ssh -i "$SSH_KEY" "$REMOTE_HOST" "rm -f $REMOTE_TAR"
        continue
    fi
    
    # Extract and reorganize locally
    echo "  Extracting and organizing files..."
    
    # Extract to temp directory
    WORK_DIR="/tmp/liquidation_${SYMBOL}_$$"
    mkdir -p "$WORK_DIR"
    tar -xzf "/tmp/liquidation_${SYMBOL}_2026.tar.gz" -C "$WORK_DIR"
    
    # Move files to flat structure with meaningful names
    FILE_COUNT=0
    find "$WORK_DIR" -name 'data.jsonl.gz' | while read -r file; do
        # Extract date and hour from path
        DATE=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
        HOUR=$(echo "$file" | grep -oP 'hr=\K[0-9]+')
        
        if [ -n "$DATE" ] && [ -n "$HOUR" ]; then
            # Create new filename
            NEW_NAME="liquidation_${DATE}_${HOUR}.jsonl.gz"
            
            # Copy to symbol directory
            cp "$file" "$LOCAL_BASE/$SYMBOL/$NEW_NAME"
            FILE_COUNT=$((FILE_COUNT + 1))
        fi
    done
    
    # Count files
    FILE_COUNT=$(ls "$LOCAL_BASE/$SYMBOL"/liquidation_*.jsonl.gz 2>/dev/null | wc -l)
    
    # Cleanup
    rm -rf "$WORK_DIR"
    rm -f "/tmp/liquidation_${SYMBOL}_2026.tar.gz"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "rm -f $REMOTE_TAR"
    
    echo "  ✓ Downloaded and organized $FILE_COUNT files for $SYMBOL"
    
    # Show sample
    SAMPLE_FILE=$(ls "$LOCAL_BASE/$SYMBOL"/liquidation_*.jsonl.gz 2>/dev/null | head -1)
    if [ -n "$SAMPLE_FILE" ]; then
        echo "  Sample file: $(basename $SAMPLE_FILE)"
        echo "  Sample data:"
        zcat "$SAMPLE_FILE" | head -1 | python3 -m json.tool 2>/dev/null | head -20
    fi
done

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================"

# Summary
for SYMBOL in "${SYMBOLS[@]}"; do
    COUNT=$(ls "$LOCAL_BASE/$SYMBOL"/liquidation_*.jsonl.gz 2>/dev/null | wc -l)
    SIZE=$(du -sh "$LOCAL_BASE/$SYMBOL" 2>/dev/null | cut -f1)
    echo "$SYMBOL: $COUNT files, $SIZE total"
done

echo "======================================================================"
