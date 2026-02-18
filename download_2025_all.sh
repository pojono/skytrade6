#!/bin/bash
# Download 2025 liquidation + ticker data for all 5 symbols via tar
# Period: May 11 - Aug 10, 2025 (92 days)

REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "DOGEUSDT" "XRPUSDT")

echo "======================================================================"
echo "DOWNLOAD 2025 DATA — Liquidations + Ticker"
echo "======================================================================"
echo "Period: 2025-05-11 to 2025-08-10 (92 days)"
echo "Symbols: ${SYMBOLS[@]}"
echo "Streams: liquidation (ws), ticker (rest)"
echo "======================================================================"

download_stream() {
    local SYMBOL=$1
    local STREAM=$2
    local SOURCE=$3
    local YEAR=$4
    local PREFIX=$5

    echo ""
    echo "  [$SYMBOL] Downloading $STREAM ($SOURCE)..."

    # Check if we already have files
    EXISTING=$(ls "$SCRIPT_DIR/$LOCAL_BASE/$SYMBOL"/${PREFIX}_${YEAR}-*.jsonl.gz 2>/dev/null | wc -l)
    if [ "$EXISTING" -gt 100 ]; then
        echo "    Already have $EXISTING files, skipping"
        return 0
    fi

    REMOTE_TAR="/tmp/${PREFIX}_${SYMBOL}_${YEAR}.tar.gz"

    # Create tar on remote
    echo -n "    Creating archive on server..."
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "cd $REMOTE_BASE && \
        find dt=${YEAR}-*/hr=*/exchange=bybit/source=${SOURCE}/market=linear/stream=${STREAM}/symbol=${SYMBOL}/ \
        -name 'data.jsonl.gz' -print0 2>/dev/null | \
        tar -czf $REMOTE_TAR --null -T - 2>/dev/null && \
        ls -lh $REMOTE_TAR | awk '{print \$5}'"

    if [ $? -ne 0 ]; then
        echo " FAILED"
        return 1
    fi

    # Download
    echo -n "    Downloading..."
    scp -q -i "$SSH_KEY" "$REMOTE_HOST:$REMOTE_TAR" "/tmp/${PREFIX}_${SYMBOL}_${YEAR}.tar.gz"
    LOCAL_SIZE=$(ls -lh "/tmp/${PREFIX}_${SYMBOL}_${YEAR}.tar.gz" 2>/dev/null | awk '{print $5}')
    echo " $LOCAL_SIZE"

    # Extract
    echo -n "    Extracting..."
    WORK_DIR="/tmp/${PREFIX}_${SYMBOL}_${YEAR}_$$"
    mkdir -p "$WORK_DIR"
    tar -xzf "/tmp/${PREFIX}_${SYMBOL}_${YEAR}.tar.gz" -C "$WORK_DIR"

    DEST_DIR="$SCRIPT_DIR/$LOCAL_BASE/$SYMBOL"
    mkdir -p "$DEST_DIR"

    find "$WORK_DIR" -name 'data.jsonl.gz' | while read -r file; do
        DATE=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
        HOUR=$(echo "$file" | grep -oP 'hr=\K[0-9]+')
        if [ -n "$DATE" ] && [ -n "$HOUR" ]; then
            cp "$file" "$DEST_DIR/${PREFIX}_${DATE}_hr${HOUR}.jsonl.gz"
        fi
    done

    FILE_COUNT=$(ls "$DEST_DIR"/${PREFIX}_${YEAR}-*.jsonl.gz 2>/dev/null | wc -l)
    echo " $FILE_COUNT files"

    # Cleanup
    rm -rf "$WORK_DIR"
    rm -f "/tmp/${PREFIX}_${SYMBOL}_${YEAR}.tar.gz"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "rm -f $REMOTE_TAR" 2>/dev/null

    echo "    ✓ Done: $FILE_COUNT files for $SYMBOL $STREAM"
}

T_START=$(date +%s)

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Processing $SYMBOL"
    echo "======================================================================"

    # Download liquidation data (ws)
    download_stream "$SYMBOL" "liquidation" "ws" "2025" "liquidation"

    # Download ticker data (rest) — we already have 2025 ticker for BTC/ETH/SOL
    download_stream "$SYMBOL" "ticker" "rest" "2025" "ticker"
done

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE — ${ELAPSED}s elapsed"
echo "======================================================================"

for SYMBOL in "${SYMBOLS[@]}"; do
    LIQ_COUNT=$(ls "$SCRIPT_DIR/$LOCAL_BASE/$SYMBOL"/liquidation_2025-*.jsonl.gz 2>/dev/null | wc -l)
    TICK_COUNT=$(ls "$SCRIPT_DIR/$LOCAL_BASE/$SYMBOL"/ticker_2025-*.jsonl.gz 2>/dev/null | wc -l)
    echo "$SYMBOL: $LIQ_COUNT liquidation files, $TICK_COUNT ticker files"
done

echo "======================================================================"
