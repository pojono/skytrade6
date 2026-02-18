#!/bin/bash
# Download liquidations and ticker data from dataminer server
#
# Remote server structure (Hive-style partitioning):
#   dataminer/data/archive/raw/
#     dt={YYYY-MM-DD}/
#       hr={00..23}/
#         exchange=bybit/
#           source=ws/
#             market=linear/
#               stream={liquidation,ticker,...}/
#                 symbol={SYMBOL}/
#                   data.jsonl.gz
#
# Local output structure:
#   ./data/{SYMBOL}/bybit/{liquidations,ticker}/{stream}_{date}_hr{HH}.jsonl.gz
#
# Usage:
#   ./download_dataminer.sh                          # all symbols, all available dates
#   ./download_dataminer.sh --symbols BTCUSDT ETHUSDT
#   ./download_dataminer.sh --start 2026-02-09 --end 2026-02-18
#   ./download_dataminer.sh --streams liquidation ticker
#   ./download_dataminer.sh --dry-run                # show what would be downloaded
#   ./download_dataminer.sh --list-dates             # list available dates on server
#   ./download_dataminer.sh --list-streams           # list available ws streams on server

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data"

DEFAULT_SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "DOGEUSDT" "XRPUSDT")
DEFAULT_STREAMS=("liquidation" "ticker")
MARKET="linear"
SOURCE="ws"
EXCHANGE="bybit"

# ── Parse arguments ───────────────────────────────────────────────────────────
SYMBOLS=()
STREAMS=()
START_DATE=""
END_DATE=""
DRY_RUN=false
LIST_DATES=false
LIST_STREAMS=false
SKIP_EXISTING=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --symbols)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SYMBOLS+=("$1")
                shift
            done
            ;;
        --streams)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STREAMS+=("$1")
                shift
            done
            ;;
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            SKIP_EXISTING=false
            shift
            ;;
        --list-dates)
            LIST_DATES=true
            shift
            ;;
        --list-streams)
            LIST_STREAMS=true
            shift
            ;;
        --help|-h)
            head -28 "$0" | tail -27
            echo ""
            echo "Options:"
            echo "  --symbols SYM1 SYM2 ...   Symbols to download (default: ${DEFAULT_SYMBOLS[*]})"
            echo "  --streams STR1 STR2 ...   Streams to download (default: ${DEFAULT_STREAMS[*]})"
            echo "  --start YYYY-MM-DD        Start date (default: earliest available)"
            echo "  --end YYYY-MM-DD          End date (default: latest available)"
            echo "  --dry-run                 Show what would be downloaded without downloading"
            echo "  --force                   Re-download even if local file exists"
            echo "  --list-dates              List available dates on remote server"
            echo "  --list-streams            List available ws streams on remote server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Apply defaults
[[ ${#SYMBOLS[@]} -eq 0 ]] && SYMBOLS=("${DEFAULT_SYMBOLS[@]}")
[[ ${#STREAMS[@]} -eq 0 ]] && STREAMS=("${DEFAULT_STREAMS[@]}")

# ── SSH helper ────────────────────────────────────────────────────────────────
remote() {
    ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "$@"
}

# ── List modes ────────────────────────────────────────────────────────────────
if $LIST_DATES; then
    echo "Available dates on remote server:"
    remote "ls -1 $REMOTE_BASE/ | sed 's/dt=//' | sort"
    exit 0
fi

if $LIST_STREAMS; then
    echo "Available ws streams (market=$MARKET) on remote server:"
    SAMPLE_DATE=$(remote "ls -1 $REMOTE_BASE/ | tail -1")
    SAMPLE_HR=$(remote "ls -1 $REMOTE_BASE/$SAMPLE_DATE/ | head -1")
    remote "ls -1 $REMOTE_BASE/$SAMPLE_DATE/$SAMPLE_HR/exchange=$EXCHANGE/source=$SOURCE/market=$MARKET/ | sed 's/stream=//'" 2>/dev/null || echo "(none found)"
    exit 0
fi

# ── Discover available dates ──────────────────────────────────────────────────
echo "======================================================================"
echo "DATAMINER DATA DOWNLOAD"
echo "======================================================================"
echo "Discovering available dates on remote server..."

ALL_DATES=$(remote "ls -1 $REMOTE_BASE/ | sed 's/dt=//' | sort")
AVAILABLE_DATES=()

while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    # Apply date filters
    if [[ -n "$START_DATE" && "$d" < "$START_DATE" ]]; then continue; fi
    if [[ -n "$END_DATE" && "$d" > "$END_DATE" ]]; then continue; fi
    AVAILABLE_DATES+=("$d")
done <<< "$ALL_DATES"

if [[ ${#AVAILABLE_DATES[@]} -eq 0 ]]; then
    echo "ERROR: No dates found matching filters (start=$START_DATE, end=$END_DATE)"
    echo "Available dates on server:"
    echo "$ALL_DATES"
    exit 1
fi

echo ""
echo "Date range:  ${AVAILABLE_DATES[0]} -> ${AVAILABLE_DATES[-1]} (${#AVAILABLE_DATES[@]} days)"
echo "Symbols:     ${SYMBOLS[*]}"
echo "Streams:     ${STREAMS[*]}"
echo "Market:      $MARKET"
echo "Source:      $SOURCE"
echo "Skip exist:  $SKIP_EXISTING"
echo "Output:      ./$LOCAL_BASE/{SYMBOL}/bybit/{stream}/"
echo "======================================================================"
echo ""

# ── Map stream name to local directory name ───────────────────────────────────
stream_to_dir() {
    case "$1" in
        liquidation) echo "liquidations" ;;
        *)           echo "$1" ;;
    esac
}

# ── Download ──────────────────────────────────────────────────────────────────
TOTAL_DOWNLOADED=0
TOTAL_SKIPPED=0
TOTAL_MISSING=0
TOTAL_FAILED=0
T_START=$SECONDS

for SYMBOL in "${SYMBOLS[@]}"; do
    for STREAM in "${STREAMS[@]}"; do
        LOCAL_DIR="$LOCAL_BASE/$SYMBOL/bybit/$(stream_to_dir "$STREAM")"
        mkdir -p "$LOCAL_DIR"

        echo "──────────────────────────────────────────────────────────────"
        echo "  $SYMBOL / $STREAM -> $LOCAL_DIR"
        echo "──────────────────────────────────────────────────────────────"

        # Build remote find pattern for all dates+hours at once, then tar+download
        # This is much faster than individual scp calls
        FIND_PATTERNS=""
        EXPECTED_FILES=()
        SKIP_COUNT=0

        for DATE in "${AVAILABLE_DATES[@]}"; do
            for HR in $(printf '%02d ' {0..23}); do
                LOCAL_FILE="$LOCAL_DIR/${STREAM}_${DATE}_hr${HR}.jsonl.gz"
                
                if $SKIP_EXISTING && [[ -f "$LOCAL_FILE" ]] && [[ -s "$LOCAL_FILE" ]]; then
                    SKIP_COUNT=$((SKIP_COUNT + 1))
                    continue
                fi
                
                REMOTE_PATH="dt=${DATE}/hr=${HR}/exchange=${EXCHANGE}/source=${SOURCE}/market=${MARKET}/stream=${STREAM}/symbol=${SYMBOL}/data.jsonl.gz"
                FIND_PATTERNS+="${REMOTE_PATH}\n"
                EXPECTED_FILES+=("${DATE}_hr${HR}")
            done
        done

        TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIP_COUNT))

        if [[ ${#EXPECTED_FILES[@]} -eq 0 ]]; then
            echo "  All ${SKIP_COUNT} files already exist, skipping"
            continue
        fi

        echo "  Files to download: ${#EXPECTED_FILES[@]} (skipped existing: $SKIP_COUNT)"

        if $DRY_RUN; then
            echo "  [DRY RUN] Would download ${#EXPECTED_FILES[@]} files"
            continue
        fi

        # Create file list on remote, find which actually exist, tar them
        REMOTE_TAR="/tmp/dataminer_${SYMBOL}_${STREAM}_$$.tar.gz"
        FILE_LIST="/tmp/dataminer_filelist_${SYMBOL}_${STREAM}_$$.txt"

        echo "  Creating archive on remote server..."
        T_ARCHIVE=$SECONDS

        # Send file list to remote, filter to existing files, create tar
        # Remote script prints progress lines like "SCAN 100/2208" and "TAR_START" / "TAR_DONE"
        TOTAL_EXPECTED=${#EXPECTED_FILES[@]}
        EXISTING_COUNT=0

        echo -e "$FIND_PATTERNS" | \
            remote "cat > $FILE_LIST && cd $REMOTE_BASE && \
                    FOUND=0; SCANNED=0; TOTAL=\$(wc -l < $FILE_LIST); \
                    > ${FILE_LIST}.found; \
                    while IFS= read -r f; do \
                        [[ -z \"\$f\" ]] && continue; \
                        SCANNED=\$((SCANNED + 1)); \
                        if [[ -f \"\$f\" ]]; then \
                            echo \"\$f\" >> ${FILE_LIST}.found; \
                            FOUND=\$((FOUND + 1)); \
                        fi; \
                        if (( SCANNED % 200 == 0 )); then \
                            echo \"SCAN \$SCANNED/\$TOTAL found=\$FOUND\"; \
                        fi; \
                    done < $FILE_LIST; \
                    echo \"SCAN_DONE \$FOUND/\$TOTAL\"; \
                    if [[ \$FOUND -gt 0 ]]; then \
                        echo \"TAR_START \$FOUND files\"; \
                        tar -czf $REMOTE_TAR -T ${FILE_LIST}.found 2>/dev/null; \
                        TAR_SIZE=\$(du -sh $REMOTE_TAR | cut -f1); \
                        echo \"TAR_DONE \$TAR_SIZE\"; \
                    fi; \
                    rm -f $FILE_LIST ${FILE_LIST}.found" | \
        while IFS= read -r line; do
            case "$line" in
                SCAN\ *)
                    printf "\r  Scanning remote: %s" "$line" ;;
                SCAN_DONE\ *)
                    EXISTING_COUNT=$(echo "$line" | grep -oP '^SCAN_DONE \K[0-9]+')
                    printf "\r  Scanning remote: %s\n" "$line" ;;
                TAR_START\ *)
                    echo "  Archiving: $line" ;;
                TAR_DONE\ *)
                    echo "  Archive ready: $line" ;;
            esac
        done

        ARCHIVE_SECS=$((SECONDS - T_ARCHIVE))

        # Re-check: get the actual count from the remote tar (the pipe subshell loses vars)
        EXISTING_COUNT=$(remote "cd $REMOTE_BASE && \
            if [[ -f $REMOTE_TAR ]]; then tar -tzf $REMOTE_TAR | wc -l; else echo 0; fi" | tr -d '[:space:]')

        if [[ "$EXISTING_COUNT" -eq 0 || -z "$EXISTING_COUNT" ]]; then
            echo "  No data found on remote for $SYMBOL/$STREAM"
            TOTAL_MISSING=$((TOTAL_MISSING + ${#EXPECTED_FILES[@]}))
            remote "rm -f $REMOTE_TAR" 2>/dev/null || true
            continue
        fi

        MISSING_ON_REMOTE=$(( ${#EXPECTED_FILES[@]} - EXISTING_COUNT ))
        TOTAL_MISSING=$((TOTAL_MISSING + MISSING_ON_REMOTE))
        echo "  Found $EXISTING_COUNT files on remote ($MISSING_ON_REMOTE missing) [${ARCHIVE_SECS}s]"

        # Download tar
        echo "  Downloading archive..."
        LOCAL_TAR="/tmp/dataminer_${SYMBOL}_${STREAM}_$$.tar.gz"
        T_DL=$SECONDS

        if ! scp -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_HOST:$REMOTE_TAR" "$LOCAL_TAR"; then
            echo "  ✗ Failed to download archive"
            remote "rm -f $REMOTE_TAR" 2>/dev/null || true
            TOTAL_FAILED=$((TOTAL_FAILED + EXISTING_COUNT))
            continue
        fi

        DL_SIZE=$(du -sh "$LOCAL_TAR" | cut -f1)
        DL_SECS=$((SECONDS - T_DL))
        echo "  Downloaded $DL_SIZE [${DL_SECS}s]"

        # Extract to temp dir and reorganize
        echo "  Extracting and organizing..."
        WORK_DIR="/tmp/dataminer_work_${SYMBOL}_${STREAM}_$$"
        mkdir -p "$WORK_DIR"
        tar -xzf "$LOCAL_TAR" -C "$WORK_DIR"

        FILE_COUNT=0
        TOTAL_FILES_IN_TAR=$(find "$WORK_DIR" -name 'data.jsonl.gz' -type f | wc -l)
        while IFS= read -r file; do
            DATE=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
            HOUR=$(echo "$file" | grep -oP 'hr=\K[0-9]+')

            if [[ -n "$DATE" && -n "$HOUR" ]]; then
                NEW_NAME="${STREAM}_${DATE}_hr${HOUR}.jsonl.gz"
                cp "$file" "$LOCAL_DIR/$NEW_NAME"
                FILE_COUNT=$((FILE_COUNT + 1))
                if (( FILE_COUNT % 100 == 0 )) || (( FILE_COUNT == TOTAL_FILES_IN_TAR )); then
                    printf "\r  Organizing: %d/%d files" "$FILE_COUNT" "$TOTAL_FILES_IN_TAR"
                fi
            fi
        done < <(find "$WORK_DIR" -name 'data.jsonl.gz' -type f | sort)
        echo ""

        TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + FILE_COUNT))

        # Cleanup temp files
        rm -rf "$WORK_DIR" "$LOCAL_TAR"
        remote "rm -f $REMOTE_TAR" 2>/dev/null || true

        echo "  ✓ Organized $FILE_COUNT files into $LOCAL_DIR"
    done
done

ELAPSED=$((SECONDS - T_START))

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE  [${ELAPSED}s elapsed]"
echo "======================================================================"
echo "  Downloaded: $TOTAL_DOWNLOADED"
echo "  Skipped:    $TOTAL_SKIPPED (already exist)"
echo "  Missing:    $TOTAL_MISSING (not on remote)"
echo "  Failed:     $TOTAL_FAILED"
echo "======================================================================"
echo ""

# Summary per symbol/stream
echo "Local data summary:"
for SYMBOL in "${SYMBOLS[@]}"; do
    for STREAM in "${STREAMS[@]}"; do
        DIR="$LOCAL_BASE/$SYMBOL/bybit/$(stream_to_dir "$STREAM")"
        if [[ -d "$DIR" ]]; then
            COUNT=$(find "$DIR" -name "${STREAM}_*.jsonl.gz" -type f | wc -l)
            SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
            echo "  $SYMBOL/bybit/$(stream_to_dir "$STREAM"): $COUNT files, $SIZE"
        fi
    done
done
echo "======================================================================"
