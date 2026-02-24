#!/bin/bash
# Download symbol=ALL ticker data from Binance and Bybit via dataminer server
#
# Data streams downloaded:
#   1. Binance  rest/linear/ticker/symbol=ALL     (~38MB/hr compressed)
#   2. Binance  rest/linear/fundingRate/symbol=ALL (~13MB/hr compressed)
#   3. Bybit    rest/linear/ticker/symbol=ALL      (~44MB/hr compressed)
#
# Approach: create one tar per exchange on remote, download, extract locally.
#
# Local output structure:
#   data_all/binance/ticker/ticker_{date}_hr{HH}.jsonl.gz
#   data_all/binance/fundingRate/fundingRate_{date}_hr{HH}.jsonl.gz
#   data_all/bybit/ticker/ticker_{date}_hr{HH}.jsonl.gz
#
# Usage:
#   ./download_ticker_all.sh                          # all available dates
#   ./download_ticker_all.sh --start 2026-02-22 --end 2026-02-23
#   ./download_ticker_all.sh --dry-run
#   ./download_ticker_all.sh --list-dates

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REMOTE_HOST="ubuntu@13.251.79.76"
SSH_KEY="$HOME/.ssh/id_ed25519_remote"
REMOTE_BASE="dataminer/data/archive/raw"
LOCAL_BASE="data_all"

# ── Parse arguments ───────────────────────────────────────────────────────────
START_DATE=""
END_DATE=""
DRY_RUN=false
LIST_DATES=false
SKIP_EXISTING=true
INCLUDE_PARTIAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)    START_DATE="$2"; shift 2 ;;
        --end)      END_DATE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --force)    SKIP_EXISTING=false; shift ;;
        --include-partial) INCLUDE_PARTIAL=true; shift ;;
        --list-dates) LIST_DATES=true; shift ;;
        --help|-h)
            head -22 "$0" | tail -21
            echo ""
            echo "Options:"
            echo "  --start YYYY-MM-DD   Start date (default: earliest available)"
            echo "  --end YYYY-MM-DD     End date (default: latest available)"
            echo "  --dry-run            Show what would be downloaded"
            echo "  --force              Re-download even if local files exist"
            echo "  --include-partial    Include the current (incomplete) hour"
            echo "  --list-dates         List available dates on server"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── SSH helper ────────────────────────────────────────────────────────────────
remote() {
    ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "$@"
}

# ── List dates mode ───────────────────────────────────────────────────────────
if $LIST_DATES; then
    echo "Available dates with symbol=ALL data:"
    remote "for d in \$(ls -1 $REMOTE_BASE/ | sed 's/dt=//'); do
        if [ -f $REMOTE_BASE/dt=\$d/hr=00/exchange=binance/source=rest/market=linear/stream=ticker/symbol=ALL/data.jsonl.gz ]; then
            echo \"  \$d\"
        fi
    done"
    exit 0
fi

# ── Discover available dates ──────────────────────────────────────────────────
echo "======================================================================"
echo "DOWNLOAD symbol=ALL TICKER DATA (Binance + Bybit)"
echo "======================================================================"
echo "Discovering dates with symbol=ALL data..."

ALL_DATES=$(remote "for d in \$(ls -1 $REMOTE_BASE/ | sed 's/dt=//'); do
    if [ -f $REMOTE_BASE/dt=\$d/hr=00/exchange=binance/source=rest/market=linear/stream=ticker/symbol=ALL/data.jsonl.gz ]; then
        echo \$d
    fi
done")

DATES=()
while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    if [[ -n "$START_DATE" && "$d" < "$START_DATE" ]]; then continue; fi
    if [[ -n "$END_DATE" && "$d" > "$END_DATE" ]]; then continue; fi
    DATES+=("$d")
done <<< "$ALL_DATES"

if [[ ${#DATES[@]} -eq 0 ]]; then
    echo "ERROR: No dates found matching filters"
    echo "Available dates: $ALL_DATES"
    exit 1
fi

echo ""
echo "Date range:  ${DATES[0]} -> ${DATES[-1]} (${#DATES[@]} days)"
echo "Streams:"
echo "  Binance: ticker (rest/linear), fundingRate (rest/linear)"
echo "  Bybit:   ticker (rest/linear)"
echo "Output:    ./$LOCAL_BASE/{exchange}/{stream}/"
echo "Skip existing: $SKIP_EXISTING"
echo "Include partial hours: $INCLUDE_PARTIAL"

# ── Partial-hour protection ──────────────────────────────────────────────────
# Current UTC date/hour — the current hour is always partial (still being written)
NOW_UTC_DATE=$(date -u +%Y-%m-%d)
NOW_UTC_HOUR=$(date -u +%H)
echo "Current UTC: ${NOW_UTC_DATE} ${NOW_UTC_HOUR}:xx"
if ! $INCLUDE_PARTIAL; then
    echo "  → Will skip hr=${NOW_UTC_HOUR} on ${NOW_UTC_DATE} (partial)"
fi
echo "======================================================================"
echo ""

# ── Define download jobs ─────────────────────────────────────────────────────
# Each job: EXCHANGE SOURCE MARKET STREAM LOCAL_SUBDIR
JOBS=(
    "binance rest linear ticker      binance/ticker"
    "binance rest linear fundingRate binance/fundingRate"
    "bybit   rest linear ticker      bybit/ticker"
)

# ── Download function ─────────────────────────────────────────────────────────
download_job() {
    local EXCHANGE="$1"
    local SOURCE="$2"
    local MARKET="$3"
    local STREAM="$4"
    local LOCAL_DIR="$LOCAL_BASE/$5"

    echo "══════════════════════════════════════════════════════════════════════"
    echo "  $EXCHANGE / $STREAM  →  $LOCAL_DIR"
    echo "══════════════════════════════════════════════════════════════════════"

    mkdir -p "$LOCAL_DIR"

    # Build list of remote paths we need
    FILE_LIST=""
    NEEDED=0
    SKIPPED=0

    for DATE in "${DATES[@]}"; do
        for HR in $(printf '%02d ' {0..23}); do
            # Skip the current (partial) hour — it's still being written
            if ! $INCLUDE_PARTIAL && [[ "$DATE" == "$NOW_UTC_DATE" && "$HR" == "$NOW_UTC_HOUR" ]]; then
                continue
            fi

            LOCAL_FILE="$LOCAL_DIR/${STREAM}_${DATE}_hr${HR}.jsonl.gz"
            PARTIAL_MARKER="${LOCAL_FILE}.partial"

            # If file was previously downloaded as partial, re-download it
            if [[ -f "$PARTIAL_MARKER" ]]; then
                rm -f "$LOCAL_FILE" "$PARTIAL_MARKER"
            fi

            if $SKIP_EXISTING && [[ -f "$LOCAL_FILE" ]] && [[ -s "$LOCAL_FILE" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            REMOTE_PATH="dt=${DATE}/hr=${HR}/exchange=${EXCHANGE}/source=${SOURCE}/market=${MARKET}/stream=${STREAM}/symbol=ALL/data.jsonl.gz"
            FILE_LIST+="${REMOTE_PATH}\n"
            NEEDED=$((NEEDED + 1))
        done
    done

    if [[ $NEEDED -eq 0 ]]; then
        echo "  All files already exist ($SKIPPED skipped)"
        echo ""
        return 0
    fi

    echo "  Need: $NEEDED files  |  Skipped: $SKIPPED existing"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would download $NEEDED files"
        echo ""
        return 0
    fi

    # Send file list to remote, filter existing, create tar
    REMOTE_TAR="/tmp/all_${EXCHANGE}_${STREAM}_$$.tar"
    REMOTE_FILELIST="/tmp/all_${EXCHANGE}_${STREAM}_$$.txt"

    echo "  Creating archive on remote server..."
    T_ARCHIVE=$SECONDS

    FOUND_COUNT=$(echo -e "$FILE_LIST" | \
        remote "cat > $REMOTE_FILELIST && cd $REMOTE_BASE && \
            FOUND=0; TOTAL=\$(grep -c . $REMOTE_FILELIST || echo 0); \
            > ${REMOTE_FILELIST}.found; \
            while IFS= read -r f; do \
                [[ -z \"\$f\" ]] && continue; \
                if [[ -f \"\$f\" ]]; then \
                    echo \"\$f\" >> ${REMOTE_FILELIST}.found; \
                    FOUND=\$((FOUND + 1)); \
                fi; \
            done < $REMOTE_FILELIST; \
            echo \$FOUND; \
            if [[ \$FOUND -gt 0 ]]; then \
                tar -cf $REMOTE_TAR -T ${REMOTE_FILELIST}.found 2>/dev/null; \
            fi; \
            rm -f $REMOTE_FILELIST ${REMOTE_FILELIST}.found" | tail -1)

    ARCHIVE_SECS=$((SECONDS - T_ARCHIVE))

    if [[ -z "$FOUND_COUNT" || "$FOUND_COUNT" -eq 0 ]]; then
        echo "  No files found on remote for $EXCHANGE/$STREAM"
        remote "rm -f $REMOTE_TAR" 2>/dev/null || true
        echo ""
        return 1
    fi

    # Get tar size
    TAR_SIZE=$(remote "ls -lh $REMOTE_TAR 2>/dev/null | awk '{print \$5}'" || echo "?")
    echo "  Archive: $FOUND_COUNT files, ${TAR_SIZE} [${ARCHIVE_SECS}s]"

    # Download tar (no compression — files inside are already gzipped)
    echo "  Downloading..."
    T_DL=$SECONDS
    LOCAL_TAR="/tmp/all_${EXCHANGE}_${STREAM}_$$.tar"

    if ! scp -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_HOST:$REMOTE_TAR" "$LOCAL_TAR"; then
        echo "  ✗ Download failed"
        remote "rm -f $REMOTE_TAR" 2>/dev/null || true
        echo ""
        return 1
    fi

    DL_SIZE=$(du -sh "$LOCAL_TAR" | cut -f1)
    DL_SECS=$((SECONDS - T_DL))
    echo "  Downloaded ${DL_SIZE} [${DL_SECS}s]"

    # Clean up remote tar immediately
    remote "rm -f $REMOTE_TAR" 2>/dev/null || true

    # Extract and reorganize
    echo "  Extracting and organizing..."
    WORK_DIR="/tmp/all_${EXCHANGE}_${STREAM}_work_$$"
    mkdir -p "$WORK_DIR"
    tar -xf "$LOCAL_TAR" -C "$WORK_DIR"

    FILE_COUNT=0
    while IFS= read -r file; do
        DATE=$(echo "$file" | grep -oP 'dt=\K[0-9-]+')
        HOUR=$(echo "$file" | grep -oP 'hr=\K[0-9]+')

        if [[ -n "$DATE" && -n "$HOUR" ]]; then
            cp "$file" "$LOCAL_DIR/${STREAM}_${DATE}_hr${HOUR}.jsonl.gz"
            FILE_COUNT=$((FILE_COUNT + 1))
            if (( FILE_COUNT % 10 == 0 )); then
                printf "\r  Organized: %d/%d files" "$FILE_COUNT" "$FOUND_COUNT"
            fi
        fi
    done < <(find "$WORK_DIR" -name 'data.jsonl.gz' -type f | sort)

    printf "\r  Organized: %d/%d files\n" "$FILE_COUNT" "$FOUND_COUNT"

    # Cleanup
    rm -rf "$WORK_DIR" "$LOCAL_TAR"

    # Local summary
    TOTAL_LOCAL=$(find "$LOCAL_DIR" -name "${STREAM}_*.jsonl.gz" -type f | wc -l)
    TOTAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
    echo "  ✓ Done: $FILE_COUNT new files  |  Total: $TOTAL_LOCAL files, $TOTAL_SIZE"
    echo ""
}

# ── Main loop ─────────────────────────────────────────────────────────────────
T_START=$SECONDS
TOTAL_NEW=0

for JOB in "${JOBS[@]}"; do
    # shellcheck disable=SC2086
    read -r EX SRC MKT STREAM LDIR <<< $JOB
    download_job "$EX" "$SRC" "$MKT" "$STREAM" "$LDIR"
done

ELAPSED=$((SECONDS - T_START))

# ── Final summary ─────────────────────────────────────────────────────────────
echo "======================================================================"
echo "DOWNLOAD COMPLETE  [${ELAPSED}s elapsed]"
echo "======================================================================"

for JOB in "${JOBS[@]}"; do
    read -r EX SRC MKT STREAM LDIR <<< $JOB
    DIR="$LOCAL_BASE/$LDIR"
    if [[ -d "$DIR" ]]; then
        COUNT=$(find "$DIR" -name "${STREAM}_*.jsonl.gz" -type f | wc -l)
        SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
        echo "  $EX/$STREAM: $COUNT files, $SIZE"
    fi
done

echo "======================================================================"
