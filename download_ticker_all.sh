#!/bin/bash
# Download symbol=ALL ticker data from Binance, Bybit, and OKX via dataminer server
#
# Data streams downloaded:
#   1. Binance  rest/linear/ticker/symbol=ALL       (~38MB/hr compressed)
#   2. Binance  rest/linear/fundingRate/symbol=ALL   (~13MB/hr compressed)
#   3. Binance  rest/spot/ticker/symbol=ALL          (~12MB/hr compressed)
#   4. Bybit    rest/linear/ticker/symbol=ALL        (~44MB/hr compressed)
#   5. Bybit    rest/spot/ticker/symbol=ALL
#   6. OKX     rest/linear/ticker/symbol=ALL         (~12MB/hr compressed)
#   7. OKX     rest/spot/ticker/symbol=ALL
#
# Approach: create one tar per exchange on remote, download, extract locally.
#
# Local output structure:
#   data_all/{exchange}/{market}/{stream}/{stream}_{date}_hr{HH}.jsonl.gz
#   e.g. data_all/binance/linear/ticker/ticker_{date}_hr{HH}.jsonl.gz
#        data_all/okx/spot/ticker/ticker_{date}_hr{HH}.jsonl.gz
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
echo "DOWNLOAD symbol=ALL TICKER DATA (Binance + Bybit + OKX)"
echo "======================================================================"
echo "Discovering dates with symbol=ALL data..."

ALL_DATES=$(remote "for d in \$(ls -1 $REMOTE_BASE/ | sed 's/dt=//'); do
    for ex in binance bybit okx; do
        if [ -f $REMOTE_BASE/dt=\$d/hr=00/exchange=\$ex/source=rest/market=linear/stream=ticker/symbol=ALL/data.jsonl.gz ]; then
            echo \$d; break;
        fi;
    done;
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
echo "  Binance: linear/ticker, linear/fundingRate, spot/ticker"
echo "  Bybit:   linear/ticker, spot/ticker"
echo "  OKX:     linear/ticker, spot/ticker"
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
    "binance rest linear ticker      binance/linear/ticker"
    "binance rest linear fundingRate binance/linear/fundingRate"
    "binance rest spot   ticker      binance/spot/ticker"
    "bybit   rest linear ticker      bybit/linear/ticker"
    "bybit   rest spot   ticker      bybit/spot/ticker"
    "okx     rest linear ticker      okx/linear/ticker"
    "okx     rest spot   ticker      okx/spot/ticker"
)

# ── Download function ─────────────────────────────────────────────────────────
# Downloads files individually via scp (no tar) for natural resume support.
download_job() {
    local EXCHANGE="$1"
    local SOURCE="$2"
    local MARKET="$3"
    local STREAM="$4"
    local LOCAL_DIR="$LOCAL_BASE/$5"

    echo "══════════════════════════════════════════════════════════════════════"
    echo "  $EXCHANGE / $MARKET / $STREAM  →  $LOCAL_DIR"
    echo "══════════════════════════════════════════════════════════════════════"

    mkdir -p "$LOCAL_DIR"

    # Build list of files we need
    declare -a NEED_DATES=()
    declare -a NEED_HOURS=()
    NEEDED=0
    SKIPPED=0

    for DATE in "${DATES[@]}"; do
        for HR in $(printf '%02d ' {0..23}); do
            # Skip the current (partial) hour — it's still being written
            if ! $INCLUDE_PARTIAL && [[ "$DATE" == "$NOW_UTC_DATE" && "$HR" == "$NOW_UTC_HOUR" ]]; then
                continue
            fi

            LOCAL_FILE="$LOCAL_DIR/${STREAM}_${DATE}_hr${HR}.jsonl.gz"

            if $SKIP_EXISTING && [[ -f "$LOCAL_FILE" ]] && [[ -s "$LOCAL_FILE" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            NEED_DATES+=("$DATE")
            NEED_HOURS+=("$HR")
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

    # Download each file individually via scp
    T_JOB=$SECONDS
    DOWNLOADED=0
    MISSING=0
    FAILED=0

    for i in $(seq 0 $((NEEDED - 1))); do
        DATE="${NEED_DATES[$i]}"
        HR="${NEED_HOURS[$i]}"
        REMOTE_PATH="$REMOTE_BASE/dt=${DATE}/hr=${HR}/exchange=${EXCHANGE}/source=${SOURCE}/market=${MARKET}/stream=${STREAM}/symbol=ALL/data.jsonl.gz"
        LOCAL_FILE="$LOCAL_DIR/${STREAM}_${DATE}_hr${HR}.jsonl.gz"

        if scp -q -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_FILE" 2>/dev/null; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            # File might not exist on remote — that's OK
            rm -f "$LOCAL_FILE"
            MISSING=$((MISSING + 1))
        fi

        # Progress every 5 files or on last file
        if (( (DOWNLOADED + MISSING) % 5 == 0 )) || (( i == NEEDED - 1 )); then
            ELAPSED=$((SECONDS - T_JOB))
            DONE=$((DOWNLOADED + MISSING))
            if [[ $ELAPSED -gt 0 && $DONE -gt 0 ]]; then
                RATE=$(echo "$DONE $ELAPSED" | awk '{printf "%.1f", $1/$2*60}')
                ETA=$(echo "$NEEDED $DONE $ELAPSED" | awk '{rem=$1-$2; if($2>0) printf "%.0f", rem*$3/$2; else print "?"}')
                printf "\r  [%3d/%d] ✓ %d  ✗ %d  (%s files/min, ETA %ss)  " "$DONE" "$NEEDED" "$DOWNLOADED" "$MISSING" "$RATE" "$ETA"
            fi
        fi
    done
    echo ""

    JOB_SECS=$((SECONDS - T_JOB))

    # Local summary
    TOTAL_LOCAL=$(find "$LOCAL_DIR" -name "${STREAM}_*.jsonl.gz" -type f | wc -l)
    TOTAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
    echo "  ✓ Done: +${DOWNLOADED} new, ${MISSING} missing  [${JOB_SECS}s]"
    echo "    Total: $TOTAL_LOCAL files, $TOTAL_SIZE"
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
        echo "  $LDIR: $COUNT files, $SIZE"
    fi
done

echo "======================================================================"
