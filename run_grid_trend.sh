#!/bin/bash
# Run grid+trend combined experiments on all symbols

EXCHANGE="bybit_futures"
START_DATE="2025-11-01"
END_DATE="2026-01-31"
SYMBOLS="BTCUSDT ETHUSDT SOLUSDT"

for SYMBOL in $SYMBOLS; do
    echo ">>> Running grid+trend on $SYMBOL ($START_DATE → $END_DATE)"
    OUTFILE="results/grid_trend_${SYMBOL}_${START_DATE}_${END_DATE}.txt"
    echo ">>> Output: $OUTFILE"
    python3 grid_trend_backtest.py \
        --exchange "$EXCHANGE" \
        --symbol "$SYMBOL" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --mode all \
        2>&1 | tee "$OUTFILE"
    echo "<<< Done $SYMBOL"
    echo ""
done

echo "All grid+trend experiments complete."
