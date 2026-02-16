#!/bin/bash
# Run all experiment suites on all symbols for full 3-month period
# Data: Bybit futures, Nov 1 2025 - Jan 31 2026

set -e

START="2025-11-01"
END="2026-01-31"
EXCHANGE="bybit_futures"

echo "============================================"
echo "  Running ALL experiments"
echo "  Period: $START → $END"
echo "  Exchange: $EXCHANGE"
echo "  Symbols: BTCUSDT ETHUSDT SOLUSDT"
echo "============================================"

# Signal experiments (E01-E15) — fast, 5m bars
echo ""
echo ">>> SUITE 1/4: Signal experiments (E01-E15)"
python3 -u run_experiment.py --suite signal --exchange $EXCHANGE --symbol all --start $START --end $END

# Novel experiments (N01-N16) — medium, richer features
echo ""
echo ">>> SUITE 2/4: Novel experiments (N01-N16)"
python3 -u run_experiment.py --suite novel --exchange $EXCHANGE --symbol all --start $START --end $END

# Grid OHLCV experiments (G01-G10) — fast, 5m bars
echo ""
echo ">>> SUITE 3/4: Grid OHLCV experiments (G01-G10)"
python3 -u run_experiment.py --suite grid_ohlcv --exchange $EXCHANGE --symbol all --start $START --end $END

# Grid tick experiments — slow, tick-by-tick
echo ""
echo ">>> SUITE 4/4: Grid tick experiments"
python3 -u run_experiment.py --suite grid_tick --exchange $EXCHANGE --symbol all --start $START --end $END

echo ""
echo "============================================"
echo "  ALL DONE"
echo "  Results in: results/"
echo "============================================"
ls -la results/
