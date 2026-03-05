# Grok-2 Strategy Development

## Objective
Develop a profitable trading strategy for crypto markets that remains viable after accounting for fees (maker 0.04%, taker 0.1%).

## Data Sources
- Utilizing existing data from `datalake/` for Binance, Bybit, and OKX.
- Additional data downloads in progress for OKX (BTCUSDT, ETHUSDT, SOLUSDT) covering 2025-07-01 to 2025-12-31.

## Approach
- Plan and validate multiple strategy ideas.
- Iterate quickly based on backtest results.
- Focus on avoiding lookahead bias and overfitting.
- Target a real, sustainable edge.

## Current Status
- **Data Preparation**: Downloads for OKX data are in progress with rate limit delays (BTCUSDT ~550-600/1288 tasks, ETHUSDT ~400-450/1288, SOLUSDT ~950-1000/1288).
- **Strategy Hypotheses**: Three strategies drafted—Funding Rate Arbitrage, Volatility Breakout, and Settlement Scalp.
- **Backtest Framework**: Under development with initial logic for all three strategies implemented.

## Key Files
- `data_assessment.md`: Initial data coverage analysis.
- `data_download_progress.md`: Ongoing OKX data download status.
- `strategy_hypotheses.md`: Detailed strategy concepts and validation approach.
- `backtest_framework.py`: Core script for simulating and evaluating trading strategies.
- `common_symbols.py`: Utility to fetch and compare symbols across exchanges.
- `requirements.txt`: List of dependencies for reproducibility.

## Next Steps
- Complete OKX data downloads and verify completeness.
- Finalize backtest framework with refined strategy logic.
- Run backtests to validate strategies, ensuring no lookahead bias or overfitting.
- Analyze results and iterate until a real edge is confirmed.
