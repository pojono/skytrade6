# Tick-Level Smart Money Divergence
## Out-Of-Sample Validation (July - August 2025)

- **Universe:** 15 highly volatile altcoins
- **Methodology:** Dynamic daily sizing (98th percentile = Whale, 20th percentile = Retail).
- **Signal:** Rolling 4h CVD Z-score > 1.5 for one cohort and < -1.5 for the other, coinciding with a 4h price extreme.

### Aggregate Results
- **Bearish Events (Short):** 30 trades | Win Rate: 33.3% | Avg 4h Edge: 1.18%
- **Bullish Events (Long):** 14 trades | Win Rate: 57.1% | Avg 4h Edge: 0.72%

### Per-Coin Breakdown
```text
          bear_count bear_4h  bear_wr  bull_count bull_4h  bull_wr
symbol                                                            
SOLUSDT            1   7.26%    0.00%           0   0.00%    0.00%
SUIUSDT            3  -1.52%   66.67%           0   0.00%    0.00%
AVAXUSDT           1   0.47%    0.00%           2   0.42%   50.00%
LINKUSDT           2   0.04%   50.00%           1   6.75%  100.00%
NEARUSDT           2   0.94%    0.00%           2   0.28%   50.00%
WLDUSDT            1  -1.23%  100.00%           2  -0.71%    0.00%
APTUSDT            5   0.97%   20.00%           2   0.31%   50.00%
ARBUSDT            6   3.76%    0.00%           0   0.00%    0.00%
AAVEUSDT           2  -0.78%  100.00%           5   0.56%   80.00%
INJUSDT            2   0.65%   50.00%           0   0.00%    0.00%
TIAUSDT            2  -0.54%   50.00%           0   0.00%    0.00%
SEIUSDT            3   1.80%   33.33%           0   0.00%    0.00%
OPUSDT             0   0.00%    0.00%           0   0.00%    0.00%
```
