# Cohort Delayed Confirmation

This retests delayed confirmation only on small vetted symbol cohorts instead of the full covered universe.

## robust_plus_sol

- Symbols (3): SOLUSDT, TIAUSDT, WLDUSDT
- Broad base test avg: -40.67 bps
- Best delayed rule: wait `30s`, `ret_delay_bps >= 11.56`, `ret_5m_bps >= 0.00`, `buy_share_5m >= 0.500`
- Broad best delayed test avg: 175.93 bps on 1 rows
- Broad improvement vs base: 216.60 bps
- Strict base test avg: -54.61 bps
- Strict best delayed test avg: 175.93 bps on 1 rows

## train_positive_2plus

- Symbols (7): AVAXUSDT, ENAUSDT, ETHUSDT, SOLUSDT, TIAUSDT, WLDUSDT, XRPUSDT
- Broad base test avg: -64.18 bps
- Best delayed rule: wait `30s`, `ret_delay_bps >= 11.56`, `ret_5m_bps >= 0.00`, `buy_share_5m >= 0.500`
- Broad best delayed test avg: 108.57 bps on 2 rows
- Broad improvement vs base: 172.75 bps
- Strict base test avg: -128.19 bps
- Strict best delayed test avg: 175.93 bps on 1 rows

## total_positive

- Symbols (13): 1000BONKUSDT, AVAXUSDT, BARDUSDT, ENAUSDT, ETHUSDT, FARTCOINUSDT, PIPPINUSDT, SOLUSDT, TIAUSDT, TRUMPUSDT, WIFUSDT, WLDUSDT, XPLUSDT
- Broad base test avg: -72.50 bps
- Best delayed rule: wait `60s`, `ret_delay_bps >= 0.00`, `ret_5m_bps >= 11.00`, `buy_share_5m >= 0.500`
- Broad best delayed test avg: -3.32 bps on 9 rows
- Broad improvement vs base: 69.18 bps
- Strict base test avg: -107.43 bps
- Strict best delayed test avg: 187.92 bps on 1 rows

## tested_positive

- Symbols (11): 1000BONKUSDT, 1000PEPEUSDT, ASTERUSDT, BARDUSDT, FARTCOINUSDT, LINKUSDT, LTCUSDT, PIPPINUSDT, TIAUSDT, TRUMPUSDT, WLDUSDT
- Broad base test avg: -60.93 bps
- Best delayed rule: wait `60s`, `ret_delay_bps >= 0.00`, `ret_5m_bps >= 0.00`, `buy_share_5m >= 0.500`
- Broad best delayed test avg: -43.22 bps on 5 rows
- Broad improvement vs base: 17.71 bps
- Strict base test avg: 43.25 bps
- Strict best delayed test avg: 33.92 bps on 1 rows

## Bottom Line

- Best cohort by absolute broad delayed test avg: `robust_plus_sol`
- That cohort reached 175.93 bps on 1 delayed test rows.
- This is only credible if it still has enough rows; tiny cohorts can look good by accident.