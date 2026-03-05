# Rolling Symbol Eligibility

This simulates a live symbol-eligibility gate with an expanding window.
A symbol only becomes eligible after enough prior 4-hour outcomes are known, so history is only counted once the prior trade is at least 4 hours old.

## Delayed Rule Used

- Delay: `30s`
- `ret_delay_bps >= 11.28`
- `ret_5m_bps >= 10.49`
- `buy_share_5m >= 0.522`

## Base Variant

- Unfiltered test avg: -91.51 bps
- Chosen eligibility: `min_hist >= 2`, `hist_avg >= 50.00 bps`
- Filtered test avg: -144.28 bps on 14 rows
- Improvement vs unfiltered: -52.78 bps
- Test symbols kept: 1000BONKUSDT,FARTCOINUSDT,ONDOUSDT,SUIUSDT,TRUMPUSDT,WIFUSDT,XLMUSDT,XPLUSDT

## Delayed Variant

- Unfiltered delayed test avg: -103.95 bps
- Chosen eligibility: `min_hist >= 1`, `hist_avg >= 0.00 bps`
- Filtered delayed test avg: no rows on 0 rows
- Improvement vs unfiltered: n/a
- Test symbols kept: none

## Bottom Line

- Rolling earned-eligibility improves selectivity, but it still does not make the covered test positive.
- For the base variant, every tested eligibility setting stayed negative; the best observed test average was -133.47 bps.
- For the delayed variant, the filtered sample is too sparse to establish a usable earned-eligibility rule.
- Any apparent success here is fragile because the filter keeps very few test trades.