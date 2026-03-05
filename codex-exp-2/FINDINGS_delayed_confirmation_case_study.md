# Delayed Confirmation Strategy Case Study
This converts short-horizon follow-through into an executable strategy variant: wait, confirm, then enter.
## Scope
- Covered symbols: 1000BONKUSDT, 1000PEPEUSDT, AAVEUSDT, ADAUSDT, AIXBTUSDT, APTUSDT, ARBUSDT, ASTERUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BARDUSDT, BCHUSDT, BIOUSDT, BNBUSDT, BTCUSDT, CRVUSDT, DASHUSDT, DOGEUSDT, DOTUSDT, EIGENUSDT, ENAUSDT, ETCUSDT, ETHFIUSDT, ETHUSDT, FARTCOINUSDT, FFUSDT, FILUSDT, HBARUSDT, HUSDT, ICPUSDT, KAITOUSDT, KITEUSDT, LDOUSDT, LINKUSDT, LTCUSDT, NEARUSDT, ONDOUSDT, OPUSDT, PAXGUSDT, PENGUUSDT, PIPPINUSDT, RIVERUSDT, SAHARAUSDT, SOLUSDT, STRKUSDT, SUIUSDT, TAOUSDT, TIAUSDT, TONUSDT, TRUMPUSDT, TRXUSDT, UNIUSDT, WIFUSDT, WLDUSDT, WLFIUSDT, XLMUSDT, XMRUSDT, XPLUSDT, XRPUSDT, ZECUSDT
- Entry is delayed by 30s or 60s after the original signal.
- Delayed entry price = last trade observed by that delayed timestamp.
- Exit remains at the original 4-hour horizon endpoint implied by the base signal.
- This is a covered-subset case study, not a full-universe result.
## Broad Research Set
- Train rows: 131
- Test rows: 139
- Base broad test avg: -65.71 bps
## Best Delayed-Entry Rule
- Delay: `60s`
- `ret_delay_bps >= 6.56`
- `ret_5m_bps >= 0.00`
- `buy_share_5m >= 0.522`
- Broad train avg after delayed confirmation: 5.74 bps on 20 rows
- Broad test avg after delayed confirmation: -39.38 bps on 20 rows
- Broad test improvement vs base: 26.34 bps
- Absolute broad test result remains negative.
## Apply Same Rule To Strict Strategy Subset
- Strict covered test rows before: 16
- Strict covered test avg before: -76.51 bps
- Strict covered test rows after: 1
- Strict covered test avg after: 187.92 bps
- Strict kept row: SOLUSDT at 2026-01-26 00:05:00+00:00
## Interpretation
- Delayed confirmation still helps relative to immediate entry on the covered broad set, which supports the 'wait for follow-through' idea.
- But no tested delayed-entry rule makes the expanded broad covered test positive in absolute terms.
- The strict-subset improvement is not broad evidence if it is driven by one surviving trade.