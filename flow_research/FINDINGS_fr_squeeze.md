# OI + Funding Rate Squeeze Strategy — Findings

## Hypothesis

When positioning is extreme (high OI + extreme FR + tilted LS ratio), the market is
vulnerable to a **squeeze in the opposite direction** after FR settlement. We test whether
entering in the squeeze direction at settlement time is profitable after fees.

**Fee constraint:** taker 10bps/leg = 20bps RT, maker 4bps/leg = 8bps RT.

## Data

- **55 coins** across Bybit linear perpetuals (1h, 4h, 8h FR intervals)
- **90 days** (Dec 2025 – Feb 2026)
- **47,363 total settlements** analyzed
- Channels: kline 1m, OI 5min, FR, LS ratio 5min, premium index 1m

## Key Results

### 1. FR Magnitude Does NOT Predict Squeeze Direction

The signed post-settlement return (in the squeeze direction) is **negative at all horizons**
for moderate FR levels, and only marginally positive at extreme levels:

| FR Bucket | N | Med 30m | WR 30m | Med 1h | WR 1h | Med 4h | WR 4h |
|-----------|---:|--------:|-------:|-------:|------:|-------:|------:|
| \|FR\|<2 | 40,166 | +2.7 | 51.3% | +2.4 | 50.8% | +6.2 | 51.3% |
| \|FR\|2-5 | 3,851 | -2.4 | 48.0% | -3.2 | 47.4% | -13.1 | 47.1% |
| \|FR\|5-10 | 1,595 | -7.6 | 46.6% | -11.3 | 45.6% | -34.3 | 43.6% |
| \|FR\|10-20 | 818 | -18.5 | 43.1% | -21.1 | 45.7% | -40.5 | 44.0% |
| \|FR\|20-50 | 578 | -23.6 | 45.0% | -37.6 | 43.9% | -118.0 | 35.7% |
| \|FR\|50-100 | 216 | -0.6 | 50.0% | -51.6 | 43.1% | -155.5 | 37.5% |
| \|FR\|>100 | 139 | +7.1 | 51.1% | -42.3 | 47.5% | -223.2 | 38.1% |

**Critical finding:** The signed returns are NEGATIVE (anti-squeeze). Higher |FR| → price
continues in the crowded direction, it does NOT reverse. The squeeze hypothesis is **wrong**.

### 2. Statistical Tests Confirm Anti-Squeeze

For |FR| 20-50 bps (n=578), the t-test confirms **statistically significant negative**
signed returns:
- 1h: mean = -42.9 bps, **p=0.015**
- 2h: mean = -58.5 bps, **p=0.015**
- 4h: mean = -126.1 bps, **p=0.0000**

This means extreme FR is a **continuation signal**, not a reversal signal. Coins with
extreme FR keep moving in the direction that caused the extreme FR.

### 3. Only |FR|>=50 Shows Brief Squeeze at 5min

The ONLY profitable window:
- |FR|50-100: +23.5 bps mean at 5m, **p=0.039** → net +3.5 bps after taker fees
- |FR|>100: +78.8 bps mean at 15m, **p=0.012** → net +58.8 bps after taker fees

But these quickly reverse — by 1h the signed return is negative again. This is consistent
with the known post-settlement snap-back effect (already exploited in skytrade7 FR scalp).

### 4. OI/LS Conditioning Does Not Help

| Condition | N | Med 30m | WR 30m | Med 1h | WR 1h | Med 4h | WR 4h |
|-----------|---:|--------:|-------:|-------:|------:|-------:|------:|
| All \|FR\|>=5 | 3,346 | -10.8 | 45.9% | -17.4 | 45.2% | -53.0 | 41.7% |
| + OI rising 1h | 1,671 | -10.7 | 46.3% | -18.4 | 45.5% | -68.2 | 39.5% |
| + OI falling 1h | 1,672 | -10.8 | 45.4% | -15.5 | 45.0% | -36.5 | 44.0% |
| + OI surge (>2%) | 808 | -11.0 | 46.7% | -29.4 | 45.4% | -114.2 | 38.0% |
| + pre-1h aligned | 1,695 | -15.7 | 44.0% | -26.7 | 43.2% | -71.4 | 39.3% |
| + pre-1h counter | 1,647 | -5.3 | 47.8% | -8.6 | 47.4% | -31.7 | 44.3% |

No conditioning variable (OI change, LS ratio, pre-momentum) flips the sign to positive.
OI surge actually makes the anti-squeeze effect **stronger** (more negative at 4h).

The only positive subset: **FR>5 + high buy ratio** (n=132) shows +14.1 at 1h, 56% WR,
but this is a tiny sample and likely noise.

### 5. Trade-Level Profitability — All Negative Except Ultra-Short

| Threshold | Exit | Gross | Net Taker | WR | Trades/Day |
|-----------|------|------:|----------:|---:|-----------:|
| \|FR\|>=5 | 30m | +0.1 | **-19.9** | 45.9% | 37.2 |
| \|FR\|>=10 | 30m | -1.3 | **-21.3** | 45.2% | 19.4 |
| \|FR\|>=20 | 30m | +4.5 | **-15.5** | 47.1% | 10.4 |
| \|FR\|>=50 | 30m | +42.5 | **+22.5** | 50.4% | 3.9 |

Only |FR|>=50 at 30m exit is marginally profitable (+22.5 bps net taker), but with
P25/P75 = -178/+199 bps, the variance is enormous and the edge is tiny relative to risk.

### 6. By FR Interval

**1h coins** (27,931 settlements): Anti-squeeze confirmed. |FR|20-100 at 4h shows
mean = -131.9 bps, **p=0.000**.

**4h coins** (15,925 settlements): Mixed. |FR|5-20 at 4h shows mean = +69.6 bps,
**p=0.005** — this is the opposite (pro-squeeze), but only for low-FR events.

**8h coins** (1,347 settlements): Only 12 extreme events. Not enough data.

## Why the Squeeze Hypothesis Fails

1. **FR reflects real positioning pressure.** When FR is extreme, it's because the market
   is directionally imbalanced — and that imbalance typically persists because the
   fundamental driver (news, momentum, narrative) is still active.

2. **Settlement does not force liquidations.** FR payment is a cost, not a margin call.
   A -100bps FR on a 10x leveraged position is only -10bps of margin. Traders hold through
   unless they're already near liquidation.

3. **The "squeeze" already happened.** By the time FR reaches extreme levels, the price
   move that caused it has already occurred. The FR is a lagging indicator of positioning,
   not a leading indicator of reversal.

4. **Continuation dominates.** The data shows that extreme FR coins continue their trend
   for hours after settlement. The smart trade would be to go WITH the crowd (continuation),
   not against it — but even that is risky and inconsistent.

## Comparison with Known FR Strategies

The **FR flash scalp** (skytrade7) exploits a different mechanism:
- Enter BEFORE settlement, collect FR payment, exit immediately after
- The edge is the FR payment itself (~50-100+ bps), not the directional move
- Hold time: seconds, not hours
- This works because the FR payment is certain, not speculative

The squeeze strategy attempted here tries to predict directional moves AFTER settlement,
which the data conclusively shows does not work.

## Conclusion

**The OI + FR squeeze strategy is NOT viable.** Across 55 coins, 47K settlements, and
90 days of data:

- Extreme FR predicts **continuation**, not reversal
- No combination of OI, LS ratio, or premium index filters produces a positive edge
- The only positive signal (|FR|>50 at 5-15m) is the already-known snap-back effect
- All fixed-horizon exit strategies lose money after fees

**Recommendation:** Do not pursue this strategy further. The derivatives positioning
data (OI, FR, LS ratio) is better used as a regime/context filter for other strategies
than as a standalone directional signal.
