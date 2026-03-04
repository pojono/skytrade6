# Phase 7 Findings

## Goal

Explain why the account-level return looked too low and test whether higher utilization or leverage can make the frozen strategy economically attractive.

## Why The Conservative Account Return Looked Small

The current preferred setup uses:

- 3 symbols
- 1 open position
- 10% capital per trade
- 1-minute holding period

That means capital is deployed only briefly.

From the current live-style run:

- total fills: `1,421`
- average allocation while in a trade: about `$10,176`
- average deployed capital across all clock time: about `$48`
- capital utilization across the full sample: about `0.0468%`

So the edge is positive, but the account spends most of its time idle.

That is why the account-level return looked weak even though the trade-level edge is real.

## Baseline Conservative Run

At `10%` allocation per trade:

- final capital: `$105,358.62`
- total return over the sample: `5.36%`
- annualized return (daily-based): about `9.54%`

This is positive but not compelling enough versus cash.

## Capital Utilization Sweep

Keeping the exact same frozen strategy and assumptions, only increasing per-trade allocation:

### 10% allocation

- final capital: `$105,358.62`
- total return: `5.36%`
- rough annualized return: `9.54%`

### 25% allocation

- final capital: `$113,935.40`
- total return: `13.94%`
- rough annualized return: `25.59%`

### 50% allocation

- final capital: `$129,796.35`
- total return: `29.80%`
- rough annualized return: `57.69%`

### 100% allocation

- final capital: `$167,409.77`
- total return: `67.41%`
- rough annualized return: `145.93%`

### 200% notional allocation

- final capital: `$284,304.73`
- total return: `184.30%`
- rough annualized return: `520.15%`

## What Leverage Does Here

In this model, increasing allocation or leverage helps a lot because:

1. The trade-level edge is positive
2. The strategy is time-sparse
3. The base version underuses capital heavily

So scaling notional exposure increases account return roughly in line with deployed size.

## What The Model Is Missing

These higher-size results are **not** production-ready return estimates.

They likely overstate reality because the current simulator does not yet include:

1. Borrow / funding costs for leveraged exposure
2. Position-size-dependent market impact
3. Venue-specific margin constraints
4. Liquidation path risk
5. Size-dependent slippage worsening beyond the current proxy

So:

- `25%` and `50%` allocation are plausible next validation targets
- `100%` and especially `200%` should be treated as stress experiments, not deployable conclusions

## Practical Conclusion

The strategy is not unattractive because the edge is weak.

It is unattractive at low size because the capital utilization is too low.

The most reasonable next step is:

1. Test `25%` and `50%` allocation as primary sizing candidates
2. Add a stronger size-dependent slippage model before trusting `100%+` notional
3. Reject any leverage level that turns fragile once slippage scales with size
