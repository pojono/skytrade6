# Findings Phase 20: Strict Fill Replay Fails

This phase runs the next bar that had not been cleared before:

- a stricter fill replay on the actual accepted trades from the current best microstructure-gated 30-day sleeve

Script:

- `strict_fill_replay.py`

Outputs:

- `out/strict_fill_replay_train_gate_30d.csv`
- `out/strict_fill_replay_train_gate_30d.md`

## What This Replay Does

It replays the actual `193` accepted fills from the current best microstructure-gated sleeve and replaces the generic cost model with a stricter entry/exit fill model:

- Bybit:
  - exact simulation against archived L2 order book snapshots
- Binance:
  - approximate simulation against archived 1% cumulative `bookDepth` snapshots
- both entry and exit are simulated
- both legs must be fillable

This is still not perfect venue-accurate execution, but it is materially stricter than the earlier generic slippage model.

## Result

Coverage:

- replayed fills: `193`
- fully fillable: `193`
- fill success rate: `100%`

But the economics fail:

- modeled avg net: `7.9684 bps`
- strict-fill avg net: `-1.6466 bps`

- modeled total PnL: `$3,917.39`
- strict-fill total PnL: `-$805.32`

- strict-fill win rate: `43.52%`

The main reason is fill cost:

- average strict execution slippage: `14.4799 bps`
- median strict execution slippage: `14.2627 bps`

That is substantially worse than what the earlier generic cost model was effectively assuming for these accepted trades.

## What This Means

This is the most important reality check in the project so far:

- the strategy survives coarse historical replay
- it improves under microstructure-aware filtering
- but it does **not** survive this stricter fill replay

So the answer to the real production question becomes:

- the signal may still exist
- but the currently modeled execution edge is too thin once stricter fill mechanics are applied

That means the current version is **not** production-ready as a real execution strategy.

## Correct Interpretation

This does not prove the signal is fake.

It does prove that at least one materially stricter fill model wipes out the edge.

So from here, the burden shifts:

- no more claiming a robust deployable edge
- no more trusting 1-minute replay alone
- future work must focus on execution realism first

## Current Status After This Test

The best honest status is now:

- research signal: still interesting
- execution model: not cleared
- production readiness: not cleared

The strategy is now better understood, but less deployable than it looked before the strict fill replay.
