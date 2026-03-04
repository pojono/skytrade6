# Deployment Guide — Cross-Exchange Mean-Reversion Strategy

## Architecture

```
runner.py          Main orchestrator (paper/live/replay modes)
├── strategy_live.py   Signal engine + risk management
├── ws_feed.py         Real-time WS data from Bybit & Binance → 5m bars
└── execution.py       Order placement (paper sim or Bybit REST API)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp numpy pandas
```

### 2. Replay Test (historical data)

```bash
cd claude-exp-1
python3 runner.py --mode replay --symbols SOLUSDT,BTCUSDT,AAVEUSDT
```

This runs the full pipeline against local datalake files with simulated fills.

### 3. Paper Trading (live WS, simulated fills)

```bash
python3 runner.py --mode paper --config production_config.json
```

Connects to Bybit + Binance WebSocket streams, constructs 5m bars in real-time, computes signals, and simulates fills. No API keys needed.

### 4. Live Trading

```bash
export BYBIT_API_KEY=your_key
export BYBIT_API_SECRET=your_secret
python3 runner.py --mode live --config production_config.json
```

**WARNING**: Places real orders on Bybit. Start with `--symbols SOLUSDT` to limit scope.

## Configuration

All parameters in `production_config.json`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| parameters | sig_threshold | 2.5 | Composite z-score entry threshold |
| parameters | vol_threshold | 2.0 | Min realized vol ratio for entry |
| parameters | spread_vol_threshold | 1.3 | Min spread vol ratio for entry |
| parameters | max_hold | 24 | Max bars (2h) before forced exit |
| parameters | min_hold | 3 | Min bars (15m) before exit allowed |
| parameters | cooldown | 3 | Bars between trades per symbol |
| risk_management | max_concurrent_positions | 5 | Max simultaneous open positions |
| risk_management | daily_loss_stop_usd | 500 | Halt new entries after daily loss |
| risk_management | max_drawdown_usd | 2000 | Halt all entries on drawdown breach |
| risk_management | max_total_exposure_usd | 50000 | Max total notional across positions |
| position_sizing | base_notional_usd | 10000 | Base trade size |
| position_sizing | tier_multipliers | A=1.5, B=1.0, C=0.5 | Size scaling by symbol tier |

## Risk Controls

1. **Max concurrent positions** (default 5) — prevents correlated blowup
2. **Daily loss stop** ($500) — halts new entries, exits still fire
3. **Max drawdown breaker** ($2000) — halts all entries permanently until manual reset
4. **Exposure cap** ($50K) — reduces position size if at limit
5. **Per-symbol cooldown** (15 min) — prevents overtrading same symbol
6. **Min hold** (15 min) — prevents whipsawing on noisy signals

## Operational Notes

### Logs

All logs go to `logs/` directory:
- `logs/{mode}_{timestamp}.log` — full console output
- `logs/trades.jsonl` — append-only trade audit log (one JSON object per line)

### State Persistence

`state.json` is saved every 5 minutes with:
- Open positions and entry prices
- Daily/total PnL
- Risk halt status
- Symbol readiness (bars ingested)

### Restart Recovery

On restart, the strategy needs **300 bars (25 hours)** of warmup before signals fire. This is by design — the rolling statistics need history. Options:

1. **Accept warmup** — just wait 25h after restart
2. **Pre-seed from datalake** — load last 300 5m bars from CSV before starting WS

### Data Requirements

The strategy requires simultaneous data from **both** exchanges:
- Bybit: 1m klines (OHLCV + turnover)
- Binance: 1m klines (OHLCV + turnover + taker buy)
- Both: OI (polled every 5min via REST)
- Both: Premium index (polled every 1min via REST)

If either exchange WS drops, bars won't emit until both are caught up.

## Recommended Paper Trading Checklist

Before going live:

- [ ] Run paper mode for 2+ weeks
- [ ] Verify signals match historical replay patterns
- [ ] Confirm WS reconnection works (kill and restart)
- [ ] Check that daily loss stop fires correctly
- [ ] Verify bar construction alignment (compare paper bars to exchange charts)
- [ ] Monitor maker fill rate assumption (paper assumes 70% maker)
- [ ] Review `trades.jsonl` for any anomalies

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `strategy_live.py` | ~790 | Signal engine: RollingStats, SymbolState, Strategy, risk mgmt |
| `ws_feed.py` | ~380 | WS data feed: Bybit+Binance klines → 5m bar aggregation |
| `execution.py` | ~340 | Order execution: paper sim + Bybit V5 REST API |
| `runner.py` | ~440 | Main orchestrator: paper/live/replay modes, state persistence |
| `production_config.json` | ~190 | Strategy parameters, whitelist, blacklist, risk limits |
| `production_backtest.py` | ~580 | Walk-forward backtest (research, not for production) |
| `load_data.py` | ~330 | Historical data loader (research + replay mode) |
| `features.py` | ~217 | Feature engineering (research, replaced by RollingStats in live) |
| `backtest.py` | ~542 | V1-V3 backtests (research) |
