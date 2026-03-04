#!/usr/bin/env python3
"""
Production Runner — Ties together strategy, data feed, execution, and ops.

Modes:
  paper   — Live WS data, simulated fills (default)
  live    — Live WS data, real orders on Bybit
  replay  — Historical data replay for testing

Usage:
    python3 runner.py --mode paper --config production_config.json
    python3 runner.py --mode live --config production_config.json --api-key KEY --api-secret SECRET
    python3 runner.py --mode replay --config production_config.json --symbols SOLUSDT,BTCUSDT
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from strategy_live import Strategy, StrategyConfig, Bar, Action, Signal
from ws_feed import DataFeed
from execution import Executor, ExecutorConfig, ExecutionMode, OrderResult

# =============================================================================
# Logging setup
# =============================================================================

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(mode: str):
    """Configure logging to file + console."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{mode}_{ts}.log"

    fmt = "%(asctime)s %(levelname)-7s %(name)-12s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    return log_file


logger = logging.getLogger("runner")

# =============================================================================
# State persistence
# =============================================================================

STATE_FILE = Path(__file__).parent / "state.json"
TRADES_LOG = Path(__file__).parent / "logs" / "trades.jsonl"


def save_state(strategy: Strategy, executor: Executor, feed_stats: dict):
    """Persist strategy state to disk for restart recovery."""
    state = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy.get_state(),
        "executor_stats": executor.get_stats(),
        "feed_stats": feed_stats,
    }
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(STATE_FILE)


def log_trade(trade_record: dict):
    """Append a closed trade to the JSONL audit log."""
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(trade_record, default=str) + "\n")


# =============================================================================
# Main orchestrator
# =============================================================================

class Runner:
    """
    Main production orchestrator.
    
    Connects strategy engine ↔ data feed ↔ executor.
    """

    def __init__(self, config_path: str, mode: str = "paper",
                 api_key: str = "", api_secret: str = "",
                 symbols_override: list[str] = None):
        self.config_path = config_path
        self.mode = mode

        # Load strategy config
        self.strategy_config = StrategyConfig.from_json(config_path)
        self.strategy = Strategy(self.strategy_config)

        # Determine symbols
        if symbols_override:
            self.symbols = symbols_override
        else:
            self.symbols = self.strategy_config.whitelist[:20]  # start conservative

        # Register symbols
        for sym in self.symbols:
            tier = self.strategy_config.symbol_tiers.get(sym, "B")
            self.strategy.add_symbol(sym, tier)

        # Setup executor
        exec_config = ExecutorConfig(
            mode=ExecutionMode.LIVE if mode == "live" else ExecutionMode.PAPER,
            exchange="bybit",
            api_key=api_key,
            api_secret=api_secret,
        )
        self.executor = Executor(exec_config, on_fill=self._on_fill)

        # Setup data feed
        self.feed = DataFeed(
            symbols=self.symbols,
            on_bar=self._on_bar,
            paper_mode=(mode != "live"),
        )

        self._running = False
        self._bar_count = 0
        self._signal_count = 0
        self._last_state_save = 0.0
        self._entry_fills: dict[str, OrderResult] = {}  # symbol -> entry fill

    def _on_bar(self, symbol: str, timestamp: str, bb_bar: Bar, bn_bar: Bar):
        """Called by DataFeed when a new 5m bar is ready."""
        self._bar_count += 1

        # Feed to strategy
        self.strategy.on_bar(timestamp, symbol, bb_bar, bn_bar)

        # Check for signals
        signals = self.strategy.get_signals()
        for sig in signals:
            self._signal_count += 1
            self._handle_signal(sig, bb_bar, bn_bar)

        # Periodic state save (every 5 minutes)
        now = time.time()
        if now - self._last_state_save > 300:
            self._save_state()
            self._last_state_save = now

    def _handle_signal(self, sig: Signal, bb_bar: Bar, bn_bar: Bar):
        """Process a strategy signal — route to executor."""
        mid = (bb_bar.close + bn_bar.close) / 2.0
        logger.info(f"SIGNAL: {sig.action.value} {sig.symbol} "
                    f"${sig.notional_usd:.0f} composite={sig.composite_score:.2f} "
                    f"rvol={sig.rvol_ratio:.2f} reason={sig.reason}")

        # Fire-and-forget execution (runs in background)
        asyncio.ensure_future(self._execute_signal(sig, mid))

    async def _execute_signal(self, sig: Signal, current_price: float):
        """Execute a signal asynchronously."""
        try:
            result = await self.executor.execute_signal(
                symbol=sig.symbol,
                side=sig.side,
                notional_usd=sig.notional_usd,
                current_price=current_price,
                use_limit=sig.use_limit_order,
                timestamp=sig.timestamp,
            )

            if result and result.success:
                if sig.action == Action.OPEN_LONG:
                    self._entry_fills[sig.symbol] = result
                elif sig.action == Action.CLOSE_LONG:
                    entry = self._entry_fills.pop(sig.symbol, None)
                    if entry:
                        trade = self.strategy.record_close(
                            symbol=sig.symbol,
                            entry_price=entry.fill_price,
                            exit_price=result.fill_price,
                            notional_usd=entry.notional_usd,
                            timestamp=sig.timestamp,
                        )
                        log_trade(trade)
            else:
                logger.warning(f"Execution failed for {sig.symbol}: "
                             f"{result.error if result else 'no result'}")
        except Exception as e:
            logger.error(f"Execution error for {sig.symbol}: {e}")

    def _on_fill(self, result: OrderResult):
        """Called by executor when an order is filled."""
        logger.info(f"FILL: {result.side} {result.symbol} "
                    f"${result.notional_usd:.0f} @ {result.fill_price:.4f} "
                    f"{'maker' if result.is_maker else 'taker'}")

    def _save_state(self):
        """Save current state to disk."""
        try:
            save_state(self.strategy, self.executor, self.feed.get_stats())
        except Exception as e:
            logger.error(f"State save failed: {e}")

    async def run(self):
        """Main run loop."""
        self._running = True
        logger.info(f"Runner starting in {self.mode} mode")
        logger.info(f"Symbols: {len(self.symbols)} — {self.symbols[:5]}...")
        logger.info(f"Config: sig_thr={self.strategy_config.sig_threshold} "
                    f"vol_thr={self.strategy_config.vol_threshold} "
                    f"max_pos={self.strategy_config.max_concurrent_positions}")

        await self.executor.start()
        await self.feed.start()

        # Status loop
        try:
            while self._running:
                await asyncio.sleep(30)
                status = self.strategy.get_status()
                feed_stats = self.feed.get_stats()
                logger.info(f"STATUS: bars={self._bar_count} signals={self._signal_count} "
                           f"open_pos={status['open_positions']} "
                           f"day_pnl=${status['daily_pnl_usd']:.1f} "
                           f"total_pnl=${status['total_pnl_usd']:.1f} "
                           f"dd=${status['drawdown_usd']:.1f} "
                           f"halted={'YES' if status['daily_halted'] or status['drawdown_halted'] else 'no'} "
                           f"feed_bars={feed_stats['bars_emitted']}")
        except asyncio.CancelledError:
            pass

        await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        self._save_state()

        # Close any open positions in live mode
        if self.mode == "live" and self.strategy.positions:
            logger.warning(f"Closing {len(self.strategy.positions)} open positions...")
            for sym, pos in list(self.strategy.positions.items()):
                try:
                    await self.executor.execute_signal(
                        symbol=sym, side="sell",
                        notional_usd=pos.notional_usd,
                        current_price=pos.entry_price,
                        use_limit=False,  # market order for emergency close
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as e:
                    logger.error(f"Emergency close failed for {sym}: {e}")

        await self.feed.stop()
        await self.executor.stop()
        logger.info("Shutdown complete")


# =============================================================================
# Replay mode (historical data)
# =============================================================================

async def run_replay(config_path: str, symbols: list[str]):
    """Run strategy against historical data for end-to-end testing."""
    import numpy as np
    from load_data import load_symbol

    config = StrategyConfig.from_json(config_path)
    strategy = Strategy(config)

    exec_config = ExecutorConfig(mode=ExecutionMode.PAPER)
    executor = Executor(exec_config)
    await executor.start()

    def safe_col(df, col):
        return df[col].fillna(0).values if col in df.columns else np.zeros(len(df))

    total_bars = 0
    total_signals = 0
    entry_fills: dict[str, OrderResult] = {}  # symbol -> entry fill result
    t0 = time.time()

    for sym in symbols:
        tier = config.symbol_tiers.get(sym, "B")
        strategy.add_symbol(sym, tier)

    for s_idx, sym in enumerate(symbols):
        print(f"  [{s_idx+1}/{len(symbols)}] Loading {sym}...", flush=True)
        df = load_symbol(sym)
        if df.empty:
            print(f"    EMPTY — skipped", flush=True)
            continue

        n = len(df)
        timestamps = [str(t) for t in df.index]

        bb_o = safe_col(df, 'bb_open')
        bb_h = safe_col(df, 'bb_high')
        bb_l = safe_col(df, 'bb_low')
        bb_c = df['bb_close'].values
        bb_v = safe_col(df, 'bb_volume')
        bb_t = safe_col(df, 'bb_turnover')
        bb_p = safe_col(df, 'bb_premium')
        bb_oi = safe_col(df, 'bb_oi')
        bn_o = safe_col(df, 'bn_open')
        bn_h = safe_col(df, 'bn_high')
        bn_l = safe_col(df, 'bn_low')
        bn_c = df['bn_close'].values
        bn_v = safe_col(df, 'bn_volume')
        bn_t = safe_col(df, 'bn_turnover')
        bn_tk = safe_col(df, 'bn_taker_buy_turnover')
        bn_p = safe_col(df, 'bn_premium')
        bn_oi = safe_col(df, 'bn_oi')

        ts_feed = time.time()
        for i in range(n):
            bb = Bar(bb_o[i], bb_h[i], bb_l[i], bb_c[i], bb_v[i], bb_t[i],
                     premium=bb_p[i], open_interest=bb_oi[i])
            bn = Bar(bn_o[i], bn_h[i], bn_l[i], bn_c[i], bn_v[i], bn_t[i],
                     taker_buy_turnover=bn_tk[i], premium=bn_p[i],
                     open_interest=bn_oi[i])

            strategy.on_bar(timestamps[i], sym, bb, bn)

            for sig in strategy.get_signals():
                total_signals += 1
                mid = (bb_c[i] + bn_c[i]) / 2.0

                result = await executor.execute_signal(
                    symbol=sig.symbol, side=sig.side,
                    notional_usd=sig.notional_usd,
                    current_price=mid,
                    use_limit=sig.use_limit_order,
                    timestamp=timestamps[i],
                )

                if result and result.success:
                    if sig.action == Action.OPEN_LONG:
                        entry_fills[sig.symbol] = result
                    elif sig.action == Action.CLOSE_LONG:
                        entry = entry_fills.pop(sig.symbol, None)
                        if entry:
                            strategy.record_close(
                                symbol=sig.symbol,
                                entry_price=entry.fill_price,
                                exit_price=result.fill_price,
                                notional_usd=entry.notional_usd,
                                timestamp=timestamps[i],
                            )

                action_str = sig.action.value
                print(f"    {timestamps[i]} {sym:15s} {action_str:12s} "
                      f"${sig.notional_usd:.0f} {sig.reason}", flush=True)

            if (i + 1) % 20000 == 0:
                elapsed = time.time() - ts_feed
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{n} bars ({rate:.0f} bar/s)", flush=True)

        total_bars += n
        elapsed_sym = time.time() - ts_feed
        print(f"    Done: {n} bars in {elapsed_sym:.1f}s ({n/max(elapsed_sym,0.01):.0f} bar/s)",
              flush=True)

    elapsed_total = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"Replay complete: {total_signals} signals across {len(symbols)} symbols "
          f"in {elapsed_total:.1f}s", flush=True)
    print(f"Strategy status: {json.dumps(strategy.get_status(), indent=2)}", flush=True)
    print(f"Executor stats: {json.dumps(executor.get_stats(), indent=2)}", flush=True)

    await executor.stop()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Exchange Strategy Runner")
    parser.add_argument("--mode", choices=["paper", "live", "replay"],
                        default="paper", help="Execution mode")
    parser.add_argument("--config", default="production_config.json",
                        help="Strategy config JSON file")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbol list (overrides config)")
    parser.add_argument("--api-key", default=os.environ.get("BYBIT_API_KEY", ""),
                        help="Bybit API key (live mode)")
    parser.add_argument("--api-secret", default=os.environ.get("BYBIT_API_SECRET", ""),
                        help="Bybit API secret (live mode)")
    args = parser.parse_args()

    log_file = setup_logging(args.mode)
    logger.info(f"Log file: {log_file}")

    symbols = args.symbols.split(",") if args.symbols else None

    if args.mode == "replay":
        # Replay mode: run against historical data
        syms = symbols or ["SOLUSDT", "BTCUSDT", "ETHUSDT", "AAVEUSDT", "ALCHUSDT"]
        logger.info(f"Replay mode: {len(syms)} symbols")
        asyncio.run(run_replay(args.config, syms))
    else:
        # Paper or live mode
        if args.mode == "live":
            if not args.api_key or not args.api_secret:
                logger.error("Live mode requires --api-key and --api-secret "
                           "(or BYBIT_API_KEY/BYBIT_API_SECRET env vars)")
                sys.exit(1)

        runner = Runner(
            config_path=args.config,
            mode=args.mode,
            api_key=args.api_key,
            api_secret=args.api_secret,
            symbols_override=symbols,
        )

        # Handle graceful shutdown
        loop = asyncio.new_event_loop()

        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            runner._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            loop.run_until_complete(runner.run())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, shutting down...")
            loop.run_until_complete(runner._shutdown())
        finally:
            loop.close()


if __name__ == "__main__":
    main()
