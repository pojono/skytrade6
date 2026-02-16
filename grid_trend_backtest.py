#!/usr/bin/env python3
"""
Grid + Trend Combined Backtester — tick-level.

The core question: can a trend-following overlay fix the grid's fatal flaw
(inventory accumulation in trending markets)?

We test several combination modes:

MODE 1 — REGIME SWITCH:
  Detect trend vs range regime using a simple moving-average slope.
  - Range regime → grid is ON, trend is OFF
  - Trend regime → grid is PAUSED (no new fills), trend follower takes over
  - When switching from trend→range, re-center the grid at current price

MODE 2 — PARALLEL (both always on):
  Grid runs continuously. Trend follower also runs continuously.
  They share no state — independent P&L tracking.
  The idea: grid profits in range, trend profits in trends, combined equity
  curve is smoother than either alone.

MODE 3 — GRID + TREND HEDGE:
  Grid runs continuously. When trend is detected, open a trend position
  that hedges the grid's inventory direction. When trend ends, close hedge.
  Grid keeps running throughout — the hedge just protects against inventory loss.

MODE 4 — ADAPTIVE GRID:
  Grid runs continuously but RE-CENTERS when a trend is detected.
  Close all open grid positions (realize the loss), rebuild grid at new price.
  This caps the maximum inventory loss at the cost of realized losses on re-center.

Trend detection: EMA crossover on tick-sampled prices (fast EMA vs slow EMA).
Trend following: simple breakout — enter on EMA cross, exit on reverse cross.

Reuses Grid class from grid_backtest.py.
"""

import sys
import time
import argparse
import psutil
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Import Grid from existing backtester
from grid_backtest import Grid, LevelState, MAKER_FEE_BPS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
TAKER_FEE_BPS = 5.5  # taker fee for trend trades (Bybit VIP0: 5.5 bps per side)
TREND_FEE_BPS = 2 * TAKER_FEE_BPS  # round-trip for trend trades


# ---------------------------------------------------------------------------
# Trend Follower — EMA crossover
# ---------------------------------------------------------------------------

class TrendState(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class EMATracker:
    """Incremental EMA computed on every N-th tick (subsampled)."""

    def __init__(self, fast_period, slow_period, subsample=1000):
        self.fast_alpha = 2.0 / (fast_period + 1)
        self.slow_alpha = 2.0 / (slow_period + 1)
        self.fast_ema = None
        self.slow_ema = None
        self.subsample = subsample
        self.tick_count = 0
        self.initialized = False

    def update(self, price):
        """Update EMAs. Returns True if this tick was sampled."""
        self.tick_count += 1
        if self.tick_count % self.subsample != 0:
            return False

        if self.fast_ema is None:
            self.fast_ema = price
            self.slow_ema = price
            self.initialized = True
            return True

        self.fast_ema = self.fast_alpha * price + (1 - self.fast_alpha) * self.fast_ema
        self.slow_ema = self.slow_alpha * price + (1 - self.slow_alpha) * self.slow_ema
        return True

    @property
    def trend_up(self):
        if not self.initialized:
            return None
        return self.fast_ema > self.slow_ema

    @property
    def spread_bps(self):
        """Spread between fast and slow EMA in bps."""
        if not self.initialized or self.slow_ema == 0:
            return 0.0
        return (self.fast_ema - self.slow_ema) / self.slow_ema * 10000


class TrendFollower:
    """
    Enhanced EMA crossover trend follower with:
    - Trailing stop: lock in profits when price moves in our favor
    - Hard stop-loss: cut losers at a fixed bps threshold
    - Cooldown: minimum ticks between trades to avoid whipsaw
    """

    def __init__(self, ema_tracker, min_spread_bps=0.0,
                 stop_loss_bps=50.0, trailing_stop_bps=30.0,
                 cooldown_ticks=5000):
        self.ema = ema_tracker
        self.min_spread_bps = min_spread_bps
        self.stop_loss_bps = stop_loss_bps
        self.trailing_stop_bps = trailing_stop_bps
        self.cooldown_ticks = cooldown_ticks
        self.state = TrendState.FLAT
        self.entry_price = 0.0
        self.best_price = 0.0  # best price since entry (for trailing stop)
        self.ticks_since_exit = cooldown_ticks  # allow immediate first trade
        self.completed_trades = []
        self.total_pnl = 0.0

    def _close_position(self, price, reason="trend"):
        """Close current position, return trade dict."""
        if self.state == TrendState.LONG:
            pnl_bps = (price - self.entry_price) / self.entry_price * 10000 - TREND_FEE_BPS
            side = 1
        elif self.state == TrendState.SHORT:
            pnl_bps = (self.entry_price - price) / self.entry_price * 10000 - TREND_FEE_BPS
            side = -1
        else:
            return None
        trade = {"entry": self.entry_price, "exit": price, "side": side,
                 "pnl_bps": pnl_bps, "type": reason}
        self.total_pnl += pnl_bps
        self.state = TrendState.FLAT
        self.entry_price = 0.0
        self.best_price = 0.0
        self.ticks_since_exit = 0
        self.completed_trades.append(trade)
        return trade

    def check_signal(self, price):
        """Check for entry/exit signals. Returns list of completed trades."""
        self.ticks_since_exit += 1

        if not self.ema.initialized:
            return []

        completed = []

        # --- Exit logic (check stops FIRST, before EMA signal) ---
        if self.state == TrendState.LONG:
            unrealized = (price - self.entry_price) / self.entry_price * 10000
            # Update best price
            if price > self.best_price:
                self.best_price = price
            # Trailing stop: if we had profit and it pulled back
            best_unrealized = (self.best_price - self.entry_price) / self.entry_price * 10000
            if self.trailing_stop_bps > 0 and best_unrealized > self.trailing_stop_bps:
                drawback = best_unrealized - unrealized
                if drawback >= self.trailing_stop_bps:
                    trade = self._close_position(price, "trailing_stop")
                    if trade:
                        completed.append(trade)
                        return completed
            # Hard stop-loss
            if self.stop_loss_bps > 0 and unrealized <= -self.stop_loss_bps:
                trade = self._close_position(price, "stop_loss")
                if trade:
                    completed.append(trade)
                    return completed
            # EMA reversal exit
            if not self.ema.trend_up:
                trade = self._close_position(price, "ema_reversal")
                if trade:
                    completed.append(trade)
                    return completed

        elif self.state == TrendState.SHORT:
            unrealized = (self.entry_price - price) / self.entry_price * 10000
            # Update best price (lowest for shorts)
            if price < self.best_price:
                self.best_price = price
            # Trailing stop
            best_unrealized = (self.entry_price - self.best_price) / self.entry_price * 10000
            if self.trailing_stop_bps > 0 and best_unrealized > self.trailing_stop_bps:
                drawback = best_unrealized - unrealized
                if drawback >= self.trailing_stop_bps:
                    trade = self._close_position(price, "trailing_stop")
                    if trade:
                        completed.append(trade)
                        return completed
            # Hard stop-loss
            if self.stop_loss_bps > 0 and unrealized <= -self.stop_loss_bps:
                trade = self._close_position(price, "stop_loss")
                if trade:
                    completed.append(trade)
                    return completed
            # EMA reversal exit
            if self.ema.trend_up:
                trade = self._close_position(price, "ema_reversal")
                if trade:
                    completed.append(trade)
                    return completed

        # --- Entry logic ---
        if self.state == TrendState.FLAT and self.ticks_since_exit >= self.cooldown_ticks:
            spread = abs(self.ema.spread_bps)
            if self.ema.trend_up and spread >= self.min_spread_bps:
                self.state = TrendState.LONG
                self.entry_price = price
                self.best_price = price
            elif not self.ema.trend_up and spread >= self.min_spread_bps:
                self.state = TrendState.SHORT
                self.entry_price = price
                self.best_price = price

        return completed

    def get_unrealized_pnl(self, price):
        if self.state == TrendState.LONG:
            return (price - self.entry_price) / self.entry_price * 10000
        elif self.state == TrendState.SHORT:
            return (self.entry_price - price) / self.entry_price * 10000
        return 0.0

    def force_close(self, price):
        """Force close any open position."""
        if self.state == TrendState.FLAT:
            return []
        trade = self._close_position(price, "forced")
        return [trade] if trade else []


# ---------------------------------------------------------------------------
# MODE 1: Regime Switch
# ---------------------------------------------------------------------------

def run_regime_switch(symbol, exchange, start_date, end_date,
                      cell_bps, n_levels, ema_fast, ema_slow, subsample,
                      trend_threshold_bps=10.0):
    """
    Grid ON in range, trend ON in trends. Re-center grid on regime change.
    """
    dates = pd.date_range(start_date, end_date)
    label = f"regime_switch_{cell_bps}bps_{n_levels}lvl_ema{ema_fast}_{ema_slow}"

    # Load first tick for center price
    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        print(f"  No data for {symbol}")
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    # Initialize
    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    ema = EMATracker(ema_fast, ema_slow, subsample)
    trend = TrendFollower(ema, min_spread_bps=trend_threshold_bps)

    grid_active = True
    regime = "range"
    recenters = 0
    grid_realized = 0.0
    grid_recenter_losses = 0.0
    all_grid_trades = []

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  MODE 1: REGIME SWITCH — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}, Trend threshold: {trend_threshold_bps} bps")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_grid_trades = 0
        day_trend_trades = 0

        for p in prices:
            last_price = p
            sampled = ema.update(p)

            if sampled and ema.initialized:
                spread = abs(ema.spread_bps)
                new_regime = "trend" if spread >= trend_threshold_bps else "range"

                if new_regime != regime:
                    if new_regime == "trend":
                        # Switching to trend: close all grid positions, pause grid
                        loss = _close_grid_positions(grid, p)
                        grid_recenter_losses += loss
                        grid_active = False
                    else:
                        # Switching to range: re-center grid, resume
                        cell = p * cell_bps / 10000
                        grid = Grid(center_price=p, cell_width=cell, n_levels_per_side=n_levels)
                        grid_active = True
                        recenters += 1
                        # Also close trend position
                        closed = trend.force_close(p)
                        day_trend_trades += len(closed)

                    regime = new_regime

            # Process grid if active
            if grid_active:
                completed = grid.process_tick(p)
                if completed:
                    day_grid_trades += len(completed)
                    for t in completed:
                        grid_realized += t["pnl_bps"]
                    all_grid_trades.extend(completed)

            # Process trend if in trend regime
            if regime == "trend":
                completed = trend.check_signal(p)
                day_trend_trades += len(completed)

        # Daily log
        longs, shorts = grid.get_open_count()
        grid_unrl = grid.get_unrealized_pnl(last_price) if grid_active else 0.0
        trend_unrl = trend.get_unrealized_pnl(last_price)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"regime={regime:5s} | grid_t=+{day_grid_trades} trend_t=+{day_trend_trades} | "
              f"grid_real={grid_realized:+.1f} grid_unrl={grid_unrl:+.1f} | "
              f"trend_pnl={trend.total_pnl:+.1f} trend_unrl={trend_unrl:+.1f} | "
              f"recenters={recenters} | {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    # Final
    grid_unrl = grid.get_unrealized_pnl(last_price) if grid_active else 0.0
    trend_unrl = trend.get_unrealized_pnl(last_price)
    total_grid_net = grid_realized + grid_recenter_losses + grid_unrl
    total_trend_net = trend.total_pnl + trend_unrl
    combined_net = total_grid_net + total_trend_net

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, grid_recenter_losses, grid_unrl,
        trend.completed_trades, trend.total_pnl, trend_unrl,
        recenters, combined_net
    )


# ---------------------------------------------------------------------------
# MODE 2: Parallel (both always on)
# ---------------------------------------------------------------------------

def run_parallel(symbol, exchange, start_date, end_date,
                 cell_bps, n_levels, ema_fast, ema_slow, subsample,
                 trend_threshold_bps=0.0):
    """
    Grid and trend follower both run independently, always on.
    """
    dates = pd.date_range(start_date, end_date)
    label = f"parallel_{cell_bps}bps_{n_levels}lvl_ema{ema_fast}_{ema_slow}"

    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    ema = EMATracker(ema_fast, ema_slow, subsample)
    trend = TrendFollower(ema, min_spread_bps=trend_threshold_bps)

    grid_realized = 0.0
    all_grid_trades = []

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  MODE 2: PARALLEL — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_grid_trades = 0
        day_trend_trades = 0

        for p in prices:
            last_price = p
            ema.update(p)

            # Grid always processes
            completed = grid.process_tick(p)
            if completed:
                day_grid_trades += len(completed)
                for t in completed:
                    grid_realized += t["pnl_bps"]
                all_grid_trades.extend(completed)

            # Trend always checks
            completed = trend.check_signal(p)
            day_trend_trades += len(completed)

        longs, shorts = grid.get_open_count()
        grid_unrl = grid.get_unrealized_pnl(last_price)
        trend_unrl = trend.get_unrealized_pnl(last_price)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"grid_t=+{day_grid_trades} trend_t=+{day_trend_trades} | "
              f"grid_real={grid_realized:+.1f} grid_unrl={grid_unrl:+.1f} | "
              f"trend_pnl={trend.total_pnl:+.1f} trend_unrl={trend_unrl:+.1f} | "
              f"inv={grid.inventory:.0f} | {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    grid_unrl = grid.get_unrealized_pnl(last_price)
    trend_unrl = trend.get_unrealized_pnl(last_price)
    combined_net = (grid_realized + grid_unrl) + (trend.total_pnl + trend_unrl)

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, 0.0, grid_unrl,
        trend.completed_trades, trend.total_pnl, trend_unrl,
        0, combined_net
    )


# ---------------------------------------------------------------------------
# MODE 3: Grid + Trend Hedge
# ---------------------------------------------------------------------------

def run_grid_hedge(symbol, exchange, start_date, end_date,
                   cell_bps, n_levels, ema_fast, ema_slow, subsample,
                   trend_threshold_bps=10.0):
    """
    Grid always runs. When trend detected AND grid has inventory,
    open a hedge position opposite to inventory. Close hedge when trend ends.
    """
    dates = pd.date_range(start_date, end_date)
    label = f"grid_hedge_{cell_bps}bps_{n_levels}lvl_ema{ema_fast}_{ema_slow}"

    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    ema = EMATracker(ema_fast, ema_slow, subsample)

    grid_realized = 0.0
    all_grid_trades = []
    hedge_trades = []
    hedge_pnl = 0.0
    hedge_state = "flat"  # flat, long, short
    hedge_entry = 0.0
    hedge_count = 0

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  MODE 3: GRID + TREND HEDGE — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_grid_trades = 0
        day_hedge_trades = 0

        for p in prices:
            last_price = p
            sampled = ema.update(p)

            # Grid always processes
            completed = grid.process_tick(p)
            if completed:
                day_grid_trades += len(completed)
                for t in completed:
                    grid_realized += t["pnl_bps"]
                all_grid_trades.extend(completed)

            # Hedge logic on sampled ticks
            if sampled and ema.initialized:
                spread = abs(ema.spread_bps)
                in_trend = spread >= trend_threshold_bps
                inv = grid.inventory

                if in_trend and hedge_state == "flat" and abs(inv) > 0:
                    # Open hedge in trend direction (which should offset inventory)
                    if ema.trend_up and inv < 0:
                        # Trend up, we're short → hedge long
                        hedge_state = "long"
                        hedge_entry = p
                        hedge_count += 1
                    elif not ema.trend_up and inv > 0:
                        # Trend down, we're long → hedge short
                        hedge_state = "short"
                        hedge_entry = p
                        hedge_count += 1
                    elif ema.trend_up and inv > 0:
                        # Trend up, we're long → no hedge needed (trend helps us)
                        pass
                    elif not ema.trend_up and inv < 0:
                        # Trend down, we're short → no hedge needed
                        pass

                elif not in_trend and hedge_state != "flat":
                    # Close hedge
                    if hedge_state == "long":
                        pnl = (p - hedge_entry) / hedge_entry * 10000 - TREND_FEE_BPS
                    else:
                        pnl = (hedge_entry - p) / hedge_entry * 10000 - TREND_FEE_BPS
                    hedge_pnl += pnl
                    hedge_trades.append({"entry": hedge_entry, "exit": p,
                                         "side": 1 if hedge_state == "long" else -1,
                                         "pnl_bps": pnl, "type": "hedge"})
                    hedge_state = "flat"
                    hedge_entry = 0.0
                    day_hedge_trades += 1

        # Daily log
        grid_unrl = grid.get_unrealized_pnl(last_price)
        hedge_unrl = 0.0
        if hedge_state == "long":
            hedge_unrl = (last_price - hedge_entry) / hedge_entry * 10000
        elif hedge_state == "short":
            hedge_unrl = (hedge_entry - last_price) / hedge_entry * 10000
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"grid_t=+{day_grid_trades} hedge_t=+{day_hedge_trades} | "
              f"grid_real={grid_realized:+.1f} grid_unrl={grid_unrl:+.1f} | "
              f"hedge_pnl={hedge_pnl:+.1f} hedge_unrl={hedge_unrl:+.1f} hedge={hedge_state} | "
              f"inv={grid.inventory:.0f} | {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    # Close any open hedge at end
    if hedge_state != "flat":
        if hedge_state == "long":
            pnl = (last_price - hedge_entry) / hedge_entry * 10000 - TREND_FEE_BPS
        else:
            pnl = (hedge_entry - last_price) / hedge_entry * 10000 - TREND_FEE_BPS
        hedge_pnl += pnl
        hedge_trades.append({"entry": hedge_entry, "exit": last_price,
                             "side": 1 if hedge_state == "long" else -1,
                             "pnl_bps": pnl, "type": "hedge_forced"})

    grid_unrl = grid.get_unrealized_pnl(last_price)
    hedge_unrl = 0.0  # closed above
    combined_net = (grid_realized + grid_unrl) + hedge_pnl

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, 0.0, grid_unrl,
        hedge_trades, hedge_pnl, hedge_unrl,
        hedge_count, combined_net
    )


# ---------------------------------------------------------------------------
# MODE 4: Adaptive Grid (re-center on trend)
# ---------------------------------------------------------------------------

def run_adaptive_grid(symbol, exchange, start_date, end_date,
                      cell_bps, n_levels, ema_fast, ema_slow, subsample,
                      trend_threshold_bps=10.0, recenter_cooldown_ticks=50000):
    """
    Grid always runs. When trend detected, close all positions and re-center
    grid at current price. Cooldown prevents excessive re-centering.
    """
    dates = pd.date_range(start_date, end_date)
    label = f"adaptive_{cell_bps}bps_{n_levels}lvl_ema{ema_fast}_{ema_slow}"

    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    ema = EMATracker(ema_fast, ema_slow, subsample)

    grid_realized = 0.0
    recenter_losses = 0.0
    all_grid_trades = []
    recenters = 0
    ticks_since_recenter = recenter_cooldown_ticks  # allow immediate first recenter
    was_trending = False

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  MODE 4: ADAPTIVE GRID — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}, Cooldown: {recenter_cooldown_ticks} ticks")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_grid_trades = 0
        day_recenters = 0

        for p in prices:
            last_price = p
            ticks_since_recenter += 1
            sampled = ema.update(p)

            # Check for re-center trigger
            if sampled and ema.initialized:
                spread = abs(ema.spread_bps)
                is_trending = spread >= trend_threshold_bps

                # Re-center when transitioning INTO trend and cooldown elapsed
                if is_trending and not was_trending and ticks_since_recenter >= recenter_cooldown_ticks:
                    # Close all grid positions
                    loss = _close_grid_positions(grid, p)
                    recenter_losses += loss
                    # Rebuild grid at current price
                    cell = p * cell_bps / 10000
                    grid = Grid(center_price=p, cell_width=cell, n_levels_per_side=n_levels)
                    recenters += 1
                    day_recenters += 1
                    ticks_since_recenter = 0

                was_trending = is_trending

            # Grid always processes
            completed = grid.process_tick(p)
            if completed:
                day_grid_trades += len(completed)
                for t in completed:
                    grid_realized += t["pnl_bps"]
                all_grid_trades.extend(completed)

        grid_unrl = grid.get_unrealized_pnl(last_price)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"grid_t=+{day_grid_trades} recenters=+{day_recenters} | "
              f"grid_real={grid_realized:+.1f} recenter_loss={recenter_losses:+.1f} "
              f"grid_unrl={grid_unrl:+.1f} | "
              f"inv={grid.inventory:.0f} total_recenters={recenters} | "
              f"{elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    grid_unrl = grid.get_unrealized_pnl(last_price)
    combined_net = grid_realized + recenter_losses + grid_unrl

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, recenter_losses, grid_unrl,
        [], 0.0, 0.0,
        recenters, combined_net
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_grid_positions(grid, price):
    """Close all open grid positions at current price. Returns total loss in bps."""
    loss = 0.0
    for lvl in grid.levels:
        if lvl.state == LevelState.LONG:
            pnl = (price - lvl.entry_price) / lvl.entry_price * 10000 - TREND_FEE_BPS
            loss += pnl
            lvl.state = LevelState.BUY
            lvl.entry_price = 0.0
            grid.inventory -= 1.0
        elif lvl.state == LevelState.SHORT:
            pnl = (lvl.entry_price - price) / lvl.entry_price * 10000 - TREND_FEE_BPS
            loss += pnl
            lvl.state = LevelState.SELL
            lvl.entry_price = 0.0
            grid.inventory += 1.0
    return loss


def _print_combined_summary(label, symbol, total_ticks, elapsed,
                            grid_trades, grid_realized, grid_recenter_losses, grid_unrl,
                            trend_trades, trend_realized, trend_unrl,
                            recenters_or_hedges, combined_net):
    """Print and return summary dict."""
    n_grid = len(grid_trades)
    n_trend = len(trend_trades)

    grid_net = grid_realized + grid_recenter_losses + grid_unrl
    trend_net = trend_realized + trend_unrl

    # Grid stats
    if n_grid > 0:
        grid_avg = sum(t["pnl_bps"] for t in grid_trades) / n_grid
        grid_wr = sum(1 for t in grid_trades if t["pnl_bps"] > 0) / n_grid
    else:
        grid_avg = grid_wr = 0.0

    # Trend stats
    if n_trend > 0:
        trend_avg = sum(t["pnl_bps"] for t in trend_trades) / n_trend
        trend_wr = sum(1 for t in trend_trades if t["pnl_bps"] > 0) / n_trend
    else:
        trend_avg = trend_wr = 0.0

    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {label} — {symbol}")
    print(f"  {'='*60}")
    print(f"  Grid:  {n_grid} trades, avg={grid_avg:+.2f} bps, WR={grid_wr:.0%}")
    print(f"         realized={grid_realized:+.1f}, recenter_loss={grid_recenter_losses:+.1f}, "
          f"unrealized={grid_unrl:+.1f}, net={grid_net:+.1f}")
    print(f"  Trend: {n_trend} trades, avg={trend_avg:+.2f} bps, WR={trend_wr:.0%}")
    print(f"         realized={trend_realized:+.1f}, unrealized={trend_unrl:+.1f}, "
          f"net={trend_net:+.1f}")
    print(f"  Recenters/Hedges: {recenters_or_hedges}")
    print(f"  COMBINED NET: {combined_net:+.1f} bps")
    print(f"  Ticks: {total_ticks:,}, Time: {elapsed:.0f}s")

    return {
        "label": label, "symbol": symbol,
        "grid_trades": n_grid, "grid_avg": grid_avg, "grid_wr": grid_wr,
        "grid_realized": grid_realized, "grid_recenter_losses": grid_recenter_losses,
        "grid_unrl": grid_unrl, "grid_net": grid_net,
        "trend_trades": n_trend, "trend_avg": trend_avg, "trend_wr": trend_wr,
        "trend_realized": trend_realized, "trend_unrl": trend_unrl, "trend_net": trend_net,
        "recenters": recenters_or_hedges,
        "combined_net": combined_net,
        "total_ticks": total_ticks, "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Baseline: pure grid (no trend) for comparison
# ---------------------------------------------------------------------------

def run_pure_grid(symbol, exchange, start_date, end_date, cell_bps, n_levels):
    """Run pure grid for baseline comparison."""
    dates = pd.date_range(start_date, end_date)
    label = f"pure_grid_{cell_bps}bps_{n_levels}lvl"

    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    grid_realized = 0.0
    all_grid_trades = []

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  BASELINE: PURE GRID — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_trades = 0
        for p in prices:
            last_price = p
            completed = grid.process_tick(p)
            if completed:
                day_trades += len(completed)
                for t in completed:
                    grid_realized += t["pnl_bps"]
                all_grid_trades.extend(completed)

        grid_unrl = grid.get_unrealized_pnl(last_price)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"grid_t=+{day_trades} | "
              f"grid_real={grid_realized:+.1f} grid_unrl={grid_unrl:+.1f} | "
              f"inv={grid.inventory:.0f} | {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    grid_unrl = grid.get_unrealized_pnl(last_price)
    combined_net = grid_realized + grid_unrl

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, 0.0, grid_unrl,
        [], 0.0, 0.0,
        0, combined_net
    )


# ---------------------------------------------------------------------------
# Parameterized parallel runner for sweep mode
# ---------------------------------------------------------------------------

def _run_parallel_with_params(symbol, exchange, start_date, end_date,
                              cell_bps, n_levels,
                              ema_fast, ema_slow, subsample, threshold,
                              stop_loss_bps, trailing_stop_bps, cooldown_ticks,
                              label):
    """Parallel mode with explicit trend follower parameters."""
    dates = pd.date_range(start_date, end_date)

    first_path = PARQUET_DIR / symbol / "trades" / exchange / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_bps / 10000
    del first_trades

    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    ema = EMATracker(ema_fast, ema_slow, subsample)
    trend = TrendFollower(ema, min_spread_bps=threshold,
                          stop_loss_bps=stop_loss_bps,
                          trailing_stop_bps=trailing_stop_bps,
                          cooldown_ticks=cooldown_ticks)

    grid_realized = 0.0
    all_grid_trades = []

    t0 = time.time()
    total_ticks = 0
    last_price = center

    print(f"\n  PARALLEL — {label}")
    print(f"  Center: {center:.2f}, Cell: {cell:.2f}")
    print(f"  SL={stop_loss_bps} TS={trailing_stop_bps} CD={cooldown_ticks}")

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if not path.exists():
            continue

        trades_df = pd.read_parquet(path)
        prices = trades_df["price"].values
        total_ticks += len(prices)
        del trades_df

        day_grid_trades = 0
        day_trend_trades = 0

        for p in prices:
            last_price = p
            ema.update(p)

            completed = grid.process_tick(p)
            if completed:
                day_grid_trades += len(completed)
                for t in completed:
                    grid_realized += t["pnl_bps"]
                all_grid_trades.extend(completed)

            completed = trend.check_signal(p)
            day_trend_trades += len(completed)

        grid_unrl = grid.get_unrealized_pnl(last_price)
        trend_unrl = trend.get_unrealized_pnl(last_price)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {len(prices):,} ticks | "
              f"grid_t=+{day_grid_trades} trend_t=+{day_trend_trades} | "
              f"grid_real={grid_realized:+.1f} grid_unrl={grid_unrl:+.1f} | "
              f"trend_pnl={trend.total_pnl:+.1f} trend_unrl={trend_unrl:+.1f} | "
              f"inv={grid.inventory:.0f} | {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    grid_unrl = grid.get_unrealized_pnl(last_price)
    trend_unrl = trend.get_unrealized_pnl(last_price)
    combined_net = (grid_realized + grid_unrl) + (trend.total_pnl + trend_unrl)

    return _print_combined_summary(
        label, symbol, total_ticks, time.time() - t0,
        all_grid_trades, grid_realized, 0.0, grid_unrl,
        trend.completed_trades, trend.total_pnl, trend_unrl,
        0, combined_net
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid + Trend Combined Backtester")
    parser.add_argument("--exchange", default="bybit_futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--mode", default="all",
                        choices=["all", "regime", "parallel", "hedge", "adaptive",
                                 "baseline", "sweep"])
    args = parser.parse_args()

    symbol = args.symbol
    exchange = args.exchange
    start = args.start
    end = args.end

    print("=" * 70)
    print("  GRID + TREND COMBINED EXPERIMENTS")
    print(f"  {symbol} on {exchange} | {start} → {end}")
    print(f"  Grid fee: {MAKER_FEE_BPS} bps maker (={2*MAKER_FEE_BPS} bps RT)")
    print(f"  Trend fee: {TAKER_FEE_BPS} bps taker (={TREND_FEE_BPS} bps RT)")
    print("=" * 70)

    all_results = []

    if args.mode == "sweep":
        # ---------------------------------------------------------------
        # PARALLEL SWEEP: test many parameter combos for the winning mode
        # ---------------------------------------------------------------
        # Grid configs
        grid_configs = [
            (30, 3, "30bps_3lvl"),
        ]

        # Trend configs: (ema_fast, ema_slow, subsample, threshold, stop_loss, trailing, cooldown, label)
        trend_configs = [
            # --- No stops (original v1 behavior) ---
            (500,  2000,  500,  15.0, 0.0,  0.0,  0,     "raw"),
            # --- Base with stops ---
            (500,  2000,  500,  15.0, 50.0, 30.0, 5000,  "sl50_ts30"),
            # --- Tighter stop ---
            (500,  2000,  500,  15.0, 30.0, 20.0, 5000,  "sl30_ts20"),
            # --- Wider stop ---
            (500,  2000,  500,  15.0, 80.0, 50.0, 5000,  "sl80_ts50"),
            # --- No trailing, just stop-loss ---
            (500,  2000,  500,  15.0, 50.0, 0.0,  5000,  "sl50_nots"),
            # --- Longer cooldown ---
            (500,  2000,  500,  15.0, 50.0, 30.0, 20000, "sl50_cd20k"),
            # --- Slower EMA with stops ---
            (1000, 4000,  500,  15.0, 50.0, 30.0, 5000,  "mid_sl50"),
        ]

        # Baselines first
        for cell_bps, n_levels, grid_label in grid_configs:
            print(f"\n{'#'*70}")
            print(f"  BASELINE: {grid_label}")
            print(f"{'#'*70}")
            result = run_pure_grid(symbol, exchange, start, end, cell_bps, n_levels)
            if result:
                result["mode"] = f"baseline_{grid_label}"
                result["ema_label"] = "-"
                result["config"] = grid_label
                all_results.append(result)

        # Parallel sweep: grid_config × trend_config
        for cell_bps, n_levels, grid_label in grid_configs:
            for (ema_f, ema_s, sub, thresh, sl, ts, cd, t_label) in trend_configs:
                combo_label = f"{grid_label}+{t_label}"
                print(f"\n{'#'*70}")
                print(f"  PARALLEL: {combo_label}")
                print(f"  Grid: {cell_bps}bps {n_levels}lvl | "
                      f"EMA: {ema_f}/{ema_s} sub={sub} thresh={thresh} | "
                      f"SL={sl} TS={ts} CD={cd}")
                print(f"{'#'*70}")

                # Need fresh EMA + TrendFollower for each combo
                result = _run_parallel_with_params(
                    symbol, exchange, start, end,
                    cell_bps, n_levels,
                    ema_f, ema_s, sub, thresh,
                    sl, ts, cd,
                    combo_label
                )
                if result:
                    result["mode"] = "parallel"
                    result["ema_label"] = t_label
                    result["config"] = combo_label
                    all_results.append(result)

    else:
        # ---------------------------------------------------------------
        # Original mode-based runs
        # ---------------------------------------------------------------
        cell_bps = 30
        n_levels = 3

        ema_configs = [
            (500, 2000, 500, 15.0, "fast"),
            (2000, 8000, 1000, 20.0, "medium"),
            (5000, 20000, 2000, 30.0, "slow"),
        ]

        if args.mode in ("all", "baseline"):
            result = run_pure_grid(symbol, exchange, start, end, cell_bps, n_levels)
            if result:
                all_results.append(result)

        for ema_fast, ema_slow, subsample, threshold, ema_label in ema_configs:
            tag = f" [{ema_label}]"

            if args.mode in ("all", "regime"):
                print(f"\n{'#'*70}")
                print(f"  MODE 1: REGIME SWITCH{tag}")
                print(f"{'#'*70}")
                result = run_regime_switch(symbol, exchange, start, end,
                                           cell_bps, n_levels, ema_fast, ema_slow,
                                           subsample, threshold)
                if result:
                    result["mode"] = "regime_switch"
                    result["ema_label"] = ema_label
                    all_results.append(result)

            if args.mode in ("all", "parallel"):
                print(f"\n{'#'*70}")
                print(f"  MODE 2: PARALLEL{tag}")
                print(f"{'#'*70}")
                result = run_parallel(symbol, exchange, start, end,
                                      cell_bps, n_levels, ema_fast, ema_slow,
                                      subsample, threshold)
                if result:
                    result["mode"] = "parallel"
                    result["ema_label"] = ema_label
                    all_results.append(result)

            if args.mode in ("all", "hedge"):
                print(f"\n{'#'*70}")
                print(f"  MODE 3: GRID + TREND HEDGE{tag}")
                print(f"{'#'*70}")
                result = run_grid_hedge(symbol, exchange, start, end,
                                        cell_bps, n_levels, ema_fast, ema_slow,
                                        subsample, threshold)
                if result:
                    result["mode"] = "grid_hedge"
                    result["ema_label"] = ema_label
                    all_results.append(result)

            if args.mode in ("all", "adaptive"):
                print(f"\n{'#'*70}")
                print(f"  MODE 4: ADAPTIVE GRID{tag}")
                print(f"{'#'*70}")
                result = run_adaptive_grid(symbol, exchange, start, end,
                                           cell_bps, n_levels, ema_fast, ema_slow,
                                           subsample, threshold,
                                           recenter_cooldown_ticks=100000)
                if result:
                    result["mode"] = "adaptive"
                    result["ema_label"] = ema_label
                    all_results.append(result)

    # Final comparison table
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"  FINAL COMPARISON — {symbol}")
        print(f"{'='*100}")
        print(f"  {'Config':<30s} {'G_Trades':>8s} {'G_Net':>9s} "
              f"{'T_Trades':>8s} {'T_WR':>5s} {'T_Avg':>8s} {'T_Net':>9s} {'COMBINED':>10s}")
        print(f"  {'-'*92}")

        for r in sorted(all_results, key=lambda x: -x["combined_net"]):
            cfg = r.get("config", r.get("mode", r["label"][:28]))
            t_wr = f"{r['trend_wr']:.0%}" if r['trend_trades'] > 0 else "-"
            t_avg = f"{r['trend_avg']:+.1f}" if r['trend_trades'] > 0 else "-"
            print(f"  {cfg:<30s} {r['grid_trades']:>8d} {r['grid_net']:>+9.1f} "
                  f"{r['trend_trades']:>8d} {t_wr:>5s} {t_avg:>8s} {r['trend_net']:>+9.1f} "
                  f"{r['combined_net']:>+10.1f}")

    print(f"\n✅ All done.")


if __name__ == "__main__":
    main()
