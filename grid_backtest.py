#!/usr/bin/env python3
"""
Tick-level grid backtester — correct mechanics.

How a real grid works:
  1. Define N grid levels at fixed prices, spaced by cell_width.
  2. Levels below current price start as BUY orders.
  3. Levels above current price start as SELL orders.
  4. When a BUY fills → open long position, place TP sell at level+1.
  5. When a TP SELL fills → close long, profit = cell - fees. Restore BUY at original level.
  6. When a SELL fills → open short position, place TP buy at level-1.
  7. When a TP BUY fills → close short, profit = cell - fees. Restore SELL at original level.
  8. Grid levels are FIXED. They don't move with price.
  9. Re-center is a manual decision: close everything, rebuild grid at new price.

Each grid level is in one of 3 states:
  - RESTING_BUY: waiting for price to drop to this level
  - RESTING_SELL: waiting for price to rise to this level
  - POSITION_LONG: filled buy, waiting for TP sell at level above
  - POSITION_SHORT: filled sell, waiting for TP buy at level below

Bybit VIP0: maker 2 bps per fill (both entry and TP).
"""

import time
import psutil
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SOURCE = "bybit_futures"
PARQUET_DIR = Path("./parquet")
MAKER_FEE_BPS = 2.0

PERIOD_7D = ("2025-12-01", "2025-12-07")
PERIOD_30D = ("2025-12-01", "2025-12-30")


# ---------------------------------------------------------------------------
# Grid level state machine
# ---------------------------------------------------------------------------

class LevelState(Enum):
    BUY = "buy"          # resting buy order at this level
    SELL = "sell"         # resting sell order at this level
    LONG = "long"         # filled buy, TP sell at level above
    SHORT = "short"       # filled sell, TP buy at level below


class GridLevel:
    __slots__ = ["price", "state", "entry_price"]

    def __init__(self, price, state):
        self.price = price
        self.state = state
        self.entry_price = 0.0  # set when position opens


class Grid:
    """
    Fixed-price grid. Levels don't move.

    Levels are indexed 0..N-1 from lowest to highest price.
    Initially: levels below center_idx are BUY, levels above are SELL.
    """

    def __init__(self, center_price, cell_width, n_levels_per_side):
        self.cell_width = cell_width
        self.n_per_side = n_levels_per_side
        self.levels = []
        self.completed_trades = []
        self.inventory = 0.0  # net position in units

        # Build levels: n below center, n above center
        # Level prices: center - n*cell, ..., center - cell, center + cell, ..., center + n*cell
        for i in range(-n_levels_per_side, 0):
            lvl = GridLevel(price=center_price + i * cell_width, state=LevelState.BUY)
            self.levels.append(lvl)
        for i in range(1, n_levels_per_side + 1):
            lvl = GridLevel(price=center_price + i * cell_width, state=LevelState.SELL)
            self.levels.append(lvl)

        # Sort by price ascending
        self.levels.sort(key=lambda l: l.price)

    def process_tick(self, price):
        """Process one trade at given price. Returns list of completed trades."""
        completed = []

        for i, lvl in enumerate(self.levels):
            if lvl.state == LevelState.BUY and price <= lvl.price:
                # Buy order fills → open long position
                lvl.state = LevelState.LONG
                lvl.entry_price = lvl.price
                self.inventory += 1.0

            elif lvl.state == LevelState.SELL and price >= lvl.price:
                # Sell order fills → open short position
                lvl.state = LevelState.SHORT
                lvl.entry_price = lvl.price
                self.inventory -= 1.0

            elif lvl.state == LevelState.LONG:
                # TP is at the next level above (entry + cell_width)
                tp_price = lvl.entry_price + self.cell_width
                if price >= tp_price:
                    # TP fills → close long, profit
                    gross_bps = (tp_price - lvl.entry_price) / lvl.entry_price * 10000
                    net_bps = gross_bps - 2 * MAKER_FEE_BPS  # entry + exit maker fees
                    completed.append({
                        "entry": lvl.entry_price,
                        "exit": tp_price,
                        "side": 1,
                        "pnl_bps": net_bps,
                    })
                    self.inventory -= 1.0
                    # Restore to BUY at this level
                    lvl.state = LevelState.BUY
                    lvl.entry_price = 0.0

            elif lvl.state == LevelState.SHORT:
                # TP is at the next level below (entry - cell_width)
                tp_price = lvl.entry_price - self.cell_width
                if price <= tp_price:
                    # TP fills → close short, profit
                    gross_bps = (lvl.entry_price - tp_price) / lvl.entry_price * 10000
                    net_bps = gross_bps - 2 * MAKER_FEE_BPS
                    completed.append({
                        "entry": lvl.entry_price,
                        "exit": tp_price,
                        "side": -1,
                        "pnl_bps": net_bps,
                    })
                    self.inventory += 1.0
                    # Restore to SELL at this level
                    lvl.state = LevelState.SELL
                    lvl.entry_price = 0.0

        self.completed_trades.extend(completed)
        return completed

    def get_unrealized_pnl(self, current_price):
        """Mark-to-market all open positions."""
        pnl = 0.0
        for lvl in self.levels:
            if lvl.state == LevelState.LONG:
                pnl += (current_price - lvl.entry_price) / lvl.entry_price * 10000
            elif lvl.state == LevelState.SHORT:
                pnl += (lvl.entry_price - current_price) / lvl.entry_price * 10000
        return pnl

    def get_open_count(self):
        longs = sum(1 for l in self.levels if l.state == LevelState.LONG)
        shorts = sum(1 for l in self.levels if l.state == LevelState.SHORT)
        return longs, shorts

    def summary_str(self):
        states = {}
        for s in LevelState:
            states[s.value] = sum(1 for l in self.levels if l.state == s)
        return f"buy={states['buy']} sell={states['sell']} long={states['long']} short={states['short']}"


# ---------------------------------------------------------------------------
# Verification test on synthetic data
# ---------------------------------------------------------------------------

def test_grid_logic():
    """Verify grid mechanics on a simple price sequence."""
    print("=== VERIFICATION TEST ===")

    # Grid: center=100, cell=1, 2 levels per side
    # Levels: 98(buy), 99(buy), 101(sell), 102(sell)
    grid = Grid(center_price=100, cell_width=1, n_levels_per_side=2)

    print(f"Initial: {grid.summary_str()}")
    for l in grid.levels:
        print(f"  {l.price:.0f}: {l.state.value}")

    # Price drops to 99 → buy at 99 fills
    grid.process_tick(99)
    print(f"\nAfter tick 99: {grid.summary_str()}, inv={grid.inventory:.0f}")

    # Price drops to 98 → buy at 98 fills
    grid.process_tick(98)
    print(f"After tick 98: {grid.summary_str()}, inv={grid.inventory:.0f}")

    # Price rises to 99 → TP for the 98-buy (TP=99) fills
    trades = grid.process_tick(99)
    print(f"After tick 99: {grid.summary_str()}, inv={grid.inventory:.0f}, trades={len(trades)}")
    if trades:
        print(f"  Trade: entry={trades[0]['entry']:.0f} exit={trades[0]['exit']:.0f} pnl={trades[0]['pnl_bps']:.1f}bps")

    # Price rises to 100 → TP for the 99-buy (TP=100) fills
    trades = grid.process_tick(100)
    print(f"After tick 100: {grid.summary_str()}, inv={grid.inventory:.0f}, trades={len(trades)}")

    # Price rises to 101 → sell at 101 fills
    grid.process_tick(101)
    print(f"After tick 101: {grid.summary_str()}, inv={grid.inventory:.0f}")

    # Price drops to 100 → TP for the 101-sell (TP=100) fills
    trades = grid.process_tick(100)
    print(f"After tick 100: {grid.summary_str()}, inv={grid.inventory:.0f}, trades={len(trades)}")

    total_trades = len(grid.completed_trades)
    total_pnl = sum(t["pnl_bps"] for t in grid.completed_trades)
    print(f"\nTotal: {total_trades} trades, {total_pnl:+.1f} bps")
    print(f"Expected: 3 trades, each ~{100/100*10000/100 - 2*MAKER_FEE_BPS:.1f} bps = ~{3*(100 - 2*MAKER_FEE_BPS):.1f} bps total")

    # Verify
    assert total_trades == 3, f"Expected 3 trades, got {total_trades}"
    assert grid.inventory == 0.0, f"Expected 0 inventory, got {grid.inventory}"
    for t in grid.completed_trades:
        assert t["pnl_bps"] > 0, f"Expected positive PnL, got {t['pnl_bps']}"
    print("✅ All assertions passed!\n")


# ---------------------------------------------------------------------------
# Run on real data
# ---------------------------------------------------------------------------

def run_grid_on_data(symbol, start_date, end_date, cell_width_bps, n_levels, label):
    dates = pd.date_range(start_date, end_date)
    print(f"\n{'='*70}")
    print(f"  {symbol} — {label}")
    print(f"  Cell: {cell_width_bps} bps, Levels: {n_levels} per side")
    print(f"{'='*70}")

    # Load first day to get initial price
    first_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{dates[0].strftime('%Y-%m-%d')}.parquet"
    if not first_path.exists():
        print("  ❌ No data")
        return None

    first_trades = pd.read_parquet(first_path)
    center = first_trades["price"].values[0]
    cell = center * cell_width_bps / 10000
    del first_trades

    print(f"  Center: {center:.2f}, Cell width: {cell:.2f}")
    grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
    print(f"  Grid levels:")
    for l in grid.levels:
        print(f"    {l.price:.2f} [{l.state.value}]")

    t0 = time.time()
    total_ticks = 0

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
        if not path.exists():
            continue

        trades = pd.read_parquet(path)
        prices = trades["price"].values
        n = len(prices)
        total_ticks += n
        del trades

        day_trades_before = len(grid.completed_trades)

        # Process all ticks
        for p in prices:
            grid.process_tick(p)

        day_trades = len(grid.completed_trades) - day_trades_before
        longs, shorts = grid.get_open_count()
        unrealized = grid.get_unrealized_pnl(prices[-1])
        realized = sum(t["pnl_bps"] for t in grid.completed_trades)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().used / (1024**3)

        print(f"  [{i}/{len(dates)}] {ds}: {n:,} ticks, +{day_trades} trades  "
              f"| {grid.summary_str()} | inv={grid.inventory:.0f} "
              f"| real={realized:+.1f} unrl={unrealized:+.1f} "
              f"| {elapsed:.0f}s RAM={mem:.1f}GB", flush=True)

    # Final summary
    n_trades = len(grid.completed_trades)
    last_price = prices[-1] if len(prices) > 0 else center
    realized = sum(t["pnl_bps"] for t in grid.completed_trades)
    unrealized = grid.get_unrealized_pnl(last_price)
    longs, shorts = grid.get_open_count()

    if n_trades > 0:
        df = pd.DataFrame(grid.completed_trades)
        avg = df["pnl_bps"].mean()
        wr = (df["pnl_bps"] > 0).mean()
        df["cum"] = df["pnl_bps"].cumsum()
        dd = (df["cum"] - df["cum"].cummax()).min()
    else:
        avg = wr = dd = 0

    net = realized + unrealized
    elapsed = time.time() - t0

    print(f"\n  SUMMARY:")
    print(f"    Completed trades: {n_trades}")
    print(f"    Open positions: {longs} long, {shorts} short (inv={grid.inventory:.0f})")
    print(f"    Avg PnL per trade: {avg:+.2f} bps")
    print(f"    Win rate: {wr:.0%}")
    print(f"    Realized PnL: {realized:+.1f} bps")
    print(f"    Unrealized PnL: {unrealized:+.1f} bps")
    print(f"    Net PnL: {net:+.1f} bps")
    print(f"    Max drawdown: {dd:+.1f} bps")
    print(f"    Total ticks: {total_ticks:,}")
    print(f"    Time: {elapsed:.0f}s")

    return {
        "symbol": symbol, "label": label,
        "cell_bps": cell_width_bps, "n_levels": n_levels,
        "trades": n_trades, "longs": longs, "shorts": shorts,
        "avg_pnl": avg, "realized": realized, "unrealized": unrealized,
        "net": net, "win_rate": wr, "max_dd": dd,
        "inventory": grid.inventory,
    }


def main():
    # First: verify logic
    test_grid_logic()

    t_start = time.time()
    print("=" * 70)
    print("  TICK-LEVEL GRID: Correct Mechanics")
    print(f"  Fee: {MAKER_FEE_BPS} bps maker per fill (entry + exit = {2*MAKER_FEE_BPS} bps RT)")
    print("=" * 70)

    configs = [
        (20, 3, "20bps_3lvl"),
        (30, 3, "30bps_3lvl"),
        (50, 2, "50bps_2lvl"),
        (50, 3, "50bps_3lvl"),
    ]

    all_results = []

    # 7-day test
    print(f"\n{'#'*70}")
    print(f"  7-DAY TEST")
    print(f"{'#'*70}")

    for symbol in SYMBOLS:
        for cell_bps, n_lvl, cfg_label in configs:
            result = run_grid_on_data(symbol, *PERIOD_7D, cell_bps, n_lvl, f"7d_{cfg_label}")
            if result:
                all_results.append(result)

    print(f"\n\n{'='*70}")
    print(f"  7-DAY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':14s} {'Symbol':>8s} {'Trades':>7s} {'L':>3s} {'S':>3s} {'Inv':>5s} "
          f"{'Avg':>7s} {'Real':>8s} {'Unrl':>7s} {'Net':>9s} {'WR':>5s}")
    print(f"  {'-'*85}")
    for r in sorted(all_results, key=lambda x: -x["net"]):
        cfg = f"{r['cell_bps']}bps_{r['n_levels']}lvl"
        print(f"  {cfg:14s} {r['symbol']:>8s} {r['trades']:>7d} "
              f"{r['longs']:>3d} {r['shorts']:>3d} {r['inventory']:>+5.0f} "
              f"{r['avg_pnl']:>+7.2f} {r['realized']:>+8.1f} "
              f"{r['unrealized']:>+7.1f} {r['net']:>+9.1f} "
              f"{r['win_rate']:>5.0%}")

    # 30d for net-positive configs
    positive_configs = set()
    for r in all_results:
        if r["net"] > 0:
            positive_configs.add((r["cell_bps"], r["n_levels"]))

    if not positive_configs:
        print("\n  No profitable configs to validate on 30d.")
        return

    configs_30d = [(c, n, f"30d_{c}bps_{n}lvl") for c, n in positive_configs]

    print(f"\n\n{'#'*70}")
    print(f"  30-DAY VALIDATION ({len(configs_30d)} configs)")
    print(f"{'#'*70}")

    val_results = []
    for symbol in SYMBOLS:
        for cell_bps, n_lvl, cfg_label in configs_30d:
            result = run_grid_on_data(symbol, *PERIOD_30D, cell_bps, n_lvl, cfg_label)
            if result:
                val_results.append(result)

    print(f"\n\n{'='*70}")
    print(f"  30-DAY RESULTS")
    print(f"{'='*70}")
    print(f"  {'Config':14s} {'Symbol':>8s} {'Trades':>7s} {'L':>3s} {'S':>3s} {'Inv':>5s} "
          f"{'Avg':>7s} {'Real':>8s} {'Unrl':>7s} {'Net':>9s} {'WR':>5s}")
    print(f"  {'-'*85}")
    for r in sorted(val_results, key=lambda x: -x["net"]):
        cfg = f"{r['cell_bps']}bps_{r['n_levels']}lvl"
        print(f"  {cfg:14s} {r['symbol']:>8s} {r['trades']:>7d} "
              f"{r['longs']:>3d} {r['shorts']:>3d} {r['inventory']:>+5.0f} "
              f"{r['avg_pnl']:>+7.2f} {r['realized']:>+8.1f} "
              f"{r['unrealized']:>+7.1f} {r['net']:>+9.1f} "
              f"{r['win_rate']:>5.0%}")

    elapsed = time.time() - t_start
    print(f"\n✅ Complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
