#!/usr/bin/env python3
"""
Unified experiment runner.

Usage:
  python run_experiment.py --suite signal    --exchange bybit_futures --symbol BTCUSDT --start 2025-11-01 --end 2026-01-31
  python run_experiment.py --suite novel     --exchange bybit_futures --symbol ETHUSDT --start 2025-11-01 --end 2026-01-31
  python run_experiment.py --suite grid_ohlcv --exchange bybit_futures --symbol SOLUSDT --start 2025-11-01 --end 2026-01-31
  python run_experiment.py --suite grid_tick  --exchange bybit_futures --symbol BTCUSDT --start 2025-11-01 --end 2026-01-31

  # Run all suites on all symbols:
  python run_experiment.py --suite all --exchange bybit_futures --symbol all --start 2025-11-01 --end 2026-01-31

Output is written to results/<suite>_<symbol>_<start>_<end>.txt
"""

import argparse
import sys
import time
import io
import os
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import psutil


RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_SUITES = ["signal", "novel", "grid_ohlcv", "grid_tick"]


class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def run_signal_suite(exchange, symbol, start, end):
    """Run E01-E15 signal experiments."""
    from experiments import (
        load_features, add_derived_features, EXPERIMENTS,
        backtest_signal, zscore_signal, rank_composite, ROUND_TRIP_FEE_BPS
    )

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    print(f"\n{'='*70}")
    print(f"  SIGNAL EXPERIMENTS (E01-E15)")
    print(f"  {symbol} on {exchange} | {start} → {end} ({days} days)")
    print(f"  Fee: {ROUND_TRIP_FEE_BPS} bps RT")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"{'='*70}")

    # Override SOURCE for the exchange
    import experiments
    experiments.SOURCE = exchange

    print(f"\n  Loading features...", flush=True)
    df = load_features(symbol, start, end)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, price {df['close'].min():.2f}–{df['close'].max():.2f}")
    print(f"  Adding derived features...", flush=True)
    df = add_derived_features(df)

    all_results = []

    for exp_name, exp_desc, exp_func in EXPERIMENTS:
        print(f"\n  📋 {exp_name}: {exp_desc}", flush=True)
        df_copy = df.copy()

        try:
            results = exp_func(df_copy)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        best_avg = -999
        best_cfg = None

        for thresh, hl, pnls in results:
            if len(pnls) < 5:
                continue
            avg = pnls.mean()
            total = pnls.sum()
            wr = (pnls > 0).mean()
            sharpe = avg / max(pnls.std(), 1e-10) if len(pnls) > 1 else 0

            if avg > best_avg:
                best_avg = avg
                best_cfg = (thresh, hl, len(pnls), avg, total, wr, sharpe)

            # Log ALL configs, not just best
            marker = "+" if avg > 0 else "-"
            print(f"    {marker} thresh={thresh}, hold={hl}, "
                  f"trades={len(pnls)}, avg={avg:+.2f}, total={total:+.1f}, "
                  f"WR={wr:.0%}, sharpe={sharpe:.2f}")

        if best_cfg:
            thresh, hl, n_trades, avg, total, wr, sharpe = best_cfg
            marker = "✅" if avg > 0 and n_trades >= 10 else "  "
            print(f"    {marker} BEST: thresh={thresh}, hold={hl}, "
                  f"trades={n_trades}, avg={avg:+.2f} bps, total={total:+.1f}, "
                  f"WR={wr:.0%}, sharpe={sharpe:.2f}")

            all_results.append({
                "experiment": exp_name, "symbol": symbol,
                "threshold": thresh, "holding": hl, "n_trades": n_trades,
                "avg_pnl_bps": avg, "total_pnl_bps": total,
                "win_rate": wr, "sharpe": sharpe,
            })

    return all_results


def run_novel_suite(exchange, symbol, start, end):
    """Run N01-N16 novel experiments."""
    from experiments_v2 import (
        load_features, add_derived, EXPERIMENTS,
        backtest, zscore, rank_composite, ROUND_TRIP_FEE_BPS
    )

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    print(f"\n{'='*70}")
    print(f"  NOVEL EXPERIMENTS (N01-N16)")
    print(f"  {symbol} on {exchange} | {start} → {end} ({days} days)")
    print(f"  Fee: {ROUND_TRIP_FEE_BPS} bps RT")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"{'='*70}")

    import experiments_v2
    experiments_v2.SOURCE = exchange

    print(f"\n  Loading & computing features...", flush=True)
    df = load_features(symbol, start, end)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, adding derived features...", flush=True)
    df = add_derived(df)

    all_results = []

    for exp_name, exp_desc, exp_func in EXPERIMENTS:
        print(f"\n  📋 {exp_name}: {exp_desc}", flush=True)
        df_copy = df.copy()

        try:
            results = exp_func(df_copy)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        best_avg = -999
        best_cfg = None

        for thresh, hl, pnls in results:
            if len(pnls) < 5:
                continue
            avg = pnls.mean()
            total = pnls.sum()
            wr = (pnls > 0).mean()
            sharpe = avg / max(pnls.std(), 1e-10) if len(pnls) > 1 else 0

            if avg > best_avg:
                best_avg = avg
                best_cfg = (thresh, hl, len(pnls), avg, total, wr, sharpe)

            marker = "+" if avg > 0 else "-"
            print(f"    {marker} thresh={thresh}, hold={hl}, "
                  f"trades={len(pnls)}, avg={avg:+.2f}, total={total:+.1f}, "
                  f"WR={wr:.0%}, sharpe={sharpe:.2f}")

        if best_cfg:
            thresh, hl, n_trades, avg, total, wr, sharpe = best_cfg
            marker = "✅" if avg > 0 and n_trades >= 10 else "  "
            print(f"    {marker} BEST: thresh={thresh}, hold={hl}, "
                  f"trades={n_trades}, avg={avg:+.2f} bps, total={total:+.1f}, "
                  f"WR={wr:.0%}, sharpe={sharpe:.2f}")

            all_results.append({
                "experiment": exp_name, "symbol": symbol,
                "threshold": thresh, "holding": hl, "n_trades": n_trades,
                "avg_pnl_bps": avg, "total_pnl_bps": total,
                "win_rate": wr, "sharpe": sharpe,
            })

    return all_results


def run_grid_ohlcv_suite(exchange, symbol, start, end):
    """Run G01-G10 OHLCV grid experiments."""
    from experiments_grid import (
        load_features, GridBacktester, GRID_EXPERIMENTS,
        GRID_RT_FEE_BPS, TREND_RT_FEE_BPS
    )

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    print(f"\n{'='*70}")
    print(f"  GRID OHLCV EXPERIMENTS (G01-G10)")
    print(f"  {symbol} on {exchange} | {start} → {end} ({days} days)")
    print(f"  Grid fees: {GRID_RT_FEE_BPS} bps RT | Trend fees: {TREND_RT_FEE_BPS} bps RT")
    print(f"  Experiments: {len(GRID_EXPERIMENTS)}")
    print(f"{'='*70}")

    import experiments_grid
    experiments_grid.SOURCE = exchange

    print(f"\n  Loading features...", flush=True)
    df = load_features(symbol, start, end)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, price {df['close'].min():.2f}–{df['close'].max():.2f}")

    all_results = []

    for exp_name, exp_desc, params in GRID_EXPERIMENTS:
        print(f"\n  📋 {exp_name}: {exp_desc}", flush=True)

        try:
            grid = GridBacktester(**params)
            trades_df, summary = grid.run(df)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        if not summary:
            print(f"    — No trades")
            continue

        nt = summary["total_trades"]
        avg = summary["avg_pnl_bps"]
        total = summary["total_pnl_bps"]
        wr = summary["win_rate"]
        gt = summary["grid_trades"]
        tt = summary["trend_trades"]
        dd = summary["max_dd"]

        marker = "✅" if avg > 0 and nt >= 20 else "  "
        print(f"    {marker} Trades={nt} (grid={gt}, trend={tt}), "
              f"Avg={avg:+.2f} bps, Total={total:+.1f}, WR={wr:.0%}, DD={dd:+.1f}")
        if gt > 0:
            print(f"       Grid: avg={summary['grid_avg_pnl']:+.2f} bps, WR={summary['grid_wr']:.0%}")
        if tt > 0:
            print(f"       Trend: avg={summary['trend_avg_pnl']:+.2f} bps, WR={summary['trend_wr']:.0%}")

        all_results.append({
            "experiment": exp_name, "symbol": symbol,
            "n_trades": nt, "grid_trades": gt, "trend_trades": tt,
            "avg_pnl_bps": avg, "total_pnl_bps": total,
            "win_rate": wr, "max_dd": dd,
        })

    return all_results


def run_grid_tick_suite(exchange, symbol, start, end):
    """Run tick-level grid experiments."""
    from grid_backtest import Grid, MAKER_FEE_BPS
    import psutil

    configs = [
        (20, 3, "20bps_3lvl"),
        (30, 3, "30bps_3lvl"),
        (50, 2, "50bps_2lvl"),
        (50, 3, "50bps_3lvl"),
    ]

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    print(f"\n{'='*70}")
    print(f"  TICK-LEVEL GRID EXPERIMENTS")
    print(f"  {symbol} on {exchange} | {start} → {end} ({days} days)")
    print(f"  Fee: {MAKER_FEE_BPS} bps maker per fill ({2*MAKER_FEE_BPS} bps RT)")
    print(f"  Configs: {len(configs)}")
    print(f"{'='*70}")

    PARQUET_DIR = Path("./parquet")
    dates = pd.date_range(start, end)

    # Get center price from first available day
    center = None
    for date in dates:
        ds = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
        if path.exists():
            first_trades = pd.read_parquet(path)
            center = first_trades["price"].values[0]
            del first_trades
            break

    if center is None:
        print(f"  ❌ No data!")
        return []

    all_results = []

    for cell_bps, n_levels, cfg_label in configs:
        cell = center * cell_bps / 10000
        print(f"\n  📋 {cfg_label}: cell={cell_bps}bps ({cell:.2f}), levels={n_levels}/side")
        print(f"  Center: {center:.2f}")

        grid = Grid(center_price=center, cell_width=cell, n_levels_per_side=n_levels)
        for l in grid.levels:
            print(f"    {l.price:.2f} [{l.state.value}]")

        t0 = time.time()
        total_ticks = 0

        for i, date in enumerate(dates, 1):
            ds = date.strftime("%Y-%m-%d")
            path = PARQUET_DIR / symbol / "trades" / exchange / f"{ds}.parquet"
            if not path.exists():
                continue

            trades = pd.read_parquet(path)
            prices = trades["price"].values
            n = len(prices)
            total_ticks += n
            del trades

            day_trades_before = len(grid.completed_trades)

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

        # Summary
        n_trades = len(grid.completed_trades)
        last_price = prices[-1] if total_ticks > 0 else center
        realized = sum(t["pnl_bps"] for t in grid.completed_trades)
        unrealized = grid.get_unrealized_pnl(last_price)
        longs, shorts = grid.get_open_count()
        net = realized + unrealized

        if n_trades > 0:
            df_trades = pd.DataFrame(grid.completed_trades)
            avg = df_trades["pnl_bps"].mean()
            wr = (df_trades["pnl_bps"] > 0).mean()
        else:
            avg = wr = 0

        elapsed = time.time() - t0
        print(f"\n  SUMMARY ({cfg_label}):")
        print(f"    Completed trades: {n_trades}")
        print(f"    Open: {longs}L {shorts}S (inv={grid.inventory:.0f})")
        print(f"    Avg PnL: {avg:+.2f} bps, WR: {wr:.0%}")
        print(f"    Realized: {realized:+.1f}, Unrealized: {unrealized:+.1f}, Net: {net:+.1f} bps")
        print(f"    Ticks: {total_ticks:,}, Time: {elapsed:.0f}s")

        all_results.append({
            "experiment": f"grid_tick_{cfg_label}", "symbol": symbol,
            "n_trades": n_trades, "longs": longs, "shorts": shorts,
            "inventory": grid.inventory,
            "avg_pnl_bps": avg, "realized_bps": realized,
            "unrealized_bps": unrealized, "net_bps": net,
            "win_rate": wr, "total_ticks": total_ticks,
        })

    return all_results


def print_summary_table(results, suite):
    """Print a summary table of all results."""
    if not results:
        print(f"\n  No results for {suite}.")
        return

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {suite.upper()}")
    print(f"{'='*70}")

    if suite == "grid_tick":
        print(f"  {'Experiment':20s} {'Symbol':>8s} {'Trades':>7s} {'Inv':>5s} "
              f"{'Avg':>7s} {'Real':>8s} {'Unrl':>7s} {'Net':>9s} {'WR':>5s}")
        print(f"  {'-'*80}")
        for r in sorted(results, key=lambda x: -x.get("net_bps", 0)):
            print(f"  {r['experiment']:20s} {r['symbol']:>8s} {r['n_trades']:>7d} "
                  f"{r.get('inventory',0):>+5.0f} {r['avg_pnl_bps']:>+7.2f} "
                  f"{r.get('realized_bps',0):>+8.1f} {r.get('unrealized_bps',0):>+7.1f} "
                  f"{r.get('net_bps',0):>+9.1f} {r['win_rate']:>5.0%}")
    elif suite == "grid_ohlcv":
        print(f"  {'Experiment':25s} {'Symbol':>8s} {'Trades':>7s} "
              f"{'Avg':>8s} {'Total':>9s} {'WR':>5s} {'DD':>8s}")
        print(f"  {'-'*75}")
        for r in sorted(results, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {r['experiment']:25s} {r['symbol']:>8s} {r['n_trades']:>7d} "
                  f"{r['avg_pnl_bps']:>+8.2f} {r['total_pnl_bps']:>+9.1f} "
                  f"{r['win_rate']:>5.0%} {r.get('max_dd',0):>+8.1f}")
    else:
        print(f"  {'Experiment':35s} {'Symbol':>8s} {'Thresh':>6s} {'Hold':>6s} "
              f"{'Trades':>7s} {'Avg':>8s} {'Total':>9s} {'WR':>5s} {'Sharpe':>7s}")
        print(f"  {'-'*95}")
        for r in sorted(results, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {r['experiment']:35s} {r['symbol']:>8s} {r.get('threshold',''):>6} "
                  f"{r.get('holding',''):>6s} {r['n_trades']:>7d} "
                  f"{r['avg_pnl_bps']:>+8.2f} {r.get('total_pnl_bps',0):>+9.1f} "
                  f"{r['win_rate']:>5.0%} {r.get('sharpe',0):>+7.2f}")

    # Count winners
    winners = [r for r in results if r["avg_pnl_bps"] > 0 and r.get("n_trades", 0) >= 10]
    print(f"\n  Winners (avg>0, trades>=10): {len(winners)}/{len(results)}")


SUITE_RUNNERS = {
    "signal": run_signal_suite,
    "novel": run_novel_suite,
    "grid_ohlcv": run_grid_ohlcv_suite,
    "grid_tick": run_grid_tick_suite,
}


def main():
    parser = argparse.ArgumentParser(description="Run trading experiments")
    parser.add_argument("--suite", required=True,
                        choices=ALL_SUITES + ["all"],
                        help="Experiment suite to run")
    parser.add_argument("--exchange", default="bybit_futures",
                        help="Exchange data source (default: bybit_futures)")
    parser.add_argument("--symbol", required=True,
                        help="Trading symbol (e.g. BTCUSDT) or 'all'")
    parser.add_argument("--start", required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    symbols = ALL_SYMBOLS if args.symbol == "all" else [args.symbol]
    suites = ALL_SUITES if args.suite == "all" else [args.suite]

    t_global = time.time()

    for suite in suites:
        runner = SUITE_RUNNERS[suite]
        all_results = []

        for symbol in symbols:
            outfile = RESULTS_DIR / f"{suite}_{symbol}_{args.start}_{args.end}.txt"
            print(f"\n>>> Running {suite} on {symbol} ({args.start} → {args.end})")
            print(f">>> Output: {outfile}")

            tee = TeeWriter(str(outfile))
            old_stdout = sys.stdout
            sys.stdout = tee

            try:
                t0 = time.time()
                print(f"# Experiment: {suite}")
                print(f"# Symbol: {symbol}")
                print(f"# Exchange: {args.exchange}")
                print(f"# Period: {args.start} → {args.end}")
                print(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
                print()

                results = runner(args.exchange, symbol, args.start, args.end)
                all_results.extend(results)

                elapsed = time.time() - t0
                print(f"\n# Completed in {elapsed:.0f}s")
            except Exception as e:
                print(f"\n❌ FATAL ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                tee.close()

            print(f"<<< Done {suite}/{symbol} → {outfile}")

        # Print cross-symbol summary to a combined file
        if len(symbols) > 1 and all_results:
            summary_file = RESULTS_DIR / f"{suite}_ALL_{args.start}_{args.end}.txt"
            tee = TeeWriter(str(summary_file))
            old_stdout = sys.stdout
            sys.stdout = tee
            try:
                print(f"# Combined Summary: {suite}")
                print(f"# Exchange: {args.exchange}")
                print(f"# Period: {args.start} → {args.end}")
                print(f"# Symbols: {', '.join(symbols)}")
                print()
                print_summary_table(all_results, suite)
            finally:
                sys.stdout = old_stdout
                tee.close()
            print(f"<<< Summary → {summary_file}")

    elapsed = time.time() - t_global
    print(f"\n✅ All done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
