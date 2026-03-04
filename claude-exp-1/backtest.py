#!/usr/bin/env python3
"""
Cross-exchange mean-reversion backtest.

KEY INSIGHT from discovery:
- When Bybit price > Binance price (price_div_bps > 0), future returns are NEGATIVE
- This is mean-reversion: the overextended exchange corrects back
- Effect is strongest when MULTIPLE signals align (price div + premium spread + OI div)
- We need extreme conditions to beat the 20bps fee hurdle

Strategy logic:
1. Compute composite signal from top cross-exchange features
2. When composite signal is EXTREME → enter trade in mean-reversion direction
3. Hold for N bars (15m-2h), exit
4. Focus on altcoins (bigger dislocations than BTC/ETH)

Two modes:
- TAKER: 10bps per leg (20bps RT). Need big moves.
- MAKER: 4bps per leg (8bps RT). More trades viable but need limit order assumptions.

Usage:
    python3 backtest.py [--symbols N] [--jobs N] [--fee-mode taker|maker]
"""

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from load_data import load_symbol, list_common_symbols
from features import compute_features

warnings.filterwarnings("ignore")


@dataclass
class TradeResult:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    gross_ret_bps: float
    fee_bps: float
    net_ret_bps: float
    signal_strength: float
    hold_bars: int


def compute_composite_signal(df: pd.DataFrame) -> pd.Series:
    """
    Composite mean-reversion signal from cross-exchange features.

    Positive signal → price is "too high" → expect drop → SHORT
    Negative signal → price is "too low" → expect rise → LONG

    Uses z-scored components so they're comparable across symbols.
    """
    components = []
    weights = []

    # 1. Price divergence z-score (strongest single signal)
    if "price_div_z72" in df.columns:
        components.append(df["price_div_z72"])
        weights.append(3.0)

    if "price_div_z288" in df.columns:
        components.append(df["price_div_z288"])
        weights.append(2.0)

    # 2. Premium spread z-score
    if "premium_z72" in df.columns:
        components.append(df["premium_z72"])
        weights.append(2.0)

    if "premium_z288" in df.columns:
        components.append(df["premium_z288"])
        weights.append(1.5)

    # 3. Price divergence raw (smoothed) — normalized
    if "price_div_ma12" in df.columns:
        raw = df["price_div_ma12"]
        mu = raw.rolling(288).mean()
        sigma = raw.rolling(288).std().replace(0, np.nan)
        components.append((raw - mu) / sigma)
        weights.append(1.5)

    # 4. OI divergence (inverted — high OI div predicts drop)
    if "oi_div" in df.columns:
        raw = df["oi_div"]
        mu = raw.rolling(288).mean()
        sigma = raw.rolling(288).std().replace(0, np.nan)
        components.append((raw - mu) / sigma)
        weights.append(1.0)

    # 5. Volume ratio z-score (where is flow going?)
    if "vol_ratio_z72" in df.columns:
        components.append(df["vol_ratio_z72"])
        weights.append(0.5)

    # 6. Return difference accumulation
    if "ret_diff_sum12" in df.columns:
        raw = df["ret_diff_sum12"]
        mu = raw.rolling(288).mean()
        sigma = raw.rolling(288).std().replace(0, np.nan)
        components.append((raw - mu) / sigma)
        weights.append(1.0)

    if not components:
        return pd.Series(0, index=df.index)

    # Weighted average
    total_weight = sum(weights)
    composite = sum(c * w for c, w in zip(components, weights)) / total_weight
    return composite


def backtest_symbol(sym: str, fee_per_leg_bps: float = 10.0,
                    entry_threshold: float = 2.0,
                    hold_bars: int = 6,
                    max_concurrent: int = 1,
                    cooldown_bars: int = 3) -> list[TradeResult]:
    """
    Backtest mean-reversion strategy on a single symbol.

    Parameters:
        fee_per_leg_bps: fee per trade leg in basis points (10 = taker, 4 = maker)
        entry_threshold: z-score threshold for entry (2.0 = top/bottom ~2.5%)
        hold_bars: number of 5m bars to hold (6 = 30min)
        max_concurrent: max concurrent positions
        cooldown_bars: minimum bars between trades
    """
    df = load_symbol(sym)
    if df.empty or len(df) < 5000:
        return []

    feat = compute_features(df)
    signal = compute_composite_signal(feat)
    feat["composite"] = signal

    # Need warmup period for rolling stats
    feat = feat.iloc[300:]
    feat = feat.dropna(subset=["composite"])

    if len(feat) < 3000:
        return []

    fee_rt_bps = fee_per_leg_bps * 2
    trades = []
    last_exit_idx = -cooldown_bars

    for i in range(len(feat) - hold_bars):
        # Cooldown check
        if i < last_exit_idx + cooldown_bars:
            continue

        sig = feat["composite"].iloc[i]

        # Entry conditions
        if abs(sig) < entry_threshold:
            continue

        # Direction: signal > threshold → SHORT (mean reversion down)
        #            signal < -threshold → LONG (mean reversion up)
        direction = -1 if sig > 0 else +1

        entry_time = feat.index[i]
        exit_idx = min(i + hold_bars, len(feat) - 1)
        exit_time = feat.index[exit_idx]

        # Use midpoint for entry/exit (average of both exchanges)
        entry_price = (feat["bb_close"].iloc[i] + feat["bn_close"].iloc[i]) / 2
        exit_price = (feat["bb_close"].iloc[exit_idx] + feat["bn_close"].iloc[exit_idx]) / 2

        gross_ret_bps = (exit_price / entry_price - 1) * 10000 * direction
        net_ret_bps = gross_ret_bps - fee_rt_bps

        trades.append(TradeResult(
            symbol=sym,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_ret_bps=gross_ret_bps,
            fee_bps=fee_rt_bps,
            net_ret_bps=net_ret_bps,
            signal_strength=abs(sig),
            hold_bars=hold_bars,
        ))
        last_exit_idx = exit_idx

    return trades


def backtest_symbol_adaptive(sym: str, fee_per_leg_bps: float = 10.0,
                              entry_threshold: float = 2.0,
                              cooldown_bars: int = 3) -> list[TradeResult]:
    """
    Adaptive hold backtest: exit when signal crosses zero OR max hold reached.

    This should capture more of the reversion move on strong signals
    and cut losses faster on weak ones.
    """
    df = load_symbol(sym)
    if df.empty or len(df) < 5000:
        return []

    feat = compute_features(df)
    signal = compute_composite_signal(feat)
    feat["composite"] = signal

    feat = feat.iloc[300:]
    feat = feat.dropna(subset=["composite"])

    if len(feat) < 3000:
        return []

    fee_rt_bps = fee_per_leg_bps * 2
    trades = []
    in_trade = False
    entry_idx = 0
    direction = 0
    sig_at_entry = 0
    max_hold = 24  # 2h max

    for i in range(len(feat)):
        sig = feat["composite"].iloc[i]

        if not in_trade:
            # Cooldown
            if trades and (i - entry_idx) < cooldown_bars:
                continue

            if abs(sig) >= entry_threshold:
                direction = -1 if sig > 0 else +1
                entry_idx = i
                sig_at_entry = sig
                in_trade = True
        else:
            # Exit conditions
            bars_held = i - entry_idx
            should_exit = False

            # 1. Signal crossed zero (mean reversion complete)
            if direction == +1 and sig >= 0:
                should_exit = True
            elif direction == -1 and sig <= 0:
                should_exit = True

            # 2. Max hold reached
            if bars_held >= max_hold:
                should_exit = True

            # 3. Signal reversed strongly (stop loss)
            if direction == +1 and sig > entry_threshold:
                should_exit = True
            elif direction == -1 and sig < -entry_threshold:
                should_exit = True

            if should_exit:
                entry_price = (feat["bb_close"].iloc[entry_idx] +
                               feat["bn_close"].iloc[entry_idx]) / 2
                exit_price = (feat["bb_close"].iloc[i] + feat["bn_close"].iloc[i]) / 2
                gross_ret_bps = (exit_price / entry_price - 1) * 10000 * direction
                net_ret_bps = gross_ret_bps - fee_rt_bps

                trades.append(TradeResult(
                    symbol=sym,
                    entry_time=feat.index[entry_idx],
                    exit_time=feat.index[i],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_ret_bps=gross_ret_bps,
                    fee_bps=fee_rt_bps,
                    net_ret_bps=net_ret_bps,
                    signal_strength=abs(sig_at_entry),
                    hold_bars=bars_held,
                ))
                in_trade = False

    return trades


def run_param_sweep(symbols: list[str], n_jobs: int = 8) -> pd.DataFrame:
    """Run backtest across multiple parameter combinations."""
    configs = []

    # Sweep entry thresholds and hold bars
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        for hold in [3, 6, 12, 24]:
            for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
                configs.append({
                    "threshold": threshold,
                    "hold": hold,
                    "fee_mode": fee_mode,
                    "fee_bps": fee_bps,
                    "mode": "fixed",
                })

    # Also test adaptive mode
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
            configs.append({
                "threshold": threshold,
                "hold": 0,
                "fee_mode": fee_mode,
                "fee_bps": fee_bps,
                "mode": "adaptive",
            })

    results = []
    total = len(configs)

    for ci, cfg in enumerate(configs):
        t0 = time.time()
        all_trades = []

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            if cfg["mode"] == "fixed":
                futures = {
                    pool.submit(backtest_symbol, sym, cfg["fee_bps"],
                                cfg["threshold"], cfg["hold"]): sym
                    for sym in symbols
                }
            else:
                futures = {
                    pool.submit(backtest_symbol_adaptive, sym, cfg["fee_bps"],
                                cfg["threshold"]): sym
                    for sym in symbols
                }

            for fut in as_completed(futures):
                try:
                    trades = fut.result()
                    all_trades.extend(trades)
                except Exception:
                    pass

        if not all_trades:
            continue

        net_rets = [t.net_ret_bps for t in all_trades]
        gross_rets = [t.gross_ret_bps for t in all_trades]

        n_trades = len(all_trades)
        n_symbols_traded = len(set(t.symbol for t in all_trades))
        avg_net = np.mean(net_rets)
        med_net = np.median(net_rets)
        win_rate = np.mean([r > 0 for r in net_rets])
        avg_gross = np.mean(gross_rets)
        total_net_bps = sum(net_rets)
        avg_hold = np.mean([t.hold_bars for t in all_trades])
        profit_factor = (sum(r for r in net_rets if r > 0) /
                         abs(sum(r for r in net_rets if r < 0)) if any(r < 0 for r in net_rets) else 999)

        # Per-day estimate (8 months ≈ 245 days)
        days = 245
        daily_net_bps = total_net_bps / days

        elapsed = time.time() - t0
        results.append({
            "mode": cfg["mode"],
            "fee_mode": cfg["fee_mode"],
            "threshold": cfg["threshold"],
            "hold": cfg["hold"],
            "n_trades": n_trades,
            "n_symbols": n_symbols_traded,
            "avg_gross_bps": avg_gross,
            "avg_net_bps": avg_net,
            "med_net_bps": med_net,
            "win_rate": win_rate,
            "total_net_bps": total_net_bps,
            "daily_net_bps": daily_net_bps,
            "profit_factor": profit_factor,
            "avg_hold_bars": avg_hold,
        })

        print(f"  [{ci+1}/{total}] {cfg['mode']:8s} fee={cfg['fee_mode']:5s} "
              f"thr={cfg['threshold']:.1f} hold={cfg['hold']:2d} → "
              f"{n_trades:5d} trades  WR={win_rate:.1%}  avg_net={avg_net:+.1f}bps  "
              f"PF={profit_factor:.2f}  daily={daily_net_bps:+.1f}bps  ({elapsed:.0f}s)")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", "-n", type=int, default=None)
    parser.add_argument("--jobs", "-j", type=int, default=8)
    parser.add_argument("--fee-mode", choices=["taker", "maker", "both"], default="both")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--hold", type=int, default=6)
    args = parser.parse_args()

    symbols = list_common_symbols()
    if args.symbols:
        symbols = symbols[:args.symbols]

    print(f"Backtesting {len(symbols)} symbols with {args.jobs} workers")

    if args.sweep:
        print(f"\n{'='*100}")
        print(f"  PARAMETER SWEEP")
        print(f"{'='*100}")
        sweep_df = run_param_sweep(symbols, args.jobs)

        sweep_df.to_csv("backtest_sweep_results.csv", index=False)
        print(f"\nSweep results saved to backtest_sweep_results.csv")

        # Best configs by net return
        print(f"\n{'='*100}")
        print(f"  TOP 10 CONFIGS BY DAILY NET BPS")
        print(f"{'='*100}")
        top = sweep_df.sort_values("daily_net_bps", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"  {row['mode']:8s} fee={row['fee_mode']:5s} "
                  f"thr={row['threshold']:.1f} hold={row['hold']:2.0f} → "
                  f"{row['n_trades']:5.0f} trades  WR={row['win_rate']:.1%}  "
                  f"avg_net={row['avg_net_bps']:+.1f}bps  PF={row['profit_factor']:.2f}  "
                  f"daily={row['daily_net_bps']:+.1f}bps")

        # Best configs by profit factor (for taker mode specifically)
        print(f"\n--- BEST TAKER CONFIGS (fee=10bps/leg) ---")
        taker = sweep_df[sweep_df["fee_mode"] == "taker"].sort_values(
            "profit_factor", ascending=False).head(10)
        for _, row in taker.iterrows():
            print(f"  {row['mode']:8s} thr={row['threshold']:.1f} "
                  f"hold={row['hold']:2.0f} → "
                  f"{row['n_trades']:5.0f} trades  WR={row['win_rate']:.1%}  "
                  f"avg_net={row['avg_net_bps']:+.1f}bps  PF={row['profit_factor']:.2f}  "
                  f"daily={row['daily_net_bps']:+.1f}bps")
        return

    # Single config run with detailed output
    print(f"\nRunning: threshold={args.threshold}, hold={args.hold}")

    for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
        if args.fee_mode != "both" and args.fee_mode != fee_mode:
            continue

        print(f"\n{'='*80}")
        print(f"  FEE MODE: {fee_mode.upper()} ({fee_bps} bps/leg = {fee_bps*2} bps RT)")
        print(f"{'='*80}")

        all_trades = []
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(backtest_symbol, sym, fee_bps,
                            args.threshold, args.hold): sym
                for sym in symbols
            }
            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    trades = fut.result()
                    all_trades.extend(trades)
                    if done % 20 == 0 or done == len(symbols):
                        print(f"  [{done}/{len(symbols)}] {len(all_trades)} trades "
                              f"so far ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"  {sym}: ERROR {e}")

        if not all_trades:
            print("  No trades generated!")
            continue

        # Analyze trades
        trade_df = pd.DataFrame([{
            "symbol": t.symbol, "entry_time": t.entry_time, "exit_time": t.exit_time,
            "direction": t.direction, "gross_bps": t.gross_ret_bps,
            "net_bps": t.net_ret_bps, "signal_strength": t.signal_strength,
            "hold_bars": t.hold_bars
        } for t in all_trades])

        trade_df.to_csv(f"trades_{fee_mode}.csv", index=False)

        n = len(trade_df)
        wr = (trade_df["net_bps"] > 0).mean()
        avg_net = trade_df["net_bps"].mean()
        avg_gross = trade_df["gross_bps"].mean()
        med_net = trade_df["net_bps"].median()
        total_net = trade_df["net_bps"].sum()
        avg_win = trade_df.loc[trade_df["net_bps"] > 0, "net_bps"].mean()
        avg_loss = trade_df.loc[trade_df["net_bps"] <= 0, "net_bps"].mean()
        n_syms = trade_df["symbol"].nunique()

        print(f"\n  RESULTS:")
        print(f"  Trades: {n} across {n_syms} symbols")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Avg gross: {avg_gross:+.2f} bps")
        print(f"  Avg net:   {avg_net:+.2f} bps")
        print(f"  Med net:   {med_net:+.2f} bps")
        print(f"  Avg win:   {avg_win:+.2f} bps")
        print(f"  Avg loss:  {avg_loss:+.2f} bps")
        print(f"  Total net: {total_net:+.0f} bps")

        # Per-symbol breakdown
        print(f"\n  --- TOP 10 SYMBOLS BY NET PNL ---")
        sym_stats = trade_df.groupby("symbol").agg(
            n_trades=("net_bps", "count"),
            wr=("net_bps", lambda x: (x > 0).mean()),
            avg_net=("net_bps", "mean"),
            total_net=("net_bps", "sum"),
        ).sort_values("total_net", ascending=False)

        for sym, row in sym_stats.head(10).iterrows():
            print(f"    {sym:20s} {row['n_trades']:4.0f} trades  "
                  f"WR={row['wr']:.1%}  avg={row['avg_net']:+.1f}bps  "
                  f"total={row['total_net']:+.0f}bps")

        # Time-based analysis: monthly
        trade_df["month"] = pd.to_datetime(trade_df["entry_time"]).dt.to_period("M")
        monthly = trade_df.groupby("month").agg(
            n=("net_bps", "count"),
            wr=("net_bps", lambda x: (x > 0).mean()),
            avg_net=("net_bps", "mean"),
            total=("net_bps", "sum"),
        )
        print(f"\n  --- MONTHLY PERFORMANCE ---")
        for month, row in monthly.iterrows():
            print(f"    {str(month):10s} {row['n']:5.0f} trades  "
                  f"WR={row['wr']:.1%}  avg={row['avg_net']:+.1f}bps  "
                  f"total={row['total']:+.0f}bps")


if __name__ == "__main__":
    main()
