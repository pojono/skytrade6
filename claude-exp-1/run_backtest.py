#!/usr/bin/env python3
"""
Efficient backtest: load each symbol ONCE, test multiple configs inline.
Writes progress to stderr (unbuffered), results to stdout.
"""

import sys, time, os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def process_symbol(sym):
    """Load symbol, compute signal, run all configs, return results dict."""
    try:
        df = load_symbol(sym)
        if df.empty or len(df) < 5000:
            return sym, None

        feat = compute_features(df)
        signal = compute_composite_signal(feat)
        feat["composite"] = signal
        feat = feat.iloc[300:].dropna(subset=["composite"])

        if len(feat) < 3000:
            return sym, None

        mid = (feat["bb_close"].values + feat["bn_close"].values) / 2
        sig = feat["composite"].values
        n = len(feat)
        idx = feat.index

        results = {}

        for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
            for hold in [3, 6, 12, 24]:
                cooldown = 3
                key = f"fixed_thr{threshold}_h{hold}"
                trades = []
                last_exit = -cooldown

                for i in range(n - hold):
                    if i < last_exit + cooldown:
                        continue
                    if abs(sig[i]) < threshold:
                        continue

                    direction = -1 if sig[i] > 0 else 1
                    exit_i = min(i + hold, n - 1)
                    gross_bps = (mid[exit_i] / mid[i] - 1) * 10000 * direction
                    trades.append((gross_bps, abs(sig[i]), hold, direction,
                                   str(idx[i]), str(idx[exit_i])))
                    last_exit = exit_i

                if trades:
                    results[key] = trades

            # Adaptive mode
            key = f"adaptive_thr{threshold}"
            trades = []
            in_trade = False
            entry_i = 0
            direction = 0
            sig_entry = 0
            max_hold = 24

            for i in range(n):
                if not in_trade:
                    if trades and (i - entry_i) < 3:
                        continue
                    if abs(sig[i]) >= threshold:
                        direction = -1 if sig[i] > 0 else 1
                        entry_i = i
                        sig_entry = sig[i]
                        in_trade = True
                else:
                    bars = i - entry_i
                    exit_now = False
                    if direction == 1 and sig[i] >= 0:
                        exit_now = True
                    elif direction == -1 and sig[i] <= 0:
                        exit_now = True
                    if bars >= max_hold:
                        exit_now = True
                    if direction == 1 and sig[i] > threshold:
                        exit_now = True
                    elif direction == -1 and sig[i] < -threshold:
                        exit_now = True

                    if exit_now:
                        gross_bps = (mid[i] / mid[entry_i] - 1) * 10000 * direction
                        trades.append((gross_bps, abs(sig_entry), bars, direction,
                                       str(idx[entry_i]), str(idx[i])))
                        in_trade = False

            if trades:
                results[key] = trades

        return sym, results
    except Exception as e:
        return sym, None


def main():
    symbols = list_common_symbols()
    log(f"Processing {len(symbols)} symbols...")
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(process_symbol, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym, res = fut.result()
            if res:
                all_results[sym] = res
            if done % 10 == 0 or done == len(symbols):
                log(f"  [{done}/{len(symbols)}] {len(all_results)} symbols with trades ({time.time()-t0:.0f}s)")

    log(f"\nLoaded {len(all_results)} symbols in {time.time()-t0:.0f}s")

    # Aggregate results per config
    config_keys = set()
    for sym_res in all_results.values():
        config_keys.update(sym_res.keys())
    config_keys = sorted(config_keys)

    rows = []
    for key in config_keys:
        all_trades = []
        syms_traded = set()
        for sym, sym_res in all_results.items():
            if key in sym_res:
                all_trades.extend(sym_res[key])
                syms_traded.add(sym)

        if not all_trades:
            continue

        grosses = [t[0] for t in all_trades]
        n = len(grosses)
        nsym = len(syms_traded)
        avg_gross = np.mean(grosses)

        for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
            fee_rt = fee_bps * 2
            nets = [g - fee_rt for g in grosses]
            avg_net = np.mean(nets)
            wr = np.mean([r > 0 for r in nets])
            total_net = sum(nets)
            wins = sum(r for r in nets if r > 0)
            losses = abs(sum(r for r in nets if r < 0))
            pf = wins / losses if losses > 0 else 999
            daily = total_net / 245
            avg_hold = np.mean([t[2] for t in all_trades])

            rows.append({
                "config": key, "fee_mode": fee_mode, "fee_bps": fee_bps,
                "n_trades": n, "n_symbols": nsym,
                "avg_gross_bps": avg_gross, "avg_net_bps": avg_net,
                "win_rate": wr, "total_net_bps": total_net,
                "daily_net_bps": daily, "profit_factor": pf,
                "avg_hold_bars": avg_hold,
            })

    df = pd.DataFrame(rows)
    df.to_csv("backtest_sweep_results.csv", index=False)

    # Print results
    log(f"\n{'='*120}")
    log(f"  ALL CONFIGS — TAKER FEES (10bps/leg = 20bps RT)")
    log(f"{'='*120}")
    log(f"{'Config':30s} {'Trades':>6} {'Syms':>4} {'AvgGross':>9} {'AvgNet':>8} "
        f"{'WR':>6} {'PF':>6} {'Daily':>8} {'AvgHold':>7}")
    log("-" * 120)

    taker = df[df["fee_mode"] == "taker"].sort_values("daily_net_bps", ascending=False)
    for _, r in taker.iterrows():
        log(f"{r['config']:30s} {r['n_trades']:6.0f} {r['n_symbols']:4.0f} "
            f"{r['avg_gross_bps']:+9.2f} {r['avg_net_bps']:+8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['daily_net_bps']:+8.1f} {r['avg_hold_bars']:7.1f}")

    log(f"\n{'='*120}")
    log(f"  ALL CONFIGS — MAKER FEES (4bps/leg = 8bps RT)")
    log(f"{'='*120}")
    log(f"{'Config':30s} {'Trades':>6} {'Syms':>4} {'AvgGross':>9} {'AvgNet':>8} "
        f"{'WR':>6} {'PF':>6} {'Daily':>8} {'AvgHold':>7}")
    log("-" * 120)

    maker = df[df["fee_mode"] == "maker"].sort_values("daily_net_bps", ascending=False)
    for _, r in maker.iterrows():
        log(f"{r['config']:30s} {r['n_trades']:6.0f} {r['n_symbols']:4.0f} "
            f"{r['avg_gross_bps']:+9.2f} {r['avg_net_bps']:+8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['daily_net_bps']:+8.1f} {r['avg_hold_bars']:7.1f}")

    # Save detailed trades for the best config
    best_key = taker.iloc[0]["config"]
    log(f"\n--- Saving detailed trades for best taker config: {best_key} ---")
    best_trades = []
    for sym, sym_res in all_results.items():
        if best_key in sym_res:
            for t in sym_res[best_key]:
                best_trades.append({
                    "symbol": sym, "gross_bps": t[0], "signal_strength": t[1],
                    "hold_bars": t[2], "direction": t[3],
                    "entry_time": t[4], "exit_time": t[5],
                    "net_bps_taker": t[0] - 20, "net_bps_maker": t[0] - 8,
                })
    trade_df = pd.DataFrame(best_trades)
    trade_df.to_csv("best_config_trades.csv", index=False)

    # Per-symbol breakdown for best config
    sym_stats = trade_df.groupby("symbol").agg(
        n=("net_bps_taker", "count"),
        wr=("net_bps_taker", lambda x: (x > 0).mean()),
        avg_net=("net_bps_taker", "mean"),
        total=("net_bps_taker", "sum"),
    ).sort_values("total", ascending=False)

    log(f"\n--- TOP 20 SYMBOLS (best taker config: {best_key}) ---")
    for sym, r in sym_stats.head(20).iterrows():
        log(f"  {sym:20s} {r['n']:4.0f} trades WR={r['wr']:.1%} "
            f"avg={r['avg_net']:+.1f}bps total={r['total']:+.0f}bps")

    log(f"\n--- BOTTOM 10 SYMBOLS ---")
    for sym, r in sym_stats.tail(10).iterrows():
        log(f"  {sym:20s} {r['n']:4.0f} trades WR={r['wr']:.1%} "
            f"avg={r['avg_net']:+.1f}bps total={r['total']:+.0f}bps")

    # Monthly performance
    trade_df["month"] = pd.to_datetime(trade_df["entry_time"]).dt.to_period("M")
    monthly = trade_df.groupby("month").agg(
        n=("net_bps_taker", "count"),
        wr=("net_bps_taker", lambda x: (x > 0).mean()),
        avg=("net_bps_taker", "mean"),
        total=("net_bps_taker", "sum"),
    )
    log(f"\n--- MONTHLY PERFORMANCE (best taker config) ---")
    for m, r in monthly.iterrows():
        log(f"  {str(m):10s} {r['n']:5.0f} trades WR={r['wr']:.1%} "
            f"avg={r['avg']:+.1f}bps total={r['total']:+.0f}bps")


if __name__ == "__main__":
    main()
