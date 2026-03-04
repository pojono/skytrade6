#!/usr/bin/env python3
"""
Run backtest parameter sweep with unbuffered output.
Processes symbols in parallel within each config, configs sequentially.
"""

import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal, backtest_symbol, backtest_symbol_adaptive

def p(msg):
    print(msg, flush=True)

def main():
    symbols = list_common_symbols()
    p(f"Sweeping {len(symbols)} symbols")

    configs = []
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for hold in [3, 6, 12, 24]:
            for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
                configs.append(("fixed", fee_mode, fee_bps, threshold, hold))
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for fee_mode, fee_bps in [("taker", 10.0), ("maker", 4.0)]:
            configs.append(("adaptive", fee_mode, fee_bps, threshold, 0))

    p(f"Total configs: {len(configs)}")
    rows = []

    for ci, (mode, fee_mode, fee_bps, threshold, hold) in enumerate(configs):
        t0 = time.time()
        all_trades = []

        with ProcessPoolExecutor(max_workers=8) as pool:
            if mode == "fixed":
                futs = {pool.submit(backtest_symbol, s, fee_bps, threshold, hold): s
                        for s in symbols}
            else:
                futs = {pool.submit(backtest_symbol_adaptive, s, fee_bps, threshold): s
                        for s in symbols}
            for fut in as_completed(futs):
                try:
                    all_trades.extend(fut.result())
                except:
                    pass

        if not all_trades:
            p(f"  [{ci+1}/{len(configs)}] {mode:8s} fee={fee_mode:5s} thr={threshold:.1f} hold={hold:2d} → NO TRADES")
            continue

        nets = [t.net_ret_bps for t in all_trades]
        grosses = [t.gross_ret_bps for t in all_trades]
        n = len(nets)
        nsym = len(set(t.symbol for t in all_trades))
        wr = np.mean([r > 0 for r in nets])
        avg_net = np.mean(nets)
        avg_gross = np.mean(grosses)
        total_net = sum(nets)
        pf = (sum(r for r in nets if r > 0) /
              abs(sum(r for r in nets if r < 0))) if any(r < 0 for r in nets) else 999
        daily = total_net / 245
        avg_hold = np.mean([t.hold_bars for t in all_trades])
        elapsed = time.time() - t0

        rows.append({
            "mode": mode, "fee_mode": fee_mode, "fee_bps": fee_bps,
            "threshold": threshold, "hold": hold,
            "n_trades": n, "n_symbols": nsym,
            "avg_gross_bps": avg_gross, "avg_net_bps": avg_net,
            "win_rate": wr, "total_net_bps": total_net,
            "daily_net_bps": daily, "profit_factor": pf,
            "avg_hold_bars": avg_hold,
        })

        p(f"  [{ci+1}/{len(configs)}] {mode:8s} fee={fee_mode:5s} thr={threshold:.1f} "
          f"hold={hold:2d} → {n:5d} trades {nsym:3d}sym WR={wr:.1%} "
          f"avg_net={avg_net:+.1f}bps PF={pf:.2f} daily={daily:+.1f}bps ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv("backtest_sweep_results.csv", index=False)
    p(f"\nSaved to backtest_sweep_results.csv")

    # Print top configs
    p(f"\n{'='*110}")
    p(f"  TOP 15 CONFIGS BY DAILY NET BPS")
    p(f"{'='*110}")
    top = df.sort_values("daily_net_bps", ascending=False).head(15)
    for _, r in top.iterrows():
        p(f"  {r['mode']:8s} fee={r['fee_mode']:5s} thr={r['threshold']:.1f} "
          f"hold={r['hold']:2.0f} → {r['n_trades']:5.0f} trades WR={r['win_rate']:.1%} "
          f"avg_net={r['avg_net_bps']:+.1f}bps PF={r['profit_factor']:.2f} "
          f"daily={r['daily_net_bps']:+.1f}bps")

    p(f"\n--- BEST TAKER-ONLY CONFIGS ---")
    taker = df[df["fee_mode"] == "taker"].sort_values("daily_net_bps", ascending=False).head(10)
    for _, r in taker.iterrows():
        p(f"  {r['mode']:8s} thr={r['threshold']:.1f} hold={r['hold']:2.0f} → "
          f"{r['n_trades']:5.0f} trades WR={r['win_rate']:.1%} "
          f"avg_net={r['avg_net_bps']:+.1f}bps PF={r['profit_factor']:.2f} "
          f"daily={r['daily_net_bps']:+.1f}bps")

    p(f"\n--- BEST BY PROFIT FACTOR (taker, PF > 1.0) ---")
    taker_pf = df[(df["fee_mode"] == "taker") & (df["profit_factor"] > 1.0)].sort_values(
        "profit_factor", ascending=False).head(10)
    for _, r in taker_pf.iterrows():
        p(f"  {r['mode']:8s} thr={r['threshold']:.1f} hold={r['hold']:2.0f} → "
          f"{r['n_trades']:5.0f} trades WR={r['win_rate']:.1%} "
          f"avg_net={r['avg_net_bps']:+.1f}bps PF={r['profit_factor']:.2f} "
          f"daily={r['daily_net_bps']:+.1f}bps")

if __name__ == "__main__":
    main()
