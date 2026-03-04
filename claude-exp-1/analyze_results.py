#!/usr/bin/env python3
"""
Deep analysis of backtest results:
1. Per-symbol breakdown
2. Monthly consistency
3. Walk-forward out-of-sample validation
4. Trade distribution analysis
5. Signal strength vs returns
6. Drawdown analysis
"""

import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def process_symbol_wfo(sym):
    """Walk-forward: train signal on first 4 months, test on last 4 months."""
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

        # Split: first half = in-sample, second half = out-of-sample
        split_idx = n // 2
        split_time = idx[split_idx]

        results = {"is_trades": [], "oos_trades": [], "split_time": str(split_time)}

        for threshold in [2.5, 3.0, 3.5, 4.0]:
            # Adaptive backtest on both halves
            for half_name, start, end in [("is", 0, split_idx), ("oos", split_idx, n)]:
                trades = []
                in_trade = False
                entry_i = 0
                direction = 0
                sig_entry = 0
                max_hold = 24

                for i in range(start, end):
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
                            trades.append({
                                "symbol": sym, "threshold": threshold,
                                "gross_bps": gross_bps, "signal_strength": abs(sig_entry),
                                "hold_bars": bars, "direction": direction,
                                "entry_time": str(idx[entry_i]),
                                "exit_time": str(idx[i]),
                                "half": half_name,
                            })
                            in_trade = False

                if half_name == "is":
                    results["is_trades"].extend(trades)
                else:
                    results["oos_trades"].extend(trades)

        return sym, results
    except Exception as e:
        return sym, None


def main():
    symbols = list_common_symbols()
    log(f"Walk-forward analysis on {len(symbols)} symbols...")
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(process_symbol_wfo, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym, res = fut.result()
            if res:
                all_results[sym] = res
            if done % 20 == 0 or done == len(symbols):
                log(f"  [{done}/{len(symbols)}] ({time.time()-t0:.0f}s)")

    log(f"\nProcessed {len(all_results)} symbols in {time.time()-t0:.0f}s")

    # Collect all trades
    is_trades = []
    oos_trades = []
    for sym, res in all_results.items():
        is_trades.extend(res["is_trades"])
        oos_trades.extend(res["oos_trades"])

    is_df = pd.DataFrame(is_trades)
    oos_df = pd.DataFrame(oos_trades)

    log(f"\nTotal IS trades: {len(is_df)}, OOS trades: {len(oos_df)}")

    # ========================================================================
    # 1. Walk-forward results by threshold
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  WALK-FORWARD VALIDATION: IN-SAMPLE vs OUT-OF-SAMPLE")
    log(f"{'='*120}")
    log(f"{'Threshold':>10} {'Half':>5} {'Trades':>7} {'Syms':>5} "
        f"{'AvgGross':>10} {'Taker Net':>10} {'Maker Net':>10} "
        f"{'WR(tk)':>8} {'WR(mk)':>8} {'PF(tk)':>8} {'PF(mk)':>8}")
    log("-" * 120)

    for thr in [2.5, 3.0, 3.5, 4.0]:
        for half, df_half in [("IS", is_df), ("OOS", oos_df)]:
            subset = df_half[df_half["threshold"] == thr]
            if len(subset) == 0:
                continue

            g = subset["gross_bps"]
            n = len(g)
            nsym = subset["symbol"].nunique()
            avg_gross = g.mean()

            for fee_label, fee_rt in [("taker", 20), ("maker", 8)]:
                nets = g - fee_rt
                wr = (nets > 0).mean()
                wins = nets[nets > 0].sum()
                losses = abs(nets[nets < 0].sum())
                pf = wins / losses if losses > 0 else 999

                if fee_label == "taker":
                    tk_net = nets.mean()
                    tk_wr = wr
                    tk_pf = pf
                else:
                    mk_net = nets.mean()
                    mk_wr = wr
                    mk_pf = pf

            log(f"{thr:10.1f} {half:>5} {n:7d} {nsym:5d} "
                f"{avg_gross:+10.2f} {tk_net:+10.2f} {mk_net:+10.2f} "
                f"{tk_wr:8.1%} {mk_wr:8.1%} {tk_pf:8.2f} {mk_pf:8.2f}")

    # ========================================================================
    # 2. Monthly consistency for best config (thr=3.5, adaptive, taker)
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  MONTHLY CONSISTENCY — adaptive thr=3.5 (ALL data, taker fees)")
    log(f"{'='*120}")

    # Load the best_config_trades.csv for monthly analysis
    try:
        best = pd.read_csv("best_config_trades.csv")
        best["month"] = pd.to_datetime(best["entry_time"]).dt.to_period("M")
        monthly = best.groupby("month").agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            avg_gross=("gross_bps", "mean"),
            avg_net=("net_bps_taker", "mean"),
            total_net=("net_bps_taker", "sum"),
            avg_signal=("signal_strength", "mean"),
            avg_hold=("hold_bars", "mean"),
        )
        for m, r in monthly.iterrows():
            status = "✓" if r["total_net"] > 0 else "✗"
            log(f"  {str(m):10s} {status} {r['n']:4.0f} trades  "
                f"WR={r['wr']:.1%}  avg_gross={r['avg_gross']:+.0f}  "
                f"avg_net={r['avg_net']:+.0f}bps  total={r['total_net']:+.0f}bps  "
                f"sig={r['avg_signal']:.2f}  hold={r['avg_hold']:.1f}bars")

        pos_months = (monthly["total_net"] > 0).sum()
        total_months = len(monthly)
        log(f"\n  Positive months: {pos_months}/{total_months}")
    except Exception as e:
        log(f"  Error loading trades: {e}")

    # ========================================================================
    # 3. Per-symbol analysis
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  PER-SYMBOL BREAKDOWN — adaptive thr=3.5 (taker)")
    log(f"{'='*120}")

    try:
        sym_stats = best.groupby("symbol").agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            avg_gross=("gross_bps", "mean"),
            avg_net=("net_bps_taker", "mean"),
            total=("net_bps_taker", "sum"),
            avg_signal=("signal_strength", "mean"),
        ).sort_values("total", ascending=False)

        profitable_syms = (sym_stats["total"] > 0).sum()
        log(f"  Profitable symbols: {profitable_syms}/{len(sym_stats)}")
        log(f"\n  TOP 15 SYMBOLS:")
        for sym, r in sym_stats.head(15).iterrows():
            log(f"    {sym:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} "
                f"avg_gross={r['avg_gross']:+.0f} avg_net={r['avg_net']:+.0f} "
                f"total={r['total']:+.0f}bps sig={r['avg_signal']:.2f}")

        log(f"\n  BOTTOM 10 SYMBOLS:")
        for sym, r in sym_stats.tail(10).iterrows():
            log(f"    {sym:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} "
                f"avg_gross={r['avg_gross']:+.0f} avg_net={r['avg_net']:+.0f} "
                f"total={r['total']:+.0f}bps sig={r['avg_signal']:.2f}")
    except Exception as e:
        log(f"  Error: {e}")

    # ========================================================================
    # 4. Signal strength vs returns
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  SIGNAL STRENGTH ANALYSIS")
    log(f"{'='*120}")

    try:
        best["sig_bucket"] = pd.cut(best["signal_strength"],
                                     bins=[3.5, 4.0, 4.5, 5.0, 6.0, 10.0, 100.0],
                                     labels=["3.5-4.0", "4.0-4.5", "4.5-5.0",
                                             "5.0-6.0", "6.0-10.0", "10.0+"])
        sig_analysis = best.groupby("sig_bucket", observed=True).agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            avg_gross=("gross_bps", "mean"),
            avg_net_tk=("net_bps_taker", "mean"),
            avg_net_mk=("net_bps_maker", "mean"),
        )
        for bucket, r in sig_analysis.iterrows():
            log(f"  Signal {bucket:10s}: {r['n']:4.0f} trades  WR(tk)={r['wr']:.1%}  "
                f"avg_gross={r['avg_gross']:+.0f}  net_tk={r['avg_net_tk']:+.0f}  "
                f"net_mk={r['avg_net_mk']:+.0f}")
    except Exception as e:
        log(f"  Error: {e}")

    # ========================================================================
    # 5. Direction analysis (long vs short)
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  DIRECTION ANALYSIS")
    log(f"{'='*120}")

    try:
        for dir_val, dir_name in [(1, "LONG"), (-1, "SHORT")]:
            subset = best[best["direction"] == dir_val]
            if len(subset) == 0:
                continue
            n = len(subset)
            wr = (subset["net_bps_taker"] > 0).mean()
            avg = subset["net_bps_taker"].mean()
            total = subset["net_bps_taker"].sum()
            log(f"  {dir_name:5s}: {n:4d} trades  WR={wr:.1%}  "
                f"avg_net={avg:+.0f}bps  total={total:+.0f}bps")
    except Exception as e:
        log(f"  Error: {e}")

    # ========================================================================
    # 6. Drawdown analysis (cumulative PnL)
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  DRAWDOWN ANALYSIS")
    log(f"{'='*120}")

    try:
        best_sorted = best.sort_values("entry_time")
        cum_pnl = best_sorted["net_bps_taker"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        final_pnl = cum_pnl.iloc[-1]

        log(f"  Final cumulative PnL: {final_pnl:+.0f} bps")
        log(f"  Max drawdown: {max_dd:+.0f} bps")
        log(f"  Max DD occurred at trade #{max_dd_idx}")
        log(f"  Calmar-like ratio: {final_pnl / abs(max_dd):.2f}" if max_dd != 0 else "  No drawdown")

        # Also compute for maker fees
        cum_pnl_mk = best_sorted["net_bps_maker"].cumsum()
        running_max_mk = cum_pnl_mk.cummax()
        dd_mk = (cum_pnl_mk - running_max_mk).min()
        log(f"\n  Maker fees:")
        log(f"  Final cumulative PnL: {cum_pnl_mk.iloc[-1]:+.0f} bps")
        log(f"  Max drawdown: {dd_mk:+.0f} bps")
    except Exception as e:
        log(f"  Error: {e}")

    # ========================================================================
    # 7. Hold time analysis
    # ========================================================================
    log(f"\n{'='*120}")
    log(f"  HOLD TIME ANALYSIS")
    log(f"{'='*120}")

    try:
        best["hold_bucket"] = pd.cut(best["hold_bars"],
                                      bins=[0, 3, 6, 12, 24, 100],
                                      labels=["1-3", "4-6", "7-12", "13-24", "24+"])
        hold_analysis = best.groupby("hold_bucket", observed=True).agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            avg_net=("net_bps_taker", "mean"),
            avg_gross=("gross_bps", "mean"),
        )
        for bucket, r in hold_analysis.iterrows():
            log(f"  Hold {bucket:6s}: {r['n']:4.0f} trades  WR={r['wr']:.1%}  "
                f"avg_gross={r['avg_gross']:+.0f}  avg_net={r['avg_net']:+.0f}")
    except Exception as e:
        log(f"  Error: {e}")

    # Save WFO results
    all_wfo = pd.concat([is_df, oos_df], ignore_index=True)
    all_wfo.to_csv("wfo_trades.csv", index=False)
    log(f"\nWFO trades saved to wfo_trades.csv")


if __name__ == "__main__":
    main()
