#!/usr/bin/env python3
"""
Fast v2 strategy sweep: load each symbol ONCE, test all configs in-memory.
This avoids the ~70s per config overhead from v2.
"""

import sys, time, os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["PYTHONUNBUFFERED"] = "1"

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def compute_enhanced_signals(df):
    """Compute multiple signal variants."""
    f = df.copy()
    f["sig_base"] = compute_composite_signal(f)

    mid_ret = ((f["bb_close"] + f["bn_close"]) / 2).pct_change()
    vol_20 = mid_ret.rolling(20).std() * np.sqrt(288)
    vol_60 = mid_ret.rolling(60).std() * np.sqrt(288)
    f["vol_ratio_regime"] = vol_20 / vol_60.replace(0, np.nan)
    f["sig_vol_adj"] = f["sig_base"] * f["vol_ratio_regime"].clip(0.5, 3.0)

    if "vol_ratio_z72" in f.columns:
        vol_confirm = np.where(
            f["sig_base"] < 0,
            np.clip(f["vol_ratio_z72"], -3, 0),
            np.clip(-f["vol_ratio_z72"], -3, 0)
        )
        f["sig_vol_confirm"] = f["sig_base"] + vol_confirm * 0.5

    if "bn_taker_imbalance" in f.columns:
        taker_z = (f["bn_taker_imbalance"] - f["bn_taker_imbalance"].rolling(72).mean()) / \
                  f["bn_taker_imbalance"].rolling(72).std().replace(0, np.nan)
        f["sig_taker"] = f["sig_base"] - taker_z * 0.3

    f["vol_5m"] = mid_ret.rolling(12).std() * 10000
    f["vol_gate"] = f["vol_5m"] > 5
    return f


def run_long_only(mid, sig, vol_ok, n, threshold, min_hold, max_hold,
                  trailing_stop_bps, cooldown=3):
    """Fast LONG-only backtest on numpy arrays."""
    trades = []
    in_trade = False
    entry_i = 0
    peak_price = 0.0
    last_exit = -cooldown

    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if sig[i] < -threshold and vol_ok[i]:
                entry_i = i
                in_trade = True
                peak_price = mid[i]
        else:
            bars = i - entry_i
            current_ret = (mid[i] / mid[entry_i] - 1) * 10000
            if mid[i] > peak_price:
                peak_price = mid[i]
            peak_ret = (peak_price / mid[entry_i] - 1) * 10000

            if bars < min_hold:
                continue

            exit_now = False
            reason = ""
            if sig[i] >= 0:
                exit_now, reason = True, "cross"
            elif bars >= max_hold:
                exit_now, reason = True, "maxhold"
            elif trailing_stop_bps > 0 and peak_ret - current_ret > trailing_stop_bps:
                exit_now, reason = True, "trail"
            elif sig[i] < -threshold * 1.5 and bars >= 6:
                exit_now, reason = True, "reversal"

            if exit_now:
                trades.append((current_ret, abs(sig[entry_i]), bars, i, entry_i, reason))
                in_trade = False
                last_exit = i

    return trades


def process_symbol(sym):
    """Load, compute all signals, run all configs, return results."""
    try:
        df = load_symbol(sym)
        if df.empty or len(df) < 5000:
            return sym, None

        feat = compute_features(df)
        feat = compute_enhanced_signals(feat)
        feat = feat.iloc[300:]

        signals = {}
        for sig_name in ["sig_base", "sig_vol_adj", "sig_vol_confirm", "sig_taker"]:
            if sig_name in feat.columns:
                valid = feat.dropna(subset=[sig_name])
                if len(valid) >= 3000:
                    signals[sig_name] = valid

        if not signals:
            return sym, None

        results = {}
        for sig_name, valid in signals.items():
            mid = (valid["bb_close"].values + valid["bn_close"].values) / 2
            sig = valid[sig_name].values
            vol_ok = valid["vol_gate"].values if "vol_gate" in valid.columns else np.ones(len(valid), dtype=bool)
            n = len(valid)
            idx = valid.index

            for threshold in [2.5, 3.0, 3.5, 4.0]:
                for max_hold in [12, 24]:
                    for trailing_stop in [0, 150, 300]:
                        # With vol gate
                        key = f"{sig_name}_thr{threshold}_h{max_hold}_ts{trailing_stop}"
                        trades = run_long_only(mid, sig, vol_ok, n, threshold, 3, max_hold, trailing_stop)
                        if trades:
                            # Store with timestamps for monthly analysis
                            results[key] = [(t[0], t[1], t[2], str(idx[t[4]]), str(idx[t[3]]), t[5])
                                            for t in trades]

                        # Without vol gate (for base signal only)
                        if sig_name == "sig_base":
                            key_novg = f"{sig_name}_thr{threshold}_h{max_hold}_ts{trailing_stop}_novg"
                            trades_novg = run_long_only(mid, sig, np.ones(n, dtype=bool), n,
                                                        threshold, 3, max_hold, trailing_stop)
                            if trades_novg:
                                results[key_novg] = [(t[0], t[1], t[2], str(idx[t[4]]), str(idx[t[3]]), t[5])
                                                      for t in trades_novg]

        return sym, results
    except Exception as e:
        return sym, None


def main():
    symbols = list_common_symbols()
    log(f"V2 Fast Sweep: LONG-ONLY on {len(symbols)} symbols")
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
            if done % 20 == 0 or done == len(symbols):
                log(f"  [{done}/{len(symbols)}] {len(all_results)} symbols ({time.time()-t0:.0f}s)")

    log(f"\nProcessed {len(all_results)} symbols in {time.time()-t0:.0f}s")

    # Aggregate per config
    config_keys = set()
    for sym_res in all_results.values():
        config_keys.update(sym_res.keys())
    config_keys = sorted(config_keys)

    rows = []
    config_trades = {}  # Store for detailed analysis of best config

    for key in config_keys:
        all_trades = []
        syms_traded = set()
        for sym, sym_res in all_results.items():
            if key in sym_res:
                for t in sym_res[key]:
                    all_trades.append({"symbol": sym, "gross_bps": t[0], "sig_str": t[1],
                                       "hold": t[2], "entry_time": t[3], "exit_time": t[4],
                                       "exit_reason": t[5]})
                syms_traded.add(sym)

        if not all_trades:
            continue

        config_trades[key] = all_trades
        df_t = pd.DataFrame(all_trades)
        n = len(df_t)
        nsym = len(syms_traded)
        avg_gross = df_t["gross_bps"].mean()

        for fee_label, fee_rt in [("taker", 20), ("maker", 8)]:
            nets = df_t["gross_bps"] - fee_rt
            wr = (nets > 0).mean()
            total = nets.sum()
            wins = nets[nets > 0].sum()
            losses = abs(nets[nets < 0].sum())
            pf = wins / losses if losses > 0 else 999
            avg_hold = df_t["hold"].mean()

            # Monthly
            df_t["month"] = pd.to_datetime(df_t["entry_time"]).dt.to_period("M")
            monthly_pnl = df_t.groupby("month")["gross_bps"].sum() - fee_rt * df_t.groupby("month").size()
            pos_m = (monthly_pnl > 0).sum()
            tot_m = len(monthly_pnl)

            # WFO split (time-based)
            df_sorted = df_t.sort_values("entry_time")
            half = len(df_sorted) // 2
            is_net = (df_sorted.iloc[:half]["gross_bps"] - fee_rt)
            oos_net = (df_sorted.iloc[half:]["gross_bps"] - fee_rt)
            is_avg = is_net.mean() if len(is_net) > 0 else 0
            oos_avg = oos_net.mean() if len(oos_net) > 0 else 0
            is_wr = (is_net > 0).mean() if len(is_net) > 0 else 0
            oos_wr = (oos_net > 0).mean() if len(oos_net) > 0 else 0
            oos_pf = (oos_net[oos_net > 0].sum() / abs(oos_net[oos_net < 0].sum())
                      if (oos_net < 0).any() else 999)

            rows.append({
                "config": key, "fee": fee_label, "n": n, "nsym": nsym,
                "avg_gross": avg_gross, "avg_net": nets.mean(),
                "wr": wr, "pf": pf, "total": total, "daily": total/245,
                "avg_hold": avg_hold, "pos_m": pos_m, "tot_m": tot_m,
                "is_avg": is_avg, "oos_avg": oos_avg,
                "is_wr": is_wr, "oos_wr": oos_wr, "oos_pf": oos_pf,
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv("v2_sweep_results.csv", index=False)

    # === PRINT TAKER RESULTS sorted by OOS profit factor ===
    log(f"\n{'='*160}")
    log(f"  V2 LONG-ONLY — TAKER (20bps RT) — sorted by OOS Profit Factor")
    log(f"{'='*160}")
    log(f"{'Config':50s} {'N':>5} {'Sym':>4} {'Gross':>7} {'Net':>7} "
        f"{'WR':>6} {'PF':>6} {'Daily':>7} {'Mo':>5} "
        f"{'IS_avg':>7} {'OOS_avg':>8} {'OOS_WR':>7} {'OOS_PF':>7}")
    log("-" * 160)

    taker = df_results[df_results["fee"] == "taker"].copy()
    taker = taker[taker["n"] >= 20]  # min 20 trades
    taker_sorted = taker.sort_values("oos_pf", ascending=False)

    for _, r in taker_sorted.head(40).iterrows():
        log(f"{r['config']:50s} {r['n']:5.0f} {r['nsym']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['wr']:6.1%} {r['pf']:6.2f} {r['daily']:+7.0f} "
            f"{r['pos_m']:.0f}/{r['tot_m']:.0f} "
            f"{r['is_avg']:+7.0f} {r['oos_avg']:+8.0f} "
            f"{r['oos_wr']:7.1%} {r['oos_pf']:7.2f}")

    # === PRINT MAKER RESULTS ===
    log(f"\n{'='*160}")
    log(f"  V2 LONG-ONLY — MAKER (8bps RT) — sorted by OOS Profit Factor")
    log(f"{'='*160}")

    maker = df_results[df_results["fee"] == "maker"].copy()
    maker = maker[maker["n"] >= 20]
    maker_sorted = maker.sort_values("oos_pf", ascending=False)

    for _, r in maker_sorted.head(30).iterrows():
        log(f"{r['config']:50s} {r['n']:5.0f} {r['nsym']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['wr']:6.1%} {r['pf']:6.2f} {r['daily']:+7.0f} "
            f"{r['pos_m']:.0f}/{r['tot_m']:.0f} "
            f"{r['is_avg']:+7.0f} {r['oos_avg']:+8.0f} "
            f"{r['oos_wr']:7.1%} {r['oos_pf']:7.2f}")

    # === BEST CONFIG DETAILED ANALYSIS ===
    # Pick best by OOS PF for taker
    if len(taker_sorted) > 0:
        best_key = taker_sorted.iloc[0]["config"]
        log(f"\n{'='*120}")
        log(f"  BEST CONFIG DETAILED: {best_key}")
        log(f"{'='*120}")

        best_trades = pd.DataFrame(config_trades[best_key])
        best_trades["net_tk"] = best_trades["gross_bps"] - 20
        best_trades["net_mk"] = best_trades["gross_bps"] - 8

        # Monthly
        best_trades["month"] = pd.to_datetime(best_trades["entry_time"]).dt.to_period("M")
        monthly = best_trades.groupby("month").agg(
            n=("net_tk", "count"),
            wr=("net_tk", lambda x: (x > 0).mean()),
            avg=("net_tk", "mean"),
            total=("net_tk", "sum"),
        )
        log(f"\n  Monthly (taker):")
        for m, r in monthly.iterrows():
            s = "✓" if r["total"] > 0 else "✗"
            log(f"    {str(m):10s} {s} {r['n']:4.0f} trades WR={r['wr']:.1%} "
                f"avg={r['avg']:+.0f}bps total={r['total']:+.0f}bps")

        # Per-symbol
        sym_stats = best_trades.groupby("symbol").agg(
            n=("net_tk", "count"),
            wr=("net_tk", lambda x: (x > 0).mean()),
            total=("net_tk", "sum"),
        ).sort_values("total", ascending=False)

        prof = (sym_stats["total"] > 0).sum()
        log(f"\n  Profitable symbols: {prof}/{len(sym_stats)}")
        log(f"  Top 15:")
        for s, r in sym_stats.head(15).iterrows():
            log(f"    {s:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} total={r['total']:+.0f}bps")

        # Exit reasons
        log(f"\n  Exit reasons:")
        for reason, grp in best_trades.groupby("exit_reason"):
            nets = grp["net_tk"]
            log(f"    {reason:15s}: {len(grp):4d} trades WR={(nets > 0).mean():.1%} "
                f"avg={nets.mean():+.0f}bps")

        # Signal strength buckets
        best_trades["sig_bucket"] = pd.cut(best_trades["sig_str"],
                                            bins=[0, 3.5, 4.0, 5.0, 7.0, 100],
                                            labels=["<3.5", "3.5-4", "4-5", "5-7", "7+"])
        log(f"\n  Signal strength:")
        for bucket, grp in best_trades.groupby("sig_bucket", observed=True):
            nets = grp["net_tk"]
            log(f"    sig {bucket:6s}: {len(grp):4d} trades WR={(nets > 0).mean():.1%} "
                f"avg={nets.mean():+.0f}bps")

        # Drawdown
        sorted_t = best_trades.sort_values("entry_time")
        cum = sorted_t["net_tk"].cumsum()
        dd = cum - cum.cummax()
        log(f"\n  Cumulative PnL: {cum.iloc[-1]:+.0f} bps")
        log(f"  Max drawdown: {dd.min():+.0f} bps")
        if dd.min() != 0:
            log(f"  Calmar: {cum.iloc[-1] / abs(dd.min()):.2f}")

        best_trades.to_csv("v2_best_trades.csv", index=False)
        log(f"\n  Saved {len(best_trades)} trades to v2_best_trades.csv")


if __name__ == "__main__":
    main()
