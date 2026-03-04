#!/usr/bin/env python3
"""
Strategy v3: Honest validation + new approaches.

Problems with v2:
- October 2025 = 58% of trades, 101% of PnL. Single-event dependency.
- Low trade count (236 trades in 8 months) → not statistically robust.

v3 approaches:
1. EXCLUDE-ONE-MONTH validation: for each month, train on other months, test on that month
2. VOLATILITY-CONDITIONED strategy: only trade when vol is elevated (dislocation regime)
3. LOWER THRESHOLD + MORE TRADES: sacrifice per-trade edge for frequency
4. CROSS-EXCHANGE MOMENTUM: instead of mean-reversion, follow the leader exchange
5. BINANCE TAKER FLOW: use Binance taker buy/sell imbalance as directional signal

The key insight: cross-exchange dislocations happen most during volatile periods.
Instead of waiting for extreme z-scores, detect when vol regime shifts and
ride the mean-reversion during the ENTIRE elevated-vol episode.
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


def process_symbol_v3(sym):
    """
    Process one symbol: compute features, run multiple strategy variants.
    Returns all trades with timestamps for proper validation.
    """
    try:
        df = load_symbol(sym)
        if df.empty or len(df) < 5000:
            return sym, None

        feat = compute_features(df)
        sig_base = compute_composite_signal(feat)
        feat["sig_base"] = sig_base
        feat = feat.iloc[300:]

        # Additional features for v3 strategies
        mid = (feat["bb_close"] + feat["bn_close"]) / 2
        mid_ret = mid.pct_change()

        # Realized vol (annualized from 5m)
        feat["rvol_1h"] = mid_ret.rolling(12).std() * np.sqrt(288) * 100  # in %
        feat["rvol_6h"] = mid_ret.rolling(72).std() * np.sqrt(288) * 100
        feat["rvol_ratio"] = feat["rvol_1h"] / feat["rvol_6h"].replace(0, np.nan)

        # Binance taker flow signal (smoothed)
        if "bn_taker_buy_pct" in feat.columns:
            feat["taker_flow_6"] = feat["bn_taker_buy_pct"].rolling(6).mean() - 0.5  # -0.5 to +0.5
            feat["taker_flow_12"] = feat["bn_taker_buy_pct"].rolling(12).mean() - 0.5
            feat["taker_flow_z"] = (feat["taker_flow_12"] - feat["taker_flow_12"].rolling(288).mean()) / \
                                    feat["taker_flow_12"].rolling(288).std().replace(0, np.nan)

        # Price divergence raw (not z-scored)
        feat["pdiv_raw"] = (feat["bb_close"] - feat["bn_close"]) / mid * 10000

        feat = feat.dropna(subset=["sig_base", "rvol_1h"])
        if len(feat) < 3000:
            return sym, None

        mid_arr = ((feat["bb_close"].values + feat["bn_close"].values) / 2)
        sig_arr = feat["sig_base"].values
        rvol_arr = feat["rvol_1h"].values
        rvol_ratio_arr = feat["rvol_ratio"].values
        pdiv_arr = feat["pdiv_raw"].values
        taker_arr = feat["taker_flow_z"].values if "taker_flow_z" in feat.columns else np.zeros(len(feat))
        idx = feat.index
        n = len(feat)

        all_trades = {}

        # =====================================================================
        # STRATEGY 1: Original long-only mean-reversion (baseline)
        # =====================================================================
        for thr in [3.0, 3.5, 4.0]:
            for max_h in [12, 24]:
                key = f"S1_meanrev_thr{thr}_h{max_h}"
                trades = _run_long_only(mid_arr, sig_arr, n, idx, thr, 3, max_h)
                if trades:
                    all_trades[key] = trades

        # =====================================================================
        # STRATEGY 2: Vol-conditioned mean-reversion
        # Only enter when rvol_ratio > 1.5 (vol expanding) AND sig extreme
        # =====================================================================
        for thr in [2.0, 2.5, 3.0]:
            for vol_thr in [1.3, 1.5, 2.0]:
                key = f"S2_volcond_thr{thr}_vol{vol_thr}_h24"
                vol_gate = rvol_ratio_arr > vol_thr
                trades = _run_long_only_gated(mid_arr, sig_arr, vol_gate, n, idx, thr, 3, 24)
                if trades:
                    all_trades[key] = trades

        # =====================================================================
        # STRATEGY 3: Cross-exchange momentum (follow the leader)
        # If BB has been outperforming BN over last N bars → go LONG on BB
        # (not mean-reversion — momentum following)
        # =====================================================================
        bb_ret = feat["bb_ret"].values
        bn_ret = feat["bn_ret"].values
        for lookback in [6, 12, 24]:
            # Cumulative BB excess return over BN in last N bars
            excess = np.full(n, np.nan)
            for i in range(lookback, n):
                excess[i] = np.sum(bb_ret[i-lookback:i]) - np.sum(bn_ret[i-lookback:i])
            excess_bps = excess * 10000

            for thr_bps in [10, 20, 30]:
                key = f"S3_momentum_lb{lookback}_thr{thr_bps}"
                trades = _run_momentum(mid_arr, excess_bps, n, idx, thr_bps, 6, 24)
                if trades:
                    all_trades[key] = trades

        # =====================================================================
        # STRATEGY 4: Taker flow + price divergence combo
        # When BN aggressive buying (taker_z > 1) AND BB is cheaper → buy BB
        # =====================================================================
        if "taker_flow_z" in feat.columns:
            for taker_thr in [1.0, 1.5, 2.0]:
                for pdiv_thr in [3, 5, 8]:
                    key = f"S4_takerflow_ft{taker_thr}_pd{pdiv_thr}"
                    trades = _run_taker_flow(mid_arr, taker_arr, pdiv_arr, n, idx,
                                             taker_thr, pdiv_thr, 3, 24)
                    if trades:
                        all_trades[key] = trades

        # =====================================================================
        # STRATEGY 5: Volatility breakout — enter when price div spikes
        # beyond N standard deviations, bet on continuation (not reversion)
        # during high-vol episodes
        # =====================================================================
        pdiv_ma = pd.Series(pdiv_arr).rolling(72).mean().values
        pdiv_std = pd.Series(pdiv_arr).rolling(72).std().values
        pdiv_z = (pdiv_arr - pdiv_ma) / np.where(pdiv_std > 0, pdiv_std, np.nan)

        for z_thr in [2.0, 2.5, 3.0]:
            for vol_thr in [1.3, 1.5]:
                key = f"S5_divbreak_z{z_thr}_vol{vol_thr}"
                trades = _run_div_breakout(mid_arr, pdiv_z, rvol_ratio_arr, n, idx,
                                           z_thr, vol_thr, 3, 12)
                if trades:
                    all_trades[key] = trades

        return sym, all_trades
    except Exception as e:
        return sym, None


def _run_long_only(mid, sig, n, idx, threshold, min_hold, max_hold, cooldown=3):
    """Long-only mean-reversion: enter when sig < -threshold."""
    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown
    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if sig[i] < -threshold:
                entry_i = i
                in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            if sig[i] >= 0:
                exit_now = True
            elif bars >= max_hold:
                exit_now = True
            elif sig[i] < -threshold * 1.5 and bars >= 6:
                exit_now = True
            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                trades.append((gross, abs(sig[entry_i]), bars, str(idx[entry_i]), str(idx[i]), "long"))
                in_trade = False
                last_exit = i
    return trades


def _run_long_only_gated(mid, sig, gate, n, idx, threshold, min_hold, max_hold, cooldown=3):
    """Long-only with volatility gate."""
    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown
    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if sig[i] < -threshold and gate[i]:
                entry_i = i
                in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            if sig[i] >= 0:
                exit_now = True
            elif bars >= max_hold:
                exit_now = True
            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                trades.append((gross, abs(sig[entry_i]), bars, str(idx[entry_i]), str(idx[i]), "long_vg"))
                in_trade = False
                last_exit = i
    return trades


def _run_momentum(mid, excess_bps, n, idx, thr_bps, min_hold, max_hold, cooldown=6):
    """
    Momentum: when BB has outperformed BN by > thr_bps in last N bars,
    go LONG expecting continuation.
    """
    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown
    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if np.isnan(excess_bps[i]):
                continue
            if excess_bps[i] > thr_bps:
                entry_i = i
                in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            if not np.isnan(excess_bps[i]) and excess_bps[i] < 0:
                exit_now = True
            elif bars >= max_hold:
                exit_now = True
            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                trades.append((gross, excess_bps[entry_i], bars, str(idx[entry_i]), str(idx[i]), "momentum"))
                in_trade = False
                last_exit = i
    return trades


def _run_taker_flow(mid, taker_z, pdiv, n, idx, taker_thr, pdiv_thr, min_hold, max_hold, cooldown=3):
    """
    Taker flow + price divergence:
    When BN taker buying (taker_z > thr) AND BB is cheaper (pdiv < -pdiv_thr) → LONG
    """
    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown
    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if np.isnan(taker_z[i]) or np.isnan(pdiv[i]):
                continue
            if taker_z[i] > taker_thr and pdiv[i] < -pdiv_thr:
                entry_i = i
                in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            if not np.isnan(taker_z[i]) and taker_z[i] < 0:
                exit_now = True
            elif not np.isnan(pdiv[i]) and pdiv[i] > 0:
                exit_now = True
            elif bars >= max_hold:
                exit_now = True
            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                trades.append((gross, taker_z[entry_i], bars, str(idx[entry_i]), str(idx[i]), "taker_flow"))
                in_trade = False
                last_exit = i
    return trades


def _run_div_breakout(mid, pdiv_z, rvol_ratio, n, idx, z_thr, vol_thr, min_hold, max_hold, cooldown=6):
    """
    Breakout on price divergence z-score during high vol.
    When pdiv_z < -z_thr AND rvol expanding → BB is diverging DOWN from BN abnormally
    → expect mean-reversion UP → LONG
    """
    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown
    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if np.isnan(pdiv_z[i]) or np.isnan(rvol_ratio[i]):
                continue
            if pdiv_z[i] < -z_thr and rvol_ratio[i] > vol_thr:
                entry_i = i
                in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            if not np.isnan(pdiv_z[i]) and pdiv_z[i] > 0:
                exit_now = True
            elif bars >= max_hold:
                exit_now = True
            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                trades.append((gross, abs(pdiv_z[entry_i]), bars, str(idx[entry_i]), str(idx[i]), "breakout"))
                in_trade = False
                last_exit = i
    return trades


def main():
    symbols = list_common_symbols()
    log(f"V3 Multi-Strategy Sweep: {len(symbols)} symbols")
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(process_symbol_v3, s): s for s in symbols}
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
    config_trades = {}

    for key in config_keys:
        all_trades = []
        for sym, sym_res in all_results.items():
            if key in sym_res:
                for t in sym_res[key]:
                    all_trades.append({
                        "symbol": sym, "gross_bps": t[0], "sig_str": t[1],
                        "hold": t[2], "entry_time": t[3], "exit_time": t[4],
                        "strategy": t[5],
                    })

        if len(all_trades) < 20:
            continue

        config_trades[key] = all_trades
        df_t = pd.DataFrame(all_trades)
        n = len(df_t)
        nsym = df_t["symbol"].nunique()
        avg_gross = df_t["gross_bps"].mean()
        df_t["entry_month"] = pd.to_datetime(df_t["entry_time"]).dt.to_period("M")

        for fee_label, fee_rt in [("taker", 20), ("maker", 8)]:
            nets = df_t["gross_bps"] - fee_rt
            wr = (nets > 0).mean()
            total = nets.sum()
            wins = nets[nets > 0].sum()
            losses = abs(nets[nets < 0].sum())
            pf = wins / losses if losses > 0 else 999

            # Monthly analysis
            monthly_pnl = []
            for m in sorted(df_t["entry_month"].unique()):
                m_trades = df_t[df_t["entry_month"] == m]
                m_net = (m_trades["gross_bps"] - fee_rt).sum()
                monthly_pnl.append(m_net)

            pos_m = sum(1 for x in monthly_pnl if x > 0)
            tot_m = len(monthly_pnl)

            # EXCLUDE-OCTOBER analysis
            no_oct = df_t[~df_t["entry_month"].astype(str).str.startswith("2025-10")]
            if len(no_oct) > 5:
                no_oct_nets = no_oct["gross_bps"] - fee_rt
                no_oct_wr = (no_oct_nets > 0).mean()
                no_oct_avg = no_oct_nets.mean()
                no_oct_pf = (no_oct_nets[no_oct_nets > 0].sum() /
                             abs(no_oct_nets[no_oct_nets < 0].sum())
                             if (no_oct_nets < 0).any() else 999)
                no_oct_n = len(no_oct)
            else:
                no_oct_wr = no_oct_avg = no_oct_pf = 0
                no_oct_n = 0

            rows.append({
                "config": key, "fee": fee_label, "n": n, "nsym": nsym,
                "avg_gross": avg_gross, "avg_net": nets.mean(),
                "wr": wr, "pf": pf, "total": total,
                "pos_m": pos_m, "tot_m": tot_m,
                "no_oct_n": no_oct_n, "no_oct_wr": no_oct_wr,
                "no_oct_avg": no_oct_avg, "no_oct_pf": no_oct_pf,
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv("v3_sweep_results.csv", index=False)

    # === PRINT: sorted by ex-October profit factor (taker) ===
    log(f"\n{'='*170}")
    log(f"  V3 MULTI-STRATEGY — TAKER (20bps RT) — sorted by EX-OCTOBER Profit Factor")
    log(f"{'='*170}")
    log(f"{'Config':45s} {'N':>5} {'Sym':>4} {'Gross':>7} {'Net':>7} "
        f"{'WR':>6} {'PF':>6} {'Mo':>5} "
        f"{'noOct_N':>7} {'noOct_WR':>8} {'noOct_Avg':>9} {'noOct_PF':>8}")
    log("-" * 170)

    taker = df_results[(df_results["fee"] == "taker") & (df_results["no_oct_n"] >= 10)].copy()
    taker_sorted = taker.sort_values("no_oct_pf", ascending=False)

    for _, r in taker_sorted.head(40).iterrows():
        log(f"{r['config']:45s} {r['n']:5.0f} {r['nsym']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['wr']:6.1%} {r['pf']:6.2f} "
            f"{r['pos_m']:.0f}/{r['tot_m']:.0f} "
            f"{r['no_oct_n']:7.0f} {r['no_oct_wr']:8.1%} "
            f"{r['no_oct_avg']:+9.1f} {r['no_oct_pf']:8.2f}")

    # === PRINT: MAKER results ===
    log(f"\n{'='*170}")
    log(f"  V3 MULTI-STRATEGY — MAKER (8bps RT) — sorted by EX-OCTOBER Profit Factor")
    log(f"{'='*170}")

    maker = df_results[(df_results["fee"] == "maker") & (df_results["no_oct_n"] >= 10)].copy()
    maker_sorted = maker.sort_values("no_oct_pf", ascending=False)

    for _, r in maker_sorted.head(30).iterrows():
        log(f"{r['config']:45s} {r['n']:5.0f} {r['nsym']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['wr']:6.1%} {r['pf']:6.2f} "
            f"{r['pos_m']:.0f}/{r['tot_m']:.0f} "
            f"{r['no_oct_n']:7.0f} {r['no_oct_wr']:8.1%} "
            f"{r['no_oct_avg']:+9.1f} {r['no_oct_pf']:8.2f}")

    # === DETAILED ANALYSIS of best ex-October config ===
    if len(taker_sorted) > 0:
        best_key = taker_sorted.iloc[0]["config"]
        log(f"\n{'='*120}")
        log(f"  BEST EX-OCTOBER CONFIG: {best_key}")
        log(f"{'='*120}")

        best = pd.DataFrame(config_trades[best_key])
        best["net_tk"] = best["gross_bps"] - 20
        best["net_mk"] = best["gross_bps"] - 8
        best["month"] = pd.to_datetime(best["entry_time"]).dt.to_period("M")

        log(f"\n  Monthly (taker):")
        monthly = best.groupby("month").agg(
            n=("net_tk", "count"),
            wr=("net_tk", lambda x: (x > 0).mean()),
            avg=("net_tk", "mean"),
            total=("net_tk", "sum"),
        )
        for m, r in monthly.iterrows():
            s = "✓" if r["total"] > 0 else "✗"
            oct_tag = " ← OCT" if "2025-10" in str(m) else ""
            log(f"    {str(m):10s} {s} {r['n']:4.0f} trades WR={r['wr']:.1%} "
                f"avg={r['avg']:+.0f}bps total={r['total']:+.0f}bps{oct_tag}")

        # Per-symbol (ex-Oct)
        no_oct = best[~best["month"].astype(str).str.startswith("2025-10")]
        if len(no_oct) > 0:
            log(f"\n  EX-OCTOBER: {len(no_oct)} trades")
            sym_stats = no_oct.groupby("symbol").agg(
                n=("net_tk", "count"),
                wr=("net_tk", lambda x: (x > 0).mean()),
                total=("net_tk", "sum"),
            ).sort_values("total", ascending=False)
            prof = (sym_stats["total"] > 0).sum()
            log(f"  Profitable symbols (ex-Oct): {prof}/{len(sym_stats)}")
            log(f"  Top 10:")
            for s, r in sym_stats.head(10).iterrows():
                log(f"    {s:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} total={r['total']:+.0f}bps")

        best.to_csv("v3_best_trades.csv", index=False)
        log(f"\n  Saved {len(best)} trades to v3_best_trades.csv")

    # === STRATEGY FAMILY COMPARISON ===
    log(f"\n{'='*120}")
    log(f"  STRATEGY FAMILY COMPARISON (taker fees, ex-October)")
    log(f"{'='*120}")

    taker["family"] = taker["config"].str.extract(r'^(S\d+_\w+)_')[0]
    family_stats = taker.groupby("family").agg(
        best_pf=("no_oct_pf", "max"),
        avg_pf=("no_oct_pf", "mean"),
        best_wr=("no_oct_wr", "max"),
        avg_trades=("n", "mean"),
        n_configs=("config", "count"),
    ).sort_values("best_pf", ascending=False)

    for fam, r in family_stats.iterrows():
        log(f"  {fam:25s}: best_PF={r['best_pf']:.2f} avg_PF={r['avg_pf']:.2f} "
            f"best_WR={r['best_wr']:.1%} avg_trades={r['avg_trades']:.0f} ({r['n_configs']:.0f} configs)")


if __name__ == "__main__":
    main()
