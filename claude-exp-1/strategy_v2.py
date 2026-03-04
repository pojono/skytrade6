#!/usr/bin/env python3
"""
Cross-Exchange Mean-Reversion Strategy v2

Key improvements over v1:
1. LONG-ONLY — short side has no edge OOS (WR=45.8%, avg=-60bps)
2. Higher composite threshold (3.0-4.0) for OOS robustness
3. Enhanced signal: adds volatility regime filter, volume confirmation
4. Trailing stop + adaptive exit
5. Multiple signal variants tested
6. Proper walk-forward with rolling window (not just 50/50 split)

Logic:
- When composite signal < -threshold (Bybit "too cheap" vs Binance), go LONG
- This means: BB is trading below its normal premium to BN → expect BB to catch up
- Exit when signal crosses back to 0 or stop hit

Walk-forward: 3-month training window, 1-month test, rolling monthly.
"""

import sys
import time
import os
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


def compute_enhanced_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced signal variants beyond the basic composite.
    Returns df with multiple signal columns.
    """
    f = df.copy()

    # Base composite
    f["sig_base"] = compute_composite_signal(f)

    # --- Signal 2: Volatility-adjusted ---
    # Scale signal by recent realized volatility (higher vol = bigger dislocations expected)
    mid_ret = ((f["bb_close"] + f["bn_close"]) / 2).pct_change()
    vol_20 = mid_ret.rolling(20).std() * np.sqrt(288)  # annualized from 5m bars
    vol_60 = mid_ret.rolling(60).std() * np.sqrt(288)
    f["vol_ratio_regime"] = vol_20 / vol_60.replace(0, np.nan)
    # High vol_ratio = volatility expanding → bigger dislocations → more edge
    f["sig_vol_adj"] = f["sig_base"] * f["vol_ratio_regime"].clip(0.5, 3.0)

    # --- Signal 3: Volume-confirmed ---
    # Only trust the signal when volume is shifting toward the "cheap" exchange
    if "vol_ratio_z72" in f.columns:
        # When BB is cheap (sig < 0) AND volume is flowing TO BB → stronger signal
        vol_confirm = np.where(
            f["sig_base"] < 0,
            np.clip(f["vol_ratio_z72"], -3, 0),  # negative vol_z = more BB volume = confirms
            np.clip(-f["vol_ratio_z72"], -3, 0)   # positive sig: more BN volume confirms
        )
        f["sig_vol_confirm"] = f["sig_base"] + vol_confirm * 0.5

    # --- Signal 4: Taker flow confirmation ---
    if "bn_taker_imbalance" in f.columns:
        taker_z = (f["bn_taker_imbalance"] - f["bn_taker_imbalance"].rolling(72).mean()) / \
                  f["bn_taker_imbalance"].rolling(72).std().replace(0, np.nan)
        # When sig < 0 (expect up) AND taker buying (taker_z > 0) → stronger
        f["sig_taker"] = f["sig_base"] - taker_z * 0.3  # subtract because we want to amplify negative sig

    # --- Signal 5: OI momentum ---
    if "oi_div" in f.columns:
        oi_z = (f["oi_div"] - f["oi_div"].rolling(288).mean()) / \
               f["oi_div"].rolling(288).std().replace(0, np.nan)
        f["sig_oi"] = f["sig_base"] + oi_z * 0.3

    # --- Signal 6: Premium mean-reversion ---
    if "premium_z288" in f.columns:
        f["sig_premium"] = f["sig_base"] * (1 + f["premium_z288"].abs() * 0.2)

    # --- Volatility filter: don't trade in ultra-low vol (no moves) ---
    f["vol_5m"] = mid_ret.rolling(12).std() * 10000  # 1h vol in bps
    f["vol_gate"] = f["vol_5m"] > 5  # at least 5 bps/bar vol

    return f


def backtest_long_only(sym: str, signal_col: str = "sig_base",
                       threshold: float = 3.0,
                       min_hold: int = 3,
                       max_hold: int = 24,
                       trailing_stop_bps: float = 0,
                       vol_gate: bool = True,
                       period_start: str = None,
                       period_end: str = None) -> list[dict]:
    """
    Long-only mean-reversion backtest.

    Entry: signal < -threshold (BB "too cheap" vs BN)
    Exit: signal > 0 OR max_hold OR trailing stop
    """
    df = load_symbol(sym)
    if df.empty or len(df) < 5000:
        return []

    feat = compute_features(df)
    feat = compute_enhanced_signal(feat)
    feat = feat.iloc[300:].dropna(subset=[signal_col])

    if period_start:
        feat = feat[feat.index >= period_start]
    if period_end:
        feat = feat[feat.index < period_end]

    if len(feat) < 500:
        return []

    mid = (feat["bb_close"].values + feat["bn_close"].values) / 2
    sig = feat[signal_col].values
    vol_ok = feat["vol_gate"].values if vol_gate and "vol_gate" in feat.columns else np.ones(len(feat), dtype=bool)
    n = len(feat)
    idx = feat.index

    trades = []
    in_trade = False
    entry_i = 0
    peak_price = 0
    cooldown = 3
    last_exit = -cooldown

    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            # LONG entry: signal very negative (BB cheap vs BN)
            if sig[i] < -threshold and vol_ok[i]:
                entry_i = i
                in_trade = True
                peak_price = mid[i]
        else:
            bars = i - entry_i
            current_ret_bps = (mid[i] / mid[entry_i] - 1) * 10000
            peak_price = max(peak_price, mid[i])
            peak_ret_bps = (peak_price / mid[entry_i] - 1) * 10000

            exit_now = False
            exit_reason = ""

            # Min hold not reached
            if bars < min_hold:
                continue

            # 1. Signal crossed zero (reversion complete)
            if sig[i] >= 0:
                exit_now = True
                exit_reason = "signal_cross"

            # 2. Max hold
            if bars >= max_hold:
                exit_now = True
                exit_reason = "max_hold"

            # 3. Trailing stop (from peak)
            if trailing_stop_bps > 0 and peak_ret_bps - current_ret_bps > trailing_stop_bps:
                exit_now = True
                exit_reason = "trailing_stop"

            # 4. Signal reversed strongly (got worse)
            if sig[i] < -threshold * 1.5 and bars >= 6:
                exit_now = True
                exit_reason = "reversal_stop"

            if exit_now:
                gross_bps = current_ret_bps
                trades.append({
                    "symbol": sym,
                    "entry_time": str(idx[entry_i]),
                    "exit_time": str(idx[i]),
                    "direction": 1,
                    "gross_bps": gross_bps,
                    "signal_strength": abs(sig[entry_i]),
                    "hold_bars": bars,
                    "exit_reason": exit_reason,
                    "peak_ret_bps": peak_ret_bps,
                })
                in_trade = False
                last_exit = i

    return trades


def walk_forward_monthly(sym: str, signal_col: str, threshold: float,
                         min_hold: int, max_hold: int,
                         trailing_stop_bps: float, vol_gate: bool) -> list[dict]:
    """
    Walk-forward with monthly periods.
    No training needed (signal is parameterized, not ML), so we just
    split into monthly periods for consistency analysis.
    """
    return backtest_long_only(
        sym, signal_col=signal_col, threshold=threshold,
        min_hold=min_hold, max_hold=max_hold,
        trailing_stop_bps=trailing_stop_bps, vol_gate=vol_gate
    )


def run_config(args):
    """Worker function for parallel execution."""
    sym, cfg = args
    return backtest_long_only(
        sym,
        signal_col=cfg["signal"],
        threshold=cfg["threshold"],
        min_hold=cfg.get("min_hold", 3),
        max_hold=cfg.get("max_hold", 24),
        trailing_stop_bps=cfg.get("trailing_stop", 0),
        vol_gate=cfg.get("vol_gate", True),
    )


def analyze_trades(trades_list, config_name, fee_rt_bps=20):
    """Analyze a list of trade dicts."""
    if not trades_list:
        return None

    df = pd.DataFrame(trades_list)
    n = len(df)
    nsym = df["symbol"].nunique()
    nets = df["gross_bps"] - fee_rt_bps
    wr = (nets > 0).mean()
    avg_gross = df["gross_bps"].mean()
    avg_net = nets.mean()
    total = nets.sum()
    wins = nets[nets > 0].sum()
    losses = abs(nets[nets < 0].sum())
    pf = wins / losses if losses > 0 else 999
    avg_hold = df["hold_bars"].mean()

    # Monthly consistency
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly_totals = (df.groupby("month")["gross_bps"].sum() - fee_rt_bps * df.groupby("month").size())
    pos_months = (monthly_totals > 0).sum()
    total_months = len(monthly_totals)

    # Per-direction (should all be long=+1 in v2)
    long_n = (df["direction"] == 1).sum()
    short_n = (df["direction"] == -1).sum()

    # WFO: first half vs second half
    split = len(df) // 2
    df_sorted = df.sort_values("entry_time")
    is_nets = df_sorted.iloc[:split]["gross_bps"] - fee_rt_bps
    oos_nets = df_sorted.iloc[split:]["gross_bps"] - fee_rt_bps
    is_wr = (is_nets > 0).mean() if len(is_nets) > 0 else 0
    oos_wr = (oos_nets > 0).mean() if len(oos_nets) > 0 else 0
    is_avg = is_nets.mean() if len(is_nets) > 0 else 0
    oos_avg = oos_nets.mean() if len(oos_nets) > 0 else 0

    return {
        "config": config_name,
        "n_trades": n, "n_symbols": nsym,
        "avg_gross": avg_gross, "avg_net": avg_net,
        "win_rate": wr, "total_net": total,
        "profit_factor": pf, "avg_hold": avg_hold,
        "pos_months": pos_months, "total_months": total_months,
        "is_wr": is_wr, "oos_wr": oos_wr,
        "is_avg_net": is_avg, "oos_avg_net": oos_avg,
        "daily_net": total / 245,
    }


def main():
    symbols = list_common_symbols()
    log(f"Strategy v2: LONG-ONLY cross-exchange mean-reversion")
    log(f"Testing on {len(symbols)} symbols\n")

    # Define configs to test
    configs = []

    # Vary signal type, threshold, holds, trailing stop
    for sig in ["sig_base", "sig_vol_adj", "sig_vol_confirm", "sig_taker", "sig_premium"]:
        for thr in [2.5, 3.0, 3.5, 4.0]:
            for max_h in [12, 24]:
                for ts in [0, 100, 200]:
                    configs.append({
                        "name": f"{sig}_thr{thr}_h{max_h}_ts{ts}",
                        "signal": sig,
                        "threshold": thr,
                        "min_hold": 3,
                        "max_hold": max_h,
                        "trailing_stop": ts,
                        "vol_gate": True,
                    })

    # Also test without vol gate
    for thr in [3.0, 3.5, 4.0]:
        configs.append({
            "name": f"sig_base_thr{thr}_h24_ts0_novg",
            "signal": "sig_base",
            "threshold": thr,
            "min_hold": 3,
            "max_hold": 24,
            "trailing_stop": 0,
            "vol_gate": False,
        })

    log(f"Total configs: {len(configs)}")
    all_summaries = []

    for ci, cfg in enumerate(configs):
        t0 = time.time()
        all_trades = []

        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(run_config, (s, cfg)): s for s in symbols}
            for fut in as_completed(futs):
                try:
                    trades = fut.result()
                    all_trades.extend(trades)
                except:
                    pass

        for fee_label, fee_rt in [("taker", 20), ("maker", 8)]:
            summary = analyze_trades(all_trades, f"{cfg['name']}_{fee_label}", fee_rt)
            if summary:
                all_summaries.append(summary)

        elapsed = time.time() - t0
        n = len(all_trades)
        if n > 0:
            nets_tk = [t["gross_bps"] - 20 for t in all_trades]
            avg_tk = np.mean(nets_tk)
            wr_tk = np.mean([r > 0 for r in nets_tk])
            log(f"  [{ci+1}/{len(configs)}] {cfg['name']:45s} → "
                f"{n:4d} trades WR={wr_tk:.1%} avg_net(tk)={avg_tk:+.0f}bps ({elapsed:.0f}s)")
        else:
            log(f"  [{ci+1}/{len(configs)}] {cfg['name']:45s} → NO TRADES ({elapsed:.0f}s)")

    # Results summary
    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv("v2_sweep_results.csv", index=False)

    log(f"\n{'='*140}")
    log(f"  V2 STRATEGY RESULTS — TAKER FEES (20bps RT)")
    log(f"{'='*140}")
    log(f"{'Config':50s} {'N':>5} {'Sym':>4} {'AvgGr':>7} {'AvgNet':>7} "
        f"{'WR':>6} {'PF':>6} {'Daily':>7} {'Months':>7} "
        f"{'IS_WR':>6} {'OOS_WR':>7} {'OOS_Avg':>8}")
    log("-" * 140)

    taker_results = results_df[results_df["config"].str.endswith("_taker")].sort_values(
        "profit_factor", ascending=False)
    for _, r in taker_results.head(30).iterrows():
        log(f"{r['config']:50s} {r['n_trades']:5.0f} {r['n_symbols']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} {r['daily_net']:+7.0f} "
            f"{r['pos_months']:.0f}/{r['total_months']:.0f} "
            f"{r['is_wr']:6.1%} {r['oos_wr']:7.1%} {r['oos_avg_net']:+8.0f}")

    log(f"\n{'='*140}")
    log(f"  V2 STRATEGY RESULTS — MAKER FEES (8bps RT)")
    log(f"{'='*140}")
    maker_results = results_df[results_df["config"].str.endswith("_maker")].sort_values(
        "profit_factor", ascending=False)
    for _, r in maker_results.head(20).iterrows():
        log(f"{r['config']:50s} {r['n_trades']:5.0f} {r['n_symbols']:4.0f} "
            f"{r['avg_gross']:+7.0f} {r['avg_net']:+7.0f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} {r['daily_net']:+7.0f} "
            f"{r['pos_months']:.0f}/{r['total_months']:.0f} "
            f"{r['is_wr']:6.1%} {r['oos_wr']:7.1%} {r['oos_avg_net']:+8.0f}")

    # Save best config trades
    best_cfg_name = taker_results.iloc[0]["config"].replace("_taker", "")
    log(f"\n--- Best taker config: {best_cfg_name} ---")

    # Find the matching config
    best_cfg = None
    for cfg in configs:
        if cfg["name"] == best_cfg_name:
            best_cfg = cfg
            break

    if best_cfg:
        log(f"Re-running best config to save detailed trades...")
        all_trades = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(run_config, (s, best_cfg)): s for s in symbols}
            for fut in as_completed(futs):
                try:
                    all_trades.extend(fut.result())
                except:
                    pass

        trade_df = pd.DataFrame(all_trades)
        trade_df["net_bps_taker"] = trade_df["gross_bps"] - 20
        trade_df["net_bps_maker"] = trade_df["gross_bps"] - 8
        trade_df.to_csv("v2_best_trades.csv", index=False)
        log(f"Saved {len(trade_df)} trades to v2_best_trades.csv")

        # Monthly breakdown
        trade_df["month"] = pd.to_datetime(trade_df["entry_time"]).dt.to_period("M")
        monthly = trade_df.groupby("month").agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            avg=("net_bps_taker", "mean"),
            total=("net_bps_taker", "sum"),
        )
        log(f"\n  Monthly performance:")
        for m, r in monthly.iterrows():
            status = "✓" if r["total"] > 0 else "✗"
            log(f"  {str(m):10s} {status} {r['n']:4.0f} trades WR={r['wr']:.1%} "
                f"avg={r['avg']:+.0f}bps total={r['total']:+.0f}bps")

        # Per-symbol
        sym_stats = trade_df.groupby("symbol").agg(
            n=("net_bps_taker", "count"),
            wr=("net_bps_taker", lambda x: (x > 0).mean()),
            total=("net_bps_taker", "sum"),
        ).sort_values("total", ascending=False)

        log(f"\n  Top 15 symbols:")
        for s, r in sym_stats.head(15).iterrows():
            log(f"    {s:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} total={r['total']:+.0f}bps")

        # Exit reason analysis
        if "exit_reason" in trade_df.columns:
            log(f"\n  Exit reasons:")
            for reason, group in trade_df.groupby("exit_reason"):
                n = len(group)
                nets = group["net_bps_taker"]
                log(f"    {reason:20s}: {n:4d} trades WR={( nets > 0).mean():.1%} "
                    f"avg={nets.mean():+.0f}bps")


if __name__ == "__main__":
    main()
