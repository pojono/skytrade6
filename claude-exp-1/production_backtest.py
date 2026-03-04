#!/usr/bin/env python3
"""
Production Strategy: Cross-Exchange Volatility-Conditioned Mean-Reversion

OPTIMIZED: 
- Data loaded once in parallel (ProcessPoolExecutor)
- Per-symbol trades computed ONCE per config (run full backtest, slice by month)
- Configs run in PARALLEL (ProcessPoolExecutor)
- Walk-forward: train on past months, test on next month, rolling

Features:
1. Monthly rolling walk-forward (no lookahead)
2. Symbol tiering from rolling profitability
3. Signal-strength position sizing
4. Maker/taker/hybrid fee models
5. Regime detection (vol + spread vol filters)
6. Production config JSON output
"""

import sys, time, os, json, pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# =============================================================================
# DATA LOADING + FEATURE COMPUTATION (parallelized, done once)
# =============================================================================

def process_symbol(sym):
    """Load data, compute features + regime, return numpy arrays."""
    try:
        df = load_symbol(sym)
        if df.empty or len(df) < 5000:
            return sym, None

        feat = compute_features(df)
        sig = compute_composite_signal(feat)
        feat["sig_base"] = sig
        feat = feat.iloc[300:]

        # Regime features
        mid = (feat["bb_close"] + feat["bn_close"]) / 2
        mid_ret = mid.pct_change()
        rvol_1h = mid_ret.rolling(12).std() * np.sqrt(288) * 100
        rvol_6h = mid_ret.rolling(72).std() * np.sqrt(288) * 100
        feat["rvol_ratio"] = rvol_1h / rvol_6h.replace(0, np.nan)

        pdiv = feat.get("price_div_bps", (feat["bb_close"] - feat["bn_close"]) / mid * 10000)
        sv_1h = pdiv.rolling(12).std()
        sv_6h = pdiv.rolling(72).std()
        feat["spread_vol_ratio"] = sv_1h / sv_6h.replace(0, np.nan)

        feat = feat.dropna(subset=["sig_base", "rvol_ratio"])
        if len(feat) < 3000:
            return sym, None

        # Pre-compute month periods for fast slicing
        months = feat.index.to_period("M")

        return sym, {
            "mid": ((feat["bb_close"].values + feat["bn_close"].values) / 2),
            "sig": feat["sig_base"].values,
            "rvol_ratio": feat["rvol_ratio"].values,
            "spread_vol_ratio": feat["spread_vol_ratio"].values,
            "months": months.values,  # array of Period objects
            "index": feat.index,
            "n": len(feat),
        }
    except Exception:
        return sym, None


# =============================================================================
# TRADE ENGINE — run once per symbol per config, returns ALL trades
# =============================================================================

def run_all_trades(mid, sig, rvol_ratio, spread_vol_ratio, idx, n, months,
                   config, symbol_tier="B"):
    """Run backtest over full history. Returns list of trade dicts with month tags."""
    sig_thr = config["sig_threshold"]
    vol_thr = config.get("vol_threshold", 0.0)
    sv_thr = config.get("spread_vol_threshold", 0.0)
    max_hold = config["max_hold"]
    min_hold = config.get("min_hold", 3)
    cooldown = config.get("cooldown", 3)
    maker_pct = config.get("maker_pct", 0.0)
    taker_rt = config.get("taker_fee_rt", 20)
    maker_rt = config.get("maker_fee_rt", 8)
    sizing = config.get("sizing_mode", "fixed")
    base_size = config.get("base_size", 1.0)
    blended_fee = maker_pct * maker_rt + (1 - maker_pct) * taker_rt
    tier_mult = {"A": 1.5, "B": 1.0, "C": 0.5}.get(symbol_tier, 1.0)

    trades = []
    in_trade = False
    entry_i = 0
    last_exit = -cooldown

    for i in range(n):
        if not in_trade:
            if i < last_exit + cooldown:
                continue
            if sig[i] >= -sig_thr:
                continue
            if vol_thr > 0 and (np.isnan(rvol_ratio[i]) or rvol_ratio[i] < vol_thr):
                continue
            if sv_thr > 0 and (np.isnan(spread_vol_ratio[i]) or spread_vol_ratio[i] < sv_thr):
                continue

            sig_strength = abs(sig[i])
            if sizing == "signal_scaled":
                size_mult = min(2.0, 0.5 + (sig_strength - sig_thr) / max(sig_thr, 0.01) * 0.5)
            else:
                size_mult = 1.0

            entry_i = i
            entry_size = base_size * size_mult * tier_mult
            in_trade = True
        else:
            bars = i - entry_i
            if bars < min_hold:
                continue
            exit_now = False
            exit_reason = ""
            if sig[i] >= 0:
                exit_now, exit_reason = True, "cross"
            elif bars >= max_hold:
                exit_now, exit_reason = True, "maxhold"
            elif sig[i] < -sig_thr * 2.0 and bars >= 6:
                exit_now, exit_reason = True, "reversal"

            if exit_now:
                gross = (mid[i] / mid[entry_i] - 1) * 10000
                net = gross - blended_fee
                trades.append({
                    "entry_i": entry_i,
                    "exit_i": i,
                    "entry_month": months[entry_i],
                    "gross_bps": gross,
                    "net_bps": net,
                    "fee_bps": blended_fee,
                    "sig_strength": abs(sig[entry_i]),
                    "hold_bars": bars,
                    "exit_reason": exit_reason,
                    "position_size": entry_size,
                    "pnl_sized": net * entry_size,
                    "rvol_ratio": rvol_ratio[entry_i] if not np.isnan(rvol_ratio[entry_i]) else 0,
                })
                in_trade = False
                last_exit = i

    return trades


# =============================================================================
# WALK-FORWARD FOR ONE CONFIG (called in parallel)
# =============================================================================

def walk_forward_one_config(args):
    """
    Monthly rolling walk-forward for a single config.
    
    For each test month M:
      - Train symbol whitelist on months before M
      - Test on month M using tiers from training
    
    args = (config_name, config, symbol_data_pickled_path)
    We receive symbol_data as a dict (shared via fork).
    """
    config_name, config, symbol_data = args
    
    # Gather all unique months across all symbols
    all_months = set()
    for data in symbol_data.values():
        all_months.update(set(data["months"]))
    months_sorted = sorted(all_months)
    
    if len(months_sorted) < 3:
        return config_name, None
    
    warmup_months = 2
    results_by_month = {}
    all_trades = []
    symbol_perf = defaultdict(list)

    # Pre-compute ALL trades for each symbol with tier=B (neutral sizing)
    # This is the key optimization: run backtest ONCE per symbol
    sym_all_trades = {}
    for sym, data in symbol_data.items():
        trades = run_all_trades(
            data["mid"], data["sig"], data["rvol_ratio"],
            data["spread_vol_ratio"], data["index"], data["n"],
            data["months"], config, symbol_tier="B"
        )
        sym_all_trades[sym] = trades

    for m_idx in range(warmup_months, len(months_sorted)):
        test_month = months_sorted[m_idx]
        train_months = set(months_sorted[:m_idx])

        # Phase 1: score symbols from training period trades
        symbol_scores = {}
        for sym, trades in sym_all_trades.items():
            train = [t for t in trades if t["entry_month"] in train_months]
            if len(train) >= 2:
                avg_net = np.mean([t["net_bps"] for t in train])
                wr = np.mean([1 if t["net_bps"] > 0 else 0 for t in train])
                score = avg_net * min(1.0, len(train) / 5) * wr
                symbol_scores[sym] = {"score": score, "avg_net": avg_net, "wr": wr, "n": len(train)}
            else:
                symbol_scores[sym] = {"score": 0, "avg_net": 0, "wr": 0.5, "n": 0}

        # Tier + blacklist
        scores_sorted = sorted(symbol_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        n_sym = len(scores_sorted)
        tier_a = set(s for s, _ in scores_sorted[:max(1, int(n_sym * 0.3))])
        tier_c = set(s for s, _ in scores_sorted[max(1, int(n_sym * 0.7)):])
        tier_b = set(s for s, _ in scores_sorted) - tier_a - tier_c
        blacklist = set(s for s, sc in symbol_scores.items()
                       if sc["avg_net"] < -50 and sc["n"] >= 3)

        # Phase 2: collect test-month trades with tier-adjusted sizing
        month_trades = []
        for sym, data in symbol_data.items():
            if sym in blacklist:
                continue

            tier = "A" if sym in tier_a else ("B" if sym in tier_b else "C")
            tier_mult = {"A": 1.5, "B": 1.0, "C": 0.5}[tier]

            # Filter pre-computed trades to this test month, adjust sizing
            for t in sym_all_trades[sym]:
                if t["entry_month"] == test_month:
                    # Re-scale sized PnL by tier
                    adjusted = dict(t)
                    adjusted["position_size"] = t["position_size"] * tier_mult
                    adjusted["pnl_sized"] = t["net_bps"] * adjusted["position_size"]
                    adjusted["symbol"] = sym
                    adjusted["tier"] = tier
                    adjusted["month"] = str(test_month)
                    adjusted["entry_time"] = str(data["index"][t["entry_i"]])
                    adjusted["exit_time"] = str(data["index"][t["exit_i"]])
                    month_trades.append(adjusted)
                    symbol_perf[sym].append(t["net_bps"])

        if month_trades:
            nets = [t["net_bps"] for t in month_trades]
            sized = [t["pnl_sized"] for t in month_trades]
            results_by_month[str(test_month)] = {
                "n_trades": len(month_trades),
                "n_symbols": len(set(t["symbol"] for t in month_trades)),
                "avg_net": np.mean(nets),
                "total_net": np.sum(nets),
                "wr": np.mean([1 if x > 0 else 0 for x in nets]),
                "total_sized": np.sum(sized),
                "whitelisted": len(symbol_scores) - len(blacklist),
                "blacklisted": len(blacklist),
                "tier_a": len(tier_a),
            }
        else:
            results_by_month[str(test_month)] = {
                "n_trades": 0, "n_symbols": 0, "avg_net": 0,
                "total_net": 0, "wr": 0, "total_sized": 0,
                "whitelisted": 0, "blacklisted": 0, "tier_a": 0,
            }

        all_trades.extend(month_trades)

    return config_name, {
        "monthly": results_by_month,
        "trades": all_trades,
        "symbol_performance": dict(symbol_perf),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    symbols = list_common_symbols()
    log(f"Production Backtest: loading {len(symbols)} symbols...")
    t0 = time.time()

    # Phase 1: load all data in parallel
    symbol_data = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(process_symbol, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym, data = fut.result()
            if data:
                symbol_data[sym] = data
            if done % 20 == 0 or done == len(symbols):
                log(f"  [{done}/{len(symbols)}] {len(symbol_data)} loaded ({time.time()-t0:.0f}s)")

    log(f"\nLoaded {len(symbol_data)} symbols in {time.time()-t0:.0f}s")

    # =========================================================================
    # CONFIG GRID
    # =========================================================================
    configs = {
        "conservative_taker":  {"sig_threshold": 3.0, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "conservative_sized":  {"sig_threshold": 3.0, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
        "moderate_taker":      {"sig_threshold": 2.5, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "moderate_sized":      {"sig_threshold": 2.5, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
        "aggressive_taker":    {"sig_threshold": 2.5, "vol_threshold": 2.0, "spread_vol_threshold": 1.3, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "aggressive_sized":    {"sig_threshold": 2.5, "vol_threshold": 2.0, "spread_vol_threshold": 1.3, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
        "conservative_maker":  {"sig_threshold": 3.0, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 1.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "moderate_maker":      {"sig_threshold": 2.5, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 1.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "aggressive_maker":    {"sig_threshold": 2.5, "vol_threshold": 2.0, "spread_vol_threshold": 1.3, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 1.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "fixed", "base_size": 1.0},
        "moderate_hybrid":     {"sig_threshold": 2.5, "vol_threshold": 1.5, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.5, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
        "aggressive_hybrid":   {"sig_threshold": 2.5, "vol_threshold": 2.0, "spread_vol_threshold": 1.3, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.5, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
        "wide_taker":          {"sig_threshold": 2.0, "vol_threshold": 2.0, "spread_vol_threshold": 0.0, "max_hold": 24, "min_hold": 3, "cooldown": 3, "maker_pct": 0.0, "taker_fee_rt": 20, "maker_fee_rt": 8, "sizing_mode": "signal_scaled", "base_size": 1.0},
    }

    # =========================================================================
    # Phase 2: run walk-forward for ALL configs in PARALLEL
    # =========================================================================
    log(f"\nRunning walk-forward on {len(configs)} configs in parallel...")
    t1 = time.time()

    config_results = {}
    # Use threads to share symbol_data (fork would copy, but data is read-only)
    # Actually with fork-based ProcessPoolExecutor the parent memory is COW-shared
    with ProcessPoolExecutor(max_workers=min(len(configs), 8)) as pool:
        futs = {}
        for name, cfg in configs.items():
            fut = pool.submit(walk_forward_one_config, (name, cfg, symbol_data))
            futs[fut] = name

        for fut in as_completed(futs):
            name, res = fut.result()
            if res and res["trades"]:
                nets = [t["net_bps"] for t in res["trades"]]
                wr = np.mean([1 if x > 0 else 0 for x in nets])
                log(f"  {name:30s}: {len(res['trades']):5d} trades, WR={wr:.1%}, "
                    f"avg={np.mean(nets):+.1f}bps ({time.time()-t1:.0f}s)")
                config_results[name] = res
            else:
                log(f"  {name:30s}: 0 trades ({time.time()-t1:.0f}s)")

    log(f"\nAll configs done in {time.time()-t1:.0f}s")

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    log(f"\n{'='*170}")
    log(f"  MONTHLY WALK-FORWARD — sorted by Ex-October Profit Factor")
    log(f"{'='*170}")
    log(f"{'Config':35s} {'N':>5} {'Sym':>4} {'AvgNet':>7} {'WR':>6} {'PF':>6} "
        f"{'Sized':>10} {'Mo+':>4} "
        f"{'noOct_N':>7} {'noOct_WR':>8} {'noOct_Avg':>9} {'noOct_PF':>8}")
    log("-" * 170)

    summary_rows = []
    for name, res in config_results.items():
        trades = res["trades"]
        if len(trades) < 10:
            continue

        nets = np.array([t["net_bps"] for t in trades])
        sized = np.array([t["pnl_sized"] for t in trades])
        wr = (nets > 0).mean()
        wins = nets[nets > 0].sum()
        losses = abs(nets[nets < 0].sum())
        pf = wins / losses if losses > 0 else 999

        months = res["monthly"]
        pos_months = sum(1 for m in months.values() if m["total_net"] > 0)

        no_oct = [t for t in trades if "2025-10" not in t.get("month", "")]
        no_oct_nets = np.array([t["net_bps"] for t in no_oct]) if no_oct else np.array([0])
        no_oct_wr = (no_oct_nets > 0).mean() if len(no_oct) > 0 else 0
        no_oct_avg = no_oct_nets.mean() if len(no_oct) > 0 else 0
        no_oct_wins = no_oct_nets[no_oct_nets > 0].sum()
        no_oct_losses = abs(no_oct_nets[no_oct_nets < 0].sum())
        no_oct_pf = no_oct_wins / no_oct_losses if no_oct_losses > 0 else 999

        nsym = len(set(t["symbol"] for t in trades))

        summary_rows.append({
            "config": name, "n_trades": len(trades), "n_symbols": nsym,
            "avg_net": nets.mean(), "wr": wr, "pf": pf,
            "total_sized": sized.sum(), "pos_months": pos_months,
            "total_months": len(months),
            "no_oct_n": len(no_oct), "no_oct_wr": no_oct_wr,
            "no_oct_avg": no_oct_avg, "no_oct_pf": no_oct_pf,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("no_oct_pf", ascending=False)
    summary_df.to_csv("production_sweep_results.csv", index=False)

    for _, r in summary_df.iterrows():
        log(f"{r['config']:35s} {r['n_trades']:5.0f} {r['n_symbols']:4.0f} "
            f"{r['avg_net']:+7.1f} {r['wr']:6.1%} {r['pf']:6.2f} "
            f"{r['total_sized']:+10.0f} {r['pos_months']:4.0f} "
            f"{r['no_oct_n']:7.0f} {r['no_oct_wr']:8.1%} "
            f"{r['no_oct_avg']:+9.1f} {r['no_oct_pf']:8.2f}")

    # =========================================================================
    # BEST CONFIG DETAIL
    # =========================================================================
    if not summary_rows:
        log("\nNo configs produced enough trades.")
        return

    best_name = summary_df.iloc[0]["config"]
    best_res = config_results[best_name]
    trades = best_res["trades"]
    df_trades = pd.DataFrame(trades)

    log(f"\n{'='*120}")
    log(f"  BEST CONFIG: {best_name}")
    log(f"{'='*120}")

    # Monthly detail
    log(f"\n  Monthly walk-forward:")
    cumulative = 0
    peak = 0
    max_dd = 0
    for month, stats in sorted(best_res["monthly"].items()):
        s = "✓" if stats["total_net"] > 0 else "✗"
        cumulative += stats["total_net"]
        peak = max(peak, cumulative)
        dd = cumulative - peak
        max_dd = min(max_dd, dd)
        oct = " ← OCT" if "2025-10" in month else ""
        log(f"    {month:10s} {s} {stats['n_trades']:4d} trades "
            f"WR={stats['wr']:.1%} avg={stats['avg_net']:+.0f}bps "
            f"total={stats['total_net']:+.0f}bps cum={cumulative:+.0f} "
            f"[wl={stats['whitelisted']} bl={stats['blacklisted']} A={stats['tier_a']}]"
            f"{oct}")

    log(f"\n  Cumulative: {cumulative:+.0f} bps | Max DD: {max_dd:+.0f} bps | "
        f"Calmar: {cumulative / abs(max_dd):.2f}" if max_dd < 0 else
        f"\n  Cumulative: {cumulative:+.0f} bps | No drawdown")

    # Ex-October per-symbol
    if len(df_trades) > 0:
        df_no_oct = df_trades[~df_trades["month"].str.contains("2025-10")]
        if len(df_no_oct) > 0:
            sym_stats = df_no_oct.groupby("symbol").agg(
                n=("net_bps", "count"),
                wr=("net_bps", lambda x: (x > 0).mean()),
                avg=("net_bps", "mean"),
                total=("net_bps", "sum"),
            ).sort_values("total", ascending=False)
            prof = (sym_stats["total"] > 0).sum()
            log(f"\n  Ex-Oct symbols: {prof}/{len(sym_stats)} profitable")
            log(f"  Top 10:")
            for s, r in sym_stats.head(10).iterrows():
                log(f"    {s:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} total={r['total']:+.0f}bps")
            log(f"  Bottom 5:")
            for s, r in sym_stats.tail(5).iterrows():
                log(f"    {s:20s} {r['n']:3.0f} trades WR={r['wr']:.1%} total={r['total']:+.0f}bps")

        # Tier performance
        if "tier" in df_no_oct.columns and len(df_no_oct) > 0:
            log(f"\n  Tier performance (ex-Oct):")
            for tier in ["A", "B", "C"]:
                sub = df_no_oct[df_no_oct["tier"] == tier]
                if len(sub) > 0:
                    log(f"    Tier {tier}: {len(sub):4d} trades WR={(sub['net_bps']>0).mean():.1%} "
                        f"avg={sub['net_bps'].mean():+.1f}bps sized={sub['pnl_sized'].sum():+.0f}")

        # Signal strength
        if len(df_no_oct) > 0:
            log(f"\n  Signal strength (ex-Oct):")
            for lo, hi in [(2, 3), (3, 4), (4, 5), (5, 7), (7, 999)]:
                sub = df_no_oct[(df_no_oct["sig_strength"] >= lo) & (df_no_oct["sig_strength"] < hi)]
                if len(sub) >= 3:
                    log(f"    sig {lo}-{hi}: {len(sub):4d} trades WR={(sub['net_bps']>0).mean():.1%} "
                        f"avg={sub['net_bps'].mean():+.1f}bps")

        # Exit reasons
        if len(df_no_oct) > 0 and "exit_reason" in df_no_oct.columns:
            log(f"\n  Exit reasons (ex-Oct):")
            for reason in sorted(df_no_oct["exit_reason"].unique()):
                sub = df_no_oct[df_no_oct["exit_reason"] == reason]
                log(f"    {reason:15s}: {len(sub):4d} trades WR={(sub['net_bps']>0).mean():.1%} "
                    f"avg={sub['net_bps'].mean():+.1f}bps")

    # Save trades
    if len(df_trades) > 0:
        df_trades.to_csv("production_best_trades.csv", index=False)
        log(f"\n  Saved {len(df_trades)} trades to production_best_trades.csv")

    # =========================================================================
    # PRODUCTION CONFIG JSON
    # =========================================================================
    best_cfg = configs[best_name]
    final_whitelist = []
    final_blacklist = []
    for sym, perf_list in best_res["symbol_performance"].items():
        total = sum(perf_list)
        if total > 0 and len(perf_list) >= 2:
            final_whitelist.append(sym)
        elif total < -100:
            final_blacklist.append(sym)

    prod_config = {
        "strategy_name": "cross_exchange_volcond_meanrev_v1",
        "config_name": best_name,
        "parameters": best_cfg,
        "direction": "LONG_ONLY",
        "exchange_pair": ["bybit", "binance"],
        "signal": {
            "type": "composite_zscore",
            "components": [
                {"feature": "price_div_z72", "weight": 3.0},
                {"feature": "price_div_z288", "weight": 2.0},
                {"feature": "premium_z72", "weight": 2.0},
                {"feature": "premium_z288", "weight": 1.5},
                {"feature": "price_div_ma12_z288", "weight": 1.5},
                {"feature": "oi_div_z288", "weight": 1.0},
                {"feature": "vol_ratio_z72", "weight": 0.5},
                {"feature": "ret_diff_sum12_z288", "weight": 1.0},
            ],
            "threshold": best_cfg["sig_threshold"],
            "entry_condition": "composite < -threshold AND rvol_ratio > vol_threshold",
        },
        "regime_filter": {
            "rvol_ratio_min": best_cfg.get("vol_threshold", 0),
            "rvol_ratio_calc": "std(mid_ret, 12bars) / std(mid_ret, 72bars)",
        },
        "execution": {
            "entry": "limit_order_with_taker_fallback" if best_cfg.get("maker_pct", 0) > 0 else "taker_market_order",
            "exit": "signal_cross_zero_or_maxhold",
            "max_hold_minutes": best_cfg["max_hold"] * 5,
            "bar_size_minutes": 5,
            "cooldown_minutes": best_cfg.get("cooldown", 3) * 5,
        },
        "position_sizing": {
            "mode": best_cfg.get("sizing_mode", "fixed"),
            "base_notional_usd": 10000,
            "max_notional_usd": 30000,
            "tier_multipliers": {"A": 1.5, "B": 1.0, "C": 0.5},
        },
        "symbol_whitelist": sorted(final_whitelist),
        "symbol_blacklist": sorted(final_blacklist),
        "backtest_summary": {
            "total_trades": int(summary_df.iloc[0]["n_trades"]),
            "win_rate": float(summary_df.iloc[0]["wr"]),
            "profit_factor": float(summary_df.iloc[0]["pf"]),
            "avg_net_bps": float(summary_df.iloc[0]["avg_net"]),
            "ex_oct_trades": int(summary_df.iloc[0]["no_oct_n"]),
            "ex_oct_pf": float(summary_df.iloc[0]["no_oct_pf"]),
            "ex_oct_wr": float(summary_df.iloc[0]["no_oct_wr"]),
            "ex_oct_avg_bps": float(summary_df.iloc[0]["no_oct_avg"]),
        },
    }

    with open("production_config.json", "w") as f:
        json.dump(prod_config, f, indent=2)
    log(f"\n  Production config → production_config.json")
    log(f"  Whitelist: {len(final_whitelist)} symbols | Blacklist: {len(final_blacklist)} symbols")
    log(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
