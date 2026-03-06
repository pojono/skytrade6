"""
Phase 3: Execution realism — rebalancing frequency, universe size,
vol-scaled sizing, maker fill rate, and capacity analysis.

Best combo from Phase 2b: funding + mom_24h (Combo A)

Tests:
  1. Rebal frequency: 8h vs 16h vs 24h
     (funding IC is strongest at fwd_24h — daily rebal may suit it better)
  2. Universe size: top N by volume (10, 20, 30, 50 per leg)
  3. Volatility-weighted position sizing vs equal-weight
  4. Maker fill rate sensitivity (100% → 60% partial fills)
  5. Capacity: at what AUM does market impact eat the edge?

Outputs:
  results/phase3_rebal_freq.csv    — frequency comparison
  results/phase3_universe_size.csv — N_long sensitivity
  results/phase3_vol_sizing.csv    — equal vs vol-scaled
  results/phase3_capacity.csv      — capacity analysis
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import time

warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
PERIODS_PER_YEAR = 365 * 3   # default: 8h bars/year; overridden per freq
FEE_MAKER        = 4    # bps per side
FEE_TAKER        = 10
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panels_fast(signals_dir, cols):
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
    data  = {c: {} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet", "")
        try:
            df = pd.read_parquet(fpath)
            for c in cols:
                if c in df.columns:
                    data[c][sym] = df[c]
        except Exception:
            pass
    panels = {}
    for c in cols:
        if data[c]:
            p = pd.DataFrame(data[c])
            p.index = pd.to_datetime(p.index, utc=True)
            panels[c] = p.sort_index()
    return panels


# ---------------------------------------------------------------------------
# Signal / return preparation for any rebal frequency
# ---------------------------------------------------------------------------

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-3, 3)
    z[n < min_valid] = np.nan
    return z


def build_combo_a(panels, freq="8h"):
    """funding + mom_24h composite, resampled to given freq."""
    fund = panels["funding"].resample(freq, closed="left", label="left").first()
    mom  = panels["mom_24h"].resample(freq, closed="left", label="left").first()
    fund_z = cs_zscore(fund)
    mom_z  = cs_zscore(mom)
    # Equal-weight composite
    comp = fund_z.add(mom_z, fill_value=0) / 2
    return comp


def compute_fwd(close_panel, n_hours, rebal_freq_hours, min_valid=5):
    """
    Compute n_hours forward return from the close panel,
    aligned to rebal timestamps (every rebal_freq_hours hours).
    """
    # Forward return at 1h resolution: fwd_Nh[t] = (close[t+N] - close[t]) / close[t]
    fwd_1h = close_panel.pct_change(n_hours, fill_method=None).shift(-n_hours)
    fwd_1h = fwd_1h.clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
    # Align to rebal timestamps
    rebal_idx = close_panel.resample(f"{rebal_freq_hours}h",
                                     closed="left", label="left").first().index
    return fwd_1h.reindex(rebal_idx)


def volume_rank(panels, freq="8h"):
    """
    At each rebal bar, rank symbols by recent avg volume.
    Returns a boolean mask: True = in top-volume universe.
    """
    vol = panels["volume"].resample(freq, closed="left", label="left").mean()
    # 30-day rolling avg volume per coin
    roll_vol = vol.rolling(90, min_periods=10).mean()  # 90 * 8h = 30 days
    return roll_vol


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def sim(sig_panel, fwd_panel, n_long=10, n_short=10,
        fee_bps=FEE_MAKER, vol_weights=None, fill_rate=1.0,
        vol_filter=None, top_n_by_vol=None, min_universe=20,
        periods_per_year=PERIODS_PER_YEAR):
    """
    Core portfolio simulation with optional:
      vol_weights   : DataFrame same shape as sig_panel (position weight = 1/vol)
      fill_rate     : fraction of limit orders that fill (maker fill rate)
      vol_filter    : DataFrame of rolling volume (to rank by liquidity)
      top_n_by_vol  : if set, only trade top N coins by volume at each bar
    """
    fee_rt  = fee_bps * 2 / 10000
    common  = sig_panel.index.intersection(fwd_panel.index)
    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = sig_panel.loc[ts].dropna()
        fwd = fwd_panel.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        # Volume filter: restrict to top N most liquid coins
        if vol_filter is not None and top_n_by_vol is not None:
            if ts in vol_filter.index:
                vols = vol_filter.loc[ts, sig.index].dropna()
                if len(vols) >= top_n_by_vol:
                    top_syms = vols.nlargest(top_n_by_vol).index
                    sig = sig.loc[sig.index.intersection(top_syms)]
                    fwd = fwd.loc[fwd.index.intersection(top_syms)]

        if len(sig) < min_universe:
            records.append(dict(timestamp=ts, gross=np.nan, net=np.nan,
                                turnover=np.nan, n=len(sig)))
            prev_longs = prev_shorts = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)

        # Get position weights (equal or vol-scaled)
        if vol_weights is not None and ts in vol_weights.index:
            wl = vol_weights.loc[ts, list(longs)].dropna()
            ws = vol_weights.loc[ts, list(shorts)].dropna()
            wl = wl / wl.sum() if wl.sum() > 0 else None
            ws = ws / ws.sum() if ws.sum() > 0 else None
        else:
            wl = ws = None

        if wl is not None:
            lr = (fwd.loc[wl.index] * wl).sum()
        else:
            lr = fwd.loc[list(longs)].mean()

        if ws is not None:
            sr = (fwd.loc[ws.index] * ws).sum()
        else:
            sr = fwd.loc[list(shorts)].mean()

        gross = (lr - sr) if not (np.isnan(lr) or np.isnan(sr)) else np.nan

        # Turnover
        all_prev = prev_longs | prev_shorts
        if all_prev:
            changed  = len((longs - prev_longs) | (shorts - prev_shorts))
            turnover = changed / (n_long + n_short)
        else:
            turnover = 1.0

        # Apply maker fill rate: fraction of orders fill at limit price
        # Unfilled orders assumed to fill at market (taker fee) or not at all
        # Simplified: effective fee = fill_rate * maker + (1-fill_rate) * taker
        eff_fee_bps = fill_rate * fee_bps + (1 - fill_rate) * FEE_TAKER
        eff_cost    = turnover * eff_fee_bps * 2 / 10000

        net = (gross - eff_cost) if not np.isnan(gross) else np.nan
        records.append(dict(timestamp=ts, gross=gross, net=net,
                            turnover=turnover, n=len(sig)))
        prev_longs, prev_shorts = longs, shorts

    df = pd.DataFrame(records).set_index("timestamp")
    return df


def port_stats(net, label="", ppy=PERIODS_PER_YEAR):
    s = net.dropna()
    if len(s) < 5:
        return dict(label=label)
    ar  = s.mean() * ppy
    av  = s.std()  * np.sqrt(ppy)
    sh  = ar / av  if av > 0 else np.nan
    dn  = s[s < 0].std() * np.sqrt(ppy)
    so  = ar / dn  if dn  > 0 else np.nan
    cum = (1 + s).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    t   = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
    return dict(label=label, n=len(s),
                gross_bps=0, net_bps=round(s.mean()*10000,2),
                ann_ret=round(ar*100,2), ann_vol=round(av*100,2),
                sharpe=round(sh,3), sortino=round(so,3),
                max_dd=round(mdd*100,2), win_rate=round((s>0).mean(),3),
                t_stat=round(t,2))


def walk_forward(sig_panel, fwd_panel, fee_bps=FEE_MAKER,
                 n_long=10, n_short=10, ppy=PERIODS_PER_YEAR,
                 vol_weights=None, fill_rate=1.0,
                 vol_filter=None, top_n_by_vol=None):
    start, end = sig_panel.index.min(), sig_panel.index.max()
    windows, t = [], start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    all_pnl, wf_rows = [], []
    for oo_s, oo_e in windows:
        sig_oos = sig_panel.loc[oo_s:oo_e]
        fwd_oos = fwd_panel.reindex(sig_oos.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
        vw_oos  = vol_weights.loc[oo_s:oo_e] if vol_weights is not None else None
        vf_oos  = vol_filter.loc[oo_s:oo_e] if vol_filter is not None else None
        if sig_oos.empty:
            continue
        pnl = sim(sig_oos, fwd_oos, n_long=n_long, n_short=n_short,
                  fee_bps=fee_bps, vol_weights=vw_oos, fill_rate=fill_rate,
                  vol_filter=vf_oos, top_n_by_vol=top_n_by_vol)
        st  = port_stats(pnl["net"], label=f"{oo_s.date()}–{oo_e.date()}", ppy=ppy)
        st["gross_bps"] = round(pnl["gross"].mean()*10000, 2)
        st["turnover"]  = round(pnl["turnover"].mean(), 3)
        wf_rows.append(st)
        all_pnl.append(pnl)

    combined = pd.concat(all_pnl) if all_pnl else pd.DataFrame()
    return combined, wf_rows


def wf_summary(wf_rows, label=""):
    if not wf_rows:
        return {}
    sh_vals = [r.get("sharpe", np.nan) for r in wf_rows]
    sh_vals = [v for v in sh_vals if not np.isnan(v)]
    return dict(
        label=label,
        avg_sharpe=round(np.mean(sh_vals), 3) if sh_vals else np.nan,
        min_sharpe=round(min(sh_vals), 3) if sh_vals else np.nan,
        pos_windows=sum(v > 0 for v in sh_vals),
        n_windows=len(sh_vals),
        avg_net=round(np.mean([r.get("net_bps",0) for r in wf_rows]), 2),
        avg_turn=round(np.mean([r.get("turnover",0) for r in wf_rows]), 3),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    cols = ["funding", "mom_24h", "close", "volume", "fwd_8h", "fwd_24h"]
    print("Loading panels...")
    panels = load_panels_fast(SIGNALS_DIR, cols)
    print(f"  Loaded: {list(panels.keys())}\n")

    all_rebal_rows   = []
    all_size_rows    = []
    all_sizing_rows  = []
    all_fill_rows    = []
    all_cap_rows     = []

    # =================================================================
    # TEST 1: Rebalancing frequency — 8h vs 16h vs 24h
    # =================================================================
    print("=" * 75)
    print("TEST 1 — Rebalancing frequency (funding + mom_24h, Maker)")
    print("=" * 75)
    print(f"\n{'Freq':<8} {'PPY':>5}  {'Gross':>8}  {'Net':>8}  {'Sharpe':>7}  "
          f"{'MaxDD':>7}  {'Turn':>6}  {'WF-Sharpe':>10}  {'WF-min':>8}")
    print("-" * 80)

    freq_configs = [
        ("8h",  8,  365 * 3),
        ("16h", 16, 365 * 3 // 2),
        ("24h", 24, 365),
    ]

    for freq_label, freq_h, ppy in freq_configs:
        comp  = build_combo_a(panels, freq=f"{freq_h}h")
        fwd   = compute_fwd(panels["close"], n_hours=freq_h,
                            rebal_freq_hours=freq_h)
        fwd   = fwd.reindex(comp.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)

        pnl   = sim(comp, fwd, fee_bps=FEE_MAKER)
        st    = port_stats(pnl["net"], ppy=ppy)
        st["gross_bps"] = round(pnl["gross"].mean()*10000, 2)
        st["turnover"]  = round(pnl["turnover"].mean(), 3)

        wf_pnl, wf_rows = walk_forward(comp, fwd, fee_bps=FEE_MAKER, ppy=ppy,
                                        n_long=10, n_short=10)
        ws = wf_summary(wf_rows)

        print(f"  {freq_label:<6} {ppy:>5}  "
              f"{st['gross_bps']:>8.2f}  {st.get('net_bps',0):>8.2f}  "
              f"{st.get('sharpe',0):>7.3f}  {st.get('max_dd',0):>7.1f}  "
              f"{st.get('turnover',0):>6.3f}  "
              f"{ws.get('avg_sharpe',0):>10.3f}  {ws.get('min_sharpe',0):>8.3f}")

        row = {**st, **ws, "freq": freq_label, "ppy": ppy}
        all_rebal_rows.append(row)

        # Per-window detail
        if wf_rows:
            for r in wf_rows:
                print(f"    {r['label']:<24} net={r.get('net_bps',0):>7.2f}bps  "
                      f"sh={r.get('sharpe',0):>6.3f}  turn={r.get('turnover',0):.3f}")
        print()

    pd.DataFrame(all_rebal_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase3_rebal_freq.csv"), index=False)

    # =================================================================
    # TEST 2: Universe size (N per leg: 5/10/15/20/30)
    # Use 24h rebal (best freq from test 1 or keep 8h if it's best)
    # =================================================================
    print("=" * 75)
    print("TEST 2 — Universe size: coins per leg (8h rebal, Maker)")
    print("=" * 75)
    print(f"\n{'N/leg':>6}  {'Gross':>8}  {'Net':>8}  {'Sharpe':>7}  "
          f"{'MaxDD':>7}  {'WF-Sharpe':>10}  {'WF-min':>8}")
    print("-" * 65)

    comp_8h = build_combo_a(panels, freq="8h")
    fwd_8h  = compute_fwd(panels["close"], n_hours=8, rebal_freq_hours=8)
    fwd_8h  = fwd_8h.reindex(comp_8h.index).clip(-0.99,3.0).replace([np.inf,-np.inf],np.nan)

    for n_per_leg in [5, 10, 15, 20, 30]:
        pnl  = sim(comp_8h, fwd_8h, n_long=n_per_leg, n_short=n_per_leg,
                   fee_bps=FEE_MAKER)
        st   = port_stats(pnl["net"])
        st["gross_bps"] = round(pnl["gross"].mean()*10000, 2)
        st["turnover"]  = round(pnl["turnover"].mean(), 3)
        wf_pnl, wf_rows = walk_forward(comp_8h, fwd_8h, n_long=n_per_leg,
                                       n_short=n_per_leg, fee_bps=FEE_MAKER)
        ws = wf_summary(wf_rows)
        print(f"  {n_per_leg:>5}   {st['gross_bps']:>8.2f}  {st.get('net_bps',0):>8.2f}  "
              f"{st.get('sharpe',0):>7.3f}  {st.get('max_dd',0):>7.1f}  "
              f"{ws.get('avg_sharpe',0):>10.3f}  {ws.get('min_sharpe',0):>8.3f}")
        all_size_rows.append({**st, **ws, "n_per_leg": n_per_leg})

    pd.DataFrame(all_size_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase3_universe_size.csv"), index=False)

    # =================================================================
    # TEST 3: Equal-weight vs volatility-scaled positions
    # =================================================================
    print("\n" + "=" * 75)
    print("TEST 3 — Equal-weight vs volatility-scaled positions (8h, Maker)")
    print("=" * 75)

    # Compute rolling 30-day realized vol per coin at 8h bars
    close_8h = panels["close"].resample("8h", closed="left", label="left").first()
    ret_8h   = close_8h.pct_change(1, fill_method=None)
    rvol_8h  = ret_8h.rolling(90, min_periods=20).std()  # 90 * 8h = 30 days
    # Weight = target_vol / coin_vol (inverse vol weighting)
    # Cap at 3x equal-weight to avoid excessive concentration
    inv_vol  = (1 / rvol_8h.replace(0, np.nan)).clip(upper=3.0)
    # Normalise cross-sectionally
    inv_vol_z = inv_vol.div(inv_vol.mean(axis=1), axis=0)

    for label, vw in [("Equal-weight", None), ("Vol-scaled (1/σ)", inv_vol_z)]:
        pnl  = sim(comp_8h, fwd_8h, fee_bps=FEE_MAKER, vol_weights=vw)
        st   = port_stats(pnl["net"], label=label)
        st["gross_bps"] = round(pnl["gross"].mean()*10000, 2)
        st["turnover"]  = round(pnl["turnover"].mean(), 3)
        wf_pnl, wf_rows = walk_forward(comp_8h, fwd_8h, fee_bps=FEE_MAKER,
                                       vol_weights=vw)
        ws = wf_summary(wf_rows)
        print(f"\n  {label}")
        print(f"    Gross={st['gross_bps']:.2f}bps  Net={st.get('net_bps',0):.2f}bps  "
              f"Sharpe={st.get('sharpe',0):.3f}  MaxDD={st.get('max_dd',0):.1f}%  "
              f"WF_Sharpe={ws.get('avg_sharpe',0):.3f}  WF_min={ws.get('min_sharpe',0):.3f}")
        all_sizing_rows.append({**st, **ws, "method": label})

    pd.DataFrame(all_sizing_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase3_vol_sizing.csv"), index=False)

    # =================================================================
    # TEST 4: Maker fill rate sensitivity
    # =================================================================
    print("\n" + "=" * 75)
    print("TEST 4 — Maker fill rate sensitivity (8h, N=10/leg)")
    print("  Unfilled fraction executes at taker rate (10 bps/side)")
    print("=" * 75)
    print(f"\n  {'Fill%':>6}  {'Eff fee/side':>12}  {'Net bps':>9}  {'Sharpe':>8}  {'WF-Sharpe':>10}")
    print("  " + "-" * 55)

    for fill_pct in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]:
        eff_fee = fill_pct * FEE_MAKER + (1 - fill_pct) * FEE_TAKER
        pnl = sim(comp_8h, fwd_8h, fee_bps=FEE_MAKER, fill_rate=fill_pct)
        st  = port_stats(pnl["net"])
        wf_pnl, wf_rows = walk_forward(comp_8h, fwd_8h,
                                       fee_bps=FEE_MAKER, fill_rate=fill_pct)
        ws = wf_summary(wf_rows)
        print(f"  {fill_pct*100:>5.0f}%  {eff_fee:>12.1f}  "
              f"{st.get('net_bps',0):>9.2f}  {st.get('sharpe',0):>8.3f}  "
              f"{ws.get('avg_sharpe',0):>10.3f}")
        all_fill_rows.append({**st, **ws, "fill_rate": fill_pct, "eff_fee_bps": eff_fee})

    pd.DataFrame(all_fill_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase3_fill_rate.csv"), index=False)

    # =================================================================
    # TEST 5: Capacity analysis
    # =================================================================
    print("\n" + "=" * 75)
    print("TEST 5 — Capacity: market impact vs AUM")
    print("  Impact model: sqrt(order_size / daily_volume) * 10 bps")
    print("  Portfolio: 10 long + 10 short positions, equal-weight")
    print("=" * 75)

    # Load avg daily volume for each symbol
    vol_panel = panels["volume"]  # 1h volume in USDT
    # Daily volume = sum of 24 hourly bars, rolling 30d avg
    daily_vol = vol_panel.resample("1D").sum()
    avg_daily_vol = daily_vol.mean()  # avg daily USDT volume per coin

    # Median of top-20 liquid coins (approximately our trading universe)
    top20_vol = avg_daily_vol.nlargest(20)
    med_daily_vol = top20_vol.median()
    print(f"\n  Median daily volume (top 20 coins): ${med_daily_vol/1e6:.1f}M USDT")
    print(f"  Min daily volume (top 20): ${top20_vol.min()/1e6:.1f}M USDT")

    print(f"\n  {'AUM ($)':>12}  {'Per pos ($)':>12}  {'Order/DailyVol':>15}  "
          f"{'Impact bps':>11}  {'Net after impact':>17}")
    print("  " + "-" * 75)

    base_gross = 31.5  # bps per 8h bar from Phase 2b
    base_net   = 27.3  # bps net (maker)
    turnover   = 0.62  # fraction of positions changed per rebal

    for aum in [100_000, 500_000, 1_000_000, 5_000_000,
                10_000_000, 50_000_000, 100_000_000]:
        pos_size    = aum / 20        # split across 20 positions (10L + 10S)
        order_frac  = pos_size * turnover / med_daily_vol
        # Square-root market impact model (standard assumption)
        impact_bps  = 10 * np.sqrt(order_frac)  # bps per trade
        # Impact is paid on both sides (entry + exit)
        total_impact = impact_bps * 2 * turnover  # × turnover since not all positions change
        net_after   = base_net - total_impact
        print(f"  ${aum:>11,.0f}  ${pos_size:>11,.0f}  "
              f"{order_frac*100:>14.2f}%  "
              f"{impact_bps:>11.1f}  "
              f"{net_after:>17.1f} bps")
        all_cap_rows.append(dict(aum=aum, pos_size=pos_size,
                                 order_frac_pct=order_frac*100,
                                 impact_bps=round(impact_bps,2),
                                 net_after_impact=round(net_after,2)))

    pd.DataFrame(all_cap_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase3_capacity.csv"), index=False)

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
