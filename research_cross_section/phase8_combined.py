"""
Phase 8: Combined universe filter + regime filter.

Tests the cumulative effect of stacking:
  1. Universe filter  — exclude Majors + worst meme (105 coins)
  2. Regime filter    — trade only when signal_strength > θ AND funding_disp > θ
                        (thresholds from walk-forward training windows)

Variants:
  Baseline           — 131 coins, no filter
  UnivFilter         — 105 coins, no regime filter
  RegimeFilter       — 131 coins, regime filter only
  Combined           — 105 coins + regime filter
  Combined+Top60     — 60 coins + regime filter  (most optimistic; partial look-ahead in coin selection)

All walk-forward OOS (6mo train / 3mo OOS). Regime thresholds fit on training only.

Outputs:
  results/phase8_summary.csv
  results/phase8_monthly.csv
  results/phase8_equity.png
  results/phase8_monthly_heatmap.png
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
N_LONG           = 10
N_SHORT          = 10
FEE_MAKER_BPS    = 4
PERIODS_PER_YEAR = 365 * 3
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3

# ============================================================
# Universe definitions
# ============================================================

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}
WORST_MEME = {
    "PENGUUSDT","BANUSDT","RESOLVUSDT","WIFUSDT","MEMEUSDT",
    "SPXUSDT","APEXUSDT","SIGNUSDT",
}

# ============================================================
# Loaders
# ============================================================

def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.parquet")))
    data = {c: {} for c in cols}
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

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-3, 3)
    z[n < min_valid] = np.nan
    return z

def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()

def restrict(panel, keep):
    cols = [c for c in panel.columns if c in keep]
    return panel[cols]

# ============================================================
# Regime features
# ============================================================

def build_regime_features(panels, composite_8h):
    """signal_strength and funding_disp — the two winners from Phase 5."""
    rebal_idx   = composite_8h.index
    funding_8h  = to_8h(panels["funding"])
    feats = pd.DataFrame(index=rebal_idx)
    feats["signal_strength"] = composite_8h.std(axis=1)
    feats["funding_disp"]    = funding_8h.std(axis=1).reindex(rebal_idx)
    return feats

# ============================================================
# Simulation
# ============================================================

def sim(composite_8h, fwd_8h, mask=None,
        n_long=N_LONG, n_short=N_SHORT,
        fee_bps=FEE_MAKER_BPS, min_universe=20):
    fee_rt = fee_bps * 2 / 10000
    common = composite_8h.index.intersection(fwd_8h.index)
    if mask is not None:
        mask = mask.reindex(common).fillna(True)

    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        trading = True if mask is None else bool(mask.loc[ts])
        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe or not trading:
            records.append(dict(timestamp=ts, gross=0., turnover=0., net=0.))
            prev_longs = prev_shorts = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)
        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        gross = lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan
        if prev_longs | prev_shorts:
            changed  = len((longs - prev_longs) | (shorts - prev_shorts))
            turnover = changed / (n_long + n_short)
        else:
            turnover = 1.0
        net = (gross - turnover * fee_rt) if not np.isnan(gross) else np.nan
        records.append(dict(timestamp=ts, gross=gross, turnover=turnover, net=net))
        prev_longs, prev_shorts = longs, shorts

    return pd.DataFrame(records).set_index("timestamp")

def port_stats(net_series, label=""):
    s = net_series.replace(0, np.nan).dropna()
    if len(s) < 5:
        return dict(label=label, n=len(s))
    ar   = s.mean() * PERIODS_PER_YEAR
    av   = s.std()  * np.sqrt(PERIODS_PER_YEAR)
    sh   = ar / av  if av > 0 else np.nan
    down = s[s < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    so   = ar / down if down > 0 else np.nan
    cum  = (1 + s).cumprod()
    mdd  = (cum / cum.cummax() - 1).min()
    t    = (s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else np.nan
    return dict(label=label, n=len(s),
                mean_bps=round(s.mean()*10000, 2),
                ann_ret=round(ar*100, 2),
                ann_vol=round(av*100, 2),
                sharpe=round(sh, 3),
                sortino=round(so, 3),
                max_dd=round(mdd*100, 2),
                win_rate=round((s > 0).mean(), 3),
                t_stat=round(t, 2))

# ============================================================
# Walk-forward with optional regime filter
# ============================================================

def wf_run(composite_8h, fwd_8h_full, feats=None, label=""):
    """
    feats: if provided, fit regime thresholds in training, apply in OOS.
    Returns (oos_net Series, wf_rows list)
    """
    start = composite_8h.index.min()
    end   = composite_8h.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    all_oos, wf_rows = [], []

    for tr_s, tr_e, oo_s, oo_e in windows:
        fwd_slice = fwd_8h_full.reindex(composite_8h.index).clip(-0.99,3.0).replace([np.inf,-np.inf],np.nan)

        mask_oos = None
        active_pct = 1.0

        if feats is not None:
            # Fit both thresholds on training P&L
            net_tr = sim(composite_8h.loc[tr_s:tr_e],
                         fwd_slice.reindex(composite_8h.loc[tr_s:tr_e].index))["net"]
            net_tr = net_tr.replace(0, np.nan).dropna()

            masks_tr = {}
            thresholds = {}
            for feat_col, direction in [("signal_strength", 1), ("funding_disp", -1)]:
                # direction=1: trade when feat > θ (high is good)
                # direction=-1 unused here; both features: trade when HIGH
                f_tr = feats[feat_col].reindex(net_tr.index).dropna()
                if len(f_tr) < 20:
                    thresholds[feat_col] = -np.inf
                    continue
                ths  = np.percentile(f_tr, np.linspace(10, 90, 17))
                best_th, best_sh = None, -np.inf
                for th in ths:
                    m = f_tr > th          # trade when feature is above threshold
                    active = m.mean()
                    if active < 0.20 or active > 0.95:
                        continue
                    n_ = net_tr.reindex(m.index)[m]
                    if len(n_) < 20 or n_.std() == 0:
                        continue
                    sh = n_.mean() / n_.std() * np.sqrt(PERIODS_PER_YEAR)
                    if sh > best_sh:
                        best_sh = sh; best_th = th
                thresholds[feat_col] = best_th

            # Apply both thresholds in OOS (AND logic)
            oos_idx = composite_8h.loc[oo_s:oo_e].index
            m1 = feats["signal_strength"].reindex(oos_idx) > thresholds.get("signal_strength", -np.inf)
            m2 = feats["funding_disp"].reindex(oos_idx)    > thresholds.get("funding_disp", -np.inf)
            mask_oos  = (m1 & m2).fillna(True)
            active_pct = mask_oos.mean()

        pnl = sim(composite_8h.loc[oo_s:oo_e],
                  fwd_slice.reindex(composite_8h.loc[oo_s:oo_e].index),
                  mask=mask_oos)
        st = port_stats(pnl["net"], label=f"{oo_s.date()}")
        st["active_pct"] = round(active_pct, 3)
        wf_rows.append(st)
        all_oos.append(pnl["net"])

    combined = pd.concat(all_oos) if all_oos else pd.Series(dtype=float)
    return combined, wf_rows

def monthly_net(net, label):
    rows = []
    net = net.replace(0, np.nan).dropna()
    for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
        rows.append(dict(month=f"{yr}-{mo:02d}", label=label,
                         net_pct=round(((1+grp).prod()-1)*100, 2)))
    return pd.DataFrame(rows)

# ============================================================
# Plots
# ============================================================

STYLES = {
    "Baseline":          ("#d62728", 2.0, "-"),
    "UnivFilter":        ("#1f77b4", 1.8, "-"),
    "RegimeFilter":      ("#ff7f0e", 1.8, "--"),
    "Combined":          ("#2ca02c", 2.2, "-"),
    "Combined+Top60":    ("#9467bd", 2.0, "-"),
}

def plot_equity(pnl_dict, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.set_title("Combined Universe + Regime Filter — OOS Equity Curves",
                  fontsize=12, fontweight="bold")

    for label, net in pnl_dict.items():
        net = net.replace(0, np.nan).dropna()
        if len(net) < 5:
            continue
        cum = (1 + net).cumprod()
        dd  = cum / cum.cummax() - 1
        sty = STYLES.get(label, ("#aaa", 1.0, "-"))
        ax1.plot(cum.index, cum.values, color=sty[0], linewidth=sty[1],
                 linestyle=sty[2], label=label)
        ax2.plot(dd.index, dd.values*100, color=sty[0],
                 linewidth=sty[1]*0.7, linestyle=sty[2], alpha=0.85)

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"))
    ax1.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Cumulative return (log, OOS)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticklabels([])

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_monthly_heatmap(monthly_df, variants, out_path):
    pivot = monthly_df.pivot(index="label", columns="month", values="net_pct").fillna(0)
    pivot = pivot.loc[[v for v in variants if v in pivot.index]]
    months = sorted(pivot.columns)
    pivot  = pivot[months]

    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 10)
    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Net Return (%) — All Variants (OOS)", fontsize=11, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(months)):
            v = pivot.values[i, j]
            if abs(v) > 1:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > vmax * 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Monthly net %", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 8: Combined Filter Analysis")
    print("=" * 60)

    print("\nLoading panels...")
    panels = load_panels(["close", "funding", "mom_24h"])
    all_syms  = set(panels["close"].columns)
    univ_105  = all_syms - MAJORS - WORST_MEME

    # Top-60 from phase 6
    coin_path = os.path.join(RESULTS_DIR, "phase6_coin_pnl.csv")
    top60_syms = set(pd.read_csv(coin_path).nlargest(60, "total_contrib_bps")["symbol"]) & all_syms \
                 if os.path.exists(coin_path) else univ_105

    # Forward returns (full universe — restrict per variant later)
    fwd_raw = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw = fwd_raw.clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
    fwd_8h  = fwd_raw.reindex(to_8h(panels["close"]).index)

    def make_composite(keep):
        f8 = cs_zscore(to_8h(restrict(panels["funding"], keep)))
        m8 = cs_zscore(to_8h(restrict(panels["mom_24h"], keep)))
        return (f8.add(m8, fill_value=0)) / 2

    universes = {
        "Baseline":       all_syms,
        "UnivFilter":     univ_105,
        "RegimeFilter":   all_syms,
        "Combined":       univ_105,
        "Combined+Top60": top60_syms,
    }
    use_regime = {
        "Baseline":       False,
        "UnivFilter":     False,
        "RegimeFilter":   True,
        "Combined":       True,
        "Combined+Top60": True,
    }

    print(f"  Universes: Baseline={len(all_syms)}, UnivFilter={len(univ_105)}, Top60={len(top60_syms)}")

    all_stats   = []
    all_oos_pnl = {}
    all_monthly = []

    for name, keep in universes.items():
        print(f"\nRunning: {name} ({len(keep)} coins, regime={use_regime[name]})...")
        comp = make_composite(keep)
        fwd  = restrict(fwd_8h, keep)

        feats = build_regime_features(panels, comp) if use_regime[name] else None

        oos_pnl, wf_rows = wf_run(comp,
                                   fwd.reindex(comp.index).clip(-0.99,3).replace([np.inf,-np.inf],np.nan),
                                   feats=feats, label=name)

        st = port_stats(oos_pnl, label=name)
        st["variant"]      = name
        st["n_syms"]       = len(keep)
        st["regime_used"]  = use_regime[name]
        pos_win = sum(1 for r in wf_rows if r.get("sharpe", -99) > 0)
        avg_active = np.mean([r.get("active_pct", 1.0) for r in wf_rows])
        st["pos_windows"]  = pos_win
        st["avg_active"]   = round(avg_active, 3)

        all_stats.append(st)
        all_oos_pnl[name] = oos_pnl
        all_monthly.append(monthly_net(oos_pnl, name))

        print(f"  OOS: Sharpe={st.get('sharpe',np.nan):.3f}  "
              f"MaxDD={st.get('max_dd',np.nan):.1f}%  "
              f"AnnRet={st.get('ann_ret',np.nan):.0f}%  "
              f"PosWin={pos_win}/4  Active={avg_active:.0%}")

    # ---- Summary ----
    stats_df   = pd.DataFrame(all_stats)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "phase8_summary.csv"), index=False)
    monthly_df.to_csv(os.path.join(RESULTS_DIR, "phase8_monthly.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY (OOS walk-forward)")
    print("=" * 80)
    print(f"  {'Variant':<22} {'Syms':>5} {'Regime':>7} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'AnnRet':>8} {'Sortino':>8} {'Active':>7} {'PosWin':>8}")
    print("  " + "-" * 82)
    for _, r in stats_df.iterrows():
        print(f"  {r['variant']:<22} {r['n_syms']:>5} "
              f"{'Yes' if r['regime_used'] else 'No':>7} "
              f"{r.get('sharpe',np.nan):>8.3f} "
              f"{r.get('max_dd',np.nan):>8.1f}% "
              f"{r.get('ann_ret',np.nan):>8.0f}% "
              f"{r.get('sortino',np.nan):>8.2f} "
              f"{r['avg_active']:>7.0%} "
              f"{int(r['pos_windows']):>7}/4")

    # Monthly table
    pivot = monthly_df.pivot(index="label", columns="month", values="net_pct").fillna(0)
    order = [v for v in universes if v in pivot.index]
    pivot = pivot.loc[order]
    print("\n  Monthly OOS net returns (%):")
    print(pivot.round(1).to_string())

    # Improvement breakdown
    base_sh  = stats_df[stats_df["variant"]=="Baseline"]["sharpe"].values[0]
    comb_sh  = stats_df[stats_df["variant"]=="Combined"]["sharpe"].values[0]
    base_dd  = stats_df[stats_df["variant"]=="Baseline"]["max_dd"].values[0]
    comb_dd  = stats_df[stats_df["variant"]=="Combined"]["max_dd"].values[0]
    print(f"\n  Stacking improvement (Baseline → Combined):")
    print(f"    Sharpe:  {base_sh:.3f} → {comb_sh:.3f}  ({comb_sh-base_sh:+.3f})")
    print(f"    Max DD:  {base_dd:.1f}% → {comb_dd:.1f}%  ({comb_dd-base_dd:+.1f}pp)")

    # Plots
    print("\nGenerating plots...")
    plot_equity(all_oos_pnl,
                os.path.join(RESULTS_DIR, "phase8_equity.png"))
    plot_monthly_heatmap(monthly_df, list(universes.keys()),
                         os.path.join(RESULTS_DIR, "phase8_monthly_heatmap.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
