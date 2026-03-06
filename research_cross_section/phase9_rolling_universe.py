"""
Phase 9: Rolling coin selection — properly walk-forward, zero look-ahead.

The key fix: coin inclusion/exclusion is determined from TRAINING data only,
applied to OOS. Same walk-forward splits as before (6mo train / 3mo OOS).

Selection methods tested:
  M1  All coins (baseline, no selection)
  M2  Training: exclude coins with negative total contribution → OOS
  M3  Training: top-N coins by per-coin Sharpe → OOS  (N=60, 80, 100)
  M4  Training: exclude coins below -X bps contribution threshold
  M5  Training: exclude Majors only (structural argument, no data snooping)
  M6  M5 + M2 combined (structural exclusion + data-driven rolling)

For each method, report per-window:
  - Which coins were selected in training
  - OOS Sharpe, MaxDD, AnnRet
  - Stability: how much does the universe change window-to-window?

Outputs:
  results/phase9_rolling_summary.csv    per-method OOS stats
  results/phase9_rolling_windows.csv    per-window per-method stats
  results/phase9_universe_stability.csv how many coins change per roll
  results/phase9_equity.png             equity curves
  results/phase9_monthly_heatmap.png    monthly P&L heatmap
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

# Structural exclusions — based on fundamental reasoning, not data snooping:
# Majors are large-cap, mean-reverting at 8h scale, dominated by
# market-wide moves rather than cross-sectional funding/momentum effects.
MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}


# ============================================================
# Loaders / helpers
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
    return panel[cols] if cols else panel


# ============================================================
# Simulation with per-coin attribution (for training selection)
# ============================================================

def sim_with_attribution(composite_8h, fwd_8h,
                          n_long=N_LONG, n_short=N_SHORT,
                          fee_bps=FEE_MAKER_BPS, min_universe=20):
    fee_rt = fee_bps * 2 / 10000
    common = composite_8h.index.intersection(fwd_8h.index)
    bar_records, attr_records = [], []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]
        if len(sig) < min_universe:
            bar_records.append(dict(timestamp=ts, net=np.nan))
            prev_longs = prev_shorts = set()
            continue
        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)
        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        gross = lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan
        turnover = (len((longs-prev_longs)|(shorts-prev_shorts)) / (n_long+n_short)
                    if prev_longs|prev_shorts else 1.0)
        net = (gross - turnover * fee_rt) if not np.isnan(gross) else np.nan
        bar_records.append(dict(timestamp=ts, net=net))
        for sym in longs:
            r = fwd.get(sym, np.nan)
            attr_records.append(dict(timestamp=ts, symbol=sym, side="long",
                                     contribution=r/n_long if not np.isnan(r) else np.nan))
        for sym in shorts:
            r = fwd.get(sym, np.nan)
            attr_records.append(dict(timestamp=ts, symbol=sym, side="short",
                                     contribution=-r/n_short if not np.isnan(r) else np.nan))
        prev_longs, prev_shorts = longs, shorts

    bar_df  = pd.DataFrame(bar_records).set_index("timestamp")
    attr_df = pd.DataFrame(attr_records) if attr_records else pd.DataFrame(
        columns=["timestamp","symbol","side","contribution"])
    if not attr_df.empty:
        attr_df["timestamp"] = pd.to_datetime(attr_df["timestamp"], utc=True)
    return bar_df, attr_df


def sim_simple(composite_8h, fwd_8h,
               n_long=N_LONG, n_short=N_SHORT,
               fee_bps=FEE_MAKER_BPS, min_universe=20):
    fee_rt = fee_bps * 2 / 10000
    common = composite_8h.index.intersection(fwd_8h.index)
    records = []
    prev_longs, prev_shorts = set(), set()
    for ts in common:
        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]
        if len(sig) < min_universe:
            records.append(dict(timestamp=ts, net=np.nan))
            prev_longs = prev_shorts = set()
            continue
        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)
        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        gross = lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan
        turnover = (len((longs-prev_longs)|(shorts-prev_shorts))/(n_long+n_short)
                    if prev_longs|prev_shorts else 1.0)
        net = (gross - turnover * fee_rt) if not np.isnan(gross) else np.nan
        records.append(dict(timestamp=ts, net=net))
        prev_longs, prev_shorts = longs, shorts
    return pd.DataFrame(records).set_index("timestamp")


def port_stats(net_series, label=""):
    s = net_series.dropna()
    if len(s) < 5:
        return dict(label=label, n=len(s))
    ar   = s.mean() * PERIODS_PER_YEAR
    av   = s.std()  * np.sqrt(PERIODS_PER_YEAR)
    sh   = ar / av  if av > 0 else np.nan
    down = s[s < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    so   = ar / down if down > 0 else np.nan
    cum  = (1 + s).cumprod()
    mdd  = (cum / cum.cummax() - 1).min()
    t    = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
    return dict(label=label, n=len(s),
                mean_bps=round(s.mean()*10000, 2),
                ann_ret=round(ar*100, 2), ann_vol=round(av*100, 2),
                sharpe=round(sh, 3), sortino=round(so, 3),
                max_dd=round(mdd*100, 2), win_rate=round((s>0).mean(), 3),
                t_stat=round(t, 2))


# ============================================================
# Per-coin attribution from training data
# ============================================================

def coin_sharpe_from_training(attr_df, bar_df):
    """
    Returns Series: symbol → Sharpe of that coin's contribution in training period.
    """
    if attr_df.empty:
        return pd.Series(dtype=float)
    rows = []
    for sym, grp in attr_df.groupby("symbol"):
        by_bar = grp.groupby("timestamp")["contribution"].sum()
        if len(by_bar) < 10 or by_bar.std() == 0:
            continue
        sh = by_bar.mean() / by_bar.std() * np.sqrt(PERIODS_PER_YEAR)
        rows.append(dict(symbol=sym, sharpe=sh,
                         total_contrib=by_bar.sum(),
                         n_bars=len(by_bar)))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).set_index("symbol")
    return df["sharpe"]


def coin_contrib_from_training(attr_df):
    """Returns Series: symbol → total contribution (bps) in training period."""
    if attr_df.empty:
        return pd.Series(dtype=float)
    return attr_df.groupby("symbol")["contribution"].sum()


# ============================================================
# Walk-forward engine
# ============================================================

def select_universe_from_training(attr_df, all_syms, method, params):
    """
    Determine which coins to include in OOS based on training attribution.
    Returns set of symbol strings.
    """
    if method == "M1":
        return all_syms

    if method == "M5":
        return all_syms - MAJORS

    contrib = coin_contrib_from_training(attr_df)
    sharpe  = coin_sharpe_from_training(attr_df, None)

    if method == "M2":
        # Exclude coins with negative total contribution in training
        bad = set(contrib[contrib < 0].index)
        return (all_syms - bad)

    if method == "M3":
        # Top-N coins by training Sharpe
        n = params.get("n", 80)
        top = set(sharpe.nlargest(n).index) if len(sharpe) >= n else set(sharpe.index)
        return top & all_syms

    if method == "M4":
        # Exclude coins below -threshold bps total contribution
        threshold = params.get("threshold", -500)   # bps
        bad = set((contrib * 10000)[contrib * 10000 < threshold].index)
        return (all_syms - bad)

    if method == "M6":
        # Structural (M5) + exclude negatives (M2)
        bad = set(contrib[contrib < 0].index)
        return (all_syms - MAJORS - bad)

    return all_syms


def run_rolling_wf(panels_all, fwd_8h_full, method, params=None, label=""):
    """
    Full walk-forward with rolling coin selection.
    In each window:
      1. Run simulation on training set (all coins) → get attribution
      2. Select universe based on method (training data only)
      3. Run OOS simulation with selected universe
      4. Return OOS P&L
    """
    if params is None:
        params = {}
    all_syms = set(panels_all["funding"].columns)

    start = fwd_8h_full.index.min()
    end   = fwd_8h_full.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    oos_pnl_list = []
    window_rows  = []
    prev_universe = None

    for win_i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):

        # --- Training: build composite with ALL coins ---
        def make_comp(keep):
            f8 = cs_zscore(to_8h(restrict(panels_all["funding"], keep)))
            m8 = cs_zscore(to_8h(restrict(panels_all["mom_24h"], keep)))
            return (f8.add(m8, fill_value=0)) / 2

        comp_tr_all = make_comp(all_syms)
        fwd_tr = fwd_8h_full.reindex(comp_tr_all.loc[tr_s:tr_e].index) \
                             .clip(-0.99,3).replace([np.inf,-np.inf],np.nan)

        _, attr_tr = sim_with_attribution(
            comp_tr_all.loc[tr_s:tr_e],
            restrict(fwd_tr, all_syms)
        )

        # --- Select universe from training ---
        universe = select_universe_from_training(attr_tr, all_syms, method, params)

        # Ensure at least 2×N coins in universe
        if len(universe) < 2 * max(N_LONG, N_SHORT):
            universe = all_syms  # fallback

        # --- OOS: run simulation with selected universe ---
        comp_oos = make_comp(universe)
        fwd_oos  = restrict(fwd_8h_full, universe) \
                       .reindex(comp_oos.loc[oo_s:oo_e].index) \
                       .clip(-0.99,3).replace([np.inf,-np.inf],np.nan)

        pnl_oos = sim_simple(comp_oos.loc[oo_s:oo_e], fwd_oos)
        st = port_stats(pnl_oos["net"], label=f"{oo_s.date()}–{oo_e.date()}")

        # Universe stability
        if prev_universe is not None:
            added   = len(universe - prev_universe)
            removed = len(prev_universe - universe)
        else:
            added = removed = 0

        window_rows.append(dict(
            window=f"{oo_s.date()}–{oo_e.date()}",
            method=label,
            n_coins=len(universe),
            coins_added=added,
            coins_removed=removed,
            **{k: v for k, v in st.items() if k != "label"},
        ))

        oos_pnl_list.append(pnl_oos["net"])
        prev_universe = universe

        sharpe_str = f"{st.get('sharpe', np.nan):.3f}" if st.get('sharpe') is not None else "n/a"
        print(f"    [{label}] {oo_s.date()}–{oo_e.date()}: "
              f"N={len(universe)} (+{added}/-{removed})  "
              f"Sharpe={sharpe_str}  "
              f"MaxDD={st.get('max_dd', np.nan):.1f}%")

    combined = pd.concat(oos_pnl_list) if oos_pnl_list else pd.Series(dtype=float)
    return combined, window_rows


def monthly_net(net, label):
    rows = []
    net = net.dropna()
    for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
        rows.append(dict(month=f"{yr}-{mo:02d}", label=label,
                         net_pct=round(((1+grp).prod()-1)*100, 2)))
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

STYLES = {
    "M1-Baseline":      ("#d62728", 2.0, "-"),
    "M5-NoMajors":      ("#1f77b4", 1.8, "--"),
    "M2-ExclNeg":       ("#ff7f0e", 1.8, "-"),
    "M6-M5+ExclNeg":    ("#2ca02c", 2.2, "-"),
    "M3-Top80":         ("#9467bd", 1.6, "--"),
    "M3-Top60":         ("#8c564b", 1.6, "-."),
    "M4-ExclBadCoins":  ("#e377c2", 1.4, ":"),
}

def plot_equity(pnl_dict, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.set_title("Rolling Universe Selection — Zero Look-Ahead (OOS only)",
                  fontsize=12, fontweight="bold")

    for label, net in pnl_dict.items():
        net = net.dropna()
        if len(net) < 5:
            continue
        cum = (1 + net).cumprod()
        dd  = cum / cum.cummax() - 1
        sty = STYLES.get(label, ("#aaa", 1.0, "-"))
        ax1.plot(cum.index, cum.values, color=sty[0], linewidth=sty[1],
                 linestyle=sty[2], label=label)
        ax2.plot(dd.index, dd.values*100, color=sty[0], linewidth=max(sty[1]*0.6, 0.8),
                 linestyle=sty[2], alpha=0.8)

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"))
    ax1.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Cumulative return (log, OOS)")
    ax1.legend(loc="upper left", fontsize=8)
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


def plot_monthly_heatmap(monthly_df, methods, out_path):
    pivot = monthly_df.pivot(index="label", columns="month", values="net_pct").fillna(0)
    order = [m for m in methods if m in pivot.index]
    pivot = pivot.loc[order, sorted(pivot.columns)]
    vmax  = max(abs(pivot.values.max()), abs(pivot.values.min()), 10)
    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Net Return (%) — Rolling Universe Selection (OOS, zero look-ahead)",
                 fontsize=11, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if abs(v) > 1:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > vmax*0.55 else "black")
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
    print("Phase 9: Rolling Universe Selection (Zero Look-Ahead)")
    print("=" * 60)

    print("\nLoading panels...")
    panels = load_panels(["close", "funding", "mom_24h"])
    all_syms = set(panels["funding"].columns)

    fwd_raw = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw = fwd_raw.clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
    fwd_8h  = fwd_raw.reindex(to_8h(panels["close"]).index)

    print(f"  Universe: {len(all_syms)} coins, "
          f"{fwd_8h.index.min().date()} – {fwd_8h.index.max().date()}")

    # Methods to test
    methods = [
        ("M1-Baseline",     "M1", {}),
        ("M5-NoMajors",     "M5", {}),
        ("M2-ExclNeg",      "M2", {}),
        ("M6-M5+ExclNeg",   "M6", {}),
        ("M3-Top80",        "M3", {"n": 80}),
        ("M3-Top60",        "M3", {"n": 60}),
        ("M4-ExclBadCoins", "M4", {"threshold": -200}),
    ]

    all_oos_pnl = {}
    all_windows = []
    all_monthly = []
    summary_rows = []

    for label, method, params in methods:
        print(f"\n{'─'*50}")
        print(f"Method: {label}")
        oos_pnl, window_rows = run_rolling_wf(panels, fwd_8h, method, params, label)
        all_oos_pnl[label]   = oos_pnl
        all_windows.extend(window_rows)
        all_monthly.append(monthly_net(oos_pnl, label))

        st = port_stats(oos_pnl, label=label)
        pos_win = sum(1 for r in window_rows if (r.get("sharpe") or -99) > 0)
        st["method"]      = label
        st["pos_windows"] = pos_win
        summary_rows.append(st)

    # ---- Summary ----
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "phase9_rolling_summary.csv"), index=False)
    windows_df = pd.DataFrame(all_windows)
    windows_df.to_csv(os.path.join(RESULTS_DIR, "phase9_rolling_windows.csv"), index=False)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    monthly_df.to_csv(os.path.join(RESULTS_DIR, "phase9_monthly.csv"), index=False)

    print("\n" + "=" * 75)
    print("FINAL SUMMARY — Rolling universe (zero look-ahead, OOS walk-forward)")
    print("=" * 75)
    print(f"  {'Method':<22} {'Sharpe':>8} {'MaxDD':>8} {'AnnRet':>8} "
          f"{'Sortino':>8} {'WinRate':>8} {'PosWin':>8}")
    print("  " + "-" * 72)
    for _, r in summary_df.iterrows():
        print(f"  {r['method']:<22} "
              f"{r.get('sharpe', np.nan):>8.3f} "
              f"{r.get('max_dd', np.nan):>8.1f}% "
              f"{r.get('ann_ret', np.nan):>8.0f}% "
              f"{r.get('sortino', np.nan):>8.2f} "
              f"{r.get('win_rate', np.nan):>8.3f} "
              f"{int(r.get('pos_windows', 0)):>7}/4")

    print("\n  Monthly OOS net returns (%):")
    pivot = monthly_df.pivot(index="label", columns="month", values="net_pct").fillna(0)
    order = [m for m, _, _ in methods if m in pivot.index]
    print(pivot.loc[order, sorted(pivot.columns)].round(1).to_string())

    # Highlight improvement vs baseline
    base_sh = summary_df[summary_df["method"]=="M1-Baseline"]["sharpe"].values[0]
    base_dd = summary_df[summary_df["method"]=="M1-Baseline"]["max_dd"].values[0]
    print(f"\n  Δ vs Baseline (Sharpe / MaxDD):")
    for _, r in summary_df.iterrows():
        if r["method"] == "M1-Baseline":
            continue
        dsh = r.get("sharpe", 0) - base_sh
        ddd = r.get("max_dd", 0) - base_dd
        print(f"    {r['method']:<22}  Sharpe {dsh:+.3f}   MaxDD {ddd:+.1f}pp")

    print("\nGenerating plots...")
    plot_equity(all_oos_pnl,
                os.path.join(RESULTS_DIR, "phase9_equity.png"))
    plot_monthly_heatmap(monthly_df, [m for m, _, _ in methods],
                         os.path.join(RESULTS_DIR, "phase9_monthly_heatmap.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
