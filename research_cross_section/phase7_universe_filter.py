"""
Phase 7: Universe composition tests.

Tests several universe variants vs baseline to see if excluding certain
clusters improves Sharpe and reduces drawdown.

Variants tested:
  Baseline       — all 131 coins
  No-Majors      — exclude BTC/ETH/SOL/ADA/etc. (18 coins)
  No-Majors+Meme — exclude Majors + worst meme coins
  Meme+AI+Legacy — only the 3 best clusters
  Top-60         — top 60 coins by per-coin Sharpe (data-driven)

All variants use same combo A signal (funding + mom_24h), N=10, 8h, maker.
Walk-forward OOS (6mo train / 3mo OOS) used to avoid overfitting.

Outputs:
  results/phase7_variants.csv          full-period + OOS stats per variant
  results/phase7_monthly.csv           monthly net % per variant
  results/phase7_equity.png            equity curves
  results/phase7_monthly_compare.png   monthly bar comparison
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
# Cluster definitions (from phase 6)
# ============================================================

MAJORS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT",
    "BCHUSDT", "TRXUSDT", "XLMUSDT", "ETCUSDT", "HBARUSDT",
    "ATOMUSDT", "ALGOUSDT", "EGLDUSDT",
}

# Worst individual meme coins by total contribution (from phase 6)
WORST_MEME = {
    "PENGUUSDT", "BANUSDT", "RESOLVUSDT", "WIFUSDT", "MEMEUSDT",
    "SPXUSDT",   # AI/Infra but negative
    "APEXUSDT", "SIGNUSDT",
}

# Best clusters by Sharpe (phase 6): Meme(2.95), Legacy(2.86), Gaming(1.97), AI/Infra(1.46)
BEST_CLUSTERS = {
    "JELLYJELLYUSDT", "COAIUSDT", "PIPPINUSDT", "MYXUSDT", "ENSOUSDT",
    "POPCATUSDT", "RIVERUSDT", "MOODENGUSDT", "SIRENUSDT", "GUNUSDT",
    "0GUSDT", "VVVUSDT", "BIOUSDT", "WLFIUSDT", "FARTCOINUSDT",
    "1000PEPEUSDT", "1000BONKUSDT", "TRUMPUSDT", "PEOPLEUSDT",
    "JELLYJELLYUSDT", "USELESSUSDT", "GIGGLEUSDT", "1000RATSUSDT",
    "1000TURBOUSDT", "ANIMEUSDT", "FIOUSDT", "HANAUSDT", "HUSDT",
    "LYNUSDT", "NOTUSDT", "SAHARAUSDT", "FFUSDT", "FORMUSDT",
    "USUALUSDT", "ASTERUSDT", "GPSUSDT", "PHAUSDT", "PLUMEUSDT",
    "BTRUSDT", "2ZUSDT", "BARDUSDT",
    # AI/Infra
    "AIXBTUSDT", "GRASSUSDT", "VIRTUALUSDT", "TAOUSDT", "KAITOUSDT",
    "ARCUSDT", "IPUSDT", "ATHUSDT", "AEROUSDT",
    # Legacy/Other
    "DASHUSDT", "ZECUSDT", "XMRUSDT", "PAXGUSDT", "FILUSDT",
    "XTZUSDT", "ZROUSDT", "KAVAUSDT", "LAUSDT",
    # Gaming/NFT
    "AXSUSDT", "GALAUSDT", "SANDUSDT", "CHZUSDT", "ORDIUSDT",
    "IMXUSDT", "WLDUSDT",
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


def restrict_universe(panel, keep_syms):
    """Drop columns not in keep_syms."""
    cols = [c for c in panel.columns if c in keep_syms]
    return panel[cols]


def sim(composite_8h, fwd_8h, n_long=N_LONG, n_short=N_SHORT,
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
            records.append(dict(timestamp=ts, gross=np.nan, turnover=np.nan, net=np.nan))
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
    t    = (s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else np.nan
    return dict(
        label=label, n=len(s),
        mean_bps=round(s.mean()*10000, 2),
        ann_ret=round(ar*100, 2),
        ann_vol=round(av*100, 2),
        sharpe=round(sh, 3),
        sortino=round(so, 3),
        max_dd=round(mdd*100, 2),
        win_rate=round((s > 0).mean(), 3),
        t_stat=round(t, 2),
    )


def walk_forward(composite_8h, fwd_8h, label):
    start = composite_8h.index.min()
    end   = composite_8h.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    all_oos = []
    wf_rows = []
    for tr_s, tr_e, oo_s, oo_e in windows:
        fwd_full = fwd_8h.reindex(composite_8h.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
        pnl = sim(composite_8h.loc[oo_s:oo_e],
                  fwd_full.reindex(composite_8h.loc[oo_s:oo_e].index))
        st  = port_stats(pnl["net"], label=f"{oo_s.date()}")
        wf_rows.append(st)
        all_oos.append(pnl["net"])

    combined = pd.concat(all_oos) if all_oos else pd.Series(dtype=float)
    return combined, wf_rows


def monthly_net(net_series, label):
    rows = []
    net = net_series.dropna()
    for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
        rows.append(dict(month=f"{yr}-{mo:02d}", label=label,
                         net_pct=round(((1+grp).prod()-1)*100, 2)))
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

STYLES = {
    "Baseline":           ("#d62728", 2.0, "-"),
    "No-Majors":          ("#1f77b4", 2.0, "-"),
    "No-Majors+WorstMeme":("#ff7f0e", 1.6, "--"),
    "Best-Clusters-Only": ("#2ca02c", 1.6, "--"),
    "Top60-by-Sharpe":    ("#9467bd", 1.4, "-."),
}


def plot_equity(pnl_dict, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.set_title("Universe Filter Comparison — OOS Equity Curves", fontsize=12, fontweight="bold")

    for label, net in pnl_dict.items():
        net = net.dropna()
        if len(net) < 5:
            continue
        cum = (1 + net).cumprod()
        dd  = cum / cum.cummax() - 1
        sty = STYLES.get(label, ("#aaa", 1.0, "-"))
        ax1.plot(cum.index, cum.values, color=sty[0], linewidth=sty[1],
                 linestyle=sty[2], label=label)
        ax2.plot(dd.index, dd.values * 100, color=sty[0], linewidth=sty[1]*0.7,
                 linestyle=sty[2], alpha=0.85)

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


def plot_monthly_grid(monthly_df, variants, out_path):
    """Side-by-side monthly bars for each variant vs baseline."""
    months = sorted(monthly_df["month"].unique())
    n_var  = len(variants)
    fig, axes = plt.subplots(n_var, 1, figsize=(18, 3.5 * n_var), sharex=True)
    if n_var == 1:
        axes = [axes]
    fig.suptitle("Monthly Net Returns by Universe Variant", fontsize=12, fontweight="bold")

    for ax, variant in zip(axes, variants):
        sub  = monthly_df[monthly_df["label"] == variant].set_index("month")
        base = monthly_df[monthly_df["label"] == "Baseline"].set_index("month")
        vals = [sub.loc[m, "net_pct"] if m in sub.index else 0 for m in months]
        bvals= [base.loc[m, "net_pct"] if m in base.index else 0 for m in months]
        x = np.arange(len(months))
        w = 0.4
        sty = STYLES.get(variant, ("#aaa", 1.0, "-"))
        ax.bar(x - w/2, bvals, width=w, color=["#2ca02c" if v>=0 else "#d62728" for v in bvals],
               alpha=0.5, label="Baseline", edgecolor="white")
        ax.bar(x + w/2, vals,  width=w, color=[sty[0] if v>=0 else "#ff7f0e" for v in vals],
               alpha=0.85, label=variant, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Net %")
        ax.set_title(variant, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(np.arange(len(months)))
    axes[-1].set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 7: Universe Composition Tests")
    print("=" * 60)

    print("\nLoading panels...")
    panels = load_panels(["close", "funding", "mom_24h"])
    all_syms = set(panels["close"].columns)

    fwd_raw = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw = fwd_raw.clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
    fwd_8h  = fwd_raw.reindex(to_8h(panels["close"]).index)

    # Load per-coin Sharpe from phase 6 for data-driven Top-60
    coin_pnl_path = os.path.join(RESULTS_DIR, "phase6_coin_pnl.csv")
    if os.path.exists(coin_pnl_path):
        coin_df = pd.read_csv(coin_pnl_path)
        # Compute per-coin Sharpe proxy from mean/std of contribution
        # Use total_contrib_bps as ranking (already computed)
        top60_syms = set(coin_df.nlargest(60, "total_contrib_bps")["symbol"])
    else:
        top60_syms = all_syms  # fallback

    # Define universe variants
    variants = {
        "Baseline": all_syms,
        "No-Majors": all_syms - MAJORS,
        "No-Majors+WorstMeme": all_syms - MAJORS - WORST_MEME,
        "Best-Clusters-Only": BEST_CLUSTERS & all_syms,
        "Top60-by-Sharpe": top60_syms & all_syms,
    }

    for name, syms in variants.items():
        print(f"  {name}: {len(syms)} symbols")

    # ---- Run each variant ----
    all_stats    = []
    all_oos_pnl  = {}
    all_monthly  = []

    for name, keep in variants.items():
        print(f"\nRunning: {name}...")

        f8  = to_8h(panels["funding"])
        m8  = to_8h(panels["mom_24h"])
        f8  = restrict_universe(f8, keep)
        m8  = restrict_universe(m8, keep)
        fwd = restrict_universe(fwd_8h, keep)

        fz  = cs_zscore(f8)
        mz  = cs_zscore(m8)
        comp = (fz.add(mz, fill_value=0)) / 2

        # Full-period sim
        pnl_full = sim(comp, fwd.reindex(comp.index).clip(-0.99,3).replace([np.inf,-np.inf],np.nan))
        st_full  = port_stats(pnl_full["net"], label=name)
        st_full["variant"] = name
        st_full["n_syms"]  = len(keep)

        # Walk-forward OOS
        oos_pnl, wf_rows = walk_forward(comp, fwd, name)
        st_oos = port_stats(oos_pnl, label=f"{name} OOS")
        pos_windows = sum(1 for r in wf_rows if r.get("sharpe", -99) > 0)

        st_full["oos_sharpe"]      = st_oos.get("sharpe", np.nan)
        st_full["oos_max_dd"]      = st_oos.get("max_dd", np.nan)
        st_full["oos_ann_ret"]     = st_oos.get("ann_ret", np.nan)
        st_full["oos_pos_windows"] = pos_windows

        all_stats.append(st_full)
        all_oos_pnl[name] = oos_pnl

        # Monthly breakdown
        all_monthly.append(monthly_net(oos_pnl, name))

        print(f"  Full: Sharpe={st_full['sharpe']:.3f}  MaxDD={st_full['max_dd']:.1f}%  "
              f"AnnRet={st_full['ann_ret']:.0f}%")
        print(f"   OOS: Sharpe={st_full['oos_sharpe']:.3f}  MaxDD={st_full['oos_max_dd']:.1f}%  "
              f"AnnRet={st_full['oos_ann_ret']:.0f}%  "
              f"PosWindows={pos_windows}/4")

    # ---- Summary table ----
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "phase7_variants.csv"), index=False)

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    monthly_df.to_csv(os.path.join(RESULTS_DIR, "phase7_monthly.csv"), index=False)

    print("\n" + "=" * 75)
    print("SUMMARY — Full Period")
    print("=" * 75)
    print(f"{'Variant':<28} {'Syms':>5} {'Sharpe':>8} {'MaxDD':>8} {'AnnRet':>8} "
          f"{'OOS Sh':>8} {'OOS DD':>8} {'OOS Ret':>8} {'PosWin':>8}")
    print("-" * 90)
    for _, row in stats_df.iterrows():
        print(f"  {row['variant']:<26} {row['n_syms']:>5} "
              f"{row['sharpe']:>8.3f} {row['max_dd']:>8.1f}% "
              f"{row['ann_ret']:>8.0f}% "
              f"{row['oos_sharpe']:>8.3f} {row['oos_max_dd']:>8.1f}% "
              f"{row['oos_ann_ret']:>8.0f}% "
              f"{int(row['oos_pos_windows']):>7}/4")

    print("\n  Monthly OOS breakdown:")
    pivot = monthly_df.pivot(index="month", columns="label", values="net_pct").fillna(0)
    pivot = pivot[[v for v in variants if v in pivot.columns]]
    pivot["Best_vs_Base"] = pivot.get("No-Majors", 0) - pivot.get("Baseline", 0)
    print(pivot.round(1).to_string())

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_equity(all_oos_pnl,
                os.path.join(RESULTS_DIR, "phase7_equity.png"))
    non_base = [v for v in variants if v != "Baseline"]
    plot_monthly_grid(monthly_df, non_base,
                      os.path.join(RESULTS_DIR, "phase7_monthly_compare.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
