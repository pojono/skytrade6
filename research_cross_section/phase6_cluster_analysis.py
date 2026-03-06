"""
Phase 6: Cluster performance analysis.

Tags all 131 universe coins into 7 clusters, then for every 8h bar tracks
which coins were longed/shorted and what return each contributed.

Outputs:
  results/phase6_cluster_pnl.csv          per-cluster P&L stats
  results/phase6_cluster_monthly.csv      cluster × month net return
  results/phase6_coin_pnl.csv             per-coin attribution summary
  results/phase6_cluster_bars.png         cluster P&L bar chart
  results/phase6_cluster_heatmap.png      cluster × month heatmap
  results/phase6_long_short_freq.png      how often each cluster is longed vs shorted
  results/phase6_coin_contribution.png    top/bottom 20 coins by total P&L contribution
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
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
N_LONG           = 10
N_SHORT          = 10
FEE_MAKER_BPS    = 4
PERIODS_PER_YEAR = 365 * 3


# ============================================================
# Cluster taxonomy
# ============================================================

CLUSTERS = {
    # --- Majors: BTC + ETH + large-cap layer-1s ---
    "Majors": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT",
        "BCHUSDT", "TRXUSDT", "XLMUSDT", "ETCUSDT", "HBARUSDT",
        "ATOMUSDT", "ALGOUSDT", "EGLDUSDT",
    ],
    # --- L1 / L2 scaling / modular ---
    "L1/L2": [
        "ARBUSDT", "OPUSDT", "STRKUSDT", "APTUSDT", "SUIUSDT",
        "SEIUSDT", "INJUSDT", "NEARUSDT", "TONUSDT", "TIAUSDT",
        "POLUSDT", "LINEAUSDT", "APTUSDT", "INITUSDT", "BERARUSDT",
        "BERAUSDT", "AKTUSDT", "AVNTUSDT",
    ],
    # --- DeFi / DEX / lending / derivatives ---
    "DeFi": [
        "AAVEUSDT", "UNIUSDT", "CRVUSDT", "LDOUSDT", "DYDXUSDT",
        "SNXUSDT", "ENAUSDT", "ONDOUSDT", "JUPUSDT", "JTOUSDT",
        "CAKEUSDT", "ETHFIUSDT", "RENDERUSDT", "MORPHOUSDT",
        "PYTHUSDT", "AUCTIONUSDT", "LINKUSDT", "EIGENUSDT",
        "AGLDUSDT", "CYBERUSDT", "DYDXUSDT",
    ],
    # --- AI / infra / data ---
    "AI/Infra": [
        "AIXBTUSDT", "GRASSUSDT", "VIRTUALUSDT", "TAOUSDT",
        "KAITOUSDT", "ICPUSDT", "ARCUSDT", "COAIUSDT", "IPUSDT",
        "ATHUSDT", "AEROUSDT", "SPXUSDT", "SIGNUSDT",
    ],
    # --- Meme / narrative / community ---
    "Meme": [
        "FARTCOINUSDT", "1000PEPEUSDT", "1000BONKUSDT", "WIFUSDT",
        "POPCATUSDT", "PENGUUSDT", "MOODENGUSDT", "TRUMPUSDT",
        "WLFIUSDT", "PEOPLEUSDT", "MEMEUSDT", "JELLYJELLYUSDT",
        "USELESSUSDT", "GIGGLEUSDT", "1000RATSUSDT", "1000TURBOUSDT",
        "PIPPINUSDT", "0GUSDT", "2ZUSDT", "BANUSDT", "BARDUSDT",
        "ANIMEUSDT", "FIOUSDT", "GUNUSDT", "HANAUSDT", "HUSDT",
        "LYNUSDT", "MYXUSDT", "NOTUSDT", "SAHARAUSDT", "SIRENUSDT",
        "XPLUSDT", "RESOLVUSDT", "RIVERUSDT", "ENSOUSDT",
        "APEXUSDT", "BTRUSDT", "FFUSDT", "FORMUSDT", "VVVUSDT",
        "USUALUSDT", "ASTERUSDT", "BIOUSDT", "GPSUSDT", "PHAUSDT",
        "PLUMEUSDT",
    ],
    # --- Gaming / NFT / metaverse ---
    "Gaming/NFT": [
        "AXSUSDT", "GALAUSDT", "SANDUSDT", "CHZUSDT", "ORDIUSDT",
        "IMXUSDT", "WLDUSDT",
    ],
    # --- Legacy / store-of-value / privacy ---
    "Legacy/Other": [
        "DASHUSDT", "XMRUSDT", "ZECUSDT", "PAXGUSDT", "FILUSDT",
        "XTZUSDT", "ZROUSDT", "KAVAUSDT", "LAUSDT",
    ],
}

# Build reverse lookup: symbol → cluster
SYM_TO_CLUSTER = {}
for cluster, syms in CLUSTERS.items():
    for s in syms:
        SYM_TO_CLUSTER[s] = cluster

CLUSTER_ORDER  = ["Majors", "L1/L2", "DeFi", "AI/Infra", "Meme", "Gaming/NFT", "Legacy/Other"]
CLUSTER_COLORS = {
    "Majors":       "#1f77b4",
    "L1/L2":        "#ff7f0e",
    "DeFi":         "#2ca02c",
    "AI/Infra":     "#9467bd",
    "Meme":         "#e377c2",
    "Gaming/NFT":   "#8c564b",
    "Legacy/Other": "#7f7f7f",
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


# ============================================================
# Simulation with per-coin attribution
# ============================================================

def sim_with_attribution(composite_8h, fwd_8h,
                          n_long=N_LONG, n_short=N_SHORT,
                          fee_bps=FEE_MAKER_BPS, min_universe=20):
    """
    Returns:
      bar_df    — per-bar summary (timestamp, gross, net, turnover)
      attr_df   — per-coin-per-bar attribution (timestamp, symbol, side, ret, contribution)
    """
    fee_rt = fee_bps * 2 / 10000
    common = composite_8h.index.intersection(fwd_8h.index)

    bar_records  = []
    attr_records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe:
            bar_records.append(dict(timestamp=ts, gross=np.nan,
                                    turnover=np.nan, net=np.nan))
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
        bar_records.append(dict(timestamp=ts, gross=gross,
                                turnover=turnover, net=net))

        # Per-coin attribution: each long contributes +ret/N, each short -ret/N
        for sym in longs:
            r = fwd.get(sym, np.nan)
            attr_records.append(dict(timestamp=ts, symbol=sym,
                                     side="long", ret=r,
                                     contribution=r / n_long if not np.isnan(r) else np.nan))
        for sym in shorts:
            r = fwd.get(sym, np.nan)
            attr_records.append(dict(timestamp=ts, symbol=sym,
                                     side="short", ret=r,
                                     contribution=-r / n_short if not np.isnan(r) else np.nan))

        prev_longs, prev_shorts = longs, shorts

    bar_df  = pd.DataFrame(bar_records).set_index("timestamp")
    attr_df = pd.DataFrame(attr_records)
    if not attr_df.empty:
        attr_df["timestamp"] = pd.to_datetime(attr_df["timestamp"], utc=True)
        attr_df["cluster"]   = attr_df["symbol"].map(SYM_TO_CLUSTER).fillna("Unknown")
    return bar_df, attr_df


# ============================================================
# Analysis functions
# ============================================================

def cluster_pnl_stats(attr_df, bar_df):
    """Aggregate P&L stats per cluster."""
    rows = []
    for cluster in CLUSTER_ORDER + ["Unknown"]:
        sub = attr_df[attr_df["cluster"] == cluster]
        if sub.empty:
            continue
        n_syms  = sub["symbol"].nunique()
        n_bars  = sub["timestamp"].nunique()
        n_long  = (sub["side"] == "long").sum()
        n_short = (sub["side"] == "short").sum()
        total_contrib = sub["contribution"].sum()
        mean_contrib  = sub["contribution"].mean()
        # annualised: total_contrib / n_bars_in_portfolio × PPY
        # use bar-level: sum contribution per bar then stats
        by_bar = sub.groupby("timestamp")["contribution"].sum()
        ann_ret = by_bar.mean() * PERIODS_PER_YEAR if len(by_bar) > 0 else np.nan
        ann_vol = by_bar.std() * np.sqrt(PERIODS_PER_YEAR) if len(by_bar) > 1 else np.nan
        sharpe  = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
        rows.append(dict(
            cluster=cluster,
            n_symbols=n_syms,
            n_long_slots=int(n_long),
            n_short_slots=int(n_short),
            long_pct=round(n_long / (n_long + n_short) * 100, 1) if (n_long + n_short) > 0 else np.nan,
            total_contrib_bps=round(total_contrib * 10000, 1),
            mean_contrib_bps=round(mean_contrib * 10000, 3),
            contrib_per_bar_bps=round(by_bar.mean() * 10000, 2),
            ann_sharpe=round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        ))
    return pd.DataFrame(rows)


def cluster_monthly(attr_df):
    """Cluster × month net contribution matrix."""
    sub = attr_df.copy()
    sub["year"]  = sub["timestamp"].dt.year
    sub["month"] = sub["timestamp"].dt.month
    sub["ym"]    = sub["timestamp"].dt.to_period("M").astype(str)

    pivot = sub.groupby(["cluster", "ym"])["contribution"].sum().unstack(fill_value=0)
    pivot = pivot * 100  # → %
    # reorder rows
    order = [c for c in CLUSTER_ORDER if c in pivot.index] + \
            [c for c in pivot.index if c not in CLUSTER_ORDER]
    pivot = pivot.loc[order]
    return pivot


def coin_pnl_stats(attr_df):
    """Per-coin total contribution."""
    rows = []
    for sym, grp in attr_df.groupby("symbol"):
        by_bar = grp.groupby("timestamp")["contribution"].sum()
        total  = grp["contribution"].sum()
        n_long  = (grp["side"] == "long").sum()
        n_short = (grp["side"] == "short").sum()
        mean_ret_long  = grp[grp["side"] == "long"]["ret"].mean()
        mean_ret_short = grp[grp["side"] == "short"]["ret"].mean()
        rows.append(dict(
            symbol=sym,
            cluster=SYM_TO_CLUSTER.get(sym, "Unknown"),
            n_long=int(n_long),
            n_short=int(n_short),
            long_pct=round(n_long / (n_long + n_short) * 100, 1) if (n_long + n_short) > 0 else np.nan,
            total_contrib_bps=round(total * 10000, 1),
            mean_contrib_bps=round(grp["contribution"].mean() * 10000, 3),
            mean_ret_when_long_bps=round(mean_ret_long * 10000, 1) if not np.isnan(mean_ret_long) else np.nan,
            mean_ret_when_short_bps=round(mean_ret_short * 10000, 1) if not np.isnan(mean_ret_short) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values("total_contrib_bps", ascending=False)


# ============================================================
# Plots
# ============================================================

def plot_cluster_bars(cluster_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cluster Performance Breakdown", fontsize=12, fontweight="bold")

    clusters = cluster_df["cluster"].tolist()
    colors   = [CLUSTER_COLORS.get(c, "#aaa") for c in clusters]

    # Total contribution
    ax = axes[0]
    vals = cluster_df["total_contrib_bps"]
    bars = ax.barh(clusters, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total P&L contribution (bps)")
    ax.set_title("Total contribution")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(v + (max(vals)*0.01), bar.get_y() + bar.get_height()/2,
                f"{v:.0f}", va="center", fontsize=8)

    # Contribution per bar
    ax = axes[1]
    vals2 = cluster_df["contrib_per_bar_bps"]
    bars2 = ax.barh(clusters, vals2, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Avg contribution per 8h bar (bps)")
    ax.set_title("Edge per bar")
    ax.invert_yaxis()
    for bar, v in zip(bars2, vals2):
        ax.text(v + (abs(max(vals2, key=abs))*0.01), bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8)

    # Long% (how often cluster is in long vs short leg)
    ax = axes[2]
    vals3 = cluster_df["long_pct"]
    colors3 = ["#2ca02c" if v >= 50 else "#d62728" for v in vals3]
    bars3 = ax.barh(clusters, vals3, color=colors3, edgecolor="white", alpha=0.8)
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("% of portfolio slots as LONG")
    ax.set_title("Long/short bias")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for bar, v in zip(bars3, vals3):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cluster_heatmap(pivot, out_path):
    fig, ax = plt.subplots(figsize=(18, 5))

    # Diverging colormap centred at 0
    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 5)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_title("Cluster × Month P&L contribution (%)", fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if abs(v) > 0.1:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < vmax*0.6 else "white")

    plt.colorbar(im, ax=ax, label="Monthly contribution (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_coin_contribution(coin_df, out_path):
    top20  = coin_df.head(20)
    bot20  = coin_df.tail(20).sort_values("total_contrib_bps")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Per-Coin P&L Contribution (full period)", fontsize=12, fontweight="bold")

    for ax, df, title in [(axes[0], top20, "Top 20 contributors"),
                           (axes[1], bot20, "Bottom 20 contributors")]:
        colors = [CLUSTER_COLORS.get(c, "#aaa") for c in df["cluster"]]
        bars = ax.barh(df["symbol"], df["total_contrib_bps"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Total P&L contribution (bps)")
        ax.set_title(title)
        ax.invert_yaxis()
        for bar, v in zip(bars, df["total_contrib_bps"]):
            ax.text(v + (5 if v >= 0 else -5),
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)

    # Legend for clusters
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=k) for k, c in CLUSTER_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower center", ncol=7,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cluster_equity(attr_df, out_path):
    """Cumulative equity curve per cluster."""
    fig, ax = plt.subplots(figsize=(15, 6))

    for cluster in CLUSTER_ORDER:
        sub = attr_df[attr_df["cluster"] == cluster]
        if sub.empty:
            continue
        by_bar = sub.groupby("timestamp")["contribution"].sum().sort_index()
        cum = (1 + by_bar).cumprod()
        color = CLUSTER_COLORS.get(cluster, "#aaa")
        ax.plot(cum.index, cum.values, color=color, linewidth=1.4, label=cluster)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}×" if x < 10 else f"{x:.0f}×"))
    ax.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative P&L contribution (log scale)")
    ax.set_title("Cluster equity curves — cumulative P&L contribution per cluster",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 6: Cluster Performance Analysis")
    print("=" * 60)

    # ---- Load ----
    print("\nLoading panels...")
    panels = load_panels(["close", "funding", "mom_24h"])

    fz = cs_zscore(to_8h(panels["funding"]))
    mz = cs_zscore(to_8h(panels["mom_24h"]))
    composite_8h = (fz.add(mz, fill_value=0)) / 2

    fwd_raw  = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw  = fwd_raw.clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
    fwd_8h   = fwd_raw.reindex(to_8h(panels["close"]).index)

    # ---- Simulation with attribution ----
    print("Running simulation with per-coin attribution...")
    bar_df, attr_df = sim_with_attribution(composite_8h,
                                            fwd_8h.reindex(composite_8h.index)
                                                  .clip(-0.99, 3.0)
                                                  .replace([np.inf,-np.inf], np.nan))

    # Coverage check
    untagged = attr_df[attr_df["cluster"] == "Unknown"]["symbol"].unique()
    if len(untagged):
        print(f"  Untagged symbols ({len(untagged)}): {sorted(untagged)}")
        # assign to nearest cluster by name heuristic — leave as Unknown

    print(f"  Total attribution rows: {len(attr_df):,}")
    print(f"  Symbols in universe: {attr_df['symbol'].nunique()}")

    # ---- Cluster P&L stats ----
    print("\nCluster P&L breakdown:")
    cluster_df = cluster_pnl_stats(attr_df, bar_df)
    cluster_df.to_csv(os.path.join(RESULTS_DIR, "phase6_cluster_pnl.csv"), index=False)

    print(f"\n  {'Cluster':<14} {'Syms':>5} {'Long%':>6} {'TotalBps':>9} "
          f"{'Bps/bar':>8} {'Sharpe':>7}")
    print("  " + "-" * 55)
    for _, row in cluster_df.iterrows():
        print(f"  {row['cluster']:<14} {row['n_symbols']:>5} "
              f"{row['long_pct']:>6.0f}% "
              f"{row['total_contrib_bps']:>9.0f} "
              f"{row['contrib_per_bar_bps']:>8.2f} "
              f"{row['ann_sharpe']:>7.3f}")

    # ---- Cluster × month ----
    pivot = cluster_monthly(attr_df)
    pivot.to_csv(os.path.join(RESULTS_DIR, "phase6_cluster_monthly.csv"))

    print("\n  Top monthly cluster contributions:")
    print(pivot.round(1).to_string())

    # ---- Per-coin ----
    coin_df = coin_pnl_stats(attr_df)
    coin_df.to_csv(os.path.join(RESULTS_DIR, "phase6_coin_pnl.csv"), index=False)

    print("\n  Top 20 coins by total P&L contribution (bps):")
    print(f"  {'Symbol':<20} {'Cluster':<14} {'LongSlots':>10} {'ShortSlots':>11} "
          f"{'Long%':>7} {'TotalBps':>9} {'Bps/slot':>9}")
    print("  " + "-" * 82)
    for _, row in coin_df.head(20).iterrows():
        print(f"  {row['symbol']:<20} {row['cluster']:<14} "
              f"{row['n_long']:>10} {row['n_short']:>11} "
              f"{row['long_pct']:>7.0f}% "
              f"{row['total_contrib_bps']:>9.0f} "
              f"{row['mean_contrib_bps']:>9.3f}")

    print("\n  Bottom 10 coins (most negative):")
    for _, row in coin_df.tail(10).iterrows():
        print(f"  {row['symbol']:<20} {row['cluster']:<14} "
              f"{row['total_contrib_bps']:>9.0f} bps")

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_cluster_bars(cluster_df,
                      os.path.join(RESULTS_DIR, "phase6_cluster_bars.png"))
    plot_cluster_heatmap(pivot,
                         os.path.join(RESULTS_DIR, "phase6_cluster_heatmap.png"))
    plot_coin_contribution(coin_df,
                           os.path.join(RESULTS_DIR, "phase6_coin_contribution.png"))
    plot_cluster_equity(attr_df,
                        os.path.join(RESULTS_DIR, "phase6_cluster_equity.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
