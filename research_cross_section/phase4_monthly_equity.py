"""
Phase 4: Monthly P&L breakdown and equity curves.

Runs Combo A (funding + mom_24h, equal-weight, N=10, 8h rebal, maker fees)
and produces:
  results/phase4_monthly.csv      — monthly stats table
  results/phase4_equity_curve.png — log-scale equity + drawdown
  results/phase4_monthly_bar.png  — monthly net return bar chart
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
FEE_MAKER_BPS    = 4        # per side
PERIODS_PER_YEAR = 365 * 3  # 8h bars


# ---------------------------------------------------------------------------
# Load panels
# ---------------------------------------------------------------------------

def load_panels_fast(signals_dir, cols):
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
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


# ---------------------------------------------------------------------------
# Signal prep
# ---------------------------------------------------------------------------

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-3, 3)
    z[n < min_valid] = np.nan
    return z


def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

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
            records.append(dict(timestamp=ts, gross=np.nan,
                                turnover=np.nan, net=np.nan, n=len(sig)))
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
        records.append(dict(timestamp=ts, gross=gross,
                            turnover=turnover, net=net, n=len(sig)))
        prev_longs, prev_shorts = longs, shorts

    return pd.DataFrame(records).set_index("timestamp")


# ---------------------------------------------------------------------------
# Monthly stats
# ---------------------------------------------------------------------------

def monthly_breakdown(pnl_df):
    net = pnl_df["net"].dropna()
    gross = pnl_df["gross"].dropna()

    # Group by month
    rows = []
    for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
        g_grp = gross.reindex(grp.index).dropna()
        n = len(grp)
        if n == 0:
            continue
        cum_ret   = (1 + grp).prod() - 1
        cum_gross = (1 + g_grp).prod() - 1 if len(g_grp) > 0 else np.nan
        vol  = grp.std() * np.sqrt(PERIODS_PER_YEAR) if n > 1 else np.nan
        sh   = (grp.mean() / grp.std() * np.sqrt(PERIODS_PER_YEAR)
                if n > 1 and grp.std() > 0 else np.nan)
        # drawdown within month
        cum_path = (1 + grp).cumprod()
        mdd = (cum_path / cum_path.cummax() - 1).min()
        turn_grp = pnl_df["turnover"].reindex(grp.index).dropna()
        avg_turn = turn_grp.mean() if len(turn_grp) > 0 else np.nan
        rows.append(dict(
            year=yr, month=mo,
            month_label=f"{yr}-{mo:02d}",
            n_bars=n,
            gross_pct=round(cum_gross * 100, 2),
            net_pct=round(cum_ret * 100, 2),
            ann_sharpe=round(sh, 2) if not np.isnan(sh) else np.nan,
            max_dd_pct=round(mdd * 100, 2),
            avg_turnover=round(avg_turn, 3),
            net_bps_per_bar=round(grp.mean() * 10000, 1),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_and_drawdown(pnl_df, out_path):
    net = pnl_df["net"].dropna()
    gross = pnl_df["gross"].dropna()

    cum_net   = (1 + net).cumprod()
    cum_gross = (1 + gross).cumprod()
    dd_net    = cum_net / cum_net.cummax() - 1

    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    # --- Equity ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum_gross.index, cum_gross.values, color="#aaa", linewidth=1,
             label="Gross (no fees)", alpha=0.7)
    ax1.plot(cum_net.index, cum_net.values, color="#1f77b4", linewidth=1.5,
             label="Net (maker 4 bps/side)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"))
    ax1.set_ylabel("Cumulative return (log scale)")
    ax1.set_title("Combo A — Funding + Mom24h | N=10 | 8h rebal | Equal-weight",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticklabels([])
    ax1.axhline(1, color="black", linewidth=0.5, linestyle="--")

    # --- Drawdown ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dd_net.index, dd_net.values * 100, 0,
                     color="#d62728", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(dd_net.min() * 100 * 1.1, 5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # shade OOS windows
    oos_windows = [
        ("2025-01-01", "2025-04-01"),
        ("2025-04-01", "2025-07-01"),
        ("2025-07-01", "2025-10-01"),
        ("2025-10-01", "2026-01-01"),
    ]
    for (s, e) in oos_windows:
        s_dt = pd.Timestamp(s, tz="UTC")
        e_dt = pd.Timestamp(e, tz="UTC")
        ax1.axvspan(s_dt, e_dt, alpha=0.06, color="green")
        ax2.axvspan(s_dt, e_dt, alpha=0.06, color="green")

    # annotate OOS windows on equity
    for (s, e) in oos_windows:
        mid = pd.Timestamp(s, tz="UTC") + (pd.Timestamp(e, tz="UTC") - pd.Timestamp(s, tz="UTC")) / 2
        if mid in cum_net.index or True:
            ax1.axvline(pd.Timestamp(s, tz="UTC"), color="green",
                        linewidth=0.5, linestyle=":", alpha=0.5)

    ax2.text(0.01, 0.05, "Shaded = OOS windows", transform=ax2.transAxes,
             fontsize=7, color="green", alpha=0.8)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_monthly_bars(monthly_df, out_path):
    df = monthly_df.copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle("Monthly P&L — Combo A (Funding + Mom24h)", fontsize=12, fontweight="bold")

    # --- Monthly net return % ---
    ax = axes[0]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["net_pct"]]
    bars = ax.bar(df["month_label"], df["net_pct"], color=colors, width=0.7, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net monthly return (%)")
    ax.set_title("Monthly net return (maker fees deducted)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # value labels on bars
    for bar, val in zip(bars, df["net_pct"]):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (0.3 if y >= 0 else -0.8),
                f"{val:.0f}%", ha="center", va="bottom", fontsize=5.5)

    # --- Monthly Sharpe ---
    ax2 = axes[1]
    sh_colors = ["#2ca02c" if (not np.isnan(v) and v >= 0) else "#d62728"
                 for v in df["ann_sharpe"]]
    ax2.bar(df["month_label"], df["ann_sharpe"].fillna(0), color=sh_colors,
            width=0.7, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe=1")
    ax2.axhline(2, color="blue", linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe=2")
    ax2.set_ylabel("Ann. Sharpe (within month)")
    ax2.set_title("Monthly annualized Sharpe")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_combo_equity(combos_pnl, out_path):
    """Overlay equity curves for top combos."""
    fig, ax = plt.subplots(figsize=(14, 6))
    style = {
        "A: funding + mom_24h":          ("#1f77b4", 2.0, "-"),
        "F: funding + mom_48h":           ("#ff7f0e", 1.5, "--"),
        "B: funding + mom_24h - prem_z":  ("#2ca02c", 1.2, "-."),
        "C: funding only":                ("#9467bd", 1.0, ":"),
        "D: mom_24h only":                ("#8c564b", 1.0, ":"),
    }
    for label, net in combos_pnl.items():
        net = net.dropna()
        cum = (1 + net).cumprod()
        kw = style.get(label, ("#aaa", 0.8, "-"))
        ax.plot(cum.index, cum.values, color=kw[0], linewidth=kw[1],
                linestyle=kw[2], label=label)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"))
    ax.set_ylabel("Cumulative return (log scale)")
    ax.set_title("Combo equity curves — net returns (maker 4 bps/side)", fontsize=11)
    ax.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading panels...")
    COLS = ["close", "funding", "mom_24h", "mom_48h", "fwd_8h"]
    panels = load_panels_fast(SIGNALS_DIR, COLS)

    # Close panel for forward-return computation
    close_8h = to_8h(panels["close"])
    fwd_raw  = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw  = fwd_raw.clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
    fwd_8h   = fwd_raw.reindex(close_8h.index)

    # CS z-score signals at 8h
    print("Building composite signals...")
    panels_z = {}
    for col in ["funding", "mom_24h", "mom_48h"]:
        if col in panels:
            panels_z[col] = cs_zscore(to_8h(panels[col]))

    # Combo A: funding + mom_24h
    comp_a = (panels_z["funding"].add(panels_z["mom_24h"], fill_value=0)) / 2

    # Combo F: funding + mom_48h
    comp_f = (panels_z["funding"].add(panels_z["mom_48h"], fill_value=0)) / 2

    # Combo C: funding only
    comp_c = panels_z["funding"]

    # Combo D: mom_24h only
    comp_d = panels_z["mom_24h"]

    # Run combo B: funding + mom_24h - prem_z (need prem_z panel)
    # prem_z may or may not be in parquets — load if available
    if "prem_z" in load_panels_fast(SIGNALS_DIR, ["prem_z"]):
        pz_panel = load_panels_fast(SIGNALS_DIR, ["prem_z"])["prem_z"]
        pz_z = cs_zscore(to_8h(pz_panel))
        comp_b = (panels_z["funding"].add(panels_z["mom_24h"], fill_value=0)
                  .sub(pz_z, fill_value=0)) / 3
    else:
        comp_b = None

    print("Running simulations...")
    combos = {
        "A: funding + mom_24h": comp_a,
        "F: funding + mom_48h": comp_f,
        "C: funding only":       comp_c,
        "D: mom_24h only":       comp_d,
    }
    if comp_b is not None:
        combos["B: funding + mom_24h - prem_z"] = comp_b

    combos_pnl = {}
    for label, comp in combos.items():
        pnl = sim(comp, fwd_8h)
        combos_pnl[label] = pnl["net"]

    pnl_a = sim(comp_a, fwd_8h)

    # --- Monthly breakdown ---
    print("Computing monthly breakdown...")
    monthly = monthly_breakdown(pnl_a)
    monthly.to_csv(os.path.join(RESULTS_DIR, "phase4_monthly.csv"), index=False)
    print(f"  Saved: results/phase4_monthly.csv")

    # Print table
    print("\n=== Monthly P&L — Combo A (Funding + Mom24h, N=10, 8h, Maker) ===")
    print(f"{'Month':<10} {'Gross%':>7} {'Net%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'bps/bar':>8} {'Turn':>6}")
    print("-" * 60)
    for _, row in monthly.iterrows():
        print(f"{row['month_label']:<10} "
              f"{row['gross_pct']:>7.1f} "
              f"{row['net_pct']:>7.1f} "
              f"{row['ann_sharpe']:>7.2f} "
              f"{row['max_dd_pct']:>7.1f} "
              f"{row['net_bps_per_bar']:>8.1f} "
              f"{row['avg_turnover']:>6.3f}")

    # Summary stats
    net_s  = pnl_a["net"].dropna()
    pos_m  = (monthly["net_pct"] > 0).sum()
    tot_m  = len(monthly)
    print(f"\nPositive months: {pos_m}/{tot_m} ({pos_m/tot_m*100:.0f}%)")
    best   = monthly.loc[monthly["net_pct"].idxmax()]
    worst  = monthly.loc[monthly["net_pct"].idxmin()]
    print(f"Best month:  {best['month_label']}  {best['net_pct']:.1f}%")
    print(f"Worst month: {worst['month_label']}  {worst['net_pct']:.1f}%")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_equity_and_drawdown(
        pnl_a,
        os.path.join(RESULTS_DIR, "phase4_equity_curve.png")
    )
    plot_monthly_bars(
        monthly,
        os.path.join(RESULTS_DIR, "phase4_monthly_bar.png")
    )
    plot_per_combo_equity(
        combos_pnl,
        os.path.join(RESULTS_DIR, "phase4_combo_equity.png")
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
