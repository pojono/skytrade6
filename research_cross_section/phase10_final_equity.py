"""
Phase 10: Final equity curve and monthly breakdown from $1000 starting capital.

Uses the honest, zero-look-ahead configuration from Phase 9:
  - Universe: 113 coins (exclude 18 Majors, structural argument)
  - Signal: funding + mom_24h, equal-weight, N=10, 8h rebal, maker 4bps/side
  - Two variants: without and with regime filter (signal_strength + funding_disp)

Shows:
  - Dollar equity curve starting from $1000 (1x notional leverage)
  - Monthly P&L table in $ and %
  - Drawdown in $ terms
  - All periods (full backtest, not just OOS windows)

Outputs:
  results/phase10_equity_dollar.png
  results/phase10_monthly_dollar.csv
  results/phase10_monthly_dollar.png
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
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3
STARTING_CAPITAL = 1000.0

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
    return panel[cols]


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
            records.append(dict(timestamp=ts, gross=0., turnover=0., net=0., active=trading))
            if not trading:
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
        records.append(dict(timestamp=ts, gross=gross, turnover=turnover,
                            net=net, active=True))
        prev_longs, prev_shorts = longs, shorts
    return pd.DataFrame(records).set_index("timestamp")


def build_regime_mask(composite_8h, funding_8h):
    """
    Walk-forward regime filter: signal_strength + funding_disp.
    Thresholds fitted on 6mo training, applied to 3mo OOS.
    Returns boolean Series aligned to composite_8h.index.
    """
    rebal_idx = composite_8h.index
    mask = pd.Series(True, index=rebal_idx)

    feats = pd.DataFrame(index=rebal_idx)
    feats["signal_strength"] = composite_8h.std(axis=1)
    feats["funding_disp"]    = funding_8h.reindex(rebal_idx).std(axis=1)

    start = rebal_idx.min()
    end   = rebal_idx.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    # Full-period simulation for training P&L reference
    fwd_dummy = None  # we only need features, not P&L for threshold fitting

    for tr_s, tr_e, oo_s, oo_e in windows:
        for feat_col in ["signal_strength", "funding_disp"]:
            f_tr = feats[feat_col].loc[tr_s:tr_e].dropna()
            if len(f_tr) < 20:
                continue
            # Trade when feature > θ (both are "higher is better")
            ths = np.percentile(f_tr, np.linspace(10, 90, 17))
            # Simple heuristic: use 30th percentile as threshold
            # (in full walk-forward we'd optimise on training P&L —
            #  here we use a fixed percentile to avoid any leakage from P&L)
            th = np.percentile(f_tr, 30)
            oos_idx = rebal_idx[(rebal_idx >= oo_s) & (rebal_idx < oo_e)]
            f_oos   = feats[feat_col].reindex(oos_idx).fillna(f_tr.median())
            active  = f_oos > th
            mask.loc[oos_idx] &= active

    return mask


# ============================================================
# Monthly breakdown
# ============================================================

def monthly_breakdown(pnl_df, starting_capital=STARTING_CAPITAL):
    net = pnl_df["net"].replace(0, np.nan).dropna()
    cum = (1 + net).cumprod() * starting_capital
    rows = []
    for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
        n        = len(grp)
        ret_pct  = (1 + grp).prod() - 1
        cum_grp  = (1 + grp).cumprod()
        mdd      = (cum_grp / cum_grp.cummax() - 1).min()
        sh       = (grp.mean() / grp.std() * np.sqrt(PERIODS_PER_YEAR)
                    if n > 1 and grp.std() > 0 else np.nan)
        # Dollar P&L: starting equity at beginning of month
        month_start_idx = cum.index.searchsorted(grp.index[0])
        start_eq = cum.iloc[month_start_idx - 1] if month_start_idx > 0 else starting_capital
        dollar_pnl = start_eq * ret_pct
        rows.append(dict(
            month=f"{yr}-{mo:02d}",
            n_bars=n,
            ret_pct=round(ret_pct * 100, 2),
            dollar_pnl=round(dollar_pnl, 2),
            end_equity=round(start_eq + dollar_pnl, 2),
            sharpe=round(sh, 2) if not np.isnan(sh) else np.nan,
            max_dd_pct=round(mdd * 100, 2),
        ))
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def plot_equity_dollar(pnl_dict, starting_capital, out_path):
    """
    pnl_dict: {label: net_series}
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    styles = {
        "No-Majors":              ("#1f77b4", 2.2, "-"),
        "No-Majors + Regime":     ("#2ca02c", 2.2, "-"),
        "Baseline (all 131)":     ("#d62728", 1.2, "--"),
    }

    ax1.set_title(f"Equity Curve from ${starting_capital:,.0f} — "
                  "Funding + Mom24h | No-Majors | N=10 | 8h rebal | Maker",
                  fontsize=11, fontweight="bold")

    for label, net in pnl_dict.items():
        net = net.replace(0, np.nan).dropna()
        if len(net) < 5:
            continue
        equity = (1 + net).cumprod() * starting_capital
        dd     = (equity / equity.cummax() - 1) * 100
        sty    = styles.get(label, ("#aaa", 1.0, "-"))

        ax1.plot(equity.index, equity.values, color=sty[0],
                 linewidth=sty[1], linestyle=sty[2], label=label)
        ax2.plot(dd.index, dd.values, color=sty[0],
                 linewidth=max(sty[1]*0.7, 0.8), linestyle=sty[2], alpha=0.85)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Portfolio value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(starting_capital, color="black", linewidth=0.5, linestyle="--")
    ax1.set_xticklabels([])

    # Shade OOS windows
    for s, e in [("2025-01-01","2025-04-01"),("2025-04-01","2025-07-01"),
                  ("2025-07-01","2025-10-01"),("2025-10-01","2026-01-01")]:
        for ax in [ax1, ax2]:
            ax.axvspan(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC"),
                       alpha=0.05, color="green")

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.fill_between(
        list(pnl_dict.values())[0].replace(0, np.nan).dropna().pipe(
            lambda n: (1+n).cumprod() * starting_capital
        ).pipe(lambda e: e / e.cummax() - 1).index,
        list(pnl_dict.values())[0].replace(0, np.nan).dropna().pipe(
            lambda n: (1+n).cumprod() * starting_capital
        ).pipe(lambda e: (e / e.cummax() - 1) * 100).values,
        0, alpha=0.25,
        color=list(styles.values())[0][0]
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticklabels([])

    # Bar chart of 8h bar returns (for first variant)
    first_net = list(pnl_dict.values())[0].replace(0, np.nan).dropna()
    first_col = list(styles.values())[0][0]
    ax3.bar(first_net.index, first_net.values * 100,
            width=pd.Timedelta("7h"), color=[first_col if v >= 0 else "#d62728"
                                              for v in first_net.values],
            alpha=0.4)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Bar ret (%)")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.grid(axis="y", alpha=0.3)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_monthly_dollar(monthly_df_dict, starting_capital, out_path):
    """Monthly bar chart with dollar P&L and % labels."""
    fig, axes = plt.subplots(len(monthly_df_dict), 1,
                              figsize=(18, 4.5 * len(monthly_df_dict)),
                              sharex=False)
    if len(monthly_df_dict) == 1:
        axes = [axes]
    fig.suptitle(f"Monthly P&L from ${starting_capital:,.0f} starting capital",
                 fontsize=12, fontweight="bold")

    for ax, (label, mdf) in zip(axes, monthly_df_dict.items()):
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mdf["dollar_pnl"]]
        bars   = ax.bar(mdf["month"], mdf["dollar_pnl"], color=colors,
                        width=0.7, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel("Monthly P&L ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:+,.0f}"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Dual label: $ on bar, % below
        for bar, row in zip(bars, mdf.itertuples()):
            y = bar.get_height()
            offset = max(abs(mdf["dollar_pnl"].max()) * 0.01, 1)
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2,
                    y + (offset if y >= 0 else -offset),
                    f"${y:+,.0f}\n({row.ret_pct:+.1f}%)",
                    ha="center", va=va, fontsize=6.5, color="black")

        # Running equity line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(mdf["month"], mdf["end_equity"], color="#666", linewidth=1.2,
                 linestyle="--", marker="o", markersize=3, label="Equity")
        ax2.set_ylabel("Portfolio ($)", color="#666", fontsize=8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"))
        ax2.tick_params(axis="y", colors="#666", labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 10: Final Equity Curve from $1,000")
    print("=" * 60)

    print("\nLoading panels...")
    panels  = load_panels(["close", "funding", "mom_24h"])
    all_syms = set(panels["funding"].columns)
    no_maj   = all_syms - MAJORS

    fwd_raw  = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw  = fwd_raw.clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
    fwd_8h   = fwd_raw.reindex(to_8h(panels["close"]).index)

    def make_comp(keep):
        f8 = cs_zscore(to_8h(restrict(panels["funding"], keep)))
        m8 = cs_zscore(to_8h(restrict(panels["mom_24h"], keep)))
        return (f8.add(m8, fill_value=0)) / 2

    # ---- Baseline (131 coins) ----
    print("\nSimulating: Baseline (131 coins)...")
    comp_all  = make_comp(all_syms)
    fwd_all   = restrict(fwd_8h, all_syms).reindex(comp_all.index).clip(-0.99,3).replace([np.inf,-np.inf],np.nan)
    pnl_base  = sim(comp_all, fwd_all)

    # ---- No-Majors (113 coins) ----
    print("Simulating: No-Majors (113 coins)...")
    comp_nm   = make_comp(no_maj)
    fwd_nm    = restrict(fwd_8h, no_maj).reindex(comp_nm.index).clip(-0.99,3).replace([np.inf,-np.inf],np.nan)
    pnl_nm    = sim(comp_nm, fwd_nm)

    # ---- No-Majors + Regime filter ----
    print("Building regime mask...")
    funding_8h = to_8h(restrict(panels["funding"], no_maj))
    regime_mask = build_regime_mask(comp_nm, funding_8h)
    active_pct  = regime_mask.mean()
    print(f"  Regime mask active: {active_pct:.1%} of bars")

    print("Simulating: No-Majors + Regime filter...")
    pnl_regime = sim(comp_nm, fwd_nm, mask=regime_mask)

    # ---- Stats ----
    def stats(pnl, label):
        net = pnl["net"].replace(0, np.nan).dropna()
        ar  = net.mean() * PERIODS_PER_YEAR
        av  = net.std()  * np.sqrt(PERIODS_PER_YEAR)
        sh  = ar / av if av > 0 else np.nan
        cum = (1 + net).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else np.nan
        final_eq  = STARTING_CAPITAL * cum.iloc[-1] if len(cum) > 0 else np.nan
        print(f"\n  {label}:")
        print(f"    Sharpe:      {sh:.3f}")
        print(f"    Ann. Return: {ar*100:.0f}%")
        print(f"    Max DD:      {mdd*100:.1f}%")
        print(f"    Total Ret:   {total_ret*100:.0f}%")
        print(f"    $1000 → ${final_eq:,.0f}")
        print(f"    Win rate:    {(net>0).mean():.1%}")

    print("\n" + "="*50 + "\nSTATISTICS (full period)")
    stats(pnl_base,   "Baseline (131 coins)")
    stats(pnl_nm,     "No-Majors (113 coins)")
    stats(pnl_regime, "No-Majors + Regime filter")

    # ---- Monthly tables ----
    print("\nBuilding monthly tables...")
    mdf_nm     = monthly_breakdown(pnl_nm)
    mdf_regime = monthly_breakdown(pnl_regime)

    # Print No-Majors table
    print(f"\n{'─'*80}")
    print("MONTHLY P&L — No-Majors (113 coins, full period)")
    print(f"{'─'*80}")
    print(f"  {'Month':<10} {'Ret%':>7} {'$P&L':>10} {'Equity':>10} "
          f"{'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'─'*55}")
    for _, r in mdf_nm.iterrows():
        flag = " ◄ BAD" if r["ret_pct"] < -10 else (" ◄ GREAT" if r["ret_pct"] > 50 else "")
        print(f"  {r['month']:<10} {r['ret_pct']:>+7.1f}% "
              f"${r['dollar_pnl']:>+9,.0f} "
              f"${r['end_equity']:>9,.0f} "
              f"{r['sharpe']:>7.2f} "
              f"{r['max_dd_pct']:>6.1f}%"
              f"{flag}")

    pos_m  = (mdf_nm["ret_pct"] > 0).sum()
    print(f"\n  Positive months: {pos_m}/{len(mdf_nm)} ({pos_m/len(mdf_nm)*100:.0f}%)")
    print(f"  Final equity:    ${mdf_nm['end_equity'].iloc[-1]:,.0f} "
          f"(from ${STARTING_CAPITAL:,.0f})")

    # Print Regime table
    print(f"\n{'─'*80}")
    print("MONTHLY P&L — No-Majors + Regime Filter")
    print(f"{'─'*80}")
    print(f"  {'Month':<10} {'Ret%':>7} {'$P&L':>10} {'Equity':>10} "
          f"{'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'─'*55}")
    for _, r in mdf_regime.iterrows():
        flag = " ◄ BAD" if r["ret_pct"] < -10 else (" ◄ GREAT" if r["ret_pct"] > 50 else "")
        print(f"  {r['month']:<10} {r['ret_pct']:>+7.1f}% "
              f"${r['dollar_pnl']:>+9,.0f} "
              f"${r['end_equity']:>9,.0f} "
              f"{r['sharpe']:>7.2f} "
              f"{r['max_dd_pct']:>6.1f}%"
              f"{flag}")

    pos_r = (mdf_regime["ret_pct"] > 0).sum()
    print(f"\n  Positive months: {pos_r}/{len(mdf_regime)} ({pos_r/len(mdf_regime)*100:.0f}%)")
    print(f"  Final equity:    ${mdf_regime['end_equity'].iloc[-1]:,.0f} "
          f"(from ${STARTING_CAPITAL:,.0f})")

    # Save CSVs
    mdf_nm.to_csv(os.path.join(RESULTS_DIR, "phase10_monthly_no_majors.csv"), index=False)
    mdf_regime.to_csv(os.path.join(RESULTS_DIR, "phase10_monthly_regime.csv"), index=False)

    # ---- Plots ----
    print("\nGenerating plots...")
    pnl_dict = {
        "No-Majors":          pnl_nm["net"],
        "No-Majors + Regime": pnl_regime["net"],
        "Baseline (all 131)": pnl_base["net"],
    }
    plot_equity_dollar(pnl_dict, STARTING_CAPITAL,
                       os.path.join(RESULTS_DIR, "phase10_equity_dollar.png"))

    monthly_dict = {
        "No-Majors (113 coins)":     mdf_nm,
        "No-Majors + Regime Filter": mdf_regime,
    }
    plot_monthly_dollar(monthly_dict, STARTING_CAPITAL,
                        os.path.join(RESULTS_DIR, "phase10_monthly_dollar.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
