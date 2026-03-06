"""
Phase 22 - Step 2: Compare all funding signal variants.

Variants tested (all cross-sectionally z-scored, combined with same mom_24h):
  A  original      : funding (lagged settled, forward-filled every 8h)
  B  predicted     : predicted_funding (running premium TWAP + 0.0001, updates every 1h)
  C  cum24h        : funding_cum24h (sum of last 3 settlements = 24h carry)
  D  cum72h        : funding_cum72h (sum of last 9 settlements = 72h carry)
  E  pred+mom      : predicted_funding + mom_24h (replaces settled in Phase 16 combo)
  F  pred+cum24+mom: predicted_funding + funding_cum24h + mom_24h (3-signal combo)
  G  pred+pred_ft  : predicted_funding + predicted_ft (real-time carry + real-time trend)
  H  best_of_all   : predicted_funding + funding_cum24h + mom_24h + predicted_ft

For each: IC at 8h, ICIR, t-stat, and full backtest (N=10, maker fees, no-Majors).
Uses Phase 21 inverse-scaling and same walk-forward setup.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SIGNALS_V1   = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
SIGNALS_V2   = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals_v2"
RESULTS_DIR  = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
FEE_RT           = 4 * 2 / 10000
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
START            = "2025-01-01"
STARTING_CAPITAL = 1000.0
SHARPE_WINDOW    = 30
INTEREST_RATE    = 0.0001

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panels(signals_dir, cols):
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
    data  = {c: {} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet", "")
        if sym in MAJORS:
            continue
        try:
            df = pd.read_parquet(fpath)
            for c in cols:
                if c in df.columns:
                    data[c][sym] = df[c]
        except Exception:
            pass
    panels = {}
    for c in cols:
        p = pd.DataFrame(data[c])
        p.index = pd.to_datetime(p.index, utc=True)
        panels[c] = p.sort_index()
    return panels

def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-CLIP, CLIP)
    z[n < min_valid] = np.nan
    return z

# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_ic(signal_8h, fwd_8h):
    """IC and ICIR at 8h horizon, non-overlapping, cross-sectional."""
    rows = []
    for ts in signal_8h.index:
        s = signal_8h.loc[ts].dropna()
        f = fwd_8h.loc[ts].dropna() if ts in fwd_8h.index else pd.Series()
        common = s.index.intersection(f.index)
        if len(common) < 15:
            continue
        ic = s[common].corr(f[common])
        rows.append(ic)
    ics = pd.Series(rows)
    mean_ic = ics.mean()
    icir    = mean_ic / ics.std() if ics.std() > 0 else 0
    t_stat  = icir * np.sqrt(len(ics))
    return mean_ic, icir, t_stat, len(ics)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def sharpe_fn(x):
    if len(x) < 5 or x.std() == 0:
        return 0.0
    return x.mean() / x.std()

def sim(composite, fwd_8h, n=10, scale_series=None):
    """Long top-n, short bottom-n equal-weight. Returns per-bar P&L series."""
    rets = []
    dates = composite.index.intersection(fwd_8h.index)
    for ts in dates:
        row = composite.loc[ts].dropna()
        if len(row) < n * 2:
            rets.append(0.0)
            continue
        longs  = row.nlargest(n).index
        shorts = row.nsmallest(n).index
        fwd    = fwd_8h.loc[ts]
        long_ret  = fwd[longs].mean()
        short_ret = fwd[shorts].mean()
        gross = 0.5 * long_ret - 0.5 * short_ret
        net   = gross - FEE_RT
        rets.append(net)
    s = pd.Series(rets, index=dates)
    if scale_series is not None:
        sc = scale_series.reindex(s.index).fillna(1.0)
        s  = s * sc
    return s

def port_stats(rets, label="", ppy=PERIODS_PER_YEAR):
    if rets.std() == 0:
        return {}
    sr  = rets.mean() / rets.std() * np.sqrt(ppy)
    ann = (1 + rets).prod() ** (ppy / len(rets)) - 1
    neg = rets[rets < 0]
    so  = rets.mean() / neg.std() * np.sqrt(ppy) if len(neg) > 0 else np.nan
    eq  = (1 + rets).cumprod()
    dd  = (eq / eq.cummax() - 1).min()
    wr  = (rets > 0).mean()
    return dict(label=label, sharpe=sr, sortino=so, ann_ret=ann, max_dd=dd,
                win_rate=wr, n_bars=len(rets))

def build_inverse_scale(rets):
    """Phase 21 inverse scaling: 0.5x when hot (Sharpe>5) or bad (Sharpe<0)."""
    rolling_sh = rets.rolling(SHARPE_WINDOW).apply(sharpe_fn, raw=True)
    scale = pd.Series(1.0, index=rets.index)
    scale[rolling_sh > 5]  = 0.5
    scale[rolling_sh > 0]  = scale[rolling_sh > 0].clip(upper=1.0)
    scale[rolling_sh <= 0] = 0.5
    scale[rolling_sh > 5]  = 0.5
    return scale

def wf_test(composite_fn, fwd_8h, label):
    """4-window walk-forward: train Jan–Jun 2025, test Jul–Sep 2025, etc."""
    windows = [
        ("2025-01-01","2025-06-30","2025-07-01","2025-09-30"),
        ("2025-01-01","2025-09-30","2025-10-01","2025-12-31"),
        ("2025-01-01","2025-12-31","2026-01-01","2026-02-28"),
        ("2025-01-01","2026-01-31","2026-02-01","2026-03-06"),
    ]
    oos_sharpes = []
    for tr_s,tr_e,oo_s,oo_e in windows:
        comp = composite_fn()
        oo   = comp[oo_s:oo_e]
        f    = fwd_8h[oo_s:oo_e]
        r    = sim(oo, f)
        st   = port_stats(r)
        oos_sharpes.append(st.get("sharpe", 0))
    pos = sum(1 for s in oos_sharpes if s > 0)
    return oos_sharpes, pos

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading V1 panels (funding, mom_24h, fwd_8h) ...")
    v1 = load_panels(SIGNALS_V1, ["funding", "funding_trend", "mom_24h", "close", "fwd_8h"])

    print("Loading V2 panels (predicted_funding, cum24h, cum72h, predicted_ft) ...")
    v2 = load_panels(SIGNALS_V2, ["predicted_funding", "funding_cum24h",
                                   "funding_cum72h", "predicted_ft"])

    # Align to 8h
    def r8(p): return to_8h(p)

    fund_8h   = r8(v1["funding"])
    ft_8h     = r8(v1["funding_trend"])
    mom_8h    = r8(v1["mom_24h"])
    fwd_8h    = r8(v1["fwd_8h"])
    pred_8h   = r8(v2["predicted_funding"])
    cum24_8h  = r8(v2["funding_cum24h"])
    cum72_8h  = r8(v2["funding_cum72h"])
    pred_ft8h = r8(v2["predicted_ft"])

    # Align symbols across all panels
    all_syms = (fund_8h.columns.intersection(pred_8h.columns)
                               .intersection(mom_8h.columns)
                               .intersection(fwd_8h.columns))
    fund_8h   = fund_8h[all_syms]
    ft_8h     = ft_8h[all_syms]
    mom_8h    = mom_8h[all_syms]
    fwd_8h    = fwd_8h[all_syms]
    pred_8h   = pred_8h[all_syms]
    cum24_8h  = cum24_8h[all_syms] if set(all_syms).issubset(cum24_8h.columns) else cum24_8h.reindex(columns=all_syms)
    cum72_8h  = cum72_8h[all_syms] if set(all_syms).issubset(cum72_8h.columns) else cum72_8h.reindex(columns=all_syms)
    pred_ft8h = pred_ft8h[all_syms] if set(all_syms).issubset(pred_ft8h.columns) else pred_ft8h.reindex(columns=all_syms)

    print(f"Universe: {len(all_syms)} symbols, "
          f"{len(fwd_8h[START:])} 8h bars from {START}")
    print()

    # Filter to START
    def filt(p): return p[START:]

    z_fund   = cs_zscore(filt(fund_8h))
    z_ft     = cs_zscore(filt(ft_8h))
    z_mom    = cs_zscore(filt(mom_8h))
    z_pred   = cs_zscore(filt(pred_8h))
    z_cum24  = cs_zscore(filt(cum24_8h))
    z_cum72  = cs_zscore(filt(cum72_8h))
    z_pft    = cs_zscore(filt(pred_ft8h))
    fwd      = filt(fwd_8h)

    # -----------------------------------------------------------------------
    # IC comparison
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("IC COMPARISON (8h non-overlapping cross-sectional IC)")
    print("=" * 65)
    print(f"{'Signal':<28} {'IC':>7} {'ICIR':>7} {'t-stat':>8} {'N':>6}")
    print("-" * 65)

    ic_results = {}
    for name, sig in [
        ("funding (lagged settled)",  z_fund),
        ("funding_trend (lagged)",     z_ft),
        ("predicted_funding",          z_pred),
        ("funding_cum24h",             z_cum24),
        ("funding_cum72h",             z_cum72),
        ("predicted_ft",               z_pft),
        ("mom_24h",                    z_mom),
    ]:
        ic, icir, t, n = compute_ic(sig, fwd)
        ic_results[name] = (ic, icir, t)
        print(f"  {name:<26} {ic:+.4f}  {icir:+.4f}  {t:+7.2f}  {n:5d}")

    print()

    # -----------------------------------------------------------------------
    # Composite IC comparison
    # -----------------------------------------------------------------------
    print("COMPOSITE SIGNAL IC")
    print("-" * 65)
    composites = {
        "A: 2×fund + mom + ft (Phase21)":
            (2*z_fund + z_mom + z_ft) / 4,
        "B: 2×pred + mom":
            (2*z_pred + z_mom) / 3,
        "C: 2×pred + mom + cum24":
            (2*z_pred + z_mom + z_cum24) / 4,
        "D: 2×pred + mom + pred_ft":
            (2*z_pred + z_mom + z_pft) / 4,
        "E: 2×pred + cum24 + mom + pred_ft":
            (2*z_pred + z_cum24 + z_mom + z_pft) / 5,
        "F: pred + cum24 (no mom)":
            (z_pred + z_cum24) / 2,
        "G: 2×pred + pred_ft + cum24":
            (2*z_pred + z_pft + z_cum24) / 4,
    }

    for name, comp in composites.items():
        ic, icir, t, n = compute_ic(comp, fwd)
        print(f"  {name:<38} IC={ic:+.4f}  ICIR={icir:+.4f}  t={t:+6.2f}")

    print()

    # -----------------------------------------------------------------------
    # Full backtest comparison
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("BACKTEST (Jan 2025–Mar 2026, N=10, maker fees, no-Majors)")
    print("=" * 65)
    print(f"{'Variant':<40} {'Sharpe':>7} {'MaxDD':>8} {'AnnRet':>8} {'$1k→':>8} {'WF+':>5}")
    print("-" * 65)

    bt_results = []
    for name, comp in composites.items():
        comp = comp.dropna(how="all")
        # First pass: build scale from unscaled rets
        r0 = sim(comp, fwd, n=10)
        scale = build_inverse_scale(r0)
        r  = sim(comp, fwd, n=10, scale_series=scale)
        st = port_stats(r, label=name)
        final = STARTING_CAPITAL * (1 + r).prod()
        # Walk-forward
        wf_sh, wf_pos = wf_test(lambda c=comp: c, fwd, name)
        wf_min = min(wf_sh)
        print(f"  {name:<38} {st['sharpe']:>7.3f} {st['max_dd']:>7.1%} "
              f"{st['ann_ret']:>8.0%}  ${final:>7.0f}  {wf_pos}/4")
        bt_results.append(dict(name=name, **st, final=final,
                               wf_pos=wf_pos, wf_min=wf_min))

    print()

    # Best variant
    best = max(bt_results, key=lambda x: x["sharpe"])
    print(f"Best Sharpe: {best['name']}")
    print(f"  Sharpe={best['sharpe']:.3f}  MaxDD={best['max_dd']:.1%}  "
          f"$1k→${best['final']:.0f}  WF {best['wf_pos']}/4")
    print()

    # -----------------------------------------------------------------------
    # Monthly breakdown for top 2 variants
    # -----------------------------------------------------------------------
    print("MONTHLY BREAKDOWN — Phase21 baseline vs best new variant")
    print("-" * 65)
    top_names = ["A: 2×fund + mom + ft (Phase21)", best["name"]]
    monthly_data = {}
    for name in top_names:
        comp = composites[name].dropna(how="all")
        r0   = sim(comp, fwd, n=10)
        scale = build_inverse_scale(r0)
        r    = sim(comp, fwd, n=10, scale_series=scale)
        eq   = STARTING_CAPITAL * (1 + r).cumprod()
        monthly = r.resample("ME").apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
        )
        monthly_data[name] = monthly

    all_months = sorted(set().union(*[m.index for m in monthly_data.values()]))
    print(f"  {'Month':<10}", end="")
    for n in top_names:
        short = n[:18]
        print(f"  {short:>18}", end="")
    print()
    for m in all_months:
        label = m.strftime("%Y-%m")
        print(f"  {label:<10}", end="")
        for n in top_names:
            v = monthly_data[n].get(m, np.nan)
            print(f"  {v:>17.1%}" if not np.isnan(v) else f"  {'—':>17}", end="")
        print()
    print()

    # -----------------------------------------------------------------------
    # Equity curve plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    colors = ["steelblue", "crimson", "green", "orange", "purple", "brown", "teal"]
    for i, (name, comp) in enumerate(composites.items()):
        comp = comp.dropna(how="all")
        r0   = sim(comp, fwd, n=10)
        scale = build_inverse_scale(r0)
        r    = sim(comp, fwd, n=10, scale_series=scale)
        eq   = STARTING_CAPITAL * (1 + r).cumprod()
        lw   = 2.5 if name == best["name"] else 1.0
        ax.plot(eq.index, eq.values, label=name[:40], color=colors[i % len(colors)], lw=lw)
    ax.set_title("Equity Curves — All Funding Variants (Phase 22, $1k start, inverse scaling)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # IC bar chart
    ax2 = axes[1]
    sig_names = ["funding\n(lagged)", "funding_trend\n(lagged)", "predicted\nfunding",
                 "cum24h", "cum72h", "predicted\nft", "mom_24h"]
    ic_vals  = [ic_results[k][0] for k in [
        "funding (lagged settled)", "funding_trend (lagged)",
        "predicted_funding", "funding_cum24h",
        "funding_cum72h", "predicted_ft", "mom_24h"]]
    icir_vals = [ic_results[k][1] for k in [
        "funding (lagged settled)", "funding_trend (lagged)",
        "predicted_funding", "funding_cum24h",
        "funding_cum72h", "predicted_ft", "mom_24h"]]
    x = np.arange(len(sig_names))
    ax2.bar(x - 0.2, ic_vals,  0.38, label="IC",   color="steelblue", alpha=0.85)
    ax2.bar(x + 0.2, icir_vals, 0.38, label="ICIR", color="crimson",   alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sig_names, fontsize=9)
    ax2.set_title("IC & ICIR by Funding Signal Variant (8h horizon)")
    ax2.set_ylabel("IC / ICIR")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "phase22_funding_comparison.png")
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"Plot saved: {out_png}")
    print()
    print("Phase 22 complete.")


if __name__ == "__main__":
    main()
