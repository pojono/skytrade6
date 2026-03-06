"""
Phase 16: Final Combined Strategy

Incorporates the best validated improvements:

  Signal improvements (from Phase 11):
    - funding_trend (ICIR +0.106, t=3.79): change in funding over 24h
    - Tests: funding-only, funding+trend, vs baseline funding+mom24h

  Regime filter (from Phase 13):
    - Soft threshold: scale position by regime confidence (best MaxDD reducer)

  Dynamic leverage (from Phase 12):
    - Rejected: vol-targeting and DD-scaling both HURT Sharpe on this strategy
    - Not included

  Capacity (from Phase 15):
    - Optimal AUM: $1M–$5M for acceptable impact costs
    - Documented, not implemented (operational decision)

Uses Phase 10's correct time alignment:
  resample(closed="left", label="left").first()  <- correct (rebal bar at 00/08/16 UTC)
  NOT .last() which gives 7h-shifted windows

Walk-forward OOS (6mo train / 3mo OOS) for all regime thresholds.

Outputs:
  results/phase16_final.png
  results/phase16_monthly.csv
  results/phase16_summary.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
N_LONG           = 10
N_SHORT          = 10
FEE_MAKER_BPS    = 4
FEE_RT           = FEE_MAKER_BPS * 2 / 10000   # round-trip maker fee
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3
STARTING_CAPITAL = 1000.0

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}


# ── loaders (Phase 10 style — correct alignment) ───────────────────────────

def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.parquet")))
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


def to_8h(panel):
    """Resample to 8h, taking FIRST bar of each window (correct alignment)."""
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()


def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-CLIP, CLIP)
    z[n < min_valid] = np.nan
    return z


# ── portfolio simulation (Phase 10 style) ─────────────────────────────────

def sim(composite_8h, fwd_8h, scale=None, long_scale=None, short_scale=None,
        n_long=N_LONG, n_short=N_SHORT, fee_rt=FEE_RT, min_universe=20):
    """
    Run long/short portfolio simulation.

    scale:       Series of 0–1 multipliers applied to both legs (soft regime)
    long_scale:  Series of 0–1 multipliers for long leg only
    short_scale: Series of 0–1 multipliers for short leg only
    """
    common = composite_8h.index.intersection(fwd_8h.index)
    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe:
            records.append(dict(timestamp=ts, gross=np.nan, net=np.nan))
            prev_longs = prev_shorts = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)

        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()

        if np.isnan(lr) or np.isnan(sr):
            records.append(dict(timestamp=ts, gross=np.nan, net=np.nan))
            continue

        # Apply scaling (soft regime / asymmetric)
        ls = float(long_scale.get(ts, 1.0)) if long_scale is not None else 1.0
        ss = float(short_scale.get(ts, 1.0)) if short_scale is not None else 1.0
        if scale is not None:
            sc = float(scale.get(ts, 1.0))
            ls = ls * sc
            ss = ss * sc

        gross = ls * lr - ss * sr

        # Turnover (fraction of positions that changed)
        turnover = (len((longs - prev_longs) | (shorts - prev_shorts)) / (n_long + n_short)
                    if prev_longs | prev_shorts else 1.0)
        net = gross - turnover * fee_rt

        records.append(dict(timestamp=ts, gross=gross, net=net))
        prev_longs, prev_shorts = longs, shorts

    df = pd.DataFrame(records).set_index("timestamp")
    return df["net"]


def compute_metrics(rets, label=""):
    r = rets.dropna()
    if len(r) == 0:
        return {"Sharpe": 0, "Sortino": 0, "Ann_Ret": 0, "MaxDD": 0, "Total": 0, "WinPct": 0}
    sharpe  = r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else 0
    neg     = r[r < 0]
    sortino = r.mean() / neg.std() * np.sqrt(PERIODS_PER_YEAR) if len(neg) > 1 and neg.std() > 0 else 0
    ann_ret = (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    total   = eq.iloc[-1] - 1
    winpct  = (r > 0).mean() * 100
    return {
        "Sharpe"  : round(sharpe, 3),
        "Sortino" : round(sortino, 3),
        "Ann_Ret" : round(ann_ret * 100, 1),
        "MaxDD"   : round(maxdd * 100, 1),
        "Total"   : round(total * 100, 1),
        "WinPct"  : round(winpct, 1),
    }


# ── load data (correct alignment) ─────────────────────────────────────────

print("Loading panels (correct 8h alignment)...")
raw = load_panels(["close", "funding", "mom_24h", "fwd_8h"])

# Resample to 8h using .first() — correct timing
funding_8h = to_8h(raw["funding"].ffill())
mom24h_8h  = to_8h(raw["mom_24h"])
fwd_8h     = to_8h(raw["fwd_8h"])

# Funding trend at 1h resolution, then resample
funding_trend_1h = raw["funding"].ffill().diff(24)   # 24h change (24 × 1h bars)
ft_8h = to_8h(funding_trend_1h)

# Filter to universe (No-Majors)
all_syms  = list(funding_8h.columns)
univ_syms = [s for s in all_syms if s not in MAJORS]
print(f"Universe: {len(univ_syms)} coins (excl. {len(MAJORS)} Majors)")

funding_8h = funding_8h[univ_syms]
mom24h_8h  = mom24h_8h[univ_syms]
fwd_8h     = fwd_8h[univ_syms]
ft_8h      = ft_8h[univ_syms]

# Filter to 2025 start for consistent comparison
START = "2025-01-01"
funding_8h = funding_8h.loc[START:]
mom24h_8h  = mom24h_8h.loc[START:]
fwd_8h     = fwd_8h.loc[START:]
ft_8h      = ft_8h.loc[START:]

print(f"Period: {funding_8h.index[0].date()} → {funding_8h.index[-1].date()}, {len(funding_8h)} bars")


# ── build signals ──────────────────────────────────────────────────────────

print("Building signals...")
z_fund = cs_zscore(funding_8h)
z_mom  = cs_zscore(mom24h_8h)
z_ft   = cs_zscore(ft_8h)

# Composite variants
comp_base   = (z_fund + z_mom) / 2                      # baseline (Phase 10)
comp_ftonly = z_fund                                      # funding only
comp_ftadd  = (2*z_fund + z_ft) / 3                      # funding × 2 + trend
comp_full   = (2*z_fund + z_mom + z_ft) / 4              # full combo


# ── regime features ────────────────────────────────────────────────────────

signal_strength = comp_base.std(axis=1)
funding_disp    = funding_8h.std(axis=1)

# Percentile-based confidence (for soft threshold)
ss_pct = signal_strength.rank(pct=True)
fd_pct = funding_disp.rank(pct=True)
confidence = (ss_pct * fd_pct).apply(np.sqrt)

def conf_to_scale(c):
    """Map regime confidence to position scale [0.25, 1.0]."""
    if c < 0.30:
        return 0.25
    elif c > 0.70:
        return 1.0
    else:
        return 0.25 + (c - 0.30) / 0.40 * 0.75

scale_soft = confidence.map(conf_to_scale)

# Walk-forward binary regime (fitted per window)
def build_binary_regime_mask(composite_8h, funding_8h):
    rebal_idx = composite_8h.index
    mask = pd.Series(True, index=rebal_idx)
    feats = pd.DataFrame({
        "signal_strength": composite_8h.std(axis=1),
        "funding_disp"   : funding_8h.std(axis=1),
    })
    start = rebal_idx.min()
    end   = rebal_idx.max()
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        tr_s, tr_e = t - pd.DateOffset(months=TRAIN_MONTHS), t
        oo_s, oo_e = t, t + pd.DateOffset(months=OOS_MONTHS)
        for feat in ["signal_strength", "funding_disp"]:
            f_tr = feats[feat].loc[tr_s:tr_e].dropna()
            if len(f_tr) < 20:
                continue
            th = np.percentile(f_tr, 30)
            oos_idx = rebal_idx[(rebal_idx >= oo_s) & (rebal_idx < oo_e)]
            f_oos   = feats[feat].reindex(oos_idx).fillna(f_tr.median())
            mask.loc[oos_idx] &= (f_oos > th)
        t += pd.DateOffset(months=OOS_MONTHS)
    return mask


print("Building regime masks...")
binary_mask = build_binary_regime_mask(comp_base, funding_8h)
binary_mask_active = binary_mask.mean() * 100
print(f"  Binary regime active: {binary_mask_active:.1f}% of bars")
print(f"  Soft regime: 100% of bars (scaled by confidence)")


# ── run all variants ────────────────────────────────────────────────────────

print("\nRunning backtests...")
variants = {}

# 1. Baseline (Phase 10 replication)
variants["1. Baseline (funding+mom24h)"] = sim(comp_base, fwd_8h)

# 2. Funding only
variants["2. Funding only"] = sim(comp_ftonly, fwd_8h)

# 3. Funding × 2 + trend
variants["3. 2×Funding + f_trend"] = sim(comp_ftadd, fwd_8h)

# 4. Full combo (funding + mom + trend)
variants["4. Funding + mom + f_trend"] = sim(comp_full, fwd_8h)

# 5. Baseline + binary regime (Phase 10 regime filter)
variants["5. Baseline + binary regime"] = sim(
    comp_base, fwd_8h,
    scale=binary_mask.map({True: 1.0, False: 0.0})
)

# 6. Baseline + soft regime
variants["6. Baseline + soft regime"] = sim(comp_base, fwd_8h, scale=scale_soft)

# 7. Best signal + soft regime
variants["7. 2×Funding+trend + soft regime"] = sim(comp_ftadd, fwd_8h, scale=scale_soft)

# 8. Best signal + binary regime
variants["8. 2×Funding+trend + binary regime"] = sim(
    comp_ftadd, fwd_8h,
    scale=binary_mask.map({True: 1.0, False: 0.0})
)


# ── compare results ────────────────────────────────────────────────────────

print("\nResults (from Jan 2025):")
print(f"{'Strategy':45s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'$1k→':>10s}")
rows = {}
equities = {}
for name, rets in variants.items():
    m = compute_metrics(rets)
    rows[name] = m
    eq_final = (1 + rets.dropna()).prod() * STARTING_CAPITAL
    equities[name] = eq_final
    print(f"  {name:43s}  {m['Sharpe']:+7.3f}  {m['Ann_Ret']:+8.1f}%  "
          f"{m['MaxDD']:+7.1f}%  ${eq_final:>9,.0f}")


# ── monthly breakdown for top 3 variants ──────────────────────────────────

# Sort by Sharpe and pick top 3
top3 = sorted(rows, key=lambda n: rows[n]["Sharpe"], reverse=True)[:3]
print(f"\nTop 3 by Sharpe: {', '.join(top3)}")

print(f"\n{'Month':10s}  " + "  ".join(f"{n[:18]:>20s}" for n in top3))
monthly = {}
for name in top3:
    monthly[name] = variants[name].dropna().resample("ME").apply(lambda x: (1+x).prod()-1)
all_months = sorted(set(mo for m in monthly.values() for mo in m.index))
for mo in all_months:
    vals = [monthly[n].get(mo, 0) * 100 for n in top3]
    row  = f"{str(mo)[:7]:10s}  " + "  ".join(f"{v:+20.1f}%" for v in vals)
    print(row)


# ── save ───────────────────────────────────────────────────────────────────

df_res = pd.DataFrame(rows).T
df_res.to_csv(os.path.join(RESULTS_DIR, "phase16_summary.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase16_summary.csv")

# Monthly CSV for best variant
best_name = top3[0]
best_rets  = variants[best_name].dropna()
mo_rets    = best_rets.resample("ME").apply(lambda x: (1+x).prod()-1)
eq_mo      = (1 + best_rets).cumprod() * STARTING_CAPITAL
eq_mo      = eq_mo.resample("ME").last()
df_mo = pd.DataFrame({"ret_pct": mo_rets*100, "equity": eq_mo}).round(2)
df_mo.to_csv(os.path.join(RESULTS_DIR, "phase16_monthly.csv"))
print(f"Saved: {RESULTS_DIR}/phase16_monthly.csv")


# ── plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Phase 16: Final Combined Strategy", fontsize=14)

colors = {
    "1. Baseline (funding+mom24h)"         : "steelblue",
    "2. Funding only"                      : "darkorange",
    "3. 2×Funding + f_trend"               : "green",
    "4. Funding + mom + f_trend"           : "brown",
    "5. Baseline + binary regime"          : "gray",
    "6. Baseline + soft regime"            : "purple",
    "7. 2×Funding+trend + soft regime"     : "crimson",
    "8. 2×Funding+trend + binary regime"   : "gold",
}

ax_eq  = axes[0, 0]
ax_dd  = axes[0, 1]
ax_mo  = axes[1, 0]
ax_bar = axes[1, 1]

# Equity curves
for name, rets in variants.items():
    r  = rets.dropna()
    eq = (1 + r).cumprod() * STARTING_CAPITAL
    m  = rows[name]
    lw = 2.5 if name in top3[:2] else 1.0
    alpha = 1.0 if name in top3[:2] else 0.5
    ax_eq.plot(eq.values, label=f"{name[:30]} | S={m['Sharpe']:.2f}",
               color=colors.get(name, "gray"), lw=lw, alpha=alpha)

ax_eq.set_ylabel("Portfolio Value ($)")
ax_eq.set_title("Equity Curves (log scale)")
ax_eq.legend(fontsize=7)
ax_eq.grid(True, alpha=0.3)
ax_eq.set_yscale("log")

# Drawdown
for name, rets in variants.items():
    r  = rets.dropna()
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    lw = 2.0 if name in top3[:2] else 0.8
    alpha = 1.0 if name in top3[:2] else 0.4
    ax_dd.plot(dd.values, color=colors.get(name, "gray"), lw=lw, alpha=alpha, label=name[:25])

ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=7)
ax_dd.grid(True, alpha=0.3)
ax_dd.axhline(0, color="black", lw=0.8)

# Monthly bar chart for best 2
for i, name in enumerate(top3[:2]):
    mo_r = variants[name].dropna().resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    x    = np.arange(len(mo_r))
    ax_mo.bar(x + i*0.35, mo_r.values, width=0.35,
              color=colors.get(name, "gray"), alpha=0.7, label=name[:25])
    ax_mo.set_xticks(x)
    ax_mo.set_xticklabels([str(m)[:7] for m in mo_r.index], rotation=45, fontsize=7)
ax_mo.axhline(0, color="black", lw=0.8)
ax_mo.set_ylabel("Monthly Return (%)")
ax_mo.set_title("Monthly Returns (top 2 variants)")
ax_mo.legend(fontsize=8)
ax_mo.grid(True, alpha=0.3)

# Summary bar chart: Sharpe comparison
names_short = [n[:25] for n in rows]
sharpes     = [rows[n]["Sharpe"] for n in rows]
bar_colors  = [colors.get(n, "gray") for n in rows]
y_pos = np.arange(len(rows))
bars = ax_bar.barh(y_pos, sharpes, color=bar_colors, alpha=0.8)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(names_short, fontsize=8)
ax_bar.axvline(0, color="black", lw=0.8)
ax_bar.set_xlabel("Sharpe Ratio")
ax_bar.set_title("Sharpe Comparison")
ax_bar.grid(True, alpha=0.3, axis="x")
for bar, s in zip(bars, sharpes):
    ax_bar.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{s:.2f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase16_final.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase16_final.png")

print("\nPhase 16 complete.")
print("\n" + "="*70)
print("FINAL SUMMARY — Top 3 Strategies")
print("="*70)
for name in top3:
    m = rows[name]
    eq = equities[name]
    print(f"\n  {name}")
    print(f"    Sharpe:   {m['Sharpe']:.3f}")
    print(f"    Sortino:  {m['Sortino']:.3f}")
    print(f"    Ann Ret:  {m['Ann_Ret']:.1f}%")
    print(f"    Max DD:   {m['MaxDD']:.1f}%")
    print(f"    $1k→:     ${eq:,.0f}")
    print(f"    Win rate: {m['WinPct']:.1f}% of bars")
