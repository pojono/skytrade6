"""
Phase 21: Final Stable Strategy — Combining Best Improvements

From Phases 17-20:
  KEEP:   Inverse scaling (Phase 18C/D) — MaxDD -32% → -20%, minimal Sharpe cost
  KEEP:   Low-funding gate (Phase 20B)  — marginal MaxDD improvement
  REJECT: Adaptive N                    — switching to N=20 in weak regimes hurts
  REJECT: MR layer                      — catastrophically bad standalone (-1.5 Sharpe)
  REJECT: High-funding gate             — high funding is actually GOOD for the strategy

Combined variants tested:
  1. Phase 16 baseline (2×fund + mom + f_trend, N=10)
  2. Phase 16 + Inverse scaling
  3. Phase 16 + Harvest + Inverse (best Sortino)
  4. Phase 16 + Low-funding gate
  5. Phase 16 + Inverse scaling + Low-funding gate
  6. Phase 16 with N=15 (smoother but lower return)
  7. Phase 16 + N=15 + Inverse scaling

Goal: find the best Sharpe/MaxDD/Sortino tradeoff.
Output: final recommended configuration with full monthly table.
"""

import os, glob, warnings
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
FEE_RT           = 4 * 2 / 10000
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
START            = "2025-01-01"
STARTING_CAPITAL = 1000.0
SHARPE_WINDOW    = 30
FUNDING_WINDOW   = 180
MONTHLY_CAP      = 0.30

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR,"*.parquet")))
    data  = {c:{} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet","")
        df  = pd.read_parquet(fpath)
        for c in cols:
            if c in df.columns:
                data[c][sym] = df[c]
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

def sim(composite, fwd_8h, n=10, scale_series=None):
    """General simulator with optional per-bar scaling."""
    common = composite.index.intersection(fwd_8h.index)
    records = []
    prev_l, prev_s = set(), set()
    for ts in common:
        sig   = composite.loc[ts].dropna()
        fwd   = fwd_8h.loc[ts, sig.index].dropna()
        sig   = sig.loc[fwd.index]
        scale = float(scale_series.get(ts, 1.0)) if scale_series is not None else 1.0
        if len(sig) < n*4:
            records.append((ts, np.nan))
            prev_l = prev_s = set()
            continue
        ranked = sig.rank()
        longs  = set(ranked.nlargest(n).index)
        shorts = set(ranked.nsmallest(n).index)
        lr, sr = fwd.loc[list(longs)].mean(), fwd.loc[list(shorts)].mean()
        if np.isnan(lr) or np.isnan(sr):
            records.append((ts, np.nan))
            continue
        gross    = lr - sr
        turnover = (len((longs-prev_l)|(shorts-prev_s))/(n+n)
                    if prev_l|prev_s else 1.0)
        records.append((ts, (gross - turnover*FEE_RT) * scale))
        prev_l, prev_s = longs, shorts
    idx, vals = zip(*records)
    return pd.Series(vals, index=idx)

def inverse_scale_fn(rets, window=SHARPE_WINDOW):
    """Reduce when rolling Sharpe very high (>5) or negative (<0)."""
    scaled = []
    for i, r in enumerate(rets):
        if np.isnan(r) or i < window:
            scaled.append(r)
            continue
        past = rets.iloc[i-window:i].dropna()
        if len(past) < 10 or past.std()==0:
            scaled.append(r)
            continue
        rs = past.mean()/past.std()*np.sqrt(PERIODS_PER_YEAR)
        scale = 0.5 if rs > 5 else (1.0 if rs > 1 else (0.75 if rs > 0 else 0.5))
        scaled.append(r * scale)
    return pd.Series(scaled, index=rets.index)

def harvest_inverse_fn(rets, window=SHARPE_WINDOW, cap=MONTHLY_CAP):
    scaled = []
    eq_mtd, cur_month = 1.0, None
    for i, (ts, r) in enumerate(rets.items()):
        month = (ts.year, ts.month)
        if month != cur_month:
            cur_month, eq_mtd = month, 1.0
        if np.isnan(r):
            scaled.append(np.nan)
            continue
        # Sharpe scale
        if i >= window:
            past = rets.iloc[i-window:i].dropna()
            rs = past.mean()/past.std()*np.sqrt(PERIODS_PER_YEAR) if (len(past)>=10 and past.std()>0) else 2.0
            sharpe_sc = 0.5 if rs > 5 else (1.0 if rs > 1 else 0.5)
        else:
            sharpe_sc = 1.0
        # Harvest scale
        mtd = eq_mtd - 1
        harvest_sc = 0.25 if mtd >= cap else (0.5 if mtd >= cap*0.7 else 1.0)
        scale    = sharpe_sc * harvest_sc
        scaled_r = r * scale
        eq_mtd  *= (1 + scaled_r)
        scaled.append(scaled_r)
    return pd.Series(scaled, index=rets.index)

def metrics(rets):
    r = rets.dropna()
    if len(r)==0: return dict(Sharpe=0,Sortino=0,Ann_Ret=0,MaxDD=0,Total=0,WinPct=0)
    sharpe  = r.mean()/r.std()*np.sqrt(PERIODS_PER_YEAR) if r.std()>0 else 0
    neg     = r[r<0]
    sortino = r.mean()/neg.std()*np.sqrt(PERIODS_PER_YEAR) if len(neg)>1 and neg.std()>0 else 0
    ann_ret = (1+r).prod()**(PERIODS_PER_YEAR/len(r))-1
    eq      = (1+r).cumprod()
    maxdd   = (eq/eq.cummax()-1).min()
    total   = eq.iloc[-1]-1
    winpct  = (r>0).mean()*100
    return dict(Sharpe=round(sharpe,3),Sortino=round(sortino,3),
                Ann_Ret=round(ann_ret*100,1),MaxDD=round(maxdd*100,1),
                Total=round(total*100,1),WinPct=round(winpct,1))

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data (correct .first() alignment)...")
raw = load_panels(["funding","mom_24h","fwd_8h"])
funding_8h = to_8h(raw["funding"].ffill())
mom24h_8h  = to_8h(raw["mom_24h"])
fwd_8h     = to_8h(raw["fwd_8h"])
ft_8h      = to_8h(raw["funding"].ffill().diff(24))

univ = [s for s in funding_8h.columns if s not in MAJORS]
funding_8h = funding_8h[univ].loc[START:]
mom24h_8h  = mom24h_8h[univ].loc[START:]
fwd_8h     = fwd_8h[univ].loc[START:]
ft_8h      = ft_8h[univ].loc[START:]

z_f  = cs_zscore(funding_8h)
z_m  = cs_zscore(mom24h_8h)
z_ft = cs_zscore(ft_8h)
composite = (2*z_f + z_m + z_ft)/4   # Phase 16 signal

# Funding level percentile (no look-ahead)
avg_funding  = funding_8h.mean(axis=1)
funding_pct  = avg_funding.rolling(FUNDING_WINDOW, min_periods=30).rank(pct=True)
low_fund_gate = funding_pct.map(lambda p: 0.5 if (not pd.isna(p) and p < 0.10) else 1.0)

print(f"Universe: {len(univ)} coins | Bars: {len(composite)}")

# ── run baseline strategies ────────────────────────────────────────────────
print("Running baseline strategies...")
rets_n10   = sim(composite, fwd_8h, n=10)
rets_n15   = sim(composite, fwd_8h, n=15)

# Apply inverse scaling
rets_inv10 = inverse_scale_fn(rets_n10)
rets_hi10  = harvest_inverse_fn(rets_n10)
rets_inv15 = inverse_scale_fn(rets_n15)

# Low-funding gate applied on top of inverse scaling
rets_inv10_lfg = rets_inv10 * low_fund_gate.reindex(rets_inv10.index, fill_value=1.0)
rets_n10_lfg   = rets_n10   * low_fund_gate.reindex(rets_n10.index, fill_value=1.0)

variants = {
    "1. Baseline N=10 (Phase 16)"         : rets_n10,
    "2. N=10 + Inverse scale"             : rets_inv10,
    "3. N=10 + Harvest+Inverse"           : rets_hi10,
    "4. N=10 + Low-funding gate"          : rets_n10_lfg,
    "5. N=10 + Inverse + Low-fund gate"   : rets_inv10_lfg,
    "6. N=15 (smoother)"                  : rets_n15,
    "7. N=15 + Inverse scale"             : rets_inv15,
}

print(f"\n{'Strategy':45s}  {'Sharpe':>7s}  {'Sortino':>7s}  {'MaxDD':>7s}  {'$1k→':>10s}")
rows = {}
for name, rets in variants.items():
    m  = metrics(rets)
    rows[name] = m
    eq = (1+rets.dropna()).prod()*STARTING_CAPITAL
    print(f"  {name:43s}  {m['Sharpe']:+7.3f}  {m['Sortino']:+7.3f}  {m['MaxDD']:+7.1f}%  ${eq:>9,.0f}")

# ── full monthly table ─────────────────────────────────────────────────────
print("\n" + "─"*90)
print("MONTHLY BREAKDOWN — All Variants")
print("─"*90)

monthly = {n: variants[n].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100
           for n in variants}
eq_monthly = {n: (1+variants[n].dropna()).cumprod().resample("ME").last()*STARTING_CAPITAL
              for n in variants}

cols_show = list(variants.keys())
header = f"{'Month':10s}  " + "  ".join(f"{n[:14]:>14s}" for n in cols_show)
print(header)

all_months = sorted(monthly[cols_show[0]].index)
for mo in all_months:
    vals = [monthly[n].get(mo, 0) for n in cols_show]
    row  = f"{str(mo)[:7]:10s}  " + "  ".join(f"{v:+14.1f}%" for v in vals)
    # flag bad months
    if vals[0] < -10:
        row += "  ◄ BAD"
    elif vals[0] > 80:
        row += "  ◄ GREAT"
    print(row)

# Equity row
print("\nFinal equity:")
for name in cols_show:
    eq = (1+variants[name].dropna()).prod()*STARTING_CAPITAL
    print(f"  {name[:43]:43s}  ${eq:>9,.0f}")

# ── best variant detail ────────────────────────────────────────────────────
best = max(rows, key=lambda n: rows[n]["Sharpe"])
# Among top-3 Sharpe, pick one that also has good MaxDD
top3 = sorted(rows, key=lambda n: rows[n]["Sharpe"], reverse=True)[:3]
# Prefer the one with MaxDD < -30%
best_balanced = min(top3, key=lambda n: abs(rows[n]["MaxDD"]+25))  # closest to -25%

print(f"\n{'='*70}")
print(f"RECOMMENDED: {best_balanced}")
print(f"{'='*70}")
m = rows[best_balanced]
eq_final = (1+variants[best_balanced].dropna()).prod()*STARTING_CAPITAL
print(f"  Sharpe:   {m['Sharpe']:.3f}")
print(f"  Sortino:  {m['Sortino']:.3f}")
print(f"  Ann Ret:  {m['Ann_Ret']:.1f}%")
print(f"  Max DD:   {m['MaxDD']:.1f}%")
print(f"  $1k→:     ${eq_final:,.0f}")
print(f"  Win rate: {m['WinPct']:.1f}% of bars")

# Consistency metrics
r = variants[best_balanced].dropna()
monthly_r = r.resample("ME").apply(lambda x:(1+x).prod()-1)*100
pos_months = (monthly_r > 0).sum()
print(f"  Positive months: {pos_months}/{len(monthly_r)} ({pos_months/len(monthly_r)*100:.0f}%)")
print(f"  Worst month: {monthly_r.min():.1f}%")
print(f"  Best month:  {monthly_r.max():.1f}%")
print(f"  Monthly vol: {monthly_r.std():.1f}%")

# ── save ───────────────────────────────────────────────────────────────────
pd.DataFrame(rows).T.to_csv(os.path.join(RESULTS_DIR,"phase21_final_stable.csv"))

fig, axes = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle("Phase 21: Final Stable Strategy — All Combinations", fontsize=14)

colors = {
    "1. Baseline N=10 (Phase 16)"       : "steelblue",
    "2. N=10 + Inverse scale"           : "darkorange",
    "3. N=10 + Harvest+Inverse"         : "crimson",
    "4. N=10 + Low-funding gate"        : "gray",
    "5. N=10 + Inverse + Low-fund gate" : "purple",
    "6. N=15 (smoother)"                : "brown",
    "7. N=15 + Inverse scale"           : "green",
}

ax_eq, ax_dd = axes[0, 0], axes[0, 1]
ax_mo, ax_bar = axes[1, 0], axes[1, 1]

for name, rets in variants.items():
    r  = rets.dropna()
    eq = (1+r).cumprod()*STARTING_CAPITAL
    m  = rows[name]
    lw = 2.5 if name in ("1. Baseline N=10 (Phase 16)", best_balanced) else 1.0
    alpha = 1.0 if name in ("1. Baseline N=10 (Phase 16)", best_balanced,
                             "2. N=10 + Inverse scale", "3. N=10 + Harvest+Inverse") else 0.4
    ax_eq.plot(eq.values, label=f"{name[:28]} | S={m['Sharpe']:.2f}",
               color=colors.get(name,"gray"), lw=lw, alpha=alpha)
    dd = (eq/eq.cummax()-1)*100
    ax_dd.plot(dd.values, color=colors.get(name,"gray"), lw=lw, alpha=alpha, label=name[:28])

ax_eq.set_yscale("log"); ax_eq.set_ylabel("Equity ($)")
ax_eq.set_title("Equity Curves (log)"); ax_eq.legend(fontsize=7); ax_eq.grid(True,alpha=0.3)
ax_dd.set_ylabel("Drawdown (%)"); ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=7); ax_dd.axhline(0,color="black",lw=0.8); ax_dd.grid(True,alpha=0.3)

# Monthly bars: baseline vs recommended
names_plot = ["1. Baseline N=10 (Phase 16)", best_balanced]
mo_b = monthly[names_plot[0]]
mo_r = monthly[best_balanced]
x    = np.arange(len(mo_b))
ax_mo.bar(x-0.2, mo_b.values, 0.4, color="steelblue", alpha=0.7, label="Baseline N=10")
ax_mo.bar(x+0.2, mo_r.values, 0.4, color="crimson",   alpha=0.7, label=best_balanced[:25])
ax_mo.set_xticks(x)
ax_mo.set_xticklabels([str(m)[:7] for m in mo_b.index], rotation=45, fontsize=7)
ax_mo.axhline(0, color="black", lw=0.8); ax_mo.grid(True,alpha=0.3)
ax_mo.set_title("Monthly Returns: Baseline vs Recommended"); ax_mo.legend(fontsize=8)

# Sharpe vs MaxDD scatter
for name, m in rows.items():
    ax_bar.scatter(abs(m["MaxDD"]), m["Sharpe"], s=100, color=colors.get(name,"gray"),
                   zorder=5, label=name[:25])
    ax_bar.annotate(name[:18], (abs(m["MaxDD"]), m["Sharpe"]),
                    xytext=(3,3), textcoords="offset points", fontsize=7)
ax_bar.set_xlabel("|MaxDD| (%)")
ax_bar.set_ylabel("Sharpe Ratio")
ax_bar.set_title("Sharpe vs MaxDD (lower-right = better)")
ax_bar.grid(True, alpha=0.3)
# ideal region
ax_bar.axhline(3.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax_bar.axvline(25, color="gray", ls="--", lw=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"phase21_final_stable.png"),dpi=150,bbox_inches="tight")
plt.close()
print(f"\nSaved. Phase 21 complete.")
