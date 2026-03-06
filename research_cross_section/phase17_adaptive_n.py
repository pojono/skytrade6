"""
Phase 17: Adaptive N (variable portfolio width)

When signal dispersion is LOW (uncertain regime): use N=20 (diversified, lower risk)
When signal dispersion is MEDIUM: use N=10 (baseline)
When signal dispersion is HIGH (strong regime): use N=5 (concentrated, max alpha)

Signal dispersion = CS std of composite score at each bar.
Thresholds: rolling percentiles (look-back 180 bars = 60 days) to avoid look-ahead.

Also tests: fixed-N alternatives (N=5, 10, 15, 20) for reference.

Uses Phase 16 signal: 2×funding + mom_24h + funding_trend (2:1:1)
Uses correct .first() resample alignment.

Outputs:
  results/phase17_adaptive_n.png
  results/phase17_adaptive_n.csv
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
FEE_MAKER_BPS    = 4
FEE_RT           = FEE_MAKER_BPS * 2 / 10000
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
START            = "2025-01-01"
PCTILE_WINDOW    = 180   # 60-day rolling for thresholds

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.parquet")))
    data  = {c: {} for c in cols}
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

def sim_adaptive(composite, fwd_8h, n_series, fee_rt=FEE_RT, min_universe=20):
    """n_series: Series of int indexed by date (per-bar N for long AND short leg)."""
    common = composite.index.intersection(fwd_8h.index)
    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = composite.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]
        n   = int(n_series.get(ts, 10))

        if len(sig) < n * 4:
            records.append((ts, np.nan))
            prev_longs = prev_shorts = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n).index)
        shorts = set(ranked.nsmallest(n).index)

        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        if np.isnan(lr) or np.isnan(sr):
            records.append((ts, np.nan))
            continue

        gross    = lr - sr
        turnover = (len((longs-prev_longs)|(shorts-prev_shorts)) / (n+n)
                    if prev_longs|prev_shorts else 1.0)
        net = gross - turnover * fee_rt
        records.append((ts, net))
        prev_longs, prev_shorts = longs, shorts

    idx, vals = zip(*records)
    return pd.Series(vals, index=idx)

def metrics(rets):
    r = rets.dropna()
    if len(r) == 0:
        return dict(Sharpe=0, Sortino=0, Ann_Ret=0, MaxDD=0, Total=0)
    sharpe  = r.mean()/r.std()*np.sqrt(PERIODS_PER_YEAR) if r.std()>0 else 0
    neg     = r[r<0]
    sortino = r.mean()/neg.std()*np.sqrt(PERIODS_PER_YEAR) if len(neg)>1 and neg.std()>0 else 0
    ann_ret = (1+r).prod()**(PERIODS_PER_YEAR/len(r))-1
    eq      = (1+r).cumprod()
    maxdd   = (eq/eq.cummax()-1).min()
    total   = eq.iloc[-1]-1
    return dict(Sharpe=round(sharpe,3), Sortino=round(sortino,3),
                Ann_Ret=round(ann_ret*100,1), MaxDD=round(maxdd*100,1),
                Total=round(total*100,1))

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
raw = load_panels(["funding","mom_24h","fwd_8h"])
funding_8h = to_8h(raw["funding"].ffill())
mom24h_8h  = to_8h(raw["mom_24h"])
fwd_8h     = to_8h(raw["fwd_8h"])
ft_1h      = raw["funding"].ffill().diff(24)
ft_8h      = to_8h(ft_1h)

univ = [s for s in funding_8h.columns if s not in MAJORS]
funding_8h = funding_8h[univ].loc[START:]
mom24h_8h  = mom24h_8h[univ].loc[START:]
fwd_8h     = fwd_8h[univ].loc[START:]
ft_8h      = ft_8h[univ].loc[START:]

z_f  = cs_zscore(funding_8h)
z_m  = cs_zscore(mom24h_8h)
z_ft = cs_zscore(ft_8h)
composite = (2*z_f + z_m + z_ft) / 4

print(f"Universe: {len(univ)} coins | Bars: {len(composite)}")

# ── signal dispersion (rolling percentile, no look-ahead) ──────────────────
signal_strength = composite.std(axis=1)

# Rolling percentile: what % of past PCTILE_WINDOW bars had lower dispersion
ss_pct = signal_strength.rolling(PCTILE_WINDOW, min_periods=30).rank(pct=True)

# Adaptive N: low dispersion → N=20, medium → N=10, high → N=5
def pct_to_n(p):
    if pd.isna(p) or p < 0.33:
        return 20
    elif p < 0.67:
        return 10
    else:
        return 5

n_adaptive = ss_pct.map(pct_to_n)

print("N distribution:")
for n_val in [5, 10, 20]:
    pct = (n_adaptive == n_val).mean() * 100
    print(f"  N={n_val}: {pct:.1f}% of bars")

# ── run strategies ─────────────────────────────────────────────────────────
print("\nRunning backtests...")
variants = {}

# Fixed N baselines
for n in [5, 10, 15, 20]:
    n_fixed = pd.Series(n, index=composite.index)
    variants[f"Fixed N={n}"] = sim_adaptive(composite, fwd_8h, n_fixed)

# Adaptive N (3-tier)
variants["Adaptive N (5/10/20)"] = sim_adaptive(composite, fwd_8h, n_adaptive)

# Adaptive N (2-tier: 5 high, 15 low)
n_2tier = ss_pct.map(lambda p: 5 if (not pd.isna(p) and p >= 0.67) else 15)
variants["Adaptive N (5/15)"] = sim_adaptive(composite, fwd_8h, n_2tier)

# Adaptive N with concentration: top 70% → N=5, bottom 30% → N=25
n_extreme = ss_pct.map(lambda p: 5 if (not pd.isna(p) and p >= 0.70) else 25)
variants["Adaptive N (5/25 extreme)"] = sim_adaptive(composite, fwd_8h, n_extreme)

print(f"\n{'Strategy':35s}  {'Sharpe':>7s}  {'Sortino':>7s}  {'MaxDD':>7s}  {'$1k→':>10s}")
rows = {}
for name, rets in variants.items():
    m   = metrics(rets)
    rows[name] = m
    eq  = (1+rets.dropna()).prod() * 1000
    print(f"  {name:33s}  {m['Sharpe']:+7.3f}  {m['Sortino']:+7.3f}  {m['MaxDD']:+7.1f}%  ${eq:>9,.0f}")

# ── monthly table: baseline vs best ───────────────────────────────────────
best = max(rows, key=lambda n: rows[n]["Sharpe"])
base = "Fixed N=10"
print(f"\nBest: {best} | Sharpe={rows[best]['Sharpe']:.3f}")

print(f"\n{'Month':10s}  {'N=10 (base)':>13s}  {best[:20]:>22s}  {'Diff':>8s}")
r_base = variants[base].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)
r_best = variants[best].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)
for mo in r_base.index:
    rb, rv = r_base.get(mo,0)*100, r_best.get(mo,0)*100
    flag = " ▲" if rv-rb>3 else (" ▼" if rv-rb<-3 else "")
    print(f"{str(mo)[:7]:10s}  {rb:+13.1f}%  {rv:+22.1f}%  {rv-rb:+8.1f}%{flag}")

# ── save ───────────────────────────────────────────────────────────────────
pd.DataFrame(rows).T.to_csv(os.path.join(RESULTS_DIR,"phase17_adaptive_n.csv"))

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
fig.suptitle("Phase 17: Adaptive N (variable portfolio width)", fontsize=14)
colors = {"Fixed N=5":"purple","Fixed N=10":"steelblue","Fixed N=15":"gray",
          "Fixed N=20":"brown","Adaptive N (5/10/20)":"crimson",
          "Adaptive N (5/15)":"darkorange","Adaptive N (5/25 extreme)":"green"}

for name, rets in variants.items():
    r  = rets.dropna()
    eq = (1+r).cumprod()*1000
    m  = rows[name]
    lw = 2.5 if name in (base, best) else 1.0
    alpha = 1.0 if name in (base, best, "Adaptive N (5/10/20)") else 0.5
    ax1.plot(eq.values, label=f"{name} | S={m['Sharpe']:.2f} DD={m['MaxDD']:.0f}%",
             color=colors.get(name,"gray"), lw=lw, alpha=alpha)
    dd = (eq/eq.cummax()-1)*100
    ax2.plot(dd.values, color=colors.get(name,"gray"), lw=lw, alpha=alpha, label=name)

ax1.set_yscale("log"); ax1.set_ylabel("Equity ($)"); ax1.legend(fontsize=8)
ax1.set_title("Equity Curves (log)"); ax1.grid(True,alpha=0.3)
ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8)
ax2.axhline(0,color="black",lw=0.8); ax2.grid(True,alpha=0.3)
ax2.set_title("Drawdown")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"phase17_adaptive_n.png"),dpi=150,bbox_inches="tight")
plt.close()
print(f"\nSaved results. Phase 17 complete.")
