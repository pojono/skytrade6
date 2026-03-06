"""
Phase 19: Mean-Reversion Counter-Cycle Layer

Phase 1 showed mom_8h has ICIR -0.172 (significant NEGATIVE IC).
This means short-term momentum reverses at the 8h horizon.
A COUNTER-TREND signal (long recent losers, short recent winners by 8h return)
should profit from reversals — opposite of the main strategy.

The goal: combine main strategy (momentum/carry) + MR layer at 20-30% allocation.
During monster months (strong momentum), MR layer loses a bit.
During reversals and flat months, MR layer gains, cushioning drawdowns.

Tests:
  - MR layer standalone performance
  - Blended portfolios: 80/20, 70/30, 60/40 momentum/MR
  - Optimal blend (by Sharpe and by Sharpe×stability)

Uses Phase 16 signal for main strategy.
MR signal: -mom_8h (inverse 8h momentum = mean reversion).
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
N_LONG = N_SHORT = 10

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

def sim_strategy(composite, fwd_8h, n=N_LONG):
    """Single composite → L/S bar returns."""
    common = composite.index.intersection(fwd_8h.index)
    records = []
    prev_l, prev_s = set(), set()
    for ts in common:
        sig = composite.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]
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
        records.append((ts, gross - turnover*FEE_RT))
        prev_l, prev_s = longs, shorts
    idx, vals = zip(*records)
    return pd.Series(vals, index=idx)

def metrics(rets, label=""):
    r = rets.dropna()
    if len(r)==0: return dict(Sharpe=0,Sortino=0,Ann_Ret=0,MaxDD=0,Corr_to_base=np.nan)
    sharpe  = r.mean()/r.std()*np.sqrt(PERIODS_PER_YEAR) if r.std()>0 else 0
    neg     = r[r<0]
    sortino = r.mean()/neg.std()*np.sqrt(PERIODS_PER_YEAR) if len(neg)>1 and neg.std()>0 else 0
    ann_ret = (1+r).prod()**(PERIODS_PER_YEAR/len(r))-1
    eq      = (1+r).cumprod()
    maxdd   = (eq/eq.cummax()-1).min()
    return dict(Sharpe=round(sharpe,3),Sortino=round(sortino,3),
                Ann_Ret=round(ann_ret*100,1),MaxDD=round(maxdd*100,1))

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
raw = load_panels(["funding","mom_24h","mom_8h","fwd_8h"])
funding_8h = to_8h(raw["funding"].ffill())
mom24h_8h  = to_8h(raw["mom_24h"])
mom8h_8h   = to_8h(raw["mom_8h"])
fwd_8h     = to_8h(raw["fwd_8h"])
ft_8h      = to_8h(raw["funding"].ffill().diff(24))

univ = [s for s in funding_8h.columns if s not in MAJORS]
funding_8h = funding_8h[univ].loc[START:]
mom24h_8h  = mom24h_8h[univ].loc[START:]
mom8h_8h   = mom8h_8h[univ].loc[START:]
fwd_8h     = fwd_8h[univ].loc[START:]
ft_8h      = ft_8h[univ].loc[START:]

# Main signal (Phase 16)
z_f  = cs_zscore(funding_8h)
z_m  = cs_zscore(mom24h_8h)
z_ft = cs_zscore(ft_8h)
comp_main = (2*z_f + z_m + z_ft)/4

# MR signal: NEGATIVE 8h momentum (short winners, long losers)
z_8h    = cs_zscore(mom8h_8h)
comp_mr = -z_8h   # flip: long recent losers, short recent winners

print(f"Universe: {len(univ)} coins | Bars: {len(comp_main)}")

# ── run strategies ─────────────────────────────────────────────────────────
print("Running backtests...")
rets_main = sim_strategy(comp_main, fwd_8h)
rets_mr   = sim_strategy(comp_mr,   fwd_8h)

# Correlation between main and MR strategies
common_idx = rets_main.dropna().index.intersection(rets_mr.dropna().index)
corr = rets_main.loc[common_idx].corr(rets_mr.loc[common_idx])
print(f"\nCorrelation between main strategy and MR layer: {corr:.3f}")

# Blended portfolios
blends = {}
for w_main in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    w_mr   = 1.0 - w_main
    name   = f"Main {int(w_main*100)}% + MR {int(w_mr*100)}%"
    blend  = w_main * rets_main + w_mr * rets_mr
    blends[name] = blend

# Print results
print(f"\n{'Strategy':40s}  {'Sharpe':>7s}  {'Sortino':>7s}  {'MaxDD':>7s}  {'$1k→':>10s}")
all_variants = {"Main only (baseline)": rets_main, "MR only": rets_mr}
all_variants.update(blends)

rows = {}
for name, rets in all_variants.items():
    m  = metrics(rets)
    rows[name] = m
    eq = (1+rets.dropna()).prod()*1000
    print(f"  {name:38s}  {m['Sharpe']:+7.3f}  {m['Sortino']:+7.3f}  {m['MaxDD']:+7.1f}%  ${eq:>9,.0f}")

# Monthly correlation analysis
print("\nMonthly returns — main vs MR:")
mo_main = rets_main.dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100
mo_mr   = rets_mr.dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100
mo_80   = blends["Main 80% + MR 20%"].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100
mo_70   = blends["Main 70% + MR 30%"].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100

print(f"\n{'Month':10s}  {'Main':>10s}  {'MR':>10s}  {'80/20':>10s}  {'70/30':>10s}")
for mo in mo_main.index:
    rm  = mo_main.get(mo,0)
    rmr = mo_mr.get(mo,0)
    r80 = mo_80.get(mo,0)
    r70 = mo_70.get(mo,0)
    flag = " ←" if abs(rm)>20 else ""
    print(f"{str(mo)[:7]:10s}  {rm:+10.1f}%  {rmr:+10.1f}%  {r80:+10.1f}%  {r70:+10.1f}%{flag}")

# save
pd.DataFrame(rows).T.to_csv(os.path.join(RESULTS_DIR,"phase19_mr_layer.csv"))

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,14))
fig.suptitle("Phase 19: Mean-Reversion Counter-Cycle Layer", fontsize=14)

colors = {
    "Main only (baseline)":"steelblue","MR only":"crimson",
    "Main 90% + MR 10%":"#aaa","Main 80% + MR 20%":"darkorange",
    "Main 70% + MR 30%":"green","Main 60% + MR 40%":"purple",
    "Main 50% + MR 50%":"brown",
}

for name, rets in all_variants.items():
    r  = rets.dropna()
    eq = (1+r).cumprod()*1000
    m  = rows[name]
    lw = 2.0 if name in ("Main only (baseline)","Main 80% + MR 20%","Main 70% + MR 30%") else 1.0
    alpha = 0.4 if name in ("Main 90% + MR 10%","Main 50% + MR 50%","MR only") else 1.0
    ax1.plot(eq.values,label=f"{name} | S={m['Sharpe']:.2f}",
             color=colors.get(name,"gray"),lw=lw,alpha=alpha)
    dd = (eq/eq.cummax()-1)*100
    ax2.plot(dd.values,color=colors.get(name,"gray"),lw=lw,alpha=alpha,label=name)

# Scatter: correlation between main and MR monthly
ax3.scatter(mo_main.values, mo_mr.values, alpha=0.7, color="steelblue", s=60)
for mo, rm, rmr in zip(mo_main.index, mo_main.values, mo_mr.values):
    ax3.annotate(str(mo)[:7], (rm, rmr), fontsize=7, alpha=0.7)
ax3.axhline(0, color="black", lw=0.8)
ax3.axvline(0, color="black", lw=0.8)
ax3.set_xlabel("Main strategy monthly return (%)")
ax3.set_ylabel("MR layer monthly return (%)")
ax3.set_title(f"Monthly scatter — correlation = {corr:.3f}")
ax3.grid(True, alpha=0.3)

ax1.set_yscale("log"); ax1.set_ylabel("Equity ($)"); ax1.legend(fontsize=8)
ax1.set_title("Equity Curves"); ax1.grid(True,alpha=0.3)
ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8)
ax2.axhline(0,color="black",lw=0.8); ax2.grid(True,alpha=0.3)
ax2.set_title("Drawdown")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"phase19_mr_layer.png"),dpi=150,bbox_inches="tight")
plt.close()
print(f"\nSaved. Phase 19 complete.")
