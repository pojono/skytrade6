"""
Phase 20: Funding Saturation Gate + Regime-Adaptive Allocation

Idea: when universe average funding is at an extreme, risk changes:
  - VERY HIGH funding (top 5%): longs pay huge, but reversal risk is highest.
    Extreme funding often precedes funding compression crashes.
  - VERY LOW / NEGATIVE funding: shorts pay longs, carry flips sign.
    Strategy's long leg gets hurt, short leg gets paid.

Tests:
  A. High-funding gate: reduce to 0.5× when funding is top 5% (rolling 180-bar)
  B. Low-funding gate: reduce to 0.5× when funding is bottom 10% (near zero/negative)
  C. Both gates: reduce at extremes (only trade in the "sweet zone")
  D. Flip mode: when funding is very negative, REVERSE the long/short legs
     (long low-funding coins = collect from shorts, short high-funding = longs pay)
  E. Funding-scaled: continuously scale by funding_level_pctile (50% → 1×, 95% → 0.5×, 10% → 0.5×)

Uses Phase 16 signal as base.
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
ROLLING_WINDOW   = 180  # 60-day rolling for funding percentiles

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

def sim_with_scale(composite, fwd_8h, scale_series, n=N_LONG, flip_series=None):
    """
    scale_series: per-bar multiplier on gross return [0,1]
    flip_series:  per-bar boolean — if True, REVERSE L/S direction
    """
    common = composite.index.intersection(fwd_8h.index)
    records = []
    prev_l, prev_s = set(), set()
    for ts in common:
        sig   = composite.loc[ts].dropna()
        fwd   = fwd_8h.loc[ts, sig.index].dropna()
        sig   = sig.loc[fwd.index]
        scale = float(scale_series.get(ts, 1.0))
        flip  = bool(flip_series.get(ts, False)) if flip_series is not None else False

        if len(sig) < n*4:
            records.append((ts, np.nan))
            prev_l = prev_s = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n).index)
        shorts = set(ranked.nsmallest(n).index)
        if flip:
            longs, shorts = shorts, longs   # reverse legs

        lr, sr = fwd.loc[list(longs)].mean(), fwd.loc[list(shorts)].mean()
        if np.isnan(lr) or np.isnan(sr):
            records.append((ts, np.nan))
            continue

        gross    = lr - sr
        turnover = (len((longs-prev_l)|(shorts-prev_s))/(n+n)
                    if prev_l|prev_s else 1.0)
        net = (gross - turnover*FEE_RT) * scale
        records.append((ts, net))
        prev_l, prev_s = longs, shorts

    idx, vals = zip(*records)
    return pd.Series(vals, index=idx)

def metrics(rets):
    r = rets.dropna()
    if len(r)==0: return dict(Sharpe=0,Sortino=0,Ann_Ret=0,MaxDD=0,Active=0)
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
raw = load_panels(["funding","mom_24h","fwd_8h"])
funding_8h = to_8h(raw["funding"].ffill())
mom24h_8h  = to_8h(raw["mom_24h"])
fwd_8h_raw = to_8h(raw["fwd_8h"])
ft_8h      = to_8h(raw["funding"].ffill().diff(24))

univ = [s for s in funding_8h.columns if s not in MAJORS]
funding_8h = funding_8h[univ].loc[START:]
mom24h_8h  = mom24h_8h[univ].loc[START:]
fwd_8h_raw = fwd_8h_raw[univ].loc[START:]
ft_8h      = ft_8h[univ].loc[START:]

z_f  = cs_zscore(funding_8h)
z_m  = cs_zscore(mom24h_8h)
z_ft = cs_zscore(ft_8h)
composite = (2*z_f + z_m + z_ft)/4

# Universe average funding rate
avg_funding = funding_8h.mean(axis=1)

# Rolling percentile of avg funding (no look-ahead)
funding_pct = avg_funding.rolling(ROLLING_WINDOW, min_periods=30).rank(pct=True)

print(f"Universe: {len(univ)} coins | Bars: {len(composite)}")
print(f"\nFunding level stats:")
print(f"  Mean funding: {avg_funding.mean()*100:.4f}% per 8h")
print(f"  >95th pctile bars: {(funding_pct>0.95).mean()*100:.1f}%")
print(f"  <10th pctile bars: {(funding_pct<0.10).mean()*100:.1f}%")
print(f"  <5th pctile bars:  {(funding_pct<0.05).mean()*100:.1f}%")

# ── build scale/flip series ────────────────────────────────────────────────

# Baseline: no scaling
scale_base  = pd.Series(1.0, index=composite.index)
flip_base   = pd.Series(False, index=composite.index)

# A. High funding gate: reduce to 0.5× when top 5%
scale_A = funding_pct.map(lambda p: 0.5 if (not pd.isna(p) and p > 0.95) else 1.0)

# B. Low funding gate: reduce to 0.5× when bottom 10%
scale_B = funding_pct.map(lambda p: 0.5 if (not pd.isna(p) and p < 0.10) else 1.0)

# C. Both gates (sweet zone)
scale_C = funding_pct.map(
    lambda p: 0.5 if (not pd.isna(p) and (p > 0.95 or p < 0.10)) else 1.0
)

# D. Flip + gate: when very negative funding, flip L/S and reduce
#    (long low-funding shorts = longs pay them, short high-funding = those pay you)
scale_D = funding_pct.map(lambda p: 0.6 if (not pd.isna(p) and p < 0.05) else 1.0)
flip_D  = funding_pct.map(lambda p: True if (not pd.isna(p) and p < 0.05) else False)

# E. Continuous scaling: parabolic around median
#    At 50th pctile → 1.0×; at 5th/95th → 0.5×; linear in between
def funding_to_scale(p):
    if pd.isna(p): return 1.0
    dist = abs(p - 0.50) / 0.45   # distance from median, normalized to [0,1] at extremes
    return max(0.5, 1.0 - 0.5 * dist)

scale_E = funding_pct.map(funding_to_scale)

# ── run strategies ─────────────────────────────────────────────────────────
print("\nRunning backtests...")
variants = {
    "Baseline"                    : (scale_base, None),
    "A: High-funding gate (>95%)" : (scale_A,    None),
    "B: Low-funding gate (<10%)"  : (scale_B,    None),
    "C: Sweet-zone gate"          : (scale_C,    None),
    "D: Flip on neg funding"      : (scale_D,    flip_D),
    "E: Continuous funding scale" : (scale_E,    None),
}

print(f"\n{'Strategy':35s}  {'Sharpe':>7s}  {'Sortino':>7s}  {'MaxDD':>7s}  {'$1k→':>10s}")
rows    = {}
results = {}
for name, (scale, flip) in variants.items():
    rets = sim_with_scale(composite, fwd_8h_raw, scale, flip_series=flip)
    m    = metrics(rets)
    rows[name]    = m
    results[name] = rets
    eq = (1+rets.dropna()).prod()*1000
    active_pct = (scale > 0.6).mean()*100 if hasattr(scale,'mean') else 100
    print(f"  {name:33s}  {m['Sharpe']:+7.3f}  {m['Sortino']:+7.3f}  {m['MaxDD']:+7.1f}%  ${eq:>9,.0f}")

# Monthly table for key variants
print(f"\n{'Month':10s}  {'Base':>10s}  {'A:HighFG':>10s}  {'C:Sweet':>10s}  {'E:Contin':>10s}")
mo = {n: results[n].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)*100
      for n in ["Baseline","A: High-funding gate (>95%)","C: Sweet-zone gate","E: Continuous funding scale"]}
for t in mo["Baseline"].index:
    rb = mo["Baseline"].get(t,0)
    ra = mo["A: High-funding gate (>95%)"].get(t,0)
    rc = mo["C: Sweet-zone gate"].get(t,0)
    re = mo["E: Continuous funding scale"].get(t,0)
    flag = " ◄" if abs(rb)>20 else ""
    print(f"{str(t)[:7]:10s}  {rb:+10.1f}%  {ra:+10.1f}%  {rc:+10.1f}%  {re:+10.1f}%{flag}")

# Funding level plot
print(f"\nFunding level analysis:")
high_funding_bars = funding_pct > 0.95
low_funding_bars  = funding_pct < 0.10
base_rets = results["Baseline"]
print(f"  Mean bar return when HIGH funding (>95%): {base_rets[high_funding_bars].mean()*10000:.1f} bps")
print(f"  Mean bar return when LOW funding (<10%):  {base_rets[low_funding_bars].mean()*10000:.1f} bps")
print(f"  Mean bar return otherwise:                {base_rets[~(high_funding_bars|low_funding_bars)].mean()*10000:.1f} bps")

# save
pd.DataFrame(rows).T.to_csv(os.path.join(RESULTS_DIR,"phase20_funding_saturation.csv"))

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,14))
fig.suptitle("Phase 20: Funding Saturation Gate", fontsize=14)
colors = {"Baseline":"steelblue","A: High-funding gate (>95%)":"darkorange",
          "B: Low-funding gate (<10%)":"brown","C: Sweet-zone gate":"green",
          "D: Flip on neg funding":"purple","E: Continuous funding scale":"crimson"}

for name, rets in results.items():
    r  = rets.dropna()
    eq = (1+r).cumprod()*1000
    m  = rows[name]
    lw = 2.0 if name=="Baseline" else 1.5
    ax1.plot(eq.values,label=f"{name} | S={m['Sharpe']:.2f} DD={m['MaxDD']:.0f}%",
             color=colors.get(name,"gray"),lw=lw)
    dd = (eq/eq.cummax()-1)*100
    ax2.plot(dd.values,color=colors.get(name,"gray"),lw=lw,label=name)

# Funding level over time
ax3.plot(avg_funding.values*100, color="steelblue", lw=1.0, alpha=0.7, label="Universe avg funding (%)")
high_th = avg_funding.rolling(ROLLING_WINDOW,min_periods=30).quantile(0.95)
low_th  = avg_funding.rolling(ROLLING_WINDOW,min_periods=30).quantile(0.10)
ax3.plot(high_th.values*100, color="crimson", lw=1.5, ls="--", label="95th pctile (gate threshold)")
ax3.plot(low_th.values*100,  color="orange",  lw=1.5, ls="--", label="10th pctile (gate threshold)")
ax3.set_ylabel("Funding rate (%)"); ax3.set_title("Universe Average Funding Rate")
ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3)

ax1.set_yscale("log"); ax1.set_ylabel("Equity ($)"); ax1.legend(fontsize=8)
ax1.set_title("Equity Curves (log)"); ax1.grid(True,alpha=0.3)
ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8)
ax2.axhline(0,color="black",lw=0.8); ax2.grid(True,alpha=0.3)
ax2.set_title("Drawdown")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"phase20_funding_saturation.png"),dpi=150,bbox_inches="tight")
plt.close()
print(f"\nSaved. Phase 20 complete.")
