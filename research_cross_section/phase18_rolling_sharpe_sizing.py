"""
Phase 18: Rolling Sharpe Sizing

Scale position size based on recent realized Sharpe of the portfolio.
When recent Sharpe is high (hot streak) → reduce to harvest profits.
When recent Sharpe is moderate → full size.
When recent Sharpe is negative → reduce to protect capital.

This is different from vol-targeting (rejected in Phase 12):
  - Vol-targeting responds to volatility (often high in GOOD months) → cuts alpha
  - Sharpe-sizing responds to edge quality (risk-adjusted recent performance)
  - High recent Sharpe = overbought regime or genuine alpha → optional harvest
  - Negative recent Sharpe = bad regime → cut risk

Three variants:
  A. Edge-scaling: high-Sharpe bars get MORE (up to 1.5×), low-Sharpe get LESS (0.5×)
  B. Harvest-mode: cap monthly gains at 30% by scaling down mid-month
  C. Drawdown-Sharpe combo: scale by Sharpe AND cut at drawdowns
  D. Inverse scaling (experimental): REDUCE when Sharpe is very high (pure mean-reversion)

Uses Phase 16 signal: 2×funding + mom_24h + funding_trend.
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
SHARPE_WINDOW    = 30    # 10-day rolling Sharpe (30 bars)
MONTHLY_CAP      = 0.30  # 30% monthly gain cap for harvest mode

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

def base_sim(composite, fwd_8h, n=N_LONG):
    """Returns raw (unscaled) bar P&L."""
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

def metrics(rets):
    r = rets.dropna()
    if len(r)==0: return dict(Sharpe=0,Sortino=0,Ann_Ret=0,MaxDD=0)
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

print(f"Universe: {len(univ)} coins | Bars: {len(composite)}")

print("Computing baseline returns...")
raw_rets = base_sim(composite, fwd_8h_raw)

# ── A. Edge-scaling: scale by rolling Sharpe ──────────────────────────────
def edge_scale(rets, window=SHARPE_WINDOW):
    """Per-bar scale: high-Sharpe → up to 1.5×, negative → 0.5×"""
    scaled = []
    for i, r in enumerate(rets):
        if np.isnan(r) or i < window:
            scaled.append(r)
            continue
        past = rets.iloc[i-window:i].dropna()
        if len(past) < 10 or past.std()==0:
            scaled.append(r)
            continue
        roll_sharpe = past.mean()/past.std()*np.sqrt(PERIODS_PER_YEAR)
        # Map Sharpe to scale: -2→0.5x, 0→0.75x, 2→1.0x, 4→1.25x, 6+→1.5x
        scale = np.clip(0.75 + roll_sharpe*0.0625, 0.5, 1.5)
        scaled.append(r * scale)
    return pd.Series(scaled, index=rets.index)

rets_edge = edge_scale(raw_rets)

# ── B. Harvest mode: cap monthly gains ────────────────────────────────────
def harvest_mode(rets, monthly_cap=MONTHLY_CAP):
    """When MTD return exceeds cap, scale down to 25% for rest of month."""
    scaled = []
    eq_mtd   = 1.0
    cur_month = None
    for ts, r in rets.items():
        month = (ts.year, ts.month)
        if month != cur_month:
            cur_month = month
            eq_mtd    = 1.0
        if np.isnan(r):
            scaled.append(np.nan)
            continue
        mtd_ret = eq_mtd - 1
        if mtd_ret >= monthly_cap:
            scale = 0.25  # harvested — sit mostly flat
        elif mtd_ret >= monthly_cap * 0.7:
            scale = 0.5   # approaching cap — reduce
        else:
            scale = 1.0
        scaled_r = r * scale
        eq_mtd  *= (1 + scaled_r)
        scaled.append(scaled_r)
    return pd.Series(scaled, index=rets.index)

rets_harvest = harvest_mode(raw_rets)

# ── C. Inverse scaling: reduce in hot months (pure mean-reversion bet) ────
def inverse_scale(rets, window=SHARPE_WINDOW):
    """When rolling Sharpe is very high, REDUCE (bet on reversal)."""
    scaled = []
    for i, r in enumerate(rets):
        if np.isnan(r) or i < window:
            scaled.append(r)
            continue
        past = rets.iloc[i-window:i].dropna()
        if len(past) < 10 or past.std()==0:
            scaled.append(r)
            continue
        roll_sharpe = past.mean()/past.std()*np.sqrt(PERIODS_PER_YEAR)
        # High Sharpe → reduce (reversal fear), low/neg Sharpe → reduce (bad regime)
        # Optimal zone: Sharpe 2-4 → full size
        if roll_sharpe > 5:
            scale = 0.5   # overbought — reduce
        elif roll_sharpe > 2:
            scale = 1.0   # sweet spot — full
        elif roll_sharpe > 0:
            scale = 0.75  # modest — slight reduce
        else:
            scale = 0.5   # bad regime — protect
        scaled.append(r * scale)
    return pd.Series(scaled, index=rets.index)

rets_inv = inverse_scale(raw_rets)

# ── D. Combo: harvest + inverse ──────────────────────────────────────────
def combo_scale(rets, window=SHARPE_WINDOW, monthly_cap=MONTHLY_CAP):
    scaled = []
    eq_mtd   = 1.0
    cur_month = None
    for i, (ts, r) in enumerate(rets.items()):
        month = (ts.year, ts.month)
        if month != cur_month:
            cur_month = month
            eq_mtd    = 1.0
        if np.isnan(r):
            scaled.append(np.nan)
            eq_mtd *= 1.0
            continue
        # Sharpe scale
        if i >= window:
            past = rets.iloc[i-window:i].dropna()
            if len(past) >= 10 and past.std() > 0:
                rs = past.mean()/past.std()*np.sqrt(PERIODS_PER_YEAR)
                sharpe_scale = 0.5 if rs > 5 else (1.0 if rs > 1 else 0.6)
            else:
                sharpe_scale = 1.0
        else:
            sharpe_scale = 1.0
        # Harvest scale
        mtd = eq_mtd - 1
        harvest_sc = 0.25 if mtd >= monthly_cap else (0.5 if mtd >= monthly_cap*0.7 else 1.0)
        scale    = sharpe_scale * harvest_sc
        scaled_r = r * scale
        eq_mtd  *= (1 + scaled_r)
        scaled.append(scaled_r)
    return pd.Series(scaled, index=rets.index)

rets_combo = combo_scale(raw_rets)

# ── compare ───────────────────────────────────────────────────────────────
variants = {
    "Baseline (1×)"         : raw_rets,
    "A: Edge-scaling"       : rets_edge,
    "B: Harvest mode (30%)" : rets_harvest,
    "C: Inverse scaling"    : rets_inv,
    "D: Harvest + Inverse"  : rets_combo,
}

print(f"\n{'Strategy':30s}  {'Sharpe':>7s}  {'Sortino':>7s}  {'MaxDD':>7s}  {'$1k→':>10s}")
rows = {}
for name, rets in variants.items():
    m  = metrics(rets)
    rows[name] = m
    eq = (1+rets.dropna()).prod()*1000
    print(f"  {name:28s}  {m['Sharpe']:+7.3f}  {m['Sortino']:+7.3f}  {m['MaxDD']:+7.1f}%  ${eq:>9,.0f}")

# monthly table
print(f"\n{'Month':10s}  {'Base':>10s}  {'B:Harvest':>10s}  {'C:Inverse':>10s}  {'D:Combo':>10s}")
for name in ["Baseline (1×)","B: Harvest mode (30%)","C: Inverse scaling","D: Harvest + Inverse"]:
    mo = variants[name].dropna().resample("ME").apply(lambda x:(1+x).prod()-1)
    variants[name+"_mo"] = mo
base_mo    = variants["Baseline (1×)_mo"]
harvest_mo = variants["B: Harvest mode (30%)_mo"]
inv_mo     = variants["C: Inverse scaling_mo"]
combo_mo   = variants["D: Harvest + Inverse_mo"]
for mo in base_mo.index:
    rb = base_mo.get(mo,0)*100
    rh = harvest_mo.get(mo,0)*100
    ri = inv_mo.get(mo,0)*100
    rc = combo_mo.get(mo,0)*100
    flag = " ◄" if abs(rb) > 20 else ""
    print(f"{str(mo)[:7]:10s}  {rb:+10.1f}%  {rh:+10.1f}%  {ri:+10.1f}%  {rc:+10.1f}%{flag}")

pd.DataFrame(rows).T.to_csv(os.path.join(RESULTS_DIR,"phase18_sharpe_sizing.csv"))

# plot
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
fig.suptitle("Phase 18: Rolling Sharpe Sizing", fontsize=14)
colors = {"Baseline (1×)":"steelblue","A: Edge-scaling":"darkorange",
          "B: Harvest mode (30%)":"green","C: Inverse scaling":"purple",
          "D: Harvest + Inverse":"crimson"}
for name, rets in [(n,v) for n,v in variants.items() if not n.endswith("_mo")]:
    r  = rets.dropna()
    eq = (1+r).cumprod()*1000
    m  = rows[name]
    lw = 2.0 if name=="Baseline (1×)" else 1.5
    ax1.plot(eq.values,label=f"{name} | S={m['Sharpe']:.2f} DD={m['MaxDD']:.0f}%",
             color=colors.get(name,"gray"),lw=lw)
    dd = (eq/eq.cummax()-1)*100
    ax2.plot(dd.values,color=colors.get(name,"gray"),lw=lw,label=name)

ax1.set_yscale("log"); ax1.set_ylabel("Equity ($)"); ax1.legend(fontsize=8)
ax1.set_title("Equity Curves (log)"); ax1.grid(True,alpha=0.3)
ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8)
ax2.axhline(0,color="black",lw=0.8); ax2.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"phase18_sharpe_sizing.png"),dpi=150,bbox_inches="tight")
plt.close()
print(f"\nSaved. Phase 18 complete.")
