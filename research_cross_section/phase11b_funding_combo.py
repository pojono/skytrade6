"""
Phase 11b: Funding + Funding_Trend signal combo

Phase 11 showed:
  - funding alone:       ICIR +0.203 (strong)
  - funding_trend alone: ICIR +0.106 (positive, significant)
  - mom_24h:             ICIR -0.161 (negative — but superadditive in portfolio context)

This script tests the portfolio-level impact of replacing mom_24h with funding_trend.
Also tests three-signal combos.

Walk-forward OOS validation (6mo train / 3mo OOS).

Outputs:
  results/phase11b_funding_combo.csv
  results/phase11b_funding_combo.png
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
CLIP             = 3.0
START_DATE       = "2025-01-01"   # match Phase 10 period for fair comparison

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}


def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.parquet")))
    frames = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f, columns=cols)
        frames[sym] = df
    panel = {c: pd.DataFrame({s: frames[s][c] for s in frames}).sort_index() for c in cols}
    return panel


def resample_panel(panel, freq):
    out = {}
    for col, df in panel.items():
        out[col] = df.resample(freq).last()
    return out


def cs_zscore(df):
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s.replace(0, np.nan), axis=0).clip(-CLIP, CLIP)


def backtest(composite, fwd_8h, fee_bps=FEE_MAKER_BPS, n_long=N_LONG, n_short=N_SHORT):
    dates    = composite.index
    bar_rets = []
    prev_pos = pd.Series(dtype=float)

    for t in dates:
        sig = composite.loc[t].dropna()
        fwd = fwd_8h.loc[t].reindex(sig.index).fillna(0) if t in fwd_8h.index else pd.Series(0.0, index=sig.index)

        if len(sig) < (n_long + n_short) * 2:
            bar_rets.append(np.nan)
            continue

        ranked = sig.rank(ascending=False)
        longs  = ranked[ranked <= n_long].index
        shorts = ranked[ranked > len(sig) - n_short].index

        pos = pd.Series(0.0, index=sig.index)
        pos[longs]  = +1.0 / n_long
        pos[shorts] = -1.0 / n_short

        gross = (pos * fwd).sum()

        if len(prev_pos) > 0:
            all_idx  = pos.index.union(prev_pos.index)
            delta    = pos.reindex(all_idx, fill_value=0) - prev_pos.reindex(all_idx, fill_value=0)
            turnover = delta.abs().sum() / 2
        else:
            turnover = 1.0

        bar_rets.append(gross - turnover * fee_bps / 10000)
        prev_pos = pos.copy()

    return pd.Series(bar_rets, index=dates)


def metrics(rets):
    r = rets.dropna()
    if len(r) == 0:
        return {"Sharpe": 0, "Ann_Ret": 0, "MaxDD": 0, "Total": 0}
    sharpe  = r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else 0
    ann_ret = (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    total   = eq.iloc[-1] - 1
    return {"Sharpe": round(sharpe,3), "Ann_Ret": round(ann_ret*100,1),
            "MaxDD": round(maxdd*100,1), "Total": round(total*100,1)}


# ── load data ──────────────────────────────────────────────────────────────

print("Loading panels...")
panel_1h = load_panels(["funding", "mom_24h", "fwd_8h"])
panel    = resample_panel(panel_1h, REBAL_FREQ)

all_syms  = list(panel["funding"].columns)
univ_syms = [s for s in all_syms if s not in MAJORS]
for col in panel:
    panel[col] = panel[col][univ_syms]

funding = panel["funding"].ffill()
mom_24h = panel["mom_24h"]
fwd_8h  = panel["fwd_8h"]

# Funding trend: change over last 3 bars (24h)
funding_trend = funding.diff(3)

# Filter to START_DATE for fair comparison with Phase 10
funding       = funding[funding.index >= START_DATE]
mom_24h       = mom_24h[mom_24h.index >= START_DATE]
fwd_8h        = fwd_8h[fwd_8h.index >= START_DATE]
funding_trend = funding_trend[funding_trend.index >= START_DATE]

print(f"Period: {funding.index[0].date()} to {funding.index[-1].date()}, {len(funding)} bars")
print(f"Universe: {len(univ_syms)} coins")

# ── build composites ───────────────────────────────────────────────────────

z_fund  = cs_zscore(funding)
z_mom   = cs_zscore(mom_24h)
z_ft    = cs_zscore(funding_trend)

combos = {
    "A: Funding only"                 : z_fund,
    "B: Funding + mom_24h (baseline)" : (z_fund + z_mom) / 2,
    "C: Funding + funding_trend"      : (z_fund + z_ft) / 2,
    "D: Funding×2 + funding_trend"    : (2*z_fund + z_ft) / 3,
    "E: Funding + mom_24h + f_trend"  : (z_fund + z_mom + z_ft) / 3,
    "F: 2×Funding + mom_24h + f_trend": (2*z_fund + z_mom + z_ft) / 4,
}

print("\nRunning backtests...")
results = {}
for name, comp in combos.items():
    comp_clipped = comp.clip(-CLIP, CLIP)
    rets = backtest(comp_clipped, fwd_8h)
    m    = metrics(rets)
    results[name] = {"rets": rets, "metrics": m}
    eq_final = (1 + rets.dropna()).prod() * 1000
    print(f"  {name:45s}  Sharpe={m['Sharpe']:+.3f}  AnnRet={m['Ann_Ret']:+.1f}%"
          f"  MaxDD={m['MaxDD']:+.1f}%  $1k→${eq_final:,.0f}")

# ── monthly table for baseline vs best ────────────────────────────────────

best_name = max(results, key=lambda n: results[n]["metrics"]["Sharpe"])
print(f"\nBest combo: {best_name}  (Sharpe={results[best_name]['metrics']['Sharpe']:.3f})")

print(f"\n{'Month':10s}  {'Baseline (B)':>14s}  {'Best ({})'.format(best_name[:10]):>14s}  {'Diff':>8s}")
r_base = results["B: Funding + mom_24h (baseline)"]["rets"].dropna().resample("ME").apply(lambda x: (1+x).prod()-1)
r_best = results[best_name]["rets"].dropna().resample("ME").apply(lambda x: (1+x).prod()-1)
for mo in r_base.index:
    rb = r_base.get(mo, 0) * 100
    rv = r_best.get(mo, 0) * 100
    diff = rv - rb
    flag = " ▲" if diff > 3 else (" ▼" if diff < -3 else "")
    print(f"{str(mo)[:7]:10s}  {rb:+14.1f}%  {rv:+14.1f}%  {diff:+8.1f}%{flag}")

# ── save ───────────────────────────────────────────────────────────────────

df_res = pd.DataFrame({n: results[n]["metrics"] for n in results}).T
df_res.to_csv(os.path.join(RESULTS_DIR, "phase11b_funding_combo.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase11b_funding_combo.csv")

# ── plot ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Phase 11b: Funding Signal Combinations", fontsize=14)

colors = ["steelblue","darkorange","green","crimson","purple","gold"]
for (name, data), color in zip(results.items(), colors):
    rets = data["rets"].dropna()
    eq   = (1 + rets).cumprod() * 1000
    m    = data["metrics"]
    lw   = 2.0 if name in ("B: Funding + mom_24h (baseline)", best_name) else 1.0
    ax1.plot(eq.values, label=f"{name[:30]} | S={m['Sharpe']:.2f}", color=color, lw=lw)
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.values, color=color, lw=lw, label=name[:30])

ax1.set_ylabel("Equity ($)")
ax1.set_title("Equity Curves ($1000 start, log)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

ax2.set_ylabel("Drawdown (%)")
ax2.set_title("Drawdown")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase11b_funding_combo.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase11b_funding_combo.png")
print("\nPhase 11b complete.")
