"""
Phase 15: Capacity Model

For each AUM level, compute net Sharpe after market impact.
Market impact model: impact_bps = IMPACT_COEFF * sqrt(order_size_usd / daily_volume_usd)

At each rebal:
  - Gross P&L = signal-weighted return (same as baseline)
  - Market impact = function of AUM, position size, and coin daily volume
  - Net P&L = Gross - Fees - Impact

Outputs:
  results/phase15_capacity.png
  results/phase15_capacity.csv
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
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
IMPACT_COEFF     = 10   # bps per sqrt(order_frac_of_daily_vol)

# AUM range to test (in USD)
AUM_LEVELS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
              10_000_000, 25_000_000, 50_000_000, 100_000_000]

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}


# ── loaders ────────────────────────────────────────────────────────────────

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


# ── main ───────────────────────────────────────────────────────────────────

print("Loading panels...")
panel_1h = load_panels(["funding", "mom_24h", "fwd_8h", "turnover"])
panel    = resample_panel(panel_1h, REBAL_FREQ)

all_syms  = list(panel["funding"].columns)
univ_syms = [s for s in all_syms if s not in MAJORS]
for col in panel:
    panel[col] = panel[col][univ_syms]

funding  = panel["funding"].ffill()
mom_24h  = panel["mom_24h"]
fwd_8h   = panel["fwd_8h"]
turnover = panel["turnover"]   # 8h turnover in quote currency (USDT)

# Estimate daily volume = sum of last 3 bars turnover (3 × 8h = 24h)
daily_vol = turnover.rolling(3, min_periods=1).sum()

z_funding = cs_zscore(funding)
z_mom24h  = cs_zscore(mom_24h)
composite = (z_funding + z_mom24h) / 2

print(f"Universe: {len(univ_syms)} coins | Bars: {len(composite)}")


# ── precompute gross returns + per-bar impact metadata ─────────────────────

print("Precomputing bar returns and impact metadata...")

dates    = composite.index
gross_bar    = []    # gross return (no fees, no impact)
fee_bar      = []    # fee cost (independent of AUM)
impact_meta  = []    # (turnover_fraction, median_daily_vol) per bar

prev_pos = None

for i, t in enumerate(dates):
    sig = composite.loc[t].dropna()
    fwd = fwd_8h.loc[t].reindex(sig.index).fillna(0) if t in fwd_8h.index else pd.Series(0.0, index=sig.index)
    dvol = daily_vol.loc[t].reindex(sig.index).fillna(1e6) if t in daily_vol.index else pd.Series(1e6, index=sig.index)

    if len(sig) < (N_LONG + N_SHORT) * 2:
        gross_bar.append(np.nan)
        fee_bar.append(np.nan)
        impact_meta.append((np.nan, np.nan))
        continue

    rank   = sig.rank(ascending=False)
    longs  = rank[rank <= N_LONG].index
    shorts = rank[rank > len(sig) - N_SHORT].index

    pos = pd.Series(0.0, index=sig.index)
    pos[longs]  = +1.0 / N_LONG
    pos[shorts] = -1.0 / N_SHORT

    gross = (pos * fwd).sum()

    if prev_pos is not None:
        all_idx  = pos.index.union(prev_pos.index)
        delta    = pos.reindex(all_idx, fill_value=0) - prev_pos.reindex(all_idx, fill_value=0)
        turnover_frac = delta.abs().sum() / 2
    else:
        turnover_frac = 1.0

    fee = turnover_frac * FEE_MAKER_BPS / 10000

    # For impact: median daily vol of traded coins (long + short positions)
    traded = longs.union(shorts)
    med_dvol = dvol[traded].median() if len(traded) > 0 else 1e6

    gross_bar.append(gross)
    fee_bar.append(fee)
    impact_meta.append((turnover_frac, med_dvol))

    prev_pos = pos.copy()

gross_bar   = pd.Series(gross_bar,   index=dates)
fee_bar     = pd.Series(fee_bar,     index=dates)
meta_df     = pd.DataFrame(impact_meta, index=dates, columns=["turnover_frac","med_dvol"])


# ── compute net returns for each AUM level ─────────────────────────────────

def compute_metrics(rets):
    r = rets.dropna()
    if len(r) < 10:
        return {"Sharpe": 0, "Ann_Ret": 0, "MaxDD": 0}
    sharpe  = r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else 0
    ann_ret = (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    return {"Sharpe": round(sharpe, 3), "Ann_Ret": round(ann_ret * 100, 1), "MaxDD": round(maxdd * 100, 1)}


print("\nRunning AUM sweep...")
results = {}

for aum in AUM_LEVELS:
    # Position size per coin (1/20 of AUM, equal-weight, 20 positions)
    pos_size_usd = aum / (N_LONG + N_SHORT)

    # Per-bar impact cost
    # turnover_frac * pos_size_usd = notional traded per position
    # impact_bps = IMPACT_COEFF * sqrt(order_usd / daily_vol_usd)
    impact_bar = pd.Series(np.nan, index=dates)
    for i, t in enumerate(dates):
        tf    = meta_df.loc[t, "turnover_frac"]
        mdvol = meta_df.loc[t, "med_dvol"]
        if np.isnan(tf) or np.isnan(mdvol) or mdvol <= 0:
            continue
        order_usd    = tf * pos_size_usd
        impact_bps_i = IMPACT_COEFF * np.sqrt(order_usd / mdvol)
        # total impact = per-position impact * turnover_frac (positions traded fraction)
        impact_bar.iloc[i] = tf * impact_bps_i / 10000

    net_rets = gross_bar - fee_bar - impact_bar
    m = compute_metrics(net_rets)
    results[aum] = m
    label = f"${aum:>12,.0f}"
    print(f"  AUM {label}  Sharpe={m['Sharpe']:+.3f}  AnnRet={m['Ann_Ret']:+.1f}%  MaxDD={m['MaxDD']:+.1f}%")


# ── also compute baseline (zero impact) ────────────────────────────────────
net_base = gross_bar - fee_bar
m_base   = compute_metrics(net_base)
results["baseline (0 impact)"] = m_base
print(f"\n  Baseline (no impact):  Sharpe={m_base['Sharpe']:.3f}  AnnRet={m_base['Ann_Ret']:.1f}%  MaxDD={m_base['MaxDD']:.1f}%")


# ── find AUM where Sharpe degrades by 25% ─────────────────────────────────
target_sharpe = m_base["Sharpe"] * 0.75
aum_vals   = AUM_LEVELS
sharpes    = [results[a]["Sharpe"] for a in aum_vals]
cap_aum    = None
for i, (a, s) in enumerate(zip(aum_vals, sharpes)):
    if s < target_sharpe:
        cap_aum = a
        break

if cap_aum:
    print(f"\nCapacity estimate (Sharpe < {target_sharpe:.2f}, -25% from baseline): ${cap_aum:,.0f}")
else:
    print(f"\nSharpe stays above {target_sharpe:.2f} even at ${max(AUM_LEVELS):,.0f}")


# ── save ───────────────────────────────────────────────────────────────────

df_res = pd.DataFrame(results).T
df_res.to_csv(os.path.join(RESULTS_DIR, "phase15_capacity.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase15_capacity.csv")


# ── plot ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase 15: Capacity Model (Market Impact)", fontsize=14)

aum_m = [a / 1e6 for a in AUM_LEVELS]

ax1.plot(aum_m, [results[a]["Sharpe"] for a in AUM_LEVELS], "o-", color="steelblue", lw=2)
ax1.axhline(m_base["Sharpe"], color="gray", ls="--", lw=1.5, label=f"No-impact baseline: {m_base['Sharpe']:.2f}")
ax1.axhline(target_sharpe, color="orange", ls="--", lw=1.5, label=f"-25% threshold: {target_sharpe:.2f}")
if cap_aum:
    ax1.axvline(cap_aum / 1e6, color="red", ls="--", lw=1.5,
                label=f"Capacity: ${cap_aum/1e6:.0f}M")
ax1.set_xlabel("AUM ($M)")
ax1.set_ylabel("Net Sharpe")
ax1.set_title("Net Sharpe vs AUM")
ax1.set_xscale("log")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(aum_m, [results[a]["Ann_Ret"] for a in AUM_LEVELS], "o-", color="darkorange", lw=2)
ax2.axhline(m_base["Ann_Ret"], color="gray", ls="--", lw=1.5, label=f"Baseline: {m_base['Ann_Ret']:.0f}%")
ax2.set_xlabel("AUM ($M)")
ax2.set_ylabel("Ann. Return (%)")
ax2.set_title("Net Ann. Return vs AUM")
ax2.set_xscale("log")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase15_capacity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase15_capacity.png")

print("\nPhase 15 complete.")
