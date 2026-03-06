"""
Phase 12: Dynamic Leverage

Applies two leverage-scaling mechanisms on top of the No-Majors baseline:

  1. Vol-targeting: scale gross exposure so annualized portfolio vol = TARGET_VOL
     Uses 30-bar (10-day) realized vol of 8h portfolio returns.

  2. Drawdown scaling: reduce leverage when in drawdown from equity peak
       0% – 10% DD:  full leverage (1.0x)
      10% – 20% DD:  0.5x
      20%+   DD:     0.25x

  3. Combined: vol-targeting THEN drawdown scaling (capped at 1.0x max)

Outputs:
  results/phase12_leverage.png
  results/phase12_leverage.csv
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
TARGET_VOL       = 0.40   # annualized target portfolio vol (40%)
VOL_WINDOW       = 30     # bars for realized vol (10 days)
MAX_LEVERAGE     = 2.0    # cap leverage at 2x

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


# ── portfolio backtest ─────────────────────────────────────────────────────

def backtest(composite, fwd_8h, fee_bps=FEE_MAKER_BPS, n_long=N_LONG, n_short=N_SHORT):
    """Run equal-weight long/short backtest, return series of 8h net returns."""
    dates    = composite.index
    bar_rets = []
    prev_pos = pd.Series(dtype=float)

    for i, t in enumerate(dates):
        sig = composite.loc[t].dropna()
        fwd = fwd_8h.loc[t] if t in fwd_8h.index else pd.Series(dtype=float)
        if len(sig) < (n_long + n_short) * 2:
            bar_rets.append(np.nan)
            continue

        ranked = sig.rank(ascending=False)
        longs  = ranked[ranked <= n_long].index
        shorts = ranked[ranked > len(sig) - n_short].index

        pos = pd.Series(0.0, index=sig.index)
        pos[longs]  = +1.0 / n_long
        pos[shorts] = -1.0 / n_short

        # gross return
        gross = (pos * fwd.reindex(pos.index).fillna(0)).sum()

        # turnover cost
        if len(prev_pos) > 0:
            all_idx = pos.index.union(prev_pos.index)
            delta   = pos.reindex(all_idx, fill_value=0) - prev_pos.reindex(all_idx, fill_value=0)
            turnover = delta.abs().sum() / 2
        else:
            turnover = 1.0
        fee_cost = turnover * fee_bps / 10000

        bar_rets.append(gross - fee_cost)
        prev_pos = pos.copy()

    return pd.Series(bar_rets, index=dates)


def metrics(rets, label=""):
    rets = rets.dropna()
    sharpe   = rets.mean() / rets.std() * np.sqrt(PERIODS_PER_YEAR) if rets.std() > 0 else 0
    ann_ret  = (1 + rets).prod() ** (PERIODS_PER_YEAR / len(rets)) - 1
    eq       = (1 + rets).cumprod()
    rolling_max = eq.cummax()
    dd       = (eq / rolling_max - 1)
    max_dd   = dd.min()
    total    = eq.iloc[-1] - 1
    return {
        "Sharpe"  : round(sharpe, 3),
        "Ann_Ret" : round(ann_ret * 100, 1),
        "MaxDD"   : round(max_dd * 100, 1),
        "Total"   : round(total * 100, 1),
        "n_bars"  : len(rets),
    }


# ── main ───────────────────────────────────────────────────────────────────

print("Loading panels...")
panel_1h = load_panels(["close", "funding", "mom_24h", "fwd_8h"])
panel    = resample_panel(panel_1h, REBAL_FREQ)

all_syms  = list(panel["close"].columns)
univ_syms = [s for s in all_syms if s not in MAJORS]
print(f"Universe: {len(univ_syms)} coins (excl. {len(MAJORS)} Majors)")

for col in panel:
    panel[col] = panel[col][univ_syms]

funding = panel["funding"].ffill()
mom_24h = panel["mom_24h"]
fwd_8h  = panel["fwd_8h"]

z_funding = cs_zscore(funding)
z_mom24h  = cs_zscore(mom_24h)
composite = (z_funding + z_mom24h) / 2

print("Running baseline backtest...")
raw_rets = backtest(composite, fwd_8h)

# ── apply leverage multipliers ─────────────────────────────────────────────

def apply_vol_target(rets, target_vol=TARGET_VOL, window=VOL_WINDOW, max_lev=MAX_LEVERAGE):
    """Scale returns by vol-targeting multiplier (computed from past window)."""
    bar_vol_target = target_vol / np.sqrt(PERIODS_PER_YEAR)
    scaled = []
    for i in range(len(rets)):
        if i < window:
            scaled.append(rets.iloc[i])
            continue
        past_vol = rets.iloc[i-window:i].std()
        if past_vol <= 0 or np.isnan(past_vol):
            lev = 1.0
        else:
            lev = min(bar_vol_target / past_vol, max_lev)
        scaled.append(rets.iloc[i] * lev)
    return pd.Series(scaled, index=rets.index)


def apply_dd_scale(rets):
    """Scale returns by drawdown-based multiplier."""
    scaled = []
    eq = 1.0
    peak = 1.0
    for r in rets:
        if np.isnan(r):
            scaled.append(np.nan)
            continue
        dd = eq / peak - 1
        if dd > -0.10:
            lev = 1.0
        elif dd > -0.20:
            lev = 0.5
        else:
            lev = 0.25
        scaled_r = r * lev
        scaled.append(scaled_r)
        eq = eq * (1 + scaled_r)
        peak = max(peak, eq)
    return pd.Series(scaled, index=rets.index)


def apply_combined(rets, target_vol=TARGET_VOL, window=VOL_WINDOW, max_lev=MAX_LEVERAGE):
    """Vol-target first, then drawdown scaling, then combine."""
    bar_vol_target = target_vol / np.sqrt(PERIODS_PER_YEAR)
    scaled = []
    eq = 1.0
    peak = 1.0
    for i in range(len(rets)):
        r = rets.iloc[i]
        if np.isnan(r):
            scaled.append(np.nan)
            eq = eq  # no change
            continue

        # vol-target leverage
        if i < window:
            vol_lev = 1.0
        else:
            past_vol = rets.iloc[i-window:i].std()
            vol_lev = min(bar_vol_target / past_vol, max_lev) if past_vol > 0 else 1.0

        # drawdown leverage
        dd = eq / peak - 1
        if dd > -0.10:
            dd_lev = 1.0
        elif dd > -0.20:
            dd_lev = 0.5
        else:
            dd_lev = 0.25

        lev      = vol_lev * dd_lev
        scaled_r = r * lev
        scaled.append(scaled_r)
        eq   = eq * (1 + scaled_r)
        peak = max(peak, eq)

    return pd.Series(scaled, index=rets.index)


print("Applying leverage variants...")

rets_vt   = apply_vol_target(raw_rets)
rets_dd   = apply_dd_scale(raw_rets)
rets_comb = apply_combined(raw_rets)

variants = {
    "Baseline (1x)"          : raw_rets,
    "Vol-target (40% ann)"   : rets_vt,
    "DD-scaling"             : rets_dd,
    "Combined (VT + DD)"     : rets_comb,
}

print("\nResults:")
rows = {}
for name, rets in variants.items():
    m = metrics(rets, name)
    rows[name] = m
    print(f"  {name:30s}  Sharpe={m['Sharpe']:+.3f}  AnnRet={m['Ann_Ret']:+.1f}%  MaxDD={m['MaxDD']:+.1f}%  Total={m['Total']:+.1f}%")

df_res = pd.DataFrame(rows).T
df_res.to_csv(os.path.join(RESULTS_DIR, "phase12_leverage.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase12_leverage.csv")


# ── plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Phase 12: Dynamic Leverage — Equity Curves", fontsize=14)

colors = {"Baseline (1x)": "steelblue", "Vol-target (40% ann)": "darkorange",
          "DD-scaling": "green", "Combined (VT + DD)": "crimson"}

ax_eq, ax_dd = axes
for name, rets in variants.items():
    r = rets.dropna()
    eq = (1 + r).cumprod() * 1000
    m  = rows[name]
    ax_eq.plot(eq.values, label=f"{name}  Sharpe={m['Sharpe']:.2f}, MaxDD={m['MaxDD']:.1f}%",
               color=colors.get(name, "gray"), lw=1.5)

    roll_max = eq.cummax()
    dd       = (eq / roll_max - 1) * 100
    ax_dd.plot(dd.values, color=colors.get(name, "gray"), lw=1.2, label=name)

ax_eq.set_ylabel("Portfolio Value ($)")
ax_eq.set_title("Equity Curves ($1000 start)")
ax_eq.legend(fontsize=9)
ax_eq.grid(True, alpha=0.3)
ax_eq.set_yscale("log")

ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=9)
ax_dd.grid(True, alpha=0.3)
ax_dd.axhline(0, color="black", lw=0.8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase12_leverage.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase12_leverage.png")

# monthly table for best variant
print("\nMonthly breakdown — Combined (VT + DD):")
rets = rets_comb.dropna()
eq   = (1 + rets).cumprod() * 1000
monthly_rets = rets.resample("ME").apply(lambda x: (1+x).prod()-1)
monthly_eq   = eq.resample("ME").last()

print(f"  {'Month':10s}  {'Ret%':>8s}  {'Equity':>10s}")
for mo in monthly_rets.index:
    r  = monthly_rets[mo]
    eq_val = monthly_eq[mo]
    flag = " ◄ GREAT" if r > 0.5 else (" ◄ BAD" if r < -0.10 else "")
    print(f"  {str(mo)[:7]:10s}  {r*100:+8.1f}%  ${eq_val:>9,.0f}{flag}")

final_eq = (1 + rets_comb.dropna()).prod() * 1000
print(f"\n  Final equity: ${final_eq:,.0f} (from $1,000)")
print(f"  Baseline:     ${(1+raw_rets.dropna()).prod()*1000:,.0f}")

print("\nPhase 12 complete.")
