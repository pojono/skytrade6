"""
Phase 11: New Signal IC Analysis

Tests four new signals for cross-sectional predictive power:
  1. funding_trend  — change in funding over last 24h (3 x 8h bars)
  2. oi_growth      — OI growth over last 24h
  3. vol_anomaly    — volume vs 30-day rolling mean
  4. btc_relative   — coin 8h return minus BTC 8h return (cross-asset lead-lag)

For each signal: cross-sectional Spearman IC at each 8h bar, ICIR, t-stat.
Also tests combos with the existing funding + mom_24h baseline.

Outputs:
  results/phase11_signal_ic.png
  results/phase11_signal_ic.csv
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
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0

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


def compute_ic(signal_df, fwd_df, min_obs=20):
    """Cross-sectional Spearman IC at each bar."""
    ics = []
    for t in signal_df.index:
        if t not in fwd_df.index:
            continue
        sig = signal_df.loc[t].dropna()
        fwd = fwd_df.loc[t].dropna()
        common = sig.index.intersection(fwd.index)
        if len(common) < min_obs:
            continue
        ic, _ = stats.spearmanr(sig[common], fwd[common])
        ics.append(ic)
    ics = pd.Series(ics)
    return ics


def icir_stats(ics):
    n     = len(ics)
    mean  = ics.mean()
    std   = ics.std()
    icir  = mean / std if std > 0 else 0.0
    tstat = icir * np.sqrt(n)
    return {"n": n, "mean_IC": mean, "std_IC": std, "ICIR": icir, "t_stat": tstat}


# ── main ───────────────────────────────────────────────────────────────────

print("Loading panels...")
cols_needed = ["close", "volume", "funding", "oi", "fwd_8h", "mom_24h"]
panel_1h    = load_panels(cols_needed)
panel       = resample_panel(panel_1h, REBAL_FREQ)

# drop Majors from universe
all_syms   = list(panel["close"].columns)
univ_syms  = [s for s in all_syms if s not in MAJORS]
print(f"Universe: {len(univ_syms)} coins (excl. {len(MAJORS)} Majors)")

for col in panel:
    panel[col] = panel[col][univ_syms]

close   = panel["close"]
volume  = panel["volume"]
funding = panel["funding"].ffill()
oi      = panel["oi"].ffill()
fwd_8h  = panel["fwd_8h"]
mom_24h = panel["mom_24h"]


# ── build new signals ──────────────────────────────────────────────────────

print("Building new signals...")

# 1. Funding trend: change in funding over last 3 bars (24h)
funding_trend = funding.diff(3)

# 2. OI growth: percentage change over last 3 bars
oi_growth = oi.pct_change(3)
# clip extreme values (data outliers)
oi_growth = oi_growth.clip(-2, 2)

# 3. Volume anomaly: current volume vs 90-bar (30-day) rolling mean
vol_ma    = volume.rolling(90, min_periods=30).mean()
vol_anomaly = (volume / vol_ma.replace(0, np.nan) - 1).clip(-3, 3)

# 4. BTC-relative return: coin 8h mom minus BTC 8h mom
# Use all_syms panel including BTC for this
panel_all = load_panels(["mom_8h"])
panel_all = resample_panel(panel_all, REBAL_FREQ)
mom_8h_all = panel_all["mom_8h"]
if "BTCUSDT" in mom_8h_all.columns:
    btc_mom  = mom_8h_all["BTCUSDT"]
    mom_8h_u = mom_8h_all[univ_syms]
    btc_relative = mom_8h_u.sub(btc_mom, axis=0)
else:
    print("  BTCUSDT not found — skipping btc_relative")
    btc_relative = None

# existing baseline signals
z_funding = cs_zscore(funding)
z_mom24h  = cs_zscore(mom_24h)
composite_base = (z_funding + z_mom24h) / 2


# ── IC for each signal ─────────────────────────────────────────────────────

print("\nComputing ICs...")

signals = {
    "funding (baseline)"    : cs_zscore(funding),
    "mom_24h (baseline)"    : cs_zscore(mom_24h),
    "composite (baseline)"  : composite_base,
    "funding_trend"         : cs_zscore(funding_trend),
    "oi_growth"             : cs_zscore(oi_growth),
    "vol_anomaly"           : cs_zscore(vol_anomaly),
}
if btc_relative is not None:
    signals["btc_relative"] = cs_zscore(btc_relative)

results = {}
ic_series = {}
for name, sig in signals.items():
    ics = compute_ic(sig, fwd_8h)
    results[name] = icir_stats(ics)
    ic_series[name] = ics
    r = results[name]
    print(f"  {name:30s}  ICIR={r['ICIR']:+.3f}  meanIC={r['mean_IC']:+.4f}  t={r['t_stat']:+.2f}  n={r['n']}")

# ── test combinations with new signals ─────────────────────────────────────

print("\nTesting combinations with baseline...")

def test_combo(name, z1, z2, w1=0.5, w2=0.5):
    combo = w1 * z1 + w2 * z2
    combo = combo.clip(-CLIP, CLIP)
    ics   = compute_ic(combo, fwd_8h)
    r     = icir_stats(ics)
    ic_series[name] = ics
    results[name]   = r
    print(f"  {name:40s}  ICIR={r['ICIR']:+.3f}  meanIC={r['mean_IC']:+.4f}  t={r['t_stat']:+.2f}")

test_combo("base + funding_trend",  composite_base, cs_zscore(funding_trend))
test_combo("base + oi_growth",      composite_base, cs_zscore(oi_growth))
test_combo("base + vol_anomaly",    composite_base, cs_zscore(vol_anomaly))
if btc_relative is not None:
    test_combo("base + btc_relative", composite_base, cs_zscore(btc_relative))

# triple combo: best new signals
z_ft  = cs_zscore(funding_trend)
z_oig = cs_zscore(oi_growth)
triple = (composite_base + z_ft + z_oig) / 3
triple = triple.clip(-CLIP, CLIP)
ics = compute_ic(triple, fwd_8h)
r   = icir_stats(ics)
results["base + funding_trend + oi_growth"] = r
ic_series["base + funding_trend + oi_growth"] = ics
print(f"  {'base + funding_trend + oi_growth':40s}  ICIR={r['ICIR']:+.3f}  meanIC={r['mean_IC']:+.4f}  t={r['t_stat']:+.2f}")


# ── save results ───────────────────────────────────────────────────────────

df_res = pd.DataFrame(results).T.round(4)
df_res.to_csv(os.path.join(RESULTS_DIR, "phase11_signal_ic.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase11_signal_ic.csv")


# ── rolling IC plot ────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle("Phase 11: New Signal Rolling IC (42-bar = ~2 week window)", fontsize=14)

plot_pairs = [
    ("funding (baseline)", "funding_trend"),
    ("mom_24h (baseline)", "oi_growth"),
    ("composite (baseline)", "vol_anomaly"),
    ("composite (baseline)", "btc_relative" if btc_relative is not None else "composite (baseline)"),
]

for ax, (n1, n2) in zip(axes, plot_pairs):
    for name, color in [(n1, "steelblue"), (n2, "darkorange")]:
        if name in ic_series:
            s = ic_series[name]
            roll = pd.Series(s.values).rolling(42, min_periods=10).mean()
            ax.plot(roll.values, label=f"{name} (ICIR={results[name]['ICIR']:+.3f})", color=color, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Rolling IC")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase11_signal_ic.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase11_signal_ic.png")

print("\nPhase 11 complete.")
print("\nSummary table:")
print(df_res[["mean_IC","ICIR","t_stat"]].sort_values("ICIR", ascending=False).to_string())
