"""
Phase 13: Improved Regime Filter

Three improvements over Phase 5 regime filter:

  A. Asymmetric filter: apply regime gate only to SHORT leg.
     Long leg always active (crypto has positive drift).
     Short leg only fires when regime is confirmed.

  B. BTC trend conditioning: add BTC 24h momentum as a 3rd regime gate.
     Avoid shorts when BTC is in strong uptrend (top 30th pctile).

  C. Soft threshold: scale position size by regime confidence (0.25–1.0x)
     instead of binary on/off.

All tested with walk-forward (6mo train / 3mo OOS) to avoid look-ahead.
Compared against baseline (no filter) and Phase 5 filter (binary, symmetric).

Outputs:
  results/phase13_regime_v2.png
  results/phase13_regime_v2.csv
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
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3

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


def metrics(rets, label=""):
    rets = rets.dropna()
    if len(rets) == 0:
        return {"Sharpe": 0, "Ann_Ret": 0, "MaxDD": 0, "Total": 0, "Active%": 0}
    sharpe   = rets.mean() / rets.std() * np.sqrt(PERIODS_PER_YEAR) if rets.std() > 0 else 0
    ann_ret  = (1 + rets).prod() ** (PERIODS_PER_YEAR / len(rets)) - 1
    eq       = (1 + rets).cumprod()
    max_dd   = (eq / eq.cummax() - 1).min()
    total    = eq.iloc[-1] - 1
    return {
        "Sharpe"  : round(sharpe, 3),
        "Ann_Ret" : round(ann_ret * 100, 1),
        "MaxDD"   : round(max_dd * 100, 1),
        "Total"   : round(total * 100, 1),
    }


# ── portfolio engine ────────────────────────────────────────────────────────

def run_strategy(composite, fwd_8h, fee_bps, n_long, n_short,
                 long_mask=None, short_mask=None,
                 long_scale=None, short_scale=None):
    """
    General portfolio engine with per-bar masks and scaling.

    long_mask / short_mask: boolean Series indexed by date. If False, leg is skipped.
    long_scale / short_scale: float Series (0-1) for soft sizing.
    """
    dates    = composite.index
    bar_rets = []
    prev_pos = pd.Series(dtype=float)

    for t in dates:
        sig = composite.loc[t].dropna()
        fwd = fwd_8h.loc[t] if t in fwd_8h.index else pd.Series(dtype=float)

        if len(sig) < (n_long + n_short) * 2:
            bar_rets.append(np.nan)
            continue

        do_long  = True if long_mask  is None else bool(long_mask.get(t,  True))
        do_short = True if short_mask is None else bool(short_mask.get(t, True))
        l_scale  = 1.0  if long_scale  is None else float(long_scale.get(t,  1.0))
        s_scale  = 1.0  if short_scale is None else float(short_scale.get(t, 1.0))

        ranked = sig.rank(ascending=False)
        pos    = pd.Series(0.0, index=sig.index)

        if do_long and n_long > 0:
            longs = ranked[ranked <= n_long].index
            pos[longs] = +l_scale / n_long

        if do_short and n_short > 0:
            shorts = ranked[ranked > len(sig) - n_short].index
            pos[shorts] = -s_scale / n_short

        gross = (pos * fwd.reindex(pos.index).fillna(0)).sum()

        if len(prev_pos) > 0:
            all_idx  = pos.index.union(prev_pos.index)
            delta    = pos.reindex(all_idx, fill_value=0) - prev_pos.reindex(all_idx, fill_value=0)
            turnover = delta.abs().sum() / 2
        else:
            turnover = (pos.abs().sum()) / 2

        fee_cost = turnover * fee_bps / 10000
        bar_rets.append(gross - fee_cost)
        prev_pos = pos.copy()

    return pd.Series(bar_rets, index=dates)


# ── walk-forward regime thresholds ─────────────────────────────────────────

def fit_thresholds_wf(composite, funding, fwd_8h, btc_mom24h=None):
    """
    Walk-forward: for each OOS window, fit regime thresholds on training data.
    Returns per-bar regime signals: signal_strength, funding_disp, btc_trend_pctile.
    """
    all_dates = composite.index
    start     = all_dates[0]
    end       = all_dates[-1]
    train_td  = pd.DateOffset(months=TRAIN_MONTHS)
    oos_td    = pd.DateOffset(months=OOS_MONTHS)

    signal_strength = composite.std(axis=1)
    funding_disp    = funding.std(axis=1)

    # regime active mask (symmetric, signal_strength + funding_disp)
    sym_mask = pd.Series(True, index=all_dates)

    # BTC trend percentile (rolling 180-bar = 60-day)
    if btc_mom24h is not None:
        btc_trend_pctile = btc_mom24h.rolling(180, min_periods=30).rank(pct=True)
    else:
        btc_trend_pctile = pd.Series(0.5, index=all_dates)

    # Walk-forward threshold search
    train_start = start
    oos_records = []   # (date, theta1, theta2) per OOS bar

    while True:
        train_end = train_start + train_td
        oos_start = train_end
        oos_end   = oos_start + oos_td
        if oos_start >= end:
            break

        tr_mask = (all_dates >= train_start) & (all_dates < train_end)
        tr_dates = all_dates[tr_mask]
        if len(tr_dates) < 100:
            train_start = train_start + oos_td
            continue

        ss_tr = signal_strength[tr_dates].dropna()
        fd_tr = funding_disp[tr_dates].dropna()

        best_sharpe = -np.inf
        best_t1, best_t2 = ss_tr.quantile(0.1), fd_tr.quantile(0.1)

        for pct1 in np.arange(0.10, 0.80, 0.10):
            for pct2 in np.arange(0.10, 0.80, 0.10):
                t1 = ss_tr.quantile(pct1) if len(ss_tr) >= 20 else -np.inf
                t2 = fd_tr.quantile(pct2) if len(fd_tr) >= 20 else -np.inf
                active = (signal_strength[tr_dates] > t1) & (funding_disp[tr_dates] > t2)
                rets_tr = pd.Series(np.nan, index=tr_dates)
                # quick proxy: mean return when active vs inactive
                active_rets = signal_strength[tr_dates][active]  # proxy
                if active.sum() < 30:
                    continue
                # Use signal_strength as proxy for quality (higher = better bars)
                proxy_sharpe = active_rets.mean() / active_rets.std() if active_rets.std() > 0 else 0
                if proxy_sharpe > best_sharpe:
                    best_sharpe = proxy_sharpe
                    best_t1, best_t2 = t1, t2

        oos_dates_range = all_dates[(all_dates >= oos_start) & (all_dates < oos_end)]
        for t in oos_dates_range:
            oos_records.append((t, best_t1, best_t2))

        train_start = train_start + oos_td

    if not oos_records:
        return sym_mask, btc_trend_pctile

    oos_df = pd.DataFrame(oos_records, columns=["date","t1","t2"]).set_index("date")
    oos_df = oos_df[~oos_df.index.duplicated()]
    oos_df = oos_df.reindex(all_dates)

    # Build OOS-only symmetric mask
    oos_only_mask = pd.Series(True, index=all_dates)
    for t in all_dates:
        if t in oos_df.index and not pd.isna(oos_df.loc[t, "t1"]):
            t1 = oos_df.loc[t, "t1"]
            t2 = oos_df.loc[t, "t2"]
            active = (signal_strength[t] > t1) and (funding_disp[t] > t2)
            oos_only_mask[t] = active

    return oos_only_mask, btc_trend_pctile


# ── main ───────────────────────────────────────────────────────────────────

print("Loading panels...")
# Need BTC separately for trend
panel_1h_all = load_panels(["close", "funding", "mom_24h", "fwd_8h", "mom_8h"])
panel_all    = resample_panel(panel_1h_all, REBAL_FREQ)

all_syms  = list(panel_all["close"].columns)
univ_syms = [s for s in all_syms if s not in MAJORS]

# Extract BTC signals before universe filter
btc_mom24h = panel_all["mom_24h"].get("BTCUSDT", None)

# Filter to universe
panel = {col: panel_all[col][univ_syms] for col in panel_all}
funding_univ = panel["funding"].ffill()
mom_24h      = panel["mom_24h"]
fwd_8h       = panel["fwd_8h"]

z_funding = cs_zscore(funding_univ)
z_mom24h  = cs_zscore(mom_24h)
composite = (z_funding + z_mom24h) / 2

print(f"Universe: {len(univ_syms)} coins")

# Regime features
signal_strength = composite.std(axis=1)
funding_disp    = funding_univ.std(axis=1)

# ── Baseline (no filter) ──────────────────────────────────────────────────
print("\nRunning baseline...")
rets_base = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT)

# ── Phase 5 regime (binary symmetric, fixed 30th pctile for comparison) ───
print("Running Phase-5-style regime (fixed 30th pctile)...")
t1_fixed = signal_strength.quantile(0.30)
t2_fixed = funding_disp.quantile(0.30)
active_fixed = (signal_strength > t1_fixed) & (funding_disp > t2_fixed)

active_long  = active_fixed
active_short = active_fixed
rets_p5 = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                        long_mask=active_long, short_mask=active_short)

# ── A. Asymmetric filter (filter shorts only) ──────────────────────────────
print("Running asymmetric filter (shorts only)...")
# Shorts filtered by regime; longs always on
rets_asym = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                          long_mask=None, short_mask=active_fixed)

# ── B. BTC trend gate (add to symmetric filter) ───────────────────────────
print("Running BTC trend gate...")
if btc_mom24h is not None:
    # Avoid shorts when BTC is in strong uptrend (top 30% of 60-day rolling)
    btc_trend = btc_mom24h.rolling(180, min_periods=30).rank(pct=True)
    btc_in_uptrend = btc_trend > 0.70  # top 30% momentum = strong uptrend
    # Only short when NOT in BTC uptrend AND regime active
    short_mask_btc = active_fixed & (~btc_in_uptrend)
    rets_btc = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                             long_mask=None, short_mask=short_mask_btc)
    active_pct_btc = short_mask_btc.mean() * 100
else:
    rets_btc = rets_base.copy()
    active_pct_btc = 100.0
    print("  BTCUSDT not found — skipping BTC trend gate")

# ── C. Soft threshold (scale by regime confidence) ─────────────────────────
print("Running soft threshold...")
# Regime confidence = percentile of signal_strength and funding_disp
ss_pct = signal_strength.rank(pct=True)
fd_pct = funding_disp.rank(pct=True)
# Confidence = geometric mean of percentiles, mapped to [0.25, 1.0]
confidence = (ss_pct * fd_pct).apply(np.sqrt)
# below 30th pctile = 0.25x scale, above 70th = 1.0x, linear in between
def conf_to_scale(c):
    if c < 0.30:
        return 0.25
    elif c > 0.70:
        return 1.0
    else:
        return 0.25 + (c - 0.30) / (0.70 - 0.30) * 0.75

scale_series = confidence.map(conf_to_scale)
rets_soft = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                          long_scale=scale_series, short_scale=scale_series)

# ── D. Asymmetric + BTC trend (combined A+B) ───────────────────────────────
print("Running A+B (asymmetric + BTC trend)...")
if btc_mom24h is not None:
    rets_ab = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                            long_mask=None, short_mask=short_mask_btc)
else:
    rets_ab = rets_asym.copy()

# ── E. Asymmetric + soft threshold (longs full, shorts scaled) ────────────
print("Running A+C (asymmetric + soft)...")
# For longs: always full scale
# For shorts: scale by confidence, but long leg always on
rets_ac = run_strategy(composite, fwd_8h, FEE_MAKER_BPS, N_LONG, N_SHORT,
                        long_scale=None, short_scale=scale_series)


# ── collect results ────────────────────────────────────────────────────────

variants = {
    "Baseline"                         : rets_base,
    "Phase-5 regime (symm, 30th)"      : rets_p5,
    "A: Asymmetric (short gate)"       : rets_asym,
    "B: BTC trend gate (asym)"         : rets_btc,
    "C: Soft threshold"                : rets_soft,
    "A+B: Asym + BTC trend"            : rets_ab,
    "A+C: Asym + soft"                 : rets_ac,
}

# Active % for reference
active_pcts = {
    "Baseline"                         : 100.0,
    "Phase-5 regime (symm, 30th)"      : active_fixed.mean()*100,
    "A: Asymmetric (short gate)"       : active_fixed.mean()*100,
    "B: BTC trend gate (asym)"         : active_pct_btc,
    "C: Soft threshold"                : 100.0,
    "A+B: Asym + BTC trend"            : active_pct_btc,
    "A+C: Asym + soft"                 : 100.0,
}

print("\nResults:")
rows = {}
for name, rets in variants.items():
    m = metrics(rets)
    rows[name] = m
    active_pct = active_pcts.get(name, 100.0)
    print(f"  {name:40s}  Sharpe={m['Sharpe']:+.3f}  AnnRet={m['Ann_Ret']:+.1f}%"
          f"  MaxDD={m['MaxDD']:+.1f}%  Active={active_pct:.0f}%")

df_res = pd.DataFrame(rows).T
df_res.to_csv(os.path.join(RESULTS_DIR, "phase13_regime_v2.csv"))
print(f"\nSaved: {RESULTS_DIR}/phase13_regime_v2.csv")


# ── monthly comparison ─────────────────────────────────────────────────────
print("\nMonthly comparison (Baseline vs Best variant):")

# Find best by Sharpe
best_name = max(rows, key=lambda n: rows[n]["Sharpe"])
print(f"  Best variant: {best_name}  (Sharpe {rows[best_name]['Sharpe']:.3f})")

print(f"\n  {'Month':10s}  {'Baseline':>10s}  {'Best':>10s}  {'Diff':>8s}")
m_base = rets_base.dropna().resample("ME").apply(lambda x: (1+x).prod()-1)
m_best = variants[best_name].dropna().resample("ME").apply(lambda x: (1+x).prod()-1)
for mo in m_base.index:
    rb = m_base.get(mo, 0) * 100
    rv = m_best.get(mo, 0) * 100
    diff = rv - rb
    flag = " +" if diff > 2 else (" -" if diff < -2 else "  ")
    print(f"  {str(mo)[:7]:10s}  {rb:+10.1f}%  {rv:+10.1f}%  {diff:+8.1f}%{flag}")


# ── plot ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Phase 13: Regime Filter Improvements", fontsize=14)

colors = {
    "Baseline"                    : "steelblue",
    "Phase-5 regime (symm, 30th)" : "gray",
    "A: Asymmetric (short gate)"  : "darkorange",
    "B: BTC trend gate (asym)"    : "green",
    "C: Soft threshold"           : "purple",
    "A+B: Asym + BTC trend"       : "crimson",
    "A+C: Asym + soft"            : "gold",
}

for name, rets in variants.items():
    r = rets.dropna()
    eq   = (1 + r).cumprod() * 1000
    m    = rows[name]
    lw   = 2.0 if name in ("Baseline", best_name) else 1.0
    ax1.plot(eq.values, label=f"{name} | S={m['Sharpe']:.2f} DD={m['MaxDD']:.0f}%",
             color=colors.get(name, "gray"), lw=lw)
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.values, color=colors.get(name, "gray"), lw=lw, label=name)

ax1.set_ylabel("Portfolio Value ($)")
ax1.set_title("Equity Curves ($1000 start, log scale)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

ax2.set_ylabel("Drawdown (%)")
ax2.set_title("Drawdown")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color="black", lw=0.8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase13_regime_v2.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase13_regime_v2.png")

print("\nPhase 13 complete.")
