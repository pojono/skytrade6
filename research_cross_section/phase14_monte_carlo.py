"""
Phase 14: Monte Carlo Permutation Test

Statistical validation of the strategy's Sharpe ratio:

  1. Permutation test: shuffle the mapping between signal and forward returns
     (destroy temporal alignment while preserving marginal distributions).
     Run 1000 permutations → empirical p-value for Sharpe.

  2. Bootstrap CI: resample 8h bars with replacement.
     95% CI on Sharpe and MaxDD.

  3. Block bootstrap: resample contiguous 7-day blocks to preserve autocorrelation.

Universe: No-Majors (113 coins), funding + mom_24h, N=10, 8h, maker 4bps.

Outputs:
  results/phase14_monte_carlo.png
  results/phase14_monte_carlo.csv
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
N_PERM           = 1000
N_BOOT           = 1000
BLOCK_SIZE       = 21   # 7 days × 3 bars/day
SEED             = 42

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


# ── fast portfolio return computation (vectorized) ─────────────────────────

def compute_bar_returns(composite, fwd_8h, n_long=N_LONG, n_short=N_SHORT, fee_bps=FEE_MAKER_BPS):
    """
    Fast vectorized backtest: compute bar-level net returns.
    Uses rank-based selection, equal weight, fixed turnover approximation.
    """
    dates    = composite.index
    bar_rets = np.full(len(dates), np.nan)
    prev_pos = None

    for i, t in enumerate(dates):
        sig = composite.loc[t].dropna()
        fwd = fwd_8h.loc[t].reindex(sig.index).fillna(0) if t in fwd_8h.index else pd.Series(0.0, index=sig.index)

        if len(sig) < (n_long + n_short) * 2:
            continue

        rank   = sig.rank(ascending=False)
        longs  = rank[rank <= n_long].index
        shorts = rank[rank > len(sig) - n_short].index

        pos = pd.Series(0.0, index=sig.index)
        pos[longs]  = +1.0 / n_long
        pos[shorts] = -1.0 / n_short

        gross = (pos * fwd).sum()

        if prev_pos is not None:
            all_idx  = pos.index.union(prev_pos.index)
            delta    = pos.reindex(all_idx, fill_value=0) - prev_pos.reindex(all_idx, fill_value=0)
            turnover = delta.abs().sum() / 2
        else:
            turnover = 1.0

        bar_rets[i] = gross - turnover * fee_bps / 10000
        prev_pos    = pos.copy()

    return pd.Series(bar_rets, index=dates)


def compute_sharpe(rets):
    r = rets.dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)


def compute_maxdd(rets):
    r  = rets.dropna()
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


# ── main ───────────────────────────────────────────────────────────────────

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

z_funding = cs_zscore(funding)
z_mom24h  = cs_zscore(mom_24h)
composite = (z_funding + z_mom24h) / 2

print(f"Universe: {len(univ_syms)} coins | Bars: {len(composite)}")

print("Computing true strategy returns...")
true_rets  = compute_bar_returns(composite, fwd_8h)
true_sharpe = compute_sharpe(true_rets)
true_maxdd  = compute_maxdd(true_rets)
print(f"  True Sharpe: {true_sharpe:.3f}")
print(f"  True MaxDD:  {true_maxdd*100:.1f}%")


# ── 1. Permutation test ────────────────────────────────────────────────────
print(f"\nPermutation test ({N_PERM} permutations)...")
rng  = np.random.default_rng(SEED)
perm_sharpes = []

# We shuffle the fwd_8h dates (breaks signal-return alignment)
fwd_dates = fwd_8h.index.tolist()
dates     = composite.index.tolist()

for p in range(N_PERM):
    if (p + 1) % 200 == 0:
        print(f"  {p+1}/{N_PERM}...")
    # Shuffle fwd_8h date mapping
    shuffled_idx = rng.permutation(len(fwd_dates))
    fwd_shuffled = pd.DataFrame(
        fwd_8h.values[shuffled_idx],
        index=fwd_8h.index,
        columns=fwd_8h.columns
    )
    rets_p = compute_bar_returns(composite, fwd_shuffled)
    perm_sharpes.append(compute_sharpe(rets_p))

perm_sharpes = np.array(perm_sharpes)
p_value = (perm_sharpes >= true_sharpe).mean()
print(f"  Permutation p-value: {p_value:.4f}")
print(f"  Null Sharpe: mean={perm_sharpes.mean():.3f}, std={perm_sharpes.std():.3f}")
print(f"  True Sharpe {true_sharpe:.3f} vs 95th pctile null {np.percentile(perm_sharpes,95):.3f}")


# ── 2. Bootstrap CI (iid resampling) ───────────────────────────────────────
print(f"\nBootstrap CI ({N_BOOT} samples, iid bars)...")
boot_sharpes = []
boot_maxdds  = []
rets_clean   = true_rets.dropna().values

for b in range(N_BOOT):
    idx     = rng.integers(0, len(rets_clean), size=len(rets_clean))
    sample  = rets_clean[idx]
    sr      = sample.mean() / sample.std() * np.sqrt(PERIODS_PER_YEAR) if sample.std() > 0 else 0
    eq      = np.cumprod(1 + sample)
    mdd     = (eq / np.maximum.accumulate(eq) - 1).min()
    boot_sharpes.append(sr)
    boot_maxdds.append(mdd)

boot_sharpes = np.array(boot_sharpes)
boot_maxdds  = np.array(boot_maxdds)
sharpe_ci    = np.percentile(boot_sharpes, [2.5, 97.5])
maxdd_ci     = np.percentile(boot_maxdds,  [2.5, 97.5])
print(f"  Sharpe 95% CI: [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]")
print(f"  MaxDD 95% CI:  [{maxdd_ci[0]*100:.1f}%, {maxdd_ci[1]*100:.1f}%]")


# ── 3. Block bootstrap ─────────────────────────────────────────────────────
print(f"\nBlock bootstrap ({N_BOOT} samples, block={BLOCK_SIZE} bars)...")
block_sharpes = []
n     = len(rets_clean)
n_blocks = int(np.ceil(n / BLOCK_SIZE))
blocks   = [rets_clean[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] for i in range(n_blocks) if i*BLOCK_SIZE < n]

for b in range(N_BOOT):
    idx     = rng.integers(0, len(blocks), size=n_blocks)
    sample  = np.concatenate([blocks[i] for i in idx])[:n]
    sr      = sample.mean() / sample.std() * np.sqrt(PERIODS_PER_YEAR) if sample.std() > 0 else 0
    block_sharpes.append(sr)

block_sharpes = np.array(block_sharpes)
block_ci      = np.percentile(block_sharpes, [2.5, 97.5])
print(f"  Block Sharpe 95% CI: [{block_ci[0]:.3f}, {block_ci[1]:.3f}]")


# ── save results ───────────────────────────────────────────────────────────

summary = {
    "True Sharpe"              : round(true_sharpe, 3),
    "True MaxDD (%)"           : round(true_maxdd * 100, 1),
    "Permutation p-value"      : round(p_value, 4),
    "Null Sharpe (mean)"       : round(perm_sharpes.mean(), 3),
    "Null Sharpe 95th pctile"  : round(np.percentile(perm_sharpes, 95), 3),
    "Bootstrap Sharpe CI low"  : round(sharpe_ci[0], 3),
    "Bootstrap Sharpe CI high" : round(sharpe_ci[1], 3),
    "Block CI low"             : round(block_ci[0], 3),
    "Block CI high"            : round(block_ci[1], 3),
    "Bootstrap MaxDD CI low"   : round(maxdd_ci[0] * 100, 1),
    "Bootstrap MaxDD CI high"  : round(maxdd_ci[1] * 100, 1),
}
pd.Series(summary).to_csv(os.path.join(RESULTS_DIR, "phase14_monte_carlo.csv"), header=["value"])
print(f"\nSaved: {RESULTS_DIR}/phase14_monte_carlo.csv")


# ── plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 14: Monte Carlo Validation", fontsize=14)

# Permutation distribution
ax = axes[0]
ax.hist(perm_sharpes, bins=50, color="steelblue", alpha=0.7, label="Null distribution")
ax.axvline(true_sharpe, color="crimson", lw=2, label=f"True Sharpe = {true_sharpe:.2f}")
ax.axvline(np.percentile(perm_sharpes, 95), color="orange", lw=1.5, ls="--",
           label=f"95th pctile = {np.percentile(perm_sharpes,95):.2f}")
ax.set_xlabel("Sharpe (permuted)")
ax.set_ylabel("Count")
ax.set_title(f"Permutation Test\np-value = {p_value:.4f}")
ax.legend(fontsize=9)

# Bootstrap Sharpe distribution
ax = axes[1]
ax.hist(boot_sharpes, bins=50, color="darkorange", alpha=0.7, label="Bootstrap samples")
ax.axvline(true_sharpe, color="crimson", lw=2, label=f"True = {true_sharpe:.2f}")
ax.axvline(sharpe_ci[0], color="gray", lw=1.5, ls="--", label=f"95% CI [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
ax.axvline(sharpe_ci[1], color="gray", lw=1.5, ls="--")
ax.set_xlabel("Sharpe (bootstrap)")
ax.set_title("Bootstrap CI (iid bars)")
ax.legend(fontsize=9)

# Block bootstrap
ax = axes[2]
ax.hist(block_sharpes, bins=50, color="green", alpha=0.7, label="Block bootstrap")
ax.axvline(true_sharpe, color="crimson", lw=2, label=f"True = {true_sharpe:.2f}")
ax.axvline(block_ci[0], color="gray", lw=1.5, ls="--", label=f"95% CI [{block_ci[0]:.2f}, {block_ci[1]:.2f}]")
ax.axvline(block_ci[1], color="gray", lw=1.5, ls="--")
ax.set_xlabel("Sharpe (block bootstrap)")
ax.set_title(f"Block Bootstrap CI\n(block={BLOCK_SIZE} bars = 7 days)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase14_monte_carlo.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/phase14_monte_carlo.png")

print("\nPhase 14 complete.")
print("\nKey findings:")
for k, v in summary.items():
    print(f"  {k}: {v}")
