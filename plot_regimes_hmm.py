#!/usr/bin/env python3
"""Quick visualization: BTC Dec 2025 candlestick chart with HMM regime background coloring.
Side-by-side comparison with GMM."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

from regime_detection import load_bars, compute_regime_features

# Load 1 month of BTC data
print("Loading BTC bars...")
df = load_bars("BTCUSDT", "2025-12-01", "2025-12-31")
print(f"Loaded {len(df)} bars")

print("Computing features...")
df = compute_regime_features(df)

# Prepare clustering features
CLUSTER_FEATURES = [
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "parkvol_1h", "parkvol_4h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h", "vol_ratio_bar",
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    "trade_intensity_ratio",
    "bar_eff_1h", "bar_eff_4h",
    "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "sign_persist_1h", "sign_persist_2h",
    "vol_sma_24h",
]

cols = [c for c in CLUSTER_FEATURES if c in df.columns]
X = df[cols].copy()
valid_mask = X.notna().all(axis=1)
X = X[valid_mask]
idx = X.index

means = X.mean()
stds = X.std().clip(lower=1e-10)
X_scaled = (X - means) / stds
X_arr = X_scaled.values

# --- GMM ---
print("Fitting GMM...")
gmm = GaussianMixture(n_components=2, covariance_type="diag", n_init=5, random_state=42)
gmm_labels = gmm.fit_predict(X_arr)

r0_vol = df.loc[idx, "rvol_1h"].values[gmm_labels == 0].mean()
r1_vol = df.loc[idx, "rvol_1h"].values[gmm_labels == 1].mean()
if r0_vol > r1_vol:
    gmm_labels = 1 - gmm_labels

gmm_trans = np.sum(np.diff(gmm_labels) != 0)
print(f"GMM: {gmm_trans} transitions, quiet={( gmm_labels == 0).mean():.1%}")

# --- HMM ---
print("Fitting HMM...")
best_hmm = None
best_score = -np.inf
for init in range(5):
    try:
        hmm = GaussianHMM(n_components=2, covariance_type="diag",
                          n_iter=200, tol=1e-4, random_state=42 + init, verbose=False)
        hmm.fit(X_arr)
        score = hmm.score(X_arr)
        if score > best_score:
            best_score = score
            best_hmm = hmm
    except:
        pass

hmm_labels = best_hmm.predict(X_arr)  # Viterbi

r0_vol = df.loc[idx, "rvol_1h"].values[hmm_labels == 0].mean()
r1_vol = df.loc[idx, "rvol_1h"].values[hmm_labels == 1].mean()
if r0_vol > r1_vol:
    hmm_labels = 1 - hmm_labels

hmm_trans = np.sum(np.diff(hmm_labels) != 0)
print(f"HMM: {hmm_trans} transitions, quiet={(hmm_labels == 0).mean():.1%}")

# --- Prepare plot data ---
df_plot = df.loc[idx].copy()
df_plot["gmm_regime"] = gmm_labels
df_plot["hmm_regime"] = hmm_labels
df_plot["dt"] = pd.to_datetime(df_plot["timestamp_us"], unit="us", utc=True)
df_plot = df_plot.set_index("dt")

ohlc_1h = df_plot.resample("1h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum",
    "gmm_regime": lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
    "hmm_regime": lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
}).dropna()

print(f"Resampled to {len(ohlc_1h)} 1h candles")


def draw_chart(ax, ohlc, regime_col, title, volatile_label=1):
    """Draw candlestick chart with regime background."""
    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        t = ohlc.index[i]
        width = pd.Timedelta(hours=1)
        color = "#ffcccc" if row[regime_col] == volatile_label else "#ccffcc"
        ax.axvspan(t, t + width, alpha=0.3, color=color, linewidth=0)

    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        t = ohlc.index[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([t, t], [l, h], color=color, linewidth=0.8)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        rect = Rectangle((t - pd.Timedelta(minutes=20), body_bottom),
                          pd.Timedelta(minutes=40), body_height,
                          facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USDT)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(ohlc.index[0], ohlc.index[-1])


# --- Side-by-side plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True, sharey=True)

gmm_quiet_pct = (gmm_labels == 0).mean()
hmm_quiet_pct = (hmm_labels == 0).mean()

draw_chart(ax1, ohlc_1h, "gmm_regime",
           f"GMM — {gmm_trans} transitions | Quiet {gmm_quiet_pct:.0%} / Volatile {1-gmm_quiet_pct:.0%}",
           volatile_label=1)

draw_chart(ax2, ohlc_1h, "hmm_regime",
           f"HMM (Viterbi) — {hmm_trans} transitions | Quiet {hmm_quiet_pct:.0%} / Volatile {1-hmm_quiet_pct:.0%}",
           volatile_label=1)

# Legend
legend_elements = [
    Patch(facecolor="#ccffcc", alpha=0.5, label="Quiet regime"),
    Patch(facecolor="#ffcccc", alpha=0.5, label="Volatile regime"),
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=11)
ax2.legend(handles=legend_elements, loc="upper left", fontsize=11)

fig.suptitle("BTC/USDT — December 2025 (1h candles)\nGMM vs HMM Regime Detection",
             fontsize=15, fontweight="bold", y=0.98)

fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = "results/regime_chart_btc_dec2025_hmm.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
