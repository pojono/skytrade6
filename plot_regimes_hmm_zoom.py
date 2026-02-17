#!/usr/bin/env python3
"""Zoomed-in 5-min bar chart showing GMM vs HMM regime differences.
24-hour window at raw 5-min resolution — no resampling."""

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

print("Loading BTC bars...")
df = load_bars("BTCUSDT", "2025-12-01", "2025-12-31")
print(f"Loaded {len(df)} bars")

print("Computing features...")
df = compute_regime_features(df)

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

# GMM
print("Fitting GMM...")
gmm = GaussianMixture(n_components=2, covariance_type="diag", n_init=5, random_state=42)
gmm_labels = gmm.fit_predict(X_arr)
r0 = df.loc[idx, "rvol_1h"].values[gmm_labels == 0].mean()
r1 = df.loc[idx, "rvol_1h"].values[gmm_labels == 1].mean()
if r0 > r1: gmm_labels = 1 - gmm_labels

# HMM
print("Fitting HMM...")
best_hmm, best_score = None, -np.inf
for init in range(5):
    try:
        hmm = GaussianHMM(n_components=2, covariance_type="diag",
                          n_iter=200, tol=1e-4, random_state=42 + init)
        hmm.fit(X_arr)
        s = hmm.score(X_arr)
        if s > best_score: best_score, best_hmm = s, hmm
    except: pass
hmm_labels = best_hmm.predict(X_arr)
r0 = df.loc[idx, "rvol_1h"].values[hmm_labels == 0].mean()
r1 = df.loc[idx, "rvol_1h"].values[hmm_labels == 1].mean()
if r0 > r1: hmm_labels = 1 - hmm_labels

# Find best 24h window with most disagreements at raw 5-min level
disagree = (gmm_labels != hmm_labels).astype(int)
window = 288  # 24h of 5-min bars
rolling_dis = np.convolve(disagree, np.ones(window), mode='valid')
best_start_i = np.argmax(rolling_dis)

# Extract zoom slice using array indices
gmm_zoom = gmm_labels[best_start_i:best_start_i + window]
hmm_zoom = hmm_labels[best_start_i:best_start_i + window]
idx_zoom = idx[best_start_i:best_start_i + window]

gmm_zoom_trans = np.sum(np.diff(gmm_zoom) != 0)
hmm_zoom_trans = np.sum(np.diff(hmm_zoom) != 0)
n_disagree = int(rolling_dis[best_start_i])

# Build plot dataframe at raw 5-min — NO resampling
df_zoom = df.loc[idx_zoom].copy()
df_zoom["gmm"] = gmm_zoom
df_zoom["hmm"] = hmm_zoom
df_zoom["dt"] = pd.to_datetime(df_zoom["timestamp_us"], unit="us", utc=True)

print(f"Zoom: {df_zoom['dt'].iloc[0]} to {df_zoom['dt'].iloc[-1]}")
print(f"Bars: {len(df_zoom)}, Disagreements: {n_disagree}")
print(f"GMM transitions: {gmm_zoom_trans}")
print(f"HMM transitions: {hmm_zoom_trans}")


def draw_chart_5m(ax, df_z, regime_col, title):
    """Draw 5-min candlestick chart with regime background — no resampling."""
    times = df_z["dt"].values
    bar_width = pd.Timedelta(minutes=5)

    # Background
    for i in range(len(df_z)):
        t = pd.Timestamp(times[i])
        color = "#ffcccc" if df_z[regime_col].iloc[i] == 1 else "#ccffcc"
        ax.axvspan(t, t + bar_width, alpha=0.4, color=color, linewidth=0)

    # Candlesticks
    for i in range(len(df_z)):
        row = df_z.iloc[i]
        t = pd.Timestamp(times[i])
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([t, t], [l, h], color=color, linewidth=0.6)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 5)
        rect = Rectangle((t - pd.Timedelta(minutes=1.5), body_bottom),
                          pd.Timedelta(minutes=3), body_height,
                          facecolor=color, edgecolor=color, linewidth=0.3)
        ax.add_patch(rect)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USDT)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(pd.Timestamp(times[0]), pd.Timestamp(times[-1]) + bar_width)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, sharey=True)

draw_chart_5m(ax1, df_zoom, "gmm",
              f"GMM — {gmm_zoom_trans} transitions in 24h")
draw_chart_5m(ax2, df_zoom, "hmm",
              f"HMM (Viterbi) — {hmm_zoom_trans} transitions in 24h")

legend_elements = [
    Patch(facecolor="#ccffcc", alpha=0.5, label="Quiet regime"),
    Patch(facecolor="#ffcccc", alpha=0.5, label="Volatile regime"),
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=11)
ax2.legend(handles=legend_elements, loc="upper left", fontsize=11)

date_str = df_zoom["dt"].iloc[0].strftime("%b %d")
date_str2 = df_zoom["dt"].iloc[-1].strftime("%b %d")
fig.suptitle(f"BTC/USDT — {date_str} to {date_str2} 2025 (raw 5-min bars, no resampling)\n"
             f"GMM: {gmm_zoom_trans} transitions vs HMM: {hmm_zoom_trans} transitions",
             fontsize=14, fontweight="bold", y=0.98)

fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = "results/regime_chart_btc_dec2025_hmm_zoom.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
