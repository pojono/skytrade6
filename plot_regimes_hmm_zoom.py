#!/usr/bin/env python3
"""Zoomed-in 5-min bar chart showing GMM vs HMM regime differences."""

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

# Prepare data
df_plot = df.loc[idx].copy()
df_plot["gmm"] = gmm_labels
df_plot["hmm"] = hmm_labels
df_plot["dt"] = pd.to_datetime(df_plot["timestamp_us"], unit="us", utc=True)

# Find a 3-day window with most disagreements to zoom into
df_plot = df_plot.set_index("dt")
df_plot["disagree"] = (df_plot["gmm"] != df_plot["hmm"]).astype(int)

# Rolling 3-day disagreement count
window = 288 * 3  # 3 days of 5-min bars
disagree_rolling = df_plot["disagree"].rolling(window).sum()
best_end = disagree_rolling.idxmax()
best_start = best_end - pd.Timedelta(days=3)

print(f"Zoom window: {best_start} to {best_end}")
print(f"Disagreements in window: {disagree_rolling.max():.0f} bars")

zoom = df_plot.loc[best_start:best_end].copy()
print(f"Bars in zoom: {len(zoom)}")

# Count transitions in zoom
gmm_zoom_trans = np.sum(np.diff(zoom["gmm"].values) != 0)
hmm_zoom_trans = np.sum(np.diff(zoom["hmm"].values) != 0)
print(f"GMM transitions in zoom: {gmm_zoom_trans}")
print(f"HMM transitions in zoom: {hmm_zoom_trans}")

# Resample to 15-min for readability but preserving more detail than 1h
ohlc = zoom.resample("15min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum",
    "gmm": lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
    "hmm": lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
}).dropna()

print(f"Resampled to {len(ohlc)} 15-min candles")


def draw_chart(ax, ohlc, regime_col, title):
    """Draw candlestick chart with regime background."""
    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        t = ohlc.index[i]
        width = pd.Timedelta(minutes=15)
        color = "#ffcccc" if row[regime_col] == 1 else "#ccffcc"
        ax.axvspan(t, t + width, alpha=0.4, color=color, linewidth=0)

    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        t = ohlc.index[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([t, t], [l, h], color=color, linewidth=0.8)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 10)
        rect = Rectangle((t - pd.Timedelta(minutes=5), body_bottom),
                          pd.Timedelta(minutes=10), body_height,
                          facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USDT)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(ohlc.index[0], ohlc.index[-1])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12), sharex=True, sharey=True)

draw_chart(ax1, ohlc, "gmm",
           f"GMM — {gmm_zoom_trans} transitions in 3 days (15-min candles)")
draw_chart(ax2, ohlc, "hmm",
           f"HMM (Viterbi) — {hmm_zoom_trans} transitions in 3 days (15-min candles)")

legend_elements = [
    Patch(facecolor="#ccffcc", alpha=0.5, label="Quiet regime"),
    Patch(facecolor="#ffcccc", alpha=0.5, label="Volatile regime"),
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=11)
ax2.legend(handles=legend_elements, loc="upper left", fontsize=11)

start_str = best_start.strftime("%b %d")
end_str = best_end.strftime("%b %d")
fig.suptitle(f"BTC/USDT — {start_str} to {end_str} 2025 (15-min candles, zoomed)\n"
             f"GMM vs HMM — showing where HMM filters noise flickers",
             fontsize=14, fontweight="bold", y=0.98)

fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = "results/regime_chart_btc_dec2025_hmm_zoom.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
