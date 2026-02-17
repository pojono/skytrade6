#!/usr/bin/env python3
"""Quick visualization: BTC 1-month candlestick chart with regime background coloring."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from sklearn.mixture import GaussianMixture

from regime_detection import load_bars, compute_regime_features

# Load 1 month of BTC data
print("Loading BTC bars...")
df = load_bars("BTCUSDT", "2025-12-01", "2025-12-31")
print(f"Loaded {len(df)} bars")

print("Computing features...")
df = compute_regime_features(df)

# Prepare clustering features (same as regime_classify.py)
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

print("Clustering...")
gmm = GaussianMixture(n_components=2, covariance_type="diag", n_init=5, random_state=42)
labels = gmm.fit_predict(X_scaled.values)

# Assign labels back
df_plot = df.loc[idx].copy()
df_plot["regime"] = labels
df_plot["dt"] = pd.to_datetime(df_plot["timestamp_us"], unit="us", utc=True)

# Figure out which regime is "volatile" (higher avg rvol_1h)
r0_vol = df_plot.loc[df_plot["regime"] == 0, "rvol_1h"].mean()
r1_vol = df_plot.loc[df_plot["regime"] == 1, "rvol_1h"].mean()
volatile_label = 0 if r0_vol > r1_vol else 1
quiet_label = 1 - volatile_label

print(f"Quiet regime (label={quiet_label}): {(labels == quiet_label).mean():.1%}")
print(f"Volatile regime (label={volatile_label}): {(labels == volatile_label).mean():.1%}")

# Resample to 1h candles for readability
df_plot = df_plot.set_index("dt")
ohlc_1h = df_plot.resample("1h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "regime": lambda x: x.mode().iloc[0] if len(x) > 0 else 0
}).dropna()

print(f"Resampled to {len(ohlc_1h)} 1h candles")

# Plot
fig, ax = plt.subplots(figsize=(20, 8))

# Draw regime background
for i in range(len(ohlc_1h)):
    row = ohlc_1h.iloc[i]
    t = ohlc_1h.index[i]
    width = pd.Timedelta(hours=1)
    color = "#ffcccc" if row["regime"] == volatile_label else "#ccffcc"
    ax.axvspan(t, t + width, alpha=0.3, color=color, linewidth=0)

# Draw candlesticks
for i in range(len(ohlc_1h)):
    row = ohlc_1h.iloc[i]
    t = ohlc_1h.index[i]
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]

    color = "#26a69a" if c >= o else "#ef5350"

    # Wick
    ax.plot([t, t], [l, h], color=color, linewidth=0.8)
    # Body
    body_bottom = min(o, c)
    body_height = abs(c - o)
    rect = Rectangle((t - pd.Timedelta(minutes=20), body_bottom),
                      pd.Timedelta(minutes=40), body_height,
                      facecolor=color, edgecolor=color, linewidth=0.5)
    ax.add_patch(rect)

ax.set_title("BTC/USDT — December 2025 (1h candles)\nGreen background = Quiet regime | Red background = Volatile regime",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Price (USDT)", fontsize=12)
ax.set_xlabel("")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
fig.autofmt_xdate()
ax.grid(True, alpha=0.2)
ax.set_xlim(ohlc_1h.index[0], ohlc_1h.index[-1])

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#ccffcc", alpha=0.5, label=f"Quiet ({(labels == quiet_label).mean():.0%})"),
    Patch(facecolor="#ffcccc", alpha=0.5, label=f"Volatile ({(labels == volatile_label).mean():.0%})"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

plt.tight_layout()
out_path = "results/regime_chart_btc_dec2025.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
