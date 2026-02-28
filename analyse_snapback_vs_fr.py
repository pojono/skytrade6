#!/usr/bin/env python3
"""Analyse snap-back magnitude vs FR value using ms-precision trade data.

For each settlement hour, compute:
  - FR value (bps) from historical FR data
  - VWAP at T+0..T+100ms
  - VWAP at T+4900..T+5100ms
  - Snap-back = (price_T0 - price_T5000) / price_T0 * 10000  (bps)

Plot scatter: X = FR (bps), Y = snap-back (bps)

Memory-efficient: loads one day at a time, vectorized numpy ops, max 4 months.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# Unbuffered prints
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/ubuntu/Projects/skytrade6/data"
FR_PARQUET = "/home/ubuntu/Projects/skytrade6/data_all/historical_fr/bybit_fr_history.parquet"
OUT_DIR = "/home/ubuntu/Projects/skytrade6"

# 10 coins: picked by having both ≥120 days trade data AND frequent |FR|>20bps
SYMBOLS = [
    "RIVERUSDT", "0GUSDT", "PIPPINUSDT", "MYXUSDT", "AXSUSDT",
    "SOONUSDT", "COAIUSDT", "TNSRUSDT", "MUSDT", "FLOWUSDT",
]

T_WINDOW_MS = 5000  # measure snap-back over 5 seconds
MAX_DAYS = 120      # ~4 months max per coin

# ── Load FR history (only needed symbols) ─────────────────────────────────────
print("Loading FR history...")
fr_df = pd.read_parquet(FR_PARQUET, columns=["symbol", "fundingTime", "fundingRate"])
fr_df = fr_df[fr_df["symbol"].isin(SYMBOLS)].copy()
fr_df["fundingTime"] = pd.to_datetime(fr_df["fundingTime"], utc=True)
fr_df["fr_bps"] = fr_df["fundingRate"].astype(np.float32) * 10000
fr_df["settle_epoch"] = (fr_df["fundingTime"].astype(np.int64) // 10**9).astype(np.int64)
fr_df.drop(columns=["fundingRate", "fundingTime"], inplace=True)
print(f"  FR records for selected symbols: {len(fr_df)}")

# Build lookup: (symbol, epoch_s) -> fr_bps
fr_lookup = dict(zip(
    zip(fr_df["symbol"], fr_df["settle_epoch"]),
    fr_df["fr_bps"]
))
del fr_df  # free memory


def process_day_fast(ts: np.ndarray, px: np.ndarray, sz: np.ndarray,
                     settle_epochs: np.ndarray, sym: str) -> list:
    """Fast O(log n) per settlement using searchsorted on sorted timestamps."""
    out = []
    for se in settle_epochs:
        # T+0 window: [se, se+0.1)
        i0_lo = np.searchsorted(ts, se, side="left")
        i0_hi = np.searchsorted(ts, se + 0.1, side="left")
        # T+5s window: [se+4.9, se+5.1)
        i5_lo = np.searchsorted(ts, se + 4.9, side="left")
        i5_hi = np.searchsorted(ts, se + 5.1, side="left")

        if i0_lo >= i0_hi or i5_lo >= i5_hi:
            continue

        p0, s0 = px[i0_lo:i0_hi], sz[i0_lo:i0_hi]
        p5, s5 = px[i5_lo:i5_hi], sz[i5_lo:i5_hi]

        vwap0 = np.dot(p0, s0) / s0.sum()
        vwap5 = np.dot(p5, s5) / s5.sum()
        snap = (vwap0 - vwap5) / vwap0 * 10000

        fr = fr_lookup.get((sym, int(se)))
        if fr is None:
            continue

        out.append((sym, int(se), float(fr), float(vwap0), float(vwap5),
                    float(snap), int(i0_hi - i0_lo), int(i5_hi - i5_lo)))
    return out


# ── Process trade data ────────────────────────────────────────────────────────
results = []
t_start = time.time()

for sym in SYMBOLS:
    trade_path = os.path.join(DATA_DIR, sym, "bybit", "futures")
    if not os.path.exists(trade_path):
        print(f"  {sym}: no trade data dir, skipping")
        continue

    files = sorted(glob.glob(os.path.join(trade_path, f"{sym}*.csv.gz")))
    if not files:
        print(f"  {sym}: no trade files, skipping")
        continue

    # Limit to most recent MAX_DAYS files
    files = files[-MAX_DAYS:]
    print(f"  {sym}: processing {len(files)} days...")

    # Precompute settlement epochs relevant for this symbol
    all_settle = np.array([k[1] for k in fr_lookup if k[0] == sym], dtype=np.float64)

    sym_count = 0
    for i, fpath in enumerate(files):
        try:
            df = pd.read_csv(
                fpath,
                usecols=["timestamp", "price", "size"],
                dtype={"timestamp": np.float64, "price": np.float64, "size": np.float64},
            )
        except Exception:
            continue

        if df.empty:
            continue

        ts = df["timestamp"].values
        px = df["price"].values
        sz = df["size"].values
        del df  # free dataframe

        # Filter settlements that fall within this day's data range
        day_start, day_end = ts[0], ts[-1]
        day_settles = all_settle[(all_settle >= day_start - 1) & (all_settle <= day_end + 1)]

        if len(day_settles) == 0:
            continue

        # Data is already time-sorted from Bybit, but verify
        if not np.all(ts[:-1] <= ts[1:]):
            order = np.argsort(ts)
            ts, px, sz = ts[order], px[order], sz[order]

        day_results = process_day_fast(ts, px, sz, day_settles, sym)
        results.extend(day_results)
        sym_count += len(day_results)

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t_start
            print(f"    {i+1}/{len(files)} days, {sym_count} settlements, {elapsed:.1f}s")

    elapsed = time.time() - t_start
    print(f"    → {sym_count} settlements  ({elapsed:.1f}s elapsed)")

print(f"\nTotal: {len(results)} settlements processed in {time.time()-t_start:.1f}s")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(results, columns=[
    "symbol", "settle_epoch", "fr_bps", "price_t0", "price_t5",
    "snapback_bps", "n_trades_t0", "n_trades_t5",
])
del results  # free list

if df_res.empty:
    print("No results to plot!")
    exit(1)

print(f"\nResults (all): {len(df_res)} settlements")

# Filter: only |FR| > 20bps
df_res = df_res[df_res["fr_bps"].abs() > 20].copy()
print(f"Results (|FR|>20bps): {len(df_res)} settlements")
print(f"FR range: {df_res['fr_bps'].min():.1f} to {df_res['fr_bps'].max():.1f} bps")
print(f"Snap-back range: {df_res['snapback_bps'].min():.1f} to {df_res['snapback_bps'].max():.1f} bps")
print(f"\nPer symbol:")
for sym in SYMBOLS:
    sub = df_res[df_res["symbol"] == sym]
    if len(sub) > 0:
        print(f"  {sym:15s}: {len(sub):5d} settlements  FR: [{sub['fr_bps'].min():+7.1f}, {sub['fr_bps'].max():+7.1f}]  "
              f"Snap: [{sub['snapback_bps'].min():+7.1f}, {sub['snapback_bps'].max():+7.1f}]")

DPI = 300  # high resolution for fine grid

# ── Chart 1: Scatter — FR vs Snap-back (10bps grid) ───────────────────────────
fig, ax = plt.subplots(figsize=(18, 11), facecolor="#FAFAFA")

# Color by symbol
sym_colors = plt.cm.tab10(np.linspace(0, 1, len(SYMBOLS)))
sym_color_map = {s: sym_colors[i] for i, s in enumerate(SYMBOLS)}

for sym in SYMBOLS:
    sub = df_res[df_res["symbol"] == sym]
    if sub.empty:
        continue
    ax.scatter(
        sub["fr_bps"], sub["snapback_bps"],
        c=[sym_color_map[sym]], label=f"{sym} (n={len(sub)})",
        alpha=0.45, s=18, edgecolors="none",
    )

# Reference lines
ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.6)
ax.axvline(x=0, color="black", linewidth=0.8, alpha=0.6)

# Regression line
mask_valid = np.isfinite(df_res["fr_bps"]) & np.isfinite(df_res["snapback_bps"])
x_all = df_res.loc[mask_valid, "fr_bps"].values
y_all = df_res.loc[mask_valid, "snapback_bps"].values

if len(x_all) > 10:
    coeffs = np.polyfit(x_all, y_all, 1)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    r2 = np.corrcoef(x_all, y_all)[0, 1] ** 2
    ax.plot(x_line, y_line, "r-", linewidth=2, alpha=0.8,
            label=f"Linear fit: slope={coeffs[0]:.3f}, R\u00b2={r2:.3f}")

    # Bin averages (10bps buckets)
    bins = np.arange(np.floor(x_all.min() / 10) * 10, np.ceil(x_all.max() / 10) * 10 + 10, 10)
    bin_idx = np.digitize(x_all, bins)
    bin_means_x, bin_means_y = [], []
    for i in range(1, len(bins)):
        mask_bin = bin_idx == i
        if mask_bin.sum() >= 3:
            bin_means_x.append((bins[i-1] + bins[i]) / 2)
            bin_means_y.append(y_all[mask_bin].mean())
    ax.plot(bin_means_x, bin_means_y, "ko-", markersize=5, linewidth=1.5, alpha=0.8,
            label="Bin average (10bps)")

# 10bps grid on both axes
y_lo = int(np.floor(df_res["snapback_bps"].min() / 10) * 10) - 10
y_hi = int(np.ceil(df_res["snapback_bps"].max() / 10) * 10) + 10
x_lo = int(np.floor(df_res["fr_bps"].min() / 10) * 10) - 10
x_hi = int(np.ceil(df_res["fr_bps"].max() / 10) * 10) + 10
ax.set_yticks(np.arange(y_lo, y_hi + 1, 10), minor=False)
ax.set_xticks(np.arange(x_lo, x_hi + 1, 10), minor=False)
ax.set_yticks(np.arange(y_lo, y_hi + 1, 50), minor=False)
ax.grid(which="major", alpha=0.15, linewidth=0.5)
ax.tick_params(axis="both", labelsize=7)

ax.set_xlabel("Funding Rate at Settlement (bps)", fontsize=12)
ax.set_ylabel("Price at T+5000ms vs T+0 (bps)\n(positive = price dropped = good for short)", fontsize=11)
ax.set_title(f"Snap-back at T+5s vs Funding Rate — {len(df_res)} Settlements (|FR|>20bps), {len(SYMBOLS)} Coins\n"
             f"Bybit Perpetuals, ms-precision trade data",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", ncol=2)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] * 0.95, ylim[1] * 0.92, "Negative FR\nPrice drops\n(SHORT profits)",
        fontsize=9, color="green", alpha=0.5, ha="left", va="top", fontweight="bold")
ax.text(xlim[1] * 0.95, ylim[0] * 0.92, "Positive FR\nPrice rises\n(LONG profits)",
        fontsize=9, color="green", alpha=0.5, ha="right", va="bottom", fontweight="bold")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "snapback_vs_fr_scatter.png")
fig.savefig(out1, dpi=DPI, bbox_inches="tight", facecolor="#FAFAFA")
print(f"\nSaved: {out1}")
plt.close(fig)


# ── Chart 2: Heatmap (10bps buckets on X and Y) ─────────────────────────────
print("Building heatmap...")
from matplotlib.colors import LogNorm

# 10bps bins for both axes
xbins = np.arange(x_lo, x_hi + 10, 10)
ybins = np.arange(y_lo, y_hi + 10, 10)

hist, xedges, yedges = np.histogram2d(
    df_res["fr_bps"].values, df_res["snapback_bps"].values,
    bins=[xbins, ybins]
)

fig3, ax3 = plt.subplots(figsize=(18, 11), facecolor="#FAFAFA")

# Mask zeros for cleaner look
hist_masked = np.ma.masked_where(hist == 0, hist)

im = ax3.pcolormesh(
    xedges, yedges, hist_masked.T,
    cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)),
    edgecolors="#CCCCCC", linewidth=0.3,
)

# Add count text in each cell
for i in range(len(xedges) - 1):
    for j in range(len(yedges) - 1):
        count = int(hist[i, j])
        if count > 0:
            cx = (xedges[i] + xedges[i+1]) / 2
            cy = (yedges[j] + yedges[j+1]) / 2
            color = "white" if count >= 5 else "black"
            ax3.text(cx, cy, str(count), ha="center", va="center",
                     fontsize=5.5, fontweight="bold", color=color)

cbar = fig3.colorbar(im, ax=ax3, shrink=0.8, label="# settlements")
ax3.axhline(y=0, color="blue", linewidth=1, alpha=0.5, linestyle="--")
ax3.axvline(x=0, color="blue", linewidth=1, alpha=0.5, linestyle="--")

# Overlay bin-average line
if len(bin_means_x) > 0:
    ax3.plot(bin_means_x, bin_means_y, "ko-", markersize=4, linewidth=1.5,
             alpha=0.9, label="Bin average")
    ax3.legend(fontsize=9, loc="upper right")

# 10bps grid ticks
ax3.set_xticks(xbins)
ax3.set_yticks(ybins)
ax3.tick_params(axis="both", labelsize=6)

ax3.set_xlabel("Funding Rate at Settlement (bps)", fontsize=12)
ax3.set_ylabel("Price at T+5000ms vs T+0 (bps)\n(positive = price dropped)", fontsize=11)
ax3.set_title(f"Heatmap: Snap-back at T+5s vs FR — {len(df_res)} Settlements (|FR|>20bps)\n"
              f"10bps buckets, Bybit Perpetuals",
              fontsize=13, fontweight="bold")

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "snapback_vs_fr_heatmap.png")
fig3.savefig(out3, dpi=DPI, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved: {out3}")
plt.close(fig3)


# ── Chart 3: Binned bar chart ───────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(14, 7), facecolor="#FAFAFA")

neg = df_res[df_res["fr_bps"] < -20].copy()
neg["fr_bucket"] = pd.cut(neg["fr_bps"], bins=[-500, -200, -150, -100, -75, -50, -30, -20])

bucket_stats = neg.groupby("fr_bucket", observed=True).agg(
    mean_snap=("snapback_bps", "mean"),
    median_snap=("snapback_bps", "median"),
    std_snap=("snapback_bps", "std"),
    count=("snapback_bps", "count"),
    pct_positive=("snapback_bps", lambda x: (x > 0).mean() * 100),
).dropna()

if not bucket_stats.empty:
    x_pos = range(len(bucket_stats))
    labels = [str(b) for b in bucket_stats.index]
    colors = ["#E74C3C" if m < 0 else "#2ECC71" for m in bucket_stats["mean_snap"]]

    bars = ax4.bar(x_pos, bucket_stats["mean_snap"], color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.8, width=0.7)
    ax4.errorbar(x_pos, bucket_stats["mean_snap"],
                 yerr=bucket_stats["std_snap"] / np.sqrt(bucket_stats["count"]),
                 fmt="none", color="black", capsize=5)

    for i, (mean, med, count, pct) in enumerate(zip(
            bucket_stats["mean_snap"], bucket_stats["median_snap"],
            bucket_stats["count"], bucket_stats["pct_positive"])):
        ax4.text(i, mean + 3,
                 f"mean={mean:+.1f}\nmed={med:+.1f}\nn={count}\n{pct:.0f}% drop",
                 ha="center", fontsize=8, fontweight="bold")

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, fontsize=9, rotation=25)
    ax4.set_xlabel("FR Bucket (bps)", fontsize=11)
    ax4.set_ylabel("Mean price change at T+5s (bps)", fontsize=11)
    ax4.set_title(f"Negative FR: Snap-back at T+5000ms by FR Magnitude\n"
                  f"{len(neg)} settlements, |FR|>20bps",
                  fontsize=13, fontweight="bold")
    ax4.axhline(y=0, color="black", linewidth=0.8)
    ax4.set_yticks(np.arange(-20, int(bucket_stats["mean_snap"].max()) + 30, 10))
    ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "snapback_vs_fr_buckets.png")
fig4.savefig(out4, dpi=DPI, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved: {out4}")
plt.close(fig4)

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY (|FR| > 20bps only)")
print("="*80)

for label, sub_df in [("All |FR|>20", df_res),
                       ("FR < -50bps", df_res[df_res["fr_bps"] < -50]),
                       ("FR < -100bps", df_res[df_res["fr_bps"] < -100]),
                       ("FR < -200bps", df_res[df_res["fr_bps"] < -200]),
                       ("FR > +50bps", df_res[df_res["fr_bps"] > 50]),
                       ("FR > +100bps", df_res[df_res["fr_bps"] > 100])]:
    if len(sub_df) < 3:
        print(f"  {label:20s}: n={len(sub_df)} (too few)")
        continue
    corr = sub_df["fr_bps"].corr(sub_df["snapback_bps"])
    print(f"  {label:20s}: n={len(sub_df):5d}  "
          f"mean_snap={sub_df['snapback_bps'].mean():+6.1f}bps  "
          f"median={sub_df['snapback_bps'].median():+6.1f}bps  "
          f"corr={corr:+.3f}  "
          f"drop_rate={(sub_df['snapback_bps'] > 0).mean()*100:.0f}%")
