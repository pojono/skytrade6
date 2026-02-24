#!/usr/bin/env python3
"""
Chart extreme funding rate events around settlement times.

For each settlement event, find the coin with the most extreme FR,
then plot a 30-min window (15 min before, 15 min after) showing:
  - Left Y-axis:  Funding rate on Binance + Bybit
  - Right Y-axis: Bid/ask prices on Binance + Bybit
  - X-axis:       Time

One chart per settlement. Uses 5-second raw data for maximum resolution.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta
import time
import gc

plt.style.use("dark_background")

DATA = Path("data_all")
OUT = Path("charts_settlement")
OUT.mkdir(exist_ok=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DETECT SETTLEMENT TIMES FROM BINANCE FR (downsampled to 1-min)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("LOADING DATA TO DETECT SETTLEMENTS")
print("=" * 80)

# Load Binance FR at 1-min to detect settlements + find extreme coins
print("  Loading Binance fundingRate (1-min downsample for settlement detection)...")
bn_fr = pd.read_parquet(
    DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"],
)
bn_fr["ts_1m"] = bn_fr["ts"].dt.floor("1min")
bn_fr_1m = bn_fr.groupby(["ts_1m", "symbol"]).agg(
    fr=("lastFundingRate", "last"),
    nextFunding=("nextFundingTime", "last"),
).reset_index()
del bn_fr
gc.collect()
print(f"    {len(bn_fr_1m):,} rows")

# Detect settlements: when nextFundingTime changes for a symbol
bn_fr_1m = bn_fr_1m.sort_values(["symbol", "ts_1m"])
bn_fr_1m["nf_prev"] = bn_fr_1m.groupby("symbol")["nextFunding"].shift(1)
bn_fr_1m["is_settlement"] = (bn_fr_1m["nextFunding"] != bn_fr_1m["nf_prev"]) & bn_fr_1m["nf_prev"].notna()

# For each settlement, get the row BEFORE (which has the FR that was just settled)
settle_idx = bn_fr_1m[bn_fr_1m["is_settlement"]].index
pre_settle = bn_fr_1m.loc[settle_idx - 1].copy()  # row just before the change
pre_settle["settlement_time"] = bn_fr_1m.loc[settle_idx, "ts_1m"].values

# Group by settlement_time, find the coin with most extreme |FR|
pre_settle["fr_abs"] = pre_settle["fr"].abs()
extreme_per_settlement = pre_settle.loc[
    pre_settle.groupby("settlement_time")["fr_abs"].idxmax()
][["settlement_time", "symbol", "fr"]].reset_index(drop=True)

# Also get Bybit settlements
print("  Loading Bybit ticker (1-min downsample for settlement detection)...")
bb_tk_1m_raw = pd.read_parquet(
    DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "fundingRate", "nextFundingTime"],
)
bb_tk_1m_raw["ts_1m"] = bb_tk_1m_raw["ts"].dt.floor("1min")
bb_tk_1m = bb_tk_1m_raw.groupby(["ts_1m", "symbol"]).agg(
    bb_fr=("fundingRate", "last"),
    bb_nextFunding=("nextFundingTime", "last"),
).reset_index()
del bb_tk_1m_raw
gc.collect()

bb_tk_1m = bb_tk_1m.sort_values(["symbol", "ts_1m"])
bb_tk_1m["nf_prev"] = bb_tk_1m.groupby("symbol")["bb_nextFunding"].shift(1)
bb_tk_1m["is_settlement"] = (bb_tk_1m["bb_nextFunding"] != bb_tk_1m["nf_prev"]) & bb_tk_1m["nf_prev"].notna()

bb_settle_idx = bb_tk_1m[bb_tk_1m["is_settlement"]].index
bb_pre_settle = bb_tk_1m.loc[bb_settle_idx - 1].copy()
bb_pre_settle["settlement_time"] = bb_tk_1m.loc[bb_settle_idx, "ts_1m"].values
bb_pre_settle["fr_abs"] = bb_pre_settle["bb_fr"].abs()

bb_extreme = bb_pre_settle.loc[
    bb_pre_settle.groupby("settlement_time")["fr_abs"].idxmax()
][["settlement_time", "symbol", "bb_fr"]].reset_index(drop=True)

del bn_fr_1m, bb_tk_1m, pre_settle, bb_pre_settle
gc.collect()

# Merge: for each Binance settlement, also show what Bybit's most extreme was
# But we chart the Binance extreme coin
print(f"\n  Found {len(extreme_per_settlement)} Binance settlement events")
print(extreme_per_settlement.to_string(index=False))

# Also find Bybit-only settlements (not coinciding with Binance)
bn_settle_times = set(extreme_per_settlement["settlement_time"].dt.floor("2min"))
bb_only = bb_extreme[~bb_extreme["settlement_time"].dt.floor("2min").isin(bn_settle_times)]
print(f"\n  Found {len(bb_only)} Bybit-only settlement events (most extreme coin)")
if len(bb_only) > 0:
    # Pick top 5 most extreme Bybit-only
    bb_only = bb_only.reindex(bb_only["bb_fr"].abs().nlargest(5).index)
    print(bb_only.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FOR EACH SETTLEMENT, LOAD RAW 5s DATA IN THE 30-MIN WINDOW AND CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("GENERATING CHARTS (30-min window around each settlement)")
print("=" * 80)

# We need bid/ask from both exchanges. Binance ticker doesn't have bid/ask,
# but we have lastPrice. Bybit has bid1/ask1.
# For Binance, we'll use lastPrice as a proxy (no order book data).
# Actually let's check what Binance ticker has...
# From schema: lastPrice, highPrice, lowPrice — no bid/ask.
# So for Binance we show lastPrice, for Bybit we show bid1/ask1.

# Load full raw data (we'll filter per chart)
print("  Loading raw parquet files for charting...")
bn_fr_raw = pd.read_parquet(
    DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "markPrice", "nextFundingTime"],
)
print(f"    Binance FR: {len(bn_fr_raw):,} rows")

bn_tk_raw = pd.read_parquet(
    DATA / "binance" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice"],
)
print(f"    Binance ticker: {len(bn_tk_raw):,} rows")

bb_tk_raw = pd.read_parquet(
    DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "fundingRate", "bid1Price", "ask1Price", "lastPrice",
             "nextFundingTime", "fundingIntervalHour"],
)
print(f"    Bybit ticker: {len(bb_tk_raw):,} rows")


def plot_settlement(settlement_time, symbol, fr_value, chart_idx, source="binance"):
    """Plot 30-min window around a settlement for a specific coin."""
    # Ensure tz-aware (UTC) for comparison with raw parquet timestamps
    if settlement_time.tzinfo is None:
        settlement_time = settlement_time.tz_localize("UTC")
    t_start = settlement_time - timedelta(minutes=15)
    t_end = settlement_time + timedelta(minutes=15)

    # Filter raw data for this symbol and time window
    mask_bn_fr = (bn_fr_raw["symbol"] == symbol) & (bn_fr_raw["ts"] >= t_start) & (bn_fr_raw["ts"] <= t_end)
    mask_bn_tk = (bn_tk_raw["symbol"] == symbol) & (bn_tk_raw["ts"] >= t_start) & (bn_tk_raw["ts"] <= t_end)
    mask_bb = (bb_tk_raw["symbol"] == symbol) & (bb_tk_raw["ts"] >= t_start) & (bb_tk_raw["ts"] <= t_end)

    df_bn_fr = bn_fr_raw.loc[mask_bn_fr].copy()
    df_bn_tk = bn_tk_raw.loc[mask_bn_tk].copy()
    df_bb = bb_tk_raw.loc[mask_bb].copy()

    if len(df_bn_fr) == 0 and len(df_bb) == 0:
        print(f"    ⚠ No data for {symbol} around {settlement_time}")
        return False

    fig, ax_fr = plt.subplots(figsize=(16, 8))
    ax_price = ax_fr.twinx()

    # ── Left axis: Funding rates ──
    colors_fr = {"bn": "#FF6B6B", "bb": "#4ECDC4"}
    if len(df_bn_fr) > 0:
        ax_fr.plot(df_bn_fr["ts"], df_bn_fr["lastFundingRate"] * 100,
                   color=colors_fr["bn"], linewidth=2, label="Binance FR", zorder=5)
    if len(df_bb) > 0:
        ax_fr.plot(df_bb["ts"], df_bb["fundingRate"] * 100,
                   color=colors_fr["bb"], linewidth=2, label="Bybit FR", zorder=5)

    ax_fr.set_ylabel("Funding Rate (%)", fontsize=12, color="#FF6B6B")
    ax_fr.tick_params(axis="y", labelcolor="#FF6B6B")
    ax_fr.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)

    # ── Right axis: Prices (bid/ask) ──
    colors_price = {"bn": "#FFD93D", "bb_bid": "#6BCB77", "bb_ask": "#FF6B6B"}

    if len(df_bn_tk) > 0:
        ax_price.plot(df_bn_tk["ts"], df_bn_tk["lastPrice"],
                      color="#FFD93D", linewidth=1, alpha=0.8, label="Binance lastPrice")

    if len(df_bb) > 0:
        ax_price.fill_between(df_bb["ts"], df_bb["bid1Price"], df_bb["ask1Price"],
                              alpha=0.15, color="#6BCB77", label="Bybit bid-ask spread")
        ax_price.plot(df_bb["ts"], df_bb["bid1Price"],
                      color="#6BCB77", linewidth=1, alpha=0.7, label="Bybit bid")
        ax_price.plot(df_bb["ts"], df_bb["ask1Price"],
                      color="#FF8585", linewidth=1, alpha=0.7, label="Bybit ask")

    ax_price.set_ylabel("Price (USDT)", fontsize=12, color="#6BCB77")
    ax_price.tick_params(axis="y", labelcolor="#6BCB77")

    # ── Settlement line ──
    ax_fr.axvline(x=settlement_time, color="white", linewidth=2, linestyle="--", alpha=0.8, label="Settlement")

    # ── Formatting ──
    settle_str = settlement_time.strftime("%Y-%m-%d %H:%M UTC")
    fr_pct = fr_value * 100
    title = f"{symbol} — Settlement at {settle_str}\nFR = {fr_pct:+.4f}% ({source.title()})"
    ax_fr.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax_fr.set_xlabel("Time (UTC)", fontsize=11)

    # X-axis formatting
    ax_fr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_fr.xaxis.set_major_locator(mdates.MinuteLocator(interval=3))
    ax_fr.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    fig.autofmt_xdate()

    # Combined legend
    lines_fr, labels_fr = ax_fr.get_legend_handles_labels()
    lines_p, labels_p = ax_price.get_legend_handles_labels()
    ax_fr.legend(lines_fr + lines_p, labels_fr + labels_p,
                 loc="upper left", fontsize=9, framealpha=0.7)

    ax_fr.grid(True, alpha=0.2)
    fig.tight_layout()

    fname = f"{chart_idx:02d}_{symbol}_{settlement_time.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{chart_idx:2d}] {fname}  ({symbol}, FR={fr_pct:+.4f}%)")
    return True


# Deduplicate: pick unique (symbol, ~FR magnitude) combos, prioritize most extreme
# Sort by absolute FR descending, then drop near-duplicates
eps = extreme_per_settlement.copy()
eps["fr_abs"] = eps["fr"].abs()
eps = eps.sort_values("fr_abs", ascending=False).reset_index(drop=True)

# Keep a coin only if it's the first time we see it, OR its FR is >50% different from last time we charted it
seen = {}  # symbol -> last charted FR
selected = []
for _, row in eps.iterrows():
    sym = row["symbol"]
    fr = row["fr"]
    if sym not in seen:
        seen[sym] = fr
        selected.append(row)
    elif abs(abs(fr) - abs(seen[sym])) / abs(seen[sym]) > 0.5:
        # FR changed significantly — chart again
        seen[sym] = fr
        selected.append(row)

selected = pd.DataFrame(selected)
# Re-sort by time for chronological charts
selected = selected.sort_values("settlement_time").reset_index(drop=True)

print(f"\n  Selected {len(selected)} unique settlement events to chart (from {len(eps)} total)")
print(selected[["settlement_time", "symbol", "fr"]].to_string(index=False))

# Plot selected Binance settlements
chart_idx = 0
for _, row in selected.iterrows():
    chart_idx += 1
    plot_settlement(row["settlement_time"], row["symbol"], row["fr"], chart_idx, source="binance")

# Plot top Bybit-only settlements
if len(bb_only) > 0:
    for _, row in bb_only.iterrows():
        chart_idx += 1
        plot_settlement(row["settlement_time"], row["symbol"], row["bb_fr"], chart_idx, source="bybit")

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"DONE — {chart_idx} charts saved to {OUT}/")
print(f"Elapsed: {elapsed:.1f}s")
print(f"{'='*80}")
