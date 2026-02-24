#!/usr/bin/env python3
"""
Research: Funding Rate Edge Analysis (Binance vs Bybit)

With 2 days of symbol=ALL data we have ~6 funding settlements.
Questions:
1. What are the extreme funding rate values?
2. How do they compare between Binance and Bybit?
3. Is there a funding rate spread between exchanges?
4. What was the bid/ask spread during extreme funding events?
5. Is there an exploitable edge?

Memory-efficient: downsample to 1-minute resolution before merging.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"
t0 = time.time()


def load_and_downsample(path, cols, agg_dict, ts_col="ts", freq="1min"):
    """Load parquet, keep only needed cols, downsample to freq."""
    print(f"  Loading {path}...")
    df = pd.read_parquet(path, columns=cols)
    n_raw = len(df)
    n_sym = df["symbol"].nunique()
    print(f"    {n_raw:,} rows, {n_sym} symbols")
    print(f"    Time: {df[ts_col].min()} → {df[ts_col].max()}")

    df["ts_1m"] = df[ts_col].dt.floor(freq)
    df = df.groupby(["ts_1m", "symbol"]).agg(**agg_dict).reset_index()
    print(f"    Downsampled to {len(df):,} rows ({freq})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & DOWNSAMPLE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("LOADING & DOWNSAMPLING DATA (1-minute resolution)")
print("=" * 80)

# Binance funding rate
bn_fr = load_and_downsample(
    DATA / "binance" / "fundingRate.parquet",
    cols=["ts", "symbol", "lastFundingRate", "markPrice", "indexPrice", "nextFundingTime"],
    agg_dict=dict(
        bn_fundingRate=("lastFundingRate", "last"),
        bn_markPrice=("markPrice", "last"),
        bn_indexPrice=("indexPrice", "last"),
        bn_nextFunding=("nextFundingTime", "last"),
    ),
)
gc.collect()

# Binance ticker
bn_tk = load_and_downsample(
    DATA / "binance" / "ticker.parquet",
    cols=["ts", "symbol", "lastPrice", "quoteVolume"],
    agg_dict=dict(
        bn_lastPrice=("lastPrice", "last"),
        bn_volume=("quoteVolume", "last"),
    ),
)
gc.collect()

# Bybit ticker
bb_tk = load_and_downsample(
    DATA / "bybit" / "ticker.parquet",
    cols=["ts", "symbol", "fundingRate", "markPrice", "indexPrice",
          "lastPrice", "bid1Price", "ask1Price", "openInterestValue",
          "volume24h", "nextFundingTime", "fundingIntervalHour"],
    agg_dict=dict(
        bb_fundingRate=("fundingRate", "last"),
        bb_markPrice=("markPrice", "last"),
        bb_indexPrice=("indexPrice", "last"),
        bb_lastPrice=("lastPrice", "last"),
        bb_bid1=("bid1Price", "last"),
        bb_ask1=("ask1Price", "last"),
        bb_openInterest=("openInterestValue", "last"),
        bb_volume24h=("volume24h", "last"),
        bb_nextFunding=("nextFundingTime", "last"),
        bb_fundingInterval=("fundingIntervalHour", "last"),
    ),
)
gc.collect()

print(f"\nLoaded & downsampled in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MATCH SYMBOLS BETWEEN EXCHANGES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MATCHING SYMBOLS BETWEEN EXCHANGES")
print("=" * 80)

bn_symbols = set(bn_fr["symbol"].unique())
bb_symbols = set(bb_tk["symbol"].unique())
common = sorted(bn_symbols & bb_symbols)

print(f"Binance symbols: {len(bn_symbols)}")
print(f"Bybit symbols:   {len(bb_symbols)}")
print(f"Common:           {len(common)}")
print(f"Binance only:     {len(bn_symbols - bb_symbols)}")
print(f"Bybit only:       {len(bb_symbols - bn_symbols)}")

# Filter to common
bn_fr = bn_fr[bn_fr["symbol"].isin(common)].copy()
bn_tk = bn_tk[bn_tk["symbol"].isin(common)].copy()
bb_tk = bb_tk[bb_tk["symbol"].isin(common)].copy()

print(f"Filtered to {len(common)} common symbols")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUNDING RATE OVERVIEW — BINANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BINANCE FUNDING RATE OVERVIEW")
print("=" * 80)

bn_fr["nextFundingTime_dt"] = pd.to_datetime(bn_fr["bn_nextFunding"], unit="ms", utc=True)
settlement_times_bn = sorted(bn_fr["nextFundingTime_dt"].unique())
print(f"Unique nextFundingTime values: {len(settlement_times_bn)}")
for st in settlement_times_bn[:20]:
    print(f"  {st}")

fr_stats = bn_fr.groupby("symbol")["bn_fundingRate"].agg(["mean", "std", "min", "max", "count"])
fr_stats.columns = ["mean", "std", "min", "max", "count"]

print(f"\nTop 20 HIGHEST max funding rate (Binance):")
print(fr_stats.sort_values("max", ascending=False).head(20).to_string())

print(f"\nTop 20 LOWEST min funding rate (Binance, most negative):")
print(fr_stats.sort_values("min").head(20).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FUNDING RATE OVERVIEW — BYBIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BYBIT FUNDING RATE OVERVIEW")
print("=" * 80)

print("Bybit fundingIntervalHour distribution:")
print(bb_tk.groupby("bb_fundingInterval")["symbol"].nunique().to_string())

fr_stats_bb = bb_tk.groupby("symbol")["bb_fundingRate"].agg(["mean", "std", "min", "max", "count"])
fr_stats_bb.columns = ["mean", "std", "min", "max", "count"]

print(f"\nTop 20 HIGHEST max funding rate (Bybit):")
print(fr_stats_bb.sort_values("max", ascending=False).head(20).to_string())

print(f"\nTop 20 LOWEST min funding rate (Bybit, most negative):")
print(fr_stats_bb.sort_values("min").head(20).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MERGE & CROSS-EXCHANGE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CROSS-EXCHANGE FUNDING RATE COMPARISON")
print("=" * 80)

merged = bn_fr.merge(bb_tk, on=["ts_1m", "symbol"], how="inner")
merged = merged.merge(bn_tk, on=["ts_1m", "symbol"], how="left")
del bn_fr, bn_tk, bb_tk
gc.collect()

print(f"Merged dataset: {len(merged):,} rows, {merged['symbol'].nunique()} symbols")
print(f"Time range: {merged['ts_1m'].min()} → {merged['ts_1m'].max()}")

# Compute derived columns
merged["fr_spread"] = merged["bn_fundingRate"] - merged["bb_fundingRate"]
merged["fr_spread_abs"] = merged["fr_spread"].abs()
merged["price_spread_pct"] = (merged["bn_lastPrice"] - merged["bb_lastPrice"]) / merged["bb_lastPrice"] * 100
merged["bb_spread_bps"] = (merged["bb_ask1"] - merged["bb_bid1"]) / merged["bb_bid1"] * 10000

print(f"\nFunding Rate Spread (Binance - Bybit) stats:")
print(merged["fr_spread"].describe().to_string())

print(f"\nAbsolute FR Spread distribution:")
for pct in [50, 75, 90, 95, 99, 99.5, 99.9]:
    val = merged["fr_spread_abs"].quantile(pct / 100)
    print(f"  p{pct:5.1f}: {val:.6f} ({val*100:.4f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXTREME FUNDING RATES — DETAILED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXTREME FUNDING RATES — TOP EVENTS")
print("=" * 80)

# Find snapshots with extreme funding rates on either exchange
# Define extreme as |FR| > 0.001 (0.1%)
extreme_bn = merged[merged["bn_fundingRate"].abs() > 0.001].copy()
extreme_bb = merged[merged["bb_fundingRate"].abs() > 0.001].copy()
extreme_any = merged[(merged["bn_fundingRate"].abs() > 0.001) | (merged["bb_fundingRate"].abs() > 0.001)].copy()

print(f"Snapshots with |Binance FR| > 0.1%: {len(extreme_bn):,} ({len(extreme_bn)/len(merged)*100:.2f}%)")
print(f"Snapshots with |Bybit FR| > 0.1%:   {len(extreme_bb):,} ({len(extreme_bb)/len(merged)*100:.2f}%)")
print(f"Snapshots with either > 0.1%:        {len(extreme_any):,}")

# Top extreme events by Binance FR
print("\n--- Top 30 most extreme Binance funding rates ---")
cols_show = ["ts_1m", "symbol", "bn_fundingRate", "bb_fundingRate", "fr_spread",
             "bn_lastPrice", "bb_lastPrice", "price_spread_pct", "bb_spread_bps",
             "bb_openInterest", "bb_fundingInterval"]
top_bn = merged.nlargest(30, "bn_fundingRate")[cols_show]
print(top_bn.to_string(index=False))

print("\n--- Top 30 most negative Binance funding rates ---")
bot_bn = merged.nsmallest(30, "bn_fundingRate")[cols_show]
print(bot_bn.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. FUNDING RATE SPREAD — EDGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FUNDING RATE SPREAD — EDGE ANALYSIS")
print("=" * 80)

# Group by symbol: which coins have the biggest persistent FR spread?
symbol_fr = merged.groupby("symbol").agg(
    bn_fr_mean=("bn_fundingRate", "mean"),
    bb_fr_mean=("bb_fundingRate", "mean"),
    fr_spread_mean=("fr_spread", "mean"),
    fr_spread_std=("fr_spread", "std"),
    fr_spread_abs_mean=("fr_spread_abs", "mean"),
    price_spread_mean=("price_spread_pct", "mean"),
    price_spread_std=("price_spread_pct", "std"),
    bb_oi_mean=("bb_openInterest", "mean"),
    bb_vol_mean=("bb_volume24h", "mean"),
    snapshots=("ts_1m", "count"),
).reset_index()

# Annualized FR (assuming 3 settlements/day for 8h, but varies)
# Just show raw for now
symbol_fr["fr_spread_sharpe"] = symbol_fr["fr_spread_mean"] / symbol_fr["fr_spread_std"].replace(0, np.nan)

print("\nTop 20 coins by LARGEST absolute mean FR spread (Binance - Bybit):")
top_spread = symbol_fr.nlargest(20, "fr_spread_abs_mean")
print(top_spread[["symbol", "bn_fr_mean", "bb_fr_mean", "fr_spread_mean", "fr_spread_std",
                   "fr_spread_abs_mean", "price_spread_mean", "bb_oi_mean", "snapshots"]].to_string(index=False))

print("\nTop 20 coins by most POSITIVE mean FR spread (Binance higher):")
top_pos = symbol_fr.nlargest(20, "fr_spread_mean")
print(top_pos[["symbol", "bn_fr_mean", "bb_fr_mean", "fr_spread_mean", "fr_spread_std",
               "fr_spread_sharpe", "bb_oi_mean"]].to_string(index=False))

print("\nTop 20 coins by most NEGATIVE mean FR spread (Bybit higher):")
top_neg = symbol_fr.nsmallest(20, "fr_spread_mean")
print(top_neg[["symbol", "bn_fr_mean", "bb_fr_mean", "fr_spread_mean", "fr_spread_std",
               "fr_spread_sharpe", "bb_oi_mean"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SETTLEMENT-TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SETTLEMENT-TIME ANALYSIS")
print("=" * 80)

# Bybit has variable funding intervals (1h, 2h, 4h, 8h)
# Binance is mostly 8h (00:00, 08:00, 16:00 UTC)
# Let's look at what happens around settlement times

# For Binance: settlements at 00:00, 08:00, 16:00 UTC
bn_settlement_hours = [0, 8, 16]

# Near Binance settlement: within 10 min
merged["hour"] = merged["ts_1m"].dt.hour
merged["minute"] = merged["ts_1m"].dt.minute
merged["near_bn_settlement"] = False
for h in bn_settlement_hours:
    mask = (merged["hour"] == h) & (merged["minute"] <= 10)
    mask |= (merged["hour"] == (h - 1) % 24) & (merged["minute"] >= 50)
    merged.loc[mask, "near_bn_settlement"] = True

print(f"Snapshots near Binance settlement (±10min): {merged['near_bn_settlement'].sum():,}")
print(f"Snapshots NOT near settlement: {(~merged['near_bn_settlement']).sum():,}")

# Compare FR spreads near vs far from settlement
for label, mask in [("Near settlement", merged["near_bn_settlement"]),
                     ("Far from settlement", ~merged["near_bn_settlement"])]:
    sub = merged[mask]
    print(f"\n{label} ({len(sub):,} snapshots):")
    print(f"  Mean |FR spread|:     {sub['fr_spread_abs'].mean():.6f}")
    print(f"  Mean FR spread:       {sub['fr_spread'].mean():.6f}")
    print(f"  Std FR spread:        {sub['fr_spread'].std():.6f}")
    print(f"  Mean |price spread|:  {sub['price_spread_pct'].abs().mean():.4f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FUNDING INTERVAL MISMATCH — THE REAL EDGE?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FUNDING INTERVAL MISMATCH ANALYSIS")
print("=" * 80)

# Bybit has 1h, 2h, 4h, 8h intervals; Binance is mostly 8h
# Coins with 1h Bybit funding settle 8x more often than Binance
# This creates a structural difference

interval_analysis = merged.groupby("bb_fundingInterval").agg(
    symbols=("symbol", "nunique"),
    snapshots=("ts_1m", "count"),
    bn_fr_mean=("bn_fundingRate", "mean"),
    bb_fr_mean=("bb_fundingRate", "mean"),
    fr_spread_mean=("fr_spread", "mean"),
    fr_spread_abs_mean=("fr_spread_abs", "mean"),
    fr_spread_std=("fr_spread", "std"),
).reset_index()

print("FR spread by Bybit funding interval:")
print(interval_analysis.to_string(index=False))

# For 1h Bybit coins: annualized FR collection rate
# If Bybit FR = 0.01% per hour, that's 0.01% * 24 * 365 = 87.6% annualized
# vs Binance 8h: 0.01% * 3 * 365 = 10.95% annualized
print("\n--- Annualized FR comparison for 1h-interval Bybit coins ---")
bb_1h_coins = merged[merged["bb_fundingInterval"] == "1"].groupby("symbol").agg(
    bn_fr_mean=("bn_fundingRate", "mean"),
    bb_fr_mean=("bb_fundingRate", "mean"),
    fr_spread_mean=("fr_spread", "mean"),
    bb_oi=("bb_openInterest", "mean"),
).reset_index()

# Binance settles 3x/day, Bybit settles 24x/day for 1h coins
bb_1h_coins["bn_annual_pct"] = bb_1h_coins["bn_fr_mean"] * 3 * 365 * 100
bb_1h_coins["bb_annual_pct"] = bb_1h_coins["bb_fr_mean"] * 24 * 365 * 100
bb_1h_coins["annual_spread_pct"] = bb_1h_coins["bn_annual_pct"] - bb_1h_coins["bb_annual_pct"]

print(f"\n1h-interval coins ({len(bb_1h_coins)}):")
print("Top 20 by absolute annual spread:")
top_annual = bb_1h_coins.reindex(bb_1h_coins["annual_spread_pct"].abs().sort_values(ascending=False).index).head(20)
print(top_annual[["symbol", "bn_fr_mean", "bb_fr_mean", "bn_annual_pct", "bb_annual_pct",
                   "annual_spread_pct", "bb_oi"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. PRICE SPREAD DURING EXTREME FUNDING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PRICE SPREAD DURING EXTREME FUNDING")
print("=" * 80)

# Bucket by funding rate magnitude
merged["bn_fr_bucket"] = pd.cut(
    merged["bn_fundingRate"].abs(),
    bins=[0, 0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01, 1],
    labels=["<1bp", "1-3bp", "3-5bp", "5-10bp", "10-50bp", "50-100bp", ">100bp"]
)

price_by_fr = merged.groupby("bn_fr_bucket", observed=True).agg(
    count=("ts_1m", "count"),
    symbols=("symbol", "nunique"),
    mean_price_spread=("price_spread_pct", lambda x: x.abs().mean()),
    std_price_spread=("price_spread_pct", "std"),
    mean_fr_spread=("fr_spread_abs", "mean"),
).reset_index()

print("Price spread between exchanges by Binance FR magnitude:")
print(price_by_fr.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 11. BYBIT BID-ASK SPREAD DURING EXTREME FUNDING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BYBIT BID-ASK SPREAD DURING EXTREME FUNDING")
print("=" * 80)

spread_by_fr = merged.groupby("bn_fr_bucket", observed=True).agg(
    count=("ts_1m", "count"),
    mean_bb_spread_bps=("bb_spread_bps", "mean"),
    median_bb_spread_bps=("bb_spread_bps", "median"),
    p95_bb_spread_bps=("bb_spread_bps", lambda x: x.quantile(0.95)),
).reset_index()

print("Bybit bid-ask spread (bps) by Binance FR magnitude:")
print(spread_by_fr.to_string(index=False))

# Also look at spread for specific extreme coins
print("\n--- Bybit spread for coins with |FR| > 50bp ---")
extreme_coins = merged[merged["bn_fundingRate"].abs() > 0.005]
if len(extreme_coins) > 0:
    ec_stats = extreme_coins.groupby("symbol").agg(
        snapshots=("ts_1m", "count"),
        bn_fr_mean=("bn_fundingRate", "mean"),
        bb_fr_mean=("bb_fundingRate", "mean"),
        bb_spread_bps_mean=("bb_spread_bps", "mean"),
        price_spread_pct=("price_spread_pct", lambda x: x.abs().mean()),
        bb_oi=("bb_openInterest", "mean"),
    ).sort_values("bn_fr_mean", key=abs, ascending=False)
    print(ec_stats.to_string())
else:
    print("No coins with |FR| > 50bp found")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY — KEY FINDINGS")
print("=" * 80)

total_time = time.time() - t0
print(f"\nAnalysis completed in {total_time:.1f}s")
print(f"Data: {merged['ts_1m'].min()} → {merged['ts_1m'].max()}")
print(f"Common symbols: {merged['symbol'].nunique()}")
print(f"Total matched snapshots: {len(merged):,}")

# Key stats
print(f"\nFunding Rate Spread (Binance - Bybit):")
print(f"  Mean:   {merged['fr_spread'].mean():.6f} ({merged['fr_spread'].mean()*10000:.2f} bp)")
print(f"  Std:    {merged['fr_spread'].std():.6f}")
print(f"  |Mean|: {merged['fr_spread_abs'].mean():.6f} ({merged['fr_spread_abs'].mean()*10000:.2f} bp)")

print(f"\nPrice Spread (Binance - Bybit):")
print(f"  Mean:   {merged['price_spread_pct'].mean():.4f}%")
print(f"  |Mean|: {merged['price_spread_pct'].abs().mean():.4f}%")

print(f"\nBybit Bid-Ask Spread:")
print(f"  Mean:   {merged['bb_spread_bps'].mean():.2f} bps")
print(f"  Median: {merged['bb_spread_bps'].median():.2f} bps")
