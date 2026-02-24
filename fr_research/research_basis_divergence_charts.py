#!/usr/bin/env python3
"""
Charts & deep analysis for cross-exchange FR spread arbitrage.

1. FR spread evolution around entry events (how does the spread move?)
2. Concurrent open positions over time
3. Bid-ask spread / slippage analysis from real-time tick data
4. P&L distribution and drawdown
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_all"
OUT_DIR = REPO_ROOT / "fr_research" / "charts"
OUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

print("=" * 80)
print("CHARTS: Cross-Exchange FR Spread Arb — Deep Analysis")
print("=" * 80)

# ── Load data ──────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
bn_fr = pd.read_parquet(DATA_DIR / "historical_fr" / "binance_fr_history.parquet")
bb_fr = pd.read_parquet(DATA_DIR / "historical_fr" / "bybit_fr_history.parquet")
bn_fr['hour'] = bn_fr['fundingTime'].dt.floor('h')
bb_fr['hour'] = bb_fr['fundingTime'].dt.floor('h')

merged = pd.merge(
    bn_fr[['symbol', 'hour', 'fundingRate']].rename(columns={'fundingRate': 'fr_bn'}),
    bb_fr[['symbol', 'hour', 'fundingRate']].rename(columns={'fundingRate': 'fr_bb'}),
    on=['symbol', 'hour'], how='inner'
)
merged['fr_bn_bps'] = merged['fr_bn'] * 10000
merged['fr_bb_bps'] = merged['fr_bb'] * 10000
merged['fr_spread'] = merged['fr_bn_bps'] - merged['fr_bb_bps']
merged['abs_spread'] = merged['fr_spread'].abs()
merged = merged.sort_values(['symbol', 'hour']).reset_index(drop=True)

data_days = (merged.hour.max() - merged.hour.min()).total_seconds() / 86400
print(f"  {len(merged):,} settlements, {merged.symbol.nunique()} symbols, {data_days:.0f} days")

# Build forward + backward looking columns
MAX_FWD = 12
MAX_BWD = 4
grp = merged.groupby('symbol')
for t in range(1, MAX_FWD + 1):
    merged[f'spread_t{t}'] = grp['fr_spread'].shift(-t)
    merged[f'abs_spread_t{t}'] = grp['abs_spread'].shift(-t)
for t in range(1, MAX_BWD + 1):
    merged[f'spread_tm{t}'] = grp['fr_spread'].shift(t)
    merged[f'abs_spread_tm{t}'] = grp['abs_spread'].shift(t)

merged.rename(columns={'fr_spread': 'spread_t0', 'abs_spread': 'abs_spread_t0'}, inplace=True)

# ══════════════════════════════════════════════════════════════════════════
# CHART 1: FR spread evolution around entry events
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Chart 1: FR spread evolution around entry events...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('FR Spread Evolution Around Entry Events\n(Short higher-FR exchange, Long lower-FR exchange)',
             fontsize=14, fontweight='bold')

for ax_idx, (min_spread, label) in enumerate([
    (10, '|spread| ≥ 10 bps'), (20, '|spread| ≥ 20 bps'),
    (30, '|spread| ≥ 30 bps'), (50, '|spread| ≥ 50 bps')
]):
    ax = axes[ax_idx // 2][ax_idx % 2]
    events = merged[merged['abs_spread_t0'] >= min_spread].copy()
    
    if len(events) < 10:
        ax.text(0.5, 0.5, f'N={len(events)} (too few)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label)
        continue
    
    # Collect spread paths: normalize so entry spread is always positive
    # (i.e., we always "short the higher side")
    sign = np.where(events['spread_t0'] > 0, 1, -1)
    
    times = list(range(-MAX_BWD, MAX_FWD + 1))
    paths = np.full((len(events), len(times)), np.nan)
    
    for i, t in enumerate(times):
        if t == 0:
            col = 'spread_t0'
        elif t > 0:
            col = f'spread_t{t}'
        else:
            col = f'spread_tm{abs(t)}'
        
        if col in events.columns:
            paths[:, i] = events[col].values * sign
    
    # Plot percentiles
    p25 = np.nanpercentile(paths, 25, axis=0)
    p50 = np.nanpercentile(paths, 50, axis=0)
    p75 = np.nanpercentile(paths, 75, axis=0)
    mean = np.nanmean(paths, axis=0)
    
    ax.fill_between(times, p25, p75, alpha=0.2, color=COLORS[0], label='P25–P75')
    ax.plot(times, p50, '-', color=COLORS[0], linewidth=2, label='Median')
    ax.plot(times, mean, '--', color=COLORS[1], linewidth=2, label='Mean')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Entry')
    
    n_events = len(events)
    per_day = n_events / data_days
    ax.set_title(f'{label} (N={n_events:,}, {per_day:.1f}/day)', fontsize=11)
    ax.set_xlabel('Settlement periods from entry')
    ax.set_ylabel('Normalized spread (bps)\n(positive = in our favor)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'spread_evolution.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'spread_evolution.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# CHART 2: Concurrent open positions over time
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Chart 2: Concurrent open positions over time...")

ENTRY_THRESH = 10  # bps
EXIT_THRESH = 5    # bps

# Simulate trade-by-trade
events_sorted = merged[merged['abs_spread_t0'] >= ENTRY_THRESH].copy()
events_sorted = events_sorted.sort_values('hour')

# For each event, find exit time
# Exit when |spread| drops below EXIT_THRESH
exit_hours = []
for _, row in events_sorted.iterrows():
    exit_t = None
    for t in range(1, MAX_FWD + 1):
        col = f'abs_spread_t{t}'
        if col in row and pd.notna(row[col]) and row[col] < EXIT_THRESH:
            exit_t = t
            break
    if exit_t is None:
        exit_t = MAX_FWD  # forced exit
    exit_hours.append(exit_t)

events_sorted['exit_periods'] = exit_hours

# Build time-series of open position count
# Create hourly timeline
all_hours = pd.date_range(merged['hour'].min(), merged['hour'].max(), freq='h')
pos_count = pd.Series(0, index=all_hours, name='positions')

# For each symbol, track settlements (variable frequency — 1h, 4h, 8h)
# Use a simpler approach: for each event, mark open from entry to entry + exit_periods
# But we need to map periods back to real hours per symbol
# Simpler: count events that are "active" at each unique hour

print(f"  Simulating {len(events_sorted):,} trades with entry≥{ENTRY_THRESH}bps, exit<{EXIT_THRESH}bps...")

# For speed, just track per hour
open_positions = {}  # hour → count
for _, row in events_sorted.iterrows():
    sym = row['symbol']
    entry_hour = row['hour']
    hold = row['exit_periods']
    
    # Get the actual settlement hours for this symbol after entry
    sym_hours = merged[(merged['symbol'] == sym) & (merged['hour'] >= entry_hour)].head(hold + 1)['hour'].values
    for h in sym_hours:
        h_ts = pd.Timestamp(h)
        open_positions[h_ts] = open_positions.get(h_ts, 0) + 1

pos_series = pd.Series(open_positions).sort_index()

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle(f'Position Management Analysis (entry ≥{ENTRY_THRESH} bps, exit <{EXIT_THRESH} bps)',
             fontsize=14, fontweight='bold')

# 2a: Positions over time
ax = axes[0]
ax.fill_between(pos_series.index, pos_series.values, alpha=0.4, color=COLORS[0])
ax.plot(pos_series.index, pos_series.values, linewidth=0.5, color=COLORS[0])
ax.set_ylabel('Concurrent positions')
ax.set_title(f'Open Positions Over Time')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Add stats
avg_pos = pos_series.mean()
max_pos = pos_series.max()
ax.axhline(y=avg_pos, color=COLORS[1], linestyle='--', label=f'Avg: {avg_pos:.1f}')
ax.axhline(y=max_pos, color=COLORS[1], linestyle=':', alpha=0.5, label=f'Max: {max_pos:.0f}')
ax.legend()

# 2b: Distribution of concurrent positions
ax = axes[1]
ax.hist(pos_series.values, bins=range(0, int(max_pos) + 2), color=COLORS[0], alpha=0.7, edgecolor='white')
ax.set_xlabel('Concurrent positions')
ax.set_ylabel('Hours')
ax.set_title(f'Distribution of Concurrent Positions (avg={avg_pos:.1f}, max={max_pos:.0f})')
ax.axvline(x=avg_pos, color=COLORS[1], linestyle='--', label=f'Mean: {avg_pos:.1f}')
ax.legend()

# 2c: Hold period distribution
ax = axes[2]
ax.hist(events_sorted['exit_periods'].values, bins=range(0, MAX_FWD + 2), 
        color=COLORS[2], alpha=0.7, edgecolor='white')
ax.set_xlabel('Hold periods (settlements)')
ax.set_ylabel('Trade count')
avg_hold = events_sorted['exit_periods'].mean()
ax.set_title(f'Hold Period Distribution (avg={avg_hold:.1f} periods, N={len(events_sorted):,})')
ax.axvline(x=avg_hold, color=COLORS[1], linestyle='--', label=f'Mean: {avg_hold:.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'position_management.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'position_management.png'}")
plt.close()

print(f"\n  Position stats:")
print(f"    Avg concurrent positions: {avg_pos:.1f}")
print(f"    Max concurrent positions: {max_pos:.0f}")
print(f"    P95 concurrent positions: {pos_series.quantile(0.95):.0f}")
print(f"    Avg hold period: {avg_hold:.1f} settlements")
print(f"    Capital needed (at max): ${max_pos * 10000:,.0f}/leg")

# ══════════════════════════════════════════════════════════════════════════
# CHART 3: Bid-Ask spread analysis from real-time tick data
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Chart 3: Bid-ask spread & slippage analysis (real-time tick data)...")

# Bybit has bid1/ask1 data
bb_tick = pd.read_parquet(DATA_DIR / "bybit" / "ticker.parquet",
    columns=['ts', 'symbol', 'lastPrice', 'bid1Price', 'ask1Price', 
             'bid1Size', 'ask1Size', 'fundingRate', 'indexPrice', 'markPrice'])

# Binance ticker has lastPrice but no bid/ask — use it for mark price comparison
bn_tick = pd.read_parquet(DATA_DIR / "binance" / "fundingRate.parquet",
    columns=['ts', 'symbol', 'markPrice', 'indexPrice', 'lastFundingRate'])

print(f"  Bybit tick: {len(bb_tick):,}, Binance tick: {len(bn_tick):,}")

# Compute Bybit bid-ask spread
bb_tick['ba_spread_bps'] = (bb_tick['ask1Price'] - bb_tick['bid1Price']) / bb_tick['bid1Price'] * 10000
bb_tick['ba_spread_bps'] = bb_tick['ba_spread_bps'].clip(0, 100)  # clip outliers

# Get symbols that appear in our FR spread arb
# Filter to symbols with ≥10bps FR spread events
arb_symbols = events_sorted['symbol'].unique()
bb_arb = bb_tick[bb_tick['symbol'].isin(arb_symbols)]
bb_all = bb_tick

print(f"  Arb symbols in tick data: {bb_arb.symbol.nunique()}")

# Per-symbol bid-ask stats
ba_stats = bb_arb.groupby('symbol').agg(
    mean_ba=('ba_spread_bps', 'mean'),
    median_ba=('ba_spread_bps', 'median'),
    p95_ba=('ba_spread_bps', lambda x: x.quantile(0.95)),
    mean_bid_size=('bid1Size', 'mean'),
    mean_ask_size=('ask1Size', 'mean'),
    n_ticks=('ts', 'size'),
).sort_values('mean_ba', ascending=False)

# Also get last price for notional calculation
last_prices = bb_arb.groupby('symbol')['lastPrice'].last()
ba_stats = ba_stats.join(last_prices)
ba_stats['bid_notional_usd'] = ba_stats['mean_bid_size'] * ba_stats['lastPrice']
ba_stats['ask_notional_usd'] = ba_stats['mean_ask_size'] * ba_stats['lastPrice']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bid-Ask Spread & Liquidity Analysis (Bybit, arb-eligible symbols)',
             fontsize=14, fontweight='bold')

# 3a: Distribution of mean bid-ask spread per symbol
ax = axes[0][0]
ax.hist(ba_stats['mean_ba'].values, bins=50, color=COLORS[0], alpha=0.7, edgecolor='white')
ax.set_xlabel('Mean bid-ask spread (bps)')
ax.set_ylabel('Symbols')
overall_mean = ba_stats['mean_ba'].mean()
ax.axvline(x=overall_mean, color=COLORS[1], linestyle='--', label=f'Mean: {overall_mean:.1f} bps')
ax.set_title(f'Bid-Ask Spread Distribution ({len(ba_stats)} symbols)')
ax.legend()

# 3b: Bid-ask spread vs avg FR spread for each symbol
ax = axes[0][1]
# Get avg FR spread per symbol from historical data
fr_spread_per_sym = merged.groupby('symbol')['abs_spread_t0'].mean()
plot_df = ba_stats.join(fr_spread_per_sym, how='inner')
ax.scatter(plot_df['abs_spread_t0'], plot_df['mean_ba'], alpha=0.5, s=20, color=COLORS[0])
ax.set_xlabel('Avg |FR spread| (bps)')
ax.set_ylabel('Avg bid-ask spread (bps)')
ax.set_title('FR Spread vs Bid-Ask Spread')
# Add break-even line: if BA spread > FR spread, we can't profit
max_val = max(plot_df['abs_spread_t0'].max(), plot_df['mean_ba'].max())
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Break-even (BA = FR spread)')
ax.legend(fontsize=8)

# 3c: Top-of-book depth (notional) distribution
ax = axes[1][0]
# Combined bid+ask notional
ba_stats['total_book_usd'] = ba_stats['bid_notional_usd'] + ba_stats['ask_notional_usd']
data = ba_stats['total_book_usd'].clip(0, 500_000)
ax.hist(data.values, bins=50, color=COLORS[2], alpha=0.7, edgecolor='white')
ax.set_xlabel('Top-of-book depth (bid+ask, USD)')
ax.set_ylabel('Symbols')
med_depth = ba_stats['total_book_usd'].median()
ax.axvline(x=med_depth, color=COLORS[1], linestyle='--', label=f'Median: ${med_depth:,.0f}')
ax.set_title('Top-of-Book Depth')
ax.legend()

# 3d: Slippage estimate: for $10K order, how much slippage?
ax = axes[1][1]
# Simple estimate: if order > top-of-book, we cross the spread
# Slippage ≈ (order_size / book_depth) * spread as a rough model
# More accurately: for $10K, if depth < $10K, we eat into the book
order_size = 10_000
ba_stats['fill_ratio'] = (order_size / ba_stats['total_book_usd']).clip(0, 10)
ba_stats['est_slippage_bps'] = ba_stats['mean_ba'] * (1 + ba_stats['fill_ratio'])

data = ba_stats['est_slippage_bps'].clip(0, 20)
ax.hist(data.values, bins=50, color=COLORS[3], alpha=0.7, edgecolor='white')
ax.set_xlabel(f'Estimated slippage for ${order_size/1000:.0f}K order (bps)')
ax.set_ylabel('Symbols')
med_slip = ba_stats['est_slippage_bps'].median()
ax.axvline(x=med_slip, color=COLORS[1], linestyle='--', label=f'Median: {med_slip:.1f} bps')
ax.set_title(f'Estimated Slippage (${order_size/1000:.0f}K taker order)')
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'bid_ask_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'bid_ask_analysis.png'}")
plt.close()

# Print summary stats
print(f"\n  Bid-ask spread stats (arb-eligible symbols):")
print(f"    Mean BA spread: {ba_stats['mean_ba'].mean():.2f} bps")
print(f"    Median BA spread: {ba_stats['mean_ba'].median():.2f} bps")
print(f"    P95 BA spread: {ba_stats['mean_ba'].quantile(0.95):.2f} bps")
print(f"    Median top-of-book depth: ${ba_stats['total_book_usd'].median():,.0f}")
print(f"    Median slippage ($10K): {ba_stats['est_slippage_bps'].median():.2f} bps")
print(f"    Symbols with BA < 5 bps: {(ba_stats['mean_ba'] < 5).sum()}")
print(f"    Symbols with depth > $10K: {(ba_stats['total_book_usd'] > 10_000).sum()}")

# Top 20 most liquid arb symbols
print(f"\n  Top 20 arb symbols by liquidity (low BA + high depth):")
ba_stats['liquidity_score'] = ba_stats['total_book_usd'] / (ba_stats['mean_ba'] + 0.1)
top_liquid = ba_stats.sort_values('liquidity_score', ascending=False).head(20)
print(f"  {'Symbol':<18} {'BA(bps)':>8} {'Depth($)':>10} {'Slippage':>10} {'Avg FR spr':>11}")
print(f"  {'-'*57}")
for sym, row in top_liquid.iterrows():
    fr_spr = fr_spread_per_sym.get(sym, 0)
    print(f"  {sym:<18} {row['mean_ba']:>8.2f} {row['total_book_usd']:>10,.0f} "
          f"{row['est_slippage_bps']:>10.2f} {fr_spr:>11.2f}")

# ══════════════════════════════════════════════════════════════════════════
# CHART 4: P&L distribution and equity curve
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Chart 4: P&L distribution and equity curve...")

# Rebuild trades with proper P&L
sign_vec = np.where(merged['spread_t0'] > 0, +1, -1)

cum_pnl = np.zeros(len(merged))
hold = np.zeros(len(merged), dtype=int)
exited = np.zeros(len(merged), dtype=bool)

cum_pnl += sign_vec * merged['fr_bn_bps'].values + (-sign_vec) * merged['fr_bb_bps'].values
hold += 1

for t in range(1, MAX_FWD + 1):
    bn_col = f'fr_bn_t{t}' if f'fr_bn_t{t}' in merged.columns else None
    abs_col = f'abs_spread_t{t}'
    if bn_col is None or bn_col not in merged.columns:
        # Need to build these columns
        break
    if abs_col not in merged.columns:
        break
    has_data = merged[abs_col].notna().values
    spread_above = merged[abs_col].fillna(0).values >= EXIT_THRESH
    active = ~exited & has_data
    # Need fr columns — let me rebuild
    break

# Simpler: use the events_sorted df which already has exit_periods
# Rebuild P&L per trade from the merged data with shifts
print("  Building equity curve from historical trades...")

# Re-derive from the merged data with forward shifts
merged2 = merged.copy()
grp2 = merged2.groupby('symbol')
for t in range(1, MAX_FWD + 1):
    merged2[f'fr_bn_t{t}'] = grp2['fr_bn_bps'].shift(-t)
    merged2[f'fr_bb_t{t}'] = grp2['fr_bb_bps'].shift(-t)

# Filter to entry events
entry_events = merged2[merged2['abs_spread_t0'] >= ENTRY_THRESH].copy()
entry_events['bn_higher'] = entry_events['spread_t0'] > 0

sign_bn = np.where(entry_events['bn_higher'], +1, -1)
sign_bb = -sign_bn

n = len(entry_events)
cum_pnl = np.zeros(n)
hold_periods = np.zeros(n, dtype=int)
exited = np.zeros(n, dtype=bool)

cum_pnl += sign_bn * entry_events['fr_bn_bps'].values + sign_bb * entry_events['fr_bb_bps'].values
hold_periods += 1

for t in range(1, MAX_FWD + 1):
    bn_col = f'fr_bn_t{t}'
    abs_col = f'abs_spread_t{t}'
    if bn_col not in entry_events.columns or abs_col not in entry_events.columns:
        break
    has_data = entry_events[bn_col].notna().values
    spread_above = entry_events[abs_col].fillna(0).values >= EXIT_THRESH
    active = ~exited & has_data
    period_pnl = sign_bn * entry_events[bn_col].fillna(0).values + sign_bb * entry_events[f'fr_bb_t{t}'].fillna(0).values
    cum_pnl += np.where(active, period_pnl, 0)
    hold_periods += active.astype(int)
    exited |= (active & ~spread_above)

NOTIONAL = 10_000
RT_FEE = 8.0  # maker

trades = pd.DataFrame({
    'hour': entry_events['hour'].values,
    'symbol': entry_events['symbol'].values,
    'initial_spread': entry_events['abs_spread_t0'].values,
    'hold': hold_periods,
    'gross_bps': cum_pnl,
    'net_bps': cum_pnl - RT_FEE,
})
trades['net_dollar'] = trades['net_bps'] * NOTIONAL / 10000
trades = trades.sort_values('hour')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'P&L Analysis (entry ≥{ENTRY_THRESH} bps, maker fees {RT_FEE} bps, ${NOTIONAL/1000:.0f}K/leg)',
             fontsize=14, fontweight='bold')

# 4a: Cumulative equity curve
ax = axes[0][0]
cum_equity = trades['net_dollar'].cumsum()
ax.plot(trades['hour'].values, cum_equity.values, color=COLORS[0], linewidth=1)
ax.fill_between(trades['hour'].values, 0, cum_equity.values, alpha=0.2, color=COLORS[0])
ax.set_ylabel('Cumulative P&L ($)')
ax.set_title(f'Equity Curve (total: ${cum_equity.iloc[-1]:,.0f} over {data_days:.0f} days)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 4b: Per-trade P&L distribution
ax = axes[0][1]
ax.hist(trades['net_bps'].clip(-100, 500).values, bins=80, color=COLORS[0], alpha=0.7, edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.axvline(x=trades['net_bps'].mean(), color=COLORS[1], linestyle='--',
           label=f'Mean: {trades["net_bps"].mean():.1f} bps')
ax.axvline(x=trades['net_bps'].median(), color=COLORS[2], linestyle='--',
           label=f'Median: {trades["net_bps"].median():.1f} bps')
ax.set_xlabel('Net P&L per trade (bps)')
ax.set_ylabel('Count')
ax.set_title(f'Per-Trade P&L Distribution (N={len(trades):,})')
ax.legend()

# 4c: Daily P&L
ax = axes[1][0]
daily_pnl = trades.groupby(trades['hour'].dt.date)['net_dollar'].sum()
colors_daily = [COLORS[2] if v >= 0 else COLORS[1] for v in daily_pnl.values]
ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors_daily, alpha=0.7, width=1)
ax.set_xlabel('Day')
ax.set_ylabel('Daily P&L ($)')
avg_daily = daily_pnl.mean()
win_days = (daily_pnl > 0).sum()
total_days = len(daily_pnl)
ax.axhline(y=avg_daily, color='blue', linestyle='--', label=f'Avg: ${avg_daily:,.0f}/day')
ax.set_title(f'Daily P&L ({win_days}/{total_days} profitable days = {win_days/total_days*100:.0f}%)')
ax.legend()

# 4d: Drawdown
ax = axes[1][1]
cum_max = cum_equity.cummax()
drawdown = cum_equity - cum_max
ax.fill_between(trades['hour'].values, drawdown.values, 0, alpha=0.4, color=COLORS[1])
ax.plot(trades['hour'].values, drawdown.values, color=COLORS[1], linewidth=0.5)
max_dd = drawdown.min()
ax.set_ylabel('Drawdown ($)')
ax.set_title(f'Drawdown (max: ${max_dd:,.0f})')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(OUT_DIR / 'pnl_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'pnl_analysis.png'}")
plt.close()

# Print P&L summary
print(f"\n  P&L Summary (entry≥{ENTRY_THRESH}bps, maker {RT_FEE}bps, ${NOTIONAL/1000:.0f}K/leg):")
print(f"    Total trades: {len(trades):,}")
print(f"    Total P&L: ${cum_equity.iloc[-1]:,.0f}")
print(f"    Avg daily P&L: ${avg_daily:,.0f}")
print(f"    Win rate: {(trades['net_bps'] > 0).mean()*100:.1f}%")
print(f"    Profitable days: {win_days}/{total_days} ({win_days/total_days*100:.0f}%)")
print(f"    Max drawdown: ${max_dd:,.0f}")
print(f"    Sharpe (daily): {daily_pnl.mean() / daily_pnl.std() * np.sqrt(365):.2f}")

# ══════════════════════════════════════════════════════════════════════════
# CHART 5: Realistic slippage-adjusted P&L
# ══════════════════════════════════════════════════════════════════════════
print("\n[6] Chart 5: Slippage-adjusted P&L comparison...")

fig, ax = plt.subplots(1, 1, figsize=(16, 6))

# Build equity curves with different slippage assumptions
for slip_bps, color, label in [
    (0, COLORS[0], 'No slippage (maker only)'),
    (2, COLORS[2], '+2 bps slippage'),
    (5, COLORS[3], '+5 bps slippage'),
    (10, COLORS[1], '+10 bps slippage'),
]:
    total_fee = RT_FEE + slip_bps
    net = (trades['gross_bps'] - total_fee) * NOTIONAL / 10000
    cum = net.cumsum()
    ax.plot(trades['hour'].values, cum.values, color=color, linewidth=1.5, label=f'{label} → ${cum.iloc[-1]:,.0f}')

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Cumulative P&L ($)')
ax.set_title(f'Equity Curves with Different Slippage Assumptions (entry ≥{ENTRY_THRESH} bps, ${NOTIONAL/1000:.0f}K/leg)')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(OUT_DIR / 'slippage_sensitivity.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'slippage_sensitivity.png'}")
plt.close()

# Print slippage sensitivity
print(f"\n  Slippage sensitivity (200 days, ${NOTIONAL/1000:.0f}K/leg):")
for slip_bps in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
    total_fee = RT_FEE + slip_bps
    net = trades['gross_bps'] - total_fee
    total_dollar = (net * NOTIONAL / 10000).sum()
    per_day = total_dollar / data_days
    wr = (net > 0).mean() * 100
    print(f"    +{slip_bps:>2} bps slippage (total fee: {total_fee:.0f} bps): "
          f"${per_day:>8,.0f}/day, WR={wr:.0f}%, total=${total_dollar:>10,.0f}")

print("\n" + "=" * 80)
print("All charts saved to:", OUT_DIR)
print("=" * 80)
