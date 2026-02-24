#!/usr/bin/env python3
"""
Research: Cross-Exchange Futures Premium (Basis) Divergence Arbitrage

The TRADE:
  When futures premium (basis = markPrice - indexPrice) diverges between
  Binance and Bybit on the same symbol:
    - Short futures on exchange with HIGHER premium (overpriced futures)
    - Long futures on exchange with LOWER premium (cheaper futures)
    - Delta-neutral: both legs are futures on same underlying
    - Premiums MUST converge → profit when they do

This is NOT about funding rates. This is about the actual price gap
between futures and spot on each exchange.

Data: Real-time ticker streams (~5s resolution, 2 days, Feb 2026)
  - Bybit: markPrice, indexPrice, lastPrice, bid1Price, ask1Price
  - Binance: markPrice, indexPrice (from fundingRate stream)
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
print("RESEARCH: Cross-Exchange Futures Premium Divergence Arbitrage")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load and align tick data
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Loading tick data...")

bb = pd.read_parquet(DATA_DIR / "bybit" / "ticker.parquet",
    columns=['ts', 'symbol', 'lastPrice', 'markPrice', 'indexPrice',
             'bid1Price', 'ask1Price', 'bid1Size', 'ask1Size'])
bn = pd.read_parquet(DATA_DIR / "binance" / "fundingRate.parquet",
    columns=['ts', 'symbol', 'markPrice', 'indexPrice'])

print(f"  Bybit:   {len(bb):,} ticks, {bb.symbol.nunique()} symbols")
print(f"  Binance: {len(bn):,} ticks, {bn.symbol.nunique()} symbols")

# Compute basis (futures premium) on each exchange
bb['basis_bb_bps'] = (bb['markPrice'] - bb['indexPrice']) / bb['indexPrice'] * 10000
bn['basis_bn_bps'] = (bn['markPrice'] - bn['indexPrice']) / bn['indexPrice'] * 10000

# Resample to 1-minute bars for alignment
print("\n[2] Resampling to 1-minute bars...")
bb['minute'] = bb['ts'].dt.floor('1min')
bn['minute'] = bn['ts'].dt.floor('1min')

bb_1m = bb.groupby(['symbol', 'minute']).agg(
    basis_bb=('basis_bb_bps', 'mean'),
    last_bb=('lastPrice', 'last'),
    bid_bb=('bid1Price', 'last'),
    ask_bb=('ask1Price', 'last'),
    bid_size_bb=('bid1Size', 'mean'),
    ask_size_bb=('ask1Size', 'mean'),
    mark_bb=('markPrice', 'last'),
    index_bb=('indexPrice', 'last'),
).reset_index()

bn_1m = bn.groupby(['symbol', 'minute']).agg(
    basis_bn=('basis_bn_bps', 'mean'),
    mark_bn=('markPrice', 'last'),
    index_bn=('indexPrice', 'last'),
).reset_index()

# Merge
merged = pd.merge(bb_1m, bn_1m, on=['symbol', 'minute'], how='inner')
merged['basis_spread'] = merged['basis_bn'] - merged['basis_bb']  # positive = BN premium higher
merged['abs_basis_spread'] = merged['basis_spread'].abs()

# Also compute the actual futures price spread (what we'd actually trade)
# Price divergence between the two exchanges' futures
merged['price_spread_bps'] = (merged['mark_bn'] - merged['mark_bb']) / merged['mark_bb'] * 10000

common_syms = merged.symbol.nunique()
total_minutes = merged.minute.nunique()
hours = total_minutes / 60

print(f"  Merged: {len(merged):,} rows, {common_syms} common symbols, {hours:.1f} hours")

# ══════════════════════════════════════════════════════════════════════════
# 3. Basis spread distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Basis spread distribution (BN basis - BB basis)...")
print(f"  Mean:   {merged['basis_spread'].mean():.2f} bps")
print(f"  Std:    {merged['basis_spread'].std():.2f} bps")
print(f"  Median: {merged['basis_spread'].median():.2f} bps")
for pct in [90, 95, 99, 99.5, 99.9]:
    val = merged['abs_basis_spread'].quantile(pct / 100)
    print(f"  P{pct} |spread|: {val:.1f} bps")

# ══════════════════════════════════════════════════════════════════════════
# 4. Event frequency by threshold
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Events by |basis spread| threshold:")
print(f"  {'Threshold':<15} {'1-min bars':>10} {'Per hour':>10} {'% of total':>12}")
print(f"  {'-'*47}")
for thresh in [5, 10, 15, 20, 30, 50, 100]:
    n = (merged['abs_basis_spread'] >= thresh).sum()
    per_hour = n / hours
    pct = n / len(merged) * 100
    print(f"  |spread|≥{thresh:<3} bps {n:>10,} {per_hour:>10.1f} {pct:>11.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# 5. Per-symbol analysis — which symbols diverge most?
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Top 30 symbols by basis spread volatility:")
sym_stats = merged.groupby('symbol').agg(
    n=('minute', 'size'),
    mean_spread=('basis_spread', 'mean'),
    std_spread=('basis_spread', 'std'),
    max_abs_spread=('abs_basis_spread', 'max'),
    pct_gt10=('abs_basis_spread', lambda x: (x >= 10).mean()),
    pct_gt20=('abs_basis_spread', lambda x: (x >= 20).mean()),
    mean_bn_basis=('basis_bn', 'mean'),
    mean_bb_basis=('basis_bb', 'mean'),
    last_price=('last_bb', 'last'),
).sort_values('std_spread', ascending=False)

print(f"  {'Symbol':<16} {'N':>6} {'Mean':>7} {'Std':>7} {'Max|Spr|':>9} {'%>10':>6} {'%>20':>6} {'BN basis':>9} {'BB basis':>9}")
print(f"  {'-'*80}")
for sym, r in sym_stats.head(30).iterrows():
    print(f"  {sym:<16} {r['n']:>6.0f} {r['mean_spread']:>7.1f} {r['std_spread']:>7.1f} "
          f"{r['max_abs_spread']:>9.1f} {r['pct_gt10']*100:>5.0f}% {r['pct_gt20']*100:>5.0f}% "
          f"{r['mean_bn_basis']:>9.1f} {r['mean_bb_basis']:>9.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. CONVERGENCE — how quickly does the basis spread revert?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[6] CONVERGENCE ANALYSIS: How quickly does basis spread revert?")
print("=" * 80)

# For each symbol, compute autocorrelation of basis_spread at various lags
print("\n  Basis spread autocorrelation by lag:")
merged_sorted = merged.sort_values(['symbol', 'minute'])

lags_to_test = [1, 2, 5, 10, 15, 30, 60]
ac_results = []
for sym, grp in merged_sorted.groupby('symbol'):
    if len(grp) < 120:  # need at least 2 hours
        continue
    s = grp['basis_spread'].values
    row = {'symbol': sym, 'n': len(grp)}
    for lag in lags_to_test:
        if lag >= len(s):
            row[f'ac_{lag}m'] = np.nan
            continue
        c = np.corrcoef(s[:-lag], s[lag:])[0, 1]
        row[f'ac_{lag}m'] = c
    ac_results.append(row)

ac_df = pd.DataFrame(ac_results)
print(f"  Symbols analyzed: {len(ac_df)}")
for lag in lags_to_test:
    col = f'ac_{lag}m'
    val = ac_df[col].mean()
    print(f"    Lag {lag:>2}m: AC = {val:.4f}")

print(f"\n  Interpretation:")
ac_60 = ac_df['ac_60m'].mean()
ac_5 = ac_df['ac_5m'].mean()
if ac_60 < 0.3:
    print(f"    → Basis spread mean-reverts FAST (AC drops to {ac_60:.3f} by 60m)")
elif ac_60 < 0.6:
    print(f"    → Moderate persistence (AC = {ac_60:.3f} at 60m)")
else:
    print(f"    → HIGH persistence — spread does NOT converge quickly (AC = {ac_60:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# 7. P&L SIMULATION — The actual trade
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[7] P&L SIMULATION: Basis premium convergence trade")
print("=" * 80)

print("""
  TRADE:
    When |basis_BN - basis_BB| exceeds threshold:
      - Short futures on exchange with HIGHER basis (premium)
      - Long futures on exchange with LOWER basis (discount/less premium)
    Exit when |basis spread| reverts below exit threshold.

  P&L SOURCE: The convergence of mark prices between exchanges.
    If BN mark is relatively higher and we short BN + long BB,
    when the gap closes, our short gains and our long gains.
    P&L ≈ change in (mark_BN - mark_BB) expressed in bps of position.

  NO LOOKAHEAD: We observe the basis spread at minute T, enter at T+1,
    and measure P&L from actual mark price changes going forward.
""")

# P&L = change in the mark price spread
# If we short the higher-basis exchange and long the lower:
#   Short higher: profit when its mark price drops (relative)
#   Long lower: profit when its mark price rises (relative)
# Combined P&L = -Δ(mark_higher) + Δ(mark_lower) = -Δ(mark_spread)
# where mark_spread = mark_higher - mark_lower (in bps)

# Build forward-looking mark price changes
# Use price_spread_bps (BN mark - BB mark) as the spread
# When we short BN (bn_higher), profit = -Δ(price_spread_bps)
# When we short BB (!bn_higher), profit = +Δ(price_spread_bps)

merged_sorted = merged.sort_values(['symbol', 'minute']).reset_index(drop=True)

# Forward changes in price_spread_bps
grp = merged_sorted.groupby('symbol')
for fwd in [1, 2, 5, 10, 15, 30, 60]:
    merged_sorted[f'price_spread_fwd_{fwd}m'] = grp['price_spread_bps'].shift(-fwd)
    merged_sorted[f'basis_spread_fwd_{fwd}m'] = grp['basis_spread'].shift(-fwd)

# Entry signal: observe basis_spread at time T
# NO t=0 in P&L — we enter AFTER observing, so first P&L is from T to T+hold

ENTRY_THRESHOLDS = [5, 10, 15, 20, 30, 50]
EXIT_THRESHOLD = 5  # exit when |basis spread| < 5 bps
MAX_HOLD = 60  # max 60 minutes

print(f"  Entry: |basis spread| >= threshold")
print(f"  Exit:  |basis spread| < {EXIT_THRESHOLD} bps OR max {MAX_HOLD}m hold")
print(f"  P&L:   Mark price convergence (NOT funding rate)")
print()

# For each entry threshold, simulate trades
# Trade: at minute T, observe |spread| >= threshold
# Enter at T+1 (first price available after signal)
# P&L = -(sign) × Δprice_spread from entry to exit
# where sign = +1 if BN basis is higher (we short BN)

# Use the basis_spread_fwd columns to find exit time
print(f"  {'Thresh':<10} {'Events':>8} {'PerHour':>8} {'Trades':>8} {'GrossBps':>9} "
      f"{'Median':>8} {'WR':>6} {'AvgHold':>8} {'Net@20':>8} {'Net@8':>7} {'$/h@8':>8}")
print(f"  {'-'*95}")

for entry_thresh in ENTRY_THRESHOLDS:
    # Find entry points: |basis_spread| >= threshold
    # Must have forward data
    mask = (merged_sorted['abs_basis_spread'] >= entry_thresh) & \
           merged_sorted[f'price_spread_fwd_1m'].notna()

    # No-overlap: 1 position per symbol at a time
    entries = merged_sorted[mask].copy()
    entries = entries.sort_values(['symbol', 'minute'])

    active_until = {}
    valid_trades = []

    for idx, row in entries.iterrows():
        sym = row['symbol']
        t = row['minute']

        # Skip if we already have a position on this symbol
        if sym in active_until and t < active_until[sym]:
            continue

        # Determine direction
        bn_higher = row['basis_spread'] > 0
        sign = 1 if bn_higher else -1  # +1 means short BN, profit when BN-BB spread shrinks

        # Entry price spread (at T+1)
        entry_spread = row.get('price_spread_fwd_1m', np.nan)
        if pd.isna(entry_spread):
            continue

        # Find exit: when basis_spread drops below exit threshold, or max hold
        exit_spread = entry_spread
        hold_minutes = 1
        for fwd in [2, 5, 10, 15, 30, 60]:
            fwd_basis = row.get(f'basis_spread_fwd_{fwd}m', np.nan)
            fwd_price = row.get(f'price_spread_fwd_{fwd}m', np.nan)
            if pd.isna(fwd_basis) or pd.isna(fwd_price):
                break
            exit_spread = fwd_price
            hold_minutes = fwd
            if abs(fwd_basis) < EXIT_THRESHOLD:
                break

        # P&L = -sign × (exit_price_spread - entry_price_spread)
        # If we shorted the higher basis exchange, profit when its price drops relative
        pnl_bps = -sign * (exit_spread - entry_spread)

        valid_trades.append({
            'symbol': sym,
            'minute': t,
            'entry_basis_spread': row['basis_spread'],
            'bn_higher': bn_higher,
            'entry_price_spread': entry_spread,
            'exit_price_spread': exit_spread,
            'hold_minutes': hold_minutes,
            'pnl_bps': pnl_bps,
        })

        active_until[sym] = t + pd.Timedelta(minutes=hold_minutes + 1)

    if not valid_trades:
        continue

    tdf = pd.DataFrame(valid_trades)
    n_events = mask.sum()
    n_trades = len(tdf)
    gross = tdf['pnl_bps'].mean()
    med = tdf['pnl_bps'].median()
    wr = (tdf['pnl_bps'] > 0).mean() * 100
    avg_hold = tdf['hold_minutes'].mean()
    per_hour = n_events / hours
    notional = 10_000
    net_20 = gross - 20  # taker fees
    net_8 = gross - 8    # maker fees
    dollar_h_8 = (tdf['pnl_bps'] - 8).sum() * notional / 10000 / hours

    print(f"  |spr|≥{entry_thresh:<3} bps {n_events:>8,} {per_hour:>8.1f} {n_trades:>8,} {gross:>9.1f} "
          f"{med:>8.1f} {wr:>5.0f}% {avg_hold:>8.1f}m {net_20:>8.1f} {net_8:>7.1f} {dollar_h_8:>8.1f}")

    # Save for charts
    if entry_thresh == 10:
        trades_10 = tdf.copy()
    if entry_thresh == 20:
        trades_20 = tdf.copy()

# ══════════════════════════════════════════════════════════════════════════
# 8. CHARTS
# ══════════════════════════════════════════════════════════════════════════
print("\n[8] Building charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cross-Exchange Futures Premium Divergence (Binance vs Bybit)\n'
             '2 days tick data, 1-min resolution',
             fontsize=14, fontweight='bold')

# 8a: Basis spread distribution
ax = axes[0][0]
data = merged['basis_spread'].clip(-100, 100)
ax.hist(data.values, bins=200, color=COLORS[0], alpha=0.7, edgecolor='none')
ax.set_xlabel('Basis spread BN-BB (bps)')
ax.set_ylabel('Count')
ax.set_title(f'Basis Spread Distribution (mean={merged["basis_spread"].mean():.1f}, std={merged["basis_spread"].std():.1f})')
ax.axvline(x=0, color='gray', linestyle=':')

# 8b: Autocorrelation decay
ax = axes[0][1]
lags = [1, 2, 5, 10, 15, 30, 60]
acs = [ac_df[f'ac_{l}m'].mean() for l in lags]
ax.plot(lags, acs, 'o-', color=COLORS[0], linewidth=2, markersize=8)
ax.set_xlabel('Lag (minutes)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Basis Spread Mean-Reversion Speed')
ax.axhline(y=0, color='gray', linestyle=':')
ax.set_ylim(-0.1, 1.0)
for i, (l, a) in enumerate(zip(lags, acs)):
    ax.annotate(f'{a:.3f}', (l, a), textcoords="offset points", xytext=(0, 10), fontsize=9, ha='center')

# 8c: Example BTC basis spread over time
ax = axes[1][0]
btc = merged_sorted[merged_sorted.symbol == 'BTCUSDT'].copy()
if len(btc) > 0:
    ax.plot(btc['minute'], btc['basis_bn'], label='BN basis', color=COLORS[0], alpha=0.7, linewidth=0.5)
    ax.plot(btc['minute'], btc['basis_bb'], label='BB basis', color=COLORS[1], alpha=0.7, linewidth=0.5)
    ax.fill_between(btc['minute'], btc['basis_bn'], btc['basis_bb'], alpha=0.2, color=COLORS[3])
    ax.set_ylabel('Basis (bps)')
    ax.set_title('BTCUSDT: Binance vs Bybit Futures Basis')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 8d: P&L distribution for 10bps threshold
ax = axes[1][1]
if 'trades_10' in dir() and len(trades_10) > 0:
    ax.hist(trades_10['pnl_bps'].clip(-50, 50).values, bins=50, color=COLORS[0], alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=trades_10['pnl_bps'].mean(), color=COLORS[1], linestyle='--',
               label=f'Mean: {trades_10["pnl_bps"].mean():.1f} bps')
    ax.set_xlabel('P&L per trade (bps)')
    ax.set_ylabel('Count')
    ax.set_title(f'Trade P&L Distribution (|spread|≥10bps, N={len(trades_10):,})')
    ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'basis_premium_arb.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'basis_premium_arb.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 9. Top diverging symbols chart
# ══════════════════════════════════════════════════════════════════════════

# Pick top 6 symbols by basis spread std and plot their time series
top_syms = sym_stats.head(6).index.tolist()
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Top 6 Symbols: Cross-Exchange Basis Spread Over Time', fontsize=14, fontweight='bold')

for i, sym in enumerate(top_syms):
    ax = axes[i // 2][i % 2]
    sdf = merged_sorted[merged_sorted.symbol == sym].copy()
    if len(sdf) == 0:
        continue
    ax.plot(sdf['minute'], sdf['basis_spread'], color=COLORS[0], linewidth=0.5, alpha=0.8)
    ax.fill_between(sdf['minute'], 0, sdf['basis_spread'],
                     where=sdf['basis_spread'] > 0, alpha=0.2, color=COLORS[2])
    ax.fill_between(sdf['minute'], 0, sdf['basis_spread'],
                     where=sdf['basis_spread'] < 0, alpha=0.2, color=COLORS[1])
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    std = sdf['basis_spread'].std()
    mx = sdf['abs_basis_spread'].max()
    ax.set_title(f'{sym} (std={std:.1f} bps, max={mx:.0f} bps)')
    ax.set_ylabel('BN-BB basis (bps)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / 'basis_premium_top_symbols.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'basis_premium_top_symbols.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 10. Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[10] SUMMARY")
print("=" * 80)
print(f"""
  DATA: {len(merged):,} 1-min bars, {common_syms} symbols, {hours:.1f} hours

  BASIS SPREAD (BN basis - BB basis):
    Mean: {merged['basis_spread'].mean():.2f} bps
    Std:  {merged['basis_spread'].std():.2f} bps
    P99:  {merged['abs_basis_spread'].quantile(0.99):.1f} bps

  MEAN-REVERSION:
    AC at 1m:  {ac_df['ac_1m'].mean():.3f}
    AC at 5m:  {ac_df['ac_5m'].mean():.3f}
    AC at 30m: {ac_df['ac_30m'].mean():.3f}
    AC at 60m: {ac_df['ac_60m'].mean():.3f}

  KEY DIFFERENCE FROM FR ANALYSIS:
    - This is about PRICE convergence, not funding rate collection
    - P&L = mark price change (continuous), not discrete FR payments
    - No t=0 lookahead issue: we enter AFTER observing the divergence
    - Convergence is faster (minutes, not settlement periods)
""")

print("Done.")
