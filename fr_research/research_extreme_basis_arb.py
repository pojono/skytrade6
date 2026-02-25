#!/usr/bin/env python3
"""
Research: Extreme Basis Divergence — Fat-Tail Event Strategy

Focus: When the futures premium gap between Binance and Bybit is HUGE
(>100bps, >200bps, >500bps), it MUST snap back. How fast, how profitable?

Uses 52 hours of 1-min tick data (Feb 22-24, 2026) from both exchanges.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_all"
OUT_DIR = REPO_ROOT / "fr_research" / "charts"
OUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

t0 = time.time()

print("=" * 80)
print("EXTREME BASIS DIVERGENCE: Fat-Tail Event Analysis")
print("=" * 80)

# ── Load and merge ──
print("\n[1] Loading tick data...")
bb = pd.read_parquet(DATA_DIR / "bybit" / "ticker.parquet",
    columns=['ts', 'symbol', 'markPrice', 'indexPrice', 'lastPrice',
             'bid1Price', 'ask1Price'])
bn = pd.read_parquet(DATA_DIR / "binance" / "fundingRate.parquet",
    columns=['ts', 'symbol', 'markPrice', 'indexPrice'])

bb['minute'] = bb['ts'].dt.floor('1min')
bn['minute'] = bn['ts'].dt.floor('1min')

bb_1m = bb.groupby(['symbol', 'minute']).agg(
    mark_bb=('markPrice', 'last'), index_bb=('indexPrice', 'last'),
    last_bb=('lastPrice', 'last'),
    bid_bb=('bid1Price', 'last'), ask_bb=('ask1Price', 'last'),
).reset_index()
bn_1m = bn.groupby(['symbol', 'minute']).agg(
    mark_bn=('markPrice', 'last'), index_bn=('indexPrice', 'last'),
).reset_index()

m = pd.merge(bb_1m, bn_1m, on=['symbol', 'minute'], how='inner')
m = m.sort_values(['symbol', 'minute']).reset_index(drop=True)
m['basis_bn'] = (m['mark_bn'] - m['index_bn']) / m['index_bn'] * 10000
m['basis_bb'] = (m['mark_bb'] - m['index_bb']) / m['index_bb'] * 10000
m['basis_spread'] = m['basis_bn'] - m['basis_bb']
m['abs_basis_spread'] = m['basis_spread'].abs()
m['price_gap_bps'] = (m['mark_bn'] - m['mark_bb']) / ((m['mark_bn'] + m['mark_bb']) / 2) * 10000

hours = m.minute.nunique() / 60
n_sym = m.symbol.nunique()
print(f"  {len(m):,} 1-min bars, {n_sym} symbols, {hours:.1f} hours  [{time.time()-t0:.1f}s]")

# ── Extreme event catalog ──
print("\n" + "=" * 80)
print("[2] EXTREME EVENT CATALOG")
print("=" * 80)

for thresh in [50, 100, 200, 300, 500, 1000]:
    n = (m['abs_basis_spread'] >= thresh).sum()
    syms = m[m['abs_basis_spread'] >= thresh]['symbol'].nunique()
    per_h = n / hours
    per_d = per_h * 24
    print(f"  |spread| >= {thresh:>5} bps: {n:>7,} min-bars ({per_h:>6.1f}/hr, ~{per_d:>6.0f}/day) across {syms:>3} symbols")

# ── Build forward shifts ──
print(f"\n[3] Building forward shifts for reversion analysis...")
grp = m.groupby('symbol')
fwd_mins = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
for f in fwd_mins:
    m[f'pg_{f}'] = grp['price_gap_bps'].shift(-f)
    m[f'bs_{f}'] = grp['basis_spread'].shift(-f)
    m[f'abs_bs_{f}'] = grp['abs_basis_spread'].shift(-f)
print(f"  Done [{time.time()-t0:.1f}s]")

# ── Reversion speed analysis ──
print("\n" + "=" * 80)
print("[4] SPREAD REVERSION SPEED")
print("=" * 80)

for thresh in [100, 200, 500]:
    events = m[m['abs_basis_spread'] >= thresh].copy()
    if len(events) < 5:
        print(f"\n  |spread| >= {thresh}: only {len(events)} events, skipping")
        continue

    entry_mean = events['abs_basis_spread'].mean()
    entry_med = events['abs_basis_spread'].median()

    print(f"\n  |spread| >= {thresh} bps ({len(events):,} events, {events.symbol.nunique()} symbols):")
    print(f"    Entry |spread|: mean={entry_mean:.0f}, median={entry_med:.0f} bps")
    header = f"    {'Fwd':>6}  {'Mean|spr|':>10}  {'Med|spr|':>10}  {'%reverted':>10}  {'P25':>8}  {'P75':>8}"
    print(header)
    print(f"    {'-' * len(header)}")

    for f in fwd_mins:
        col = f'abs_bs_{f}'
        if col not in events.columns:
            break
        fwd_abs = events[col].dropna().values
        if len(fwd_abs) < 5:
            break
        mean_f = fwd_abs.mean()
        med_f = np.median(fwd_abs)
        pct_reverted = (1 - mean_f / entry_mean) * 100
        p25 = np.percentile(fwd_abs, 25)
        p75 = np.percentile(fwd_abs, 75)
        print(f"    t+{f:>3}m  {mean_f:>10.1f}  {med_f:>10.1f}  {pct_reverted:>9.0f}%  {p25:>8.1f}  {p75:>8.1f}")

# ── Reversion completeness ──
print("\n" + "=" * 80)
print("[5] HOW MANY EXTREME EVENTS FULLY REVERT WITHIN 120 MINUTES?")
print("=" * 80)

for entry_thresh in [100, 200, 500]:
    events = m[m['abs_basis_spread'] >= entry_thresh].copy()
    if len(events) < 3:
        continue

    n_total = len(events)
    reverted = {5: 0, 20: 0, 50: 0}

    for _, row in events.iterrows():
        for f in fwd_mins:
            col = f'abs_bs_{f}'
            val = row.get(col, np.nan)
            if pd.isna(val):
                break
            for tgt in sorted(reverted.keys()):
                if val < tgt:
                    reverted[tgt] += 1
            if val < 5:
                break

    # Fix: each event should only be counted once at its best reversion level
    # Re-do properly
    rev_5 = 0
    rev_20 = 0
    rev_50 = 0
    for _, row in events.iterrows():
        min_fwd = 9999
        for f in fwd_mins:
            col = f'abs_bs_{f}'
            val = row.get(col, np.nan)
            if pd.isna(val):
                break
            min_fwd = min(min_fwd, val)
        if min_fwd < 5:
            rev_5 += 1
        if min_fwd < 20:
            rev_20 += 1
        if min_fwd < 50:
            rev_50 += 1

    print(f"\n  |spread| >= {entry_thresh} bps ({n_total:,} events):")
    print(f"    Reverted to <  5 bps within 120m: {rev_5:>5}/{n_total} ({rev_5/n_total*100:>4.0f}%)")
    print(f"    Reverted to < 20 bps within 120m: {rev_20:>5}/{n_total} ({rev_20/n_total*100:>4.0f}%)")
    print(f"    Reverted to < 50 bps within 120m: {rev_50:>5}/{n_total} ({rev_50/n_total*100:>4.0f}%)")

# ── P&L simulation ──
print("\n" + "=" * 80)
print("[6] P&L ON EXTREME EVENTS (entry at T+1, no lookahead)")
print("=" * 80)

NOTIONAL = 10_000

for entry_thresh in [100, 200, 300, 500]:
    for exit_thresh in [20, 50]:
        mask = (m['abs_basis_spread'] >= entry_thresh) & m['pg_1'].notna()
        entries = m[mask].copy()
        if len(entries) < 3:
            continue

        entries['sign'] = np.where(entries['basis_spread'] > 0, 1, -1)
        entry_pg = entries['pg_1'].values
        sign_arr = entries['sign'].values

        # Find exit: basis_spread < exit_thresh or max 120m
        exit_pg = entry_pg.copy()
        hold = np.ones(len(entries))
        prev_f = 1
        for f in fwd_mins[1:]:
            abs_col = f'abs_bs_{f}'
            pg_col = f'pg_{f}'
            if pg_col not in entries.columns:
                break
            has = entries[pg_col].notna().values
            below = entries[abs_col].fillna(9999).values < exit_thresh
            still_open = (hold == prev_f)
            should_exit = still_open & has & below
            should_continue = still_open & has & ~below

            exit_pg = np.where(should_exit | should_continue,
                               entries[pg_col].fillna(0).values, exit_pg)
            hold = np.where(should_exit, f,
                   np.where(should_continue, f, hold))
            prev_f = f

        pnl = sign_arr * (entry_pg - exit_pg)
        entries['pnl_bps'] = pnl
        entries['hold'] = hold

        # No overlap: 1 position per symbol at a time
        entries_s = entries.sort_values('minute')
        keep = []
        active = {}
        for idx, row in entries_s[['symbol', 'minute', 'hold', 'pnl_bps']].iterrows():
            sym = row['symbol']
            t = row['minute']
            if sym in active and t < active[sym]:
                continue
            keep.append(idx)
            active[sym] = t + pd.Timedelta(minutes=row['hold'] + 1)

        trades = entries.loc[keep]
        if len(trades) < 2:
            continue

        g = trades['pnl_bps'].mean()
        med = trades['pnl_bps'].median()
        wr = (trades['pnl_bps'] > 0).mean() * 100
        h = trades['hold'].mean()
        dollar_h_8 = (trades['pnl_bps'] - 8).sum() * NOTIONAL / 10000 / hours
        dollar_d_8 = dollar_h_8 * 24
        dollar_h_20 = (trades['pnl_bps'] - 20).sum() * NOTIONAL / 10000 / hours
        dollar_d_20 = dollar_h_20 * 24

        print(f"\n  ENTRY >= {entry_thresh} bps, EXIT < {exit_thresh} bps, max 120m:")
        print(f"    Trades: {len(trades)} (no overlap), {trades.symbol.nunique()} symbols")
        print(f"    Entry |spread|: mean={trades['abs_basis_spread'].mean():.0f} bps")
        print(f"    Gross: mean={g:.1f} bps, median={med:.1f} bps")
        print(f"    Win rate: {wr:.0f}%")
        print(f"    Avg hold: {h:.0f}m")
        print(f"    Net @8 bps (maker):  ${dollar_h_8:>7.0f}/hr = ${dollar_d_8:>9,.0f}/day")
        print(f"    Net @20 bps (taker): ${dollar_h_20:>7.0f}/hr = ${dollar_d_20:>9,.0f}/day")

        # Show all individual trades if <= 40
        if len(trades) <= 40:
            print(f"    ── Individual trades ──")
            cols = ['symbol', 'minute', 'abs_basis_spread', 'hold', 'pnl_bps']
            print(f"    {'Symbol':<16} {'Time':<20} {'Entry|spr|':>10} {'Hold':>6} {'Gross':>8} {'Net@8':>7}")
            for _, t in trades.sort_values('pnl_bps', ascending=False).iterrows():
                net = t['pnl_bps'] - 8
                ts = str(t['minute'])[:16]
                print(f"    {t['symbol']:<16} {ts:<20} {t['abs_basis_spread']:>10.0f} {t['hold']:>5.0f}m {t['pnl_bps']:>8.1f} {net:>7.1f}")

# ── Persistent offset check ──
print("\n" + "=" * 80)
print("[7] PERSISTENT OFFSET CHECK — are 'extreme' events really extreme?")
print("=" * 80)

sym_means = m.groupby('symbol')['basis_spread'].mean()
sym_stds = m.groupby('symbol')['basis_spread'].std()

# For each extreme event, check if it's extreme relative to its own symbol's mean
for thresh in [100, 200, 500]:
    events = m[m['abs_basis_spread'] >= thresh].copy()
    if len(events) < 3:
        continue

    events['sym_mean'] = events['symbol'].map(sym_means)
    events['sym_std'] = events['symbol'].map(sym_stds)
    events['deviation_from_mean'] = (events['basis_spread'] - events['sym_mean']).abs()
    events['z_score'] = events['deviation_from_mean'] / events['sym_std'].clip(lower=0.01)

    print(f"\n  |spread| >= {thresh} bps ({len(events):,} events, {events.symbol.nunique()} symbols):")
    print(f"    Raw |spread|:    mean={events['abs_basis_spread'].mean():.0f} bps")
    print(f"    |dev from mean|: mean={events['deviation_from_mean'].mean():.0f} bps")
    print(f"    Z-score:         mean={events['z_score'].mean():.1f}")

    # How much is due to persistent offset vs actual divergence?
    pct_offset = (1 - events['deviation_from_mean'].mean() / events['abs_basis_spread'].mean()) * 100
    print(f"    Persistent offset explains: {pct_offset:.0f}% of raw spread")

    # Which symbols dominate?
    sym_counts = events['symbol'].value_counts().head(10)
    print(f"    Top symbols: {', '.join(f'{s}({c})' for s, c in sym_counts.items())}")

# ── Charts ──
print(f"\n[8] Building charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Extreme Basis Divergence: Fat-Tail Events\n"
             "Cross-Exchange Futures Premium (Binance vs Bybit), 52h data",
             fontsize=14, fontweight='bold')

# 8a: Spread reversion path for different thresholds
ax = axes[0][0]
for i, thresh in enumerate([50, 100, 200, 500]):
    events = m[m['abs_basis_spread'] >= thresh].copy()
    if len(events) < 5:
        continue
    entry_mean = events['abs_basis_spread'].mean()
    means = [entry_mean]
    times = [0]
    for f in fwd_mins:
        col = f'abs_bs_{f}'
        if col not in events.columns:
            break
        vals = events[col].dropna().values
        if len(vals) < 5:
            break
        means.append(vals.mean())
        times.append(f)

    # Normalize to % of entry
    norm = [x / entry_mean * 100 for x in means]
    ax.plot(times, norm, 'o-', color=COLORS[i], linewidth=2, markersize=5,
            label=f'>={thresh} bps (n={len(events):,})')

ax.set_xlabel('Minutes after entry')
ax.set_ylabel('% of entry |spread| remaining')
ax.set_title('Spread Reversion Speed by Entry Threshold')
ax.legend()
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylim(0, 110)

# 8b: Scatter of entry spread vs P&L (for >=50 bps events)
ax = axes[0][1]
events_50 = m[(m['abs_basis_spread'] >= 50) & m['pg_1'].notna()].copy()
if len(events_50) > 0:
    events_50['sign'] = np.where(events_50['basis_spread'] > 0, 1, -1)
    # Quick P&L using 60m forward
    fwd_col = 'pg_60'
    if fwd_col in events_50.columns:
        entry_pg = events_50['pg_1'].values
        exit_pg = events_50[fwd_col].fillna(events_50['pg_1']).values
        pnl = events_50['sign'].values * (entry_pg - exit_pg)
        events_50['pnl_60m'] = pnl

        # Subsample for plotting
        sample = events_50.sample(min(5000, len(events_50)), random_state=42)
        ax.scatter(sample['abs_basis_spread'], sample['pnl_60m'],
                   alpha=0.15, s=5, color=COLORS[0])
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=8, color='green', linestyle='--', alpha=0.5, label='8 bps maker fee')
        ax.set_xlabel('Entry |basis spread| (bps)')
        ax.set_ylabel('P&L at 60m (bps)')
        ax.set_title('Entry Spread vs 60m P&L')
        ax.legend()
        ax.set_xlim(50, min(500, events_50['abs_basis_spread'].quantile(0.99)))

# 8c: Time series of top extreme-event symbols
ax = axes[1][0]
# Pick 3 symbols with biggest events
top_extreme = m.nlargest(20, 'abs_basis_spread')['symbol'].unique()[:3]
for i, sym in enumerate(top_extreme):
    sdf = m[m.symbol == sym]
    ax.plot(sdf['minute'], sdf['basis_spread'], label=sym,
            color=COLORS[i], linewidth=0.7, alpha=0.8)
ax.set_ylabel('BN-BB basis spread (bps)')
ax.set_title('Top 3 Extreme-Divergence Symbols Over Time')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

# 8d: Histogram of extreme event P&L
ax = axes[1][1]
for thresh, color, label in [(100, COLORS[0], '>=100'), (200, COLORS[1], '>=200')]:
    events = m[(m['abs_basis_spread'] >= thresh) & m['pg_1'].notna()].copy()
    if len(events) < 10:
        continue
    events['sign'] = np.where(events['basis_spread'] > 0, 1, -1)
    if 'pg_60' in events.columns:
        entry_pg = events['pg_1'].values
        exit_pg = events['pg_60'].fillna(events['pg_1']).values
        pnl = events['sign'].values * (entry_pg - exit_pg)
        ax.hist(np.clip(pnl, -200, 200), bins=80, alpha=0.5, color=color,
                label=f'{label} bps (n={len(events):,})', edgecolor='none')

ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.axvline(x=8, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('P&L at 60m (bps)')
ax.set_ylabel('Count')
ax.set_title('Extreme Event P&L Distribution (60m hold)')
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'extreme_basis_arb.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'extreme_basis_arb.png'}")
plt.close()

# ── Summary ──
print("\n" + "=" * 80)
print("[9] SUMMARY")
print("=" * 80)
print(f"""
  STRATEGY: Wait for extreme cross-exchange basis divergences, enter convergence trade.
  
  DATA: {len(m):,} 1-min bars, {n_sym} symbols, {hours:.1f} hours (Feb 22-24, 2026)
  
  KEY QUESTION: Do the biggest divergences snap back fast enough to profit?
  
  This is different from the small-spread analysis because:
  1. Extreme events (>100 bps) are rare but the convergence is more certain
  2. The absolute P&L per trade is much larger
  3. The key risk is: does the spread revert or does it stay permanently different?
""")

print(f"Done [{time.time()-t0:.1f}s]")
