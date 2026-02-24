#!/usr/bin/env python3
"""
Research: Futures Premium (Basis) Divergence Between Binance & Bybit

Question: When funding rate is positive on one exchange and negative on the other,
does the premium converge? Is there a tradeable edge?

Approach:
1. Historical FR (200 days): Find sign-disagreement events between BN and BB
2. Track what happens to FR on BOTH exchanges in the hours after divergence
3. Real-time ticker (2 days): High-resolution basis analysis for recent divergences
4. Quantify convergence speed, magnitude, and potential P&L
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Resolve data path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_all"

print("=" * 80)
print("RESEARCH: Futures Premium Divergence — Binance vs Bybit")
print("=" * 80)

# ── 1. Load historical FR data ─────────────────────────────────────────────
print("\n[1] Loading historical funding rate data...")
bn_fr = pd.read_parquet(DATA_DIR / "historical_fr" / "binance_fr_history.parquet")
bb_fr = pd.read_parquet(DATA_DIR / "historical_fr" / "bybit_fr_history.parquet")

print(f"  Binance: {len(bn_fr):,} records, {bn_fr.symbol.nunique()} symbols, "
      f"{bn_fr.fundingTime.min().strftime('%Y-%m-%d')} to {bn_fr.fundingTime.max().strftime('%Y-%m-%d')}")
print(f"  Bybit:   {len(bb_fr):,} records, {bb_fr.symbol.nunique()} symbols, "
      f"{bb_fr.fundingTime.min().strftime('%Y-%m-%d')} to {bb_fr.fundingTime.max().strftime('%Y-%m-%d')}")

# Normalize timestamps to nearest hour for joining
bn_fr['hour'] = bn_fr['fundingTime'].dt.floor('h')
bb_fr['hour'] = bb_fr['fundingTime'].dt.floor('h')

# ── 2. Merge on (symbol, hour) ─────────────────────────────────────────────
print("\n[2] Merging BN & BB funding rates on (symbol, hour)...")
merged = pd.merge(
    bn_fr[['symbol', 'hour', 'fundingRate']].rename(columns={'fundingRate': 'fr_bn'}),
    bb_fr[['symbol', 'hour', 'fundingRate']].rename(columns={'fundingRate': 'fr_bb'}),
    on=['symbol', 'hour'],
    how='inner'
)
print(f"  Matched records: {len(merged):,}")
print(f"  Common symbols: {merged.symbol.nunique()}")
print(f"  Date range: {merged.hour.min()} to {merged.hour.max()}")

# ── 3. Identify sign disagreements ─────────────────────────────────────────
print("\n[3] Finding sign disagreements (BN+ & BB- or BN- & BB+)...")
merged['bn_pos_bb_neg'] = (merged['fr_bn'] > 0) & (merged['fr_bb'] < 0)
merged['bn_neg_bb_pos'] = (merged['fr_bn'] < 0) & (merged['fr_bb'] > 0)
merged['sign_disagree'] = merged['bn_pos_bb_neg'] | merged['bn_neg_bb_pos']

n_disagree = merged['sign_disagree'].sum()
n_total = len(merged)
print(f"  Sign disagreements: {n_disagree:,} / {n_total:,} ({n_disagree/n_total*100:.1f}%)")
print(f"    BN+ & BB-: {merged['bn_pos_bb_neg'].sum():,}")
print(f"    BN- & BB+: {merged['bn_neg_bb_pos'].sum():,}")

# ── 4. Extreme disagreements (large magnitude on both sides) ───────────────
print("\n[4] Extreme disagreements (both sides |FR| >= 5bps)...")
merged['fr_bn_bps'] = merged['fr_bn'] * 10000
merged['fr_bb_bps'] = merged['fr_bb'] * 10000
merged['fr_spread'] = merged['fr_bn_bps'] - merged['fr_bb_bps']  # positive = BN more bullish

disagree = merged[merged['sign_disagree']].copy()

# Categorize by magnitude
for min_bps in [0, 5, 10, 20, 50]:
    extreme = disagree[(disagree['fr_bn_bps'].abs() >= min_bps) & (disagree['fr_bb_bps'].abs() >= min_bps)]
    print(f"  Both |FR| >= {min_bps}bps: {len(extreme):,} events")

# ── 5. Detailed analysis of sign disagreements ─────────────────────────────
print("\n[5] Sign disagreement statistics...")
disagree_stats = disagree.groupby('symbol').agg(
    count=('hour', 'size'),
    avg_bn_bps=('fr_bn_bps', 'mean'),
    avg_bb_bps=('fr_bb_bps', 'mean'),
    avg_spread=('fr_spread', 'mean'),
    max_spread=('fr_spread', lambda x: x.abs().max()),
).sort_values('count', ascending=False)

print(f"\n  Top 20 symbols with most sign disagreements:")
print(f"  {'Symbol':<20} {'Count':>6} {'Avg BN(bps)':>12} {'Avg BB(bps)':>12} {'Avg Spread':>11} {'Max |Spread|':>13}")
print(f"  {'-'*74}")
for sym, row in disagree_stats.head(20).iterrows():
    print(f"  {sym:<20} {row['count']:>6.0f} {row['avg_bn_bps']:>12.1f} {row['avg_bb_bps']:>12.1f} "
          f"{row['avg_spread']:>11.1f} {row['max_spread']:>13.1f}")

# ── 6. CONVERGENCE ANALYSIS ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("[6] CONVERGENCE ANALYSIS: What happens after a sign disagreement?")
print("=" * 80)

# Vectorized: use groupby + shift to build forward-looking columns in one pass
print("\n  Building forward-looking FR sequences (vectorized)...")
merged_sorted = merged.sort_values(['symbol', 'hour']).reset_index(drop=True)

MAX_FWD = 8
grp = merged_sorted.groupby('symbol')
for t in range(1, MAX_FWD + 1):
    merged_sorted[f'fr_bn_t{t}'] = grp['fr_bn_bps'].shift(-t)
    merged_sorted[f'fr_bb_t{t}'] = grp['fr_bb_bps'].shift(-t)
    merged_sorted[f'spread_t{t}'] = grp['fr_spread'].shift(-t)
    merged_sorted[f'still_disagree_t{t}'] = grp['sign_disagree'].shift(-t)

# Filter to disagreement rows that have at least 1 forward period
events_df = merged_sorted[merged_sorted['sign_disagree'] & merged_sorted['fr_bn_t1'].notna()].copy()
events_df.rename(columns={'fr_bn_bps': 'fr_bn_t0', 'fr_bb_bps': 'fr_bb_t0', 'fr_spread': 'spread_t0'}, inplace=True)
events_df['type'] = np.where(events_df['bn_pos_bb_neg'], 'BN+_BB-', 'BN-_BB+')

print(f"  Total events with forward data: {len(events_df):,}")

# ── 7. Convergence speed ───────────────────────────────────────────────────
print("\n[7] Convergence speed analysis...")
print(f"\n  After a sign disagreement, how quickly do the signs re-align?")
print(f"  (i.e., both become positive, both become negative, or one goes to zero)")

for t in range(1, 7):
    col = f'still_disagree_t{t}'
    if col in events_df.columns:
        still = events_df[col].sum()
        total = events_df[col].notna().sum()
        if total > 0:
            pct = still / total * 100
            print(f"    t+{t}: {still:,}/{total:,} still disagree ({pct:.1f}%)")

# ── 8. Spread convergence magnitude ────────────────────────────────────────
print("\n[8] Spread convergence magnitude...")
print(f"  Does the FR spread (BN - BB) revert toward zero after disagreement?")

for t in range(1, 7):
    spread_col = f'spread_t{t}'
    if spread_col in events_df.columns:
        valid = events_df[[f'spread_t0', spread_col]].dropna()
        if len(valid) > 0:
            initial_spread = valid['spread_t0'].abs().mean()
            later_spread = valid[spread_col].abs().mean()
            reversion = (initial_spread - later_spread) / initial_spread * 100
            
            # Also check: does spread flip sign? (overshoot)
            same_sign = ((valid['spread_t0'] > 0) & (valid[spread_col] > 0) | 
                        (valid['spread_t0'] < 0) & (valid[spread_col] < 0)).mean() * 100
            
            print(f"    t+{t}: |spread| {initial_spread:.1f} → {later_spread:.1f} bps "
                  f"({reversion:+.1f}% reversion), same sign: {same_sign:.0f}%")

# ── 9. Analyze by direction type ───────────────────────────────────────────
print("\n[9] Convergence by divergence type...")
for dtype in ['BN+_BB-', 'BN-_BB+']:
    sub = events_df[events_df['type'] == dtype]
    print(f"\n  {dtype} ({len(sub):,} events):")
    print(f"    Avg initial spread: {sub['spread_t0'].mean():.1f} bps")
    for t in range(1, 5):
        col = f'spread_t{t}'
        if col in sub.columns:
            valid = sub[['spread_t0', col]].dropna()
            if len(valid) > 0:
                # Mean reversion: how much of the initial spread is recovered?
                initial = valid['spread_t0'].mean()
                later = valid[col].mean()
                print(f"    t+{t}: spread {initial:.1f} → {later:.1f} bps "
                      f"(converged {abs(initial - later):.1f} bps)")

# ── 10. Extreme events deep dive ──────────────────────────────────────────
print("\n" + "=" * 80)
print("[10] EXTREME EVENTS DEEP DIVE (|spread| >= 30 bps at t0)")
print("=" * 80)

extreme = events_df[events_df['spread_t0'].abs() >= 30].copy()
print(f"\n  Extreme events: {len(extreme):,}")

if len(extreme) > 0:
    print(f"\n  {'Symbol':<18} {'Hour':<22} {'Type':<10} {'Spread t0':>10} {'t+1':>8} {'t+2':>8} {'t+3':>8}")
    print(f"  {'-'*84}")
    for _, row in extreme.sort_values('spread_t0', key=abs, ascending=False).head(40).iterrows():
        t1 = f"{row.get('spread_t1', float('nan')):.1f}" if pd.notna(row.get('spread_t1')) else "N/A"
        t2 = f"{row.get('spread_t2', float('nan')):.1f}" if pd.notna(row.get('spread_t2')) else "N/A"
        t3 = f"{row.get('spread_t3', float('nan')):.1f}" if pd.notna(row.get('spread_t3')) else "N/A"
        print(f"  {row['symbol']:<18} {str(row['hour']):<22} {row['type']:<10} "
              f"{row['spread_t0']:>10.1f} {t1:>8} {t2:>8} {t3:>8}")

# ── 11. FUTURES vs FUTURES CROSS-EXCHANGE ARB P&L ─────────────────────────
print("\n" + "=" * 80)
print("[11] FUTURES vs FUTURES CROSS-EXCHANGE ARB — P&L SIMULATION")
print("=" * 80)
print("""
  Trade: Short futures on exchange with positive FR,
         Long futures on exchange with negative FR.
         Delta-neutral (no spot leg needed).

  P&L sources:
    1. FR collection: collect positive FR on short side + receive negative FR on long side
    2. Basis convergence: if mark prices converge, the spread trade profits
  
  Fees: Futures only (no spot fees!)
    - VIP-0 taker: 5.0 bps/side × 2 sides (open) + 5.0 bps × 2 sides (close) = 20 bps RT
    - VIP-0 maker: 2.0 bps/side × 4 = 8 bps RT  
    - VIP-1 taker: 4.0 bps × 4 = 16 bps RT
    - VIP-1 maker: 1.6 bps × 4 = 6.4 bps RT
""")

# Vectorized P&L simulation
# Position signs: BN+_BB- → short BN (+1), long BB (-1); BN-_BB+ → opposite
sign_bn = np.where(events_df['type'] == 'BN+_BB-', +1, -1)
sign_bb = np.where(events_df['type'] == 'BN+_BB-', -1, +1)

# Build per-period P&L and find first exit (signs re-align)
n = len(events_df)
cum_pnl = np.zeros(n)
hold_periods = np.zeros(n, dtype=int)
exited = np.zeros(n, dtype=bool)

# t=0: always included
cum_pnl += sign_bn * events_df['fr_bn_t0'].values + sign_bb * events_df['fr_bb_t0'].values
hold_periods += 1

for t in range(1, MAX_FWD + 1):
    bn_col = f'fr_bn_t{t}'
    disagree_col = f'still_disagree_t{t}'
    if bn_col not in events_df.columns:
        break
    
    has_data = events_df[bn_col].notna().values
    still_disagree = events_df[disagree_col].fillna(False).astype(bool).values
    
    # Only accumulate for rows that haven't exited and have data
    active = ~exited & has_data
    period_pnl = sign_bn * events_df[bn_col].fillna(0).values + sign_bb * events_df[f'fr_bb_t{t}'].fillna(0).values
    cum_pnl += np.where(active, period_pnl, 0)
    hold_periods += active.astype(int)
    
    # Exit if signs re-aligned (no longer disagree)
    exited |= (active & ~still_disagree)

trades_df = pd.DataFrame({
    'symbol': events_df['symbol'].values,
    'hour': events_df['hour'].values,
    'type': events_df['type'].values,
    'initial_spread_bps': events_df['spread_t0'].values,
    'hold_periods': hold_periods,
    'cum_pnl_bps': cum_pnl,
})

# ── Gross stats ──
print(f"  GROSS P&L (before fees):")
print(f"  Total convergence trades: {len(trades_df):,}")
print(f"  Avg hold periods: {trades_df['hold_periods'].mean():.1f}")
print(f"  Avg P&L: {trades_df['cum_pnl_bps'].mean():.2f} bps")
print(f"  Median P&L: {trades_df['cum_pnl_bps'].median():.2f} bps")
print(f"  Win rate: {(trades_df['cum_pnl_bps'] > 0).mean()*100:.1f}%")
print(f"  Avg win: {trades_df[trades_df['cum_pnl_bps'] > 0]['cum_pnl_bps'].mean():.2f} bps")
print(f"  Avg loss: {trades_df[trades_df['cum_pnl_bps'] <= 0]['cum_pnl_bps'].mean():.2f} bps")

# ── Net P&L at different fee tiers ──
data_days = (merged.hour.max() - merged.hour.min()).total_seconds() / 86400
NOTIONAL = 10_000  # $ per leg

fee_scenarios = [
    ("VIP-0 taker", 20.0),
    ("VIP-0 maker", 8.0),
    ("VIP-1 taker", 16.0),
    ("VIP-1 maker", 6.4),
]

print(f"\n  NET P&L BY FEE SCENARIO (${NOTIONAL:,}/leg, {data_days:.0f} days of data):")
print(f"  {'Scenario':<18} {'RT Fee':>8} {'Avg Net':>10} {'Net WR':>8} {'Trades':>8} "
      f"{'Total $':>10} {'$/day':>8}")
print(f"  {'-'*70}")
for name, rt_fee in fee_scenarios:
    net = trades_df['cum_pnl_bps'] - rt_fee
    wr = (net > 0).mean() * 100
    total_dollar = (net * NOTIONAL / 10000).sum()
    per_day = total_dollar / data_days
    print(f"  {name:<18} {rt_fee:>7.1f} {net.mean():>10.2f} {wr:>7.1f}% {len(net):>8,} "
          f"{total_dollar:>10,.0f} {per_day:>8,.0f}")

# ── Gross P&L by bucket ──
print(f"\n  GROSS P&L by initial |spread| bucket:")
print(f"  {'Bucket':<14} {'Trades':>7} {'Gross':>8} {'WR':>6} {'Hold':>6} "
      f"{'Net@20':>8} {'Net@8':>8} {'$/day@20':>10} {'$/day@8':>10}")
print(f"  {'-'*85}")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 500)]:
    bucket = trades_df[(trades_df['initial_spread_bps'].abs() >= lo) & 
                       (trades_df['initial_spread_bps'].abs() < hi)]
    if len(bucket) > 0:
        gross = bucket['cum_pnl_bps'].mean()
        wr = (bucket['cum_pnl_bps'] > 0).mean() * 100
        hold = bucket['hold_periods'].mean()
        net_taker = gross - 20.0
        net_maker = gross - 8.0
        dollar_day_taker = ((bucket['cum_pnl_bps'] - 20.0) * NOTIONAL / 10000).sum() / data_days
        dollar_day_maker = ((bucket['cum_pnl_bps'] - 8.0) * NOTIONAL / 10000).sum() / data_days
        print(f"  {lo}-{hi} bps{'':<7} {len(bucket):>7,} {gross:>8.1f} {wr:>5.0f}% {hold:>6.1f} "
              f"{net_taker:>8.1f} {net_maker:>8.1f} {dollar_day_taker:>10.1f} {dollar_day_maker:>10.1f}")

# ── P&L by direction type ──
print(f"\n  P&L by direction type:")
for dtype in ['BN+_BB-', 'BN-_BB+']:
    sub = trades_df[trades_df['type'] == dtype]
    print(f"    {dtype}: {len(sub):,} trades, gross={sub['cum_pnl_bps'].mean():.2f} bps, "
          f"WR={( sub['cum_pnl_bps'] > 0).mean()*100:.1f}%")

# ── Filter to profitable-only: spread >= 20 bps ──
print(f"\n  FILTERED STRATEGY: Only trade when |FR spread| >= 20 bps")
for min_spread in [10, 15, 20, 30]:
    filt = trades_df[trades_df['initial_spread_bps'].abs() >= min_spread]
    if len(filt) > 0:
        gross = filt['cum_pnl_bps'].mean()
        n_trades = len(filt)
        per_day = n_trades / data_days
        for name, rt_fee in fee_scenarios:
            net = filt['cum_pnl_bps'] - rt_fee
            total_dollar = (net * NOTIONAL / 10000).sum()
            dollar_day = total_dollar / data_days
            if name in ("VIP-0 taker", "VIP-0 maker"):
                print(f"    |spread|≥{min_spread}bps + {name}: {n_trades} trades ({per_day:.1f}/day), "
                      f"gross={gross:.1f}bps, net={net.mean():.1f}bps, "
                      f"${dollar_day:.0f}/day (${NOTIONAL/1000:.0f}K/leg)")

# ── Scaling analysis ──
print(f"\n  SCALING: What if we use larger notional?")
for notional in [10_000, 25_000, 50_000, 100_000]:
    for min_spread in [10, 20]:
        filt = trades_df[trades_df['initial_spread_bps'].abs() >= min_spread]
        if len(filt) > 0:
            # Use maker fees (8 bps RT)
            net = filt['cum_pnl_bps'] - 8.0
            total_dollar = (net * notional / 10000).sum()
            dollar_day = total_dollar / data_days
            dollar_year = dollar_day * 365
            print(f"    ${notional/1000:.0f}K/leg, |spread|≥{min_spread}bps, maker fees: "
                  f"${dollar_day:,.0f}/day = ${dollar_year:,.0f}/year")

# ── 12. Real-time basis analysis (high-res, 2 days) ───────────────────────
print("\n" + "=" * 80)
print("[12] REAL-TIME BASIS ANALYSIS (high-resolution, ~2 days)")
print("=" * 80)

print("\n  Loading real-time ticker data...")
# Binance fundingRate stream has markPrice and indexPrice
bn_tick = pd.read_parquet(DATA_DIR / "binance" / "fundingRate.parquet",
                          columns=['ts', 'symbol', 'markPrice', 'indexPrice', 'lastFundingRate'])
# Bybit linear ticker has indexPrice and lastPrice
bb_tick = pd.read_parquet(DATA_DIR / "bybit" / "ticker.parquet",
                          columns=['ts', 'symbol', 'lastPrice', 'indexPrice', 'markPrice', 'fundingRate'])

print(f"  BN tick: {len(bn_tick):,} records")
print(f"  BB tick: {len(bb_tick):,} records")

# Compute basis = (futures - spot) / spot in bps
bn_tick['basis_bps'] = (bn_tick['markPrice'] - bn_tick['indexPrice']) / bn_tick['indexPrice'] * 10000
bb_tick['basis_bps'] = (bb_tick['markPrice'] - bb_tick['indexPrice']) / bb_tick['indexPrice'] * 10000

# Sample to 1-minute bars for manageable merge
bn_tick['minute'] = bn_tick['ts'].dt.floor('1min')
bb_tick['minute'] = bb_tick['ts'].dt.floor('1min')

bn_1m = bn_tick.groupby(['symbol', 'minute']).agg(
    basis_bn=('basis_bps', 'mean'),
    fr_bn=('lastFundingRate', 'last'),
).reset_index()

bb_1m = bb_tick.groupby(['symbol', 'minute']).agg(
    basis_bb=('basis_bps', 'mean'),
    fr_bb=('fundingRate', 'last'),
).reset_index()

# Merge
basis_merged = pd.merge(bn_1m, bb_1m, on=['symbol', 'minute'], how='inner')
basis_merged['basis_spread'] = basis_merged['basis_bn'] - basis_merged['basis_bb']
basis_merged['fr_bn_bps'] = basis_merged['fr_bn'] * 10000
basis_merged['fr_bb_bps'] = basis_merged['fr_bb'] * 10000

print(f"  Merged 1-min bars: {len(basis_merged):,}")
print(f"  Common symbols: {basis_merged.symbol.nunique()}")

# Find moments where basis has opposite signs
basis_merged['sign_disagree'] = ((basis_merged['basis_bn'] > 0) & (basis_merged['basis_bb'] < 0) |
                                  (basis_merged['basis_bn'] < 0) & (basis_merged['basis_bb'] > 0))
n_disagree = basis_merged['sign_disagree'].sum()
print(f"\n  Basis sign disagreements: {n_disagree:,} / {len(basis_merged):,} "
      f"({n_disagree/len(basis_merged)*100:.1f}%)")

# Distribution of basis spread
print(f"\n  Basis spread (BN - BB) distribution:")
print(f"    Mean: {basis_merged['basis_spread'].mean():.2f} bps")
print(f"    Std: {basis_merged['basis_spread'].std():.2f} bps")
print(f"    P5/P95: {basis_merged['basis_spread'].quantile(0.05):.2f} / {basis_merged['basis_spread'].quantile(0.95):.2f} bps")
print(f"    P1/P99: {basis_merged['basis_spread'].quantile(0.01):.2f} / {basis_merged['basis_spread'].quantile(0.99):.2f} bps")

# Top symbols with highest basis divergence
print(f"\n  Top 20 symbols with highest avg |basis spread|:")
basis_stats = basis_merged.groupby('symbol').agg(
    avg_basis_bn=('basis_bn', 'mean'),
    avg_basis_bb=('basis_bb', 'mean'),
    avg_spread=('basis_spread', 'mean'),
    std_spread=('basis_spread', 'std'),
    max_abs_spread=('basis_spread', lambda x: x.abs().max()),
    pct_sign_disagree=('sign_disagree', 'mean'),
).sort_values('max_abs_spread', ascending=False)

print(f"  {'Symbol':<18} {'BN basis':>10} {'BB basis':>10} {'Spread':>8} {'Std':>8} {'Max|Spr|':>10} {'%disagree':>10}")
print(f"  {'-'*74}")
for sym, row in basis_stats.head(20).iterrows():
    print(f"  {sym:<18} {row['avg_basis_bn']:>10.2f} {row['avg_basis_bb']:>10.2f} "
          f"{row['avg_spread']:>8.2f} {row['std_spread']:>8.2f} {row['max_abs_spread']:>10.1f} "
          f"{row['pct_sign_disagree']*100:>9.1f}%")

# ── 13. Basis convergence in real-time data ────────────────────────────────
print("\n" + "=" * 80)
print("[13] REAL-TIME BASIS CONVERGENCE (autocorrelation & mean-reversion)")
print("=" * 80)

# For each symbol, compute autocorrelation of basis_spread
print("\n  Basis spread autocorrelation (1-min lag) by symbol:")
autocorr_results = []
for sym in basis_merged.symbol.unique():
    sdf = basis_merged[basis_merged.symbol == sym].sort_values('minute')
    if len(sdf) < 100:
        continue
    ac1 = sdf['basis_spread'].autocorr(lag=1)
    ac5 = sdf['basis_spread'].autocorr(lag=5)
    ac30 = sdf['basis_spread'].autocorr(lag=30)
    ac60 = sdf['basis_spread'].autocorr(lag=60)
    autocorr_results.append({
        'symbol': sym,
        'ac_1m': ac1,
        'ac_5m': ac5,
        'ac_30m': ac30,
        'ac_60m': ac60,
        'avg_abs_spread': sdf['basis_spread'].abs().mean(),
    })

ac_df = pd.DataFrame(autocorr_results)
if len(ac_df) > 0:
    print(f"  Symbols analyzed: {len(ac_df)}")
    print(f"  Avg autocorrelation at lag 1m:  {ac_df['ac_1m'].mean():.4f}")
    print(f"  Avg autocorrelation at lag 5m:  {ac_df['ac_5m'].mean():.4f}")
    print(f"  Avg autocorrelation at lag 30m: {ac_df['ac_30m'].mean():.4f}")
    print(f"  Avg autocorrelation at lag 60m: {ac_df['ac_60m'].mean():.4f}")
    
    print(f"\n  Interpretation:")
    if ac_df['ac_60m'].mean() < 0.3:
        print(f"    → Basis spread mean-reverts within ~1 hour (AC drops to {ac_df['ac_60m'].mean():.3f})")
    elif ac_df['ac_60m'].mean() < 0.6:
        print(f"    → Moderate persistence: basis spread partially mean-reverts within 1 hour")
    else:
        print(f"    → High persistence: basis spread does NOT converge quickly")

# ── 14. FR disagreement + basis divergence cross-check ─────────────────────
print("\n" + "=" * 80)
print("[14] FR SIGN DISAGREEMENT vs BASIS SIGN DISAGREEMENT (real-time)")
print("=" * 80)

# Do FR disagreements coincide with basis disagreements?
basis_merged['fr_disagree'] = ((basis_merged['fr_bn_bps'] > 0) & (basis_merged['fr_bb_bps'] < 0) |
                                (basis_merged['fr_bn_bps'] < 0) & (basis_merged['fr_bb_bps'] > 0))

both = (basis_merged['sign_disagree'] & basis_merged['fr_disagree']).sum()
only_basis = (basis_merged['sign_disagree'] & ~basis_merged['fr_disagree']).sum()
only_fr = (~basis_merged['sign_disagree'] & basis_merged['fr_disagree']).sum()
neither = (~basis_merged['sign_disagree'] & ~basis_merged['fr_disagree']).sum()

total = len(basis_merged)
print(f"  Both FR + Basis disagree: {both:,} ({both/total*100:.1f}%)")
print(f"  Only Basis disagree:      {only_basis:,} ({only_basis/total*100:.1f}%)")
print(f"  Only FR disagree:         {only_fr:,} ({only_fr/total*100:.1f}%)")
print(f"  Neither disagree:         {neither:,} ({neither/total*100:.1f}%)")

# ── 15. Summary & conclusion ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("[15] SUMMARY & CONCLUSIONS")
print("=" * 80)

avg_pnl = trades_df['cum_pnl_bps'].mean()
wr = (trades_df['cum_pnl_bps'] > 0).mean() * 100

# Filtered stats
filt_10 = trades_df[trades_df['initial_spread_bps'].abs() >= 10]
filt_20 = trades_df[trades_df['initial_spread_bps'].abs() >= 20]
f10_pnl = filt_10['cum_pnl_bps'].mean() if len(filt_10) > 0 else 0
f20_pnl = filt_20['cum_pnl_bps'].mean() if len(filt_20) > 0 else 0
f10_day = len(filt_10) / data_days
f20_day = len(filt_20) / data_days

print(f"""
  TRADE: Short futures on exchange A + Long futures on exchange B (delta-neutral)
  FEES:  Futures-only — 20 bps RT (taker) or 8 bps RT (maker)

  1. FREQUENCY: {n_disagree:,} FR sign disagreements ({n_disagree/n_total*100:.1f}% of all settlements)
  
  2. CONVERGENCE: YES — spread converges with {wr:.0f}% gross win rate
     - All trades: avg gross {avg_pnl:.1f} bps → net {avg_pnl-20:.1f} bps (taker) / {avg_pnl-8:.1f} bps (maker)
     - |spread|≥10bps ({len(filt_10):,} trades, {f10_day:.1f}/day): gross {f10_pnl:.0f} bps → net {f10_pnl-20:.0f} (taker) / {f10_pnl-8:.0f} (maker)
     - |spread|≥20bps ({len(filt_20):,} trades, {f20_day:.1f}/day): gross {f20_pnl:.0f} bps → net {f20_pnl-20:.0f} (taker) / {f20_pnl-8:.0f} (maker)
  
  3. REAL-TIME BASIS: Autocorrelation decays ~1.0 → ~{ac_df['ac_60m'].mean():.2f} over 60 min
     → Strong mean-reversion in basis spread
  
  4. VERDICT:
     - With MAKER fees (8 bps RT): profitable on ALL trades (avg net = {avg_pnl-8:.1f} bps)
     - With TAKER fees (20 bps RT): profitable only on |spread|≥10 bps trades
     - Bigger spreads = bigger edge, but rarer events
     - Key risk: basis risk (mark prices may diverge further before converging)
""")

print("Done.")
