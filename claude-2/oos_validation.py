#!/usr/bin/env python3
"""
OUT-OF-SAMPLE VALIDATION: Bybit 2024-01 → 2025-05
====================================================
The 2 surviving signals were discovered on 2025-06 → 2026-03.
This runs them on the COMPLETELY UNSEEN earlier period.

Same audit rigor:
 - T+1 entry (realistic latency)
 - 30-minute declustering (independent signals only)
 - Shuffle test (1000x)
 - Bootstrap 95% CI
 - Monthly breakdown

Signals tested:
 1. Idea 4: BTC 3m return > 150 bps → long alts
 2. Idea 5: Spot leads futures > 40 bps (3m) → long futures
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import load_kline, progress_bar, RT_TAKER_BPS

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

# UNSEEN period — signals were discovered on 2025-06+
OOS_START = '2024-01-01'
OOS_END = '2025-05-31'

# Original discovery period for comparison
DISC_START = '2025-06-01'
DISC_END = '2026-03-04'

N_SHUFFLE = 1000
N_BOOTSTRAP = 2000
MIN_GAP = 30  # 30-bar declustering

np.random.seed(42)

# Coins to test — focus on ones with 2024 data
TEST_SYMBOLS = [
    'SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
    'LINKUSDT', 'ADAUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT',
    'NEARUSDT', 'ATOMUSDT', 'DOTUSDT', 'WIFUSDT',
    'OPUSDT', 'INJUSDT', 'TIAUSDT', 'SEIUSDT', 'MKRUSDT',
    'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'AAVEUSDT',
    'RUNEUSDT', 'FETUSDT', 'GRTUSDT', 'IMXUSDT', 'STXUSDT',
    'BNBUSDT',
]


def decluster(mask, gap=MIN_GAP):
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return mask & False
    kept = [indices[0]]
    for idx in indices[1:]:
        if idx - kept[-1] >= gap:
            kept.append(idx)
    out = pd.Series(False, index=mask.index)
    out.iloc[kept] = True
    return out


def bootstrap_ci(rets, n=N_BOOTSTRAP):
    if len(rets) < 5:
        return np.nan, np.nan
    means = [np.random.choice(rets.values, len(rets), replace=True).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def shuffle_pval(signal_rets, all_rets, n=N_SHUFFLE):
    if len(signal_rets) < 5 or len(all_rets) < len(signal_rets):
        return np.nan
    obs = signal_rets.mean()
    vals = all_rets.dropna().values
    count = sum(np.random.choice(vals, len(signal_rets), replace=False).mean() >= obs for _ in range(n))
    return count / n


def run_period(symbols, start, end, period_label):
    """Run both signals on all symbols for one time period."""
    print(f"\n{'='*70}")
    print(f"  PERIOD: {period_label} ({start} → {end})")
    print(f"{'='*70}")

    # Load BTC
    btc = load_kline('BTCUSDT', start, end)
    if btc.empty or len(btc) < 5000:
        print(f"  ❌ BTC data insufficient for {period_label}")
        return []
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.columns = ['btc_close']
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000

    results = []
    t0 = time.time()

    for si, sym in enumerate(symbols):
        if sym == 'BTCUSDT':
            continue
        progress_bar(si, len(symbols), prefix=f'  {period_label}', start_time=t0)

        kf = load_kline(sym, start, end, spot=False)
        ks = load_kline(sym, start, end, spot=True)

        if kf.empty or len(kf) < 5000:
            continue

        k = kf[['ts', 'close']].set_index('ts').sort_index()
        k = k[~k.index.duplicated(keep='first')]
        k.columns = ['close']
        k = k.join(btc, how='inner')
        if len(k) < 5000:
            continue

        # Forward returns
        for h in [30, 60, 240]:
            k[f'fwd_{h}m'] = (k['close'].shift(-h) / k['close'] - 1) * 10000

        # Delayed forward returns (T+1 entry)
        for h in [30, 60, 240]:
            k[f'dfwd_{h}m'] = k[f'fwd_{h}m'].shift(-1)

        # === SIGNAL 4: BTC pump > 150 bps in 3m ===
        sig4 = k['btc_ret_3m'] > 150

        # === SIGNAL 5: Spot leads futures > 40 bps ===
        sig5 = pd.Series(False, index=k.index)
        has_spot = False
        if not ks.empty and len(ks) > 5000:
            ks_s = ks[['ts', 'close']].set_index('ts').sort_index()
            ks_s = ks_s[~ks_s.index.duplicated(keep='first')]
            ks_s.columns = ['spot_close']
            k = k.join(ks_s, how='left')
            if 'spot_close' in k.columns:
                k['spot_ret_3m'] = (k['spot_close'] / k['spot_close'].shift(3) - 1) * 10000
                k['fut_ret_3m'] = (k['close'] / k['close'].shift(3) - 1) * 10000
                sig5 = ((k['spot_ret_3m'] - k['fut_ret_3m']) > 40).fillna(False)
                has_spot = True

        for sig_name, sig_mask in [('Idea4_BTC_pump', sig4), ('Idea5_spot_lead', sig5)]:
            if sig_name == 'Idea5_spot_lead' and not has_spot:
                continue
            if sig_mask.sum() < 5:
                continue

            sig_dc = decluster(sig_mask)

            for h in [30, 60, 240]:
                fwd_col = f'fwd_{h}m'
                dfwd_col = f'dfwd_{h}m'
                all_fwd = k[fwd_col].dropna()

                # Raw
                raw_rets = k.loc[sig_mask, fwd_col].dropna()
                # Declustered + delayed (HONEST)
                honest_rets = k.loc[sig_dc, dfwd_col].dropna()

                if len(raw_rets) < 5:
                    continue

                row = {
                    'period': period_label,
                    'symbol': sym,
                    'signal': sig_name,
                    'horizon_m': h,
                    'raw_n': len(raw_rets),
                    'raw_mean': raw_rets.mean(),
                    'raw_net': raw_rets.mean() - RT_TAKER_BPS,
                    'raw_wr': (raw_rets > 0).mean() * 100,
                }

                if len(honest_rets) >= 3:
                    row['honest_n'] = len(honest_rets)
                    row['honest_mean'] = honest_rets.mean()
                    row['honest_net'] = honest_rets.mean() - RT_TAKER_BPS
                    row['honest_wr'] = (honest_rets > 0).mean() * 100

                    # Shuffle test on honest returns
                    row['shuffle_p'] = shuffle_pval(honest_rets, all_fwd)

                    # Bootstrap CI
                    ci_lo, ci_hi = bootstrap_ci(honest_rets)
                    row['ci95_lo'] = ci_lo - RT_TAKER_BPS if not np.isnan(ci_lo) else np.nan
                    row['ci95_hi'] = ci_hi - RT_TAKER_BPS if not np.isnan(ci_hi) else np.nan
                else:
                    row['honest_n'] = 0
                    row['honest_net'] = np.nan

                # Monthly breakdown
                months_prof = 0
                months_total = 0
                worst_month = np.nan
                for ms in pd.date_range(start, end, freq='MS'):
                    me = ms + pd.DateOffset(months=1)
                    mm = sig_dc & (sig_dc.index >= ms) & (sig_dc.index < me)
                    mr = k.loc[mm, dfwd_col].dropna()
                    if len(mr) > 0:
                        months_total += 1
                        mnet = mr.mean() - RT_TAKER_BPS
                        if mnet > 0:
                            months_prof += 1
                        if np.isnan(worst_month) or mnet < worst_month:
                            worst_month = mnet
                row['months_total'] = months_total
                row['months_profitable'] = months_prof
                row['months_pct'] = months_prof / max(months_total, 1) * 100
                row['worst_month'] = worst_month

                results.append(row)

    progress_bar(len(symbols), len(symbols), prefix=f'  {period_label}', start_time=t0)
    return results


def main():
    print("=" * 70)
    print("  OUT-OF-SAMPLE VALIDATION")
    print("  Unseen: 2024-01 → 2025-05 | Discovery: 2025-06 → 2026-03")
    print("=" * 70)

    # Run both periods
    oos_results = run_period(TEST_SYMBOLS, OOS_START, OOS_END, 'OOS_2024')
    disc_results = run_period(TEST_SYMBOLS, DISC_START, DISC_END, 'DISC_2025')

    all_results = oos_results + disc_results
    if not all_results:
        print("\n❌ No results!")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/oos_validation_raw.csv', index=False)

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  COMPARISON: DISCOVERY vs OUT-OF-SAMPLE")
    print(f"{'='*70}")

    print(f"\n{'Signal':<22s} {'H':>4s} │ {'DISC net':>9s} {'WR':>5s} {'n':>5s} {'p':>6s} │ {'OOS net':>9s} {'WR':>5s} {'n':>5s} {'p':>6s} │ {'Verdict'}")
    print("─" * 95)

    for sig in ['Idea4_BTC_pump', 'Idea5_spot_lead']:
        for h in [30, 60, 240]:
            disc = df[(df['signal'] == sig) & (df['horizon_m'] == h) & (df['period'] == 'DISC_2025')]
            oos = df[(df['signal'] == sig) & (df['horizon_m'] == h) & (df['period'] == 'OOS_2024')]

            if len(disc) == 0 and len(oos) == 0:
                continue

            d_net = disc['honest_net'].mean() if len(disc) > 0 else np.nan
            d_wr = disc['honest_wr'].mean() if len(disc) > 0 else np.nan
            d_n = disc['honest_n'].mean() if len(disc) > 0 else 0
            d_p = disc['shuffle_p'].mean() if len(disc) > 0 else np.nan

            o_net = oos['honest_net'].mean() if len(oos) > 0 else np.nan
            o_wr = oos['honest_wr'].mean() if len(oos) > 0 else np.nan
            o_n = oos['honest_n'].mean() if len(oos) > 0 else 0
            o_p = oos['shuffle_p'].mean() if len(oos) > 0 else np.nan

            # Verdict
            if np.isnan(o_net):
                verdict = '❓ No OOS data'
            elif o_net > 0 and (np.isnan(o_p) or o_p < 0.05):
                verdict = '✅ CONFIRMED'
            elif o_net > 0:
                verdict = '⚠️ Profitable but p>{:.2f}'.format(o_p)
            else:
                verdict = '❌ FAILED OOS'

            d_net_s = f'{d_net:+8.1f}' if not np.isnan(d_net) else '     N/A'
            o_net_s = f'{o_net:+8.1f}' if not np.isnan(o_net) else '     N/A'
            d_wr_s = f'{d_wr:4.0f}%' if not np.isnan(d_wr) else '  N/A'
            o_wr_s = f'{o_wr:4.0f}%' if not np.isnan(o_wr) else '  N/A'
            d_p_s = f'{d_p:.4f}' if not np.isnan(d_p) else '   N/A'
            o_p_s = f'{o_p:.4f}' if not np.isnan(o_p) else '   N/A'

            print(f"{sig:<22s} {h:>4d}m │ {d_net_s} {d_wr_s} {d_n:>5.0f} {d_p_s} │ {o_net_s} {o_wr_s} {o_n:>5.0f} {o_p_s} │ {verdict}")

    # Per-symbol comparison for top signal
    for sig in ['Idea4_BTC_pump', 'Idea5_spot_lead']:
        print(f"\n\n{'─'*70}")
        print(f"  {sig} @ 240m — PER SYMBOL")
        print(f"{'─'*70}")
        print(f"  {'Symbol':<12s} │ {'DISC net':>9s} {'WR':>5s} │ {'OOS net':>9s} {'WR':>5s} │ Status")

        disc_sub = df[(df['signal'] == sig) & (df['horizon_m'] == 240) & (df['period'] == 'DISC_2025')]
        oos_sub = df[(df['signal'] == sig) & (df['horizon_m'] == 240) & (df['period'] == 'OOS_2024')]

        all_syms = sorted(set(disc_sub['symbol'].tolist() + oos_sub['symbol'].tolist()))
        pass_count = 0
        fail_count = 0
        for sym in all_syms:
            d = disc_sub[disc_sub['symbol'] == sym]
            o = oos_sub[oos_sub['symbol'] == sym]
            dn = d['honest_net'].values[0] if len(d) > 0 and not np.isnan(d['honest_net'].values[0]) else np.nan
            on = o['honest_net'].values[0] if len(o) > 0 and not np.isnan(o['honest_net'].values[0]) else np.nan
            dw = d['honest_wr'].values[0] if len(d) > 0 and not np.isnan(d['honest_wr'].values[0]) else np.nan
            ow = o['honest_wr'].values[0] if len(o) > 0 and not np.isnan(o['honest_wr'].values[0]) else np.nan

            dn_s = f'{dn:+8.1f}' if not np.isnan(dn) else '     N/A'
            on_s = f'{on:+8.1f}' if not np.isnan(on) else '     N/A'
            dw_s = f'{dw:4.0f}%' if not np.isnan(dw) else '  N/A'
            ow_s = f'{ow:4.0f}%' if not np.isnan(ow) else '  N/A'

            if not np.isnan(on) and on > 0:
                status = '✅'
                pass_count += 1
            elif np.isnan(on):
                status = '—'
            else:
                status = '❌'
                fail_count += 1

            print(f"  {sym:<12s} │ {dn_s} {dw_s} │ {on_s} {ow_s} │ {status}")

        tested = pass_count + fail_count
        if tested > 0:
            print(f"\n  Cross-symbol consistency: {pass_count}/{tested} = {pass_count/tested*100:.0f}%")

    # Monthly consistency in OOS
    print(f"\n\n{'─'*70}")
    print(f"  MONTHLY CONSISTENCY (OOS period)")
    print(f"{'─'*70}")
    for sig in ['Idea4_BTC_pump', 'Idea5_spot_lead']:
        oos = df[(df['signal'] == sig) & (df['horizon_m'] == 240) & (df['period'] == 'OOS_2024')]
        if len(oos) > 0:
            mo_pct = oos['months_pct'].mean()
            worst = oos['worst_month'].mean()
            print(f"  {sig}: {mo_pct:.0f}% of months profitable, worst month: {worst:+.0f} bps")

    elapsed_total = sum(r.get('_elapsed', 0) for r in all_results) if all_results else 0
    print(f"\n✅ Results saved to {OUT}/oos_validation_raw.csv")


if __name__ == '__main__':
    main()
