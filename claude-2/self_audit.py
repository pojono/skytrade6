#!/usr/bin/env python3
"""
SELF-AUDIT: No lookahead, no overfitting
==========================================
Systematic check for every bias that could inflate results.

Tests:
1. ENTRY DELAY: Re-run top signals entering at T+1 instead of T (realistic latency)
2. SIGNAL CLUSTERING: Count truly independent signals (min 30m gap between entries)
3. SHUFFLE TEST: Randomize signal timestamps → null distribution of returns
4. BOOTSTRAP CI: 95% confidence intervals on mean return
5. BONFERRONI: Multiple comparison correction for p-values
6. REGIME SPLIT: Monthly P&L to check if edge is concentrated in one period
7. EFFECTIVE N: Adjust for serial correlation in overlapping returns
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from scipy import stats
from data_loader import get_symbols, load_kline, load_premium, progress_bar, RT_TAKER_BPS

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2025-06-01'
END = '2026-03-04'
N_SHUFFLE = 1000
N_BOOTSTRAP = 2000
MIN_GAP_BARS = 30  # minimum 30 bars (30min) between independent signals

np.random.seed(42)


def load_btc():
    btc = load_kline('BTCUSDT', START, END)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.columns = ['btc_close']
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000
    return btc


def decluster_signals(signal_mask, min_gap=MIN_GAP_BARS):
    """Remove overlapping signals, keeping only the first in each cluster."""
    indices = np.where(signal_mask.values)[0]
    if len(indices) == 0:
        return signal_mask & False  # empty mask
    
    kept = [indices[0]]
    for idx in indices[1:]:
        if idx - kept[-1] >= min_gap:
            kept.append(idx)
    
    new_mask = pd.Series(False, index=signal_mask.index)
    new_mask.iloc[kept] = True
    return new_mask


def bootstrap_ci(returns, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap confidence interval for mean return."""
    if len(returns) < 5:
        return returns.mean(), returns.mean(), returns.mean()
    
    means = np.array([
        np.random.choice(returns.values, size=len(returns), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.mean(means), np.percentile(means, (1 - alpha) * 100)


def shuffle_test(returns_at_signal, all_returns, n_shuffle=N_SHUFFLE):
    """Shuffle test: how likely is the observed mean under random timing?"""
    observed = returns_at_signal.mean()
    n = len(returns_at_signal)
    
    random_means = []
    all_vals = all_returns.dropna().values
    if len(all_vals) < n:
        return observed, 0.5, 0.0  # can't test
    
    for _ in range(n_shuffle):
        sample = np.random.choice(all_vals, size=n, replace=False)
        random_means.append(sample.mean())
    
    random_means = np.array(random_means)
    # p-value: fraction of random samples with mean >= observed
    p_value = (random_means >= observed).mean()
    effect_size = (observed - random_means.mean()) / max(random_means.std(), 0.001)
    
    return observed, p_value, effect_size


def effective_n(returns, lag=30):
    """Effective sample size accounting for autocorrelation."""
    n = len(returns)
    if n < lag * 2:
        return n
    
    # Autocorrelation
    centered = returns - returns.mean()
    var = centered.var()
    if var == 0:
        return n
    
    rho_sum = 0
    for k in range(1, min(lag, n // 2)):
        rho_k = (centered.iloc[:-k].values * centered.iloc[k:].values).mean() / var
        if abs(rho_k) < 2 / np.sqrt(n):  # insignificant
            break
        rho_sum += rho_k
    
    ess = n / (1 + 2 * rho_sum)
    return max(1, ess)


def audit_signal(name, signal_mask, fwd_returns, all_fwd_returns, monthly_index):
    """Full audit of one signal configuration."""
    result = {'signal': name}
    
    # ============ RAW (as reported) ============
    raw_rets = fwd_returns[signal_mask].dropna()
    result['raw_n'] = len(raw_rets)
    result['raw_mean'] = raw_rets.mean() if len(raw_rets) > 0 else 0
    result['raw_wr'] = (raw_rets > 0).mean() * 100 if len(raw_rets) > 0 else 0
    result['raw_net'] = result['raw_mean'] - RT_TAKER_BPS
    
    if len(raw_rets) < 10:
        return result
    
    # ============ TEST 1: Entry delay (T+1) ============
    # Forward return from T+1 to T+1+h instead of T to T+h
    # This is equivalent to fwd_returns shifted by 1
    delayed_fwd = fwd_returns.shift(-1)  # return starting 1 bar later
    delayed_rets = delayed_fwd[signal_mask].dropna()
    result['delayed_n'] = len(delayed_rets)
    result['delayed_mean'] = delayed_rets.mean() if len(delayed_rets) > 0 else 0
    result['delayed_net'] = result['delayed_mean'] - RT_TAKER_BPS
    result['delay_cost'] = result['raw_mean'] - result['delayed_mean']
    
    # ============ TEST 2: Declustered signals ============
    declustered = decluster_signals(signal_mask, MIN_GAP_BARS)
    declust_rets = fwd_returns[declustered].dropna()
    result['declust_n'] = len(declust_rets)
    result['declust_mean'] = declust_rets.mean() if len(declust_rets) > 0 else 0
    result['declust_net'] = result['declust_mean'] - RT_TAKER_BPS
    result['cluster_ratio'] = result['raw_n'] / max(result['declust_n'], 1)
    
    # Declustered + delayed
    declust_delayed_rets = delayed_fwd[declustered].dropna()
    result['declust_delayed_n'] = len(declust_delayed_rets)
    result['declust_delayed_mean'] = declust_delayed_rets.mean() if len(declust_delayed_rets) > 0 else 0
    result['declust_delayed_net'] = result['declust_delayed_mean'] - RT_TAKER_BPS
    
    # ============ TEST 3: Shuffle test ============
    obs, p_val, eff_sz = shuffle_test(raw_rets, all_fwd_returns)
    result['shuffle_p'] = p_val
    result['shuffle_effect_size'] = eff_sz
    
    # ============ TEST 4: Bootstrap CI ============
    ci_lo, ci_mid, ci_hi = bootstrap_ci(raw_rets)
    result['ci95_lo'] = ci_lo - RT_TAKER_BPS
    result['ci95_hi'] = ci_hi - RT_TAKER_BPS
    result['ci95_lo_profitable'] = ci_lo > RT_TAKER_BPS
    
    # ============ TEST 5: Effective N ============
    eff = effective_n(raw_rets)
    result['effective_n'] = eff
    result['n_inflation'] = result['raw_n'] / max(eff, 1)
    # Adjusted t-stat with effective N
    if raw_rets.std() > 0:
        result['t_stat_raw'] = raw_rets.mean() / (raw_rets.std() / np.sqrt(len(raw_rets)))
        result['t_stat_adj'] = raw_rets.mean() / (raw_rets.std() / np.sqrt(eff))
    else:
        result['t_stat_raw'] = 0
        result['t_stat_adj'] = 0
    
    # ============ TEST 6: Monthly regime split ============
    monthly_rets = []
    for month_start in pd.date_range(START, END, freq='MS'):
        month_end = month_start + pd.DateOffset(months=1)
        month_mask = signal_mask & (signal_mask.index >= month_start) & (signal_mask.index < month_end)
        m_rets = fwd_returns[month_mask].dropna()
        if len(m_rets) > 0:
            monthly_rets.append({
                'month': month_start.strftime('%Y-%m'),
                'n': len(m_rets),
                'mean': m_rets.mean(),
                'net': m_rets.mean() - RT_TAKER_BPS,
            })
    
    if monthly_rets:
        mdf = pd.DataFrame(monthly_rets)
        result['months_tested'] = len(mdf)
        result['months_profitable'] = (mdf['net'] > 0).sum()
        result['months_pct_profitable'] = result['months_profitable'] / result['months_tested'] * 100
        result['worst_month_net'] = mdf['net'].min()
        result['best_month_net'] = mdf['net'].max()
        result['monthly_sharpe'] = mdf['net'].mean() / max(mdf['net'].std(), 0.01)
    
    return result


def main():
    print("=" * 80)
    print("SELF-AUDIT: No Lookahead, No Overfitting")
    print("=" * 80)
    print(f"\nTests: entry delay, declustering, shuffle ({N_SHUFFLE}x), bootstrap CI,")
    print(f"       effective N, monthly regime, Bonferroni correction")
    print(f"Period: {START} → {END}")
    print()
    
    btc = load_btc()
    
    # ================================================================
    # Test the TOP signals from each idea
    # ================================================================
    test_symbols = ['SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
                    'LINKUSDT', 'ADAUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT',
                    'NEARUSDT', 'ATOMUSDT', 'DOTUSDT', 'HYPEUSDT', 'WIFUSDT',
                    'OPUSDT', 'INJUSDT', 'TIAUSDT', 'SEIUSDT', 'MKRUSDT']
    
    all_audits = []
    t0 = time.time()
    
    for si, sym in enumerate(test_symbols):
        progress_bar(si, len(test_symbols), prefix='Auditing', start_time=t0)
        
        kf = load_kline(sym, START, END, spot=False)
        ks = load_kline(sym, START, END, spot=True)
        prem = load_premium(sym, START, END)
        
        if kf.empty or len(kf) < 10000:
            continue
        
        k = kf[['ts', 'close']].set_index('ts').sort_index()
        k = k[~k.index.duplicated(keep='first')]
        k.columns = ['close']
        k = k.join(btc, how='inner')
        
        if len(k) < 10000:
            continue
        
        # Build all forward returns
        for h in [15, 30, 60, 240]:
            k[f'fwd_{h}m'] = (k['close'].shift(-h) / k['close'] - 1) * 10000
        
        # ---- SIGNAL 4: BTC pump → long alt ----
        sig4 = k['btc_ret_3m'] > 150
        
        # ---- SIGNAL 5: Spot leads futures ----
        sig5 = pd.Series(False, index=k.index)
        if not ks.empty and len(ks) > 5000:
            ks_s = ks[['ts', 'close']].set_index('ts').sort_index()
            ks_s = ks_s[~ks_s.index.duplicated(keep='first')]
            ks_s.columns = ['spot_close']
            k = k.join(ks_s, how='left')
            if 'spot_close' in k.columns:
                k['spot_ret_3m'] = (k['spot_close'] / k['spot_close'].shift(3) - 1) * 10000
                k['fut_ret_3m'] = (k['close'] / k['close'].shift(3) - 1) * 10000
                sig5 = (k['spot_ret_3m'] - k['fut_ret_3m']) > 40
                sig5 = sig5.fillna(False)
        
        # ---- SIGNAL 3: High implied FR ----
        sig3 = pd.Series(False, index=k.index)
        if not prem.empty and len(prem) > 1000:
            prem_s = prem[['ts', 'close']].set_index('ts').sort_index()
            prem_s = prem_s[~prem_s.index.duplicated(keep='first')]
            prem_s.columns = ['premium']
            k = k.join(prem_s, how='left')
            k['premium'] = k['premium'].ffill()
            if 'premium' in k.columns:
                sig3 = (k['premium'] * 10000) > 20
                sig3 = sig3.fillna(False)
        
        # ---- COMBINED: Spot + HighIFR ----
        sig_combo = sig5 & sig3
        
        # Audit each signal at key horizons
        for horizon in [30, 60, 240]:
            fwd_col = f'fwd_{h}m'  
            fwd_col = f'fwd_{horizon}m'
            all_fwd = k[fwd_col].dropna()
            
            for sig_name, sig_mask in [
                (f'{sym}_Idea4_BTC_pump_{horizon}m', sig4),
                (f'{sym}_Idea5_spot_lead_{horizon}m', sig5),
                (f'{sym}_Idea3_high_IFR_{horizon}m', sig3),
                (f'{sym}_Combo_spot+IFR_{horizon}m', sig_combo),
            ]:
                if sig_mask.sum() < 5:
                    continue
                
                audit = audit_signal(sig_name, sig_mask, k[fwd_col], all_fwd, k.index)
                audit['symbol'] = sym
                audit['horizon_m'] = horizon
                all_audits.append(audit)
    
    progress_bar(len(test_symbols), len(test_symbols), prefix='Auditing', start_time=t0)
    
    if not all_audits:
        print("\n❌ No audit results!")
        return
    
    df = pd.DataFrame(all_audits)
    df.to_csv(f'{OUT}/self_audit_raw.csv', index=False)
    
    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)
    
    # Extract signal type from name
    df['sig_type'] = df['signal'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
    
    # Bonferroni correction
    n_tests = len(df)
    bonferroni_alpha = 0.05 / n_tests
    df['bonferroni_sig'] = df['shuffle_p'] < bonferroni_alpha
    
    # BH-FDR correction
    sorted_p = df['shuffle_p'].sort_values()
    m = len(sorted_p)
    bh_thresh = pd.Series([(i+1) / m * 0.05 for i in range(m)], index=sorted_p.index)
    df['bh_fdr_sig'] = False
    rejected = sorted_p <= bh_thresh
    if rejected.any():
        max_reject = rejected[rejected].index[-1]
        cutoff = bh_thresh[max_reject]
        df.loc[df['shuffle_p'] <= cutoff, 'bh_fdr_sig'] = True
    
    # ---- Print by signal type ----
    for sig_type in ['Idea4_BTC_pump', 'Idea5_spot_lead', 'Idea3_high_IFR', 'Combo_spot+IFR']:
        sub = df[df['sig_type'] == sig_type]
        if len(sub) == 0:
            continue
        
        print(f"\n{'─'*80}")
        print(f"  SIGNAL: {sig_type}")
        print(f"{'─'*80}")
        
        for horizon in [30, 60, 240]:
            hsub = sub[sub['horizon_m'] == horizon]
            if len(hsub) == 0:
                continue
            
            print(f"\n  [{horizon}m horizon] ({len(hsub)} coins)")
            
            # Raw vs delayed vs declustered
            raw = hsub['raw_net'].mean()
            delayed = hsub['delayed_net'].mean()
            declust = hsub['declust_net'].mean()
            dd_net = hsub['declust_delayed_net'].mean()
            delay_cost = hsub['delay_cost'].mean()
            
            print(f"    Raw (as reported):      {raw:+8.1f} bps  (n={hsub['raw_n'].mean():.0f} avg)")
            print(f"    + Entry delay (T+1):    {delayed:+8.1f} bps  (cost of 1-bar delay: {delay_cost:.1f} bps)")
            print(f"    + Declustered (30m gap): {declust:+8.1f} bps  (n={hsub['declust_n'].mean():.0f} avg, {hsub['cluster_ratio'].mean():.1f}x inflation)")
            print(f"    + BOTH (realistic):     {dd_net:+8.1f} bps  ← HONEST ESTIMATE")
            
            # Statistical significance
            avg_p = hsub['shuffle_p'].mean()
            bonf_pass = hsub['bonferroni_sig'].mean() * 100
            fdr_pass = hsub['bh_fdr_sig'].mean() * 100
            ci_lo = hsub['ci95_lo'].mean()
            ci_hi = hsub['ci95_hi'].mean()
            
            print(f"    Shuffle p-value:        {avg_p:.4f} (Bonferroni pass: {bonf_pass:.0f}%, BH-FDR pass: {fdr_pass:.0f}%)")
            print(f"    Bootstrap 95% CI net:   [{ci_lo:+.1f}, {ci_hi:+.1f}] bps")
            
            # Effective N
            eff_n = hsub['effective_n'].mean()
            n_infl = hsub['n_inflation'].mean()
            t_raw = hsub['t_stat_raw'].mean()
            t_adj = hsub['t_stat_adj'].mean()
            
            print(f"    Effective N:            {eff_n:.0f} (inflation: {n_infl:.1f}x)  t-stat: {t_raw:.2f} → {t_adj:.2f}")
            
            # Monthly
            mo_pct = hsub['months_pct_profitable'].mean()
            worst = hsub['worst_month_net'].mean()
            mo_sharpe = hsub['monthly_sharpe'].mean()
            
            print(f"    Monthly profitable:     {mo_pct:.0f}%  worst month: {worst:+.0f} bps  monthly Sharpe: {mo_sharpe:.2f}")
            
            # CI check
            if ci_lo > 0:
                print(f"    ✅ CI lower bound > 0 after fees")
            elif dd_net > 0:
                print(f"    ⚠️  Profitable but CI includes 0")
            else:
                print(f"    ❌ NOT profitable after realistic adjustments")
    
    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")
    
    print(f"\nTotal configs tested across all ideas: ~1500+")
    print(f"Bonferroni α threshold: {bonferroni_alpha:.6f}")
    print(f"Configs passing Bonferroni: {df['bonferroni_sig'].sum()}/{len(df)}")
    print(f"Configs passing BH-FDR: {df['bh_fdr_sig'].sum()}/{len(df)}")
    
    # Summary table: honest estimates
    print(f"\n{'Signal':<30s} {'H':>4s} {'Raw':>8s} {'Delayed':>8s} {'Declust':>8s} {'HONEST':>8s} {'ShufP':>7s} {'CI_lo':>7s} {'Mo%':>5s}")
    for sig_type in ['Idea4_BTC_pump', 'Idea5_spot_lead', 'Idea3_high_IFR', 'Combo_spot+IFR']:
        for horizon in [30, 60, 240]:
            sub = df[(df['sig_type'] == sig_type) & (df['horizon_m'] == horizon)]
            if len(sub) == 0:
                continue
            
            raw = sub['raw_net'].mean()
            delayed = sub['delayed_net'].mean()
            declust = sub['declust_net'].mean()
            honest = sub['declust_delayed_net'].mean()
            p = sub['shuffle_p'].mean()
            ci = sub['ci95_lo'].mean()
            mo = sub['months_pct_profitable'].mean()
            
            verdict = '✅' if honest > 0 and p < 0.05 and ci > -RT_TAKER_BPS else '⚠️' if honest > 0 else '❌'
            
            print(f"{verdict} {sig_type:<28s} {horizon:>4d}m {raw:>+8.1f} {delayed:>+8.1f} {declust:>+8.1f} {honest:>+8.1f} {p:>7.4f} {ci:>+7.1f} {mo:>4.0f}%")
    
    # Key findings
    print(f"\n\nKEY FINDINGS:")
    
    # Entry delay impact
    avg_delay = df['delay_cost'].mean()
    print(f"  1. Entry delay (1 bar = 1min) costs avg {avg_delay:.1f} bps — "
          f"{'SIGNIFICANT' if avg_delay > 50 else 'moderate' if avg_delay > 10 else 'negligible'}")
    
    # Signal clustering
    avg_cluster = df['cluster_ratio'].mean()
    print(f"  2. Signal clustering: avg {avg_cluster:.1f}x inflation in signal count")
    print(f"     Raw signals are NOT independent — same event triggers many consecutive bars")
    
    # Multiple comparisons
    print(f"  3. Multiple comparisons: {n_tests} tests run in this audit alone")
    print(f"     Full research tested ~1500+ configs — some winners expected by chance")
    
    # Regime concentration
    avg_mo_pct = df['months_pct_profitable'].mean()
    print(f"  4. Monthly consistency: avg {avg_mo_pct:.0f}% of months profitable")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
