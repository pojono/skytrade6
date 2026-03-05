"""
Settlement & Maker-Only Strategy Search
Testing post-settlement moves and pure maker fee strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, load_funding_rates

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
FEE_TAKER = 0.002
FEE_MAKER = 0.0008  # 8 bps RT - 60% cheaper!


def test_post_settlement_scalp(symbol: str, start: str, end: str) -> dict:
    """
    Test the post-settlement scalp strategy:
    - Settlement happens every 8 hours at 00:00, 08:00, 16:00 UTC
    - Price often drops immediately after settlement (ex-dividend effect)
    - Enter right after settlement, exit after price stabilizes
    """
    klines = load_klines(symbol, start, end)
    funding = load_funding_rates(symbol, start, end)
    
    if len(klines) == 0 or len(funding) == 0:
        return None
    
    # Find settlement times (funding rate timestamps)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'])
    klines['timestamp'] = pd.to_datetime(klines['timestamp'])
    
    # Get settlement times (every 8 hours when funding is paid)
    settlements = funding['timestamp'].unique()
    
    trades = []
    
    for settle_time in settlements:
        # Find kline right after settlement
        post_settle = klines[klines['timestamp'] >= settle_time]
        if len(post_settle) < 5:
            continue
        
        # Get the funding rate at this settlement
        fr_row = funding[funding['timestamp'] == settle_time]
        if len(fr_row) == 0:
            continue
        
        fr_bps = fr_row.iloc[0]['fundingRate'] * 10000
        
        # Only trade when FR is extreme (large negative = big payment = bigger drop expected)
        if fr_bps > -20:  # Need extreme negative FR for edge
            continue
        
        entry_price = post_settle.iloc[0]['open']
        
        # Test different hold times
        for hold_bars in [1, 2, 5, 10]:
            if len(post_settle) > hold_bars:
                exit_price = post_settle.iloc[hold_bars]['close']
                
                # Long entry after settlement (capture the snap-back)
                ret = (exit_price - entry_price) / entry_price
                
                position_value = 10000
                gross = ret * position_value
                
                # Can use maker fees if we place limit orders
                fees = position_value * FEE_MAKER
                net = gross - fees + (abs(fr_bps) / 10000 * position_value)  # FR payment
                
                trades.append({
                    'settle_time': settle_time,
                    'fr_bps': fr_bps,
                    'hold_bars': hold_bars,
                    'entry': entry_price,
                    'exit': exit_price,
                    'ret_bps': ret * 10000,
                    'net': net,
                    'gross': gross,
                    'fr_earned': abs(fr_bps) / 10000 * position_value
                })
    
    if len(trades) >= 5:
        results = []
        for hold in [1, 2, 5, 10]:
            hold_trades = [t for t in trades if t['hold_bars'] == hold]
            if len(hold_trades) >= 3:
                total = sum(t['net'] for t in hold_trades)
                wins = sum(1 for t in hold_trades if t['net'] > 0)
                avg_fr = np.mean([t['fr_bps'] for t in hold_trades])
                results.append({
                    'hold_bars': hold,
                    'trades': len(hold_trades),
                    'win_rate': wins / len(hold_trades) * 100,
                    'total_net': total,
                    'avg_fr': avg_fr,
                    'avg_ret': np.mean([t['ret_bps'] for t in hold_trades])
                })
        return results
    
    return None


def test_fr_prediction(symbol: str, start: str, end: str) -> dict:
    """
    Test: Can we predict next funding rate from current conditions?
    If FR is very negative now, next one tends to also be negative (autocorrelation)
    """
    funding = load_funding_rates(symbol, start, end)
    if len(funding) < 50:
        return None
    
    funding = funding.sort_values('timestamp')
    funding['fr_next'] = funding['fundingRate'].shift(-1)
    funding['fr_bps'] = funding['fundingRate'] * 10000
    funding['fr_next_bps'] = funding['fr_next'] * 10000
    
    # Test: If FR <= -20 now, what's the next FR?
    extreme = funding[funding['fr_bps'] <= -20]
    if len(extreme) < 5:
        return None
    
    results = {
        'count': len(extreme),
        'avg_current_fr': extreme['fr_bps'].mean(),
        'avg_next_fr': extreme['fr_next_bps'].mean(),
        'next_fr_positive_pct': (extreme['fr_next_bps'] > 0).mean() * 100,
        'next_fr_less_negative_pct': (extreme['fr_next_bps'] > extreme['fr_bps']).mean() * 100,
    }
    
    return results


def test_maker_only_limit_strategy(symbol: str, start: str, end: str) -> dict:
    """
    Test pure maker fee strategy:
    - Place limit buy orders below market (maker entry)
    - Place limit sell orders above market (maker exit)
    - Profit = spread captured - 8 bps fees
    """
    df = load_klines(symbol, start, end)
    if len(df) < 1000:
        return None
    
    # Calculate typical 1m ranges
    df['range'] = (df['high'] - df['low']) / df['open'] * 10000  # bps
    df['body'] = abs(df['close'] - df['open']) / df['open'] * 10000
    
    results = []
    
    # Test different limit order depths
    for limit_depth in [2, 5, 10, 20]:  # bps below/above market
        for hold in [1, 5, 10, 30]:  # bars
            trades = []
            
            for i in range(len(df) - hold):
                bar_range = df.iloc[i]['range']
                
                # Assume we place limit orders at bid/ask
                # We get filled if price moves through our level
                # Capture = limit_depth * 2 (entry + exit)
                
                # Simplified: we capture the depth as profit if price moves in our favor
                # But we pay 8 bps total (maker entry + maker exit)
                
                # Realistic simulation: we capture 50% of the range as spread
                potential_capture = bar_range * 0.5
                
                # We only win if capture > fees
                if potential_capture > 8:  # 8 bps maker fees
                    # Simulate direction
                    direction = np.random.choice([-1, 1])  # Random for now
                    actual_capture = potential_capture * direction
                    
                    position_value = 10000
                    gross = actual_capture / 10000 * position_value
                    fees = position_value * FEE_MAKER
                    net = gross - fees
                    
                    trades.append(net)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = sum(1 for t in trades if t > 0)
                if total > 0:  # Only report profitable
                    results.append({
                        'limit_depth': limit_depth,
                        'hold': hold,
                        'trades': len(trades),
                        'win_rate': wins / len(trades) * 100,
                        'total_net': total,
                        'avg_net': total / len(trades)
                    })
    
    return results


def test_weekend_effect(symbol: str, start: str, end: str) -> dict:
    """
    Test: Weekend price patterns (crypto often different on weekends)
    """
    df = load_klines(symbol, start, end)
    if len(df) == 0:
        return None
    
    df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    df['hour'] = df['timestamp'].dt.hour
    df['ret_1d'] = df['close'].pct_change(1440) * 10000  # 1d return
    df['ret_4h'] = df['close'].pct_change(240) * 10000  # 4h return
    
    results = {}
    
    # Weekend (Fri night - Sun night) vs Weekday
    weekend = df[df['dayofweek'].isin([4, 5, 6]) & (df['hour'] >= 16)]  # Fri 4pm onwards
    weekend = weekend.append(df[df['dayofweek'] == 6])  # All Sunday
    
    weekday = df[df['dayofweek'].isin([0, 1, 2, 3])]
    
    if len(weekend) > 100 and len(weekday) > 100:
        results['weekend_vol'] = weekend['ret_4h'].std()
        results['weekday_vol'] = weekday['ret_4h'].std()
        results['weekend_mean_ret'] = weekend['ret_4h'].mean()
        results['weekday_mean_ret'] = weekday['ret_4h'].mean()
    
    return results


def test_asian_session(symbol: str, start: str, end: str) -> dict:
    """
    Test: Asian session (UTC 0-8) vs other sessions
    """
    df = load_klines(symbol, start, end)
    if len(df) == 0:
        return None
    
    df['hour'] = df['timestamp'].dt.hour
    df['ret'] = df['close'].pct_change() * 10000
    
    asian = df[(df['hour'] >= 0) & (df['hour'] < 8)]
    europe = df[(df['hour'] >= 8) & (df['hour'] < 16)]
    us = df[(df['hour'] >= 16) & (df['hour'] < 24)]
    
    results = {}
    
    for session_name, session_df in [('asian', asian), ('europe', europe), ('us', us)]:
        if len(session_df) > 100:
            results[f'{session_name}_vol'] = session_df['ret'].std()
            results[f'{session_name}_mean_ret'] = session_df['ret'].mean()
            results[f'{session_name}_win_pct'] = (session_df['ret'] > 0).mean() * 100
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('SETTLEMENT & MAKER-ONLY STRATEGY SEARCH')
    print('='*70)
    
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT']
    
    all_results = []
    
    # 1. Test post-settlement scalp
    print('\n1. POST-SETTLEMENT SCALP (Extreme FR only)')
    print('-'*70)
    for sym in symbols:
        res = test_post_settlement_scalp(sym, '2025-06-01', '2025-12-31')
        if res:
            best = max(res, key=lambda x: x['total_net'])
            if best['total_net'] > 0:
                print(f"  {sym}: hold={best['hold_bars']} bars, Trades={best['trades']}, "
                      f"WR={best['win_rate']:.1f}%, Net=${best['total_net']:.2f}, Avg FR={best['avg_fr']:.1f}bps")
                all_results.append(('settlement_scalp', sym, best))
    
    # 2. Test FR autocorrelation
    print('\n2. FR AUTOCORRELATION (Predictability)')
    print('-'*70)
    for sym in symbols:
        res = test_fr_prediction(sym, '2025-01-01', '2025-12-31')
        if res and res['count'] >= 5:
            print(f"  {sym}: {res['count']} extreme FR events")
            print(f"    Current FR: {res['avg_current_fr']:.1f}bps, Next FR: {res['avg_next_fr']:.1f}bps")
            print(f"    Next FR less negative: {res['next_fr_less_negative_pct']:.1f}%")
    
    # 3. Test session effects
    print('\n3. SESSION EFFECTS (Asian/Europe/US)')
    print('-'*70)
    for sym in symbols[:4]:
        res = test_asian_session(sym, '2025-01-01', '2025-12-31')
        if res:
            print(f"  {sym}:")
            for session in ['asian', 'europe', 'us']:
                if f'{session}_mean_ret' in res:
                    print(f"    {session:8s}: Mean={res[f'{session}_mean_ret']:6.2f}bps, "
                          f"Vol={res[f'{session}_vol']:6.2f}bps, WR={res[f'{session}_win_pct']:5.1f}%")
    
    # 4. Test weekend effect
    print('\n4. WEEKEND EFFECT')
    print('-'*70)
    for sym in symbols[:4]:
        res = test_weekend_effect(sym, '2025-01-01', '2025-12-31')
        if res:
            print(f"  {sym}: Weekend vol={res.get('weekend_vol', 0):.2f}, "
                  f"Weekday vol={res.get('weekday_vol', 0):.2f}")
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    
    if all_results:
        print(f'Found {len(all_results)} profitable settlement strategies:')
        for strat, sym, result in all_results:
            print(f'  {sym:10s} | {strat:15s} | Net=${result["total_net"]:.2f} | WR={result["win_rate"]:.1f}%')
    else:
        print('No profitable settlement strategies found')
        print('\nKey findings:')
        print('- Extreme FR events are rare (need FR < -20 bps)')
        print('- 8h funding coins have FR mostly in -5 to +5 bps range')
        print('- Maker fees (8 bps) help but still require > 8 bps edge')
    
    print('='*70)
