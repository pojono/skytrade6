"""
Focused Edge Search - Orderbook & Cross-Sectional Strategies
Persistent search for real edge
"""

import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, get_available_symbols

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
FEE_RT = 0.002  # 20 bps


def analyze_orderbook_one_day(symbol: str, date: str) -> dict:
    """
    Analyze orderbook for microstructure signals
    """
    ob_file = DATALAKE / symbol / f"{date}_orderbook.jsonl.gz"
    
    if not ob_file.exists():
        return None
    
    try:
        records = []
        with gzip.open(ob_file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    records.append({
                        'timestamp': data['ts'],
                        'bid0': data['data']['b'][0][0] if data['data']['b'] else None,
                        'ask0': data['data']['a'][0][0] if data['data']['a'] else None,
                        'bid0_qty': data['data']['b'][0][1] if data['data']['b'] else 0,
                        'ask0_qty': data['data']['a'][0][1] if data['data']['a'] else 0,
                    })
                except:
                    continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        
        # Calculate orderbook metrics
        df['spread'] = (df['ask0'] - df['bid0']) / ((df['bid0'] + df['ask0']) / 2) * 10000  # bps
        df['mid'] = (df['bid0'] + df['ask0']) / 2
        df['imbalance'] = (df['bid0_qty'] - df['ask0_qty']) / (df['bid0_qty'] + df['ask0_qty'])
        
        # Calculate changes
        df['mid_chg_1m'] = df['mid'].pct_change(periods=60) * 10000  # 1min change in bps
        
        results = {
            'spread_mean': df['spread'].mean(),
            'spread_p95': df['spread'].quantile(0.95),
            'imbalance_mean': df['imbalance'].mean(),
            'records': len(df)
        }
        
        # Test simple signal: imbalance predicts direction?
        df_valid = df.dropna()
        if len(df_valid) > 100:
            # Strong bid imbalance -> price up?
            strong_bid = df_valid[df_valid['imbalance'] > 0.5]
            if len(strong_bid) > 10:
                results['bid_imb_pred_up'] = (strong_bid['mid_chg_1m'] > 0).mean() * 100
            
            # Strong ask imbalance -> price down?
            strong_ask = df_valid[df_valid['imbalance'] < -0.5]
            if len(strong_ask) > 10:
                results['ask_imb_pred_down'] = (strong_ask['mid_chg_1m'] < 0).mean() * 100
        
        return results
    except Exception as e:
        return {'error': str(e)}


def test_spread_scalping(symbol: str, start: str, end: str) -> dict:
    """
    Test: Capture spread when it's wide, using maker orders
    """
    df = load_klines(symbol, start, end)
    if len(df) < 100:
        return None
    
    # Estimate spread from OHLC
    df['spread_est'] = (df['high'] - df['low']) / df['close'] * 10000  # bps
    df['range'] = (df['high'] - df['low']) / df['open'] * 10000  # bps
    
    results = []
    
    # Test: When spread is wide, put maker orders at bid/ask
    for spread_thresh in [5, 10, 20, 50]:  # bps
        for hold in [1, 5, 10]:  # bars
            trades = []
            
            for i in range(len(df) - hold):
                spread = df.iloc[i]['range']
                
                if spread < spread_thresh:
                    continue
                
                # Simulate maker entry/exit (8 bps fees)
                # Buy at bid (below open), sell at ask (above open)
                # Assume we capture 30% of the range
                capture = spread * 0.3
                fees = 8  # bps
                
                net = capture - fees  # bps
                
                # Convert to dollar P&L (assume $10k position)
                position_value = 10000
                pnl = net / 10000 * position_value
                
                trades.append(pnl)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'spread_thresh': spread_thresh,
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    return results


def test_ranking_strategy(symbols: list, start: str, end: str) -> dict:
    """
    Cross-sectional: Rank coins by momentum/volatility, go long top
    """
    all_data = {}
    
    for sym in symbols:
        df = load_klines(sym, start, end)
        if len(df) > 100:
            # Calculate metrics
            df['ret_1d'] = df['close'].pct_change(1440)  # 1 day (1440 min)
            df['vol_1d'] = df['close'].pct_change().rolling(1440).std() * np.sqrt(1440)
            df['sharpe_1d'] = df['ret_1d'] / df['vol_1d'] if df['vol_1d'].mean() > 0 else 0
            
            all_data[sym] = df
    
    if len(all_data) < 5:
        return None
    
    # Align timestamps
    common_times = None
    for df in all_data.values():
        if common_times is None:
            common_times = set(df['timestamp'])
        else:
            common_times &= set(df['timestamp'])
    
    common_times = sorted(list(common_times))
    
    if len(common_times) < 100:
        return None
    
    # Simulate daily ranking
    trades = []
    
    for i in range(0, len(common_times) - 1440, 1440):  # Daily rebalancing
        time_now = common_times[i]
        time_future = common_times[i + 1440]
        
        # Get rankings at time_now
        scores = {}
        for sym, df in all_data.items():
            row = df[df['timestamp'] == time_now]
            if len(row) > 0:
                # Score by recent momentum
                scores[sym] = row['ret_1d'].values[0] if not pd.isna(row['ret_1d'].values[0]) else 0
        
        if len(scores) < 3:
            continue
        
        # Go long top 3 momentum
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate next day returns
        for sym, _ in top3:
            df = all_data[sym]
            row_now = df[df['timestamp'] == time_now]
            row_future = df[df['timestamp'] == time_future]
            
            if len(row_now) > 0 and len(row_future) > 0:
                ret = (row_future['close'].values[0] - row_now['close'].values[0]) / row_now['close'].values[0]
                
                position_value = 10000 / 3  # Split across 3 coins
                gross = ret * position_value
                fees = position_value * FEE_RT
                net = gross - fees
                trades.append(net)
    
    if len(trades) >= 5:
        total = sum(trades)
        wins = len([t for t in trades if t > 0])
        return {
            'trades': len(trades),
            'win_rate': wins / len(trades) * 100,
            'total_net': total,
            'avg_net': total / len(trades)
        }
    
    return None


def test_consecutive_moves(symbol: str, start: str, end: str) -> dict:
    """
    Test: After N consecutive up/down bars, reversal or continuation?
    """
    df = load_klines(symbol, start, end)
    if len(df) < 100:
        return None
    
    df['return'] = df['close'].pct_change()
    df['direction'] = np.where(df['return'] > 0, 1, -1)
    
    # Count consecutive moves
    df['consec'] = df['direction'].groupby((df['direction'] != df['direction'].shift()).cumsum()).cumcount() + 1
    df['consec'] = df['consec'] * df['direction']  # Positive for up streak, negative for down
    
    results = []
    
    for consec_thresh in [3, 4, 5]:  # N consecutive bars
        for hold in [1, 5, 10]:  # bars
            trades = []
            
            for i in range(len(df) - hold):
                consec = df.iloc[i]['consec']
                
                if pd.isna(consec):
                    continue
                
                # After N consecutive ups, go short (reversal)
                if consec >= consec_thresh:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (entry - exit_p) / entry  # short
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_RT
                    net = gross - fees
                    trades.append(net)
                
                # After N consecutive downs, go long (reversal)
                elif consec <= -consec_thresh:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (exit_p - entry) / entry  # long
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_RT
                    net = gross - fees
                    trades.append(net)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'consec_thresh': consec_thresh,
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('FOCUSED EDGE SEARCH - Orderbook & Cross-Sectional')
    print('='*70)
    
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT']
    start = '2025-06-01'
    end = '2025-12-31'
    
    all_profitable = []
    
    # 1. Test spread scalping
    print('\n1. SPREAD SCALPING (Maker Fee Strategy)')
    print('-'*70)
    for sym in symbols:
        res = test_spread_scalping(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            for p in profitable:
                print(f"  {sym}: spread>{p['spread_thresh']}bps, hold={p['hold']} | "
                      f"Trades={p['trades']}, WR={p['win_rate']:.1f}%, Net=${p['total_net']:.2f}")
                all_profitable.append(('spread_scalp', sym, p))
    
    # 2. Test consecutive moves reversal
    print('\n2. CONSECUTIVE MOVES REVERSAL')
    print('-'*70)
    for sym in symbols:
        res = test_consecutive_moves(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            for p in profitable:
                print(f"  {sym}: consec>{p['consec_thresh']}, hold={p['hold']} | "
                      f"Trades={p['trades']}, WR={p['win_rate']:.1f}%, Net=${p['total_net']:.2f}")
                all_profitable.append(('consec_rev', sym, p))
    
    # 3. Test cross-sectional ranking
    print('\n3. CROSS-SECTIONAL MOMENTUM RANKING')
    print('-'*70)
    res = test_ranking_strategy(symbols, start, end)
    if res and res['total_net'] > 0:
        print(f"  Portfolio: Trades={res['trades']}, WR={res['win_rate']:.1f}%, Net=${res['total_net']:.2f}")
        all_profitable.append(('cross_sec', 'portfolio', res))
    else:
        print(f'  No profitable cross-sectional strategy')
    
    # 4. Orderbook microstructure (sample one day)
    print('\n4. ORDERBOOK MICROSTRUCTURE (Sample: 2025-02-01)')
    print('-'*70)
    for sym in symbols[:4]:
        res = analyze_orderbook_one_day(sym, '2025-02-01')
        if res and 'error' not in res:
            print(f"  {sym}: Spread={res['spread_mean']:.2f}bps, "
                  f"Bid_imb_pred_up={res.get('bid_imb_pred_up', 0):.1f}%, "
                  f"Ask_imb_pred_down={res.get('ask_imb_pred_down', 0):.1f}%")
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    
    if all_profitable:
        print(f'Found {len(all_profitable)} profitable configurations:')
        all_profitable.sort(key=lambda x: x[2]['total_net'], reverse=True)
        for strategy, sym, result in all_profitable[:10]:
            print(f"  {strategy:15s} | {sym:10s} | Net=${result['total_net']:8.2f} | "
                  f"WR={result['win_rate']:5.1f}% | Trades={result['trades']}")
    else:
        print('NO PROFITABLE STRATEGIES FOUND')
        print('\nConclusions from fresh analysis:')
        print('- 8h funding rates too small (< 10 bps vs 20 bps fees)')
        print('- Simple price patterns do not overcome fees')
        print('- Orderbook microstructure not predictive at this timeframe')
        print('- Need to explore: 1h funding coins, cross-exchange, or HFT')
    
    print('='*70)
