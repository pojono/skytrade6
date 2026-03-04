"""
Kimi-1 Cross-Exchange FR Arbitrage Research
Test FR differential between Bybit and Binance.
"""
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

ROUND_TRIP_FEE = 0.002  # 0.2% taker round-trip

def load_fr_data(symbol, date, exchange):
    """Load funding rate data for a specific exchange."""
    try:
        if exchange == 'binance':
            # Binance stores funding in metrics
            file = f'/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/{date}_metrics.csv'
            df = pd.read_csv(file)
            if 'funding_rate' in df.columns:
                df = df[['timestamp', 'funding_rate']].copy()
                df.rename(columns={'funding_rate': 'fr'}, inplace=True)
            else:
                return None
        else:
            file = f'/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/{date}_funding_rate.csv'
            df = pd.read_csv(file)
            df.rename(columns={'fundingRate': 'fr'}, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['fr_bps'] = df['fr'] * 10000
        return df[['timestamp', 'fr_bps']]
    except Exception as e:
        return None


def test_cross_exchange_arb(symbol, date, entry_diff_bps=5.0, exit_diff_bps=2.0):
    """
    Test cross-exchange FR arbitrage.
    When FR_diff > threshold, go long on lower FR exchange, short on higher.
    """
    fr_bb = load_fr_data(symbol, date, 'bybit')
    fr_bn = load_fr_data(symbol, date, 'binance')
    
    if fr_bb is None or fr_bn is None:
        return None
    
    if len(fr_bb) < 2 or len(fr_bn) < 2:
        return None
    
    # Merge on timestamp (approximate)
    merged = pd.merge_asof(
        fr_bb.sort_values('timestamp'),
        fr_bn.sort_values('timestamp'),
        on='timestamp',
        suffixes=('_bb', '_bn'),
        tolerance=pd.Timedelta('1H')
    )
    
    if merged.empty:
        return None
    
    merged['fr_diff'] = abs(merged['fr_bps_bb'] - merged['fr_bps_bn'])
    merged['long_bb'] = merged['fr_bps_bb'] < merged['fr_bps_bn']  # Long BB if FR lower
    
    # Simulate trades (simplified: just capture FR differential)
    trades = []
    position = 0
    entry_diff = 0
    long_exchange = None
    
    for _, row in merged.iterrows():
        diff = row['fr_diff']
        
        if position == 0 and diff >= entry_diff_bps:
            position = 1
            entry_diff = diff
            long_exchange = 'bb' if row['long_bb'] else 'bn'
        
        elif position == 1 and diff <= exit_diff_bps:
            # Profit = captured differential - fees
            # Assume we capture 50% of the differential
            captured = entry_diff * 0.5
            fees = 20  # 20 bps round trip
            pnl = captured - fees
            
            trades.append({
                'captured_bps': captured,
                'fees_bps': fees,
                'pnl_bps': pnl,
                'long_ex': long_exchange,
                'won': pnl > 0
            })
            position = 0
    
    if len(trades) < 2:
        return None
    
    total_pnl_bps = sum(t['pnl_bps'] for t in trades)
    wins = sum(t['won'] for t in trades)
    
    return {
        'symbol': symbol,
        'date': date,
        'entry_diff': entry_diff_bps,
        'exit_diff': exit_diff_bps,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl_bps': total_pnl_bps,
        'avg_trade_bps': total_pnl_bps / len(trades)
    }


def test_price_momentum(symbol, date, lookback=60, threshold=0.005):
    """Test simple price momentum on 1m data."""
    try:
        file = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{date}_kline_1m.csv'
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    except:
        return None
    
    if len(df) < lookback + 10:
        return None
    
    # Calculate momentum (no lookahead)
    df['ret'] = df['close'].pct_change()
    df['momentum'] = df['ret'].rolling(window=lookback).sum().shift(1)
    
    trades = []
    position = 0
    entry_price = 0
    max_bars = 30
    bars_held = 0
    
    for i in range(lookback + 1, len(df)):
        mom = df['momentum'].iloc[i]
        price = df['close'].iloc[i]
        
        if position == 0 and mom > threshold:
            position = 1
            entry_price = price
            bars_held = 0
        
        elif position == 1:
            bars_held += 1
            pnl_pct = (price - entry_price) / entry_price
            
            # Exit on profit target, stop loss, or max bars
            exit_signal = (pnl_pct > 0.003) or (pnl_pct < -0.003) or (bars_held >= max_bars)
            
            if exit_signal:
                pnl_gross = pnl_pct * 10000
                fees = 10000 * ROUND_TRIP_FEE
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'pnl_net': pnl_net,
                    'won': pnl_net > 0
                })
                position = 0
    
    if len(trades) < 3:
        return None
    
    total_pnl = sum(t['pnl_net'] for t in trades)
    wins = sum(t['won'] for t in trades)
    
    return {
        'symbol': symbol,
        'date': date,
        'strategy': 'momentum',
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'fees': len(trades) * 10000 * ROUND_TRIP_FEE,
        'edge': total_pnl - len(trades) * 10000 * ROUND_TRIP_FEE,
        'avg_trade': total_pnl / len(trades)
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 CROSS-EXCHANGE & MOMENTUM RESEARCH")
    print("=" * 100)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    dates = ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05']
    
    # Test 1: Cross-exchange FR arb
    print("\n" + "=" * 100)
    print("TEST 1: Cross-Exchange FR Arbitrage")
    print("=" * 100)
    
    arb_results = []
    for symbol in symbols:
        for date in dates:
            for entry_diff in [3, 5, 10]:
                result = test_cross_exchange_arb(symbol, date, entry_diff, entry_diff * 0.3)
                if result and result['trades'] >= 2:
                    arb_results.append(result)
                    if result['total_pnl_bps'] > 0:
                        print(f"{symbol:12s} {date} | Diff>={entry_diff}bps | "
                              f"{result['trades']} trades | WR={result['win_rate']:.0%} | "
                              f"PnL={result['total_pnl_bps']:.1f}bps")
    
    if arb_results:
        df_arb = pd.DataFrame(arb_results)
        print(f"\nCross-exchange results: {len(arb_results)} tests")
        print(f"Profitable: {(df_arb['total_pnl_bps'] > 0).sum()}/{len(df_arb)}")
    
    # Test 2: Momentum
    print("\n" + "=" * 100)
    print("TEST 2: Price Momentum (1m bars)")
    print("=" * 100)
    
    mom_results = []
    for symbol in symbols:
        print(f"\n{symbol}:")
        for date in dates:
            result = test_price_momentum(symbol, date, lookback=60, threshold=0.005)
            if result and result['trades'] >= 3:
                mom_results.append(result)
                edge_str = f"Edge=${result['edge']:.2f}" if result['edge'] > 0 else f"Loss=${result['edge']:.2f}"
                print(f"  {date} | {result['trades']} trades | "
                      f"WR={result['win_rate']:.0%} | Net=${result['total_pnl']:.2f} | {edge_str}")
    
    if mom_results:
        df_mom = pd.DataFrame(mom_results)
        print(f"\nMomentum results: {len(mom_results)} tests")
        profitable = df_mom[df_mom['edge'] > 0]
        print(f"Profitable after fees: {len(profitable)}/{len(df_mom)}")
        
        if len(profitable) > 0:
            print("\nTop 5 by edge:")
            for _, r in profitable.nlargest(5, 'edge').iterrows():
                print(f"  {r['symbol']:12s} {r['date']} | {r['trades']} trades | Edge=${r['edge']:.2f}")
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    
    if not arb_results and not mom_results:
        print("\nNo profitable edges found in quick tests.")
        print("\nPossible reasons:")
        print("  1. Date range too short (need more data for statistical significance)")
        print("  2. Parameters need optimization")
        print("  3. Need different strategy types (OI + FR, liquidation cascades)")
        print("\nRecommendations:")
        print("  - Test on 3-6 months of data")
        print("  - Walk-forward optimization")
        print("  - Use 1h funding rate coins if available")
        print("  - Combine multiple signals (FR + OI + price)")
    
    print("\n" + "=" * 100)
