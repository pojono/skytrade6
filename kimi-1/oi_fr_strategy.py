"""
Kimi-1 OI + FR Combined Strategy
Test if Open Interest changes + Funding Rate signals create edge.
"""
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ROUND_TRIP_FEE = 0.002

def load_oi_fr(symbol, date):
    """Load OI and FR for a symbol on a date."""
    try:
        # Load OI
        oi_file = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{date}_open_interest_5min.csv'
        oi = pd.read_csv(oi_file)
        oi['timestamp'] = pd.to_datetime(oi['timestamp'], unit='ms')
        oi = oi.sort_values('timestamp')
        oi['oi_change'] = oi['openInterest'].pct_change() * 100  # % change
        
        # Load FR
        fr_file = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{date}_funding_rate.csv'
        fr = pd.read_csv(fr_file)
        fr['timestamp'] = pd.to_datetime(fr['timestamp'], unit='ms')
        fr['fr_bps'] = fr['fundingRate'] * 10000
        
        return oi, fr
    except:
        return None, None


def test_oi_fr_strategy(symbol, date, oi_threshold=5.0, fr_threshold=1.0):
    """
    Strategy: When OI increases > threshold and FR > threshold, go long.
    Theory: Rising OI + high FR = crowded longs, but momentum continues short-term.
    """
    oi, fr = load_oi_fr(symbol, date)
    if oi is None or fr is None:
        return None
    
    # Resample OI to hourly for cleaner signals
    oi_hourly = oi.set_index('timestamp').resample('1H')['oi_change'].last().reset_index()
    
    # Merge with FR
    merged = pd.merge_asof(
        oi_hourly.sort_values('timestamp'),
        fr[['timestamp', 'fr_bps']].sort_values('timestamp'),
        on='timestamp',
        tolerance=pd.Timedelta('2H')
    )
    
    merged = merged.dropna()
    if len(merged) < 3:
        return None
    
    # Generate signals
    merged['signal'] = (merged['oi_change'] > oi_threshold) & (merged['fr_bps'] > fr_threshold)
    
    # Simulate trades (simplified: signal = entry, hold for 4 hours)
    trades = []
    position = 0
    entry_time = None
    
    for i in range(len(merged) - 4):  # Need room for exit
        if position == 0 and merged['signal'].iloc[i]:
            position = 1
            entry_time = merged['timestamp'].iloc[i]
            entry_fr = merged['fr_bps'].iloc[i]
            
            # Exit 4 periods later
            exit_fr = merged['fr_bps'].iloc[i + 4]
            
            # P&L: collected funding over 4 periods minus fees
            avg_fr = (entry_fr + exit_fr) / 2
            gross_bps = avg_fr * 4
            fee_bps = 20
            net_bps = gross_bps - fee_bps
            
            trades.append({
                'entry_fr': entry_fr,
                'exit_fr': exit_fr,
                'gross_bps': gross_bps,
                'net_bps': net_bps,
                'pnl': net_bps,  # Simplified: $1 per bps
                'won': net_bps > 0
            })
            position = 0
    
    if len(trades) < 3:
        return None
    
    wins = sum(t['won'] for t in trades)
    total_pnl = sum(t['pnl'] for t in trades)
    
    return {
        'symbol': symbol,
        'date': date,
        'oi_thresh': oi_threshold,
        'fr_thresh': fr_threshold,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'avg_trade': total_pnl / len(trades)
    }


def test_fr_contrarian(symbol, start_date, end_date, entry_bps=5.0):
    """
    Contrarian strategy: When FR spikes high, short (expect reversion).
    Entry: FR > entry_bps (extreme)
    Exit: FR normalizes
    """
    all_trades = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        try:
            fr_file = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{date_str}_funding_rate.csv'
            fr = pd.read_csv(fr_file)
            fr['timestamp'] = pd.to_datetime(fr['timestamp'], unit='ms')
            fr['fr_bps'] = fr['fundingRate'] * 10000
            
            for i in range(len(fr) - 1):
                if fr['fr_bps'].iloc[i] > entry_bps:
                    # Short high FR, exit next period
                    entry_fr = fr['fr_bps'].iloc[i]
                    exit_fr = fr['fr_bps'].iloc[i + 1]
                    
                    # Contrarian profit: if FR drops, we win
                    fr_change = entry_fr - exit_fr  # Profit if FR drops
                    
                    # Assume we capture 50% of FR change
                    gross_bps = fr_change * 0.5
                    fee_bps = 20
                    net_bps = gross_bps - fee_bps
                    
                    all_trades.append({
                        'net_bps': net_bps,
                        'won': net_bps > 0
                    })
        except:
            pass
        
        current += timedelta(days=1)
    
    if len(all_trades) < 5:
        return None
    
    wins = sum(t['won'] for t in all_trades)
    total_pnl = sum(t['net_bps'] for t in all_trades)
    
    return {
        'symbol': symbol,
        'strategy': 'fr_contrarian',
        'entry_bps': entry_bps,
        'trades': len(all_trades),
        'win_rate': wins / len(all_trades),
        'total_pnl': total_pnl,
        'avg_trade': total_pnl / len(all_trades)
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 OI + FR COMBINED STRATEGY")
    print("=" * 100)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    dates = ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05']
    
    # Test 1: OI + FR combined
    print("\n" + "=" * 100)
    print("TEST 1: OI Change + FR Combined")
    print("=" * 100)
    
    oi_results = []
    for symbol in symbols:
        print(f"\n{symbol}:")
        for date in dates:
            for oi_thresh in [3, 5, 10]:
                for fr_thresh in [0.5, 1.0, 2.0]:
                    result = test_oi_fr_strategy(symbol, date, oi_thresh, fr_thresh)
                    if result and result['trades'] >= 2:
                        oi_results.append(result)
                        if result['total_pnl'] > 0:
                            print(f"  {date} OI>{oi_thresh}% FR>{fr_thresh}bps | "
                                  f"{result['trades']} trades | WR={result['win_rate']:.0%} | "
                                  f"PnL={result['total_pnl']:.1f}bps")
    
    # Test 2: FR Contrarian
    print("\n" + "=" * 100)
    print("TEST 2: FR Contrarian (Short High FR)")
    print("=" * 100)
    
    contrarian_results = []
    for symbol in symbols:
        print(f"\n{symbol}:")
        for entry_bps in [3, 5, 10]:
            result = test_fr_contrarian(symbol, '2025-07-01', '2025-12-31', entry_bps)
            if result:
                contrarian_results.append(result)
                status = "PROFIT" if result['avg_trade'] > 0 else "LOSS"
                print(f"  Entry>={entry_bps}bps | {result['trades']} trades | "
                      f"WR={result['win_rate']:.0%} | Avg={result['avg_trade']:+.1f}bps | {status}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if oi_results:
        df_oi = pd.DataFrame(oi_results)
        profitable_oi = df_oi[df_oi['total_pnl'] > 0]
        print(f"\nOI+FR Results: {len(profitable_oi)}/{len(df_oi)} profitable")
    
    if contrarian_results:
        df_con = pd.DataFrame(contrarian_results)
        profitable_con = df_con[df_con['avg_trade'] > 0]
        print(f"Contrarian Results: {len(profitable_con)}/{len(df_con)} profitable")
        
        if len(profitable_con) > 0:
            print("\nProfitable contrarian configs:")
            for _, r in profitable_con.iterrows():
                print(f"  {r['symbol']:12s} E={r['entry_bps']}bps | "
                      f"{r['trades']} trades | PnL={r['total_pnl']:.1f}bps")
    
    print("\n" + "=" * 100)
