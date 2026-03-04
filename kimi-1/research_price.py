"""
Kimi-1 Price-Based Strategy Research
Focus on momentum, volatility breakout, and mean reversion strategies.
"""
import pandas as pd
import numpy as np
from framework import DataLoader, BacktestEngine, Strategy
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy using EMA crossovers with volatility filter.
    Entry: Price > EMA_fast and EMA_fast > EMA_slow and volatility > threshold
    Exit: Price < EMA_fast or stop loss hit
    """
    
    def __init__(self, fast_ema=20, slow_ema=50, vol_lookback=20, vol_threshold=0.5):
        super().__init__("Momentum_EMA")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        klines = data.get('klines', pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # Calculate volatility (ATR-based)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.vol_lookback).mean()
        df['volatility'] = df['atr'] / df['close']
        
        vol_median = df['volatility'].median()
        
        # Generate signals
        df['signal'] = 0
        
        position = 0
        for i in range(self.slow_ema + 1, len(df)):
            price = df['close'].iloc[i]
            ema_f = df['ema_fast'].iloc[i]
            ema_s = df['ema_slow'].iloc[i]
            vol = df['volatility'].iloc[i]
            
            if position == 0:
                # Entry: Bullish crossover with elevated volatility
                if price > ema_f and ema_f > ema_s and vol > vol_median * self.vol_threshold:
                    df.loc[df.index[i], 'signal'] = 1
                    position = 1
            else:
                # Exit: Bearish crossover
                if price < ema_f or ema_f < ema_s:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                else:
                    df.loc[df.index[i], 'signal'] = 1
        
        return df


class MeanReversionStrategy(Strategy):
    """
    Mean reversion after large moves.
    Entry: Price deviates > threshold from moving average (RSI extreme)
    Exit: Price returns to mean
    """
    
    def __init__(self, lookback=20, entry_threshold=2.0, exit_threshold=0.5):
        super().__init__("MeanReversion")
        self.lookback = lookback
        self.entry_threshold = entry_threshold  # Std devs from mean
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        klines = data.get('klines', pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate z-score
        df['ma'] = df['close'].rolling(window=self.lookback).mean()
        df['std'] = df['close'].rolling(window=self.lookback).std()
        df['zscore'] = (df['close'] - df['ma']) / df['std']
        
        # Generate signals
        df['signal'] = 0
        
        position = 0
        for i in range(self.lookback + 1, len(df)):
            zscore = df['zscore'].iloc[i]
            
            if position == 0:
                # Entry: Extreme deviation (short overbought, long oversold)
                if zscore < -self.entry_threshold:
                    df.loc[df.index[i], 'signal'] = 1  # Long
                    position = 1
                elif zscore > self.entry_threshold:
                    df.loc[df.index[i], 'signal'] = -1  # Short
                    position = -1
            elif position == 1:
                # Exit long when price reverts
                if zscore > -self.exit_threshold:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                else:
                    df.loc[df.index[i], 'signal'] = 1
            elif position == -1:
                # Exit short when price reverts
                if zscore < self.exit_threshold:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                else:
                    df.loc[df.index[i], 'signal'] = -1
        
        return df


class VolatilityBreakoutStrategy(Strategy):
    """
    Breakout when price breaks above/below volatility bands.
    Uses Bollinger Bands with volume confirmation.
    """
    
    def __init__(self, lookback=20, num_std=2.0, min_volume_percentile=50):
        super().__init__("VolBreakout")
        self.lookback = lookback
        self.num_std = num_std
        self.min_volume_percentile = min_volume_percentile
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        klines = data.get('klines', pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate Bollinger Bands
        df['ma'] = df['close'].rolling(window=self.lookback).mean()
        df['std'] = df['close'].rolling(window=self.lookback).std()
        df['upper'] = df['ma'] + self.num_std * df['std']
        df['lower'] = df['ma'] - self.num_std * df['std']
        
        # Volume filter
        df['volume_sma'] = df['volume'].rolling(window=self.lookback).mean()
        df['high_volume'] = df['volume'] > df['volume_sma']
        
        # Generate signals
        df['signal'] = 0
        
        position = 0
        for i in range(self.lookback + 1, len(df)):
            close = df['close'].iloc[i]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]
            ma = df['ma'].iloc[i]
            high_vol = df['high_volume'].iloc[i]
            
            if position == 0:
                # Entry: Breakout with volume
                if close > upper and high_vol:
                    df.loc[df.index[i], 'signal'] = 1
                    position = 1
                elif close < lower and high_vol:
                    df.loc[df.index[i], 'signal'] = -1
                    position = -1
            elif position == 1:
                # Exit: Return to mean
                if close < ma:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                else:
                    df.loc[df.index[i], 'signal'] = 1
            elif position == -1:
                if close > ma:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                else:
                    df.loc[df.index[i], 'signal'] = -1
        
        return df


def quick_backtest(strategy, symbol, exchange, start, end):
    """Run quick backtest for a strategy."""
    loader = DataLoader()
    
    klines = loader.load_klines(exchange, symbol, start, end)
    if len(klines) < 1000:
        return None
    
    data = {'klines': klines}
    
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(strategy, data, position_size=1.0)
    
    if result.total_trades < 10:
        return None
    
    net_pnl = sum(t.pnl_net for t in result.trades)
    total_fees = sum(t.fees for t in result.trades)
    
    return {
        'strategy': strategy.name,
        'symbol': symbol,
        'trades': result.total_trades,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'total_return': result.total_return,
        'max_dd': result.max_drawdown,
        'sharpe': result.sharpe_ratio,
        'net_pnl': net_pnl,
        'fees': total_fees,
        'edge_after_fees': net_pnl - total_fees,
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 PRICE-BASED STRATEGY RESEARCH")
    print("=" * 100)
    
    # Test symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT']
    exchange = 'bybit'
    start = '2025-07-01'
    end = '2025-12-31'
    
    # Strategy configurations
    strategies = [
        ('Momentum_20_50', MomentumStrategy(fast_ema=20, slow_ema=50)),
        ('Momentum_10_30', MomentumStrategy(fast_ema=10, slow_ema=30)),
        ('MeanRev_2std', MeanReversionStrategy(entry_threshold=2.0)),
        ('MeanRev_1.5std', MeanReversionStrategy(entry_threshold=1.5)),
        ('VolBreakout_2std', VolatilityBreakoutStrategy(num_std=2.0)),
        ('VolBreakout_1.5std', VolatilityBreakoutStrategy(num_std=1.5)),
    ]
    
    results = []
    
    print(f"\nTesting {len(strategies)} strategies on {len(symbols)} symbols...")
    print("=" * 100)
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        for name, strat in strategies:
            result = quick_backtest(strat, symbol, exchange, start, end)
            if result:
                results.append(result)
                print(f"  {name:20s} | {result['trades']:3d} trades | "
                      f"WR={result['win_rate']:5.1%} | PF={result['profit_factor']:5.2f} | "
                      f"Net=${result['net_pnl']:8.2f} | Edge=${result['edge_after_fees']:8.2f}")
    
    print("\n" + "=" * 100)
    print("SUMMARY - TOP PERFORMERS")
    print("=" * 100)
    
    if results:
        df = pd.DataFrame(results)
        
        # Sort by edge after fees
        df_sorted = df.sort_values('edge_after_fees', ascending=False)
        
        print("\nTop 10 by Edge (Net P&L - Fees):")
        for idx, row in df_sorted.head(10).iterrows():
            print(f"  {row['symbol']:10s} | {row['strategy']:20s} | "
                  f"Trades={row['trades']:3d} | WR={row['win_rate']:5.1%} | "
                  f"PF={row['profit_factor']:5.2f} | Sharpe={row['sharpe']:5.2f} | "
                  f"Edge=${row['edge_after_fees']:8.2f}")
        
        # Profitable configs
        profitable = df[df['edge_after_fees'] > 0]
        print(f"\nProfitable configurations: {len(profitable)}/{len(df)} ({len(profitable)/len(df):.1%})")
        
        if len(profitable) > 0:
            print("\nProfitable after fees:")
            for idx, row in profitable.head(5).iterrows():
                print(f"  {row['symbol']:10s} | {row['strategy']:20s} | Edge=${row['edge_after_fees']:8.2f}")
        
        # Save results
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/price_strategy_results.csv', index=False)
        print(f"\nResults saved to price_strategy_results.csv")
    else:
        print("No results generated.")
    
    print("\n" + "=" * 100)
    print("RESEARCH COMPLETE")
    print("=" * 100)
