"""
Strategy implementations based on research findings.

Key insights from prior research:
1. Funding Rate HOLD (entry>=20bps, exit<8bps): 65-75% WR, profitable across exchanges
2. Post-settlement snap-back: -30 to -50 bps drop in first 100ms after settlement
3. FR autocorrelation is strong (r=0.6-0.7), enabling prediction
4. Maker 0.04%, Taker 0.1% - need 20+ bps edge for profitability
"""
import pandas as pd
import numpy as np
from framework import Strategy, DataLoader, BacktestEngine, walk_forward_validation
from typing import Dict, List, Optional, Tuple


class FundingRateHoldStrategy(Strategy):
    """
    Funding Rate Hold Strategy.
    
    Entry: When funding rate >= threshold (e.g., 20 bps)
    Exit: When funding rate < exit_threshold (e.g., 8 bps) or after N periods
    
    Based on research showing FR autocorrelation and profitability of holding
    positions to collect funding payments.
    """
    
    def __init__(self, entry_threshold_bps: float = 20.0,
                 exit_threshold_bps: float = 8.0,
                 max_hold_periods: int = 3,
                 use_ml_filter: bool = False):
        super().__init__("FR_Hold")
        self.entry_threshold = entry_threshold_bps / 10000  # Convert to decimal
        self.exit_threshold = exit_threshold_bps / 10000
        self.max_hold_periods = max_hold_periods
        self.use_ml_filter = use_ml_filter
        self.position_age = 0
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate long signals when funding rate is high (collecting funding)."""
        klines = data.get('klines', pd.DataFrame())
        funding = data.get('funding', pd.DataFrame())
        
        if klines.empty or funding.empty:
            return pd.DataFrame()
        
        # Merge klines with funding rates
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        funding['timestamp'] = pd.to_datetime(funding['timestamp'])
        
        # Forward-fill funding rates to match kline timestamps
        df = df.merge(funding[['timestamp', 'funding_rate']], on='timestamp', how='left')
        df['funding_rate'] = df['funding_rate'].ffill()
        
        # Generate signals
        df['signal'] = 0
        df['exit_signal'] = False
        
        position = 0
        entry_idx = 0
        
        for i in range(len(df)):
            fr = df.loc[df.index[i], 'funding_rate']
            
            if pd.isna(fr):
                continue
                
            # Entry condition: FR >= threshold and no position
            if position == 0 and fr >= self.entry_threshold:
                df.loc[df.index[i], 'signal'] = 1  # Long
                position = 1
                entry_idx = i
                self.position_age = 0
            
            # Exit conditions
            elif position == 1:
                self.position_age += 1
                
                exit_condition = (
                    fr < self.exit_threshold or  # FR dropped below threshold
                    self.position_age >= self.max_hold_periods  # Max hold time
                )
                
                if exit_condition:
                    df.loc[df.index[i], 'signal'] = 0
                    df.loc[df.index[i], 'exit_signal'] = True
                    position = 0
                    self.position_age = 0
                else:
                    df.loc[df.index[i], 'signal'] = 1  # Maintain position
        
        df.set_index('timestamp', inplace=True)
        return df


class SettlementSnapStrategy(Strategy):
    """
    Post-Settlement Snap-back Strategy.
    
    Based on research showing consistent price drops of 30-50 bps 
    in the first 100ms after settlement when FR <= -50 bps.
    
    Entry: Short immediately after settlement if FR <= threshold
    Exit: Close short after fixed time (e.g., 100ms - 500ms)
    
    Note: This requires millisecond-level data for accurate backtesting.
    """
    
    def __init__(self, fr_threshold_bps: float = -50.0,
                 entry_delay_ms: int = 10,
                 exit_delay_ms: int = 100,
                 min_spread_bps: float = 2.0):
        super().__init__("Settlement_Snap")
        self.fr_threshold = fr_threshold_bps / 10000
        self.entry_delay_ms = entry_delay_ms
        self.exit_delay_ms = exit_delay_ms
        self.min_spread = min_spread_bps / 10000
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate short signals after settlement when funding is very negative.
        """
        klines = data.get('klines', pd.DataFrame())
        funding = data.get('funding', pd.DataFrame())
        
        if klines.empty or funding.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        funding['timestamp'] = pd.to_datetime(funding['timestamp'])
        
        # Identify settlement times (typically every 8 hours: 00:00, 08:00, 16:00 UTC)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['is_settlement'] = (df['hour'].isin([0, 8, 16])) & (df['minute'] == 0)
        
        # Merge funding rates
        df = df.merge(funding[['timestamp', 'funding_rate']], on='timestamp', how='left')
        df['funding_rate'] = df['funding_rate'].ffill()
        
        # Generate signals
        df['signal'] = 0
        df['exit_signal'] = False
        
        position = 0
        entry_time = None
        bars_since_entry = 0
        
        for i in range(len(df)):
            # Entry: Just after settlement with very negative FR
            if position == 0 and df.loc[df.index[i], 'is_settlement']:
                fr = df.loc[df.index[i], 'funding_rate']
                if pd.notna(fr) and fr <= self.fr_threshold:
                    df.loc[df.index[i], 'signal'] = -1  # Short
                    position = -1
                    entry_time = df.loc[df.index[i], 'timestamp']
                    bars_since_entry = 0
            
            # Exit: After fixed hold period
            elif position == -1:
                bars_since_entry += 1
                # Exit after approximately 100ms (1-2 bars at 1-min resolution)
                if bars_since_entry >= 2:  # Hold for ~2 minutes (conservative)
                    df.loc[df.index[i], 'signal'] = 0
                    df.loc[df.index[i], 'exit_signal'] = True
                    position = 0
                    bars_since_entry = 0
                else:
                    df.loc[df.index[i], 'signal'] = -1
        
        df.set_index('timestamp', inplace=True)
        return df


class CrossExchangeFRArbStrategy(Strategy):
    """
    Cross-Exchange Funding Rate Arbitrage.
    
    When funding rates diverge significantly between exchanges,
    go long on the exchange with lower FR, short on the exchange with higher FR.
    
    Entry: FR differential >= 30 bps
    Exit: When differential < 10 bps or after 1 funding period
    """
    
    def __init__(self, entry_diff_bps: float = 30.0,
                 exit_diff_bps: float = 10.0,
                 exchanges: List[str] = None):
        super().__init__("CrossExchange_FR_Arb")
        self.entry_diff = entry_diff_bps / 10000
        self.exit_diff = exit_diff_bps / 10000
        self.exchanges = exchanges or ['bybit', 'binance', 'okx']
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Requires data from multiple exchanges with keys like:
        'bybit_klines', 'binance_klines', 'bybit_funding', etc.
        """
        # Extract funding data for each exchange
        funding_data = {}
        for ex in self.exchanges:
            fr_key = f'{ex}_funding'
            if fr_key in data and not data[fr_key].empty:
                funding_data[ex] = data[fr_key].copy()
                funding_data[ex]['timestamp'] = pd.to_datetime(funding_data[ex]['timestamp'])
        
        if len(funding_data) < 2:
            return pd.DataFrame()
        
        # Get primary klines (use first available exchange)
        primary_ex = list(funding_data.keys())[0]
        klines_key = f'{primary_ex}_klines'
        if klines_key not in data:
            return pd.DataFrame()
        
        df = data[klines_key].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge funding rates from all exchanges
        for ex, fr_df in funding_data.items():
            df = df.merge(fr_df[['timestamp', 'funding_rate']], 
                         on='timestamp', how='left', suffixes=('', f'_{ex}'))
            df[f'funding_rate_{ex}'] = df[f'funding_rate_{ex}'].ffill()
        
        # Calculate min/max FR across exchanges
        fr_cols = [f'funding_rate_{ex}' for ex in funding_data.keys()]
        df['fr_min'] = df[fr_cols].min(axis=1)
        df['fr_max'] = df[fr_cols].max(axis=1)
        df['fr_diff'] = df['fr_max'] - df['fr_min']
        
        # Generate signals (simplified - just use primary exchange for execution)
        df['signal'] = 0
        df['exit_signal'] = False
        
        position = 0
        
        for i in range(len(df)):
            diff = df.loc[df.index[i], 'fr_diff']
            
            if pd.isna(diff):
                continue
            
            # Entry: Significant FR differential
            if position == 0 and diff >= self.entry_diff:
                # Go long on exchange with lowest FR
                lowest_fr_ex = min(funding_data.keys(), 
                                  key=lambda x: df.loc[df.index[i], f'funding_rate_{x}'])
                if lowest_fr_ex == primary_ex:
                    df.loc[df.index[i], 'signal'] = 1
                else:
                    df.loc[df.index[i], 'signal'] = 0  # Skip if not primary
                position = 1
            
            # Exit: Differential narrowed
            elif position == 1 and diff <= self.exit_diff:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'exit_signal'] = True
                position = 0
            elif position == 1:
                df.loc[df.index[i], 'signal'] = 1
        
        df.set_index('timestamp', inplace=True)
        return df


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy with Regime Filter.
    
    Entry: Price breaks above/below volatility band (ATR-based)
    Filter: Only trade in trending regimes (not choppy/sideways)
    """
    
    def __init__(self, lookback: int = 20,
                 atr_multiplier: float = 1.5,
                 regime_lookback: int = 50,
                 trend_threshold: float = 0.3):
        super().__init__("Volatility_Breakout")
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.regime_lookback = regime_lookback
        self.trend_threshold = trend_threshold
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        klines = data.get('klines', pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.lookback).mean()
        
        # Calculate volatility bands
        df['middle'] = df['close'].rolling(window=self.lookback).mean()
        df['upper'] = df['middle'] + self.atr_multiplier * df['atr']
        df['lower'] = df['middle'] - self.atr_multiplier * df['atr']
        
        # Regime detection using ADX-like measure
        df['directional_movement'] = abs(df['close'] - df['close'].shift(self.regime_lookback))
        df['total_movement'] = df['tr'].rolling(window=self.regime_lookback).sum()
        df['trend_strength'] = df['directional_movement'] / df['total_movement']
        df['is_trending'] = df['trend_strength'] > self.trend_threshold
        
        # Generate signals
        df['signal'] = 0
        df['exit_signal'] = False
        
        position = 0
        
        for i in range(self.lookback + 1, len(df)):
            if not df.loc[df.index[i], 'is_trending']:
                if position != 0:
                    df.loc[df.index[i], 'signal'] = 0
                    df.loc[df.index[i], 'exit_signal'] = True
                    position = 0
                continue
            
            close = df.loc[df.index[i], 'close']
            upper = df.loc[df.index[i], 'upper']
            lower = df.loc[df.index[i], 'lower']
            
            # Breakout up
            if position == 0 and close > upper:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
            # Breakout down
            elif position == 0 and close < lower:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
            # Exit on mean reversion
            elif position == 1 and close < df.loc[df.index[i], 'middle']:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'exit_signal'] = True
                position = 0
            elif position == -1 and close > df.loc[df.index[i], 'middle']:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'exit_signal'] = True
                position = 0
            else:
                df.loc[df.index[i], 'signal'] = position
        
        df.set_index('timestamp', inplace=True)
        return df


def run_strategy_scan(symbols: List[str], start: str, end: str,
                     exchanges: List[str] = None) -> pd.DataFrame:
    """
    Scan multiple strategies across symbols to find best performers.
    """
    exchanges = exchanges or ['bybit']
    loader = DataLoader()
    
    results = []
    
    # Strategy configs to test
    strategies = [
        ('FR_Hold_20_8', FundingRateHoldStrategy(20, 8, max_hold_periods=3)),
        ('FR_Hold_30_10', FundingRateHoldStrategy(30, 10, max_hold_periods=3)),
        ('FR_Hold_15_5', FundingRateHoldStrategy(15, 5, max_hold_periods=2)),
        ('Settlement_Snap', SettlementSnapStrategy(fr_threshold_bps=-50)),
        ('Vol_Breakout', VolatilityBreakoutStrategy(lookback=20)),
    ]
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        for ex in exchanges:
            # Load data
            klines = loader.load_klines(ex, symbol, start, end)
            funding = loader.load_funding_rates(ex, symbol, start, end)
            
            if klines.empty or funding.empty:
                continue
            
            data = {'klines': klines, 'funding': funding}
            
            for name, strategy in strategies:
                try:
                    engine = BacktestEngine(initial_capital=10000)
                    result = engine.run(strategy, data, position_size=1.0)
                    
                    if result.total_trades >= 10:
                        results.append({
                            'symbol': symbol,
                            'exchange': ex,
                            'strategy': name,
                            'trades': result.total_trades,
                            'win_rate': result.win_rate,
                            'profit_factor': result.profit_factor,
                            'total_return': result.total_return,
                            'max_drawdown': result.max_drawdown,
                            'sharpe': result.sharpe_ratio,
                            'total_fees': sum(t.fees for t in result.trades),
                            'net_pnl': sum(t.pnl_net for t in result.trades),
                        })
                        print(f"  {name}: {result.total_trades} trades, "
                              f"WR={result.win_rate:.1%}, PF={result.profit_factor:.2f}")
                except Exception as e:
                    print(f"  {name}: Error - {e}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Quick test run
    loader = DataLoader()
    symbols = loader.get_common_symbols()[:5]  # Test first 5 symbols
    
    print("Running strategy scan on top symbols...")
    results = run_strategy_scan(symbols, '2025-07-01', '2025-12-31')
    
    if not results.empty:
        print("\n" + "="*80)
        print("TOP PERFORMERS (by net P&L):")
        print("="*80)
        top = results.nlargest(10, 'net_pnl')
        print(top.to_string())
    else:
        print("No results generated")
