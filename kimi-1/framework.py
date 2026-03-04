"""
Kimi-1 Strategy Research Framework
Data-driven strategy development with proper fee accounting and walk-forward validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fee structure
MAKER_FEE = 0.0004  # 0.04%
TAKER_FEE = 0.001   # 0.10%
ROUND_TRIP_MAKER = 2 * MAKER_FEE  # 0.08%
ROUND_TRIP_TAKER = 2 * TAKER_FEE  # 0.20%
ROUND_TRIP_MIXED = MAKER_FEE + TAKER_FEE  # 0.14%

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    exchange: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    size: float
    fees: float
    pnl_gross: float
    pnl_net: float
    exit_reason: str

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_net > 0)
        return wins / len(self.trades)
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_net for t in self.trades if t.pnl_net > 0)
        gross_loss = abs(sum(t.pnl_net for t in self.trades if t.pnl_net < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(365 * 24)  # Hourly Sharpe
    
    @property
    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()
    
    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1


class DataLoader:
    """Load and merge data from multiple exchanges."""
    
    def __init__(self, datalake_path: str = '/home/ubuntu/Projects/skytrade6/datalake'):
        self.datalake_path = Path(datalake_path)
        
    def load_klines(self, exchange: str, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load 1-minute kline data for a symbol from specified exchange."""
        path = self.datalake_path / exchange / symbol
        if not path.exists():
            return pd.DataFrame()
        
        files = sorted(path.glob('*_kline_1m.csv'))
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            date_str = f.name.split('_')[0]
            if start <= date_str <= end:
                df = pd.read_csv(f)
                if not df.empty:
                    dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def load_funding_rates(self, exchange: str, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load funding rate data."""
        path = self.datalake_path / exchange / symbol
        
        if exchange == 'binance':
            # Binance stores funding in metrics.csv
            files = sorted(path.glob('*_metrics.csv'))
        else:
            files = sorted(path.glob('*_funding_rate.csv'))
        
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            date_str = f.name.split('_')[0]
            if start <= date_str <= end:
                df = pd.read_csv(f)
                if not df.empty:
                    dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
        df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True, errors='ignore')
        return df
    
    def load_open_interest(self, exchange: str, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load open interest data."""
        path = self.datalake_path / exchange / symbol
        files = sorted(path.glob('*_open_interest_5min.csv'))
        
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            date_str = f.name.split('_')[0]
            if start <= date_str <= end:
                df = pd.read_csv(f)
                if not df.empty:
                    dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def get_common_symbols(self) -> List[str]:
        """Get list of symbols available on all three exchanges."""
        bybit_symbols = set(p.name for p in (self.datalake_path / 'bybit').iterdir() if p.is_dir())
        binance_symbols = set(p.name for p in (self.datalake_path / 'binance').iterdir() if p.is_dir())
        okx_symbols = set(p.name for p in (self.datalake_path / 'okx').iterdir() if p.is_dir())
        return sorted(bybit_symbols & binance_symbols & okx_symbols)


class Strategy:
    """Base class for strategies."""
    
    def __init__(self, name: str, maker_fee: float = MAKER_FEE, taker_fee: float = TAKER_FEE):
        self.name = name
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals. Returns DataFrame with 'signal' column (1, -1, 0)."""
        raise NotImplementedError
    
    def calculate_fees(self, price: float, size: float, is_maker: bool = False) -> float:
        """Calculate trading fees for a trade."""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return price * size * fee_rate


class BacktestEngine:
    """Backtest engine with proper fee accounting."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
    
    def run(self, strategy: Strategy, data: Dict[str, pd.DataFrame], 
            position_size: float = 1.0, max_positions: int = 3) -> BacktestResult:
        """Run backtest for a strategy."""
        
        signals = strategy.generate_signals(data)
        if signals.empty or 'signal' not in signals.columns:
            return BacktestResult([], pd.Series(), {})
        
        trades = []
        equity = [self.initial_capital]
        equity_times = [signals.index[0]]
        
        current_position = None
        capital = self.initial_capital
        
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            signal = row['signal']
            price = row.get('close', row.get('mark_price', 0))
            
            # Close existing position if signal changes or exit condition met
            if current_position is not None:
                should_exit = False
                exit_reason = ""
                
                if signal != current_position['direction'] and signal != 0:
                    should_exit = True
                    exit_reason = "signal_flip"
                elif 'exit_signal' in row and row['exit_signal']:
                    should_exit = True
                    exit_reason = "exit_trigger"
                
                if should_exit and price > 0:
                    # Close position
                    entry_price = current_position['entry_price']
                    direction = current_position['direction']
                    size = current_position['size']
                    
                    # Calculate P&L
                    price_change = (price - entry_price) / entry_price
                    pnl_gross = price_change * direction * capital * size
                    
                    # Calculate fees (exit fee)
                    position_value = size * capital
                    exit_fee = strategy.calculate_fees(price, position_value / price, is_maker=False)
                    entry_fee = current_position['entry_fee']
                    total_fees = entry_fee + exit_fee
                    
                    pnl_net = pnl_gross - total_fees
                    
                    trade = Trade(
                        entry_time=current_position['entry_time'],
                        exit_time=timestamp,
                        symbol=current_position['symbol'],
                        exchange=current_position['exchange'],
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=price,
                        size=size,
                        fees=total_fees,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    
                    capital += pnl_net
                    current_position = None
                    
                    equity.append(capital)
                    equity_times.append(timestamp)
            
            # Open new position if no current position and signal exists
            if current_position is None and signal != 0 and price > 0:
                position_value = position_size * capital
                entry_fee = strategy.calculate_fees(price, position_value / price, is_maker=False)
                
                current_position = {
                    'entry_time': timestamp,
                    'entry_price': price,
                    'direction': signal,
                    'size': position_size,
                    'symbol': row.get('symbol', 'UNKNOWN'),
                    'exchange': row.get('exchange', 'UNKNOWN'),
                    'entry_fee': entry_fee
                }
        
        # Close any open position at the end
        if current_position is not None and len(signals) > 0:
            last_price = signals.iloc[-1].get('close', signals.iloc[-1].get('mark_price', 0))
            entry_price = current_position['entry_price']
            direction = current_position['direction']
            size = current_position['size']
            
            price_change = (last_price - entry_price) / entry_price
            pnl_gross = price_change * direction * capital * size
            
            exit_fee = strategy.calculate_fees(last_price, size * capital / last_price, is_maker=False)
            total_fees = current_position['entry_fee'] + exit_fee
            pnl_net = pnl_gross - total_fees
            
            trade = Trade(
                entry_time=current_position['entry_time'],
                exit_time=signals.index[-1],
                symbol=current_position['symbol'],
                exchange=current_position['exchange'],
                direction=direction,
                entry_price=entry_price,
                exit_price=last_price,
                size=size,
                fees=total_fees,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                exit_reason="end_of_data"
            )
            trades.append(trade)
            capital += pnl_net
            equity.append(capital)
            equity_times.append(signals.index[-1])
        
        equity_curve = pd.Series(equity, index=equity_times)
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0,
            'avg_trade': sum(t.pnl_net for t in trades) / len(trades) if trades else 0,
            'total_pnl': sum(t.pnl_net for t in trades),
            'total_fees': sum(t.fees for t in trades),
        }
        
        return BacktestResult(trades, equity_curve, metrics)


def walk_forward_validation(strategy: Strategy, data: Dict[str, pd.DataFrame], 
                           train_size: int = 30, test_size: int = 7,
                           n_splits: int = 5) -> List[BacktestResult]:
    """
    Perform walk-forward validation to avoid overfitting.
    
    Args:
        strategy: Strategy to test
        data: Data dictionary with 'klines' etc.
        train_size: Training window in days
        test_size: Test window in days
        n_splits: Number of walk-forward splits
    """
    results = []
    
    # Get date range
    klines = data.get('klines', pd.DataFrame())
    if klines.empty:
        return results
    
    start_date = klines['timestamp'].min()
    end_date = klines['timestamp'].max()
    total_days = (end_date - start_date).days
    
    # Calculate split points
    step = (total_days - train_size) // n_splits
    
    for i in range(n_splits):
        train_start = start_date + timedelta(days=i * step)
        train_end = train_start + timedelta(days=train_size)
        test_start = train_end
        test_end = min(test_start + timedelta(days=test_size), end_date)
        
        # Filter data for this split
        train_mask = (klines['timestamp'] >= train_start) & (klines['timestamp'] < train_end)
        test_mask = (klines['timestamp'] >= test_start) & (klines['timestamp'] < test_end)
        
        train_data = {'klines': klines[train_mask]}
        test_data = {'klines': klines[test_mask]}
        
        # Run backtest on test set
        engine = BacktestEngine()
        result = engine.run(strategy, test_data)
        results.append(result)
    
    return results


if __name__ == '__main__':
    # Quick test
    loader = DataLoader()
    symbols = loader.get_common_symbols()
    print(f"Found {len(symbols)} common symbols across exchanges")
    print(f"Top 10: {symbols[:10]}")
