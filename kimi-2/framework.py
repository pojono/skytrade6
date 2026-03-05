"""
Kimi-2 Strategy Research Framework
===================================
Goal: Find profitable strategies surviving maker 0.04% / taker 0.1% fees
Data: Bybit 1m klines, funding rates, OI, L/S ratio, orderbook, trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fee structure (user's account)
MAKER_FEE = 0.0004  # 0.04%
TAKER_FEE = 0.001   # 0.1%

# Round-trip costs
RT_TAKER = 2 * TAKER_FEE  # 0.2% = 20 bps
RT_MAKER = 2 * MAKER_FEE  # 0.08% = 8 bps
RT_MIXED = TAKER_FEE + MAKER_FEE  # 0.14% = 14 bps

DATALAKE_PATH = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size_usd: float
    fees: float
    pnl_gross: float
    pnl_net: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Backtest results container"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    trades: List[Trade]
    
    # Performance metrics
    total_pnl_net: float = 0
    total_pnl_gross: float = 0
    total_fees: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    num_trades: int = 0
    avg_trade_pnl: float = 0
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        if not self.trades:
            return
        
        self.num_trades = len(self.trades)
        self.total_pnl_net = sum(t.pnl_net for t in self.trades)
        self.total_pnl_gross = sum(t.pnl_gross for t in self.trades)
        self.total_fees = sum(t.fees for t in self.trades)
        self.avg_trade_pnl = self.total_pnl_net / self.num_trades
        
        wins = [t.pnl_net for t in self.trades if t.pnl_net > 0]
        losses = [t.pnl_net for t in self.trades if t.pnl_net <= 0]
        
        self.win_rate = len(wins) / self.num_trades * 100 if self.num_trades > 0 else 0
        
        gross_profits = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        self.profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Equity curve for Sharpe and drawdown
        equity = [0]
        for trade in self.trades:
            equity.append(equity[-1] + trade.pnl_net)
        
        returns = np.diff(equity)
        if len(returns) > 1 and np.std(returns) > 0:
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        self.max_drawdown = max_dd
    
    def summary(self) -> str:
        """Return formatted summary"""
        return f"""
{'='*60}
Strategy: {self.strategy_name} | Symbol: {self.symbol}
Period: {self.start_date.date()} to {self.end_date.date()}
{'='*60}
Trades:          {self.num_trades}
Win Rate:        {self.win_rate:.1f}%
Net P&L:         ${self.total_pnl_net:,.2f}
Gross P&L:       ${self.total_pnl_gross:,.2f}
Total Fees:      ${self.total_fees:,.2f} ({self.total_fees/self.total_pnl_gross*100:.1f}% of gross)
Avg Trade:       ${self.avg_trade_pnl:.2f}
Profit Factor:   {self.profit_factor:.2f}
Sharpe Ratio:    {self.sharpe_ratio:.2f}
Max Drawdown:    {self.max_drawdown*100:.2f}%
{'='*60}
"""


def load_klines(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load 1m klines from datalake"""
    symbol_path = DATALAKE_PATH / symbol
    if not symbol_path.exists():
        return pd.DataFrame()
    
    files = sorted(symbol_path.glob('*_kline_1m.csv'))
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    dfs = []
    for f in files:
        # Extract date from filename
        date_str = f.stem.split('_')[0]
        try:
            file_date = pd.to_datetime(date_str)
            if start_dt <= file_date <= end_dt:
                df = pd.read_csv(f)
                df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_funding_rates(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load funding rate history"""
    symbol_path = DATALAKE_PATH / symbol
    if not symbol_path.exists():
        return pd.DataFrame()
    
    files = sorted(symbol_path.glob('*_funding_rate.csv'))
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    dfs = []
    for f in files:
        date_str = f.stem.split('_')[0]
        try:
            file_date = pd.to_datetime(date_str)
            if start_dt <= file_date <= end_dt:
                df = pd.read_csv(f)
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_open_interest(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load open interest data"""
    symbol_path = DATALAKE_PATH / symbol
    if not symbol_path.exists():
        return pd.DataFrame()
    
    files = sorted(symbol_path.glob('*_open_interest_5min.csv'))
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    dfs = []
    for f in files:
        date_str = f.stem.split('_')[0]
        try:
            file_date = pd.to_datetime(date_str)
            if start_dt <= file_date <= end_dt:
                df = pd.read_csv(f)
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_long_short_ratio(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load long/short ratio data"""
    symbol_path = DATALAKE_PATH / symbol
    if not symbol_path.exists():
        return pd.DataFrame()
    
    files = sorted(symbol_path.glob('*_long_short_ratio_5min.csv'))
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    dfs = []
    for f in files:
        date_str = f.stem.split('_')[0]
        try:
            file_date = pd.to_datetime(date_str)
            if start_dt <= file_date <= end_dt:
                df = pd.read_csv(f)
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def get_available_symbols() -> List[str]:
    """Get list of available symbols in datalake"""
    if not DATALAKE_PATH.exists():
        return []
    return sorted([d.name for d in DATALAKE_PATH.iterdir() if d.is_dir()])


def check_symbol_data(symbol: str) -> Dict:
    """Check what data is available for a symbol"""
    symbol_path = DATALAKE_PATH / symbol
    if not symbol_path.exists():
        return {}
    
    data_types = {
        'klines': len(list(symbol_path.glob('*_kline_1m.csv'))),
        'funding': len(list(symbol_path.glob('*_funding_rate.csv'))),
        'oi': len(list(symbol_path.glob('*_open_interest_5min.csv'))),
        'ls_ratio': len(list(symbol_path.glob('*_long_short_ratio_5min.csv'))),
        'trades': len(list(symbol_path.glob('*_trades.csv.gz'))),
        'orderbook': len(list(symbol_path.glob('*_orderbook.jsonl.gz'))),
    }
    return data_types


if __name__ == '__main__':
    # Test data loading
    symbols = get_available_symbols()
    print(f"Available symbols: {len(symbols)}")
    print(f"First 10: {symbols[:10]}")
    
    # Check SOLUSDT data
    sol_data = check_symbol_data('SOLUSDT')
    print(f"\nSOLUSDT data: {sol_data}")
