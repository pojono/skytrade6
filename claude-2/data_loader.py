"""
Shared RAM-efficient data loader for Bybit datalake.
Loads CSV data in streaming fashion, returns pandas DataFrames.
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")

# Fee structure
MAKER_FEE_BPS = 4.0   # 0.04%
TAKER_FEE_BPS = 10.0  # 0.1%
MAKER_FEE = MAKER_FEE_BPS / 10000
TAKER_FEE = TAKER_FEE_BPS / 10000
RT_TAKER_BPS = 20.0   # round-trip taker
RT_MAKER_BPS = 8.0    # round-trip maker
RT_MIXED_BPS = 14.0   # maker entry, taker exit


def get_symbols(min_days=90):
    """Get symbols with at least min_days of data."""
    symbols = []
    for d in sorted(DATALAKE.iterdir()):
        if not d.is_dir():
            continue
        klines = list(d.glob("*_kline_1m.csv"))
        if len(klines) >= min_days:
            symbols.append(d.name)
    return symbols


def load_csv_daterange(symbol, dtype, start_date=None, end_date=None):
    """Load CSV files for a symbol/dtype, filtered by date range.
    RAM-efficient: only reads files within the date range.
    
    dtype: 'kline_1m', 'open_interest_5min', 'long_short_ratio_5min',
           'premium_index_kline_1m', 'mark_price_kline_1m', 'funding_rate',
           'kline_1m_spot'
    """
    sym_dir = DATALAKE / symbol
    # Use regex-based matching to avoid glob ambiguity
    # e.g. *_kline_1m.csv must NOT match *_mark_price_kline_1m.csv
    import re
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_' + re.escape(dtype) + r'\.csv$')
    all_files = sorted(os.listdir(sym_dir)) if sym_dir.exists() else []
    files = [str(sym_dir / f) for f in all_files if date_pattern.match(f)]
    
    if not files:
        return pd.DataFrame()
    
    # Filter files by date range from filename
    filtered = []
    for f in files:
        fname = os.path.basename(f)
        file_date = fname[:10]  # YYYY-MM-DD
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue
        filtered.append(f)
    
    if not filtered:
        return pd.DataFrame()
    
    chunks = []
    for f in filtered:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                chunks.append(df)
        except Exception:
            continue
    
    if not chunks:
        return pd.DataFrame()
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Convert timestamp columns
    ts_col = None
    for c in ['startTime', 'timestamp']:
        if c in df.columns:
            ts_col = c
            break
    
    if ts_col:
        df['ts'] = pd.to_datetime(df[ts_col], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
    
    return df


def load_kline(symbol, start_date=None, end_date=None, spot=False):
    """Load 1m kline data. Returns df with ts, open, high, low, close, volume, turnover."""
    dtype = 'kline_1m_spot' if spot else 'kline_1m'
    return load_csv_daterange(symbol, dtype, start_date, end_date)


def load_oi(symbol, start_date=None, end_date=None):
    """Load 5min open interest. Returns df with ts, openInterest."""
    return load_csv_daterange(symbol, 'open_interest_5min', start_date, end_date)


def load_ls_ratio(symbol, start_date=None, end_date=None):
    """Load 5min long/short ratio. Returns df with ts, buyRatio, sellRatio."""
    return load_csv_daterange(symbol, 'long_short_ratio_5min', start_date, end_date)


def load_premium(symbol, start_date=None, end_date=None):
    """Load 1m premium index. Returns df with ts, open, high, low, close (premium values)."""
    return load_csv_daterange(symbol, 'premium_index_kline_1m', start_date, end_date)


def load_mark(symbol, start_date=None, end_date=None):
    """Load 1m mark price kline. Returns df with ts, open, high, low, close."""
    return load_csv_daterange(symbol, 'mark_price_kline_1m', start_date, end_date)


def load_funding_rate(symbol, start_date=None, end_date=None):
    """Load funding rate data. Returns df with ts, fundingRate."""
    return load_csv_daterange(symbol, 'funding_rate', start_date, end_date)


def progress_bar(current, total, prefix='', suffix='', start_time=None, width=40):
    """Print a progress bar with ETA."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    eta_str = ''
    if start_time and current > 0:
        elapsed = time.time() - start_time
        eta = elapsed / current * (total - current)
        if eta > 60:
            eta_str = f' ETA {eta/60:.1f}m'
        else:
            eta_str = f' ETA {eta:.0f}s'
    print(f'\r{prefix} |{bar}| {current}/{total} ({pct*100:.0f}%){eta_str} {suffix}', end='', flush=True)
    if current >= total:
        print()


def backtest_signals(prices, signals, hold_periods=12, fee_bps=RT_TAKER_BPS):
    """
    Simple vectorized backtest.
    prices: Series of close prices (1m)
    signals: Series of +1 (long), -1 (short), 0 (flat), aligned with prices
    hold_periods: how many bars to hold
    fee_bps: round-trip fee in bps
    
    Returns dict with stats.
    """
    fee = fee_bps / 10000
    
    # Find signal entries (non-zero after zero or sign change)
    entries = signals[(signals != 0) & (signals.shift(1) != signals)].index
    
    trades = []
    for idx in entries:
        pos = int(idx)
        if pos + hold_periods >= len(prices):
            break
        direction = signals.iloc[pos]
        entry_price = prices.iloc[pos]
        exit_price = prices.iloc[pos + hold_periods]
        ret_bps = direction * (exit_price / entry_price - 1) * 10000 - fee_bps
        trades.append({
            'entry_idx': pos,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'ret_bps': ret_bps,
        })
    
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'avg_ret_bps': 0, 'total_ret_bps': 0,
                'sharpe': 0, 'max_dd_bps': 0, 'profit_factor': 0}
    
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['ret_bps'] > 0]
    losses = tdf[tdf['ret_bps'] <= 0]
    
    cum = tdf['ret_bps'].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    
    gross_win = wins['ret_bps'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['ret_bps'].sum()) if len(losses) > 0 else 0.001
    
    return {
        'n_trades': len(tdf),
        'win_rate': len(wins) / len(tdf) * 100,
        'avg_ret_bps': tdf['ret_bps'].mean(),
        'total_ret_bps': tdf['ret_bps'].sum(),
        'sharpe': tdf['ret_bps'].mean() / max(tdf['ret_bps'].std(), 0.01),
        'max_dd_bps': dd,
        'profit_factor': gross_win / gross_loss,
        'trades': tdf,
    }


if __name__ == '__main__':
    syms = get_symbols(min_days=90)
    print(f"Found {len(syms)} symbols with 90+ days of data")
    syms30 = get_symbols(min_days=30)
    print(f"Found {len(syms30)} symbols with 30+ days of data")
    
    # Quick test load
    df = load_kline('SOLUSDT', '2026-02-01', '2026-02-07')
    print(f"SOL 1w kline: {len(df)} rows, {df['ts'].min()} → {df['ts'].max()}")
    print(f"RAM: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
