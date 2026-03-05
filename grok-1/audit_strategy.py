"""
Independent audit of grok-1 Momentum Continuation strategy.
Tests: signal correctness, backtest logic, Sharpe calculation, random baseline.
"""
import pandas as pd
import numpy as np
import sys, glob
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/grok-1')
from data.data_loader import load_bybit_data

# ── 1. Load data for a few symbols ──────────────────────────────────
symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']
START, END = '2025-07-01', '2026-02-28'

for sym in symbols:
    print(f"\n{'='*60}")
    print(f"  AUDITING: {sym}")
    print(f"{'='*60}")
    
    try:
        df = load_bybit_data(sym, START, END)
    except Exception as e:
        print(f"  SKIP: {e}")
        continue
    
    n_rows = len(df)
    print(f"  Rows: {n_rows}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # ── 2. Replicate signals independently ──────────────────────────
    consecutive = 2
    up = (df['close'] > df['open']).astype(int)
    down = (df['close'] < df['open']).astype(int)
    up_count = up.rolling(consecutive).sum()
    down_count = down.rolling(consecutive).sum()
    
    signal = pd.Series(0, index=df.index)
    signal[up_count >= consecutive] = 1
    signal[down_count >= consecutive] = -1
    
    # Check: signal uses only current and previous bar (no lookahead)
    # up_count at bar i = up[i] + up[i-1] for consecutive=2 → OK, no future data
    print(f"  Signal distribution: long={int((signal==1).sum())}, short={int((signal==-1).sum())}, flat={int((signal==0).sum())}")
    
    # ── 3. Replicate backtest independently ─────────────────────────
    hold_period = 2
    fees = 0.002  # 20bps RT
    
    trades = []
    position = 0
    entry_price = np.nan
    entry_idx = -1
    hold_counter = 0
    
    for i in range(len(df)):
        s = signal.iloc[i]
        if position == 0 and s != 0:
            position = s
            entry_price = df.iloc[i]['open']  # entry at OPEN of signal bar
            entry_idx = i
            hold_counter = 0
        elif position != 0:
            hold_counter += 1
            if hold_counter >= hold_period:
                exit_price = df.iloc[i]['close']  # exit at CLOSE after hold
                if position == 1:
                    raw_pnl = (exit_price - entry_price) / entry_price
                else:
                    raw_pnl = (entry_price - exit_price) / entry_price
                net_pnl = raw_pnl - fees
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': df.iloc[i]['timestamp'],
                    'direction': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'raw_pnl': raw_pnl,
                    'net_pnl': net_pnl,
                })
                position = 0
                entry_price = np.nan
                hold_counter = 0
    
    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("  NO TRADES")
        continue
    
    n_trades = len(tdf)
    win_rate = (tdf['net_pnl'] > 0).mean()
    avg_pnl = tdf['net_pnl'].mean()
    total_ret = tdf['net_pnl'].sum()
    
    print(f"\n  --- Backtest Results ---")
    print(f"  Trades: {n_trades}")
    print(f"  Win rate: {win_rate:.2%}")
    print(f"  Avg net PnL: {avg_pnl:.4f} ({avg_pnl*100:.2f}%)")
    print(f"  Total return: {total_ret:.4f} ({total_ret*100:.2f}%)")
    print(f"  Avg raw PnL (before fees): {tdf['raw_pnl'].mean():.4f} ({tdf['raw_pnl'].mean()*100:.2f}%)")
    
    # ── 4. CORRECT Sharpe calculation ───────────────────────────────
    # The original uses daily_pnl = df.groupby('date')['pnl'].sum() 
    # which includes zeros for non-trade bars → deflates std → inflates Sharpe
    
    # Method A: Per-trade Sharpe (standard for strategies)
    if tdf['net_pnl'].std() > 0:
        trades_per_year = n_trades / ((df['timestamp'].max() - df['timestamp'].min()).days / 365.25)
        sharpe_per_trade = tdf['net_pnl'].mean() / tdf['net_pnl'].std() * np.sqrt(trades_per_year)
    else:
        sharpe_per_trade = 0
    
    # Method B: Daily PnL Sharpe (correct: only days with exits)
    tdf['exit_date'] = pd.to_datetime(tdf['exit_time']).dt.date
    daily_pnl = tdf.groupby('exit_date')['net_pnl'].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe_daily = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
    else:
        sharpe_daily = 0
    
    # Method C: ALL calendar days (including zero-PnL days = original method bug)
    all_dates = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='D')
    daily_pnl_all = daily_pnl.reindex([d.date() for d in all_dates], fill_value=0)
    if daily_pnl_all.std() > 0:
        sharpe_all_days = daily_pnl_all.mean() / daily_pnl_all.std() * np.sqrt(365)
    else:
        sharpe_all_days = 0
    
    # Method D: Replicate original buggy Sharpe
    # Original: df['pnl'] column has values only on exit bars, rest are 0 or NaN
    # In broad_test.py: pnl column initialized as NaN, then sum() per day
    # .sum() on NaN = 0 → days without exits = 0 → inflates denominator
    
    print(f"\n  --- Sharpe Comparison ---")
    print(f"  Per-trade Sharpe:           {sharpe_per_trade:.2f}")
    print(f"  Daily Sharpe (trade days):  {sharpe_daily:.2f}")
    print(f"  Daily Sharpe (all days):    {sharpe_all_days:.2f}")
    
    # ── 5. Entry timing audit ───────────────────────────────────────
    # CRITICAL: Signal at bar i is based on bars i and i-1 (close>open).
    # But entry is at bar i's OPEN. 
    # The signal uses bar i's CLOSE to determine direction, but enters at bar i's OPEN.
    # This means: we decide to enter based on bar i being an up/down bar,
    # but we enter at bar i's OPEN *before* we know if bar i is up/down.
    # THIS IS LOOKAHEAD BIAS!
    
    # Let's check: at entry bar, is the signal determined from current bar's close?
    # signal[i] = 1 if up_count[i] >= 2, where up_count[i] = up[i] + up[i-1]
    # up[i] = 1 if close[i] > open[i]
    # So signal[i] depends on close[i], but entry_price = open[i]
    # We're using close[i] info to decide to enter at open[i] → LOOKAHEAD!
    
    # Correct: entry should be at open[i+1] (next bar open)
    trades_fixed = []
    position = 0
    entry_price = np.nan
    entry_idx = -1
    hold_counter = 0
    pending_signal = 0
    
    for i in range(len(df)):
        # Execute pending entry from previous bar's signal
        if position == 0 and pending_signal != 0:
            position = pending_signal
            entry_price = df.iloc[i]['open']  # enter at THIS bar's open
            entry_idx = i
            hold_counter = 0
            pending_signal = 0
        elif position != 0:
            hold_counter += 1
            if hold_counter >= hold_period:
                exit_price = df.iloc[i]['close']
                if position == 1:
                    raw_pnl = (exit_price - entry_price) / entry_price
                else:
                    raw_pnl = (entry_price - exit_price) / entry_price
                net_pnl = raw_pnl - fees
                trades_fixed.append({
                    'net_pnl': net_pnl,
                    'raw_pnl': raw_pnl,
                })
                position = 0
                entry_price = np.nan
                hold_counter = 0
        
        # Queue signal for next bar execution
        s = signal.iloc[i]
        if position == 0 and s != 0:
            pending_signal = s
    
    tdf_fixed = pd.DataFrame(trades_fixed)
    if len(tdf_fixed) > 0:
        n_fixed = len(tdf_fixed)
        wr_fixed = (tdf_fixed['net_pnl'] > 0).mean()
        avg_fixed = tdf_fixed['net_pnl'].mean()
        total_fixed = tdf_fixed['net_pnl'].sum()
        
        print(f"\n  --- FIXED (no lookahead: enter next bar open) ---")
        print(f"  Trades: {n_fixed}")
        print(f"  Win rate: {wr_fixed:.2%}")
        print(f"  Avg net PnL: {avg_fixed:.4f} ({avg_fixed*100:.2f}%)")
        print(f"  Total return: {total_fixed:.4f} ({total_fixed*100:.2f}%)")
    
    # ── 6. Random baseline ──────────────────────────────────────────
    np.random.seed(42)
    n_random_trials = 100
    random_returns = []
    for trial in range(n_random_trials):
        # Random entry: same frequency as strategy
        rand_trades = []
        pos = 0
        ep = np.nan
        hc = 0
        # Generate random signals with same frequency
        n_signals = int((signal != 0).sum())
        rand_signal = np.zeros(len(df))
        signal_indices = np.random.choice(len(df), size=min(n_signals, len(df)), replace=False)
        rand_signal[signal_indices] = np.random.choice([-1, 1], size=len(signal_indices))
        
        for i in range(1, len(df)):
            if pos == 0 and rand_signal[i] != 0:
                pos = rand_signal[i]
                ep = df.iloc[i]['open']
                hc = 0
            elif pos != 0:
                hc += 1
                if hc >= hold_period:
                    xp = df.iloc[i]['close']
                    if pos == 1:
                        rpnl = (xp - ep) / ep
                    else:
                        rpnl = (ep - xp) / ep
                    rand_trades.append(rpnl - fees)
                    pos = 0
                    hc = 0
        
        if rand_trades:
            random_returns.append(np.mean(rand_trades))
    
    rand_avg = np.mean(random_returns)
    rand_std = np.std(random_returns)
    print(f"\n  --- Random Baseline (100 trials) ---")
    print(f"  Random avg net PnL: {rand_avg:.4f} ({rand_avg*100:.2f}%)")
    print(f"  Random std: {rand_std:.4f}")
    print(f"  Strategy avg: {avg_pnl:.4f}")
    print(f"  Edge over random: {(avg_pnl - rand_avg)*100:.2f}%")
    if rand_std > 0:
        z_score = (avg_pnl - rand_avg) / rand_std
        print(f"  Z-score vs random: {z_score:.2f}")

print("\n" + "="*60)
print("AUDIT COMPLETE")
print("="*60)
