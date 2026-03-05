import pandas as pd
import numpy as np

def add_ma_crossover_signals(df):
    """
    Add MA Crossover signals.
    Long when fast MA > slow MA, short when fast MA < slow MA.
    """
    df['fast_ma'] = df['close'].rolling(10).mean()
    df['slow_ma'] = df['close'].rolling(50).mean()
    df['ma_signal'] = 0
    df.loc[df['fast_ma'] > df['slow_ma'], 'ma_signal'] = 1
    df.loc[df['fast_ma'] < df['slow_ma'], 'ma_signal'] = -1
    return df

def add_combined_signals(df):
    """
    Set combined signals to MA Crossover signals.
    """
    df['combined_signal'] = df['ma_signal']

    return df
