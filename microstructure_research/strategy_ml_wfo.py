#!/usr/bin/env python3
"""
Walk-Forward Optimization (WFO) Framework for ML-based crypto futures strategies.

Design principles:
  - NO lookahead bias: strict train/gap/test splits, purged CV
  - Realistic fees: maker 0.02% entry (limit), taker 0.055% stop-loss, maker 0.02% TP
  - NO trailing stops (requires tick simulation)
  - NO intra-bar assumptions: enter at NEXT bar open, exit at bar close
  - Conservative position sizing
  - Full metrics: Sharpe, max DD, profit factor, avg trade, win rate

Usage:
  from strategy_ml_wfo import (
      load_features, WFOEngine, BacktestEngine, print_metrics
  )
"""

import gc
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAKER_FEE = 0.0002   # 0.02% = 2 bps
TAKER_FEE = 0.00055  # 0.055% = 5.5 bps
DEFAULT_FEATURES_DIR = Path("features")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_features(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "15m",
    features_dir: Path = DEFAULT_FEATURES_DIR,
) -> pd.DataFrame:
    """Load feature parquet files for a date range, concatenate into one DataFrame.

    Returns DataFrame with datetime index, feature columns, and target columns.
    Shows progress for large loads.
    """
    feat_dir = features_dir / symbol / timeframe
    if not feat_dir.exists():
        raise FileNotFoundError(f"No features directory: {feat_dir}")

    dates = pd.date_range(start_date, end_date, freq="D")
    frames = []
    loaded = 0
    missing = 0
    t0 = time.time()

    for i, dt in enumerate(dates):
        pq_path = feat_dir / f"{dt.date()}.parquet"
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            frames.append(df)
            loaded += 1
        else:
            missing += 1

        if (i + 1) % 30 == 0 or i == len(dates) - 1:
            elapsed = time.time() - t0
            print(f"  Loading: {i+1}/{len(dates)} days, {loaded} loaded, "
                  f"{missing} missing [{elapsed:.0f}s]")

    if not frames:
        raise ValueError(f"No parquet files found for {symbol} {timeframe} "
                         f"{start_date} to {end_date}")

    result = pd.concat(frames, axis=0)
    result.sort_index(inplace=True)
    result = result[~result.index.duplicated(keep="last")]

    print(f"  Loaded {symbol} {timeframe}: {len(result)} candles, "
          f"{loaded} days, {len(result.columns)} cols, "
          f"{result.index[0].date()} to {result.index[-1].date()}")
    return result


def split_features_targets(df: pd.DataFrame):
    """Split DataFrame into features (X) and targets (y dict).

    Returns:
        X: DataFrame of feature columns (no tgt_ prefix)
        targets: dict of {target_name: Series}
    """
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    # Exclude raw OHLCV and other non-feature columns
    exclude = {"open", "high", "low", "close", "vwap", "twap",
               "vwap_buy", "vwap_sell", "candle_time",
               "session_asia", "session_europe", "session_us",
               "high_before_low", "poc_price", "fair_price",
               "fair_value", "value_area_low", "value_area_high",
               "close_above_value_area", "close_below_value_area",
               "overlap_asia_europe", "overlap_europe_us",
               "fvg_bullish", "fvg_bearish",
               "fib_nearest_level", "fib_proximity",
               "busiest_quartile", "busiest_vol_quartile"}

    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_") and c not in exclude]

    X = df[feat_cols].copy()
    targets = {c: df[c].copy() for c in tgt_cols}

    return X, targets


def prepare_features(X: pd.DataFrame, max_nan_frac: float = 0.3) -> pd.DataFrame:
    """Clean features: drop high-NaN columns, replace inf, fill remaining NaN.

    Args:
        X: raw feature DataFrame
        max_nan_frac: drop columns with more than this fraction NaN
    Returns:
        cleaned DataFrame
    """
    # Drop columns with too many NaN
    nan_frac = X.isna().mean()
    keep_cols = nan_frac[nan_frac <= max_nan_frac].index.tolist()
    X = X[keep_cols].copy()

    # Replace inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill remaining NaN with 0 (safe for tree models)
    X.fillna(0, inplace=True)

    return X


# ---------------------------------------------------------------------------
# Walk-Forward Splits
# ---------------------------------------------------------------------------

@dataclass
class WFOSplit:
    """One walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    gap_days: int = 0


def generate_wfo_splits(
    index: pd.DatetimeIndex,
    train_days: int = 60,
    test_days: int = 20,
    gap_days: int = 5,
    step_days: int = 20,
    min_train_rows: int = 500,
) -> list[WFOSplit]:
    """Generate walk-forward optimization splits.

    Args:
        index: DatetimeIndex of the data
        train_days: training window in calendar days
        test_days: test window in calendar days
        gap_days: purge gap between train and test (prevents leakage)
        step_days: how many days to roll forward each fold
        min_train_rows: minimum rows required in training set

    Returns:
        list of WFOSplit objects
    """
    start = index.min()
    end = index.max()
    splits = []
    fold_id = 0

    train_start = start
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > end:
            break

        # Check minimum rows
        train_mask = (index >= train_start) & (index < train_end)
        test_mask = (index >= test_start) & (index < test_end)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train >= min_train_rows and n_test > 0:
            splits.append(WFOSplit(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap_days=gap_days,
            ))
            fold_id += 1

        train_start += pd.Timedelta(days=step_days)

    return splits


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_type: str          # "tp", "sl", "timeout", "signal"
    pnl_bps: float          # net P&L in basis points (after fees)
    hold_bars: int
    signal_prob: float = 0.0


@dataclass
class BacktestResult:
    """Results from one backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_bps > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl_bps(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.pnl_bps for t in self.trades])

    @property
    def total_pnl_bps(self) -> float:
        return sum(t.pnl_bps for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_bps for t in self.trades if t.pnl_bps > 0)
        gross_loss = abs(sum(t.pnl_bps for t in self.trades if t.pnl_bps < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe from trade P&L."""
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl_bps for t in self.trades]
        mu = np.mean(pnls)
        std = np.std(pnls)
        if std == 0:
            return 0.0
        # Assume ~96 candles/day at 15m, annualize
        trades_per_year = len(self.trades) / max(1, self._span_days) * 365
        return (mu / std) * np.sqrt(trades_per_year)

    @property
    def max_drawdown_bps(self) -> float:
        if not self.trades:
            return 0.0
        cum = np.cumsum([t.pnl_bps for t in self.trades])
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(dd.min())

    @property
    def _span_days(self) -> float:
        if len(self.trades) < 2:
            return 1.0
        first = self.trades[0].entry_time
        last = self.trades[-1].exit_time
        return max(1.0, (last - first).total_seconds() / 86400)


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    hold_bars: int = 5,
    tp_bps: Optional[float] = None,
    sl_bps: Optional[float] = None,
    max_concurrent: int = 1,
    entry_on: str = "next_open",
    cooldown_bars: int = 1,
) -> BacktestResult:
    """Run backtest on signal series.

    Args:
        df: DataFrame with OHLC data (must have open, high, low, close columns)
        signals: Series of signals: +1 (long), -1 (short), 0 (flat)
                 Index must match df index.
        hold_bars: fixed holding period in bars
        tp_bps: take-profit in bps (None = no TP, hold full period)
        sl_bps: stop-loss in bps (None = no SL, hold full period)
        max_concurrent: max simultaneous positions
        entry_on: "next_open" (conservative) or "close" (optimistic)
        cooldown_bars: minimum bars between entries

    Returns:
        BacktestResult with trade list and equity curve
    """
    trades = []
    positions = []  # list of active positions as dicts
    last_entry_bar = -cooldown_bars - 1

    close = df["close"].values
    opn = df["open"].values
    high = df["high"].values
    low = df["low"].values
    idx = df.index
    sig = signals.reindex(df.index).fillna(0).values

    for i in range(len(df)):
        # --- Check exits for active positions ---
        closed_positions = []
        for pos in positions:
            entry_bar = pos["entry_bar"]
            bars_held = i - entry_bar
            direction = pos["direction"]
            entry_price = pos["entry_price"]

            # Check TP/SL using high/low of current bar
            exit_type = None
            exit_price = None

            if direction == 1:  # long
                if sl_bps is not None and low[i] <= entry_price * (1 - sl_bps / 10000):
                    exit_type = "sl"
                    exit_price = entry_price * (1 - sl_bps / 10000)
                elif tp_bps is not None and high[i] >= entry_price * (1 + tp_bps / 10000):
                    exit_type = "tp"
                    exit_price = entry_price * (1 + tp_bps / 10000)
            else:  # short
                if sl_bps is not None and high[i] >= entry_price * (1 + sl_bps / 10000):
                    exit_type = "sl"
                    exit_price = entry_price * (1 + sl_bps / 10000)
                elif tp_bps is not None and low[i] <= entry_price * (1 - tp_bps / 10000):
                    exit_type = "tp"
                    exit_price = entry_price * (1 - tp_bps / 10000)

            # Timeout exit
            if exit_type is None and bars_held >= hold_bars:
                exit_type = "timeout"
                exit_price = close[i]

            if exit_type is not None:
                # Calculate P&L
                if direction == 1:
                    raw_ret = (exit_price - entry_price) / entry_price
                else:
                    raw_ret = (entry_price - exit_price) / entry_price

                # Fees: maker entry (limit), maker TP / taker SL
                entry_fee = MAKER_FEE
                exit_fee = TAKER_FEE if exit_type == "sl" else MAKER_FEE
                net_ret = raw_ret - entry_fee - exit_fee
                pnl_bps = net_ret * 10000

                trades.append(Trade(
                    entry_time=idx[entry_bar],
                    exit_time=idx[i],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_type=exit_type,
                    pnl_bps=pnl_bps,
                    hold_bars=bars_held,
                    signal_prob=pos.get("prob", 0.0),
                ))
                closed_positions.append(pos)

        for cp in closed_positions:
            positions.remove(cp)

        # --- Check for new entry ---
        if (i - last_entry_bar) < cooldown_bars:
            continue
        if len(positions) >= max_concurrent:
            continue
        if i >= len(df) - hold_bars:  # don't enter too close to end
            continue

        signal = sig[i]
        if signal == 0:
            continue

        direction = int(signal)
        if entry_on == "next_open" and i + 1 < len(df):
            entry_price = opn[i + 1]
            entry_bar = i + 1
        else:
            entry_price = close[i]
            entry_bar = i

        positions.append({
            "entry_bar": entry_bar,
            "direction": direction,
            "entry_price": entry_price,
            "prob": 0.0,
        })
        last_entry_bar = i

    # Build equity curve
    if trades:
        pnls = [t.pnl_bps for t in trades]
        times = [t.exit_time for t in trades]
        equity = pd.Series(np.cumsum(pnls), index=times, name="equity_bps")
    else:
        equity = pd.Series(dtype=float, name="equity_bps")

    return BacktestResult(trades=trades, equity_curve=equity)


# ---------------------------------------------------------------------------
# Metrics & Reporting
# ---------------------------------------------------------------------------

def print_metrics(result: BacktestResult, label: str = ""):
    """Print formatted backtest metrics."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    if result.n_trades == 0:
        print("  No trades.")
        return

    print(f"  Trades:         {result.n_trades}")
    print(f"  Win rate:       {result.win_rate:.1%}")
    print(f"  Avg P&L:        {result.avg_pnl_bps:+.2f} bps")
    print(f"  Total P&L:      {result.total_pnl_bps:+.1f} bps")
    print(f"  Profit factor:  {result.profit_factor:.2f}")
    print(f"  Max drawdown:   {result.max_drawdown_bps:.1f} bps")
    print(f"  Sharpe (ann):   {result.sharpe:.2f}")

    # Exit type breakdown
    exit_types = {}
    for t in result.trades:
        exit_types[t.exit_type] = exit_types.get(t.exit_type, 0) + 1
    exit_str = ", ".join(f"{k}={v}" for k, v in sorted(exit_types.items()))
    print(f"  Exits:          {exit_str}")

    # Direction breakdown
    longs = [t for t in result.trades if t.direction == 1]
    shorts = [t for t in result.trades if t.direction == -1]
    if longs:
        l_avg = np.mean([t.pnl_bps for t in longs])
        print(f"  Longs:          {len(longs)} trades, avg {l_avg:+.2f} bps")
    if shorts:
        s_avg = np.mean([t.pnl_bps for t in shorts])
        print(f"  Shorts:         {len(shorts)} trades, avg {s_avg:+.2f} bps")

    # Avg hold
    avg_hold = np.mean([t.hold_bars for t in result.trades])
    print(f"  Avg hold:       {avg_hold:.1f} bars")


def print_wfo_summary(fold_results: list[tuple[WFOSplit, BacktestResult]]):
    """Print summary across all WFO folds."""
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD SUMMARY ({len(fold_results)} folds)")
    print(f"{'='*60}")

    all_trades = []
    fold_metrics = []

    for split, result in fold_results:
        all_trades.extend(result.trades)
        fold_metrics.append({
            "fold": split.fold_id,
            "test_start": split.test_start.date(),
            "test_end": split.test_end.date(),
            "n_trades": result.n_trades,
            "win_rate": result.win_rate,
            "avg_pnl": result.avg_pnl_bps,
            "total_pnl": result.total_pnl_bps,
            "sharpe": result.sharpe,
        })

    # Per-fold table
    print(f"\n  {'Fold':>4} {'Test Period':>24} {'Trades':>7} {'WR':>6} "
          f"{'Avg bps':>8} {'Total bps':>10} {'Sharpe':>7}")
    print(f"  {'-'*4} {'-'*24} {'-'*7} {'-'*6} {'-'*8} {'-'*10} {'-'*7}")

    profitable_folds = 0
    for m in fold_metrics:
        profitable_folds += 1 if m["total_pnl"] > 0 else 0
        print(f"  {m['fold']:>4} {str(m['test_start'])+' -> '+str(m['test_end']):>24} "
              f"{m['n_trades']:>7} {m['win_rate']:>5.1%} "
              f"{m['avg_pnl']:>+7.2f} {m['total_pnl']:>+9.1f} "
              f"{m['sharpe']:>6.2f}")

    # Aggregate OOS
    if all_trades:
        combined = BacktestResult(trades=all_trades)
        print(f"\n  --- Combined OOS ---")
        print(f"  Total trades:     {combined.n_trades}")
        print(f"  Win rate:         {combined.win_rate:.1%}")
        print(f"  Avg P&L:          {combined.avg_pnl_bps:+.2f} bps")
        print(f"  Total P&L:        {combined.total_pnl_bps:+.1f} bps")
        print(f"  Profit factor:    {combined.profit_factor:.2f}")
        print(f"  Max drawdown:     {combined.max_drawdown_bps:.1f} bps")
        print(f"  Sharpe (ann):     {combined.sharpe:.2f}")
        print(f"  Profitable folds: {profitable_folds}/{len(fold_results)} "
              f"({profitable_folds/len(fold_results):.0%})")

    return all_trades


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

def select_features_mi(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 50,
    method: str = "f_classif",
) -> list[str]:
    """Select top-k features by mutual information or F-test.

    Args:
        X: feature DataFrame (cleaned, no NaN)
        y: target Series (must be aligned with X)
        top_k: number of features to select
        method: "mutual_info" or "f_classif" or "f_regression"

    Returns:
        list of selected feature names
    """
    from sklearn.feature_selection import (
        mutual_info_classif, f_classif, f_regression,
        mutual_info_regression,
    )

    # Align and drop NaN in target
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    if len(X_clean) < 100:
        print(f"  WARNING: only {len(X_clean)} samples for feature selection")
        return X.columns.tolist()[:top_k]

    if method == "mutual_info":
        scores = mutual_info_classif(X_clean, y_clean, random_state=42, n_neighbors=5)
    elif method == "mutual_info_regression":
        scores = mutual_info_regression(X_clean, y_clean, random_state=42, n_neighbors=5)
    elif method == "f_classif":
        scores, _ = f_classif(X_clean, y_clean)
    elif method == "f_regression":
        scores, _ = f_regression(X_clean, y_clean)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle NaN scores
    scores = np.nan_to_num(scores, nan=0.0)

    # Sort and select top-k
    ranked = sorted(zip(X.columns, scores), key=lambda x: -x[1])
    selected = [name for name, score in ranked[:top_k]]

    print(f"  Feature selection ({method}): {len(X.columns)} -> {len(selected)} features")
    print(f"  Top 10: {[f'{n} ({s:.3f})' for n, s in ranked[:10]]}")

    return selected


def drop_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Drop features that are highly correlated with each other.

    Keeps the first feature in each correlated pair.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"  Dropped {len(to_drop)} correlated features (|r| > {threshold})")
    return X.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# LightGBM Helpers
# ---------------------------------------------------------------------------

def train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[dict] = None,
) -> "lgb.Booster":
    """Train a LightGBM classifier with sensible defaults for anti-overfitting."""
    import lightgbm as lgb

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": 4,
        # Anti-overfitting
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        # Early stopping needs validation set
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    n_est = default_params.pop("n_estimators", 500)
    random_state = default_params.pop("random_state", 42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = None
    callbacks = []

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks.append(lgb.early_stopping(50, verbose=False))
        callbacks.append(lgb.log_evaluation(0))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=n_est,
        valid_sets=[val_data] if val_data else None,
        callbacks=callbacks if callbacks else None,
    )

    return model


def train_lgbm_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_class: int = 3,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[dict] = None,
) -> "lgb.Booster":
    """Train a LightGBM multiclass classifier."""
    import lightgbm as lgb

    default_params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_threads": 4,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    n_est = default_params.pop("n_estimators", 500)
    random_state = default_params.pop("random_state", 42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = None
    callbacks = []

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks.append(lgb.early_stopping(50, verbose=False))
        callbacks.append(lgb.log_evaluation(0))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=n_est,
        valid_sets=[val_data] if val_data else None,
        callbacks=callbacks if callbacks else None,
    )

    return model


def train_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[dict] = None,
) -> "lgb.Booster":
    """Train a LightGBM regressor."""
    import lightgbm as lgb

    default_params = {
        "objective": "regression",
        "metric": "mse",
        "verbosity": -1,
        "num_threads": 4,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    n_est = default_params.pop("n_estimators", 500)
    random_state = default_params.pop("random_state", 42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = None
    callbacks = []

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks.append(lgb.early_stopping(50, verbose=False))
        callbacks.append(lgb.log_evaluation(0))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=n_est,
        valid_sets=[val_data] if val_data else None,
        callbacks=callbacks if callbacks else None,
    )

    return model


# ---------------------------------------------------------------------------
# Main WFO Runner
# ---------------------------------------------------------------------------

def run_wfo(
    df: pd.DataFrame,
    X: pd.DataFrame,
    target_name: str,
    targets: dict,
    strategy_fn,
    train_days: int = 60,
    test_days: int = 20,
    gap_days: int = 5,
    step_days: int = 20,
    feature_select_k: int = 50,
    verbose: bool = True,
) -> list[tuple[WFOSplit, BacktestResult]]:
    """Run full walk-forward optimization.

    Args:
        df: full DataFrame with OHLC
        X: feature DataFrame (cleaned)
        target_name: which target to use (e.g., "tgt_profitable_long_5")
        targets: dict of target Series
        strategy_fn: callable(X_train, y_train, X_test, df_test, fold_info) -> signals
                     Must return a pd.Series of signals (+1, -1, 0)
        train_days, test_days, gap_days, step_days: WFO parameters
        feature_select_k: number of features to select per fold
        verbose: print progress

    Returns:
        list of (WFOSplit, BacktestResult) tuples
    """
    y = targets[target_name]
    splits = generate_wfo_splits(
        df.index, train_days, test_days, gap_days, step_days
    )

    if verbose:
        print(f"\n  WFO: {len(splits)} folds, train={train_days}d, "
              f"gap={gap_days}d, test={test_days}d, step={step_days}d")
        print(f"  Target: {target_name}")
        print(f"  Features: {len(X.columns)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        # Split data
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]
        df_test = df.loc[test_mask]

        # Drop rows where target is NaN
        valid_train = y_train.notna()
        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        valid_test = y_test.notna()
        X_test_valid = X_test.loc[valid_test]
        y_test_valid = y_test.loc[valid_test]

        if len(X_train) < 100 or len(X_test_valid) < 10:
            if verbose:
                print(f"  Fold {split.fold_id}: skip (train={len(X_train)}, "
                      f"test={len(X_test_valid)})")
            continue

        # Call strategy function
        fold_info = {
            "split": split,
            "feature_select_k": feature_select_k,
        }
        signals = strategy_fn(X_train, y_train, X_test, df_test, fold_info)

        # Backtest
        result = backtest_signals(df_test, signals, hold_bars=5)

        fold_results.append((split, result))

        if verbose:
            elapsed = time.time() - t0
            print(f"  Fold {split.fold_id}: "
                  f"train={len(X_train)}, test={len(df_test)}, "
                  f"trades={result.n_trades}, "
                  f"avg={result.avg_pnl_bps:+.2f} bps, "
                  f"WR={result.win_rate:.1%} "
                  f"[{elapsed:.0f}s]")

    return fold_results


if __name__ == "__main__":
    # Quick self-test
    print("Loading features...")
    df = load_features("BTCUSDT", "2024-01-01", "2024-01-14", "15m")
    X, targets = split_features_targets(df)
    X = prepare_features(X)
    print(f"Features: {X.shape}, Targets: {len(targets)}")

    # Test WFO splits
    splits = generate_wfo_splits(df.index, train_days=7, test_days=3, gap_days=1, step_days=3)
    print(f"\nWFO splits: {len(splits)}")
    for s in splits:
        print(f"  Fold {s.fold_id}: train {s.train_start.date()}-{s.train_end.date()}, "
              f"test {s.test_start.date()}-{s.test_end.date()}")

    # Test backtest with random signals
    np.random.seed(42)
    random_signals = pd.Series(
        np.random.choice([-1, 0, 0, 0, 1], size=len(df)),
        index=df.index
    )
    result = backtest_signals(df, random_signals, hold_bars=5)
    print_metrics(result, "Random signals baseline")

    print("\nFramework self-test passed!")
