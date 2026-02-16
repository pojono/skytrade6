#!/usr/bin/env python3
"""
Grid Bot Simulator — Vol-Adaptive vs Fixed Grid.

Uses proven findings from v9-v13:
- Ridge vol prediction (R²=0.34 at 1h) for adaptive grid spacing
- Range ≈ vol × k (k=5.6 for 1h, 11.0 for 4h)
- Symmetric grid (direction unpredictable)
- P50 (aggressive) and P90 (conservative) grid widths

Grid mechanics:
- N grid levels placed symmetrically around current price
- Each level is a limit order: buy below, sell above
- When price crosses a level, the order fills → position changes
- Profit = captured spread between grid levels
- Risk = holding inventory when price trends away

Strategies compared:
1. Fixed grid: constant spacing based on historical median range
2. Adaptive grid: spacing from Ridge vol prediction × k
3. Adaptive + pause: same as #2 but pause when vol > 3× median
4. Adaptive + rebalance: same as #2 but recenter grid periodically

All use walk-forward vol prediction (no lookahead).
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from regime_detection import load_bars, compute_regime_features

PARQUET_DIR = Path("./parquet")
ROUND_TRIP_FEE_BPS = 7.0  # Bybit VIP0

VOL_FEATURES = [
    "parkvol_1h", "parkvol_2h", "parkvol_4h", "parkvol_8h", "parkvol_24h",
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_sma_24h", "vol_ratio_bar",
    "trade_intensity_ratio", "parkinson_vol",
    "bar_eff_1h", "bar_eff_4h", "bar_efficiency",
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_2h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    "sign_persist_1h", "sign_persist_2h",
    "imbalance_1h", "imbalance_4h", "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "price_vs_sma_2h", "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "vol_imbalance",
]


# ---------------------------------------------------------------------------
# Walk-forward vol prediction
# ---------------------------------------------------------------------------

def predict_vol_walkforward(df, feature_cols, target_col="fwd_vol", min_train=2000):
    """
    Walk-forward Ridge vol prediction. Retrain every 288 bars (1 day).
    Returns array of predicted vol (NaN where unavailable).
    """
    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    n = len(df)

    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)

    retrain_interval = 288
    last_train = -retrain_interval  # force first train

    for i in range(min_train, n):
        if i - last_train >= retrain_interval:
            train_mask = ~np.isnan(y[:i])
            X_train = np.nan_to_num(X[:i][train_mask], nan=0, posinf=0, neginf=0)
            y_train = y[:i][train_mask]
            if len(y_train) < 100:
                continue
            scaler.fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            last_train = i

        x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
        pred = model.predict(scaler.transform(x_i))[0]
        predictions[i] = max(pred, 1e-8)  # floor at tiny positive

    return predictions


# ---------------------------------------------------------------------------
# Grid Bot Simulator
# ---------------------------------------------------------------------------

class GridBotSimulator:
    """
    Simulates a symmetric grid bot on historical OHLC data.

    Grid mechanics:
    - N_LEVELS buy orders below center, N_LEVELS sell orders above
    - When price crosses a buy level → fill buy, position += 1 unit
    - When price crosses a sell level → fill sell, position -= 1 unit
    - Each completed round-trip (buy+sell or sell+buy) captures 1 grid spacing
    - Fees deducted per fill

    Position limits prevent unlimited accumulation.
    """

    def __init__(self, n_levels=5, max_position=5, fee_bps=7.0,
                 capital_usd=10000, rebalance_interval=12,
                 pause_vol_mult=3.0):
        self.n_levels = n_levels
        self.max_position = max_position
        self.fee_bps = fee_bps
        self.capital_usd = capital_usd
        self.rebalance_interval = rebalance_interval  # bars between grid recentering
        self.pause_vol_mult = pause_vol_mult

    def run(self, prices_close, prices_high, prices_low,
            grid_spacings, strategy_name="fixed", paused=None):
        """
        Run grid bot simulation.

        Args:
            prices_close: array of close prices
            prices_high: array of high prices
            prices_low: array of low prices
            grid_spacings: array of grid spacing (in price units) per bar
            strategy_name: label for logging
            paused: optional boolean array, True = don't trade this bar

        Returns: dict of metrics
        """
        n = len(prices_close)
        if paused is None:
            paused = np.zeros(n, dtype=bool)

        # State
        position = 0  # in grid units (-max to +max)
        cash = 0.0  # realized PnL in USD
        total_fees = 0.0
        fills = 0
        round_trips = 0

        # Grid state
        grid_center = prices_close[0]
        grid_spacing = grid_spacings[0]
        last_rebalance = 0

        # Tracking
        equity_curve = np.zeros(n)
        position_history = np.zeros(n)
        spacing_history = np.zeros(n)
        fill_bars = []

        for i in range(n):
            price = prices_close[i]
            high = prices_high[i]
            low = prices_low[i]
            spacing = grid_spacings[i]

            # Rebalance grid center periodically
            if i - last_rebalance >= self.rebalance_interval:
                grid_center = price
                grid_spacing = spacing
                last_rebalance = i

            spacing_history[i] = grid_spacing

            if paused[i]:
                # Still track equity but don't trade
                unrealized = position * (price - grid_center) * (self.capital_usd / price / self.n_levels)
                equity_curve[i] = cash + unrealized
                position_history[i] = position
                continue

            # Check each grid level for fills using high/low
            # Buy levels below center, sell levels above
            for level_idx in range(1, self.n_levels + 1):
                buy_level = grid_center - level_idx * grid_spacing
                sell_level = grid_center + level_idx * grid_spacing

                # Size per grid level: spread capital across levels
                size_usd = self.capital_usd / self.n_levels
                size_units = size_usd / price

                # Buy fill: low touched buy level
                if low <= buy_level and position < self.max_position:
                    fill_price = buy_level
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    # Track the buy: we'll profit when we sell higher
                    cash -= size_units * fill_price  # pay for units
                    position += 1
                    fills += 1
                    fill_bars.append(i)

                # Sell fill: high touched sell level
                if high >= sell_level and position > -self.max_position:
                    fill_price = sell_level
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    cash += size_units * fill_price  # receive for units
                    position -= 1
                    fills += 1
                    fill_bars.append(i)

                    # Count round trips (simplified: each sell after buy or vice versa)
                    if position >= 0:
                        round_trips += 1

            # Mark-to-market equity
            size_usd = self.capital_usd / self.n_levels
            unrealized = position * (price - grid_center) * (size_usd / price)
            equity_curve[i] = cash + unrealized
            position_history[i] = position

        # Final metrics
        final_equity = equity_curve[-1]
        max_equity = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - max_equity
        max_drawdown = np.min(drawdowns)

        # Annualized metrics
        n_days = n / 288  # 5m bars
        daily_returns = np.diff(equity_curve[::288]) if n > 288 else np.array([0])
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe = 0

        avg_spacing_pct = np.nanmean(spacing_history) / np.nanmean(prices_close) * 100

        return {
            "strategy": strategy_name,
            "total_pnl": final_equity,
            "total_fees": total_fees,
            "fills": fills,
            "round_trips": round_trips,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "avg_spacing_pct": avg_spacing_pct,
            "avg_position": np.mean(np.abs(position_history)),
            "max_position": np.max(np.abs(position_history)),
            "n_days": n_days,
            "pnl_per_day": final_equity / max(n_days, 1),
            "fills_per_day": fills / max(n_days, 1),
            "equity_curve": equity_curve,
            "position_history": position_history,
            "spacing_history": spacing_history,
        }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_grid_experiment(symbol, start_date, end_date):
    """Run grid bot comparison: fixed vs adaptive vs adaptive+pause."""
    t_total = time.time()

    print("=" * 70)
    print(f"  GRID BOT SIMULATOR")
    print(f"  Symbol:   {symbol}")
    print(f"  Period:   {start_date} → {end_date}")
    print(f"  Fee:      {ROUND_TRIP_FEE_BPS} bps RT")
    print(f"  Capital:  $10,000")
    print("=" * 70)

    # --- Load data ---
    print(f"\n  Step 1: Loading 5m bars...", flush=True)
    df = load_bars(symbol, start_date, end_date)
    print(f"  Loaded {len(df):,} bars in {time.time()-t_total:.0f}s")

    # --- Compute features ---
    print(f"\n  Step 2: Computing regime features...", flush=True)
    t1 = time.time()
    df = compute_regime_features(df)
    print(f"  Features in {time.time()-t1:.0f}s")

    # --- Compute forward vol target ---
    print(f"\n  Step 3: Computing forward vol target...", flush=True)
    ret = df["returns"].values
    n = len(df)
    fwd_vol = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol[i] = np.std(ret[i+1:i+13])  # 1h forward vol
    df["fwd_vol"] = fwd_vol

    # --- Walk-forward vol prediction ---
    print(f"\n  Step 4: Walk-forward vol prediction...", flush=True)
    t2 = time.time()
    predicted_vol = predict_vol_walkforward(df, VOL_FEATURES, "fwd_vol")
    valid = np.sum(~np.isnan(predicted_vol))
    print(f"  {valid:,} valid predictions ({100*valid/n:.0f}%) in {time.time()-t2:.0f}s")

    # --- Prepare simulation data ---
    # Skip warmup period (need features + training data)
    warmup = 2500  # ~8.7 days
    df_sim = df.iloc[warmup:].copy().reset_index(drop=True)
    pred_vol_sim = predicted_vol[warmup:]
    n_sim = len(df_sim)

    prices_close = df_sim["close"].values.astype(float)
    prices_high = df_sim["high"].values.astype(float)
    prices_low = df_sim["low"].values.astype(float)

    # Range scaling factor (from v11: range/vol ≈ 5.6 for 1h)
    K_RANGE = 5.6

    # --- Strategy 1: Fixed grid ---
    # Use historical median range as fixed spacing
    actual_ranges = np.abs(prices_high - prices_low)
    median_range = np.nanmedian(actual_ranges)
    fixed_spacings = np.full(n_sim, median_range)

    print(f"\n  Simulation period: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price range: ${prices_close.min():.0f} - ${prices_close.max():.0f}")
    print(f"  Median bar range: ${median_range:.2f}")
    print(f"  Median 1h range: ${median_range * 12:.0f} (approx)")

    # --- Strategy 2: Adaptive grid ---
    # Grid spacing = predicted_vol × K × price
    adaptive_spacings = np.full(n_sim, median_range)  # default to fixed
    for i in range(n_sim):
        if not np.isnan(pred_vol_sim[i]) and pred_vol_sim[i] > 0:
            # predicted_vol is in return units, convert to price
            adaptive_spacings[i] = pred_vol_sim[i] * K_RANGE * prices_close[i]
    # Clip to reasonable range (0.2x to 5x of median)
    adaptive_spacings = np.clip(adaptive_spacings, median_range * 0.2, median_range * 5.0)

    # --- Strategy 3: Adaptive + pause during extreme vol ---
    median_pred_vol = np.nanmedian(pred_vol_sim[~np.isnan(pred_vol_sim)])
    paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pred_vol_sim[i]) and pred_vol_sim[i] > 3.0 * median_pred_vol:
            paused[i] = True
    pct_paused = 100 * paused.sum() / n_sim

    # --- Strategy 4: Wider adaptive (P90-like, 1.7x safety factor) ---
    wide_spacings = adaptive_spacings * 1.7

    print(f"\n  Adaptive spacing stats:")
    print(f"    Mean: ${np.mean(adaptive_spacings):.2f} ({np.mean(adaptive_spacings)/np.mean(prices_close)*100:.3f}%)")
    print(f"    Fixed: ${median_range:.2f} ({median_range/np.mean(prices_close)*100:.3f}%)")
    print(f"    Ratio adaptive/fixed: {np.mean(adaptive_spacings)/median_range:.2f}x")
    print(f"    Paused bars: {paused.sum():,} ({pct_paused:.1f}%)")

    # --- Run simulations ---
    configs = [
        ("Fixed", fixed_spacings, None, 12),
        ("Adaptive (P50)", adaptive_spacings, None, 12),
        ("Adaptive + Pause", adaptive_spacings, paused, 12),
        ("Adaptive Wide (P90)", wide_spacings, None, 12),
        ("Fixed (rebal 1h)", fixed_spacings, None, 12),
        ("Fixed (rebal 4h)", fixed_spacings, None, 48),
        ("Adaptive (rebal 4h)", adaptive_spacings, None, 48),
    ]

    print(f"\n  Running {len(configs)} strategies...")
    results = []

    for name, spacings, pause_mask, rebal_interval in configs:
        t_s = time.time()
        bot = GridBotSimulator(
            n_levels=5, max_position=5, fee_bps=ROUND_TRIP_FEE_BPS,
            capital_usd=10000, rebalance_interval=rebal_interval,
            pause_vol_mult=3.0
        )
        r = bot.run(prices_close, prices_high, prices_low,
                     spacings, strategy_name=name, paused=pause_mask)
        results.append(r)
        elapsed = time.time() - t_s

        print(f"\n  {name}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} | Fees: ${r['total_fees']:.2f} | "
              f"Fills: {r['fills']:,} | RTs: {r['round_trips']:,}")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills/day: {r['fills_per_day']:.1f} | "
              f"Sharpe: {r['sharpe']:.2f}")
        print(f"    Max DD: ${r['max_drawdown']:.2f} | Avg pos: {r['avg_position']:.1f} | "
              f"Max pos: {r['max_position']:.0f}")
        print(f"    Avg spacing: {r['avg_spacing_pct']:.3f}% | ({elapsed:.1f}s)")

    # --- Summary table ---
    print(f"\n  {'='*70}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days)")
    print(f"  {'='*70}")
    print(f"  {'Strategy':<25s} {'PnL':>10s} {'PnL/day':>10s} {'Fills':>8s} {'Sharpe':>8s} {'MaxDD':>10s} {'Spacing':>8s}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['strategy']:<25s} ${r['total_pnl']:>+9.2f} ${r['pnl_per_day']:>+8.2f} "
              f"{r['fills']:>7,d} {r['sharpe']:>8.2f} ${r['max_drawdown']:>9.2f} "
              f"{r['avg_spacing_pct']:>7.3f}%")

    # --- Regime analysis ---
    print(f"\n  {'='*70}")
    print(f"  REGIME ANALYSIS")
    print(f"  {'='*70}")

    # Split into calm vs volatile periods
    if median_pred_vol > 0:
        calm_mask = pred_vol_sim[~np.isnan(pred_vol_sim)] < median_pred_vol
        vol_mask = pred_vol_sim[~np.isnan(pred_vol_sim)] >= median_pred_vol

        # Compare adaptive vs fixed spacing in each regime
        valid_mask = ~np.isnan(pred_vol_sim)
        calm_idx = valid_mask & (pred_vol_sim < median_pred_vol)
        vol_idx = valid_mask & (pred_vol_sim >= median_pred_vol)

        calm_adaptive = np.mean(adaptive_spacings[calm_idx[:n_sim]])
        calm_fixed = median_range
        vol_adaptive = np.mean(adaptive_spacings[vol_idx[:n_sim]])
        vol_fixed = median_range

        print(f"  Calm periods ({calm_idx.sum():,} bars):")
        print(f"    Adaptive spacing: ${calm_adaptive:.2f} | Fixed: ${calm_fixed:.2f} | "
              f"Savings: ${calm_fixed - calm_adaptive:.2f} ({(1-calm_adaptive/calm_fixed)*100:.0f}% tighter)")
        print(f"  Volatile periods ({vol_idx.sum():,} bars):")
        print(f"    Adaptive spacing: ${vol_adaptive:.2f} | Fixed: ${vol_fixed:.2f} | "
              f"Wider by: ${vol_adaptive - vol_fixed:.2f} ({(vol_adaptive/vol_fixed-1)*100:.0f}% wider)")

    elapsed_total = time.time() - t_total
    print(f"\n✅ {symbol} grid bot simulation complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {symbol} in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Grid bot simulator")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    run_grid_experiment(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
