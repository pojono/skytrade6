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
MAKER_FEE_BPS = 2.0   # Bybit VIP0 maker: 0.0200% per fill
TAKER_FEE_BPS = 5.5   # Bybit VIP0 taker: 0.0550% per fill
# Grid bot uses limit orders → maker fee applies
ROUND_TRIP_FEE_BPS = MAKER_FEE_BPS  # per fill (not per RT)

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

    Clean mechanics:
    - Grid has N levels above and N levels below center price.
    - Each level is a pending order: buy below, sell above.
    - A level can only fill ONCE until the grid is reset.
    - When a buy fills, the corresponding sell level (one step up) becomes active.
    - When a sell fills, the corresponding buy level (one step down) becomes active.
    - This captures the spread between adjacent levels on each round-trip.
    - Grid resets (recenters) every rebalance_interval bars.
    - At reset, any open inventory is closed at market (realized PnL).

    PnL tracking:
    - Each fill is tracked with its price.
    - Inventory is valued at cost basis (FIFO).
    - Equity = cash + unrealized (inventory × current price - cost basis).

    Position sizing:
    - Fixed USD per grid level (capital / n_levels / 2 per side).
    """

    def __init__(self, n_levels=5, fee_bps=7.0,
                 capital_usd=10000, rebalance_interval=12):
        self.n_levels = n_levels
        self.fee_bps = fee_bps
        self.capital_usd = capital_usd
        self.rebalance_interval = rebalance_interval

    def _setup_grid(self, center, spacing):
        """Create grid levels and pending order state."""
        levels = {}
        for i in range(1, self.n_levels + 1):
            levels[f"buy_{i}"] = {"price": center - i * spacing, "active": True, "type": "buy"}
            levels[f"sell_{i}"] = {"price": center + i * spacing, "active": True, "type": "sell"}
        return levels

    def run(self, prices_close, prices_high, prices_low,
            grid_spacings_pct, strategy_name="fixed", paused=None):
        """
        Run grid bot simulation.

        Args:
            prices_close/high/low: price arrays
            grid_spacings_pct: array of grid spacing as FRACTION of price (e.g., 0.005 = 0.5%)
            strategy_name: label
            paused: optional boolean array

        Returns: dict of metrics
        """
        n = len(prices_close)
        if paused is None:
            paused = np.zeros(n, dtype=bool)

        # Per-level USD size
        size_usd = self.capital_usd / (self.n_levels * 2)

        # State
        inventory = []  # list of (quantity, cost_price) for longs; negative qty for shorts
        cash = 0.0
        total_fees = 0.0
        fills = 0
        grid_profits = 0.0  # profit from completed grid round-trips

        # Grid state
        grid_center = prices_close[0]
        grid_spacing = grid_spacings_pct[0] * grid_center
        levels = self._setup_grid(grid_center, grid_spacing)
        last_rebalance = 0

        # Tracking
        equity_curve = np.zeros(n)
        position_history = np.zeros(n)
        spacing_history = np.zeros(n)

        for i in range(n):
            price = prices_close[i]
            high = prices_high[i]
            low = prices_low[i]

            # Rebalance: close inventory at market, recenter grid
            if i - last_rebalance >= self.rebalance_interval and i > 0:
                # Close all inventory at current price
                for qty, cost_p in inventory:
                    pnl = qty * (price - cost_p)
                    cash += pnl
                    fee = abs(qty) * price * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    fills += 1
                inventory = []

                # Recenter grid with current spacing
                grid_center = price
                grid_spacing = grid_spacings_pct[i] * price
                levels = self._setup_grid(grid_center, grid_spacing)
                last_rebalance = i

            spacing_history[i] = grid_spacing

            if paused[i]:
                net_qty = sum(q for q, _ in inventory)
                cost_basis = sum(q * p for q, p in inventory)
                unrealized = net_qty * price - cost_basis
                equity_curve[i] = cash + unrealized
                position_history[i] = net_qty * price  # in USD
                continue

            # Check fills against grid levels
            for key, level in levels.items():
                if not level["active"]:
                    continue

                lp = level["price"]
                qty = size_usd / lp  # units to trade

                if level["type"] == "buy" and low <= lp:
                    # Buy fill
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    inventory.append((qty, lp))
                    fills += 1
                    level["active"] = False

                    # Activate corresponding sell level (one step up)
                    sell_key = key.replace("buy", "sell")
                    if sell_key in levels:
                        levels[sell_key]["active"] = True

                elif level["type"] == "sell" and high >= lp:
                    # Sell fill — close oldest long inventory (FIFO) or open short
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee

                    if inventory and inventory[0][0] > 0:
                        # Close long position
                        old_qty, old_price = inventory.pop(0)
                        profit = old_qty * (lp - old_price)
                        cash += profit
                        grid_profits += profit
                    else:
                        # Open short
                        inventory.append((-qty, lp))

                    fills += 1
                    level["active"] = False

                    # Activate corresponding buy level (one step down)
                    buy_key = key.replace("sell", "buy")
                    if buy_key in levels:
                        levels[buy_key]["active"] = True

            # Mark-to-market
            net_qty = sum(q for q, _ in inventory)
            cost_basis = sum(q * p for q, p in inventory)
            unrealized = net_qty * price - cost_basis
            equity_curve[i] = cash + unrealized
            position_history[i] = net_qty * price  # inventory in USD

        # Close remaining inventory at final price
        final_price = prices_close[-1]
        for qty, cost_p in inventory:
            pnl = qty * (final_price - cost_p)
            cash += pnl
            fee = abs(qty) * final_price * self.fee_bps / 10000
            cash -= fee
            total_fees += fee
        equity_curve[-1] = cash

        # Metrics
        final_equity = cash
        max_equity = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - max_equity
        max_drawdown = np.min(drawdowns)

        n_days = n / 288
        daily_eq = equity_curve[::288]
        daily_returns = np.diff(daily_eq)
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe = 0

        avg_spacing_pct = np.nanmean(spacing_history) / np.nanmean(prices_close) * 100

        return {
            "strategy": strategy_name,
            "total_pnl": final_equity,
            "grid_profits": grid_profits,
            "total_fees": total_fees,
            "fills": fills,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "avg_spacing_pct": avg_spacing_pct,
            "avg_inventory_usd": np.mean(np.abs(position_history)),
            "max_inventory_usd": np.max(np.abs(position_history)),
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
    """Run grid bot comparison: fixed vs adaptive."""
    t_total = time.time()

    print("=" * 70)
    print(f"  GRID BOT SIMULATOR (v2 — proper mechanics)")
    print(f"  Symbol:   {symbol}")
    print(f"  Period:   {start_date} → {end_date}")
    print(f"  Fee:      {MAKER_FEE_BPS} bps maker per fill ({MAKER_FEE_BPS*2} bps RT)")
    print(f"  Capital:  $10,000")
    print(f"  Levels:   5 buy + 5 sell")
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
    warmup = 2500  # ~8.7 days for features + training
    df_sim = df.iloc[warmup:].copy().reset_index(drop=True)
    pred_vol_sim = predicted_vol[warmup:]
    n_sim = len(df_sim)

    prices_close = df_sim["close"].values.astype(float)
    prices_high = df_sim["high"].values.astype(float)
    prices_low = df_sim["low"].values.astype(float)

    # Compute actual 1h range as % of price (for calibration)
    actual_1h_range_pct = []
    for i in range(0, n_sim - 12, 12):
        h = prices_high[i:i+12].max()
        l = prices_low[i:i+12].min()
        actual_1h_range_pct.append((h - l) / prices_close[i])
    median_1h_range_pct = np.median(actual_1h_range_pct)

    # Range scaling factor (from v11: range ≈ vol × 5.6 for 1h)
    K_RANGE = 5.6

    print(f"\n  Simulation period: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price range: ${prices_close.min():.0f} - ${prices_close.max():.0f}")
    print(f"  Median 1h range: {median_1h_range_pct*100:.3f}% (${median_1h_range_pct*np.mean(prices_close):.0f})")

    # --- Build spacing arrays (as fraction of price) ---
    #
    # KEY INSIGHT: Fee per round-trip = 2 × 7 bps = 0.14%.
    # Grid profit per round-trip = spacing between levels.
    # BREAKEVEN: spacing > 0.14%. Need ~2x margin → spacing ≥ 0.25%.
    #
    # With 5 levels each side and spacing S:
    #   Total grid width = 5 × S × 2 = 10S
    #   At S=0.25%, total grid = 2.5% of price
    #   At S=0.50%, total grid = 5.0% of price
    #
    # Median 1h range is ~0.5%, so 0.25% spacing = 2 levels per 1h range.
    # This means ~2 fills per hour in ranging markets.

    fee_per_rt = 2 * MAKER_FEE_BPS / 10000  # 0.0004 at 2 bps maker

    # Fixed spacings at different widths
    fixed_025 = np.full(n_sim, 0.0025)  # 0.25% — minimum profitable
    fixed_050 = np.full(n_sim, 0.0050)  # 0.50% — comfortable margin
    fixed_100 = np.full(n_sim, 0.0100)  # 1.00% — wide, fewer fills

    # Adaptive: predicted_vol × K_RANGE gives predicted 1h range
    # Grid spacing = predicted_range / (2 × n_levels)
    # But floor at a minimum to prevent inventory blowup during trends
    median_pred_vol = np.nanmedian(pred_vol_sim[~np.isnan(pred_vol_sim)])

    def build_adaptive(floor_pct, divisor=2.0):
        """
        Build adaptive spacing array.
        spacing = predicted_1h_range / divisor, floored at floor_pct.

        divisor=2: each level = half the predicted 1h range
          → 5 levels cover 2.5× the 1h range each side
        divisor=5: each level = 1/5 of predicted 1h range
          → 5 levels cover exactly the 1h range each side
        """
        spacings = np.full(n_sim, floor_pct)
        for i in range(n_sim):
            if not np.isnan(pred_vol_sim[i]) and pred_vol_sim[i] > 0:
                predicted_range_pct = pred_vol_sim[i] * K_RANGE
                spacings[i] = max(predicted_range_pct / divisor, floor_pct)
        return np.clip(spacings, floor_pct, 0.0500)

    # Adaptive variants: spacing = predicted_range / divisor
    # divisor=2 → wide grid (each level = half the 1h range)
    # divisor=5 → medium grid (5 levels = 1h range per side)
    adapt_d2 = build_adaptive(0.0010, divisor=2)   # wide: ~0.24% avg
    adapt_d5 = build_adaptive(0.0010, divisor=5)   # medium: ~0.10% avg
    adapt_d2_f025 = build_adaptive(0.0025, divisor=2)  # wide, floor 0.25%
    adapt_d2_f050 = build_adaptive(0.0050, divisor=2)  # wide, floor 0.50%

    # Adaptive + pause during extreme vol
    paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pred_vol_sim[i]) and pred_vol_sim[i] > 3.0 * median_pred_vol:
            paused[i] = True
    pct_paused = 100 * paused.sum() / n_sim

    print(f"  Fee per round-trip: {fee_per_rt*100:.3f}%")
    print(f"  Adapt /2: mean={np.mean(adapt_d2)*100:.3f}%, min={np.min(adapt_d2)*100:.3f}%, max={np.max(adapt_d2)*100:.3f}%")
    print(f"  Adapt /5: mean={np.mean(adapt_d5)*100:.3f}%")
    print(f"  Adapt /2 f025: mean={np.mean(adapt_d2_f025)*100:.3f}%")
    print(f"  Adapt /2 f050: mean={np.mean(adapt_d2_f050)*100:.3f}%")
    print(f"  Paused bars (vol>3×median): {paused.sum():,} ({pct_paused:.1f}%)")

    # --- Run simulations ---
    configs = [
        # (name, spacings_pct, paused, rebalance_bars)
        # Fixed baselines
        ("Fix 0.25% (8h)", fixed_025, None, 96),
        ("Fix 0.25% (24h)", fixed_025, None, 288),
        ("Fix 0.50% (8h)", fixed_050, None, 96),
        ("Fix 0.50% (24h)", fixed_050, None, 288),
        ("Fix 1.00% (24h)", fixed_100, None, 288),
        # Adaptive /2 (each level = half predicted 1h range)
        ("Adapt /2 (8h)", adapt_d2, None, 96),
        ("Adapt /2 (24h)", adapt_d2, None, 288),
        # Adaptive /5 (5 levels = 1h range per side)
        ("Adapt /5 (8h)", adapt_d5, None, 96),
        ("Adapt /5 (24h)", adapt_d5, None, 288),
        # Adaptive /2 with floors
        ("Adapt /2 f025 (8h)", adapt_d2_f025, None, 96),
        ("Adapt /2 f025 (24h)", adapt_d2_f025, None, 288),
        ("Adapt /2 f050 (24h)", adapt_d2_f050, None, 288),
        # Adaptive + pause
        ("Adapt /2 +P (24h)", adapt_d2, paused, 288),
    ]

    print(f"\n  Running {len(configs)} strategies...\n")
    results = []

    for name, spacings, pause_mask, rebal_interval in configs:
        t_s = time.time()
        bot = GridBotSimulator(
            n_levels=5, fee_bps=ROUND_TRIP_FEE_BPS,
            capital_usd=10000, rebalance_interval=rebal_interval,
        )
        r = bot.run(prices_close, prices_high, prices_low,
                     spacings, strategy_name=name, paused=pause_mask)
        results.append(r)
        elapsed = time.time() - t_s

        print(f"  {name}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} (grid: ${r['grid_profits']:+.2f}, fees: -${r['total_fees']:.2f})")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills: {r['fills']:,} ({r['fills_per_day']:.1f}/day) | "
              f"Sharpe: {r['sharpe']:.2f}")
        print(f"    Max DD: ${r['max_drawdown']:.2f} | Avg inv: ${r['avg_inventory_usd']:.0f} | "
              f"Max inv: ${r['max_inventory_usd']:.0f}")
        print(f"    Avg spacing: {r['avg_spacing_pct']:.4f}% ({elapsed:.1f}s)\n")

    # --- Summary table ---
    print(f"  {'='*90}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days, ${prices_close[0]:.0f}→${prices_close[-1]:.0f})")
    print(f"  {'='*90}")
    print(f"  {'Strategy':<25s} {'PnL':>10s} {'Grid$':>10s} {'Fees':>8s} {'PnL/d':>8s} "
          f"{'Fills':>7s} {'Sharpe':>7s} {'MaxDD':>10s} {'Spc%':>7s}")
    print(f"  {'-'*92}")
    for r in results:
        print(f"  {r['strategy']:<25s} ${r['total_pnl']:>+9.2f} ${r['grid_profits']:>+9.2f} "
              f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+7.2f} "
              f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>9.2f} "
              f"{r['avg_spacing_pct']:>6.4f}%")

    # --- Regime analysis ---
    print(f"\n  {'='*70}")
    print(f"  REGIME ANALYSIS")
    print(f"  {'='*70}")

    valid_mask = ~np.isnan(pred_vol_sim[:n_sim])
    if valid_mask.sum() > 0:
        calm_idx = valid_mask & (pred_vol_sim[:n_sim] < median_pred_vol)
        vol_idx = valid_mask & (pred_vol_sim[:n_sim] >= median_pred_vol)

        calm_adapt = np.mean(adapt_d2[calm_idx]) * 100
        vol_adapt = np.mean(adapt_d2[vol_idx]) * 100
        fixed_pct = 0.50  # reference: 0.50% fixed

        print(f"  Calm periods ({calm_idx.sum():,} bars, {calm_idx.sum()/288:.0f} days):")
        print(f"    Adaptive: {calm_adapt:.4f}% | Fixed: {fixed_pct:.4f}% | "
              f"{(1-calm_adapt/fixed_pct)*100:+.0f}% vs fixed")
        print(f"  Volatile periods ({vol_idx.sum():,} bars, {vol_idx.sum()/288:.0f} days):")
        print(f"    Adaptive: {vol_adapt:.4f}% | Fixed: {fixed_pct:.4f}% | "
              f"{(vol_adapt/fixed_pct-1)*100:+.0f}% vs fixed")

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
