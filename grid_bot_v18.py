#!/usr/bin/env python3
"""
grid_bot_v18.py — Grid Bot with Orderbook Features

Building on v17 (regime-filtered grid bot), adds:
  1. ob_mid_volatility for better vol prediction → smarter pause/spacing
  2. Futures-spot depth ratio for rebalance timing (IC=-0.10 at 4h)
  3. OB-enhanced vol-adaptive spacing

Key findings from v23 research:
  - ob_mid_volatility is #1 feature for vol prediction (Ridge R² +5.4% walk-forward)
  - Depth ratio z>1.5 at 4h: +12.9 bps avg, 53.1% WR, Sharpe 4.87
  - Depth imbalance is contrarian — NOT useful for direction
  - OB spread_std is 4.4× higher in volatile regimes → good pause signal
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
warnings.filterwarnings("ignore", category=DeprecationWarning)

from regime_detection import compute_regime_features, PARQUET_DIR, SOURCE, INTERVAL_5M_US

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE_BPS = 2.0

# v17 vol features (baseline)
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

# OB features that improve vol prediction
OB_VOL_FEATURES = [
    "ob_mid_volatility", "ob_spread_std", "ob_spread_max",
    "ob_bid_depth_cv", "ob_ask_depth_cv",
    "ob_bid_wall_frac", "ob_ask_wall_frac",
    "ob_total_depth_mean",
    "ob_imb_1bps_std", "ob_imb_2bps_std",
]


# ---------------------------------------------------------------------------
# Data loading (reuse v17 bar loading + add OB features)
# ---------------------------------------------------------------------------

def load_ob_features_5m(symbol, market="futures"):
    """Load 5-min OB features."""
    feat_dir = PARQUET_DIR / symbol / "ob_features_5m" / f"bybit_{market}"
    if not feat_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(feat_dir.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_bars_v18(symbol, start_date, end_date):
    """Load 5m bars from v17 pipeline + merge OB features."""
    from grid_bot_v17 import load_bars_v17, compute_informed_composite

    df = load_bars_v17(symbol, start_date, end_date)
    if df.empty:
        return df

    df = compute_regime_features(df)
    df = compute_informed_composite(df)

    # Load and merge futures OB features
    fut_ob = load_ob_features_5m(symbol, "futures")
    if not fut_ob.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp_us"),
            fut_ob.sort_values("timestamp_us"),
            on="timestamp_us", tolerance=300_000_000, direction="nearest",
        )
        n_matched = df["ob_mid_volatility"].notna().sum() if "ob_mid_volatility" in df.columns else 0
        print(f"  Futures OB merged: {n_matched}/{len(df)} bars ({n_matched/len(df)*100:.1f}%)")
    else:
        print(f"  No futures OB features found for {symbol}")

    # Load and merge spot OB features (for basis)
    spot_ob = load_ob_features_5m(symbol, "spot")
    if not spot_ob.empty:
        spot_cols = {c: f"spot_{c}" for c in spot_ob.columns if c != "timestamp_us"}
        spot_renamed = spot_ob.rename(columns=spot_cols)
        df = pd.merge_asof(
            df.sort_values("timestamp_us"),
            spot_renamed.sort_values("timestamp_us"),
            on="timestamp_us", tolerance=300_000_000, direction="nearest",
        )
        # Compute basis features
        if "ob_total_depth_mean" in df.columns and "spot_ob_total_depth_mean" in df.columns:
            df["basis_depth_ratio"] = df["ob_total_depth_mean"] / df["spot_ob_total_depth_mean"].clip(lower=0.01)
            n_basis = df["basis_depth_ratio"].notna().sum()
            print(f"  Basis features computed: {n_basis} bars")
    else:
        print(f"  No spot OB features found for {symbol}")

    return df


# ---------------------------------------------------------------------------
# Walk-forward Ridge (enhanced with OB features)
# ---------------------------------------------------------------------------

def walkforward_ridge(df, feature_cols, target_col, min_train=2000):
    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    n = len(df)
    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)
    retrain_interval = 288

    last_train = -retrain_interval
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
        predictions[i] = max(model.predict(scaler.transform(x_i))[0], 1e-8)
    return predictions


# ---------------------------------------------------------------------------
# GridBotSimulator (from v17)
# ---------------------------------------------------------------------------

class GridBotSimulator:
    def __init__(self, n_levels=5, fee_bps=2.0, capital_usd=10000):
        self.n_levels = n_levels
        self.fee_bps = fee_bps
        self.capital_usd = capital_usd

    def _setup_grid(self, center, spacing):
        levels = {}
        for i in range(1, self.n_levels + 1):
            levels[f"buy_{i}"] = {"price": center - i * spacing, "active": True, "type": "buy"}
            levels[f"sell_{i}"] = {"price": center + i * spacing, "active": True, "type": "sell"}
        return levels

    def run(self, prices_close, prices_high, prices_low, spacings,
            strategy_name="", paused=None, rebalance_intervals=None,
            close_on_pause=False):
        n = len(prices_close)
        if paused is None:
            paused = np.zeros(n, dtype=bool)
        if rebalance_intervals is None:
            rebalance_intervals = np.full(n, 288, dtype=int)
        elif isinstance(rebalance_intervals, (int, float)):
            rebalance_intervals = np.full(n, int(rebalance_intervals), dtype=int)
        if isinstance(spacings, (int, float)):
            spacings = np.full(n, spacings)

        size_usd = self.capital_usd / (self.n_levels * 2)
        inventory = []
        cash = 0.0
        total_fees = 0.0
        fills = 0
        grid_profits = 0.0

        grid_center = prices_close[0]
        grid_spacing = spacings[0] * grid_center
        levels = self._setup_grid(grid_center, grid_spacing)
        last_rebalance = 0

        equity_curve = np.zeros(n)

        for i in range(n):
            price = prices_close[i]
            high = prices_high[i]
            low = prices_low[i]

            rebal_interval = int(rebalance_intervals[i])
            if i - last_rebalance >= rebal_interval and i > 0:
                for qty, cost_p in inventory:
                    pnl = qty * (price - cost_p)
                    cash += pnl
                    fee = abs(qty) * price * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    fills += 1
                inventory = []
                grid_center = price
                grid_spacing = spacings[i] * price
                levels = self._setup_grid(grid_center, grid_spacing)
                last_rebalance = i

            if paused[i]:
                if close_on_pause and inventory:
                    for qty, cost_p in inventory:
                        pnl = qty * (price - cost_p)
                        cash += pnl
                        fee = abs(qty) * price * self.fee_bps / 10000
                        cash -= fee
                        total_fees += fee
                        fills += 1
                    inventory = []
                    grid_center = price
                    grid_spacing = spacings[i] * price
                    levels = self._setup_grid(grid_center, grid_spacing)
                    last_rebalance = i
                net_qty = sum(q for q, _ in inventory)
                cost_basis = sum(q * p for q, p in inventory)
                unrealized = net_qty * price - cost_basis
                equity_curve[i] = cash + unrealized
                continue

            for key, level in levels.items():
                if not level["active"]:
                    continue
                lp = level["price"]
                qty = size_usd / lp

                if level["type"] == "buy" and low <= lp:
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    inventory.append((qty, lp))
                    fills += 1
                    level["active"] = False
                    sell_key = key.replace("buy", "sell")
                    if sell_key in levels:
                        levels[sell_key]["active"] = True

                elif level["type"] == "sell" and high >= lp:
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    if inventory and inventory[0][0] > 0:
                        old_qty, old_price = inventory.pop(0)
                        profit = old_qty * (lp - old_price)
                        cash += profit
                        grid_profits += profit
                    else:
                        inventory.append((-qty, lp))
                    fills += 1
                    level["active"] = False
                    buy_key = key.replace("sell", "buy")
                    if buy_key in levels:
                        levels[buy_key]["active"] = True

            net_qty = sum(q for q, _ in inventory)
            cost_basis = sum(q * p for q, p in inventory)
            unrealized = net_qty * price - cost_basis
            equity_curve[i] = cash + unrealized

        final_price = prices_close[-1]
        for qty, cost_p in inventory:
            pnl = qty * (final_price - cost_p)
            cash += pnl
            fee = abs(qty) * final_price * self.fee_bps / 10000
            cash -= fee
            total_fees += fee
        equity_curve[-1] = cash

        final_equity = cash
        max_equity = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - max_equity
        max_dd = np.min(drawdowns)

        n_days = n / 288
        daily_eq = equity_curve[::288]
        daily_returns = np.diff(daily_eq)
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe = 0

        return {
            "strategy": strategy_name,
            "total_pnl": final_equity,
            "grid_profits": grid_profits,
            "total_fees": total_fees,
            "fills": fills,
            "fills_per_day": fills / max(n_days, 1),
            "pnl_per_day": final_equity / max(n_days, 1),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_v18(symbol, start_date, end_date):
    t_total = time.time()
    print("=" * 70)
    print(f"  v18: Grid Bot with Orderbook Features — {symbol}")
    print(f"  Period: {start_date} → {end_date}")
    print("=" * 70)

    # --- Load data ---
    print(f"\n  Loading data...", flush=True)
    df = load_bars_v18(symbol, start_date, end_date)
    if df.empty:
        print("  No data loaded!")
        return
    print(f"  {len(df):,} bars loaded in {time.time()-t_total:.0f}s")

    # --- Forward targets ---
    ret = df["returns"].values
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    fwd_vol_1h = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol_1h[i] = np.std(ret[i+1:i+13])
    df["fwd_vol_1h"] = fwd_vol_1h

    # --- Walk-forward vol prediction: OHLCV only vs OHLCV+OB ---
    warmup = 2500

    print(f"\n  Walk-forward vol prediction (OHLCV only)...", flush=True)
    t1 = time.time()
    pred_vol_ohlcv = walkforward_ridge(df, VOL_FEATURES, "fwd_vol_1h", min_train=2000)
    print(f"    Done in {time.time()-t1:.0f}s")

    print(f"  Walk-forward vol prediction (OHLCV + OB)...", flush=True)
    t1 = time.time()
    pred_vol_ob = walkforward_ridge(df, VOL_FEATURES + OB_VOL_FEATURES, "fwd_vol_1h", min_train=2000)
    print(f"    Done in {time.time()-t1:.0f}s")

    # Compare predictions
    valid_mask = ~np.isnan(pred_vol_ohlcv[warmup:]) & ~np.isnan(fwd_vol_1h[warmup:])
    if valid_mask.sum() > 100:
        actual = fwd_vol_1h[warmup:][valid_mask]
        pred_o = pred_vol_ohlcv[warmup:][valid_mask]
        pred_ob_v = pred_vol_ob[warmup:][valid_mask]
        corr_o = np.corrcoef(actual, pred_o)[0, 1]
        corr_ob = np.corrcoef(actual, pred_ob_v)[0, 1]
        print(f"    OHLCV corr: {corr_o:.4f}")
        print(f"    OHLCV+OB corr: {corr_ob:.4f} ({(corr_ob-corr_o)/corr_o*100:+.1f}%)")

    # --- Prepare simulation data ---
    df_sim = df.iloc[warmup:].copy().reset_index(drop=True)
    n_sim = len(df_sim)
    prices_close = df_sim["close"].values.astype(float)
    prices_high = df_sim["high"].values.astype(float)
    prices_low = df_sim["low"].values.astype(float)

    pv_ohlcv = pred_vol_ohlcv[warmup:]
    pv_ob = pred_vol_ob[warmup:]
    median_vol_ohlcv = np.nanmedian(pv_ohlcv[~np.isnan(pv_ohlcv)])
    median_vol_ob = np.nanmedian(pv_ob[~np.isnan(pv_ob)])

    # Backward-looking features
    eff_4h = df_sim["efficiency_4h"].values if "efficiency_4h" in df_sim.columns else np.full(n_sim, 0.3)
    informed_z = df_sim["informed_composite_z"].values if "informed_composite_z" in df_sim.columns else np.zeros(n_sim)
    parkvol_1h = df_sim["parkvol_1h"].values if "parkvol_1h" in df_sim.columns else np.full(n_sim, np.nan)
    parkvol_median = np.nanmedian(parkvol_1h[~np.isnan(parkvol_1h)])

    # OB features for direct use
    ob_mid_vol = df_sim["ob_mid_volatility"].values if "ob_mid_volatility" in df_sim.columns else np.full(n_sim, np.nan)
    ob_spread_std = df_sim["ob_spread_std"].values if "ob_spread_std" in df_sim.columns else np.full(n_sim, np.nan)
    ob_mid_vol_median = np.nanmedian(ob_mid_vol[~np.isnan(ob_mid_vol)])
    ob_spread_std_median = np.nanmedian(ob_spread_std[~np.isnan(ob_spread_std)])

    # Basis depth ratio
    has_basis = "basis_depth_ratio" in df_sim.columns
    if has_basis:
        basis_ratio = df_sim["basis_depth_ratio"].values
        basis_mean = np.nanmean(basis_ratio[~np.isnan(basis_ratio)])
        basis_std = np.nanstd(basis_ratio[~np.isnan(basis_ratio)])
        basis_z = np.full(n_sim, np.nan)
        # Walk-forward z-score with 7-day lookback
        for i in range(288, n_sim):
            window = basis_ratio[max(0, i-288*7):i]
            window = window[~np.isnan(window)]
            if len(window) > 50:
                basis_z[i] = (basis_ratio[i] - np.mean(window)) / max(np.std(window), 1e-10)
        print(f"  Basis depth ratio: mean={basis_mean:.2f}, std={basis_std:.2f}")
    else:
        basis_z = np.full(n_sim, np.nan)
        print(f"  No basis depth ratio available")

    print(f"\n  Simulation: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price: ${prices_close.min():.2f} - ${prices_close.max():.2f}")
    print(f"  Median vol (OHLCV): {median_vol_ohlcv:.6f}")
    print(f"  Median vol (OB): {median_vol_ob:.6f}")
    print(f"  Median ob_mid_vol: {ob_mid_vol_median:.4f}" if not np.isnan(ob_mid_vol_median) else "  ob_mid_vol: N/A")

    bot = GridBotSimulator(n_levels=5, fee_bps=MAKER_FEE_BPS)
    strategies = []

    fixed_spacing = np.full(n_sim, 0.0100)
    fixed_rebal = np.full(n_sim, 288, dtype=int)

    # N5b informed rebalance
    n5b_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            n5b_rebal[i] = 12

    # =====================================================================
    # BASELINE STRATEGIES (from v17 for comparison)
    # =====================================================================

    # S0: Baseline
    strategies.append({
        "name": "S0: Fix 1.00% (24h)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # B1: v17 best — vol pause (OHLCV) + informed rebalance
    b1_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pv_ohlcv[i]) and pv_ohlcv[i] > median_vol_ohlcv * 1.5:
            b1_paused[i] = True
    pct = b1_paused.sum() / n_sim * 100
    print(f"\n  B1 OHLCV vol pause (>1.5x): {pct:.1f}% paused")
    strategies.append({
        "name": "B1: OHLCV VolP+InfR",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": b1_paused,
    })

    # B2: v17 best with 1h rebalance
    rebal_1h = np.full(n_sim, 12, dtype=int)
    strategies.append({
        "name": "B2: OHLCV 1%+1hR+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_1h.copy(),
        "paused": b1_paused.copy(),
    })

    # =====================================================================
    # OB-ENHANCED STRATEGIES
    # =====================================================================

    # --- O1: OB-enhanced vol pause (OHLCV+OB prediction) ---
    o1_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pv_ob[i]) and pv_ob[i] > median_vol_ob * 1.5:
            o1_paused[i] = True
    pct = o1_paused.sum() / n_sim * 100
    print(f"  O1 OB vol pause (>1.5x): {pct:.1f}% paused")
    strategies.append({
        "name": "O1: OB VolP+InfR",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o1_paused,
    })

    # --- O2: Direct ob_mid_volatility pause (no ML, pure backward-looking) ---
    o2_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(ob_mid_vol[i]) and ob_mid_vol[i] > ob_mid_vol_median * 1.5:
            o2_paused[i] = True
    pct = o2_paused.sum() / n_sim * 100
    print(f"  O2 ob_mid_vol pause (>1.5x): {pct:.1f}% paused")
    strategies.append({
        "name": "O2: OBmidVol pause",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o2_paused,
    })

    # --- O3: ob_spread_std pause (4.4× higher in volatile — strong discriminator) ---
    o3_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(ob_spread_std[i]) and ob_spread_std[i] > ob_spread_std_median * 2.0:
            o3_paused[i] = True
    pct = o3_paused.sum() / n_sim * 100
    print(f"  O3 ob_spread_std pause (>2x): {pct:.1f}% paused")
    strategies.append({
        "name": "O3: SpreadStd pause",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o3_paused,
    })

    # --- O4: Combined OB pause (ob_mid_vol OR ob_spread_std) ---
    o4_paused = o2_paused | o3_paused
    pct = o4_paused.sum() / n_sim * 100
    print(f"  O4 combined OB pause: {pct:.1f}% paused")
    strategies.append({
        "name": "O4: OBmidVol|SpreadStd",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o4_paused,
    })

    # --- O5: OHLCV vol pause + OB pause (belt and suspenders) ---
    o5_paused = b1_paused | o2_paused
    pct = o5_paused.sum() / n_sim * 100
    print(f"  O5 OHLCV+OB vol pause: {pct:.1f}% paused")
    strategies.append({
        "name": "O5: OHLCV+OB VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o5_paused,
    })

    # =====================================================================
    # DEPTH RATIO REBALANCE (from walk-forward: z>1.5 at 4h = Sharpe 4.87)
    # =====================================================================

    # --- D1: Rebalance when depth ratio z > 1.5 (excess futures depth = bearish) ---
    d1_rebal = n5b_rebal.copy()
    depth_triggers = 0
    for i in range(n_sim):
        if not np.isnan(basis_z[i]) and abs(basis_z[i]) > 1.5:
            d1_rebal[i] = min(d1_rebal[i], 6)  # 30-min rebalance
            depth_triggers += 1
    print(f"  D1 depth ratio rebalance triggers: {depth_triggers} ({depth_triggers/n_sim*100:.1f}%)")
    strategies.append({
        "name": "D1: DepthR+InfR+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": d1_rebal,
        "paused": b1_paused.copy(),
    })

    # --- D2: Depth ratio rebalance + OB vol pause ---
    strategies.append({
        "name": "D2: DepthR+InfR+OBVolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": d1_rebal.copy(),
        "paused": o2_paused.copy(),
    })

    # --- D3: Depth ratio rebalance + combined pause + 1h base rebalance ---
    d3_rebal = np.full(n_sim, 12, dtype=int)  # 1h base
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            d3_rebal[i] = 1
        elif not np.isnan(basis_z[i]) and abs(basis_z[i]) > 1.5:
            d3_rebal[i] = min(d3_rebal[i], 6)
    strategies.append({
        "name": "D3: 1hR+DepthR+Inf+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": d3_rebal,
        "paused": b1_paused.copy(),
    })

    # =====================================================================
    # OB-ADAPTIVE SPACING
    # =====================================================================

    # --- A1: OB vol-adaptive spacing (tighter when OB is calm) ---
    a1_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(ob_mid_vol[i]):
            if ob_mid_vol[i] < ob_mid_vol_median * 0.5:
                a1_spacings[i] = 0.0060  # very calm
            elif ob_mid_vol[i] < ob_mid_vol_median * 0.8:
                a1_spacings[i] = 0.0080  # calm
            elif ob_mid_vol[i] > ob_mid_vol_median * 2.0:
                a1_spacings[i] = 0.0150  # widen in volatile
    strategies.append({
        "name": "A1: OBAdaptSpc+VolP+InfR",
        "spacings": a1_spacings,
        "rebalance": n5b_rebal.copy(),
        "paused": b1_paused.copy(),
    })

    # --- A2: OB vol-adaptive spacing + OB pause ---
    strategies.append({
        "name": "A2: OBAdaptSpc+OBVolP",
        "spacings": a1_spacings.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": o2_paused.copy(),
    })

    # --- A3: OB vol-adaptive spacing + depth rebalance + OB pause ---
    strategies.append({
        "name": "A3: OBAdapt+DepthR+OBP",
        "spacings": a1_spacings.copy(),
        "rebalance": d1_rebal.copy(),
        "paused": o2_paused.copy(),
    })

    # =====================================================================
    # BEST COMBOS with 1h rebalance (v17 showed shorter rebal helps)
    # =====================================================================

    # --- X1: 1h rebal + OB vol pause + informed ---
    x1_rebal = np.full(n_sim, 12, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            x1_rebal[i] = 1
    strategies.append({
        "name": "X1: 1hR+OBVolP+Inf",
        "spacings": fixed_spacing.copy(),
        "rebalance": x1_rebal,
        "paused": o2_paused.copy(),
    })

    # --- X2: 1h rebal + OB adaptive spacing + OB vol pause ---
    strategies.append({
        "name": "X2: 1hR+OBAdapt+OBVolP",
        "spacings": a1_spacings.copy(),
        "rebalance": x1_rebal.copy(),
        "paused": o2_paused.copy(),
    })

    # --- X3: 1h rebal + OB adaptive + depth rebal + OB pause (kitchen sink) ---
    x3_rebal = np.full(n_sim, 12, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            x3_rebal[i] = 1
        elif not np.isnan(basis_z[i]) and abs(basis_z[i]) > 1.5:
            x3_rebal[i] = min(x3_rebal[i], 6)
    strategies.append({
        "name": "X3: 1hR+OBAll",
        "spacings": a1_spacings.copy(),
        "rebalance": x3_rebal,
        "paused": o2_paused.copy(),
    })

    # --- X4: 30min rebal + OB vol pause ---
    rebal_30m = np.full(n_sim, 6, dtype=int)
    strategies.append({
        "name": "X4: 30mR+OBVolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_30m.copy(),
        "paused": o2_paused.copy(),
    })

    # --- X5: 30min rebal + OB adaptive + OB vol pause ---
    strategies.append({
        "name": "X5: 30mR+OBAdapt+OBVolP",
        "spacings": a1_spacings.copy(),
        "rebalance": rebal_30m.copy(),
        "paused": o2_paused.copy(),
    })

    # =====================================================================
    # Run all strategies
    # =====================================================================
    print(f"\n  Running {len(strategies)} strategies...\n")
    results = []

    for strat in strategies:
        t_s = time.time()
        r = bot.run(
            prices_close, prices_high, prices_low,
            strat["spacings"],
            strategy_name=strat["name"],
            paused=strat.get("paused"),
            rebalance_intervals=strat["rebalance"],
            close_on_pause=True,
        )
        results.append(r)
        elapsed = time.time() - t_s

        print(f"  {r['strategy']}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} (grid: ${r['grid_profits']:+.2f}, fees: -${r['total_fees']:.2f})")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills: {r['fills']:,} ({r['fills_per_day']:.1f}/d) | "
              f"Sharpe: {r['sharpe']:.2f} | MaxDD: ${r['max_drawdown']:.2f} ({elapsed:.1f}s)\n")

    # --- Summary table ---
    baseline = results[0]
    print(f"  {'='*120}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days)")
    print(f"  {'='*120}")
    print(f"  {'Strategy':<30s} {'PnL':>12s} {'Grid$':>12s} {'Fees':>8s} {'PnL/d':>9s} "
          f"{'Fills':>7s} {'Sharpe':>7s} {'MaxDD':>12s} {'vs Base':>10s}")
    print(f"  {'-'*120}")
    for r in results:
        delta = r['total_pnl'] - baseline['total_pnl']
        marker = "✅" if delta > 0 and r is not baseline else "  "
        if r is baseline:
            print(f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+11.2f} ${r['grid_profits']:>+11.2f} "
                  f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+8.2f} "
                  f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>11.2f} "
                  f"{'':>10s}")
        else:
            print(f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+11.2f} ${r['grid_profits']:>+11.2f} "
                  f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+8.2f} "
                  f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>11.2f} "
                  f"${delta:>+9.0f}")

    # Best strategies
    sorted_results = sorted(results, key=lambda r: r['total_pnl'], reverse=True)
    print(f"\n  Top 5 strategies:")
    for i, r in enumerate(sorted_results[:5], 1):
        delta = r['total_pnl'] - baseline['total_pnl']
        print(f"    {i}. {r['strategy']}: ${r['total_pnl']:+.2f} (Sharpe: {r['sharpe']:.2f}, vs base: ${delta:+.0f})")

    positive = [r for r in results if r['total_pnl'] > 0]
    if positive:
        print(f"\n  🎉 {len(positive)} strategies achieved POSITIVE PnL!")
    else:
        print(f"\n  ⚠️ No strategy achieved positive PnL yet.")

    # OB vs non-OB comparison
    ob_strats = [r for r in results if any(x in r['strategy'] for x in ['OB', 'Depth', 'Spread', 'Adapt'])]
    non_ob_strats = [r for r in results if r not in ob_strats and r is not baseline]
    if ob_strats and non_ob_strats:
        best_ob = max(ob_strats, key=lambda r: r['total_pnl'])
        best_non_ob = max(non_ob_strats, key=lambda r: r['total_pnl'])
        print(f"\n  Best OB strategy:     {best_ob['strategy']}: ${best_ob['total_pnl']:+.2f}")
        print(f"  Best non-OB strategy: {best_non_ob['strategy']}: ${best_non_ob['total_pnl']:+.2f}")
        print(f"  OB advantage: ${best_ob['total_pnl'] - best_non_ob['total_pnl']:+.2f}")

    elapsed_total = time.time() - t_total
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-12-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    run_v18(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
