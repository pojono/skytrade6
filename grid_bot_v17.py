#!/usr/bin/env python3
"""
grid_bot_v17.py — Regime-Filtered Grid Bot

Key insight from v8/v9: Markets are ranging 85-94% of the time.
Grid bots profit in ranging markets but bleed in trends.
The rare trending bars (1-2.5%) cause nearly ALL the losses.

Problem: Trend detection FAILS (F1=0.05) — can't predict trend onset.
Solution: Use what WORKS:
  1. Vol detection (F1=0.35-0.43) — high vol contains dangerous trends
  2. Backward-looking efficiency ratio — continuous trending indicator
  3. N5b informed rebalance (v16 winner) — cut inventory when informed flow spikes

Strategy: PAUSE the grid during high-vol and/or high-efficiency periods.
Only run when conditions favor mean-reversion (low vol + low efficiency).

Baseline: Fix 1.00% (24h) from v15.
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

MAKER_FEE_BPS = 2.0

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
# Extended 5m bar aggregation (reuse from v16 if cache exists, else basic)
# ---------------------------------------------------------------------------

def _aggregate_5m_with_novel(trades):
    """Aggregate tick data into 5-minute bars with novel features for informed flow."""
    bucket = (trades["timestamp_us"].values // INTERVAL_5M_US) * INTERVAL_5M_US
    trades = trades.copy()
    trades["bucket"] = bucket

    bars = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        n = len(grp)
        if n < 5:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        ret = (close_p - open_p) / max(open_p, 1e-10)

        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)

        price_changes = np.abs(np.diff(p))
        total_path = price_changes.sum()
        net_move = abs(p[-1] - p[0])
        bar_efficiency = net_move / max(total_path, 1e-10)

        if n > 10:
            iti = np.diff(grp["timestamp_us"].values).astype(np.float64)
            iti_mean = iti.mean()
            iti_cv = iti.std() / max(iti_mean, 1) if iti_mean > 0 else 0
        else:
            iti_cv = 0.0

        if n > 5:
            sign_changes = np.sum(np.diff(s) != 0)
            sign_persistence = 1.0 - sign_changes / max(n - 1, 1)
        else:
            sign_persistence = 0.5

        if n > 20:
            q90 = np.percentile(q, 90)
            large_frac = q[q >= q90].sum() / max(total_vol, 1e-10)
        else:
            large_frac = 0.0

        parkinson_vol = np.sqrt(np.log(high_p / max(low_p, 1e-10))**2 / (4 * np.log(2))) if low_p > 0 else 0

        vwap = qq.sum() / max(total_vol, 1e-10)
        vwap_dev = (close_p - vwap) / max(vwap, 1e-10)

        # --- Novel features for informed flow composite ---
        # VPIN
        cum_vol = np.cumsum(q)
        vol_per_bucket = total_vol / 5
        vpin_imbalances = []
        for vb in range(5):
            lo = vb * vol_per_bucket
            hi = (vb + 1) * vol_per_bucket
            mask = (cum_vol > lo) & (cum_vol <= hi)
            if mask.sum() > 0:
                bv = q[mask & buy_mask].sum()
                sv = q[mask & sell_mask].sum()
                vpin_imbalances.append(abs(bv - sv) / max(bv + sv, 1e-10))
        vpin = np.mean(vpin_imbalances) if vpin_imbalances else 0.0

        # Aggressive ratio
        price_changed = np.diff(p) != 0
        if len(price_changed) > 0:
            aggressive_vol = q[1:][price_changed].sum()
            passive_vol = q[1:][~price_changed].sum()
            aggressive_ratio = aggressive_vol / max(aggressive_vol + passive_vol, 1e-10)
        else:
            aggressive_ratio = 0.5

        # Herding runs
        if n > 2:
            sign_changes_arr = np.diff(s) != 0
            runs = np.split(np.arange(len(s) - 1), np.where(sign_changes_arr)[0] + 1)
            run_lengths = [len(r) for r in runs if len(r) > 0]
            avg_run_length = np.mean(run_lengths) if run_lengths else 1
        else:
            avg_run_length = 1

        bars.append({
            "timestamp_us": bkt,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "trade_count": n,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "returns": ret,
            "vol_imbalance": vol_imbalance,
            "bar_efficiency": bar_efficiency,
            "iti_cv": iti_cv,
            "sign_persistence": sign_persistence,
            "large_trade_frac": large_frac,
            "parkinson_vol": parkinson_vol,
            "vwap_dev": vwap_dev,
            # Novel (for informed flow composite)
            "vpin": vpin,
            "aggressive_ratio": aggressive_ratio,
            "avg_run_length": avg_run_length,
        })

    return pd.DataFrame(bars)


def load_bars_v17(symbol, start_date, end_date):
    """Load tick data, aggregate to 5m bars. Tries v16 novel cache first, else builds."""
    import psutil

    # Try v16 cache first (has all novel features)
    novel_cache = PARQUET_DIR / symbol / "novel_5m_cache" / SOURCE
    regime_cache = PARQUET_DIR / symbol / "regime_5m_cache" / SOURCE
    build_cache = PARQUET_DIR / symbol / "v17_5m_cache" / SOURCE
    build_cache.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date)
    all_bars = []
    t0 = time.time()
    sources = {"novel": 0, "regime": 0, "built": 0}

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")

        # Priority: novel_cache (has all features) > v17 cache > build from ticks
        loaded = False
        for cache_dir, src_name in [(novel_cache, "novel"), (build_cache, "built")]:
            cache_path = cache_dir / f"{ds}.parquet"
            if cache_path.exists():
                bars = pd.read_parquet(cache_path)
                all_bars.append(bars)
                sources[src_name] += 1
                loaded = True
                break

        if not loaded:
            # Check if regime cache exists (missing novel features, but has basics)
            regime_path = regime_cache / f"{ds}.parquet"
            if regime_path.exists():
                bars = pd.read_parquet(regime_path)
                all_bars.append(bars)
                sources["regime"] += 1
                loaded = True

        if not loaded:
            tick_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
            if not tick_path.exists():
                continue
            trades = pd.read_parquet(tick_path)
            bars = _aggregate_5m_with_novel(trades)
            del trades
            if not bars.empty:
                bars.to_parquet(build_cache / f"{ds}.parquet", index=False, compression="snappy")
                all_bars.append(bars)
            sources["built"] += 1

        if i % 20 == 0 or i == len(dates):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(dates) - i) / max(rate, 0.01)
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  [{i}/{len(dates)}] {ds} | {elapsed:.0f}s ETA={eta:.0f}s "
                  f"RAM={mem:.1f}GB novel={sources['novel']} regime={sources['regime']} built={sources['built']}", flush=True)

    if not all_bars:
        return pd.DataFrame()

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    print(f"  Loaded {len(df):,} bars ({len(dates)} days, "
          f"novel={sources['novel']} regime={sources['regime']} built={sources['built']})")
    return df


def compute_informed_composite(df):
    """Compute the informed flow composite z-score (N5b from v16)."""
    # Need: vpin, sign_persistence, aggressive_ratio, avg_run_length
    required = ["vpin", "sign_persistence", "aggressive_ratio", "avg_run_length"]
    available = [c for c in required if c in df.columns]

    if len(available) < 2:
        print(f"  WARNING: Only {len(available)} informed flow features available, skipping composite")
        df["informed_composite_z"] = 0.0
        return df

    for col in available:
        df[f"_r_{col}"] = df[col].rolling(288, min_periods=60).rank(pct=True)

    rank_cols = [f"_r_{col}" for col in available]
    composite = df[rank_cols].mean(axis=1)
    mean = composite.rolling(288, min_periods=60).mean()
    std = composite.rolling(288, min_periods=60).std().clip(lower=1e-10)
    df["informed_composite_z"] = (composite - mean) / std
    df.drop(columns=rank_cols, inplace=True)

    return df


# ---------------------------------------------------------------------------
# GridBotSimulator — ported from v15 (correct implementation)
# Key features: level deactivation, proper position sizing, paired buy/sell
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

            # Rebalance check
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

            # Pause handling
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

            # Fill logic with level deactivation
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

        # Close remaining inventory at end
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
# Walk-forward Ridge (same as v15/v16)
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
# Main experiment
# ---------------------------------------------------------------------------

def run_v17(symbol, start_date, end_date):
    t_total = time.time()
    print("=" * 70)
    print(f"  v17: Regime-Filtered Grid Bot — {symbol}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Goal: Only run grid in ranging/low-vol regimes")
    print("=" * 70)

    # --- Load data ---
    print(f"\n  Loading data...", flush=True)
    df = load_bars_v17(symbol, start_date, end_date)
    print(f"  {len(df):,} bars loaded in {time.time()-t_total:.0f}s")

    print(f"  Computing regime features...", flush=True)
    t1 = time.time()
    df = compute_regime_features(df)
    print(f"  Regime features in {time.time()-t1:.0f}s")

    print(f"  Computing informed flow composite...", flush=True)
    t1 = time.time()
    df = compute_informed_composite(df)
    print(f"  Informed composite in {time.time()-t1:.0f}s")

    # --- Forward targets ---
    print(f"  Computing forward targets...", flush=True)
    ret = df["returns"].values
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    fwd_vol_1h = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol_1h[i] = np.std(ret[i+1:i+13])
    df["fwd_vol_1h"] = fwd_vol_1h

    # --- Walk-forward vol prediction ---
    warmup = 2500

    print(f"\n  Walk-forward 1h vol prediction (Ridge)...", flush=True)
    t1 = time.time()
    pred_vol_1h = walkforward_ridge(df, VOL_FEATURES, "fwd_vol_1h", min_train=2000)
    valid = np.sum(~np.isnan(pred_vol_1h[warmup:]))
    print(f"    {valid:,} valid predictions in {time.time()-t1:.0f}s")

    # --- Prepare simulation data ---
    df_sim = df.iloc[warmup:].copy().reset_index(drop=True)
    n_sim = len(df_sim)
    prices_close = df_sim["close"].values.astype(float)
    prices_high = df_sim["high"].values.astype(float)
    prices_low = df_sim["low"].values.astype(float)

    pv1h = pred_vol_1h[warmup:]
    median_pred_vol = np.nanmedian(pv1h[~np.isnan(pv1h)])

    # Backward-looking efficiency (4h) — continuous trend indicator
    eff_4h = df_sim["efficiency_4h"].values if "efficiency_4h" in df_sim.columns else np.full(n_sim, 0.3)
    eff_8h = df_sim["efficiency_8h"].values if "efficiency_8h" in df_sim.columns else np.full(n_sim, 0.3)

    # Informed flow composite
    informed_z = df_sim["informed_composite_z"].values if "informed_composite_z" in df_sim.columns else np.zeros(n_sim)

    # Parkinson vol (backward-looking, no ML needed)
    parkvol_1h = df_sim["parkvol_1h"].values if "parkvol_1h" in df_sim.columns else np.full(n_sim, np.nan)
    parkvol_median = np.nanmedian(parkvol_1h[~np.isnan(parkvol_1h)])

    # ADX (backward-looking trend strength)
    adx_4h = df_sim["adx_4h"].values if "adx_4h" in df_sim.columns else np.full(n_sim, 0.2)

    print(f"\n  Simulation: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price: ${prices_close.min():.2f} - ${prices_close.max():.2f}")
    print(f"  Median pred vol: {median_pred_vol:.6f}")
    print(f"  Median parkvol_1h: {parkvol_median:.6f}")
    print(f"  Median eff_4h: {np.nanmedian(eff_4h):.4f}")
    print(f"  Median adx_4h: {np.nanmedian(adx_4h):.4f}")

    bot = GridBotSimulator(n_levels=5, fee_bps=MAKER_FEE_BPS)
    strategies = []

    fixed_spacing = np.full(n_sim, 0.0100)
    fixed_rebal = np.full(n_sim, 288, dtype=int)

    # --- S0: Baseline Fix 1.00% (24h) ---
    strategies.append({
        "name": "S0: Fix 1.00% (24h)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # --- Ref: N5b from v16 (informed rebalance only) ---
    n5b_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            n5b_rebal[i] = 12
    strategies.append({
        "name": "Ref: N5b InfRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": None,
    })

    # =====================================================================
    # REGIME FILTER STRATEGIES
    # =====================================================================

    # --- R1: Pause when predicted vol > 2× median ---
    # High vol = dangerous for grids. 22% of bars are high-vol.
    r1_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]) and pv1h[i] > median_pred_vol * 2.0:
            r1_paused[i] = True
    pct = r1_paused.sum() / n_sim * 100
    print(f"\n  R1 vol pause (>2x): {pct:.1f}% paused")
    strategies.append({
        "name": "R1: Pause highVol(2x)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r1_paused,
    })

    # --- R2: Pause when predicted vol > 1.5× median ---
    r2_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]) and pv1h[i] > median_pred_vol * 1.5:
            r2_paused[i] = True
    pct = r2_paused.sum() / n_sim * 100
    print(f"  R2 vol pause (>1.5x): {pct:.1f}% paused")
    strategies.append({
        "name": "R2: Pause highVol(1.5x)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r2_paused,
    })

    # --- R3: Pause when efficiency_4h > 0.35 (trending) ---
    # Efficiency > 0.4 = trending (v8 definition), use 0.35 for earlier detection
    r3_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(eff_4h[i]) and eff_4h[i] > 0.35:
            r3_paused[i] = True
    pct = r3_paused.sum() / n_sim * 100
    print(f"  R3 eff pause (>0.35): {pct:.1f}% paused")
    strategies.append({
        "name": "R3: Pause trend(eff>0.35)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r3_paused,
    })

    # --- R4: Pause when efficiency_4h > 0.30 ---
    r4_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(eff_4h[i]) and eff_4h[i] > 0.30:
            r4_paused[i] = True
    pct = r4_paused.sum() / n_sim * 100
    print(f"  R4 eff pause (>0.30): {pct:.1f}% paused")
    strategies.append({
        "name": "R4: Pause trend(eff>0.30)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r4_paused,
    })

    # --- R5: Pause when ADX > 0.30 (strong trend) ---
    r5_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(adx_4h[i]) and adx_4h[i] > 0.30:
            r5_paused[i] = True
    pct = r5_paused.sum() / n_sim * 100
    print(f"  R5 adx pause (>0.30): {pct:.1f}% paused")
    strategies.append({
        "name": "R5: Pause trend(adx>0.30)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r5_paused,
    })

    # --- R6: Pause when parkvol_1h > 1.5× median (no ML, pure backward-looking) ---
    r6_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(parkvol_1h[i]) and parkvol_1h[i] > parkvol_median * 1.5:
            r6_paused[i] = True
    pct = r6_paused.sum() / n_sim * 100
    print(f"  R6 parkvol pause (>1.5x): {pct:.1f}% paused")
    strategies.append({
        "name": "R6: Pause parkvol(1.5x)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r6_paused,
    })

    # --- R7: Combined: pause when highVol OR trending ---
    r7_paused = r2_paused | r3_paused
    pct = r7_paused.sum() / n_sim * 100
    print(f"  R7 vol+eff pause: {pct:.1f}% paused")
    strategies.append({
        "name": "R7: Pause vol|eff",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r7_paused,
    })

    # --- R8: Combined: pause when highVol OR trending OR ADX ---
    r8_paused = r2_paused | r3_paused | r5_paused
    pct = r8_paused.sum() / n_sim * 100
    print(f"  R8 vol+eff+adx pause: {pct:.1f}% paused")
    strategies.append({
        "name": "R8: Pause vol|eff|adx",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": r8_paused,
    })

    # =====================================================================
    # COMBOS: Best regime filter + N5b informed rebalance
    # =====================================================================

    # --- C1: R2 (vol pause) + N5b (informed rebalance) ---
    strategies.append({
        "name": "C1: VolPause+InfRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r2_paused.copy(),
    })

    # --- C2: R3 (eff pause) + N5b ---
    strategies.append({
        "name": "C2: EffPause+InfRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r3_paused.copy(),
    })

    # --- C3: R7 (vol|eff pause) + N5b ---
    strategies.append({
        "name": "C3: Vol|Eff+InfRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r7_paused.copy(),
    })

    # --- C4: R6 (parkvol pause, no ML) + N5b ---
    strategies.append({
        "name": "C4: Parkvol+InfRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r6_paused.copy(),
    })

    # =====================================================================
    # ROUND 2: Tighter grids + wider pauses + shorter rebalance
    # C1 was best at -$4K. Grid profits are negative → need more fills
    # or better fill quality (tighter grid in calm markets).
    # =====================================================================

    # --- T1: Tighter grid (0.70%) + vol pause + informed rebalance ---
    tight_spacing = np.full(n_sim, 0.0070)
    strategies.append({
        "name": "T1: 0.70%+VolP+InfR",
        "spacings": tight_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r2_paused.copy(),
    })

    # --- T2: Tighter grid (0.50%) + vol pause + informed rebalance ---
    tighter_spacing = np.full(n_sim, 0.0050)
    strategies.append({
        "name": "T2: 0.50%+VolP+InfR",
        "spacings": tighter_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r2_paused.copy(),
    })

    # --- T3: Tighter grid (0.70%) + vol|eff pause + informed rebalance ---
    strategies.append({
        "name": "T3: 0.70%+V|E+InfR",
        "spacings": tight_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r7_paused.copy(),
    })

    # --- T4: Tighter grid (0.50%) + vol|eff pause + informed rebalance ---
    strategies.append({
        "name": "T4: 0.50%+V|E+InfR",
        "spacings": tighter_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r7_paused.copy(),
    })

    # --- T5: Wider pause (vol>1.2x) + tight grid + informed rebalance ---
    r_wide_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]) and pv1h[i] > median_pred_vol * 1.2:
            r_wide_paused[i] = True
    pct = r_wide_paused.sum() / n_sim * 100
    print(f"  T5 wide vol pause (>1.2x): {pct:.1f}% paused")
    strategies.append({
        "name": "T5: 0.70%+WideVP+InfR",
        "spacings": tight_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r_wide_paused.copy(),
    })

    # --- T6: Very wide pause (vol>1.2x | eff>0.25) + tight grid + inf rebal ---
    r_vwide_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        v_pause = not np.isnan(pv1h[i]) and pv1h[i] > median_pred_vol * 1.2
        e_pause = not np.isnan(eff_4h[i]) and eff_4h[i] > 0.25
        if v_pause or e_pause:
            r_vwide_paused[i] = True
    pct = r_vwide_paused.sum() / n_sim * 100
    print(f"  T6 vwide pause (vol1.2|eff0.25): {pct:.1f}% paused")
    strategies.append({
        "name": "T6: 0.70%+VWideP+InfR",
        "spacings": tight_spacing.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r_vwide_paused.copy(),
    })

    # --- T7: Shorter rebalance (4h) + vol pause + tight grid ---
    short_rebal = np.full(n_sim, 48, dtype=int)  # 4h instead of 24h
    # Override with informed rebalance when triggered
    t7_rebal = short_rebal.copy()
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            t7_rebal[i] = 12
    strategies.append({
        "name": "T7: 0.70%+4hR+VolP+Inf",
        "spacings": tight_spacing.copy(),
        "rebalance": t7_rebal,
        "paused": r2_paused.copy(),
    })

    # --- T8: 8h rebalance + vol pause + tight grid + informed ---
    t8_rebal = np.full(n_sim, 96, dtype=int)  # 8h
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            t8_rebal[i] = 12
    strategies.append({
        "name": "T8: 0.70%+8hR+VolP+Inf",
        "spacings": tight_spacing.copy(),
        "rebalance": t8_rebal,
        "paused": r2_paused.copy(),
    })

    # =====================================================================
    # ROUND 3: Different approach — short rebalance + more levels
    # The remaining loss is from inventory drift. Solution: hold less time.
    # Also try more grid levels (more fills per bar).
    # =====================================================================

    # --- U1: 1% + 2h rebalance + vol pause (no informed, just fast rebal) ---
    rebal_2h = np.full(n_sim, 24, dtype=int)
    strategies.append({
        "name": "U1: 1%+2hR+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_2h.copy(),
        "paused": r2_paused.copy(),
    })

    # --- U2: 1% + 1h rebalance + vol pause ---
    rebal_1h = np.full(n_sim, 12, dtype=int)
    strategies.append({
        "name": "U2: 1%+1hR+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_1h.copy(),
        "paused": r2_paused.copy(),
    })

    # --- U3: 1% + 2h rebalance + vol|eff pause ---
    strategies.append({
        "name": "U3: 1%+2hR+V|EP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_2h.copy(),
        "paused": r7_paused.copy(),
    })

    # --- U4: 1% + 2h rebalance + vol pause + informed override ---
    u4_rebal = np.full(n_sim, 24, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            u4_rebal[i] = 1  # immediate
    strategies.append({
        "name": "U4: 1%+2hR+VolP+Inf",
        "spacings": fixed_spacing.copy(),
        "rebalance": u4_rebal,
        "paused": r2_paused.copy(),
    })

    # --- U5: 10 levels (more fills) + 1% + vol pause + informed rebal ---
    bot10 = GridBotSimulator(n_levels=10, fee_bps=MAKER_FEE_BPS)

    # --- U6: 1% + vol-adaptive spacing in calm (tighter when very calm) ---
    u6_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]):
            if pv1h[i] < median_pred_vol * 0.7:
                u6_spacings[i] = 0.0060  # very calm → tight grid
            elif pv1h[i] < median_pred_vol * 1.0:
                u6_spacings[i] = 0.0080  # calm → slightly tight
    strategies.append({
        "name": "U6: VolAdapt+VolP+InfR",
        "spacings": u6_spacings.copy(),
        "rebalance": n5b_rebal.copy(),
        "paused": r2_paused.copy(),
    })

    # --- U7: Vol-adaptive spacing + 2h rebalance + vol pause ---
    strategies.append({
        "name": "U7: VolAdapt+2hR+VolP",
        "spacings": u6_spacings.copy(),
        "rebalance": rebal_2h.copy(),
        "paused": r2_paused.copy(),
    })

    # =====================================================================
    # ROUND 4: Ultra-short rebalance — minimize inventory holding time
    # U2 (1h rebal) was best at -$1.5K. Push shorter.
    # =====================================================================

    # --- V1: 1% + 30min rebalance + vol pause ---
    rebal_30m = np.full(n_sim, 6, dtype=int)
    strategies.append({
        "name": "V1: 1%+30mR+VolP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_30m.copy(),
        "paused": r2_paused.copy(),
    })

    # --- V2: 1% + 1h rebalance + vol|eff pause ---
    strategies.append({
        "name": "V2: 1%+1hR+V|EP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_1h.copy(),
        "paused": r7_paused.copy(),
    })

    # --- V3: 1% + 30min rebalance + vol|eff pause ---
    strategies.append({
        "name": "V3: 1%+30mR+V|EP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_30m.copy(),
        "paused": r7_paused.copy(),
    })

    # --- V4: 1% + 1h rebalance + vol pause + informed immediate ---
    v4_rebal = np.full(n_sim, 12, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            v4_rebal[i] = 1
    strategies.append({
        "name": "V4: 1%+1hR+VolP+Inf",
        "spacings": fixed_spacing.copy(),
        "rebalance": v4_rebal,
        "paused": r2_paused.copy(),
    })

    # --- V5: 1% + 30min rebalance + wider vol pause (1.2x) ---
    strategies.append({
        "name": "V5: 1%+30mR+WideVP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_30m.copy(),
        "paused": r_wide_paused.copy(),
    })

    # --- V6: 1% + 1h rebalance + wider vol pause (1.2x) ---
    strategies.append({
        "name": "V6: 1%+1hR+WideVP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_1h.copy(),
        "paused": r_wide_paused.copy(),
    })

    # --- V7: 1% + 1h rebalance + very wide pause (vol1.2|eff0.25) ---
    strategies.append({
        "name": "V7: 1%+1hR+VWideP",
        "spacings": fixed_spacing.copy(),
        "rebalance": rebal_1h.copy(),
        "paused": r_vwide_paused.copy(),
    })

    # --- Run all strategies ---
    print(f"\n  Running {len(strategies)} strategies + U5 (10-level)...\n")
    results = []

    for strat in strategies:
        t_s = time.time()
        r = bot.run(
            prices_close, prices_high, prices_low,
            strat["spacings"],
            strategy_name=strat["name"],
            paused=strat.get("paused"),
            rebalance_intervals=strat["rebalance"],
            close_on_pause=True,  # regime filter: always close inventory on pause
        )
        results.append(r)
        elapsed = time.time() - t_s

        print(f"  {r['strategy']}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} (grid: ${r['grid_profits']:+.2f}, fees: -${r['total_fees']:.2f})")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills: {r['fills']:,} ({r['fills_per_day']:.1f}/d) | "
              f"Sharpe: {r['sharpe']:.2f} | MaxDD: ${r['max_drawdown']:.2f} ({elapsed:.1f}s)\n")

    # U5: 10 grid levels (separate bot instance)
    t_s = time.time()
    r = bot10.run(
        prices_close, prices_high, prices_low,
        fixed_spacing.copy(),
        strategy_name="U5: 10lvl+1%+VolP+InfR",
        paused=r2_paused.copy(),
        rebalance_intervals=n5b_rebal.copy(),
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
    print(f"  {'='*110}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days)")
    print(f"  {'='*110}")
    print(f"  {'Strategy':<30s} {'PnL':>12s} {'Grid$':>12s} {'Fees':>8s} {'PnL/d':>9s} "
          f"{'Fills':>7s} {'Sharpe':>7s} {'MaxDD':>12s} {'vs Base':>10s}")
    print(f"  {'-'*113}")
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

    # Check for positive PnL
    positive = [r for r in results if r['total_pnl'] > 0]
    if positive:
        print(f"\n  🎉 {len(positive)} strategies achieved POSITIVE PnL!")
        for r in positive:
            print(f"    {r['strategy']}: ${r['total_pnl']:+.2f} (Sharpe: {r['sharpe']:.2f})")
    else:
        print(f"\n  ⚠️ No strategy achieved positive PnL yet.")

    elapsed_total = time.time() - t_total
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()
    run_v17(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
