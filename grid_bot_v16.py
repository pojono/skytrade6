#!/usr/bin/env python3
"""
grid_bot_v16.py — Novel Microstructure Signals for Grid Bot Enhancement

Key insight from v6: Novel signals (VPIN, toxic flow, herding, efficiency)
were tested as DIRECTIONAL strategies and mostly failed (Sharpe < 0.10).
But they contain rich information about market MICROSTRUCTURE STATE.

Creative pivot: Use them not for direction, but for GRID BOT REGIME CONTROL:
  - Toxic flow  → PAUSE grid (informed traders will pick off your limits)
  - Herding runs → WIDEN grid (momentum building, avoid getting run over)
  - Efficiency   → TIGHTEN when low (choppy = grid paradise)
  - VPIN         → WIDEN when high (informed trading = adverse selection)
  - Composite    → SMART REBALANCE (rebalance when informed flow spikes)

Baseline: Fix 1.00% (24h) from v15.
Also includes S5 (adaptive rebalance) as reference since it won on BTC/ETH.

Data: Extends 5m bar aggregation with novel tick-level features.
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
# Extended 5m bar aggregation with novel microstructure features
# ---------------------------------------------------------------------------

def _aggregate_5m_extended(trades):
    """Aggregate tick data into 5-minute bars with EXTENDED microstructure features.
    Adds: VPIN, toxic_flow, herding (run_imbalance, avg_run_length),
          aggressive_ratio, size_entropy, efficiency, multifractal_ratio.
    """
    bucket = (trades["timestamp_us"].values // INTERVAL_5M_US) * INTERVAL_5M_US
    trades = trades.copy()
    trades["bucket"] = bucket

    bars = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        t = grp["timestamp_us"].values
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

        # --- Standard features (same as regime_detection._aggregate_5m) ---
        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)

        price_changes = np.abs(np.diff(p))
        total_path = price_changes.sum()
        net_move = abs(p[-1] - p[0])
        bar_efficiency = net_move / max(total_path, 1e-10)

        if n > 10:
            iti = np.diff(t).astype(np.float64)
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

        # --- NOVEL FEATURE: VPIN ---
        # Volume-Synchronized Probability of Informed Trading
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

        # --- NOVEL FEATURE: Toxic Flow ---
        # Price moves against majority of volume = informed traders on other side
        toxic_flow = -vol_imbalance * np.sign(ret) if abs(ret) > 1e-10 else 0.0

        # --- NOVEL FEATURE: Herding (Signed Volume Runs) ---
        if n > 2:
            sign_changes_arr = np.diff(s) != 0
            runs = np.split(np.arange(len(s) - 1), np.where(sign_changes_arr)[0] + 1)
            run_lengths = [len(r) for r in runs if len(r) > 0]
            avg_run_length = np.mean(run_lengths) if run_lengths else 1
            max_run_length = max(run_lengths) if run_lengths else 1
            buy_runs = []
            sell_runs = []
            for r in runs:
                if len(r) > 0 and s[r[0]] == 1:
                    buy_runs.append(len(r))
                elif len(r) > 0:
                    sell_runs.append(len(r))
            avg_buy_run = np.mean(buy_runs) if buy_runs else 0
            avg_sell_run = np.mean(sell_runs) if sell_runs else 0
            run_imbalance = (avg_buy_run - avg_sell_run) / max(avg_buy_run + avg_sell_run, 1e-10)
        else:
            avg_run_length = 1; max_run_length = 1; run_imbalance = 0

        # --- NOVEL FEATURE: Aggressive Volume Ratio ---
        # Trades that moved the price vs trades at same price
        price_changed = np.diff(p) != 0
        if len(price_changed) > 0:
            aggressive_vol = q[1:][price_changed].sum()
            passive_vol = q[1:][~price_changed].sum()
            aggressive_ratio = aggressive_vol / max(aggressive_vol + passive_vol, 1e-10)
        else:
            aggressive_ratio = 0.5

        # --- NOVEL FEATURE: Size Entropy ---
        # Low entropy = concentrated (informed), high = dispersed (noise)
        q_nonzero = q[q > 0]
        if len(q_nonzero) > 1:
            try:
                hist, _ = np.histogram(q_nonzero, bins=min(10, len(q_nonzero)))
                hist = hist[hist > 0]
                probs = hist / hist.sum()
                size_entropy = -np.sum(probs * np.log2(probs))
            except Exception:
                size_entropy = 0.0
        else:
            size_entropy = 0.0

        # --- NOVEL FEATURE: Multifractal Ratio ---
        # vol(small scale) / vol(large scale) — clustered vol detection
        if n > 20:
            ret_1 = np.diff(p) / np.maximum(p[:-1], 1e-10)
            ret_10 = (p[10:] - p[:-10]) / np.maximum(p[:-10], 1e-10) if n > 20 else ret_1
            vol_1 = np.std(ret_1) if len(ret_1) > 1 else 0
            vol_10 = np.std(ret_10) / max(10**0.5, 1) if len(ret_10) > 1 else 0
            multifractal_ratio = vol_1 / max(vol_10, 1e-10)
        else:
            multifractal_ratio = 1.0

        bars.append({
            "timestamp_us": bkt,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "trade_count": n,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "returns": ret,
            # Standard microstructure
            "vol_imbalance": vol_imbalance,
            "bar_efficiency": bar_efficiency,
            "iti_cv": iti_cv,
            "sign_persistence": sign_persistence,
            "large_trade_frac": large_frac,
            "parkinson_vol": parkinson_vol,
            "vwap_dev": vwap_dev,
            # Novel microstructure
            "vpin": vpin,
            "toxic_flow": toxic_flow,
            "avg_run_length": avg_run_length,
            "max_run_length": max_run_length,
            "run_imbalance": run_imbalance,
            "aggressive_ratio": aggressive_ratio,
            "size_entropy": size_entropy,
            "multifractal_ratio": multifractal_ratio,
        })

    return pd.DataFrame(bars)


def load_bars_extended(symbol, start_date, end_date):
    """Load tick data, aggregate to 5m bars with EXTENDED novel features.
    Uses separate cache from regime_detection to avoid conflicts."""
    import psutil
    cache_dir = PARQUET_DIR / symbol / "novel_5m_cache" / SOURCE
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date)
    all_bars = []
    t0 = time.time()
    processed = 0
    cached_hits = 0

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = cache_dir / f"{ds}.parquet"

        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            all_bars.append(bars)
            cached_hits += 1
        else:
            tick_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
            if not tick_path.exists():
                continue
            trades = pd.read_parquet(tick_path)
            bars = _aggregate_5m_extended(trades)
            del trades
            if not bars.empty:
                bars.to_parquet(cache_path, index=False, compression="snappy")
                all_bars.append(bars)
            processed += 1

        if i % 20 == 0 or i == len(dates):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(dates) - i) / max(rate, 0.01)
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  [{i}/{len(dates)}] {ds} | {elapsed:.0f}s ETA={eta:.0f}s "
                  f"RAM={mem:.1f}GB cache={cached_hits} new={processed}", flush=True)

    if not all_bars:
        return pd.DataFrame()

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    print(f"  Loaded {len(df):,} bars ({len(dates)} days, {cached_hits} cached, {processed} new)")
    return df


def compute_novel_rolling(df):
    """Compute rolling z-scores and derived features from novel signals."""
    # Rolling z-scores (24h = 288 bars lookback)
    for col in ["vpin", "toxic_flow", "avg_run_length", "run_imbalance",
                "aggressive_ratio", "size_entropy", "multifractal_ratio"]:
        if col in df.columns:
            mean = df[col].rolling(288, min_periods=60).mean()
            std = df[col].rolling(288, min_periods=60).std().clip(lower=1e-10)
            df[f"{col}_z"] = (df[col] - mean) / std

    # Cumulative toxic flow (1h = 12 bars)
    if "toxic_flow" in df.columns:
        df["cum_toxic_12"] = df["toxic_flow"].rolling(12).sum()
        df["cum_toxic_24"] = df["toxic_flow"].rolling(24).sum()
        mean = df["cum_toxic_12"].rolling(288, min_periods=60).mean()
        std = df["cum_toxic_12"].rolling(288, min_periods=60).std().clip(lower=1e-10)
        df["cum_toxic_z"] = (df["cum_toxic_12"] - mean) / std

    # Herding intensity (rolling avg of run lengths)
    if "avg_run_length" in df.columns:
        df["herding_1h"] = df["avg_run_length"].rolling(12).mean()
        df["herding_4h"] = df["avg_run_length"].rolling(48).mean()
        mean = df["herding_1h"].rolling(288, min_periods=60).mean()
        std = df["herding_1h"].rolling(288, min_periods=60).std().clip(lower=1e-10)
        df["herding_z"] = (df["herding_1h"] - mean) / std

    # VPIN rolling (smoothed)
    if "vpin" in df.columns:
        df["vpin_1h"] = df["vpin"].rolling(12).mean()
        df["vpin_4h"] = df["vpin"].rolling(48).mean()

    # Efficiency rolling
    if "bar_efficiency" in df.columns:
        df["eff_1h"] = df["bar_efficiency"].rolling(12).mean()
        df["eff_4h"] = df["bar_efficiency"].rolling(48).mean()
        mean = df["eff_1h"].rolling(288, min_periods=60).mean()
        std = df["eff_1h"].rolling(288, min_periods=60).std().clip(lower=1e-10)
        df["eff_z"] = (df["eff_1h"] - mean) / std

    # Composite informed flow: rank-based combination
    # (VPIN + sign_persistence + aggressive_ratio + avg_run_length)
    for col in ["vpin", "sign_persistence", "aggressive_ratio", "avg_run_length"]:
        if col in df.columns:
            df[f"_r_{col}"] = df[col].rolling(288, min_periods=60).rank(pct=True)
    rank_cols = [f"_r_{col}" for col in ["vpin", "sign_persistence", "aggressive_ratio", "avg_run_length"]
                 if f"_r_{col}" in df.columns]
    if rank_cols:
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
# Walk-forward Ridge (same as v15)
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

def run_v16(symbol, start_date, end_date):
    t_total = time.time()
    print("=" * 70)
    print(f"  v16: Novel Microstructure Signals for Grid Bot — {symbol}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Baseline: Fix 1.00% (24h rebalance)")
    print("=" * 70)

    # --- Load data with extended features ---
    print(f"\n  Loading data with novel features...", flush=True)
    df = load_bars_extended(symbol, start_date, end_date)
    print(f"  {len(df):,} bars loaded in {time.time()-t_total:.0f}s")

    print(f"  Computing regime features...", flush=True)
    t1 = time.time()
    df = compute_regime_features(df)
    print(f"  Regime features in {time.time()-t1:.0f}s")

    print(f"  Computing novel rolling features...", flush=True)
    t1 = time.time()
    df = compute_novel_rolling(df)
    print(f"  Novel rolling features in {time.time()-t1:.0f}s")

    # --- Forward targets (for vol prediction used in S5 reference) ---
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

    # --- Walk-forward vol prediction (for S5 reference) ---
    warmup = 2500

    print(f"\n  [Ref] Walk-forward 1h vol prediction (Ridge)...", flush=True)
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

    # Extract novel signal arrays
    cum_toxic_z = df_sim["cum_toxic_z"].values if "cum_toxic_z" in df_sim.columns else np.zeros(n_sim)
    herding_z = df_sim["herding_z"].values if "herding_z" in df_sim.columns else np.zeros(n_sim)
    eff_z = df_sim["eff_z"].values if "eff_z" in df_sim.columns else np.zeros(n_sim)
    vpin_1h = df_sim["vpin_1h"].values if "vpin_1h" in df_sim.columns else np.full(n_sim, 0.5)
    informed_z = df_sim["informed_composite_z"].values if "informed_composite_z" in df_sim.columns else np.zeros(n_sim)
    multifrac = df_sim["multifractal_ratio"].values if "multifractal_ratio" in df_sim.columns else np.ones(n_sim)

    # Rolling stats for thresholds
    vpin_median = np.nanmedian(vpin_1h[~np.isnan(vpin_1h)])

    print(f"\n  Simulation: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price: ${prices_close.min():.2f} - ${prices_close.max():.2f}")
    print(f"  Median pred vol: {median_pred_vol:.6f}")
    print(f"  Median VPIN(1h): {vpin_median:.4f}")

    bot = GridBotSimulator(n_levels=5, fee_bps=MAKER_FEE_BPS)
    strategies = []

    # --- S0: Baseline Fix 1.00% (24h) ---
    fixed_spacing = np.full(n_sim, 0.0100)
    fixed_rebal = np.full(n_sim, 288, dtype=int)
    strategies.append({
        "name": "S0: Fix 1.00% (24h)",
        "spacings": fixed_spacing,
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # --- S5: Adaptive rebalance (v15 winner for BTC/ETH) ---
    s5_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]):
            if pv1h[i] > median_pred_vol * 2.0:
                s5_rebal[i] = 96   # 8h in high vol
            elif pv1h[i] < median_pred_vol * 0.5:
                s5_rebal[i] = 576  # 48h in low vol
    strategies.append({
        "name": "S5ref: AdaptRebal (v15)",
        "spacings": fixed_spacing.copy(),
        "rebalance": s5_rebal,
        "paused": None,
    })

    # =====================================================================
    # NOVEL SIGNAL STRATEGIES
    # =====================================================================

    # --- N1: Toxic Flow Pause ---
    # When cumulative toxic flow z-score > 2.0, informed traders are active.
    # Pause the grid to avoid adverse selection.
    n1_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(cum_toxic_z[i]) and abs(cum_toxic_z[i]) > 2.0:
            n1_paused[i] = True
    pause_pct = n1_paused.sum() / n_sim * 100
    print(f"\n  N1 toxic pause: {pause_pct:.1f}% of bars paused")
    strategies.append({
        "name": "N1: Toxic pause (z>2)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": n1_paused,
    })

    # --- N1b: Toxic Flow Pause (tighter threshold) ---
    n1b_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(cum_toxic_z[i]) and abs(cum_toxic_z[i]) > 1.5:
            n1b_paused[i] = True
    pause_pct = n1b_paused.sum() / n_sim * 100
    print(f"  N1b toxic pause: {pause_pct:.1f}% of bars paused")
    strategies.append({
        "name": "N1b: Toxic pause (z>1.5)",
        "spacings": fixed_spacing.copy(),
        "rebalance": fixed_rebal.copy(),
        "paused": n1b_paused,
    })

    # --- N2: Herding Widen ---
    # When herding z-score > 1.5, momentum is building. Widen grid to 1.5%.
    n2_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(herding_z[i]):
            if abs(herding_z[i]) > 1.5:
                n2_spacings[i] = 0.0150  # widen to 1.5%
            elif abs(herding_z[i]) > 2.0:
                n2_spacings[i] = 0.0200  # widen to 2.0%
    strategies.append({
        "name": "N2: Herding widen",
        "spacings": n2_spacings,
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # --- N3: Efficiency Tighten ---
    # Low efficiency = choppy/noisy = perfect for grids → tighten to 0.7%
    # High efficiency = trending = bad for grids → widen to 1.5%
    n3_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(eff_z[i]):
            if eff_z[i] < -1.0:  # very choppy
                n3_spacings[i] = 0.0070
            elif eff_z[i] > 1.5:  # very trending
                n3_spacings[i] = 0.0150
    strategies.append({
        "name": "N3: Efficiency adapt",
        "spacings": n3_spacings,
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # --- N4: VPIN Adaptive Spacing ---
    # High VPIN = informed trading = widen (market maker logic)
    # Low VPIN = noise traders = tighten
    n4_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(vpin_1h[i]):
            if vpin_1h[i] > vpin_median * 1.3:
                n4_spacings[i] = 0.0130  # widen 30%
            elif vpin_1h[i] < vpin_median * 0.7:
                n4_spacings[i] = 0.0080  # tighten 20%
    strategies.append({
        "name": "N4: VPIN adapt spacing",
        "spacings": n4_spacings,
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # --- N5: Composite Informed → Smart Rebalance ---
    # Rebalance immediately when informed flow spikes (z > 2.0)
    # Otherwise use normal 24h
    n5_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 2.0:
            n5_rebal[i] = 1  # force immediate rebalance
    strategies.append({
        "name": "N5: Informed rebalance",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5_rebal,
        "paused": None,
    })

    # --- N5b: Composite Informed → Smart Rebalance (softer) ---
    n5b_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(informed_z[i]) and abs(informed_z[i]) > 1.5:
            n5b_rebal[i] = 12  # rebalance within 1h
    strategies.append({
        "name": "N5b: Informed rebal soft",
        "spacings": fixed_spacing.copy(),
        "rebalance": n5b_rebal,
        "paused": None,
    })

    # --- N6: Multifractal Vol Boost ---
    # High multifractal ratio = clustered vol = widen grid preemptively
    n6_spacings = np.full(n_sim, 0.0100)
    mf_median = np.nanmedian(multifrac[~np.isnan(multifrac)])
    for i in range(n_sim):
        if not np.isnan(multifrac[i]):
            if multifrac[i] > mf_median * 1.5:
                n6_spacings[i] = 0.0140  # vol about to spike, widen
            elif multifrac[i] < mf_median * 0.7:
                n6_spacings[i] = 0.0080  # smooth vol, tighten
    strategies.append({
        "name": "N6: Multifractal adapt",
        "spacings": n6_spacings,
        "rebalance": fixed_rebal.copy(),
        "paused": None,
    })

    # =====================================================================
    # COMBO STRATEGIES (best novel + best v15)
    # =====================================================================

    # --- C1: Toxic Pause + Adaptive Rebalance (S5) ---
    strategies.append({
        "name": "C1: Toxic+AdaptRebal",
        "spacings": fixed_spacing.copy(),
        "rebalance": s5_rebal.copy(),
        "paused": n1_paused.copy(),
    })

    # --- C2: Efficiency Adapt + Adaptive Rebalance ---
    strategies.append({
        "name": "C2: Eff+AdaptRebal",
        "spacings": n3_spacings.copy(),
        "rebalance": s5_rebal.copy(),
        "paused": None,
    })

    # --- C3: VPIN + Toxic Pause + Adaptive Rebalance ---
    strategies.append({
        "name": "C3: VPIN+Toxic+AdRebal",
        "spacings": n4_spacings.copy(),
        "rebalance": s5_rebal.copy(),
        "paused": n1_paused.copy(),
    })

    # --- C4: Herding Widen + Informed Rebalance ---
    strategies.append({
        "name": "C4: Herd+InfRebal",
        "spacings": n2_spacings.copy(),
        "rebalance": n5_rebal.copy(),
        "paused": None,
    })

    # --- C5: Kitchen Sink (Efficiency + Toxic Pause + Informed Rebalance) ---
    strategies.append({
        "name": "C5: Eff+Toxic+InfRebal",
        "spacings": n3_spacings.copy(),
        "rebalance": n5_rebal.copy(),
        "paused": n1_paused.copy(),
    })

    # --- Run all strategies ---
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
        )
        results.append(r)
        elapsed = time.time() - t_s

        print(f"  {r['strategy']}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} (grid: ${r['grid_profits']:+.2f}, fees: -${r['total_fees']:.2f})")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills: {r['fills']:,} ({r['fills_per_day']:.1f}/d) | "
              f"Sharpe: {r['sharpe']:.2f} | MaxDD: ${r['max_drawdown']:.2f} ({elapsed:.1f}s)\n")

    # --- Summary table ---
    baseline = results[0]
    print(f"  {'='*100}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days)")
    print(f"  {'='*100}")
    print(f"  {'Strategy':<30s} {'PnL':>10s} {'Grid$':>10s} {'Fees':>8s} {'PnL/d':>8s} "
          f"{'Fills':>7s} {'Sharpe':>7s} {'MaxDD':>10s} {'vs Base':>8s}")
    print(f"  {'-'*103}")
    for r in results:
        delta = r['total_pnl'] - baseline['total_pnl']
        marker = "✅" if delta > 0 and r is not baseline else "  "
        if r is baseline:
            print(f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+9.2f} ${r['grid_profits']:>+9.2f} "
                  f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+7.2f} "
                  f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>9.2f} "
                  f"{'':>8s}")
        else:
            print(f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+9.2f} ${r['grid_profits']:>+9.2f} "
                  f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+7.2f} "
                  f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>9.2f} "
                  f"${delta:>+7.0f}")

    elapsed_total = time.time() - t_total
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()
    run_v16(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
