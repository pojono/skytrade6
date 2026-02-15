#!/usr/bin/env python3
"""
Grid trading experiments on Bybit futures (BTC, ETH, SOL).

Tests multiple grid variants:
  G01: Fixed symmetric grid (baseline)
  G02: Volatility-scaled grid (cell width = k * realized_vol)
  G03: Asymmetric grid (bias cells based on microstructure direction)
  G04: Regime-adaptive grid (grid in range, pause in trend)
  G05: Time-of-day filtered grid (avoid low-liquidity hours)
  G06: Trend overlay grid (grid + momentum signal in trends)
  G07: Dynamic sizing grid (reduce size as inventory grows)
  G08: Full hybrid (G02 + G03 + G04 + G05 + G07 combined)

All use Bybit VIP0 fees: maker 2 bps + taker 5 bps.
Grid orders are limit orders (maker fee), trend orders are market (taker fee).
"""

import sys
import time
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SOURCE = "bybit_futures"
PARQUET_DIR = Path("./parquet")

# Bybit VIP0 fees
MAKER_FEE_BPS = 2.0   # grid fills (limit orders)
TAKER_FEE_BPS = 5.0   # trend entries (market orders)
GRID_RT_FEE_BPS = MAKER_FEE_BPS * 2  # grid: maker both sides = 4 bps
TREND_RT_FEE_BPS = TAKER_FEE_BPS + MAKER_FEE_BPS  # trend: taker entry + maker exit = 7 bps

PERIOD_7D = ("2025-12-01", "2025-12-07")
PERIOD_30D = ("2025-12-01", "2025-12-30")


# ---------------------------------------------------------------------------
# Feature computation (streamlined for grid)
# ---------------------------------------------------------------------------

def compute_features(trades, interval_us=300_000_000):
    bucket = (trades["timestamp_us"].values // interval_us) * interval_us
    trades = trades.copy()
    trades["bucket"] = bucket

    features = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        t = grp["timestamp_us"].values
        n = len(grp)
        if n < 10:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()

        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        vwap = qq.sum() / max(total_vol, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        ret = (close_p - open_p) / max(open_p, 1e-10)
        price_range = (high_p - low_p) / max(vwap, 1e-10)

        # Price efficiency (Kaufman)
        price_changes = np.abs(np.diff(p))
        total_path = price_changes.sum()
        net_move = abs(p[-1] - p[0])
        efficiency = net_move / max(total_path, 1e-10)

        # Sign autocorrelation
        signs = s.astype(float)
        sign_ac1 = np.corrcoef(signs[:-1], signs[1:])[0, 1] if n > 5 else 0

        # Cumulative imbalance (rolling will be added later)
        # Hour of day (UTC)
        hour_utc = int((bkt // 3_600_000_000) % 24)

        features.append({
            "timestamp_us": bkt,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "trade_count": n,
            "returns": ret, "price_range": price_range,
            "vol_imbalance": vol_imbalance,
            "count_imbalance": count_imbalance,
            "close_vs_vwap": close_vs_vwap,
            "efficiency": efficiency,
            "sign_ac1": sign_ac1,
            "hour_utc": hour_utc,
        })

    return pd.DataFrame(features)


def load_features(symbol, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    all_feat = []
    t0 = time.time()

    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{date_str}.parquet"
        if not path.exists():
            continue

        trades = pd.read_parquet(path)
        feat = compute_features(trades)
        del trades
        all_feat.append(feat)

        elapsed = time.time() - t0
        eta = (len(dates) - i) / (i / elapsed) if i > 0 else 0
        mem_gb = psutil.virtual_memory().used / (1024**3)
        if i % 5 == 0 or i == len(dates):
            print(f"    [{i:2d}/{len(dates)}] {date_str}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  RAM={mem_gb:.1f}GB", flush=True)

    if not all_feat:
        return pd.DataFrame()

    df = pd.concat(all_feat, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)

    # Add derived features
    df["rvol_12"] = df["returns"].rolling(12).std()      # 1h realized vol
    df["rvol_288"] = df["returns"].rolling(288).std()     # 1d realized vol
    df["vol_ratio"] = df["rvol_12"] / df["rvol_288"].clip(lower=1e-10)
    df["cum_imbalance_12"] = df["vol_imbalance"].rolling(12).sum()
    df["mom_12"] = df["close"].pct_change(12)

    # Regime: efficiency + sign_ac rolling
    df["eff_rolling"] = df["efficiency"].rolling(12).mean()
    df["sign_ac_rolling"] = df["sign_ac1"].rolling(12).mean()

    return df


# ---------------------------------------------------------------------------
# Grid Backtester
# ---------------------------------------------------------------------------

class GridBacktester:
    """
    Simulates a grid trading strategy on 5m OHLCV bars.

    The grid places limit buy orders below current price and limit sell orders above.
    When price crosses a grid level, the order fills (at maker fee).
    Each fill is a round-trip: buy at level, sell at level + cell_width (or vice versa).
    """

    def __init__(self, n_levels=5, cell_width_mult=1.0, size_decay=0.85,
                 asymmetry=0.0, vol_scale=True, regime_filter=False,
                 time_filter=False, trend_overlay=False, dynamic_sizing=True,
                 max_inventory=3.0, handbrake_vol_ratio=3.0,
                 grid_fee_bps=4.0, trend_fee_bps=7.0):
        self.n_levels = n_levels
        self.cell_width_mult = cell_width_mult
        self.size_decay = size_decay
        self.asymmetry = asymmetry  # -1 to +1: bias toward buys or sells
        self.vol_scale = vol_scale
        self.regime_filter = regime_filter
        self.time_filter = time_filter
        self.trend_overlay = trend_overlay
        self.dynamic_sizing = dynamic_sizing
        self.max_inventory = max_inventory
        self.handbrake_vol_ratio = handbrake_vol_ratio
        self.grid_fee_bps = grid_fee_bps
        self.trend_fee_bps = trend_fee_bps

    def run(self, df):
        """Run grid backtest on DataFrame with OHLCV + features."""
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        rvol = df["rvol_12"].values
        vol_ratio = df["vol_ratio"].values
        vol_imbalance = df["vol_imbalance"].values
        cum_imbalance = df["cum_imbalance_12"].values
        efficiency = df["eff_rolling"].values
        sign_ac = df["sign_ac_rolling"].values
        hours = df["hour_utc"].values
        n = len(df)

        trades = []       # completed round-trips
        inventory = 0.0   # net position in units (+ = long, - = short)
        total_pnl = 0.0

        for i in range(288, n):  # skip warmup
            price = closes[i]
            rv = rvol[i] if not np.isnan(rvol[i]) else 0.001
            vr = vol_ratio[i] if not np.isnan(vol_ratio[i]) else 1.0

            # --- Handbrake: extreme volatility ---
            if vr > self.handbrake_vol_ratio:
                continue

            # --- Time filter: skip low-liquidity hours ---
            if self.time_filter:
                h = hours[i]
                # Skip 2-6 UTC (lowest liquidity in crypto)
                if 2 <= h <= 5:
                    continue

            # --- Regime detection ---
            eff = efficiency[i] if not np.isnan(efficiency[i]) else 0.3
            sac = sign_ac[i] if not np.isnan(sign_ac[i]) else 0.0
            is_trending = (eff > 0.4 and abs(sac) > 0.1)

            if self.regime_filter and is_trending:
                # In trend regime: skip grid, use trend overlay if enabled
                if self.trend_overlay:
                    ci = cum_imbalance[i] if not np.isnan(cum_imbalance[i]) else 0
                    if abs(ci) > 1.5 and abs(inventory) < self.max_inventory:
                        direction = np.sign(ci)
                        # Trend trade: follow cumulative imbalance
                        size = 1.0
                        if self.dynamic_sizing:
                            # Reduce if already exposed in same direction
                            same_dir = (inventory * direction) > 0
                            if same_dir:
                                size *= max(0.3, 1.0 - abs(inventory) / self.max_inventory)

                        # Simulate: enter now, exit in 24 bars (2h)
                        if i + 24 < n:
                            exit_price = closes[i + 24]
                            raw_ret = (exit_price / price - 1) * 10000 * direction
                            net_ret = (raw_ret - self.trend_fee_bps) * size
                            trades.append({
                                "bar": i, "type": "trend",
                                "direction": direction, "size": size,
                                "entry_price": price, "exit_price": exit_price,
                                "pnl_bps": net_ret,
                            })
                            total_pnl += net_ret
                continue

            # --- Grid logic ---
            # Cell width based on realized volatility
            if self.vol_scale:
                cell_width = max(rv * price * self.cell_width_mult, price * 0.0005)
            else:
                cell_width = price * 0.002 * self.cell_width_mult  # fixed 0.2%

            # Asymmetry: shift grid based on microstructure signal
            if self.asymmetry != 0:
                vi = vol_imbalance[i] if not np.isnan(vol_imbalance[i]) else 0
                # Positive imbalance (buying pressure) → shift grid up (more sells)
                # Negative imbalance → shift grid down (more buys)
                bias = -vi * self.asymmetry  # contrarian bias
            else:
                bias = 0

            # Check if price moved through any grid levels
            bar_high = highs[i]
            bar_low = lows[i]

            for level in range(1, self.n_levels + 1):
                # Size for this level
                size = self.size_decay ** (level - 1)
                if self.dynamic_sizing:
                    # Reduce size when inventory is high
                    inv_penalty = max(0.2, 1.0 - abs(inventory) / self.max_inventory)
                    size *= inv_penalty

                # Buy level (below price)
                buy_offset = level + max(0, bias)
                buy_level = price - cell_width * buy_offset
                # Sell level (above price)
                sell_offset = level + max(0, -bias)
                sell_level = price + cell_width * sell_offset

                # Check if bar's low touched buy level
                if bar_low <= buy_level and inventory < self.max_inventory:
                    # Buy filled → will sell at buy_level + cell_width
                    take_profit = buy_level + cell_width
                    # Simplified: assume TP fills within next few bars
                    # Check next 48 bars (4h) for TP
                    filled = False
                    for j in range(i + 1, min(i + 49, n)):
                        if highs[j] >= take_profit:
                            raw_ret = (take_profit - buy_level) / buy_level * 10000
                            net_ret = (raw_ret - self.grid_fee_bps) * size
                            trades.append({
                                "bar": i, "type": "grid_buy",
                                "direction": 1, "size": size,
                                "entry_price": buy_level, "exit_price": take_profit,
                                "pnl_bps": net_ret,
                            })
                            total_pnl += net_ret
                            filled = True
                            break
                    if not filled:
                        # Didn't hit TP in 4h → close at market (loss)
                        if i + 48 < n:
                            exit_p = closes[min(i + 48, n - 1)]
                            raw_ret = (exit_p - buy_level) / buy_level * 10000
                            net_ret = (raw_ret - self.grid_fee_bps) * size
                            trades.append({
                                "bar": i, "type": "grid_buy_timeout",
                                "direction": 1, "size": size,
                                "entry_price": buy_level, "exit_price": exit_p,
                                "pnl_bps": net_ret,
                            })
                            total_pnl += net_ret

                # Check if bar's high touched sell level
                if bar_high >= sell_level and inventory > -self.max_inventory:
                    take_profit = sell_level - cell_width
                    filled = False
                    for j in range(i + 1, min(i + 49, n)):
                        if lows[j] <= take_profit:
                            raw_ret = (sell_level - take_profit) / sell_level * 10000
                            net_ret = (raw_ret - self.grid_fee_bps) * size
                            trades.append({
                                "bar": i, "type": "grid_sell",
                                "direction": -1, "size": size,
                                "entry_price": sell_level, "exit_price": take_profit,
                                "pnl_bps": net_ret,
                            })
                            total_pnl += net_ret
                            filled = True
                            break
                    if not filled:
                        if i + 48 < n:
                            exit_p = closes[min(i + 48, n - 1)]
                            raw_ret = (sell_level - exit_p) / sell_level * 10000
                            net_ret = (raw_ret - self.grid_fee_bps) * size
                            trades.append({
                                "bar": i, "type": "grid_sell_timeout",
                                "direction": -1, "size": size,
                                "entry_price": sell_level, "exit_price": exit_p,
                                "pnl_bps": net_ret,
                            })
                            total_pnl += net_ret

        if not trades:
            return pd.DataFrame(), {}

        trades_df = pd.DataFrame(trades)
        trades_df["cum_pnl"] = trades_df["pnl_bps"].cumsum()

        grid_trades = trades_df[trades_df["type"].str.startswith("grid")]
        trend_trades = trades_df[trades_df["type"] == "trend"]

        summary = {
            "total_trades": len(trades_df),
            "grid_trades": len(grid_trades),
            "trend_trades": len(trend_trades),
            "total_pnl_bps": trades_df["pnl_bps"].sum(),
            "avg_pnl_bps": trades_df["pnl_bps"].mean(),
            "win_rate": (trades_df["pnl_bps"] > 0).mean(),
            "grid_avg_pnl": grid_trades["pnl_bps"].mean() if len(grid_trades) > 0 else 0,
            "grid_wr": (grid_trades["pnl_bps"] > 0).mean() if len(grid_trades) > 0 else 0,
            "trend_avg_pnl": trend_trades["pnl_bps"].mean() if len(trend_trades) > 0 else 0,
            "trend_wr": (trend_trades["pnl_bps"] > 0).mean() if len(trend_trades) > 0 else 0,
            "max_dd": (trades_df["cum_pnl"] - trades_df["cum_pnl"].cummax()).min(),
        }

        return trades_df, summary


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

GRID_EXPERIMENTS = [
    ("G01_fixed_symmetric", "Fixed symmetric grid (baseline)", {
        "n_levels": 5, "cell_width_mult": 1.0, "vol_scale": False,
        "asymmetry": 0, "regime_filter": False, "time_filter": False,
        "trend_overlay": False, "dynamic_sizing": False,
    }),
    ("G02_vol_scaled", "Volatility-scaled cell width", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 0, "regime_filter": False, "time_filter": False,
        "trend_overlay": False, "dynamic_sizing": False,
    }),
    ("G03_asymmetric", "Asymmetric grid (contrarian bias from imbalance)", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 1.0, "regime_filter": False, "time_filter": False,
        "trend_overlay": False, "dynamic_sizing": False,
    }),
    ("G04_regime_adaptive", "Regime-adaptive: grid in range, pause in trend", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 0, "regime_filter": True, "time_filter": False,
        "trend_overlay": False, "dynamic_sizing": False,
    }),
    ("G05_time_filtered", "Time-of-day filter: skip 2-5 UTC", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 0, "regime_filter": False, "time_filter": True,
        "trend_overlay": False, "dynamic_sizing": False,
    }),
    ("G06_trend_overlay", "Grid + trend overlay (momentum in trends)", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 0, "regime_filter": True, "time_filter": False,
        "trend_overlay": True, "dynamic_sizing": False,
    }),
    ("G07_dynamic_sizing", "Dynamic sizing (reduce at high inventory)", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 0, "regime_filter": False, "time_filter": False,
        "trend_overlay": False, "dynamic_sizing": True,
    }),
    ("G08_full_hybrid", "Full hybrid: vol-scaled + asymmetric + regime + time + trend + dynamic", {
        "n_levels": 5, "cell_width_mult": 1.5, "vol_scale": True,
        "asymmetry": 1.0, "regime_filter": True, "time_filter": True,
        "trend_overlay": True, "dynamic_sizing": True,
    }),
    ("G09_tight_grid", "Tight grid: more levels, smaller cells", {
        "n_levels": 7, "cell_width_mult": 0.8, "vol_scale": True,
        "asymmetry": 0.5, "regime_filter": True, "time_filter": True,
        "trend_overlay": True, "dynamic_sizing": True,
    }),
    ("G10_wide_grid", "Wide grid: fewer levels, larger cells", {
        "n_levels": 3, "cell_width_mult": 2.5, "vol_scale": True,
        "asymmetry": 0.5, "regime_filter": True, "time_filter": True,
        "trend_overlay": True, "dynamic_sizing": True,
    }),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_grid_experiments(symbol, start_date, end_date, label):
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    print(f"\n{'='*70}")
    print(f"  {symbol} — {label} ({days} days: {start_date} → {end_date})")
    print(f"{'='*70}")

    print(f"  Loading features...", flush=True)
    df = load_features(symbol, start_date, end_date)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, price {df['close'].min():.2f}–{df['close'].max():.2f}")

    winners = []

    for exp_name, exp_desc, params in GRID_EXPERIMENTS:
        print(f"\n  📋 {exp_name}: {exp_desc}", flush=True)

        try:
            grid = GridBacktester(**params)
            trades_df, summary = grid.run(df)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        if not summary:
            print(f"    — No trades")
            continue

        nt = summary["total_trades"]
        avg = summary["avg_pnl_bps"]
        total = summary["total_pnl_bps"]
        wr = summary["win_rate"]
        gt = summary["grid_trades"]
        tt = summary["trend_trades"]
        gavg = summary["grid_avg_pnl"]
        tavg = summary["trend_avg_pnl"]
        dd = summary["max_dd"]

        marker = "✅" if avg > 0 and nt >= 20 else "  "
        print(f"    {marker} Trades={nt} (grid={gt}, trend={tt}), "
              f"Avg={avg:+.2f} bps, Total={total:+.1f}, WR={wr:.0%}, DD={dd:+.1f}")
        if gt > 0:
            print(f"       Grid: avg={gavg:+.2f} bps, WR={summary['grid_wr']:.0%}")
        if tt > 0:
            print(f"       Trend: avg={tavg:+.2f} bps, WR={summary['trend_wr']:.0%}")

        if avg > 0 and nt >= 20:
            winners.append({
                "experiment": exp_name, "symbol": symbol, "period": label,
                "n_trades": nt, "grid_trades": gt, "trend_trades": tt,
                "avg_pnl_bps": avg, "total_pnl_bps": total,
                "win_rate": wr, "max_dd": dd,
            })

    return winners


def main():
    t_start = time.time()
    print("=" * 70)
    print("  GRID TRADING EXPERIMENTS: Adaptive Hybrid Grid on Bybit Futures")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Grid fees: {GRID_RT_FEE_BPS} bps RT (maker+maker)")
    print(f"  Trend fees: {TREND_RT_FEE_BPS} bps RT (taker+maker)")
    print(f"  Experiments: {len(GRID_EXPERIMENTS)}")
    print("=" * 70)

    all_winners = []

    # Phase 1: 7-day screen
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: 7-DAY SCREENING")
    print(f"{'#'*70}")

    for symbol in SYMBOLS:
        winners = run_grid_experiments(symbol, *PERIOD_7D, "7d")
        all_winners.extend(winners)

    print(f"\n\n{'='*70}")
    print(f"  PHASE 1 RESULTS: 7-Day Winners")
    print(f"{'='*70}")

    if all_winners:
        print(f"  {'Experiment':25s} {'Symbol':>10s} {'Trades':>7s} {'Grid':>6s} {'Trend':>6s} "
              f"{'Avg':>8s} {'Total':>9s} {'WR':>5s} {'DD':>8s}")
        print(f"  {'-'*90}")
        for w in sorted(all_winners, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {w['experiment']:25s} {w['symbol']:>10s} {w['n_trades']:>7d} "
                  f"{w['grid_trades']:>6d} {w['trend_trades']:>6d} "
                  f"{w['avg_pnl_bps']:>+8.2f} {w['total_pnl_bps']:>+9.1f} "
                  f"{w['win_rate']:>5.0%} {w['max_dd']:>+8.1f}")
    else:
        print("  ❌ No winners!")

    # Phase 2: 30-day validation
    winning_experiments = set(w["experiment"] for w in all_winners)
    if not winning_experiments:
        print("\n  No experiments to validate.")
        return

    print(f"\n\n{'#'*70}")
    print(f"  PHASE 2: 30-DAY VALIDATION ({len(winning_experiments)} experiments)")
    print(f"{'#'*70}")

    validated = []
    for symbol in SYMBOLS:
        w30 = run_grid_experiments(symbol, *PERIOD_30D, "30d")
        for w in w30:
            if w["experiment"] in winning_experiments:
                validated.append(w)

    print(f"\n\n{'='*70}")
    print(f"  FINAL: 30-Day Validated Grid Winners")
    print(f"{'='*70}")

    if validated:
        print(f"  {'Experiment':25s} {'Symbol':>10s} {'Trades':>7s} {'Grid':>6s} {'Trend':>6s} "
              f"{'Avg':>8s} {'Total':>9s} {'WR':>5s} {'DD':>8s}")
        print(f"  {'-'*90}")
        for w in sorted(validated, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {w['experiment']:25s} {w['symbol']:>10s} {w['n_trades']:>7d} "
                  f"{w['grid_trades']:>6d} {w['trend_trades']:>6d} "
                  f"{w['avg_pnl_bps']:>+8.2f} {w['total_pnl_bps']:>+9.1f} "
                  f"{w['win_rate']:>5.0%} {w['max_dd']:>+8.1f}")
    else:
        print("  ❌ No grid experiments survived 30-day validation!")

    elapsed = time.time() - t_start
    print(f"\n✅ Grid experiments complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
