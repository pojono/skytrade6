#!/usr/bin/env python3
"""
XS-7 — Bracket / Convex Execution on S07 (Production-grade Backtest)

Goal: Can we turn S07's "elevated big move probability" into positive net EV?

Approach:
  - On S07 signal, place OCO bracket: buy-stop at P0+X, sell-stop at P0-X
  - After one leg fills, cancel the other
  - Exit via TP, SL, or 24h time stop
  - Model double-trigger (whipsaw) explicitly with cancel latency
  - Realistic fee models (maker/taker) and slippage

Signal: S07 compress_oi (rv_6h <= P20 AND oi_z >= 1.5), per-coin causal
Cooldown: 24h per coin after position open

Parameter grid:
  a (bracket width):  {0.8, 1.0, 1.2, 1.5} * ATR_1h
  b (TP distance):    {3, 4, 5} * ATR_1h
  c (SL distance):    {1.5, 2.0, 2.5} * ATR_1h
  cancel_delay:       {0, 60, 300} seconds (0=instant, 60=1min, 300=5min)
  fee_model:          M (maker TP), T (all taker)
  slippage:           {0, 5, 10} bp each side

Walk-forward: Train Jan / Test Feb (and reverse) with ±24h purge.
"""

import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# Reuse xs6 infrastructure
from xs6_bigmove_uplift import (
    DATA_DIR, START, END, MIN_DAYS, SIGNAL_STEP_MIN,
    TRAIN_END, TEST_START, PURGE_HOURS, SEED,
    discover_symbols, load_symbol, build_sym_1m, compute_features,
    compute_states,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# XS-7 Config
# ---------------------------------------------------------------------------

# Bracket width: a * ATR_1h
A_GRID = [0.8, 1.0, 1.2, 1.5]
# TP distance: b * ATR_1h from entry
B_GRID = [3, 4, 5]
# SL distance: c * ATR_1h from entry
C_GRID = [1.5, 2.0, 2.5]

# Cancel delay in seconds (for double-trigger modeling)
# 0 = instant cancel (ideal), 60 = 1min, 300 = 5min
CANCEL_DELAY_GRID = [0, 60, 300]

# Fee models
# M = entry taker + TP maker + SL taker
# T = all taker (worst case)
FEE_TAKER_BP = 10.0   # 10 bp per side (0.1%)
FEE_MAKER_BP = 2.0    # 2 bp per side (0.02%)

# Slippage grid (bp, applied to entry and exit)
SLIP_GRID = [0, 5, 10]

# Cooldown per coin after opening position
COOLDOWN_MIN = 24 * 60  # 24h in minutes

# Time stop from signal time t0
TIME_STOP_MIN = 24 * 60  # 24h

# Walk-forward splits
PURGE_START_FWD = TRAIN_END - pd.Timedelta(hours=PURGE_HOURS)
PURGE_END_FWD = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
PURGE_START_REV = TEST_START - pd.Timedelta(hours=PURGE_HOURS)
PURGE_END_REV = TRAIN_END + pd.Timedelta(hours=PURGE_HOURS)


# ---------------------------------------------------------------------------
# §1: Build 1m data with S07 signal and ATR for each symbol
# ---------------------------------------------------------------------------

def build_symbol_data(sym: str, raw: dict, grid_1m: pd.DatetimeIndex):
    """Build 1m DataFrame with close, high, low, atr_1h_raw, and S07 signal on 5m grid.
    Returns (df_1m, signal_times_5m) or (None, None) if insufficient data."""
    df_1m = build_sym_1m(sym, raw, grid_1m)
    valid_pct = 1 - df_1m["is_invalid"].mean()
    if valid_pct < 0.5:
        return None, None

    df_1m = compute_features(df_1m)

    # Sample to 5m for signal detection
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]
    df_5m = df_1m.loc[grid_5m].copy()
    df_5m = compute_states(df_5m)

    # S07 signal times
    s07 = df_5m["S07_compress_oi"].values == 1.0
    signal_times = df_5m.index[s07]

    return df_1m, signal_times


# ---------------------------------------------------------------------------
# §2: Bracket simulation engine (1m resolution)
# ---------------------------------------------------------------------------

def simulate_bracket_trades(df_1m: pd.DataFrame, signal_times: pd.DatetimeIndex,
                            sym: str, a: float, b: float, c: float,
                            cancel_delay_s: int) -> list[dict]:
    """Simulate bracket OCO trades on 1m data.

    For each signal:
      1. Place buy-stop at P0 + a*ATR and sell-stop at P0 - a*ATR
      2. Wait for one to trigger (within 24h time window)
      3. After trigger, check for double-trigger within cancel_delay
      4. If no double-trigger, manage position with TP/SL/time-stop
      5. Record trade

    Returns list of trade dicts.
    """
    close = df_1m["close"].values
    high = df_1m["high"].values if "high" in df_1m.columns else close
    low = df_1m["low"].values if "low" in df_1m.columns else close
    atr = df_1m["atr_1h_raw"].values
    ts = df_1m.index
    is_invalid = df_1m["is_invalid"].values

    # Build timestamp -> index lookup
    ts_to_idx = {t: i for i, t in enumerate(ts)}

    cancel_delay_bars = max(1, cancel_delay_s // 60)  # convert to 1m bars

    trades = []
    last_entry_time = {}  # sym -> last entry timestamp (for cooldown)

    for t_signal in signal_times:
        # Cooldown check
        if sym in last_entry_time:
            if (t_signal - last_entry_time[sym]).total_seconds() < COOLDOWN_MIN * 60:
                continue

        idx0 = ts_to_idx.get(t_signal)
        if idx0 is None or idx0 >= len(ts) - 10:
            continue

        p0 = close[idx0]
        atr_val = atr[idx0]
        if np.isnan(p0) or np.isnan(atr_val) or atr_val <= 0 or p0 <= 0:
            continue

        # ATR as fraction of price (for return calculations)
        atr_ret = atr_val / p0

        # Bracket levels
        buy_stop = p0 * (1 + a * atr_ret)
        sell_stop = p0 * (1 - a * atr_ret)

        # Time stop window (from signal)
        max_idx = min(idx0 + TIME_STOP_MIN, len(ts) - 1)

        # --- Phase 1: Wait for bracket trigger ---
        entry_idx = None
        entry_side = None  # 'long' or 'short'
        entry_px = None

        for i in range(idx0 + 1, max_idx + 1):
            if is_invalid[i]:
                continue
            # Check if high >= buy_stop (buy-stop triggered)
            h = high[i] if not np.isnan(high[i]) else close[i]
            l = low[i] if not np.isnan(low[i]) else close[i]

            buy_triggered = h >= buy_stop
            sell_triggered = l <= sell_stop

            if buy_triggered and sell_triggered:
                # Both triggered in same bar — this IS a double-trigger
                entry_idx = i
                entry_side = "long"  # arbitrary, take buy first
                entry_px = buy_stop
                # Double trigger happens regardless of cancel delay
                double_trigger = True
                opposite_px = sell_stop
                break
            elif buy_triggered:
                entry_idx = i
                entry_side = "long"
                entry_px = buy_stop
                break
            elif sell_triggered:
                entry_idx = i
                entry_side = "short"
                entry_px = sell_stop
                break

        if entry_idx is None:
            # No trigger within time window — bracket expired
            trades.append({
                "symbol": sym,
                "t_signal": t_signal,
                "t_entry": pd.NaT,
                "side": "none",
                "P0": p0,
                "X_atr_mult": a,
                "X_bp": a * atr_ret * 10000,
                "entry_px": np.nan,
                "tp_px": np.nan,
                "sl_px": np.nan,
                "exit_reason": "NO_TRIGGER",
                "exit_px": np.nan,
                "gross_pnl_bp": 0.0,
                "MAE_bp": 0.0,
                "MFE_bp": 0.0,
                "double_trigger": False,
                "time_to_entry_min": np.nan,
                "time_in_trade_min": 0,
                "atr_at_signal": atr_val,
            })
            continue

        # --- Phase 1b: Check double-trigger within cancel_delay ---
        double_trigger = False
        opposite_px = None

        if entry_side == "long":
            # Check if sell_stop was hit within cancel_delay bars after entry
            for j in range(entry_idx, min(entry_idx + cancel_delay_bars + 1, max_idx + 1)):
                lj = low[j] if not np.isnan(low[j]) else close[j]
                if lj <= sell_stop:
                    double_trigger = True
                    opposite_px = sell_stop
                    break
        else:  # short
            for j in range(entry_idx, min(entry_idx + cancel_delay_bars + 1, max_idx + 1)):
                hj = high[j] if not np.isnan(high[j]) else close[j]
                if hj >= buy_stop:
                    double_trigger = True
                    opposite_px = buy_stop
                    break

        # --- Phase 2: Manage position (TP / SL / Time-stop) ---
        if entry_side == "long":
            tp_px = entry_px * (1 + b * atr_ret)
            sl_px = entry_px * (1 - c * atr_ret)
        else:
            tp_px = entry_px * (1 - b * atr_ret)
            sl_px = entry_px * (1 + c * atr_ret)

        exit_reason = "TIME"
        exit_px = close[max_idx]  # default: time stop at close
        exit_idx = max_idx
        mae = 0.0  # max adverse excursion (bp from entry, always negative or 0)
        mfe = 0.0  # max favorable excursion (bp from entry, always positive or 0)

        for i in range(entry_idx + 1, max_idx + 1):
            if is_invalid[i]:
                continue
            h = high[i] if not np.isnan(high[i]) else close[i]
            l = low[i] if not np.isnan(low[i]) else close[i]
            c_px = close[i]

            if entry_side == "long":
                # Favorable = price going up
                cur_mfe = (h / entry_px - 1) * 10000
                cur_mae = (l / entry_px - 1) * 10000
                mfe = max(mfe, cur_mfe)
                mae = min(mae, cur_mae)

                # Check SL first (conservative: SL before TP on same bar)
                if l <= sl_px:
                    exit_reason = "SL"
                    exit_px = sl_px
                    exit_idx = i
                    break
                if h >= tp_px:
                    exit_reason = "TP"
                    exit_px = tp_px
                    exit_idx = i
                    break
            else:  # short
                # Favorable = price going down
                cur_mfe = (1 - l / entry_px) * 10000
                cur_mae = (1 - h / entry_px) * 10000  # negative when price goes up
                mfe = max(mfe, cur_mfe)
                mae = min(mae, cur_mae)

                # Check SL first
                if h >= sl_px:
                    exit_reason = "SL"
                    exit_px = sl_px
                    exit_idx = i
                    break
                if l <= tp_px:
                    exit_reason = "TP"
                    exit_px = tp_px
                    exit_idx = i
                    break

        # Compute gross PnL
        if entry_side == "long":
            gross_pnl_bp = (exit_px / entry_px - 1) * 10000
        else:
            gross_pnl_bp = (1 - exit_px / entry_px) * 10000

        # If double trigger, add the loss from opposite leg
        double_loss_bp = 0.0
        if double_trigger and opposite_px is not None:
            # The opposite position was also opened and needs to be closed
            # Assume it's closed immediately at market (worst case: at the entry bar close)
            double_close_px = close[entry_idx]
            if entry_side == "long":
                # Opposite was a short at sell_stop, closed at close[entry_idx]
                double_loss_bp = (1 - double_close_px / opposite_px) * 10000
            else:
                # Opposite was a long at buy_stop, closed at close[entry_idx]
                double_loss_bp = (double_close_px / opposite_px - 1) * 10000

        t_entry = ts[entry_idx]
        t_exit = ts[exit_idx]
        last_entry_time[sym] = t_entry

        trades.append({
            "symbol": sym,
            "t_signal": t_signal,
            "t_entry": t_entry,
            "side": entry_side,
            "P0": p0,
            "X_atr_mult": a,
            "X_bp": a * atr_ret * 10000,
            "entry_px": entry_px,
            "tp_px": tp_px,
            "sl_px": sl_px,
            "exit_reason": "DOUBLE" if double_trigger else exit_reason,
            "exit_px": exit_px,
            "gross_pnl_bp": gross_pnl_bp,
            "double_loss_bp": double_loss_bp,
            "MAE_bp": mae,
            "MFE_bp": mfe,
            "double_trigger": double_trigger,
            "time_to_entry_min": (t_entry - t_signal).total_seconds() / 60,
            "time_in_trade_min": (t_exit - t_entry).total_seconds() / 60,
            "atr_at_signal": atr_val,
        })

    return trades


# ---------------------------------------------------------------------------
# §3: Fee / slippage computation
# ---------------------------------------------------------------------------

def apply_fees(trades_df: pd.DataFrame, fee_model: str,
               slip_entry_bp: float, slip_exit_bp: float) -> pd.DataFrame:
    """Apply fee model and slippage to trade log. Returns copy with net columns."""
    df = trades_df.copy()

    # Entry is always taker (stop order)
    entry_fee = FEE_TAKER_BP

    # Exit fee depends on model and exit reason
    if fee_model == "M":
        # TP = maker, SL/TIME/DOUBLE = taker
        exit_fee = df["exit_reason"].map({
            "TP": FEE_MAKER_BP,
            "SL": FEE_TAKER_BP,
            "TIME": FEE_TAKER_BP,
            "DOUBLE": FEE_TAKER_BP,
            "NO_TRIGGER": 0.0,
        }).fillna(FEE_TAKER_BP)
    else:  # "T" — all taker
        exit_fee = df["exit_reason"].map({
            "NO_TRIGGER": 0.0,
        }).fillna(FEE_TAKER_BP)

    # Slippage: entry slip works against us, exit SL slip works against us
    # For TP, slippage is less (limit order), but we model it anyway for stress
    entry_slip = slip_entry_bp
    # SL slippage is typically worse
    exit_slip = df["exit_reason"].map({
        "TP": slip_exit_bp * 0.5 if fee_model == "M" else slip_exit_bp,
        "SL": slip_exit_bp,
        "TIME": slip_exit_bp,
        "DOUBLE": slip_exit_bp,
        "NO_TRIGGER": 0.0,
    }).fillna(slip_exit_bp)

    df["fees_bp"] = entry_fee + exit_fee
    df["slippage_bp"] = entry_slip + exit_slip

    # Double trigger: extra fees for the opposite leg (entry + immediate close)
    double_mask = df["double_trigger"] == True
    df.loc[double_mask, "fees_bp"] += FEE_TAKER_BP * 2  # entry + close of opposite
    df.loc[double_mask, "slippage_bp"] += slip_entry_bp + slip_exit_bp

    # Net PnL
    has_trade = df["exit_reason"] != "NO_TRIGGER"
    df["net_pnl_bp"] = 0.0
    df.loc[has_trade, "net_pnl_bp"] = (
        df.loc[has_trade, "gross_pnl_bp"]
        + df.loc[has_trade, "double_loss_bp"].fillna(0)
        - df.loc[has_trade, "fees_bp"]
        - df.loc[has_trade, "slippage_bp"]
    )

    return df


# ---------------------------------------------------------------------------
# §4: Summary statistics per config
# ---------------------------------------------------------------------------

def compute_summary(trades_df: pd.DataFrame, config: dict) -> dict:
    """Compute summary statistics for a config's trade log."""
    actual = trades_df[trades_df["exit_reason"] != "NO_TRIGGER"].copy()
    n_signals = len(trades_df)
    n_trades = len(actual)

    if n_trades == 0:
        return {**config, "n_signals": n_signals, "n_trades": 0}

    net = actual["net_pnl_bp"].values
    gross = actual["gross_pnl_bp"].values

    # Exit reason distribution
    reasons = actual["exit_reason"].value_counts()
    n_tp = reasons.get("TP", 0)
    n_sl = reasons.get("SL", 0)
    n_time = reasons.get("TIME", 0)
    n_double = reasons.get("DOUBLE", 0)

    # Weekly stability
    actual["_week"] = actual["t_signal"].dt.isocalendar().week.values
    weekly = actual.groupby("_week")["net_pnl_bp"].sum()
    weeks_positive = (weekly > 0).sum()
    weeks_total = len(weekly)

    # Profit factor
    wins = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    pf = wins / losses if losses > 0 else np.inf

    # Max drawdown (cumulative)
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    return {
        **config,
        "n_signals": n_signals,
        "n_trades": n_trades,
        "trigger_rate": n_trades / max(n_signals, 1),
        "net_mean_bp": np.mean(net),
        "net_median_bp": np.median(net),
        "net_total_bp": np.sum(net),
        "gross_mean_bp": np.mean(gross),
        "pf": pf,
        "hit_rate_tp": n_tp / n_trades,
        "pct_sl": n_sl / n_trades,
        "pct_time": n_time / n_trades,
        "pct_double": n_double / n_trades,
        "max_dd_bp": max_dd,
        "p5_net_bp": np.percentile(net, 5),
        "p95_net_bp": np.percentile(net, 95),
        "weeks_positive": weeks_positive,
        "weeks_total": weeks_total,
        "mean_time_to_entry_min": actual["time_to_entry_min"].mean(),
        "mean_time_in_trade_min": actual["time_in_trade_min"].mean(),
        "mean_MAE_bp": actual["MAE_bp"].mean(),
        "mean_MFE_bp": actual["MFE_bp"].mean(),
    }


# ---------------------------------------------------------------------------
# §5: Walk-forward splits
# ---------------------------------------------------------------------------

def split_signals(signal_times, split="fwd"):
    """Split signal times into train/test with purge."""
    if split == "fwd":
        train = signal_times[signal_times <= PURGE_START_FWD]
        test = signal_times[signal_times >= PURGE_END_FWD]
    else:  # reverse
        train = signal_times[signal_times >= PURGE_END_REV]
        test = signal_times[signal_times <= PURGE_START_REV]
    return train, test


# ---------------------------------------------------------------------------
# §6: Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.monotonic()
    print("=" * 70)
    print("XS-7 — Bracket / Convex Execution on S07")
    print("=" * 70)
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Bracket a: {A_GRID}, TP b: {B_GRID}, SL c: {C_GRID}")
    print(f"Cancel delay: {CANCEL_DELAY_GRID}s")
    print(f"Fee: taker={FEE_TAKER_BP}bp, maker={FEE_MAKER_BP}bp")
    print(f"Slippage: {SLIP_GRID}bp")
    print(f"Cooldown: {COOLDOWN_MIN}min")
    print()

    symbols = discover_symbols()
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"Symbols: {len(symbols)}, Grid: {len(grid_1m)} 1m bars")
    print()

    # --- Phase 1: Load data and detect signals for all symbols ---
    sym_data = {}
    total_signals = 0
    for i, sym in enumerate(symbols, 1):
        raw = load_symbol(sym)
        df_1m, signal_times = build_symbol_data(sym, raw, grid_1m)
        if df_1m is None:
            continue
        sym_data[sym] = (df_1m, signal_times)
        total_signals += len(signal_times)
        if i % 10 == 0 or i == len(symbols):
            print(f"  Loaded {i}/{len(symbols)} symbols, {total_signals} S07 signals so far")

    print(f"\nTotal: {len(sym_data)} symbols with data, {total_signals} S07 signals")
    print()

    # --- Phase 2: Run bracket simulation for all (a, cancel_delay) combos ---
    # We separate the bracket sim (which depends on a and cancel_delay)
    # from the fee/slip application (which we do later on the trade log)

    print("Phase 2: Simulating brackets...")
    # Key: (a, cancel_delay) -> list of trade dicts across all symbols
    all_raw_trades = {}

    bracket_configs = list(product(A_GRID, CANCEL_DELAY_GRID))
    for cfg_idx, (a_val, cd_val) in enumerate(bracket_configs, 1):
        t0 = time.monotonic()
        trades = []
        for sym, (df_1m, signal_times) in sym_data.items():
            # Run for all b/c values at once? No — b/c affect exit only.
            # We simulate with max b and max c, then compute exits for each b/c from the trade path.
            # Actually, b/c change TP/SL levels, so we need separate simulations.
            # But the entry trigger is the same for all b/c — only exit differs.
            # Optimization: simulate entry once, then apply different exits.

            # For now, simulate with each (b, c) combo
            for b_val in B_GRID:
                for c_val in C_GRID:
                    sym_trades = simulate_bracket_trades(
                        df_1m, signal_times, sym,
                        a=a_val, b=b_val, c=c_val,
                        cancel_delay_s=cd_val,
                    )
                    for t in sym_trades:
                        t["a"] = a_val
                        t["b"] = b_val
                        t["c"] = c_val
                        t["cancel_delay_s"] = cd_val
                    trades.extend(sym_trades)

        all_raw_trades[(a_val, cd_val)] = trades
        elapsed = time.monotonic() - t0
        n_actual = sum(1 for t in trades if t["exit_reason"] != "NO_TRIGGER")
        print(f"  [{cfg_idx}/{len(bracket_configs)}] a={a_val}, cancel={cd_val}s: "
              f"{len(trades)} signals, {n_actual} triggered, {elapsed:.1f}s")

    # --- Phase 3: Flatten trades, apply fees, compute summaries ---
    print("\nPhase 3: Applying fees and computing summaries...")

    all_trades = []
    for trades in all_raw_trades.values():
        all_trades.extend(trades)

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        print("No trades generated!")
        return

    print(f"  Total raw trade records: {len(df_trades)}")

    # Apply walk-forward split labels
    purge_end_fwd = PURGE_END_FWD
    purge_start_fwd = PURGE_START_FWD
    df_trades["split"] = "purge"
    df_trades.loc[df_trades["t_signal"] <= purge_start_fwd, "split"] = "train_fwd"
    df_trades.loc[df_trades["t_signal"] >= purge_end_fwd, "split"] = "test_fwd"
    # Reverse split
    df_trades.loc[df_trades["t_signal"] >= PURGE_END_REV, "split_rev"] = "train_rev"
    df_trades.loc[df_trades["t_signal"] <= PURGE_START_REV, "split_rev"] = "test_rev"

    # Compute summaries for all configs
    summaries = []
    fee_models = ["M", "T"]
    slip_configs = [(0, 0), (5, 5), (10, 10)]

    config_count = 0
    total_configs = (len(A_GRID) * len(B_GRID) * len(C_GRID) *
                     len(CANCEL_DELAY_GRID) * len(fee_models) * len(slip_configs) * 2)

    for a_val in A_GRID:
        for b_val in B_GRID:
            for c_val in C_GRID:
                for cd_val in CANCEL_DELAY_GRID:
                    mask = ((df_trades["a"] == a_val) &
                            (df_trades["b"] == b_val) &
                            (df_trades["c"] == c_val) &
                            (df_trades["cancel_delay_s"] == cd_val))
                    subset = df_trades[mask]

                    for fm in fee_models:
                        for slip_e, slip_x in slip_configs:
                            # Apply fees
                            with_fees = apply_fees(subset, fm, slip_e, slip_x)

                            for split_name, split_col, split_val in [
                                ("test_fwd", "split", "test_fwd"),
                                ("test_rev", "split_rev", "test_rev"),
                            ]:
                                if split_col not in with_fees.columns:
                                    continue
                                split_df = with_fees[with_fees[split_col] == split_val]
                                config = {
                                    "a": a_val, "b": b_val, "c": c_val,
                                    "cancel_delay_s": cd_val,
                                    "fee_model": fm,
                                    "slip_entry_bp": slip_e,
                                    "slip_exit_bp": slip_x,
                                    "split": split_name,
                                }
                                summary = compute_summary(split_df, config)
                                summaries.append(summary)
                                config_count += 1

    print(f"  Computed {config_count} config summaries")

    # --- Phase 4: Save outputs ---
    print("\nPhase 4: Saving outputs...")

    # Save full trade log (with default fee model M, slip 0)
    trades_with_fees = apply_fees(df_trades, "M", 0, 0)
    trades_path = OUTPUT_DIR / "xs7_trades.csv"
    trades_with_fees.to_csv(trades_path, index=False, float_format="%.4f")
    print(f"  Trade log: {trades_path} ({len(trades_with_fees)} rows)")

    # Save summary report
    df_summary = pd.DataFrame(summaries)
    report_path = OUTPUT_DIR / "xs7_report.csv"
    df_summary.to_csv(report_path, index=False, float_format="%.4f")
    print(f"  Summary report: {report_path} ({len(df_summary)} rows)")

    # --- Phase 5: Print key results ---
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    # Best configs on test_fwd (Feb) with realistic fees
    fwd_real = df_summary[
        (df_summary["split"] == "test_fwd") &
        (df_summary["fee_model"] == "M") &
        (df_summary["slip_entry_bp"] == 5) &
        (df_summary["n_trades"] >= 5)
    ].sort_values("net_mean_bp", ascending=False)

    print("\n## Best configs (test Feb, fee=M, slip=5bp, n>=5 trades):")
    if len(fwd_real) > 0:
        cols = ["a", "b", "c", "cancel_delay_s", "n_trades", "trigger_rate",
                "net_mean_bp", "net_median_bp", "pf", "hit_rate_tp",
                "pct_sl", "pct_time", "pct_double", "max_dd_bp",
                "p5_net_bp", "weeks_positive", "weeks_total"]
        print(fwd_real.head(15)[cols].to_string(index=False, float_format="%.2f"))
    else:
        print("  No configs with enough trades.")

    # Worst-case stress test
    fwd_stress = df_summary[
        (df_summary["split"] == "test_fwd") &
        (df_summary["fee_model"] == "T") &
        (df_summary["slip_entry_bp"] == 10) &
        (df_summary["n_trades"] >= 5)
    ].sort_values("net_mean_bp", ascending=False)

    print("\n## Stress test (test Feb, fee=T all-taker, slip=10bp):")
    if len(fwd_stress) > 0:
        print(fwd_stress.head(10)[cols].to_string(index=False, float_format="%.2f"))

    # Double-trigger rates
    print("\n## Double-trigger rates by cancel_delay:")
    for cd in CANCEL_DELAY_GRID:
        subset = df_summary[
            (df_summary["split"] == "test_fwd") &
            (df_summary["cancel_delay_s"] == cd) &
            (df_summary["fee_model"] == "M") &
            (df_summary["slip_entry_bp"] == 0) &
            (df_summary["n_trades"] >= 3)
        ]
        if len(subset) > 0:
            mean_double = subset["pct_double"].mean()
            print(f"  cancel_delay={cd}s: mean DOUBLE rate = {mean_double:.1%}")

    # Tradeability diagnostics
    print("\n## Tradeability diagnostics (all trades, fee=M, slip=0):")
    actual = trades_with_fees[trades_with_fees["exit_reason"] != "NO_TRIGGER"]
    if len(actual) > 0:
        print(f"  Total triggered trades: {len(actual)}")
        print(f"  Mean time to entry: {actual['time_to_entry_min'].mean():.0f} min "
              f"({actual['time_to_entry_min'].mean()/60:.1f}h)")
        print(f"  Median time to entry: {actual['time_to_entry_min'].median():.0f} min")
        print(f"  Mean time in trade: {actual['time_in_trade_min'].mean():.0f} min "
              f"({actual['time_in_trade_min'].mean()/60:.1f}h)")

        # Time to entry distribution
        tte = actual["time_to_entry_min"].values
        print(f"  Entry in 0-1h: {(tte < 60).mean():.1%}")
        print(f"  Entry in 1-6h: {((tte >= 60) & (tte < 360)).mean():.1%}")
        print(f"  Entry in 6-12h: {((tte >= 360) & (tte < 720)).mean():.1%}")
        print(f"  Entry in 12-24h: {(tte >= 720).mean():.1%}")

    # GO/NO-GO
    print("\n" + "=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    if len(fwd_real) > 0:
        best = fwd_real.iloc[0]
        print(f"\nBest config: a={best['a']}, b={best['b']}, c={best['c']}, "
              f"cancel={best['cancel_delay_s']}s")
        print(f"  Net mean: {best['net_mean_bp']:.1f} bp/trade")
        print(f"  N trades (Feb): {best['n_trades']:.0f}")
        print(f"  PF: {best['pf']:.2f}")
        print(f"  Hit rate (TP): {best['hit_rate_tp']:.1%}")
        print(f"  DOUBLE rate: {best['pct_double']:.1%}")
        print(f"  Max DD: {best['max_dd_bp']:.0f} bp")
        print(f"  p5 net: {best['p5_net_bp']:.0f} bp")
        print(f"  Weekly: {best['weeks_positive']:.0f}/{best['weeks_total']:.0f}")

        go = (best["net_mean_bp"] > 0 and
              best["p5_net_bp"] > -300 and
              best["pct_double"] < 0.10)

        # Check reverse split too
        rev_match = df_summary[
            (df_summary["split"] == "test_rev") &
            (df_summary["a"] == best["a"]) &
            (df_summary["b"] == best["b"]) &
            (df_summary["c"] == best["c"]) &
            (df_summary["cancel_delay_s"] == best["cancel_delay_s"]) &
            (df_summary["fee_model"] == "M") &
            (df_summary["slip_entry_bp"] == 5)
        ]
        if len(rev_match) > 0:
            rev = rev_match.iloc[0]
            print(f"\n  Reverse split (test Jan): net_mean={rev['net_mean_bp']:.1f}bp, "
                  f"n={rev['n_trades']:.0f}, PF={rev['pf']:.2f}")
            go = go and rev["net_mean_bp"] > 0

        print(f"\n  >>> {'GO ✅' if go else 'NO-GO ❌'} <<<")
    else:
        print("\n  >>> NO-GO ❌ (insufficient trades) <<<")

    # Generate findings
    generate_findings(df_summary, trades_with_fees)

    elapsed = time.monotonic() - t_start
    print(f"\nXS-7 done in {elapsed:.0f}s ({elapsed/60:.1f}min)")


# ---------------------------------------------------------------------------
# §7: Generate findings markdown
# ---------------------------------------------------------------------------

def generate_findings(df_summary: pd.DataFrame, trades_df: pd.DataFrame):
    """Write FINDINGS_xs7_bracket.md."""
    lines = [
        "# XS-7 — Bracket / Convex Execution on S07",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Best configs
    fwd = df_summary[
        (df_summary["split"] == "test_fwd") &
        (df_summary["fee_model"] == "M") &
        (df_summary["slip_entry_bp"] == 5) &
        (df_summary["n_trades"] >= 3)
    ].sort_values("net_mean_bp", ascending=False)

    lines.append("## Best Configs (Test Feb, fee=M, slip=5bp)")
    lines.append("")
    if len(fwd) > 0:
        lines.append("| a | b | c | cancel_s | n | net_mean | PF | TP% | SL% | TIME% | DBL% | max_DD | p5 | wk+/tot |")
        lines.append("|---|---|---|----------|---|----------|----|----|-----|------|------|--------|----|----|")
        for _, r in fwd.head(20).iterrows():
            lines.append(
                f"| {r['a']:.1f} | {r['b']:.0f} | {r['c']:.1f} | {r['cancel_delay_s']:.0f} | "
                f"{r['n_trades']:.0f} | {r.get('net_mean_bp',0):.1f} | {r.get('pf',0):.2f} | "
                f"{r.get('hit_rate_tp',0):.0%} | {r.get('pct_sl',0):.0%} | "
                f"{r.get('pct_time',0):.0%} | {r.get('pct_double',0):.0%} | "
                f"{r.get('max_dd_bp',0):.0f} | {r.get('p5_net_bp',0):.0f} | "
                f"{r.get('weeks_positive',0):.0f}/{r.get('weeks_total',0):.0f} |"
            )
    else:
        lines.append("No configs with enough trades.")
    lines.append("")

    # Double-trigger analysis
    lines.append("## Double-Trigger Analysis")
    lines.append("")
    for cd in CANCEL_DELAY_GRID:
        subset = fwd[fwd["cancel_delay_s"] == cd] if len(fwd) > 0 else pd.DataFrame()
        if len(subset) > 0:
            lines.append(f"- cancel_delay={cd}s: mean DOUBLE% = {subset['pct_double'].mean():.1%}")
    lines.append("")

    # Tradeability
    actual = trades_df[trades_df["exit_reason"] != "NO_TRIGGER"]
    if len(actual) > 0:
        lines.append("## Tradeability Diagnostics")
        lines.append("")
        tte = actual["time_to_entry_min"]
        lines.append(f"- Mean time to entry: {tte.mean():.0f}min ({tte.mean()/60:.1f}h)")
        lines.append(f"- Median time to entry: {tte.median():.0f}min")
        lines.append(f"- Entry in 0-1h: {(tte < 60).mean():.1%}")
        lines.append(f"- Entry in 1-6h: {((tte >= 60) & (tte < 360)).mean():.1%}")
        lines.append(f"- Entry in 6-12h: {((tte >= 360) & (tte < 720)).mean():.1%}")
        lines.append(f"- Entry in 12-24h: {(tte >= 720).mean():.1%}")
        lines.append("")

    md_path = OUTPUT_DIR / "FINDINGS_xs7_bracket.md"
    md_path.write_text("\n".join(lines))
    print(f"\nFindings written to {md_path}")


if __name__ == "__main__":
    main()
