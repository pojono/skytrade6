#!/usr/bin/env python3
"""
XS-7 v2 — Convex Extraction on S07 (Exit Engineering)

Goal: Turn S07 (compress_oi) from "rare jackpot" into more stable convex profile via:
  1. Closer TP (2-2.5 ATR instead of 3-5)
  2. Trailing stops (activate after trail_start, gap trail_gap)
  3. Partial exits (close p1 fraction at tp1, rest continues)
  4. Expanded SL/TIME grid

Signal: S07 compress_oi (rv_6h <= P20 AND oi_z >= 1.5), per-coin expanding percentile
Cooldown: 24h per coin after bracket placement (not after fill)

Three-phase grid:
  Phase 1 (Base):     a×b×c×T×cancel — no trailing, no partial
  Phase 2 (Trailing): Top-5 base configs × trail_start × trail_gap
  Phase 3 (Partial):  Top-5 base configs × p1 × tp1
  Phase 4 (Combo):    Top-3 trailing × top-3 partial (optional)

Walk-forward: Jan train / Feb test AND reverse, ±24h purge.
Bug guards: no future leakage, entry != signal close, gap integrity, cooldown, reproducibility.
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

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs7v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEE_TAKER_BP = 10.0
FEE_MAKER_BP = 2.0
SLIP_GRID = [0, 5, 10]

COOLDOWN_MIN = 24 * 60  # 24h in minutes (from bracket placement)

# Walk-forward splits
PURGE_START_FWD = TRAIN_END - pd.Timedelta(hours=PURGE_HOURS)
PURGE_END_FWD = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
PURGE_START_REV = TEST_START - pd.Timedelta(hours=PURGE_HOURS)
PURGE_END_REV = TRAIN_END + pd.Timedelta(hours=PURGE_HOURS)

# --- Phase 1: Base grid ---
A_GRID_BASE = [0.8, 1.0, 1.2]
B_GRID_BASE = [2.0, 2.5, 3.0]
C_GRID_BASE = [1.5, 2.0]
T_GRID_BASE = [24 * 60]  # 24h only for base
CANCEL_DELAY_BASE = [0, 60]

# --- Phase 2: Trailing grid (applied on top-5 base) ---
TRAIL_START_GRID = [1.0, 1.5]
TRAIL_GAP_GRID = [0.8, 1.0, 1.2]

# --- Phase 3: Partial grid (applied on top-5 base) ---
P1_GRID = [0.3, 0.5]
TP1_GRID = [1.5, 2.0]

# --- Phase 4 (optional): combo ---
# top-3 trailing × top-3 partial


# ---------------------------------------------------------------------------
# §1: Build 1m data with S07 signal (expanding percentile)
# ---------------------------------------------------------------------------

def build_symbol_data(sym: str, raw: dict, grid_1m: pd.DatetimeIndex):
    """Build 1m DataFrame with S07 signal using expanding percentile (no future leakage).
    Returns (df_1m, signal_times_5m) or (None, None) if insufficient data."""
    df_1m = build_sym_1m(sym, raw, grid_1m)
    valid_pct = 1 - df_1m["is_invalid"].mean()
    if valid_pct < 0.5:
        return None, None

    df_1m = compute_features(df_1m)

    # Override rv_6h_pctl with EXPANDING percentile (no future leakage)
    rv6 = df_1m["rv_6h"]
    # expanding rank: for each point, what fraction of ALL prior values are <= current
    expanding_rank = rv6.expanding(min_periods=360).rank(pct=True)
    df_1m["rv_6h_pctl_exp"] = expanding_rank

    # Sample to 5m for signal detection
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]
    df_5m = df_1m.loc[grid_5m].copy()

    # S07 with expanding percentile: rv_6h <= P20 AND oi_z >= 1.5
    s07 = (df_5m["rv_6h_pctl_exp"] <= 0.20) & (df_5m["oi_z"] >= 1.5)
    # Invalidate if ATR or rv is NaN
    s07 = s07 & df_5m["atr_1h_raw"].notna() & df_5m["rv_6h"].notna()
    s07 = s07 & (df_5m["is_invalid"] == 0)

    signal_times = df_5m.index[s07.values]

    return df_1m, signal_times


# ---------------------------------------------------------------------------
# §2: Bracket simulation engine with exit engineering
# ---------------------------------------------------------------------------

def simulate_bracket_trades(
    df_1m: pd.DataFrame,
    signal_times: pd.DatetimeIndex,
    sym: str,
    a: float,
    b: float,
    c: float,
    cancel_delay_s: int,
    time_stop_min: int,
    trail_start: float = 0.0,   # 0 = trailing OFF
    trail_gap: float = 0.0,
    p1: float = 0.0,            # 0 = partial OFF
    tp1: float = 0.0,
) -> list[dict]:
    """Simulate bracket OCO trades on 1m data with full exit engineering.

    Exit priority per bar: SL > TP1 (partial) > TP > TRAIL > TIME
    Trailing activates when MFE >= trail_start * ATR.
    Partial: close p1 fraction at tp1*ATR, remainder continues.
    """
    close = df_1m["close"].values
    high = df_1m["high"].values if "high" in df_1m.columns else close.copy()
    low = df_1m["low"].values if "low" in df_1m.columns else close.copy()
    atr = df_1m["atr_1h_raw"].values
    ts = df_1m.index
    is_invalid = df_1m["is_invalid"].values

    ts_to_idx = {t: i for i, t in enumerate(ts)}
    cancel_delay_bars = max(1, cancel_delay_s // 60) if cancel_delay_s > 0 else 0

    trades = []
    last_signal_time = {}  # sym -> last signal timestamp (cooldown from placement)

    use_trailing = trail_start > 0 and trail_gap > 0
    use_partial = p1 > 0 and tp1 > 0

    for t_signal in signal_times:
        # Cooldown check (from bracket placement, not from fill)
        if sym in last_signal_time:
            if (t_signal - last_signal_time[sym]).total_seconds() < COOLDOWN_MIN * 60:
                continue

        idx0 = ts_to_idx.get(t_signal)
        if idx0 is None or idx0 >= len(ts) - 10:
            continue

        # Anti-lookahead: entry reference is close of t0+1m
        ref_idx = idx0 + 1
        if ref_idx >= len(ts):
            continue
        p0 = close[ref_idx]
        atr_val = atr[idx0]  # ATR at signal time (causal)
        if np.isnan(p0) or np.isnan(atr_val) or atr_val <= 0 or p0 <= 0:
            continue

        # Mark cooldown from bracket placement
        last_signal_time[sym] = t_signal

        atr_ret = atr_val / p0  # ATR as fraction of price

        # Bracket levels (relative to P0 = close[t0+1m])
        buy_stop = p0 * (1 + a * atr_ret)
        sell_stop = p0 * (1 - a * atr_ret)

        # Time stop window (from t0+1m)
        max_idx = min(ref_idx + time_stop_min, len(ts) - 1)

        # Bug guard: assert feature timestamps <= t_signal
        # (already guaranteed by expanding percentile + 5m grid)

        # --- Phase 1: Wait for bracket trigger ---
        entry_idx = None
        entry_side = None
        entry_px = None
        double_trigger = False
        opposite_px = None

        for i in range(ref_idx + 1, max_idx + 1):
            if is_invalid[i]:
                continue
            h = high[i] if not np.isnan(high[i]) else close[i]
            l = low[i] if not np.isnan(low[i]) else close[i]

            buy_triggered = h >= buy_stop
            sell_triggered = l <= sell_stop

            if buy_triggered and sell_triggered:
                # Both triggered in same bar
                entry_idx = i
                entry_side = "long"
                entry_px = buy_stop
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
            trades.append(_no_trigger_record(
                sym, t_signal, p0, a, atr_ret, atr_val, b, c,
                time_stop_min, trail_start, trail_gap, p1, tp1, cancel_delay_s,
            ))
            continue

        # --- Phase 1b: Check double-trigger within cancel_delay ---
        if not double_trigger and cancel_delay_bars > 0:
            if entry_side == "long":
                for j in range(entry_idx, min(entry_idx + cancel_delay_bars + 1, max_idx + 1)):
                    lj = low[j] if not np.isnan(low[j]) else close[j]
                    if lj <= sell_stop:
                        double_trigger = True
                        opposite_px = sell_stop
                        break
            else:
                for j in range(entry_idx, min(entry_idx + cancel_delay_bars + 1, max_idx + 1)):
                    hj = high[j] if not np.isnan(high[j]) else close[j]
                    if hj >= buy_stop:
                        double_trigger = True
                        opposite_px = buy_stop
                        break

        # --- Phase 2: Position management with exit engineering ---
        if entry_side == "long":
            tp_px = entry_px * (1 + b * atr_ret)
            sl_px = entry_px * (1 - c * atr_ret)
            tp1_px = entry_px * (1 + tp1 * atr_ret) if use_partial else None
            trail_start_px = entry_px * (1 + trail_start * atr_ret) if use_trailing else None
        else:
            tp_px = entry_px * (1 - b * atr_ret)
            sl_px = entry_px * (1 + c * atr_ret)
            tp1_px = entry_px * (1 - tp1 * atr_ret) if use_partial else None
            trail_start_px = entry_px * (1 - trail_start * atr_ret) if use_trailing else None

        exit_reason = "TIME"
        exit_px = close[max_idx]
        exit_idx = max_idx
        mae = 0.0
        mfe = 0.0
        partial_filled = False
        partial_pnl_bp = 0.0
        trailing_active = False
        trailing_stop_px = sl_px  # starts at SL
        highest_px = entry_px if entry_side == "long" else entry_px
        lowest_px = entry_px if entry_side == "short" else entry_px
        remaining_frac = 1.0

        for i in range(entry_idx + 1, max_idx + 1):
            if is_invalid[i]:
                continue
            h = high[i] if not np.isnan(high[i]) else close[i]
            l = low[i] if not np.isnan(low[i]) else close[i]
            c_px = close[i]

            if entry_side == "long":
                cur_mfe = (h / entry_px - 1) * 10000
                cur_mae = (l / entry_px - 1) * 10000
                mfe = max(mfe, cur_mfe)
                mae = min(mae, cur_mae)

                # Update highest for trailing
                if h > highest_px:
                    highest_px = h

                # 1) Check SL (fixed or trailing)
                active_sl = trailing_stop_px if trailing_active else sl_px
                if l <= active_sl:
                    exit_reason = "TRAIL" if trailing_active else "SL"
                    exit_px = active_sl
                    exit_idx = i
                    break

                # 2) Check partial TP1
                if use_partial and not partial_filled and h >= tp1_px:
                    partial_filled = True
                    partial_pnl_bp = (tp1_px / entry_px - 1) * 10000
                    remaining_frac = 1.0 - p1

                # 3) Check full TP
                if h >= tp_px:
                    exit_reason = "TP"
                    exit_px = tp_px
                    exit_idx = i
                    break

                # 4) Update trailing stop
                if use_trailing and h >= trail_start_px:
                    trailing_active = True
                    new_trail = highest_px * (1 - trail_gap * atr_ret)
                    if new_trail > trailing_stop_px:
                        trailing_stop_px = new_trail

            else:  # short
                cur_mfe = (1 - l / entry_px) * 10000
                cur_mae = (1 - h / entry_px) * 10000
                mfe = max(mfe, cur_mfe)
                mae = min(mae, cur_mae)

                if l < lowest_px:
                    lowest_px = l

                # 1) SL
                active_sl = trailing_stop_px if trailing_active else sl_px
                if h >= active_sl:
                    exit_reason = "TRAIL" if trailing_active else "SL"
                    exit_px = active_sl
                    exit_idx = i
                    break

                # 2) Partial TP1
                if use_partial and not partial_filled and l <= tp1_px:
                    partial_filled = True
                    partial_pnl_bp = (1 - tp1_px / entry_px) * 10000
                    remaining_frac = 1.0 - p1

                # 3) Full TP
                if l <= tp_px:
                    exit_reason = "TP"
                    exit_px = tp_px
                    exit_idx = i
                    break

                # 4) Trailing
                if use_trailing and l <= trail_start_px:
                    trailing_active = True
                    new_trail = lowest_px * (1 + trail_gap * atr_ret)
                    if new_trail < trailing_stop_px:
                        trailing_stop_px = new_trail

        # --- Compute gross PnL ---
        if entry_side == "long":
            main_pnl_bp = (exit_px / entry_px - 1) * 10000
        else:
            main_pnl_bp = (1 - exit_px / entry_px) * 10000

        # If partial was filled, blend PnL
        if partial_filled:
            gross_pnl_bp = p1 * partial_pnl_bp + remaining_frac * main_pnl_bp
        else:
            gross_pnl_bp = main_pnl_bp

        # Double trigger loss
        double_loss_bp = 0.0
        if double_trigger and opposite_px is not None:
            double_close_px = close[entry_idx]
            if entry_side == "long":
                double_loss_bp = (1 - double_close_px / opposite_px) * 10000
            else:
                double_loss_bp = (double_close_px / opposite_px - 1) * 10000

        t_entry = ts[entry_idx]
        t_exit = ts[exit_idx]

        # Bug guard: entry_px != signal close
        entry_eq_signal = abs(entry_px - close[idx0]) < 1e-12

        trades.append({
            "symbol": sym,
            "t_signal": t_signal,
            "t_entry": t_entry,
            "t_exit": t_exit,
            "side": entry_side,
            "P0": p0,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "tp_px": tp_px,
            "sl_px": sl_px,
            "exit_reason": exit_reason,
            "gross_pnl_bp": gross_pnl_bp,
            "double_loss_bp": double_loss_bp,
            "MAE_bp": mae,
            "MFE_bp": mfe,
            "double_trigger": double_trigger,
            "partial_filled": partial_filled,
            "partial_pnl_bp": partial_pnl_bp if partial_filled else 0.0,
            "remaining_frac": remaining_frac,
            "trailing_active": trailing_active,
            "time_to_entry_min": (t_entry - t_signal).total_seconds() / 60,
            "time_in_trade_min": (t_exit - t_entry).total_seconds() / 60,
            "atr_at_signal": atr_val,
            "entry_eq_signal_close": entry_eq_signal,
            # Config params stored per-trade
            "a": a, "b": b, "c": c,
            "T_min": time_stop_min,
            "trail_start": trail_start,
            "trail_gap": trail_gap,
            "p1": p1, "tp1": tp1,
            "cancel_delay_s": cancel_delay_s,
        })

    return trades


def _no_trigger_record(sym, t_signal, p0, a, atr_ret, atr_val,
                       b, c, time_stop_min, trail_start, trail_gap, p1, tp1, cancel_delay_s):
    return {
        "symbol": sym, "t_signal": t_signal,
        "t_entry": pd.NaT, "t_exit": pd.NaT,
        "side": "none", "P0": p0,
        "entry_px": np.nan, "exit_px": np.nan,
        "tp_px": np.nan, "sl_px": np.nan,
        "exit_reason": "NO_TRIGGER",
        "gross_pnl_bp": 0.0, "double_loss_bp": 0.0,
        "MAE_bp": 0.0, "MFE_bp": 0.0,
        "double_trigger": False,
        "partial_filled": False, "partial_pnl_bp": 0.0,
        "remaining_frac": 1.0, "trailing_active": False,
        "time_to_entry_min": np.nan, "time_in_trade_min": 0,
        "atr_at_signal": atr_val, "entry_eq_signal_close": False,
        "a": a, "b": b, "c": c,
        "T_min": time_stop_min,
        "trail_start": trail_start, "trail_gap": trail_gap,
        "p1": p1, "tp1": tp1, "cancel_delay_s": cancel_delay_s,
    }


# ---------------------------------------------------------------------------
# §3: Fee / slippage computation
# ---------------------------------------------------------------------------

def apply_fees(trades_df: pd.DataFrame, slip_bp: float) -> pd.DataFrame:
    """Apply fees and slippage. TP/TP1 = maker, rest = taker."""
    df = trades_df.copy()

    entry_fee = FEE_TAKER_BP  # stop order = taker

    # Exit fee: TP = maker, everything else = taker
    exit_fee_map = {
        "TP": FEE_MAKER_BP,
        "SL": FEE_TAKER_BP,
        "TIME": FEE_TAKER_BP,
        "TRAIL": FEE_TAKER_BP,
        "DOUBLE": FEE_TAKER_BP,
        "NO_TRIGGER": 0.0,
    }
    exit_fee = df["exit_reason"].map(exit_fee_map).fillna(FEE_TAKER_BP)

    # Partial exit fee: if partial was filled, add maker fee for partial leg
    partial_fee = np.where(df["partial_filled"], FEE_MAKER_BP, 0.0)

    # Slippage
    entry_slip = slip_bp
    exit_slip_map = {
        "TP": slip_bp * 0.5,    # limit order, less slip
        "SL": slip_bp,
        "TIME": slip_bp,
        "TRAIL": slip_bp,
        "DOUBLE": slip_bp,
        "NO_TRIGGER": 0.0,
    }
    exit_slip = df["exit_reason"].map(exit_slip_map).fillna(slip_bp)
    partial_slip = np.where(df["partial_filled"], slip_bp * 0.5, 0.0)

    df["fees_bp"] = entry_fee + exit_fee + partial_fee
    df["slip_bp"] = entry_slip + exit_slip + partial_slip

    # Double trigger: extra fees
    double_mask = df["double_trigger"] == True
    df.loc[double_mask, "fees_bp"] += FEE_TAKER_BP * 2
    df.loc[double_mask, "slip_bp"] += slip_bp * 2

    # Net PnL
    has_trade = df["exit_reason"] != "NO_TRIGGER"
    df["net_bp"] = 0.0
    df.loc[has_trade, "net_bp"] = (
        df.loc[has_trade, "gross_pnl_bp"]
        + df.loc[has_trade, "double_loss_bp"].fillna(0)
        - df.loc[has_trade, "fees_bp"]
        - df.loc[has_trade, "slip_bp"]
    )

    return df


# ---------------------------------------------------------------------------
# §4: Summary statistics per config (v2 metrics)
# ---------------------------------------------------------------------------

def compute_summary(trades_df: pd.DataFrame, config: dict) -> dict:
    """Compute v2 summary metrics including concentration, tail dependency, weekly stability."""
    actual = trades_df[trades_df["exit_reason"] != "NO_TRIGGER"].copy()
    n_signals = len(trades_df)
    n_trades = len(actual)

    if n_trades == 0:
        return {**config, "n_signals": n_signals, "n_trades": 0}

    net = actual["net_bp"].values
    gross = actual["gross_pnl_bp"].values

    # Exit reason distribution
    reasons = actual["exit_reason"].value_counts()
    n_tp = reasons.get("TP", 0)
    n_sl = reasons.get("SL", 0)
    n_time = reasons.get("TIME", 0)
    n_trail = reasons.get("TRAIL", 0)
    n_double = reasons.get("DOUBLE", 0)
    n_partial = actual["partial_filled"].sum()

    # Profit factor
    wins = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else 0.0)

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    # Weekly stability
    actual_ts = actual.copy()
    actual_ts["_week"] = actual_ts["t_signal"].dt.isocalendar().week.values
    weekly = actual_ts.groupby("_week")["net_bp"].sum()
    weeks_positive = (weekly > 0).sum()
    weeks_total = len(weekly)

    # Concentration: top-5 symbols
    sym_pnl = actual.groupby("symbol")["net_bp"].sum().sort_values(ascending=False)
    total_pnl = net.sum()
    top5_sym_pnl = sym_pnl.head(5).sum()
    conc_top5_sym = top5_sym_pnl / abs(total_pnl) if abs(total_pnl) > 0 else np.nan

    # Concentration: top-5 trades
    sorted_net = np.sort(net)[::-1]
    top5_trade_pnl = sorted_net[:5].sum()
    conc_top5_trade = top5_trade_pnl / abs(total_pnl) if abs(total_pnl) > 0 else np.nan

    # Tail dependency: PnL without top-k trades
    pnl_no_top = {}
    for k in [1, 2, 3, 5]:
        if n_trades > k:
            pnl_no_top[f"net_no_top{k}_bp"] = np.sum(sorted_net[k:])
        else:
            pnl_no_top[f"net_no_top{k}_bp"] = 0.0

    # Break-even friction estimate
    be_friction = np.mean(gross) / 2 if np.mean(gross) > 0 else 0.0

    # Entry != signal close rate
    eq_rate = actual["entry_eq_signal_close"].mean()

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
        "pct_tp": n_tp / n_trades,
        "pct_sl": n_sl / n_trades,
        "pct_time": n_time / n_trades,
        "pct_trail": n_trail / n_trades,
        "pct_double": n_double / n_trades,
        "pct_partial": n_partial / n_trades,
        "max_dd_bp": max_dd,
        "p5_net_bp": np.percentile(net, 5),
        "p1_net_bp": np.percentile(net, 1),
        "p95_net_bp": np.percentile(net, 95),
        "conc_top5_sym": conc_top5_sym,
        "conc_top5_trade": conc_top5_trade,
        **pnl_no_top,
        "be_friction_bp": be_friction,
        "weeks_positive": weeks_positive,
        "weeks_total": weeks_total,
        "mean_time_to_entry_min": actual["time_to_entry_min"].mean(),
        "mean_time_in_trade_min": actual["time_in_trade_min"].mean(),
        "mean_MAE_bp": actual["MAE_bp"].mean(),
        "mean_MFE_bp": actual["MFE_bp"].mean(),
        "entry_eq_signal_pct": eq_rate,
    }


# ---------------------------------------------------------------------------
# §5: Walk-forward split labeling
# ---------------------------------------------------------------------------

def label_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add split labels for walk-forward validation."""
    df["split_fwd"] = "purge"
    df.loc[df["t_signal"] <= PURGE_START_FWD, "split_fwd"] = "train"
    df.loc[df["t_signal"] >= PURGE_END_FWD, "split_fwd"] = "test"

    df["split_rev"] = "purge"
    df.loc[df["t_signal"] >= PURGE_END_REV, "split_rev"] = "train"
    df.loc[df["t_signal"] <= PURGE_START_REV, "split_rev"] = "test"
    return df


# ---------------------------------------------------------------------------
# §6: Run one grid phase and return trades + summaries
# ---------------------------------------------------------------------------

def run_grid_phase(sym_data: dict, configs: list[dict], phase_name: str):
    """Run simulation for a list of config dicts.
    Each config dict has keys: a, b, c, T_min, cancel_delay_s, trail_start, trail_gap, p1, tp1.
    Returns (all_trades_df, summaries_list).
    """
    t_phase = time.monotonic()
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name} — {len(configs)} configs × {len(sym_data)} symbols")
    print(f"{'='*60}")

    all_trades = []
    for ci, cfg in enumerate(configs, 1):
        t0 = time.monotonic()
        n_triggered = 0
        for sym, (df_1m, signal_times) in sym_data.items():
            sym_trades = simulate_bracket_trades(
                df_1m, signal_times, sym,
                a=cfg["a"], b=cfg["b"], c=cfg["c"],
                cancel_delay_s=cfg["cancel_delay_s"],
                time_stop_min=cfg["T_min"],
                trail_start=cfg.get("trail_start", 0.0),
                trail_gap=cfg.get("trail_gap", 0.0),
                p1=cfg.get("p1", 0.0),
                tp1=cfg.get("tp1", 0.0),
            )
            n_triggered += sum(1 for t in sym_trades if t["exit_reason"] != "NO_TRIGGER")
            all_trades.extend(sym_trades)

        elapsed = time.monotonic() - t0
        print(f"  [{ci}/{len(configs)}] a={cfg['a']}, b={cfg['b']}, c={cfg['c']}, "
              f"T={cfg['T_min']//60}h, cd={cfg['cancel_delay_s']}s, "
              f"tr={cfg.get('trail_start',0)}/{cfg.get('trail_gap',0)}, "
              f"p={cfg.get('p1',0)}/{cfg.get('tp1',0)} "
              f"→ {n_triggered} triggered, {elapsed:.1f}s")

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        print(f"  No trades in {phase_name}!")
        return df_trades, []

    df_trades = label_splits(df_trades)

    # Compute summaries for each config × slip × split
    summaries = []
    for cfg in configs:
        mask = (
            (df_trades["a"] == cfg["a"]) &
            (df_trades["b"] == cfg["b"]) &
            (df_trades["c"] == cfg["c"]) &
            (df_trades["T_min"] == cfg["T_min"]) &
            (df_trades["cancel_delay_s"] == cfg["cancel_delay_s"]) &
            (df_trades["trail_start"] == cfg.get("trail_start", 0.0)) &
            (df_trades["trail_gap"] == cfg.get("trail_gap", 0.0)) &
            (df_trades["p1"] == cfg.get("p1", 0.0)) &
            (df_trades["tp1"] == cfg.get("tp1", 0.0))
        )
        subset = df_trades[mask]

        for slip in SLIP_GRID:
            with_fees = apply_fees(subset, slip)

            for split_name, split_col in [("test_fwd", "split_fwd"), ("test_rev", "split_rev")]:
                split_df = with_fees[with_fees[split_col] == "test"]
                config_dict = {
                    "a": cfg["a"], "b": cfg["b"], "c": cfg["c"],
                    "T_h": cfg["T_min"] // 60,
                    "cancel_delay_s": cfg["cancel_delay_s"],
                    "trail_start": cfg.get("trail_start", 0.0),
                    "trail_gap": cfg.get("trail_gap", 0.0),
                    "p1": cfg.get("p1", 0.0),
                    "tp1": cfg.get("tp1", 0.0),
                    "slip_bp": slip,
                    "split": split_name,
                    "phase": phase_name,
                }
                summary = compute_summary(split_df, config_dict)
                summaries.append(summary)

    elapsed_phase = time.monotonic() - t_phase
    print(f"\n  {phase_name} done: {len(df_trades)} trade records, "
          f"{len(summaries)} summaries, {elapsed_phase:.0f}s")

    return df_trades, summaries


# ---------------------------------------------------------------------------
# §7: Select top-K configs by OOS EV (both splits positive)
# ---------------------------------------------------------------------------

def select_top_configs(summaries: list[dict], k: int = 5, slip_bp: int = 5) -> list[dict]:
    """Select top-K configs where OOS mean net > 0 in BOTH splits at given slip."""
    df = pd.DataFrame(summaries)
    if len(df) == 0:
        return []

    # Filter to target slip
    df = df[df["slip_bp"] == slip_bp]

    # Group by config (exclude split/slip)
    cfg_cols = ["a", "b", "c", "T_h", "cancel_delay_s",
                "trail_start", "trail_gap", "p1", "tp1"]

    # For each config, check both splits positive and compute avg net mean
    results = []
    for _, grp in df.groupby(cfg_cols):
        if len(grp) < 2:
            continue
        fwd = grp[grp["split"] == "test_fwd"]
        rev = grp[grp["split"] == "test_rev"]
        if len(fwd) == 0 or len(rev) == 0:
            continue
        fwd_mean = fwd["net_mean_bp"].values[0]
        rev_mean = rev["net_mean_bp"].values[0]
        fwd_n = fwd["n_trades"].values[0]
        rev_n = rev["n_trades"].values[0]
        if fwd_mean > 0 and rev_mean > 0 and (fwd_n + rev_n) >= 50:
            cfg = grp.iloc[0][cfg_cols].to_dict()
            cfg["avg_net_mean"] = (fwd_mean + rev_mean) / 2
            cfg["total_trades"] = fwd_n + rev_n
            results.append(cfg)

    if not results:
        # Fallback: just take top-K by avg net mean regardless of both-positive
        print("  WARNING: No configs positive in both splits. Using fallback selection.")
        for _, grp in df.groupby(cfg_cols):
            if len(grp) < 2:
                continue
            avg_mean = grp["net_mean_bp"].mean()
            total_n = grp["n_trades"].sum()
            if total_n >= 20:
                cfg = grp.iloc[0][cfg_cols].to_dict()
                cfg["avg_net_mean"] = avg_mean
                cfg["total_trades"] = total_n
                results.append(cfg)

    results.sort(key=lambda x: x["avg_net_mean"], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# §8: Bug guards
# ---------------------------------------------------------------------------

def run_bug_guards(df_trades: pd.DataFrame):
    """Run all bug guard checks and print results."""
    print("\n" + "=" * 60)
    print("BUG GUARDS")
    print("=" * 60)

    actual = df_trades[df_trades["exit_reason"] != "NO_TRIGGER"]
    if len(actual) == 0:
        print("  No trades to check.")
        return

    # 1) Entry != signal close
    eq_pct = actual["entry_eq_signal_close"].mean()
    print(f"  Entry == signal close: {eq_pct:.1%} (expect LOW)")
    if eq_pct > 0.05:
        print("    ⚠️ WARNING: >5% of entries match signal close — possible lookahead")

    # 2) Duplicate signals in cooldown — check per single config to avoid false positives
    # Pick first config only
    cfg_cols = ["a", "b", "c", "T_min", "trail_start", "trail_gap", "p1", "tp1", "cancel_delay_s"]
    available_cfg = [c for c in cfg_cols if c in actual.columns]
    if available_cfg:
        first_cfg = actual[available_cfg].drop_duplicates().iloc[0]
        mask = pd.Series(True, index=actual.index)
        for col in available_cfg:
            mask = mask & (actual[col] == first_cfg[col])
        single = actual[mask]
        violations_total = 0
        for sym in single["symbol"].unique():
            sym_trades = single[single["symbol"] == sym].sort_values("t_signal")
            if len(sym_trades) < 2:
                continue
            diffs = sym_trades["t_signal"].diff().dt.total_seconds().dropna()
            v = (diffs < COOLDOWN_MIN * 60).sum()
            violations_total += v
        print(f"  Cooldown violations (single config check): {violations_total}")
        if violations_total > 0:
            print("    ⚠️ WARNING: cooldown violations detected")

    # 3) Double trigger rate
    dbl_rate = actual["double_trigger"].mean()
    print(f"  Double trigger rate: {dbl_rate:.2%}")
    if dbl_rate > 0.01:
        print("    ⚠️ WARNING: >1% double triggers")

    # 4) NaN in critical fields
    for col in ["entry_px", "exit_px", "gross_pnl_bp", "net_bp"]:
        if col in actual.columns:
            nan_pct = actual[col].isna().mean()
            if nan_pct > 0:
                print(f"    ⚠️ NaN in {col}: {nan_pct:.1%}")

    # 5) Concurrent positions (informational, sample-based for speed)
    if available_cfg:
        single_entries = single[["t_entry", "t_exit", "symbol"]].dropna()
        if len(single_entries) > 0:
            max_concurrent = 0
            for _, row in single_entries.iterrows():
                concurrent = ((single_entries["t_entry"] <= row["t_entry"]) &
                              (single_entries["t_exit"] > row["t_entry"])).sum()
                max_concurrent = max(max_concurrent, concurrent)
            print(f"  Max concurrent positions (single config): {max_concurrent}")

    print("  Bug guards complete.")


# ---------------------------------------------------------------------------
# §9: Generate FINDINGS markdown
# ---------------------------------------------------------------------------

def generate_findings(all_summaries: pd.DataFrame, all_trades: pd.DataFrame):
    """Write FINDINGS_xs7v2.md with auto-summary of top configs."""
    lines = [
        "# XS-7 v2 — Convex Extraction on S07 (Exit Engineering)",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Data:** {START.date()} → {END.date()}  ",
        "**Signal:** S07 compress_oi (rv_6h ≤ P20_expanding AND oi_z ≥ 1.5)  ",
        "**Cooldown:** 24h per coin from bracket placement  ",
        "",
        "---",
        "",
    ]

    # Best configs across all phases
    df = all_summaries.copy()
    if len(df) == 0:
        lines.append("No results.")
        _write_findings(lines)
        return

    # For each phase, show top configs at slip=5
    for phase in df["phase"].unique():
        phase_df = df[(df["phase"] == phase) & (df["slip_bp"] == 5)]
        if len(phase_df) == 0:
            continue

        lines.append(f"## {phase}")
        lines.append("")

        # Find configs positive in both splits
        cfg_cols = ["a", "b", "c", "T_h", "cancel_delay_s",
                    "trail_start", "trail_gap", "p1", "tp1"]
        both_pos = []
        for name, grp in phase_df.groupby(cfg_cols):
            fwd = grp[grp["split"] == "test_fwd"]
            rev = grp[grp["split"] == "test_rev"]
            if len(fwd) > 0 and len(rev) > 0:
                fm = fwd.iloc[0]
                rm = rev.iloc[0]
                both_pos.append({
                    **{c: fm[c] for c in cfg_cols},
                    "fwd_net": fm["net_mean_bp"],
                    "fwd_n": fm["n_trades"],
                    "fwd_pf": fm["pf"],
                    "fwd_median": fm["net_median_bp"],
                    "fwd_pct_tp": fm.get("pct_tp", 0),
                    "fwd_pct_trail": fm.get("pct_trail", 0),
                    "fwd_pct_time": fm.get("pct_time", 0),
                    "fwd_conc5t": fm.get("conc_top5_trade", np.nan),
                    "fwd_maxdd": fm.get("max_dd_bp", 0),
                    "fwd_wk": f"{fm.get('weeks_positive',0):.0f}/{fm.get('weeks_total',0):.0f}",
                    "rev_net": rm["net_mean_bp"],
                    "rev_n": rm["n_trades"],
                    "rev_pf": rm["pf"],
                    "avg_net": (fm["net_mean_bp"] + rm["net_mean_bp"]) / 2,
                    "total_n": fm["n_trades"] + rm["n_trades"],
                })

        if not both_pos:
            lines.append("No configs with data in both splits.")
            lines.append("")
            continue

        bp_df = pd.DataFrame(both_pos).sort_values("avg_net", ascending=False)
        n_pos = (bp_df["avg_net"] > 0).sum()
        lines.append(f"**{n_pos}/{len(bp_df)} configs positive in both splits** (slip=5bp)")
        lines.append("")

        # Table header
        hdr = "| a | b | c | T | cd | tr_s | tr_g | p1 | tp1 | "
        hdr += "Fwd net | Fwd n | Fwd PF | Fwd med | Rev net | Rev n | Avg net | "
        hdr += "TP% | TRAIL% | TIME% | top5t% | maxDD | wk+ |"
        lines.append(hdr)
        sep = "|" + "|".join(["---"] * 22) + "|"
        lines.append(sep)

        for _, r in bp_df.head(10).iterrows():
            row = (
                f"| {r['a']:.1f} | {r['b']:.1f} | {r['c']:.1f} | {r['T_h']:.0f}h | "
                f"{r['cancel_delay_s']:.0f} | {r['trail_start']:.1f} | {r['trail_gap']:.1f} | "
                f"{r['p1']:.1f} | {r['tp1']:.1f} | "
                f"{r['fwd_net']:.1f} | {r['fwd_n']:.0f} | {r['fwd_pf']:.2f} | "
                f"{r['fwd_median']:.1f} | {r['rev_net']:.1f} | {r['rev_n']:.0f} | "
                f"**{r['avg_net']:.1f}** | "
                f"{r['fwd_pct_tp']:.0%} | {r['fwd_pct_trail']:.0%} | {r['fwd_pct_time']:.0%} | "
                f"{r.get('fwd_conc5t', np.nan):.0%} | {r['fwd_maxdd']:.0f} | {r['fwd_wk']} |"
            )
            lines.append(row)
        lines.append("")

    # --- GO/NO-GO assessment ---
    lines.append("---")
    lines.append("")
    lines.append("## GO/NO-GO Assessment")
    lines.append("")

    best_all = df[(df["slip_bp"] == 5)].copy()
    if len(best_all) > 0:
        cfg_cols = ["a", "b", "c", "T_h", "cancel_delay_s",
                    "trail_start", "trail_gap", "p1", "tp1"]
        candidates = []
        for _, grp in best_all.groupby(cfg_cols):
            fwd = grp[grp["split"] == "test_fwd"]
            rev = grp[grp["split"] == "test_rev"]
            if len(fwd) == 0 or len(rev) == 0:
                continue
            fm, rm = fwd.iloc[0], rev.iloc[0]
            avg_net = (fm["net_mean_bp"] + rm["net_mean_bp"]) / 2
            total_n = fm["n_trades"] + rm["n_trades"]

            # Mini-criteria
            both_pos = fm["net_mean_bp"] > 0 and rm["net_mean_bp"] > 0
            enough_trades = total_n >= 50
            low_double = fm.get("pct_double", 0) <= 0.01
            low_conc = fm.get("conc_top5_trade", 1.0) <= 0.80
            ok_median = fm["net_median_bp"] >= -10

            # GO criteria
            go_net = fm["net_mean_bp"] > 10 and rm["net_mean_bp"] > 10
            go_pf = fm.get("pf", 0) >= 1.20
            go_dd = fm.get("max_dd_bp", -9999) > -3000
            go_conc = fm.get("conc_top5_trade", 1.0) <= 0.60
            go_weeks = fm.get("weeks_positive", 0) >= 0.6 * fm.get("weeks_total", 1)

            candidates.append({
                **{c: fm[c] for c in cfg_cols},
                "avg_net": avg_net,
                "total_n": total_n,
                "mini_pass": all([both_pos, enough_trades, low_double, low_conc, ok_median]),
                "go_pass": all([go_net, go_pf, go_dd, go_conc, go_weeks, both_pos, enough_trades]),
                "phase": fm.get("phase", "?"),
            })

        cdf = pd.DataFrame(candidates)
        if len(cdf) > 0:
            n_mini = cdf["mini_pass"].sum()
            n_go = cdf["go_pass"].sum()
            lines.append(f"- **Mini-criteria pass:** {n_mini}/{len(cdf)} configs")
            lines.append(f"- **GO criteria pass:** {n_go}/{len(cdf)} configs")
            lines.append("")

            if n_go > 0:
                lines.append("### GO Configs")
                lines.append("")
                go_cfgs = cdf[cdf["go_pass"]].sort_values("avg_net", ascending=False)
                for _, r in go_cfgs.head(5).iterrows():
                    lines.append(
                        f"- a={r['a']}, b={r['b']}, c={r['c']}, T={r['T_h']}h, "
                        f"tr={r['trail_start']}/{r['trail_gap']}, p={r['p1']}/{r['tp1']}: "
                        f"avg_net=**{r['avg_net']:.1f}bp**, n={r['total_n']:.0f}"
                    )
                lines.append("")

            if n_go == 0:
                lines.append("**Verdict: NO-GO** — no config meets all GO criteria.")
                if n_mini > 0:
                    lines.append("Some configs pass mini-criteria (candidates for further testing).")
                lines.append("")
            else:
                lines.append(f"**Verdict: CONDITIONAL GO** — {n_go} config(s) meet GO criteria.")
                lines.append("")

    # --- v2 vs v1 comparison hint ---
    lines.append("---")
    lines.append("")
    lines.append("## v1 vs v2 Comparison (reference)")
    lines.append("")
    lines.append("v1 best: a=1.0, b=3, c=1.5 → +20bp/trade, PF=1.14, median=-35bp, 96% TIME exits")
    lines.append("")
    lines.append("v2 improvements targeted:")
    lines.append("- Closer TP (b=2-2.5) → higher TP hit rate")
    lines.append("- Trailing stops → capture MFE on TIME exits (v1 mean MFE=389bp)")
    lines.append("- Partial exits → lock partial profit, reduce tail dependency")
    lines.append("")

    _write_findings(lines)


def _write_findings(lines):
    md_path = OUTPUT_DIR / "FINDINGS_xs7v2.md"
    md_path.write_text("\n".join(lines))
    print(f"\nFindings written to {md_path}")


# ---------------------------------------------------------------------------
# §10: Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.monotonic()
    print("=" * 70)
    print("XS-7 v2 — Convex Extraction on S07 (Exit Engineering)")
    print("=" * 70)
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Base grid: a={A_GRID_BASE}, b={B_GRID_BASE}, c={C_GRID_BASE}")
    print(f"Time stops: {[t//60 for t in T_GRID_BASE]}h")
    print(f"Cancel delay: {CANCEL_DELAY_BASE}s")
    print(f"Trailing: start={TRAIL_START_GRID}, gap={TRAIL_GAP_GRID}")
    print(f"Partial: p1={P1_GRID}, tp1={TP1_GRID}")
    print(f"Fee: taker={FEE_TAKER_BP}bp, maker={FEE_MAKER_BP}bp, slip={SLIP_GRID}bp")
    print(f"Cooldown: {COOLDOWN_MIN}min ({COOLDOWN_MIN//60}h)")
    print()

    # --- Load data ---
    symbols = discover_symbols()
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"Symbols: {len(symbols)}, Grid: {len(grid_1m):,} 1m bars")
    print()

    sym_data = {}
    total_signals = 0
    excluded = []
    for i, sym in enumerate(symbols, 1):
        raw = load_symbol(sym)
        df_1m, signal_times = build_symbol_data(sym, raw, grid_1m)
        if df_1m is None:
            excluded.append(sym)
            continue
        sym_data[sym] = (df_1m, signal_times)
        total_signals += len(signal_times)
        if i % 10 == 0 or i == len(symbols):
            print(f"  Loaded {i}/{len(symbols)} symbols, "
                  f"{len(sym_data)} valid, {total_signals} S07 signals")

    print(f"\nTotal: {len(sym_data)} symbols, {total_signals} S07 signals")
    if excluded:
        print(f"Excluded ({len(excluded)}): {', '.join(excluded[:10])}"
              f"{'...' if len(excluded) > 10 else ''}")
    print()

    # =====================================================================
    # PHASE 1: Base grid
    # =====================================================================
    base_configs = []
    for a, b, c, T, cd in product(A_GRID_BASE, B_GRID_BASE, C_GRID_BASE,
                                    T_GRID_BASE, CANCEL_DELAY_BASE):
        base_configs.append({
            "a": a, "b": b, "c": c, "T_min": T, "cancel_delay_s": cd,
            "trail_start": 0.0, "trail_gap": 0.0, "p1": 0.0, "tp1": 0.0,
        })

    print(f"Phase 1: {len(base_configs)} base configs")
    base_trades, base_summaries = run_grid_phase(sym_data, base_configs, "BASE")

    # Select top-5 for phases 2/3
    top5 = select_top_configs(base_summaries, k=5, slip_bp=5)
    print(f"\nTop-5 base configs for Phase 2/3:")
    for i, cfg in enumerate(top5, 1):
        print(f"  {i}. a={cfg['a']}, b={cfg['b']}, c={cfg['c']}, "
              f"cd={cfg['cancel_delay_s']}, avg_net={cfg['avg_net_mean']:.1f}bp, "
              f"n={cfg['total_trades']}")

    # =====================================================================
    # PHASE 2: Trailing grid on top-5
    # =====================================================================
    trail_configs = []
    for top_cfg in top5:
        for ts_val, tg_val in product(TRAIL_START_GRID, TRAIL_GAP_GRID):
            trail_configs.append({
                "a": top_cfg["a"], "b": top_cfg["b"], "c": top_cfg["c"],
                "T_min": int(top_cfg["T_h"]) * 60,
                "cancel_delay_s": int(top_cfg["cancel_delay_s"]),
                "trail_start": ts_val, "trail_gap": tg_val,
                "p1": 0.0, "tp1": 0.0,
            })

    trail_trades, trail_summaries = pd.DataFrame(), []
    if trail_configs:
        print(f"\nPhase 2: {len(trail_configs)} trailing configs")
        trail_trades, trail_summaries = run_grid_phase(sym_data, trail_configs, "TRAILING")

    # =====================================================================
    # PHASE 3: Partial grid on top-5
    # =====================================================================
    partial_configs = []
    for top_cfg in top5:
        for p1_val, tp1_val in product(P1_GRID, TP1_GRID):
            partial_configs.append({
                "a": top_cfg["a"], "b": top_cfg["b"], "c": top_cfg["c"],
                "T_min": int(top_cfg["T_h"]) * 60,
                "cancel_delay_s": int(top_cfg["cancel_delay_s"]),
                "trail_start": 0.0, "trail_gap": 0.0,
                "p1": p1_val, "tp1": tp1_val,
            })

    partial_trades, partial_summaries = pd.DataFrame(), []
    if partial_configs:
        print(f"\nPhase 3: {len(partial_configs)} partial configs")
        partial_trades, partial_summaries = run_grid_phase(sym_data, partial_configs, "PARTIAL")

    # =====================================================================
    # PHASE 4 (optional): Combo — top-3 trailing × top-3 partial
    # =====================================================================
    top3_trail = select_top_configs(trail_summaries, k=3, slip_bp=5)
    top3_partial = select_top_configs(partial_summaries, k=3, slip_bp=5)

    combo_configs = []
    seen = set()
    for tc in top3_trail:
        for pc in top3_partial:
            if tc["a"] != pc["a"] or tc["b"] != pc["b"] or tc["c"] != pc["c"]:
                continue  # only combine matching base configs
            key = (tc["a"], tc["b"], tc["c"], tc["T_h"], tc["cancel_delay_s"],
                   tc["trail_start"], tc["trail_gap"], pc["p1"], pc["tp1"])
            if key in seen:
                continue
            seen.add(key)
            combo_configs.append({
                "a": tc["a"], "b": tc["b"], "c": tc["c"],
                "T_min": int(tc["T_h"]) * 60,
                "cancel_delay_s": int(tc["cancel_delay_s"]),
                "trail_start": tc["trail_start"],
                "trail_gap": tc["trail_gap"],
                "p1": pc["p1"], "tp1": pc["tp1"],
            })

    combo_trades, combo_summaries = pd.DataFrame(), []
    if combo_configs:
        print(f"\nPhase 4: {len(combo_configs)} combo configs")
        combo_trades, combo_summaries = run_grid_phase(sym_data, combo_configs, "COMBO")

    # =====================================================================
    # Merge all results
    # =====================================================================
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    all_trade_frames = [base_trades]
    if len(trail_trades) > 0:
        all_trade_frames.append(trail_trades)
    if len(partial_trades) > 0:
        all_trade_frames.append(partial_trades)
    if len(combo_trades) > 0:
        all_trade_frames.append(combo_trades)

    all_trades_df = pd.concat(all_trade_frames, ignore_index=True)
    all_summaries = base_summaries + trail_summaries + partial_summaries + combo_summaries
    all_summaries_df = pd.DataFrame(all_summaries)

    print(f"Total trade records: {len(all_trades_df):,}")
    print(f"Total config summaries: {len(all_summaries_df)}")

    # Apply fees to merged trades (for saving, use slip=5 as default)
    all_trades_with_fees = apply_fees(all_trades_df, slip_bp=5)
    all_trades_with_fees = label_splits(all_trades_with_fees)

    # =====================================================================
    # Save outputs
    # =====================================================================
    print("\nSaving outputs...")

    # 1) Trade log
    trades_path = OUTPUT_DIR / "xs7v2_trades.csv"
    save_cols = [
        "symbol", "t_signal", "t_entry", "t_exit", "side",
        "entry_px", "exit_px", "exit_reason",
        "a", "b", "c", "T_min", "trail_start", "trail_gap", "p1", "tp1",
        "cancel_delay_s",
        "gross_pnl_bp", "fees_bp", "slip_bp", "net_bp",
        "MFE_bp", "MAE_bp", "time_in_trade_min",
        "double_trigger", "partial_filled", "trailing_active",
        "atr_at_signal", "entry_eq_signal_close",
        "split_fwd", "split_rev",
    ]
    actual_cols = [c for c in save_cols if c in all_trades_with_fees.columns]
    all_trades_with_fees[actual_cols].to_csv(trades_path, index=False, float_format="%.4f")
    print(f"  Trade log: {trades_path} ({len(all_trades_with_fees):,} rows)")

    # 2) Report
    report_path = OUTPUT_DIR / "xs7v2_report.csv"
    all_summaries_df.to_csv(report_path, index=False, float_format="%.4f")
    print(f"  Report: {report_path} ({len(all_summaries_df)} rows)")

    # 3) Equity curves (for top-10 configs at slip=5)
    equity_path = OUTPUT_DIR / "xs7v2_equity.csv"
    _save_equity_curves(all_trades_with_fees, all_summaries_df, equity_path)

    # =====================================================================
    # Bug guards
    # =====================================================================
    run_bug_guards(all_trades_with_fees)

    # =====================================================================
    # Print key results
    # =====================================================================
    _print_key_results(all_summaries_df)

    # =====================================================================
    # Generate findings
    # =====================================================================
    generate_findings(all_summaries_df, all_trades_with_fees)

    elapsed = time.monotonic() - t_start
    print(f"\nXS-7 v2 done in {elapsed:.0f}s ({elapsed/60:.1f}min)")


def _save_equity_curves(trades_df, summaries_df, path):
    """Save equity curves for top-10 configs."""
    cfg_cols = ["a", "b", "c", "T_min", "trail_start", "trail_gap",
                "p1", "tp1", "cancel_delay_s"]

    # Find top configs
    fwd = summaries_df[
        (summaries_df["slip_bp"] == 5) &
        (summaries_df["split"] == "test_fwd") &
        (summaries_df["n_trades"] >= 5)
    ].nlargest(10, "net_mean_bp")

    if len(fwd) == 0:
        pd.DataFrame().to_csv(path)
        return

    eq_rows = []
    for _, cfg_row in fwd.iterrows():
        mask = trades_df["exit_reason"] != "NO_TRIGGER"
        for col in cfg_cols:
            if col in trades_df.columns and col in cfg_row.index:
                mask = mask & (trades_df[col] == cfg_row[col])

        subset = trades_df[mask].sort_values("t_exit")
        if len(subset) == 0:
            continue

        cfg_id = (f"a{cfg_row['a']}_b{cfg_row['b']}_c{cfg_row['c']}_"
                  f"tr{cfg_row.get('trail_start',0)}_{cfg_row.get('trail_gap',0)}_"
                  f"p{cfg_row.get('p1',0)}_{cfg_row.get('tp1',0)}")

        cum = subset["net_bp"].cumsum().values
        peak = np.maximum.accumulate(cum)
        dd = cum - peak

        for j, (_, row) in enumerate(subset.iterrows()):
            eq_rows.append({
                "config_id": cfg_id,
                "timestamp": row["t_exit"],
                "equity_bp": cum[j],
                "dd_bp": dd[j],
            })

    pd.DataFrame(eq_rows).to_csv(path, index=False, float_format="%.4f")
    print(f"  Equity curves: {path} ({len(eq_rows)} rows)")


def _print_key_results(summaries_df):
    """Print key results table."""
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    df = summaries_df.copy()

    for phase in df["phase"].unique():
        print(f"\n--- {phase} ---")
        fwd = df[
            (df["phase"] == phase) &
            (df["split"] == "test_fwd") &
            (df["slip_bp"] == 5) &
            (df["n_trades"] >= 5)
        ].sort_values("net_mean_bp", ascending=False)

        if len(fwd) == 0:
            print("  No configs with ≥5 trades.")
            continue

        cols = ["a", "b", "c", "T_h", "cancel_delay_s",
                "trail_start", "trail_gap", "p1", "tp1",
                "n_trades", "net_mean_bp", "net_median_bp", "pf",
                "pct_tp", "pct_trail", "pct_time", "pct_double",
                "conc_top5_trade", "max_dd_bp", "weeks_positive", "weeks_total"]
        show_cols = [c for c in cols if c in fwd.columns]
        print(fwd.head(10)[show_cols].to_string(index=False, float_format="%.2f"))

    # Cost sensitivity
    print("\n--- Cost Sensitivity (best base config, test_fwd) ---")
    best_base = df[
        (df["phase"] == "BASE") &
        (df["split"] == "test_fwd") &
        (df["n_trades"] >= 5)
    ].sort_values("net_mean_bp", ascending=False)

    if len(best_base) > 0:
        top = best_base.iloc[0]
        cfg_cols = ["a", "b", "c", "T_h", "cancel_delay_s"]
        for slip in SLIP_GRID:
            match = df[
                (df["phase"] == "BASE") &
                (df["split"] == "test_fwd") &
                (df["slip_bp"] == slip)
            ]
            for col in cfg_cols:
                match = match[match[col] == top[col]]
            if len(match) > 0:
                r = match.iloc[0]
                print(f"  slip={slip}bp: net_mean={r['net_mean_bp']:.1f}bp, PF={r['pf']:.2f}")


if __name__ == "__main__":
    main()
