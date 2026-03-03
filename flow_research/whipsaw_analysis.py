#!/usr/bin/env python3
"""
Whipsaw Structure Analysis + Fade-First-Break Simulation on REG_OI_FUND

Two phases:
  Phase 1: Measure whipsaw structure (retrace rates, timing, depth)
  Phase 2: If structure confirms → simulate fade-first-break trades

Uses mark price 1m bars + regime_dataset.parquet.

Output:
  flow_research/output/regime/whipsaw_report.csv
  flow_research/output/regime/fade_report.csv
"""

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_PARQUET = OUTPUT_DIR / "regime_dataset.parquet"

SYMBOLS = [
    "1000BONKUSDT", "ARBUSDT", "APTUSDT", "ATOMUSDT",
    "AIXBTUSDT", "1000RATSUSDT", "ARCUSDT", "1000TURBOUSDT",
]

START_TS = pd.Timestamp("2026-01-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# §2: First Break Event
K_BREAK_GRID = [0.3, 0.5, 0.7]
T_BREAK_MAX = 30  # minutes to find first break

# §2.2: Cooldown
COOLDOWN_MIN = 60

# ATR
ATR_PERIOD = 15
ATR_MIN_PERCENTILE = 5

# §3: Whipsaw measurement window after break
WHIPSAW_WINDOW = 30  # minutes after break

# §4: Fade trade params
DELAY_GRID = [0, 1, 3]  # minutes after break to enter fade
K_SL_GRID = [0.7, 1.0, 1.3]
TP_MODE = ["P0", "0.5ATR"]  # TP targets
FADE_TO = 30  # minutes timeout from fade entry

# Fees
FEE_RT_BP = 20.0
SLIPPAGE_GRID_BP = [0, 2, 5]

# Walk-forward
WF_SPLIT = pd.Timestamp("2026-02-01", tz="UTC")


# ---------------------------------------------------------------------------
# Data loading (reused from backtest_breakout.py)
# ---------------------------------------------------------------------------


def load_mark_1m(symbol: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_mark_price_kline_1m.csv"))
    if not files:
        files = sorted(
            f for f in sym_dir.glob("*_kline_1m.csv")
            if "mark_price" not in f.name and "premium_index" not in f.name
        )
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)].reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close"]]


def compute_atr_series(df_1m: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df_1m["close"].shift(1)
    tr = pd.concat([
        df_1m["high"] - df_1m["low"],
        (df_1m["high"] - prev_close).abs(),
        (df_1m["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Phase 1: Whipsaw structure measurement
# ---------------------------------------------------------------------------


def measure_whipsaw_events(
    bars_1m: pd.DataFrame,
    signal_indices: list[int],
    atr_at_signal: list[float],
    k: float,
) -> list[dict]:
    """For each signal, find first break and measure retrace structure."""
    n_bars = len(bars_1m)
    highs = bars_1m["high"].values
    lows = bars_1m["low"].values
    closes = bars_1m["close"].values
    ts_vals = bars_1m["ts"].values

    events = []

    for sig_idx, atr in zip(signal_indices, atr_at_signal):
        P0 = closes[sig_idx]
        upper = P0 + k * atr
        lower = P0 - k * atr

        ev = {
            "signal_idx": sig_idx,
            "signal_ts": ts_vals[sig_idx],
            "P0": P0,
            "ATR": atr,
            "k": k,
            "break_found": False,
            "break_side": None,
            "break_bar": None,
            "time_to_break_min": np.nan,
            "break_price": np.nan,
            # Retrace metrics
            "touch_P0_5m": False,
            "touch_P0_15m": False,
            "touch_P0_30m": False,
            "time_to_touch_P0_min": np.nan,
            "mfe_after_break_atr": np.nan,  # in ATR units, in break direction
            "mae_after_break_atr": np.nan,  # in ATR units, against break (retrace depth)
            "whipsaw": False,
            "max_retrace_atr": np.nan,
        }

        # Phase 1a: Find first break within T_BREAK_MAX
        break_bar = None
        break_side = None
        max_search = min(sig_idx + T_BREAK_MAX + 1, n_bars)

        for i in range(sig_idx + 1, max_search):
            up_hit = highs[i] >= upper
            dn_hit = lows[i] <= lower

            if up_hit and dn_hit:
                # Both in same bar — skip (ambiguous)
                break
            if up_hit:
                break_bar = i
                break_side = "UP"
                ev["break_price"] = max(upper, bars_1m.iloc[i]["open"])
                break
            if dn_hit:
                break_bar = i
                break_side = "DOWN"
                ev["break_price"] = min(lower, bars_1m.iloc[i]["open"])
                break

        if break_bar is None:
            events.append(ev)
            continue

        ev["break_found"] = True
        ev["break_side"] = break_side
        ev["break_bar"] = break_bar
        ev["time_to_break_min"] = break_bar - sig_idx

        # Phase 1b: Measure retrace structure after break
        measure_end = min(break_bar + WHIPSAW_WINDOW + 1, n_bars)

        mfe = 0.0  # max favorable excursion (in break direction)
        mae = 0.0  # max adverse excursion (against break = retrace)
        touched_P0 = False
        touch_time = np.nan

        for j in range(break_bar + 1, measure_end):
            dt = j - break_bar  # minutes after break

            if break_side == "UP":
                # Favorable = higher, adverse = lower (retrace toward P0)
                excursion_fav = (highs[j] / P0 - 1)
                excursion_adv = (lows[j] / P0 - 1)  # negative = retrace
                mfe = max(mfe, excursion_fav)
                mae = min(mae, excursion_adv)

                if not touched_P0 and lows[j] <= P0:
                    touched_P0 = True
                    touch_time = dt
            else:
                # Favorable = lower, adverse = higher (retrace toward P0)
                excursion_fav = (1 - lows[j] / P0)
                excursion_adv = (1 - highs[j] / P0)  # negative = retrace
                mfe = max(mfe, excursion_fav)
                mae = min(mae, excursion_adv)

                if not touched_P0 and highs[j] >= P0:
                    touched_P0 = True
                    touch_time = dt

            # Check touch_P0 milestones
            if touched_P0:
                if dt <= 5:
                    ev["touch_P0_5m"] = True
                if dt <= 15:
                    ev["touch_P0_15m"] = True
                if dt <= 30:
                    ev["touch_P0_30m"] = True

        if touched_P0:
            ev["time_to_touch_P0_min"] = touch_time

        ev["mfe_after_break_atr"] = mfe * P0 / atr if atr > 0 else 0
        ev["mae_after_break_atr"] = abs(mae) * P0 / atr if atr > 0 else 0
        ev["max_retrace_atr"] = ev["mae_after_break_atr"]

        # Whipsaw definition: touched P0 within 30m AND retrace >= 0.2 ATR
        ev["whipsaw"] = touched_P0 and ev["max_retrace_atr"] >= 0.2

        events.append(ev)

    return events


# ---------------------------------------------------------------------------
# Phase 2: Fade-First-Break trade simulation
# ---------------------------------------------------------------------------


def simulate_fade_trade(
    bars_1m: pd.DataFrame,
    break_bar: int,
    break_side: str,
    P0: float,
    atr: float,
    delay: int,
    k_sl: float,
    tp_mode: str,
    slip_bp: float,
) -> dict:
    """Simulate a single fade trade: enter against the break direction."""
    n_bars = len(bars_1m)
    entry_bar = break_bar + delay

    if entry_bar >= n_bars:
        return {"triggered": False}

    # Entry price
    entry_price = bars_1m.iloc[entry_bar]["close"]

    # Fade direction: opposite of break
    if break_side == "UP":
        fade_side = "SHORT"
        entry_price *= (1 + slip_bp / 10000)  # worse for short entry
    else:
        fade_side = "LONG"
        entry_price *= (1 - slip_bp / 10000)  # worse for long entry

    # TP level
    if tp_mode == "P0":
        if fade_side == "LONG":
            tp_level = P0
        else:
            tp_level = P0
    elif tp_mode == "0.5ATR":
        if fade_side == "LONG":
            tp_level = entry_price + 0.5 * atr
        else:
            tp_level = entry_price - 0.5 * atr
    else:
        tp_level = P0

    # SL level
    if fade_side == "LONG":
        sl_level = entry_price - k_sl * atr
    else:
        sl_level = entry_price + k_sl * atr

    # Simulate exit
    max_bar = min(entry_bar + FADE_TO + 1, n_bars)
    exit_price = np.nan
    exit_reason = "TO"
    exit_bar = None
    mfe = 0.0
    mae = 0.0

    for j in range(entry_bar + 1, max_bar):
        h = bars_1m.iloc[j]["high"]
        l = bars_1m.iloc[j]["low"]

        if fade_side == "LONG":
            mfe = max(mfe, (h / entry_price - 1) * 10000)
            mae = min(mae, (l / entry_price - 1) * 10000)
            tp_hit = h >= tp_level
            sl_hit = l <= sl_level
        else:
            mfe = max(mfe, (1 - l / entry_price) * 10000)
            mae = min(mae, (1 - h / entry_price) * 10000)
            tp_hit = l <= tp_level
            sl_hit = h >= sl_level

        if tp_hit and sl_hit:
            exit_bar = j
            exit_price = sl_level  # conservative
            exit_reason = "SL"
            break
        elif sl_hit:
            exit_bar = j
            exit_price = sl_level
            exit_reason = "SL"
            break
        elif tp_hit:
            exit_bar = j
            exit_price = tp_level
            exit_reason = "TP"
            break

    if exit_bar is None:
        exit_bar = max_bar - 1
        if exit_bar > entry_bar and exit_bar < n_bars:
            exit_price = bars_1m.iloc[exit_bar]["close"]
            exit_reason = "TO"
        else:
            return {"triggered": False}

    # Apply exit slippage
    if fade_side == "LONG":
        exit_price *= (1 - slip_bp / 10000)
        gross_pnl_bp = (exit_price / entry_price - 1) * 10000
    else:
        exit_price *= (1 + slip_bp / 10000)
        gross_pnl_bp = (1 - exit_price / entry_price) * 10000

    return {
        "triggered": True,
        "fade_side": fade_side,
        "entry_bar": entry_bar,
        "entry_ts": bars_1m.iloc[entry_bar]["ts"],
        "entry_price": entry_price,
        "exit_bar": exit_bar,
        "exit_ts": bars_1m.iloc[exit_bar]["ts"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl_bp": gross_pnl_bp,
        "net_pnl_bp": gross_pnl_bp - FEE_RT_BP,
        "mfe_bp": mfe,
        "mae_bp": mae,
        "hold_min": exit_bar - entry_bar,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("WHIPSAW STRUCTURE ANALYSIS + FADE-FIRST-BREAK")
    print("=" * 80)

    regime_df = pd.read_parquet(REGIME_PARQUET)
    regime_df["ts"] = pd.to_datetime(regime_df["ts"], utc=True)

    all_whipsaw_events = []
    all_fade_trades = []

    for sym_i, sym in enumerate(SYMBOLS, 1):
        sym_t0 = time.monotonic()
        print(f"\n{'─'*70}")
        print(f"[{sym_i}/{len(SYMBOLS)}] {sym}")
        print(f"{'─'*70}")

        bars_1m = load_mark_1m(sym)
        if len(bars_1m) == 0:
            print("  SKIP: no 1m data")
            continue
        print(f"  1m bars: {len(bars_1m):,}")

        atr_series = compute_atr_series(bars_1m)
        valid_atr = atr_series.dropna()
        atr_min = valid_atr.quantile(ATR_MIN_PERCENTILE / 100)

        # Get signals with cooldown
        sym_signals = regime_df[
            (regime_df["symbol"] == sym) & (regime_df["REG_OI_FUND"] == 1)
        ].sort_values("ts")

        ts_1m = bars_1m["ts"].values.astype("int64")
        cooldown_delta = pd.Timedelta(minutes=COOLDOWN_MIN)
        last_used = pd.Timestamp.min.tz_localize("UTC")

        signal_indices = []
        signal_atrs = []
        signal_ts_list = []

        for _, row in sym_signals.iterrows():
            sig_ts = row["ts"]
            if sig_ts < last_used + cooldown_delta:
                continue
            bar_idx = np.searchsorted(ts_1m, sig_ts.value, side="right") - 1
            if bar_idx < 0 or bar_idx >= len(bars_1m):
                continue
            atr_val = atr_series.iloc[bar_idx]
            if np.isnan(atr_val) or atr_val <= 0 or atr_val < atr_min:
                continue
            signal_indices.append(bar_idx)
            signal_atrs.append(atr_val)
            signal_ts_list.append(sig_ts)
            last_used = sig_ts

        print(f"  Signals (after cooldown): {len(signal_indices)}")

        if len(signal_indices) == 0:
            continue

        # Phase 1: Whipsaw structure for each k
        for k in K_BREAK_GRID:
            events = measure_whipsaw_events(bars_1m, signal_indices, signal_atrs, k)
            for ev in events:
                ev["symbol"] = sym
            all_whipsaw_events.extend(events)

            n_break = sum(1 for e in events if e["break_found"])
            n_touch = sum(1 for e in events if e["touch_P0_30m"])
            n_ws = sum(1 for e in events if e["whipsaw"])
            print(f"  k={k}: {len(events)} signals, {n_break} breaks, "
                  f"touchP0_30m={n_touch} ({n_touch/max(n_break,1):.0%}), "
                  f"whipsaw={n_ws} ({n_ws/max(n_break,1):.0%})")

        # Phase 2: Fade trades (iterate over all param combos)
        for k in K_BREAK_GRID:
            # Get break events for this k
            events = measure_whipsaw_events(bars_1m, signal_indices, signal_atrs, k)

            for ev in events:
                if not ev["break_found"]:
                    continue

                for delay, k_sl, tp_mode, slip_bp in itertools.product(
                    DELAY_GRID, K_SL_GRID, TP_MODE, SLIPPAGE_GRID_BP
                ):
                    trade = simulate_fade_trade(
                        bars_1m,
                        break_bar=ev["break_bar"],
                        break_side=ev["break_side"],
                        P0=ev["P0"],
                        atr=ev["ATR"],
                        delay=delay,
                        k_sl=k_sl,
                        tp_mode=tp_mode,
                        slip_bp=slip_bp,
                    )
                    if not trade["triggered"]:
                        continue

                    trade["symbol"] = sym
                    trade["signal_ts"] = ev["signal_ts"]
                    trade["k_break"] = k
                    trade["delay"] = delay
                    trade["k_sl"] = k_sl
                    trade["tp_mode"] = tp_mode
                    trade["slip_bp"] = slip_bp
                    trade["P0"] = ev["P0"]
                    trade["break_side"] = ev["break_side"]
                    all_fade_trades.append(trade)

        print(f"  {sym} done in {time.monotonic()-sym_t0:.1f}s")

    # ===================================================================
    # Phase 1 report: Whipsaw structure
    # ===================================================================
    ws_df = pd.DataFrame(all_whipsaw_events)

    print(f"\n{'='*80}")
    print("PHASE 1: WHIPSAW STRUCTURE")
    print("=" * 80)

    ws_rows = []
    for (sym, k), grp in ws_df.groupby(["symbol", "k"]):
        breaks = grp[grp["break_found"]]
        n_sig = len(grp)
        n_break = len(breaks)
        if n_break == 0:
            continue

        ws_rows.append({
            "symbol": sym,
            "k": k,
            "n_signals": n_sig,
            "n_breaks": n_break,
            "break_rate": n_break / n_sig,
            "median_time_to_break_min": breaks["time_to_break_min"].median(),
            "touch_P0_5m_rate": breaks["touch_P0_5m"].mean(),
            "touch_P0_15m_rate": breaks["touch_P0_15m"].mean(),
            "touch_P0_30m_rate": breaks["touch_P0_30m"].mean(),
            "median_time_to_touch_P0_min": breaks["time_to_touch_P0_min"].median(),
            "median_mfe_atr": breaks["mfe_after_break_atr"].median(),
            "median_retrace_atr": breaks["max_retrace_atr"].median(),
            "whipsaw_rate": breaks["whipsaw"].mean(),
            "p25_retrace_atr": breaks["max_retrace_atr"].quantile(0.25),
            "p75_retrace_atr": breaks["max_retrace_atr"].quantile(0.75),
        })

    ws_report = pd.DataFrame(ws_rows)
    ws_report.to_csv(OUTPUT_DIR / "whipsaw_report.csv", index=False)

    print(f"\n{'Sym':<18} {'k':>3} {'N_brk':>5} {'Brk%':>5} | "
          f"{'tP0_5':>5} {'tP0_15':>6} {'tP0_30':>6} {'MedT':>5} | "
          f"{'MFE':>5} {'Retr':>5} {'WS%':>5}")
    print("-" * 95)

    for _, r in ws_report.iterrows():
        print(f"{r['symbol']:<18} {r['k']:>3.1f} {r['n_breaks']:>5.0f} {r['break_rate']:>5.0%} | "
              f"{r['touch_P0_5m_rate']:>5.0%} {r['touch_P0_15m_rate']:>6.0%} {r['touch_P0_30m_rate']:>6.0%} "
              f"{r['median_time_to_touch_P0_min']:>5.0f} | "
              f"{r['median_mfe_atr']:>5.2f} {r['median_retrace_atr']:>5.2f} {r['whipsaw_rate']:>5.0%}")

    # Cross-symbol averages
    print(f"\n  Cross-symbol averages:")
    for k in K_BREAK_GRID:
        subset = ws_report[ws_report["k"] == k]
        if len(subset) == 0:
            continue
        print(f"    k={k}: break_rate={subset['break_rate'].mean():.0%}, "
              f"touchP0_30m={subset['touch_P0_30m_rate'].mean():.0%}, "
              f"med_retrace={subset['median_retrace_atr'].mean():.2f} ATR, "
              f"whipsaw={subset['whipsaw_rate'].mean():.0%}")

    # ===================================================================
    # §6 structural criteria check
    # ===================================================================
    print(f"\n{'─'*70}")
    print("STRUCTURAL CRITERIA CHECK (§6)")
    print(f"{'─'*70}")

    for _, r in ws_report.iterrows():
        c1 = r["touch_P0_30m_rate"] >= 0.65
        c2 = r["median_time_to_touch_P0_min"] <= 10
        c3 = r["median_retrace_atr"] >= 0.4
        verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
        flags = []
        if not c1: flags.append(f"tP0_30m={r['touch_P0_30m_rate']:.0%}<65%")
        if not c2: flags.append(f"medT={r['median_time_to_touch_P0_min']:.0f}m>10m")
        if not c3: flags.append(f"retrace={r['median_retrace_atr']:.2f}<0.4ATR")
        print(f"  {r['symbol']:<18} k={r['k']:.1f}  {verdict}  {', '.join(flags) if flags else 'all criteria met'}")

    # ===================================================================
    # Phase 2 report: Fade trades
    # ===================================================================
    if not all_fade_trades:
        print("\nNo fade trades generated.")
        return

    fade_df = pd.DataFrame(all_fade_trades)
    fade_df["signal_ts"] = pd.to_datetime(fade_df["signal_ts"], utc=True)

    # Build summary
    group_cols = ["symbol", "k_break", "delay", "k_sl", "tp_mode", "slip_bp"]
    fade_rows = []

    for keys, grp in fade_df.groupby(group_cols):
        net = grp["net_pnl_bp"]
        n = len(grp)
        wins = net[net > 0]
        losses = net[net <= 0]
        gw = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)

        fade_rows.append({
            **dict(zip(group_cols, keys)),
            "n_trades": n,
            "win_rate": (net > 0).mean(),
            "tp_rate": (grp["exit_reason"] == "TP").mean(),
            "sl_rate": (grp["exit_reason"] == "SL").mean(),
            "to_rate": (grp["exit_reason"] == "TO").mean(),
            "mean_net_bp": net.mean(),
            "median_net_bp": net.median(),
            "p5_net_bp": net.quantile(0.05),
            "p95_net_bp": net.quantile(0.95),
            "profit_factor": pf,
            "avg_hold_min": grp["hold_min"].mean(),
        })

    fade_report = pd.DataFrame(fade_rows)
    fade_report.to_csv(OUTPUT_DIR / "fade_report.csv", index=False)

    # Print best configs per symbol (slip=0)
    print(f"\n{'='*80}")
    print("PHASE 2: FADE-FIRST-BREAK — TOP CONFIGS (slip=0)")
    print("=" * 80)

    for sym in sorted(fade_report["symbol"].unique()):
        sym_f = fade_report[(fade_report["symbol"] == sym) & (fade_report["slip_bp"] == 0)]
        sym_f = sym_f.sort_values("mean_net_bp", ascending=False)
        top = sym_f.head(5)

        print(f"\n  {sym}:")
        print(f"  {'k':>3} {'del':>3} {'SL':>4} {'TP':>6} | {'N':>3} {'WR':>5} {'TP%':>5} {'SL%':>5} | "
              f"{'Med':>7} {'Mean':>7} {'PF':>5} | {'P5':>7} {'P95':>7}")
        for _, r in top.iterrows():
            print(f"  {r['k_break']:>3.1f} {r['delay']:>3.0f} {r['k_sl']:>4.1f} {r['tp_mode']:>6} | "
                  f"{r['n_trades']:>3.0f} {r['win_rate']:>5.1%} {r['tp_rate']:>5.1%} {r['sl_rate']:>5.1%} | "
                  f"{r['median_net_bp']:>+7.1f} {r['mean_net_bp']:>+7.1f} {r['profit_factor']:>5.2f} | "
                  f"{r['p5_net_bp']:>+7.1f} {r['p95_net_bp']:>+7.1f}")

    # Slippage sensitivity for best config per symbol
    print(f"\n{'='*80}")
    print("SLIPPAGE SENSITIVITY (best config per symbol)")
    print("=" * 80)

    for sym in sorted(fade_report["symbol"].unique()):
        sym_f0 = fade_report[(fade_report["symbol"] == sym) & (fade_report["slip_bp"] == 0)]
        if len(sym_f0) == 0:
            continue
        best = sym_f0.loc[sym_f0["mean_net_bp"].idxmax()]
        params = {c: best[c] for c in ["k_break", "delay", "k_sl", "tp_mode"]}

        print(f"\n  {sym} (k={params['k_break']}, del={params['delay']:.0f}, "
              f"SL={params['k_sl']}, TP={params['tp_mode']}):")
        for slip in SLIPPAGE_GRID_BP:
            mask = (
                (fade_report["symbol"] == sym) &
                (fade_report["k_break"] == params["k_break"]) &
                (fade_report["delay"] == params["delay"]) &
                (fade_report["k_sl"] == params["k_sl"]) &
                (fade_report["tp_mode"] == params["tp_mode"]) &
                (fade_report["slip_bp"] == slip)
            )
            row = fade_report[mask]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"    slip={slip:>2}: N={r['n_trades']:.0f}, mean={r['mean_net_bp']:+.1f}bp, "
                  f"PF={r['profit_factor']:.2f}, WR={r['win_rate']:.1%}")

    # Walk-forward
    print(f"\n{'='*80}")
    print("WALK-FORWARD (best config per symbol, slip=0)")
    print("=" * 80)

    fade_df["period"] = np.where(fade_df["signal_ts"] < WF_SPLIT, "JAN", "FEB")
    param_cols = ["k_break", "delay", "k_sl", "tp_mode"]

    for sym in sorted(fade_df["symbol"].unique()):
        sym_fd = fade_df[(fade_df["symbol"] == sym) & (fade_df["slip_bp"] == 0)]
        if len(sym_fd) == 0:
            continue
        jan = sym_fd[sym_fd["period"] == "JAN"]
        feb = sym_fd[sym_fd["period"] == "FEB"]
        if len(jan) == 0 or len(feb) == 0:
            print(f"\n  {sym}: insufficient data (JAN={len(jan)}, FEB={len(feb)})")
            continue

        jan_perf = jan.groupby(param_cols)["net_pnl_bp"].mean().reset_index()
        best_jan = jan_perf.loc[jan_perf["net_pnl_bp"].idxmax()]
        bp_j = {c: best_jan[c] for c in param_cols}

        feb_test = feb
        for c in param_cols:
            feb_test = feb_test[feb_test[c] == bp_j[c]]

        feb_perf = feb.groupby(param_cols)["net_pnl_bp"].mean().reset_index()
        best_feb = feb_perf.loc[feb_perf["net_pnl_bp"].idxmax()]
        bp_f = {c: best_feb[c] for c in param_cols}

        jan_test = jan
        for c in param_cols:
            jan_test = jan_test[jan_test[c] == bp_f[c]]

        print(f"\n  {sym}:")
        print(f"    Train=JAN best: k={bp_j['k_break']}, del={bp_j['delay']}, SL={bp_j['k_sl']}, TP={bp_j['tp_mode']}")
        if len(feb_test) > 0:
            print(f"    Test=FEB: N={len(feb_test)}, mean={feb_test['net_pnl_bp'].mean():+.1f}bp, "
                  f"WR={(feb_test['net_pnl_bp']>0).mean():.0%}")
        else:
            print(f"    Test=FEB: no trades")
        print(f"    Train=FEB best: k={bp_f['k_break']}, del={bp_f['delay']}, SL={bp_f['k_sl']}, TP={bp_f['tp_mode']}")
        if len(jan_test) > 0:
            print(f"    Test=JAN: N={len(jan_test)}, mean={jan_test['net_pnl_bp'].mean():+.1f}bp, "
                  f"WR={(jan_test['net_pnl_bp']>0).mean():.0%}")
        else:
            print(f"    Test=JAN: no trades")

    # Go/No-Go
    print(f"\n{'='*80}")
    print("GO / NO-GO ASSESSMENT")
    print("=" * 80)

    for sym in sorted(fade_report["symbol"].unique()):
        sym_f0 = fade_report[(fade_report["symbol"] == sym) & (fade_report["slip_bp"] == 0)]
        if len(sym_f0) == 0:
            continue
        best = sym_f0.loc[sym_f0["mean_net_bp"].idxmax()]

        exp_ok = best["mean_net_bp"] > 0
        pf_ok = best["profit_factor"] > 1.1

        # Slip=2
        slip2 = fade_report[
            (fade_report["symbol"] == sym) &
            (fade_report["k_break"] == best["k_break"]) &
            (fade_report["delay"] == best["delay"]) &
            (fade_report["k_sl"] == best["k_sl"]) &
            (fade_report["tp_mode"] == best["tp_mode"]) &
            (fade_report["slip_bp"] == 2)
        ]
        slip2_ok = len(slip2) > 0 and slip2.iloc[0]["mean_net_bp"] > 0

        verdict = "GO" if (exp_ok and pf_ok and slip2_ok) else "NO-GO"
        note = ""
        if not exp_ok: note += " EV≤0"
        if not pf_ok: note += " PF≤1.1"
        if not slip2_ok: note += " dies@2bp"

        print(f"  {sym:<18} {verdict:<6}  mean={best['mean_net_bp']:+.1f}bp  "
              f"PF={best['profit_factor']:.2f}  WR={best['win_rate']:.0%}  "
              f"N={best['n_trades']:.0f}  slip2={'OK' if slip2_ok else 'FAIL'}{note}")

    elapsed = time.monotonic() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Outputs: {OUTPUT_DIR}/whipsaw_report.csv, fade_report.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
