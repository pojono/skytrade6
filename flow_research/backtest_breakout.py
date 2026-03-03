#!/usr/bin/env python3
"""
Dual-Stop Breakout Backtest on REG_OI_FUND

Monetizes the confirmed 1.5–3.9x range expansion during OI_FUND regime
via a non-directional breakout strategy on perp futures.

Uses mark price 1m bars for execution simulation.
Reads regime_dataset.parquet for signal timestamps.

Output:
  flow_research/output/regime/breakout_trades.parquet
  flow_research/output/regime/breakout_report.csv
  flow_research/output/regime/breakout_weekly.csv
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

# §1: Period
START_TS = pd.Timestamp("2026-01-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# §4.1: Parameter grid
K_ENTRY_GRID = [0.3, 0.5, 0.7]
K_SL_GRID = [0.5, 0.8, 1.0]
K_TP_GRID = [1.0, 1.5, 2.0]
TO_GRID = [30, 60]  # minutes

# §4.8: Fees (bp) — taker RT = 20bp
FEE_ENTRY_BP = 10.0
FEE_EXIT_BP = 10.0

# Slippage sensitivity
SLIPPAGE_GRID_BP = [0, 2, 5]

# §2.2: Cooldown after signal (minutes)
COOLDOWN_MIN = 60

# §3.1: ATR period (1m bars)
ATR_PERIOD = 15

# §5: Quality filter — skip if ATR below P5 (computed per-symbol)
ATR_MIN_PERCENTILE = 5

# §9: Walk-forward split
WF_SPLIT = pd.Timestamp("2026-02-01", tz="UTC")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mark_1m(symbol: str) -> pd.DataFrame:
    """Load mark price 1m bars for a symbol from per-day CSVs."""
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_mark_price_kline_1m.csv"))
    if not files:
        # Fallback to regular kline
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


def load_signals(symbol: str, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Extract REG_OI_FUND signal timestamps for a symbol."""
    sym_df = regime_df[
        (regime_df["symbol"] == symbol) & (regime_df["REG_OI_FUND"] == 1)
    ].copy()
    sym_df = sym_df.sort_values("ts").reset_index(drop=True)
    return sym_df


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------


def compute_atr_series(df_1m: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Compute SMA-based ATR on 1m bars. Returns Series aligned to df_1m index."""
    prev_close = df_1m["close"].shift(1)
    tr = pd.concat([
        df_1m["high"] - df_1m["low"],
        (df_1m["high"] - prev_close).abs(),
        (df_1m["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


# ---------------------------------------------------------------------------
# Single trade simulation
# ---------------------------------------------------------------------------


def simulate_trade(
    bars_1m: pd.DataFrame,  # 1m bars from signal_ts onward (pre-sliced)
    signal_idx: int,  # index in bars_1m of the signal bar
    P0: float,
    atr: float,
    k_entry: float,
    k_sl: float,
    k_tp: float,
    timeout_min: int,
    fee_entry_bp: float,
    fee_exit_bp: float,
    slip_bp: float,
) -> dict:
    """Simulate a single dual-stop breakout trade.

    Returns a dict with trade details.
    """
    buy_stop = P0 + k_entry * atr
    sell_stop = P0 - k_entry * atr

    result = {
        "P0": P0,
        "ATR_15m": atr,
        "buy_stop": buy_stop,
        "sell_stop": sell_stop,
        "triggered_side": "NONE",
        "entry_ts": pd.NaT,
        "entry_price": np.nan,
        "exit_ts": pd.NaT,
        "exit_price": np.nan,
        "exit_reason": "NO_TRIGGER",
        "gross_pnl_bp": 0.0,
        "fees_bp": 0.0,
        "slippage_bp": 0.0,
        "net_pnl_bp": 0.0,
        "mfe_bp": 0.0,
        "mae_bp": 0.0,
        "hold_minutes": 0,
    }

    n_bars = len(bars_1m)
    max_bar = min(signal_idx + timeout_min + 1, n_bars)

    # Phase 1: Find trigger in bars after signal
    trigger_bar = None
    triggered_side = None

    for i in range(signal_idx + 1, max_bar):
        bar_high = bars_1m.iloc[i]["high"]
        bar_low = bars_1m.iloc[i]["low"]

        long_hit = bar_high >= buy_stop
        short_hit = bar_low <= sell_stop

        if long_hit and short_hit:
            # Ambiguous — both hit in same bar → mark worst case
            result["triggered_side"] = "AMBIG"
            result["exit_reason"] = "AMBIG"
            return result

        if long_hit:
            trigger_bar = i
            triggered_side = "LONG"
            # Entry price: max of stop level and open (gap scenario)
            entry_price = max(buy_stop, bars_1m.iloc[i]["open"])
            break
        elif short_hit:
            trigger_bar = i
            triggered_side = "SHORT"
            entry_price = min(sell_stop, bars_1m.iloc[i]["open"])
            break

    if trigger_bar is None:
        return result

    # Apply slippage to entry
    if triggered_side == "LONG":
        entry_price *= (1 + slip_bp / 10000)
    else:
        entry_price *= (1 - slip_bp / 10000)

    entry_ts = bars_1m.iloc[trigger_bar]["ts"]

    # Phase 2: Set SL/TP levels
    if triggered_side == "LONG":
        sl_level = entry_price - k_sl * atr
        tp_level = entry_price + k_tp * atr
    else:
        sl_level = entry_price + k_sl * atr
        tp_level = entry_price - k_tp * atr

    # Phase 3: Simulate exit on 1m bars after entry
    # Timeout is from SIGNAL, not from entry
    remaining_timeout = max_bar  # already computed from signal
    exit_bar = None
    exit_price = np.nan
    exit_reason = "TO"
    mfe = 0.0
    mae = 0.0

    for j in range(trigger_bar + 1, remaining_timeout):
        bar = bars_1m.iloc[j]
        bar_high = bar["high"]
        bar_low = bar["low"]

        # Track MFE/MAE
        if triggered_side == "LONG":
            mfe = max(mfe, (bar_high / entry_price - 1) * 10000)
            mae = min(mae, (bar_low / entry_price - 1) * 10000)

            tp_hit = bar_high >= tp_level
            sl_hit = bar_low <= sl_level
        else:
            mfe = max(mfe, (1 - bar_low / entry_price) * 10000)
            mae = min(mae, (1 - bar_high / entry_price) * 10000)

            tp_hit = bar_low <= tp_level
            sl_hit = bar_high >= sl_level

        if tp_hit and sl_hit:
            # Both hit → conservative: SL first
            exit_bar = j
            exit_price = sl_level
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
        # Timeout: exit at close of last bar in window
        last_bar_idx = remaining_timeout - 1
        if last_bar_idx >= trigger_bar + 1 and last_bar_idx < n_bars:
            exit_bar = last_bar_idx
            exit_price = bars_1m.iloc[last_bar_idx]["close"]
            exit_reason = "TO"
        else:
            return result

    # Apply slippage to exit
    if triggered_side == "LONG":
        exit_price *= (1 - slip_bp / 10000)
    else:
        exit_price *= (1 + slip_bp / 10000)

    exit_ts = bars_1m.iloc[exit_bar]["ts"]

    # Compute PnL
    if triggered_side == "LONG":
        gross_pnl_bp = (exit_price / entry_price - 1) * 10000
    else:
        gross_pnl_bp = (1 - exit_price / entry_price) * 10000

    fees = fee_entry_bp + fee_exit_bp
    total_slip = slip_bp * 2  # entry + exit

    result.update({
        "triggered_side": triggered_side,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl_bp": gross_pnl_bp,
        "fees_bp": fees,
        "slippage_bp": total_slip,
        "net_pnl_bp": gross_pnl_bp - fees,
        "mfe_bp": mfe,
        "mae_bp": mae,
        "hold_minutes": int((exit_ts - entry_ts).total_seconds() / 60),
    })

    return result


# ---------------------------------------------------------------------------
# Backtest engine for one symbol
# ---------------------------------------------------------------------------


def backtest_symbol(
    symbol: str,
    signals_df: pd.DataFrame,
    bars_1m: pd.DataFrame,
    atr_series: pd.Series,
    atr_min: float,
) -> list[dict]:
    """Run all parameter configs for a symbol. Returns list of trade dicts."""
    if len(signals_df) == 0 or len(bars_1m) == 0:
        return []

    # Build 1m timestamp → index lookup
    ts_1m = bars_1m["ts"].values.astype("int64")

    # Pre-compute signal → 1m bar mapping
    signal_ts_list = signals_df["ts"].values

    all_trades = []
    param_grid = list(itertools.product(K_ENTRY_GRID, K_SL_GRID, K_TP_GRID, TO_GRID))

    for pi, (k_entry, k_sl, k_tp, timeout) in enumerate(param_grid):
        # Apply cooldown: iterate signals, skip those within cooldown of last used signal
        last_trade_end = pd.Timestamp.min.tz_localize("UTC")
        cooldown_delta = pd.Timedelta(minutes=COOLDOWN_MIN)

        trade_id = 0
        for si, sig_ts in enumerate(signal_ts_list):
            sig_ts_pd = pd.Timestamp(sig_ts, tz="UTC")
            if sig_ts_pd < last_trade_end + cooldown_delta:
                continue

            # Find signal bar in 1m data
            sig_ns = sig_ts_pd.value
            bar_idx = np.searchsorted(ts_1m, sig_ns, side="right") - 1
            if bar_idx < 0 or bar_idx >= len(bars_1m):
                continue

            # Get P0 and ATR at signal time
            P0 = bars_1m.iloc[bar_idx]["close"]
            atr_val = atr_series.iloc[bar_idx]

            if np.isnan(atr_val) or atr_val <= 0:
                continue
            if atr_val < atr_min:
                continue

            # Simulate for each slippage setting
            for slip_bp in SLIPPAGE_GRID_BP:
                trade = simulate_trade(
                    bars_1m=bars_1m,
                    signal_idx=bar_idx,
                    P0=P0,
                    atr=atr_val,
                    k_entry=k_entry,
                    k_sl=k_sl,
                    k_tp=k_tp,
                    timeout_min=timeout,
                    fee_entry_bp=FEE_ENTRY_BP,
                    fee_exit_bp=FEE_EXIT_BP,
                    slip_bp=slip_bp,
                )
                trade["trade_id"] = f"{symbol}_{pi}_{si}_{slip_bp}"
                trade["symbol"] = symbol
                trade["signal_ts"] = sig_ts_pd
                trade["k_entry"] = k_entry
                trade["k_sl"] = k_sl
                trade["k_tp"] = k_tp
                trade["TO"] = timeout
                trade["slip_setting"] = slip_bp
                all_trades.append(trade)

            # Update cooldown based on zero-slip trade (canonical timing)
            last_trade_end = sig_ts_pd

        if (pi + 1) % 18 == 0:
            print(f"    {pi+1}/{len(param_grid)} configs done, {len(all_trades)} trades so far")

    return all_trades


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """§6.2: Aggregate by (symbol, params, slippage)."""
    group_cols = ["symbol", "k_entry", "k_sl", "k_tp", "TO", "slip_setting"]

    rows = []
    for keys, grp in trades_df.groupby(group_cols):
        symbol, k_e, k_s, k_t, to, slip = keys
        n_signals = len(grp)
        triggered = grp[grp["triggered_side"].isin(["LONG", "SHORT"])]
        n_trades = len(triggered)
        trigger_rate = n_trades / n_signals if n_signals > 0 else 0

        if n_trades == 0:
            rows.append({
                **dict(zip(group_cols, keys)),
                "n_signals": n_signals, "n_trades": 0, "trigger_rate": trigger_rate,
                "tp_rate": 0, "sl_rate": 0, "to_rate": 0,
                "median_net_bp": 0, "mean_net_bp": 0,
                "p5_net_bp": 0, "p95_net_bp": 0,
                "profit_factor": 0,
                "avg_win_bp": 0, "avg_loss_bp": 0,
                "expectancy_per_signal_bp": 0,
                "expectancy_per_trade_bp": 0,
                "win_rate": 0,
            })
            continue

        tp_rate = (triggered["exit_reason"] == "TP").mean()
        sl_rate = (triggered["exit_reason"] == "SL").mean()
        to_rate = (triggered["exit_reason"] == "TO").mean()

        net = triggered["net_pnl_bp"]
        wins = net[net > 0]
        losses = net[net <= 0]

        gross_wins = wins.sum() if len(wins) > 0 else 0
        gross_losses = abs(losses.sum()) if len(losses) > 0 else 0
        pf = gross_wins / gross_losses if gross_losses > 0 else (999 if gross_wins > 0 else 0)

        # Expectancy per signal: count NO_TRIGGER/AMBIG as 0
        all_pnl = grp["net_pnl_bp"].copy()
        all_pnl[~grp["triggered_side"].isin(["LONG", "SHORT"])] = 0

        rows.append({
            **dict(zip(group_cols, keys)),
            "n_signals": n_signals,
            "n_trades": n_trades,
            "trigger_rate": trigger_rate,
            "tp_rate": tp_rate,
            "sl_rate": sl_rate,
            "to_rate": to_rate,
            "median_net_bp": net.median(),
            "mean_net_bp": net.mean(),
            "p5_net_bp": net.quantile(0.05),
            "p95_net_bp": net.quantile(0.95),
            "profit_factor": pf,
            "avg_win_bp": wins.mean() if len(wins) > 0 else 0,
            "avg_loss_bp": losses.mean() if len(losses) > 0 else 0,
            "expectancy_per_signal_bp": all_pnl.mean(),
            "expectancy_per_trade_bp": net.mean(),
            "win_rate": (net > 0).mean(),
        })

    return pd.DataFrame(rows)


def build_weekly(trades_df: pd.DataFrame) -> pd.DataFrame:
    """§6.3: Weekly stability."""
    triggered = trades_df[trades_df["triggered_side"].isin(["LONG", "SHORT"])].copy()
    if len(triggered) == 0:
        return pd.DataFrame()

    triggered["week"] = triggered["signal_ts"].dt.isocalendar().week.astype(int)
    triggered["year"] = triggered["signal_ts"].dt.year

    group_cols = ["symbol", "k_entry", "k_sl", "k_tp", "TO", "slip_setting", "year", "week"]
    rows = []
    for keys, grp in triggered.groupby(group_cols):
        net = grp["net_pnl_bp"]
        wins = net[net > 0]
        losses = net[net <= 0]
        gross_wins = wins.sum() if len(wins) > 0 else 0
        gross_losses = abs(losses.sum()) if len(losses) > 0 else 0
        pf = gross_wins / gross_losses if gross_losses > 0 else (999 if gross_wins > 0 else 0)

        rows.append({
            **dict(zip(group_cols, keys)),
            "n_trades": len(grp),
            "mean_net_bp": net.mean(),
            "median_net_bp": net.median(),
            "profit_factor": pf,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------


def walk_forward_analysis(summary_df: pd.DataFrame, trades_df: pd.DataFrame):
    """§9: Simple walk-forward — train on Jan, test on Feb, then swap."""
    triggered = trades_df[trades_df["triggered_side"].isin(["LONG", "SHORT"])].copy()
    if len(triggered) == 0:
        return

    triggered["period"] = np.where(triggered["signal_ts"] < WF_SPLIT, "JAN", "FEB")

    param_cols = ["k_entry", "k_sl", "k_tp", "TO"]

    print(f"\n{'─'*70}")
    print("WALK-FORWARD ANALYSIS (slip=0bp)")
    print(f"{'─'*70}")

    for sym in sorted(triggered["symbol"].unique()):
        sym_df = triggered[(triggered["symbol"] == sym) & (triggered["slip_setting"] == 0)]
        if len(sym_df) == 0:
            continue

        jan = sym_df[sym_df["period"] == "JAN"]
        feb = sym_df[sym_df["period"] == "FEB"]

        if len(jan) == 0 or len(feb) == 0:
            print(f"\n  {sym}: insufficient data for WF (JAN={len(jan)}, FEB={len(feb)})")
            continue

        # Train on JAN: find best config by mean_net_bp
        jan_perf = jan.groupby(param_cols)["net_pnl_bp"].mean().reset_index()
        best_jan = jan_perf.loc[jan_perf["net_pnl_bp"].idxmax()]
        best_params_jan = {c: best_jan[c] for c in param_cols}

        # Test on FEB with JAN-best params
        feb_test = feb
        for c in param_cols:
            feb_test = feb_test[feb_test[c] == best_params_jan[c]]

        # Train on FEB: find best config
        feb_perf = feb.groupby(param_cols)["net_pnl_bp"].mean().reset_index()
        best_feb = feb_perf.loc[feb_perf["net_pnl_bp"].idxmax()]
        best_params_feb = {c: best_feb[c] for c in param_cols}

        # Test on JAN with FEB-best params
        jan_test = jan
        for c in param_cols:
            jan_test = jan_test[jan_test[c] == best_params_feb[c]]

        print(f"\n  {sym}:")
        print(f"    Train=JAN best: k_e={best_params_jan['k_entry']}, k_sl={best_params_jan['k_sl']}, "
              f"k_tp={best_params_jan['k_tp']}, TO={best_params_jan['TO']}")
        if len(feb_test) > 0:
            print(f"    Test=FEB: N={len(feb_test)}, mean={feb_test['net_pnl_bp'].mean():+.1f}bp, "
                  f"WR={( feb_test['net_pnl_bp'] > 0).mean():.3f}, "
                  f"PF={feb_test['net_pnl_bp'][feb_test['net_pnl_bp']>0].sum() / max(abs(feb_test['net_pnl_bp'][feb_test['net_pnl_bp']<=0].sum()), 1):.2f}")
        else:
            print(f"    Test=FEB: no trades")

        print(f"    Train=FEB best: k_e={best_params_feb['k_entry']}, k_sl={best_params_feb['k_sl']}, "
              f"k_tp={best_params_feb['k_tp']}, TO={best_params_feb['TO']}")
        if len(jan_test) > 0:
            print(f"    Test=JAN: N={len(jan_test)}, mean={jan_test['net_pnl_bp'].mean():+.1f}bp, "
                  f"WR={(jan_test['net_pnl_bp'] > 0).mean():.3f}, "
                  f"PF={jan_test['net_pnl_bp'][jan_test['net_pnl_bp']>0].sum() / max(abs(jan_test['net_pnl_bp'][jan_test['net_pnl_bp']<=0].sum()), 1):.2f}")
        else:
            print(f"    Test=JAN: no trades")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("DUAL-STOP BREAKOUT BACKTEST on REG_OI_FUND")
    print(f"Grid: {len(K_ENTRY_GRID)}×{len(K_SL_GRID)}×{len(K_TP_GRID)}×{len(TO_GRID)} "
          f"= {len(list(itertools.product(K_ENTRY_GRID, K_SL_GRID, K_TP_GRID, TO_GRID)))} configs/symbol")
    print(f"Slippage settings: {SLIPPAGE_GRID_BP} bp")
    print(f"Fees: {FEE_ENTRY_BP}+{FEE_EXIT_BP} = {FEE_ENTRY_BP+FEE_EXIT_BP} bp RT")
    print(f"Cooldown: {COOLDOWN_MIN} min")
    print("=" * 80)

    # Load regime dataset
    print("\nLoading regime dataset...")
    regime_df = pd.read_parquet(REGIME_PARQUET)
    regime_df["ts"] = pd.to_datetime(regime_df["ts"], utc=True)
    print(f"  {len(regime_df):,} rows, {regime_df['symbol'].nunique()} symbols")
    print(f"  REG_OI_FUND total: {regime_df['REG_OI_FUND'].sum():.0f}")

    all_trades = []

    for sym_i, sym in enumerate(SYMBOLS, 1):
        sym_t0 = time.monotonic()
        print(f"\n{'─'*70}")
        print(f"[{sym_i}/{len(SYMBOLS)}] {sym}")
        print(f"{'─'*70}")

        # Load 1m mark price bars
        bars_1m = load_mark_1m(sym)
        if len(bars_1m) == 0:
            print(f"  SKIP: no 1m data")
            continue
        print(f"  1m bars: {len(bars_1m):,} ({bars_1m['ts'].min()} → {bars_1m['ts'].max()})")

        # Compute ATR
        atr_series = compute_atr_series(bars_1m)
        valid_atr = atr_series.dropna()
        if len(valid_atr) == 0:
            print(f"  SKIP: no valid ATR")
            continue
        atr_min = valid_atr.quantile(ATR_MIN_PERCENTILE / 100)
        print(f"  ATR_15: median={valid_atr.median():.6f}, P5={atr_min:.6f}")

        # Load signals
        signals = load_signals(sym, regime_df)
        print(f"  REG_OI_FUND signals: {len(signals)}")

        if len(signals) == 0:
            print(f"  SKIP: no signals")
            continue

        # Run backtest
        print(f"  Running {len(list(itertools.product(K_ENTRY_GRID, K_SL_GRID, K_TP_GRID, TO_GRID)))} configs × {len(SLIPPAGE_GRID_BP)} slip × ~{len(signals)} signals...")
        trades = backtest_symbol(sym, signals, bars_1m, atr_series, atr_min)
        all_trades.extend(trades)

        sym_elapsed = time.monotonic() - sym_t0
        triggered = sum(1 for t in trades if t["triggered_side"] in ("LONG", "SHORT"))
        print(f"  {len(trades)} trade records ({triggered} triggered) in {sym_elapsed:.1f}s")

    if not all_trades:
        print("\nNo trades generated. Exiting.")
        return

    # Build DataFrame
    trades_df = pd.DataFrame(all_trades)
    trades_df["signal_ts"] = pd.to_datetime(trades_df["signal_ts"], utc=True)
    for col in ["entry_ts", "exit_ts"]:
        trades_df[col] = pd.to_datetime(trades_df[col], utc=True)

    # Save trades
    trades_df.to_parquet(OUTPUT_DIR / "breakout_trades.parquet", index=False)
    print(f"\nSaved {len(trades_df):,} trade records → breakout_trades.parquet")

    # ===================================================================
    # Summary report
    # ===================================================================
    summary = build_summary(trades_df)
    summary.to_csv(OUTPUT_DIR / "breakout_report.csv", index=False)

    # Print top configs per symbol (slip=0)
    print(f"\n{'='*80}")
    print("TOP CONFIGS PER SYMBOL (slip=0bp, sorted by expectancy/signal)")
    print("=" * 80)

    for sym in sorted(summary["symbol"].unique()):
        sym_s = summary[(summary["symbol"] == sym) & (summary["slip_setting"] == 0)]
        sym_s = sym_s.sort_values("expectancy_per_signal_bp", ascending=False)
        top = sym_s.head(5)

        print(f"\n  {sym}:")
        print(f"  {'k_e':>4} {'k_sl':>4} {'k_tp':>4} {'TO':>3} | {'N_sig':>5} {'N_trd':>5} {'Trig%':>5} "
              f"{'TP%':>5} {'SL%':>5} {'WR':>5} | {'Med_net':>8} {'Mean_net':>8} {'PF':>5} | {'E/sig':>7} {'P5':>7} {'P95':>7}")
        for _, r in top.iterrows():
            print(f"  {r['k_entry']:>4.1f} {r['k_sl']:>4.1f} {r['k_tp']:>4.1f} {r['TO']:>3.0f} | "
                  f"{r['n_signals']:>5.0f} {r['n_trades']:>5.0f} {r['trigger_rate']:>5.1%} "
                  f"{r['tp_rate']:>5.1%} {r['sl_rate']:>5.1%} {r['win_rate']:>5.1%} | "
                  f"{r['median_net_bp']:>+8.1f} {r['mean_net_bp']:>+8.1f} {r['profit_factor']:>5.2f} | "
                  f"{r['expectancy_per_signal_bp']:>+7.1f} {r['p5_net_bp']:>+7.1f} {r['p95_net_bp']:>+7.1f}")

    # ===================================================================
    # Slippage sensitivity
    # ===================================================================
    print(f"\n{'='*80}")
    print("SLIPPAGE SENSITIVITY (best config per symbol, all slippage levels)")
    print("=" * 80)

    for sym in sorted(summary["symbol"].unique()):
        # Find best config at slip=0
        sym_s0 = summary[(summary["symbol"] == sym) & (summary["slip_setting"] == 0)]
        if len(sym_s0) == 0:
            continue
        best = sym_s0.loc[sym_s0["expectancy_per_signal_bp"].idxmax()]
        params = {
            "k_entry": best["k_entry"],
            "k_sl": best["k_sl"],
            "k_tp": best["k_tp"],
            "TO": best["TO"],
        }

        print(f"\n  {sym} (k_e={params['k_entry']}, k_sl={params['k_sl']}, "
              f"k_tp={params['k_tp']}, TO={params['TO']:.0f}):")
        print(f"  {'Slip':>5} {'N_trd':>5} {'Mean_net':>9} {'PF':>6} {'WR':>5} {'E/sig':>8}")

        for slip in SLIPPAGE_GRID_BP:
            mask = (
                (summary["symbol"] == sym) &
                (summary["k_entry"] == params["k_entry"]) &
                (summary["k_sl"] == params["k_sl"]) &
                (summary["k_tp"] == params["k_tp"]) &
                (summary["TO"] == params["TO"]) &
                (summary["slip_setting"] == slip)
            )
            row = summary[mask]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"  {slip:>5} {r['n_trades']:>5.0f} {r['mean_net_bp']:>+9.1f} "
                  f"{r['profit_factor']:>6.2f} {r['win_rate']:>5.1%} {r['expectancy_per_signal_bp']:>+8.1f}")

    # ===================================================================
    # Weekly stability
    # ===================================================================
    weekly = build_weekly(trades_df)
    if len(weekly) > 0:
        weekly.to_csv(OUTPUT_DIR / "breakout_weekly.csv", index=False)

        print(f"\n{'='*80}")
        print("WEEKLY STABILITY (best config per symbol, slip=0)")
        print("=" * 80)

        for sym in sorted(summary["symbol"].unique()):
            sym_s0 = summary[(summary["symbol"] == sym) & (summary["slip_setting"] == 0)]
            if len(sym_s0) == 0:
                continue
            best = sym_s0.loc[sym_s0["expectancy_per_signal_bp"].idxmax()]

            sym_w = weekly[
                (weekly["symbol"] == sym) &
                (weekly["k_entry"] == best["k_entry"]) &
                (weekly["k_sl"] == best["k_sl"]) &
                (weekly["k_tp"] == best["k_tp"]) &
                (weekly["TO"] == best["TO"]) &
                (weekly["slip_setting"] == 0)
            ].sort_values(["year", "week"])

            if len(sym_w) == 0:
                continue

            pos_weeks = (sym_w["mean_net_bp"] > 0).sum()
            total_weeks = len(sym_w)

            print(f"\n  {sym} ({pos_weeks}/{total_weeks} positive weeks):")
            print(f"  {'Wk':>4} {'N':>3} {'Mean_net':>9} {'Med_net':>9} {'PF':>6}")
            for _, r in sym_w.iterrows():
                print(f"  W{r['week']:>3.0f} {r['n_trades']:>3.0f} {r['mean_net_bp']:>+9.1f} "
                      f"{r['median_net_bp']:>+9.1f} {r['profit_factor']:>6.2f}")

    # ===================================================================
    # Walk-forward
    # ===================================================================
    walk_forward_analysis(summary, trades_df)

    # ===================================================================
    # §7: Go/No-Go assessment
    # ===================================================================
    print(f"\n{'='*80}")
    print("GO / NO-GO ASSESSMENT (§7)")
    print("=" * 80)

    for sym in sorted(summary["symbol"].unique()):
        sym_s0 = summary[(summary["symbol"] == sym) & (summary["slip_setting"] == 0)]
        if len(sym_s0) == 0:
            continue
        best = sym_s0.loc[sym_s0["expectancy_per_signal_bp"].idxmax()]

        # Check criteria
        exp_ok = best["expectancy_per_signal_bp"] > 0
        pf_ok = best["profit_factor"] > 1.1
        p5_ok = best["p5_net_bp"] > -250

        # Weekly stability
        sym_w = weekly[
            (weekly["symbol"] == sym) &
            (weekly["k_entry"] == best["k_entry"]) &
            (weekly["k_sl"] == best["k_sl"]) &
            (weekly["k_tp"] == best["k_tp"]) &
            (weekly["TO"] == best["TO"]) &
            (weekly["slip_setting"] == 0)
        ] if len(weekly) > 0 else pd.DataFrame()
        pos_weeks = (sym_w["mean_net_bp"] > 0).sum() if len(sym_w) > 0 else 0
        total_weeks = len(sym_w)
        stable = pos_weeks >= 6 if total_weeks >= 8 else pos_weeks >= total_weeks * 0.6

        # Survive slip=5?
        slip5 = summary[
            (summary["symbol"] == sym) &
            (summary["k_entry"] == best["k_entry"]) &
            (summary["k_sl"] == best["k_sl"]) &
            (summary["k_tp"] == best["k_tp"]) &
            (summary["TO"] == best["TO"]) &
            (summary["slip_setting"] == 5)
        ]
        slip5_ok = len(slip5) > 0 and slip5.iloc[0]["expectancy_per_signal_bp"] > 0

        verdict = "GO" if (exp_ok and pf_ok and p5_ok and stable) else "NO-GO"
        note = ""
        if not exp_ok: note += " E/sig≤0"
        if not pf_ok: note += " PF≤1.1"
        if not p5_ok: note += " P5<-250"
        if not stable: note += f" unstable({pos_weeks}/{total_weeks}wk)"
        if not slip5_ok: note += " dies@5bp_slip"

        print(f"  {sym:<18} {verdict:<6}  E/sig={best['expectancy_per_signal_bp']:+.1f}bp  "
              f"PF={best['profit_factor']:.2f}  P5={best['p5_net_bp']:+.0f}bp  "
              f"weeks={pos_weeks}/{total_weeks}  slip5={'OK' if slip5_ok else 'FAIL'}{note}")

    elapsed = time.monotonic() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
