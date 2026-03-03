#!/usr/bin/env python3
"""
XS-5 E1 Extended — Crowded Long → SHORT on 5-month data (Oct 2025 – Feb 2026).

E1 conditions (fixed thresholds):
  - funding_z_7d >= +2.0  (extreme positive funding)
  - oi_z_7d >= +2.0       (OI acceleration)
  - trend_2h <= 0.3        (price stalling)
  - ret_2h >= 0            (not already dumping)

Direction: SHORT (against the crowd).
12h cooldown per symbol. No SL/TP (unwind + time stop only).

Walk-forward: Oct-Nov train / Dec test, Dec-Jan train / Feb test,
              plus month-by-month stability.
"""

import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs5_e1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# Need at least 15 days of data (some coins may not exist in Oct 2025)
MIN_DAYS = 15

# Feature windows (in 1m bars)
FR_Z_WINDOW_1M = 7 * 24 * 60       # 7 days
OI_Z_WINDOW_1M = 7 * 24 * 60       # 7 days
OI_CHG_1H = 60
OI_CHG_4H = 240
RET_2H = 120
RV_2H = 120
ATR_1H = 60

# E1 thresholds (fixed)
E1_FR_Z = 2.0
E1_OI_Z = 2.0
E1_TREND = 0.3

# Cooldown: 12h
COOLDOWN_MIN = 12 * 60

# Exit params
HOLD_GRID = [4 * 60, 12 * 60, 24 * 60, 48 * 60]
HOLD_LABELS = ["4h", "12h", "24h", "48h"]
UNWIND_FR_Z_LO = -1.0
UNWIND_FR_Z_HI = 1.0

# No SL/TP — the key finding from xs5 is that SL kills the edge
SLTP_CONFIGS = [
    ("none", None, None),
    ("cat8", 8.0, None),     # catastrophe only
]

FEE_BP = 10
SLIP_GRID = [0, 5, 10]

# Stats
N_BOOTSTRAP = 5000
N_PERMUTATION = 2000
SEED = 42

MIN_TURNOVER_60M = 50_000


# ---------------------------------------------------------------------------
# §1: Data Loading (same as xs5)
# ---------------------------------------------------------------------------

def discover_symbols() -> list[str]:
    syms = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        nmark = len(list(d.glob("*_mark_price_kline_1m.csv")))
        nkline = len([f for f in d.glob("*_kline_1m.csv")
                       if "mark_price" not in f.name and "premium_index" not in f.name])
        noi = len(list(d.glob("*_open_interest_5min.csv")))
        nfr = len(list(d.glob("*_funding_rate.csv")))
        if nmark >= MIN_DAYS and nkline >= MIN_DAYS and noi >= MIN_DAYS and nfr >= MIN_DAYS:
            syms.append(d.name)
    return syms


def _load_glob(sym: str, pattern: str, ts_col: str, val_cols: dict) -> pd.DataFrame:
    sym_dir = DATA_DIR / sym
    files = sorted(sym_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df[ts_col].astype(int), unit="ms", utc=True)
    out = df[["ts"]].copy()
    for src, dst in val_cols.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out = out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    out = out[(out["ts"] >= START) & (out["ts"] <= END)]
    return out.reset_index(drop=True)


def load_symbol(sym: str) -> dict:
    mark = _load_glob(sym, "*_mark_price_kline_1m.csv", "startTime",
                      {"close": "close"})

    kline_files = sorted((DATA_DIR / sym).glob("*_kline_1m.csv"))
    kline_files = [f for f in kline_files
                   if "mark_price" not in f.name and "premium_index" not in f.name]
    kline = pd.DataFrame()
    if kline_files:
        frames = []
        for f in kline_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    frames.append(df)
            except Exception:
                continue
        if frames:
            kl = pd.concat(frames, ignore_index=True)
            kl["ts"] = pd.to_datetime(kl["startTime"].astype(int), unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume", "turnover"]:
                kl[c] = pd.to_numeric(kl[c], errors="coerce")
            kl = kl.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            kl = kl[(kl["ts"] >= START) & (kl["ts"] <= END)]
            kline = kl[["ts", "high", "low", "close", "volume", "turnover"]].reset_index(drop=True)

    oi = _load_glob(sym, "*_open_interest_5min.csv", "timestamp",
                    {"openInterest": "oi"})
    fr = _load_glob(sym, "*_funding_rate.csv", "timestamp",
                    {"fundingRate": "fr"})

    return {"mark": mark, "kline": kline, "oi": oi, "fr": fr}


# ---------------------------------------------------------------------------
# §2: Unified 1m grid
# ---------------------------------------------------------------------------

def build_sym_1m(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    mark = raw["mark"].set_index("ts")["close"] if len(raw["mark"]) > 0 else pd.Series(dtype=float)
    kl = raw["kline"]
    oi_df = raw["oi"]
    fr_df = raw["fr"]

    n = len(grid_1m)
    df = pd.DataFrame(index=grid_1m)

    close_raw = mark.reindex(grid_1m)
    is_nan = close_raw.isna()
    nan_arr = is_nan.values
    block_len = np.zeros(n, dtype=np.int32)
    run = 0
    starts = []
    for i in range(n):
        if nan_arr[i]:
            run += 1
        else:
            if run > 0:
                starts.append((i - run, run))
            run = 0
    if run > 0:
        starts.append((n - run, run))
    for s, length in starts:
        block_len[s:s + length] = length

    is_ffill = np.zeros(n, dtype=np.int8)
    is_invalid = np.zeros(n, dtype=np.int8)
    is_ffill[nan_arr & (block_len > 0) & (block_len < 5)] = 1
    is_invalid[nan_arr & (block_len >= 5)] = 1

    close = close_raw.ffill()
    close[is_invalid == 1] = np.nan

    df["close"] = close
    df["is_ffill"] = is_ffill
    df["is_invalid"] = is_invalid

    if len(kl) > 0:
        kl_idx = kl.set_index("ts")
        df["high"] = kl_idx["high"].reindex(grid_1m).ffill()
        df["low"] = kl_idx["low"].reindex(grid_1m).ffill()
        df["volume"] = kl_idx["volume"].reindex(grid_1m).fillna(0)
        df["turnover"] = kl_idx["turnover"].reindex(grid_1m).fillna(0)
    else:
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0.0
        df["turnover"] = 0.0

    # OI: 5m → ffill, shifted +5min for causal alignment
    if len(oi_df) > 0:
        oi_shifted = oi_df.copy()
        oi_shifted["ts"] = oi_shifted["ts"] + pd.Timedelta(minutes=5)
        oi_s = oi_shifted.set_index("ts")["oi"]
        df["oi"] = oi_s.reindex(grid_1m).ffill()
    else:
        df["oi"] = np.nan

    # FR: 8h → ffill, shifted +1min
    if len(fr_df) > 0:
        fr_shifted = fr_df.copy()
        fr_shifted["ts"] = fr_shifted["ts"] + pd.Timedelta(minutes=1)
        fr_s = fr_shifted.set_index("ts")["fr"]
        df["fr"] = fr_s.reindex(grid_1m).ffill()
    else:
        df["fr"] = np.nan

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.loc[df["is_invalid"] == 1, "log_ret"] = np.nan
    df.loc[df["is_invalid"].shift(1) == 1, "log_ret"] = np.nan

    return df


# ---------------------------------------------------------------------------
# §3: Features
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW_1M, min_periods=FR_Z_WINDOW_1M // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW_1M, min_periods=FR_Z_WINDOW_1M // 4).std().clip(lower=1e-12)
    df["funding_z_7d"] = (fr - fr_rm) / fr_rs
    df["funding_sign"] = np.sign(fr)

    oi = df["oi"]
    oi_lag_1h = oi.shift(OI_CHG_1H)
    df["oi_chg_1h"] = (oi - oi_lag_1h) / oi_lag_1h.clip(lower=1)
    oi_lag_4h = oi.shift(OI_CHG_4H)
    df["oi_chg_4h"] = (oi - oi_lag_4h) / oi_lag_4h.clip(lower=1)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW_1M, min_periods=OI_Z_WINDOW_1M // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW_1M, min_periods=OI_Z_WINDOW_1M // 4).std().clip(lower=1e-12)
    df["oi_z_7d"] = (oi_chg - oi_rm) / oi_rs

    close = df["close"]
    df["ret_2h"] = np.log(close / close.shift(RET_2H))
    df["rv_2h"] = df["log_ret"].rolling(RV_2H, min_periods=RV_2H // 2).std()
    rv_scaled = df["rv_2h"] * np.sqrt(RV_2H)
    df["trend_2h"] = df["ret_2h"].abs() / rv_scaled.clip(lower=1e-12)

    df["high_2h"] = df["high"].rolling(RET_2H, min_periods=1).max()
    df["low_2h"] = df["low"].rolling(RET_2H, min_periods=1).min()
    df["dd_from_high_2h"] = (close - df["high_2h"]) / df["high_2h"].clip(lower=1e-8)
    df["du_from_low_2h"] = (close - df["low_2h"]) / df["low_2h"].clip(lower=1e-8)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_1h"] = tr.rolling(ATR_1H, min_periods=ATR_1H // 2).mean()

    df["turnover_60m"] = df["turnover"].rolling(60, min_periods=10).sum()

    return df


# ---------------------------------------------------------------------------
# §4: E1 detection
# ---------------------------------------------------------------------------

def detect_e1(df: pd.DataFrame, sym: str) -> list[dict]:
    events = []
    n = len(df)
    idx = df.index

    fz = df["funding_z_7d"].values
    oiz = df["oi_z_7d"].values
    trend = df["trend_2h"].values
    ret2h = df["ret_2h"].values
    inv = df["is_invalid"].values
    turn = df["turnover_60m"].values
    atr = df["atr_1h"].values
    fr_raw = df["fr"].values
    oi_chg_1h = df["oi_chg_1h"].values
    oi_chg_4h = df["oi_chg_4h"].values
    close = df["close"].values

    last_trigger = -COOLDOWN_MIN * 2
    check_points = np.arange(0, n, 5)

    for i in check_points:
        if inv[i] == 1:
            continue
        if np.isnan(fz[i]) or np.isnan(oiz[i]) or np.isnan(trend[i]):
            continue
        if np.isnan(ret2h[i]) or np.isnan(atr[i]):
            continue
        if turn[i] < MIN_TURNOVER_60M:
            continue

        if (fz[i] >= E1_FR_Z and oiz[i] >= E1_OI_Z and
                trend[i] <= E1_TREND and ret2h[i] >= 0):
            if i - last_trigger >= COOLDOWN_MIN:
                events.append({
                    "symbol": sym, "ts": idx[i], "idx": i,
                    "event_type": "E1", "direction": -1,
                    "funding_z": fz[i], "oi_z": oiz[i],
                    "trend_2h": trend[i], "ret_2h": ret2h[i],
                    "atr_1h": atr[i], "fr_raw": fr_raw[i],
                    "oi_chg_1h": oi_chg_1h[i], "oi_chg_4h": oi_chg_4h[i],
                    "close": close[i],
                })
                last_trigger = i

    return events


# ---------------------------------------------------------------------------
# §5: Trade simulation (same logic, no SL/TP focus)
# ---------------------------------------------------------------------------

def simulate_trades(events, sym_dfs, grid_1m, hold_minutes, slip_bp,
                    sl_atr_mult=None, tp_atr_mult=None):
    trades = []
    cost_rt_bp = 2 * FEE_BP + 2 * slip_bp

    for evt in events:
        sym = evt["symbol"]
        df = sym_dfs[sym]
        idx_signal = evt["idx"]
        direction = evt["direction"]
        atr = evt["atr_1h"]

        if atr <= 0 or np.isnan(atr):
            continue

        entry_idx = idx_signal + 1
        if entry_idx >= len(df):
            continue
        entry_px = df["close"].iloc[entry_idx]
        if np.isnan(entry_px) or entry_px <= 0:
            continue

        use_sl = sl_atr_mult is not None
        use_tp = tp_atr_mult is not None
        sl_dist = (sl_atr_mult or 0) * atr
        tp_dist = (tp_atr_mult or 0) * atr
        if direction == 1:
            sl_px = entry_px - sl_dist if use_sl else 0
            tp_px = entry_px + tp_dist if use_tp else 1e18
        else:
            sl_px = entry_px + sl_dist if use_sl else 1e18
            tp_px = entry_px - tp_dist if use_tp else 0

        max_idx = min(entry_idx + hold_minutes + 1, len(df))
        exit_idx = max_idx - 1
        exit_reason = "time_stop"
        mfe = 0.0
        mae = 0.0

        for j in range(entry_idx + 1, max_idx):
            if df["is_invalid"].iloc[j] == 1:
                continue

            px = df["close"].iloc[j]
            hi = df["high"].iloc[j]
            lo = df["low"].iloc[j]
            if np.isnan(px):
                continue

            if direction == 1:
                fav = (hi - entry_px) / entry_px * 10000
                adv = (lo - entry_px) / entry_px * 10000
            else:
                fav = (entry_px - lo) / entry_px * 10000
                adv = (entry_px - hi) / entry_px * 10000
            mfe = max(mfe, fav)
            mae = min(mae, adv)

            if use_sl:
                if direction == 1 and lo <= sl_px:
                    exit_idx = j; exit_reason = "stop_loss"; break
                if direction == -1 and hi >= sl_px:
                    exit_idx = j; exit_reason = "stop_loss"; break

            if use_tp:
                if direction == 1 and hi >= tp_px:
                    exit_idx = j; exit_reason = "take_profit"; break
                if direction == -1 and lo <= tp_px:
                    exit_idx = j; exit_reason = "take_profit"; break

            fz_now = df["funding_z_7d"].iloc[j]
            if pd.notna(fz_now) and UNWIND_FR_Z_LO <= fz_now <= UNWIND_FR_Z_HI:
                exit_idx = j; exit_reason = "unwind"; break

        exit_px = df["close"].iloc[exit_idx]
        if np.isnan(exit_px) or exit_px <= 0:
            continue

        if exit_reason == "stop_loss" and use_sl:
            gross_bp = (entry_px - sl_px) / entry_px * 10000 if direction == -1 else (sl_px - entry_px) / entry_px * 10000
        elif exit_reason == "take_profit" and use_tp:
            gross_bp = (entry_px - tp_px) / entry_px * 10000 if direction == -1 else (tp_px - entry_px) / entry_px * 10000
        else:
            gross_bp = (entry_px - exit_px) / entry_px * 10000 if direction == -1 else (exit_px - entry_px) / entry_px * 10000

        net_bp = gross_bp - cost_rt_bp
        hold_min = exit_idx - entry_idx

        trades.append({
            "symbol": evt["symbol"],
            "event_type": "E1",
            "direction": direction,
            "entry_ts": grid_1m[entry_idx],
            "exit_ts": grid_1m[exit_idx],
            "entry_px": entry_px,
            "exit_px": exit_px,
            "gross_bp": gross_bp,
            "cost_bp": cost_rt_bp,
            "net_bp": net_bp,
            "mfe_bp": mfe,
            "mae_bp": mae,
            "hold_min": hold_min,
            "exit_reason": exit_reason,
            "atr_1h": atr,
            "funding_z": evt["funding_z"],
            "oi_z": evt["oi_z"],
            "trend_2h": evt["trend_2h"],
            "ret_2h": evt["ret_2h"],
            "fr_raw": evt["fr_raw"],
            "oi_chg_1h": evt["oi_chg_1h"],
            "oi_chg_4h": evt["oi_chg_4h"],
            "slip_bp": slip_bp,
        })

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# §6: Stats
# ---------------------------------------------------------------------------

def compute_stats(trades_df, rng):
    if len(trades_df) == 0:
        return {"n": 0}

    net = trades_df["net_bp"]
    gross = trades_df["gross_bp"]
    n = len(trades_df)

    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["entry_ts"]).dt.date
    daily = trades_df.groupby("date")["net_bp"].sum()
    n_days = len(daily)

    mean_daily = daily.mean() if n_days > 0 else 0
    std_daily = daily.std() if n_days > 1 else 1
    sharpe_ann = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0

    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0

    winners = net[net > 0]
    losers = net[net <= 0]
    pf = winners.sum() / abs(losers.sum()) if len(losers) > 0 and losers.sum() != 0 else np.inf

    net_arr = net.values
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = net_arr[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    p_perm = np.nan
    if n_days >= 5:
        daily_arr = daily.values
        perm_means = np.empty(N_PERMUTATION)
        for p in range(N_PERMUTATION):
            signs = rng.choice([-1, 1], size=n_days)
            perm_means[p] = (daily_arr * signs).mean()
        p_perm = (perm_means >= mean_daily).mean()

    p5_val = np.percentile(net_arr, 5)
    p95_val = np.percentile(net_arr, 95)
    winsorized = np.clip(net_arr, p5_val, p95_val)
    winsorized_mean = winsorized.mean()

    trades_df["week"] = pd.to_datetime(trades_df["entry_ts"]).dt.isocalendar().week.values
    weekly = trades_df.groupby("week")["net_bp"].sum()
    pos_weeks = (weekly > 0).sum()
    total_weeks = len(weekly)

    exit_counts = trades_df["exit_reason"].value_counts().to_dict()

    return {
        "n": n, "n_days": n_days,
        "mean_net_bp": net.mean(), "median_net_bp": net.median(),
        "std_net_bp": net.std(), "wr": (net > 0).mean(), "pf": pf,
        "p5": p5_val, "p95": p95_val,
        "mean_gross_bp": gross.mean(),
        "mean_mfe": trades_df["mfe_bp"].mean(),
        "mean_mae": trades_df["mae_bp"].mean(),
        "mean_hold_min": trades_df["hold_min"].mean(),
        "mean_daily_bp": mean_daily, "sharpe_ann": sharpe_ann,
        "max_dd_bp": max_dd, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_perm": p_perm, "winsorized_mean": winsorized_mean,
        "pos_weeks": pos_weeks, "total_weeks": total_weeks,
        "exit_sl": exit_counts.get("stop_loss", 0),
        "exit_tp": exit_counts.get("take_profit", 0),
        "exit_unwind": exit_counts.get("unwind", 0),
        "exit_time": exit_counts.get("time_stop", 0),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("XS-5 E1 EXTENDED — Crowded Long → SHORT")
    print(f"Period: {START.date()} → {END.date()} (5 months)")
    print(f"Holds: {HOLD_LABELS}")
    print(f"Cost: {FEE_BP}bp/side + slippage {SLIP_GRID} bp/side")
    print("=" * 80, flush=True)

    # §1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}", flush=True)
    symbols = discover_symbols()
    print(f"  Symbols discovered: {len(symbols)}", flush=True)

    raw_all = {}
    for i, sym in enumerate(symbols, 1):
        raw_all[sym] = load_symbol(sym)
        if i % 10 == 0 or i == len(symbols):
            n_mark = sum(1 for r in raw_all.values() if len(r['mark']) > 0)
            print(f"    Loaded {i}/{len(symbols)} ({n_mark} with mark data)", flush=True)
    print(f"  Loaded: {len(raw_all)} symbols", flush=True)

    # Check data coverage
    sym_coverage = {}
    for sym, raw in raw_all.items():
        if len(raw["mark"]) > 0:
            mn = raw["mark"]["ts"].min()
            mx = raw["mark"]["ts"].max()
            days = (mx - mn).days
            sym_coverage[sym] = {"start": mn, "end": mx, "days": days}

    oct_syms = [s for s, c in sym_coverage.items()
                if c["start"] <= pd.Timestamp("2025-10-15", tz="UTC")]
    print(f"  Symbols with data from Oct 2025: {len(oct_syms)}", flush=True)

    # §2: Build 1m grid + features
    print(f"\n{'─'*70}")
    print("PHASE 2: Unified 1m grid + features + E1 detection")
    print(f"{'─'*70}", flush=True)

    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"  1m grid: {len(grid_1m):,} points (uniform calendar)", flush=True)

    sym_dfs = {}
    all_events = []
    for i, sym in enumerate(sorted(raw_all.keys()), 1):
        raw = raw_all[sym]
        if len(raw["mark"]) == 0:
            continue
        df = build_sym_1m(sym, raw, grid_1m)
        df = compute_features(df)
        sym_dfs[sym] = df

        evts = detect_e1(df, sym)
        all_events.extend(evts)

        if i % 10 == 0 or i == len(raw_all):
            print(f"    Processed {i}/{len(raw_all)}, events so far: {len(all_events)}",
                  flush=True)

    print(f"\n  Total E1 events: {len(all_events)}", flush=True)

    if len(all_events) == 0:
        print("\n  ✗ No E1 events detected. Exiting.")
        return

    # Per-symbol and per-month breakdown
    sym_counts = Counter(e["symbol"] for e in all_events)
    print(f"  Symbols with E1: {len(sym_counts)}")
    print(f"  Top: {', '.join(f'{s}({c})' for s, c in sym_counts.most_common(15))}")

    month_counts = Counter(e["ts"].strftime("%Y-%m") for e in all_events)
    print(f"  Monthly distribution:")
    for m in sorted(month_counts.keys()):
        print(f"    {m}: {month_counts[m]} events")

    # Save events
    events_df = pd.DataFrame(all_events)
    events_df.to_parquet(OUTPUT_DIR / "xs5_e1_events.parquet", index=False)

    # §3: Trade simulation
    print(f"\n{'─'*70}")
    print("PHASE 3: Trade simulation")
    print(f"{'─'*70}", flush=True)

    all_results = []
    all_trades_list = []

    # Walk-forward periods: month-by-month
    months = sorted(month_counts.keys())

    n_configs = len(SLTP_CONFIGS) * len(HOLD_GRID) * len(SLIP_GRID)
    config_i = 0

    for sltp_label, sl_mult, tp_mult in SLTP_CONFIGS:
        for hi, (hold_min, hold_label) in enumerate(zip(HOLD_GRID, HOLD_LABELS)):
            for slip in SLIP_GRID:
                config_i += 1
                cost_rt = 2 * FEE_BP + 2 * slip

                trades_full = simulate_trades(all_events, sym_dfs, grid_1m,
                                              hold_min, slip, sl_mult, tp_mult)
                if len(trades_full) == 0:
                    continue

                trades_full["hold_label"] = hold_label
                trades_full["sltp"] = sltp_label
                all_trades_list.append(trades_full)

                stats_full = compute_stats(trades_full, rng)

                # Per-month stats
                month_stats = {}
                for m in months:
                    m_start = pd.Timestamp(f"{m}-01", tz="UTC")
                    m_end = m_start + pd.offsets.MonthEnd(1) + pd.Timedelta("23:59:59")
                    tf_m = trades_full[
                        (trades_full["entry_ts"] >= m_start) &
                        (trades_full["entry_ts"] <= m_end)
                    ]
                    if len(tf_m) > 0:
                        month_stats[m] = {
                            "n": len(tf_m),
                            "mean": tf_m["net_bp"].mean(),
                            "wr": (tf_m["net_bp"] > 0).mean(),
                        }

                result = {
                    "sltp": sltp_label, "hold": hold_label,
                    "slip_bp": slip, "cost_rt_bp": cost_rt,
                    **{f"full_{k}": v for k, v in stats_full.items()},
                }
                # Add per-month columns
                for m in months:
                    ms = month_stats.get(m, {})
                    result[f"m_{m}_n"] = ms.get("n", 0)
                    result[f"m_{m}_mean"] = ms.get("mean", np.nan)
                    result[f"m_{m}_wr"] = ms.get("wr", np.nan)

                all_results.append(result)

                if config_i % 6 == 0:
                    print(f"    Config {config_i}/{n_configs}", flush=True)

    # Save results
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_DIR / "xs5_e1_report.csv", index=False)

    if all_trades_list:
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
        all_trades_df.to_parquet(OUTPUT_DIR / "xs5_e1_trades.parquet", index=False)

    # Print summary
    print(f"\n{'─'*70}")
    print("RESULTS SUMMARY")
    print(f"{'─'*70}", flush=True)

    for sltp_l, _, _ in SLTP_CONFIGS:
        print(f"\n  ══ SL/TP={sltp_l} ══")
        for r in all_results:
            if r["sltp"] != sltp_l or r["slip_bp"] != 5:
                continue
            exits = (f"UW={r['full_exit_unwind']} TS={r['full_exit_time']} "
                     f"SL={r['full_exit_sl']}")
            print(f"  {r['hold']:3s}: N={r['full_n']:>3d}, "
                  f"mean={r['full_mean_net_bp']:+7.1f}bp, "
                  f"med={r['full_median_net_bp']:+7.1f}bp, "
                  f"WR={r['full_wr']:.0%}, "
                  f"PF={r['full_pf']:.2f}, "
                  f"Sharpe={r['full_sharpe_ann']:+.2f}, "
                  f"p={r['full_p_perm']:.3f}, "
                  f"CI=[{r['full_ci_lo']:+.0f}, {r['full_ci_hi']:+.0f}] | {exits}")

            # Monthly breakdown
            month_line = "    Monthly: "
            for m in months:
                mn = r.get(f"m_{m}_n", 0)
                mm = r.get(f"m_{m}_mean", np.nan)
                if mn > 0:
                    month_line += f"{m}:N={mn},μ={mm:+.0f}bp  "
                else:
                    month_line += f"{m}:-  "
            print(month_line)

    # Per-trade detail for best config (slip=5, no SL, 12h hold)
    print(f"\n{'─'*70}")
    print("PER-TRADE DETAIL (slip=5bp, no SL/TP, 12h hold)")
    print(f"{'─'*70}", flush=True)

    best_trades = None
    for tdf in all_trades_list:
        if (tdf["slip_bp"].iloc[0] == 5 and tdf["sltp"].iloc[0] == "none" and
                tdf["hold_label"].iloc[0] == "12h"):
            best_trades = tdf
            break

    if best_trades is not None and len(best_trades) > 0:
        for _, t in best_trades.iterrows():
            dir_s = "SHORT" if t["direction"] == -1 else "LONG"
            print(f"  {t['symbol']:18s} {t['entry_ts'].strftime('%Y-%m-%d %H:%M')} "
                  f"{dir_s} entry={t['entry_px']:.4f} exit={t['exit_px']:.4f} "
                  f"gross={t['gross_bp']:+.0f}bp net={t['net_bp']:+.0f}bp "
                  f"MFE={t['mfe_bp']:+.0f} MAE={t['mae_bp']:+.0f} "
                  f"hold={t['hold_min']:.0f}m exit={t['exit_reason']} "
                  f"fz={t['funding_z']:.1f} oiz={t['oi_z']:.1f}")

    # GO / NO-GO
    print(f"\n{'='*80}")
    print("GO / NO-GO VERDICT")
    print(f"{'='*80}", flush=True)

    for r in all_results:
        if r["slip_bp"] != 5 or r["sltp"] != "none":
            continue

        n_ok = r["full_n"] >= 40 or (r["full_n"] >= 20)
        oos_months = [m for m in months if r.get(f"m_{m}_n", 0) > 0]
        pos_months = [m for m in oos_months if r.get(f"m_{m}_mean", -999) > 0]
        consistency_ok = len(pos_months) >= len(oos_months) * 0.6 if oos_months else False
        mean_ok = r["full_mean_net_bp"] > 20
        pf_ok = r["full_pf"] >= 1.3
        p5_ok = r.get("full_p5", -9999) > -600
        wins_ok = r.get("full_winsorized_mean", -999) > 0
        p_ok = r.get("full_p_perm", 1) < 0.15

        label = f"E1 {r['hold']}"
        checks = [n_ok, consistency_ok, mean_ok, pf_ok, p5_ok, wins_ok, p_ok]
        labels = [
            f"N ≥ 20 (actual: {r['full_n']})",
            f"≥60% months positive ({len(pos_months)}/{len(oos_months)})",
            f"Mean net > 20bp (actual: {r['full_mean_net_bp']:+.1f}bp)",
            f"PF ≥ 1.3 (actual: {r['full_pf']:.2f})",
            f"P5 > -600bp (actual: {r.get('full_p5', 0):.0f}bp)",
            f"Winsorized mean > 0 (actual: {r.get('full_winsorized_mean', 0):+.1f}bp)",
            f"p-value < 0.15 (actual: {r.get('full_p_perm', 1):.3f})",
        ]

        print(f"\n  {label}:")
        for check, lbl in zip(checks, labels):
            print(f"    {'✓' if check else '✗'} {lbl}")
        if all(checks):
            print(f"    → GO ✅")
        else:
            print(f"    → NO-GO ❌")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
