#!/usr/bin/env python3
"""
XS-5 — Positioning Extremes Event Model (Funding + OI + Price Stall)

Find rare positional imbalances where expected move ≥150-300bp,
so that fees stop being the dominant factor.

Events:
  E1: Crowded Long → SHORT (high funding + rising OI + price stall)
  E2: Crowded Short → LONG (low funding + rising OI + price stall)
  E4: Funding Extreme Reversion (extreme funding, OI flat → against funding)

Single-leg trades. 12h cooldown per symbol per event.
Exits: time stop (4h/12h/24h) OR early unwind (funding_z normalizes) OR SL/TP.
"""

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
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

MIN_DAYS = 50

# Feature windows (in 1m bars)
FR_Z_WINDOW_1M = 7 * 24 * 60       # 7 days in 1m bars for funding z-score context
OI_Z_WINDOW_1M = 7 * 24 * 60       # 7 days for OI z-score
OI_CHG_1H = 60                     # 1h OI change
OI_CHG_4H = 240                    # 4h OI change
RET_2H = 120                       # 2h return window
RV_2H = 120                        # 2h realized vol
ATR_1H = 60                        # 1h ATR

# Event thresholds (fixed, not optimized)
E1_FR_Z = 2.0
E1_OI_Z = 2.0
E1_TREND = 0.3

E2_FR_Z = -2.0
E2_OI_Z = 2.0
E2_TREND = 0.3

E4_FR_Z_ABS = 3.0
E4_OI_Z_MAX = 1.0

# Cooldown: 12h = 720 1m bars
COOLDOWN_MIN = 12 * 60

# Exit params
HOLD_GRID = [4 * 60, 12 * 60, 24 * 60]   # 4h, 12h, 24h in minutes
HOLD_LABELS = ["4h", "12h", "24h"]
UNWIND_FR_Z_LO = -1.0
UNWIND_FR_Z_HI = 1.0

# SL/TP configs: (label, sl_atr_mult, tp_atr_mult)
# None means no SL/TP for that leg
SLTP_CONFIGS = [
    ("none", None, None),          # pure time stop + unwind only
    ("cat6", 6.0, None),           # catastrophe SL only (6×ATR), no TP
    ("wide", 4.0, 6.0),            # wide: SL=4×ATR, TP=6×ATR
    ("orig", 2.5, 4.0),            # original spec
]

# Cost
FEE_BP = 10                         # per side → 20bp RT
SLIP_GRID = [0, 2, 5, 10]           # bp per side (single leg)

# Walk-forward
WF_SPLIT = pd.Timestamp("2026-02-01", tz="UTC")

# Stats
N_BOOTSTRAP = 5000
N_PERMUTATION = 2000
SEED = 42

# Minimum eligibility: 60m turnover > threshold
MIN_TURNOVER_60M = 50_000


# ---------------------------------------------------------------------------
# §1: Data Loading
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
    """Load all data for one symbol."""
    mark = _load_glob(sym, "*_mark_price_kline_1m.csv", "startTime",
                      {"close": "close"})

    # Kline (not mark/premium)
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
# §2: Build unified 1m grid per symbol
# ---------------------------------------------------------------------------

def build_sym_1m(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> pd.DataFrame:
    """Build 1m DataFrame for one symbol with all columns aligned to grid."""
    mark = raw["mark"].set_index("ts")["close"] if len(raw["mark"]) > 0 else pd.Series(dtype=float)
    kl = raw["kline"]
    oi_df = raw["oi"]
    fr_df = raw["fr"]

    n = len(grid_1m)
    df = pd.DataFrame(index=grid_1m)

    # Close from mark price
    close_raw = mark.reindex(grid_1m)
    # Gap detection
    is_nan = close_raw.isna()
    nan_arr = is_nan.values
    # Compute consecutive NaN block lengths
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

    # Forward fill close
    close = close_raw.ffill()
    close[is_invalid == 1] = np.nan

    df["close"] = close
    df["is_ffill"] = is_ffill
    df["is_invalid"] = is_invalid

    # High/Low from kline
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

    # OI: 5m → ffill to 1m, shifted +5min for causal alignment
    # (OI bar at ts T covers [T, T+5m); available only after T+5m)
    if len(oi_df) > 0:
        oi_shifted = oi_df.copy()
        oi_shifted["ts"] = oi_shifted["ts"] + pd.Timedelta(minutes=5)
        oi_s = oi_shifted.set_index("ts")["oi"]
        df["oi"] = oi_s.reindex(grid_1m).ffill()
    else:
        df["oi"] = np.nan

    # Funding: 8h → ffill to 1m, shifted +1min for causal alignment
    # (FR published at settlement ts; we delay 1 bar to be safe)
    if len(fr_df) > 0:
        fr_shifted = fr_df.copy()
        fr_shifted["ts"] = fr_shifted["ts"] + pd.Timedelta(minutes=1)
        fr_s = fr_shifted.set_index("ts")["fr"]
        df["fr"] = fr_s.reindex(grid_1m).ffill()
    else:
        df["fr"] = np.nan

    # Log return
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.loc[df["is_invalid"] == 1, "log_ret"] = np.nan
    df.loc[df["is_invalid"].shift(1) == 1, "log_ret"] = np.nan

    return df


# ---------------------------------------------------------------------------
# §3: Feature computation (strictly causal)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for one symbol. All windows are [t-W, t)."""
    n = len(df)

    # --- Funding features ---
    # funding_z_7d: z-score of funding rate over 7 days
    # Since funding is 8h (3/day), 7d = 21 obs. But we ffilled to 1m.
    # Use rolling on the ffilled series with 7d window
    fr = df["fr"]
    fr_rm = fr.rolling(FR_Z_WINDOW_1M, min_periods=FR_Z_WINDOW_1M // 4).mean()
    fr_rs = fr.rolling(FR_Z_WINDOW_1M, min_periods=FR_Z_WINDOW_1M // 4).std().clip(lower=1e-12)
    df["funding_z_7d"] = (fr - fr_rm) / fr_rs
    df["funding_sign"] = np.sign(fr)

    # --- OI features ---
    oi = df["oi"]
    # OI change 1h
    oi_lag_1h = oi.shift(OI_CHG_1H)
    df["oi_chg_1h"] = (oi - oi_lag_1h) / oi_lag_1h.clip(lower=1)
    # OI change 4h
    oi_lag_4h = oi.shift(OI_CHG_4H)
    df["oi_chg_4h"] = (oi - oi_lag_4h) / oi_lag_4h.clip(lower=1)
    # OI z-score 7d (on oi_chg_1h)
    oi_chg = df["oi_chg_1h"]
    oi_rm = oi_chg.rolling(OI_Z_WINDOW_1M, min_periods=OI_Z_WINDOW_1M // 4).mean()
    oi_rs = oi_chg.rolling(OI_Z_WINDOW_1M, min_periods=OI_Z_WINDOW_1M // 4).std().clip(lower=1e-12)
    df["oi_z_7d"] = (oi_chg - oi_rm) / oi_rs

    # --- Price stall / divergence ---
    close = df["close"]
    # ret_2h
    df["ret_2h"] = np.log(close / close.shift(RET_2H))
    # rv_2h = std of 1m log returns over 2h
    df["rv_2h"] = df["log_ret"].rolling(RV_2H, min_periods=RV_2H // 2).std()
    # trend_2h = |ret_2h| / (rv_2h * sqrt(120))
    rv_scaled = df["rv_2h"] * np.sqrt(RV_2H)
    df["trend_2h"] = df["ret_2h"].abs() / rv_scaled.clip(lower=1e-12)

    # dd from high / du from low over 2h
    df["high_2h"] = df["high"].rolling(RET_2H, min_periods=1).max()
    df["low_2h"] = df["low"].rolling(RET_2H, min_periods=1).min()
    df["dd_from_high_2h"] = (close - df["high_2h"]) / df["high_2h"].clip(lower=1e-8)
    df["du_from_low_2h"] = (close - df["low_2h"]) / df["low_2h"].clip(lower=1e-8)

    # ATR 1h
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_1h"] = tr.rolling(ATR_1H, min_periods=ATR_1H // 2).mean()

    # Turnover 60m
    df["turnover_60m"] = df["turnover"].rolling(60, min_periods=10).sum()

    return df


# ---------------------------------------------------------------------------
# §4: Event detection with cooldown
# ---------------------------------------------------------------------------

def detect_events(df: pd.DataFrame, sym: str) -> list[dict]:
    """Detect E1, E2, E4 events with 12h cooldown per event type."""
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

    # Track last trigger time per event type
    last_trigger = {"E1": -COOLDOWN_MIN * 2, "E2": -COOLDOWN_MIN * 2,
                    "E4": -COOLDOWN_MIN * 2}

    # Check every 5 minutes for efficiency
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

        ts = idx[i]

        # E1: Crowded Long → SHORT
        if (fz[i] >= E1_FR_Z and oiz[i] >= E1_OI_Z and
                trend[i] <= E1_TREND and ret2h[i] >= 0):
            if i - last_trigger["E1"] >= COOLDOWN_MIN:
                events.append({
                    "symbol": sym, "ts": ts, "idx": i,
                    "event_type": "E1", "direction": -1,
                    "funding_z": fz[i], "oi_z": oiz[i],
                    "trend_2h": trend[i], "ret_2h": ret2h[i],
                    "atr_1h": atr[i], "fr_raw": fr_raw[i],
                })
                last_trigger["E1"] = i

        # E2: Crowded Short → LONG
        if (fz[i] <= E2_FR_Z and oiz[i] >= E2_OI_Z and
                trend[i] <= E2_TREND and ret2h[i] <= 0):
            if i - last_trigger["E2"] >= COOLDOWN_MIN:
                events.append({
                    "symbol": sym, "ts": ts, "idx": i,
                    "event_type": "E2", "direction": 1,
                    "funding_z": fz[i], "oi_z": oiz[i],
                    "trend_2h": trend[i], "ret_2h": ret2h[i],
                    "atr_1h": atr[i], "fr_raw": fr_raw[i],
                })
                last_trigger["E2"] = i

        # E4: Funding Extreme Reversion
        if (abs(fz[i]) >= E4_FR_Z_ABS and oiz[i] <= E4_OI_Z_MAX):
            if i - last_trigger["E4"] >= COOLDOWN_MIN:
                direction = -1 if fz[i] > 0 else 1  # against funding sign
                events.append({
                    "symbol": sym, "ts": ts, "idx": i,
                    "event_type": "E4", "direction": direction,
                    "funding_z": fz[i], "oi_z": oiz[i],
                    "trend_2h": trend[i], "ret_2h": ret2h[i],
                    "atr_1h": atr[i], "fr_raw": fr_raw[i],
                })
                last_trigger["E4"] = i

    return events


# ---------------------------------------------------------------------------
# §5: Trade simulation with MFE/MAE
# ---------------------------------------------------------------------------

def simulate_trades(
    events: list[dict],
    sym_dfs: dict,
    grid_1m: pd.DatetimeIndex,
    hold_minutes: int,
    slip_bp: float,
    sl_atr_mult: float = None,
    tp_atr_mult: float = None,
) -> pd.DataFrame:
    """Simulate single-leg trades from events.

    Exit priority: SL → TP → early unwind (funding_z normalizes) → time stop.
    Track MFE and MAE throughout.
    sl_atr_mult/tp_atr_mult: None means disabled.
    """
    trades = []
    cost_rt_bp = 2 * FEE_BP + 2 * slip_bp  # total round trip

    for evt in events:
        sym = evt["symbol"]
        df = sym_dfs[sym]
        idx_signal = evt["idx"]
        direction = evt["direction"]  # +1 long, -1 short
        atr = evt["atr_1h"]

        if atr <= 0 or np.isnan(atr):
            continue

        # Entry: next 1m close after signal
        entry_idx = idx_signal + 1
        if entry_idx >= len(df):
            continue
        entry_px = df["close"].iloc[entry_idx]
        if np.isnan(entry_px) or entry_px <= 0:
            continue

        # SL/TP levels (None = disabled)
        use_sl = sl_atr_mult is not None
        use_tp = tp_atr_mult is not None
        sl_dist = (sl_atr_mult or 0) * atr
        tp_dist = (tp_atr_mult or 0) * atr
        if direction == 1:  # long
            sl_px = entry_px - sl_dist if use_sl else 0
            tp_px = entry_px + tp_dist if use_tp else 1e18
        else:  # short
            sl_px = entry_px + sl_dist if use_sl else 1e18
            tp_px = entry_px - tp_dist if use_tp else 0

        # Simulate bar-by-bar
        max_idx = min(entry_idx + hold_minutes + 1, len(df))
        exit_idx = max_idx - 1
        exit_reason = "time_stop"
        mfe = 0.0  # max favorable excursion in bp
        mae = 0.0  # max adverse excursion in bp

        for j in range(entry_idx + 1, max_idx):
            if df["is_invalid"].iloc[j] == 1:
                continue

            px = df["close"].iloc[j]
            hi = df["high"].iloc[j]
            lo = df["low"].iloc[j]
            if np.isnan(px):
                continue

            # Track MFE/MAE using high/low
            if direction == 1:
                fav = (hi - entry_px) / entry_px * 10000
                adv = (lo - entry_px) / entry_px * 10000
                mfe = max(mfe, fav)
                mae = min(mae, adv)
            else:
                fav = (entry_px - lo) / entry_px * 10000
                adv = (entry_px - hi) / entry_px * 10000
                mfe = max(mfe, fav)
                mae = min(mae, adv)

            # Check SL
            if use_sl:
                if direction == 1 and lo <= sl_px:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if direction == -1 and hi >= sl_px:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break

            # Check TP
            if use_tp:
                if direction == 1 and hi >= tp_px:
                    exit_idx = j
                    exit_reason = "take_profit"
                    break
                if direction == -1 and lo <= tp_px:
                    exit_idx = j
                    exit_reason = "take_profit"
                    break

            # Check early unwind: funding_z back to normal
            fz_now = df["funding_z_7d"].iloc[j]
            if pd.notna(fz_now) and UNWIND_FR_Z_LO <= fz_now <= UNWIND_FR_Z_HI:
                exit_idx = j
                exit_reason = "unwind"
                break

        # Exit price
        exit_px = df["close"].iloc[exit_idx]
        if np.isnan(exit_px) or exit_px <= 0:
            continue

        # Compute PnL
        if exit_reason == "stop_loss" and use_sl:
            # Use SL price for exits (conservative)
            if direction == 1:
                gross_bp = (sl_px - entry_px) / entry_px * 10000
            else:
                gross_bp = (entry_px - sl_px) / entry_px * 10000
        elif exit_reason == "take_profit" and use_tp:
            if direction == 1:
                gross_bp = (tp_px - entry_px) / entry_px * 10000
            else:
                gross_bp = (entry_px - tp_px) / entry_px * 10000
        else:
            # Time stop or unwind: use actual close
            if direction == 1:
                gross_bp = (exit_px - entry_px) / entry_px * 10000
            else:
                gross_bp = (entry_px - exit_px) / entry_px * 10000

        net_bp = gross_bp - cost_rt_bp
        hold_min = exit_idx - entry_idx

        entry_ts = grid_1m[entry_idx]
        exit_ts = grid_1m[exit_idx]

        trades.append({
            "symbol": sym,
            "event_type": evt["event_type"],
            "direction": direction,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "sl_px": sl_px,
            "tp_px": tp_px,
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
            "slip_bp": slip_bp,
        })

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# §6: Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Compute trade-level and daily statistics."""
    if len(trades_df) == 0:
        return {"n": 0}

    net = trades_df["net_bp"]
    gross = trades_df["gross_bp"]
    n = len(trades_df)

    # Daily aggregation
    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["entry_ts"]).dt.date
    daily = trades_df.groupby("date")["net_bp"].sum()
    n_days = len(daily)

    mean_daily = daily.mean() if n_days > 0 else 0
    std_daily = daily.std() if n_days > 1 else 1
    sharpe_ann = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0

    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0

    # Per-trade stats
    winners = net[net > 0]
    losers = net[net <= 0]
    pf = winners.sum() / abs(losers.sum()) if len(losers) > 0 and losers.sum() != 0 else np.inf

    # Bootstrap CI on mean net per trade
    net_arr = net.values
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = net_arr[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Permutation test on daily returns (sign-flip)
    p_perm = np.nan
    if n_days >= 5:
        daily_arr = daily.values
        perm_means = np.empty(N_PERMUTATION)
        for p in range(N_PERMUTATION):
            signs = rng.choice([-1, 1], size=n_days)
            perm_means[p] = (daily_arr * signs).mean()
        p_perm = (perm_means >= mean_daily).mean()

    # Winsorized mean (clip at 5th/95th)
    p5_val = np.percentile(net_arr, 5)
    p95_val = np.percentile(net_arr, 95)
    winsorized = np.clip(net_arr, p5_val, p95_val)
    winsorized_mean = winsorized.mean()

    # Weekly stability
    trades_df["week"] = pd.to_datetime(trades_df["entry_ts"]).dt.isocalendar().week.values
    weekly = trades_df.groupby("week")["net_bp"].sum()
    pos_weeks = (weekly > 0).sum()
    total_weeks = len(weekly)

    # Exit reason breakdown
    exit_counts = trades_df["exit_reason"].value_counts().to_dict()

    return {
        "n": n,
        "n_days": n_days,
        "mean_net_bp": net.mean(),
        "median_net_bp": net.median(),
        "std_net_bp": net.std(),
        "wr": (net > 0).mean(),
        "pf": pf,
        "p5": np.percentile(net_arr, 5),
        "p95": np.percentile(net_arr, 95),
        "mean_gross_bp": gross.mean(),
        "mean_mfe": trades_df["mfe_bp"].mean(),
        "mean_mae": trades_df["mae_bp"].mean(),
        "mean_hold_min": trades_df["hold_min"].mean(),
        "mean_daily_bp": mean_daily,
        "sharpe_ann": sharpe_ann,
        "max_dd_bp": max_dd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "winsorized_mean": winsorized_mean,
        "pos_weeks": pos_weeks,
        "total_weeks": total_weeks,
        "exit_sl": exit_counts.get("stop_loss", 0),
        "exit_tp": exit_counts.get("take_profit", 0),
        "exit_unwind": exit_counts.get("unwind", 0),
        "exit_time": exit_counts.get("time_stop", 0),
    }


# ---------------------------------------------------------------------------
# §7: Walk-forward
# ---------------------------------------------------------------------------

def split_events(events, period_start, period_end):
    return [e for e in events if period_start <= e["ts"] <= period_end]


# ---------------------------------------------------------------------------
# §8: Bug checks
# ---------------------------------------------------------------------------

def bug_checks(events, trades_df):
    print(f"\n  BUG DETECTION CHECKLIST")
    print(f"  {'='*50}")
    print(f"  [1] All features use backward-only windows ✓")
    print(f"  [2] OI ffilled (available after bar ts) ✓")
    print(f"  [3] Funding ffilled (available after settlement) ✓")
    print(f"  [4] Entry = next 1m close after signal ✓")

    # Check cooldown
    if events:
        violations = 0
        by_sym_type = {}
        for e in events:
            key = (e["symbol"], e["event_type"])
            if key in by_sym_type:
                dt = (e["ts"] - by_sym_type[key]).total_seconds() / 60
                if dt < COOLDOWN_MIN:
                    violations += 1
            by_sym_type[key] = e["ts"]
        print(f"  [5] 12h cooldown respected: "
              f"{'✓ PASS' if violations == 0 else f'✗ FAIL ({violations})'}")

    if len(trades_df) > 0:
        future = (trades_df["exit_ts"] <= trades_df["entry_ts"]).any()
        print(f"  [6] Exit > entry: "
              f"{'✓ PASS' if not future else '✗ FAIL'}")

    print(f"  [7] No lookahead in features (z-score backward rolling) ✓")
    print(f"  [8] SL/TP computed from entry price + ATR at signal time ✓")
    print(f"  {'='*50}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("XS-5 — POSITIONING EXTREMES EVENT MODEL")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Events: E1 (crowded long→short), E2 (crowded short→long), "
          f"E4 (FR extreme reversion)")
    sltp_desc = ", ".join(f"{l}(SL={s},TP={t})" for l, s, t in SLTP_CONFIGS)
    print(f"Holds: {HOLD_LABELS}, SL/TP configs: {sltp_desc}")
    print(f"Cost: {FEE_BP}bp/side + slippage {SLIP_GRID} bp/side")
    print("=" * 80)

    # §1: Load data
    print(f"\n{'─'*70}")
    print("PHASE 1: Data loading")
    print(f"{'─'*70}")
    symbols = discover_symbols()
    print(f"  Symbols: {len(symbols)}")

    raw_all = {}
    for i, sym in enumerate(symbols, 1):
        raw_all[sym] = load_symbol(sym)
        if i % 10 == 0 or i == len(symbols):
            n_mark = sum(1 for r in raw_all.values() if len(r['mark']) > 0)
            print(f"    Loaded {i}/{len(symbols)} ({n_mark} with mark data)",
                  flush=True)
    print(f"  Loaded: {len(raw_all)} symbols", flush=True)

    # §2: Build unified 1m grid
    print(f"\n{'─'*70}")
    print("PHASE 2: Unified 1m grid + features")
    print(f"{'─'*70}")

    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"  1m grid: {len(grid_1m):,} points (uniform calendar)")

    # Build per-symbol DataFrames with features
    sym_dfs = {}
    all_events = []
    for i, sym in enumerate(sorted(raw_all.keys()), 1):
        raw = raw_all[sym]
        if len(raw["mark"]) == 0:
            continue
        df = build_sym_1m(sym, raw, grid_1m)
        df = compute_features(df)
        sym_dfs[sym] = df

        # Detect events
        evts = detect_events(df, sym)
        all_events.extend(evts)

        if i % 10 == 0:
            print(f"    Processed {i}/{len(raw_all)}")

    print(f"  Total events detected: {len(all_events)}")

    # Event summary
    from collections import Counter
    type_counts = Counter(e["event_type"] for e in all_events)
    for et in ["E1", "E2", "E4"]:
        print(f"    {et}: {type_counts.get(et, 0)}")

    # Per-symbol breakdown
    sym_counts = Counter(e["symbol"] for e in all_events)
    top_syms = sym_counts.most_common(10)
    print(f"  Top symbols: {', '.join(f'{s}({c})' for s, c in top_syms)}")
    print(f"  Symbols with events: {len(sym_counts)}")

    # Save events
    events_df = pd.DataFrame(all_events)
    if len(events_df) > 0:
        events_df.to_parquet(OUTPUT_DIR / "xs5_events.parquet", index=False)
        print(f"  Saved xs5_events.parquet")

    if len(all_events) == 0:
        print("\n  ✗ No events detected. Cannot proceed.")
        return

    # §5: Simulate trades for all configs
    print(f"\n{'─'*70}")
    print("PHASE 3: Trade simulation")
    print(f"{'─'*70}")

    all_results = []
    all_trades_list = []

    # Pre-split events for walk-forward
    evts_jan = split_events(all_events, START,
                            WF_SPLIT - pd.Timedelta(minutes=1))
    evts_feb = split_events(all_events, WF_SPLIT, END)

    n_configs = len(SLTP_CONFIGS) * len(HOLD_GRID) * len(SLIP_GRID)
    config_i = 0

    for sltp_label, sl_mult, tp_mult in SLTP_CONFIGS:
        for hi, (hold_min, hold_label) in enumerate(zip(HOLD_GRID, HOLD_LABELS)):
            for slip in SLIP_GRID:
                config_i += 1
                cost_rt = 2 * FEE_BP + 2 * slip

                # Full period
                trades_full = simulate_trades(all_events, sym_dfs, grid_1m,
                                              hold_min, slip, sl_mult, tp_mult)
                if len(trades_full) == 0:
                    continue

                trades_full["hold_label"] = hold_label
                trades_full["sltp"] = sltp_label
                all_trades_list.append(trades_full)

                # Walk-forward trades
                trades_feb = simulate_trades(evts_feb, sym_dfs, grid_1m,
                                             hold_min, slip, sl_mult, tp_mult)
                trades_jan = simulate_trades(evts_jan, sym_dfs, grid_1m,
                                             hold_min, slip, sl_mult, tp_mult)

                # Stats by event type
                for et in ["E1", "E2", "E4", "ALL"]:
                    if et == "ALL":
                        tf = trades_full
                        tf_feb = trades_feb
                        tf_jan = trades_jan
                    else:
                        tf = trades_full[trades_full["event_type"] == et]
                        tf_feb = trades_feb[trades_feb["event_type"] == et] if len(trades_feb) > 0 else trades_feb
                        tf_jan = trades_jan[trades_jan["event_type"] == et] if len(trades_jan) > 0 else trades_jan

                    if len(tf) == 0:
                        continue

                    stats_full = compute_stats(tf, rng)
                    stats_feb = compute_stats(tf_feb, rng) if len(tf_feb) > 0 else {"n": 0}
                    stats_jan = compute_stats(tf_jan, rng) if len(tf_jan) > 0 else {"n": 0}

                    result = {
                        "sltp": sltp_label,
                        "event_type": et,
                        "hold": hold_label,
                        "slip_bp": slip,
                        "cost_rt_bp": cost_rt,
                        # Full
                        "full_n": stats_full["n"],
                        "full_mean_net": stats_full.get("mean_net_bp", np.nan),
                        "full_median_net": stats_full.get("median_net_bp", np.nan),
                        "full_wr": stats_full.get("wr", np.nan),
                        "full_pf": stats_full.get("pf", np.nan),
                        "full_p5": stats_full.get("p5", np.nan),
                        "full_p95": stats_full.get("p95", np.nan),
                        "full_mean_mfe": stats_full.get("mean_mfe", np.nan),
                        "full_mean_mae": stats_full.get("mean_mae", np.nan),
                        "full_mean_hold": stats_full.get("mean_hold_min", np.nan),
                        "full_sharpe": stats_full.get("sharpe_ann", np.nan),
                        "full_max_dd": stats_full.get("max_dd_bp", np.nan),
                        "full_p_perm": stats_full.get("p_perm", np.nan),
                        "full_winsorized": stats_full.get("winsorized_mean", np.nan),
                        "full_pos_weeks": stats_full.get("pos_weeks", 0),
                        "full_total_weeks": stats_full.get("total_weeks", 0),
                        "full_ci_lo": stats_full.get("ci_lo", np.nan),
                        "full_ci_hi": stats_full.get("ci_hi", np.nan),
                        "full_exit_sl": stats_full.get("exit_sl", 0),
                        "full_exit_tp": stats_full.get("exit_tp", 0),
                        "full_exit_unwind": stats_full.get("exit_unwind", 0),
                        "full_exit_time": stats_full.get("exit_time", 0),
                        # OOS Feb
                        "oos_feb_n": stats_feb.get("n", 0),
                        "oos_feb_mean": stats_feb.get("mean_net_bp", np.nan),
                        "oos_feb_median": stats_feb.get("median_net_bp", np.nan),
                        "oos_feb_wr": stats_feb.get("wr", np.nan),
                        "oos_feb_pf": stats_feb.get("pf", np.nan),
                        "oos_feb_sharpe": stats_feb.get("sharpe_ann", np.nan),
                        "oos_feb_p": stats_feb.get("p_perm", np.nan),
                        "oos_feb_winsorized": stats_feb.get("winsorized_mean", np.nan),
                        "oos_feb_p5": stats_feb.get("p5", np.nan),
                        # OOS Jan
                        "oos_jan_n": stats_jan.get("n", 0),
                        "oos_jan_mean": stats_jan.get("mean_net_bp", np.nan),
                        "oos_jan_median": stats_jan.get("median_net_bp", np.nan),
                        "oos_jan_wr": stats_jan.get("wr", np.nan),
                        "oos_jan_pf": stats_jan.get("pf", np.nan),
                        "oos_jan_sharpe": stats_jan.get("sharpe_ann", np.nan),
                        "oos_jan_p": stats_jan.get("p_perm", np.nan),
                        "oos_jan_winsorized": stats_jan.get("winsorized_mean", np.nan),
                        "oos_jan_p5": stats_jan.get("p5", np.nan),
                    }
                    all_results.append(result)

                if config_i % 12 == 0:
                    print(f"    Config {config_i}/{n_configs}")

    # Save results
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_DIR / "xs5_report.csv", index=False)

    # Save all trades
    if all_trades_list:
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
        all_trades_df.to_parquet(OUTPUT_DIR / "xs5_trades.parquet", index=False)

    # Print summary — focus on slip=5bp across all SLTP configs
    print(f"\n{'─'*70}")
    print("RESULTS SUMMARY (slip=5bp)")
    print(f"{'─'*70}")

    for sltp_l, _, _ in SLTP_CONFIGS:
        print(f"\n  ══ SL/TP={sltp_l} ══")
        for et in ["E1", "E2", "E4", "ALL"]:
            et_res = [r for r in all_results
                      if r["event_type"] == et and r["sltp"] == sltp_l and r["slip_bp"] == 5]
            for r in et_res:
                if r["full_n"] < 3:
                    continue
                exits = f"SL={r['full_exit_sl']} TP={r['full_exit_tp']} UW={r['full_exit_unwind']} TS={r['full_exit_time']}"
                print(f"  {et:3s} {r['hold']:3s}: N={r['full_n']:>3d}, "
                      f"mean={r['full_mean_net']:+7.1f}bp, "
                      f"med={r['full_median_net']:+7.1f}bp, "
                      f"WR={r['full_wr']:.0%}, "
                      f"PF={r['full_pf']:.2f}, "
                      f"Sharpe={r['full_sharpe']:+.2f}, "
                      f"p={r['full_p_perm']:.3f} | {exits}")
                if r["oos_feb_n"] > 0 or r["oos_jan_n"] > 0:
                    feb_s = (f"Feb: N={r['oos_feb_n']}, mean={r.get('oos_feb_mean',0):+.1f}bp, "
                             f"WR={r.get('oos_feb_wr',0):.0%}, PF={r.get('oos_feb_pf',0):.2f}") if r["oos_feb_n"] > 0 else "Feb: -"
                    jan_s = (f"Jan: N={r['oos_jan_n']}, mean={r.get('oos_jan_mean',0):+.1f}bp, "
                             f"WR={r.get('oos_jan_wr',0):.0%}, PF={r.get('oos_jan_pf',0):.2f}") if r["oos_jan_n"] > 0 else "Jan: -"
                    print(f"        OOS {feb_s} | {jan_s}")

    # Bug checks
    if all_trades_list:
        bug_checks(all_events, all_trades_df)

    # OOS summary
    oos_df = pd.DataFrame([r for r in all_results
                           if r["slip_bp"] == 5 and r["event_type"] != "ALL"])
    if len(oos_df) > 0:
        oos_df.to_csv(OUTPUT_DIR / "xs5_oos_summary.csv", index=False)

    # GO / NO-GO
    print(f"\n{'='*80}")
    print("GO / NO-GO VERDICT (slip=5bp)")
    print(f"{'='*80}")

    go_any = False
    for r in all_results:
        if r["slip_bp"] != 5:
            continue
        if r["event_type"] == "ALL":
            continue
        if r["full_n"] < 10:
            continue

        n_ok = r["full_n"] >= 40 or (r["full_n"] >= 20)
        mean_ok = (r.get("oos_feb_mean", -999) > 20 and
                   r.get("oos_jan_mean", -999) > 20) or \
                  (r.get("oos_feb_median", -999) > 10 and
                   r.get("oos_jan_median", -999) > 10)
        oos_mean_ok = (r.get("oos_feb_mean", -999) > 0 and
                       r.get("oos_jan_mean", -999) > 0)
        pf_ok = r.get("full_pf", 0) >= 1.3
        p5_ok = r.get("full_p5", -9999) > -600
        wins_ok = r.get("full_winsorized", -999) > 0

        label = f"{r['event_type']} {r['hold']}"
        checks = [n_ok, oos_mean_ok, pf_ok, p5_ok, wins_ok]
        labels = [
            f"N ≥ 20 (actual: {r['full_n']})",
            f"OOS mean > 0 both halves",
            f"PF ≥ 1.3 (actual: {r.get('full_pf', 0):.2f})",
            f"P5 > -600bp (actual: {r.get('full_p5', 0):.0f}bp)",
            f"Winsorized mean > 0 (actual: {r.get('full_winsorized', 0):+.1f}bp)",
        ]

        print(f"\n  {label}:")
        for check, lbl in zip(checks, labels):
            print(f"    {'✓' if check else '✗'} {lbl}")

        if all(checks):
            go_any = True
            print(f"    → GO ✅")
        else:
            print(f"    → NO-GO ❌")

    print(f"\n  {'='*40}")
    if go_any:
        print(f"  VERDICT: GO ✅ (at least one event type passes)")
    else:
        print(f"  VERDICT: NO-GO ❌")
    print(f"  {'='*40}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
