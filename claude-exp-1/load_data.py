#!/usr/bin/env python3
"""
Unified cross-exchange data loader for 116 symbols.

Loads 1m klines, premium index, funding rate, OI, and LS ratio from both
Bybit and Binance datalake, resamples to 5m bars, and merges into a single
DataFrame per symbol with cross-exchange features ready to compute.

Usage:
    from load_data import load_symbol, load_all_symbols, list_common_symbols
    df = load_symbol("SOLUSDT")           # single symbol
    all_data = load_all_symbols(n_jobs=8)  # dict of {symbol: df}
"""

import os
import glob
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATALAKE = Path(__file__).resolve().parent.parent / "datalake"
BYBIT_DIR = DATALAKE / "bybit"
BINANCE_DIR = DATALAKE / "binance"


def list_common_symbols() -> list[str]:
    """Return sorted list of symbols present on both exchanges."""
    bybit_syms = set(os.listdir(BYBIT_DIR)) if BYBIT_DIR.exists() else set()
    binance_syms = set(os.listdir(BINANCE_DIR)) if BINANCE_DIR.exists() else set()
    common = sorted(bybit_syms & binance_syms)
    return [s for s in common if not s.startswith(".")]


def _load_csvs(pattern: str, ts_col: str = "startTime",
               cols: list[str] | None = None,
               parse_ts: bool = True) -> pd.DataFrame:
    """Load and concatenate daily CSV files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=cols if cols else None)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)

    if parse_ts and ts_col in result.columns:
        if result[ts_col].dtype == object and "-" in str(result[ts_col].iloc[0]):
            ts = pd.to_datetime(result[ts_col], utc=True)
        else:
            ts = pd.to_datetime(result[ts_col], unit="ms", utc=True)
        # Drop original ts column(s), assign parsed ts as index
        for c in [ts_col, "startTime", "open_time", "create_time"]:
            if c in result.columns:
                result = result.drop(columns=[c])
        result["_ts"] = ts.values
        result = result.sort_values("_ts").drop_duplicates(subset=["_ts"])
        result = result.set_index("_ts")
        result.index.name = "timestamp"
    return result


def _load_bybit_klines(sym: str) -> pd.DataFrame:
    """Load Bybit 1m klines for a symbol."""
    # Use explicit date pattern to avoid matching premium_index_kline, mark_price_kline, etc.
    pattern = str(BYBIT_DIR / sym / "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_kline_1m.csv")
    df = _load_csvs(pattern, ts_col="startTime",
                    cols=["startTime", "open", "high", "low", "close", "volume", "turnover"])
    if df.empty:
        return df
    df = df.rename(columns={
        "open": "bb_open", "high": "bb_high", "low": "bb_low",
        "close": "bb_close", "volume": "bb_volume", "turnover": "bb_turnover"
    })
    return df


def _load_binance_klines(sym: str) -> pd.DataFrame:
    """Load Binance 1m klines for a symbol."""
    # Use explicit date pattern to avoid matching premium_index_kline, mark_price_kline, etc.
    pattern = str(BINANCE_DIR / sym / "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_kline_1m.csv")
    df = _load_csvs(pattern, ts_col="open_time",
                    cols=["open_time", "open", "high", "low", "close", "volume",
                          "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"])
    if df.empty:
        return df
    df = df.rename(columns={
        "open": "bn_open", "high": "bn_high", "low": "bn_low",
        "close": "bn_close", "volume": "bn_volume", "quote_volume": "bn_turnover",
        "count": "bn_trades", "taker_buy_volume": "bn_taker_buy_vol",
        "taker_buy_quote_volume": "bn_taker_buy_turnover"
    })
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_bybit_premium(sym: str) -> pd.DataFrame:
    """Load Bybit premium index klines."""
    pattern = str(BYBIT_DIR / sym / "*_premium_index_kline_1m.csv")
    df = _load_csvs(pattern, ts_col="startTime", cols=["startTime", "close"])
    if df.empty:
        return df
    df = df.rename(columns={"close": "bb_premium"})
    return df


def _load_binance_premium(sym: str) -> pd.DataFrame:
    """Load Binance premium index klines."""
    pattern = str(BINANCE_DIR / sym / "*_premium_index_kline_1m.csv")
    df = _load_csvs(pattern, ts_col="open_time", cols=["open_time", "close"])
    if df.empty:
        return df
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"close": "bn_premium"})
    return df


def _load_bybit_funding(sym: str) -> pd.DataFrame:
    """Load Bybit funding rate."""
    pattern = str(BYBIT_DIR / sym / "*_funding_rate.csv")
    df = _load_csvs(pattern, ts_col="timestamp",
                    cols=["timestamp", "fundingRate"])
    if df.empty:
        return df
    df = df.rename(columns={"fundingRate": "bb_fr"})
    df["bb_fr"] = pd.to_numeric(df["bb_fr"], errors="coerce")
    return df


def _load_bybit_oi(sym: str) -> pd.DataFrame:
    """Load Bybit open interest (5min)."""
    pattern = str(BYBIT_DIR / sym / "*_open_interest_5min.csv")
    df = _load_csvs(pattern, ts_col="timestamp",
                    cols=["timestamp", "openInterest"])
    if df.empty:
        return df
    df = df.rename(columns={"openInterest": "bb_oi"})
    df["bb_oi"] = pd.to_numeric(df["bb_oi"], errors="coerce")
    return df


def _load_bybit_ls(sym: str) -> pd.DataFrame:
    """Load Bybit long/short ratio (5min)."""
    pattern = str(BYBIT_DIR / sym / "*_long_short_ratio_5min.csv")
    df = _load_csvs(pattern, ts_col="timestamp",
                    cols=["timestamp", "buyRatio", "sellRatio"])
    if df.empty:
        return df
    df["bb_ls_ratio"] = pd.to_numeric(df["buyRatio"], errors="coerce") / \
                        pd.to_numeric(df["sellRatio"], errors="coerce").replace(0, np.nan)
    df = df[["bb_ls_ratio"]]
    return df


def _load_binance_metrics(sym: str) -> pd.DataFrame:
    """Load Binance metrics (OI, LS ratio, taker vol ratio) at 5min."""
    pattern = str(BINANCE_DIR / sym / "*_metrics.csv")
    df = _load_csvs(pattern, ts_col="create_time",
                    cols=["create_time", "sum_open_interest", "sum_open_interest_value",
                          "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    if df.empty:
        return df
    df = df.rename(columns={
        "sum_open_interest": "bn_oi",
        "sum_open_interest_value": "bn_oi_value",
        "count_long_short_ratio": "bn_ls_ratio",
        "sum_taker_long_short_vol_ratio": "bn_taker_ls_ratio"
    })
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a merged 1m DataFrame to 5m bars."""
    agg = {}
    for prefix in ["bb", "bn"]:
        o, h, l, c = f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"
        if o in df.columns:
            agg[o] = "first"
            agg[h] = "max"
            agg[l] = "min"
            agg[c] = "last"
        vol = f"{prefix}_volume"
        if vol in df.columns:
            agg[vol] = "sum"
        turn = f"{prefix}_turnover"
        if turn in df.columns:
            agg[turn] = "sum"

    # Binance extra columns
    for col in ["bn_trades", "bn_taker_buy_vol", "bn_taker_buy_turnover"]:
        if col in df.columns:
            agg[col] = "sum"

    # Premium — take last value
    for col in ["bb_premium", "bn_premium"]:
        if col in df.columns:
            agg[col] = "last"

    if not agg:
        return df

    return df.resample("5min").agg(agg).dropna(subset=[c for c in agg if c.endswith("_close")])


def load_symbol(sym: str, resample: str = "5min") -> pd.DataFrame:
    """
    Load all data for a symbol, merge Bybit + Binance, resample to 5m.

    Returns a DataFrame indexed by timestamp with columns:
    - bb_open/high/low/close/volume/turnover, bb_premium
    - bn_open/high/low/close/volume/turnover/trades/taker_buy_vol, bn_premium
    - bb_fr, bb_oi, bb_ls_ratio
    - bn_oi, bn_oi_value, bn_ls_ratio, bn_taker_ls_ratio
    """
    # Load 1m klines
    bb_kl = _load_bybit_klines(sym)
    bn_kl = _load_binance_klines(sym)

    if bb_kl.empty or bn_kl.empty:
        return pd.DataFrame()

    # Merge klines on timestamp
    merged = bb_kl.join(bn_kl, how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Load and merge premium indices
    bb_prem = _load_bybit_premium(sym)
    bn_prem = _load_binance_premium(sym)
    if not bb_prem.empty:
        merged = merged.join(bb_prem, how="left")
    if not bn_prem.empty:
        merged = merged.join(bn_prem, how="left")

    # Resample to 5m
    if resample == "5min":
        merged = _resample_5m(merged)

    # Load and merge 5min data (OI, LS, funding, metrics)
    bb_fr = _load_bybit_funding(sym)
    bb_oi = _load_bybit_oi(sym)
    bb_ls = _load_bybit_ls(sym)
    bn_met = _load_binance_metrics(sym)

    if not bb_fr.empty:
        merged = merged.join(bb_fr, how="left")
        merged["bb_fr"] = merged["bb_fr"].ffill()
    if not bb_oi.empty:
        merged = merged.join(bb_oi, how="left")
        merged["bb_oi"] = merged["bb_oi"].ffill()
    if not bb_ls.empty:
        merged = merged.join(bb_ls, how="left")
        merged["bb_ls_ratio"] = merged["bb_ls_ratio"].ffill()
    if not bn_met.empty:
        merged = merged.join(bn_met, how="left")
        for col in ["bn_oi", "bn_oi_value", "bn_ls_ratio", "bn_taker_ls_ratio"]:
            if col in merged.columns:
                merged[col] = merged[col].ffill()

    merged = merged.dropna(subset=["bb_close", "bn_close"])
    return merged


def load_all_symbols(symbols: list[str] | None = None,
                     n_jobs: int = 8,
                     resample: str = "5min") -> dict[str, pd.DataFrame]:
    """Load all symbols in parallel. Returns {symbol: DataFrame}."""
    if symbols is None:
        symbols = list_common_symbols()

    result = {}
    t0 = time.time()
    total = len(symbols)

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(load_symbol, sym, resample): sym for sym in symbols}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            try:
                df = fut.result()
                if not df.empty:
                    result[sym] = df
                    if done % 10 == 0 or done == total:
                        elapsed = time.time() - t0
                        print(f"  [{done}/{total}] loaded {sym}: {len(df)} rows "
                              f"({elapsed:.1f}s elapsed)")
                else:
                    print(f"  [{done}/{total}] {sym}: EMPTY — skipped")
            except Exception as e:
                print(f"  [{done}/{total}] {sym}: ERROR — {e}")

    elapsed = time.time() - t0
    print(f"\nLoaded {len(result)}/{total} symbols in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    symbols = list_common_symbols()
    print(f"Found {len(symbols)} common symbols")

    # Load one symbol as a test
    t0 = time.time()
    df = load_symbol("SOLUSDT")
    print(f"\nSOLUSDT: {len(df)} rows, {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} — {df.index.max()}")
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"\nSample:\n{df.head(3).to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
