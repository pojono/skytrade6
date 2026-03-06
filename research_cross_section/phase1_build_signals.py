"""
Phase 1 - Step 1: Build per-symbol hourly signal files.

RAM-efficient: streams one day at a time, resamples to 1h on the fly,
never holds more than one day of 1m data in memory at once.

Output: research_cross_section/signals/{SYMBOL}.parquet
  Columns: close, volume, funding, premium, oi, ls_buy,
           mom_1h/2h/4h/8h/24h/48h, prem_z, oi_div, ls_z,
           fwd_1h/4h/8h/24h
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import psutil
import time

warnings.filterwarnings("ignore")

DATALAKE     = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
OUT_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
UNIVERSE_FILE = "/home/ubuntu/Projects/skytrade6/research_cross_section/universe.txt"
Z_WINDOW     = 30 * 24   # 30-day rolling window for z-scores (hours)


def ram_mb():
    return psutil.virtual_memory().used / 1e6


# ---------------------------------------------------------------------------
# Streaming loaders — one day file at a time
# ---------------------------------------------------------------------------

def stream_kline_1h(symbol_dir):
    """Stream 1m kline day-files → yield concatenated 1h OHLCV bars."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_kline_1m.csv")))
    chunks = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["startTime", "open", "high", "low", "close", "volume", "turnover"])
            d["startTime"] = pd.to_datetime(d["startTime"], unit="ms", utc=True)
            d = d.set_index("startTime").sort_index()
            h = d.resample("1h").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"),   close=("close", "last"),
                volume=("volume", "sum"), turnover=("turnover", "sum"),
            ).dropna(subset=["close"])
            chunks.append(h)
        except Exception:
            pass
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).sort_index()


def stream_funding(symbol_dir, h_index):
    """Load all funding rate settlements → forward-fill onto 1h index."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_funding_rate.csv")))
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["timestamp", "fundingRate"])
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.Series(np.nan, index=h_index, name="funding")
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()["fundingRate"]
    df = df[~df.index.duplicated(keep="last")]
    return df.reindex(h_index, method="ffill").rename("funding")


def stream_premium_1h(symbol_dir, h_index):
    """Stream 1m premium index day-files → resample to 1h close."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_premium_index_kline_1m.csv")))
    chunks = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["startTime", "close"])
            d["startTime"] = pd.to_datetime(d["startTime"], unit="ms", utc=True)
            d = d.set_index("startTime").sort_index()
            h = d["close"].resample("1h").last().dropna()
            chunks.append(h)
        except Exception:
            pass
    if not chunks:
        return pd.Series(np.nan, index=h_index, name="premium")
    s = pd.concat(chunks).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(h_index).rename("premium")


def stream_oi_1h(symbol_dir, h_index):
    """Stream 5min OI day-files → resample to 1h last."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_open_interest_5min.csv")))
    chunks = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["timestamp", "openInterest"])
            d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
            d = d.set_index("timestamp").sort_index()
            h = d["openInterest"].resample("1h").last().dropna()
            chunks.append(h)
        except Exception:
            pass
    if not chunks:
        return pd.Series(np.nan, index=h_index, name="oi")
    s = pd.concat(chunks).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(h_index).rename("oi")


def stream_ls_1h(symbol_dir, h_index):
    """Stream 5min L/S ratio day-files → resample to 1h last."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_long_short_ratio_5min.csv")))
    chunks = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["timestamp", "buyRatio"])
            d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
            d = d.set_index("timestamp").sort_index()
            h = d["buyRatio"].resample("1h").last().dropna()
            chunks.append(h)
        except Exception:
            pass
    if not chunks:
        return pd.Series(np.nan, index=h_index, name="ls_buy")
    s = pd.concat(chunks).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(h_index).rename("ls_buy")


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------

def build_symbol(symbol):
    base = os.path.join(DATALAKE, symbol)

    # 1h OHLCV — this is the core; peak RAM = 1 day of 1m data during streaming
    df = stream_kline_1h(base)
    if df.empty or len(df) < 200:
        return None

    h_index = df.index

    # Auxiliary series aligned to 1h index
    df["funding"] = stream_funding(base, h_index)
    df["premium"] = stream_premium_1h(base, h_index)
    df["oi"]      = stream_oi_1h(base, h_index)
    df["ls_buy"]  = stream_ls_1h(base, h_index)

    # ------------------------------------------------------------------
    # Data quality: remove zero/corrupt close prices before any signal
    # (observed: 2025-01-01 has zeroed close prices for many symbols)
    # ------------------------------------------------------------------
    price_median = df["close"].median()
    bad_price = (df["close"] <= 0) | (df["close"] < price_median * 0.01)
    if bad_price.sum() > 0:
        df.loc[bad_price, "close"] = np.nan
        df["close"] = df["close"].interpolate(method="time", limit=12)
        df.dropna(subset=["close"], inplace=True)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    # A — momentum at multiple lookbacks
    for w in [1, 2, 4, 8, 24, 48]:
        df[f"mom_{w}h"] = df["close"].pct_change(w, fill_method=None)

    # B — funding carry: raw rate (forward-filled settlement value)
    #     already in df["funding"]

    # E — premium z-score (positive = futures rich, short signal)
    prem_mean    = df["premium"].rolling(Z_WINDOW, min_periods=48).mean()
    prem_std     = df["premium"].rolling(Z_WINDOW, min_periods=48).std().replace(0, np.nan)
    df["prem_z"] = (df["premium"] - prem_mean) / prem_std

    # D — OI-price divergence: product of 1h returns
    #     positive = price & OI moved together (momentum)
    #     negative = divergence (mean-reversion signal)
    price_ret1h  = df["close"].pct_change(1, fill_method=None)
    oi_ret1h     = df["oi"].pct_change(1, fill_method=None)
    df["oi_div"] = price_ret1h * oi_ret1h

    # F — L/S ratio z-score (positive = longs crowded, bearish contrarian)
    ls_mean     = df["ls_buy"].rolling(Z_WINDOW, min_periods=48).mean()
    ls_std      = df["ls_buy"].rolling(Z_WINDOW, min_periods=48).std().replace(0, np.nan)
    df["ls_z"]  = (df["ls_buy"] - ls_mean) / ls_std

    # ------------------------------------------------------------------
    # Forward returns — pre-computed for IC step
    # fwd_Nh[t] = (close[t+N] - close[t]) / close[t]
    # ------------------------------------------------------------------
    for h in [1, 4, 8, 24]:
        df[f"fwd_{h}h"] = df["close"].pct_change(h, fill_method=None).shift(-h)

    # Drop OHLC columns we no longer need to save space
    keep = (
        ["close", "volume", "turnover", "funding", "premium", "oi", "ls_buy"]
        + [f"mom_{w}h" for w in [1, 2, 4, 8, 24, 48]]
        + ["prem_z", "oi_div", "ls_z"]
        + [f"fwd_{h}h" for h in [1, 4, 8, 24]]
    )
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(UNIVERSE_FILE) as f:
        universe = [l.strip() for l in f if l.strip()]

    print(f"Universe: {len(universe)} symbols")
    print(f"Output: {OUT_DIR}")
    print()

    t0   = time.time()
    ok   = 0
    skip = 0

    for i, symbol in enumerate(universe):
        out_path = os.path.join(OUT_DIR, f"{symbol}.parquet")
        if os.path.exists(out_path):
            ok += 1
            if (i + 1) % 25 == 0 or i == len(universe) - 1:
                print(f"  [{i+1:3d}/{len(universe)}] {symbol:<20} cached   "
                      f"RAM={ram_mb():.0f}MB  {time.time()-t0:.0f}s")
            continue

        try:
            df = build_symbol(symbol)
            if df is None:
                skip += 1
                print(f"  [{i+1:3d}/{len(universe)}] {symbol:<20} SKIP (too few bars)")
                continue
            df.to_parquet(out_path)
            ok += 1
            print(f"  [{i+1:3d}/{len(universe)}] {symbol:<20} {len(df):5d} h-bars  "
                  f"RAM={ram_mb():.0f}MB  {time.time()-t0:.0f}s")
        except Exception as e:
            skip += 1
            print(f"  [{i+1:3d}/{len(universe)}] {symbol:<20} ERROR: {e}")

    print()
    print(f"Done. {ok} built, {skip} skipped. Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
