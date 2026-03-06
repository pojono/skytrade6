"""
Phase 22 - Step 1: Build augmented signal parquets with real-time funding variants.

Takes existing signals/*.parquet (already built) and adds:
  predicted_funding  : running TWAP of premium_index_1m since last 8h settlement
                       + Bybit interest rate (0.0001). Updates every hour.
                       At T-1h before settlement = 87.5% complete → near-exact prediction.
  funding_cum24h     : sum of last 3 settled rates (24h carry accumulation)
  funding_cum72h     : sum of last 9 settled rates (72h carry trend)
  predicted_ft       : diff of predicted_funding vs 8h ago (trend of accruing funding)

Output: research_cross_section/signals_v2/{SYMBOL}.parquet
  Same columns as signals/ PLUS the four new funding variants.
"""

import os, glob, warnings, time
import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")

DATALAKE    = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SIGNALS_IN  = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
SIGNALS_OUT = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals_v2"
INTEREST_RATE = 0.0001   # Bybit standard: 0.01% per 8h settlement
FUNDING_CAP   = 0.0075   # Bybit max absolute funding rate per settlement


def ram_mb():
    return psutil.virtual_memory().used / 1e6


def load_premium_1m(symbol_dir, h_index):
    """Load all 1m premium index files → compute running 8h TWAP per 1h bar."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_premium_index_kline_1m.csv")))
    if not files:
        return pd.Series(np.nan, index=h_index, name="predicted_funding")

    chunks = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["startTime", "close"])
            d["startTime"] = pd.to_datetime(d["startTime"], unit="ms", utc=True)
            d = d.set_index("startTime").sort_index()
            # 1h mean (TWAP within hour, not just last tick)
            h = d["close"].resample("1h").mean().dropna()
            chunks.append(h)
        except Exception:
            pass

    if not chunks:
        return pd.Series(np.nan, index=h_index, name="predicted_funding")

    prem_1h = pd.concat(chunks).sort_index()
    prem_1h = prem_1h[~prem_1h.index.duplicated(keep="last")]
    prem_1h = prem_1h.reindex(h_index)

    # Running TWAP since last 8h settlement window start.
    # Bybit settles at 00:00, 08:00, 16:00 UTC.
    # Window start for each bar = floor(timestamp to 8h).
    window_start = h_index.floor("8h")

    df_tmp = pd.DataFrame({
        "prem":   prem_1h.values,
        "window": window_start,
    }, index=h_index)

    # Cumulative sum and count within each 8h window
    df_tmp["cumsum"]   = df_tmp.groupby("window")["prem"].cumsum()
    df_tmp["cumcount"] = df_tmp.groupby("window").cumcount() + 1
    df_tmp["running_twap"] = df_tmp["cumsum"] / df_tmp["cumcount"]

    # Add Bybit interest rate and clamp
    predicted = (df_tmp["running_twap"] + INTEREST_RATE).clip(-FUNDING_CAP, FUNDING_CAP)
    return predicted.rename("predicted_funding")


def load_funding_settlements(symbol_dir):
    """Load raw settlement timestamps → return Series indexed by settlement time."""
    files = sorted(glob.glob(os.path.join(symbol_dir, "*_funding_rate.csv")))
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["timestamp", "fundingRate"])
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.Series(dtype=float)
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()["fundingRate"]
    return df[~df.index.duplicated(keep="last")]


def build_cumulative_funding(settlements, h_index, n_periods):
    """Rolling sum of last n_periods settlements, forward-filled to 1h index."""
    if settlements.empty:
        return pd.Series(np.nan, index=h_index)
    cum = settlements.rolling(n_periods, min_periods=1).sum()
    return cum.reindex(h_index, method="ffill")


def augment_symbol(symbol):
    in_path  = os.path.join(SIGNALS_IN,  f"{symbol}.parquet")
    out_path = os.path.join(SIGNALS_OUT, f"{symbol}.parquet")

    if not os.path.exists(in_path):
        return None, "no source parquet"

    df = pd.read_parquet(in_path)
    h_index = df.index

    sym_dir = os.path.join(DATALAKE, symbol)

    # --- predicted_funding: real-time running TWAP of premium ---
    df["predicted_funding"] = load_premium_1m(sym_dir, h_index)

    # --- predicted_ft: trend of predicted_funding (8h diff) ---
    df["predicted_ft"] = df["predicted_funding"].diff(8)

    # --- cumulative settled funding ---
    settlements = load_funding_settlements(sym_dir)
    df["funding_cum24h"] = build_cumulative_funding(settlements, h_index, 3)   # 3 × 8h = 24h
    df["funding_cum72h"] = build_cumulative_funding(settlements, h_index, 9)   # 9 × 8h = 72h

    df.to_parquet(out_path)
    return len(df), None


def main():
    os.makedirs(SIGNALS_OUT, exist_ok=True)

    symbols = sorted([
        os.path.basename(f).replace(".parquet", "")
        for f in glob.glob(os.path.join(SIGNALS_IN, "*.parquet"))
    ])
    print(f"Augmenting {len(symbols)} symbols → {SIGNALS_OUT}")
    print()

    t0 = time.time()
    ok = skip = 0
    for i, sym in enumerate(symbols):
        out_path = os.path.join(SIGNALS_OUT, f"{sym}.parquet")
        if os.path.exists(out_path):
            ok += 1
            if (i + 1) % 25 == 0 or i == len(symbols) - 1:
                print(f"  [{i+1:3d}/{len(symbols)}] {sym:<22} cached   "
                      f"RAM={ram_mb():.0f}MB  {time.time()-t0:.0f}s")
            continue
        n, err = augment_symbol(sym)
        if err:
            skip += 1
            print(f"  [{i+1:3d}/{len(symbols)}] {sym:<22} SKIP: {err}")
        else:
            ok += 1
            if (i + 1) % 10 == 0 or i == len(symbols) - 1:
                print(f"  [{i+1:3d}/{len(symbols)}] {sym:<22} {n:5d} bars  "
                      f"RAM={ram_mb():.0f}MB  {time.time()-t0:.0f}s")

    print()
    print(f"Done. {ok} built/cached, {skip} skipped. Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
