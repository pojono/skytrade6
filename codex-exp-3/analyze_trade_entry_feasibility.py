from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
TRADES_INPUT = OUT_DIR / "revalidated_exp2_symbol_trades.csv"

DETAIL_CSV = OUT_DIR / "trade_entry_feasibility_top3.csv"
SUMMARY_CSV = OUT_DIR / "trade_entry_feasibility_summary_top3.csv"

TOP_SYMBOLS = ("WIFUSDT", "LINKUSDT", "LTCUSDT")
WINDOW_MS = 60_000


def load_binance_kline_close(symbol: str, date_str: str) -> pd.DataFrame:
    path = ROOT / "datalake" / "binance" / symbol / f"{date_str}_kline_1m.csv"
    df = pd.read_csv(path, usecols=["open_time", "close"])
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna()


def load_bybit_kline_close(symbol: str, date_str: str) -> pd.DataFrame:
    path = ROOT / "datalake" / "bybit" / symbol / f"{date_str}_kline_1m.csv"
    df = pd.read_csv(path, usecols=["startTime", "close"])
    df["startTime"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna()


def load_binance_trades(symbol: str, date_str: str) -> pd.DataFrame:
    path = ROOT / "datalake" / "binance" / symbol / f"{date_str}_trades.csv"
    df = pd.read_csv(path, usecols=["time", "price", "qty"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    return df.dropna().sort_values("time")


def load_bybit_trades(symbol: str, date_str: str) -> pd.DataFrame:
    path = ROOT / "datalake" / "bybit" / symbol / f"{date_str}_trades.csv"
    df = pd.read_csv(path, usecols=["timestamp", "price", "size"])
    df["time"] = pd.to_numeric(df["timestamp"], errors="coerce")
    # Bybit timestamps are in seconds with fractions; convert to ms.
    df["time"] = (df["time"] * 1000.0).round().astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["size"], errors="coerce")
    return df[["time", "price", "qty"]].dropna().sort_values("time")


def _compute_window_stats(window: pd.DataFrame, ref_price: float) -> dict[str, float]:
    if window.empty or ref_price <= 0:
        return {
            "trade_count": 0,
            "maker_fill": 0.0,
            "first_trade_bps": float("nan"),
            "vwap_60s_bps": float("nan"),
            "best_price_bps": float("nan"),
            "worst_price_bps": float("nan"),
        }

    first_price = float(window.iloc[0]["price"])
    best_price = float(window["price"].min())
    worst_price = float(window["price"].max())
    qty_sum = float(window["qty"].sum())
    if qty_sum > 0:
        vwap = float((window["price"] * window["qty"]).sum() / qty_sum)
    else:
        vwap = first_price

    return {
        "trade_count": int(window.shape[0]),
        "maker_fill": float(best_price <= ref_price),
        "first_trade_bps": (first_price / ref_price - 1.0) * 10000.0,
        "vwap_60s_bps": (vwap / ref_price - 1.0) * 10000.0,
        "best_price_bps": (best_price / ref_price - 1.0) * 10000.0,
        "worst_price_bps": (worst_price / ref_price - 1.0) * 10000.0,
    }


def analyze() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(TRADES_INPUT, parse_dates=["ts"])
    focus = trades.loc[
        (trades["ts"] >= pd.Timestamp("2026-01-01 00:00:00+00:00"))
        & (trades["symbol"].isin(TOP_SYMBOLS))
    ].copy()
    focus = focus.sort_values(["ts", "symbol"]).reset_index(drop=True)

    if focus.empty:
        raise RuntimeError("No test-period trades found for top symbols.")

    kline_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    trade_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows = []

    for _, row in focus.iterrows():
        symbol = row["symbol"]
        ts = row["ts"]
        date_str = ts.strftime("%Y-%m-%d")
        signal_minute_ms = int((ts - pd.Timedelta(minutes=1)).timestamp() * 1000)
        start_ms = int(ts.timestamp() * 1000)
        end_ms = start_ms + WINDOW_MS

        bn_k_key = ("binance", symbol, date_str)
        if bn_k_key not in kline_cache:
            kline_cache[bn_k_key] = load_binance_kline_close(symbol, date_str)
        bn_ref_row = kline_cache[bn_k_key].loc[kline_cache[bn_k_key]["open_time"] == signal_minute_ms]
        bn_ref = float(bn_ref_row.iloc[0]["close"])

        bb_k_key = ("bybit", symbol, date_str)
        if bb_k_key not in kline_cache:
            kline_cache[bb_k_key] = load_bybit_kline_close(symbol, date_str)
        bb_ref_row = kline_cache[bb_k_key].loc[kline_cache[bb_k_key]["startTime"] == signal_minute_ms]
        bb_ref = float(bb_ref_row.iloc[0]["close"])

        bn_t_key = ("binance", symbol, date_str)
        if bn_t_key not in trade_cache:
            trade_cache[bn_t_key] = load_binance_trades(symbol, date_str)
        bn_window = trade_cache[bn_t_key].loc[
            (trade_cache[bn_t_key]["time"] >= start_ms) & (trade_cache[bn_t_key]["time"] < end_ms)
        ]
        bn_stats = _compute_window_stats(bn_window, bn_ref)

        bb_t_key = ("bybit", symbol, date_str)
        if bb_t_key not in trade_cache:
            trade_cache[bb_t_key] = load_bybit_trades(symbol, date_str)
        bb_window = trade_cache[bb_t_key].loc[
            (trade_cache[bb_t_key]["time"] >= start_ms) & (trade_cache[bb_t_key]["time"] < end_ms)
        ]
        bb_stats = _compute_window_stats(bb_window, bb_ref)

        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "binance_ref_price": bn_ref,
                "bybit_ref_price": bb_ref,
                "binance_trade_count_60s": bn_stats["trade_count"],
                "binance_maker_fill_60s": bn_stats["maker_fill"],
                "binance_first_trade_bps": bn_stats["first_trade_bps"],
                "binance_vwap_60s_bps": bn_stats["vwap_60s_bps"],
                "binance_best_price_bps": bn_stats["best_price_bps"],
                "binance_worst_price_bps": bn_stats["worst_price_bps"],
                "bybit_trade_count_60s": bb_stats["trade_count"],
                "bybit_maker_fill_60s": bb_stats["maker_fill"],
                "bybit_first_trade_bps": bb_stats["first_trade_bps"],
                "bybit_vwap_60s_bps": bb_stats["vwap_60s_bps"],
                "bybit_best_price_bps": bb_stats["best_price_bps"],
                "bybit_worst_price_bps": bb_stats["worst_price_bps"],
            }
        )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("symbol", as_index=False)
        .agg(
            signals=("symbol", "count"),
            bn_maker_fill_rate=("binance_maker_fill_60s", "mean"),
            bb_maker_fill_rate=("bybit_maker_fill_60s", "mean"),
            bn_first_trade_bps=("binance_first_trade_bps", "mean"),
            bb_first_trade_bps=("bybit_first_trade_bps", "mean"),
            bn_vwap_60s_bps=("binance_vwap_60s_bps", "mean"),
            bb_vwap_60s_bps=("bybit_vwap_60s_bps", "mean"),
        )
        .sort_values("signals", ascending=False)
        .reset_index(drop=True)
    )
    return detail, summary


def main() -> None:
    detail, summary = analyze()
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
