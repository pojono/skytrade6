from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
TRADES_INPUT = OUT_DIR / "revalidated_exp2_symbol_trades.csv"

DETAIL_CSV = OUT_DIR / "trade_entry_feasibility_full.csv"
SUMMARY_CSV = OUT_DIR / "trade_entry_feasibility_summary_full.csv"
SUMMARY_MD = OUT_DIR / "TRADE_ENTRY_FINDINGS_full.md"

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


def load_binance_trades(symbol: str, date_str: str) -> pd.DataFrame | None:
    path = ROOT / "datalake" / "binance" / symbol / f"{date_str}_trades.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["time", "price", "qty"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    return df.dropna().sort_values("time")


def load_bybit_trades(symbol: str, date_str: str) -> pd.DataFrame | None:
    path = ROOT / "datalake" / "bybit" / symbol / f"{date_str}_trades.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["timestamp", "price", "size"])
    df["time"] = pd.to_numeric(df["timestamp"], errors="coerce")
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
        }

    first_price = float(window.iloc[0]["price"])
    qty_sum = float(window["qty"].sum())
    vwap = first_price if qty_sum <= 0 else float((window["price"] * window["qty"]).sum() / qty_sum)

    return {
        "trade_count": int(window.shape[0]),
        "maker_fill": float(float(window["price"].min()) <= ref_price),
        "first_trade_bps": (first_price / ref_price - 1.0) * 10000.0,
        "vwap_60s_bps": (vwap / ref_price - 1.0) * 10000.0,
    }


def analyze() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(TRADES_INPUT, parse_dates=["ts"])
    focus = trades.loc[trades["ts"] >= pd.Timestamp("2026-01-01 00:00:00+00:00")].copy()
    focus = focus.sort_values(["ts", "symbol"]).reset_index(drop=True)
    if focus.empty:
        raise RuntimeError("No test-period trades found.")

    kline_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    trade_cache: dict[tuple[str, str, str], pd.DataFrame | None] = {}
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
        if bn_ref_row.empty:
            continue
        bn_ref = float(bn_ref_row.iloc[0]["close"])

        bb_k_key = ("bybit", symbol, date_str)
        if bb_k_key not in kline_cache:
            kline_cache[bb_k_key] = load_bybit_kline_close(symbol, date_str)
        bb_ref_row = kline_cache[bb_k_key].loc[kline_cache[bb_k_key]["startTime"] == signal_minute_ms]
        if bb_ref_row.empty:
            continue
        bb_ref = float(bb_ref_row.iloc[0]["close"])

        bn_t_key = ("binance", symbol, date_str)
        if bn_t_key not in trade_cache:
            trade_cache[bn_t_key] = load_binance_trades(symbol, date_str)
        bb_t_key = ("bybit", symbol, date_str)
        if bb_t_key not in trade_cache:
            trade_cache[bb_t_key] = load_bybit_trades(symbol, date_str)

        bn_df = trade_cache[bn_t_key]
        bb_df = trade_cache[bb_t_key]
        if bn_df is None or bb_df is None:
            continue

        bn_window = bn_df.loc[(bn_df["time"] >= start_ms) & (bn_df["time"] < end_ms)]
        bb_window = bb_df.loc[(bb_df["time"] >= start_ms) & (bb_df["time"] < end_ms)]
        bn_stats = _compute_window_stats(bn_window, bn_ref)
        bb_stats = _compute_window_stats(bb_window, bb_ref)

        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "binance_maker_fill_60s": bn_stats["maker_fill"],
                "bybit_maker_fill_60s": bb_stats["maker_fill"],
                "binance_first_trade_bps": bn_stats["first_trade_bps"],
                "bybit_first_trade_bps": bb_stats["first_trade_bps"],
                "binance_vwap_60s_bps": bn_stats["vwap_60s_bps"],
                "bybit_vwap_60s_bps": bb_stats["vwap_60s_bps"],
            }
        )

    if not rows:
        raise RuntimeError("No rows were analyzable; trade files may still be missing.")

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
        .sort_values(["signals", "bn_vwap_60s_bps"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return detail, summary


def write_summary(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    total_signals = int(detail.shape[0])
    bn_fill = detail["binance_maker_fill_60s"].mean()
    bb_fill = detail["bybit_maker_fill_60s"].mean()
    bn_first = detail["binance_first_trade_bps"].mean()
    bb_first = detail["bybit_first_trade_bps"].mean()
    bn_vwap = detail["binance_vwap_60s_bps"].mean()
    bb_vwap = detail["bybit_vwap_60s_bps"].mean()
    worst = summary.sort_values("bn_vwap_60s_bps", ascending=False).head(10)

    lines = [
        "# Trade Entry Findings: Full Available 2026 Test Basket",
        "",
        "This extends the earlier top-3 trade-level check to all 2026 test-period strategy entries that currently have downloaded trade files on both Binance and Bybit.",
        "",
        "## Aggregate",
        "",
        f"- Signal rows checked: `{total_signals}`",
        f"- Symbols covered: `{summary.shape[0]}`",
        f"- Binance maker-fill proxy within 60s: `{bn_fill:.1%}`",
        f"- Bybit maker-fill proxy within 60s: `{bb_fill:.1%}`",
        f"- Binance first-trade drift: `{bn_first:+.2f} bps`",
        f"- Bybit first-trade drift: `{bb_first:+.2f} bps`",
        f"- Binance 60s VWAP drift: `{bn_vwap:+.2f} bps`",
        f"- Bybit 60s VWAP drift: `{bb_vwap:+.2f} bps`",
        "",
        "## Highest VWAP Drift Names",
        "",
        "| Symbol | Signals | Binance Fill | Bybit Fill | Binance VWAP | Bybit VWAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in worst.iterrows():
        lines.append(
            f"| {row['symbol']} | {int(row['signals'])} | {row['bn_maker_fill_rate']:.0%} | "
            f"{row['bb_maker_fill_rate']:.0%} | {row['bn_vwap_60s_bps']:+.2f} bps | "
            f"{row['bb_vwap_60s_bps']:+.2f} bps |"
        )
    lines.append("")
    lines.append("Files:")
    lines.append("")
    lines.append("- `trade_entry_feasibility_full.csv`")
    lines.append("- `trade_entry_feasibility_summary_full.csv`")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    detail, summary = analyze()
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_summary(detail, summary)
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
