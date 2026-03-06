#!/usr/bin/env python3
"""
Build a daily cross-exchange money-flow dataset for the shared Bybit/Binance universe.

The script streams one symbol/day at a time from the datalake and writes:
  - codex-7/out/symbol_daily_flows.csv
  - codex-7/out/dataset_summary.json

It keeps the feature set intentionally interpretable:
  futures turnover, spot turnover, premium, funding, open interest, long/short ratio,
  exchange share, and a few basic coverage flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BYBIT_DIR = ROOT / "datalake" / "bybit"
BINANCE_DIR = ROOT / "datalake" / "binance"
OUT_DIR = ROOT / "codex-7" / "out"
OUT_CSV = OUT_DIR / "symbol_daily_flows.csv"
OUT_SUMMARY = OUT_DIR / "dataset_summary.json"

BINANCE_SPOT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class DayPaths:
    bybit_fut: Path | None
    bybit_spot: Path | None
    bybit_premium: Path | None
    bybit_funding: Path | None
    bybit_oi: Path | None
    bybit_ls: Path | None
    binance_fut: Path | None
    binance_spot: Path | None
    binance_premium: Path | None
    binance_metrics: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="")
    return parser.parse_args()


def all_symbols() -> list[str]:
    bybit_symbols = {p.name for p in BYBIT_DIR.iterdir() if p.is_dir()}
    binance_symbols = {p.name for p in BINANCE_DIR.iterdir() if p.is_dir()}
    return sorted(bybit_symbols & binance_symbols)


def extract_date(path: Path, suffix: str) -> str | None:
    if not path.name.endswith(suffix):
        return None
    date_part = path.name[: -len(suffix)]
    if len(date_part) != 10:
        return None
    if date_part[4] != "-" or date_part[7] != "-":
        return None
    return date_part


def date_set(symbol_dir: Path, suffix: str) -> set[str]:
    out: set[str] = set()
    for path in symbol_dir.glob(f"*{suffix}"):
        dt = extract_date(path, suffix)
        if dt:
            out.add(dt)
    return out


def read_bybit_kline(path: Path) -> dict[str, float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "close" not in df.columns:
        return {}
    close = pd.to_numeric(df["close"], errors="coerce")
    if "turnover" in df.columns:
        turnover = pd.to_numeric(df["turnover"], errors="coerce")
    elif "volume" in df.columns:
        turnover = pd.to_numeric(df["volume"], errors="coerce") * close
    else:
        turnover = pd.Series(np.nan, index=df.index)
    close = close.dropna()
    if close.empty:
        return {}
    return {
        "close": float(close.iloc[-1]),
        "turnover": float(turnover.fillna(0).sum()),
    }


def read_bybit_premium(path: Path) -> float:
    try:
        df = pd.read_csv(path, usecols=["close"])
    except Exception:
        return np.nan
    if df.empty:
        return np.nan
    return float(df["close"].mean())


def read_bybit_funding(path: Path) -> float:
    try:
        df = pd.read_csv(path, usecols=["fundingRate"])
    except Exception:
        return np.nan
    if df.empty:
        return np.nan
    return float(df["fundingRate"].sum())


def read_bybit_oi(path: Path, close_px: float) -> float:
    try:
        df = pd.read_csv(path, usecols=["openInterest"])
    except Exception:
        return np.nan
    if df.empty or not np.isfinite(close_px):
        return np.nan
    return float(df["openInterest"].iloc[-1]) * float(close_px)


def read_bybit_ls(path: Path) -> float:
    try:
        df = pd.read_csv(path, usecols=["buyRatio"])
    except Exception:
        return np.nan
    if df.empty:
        return np.nan
    return float(df["buyRatio"].iloc[-1])


def read_binance_fut_kline(path: Path) -> dict[str, float]:
    try:
        df = pd.read_csv(path, usecols=["close", "quote_volume", "taker_buy_quote_volume"])
    except Exception:
        return {}
    if df.empty:
        return {}
    return {
        "close": float(df["close"].iloc[-1]),
        "turnover": float(df["quote_volume"].sum()),
        "taker_buy_quote": float(df["taker_buy_quote_volume"].sum()),
    }


def read_binance_spot_kline(path: Path) -> dict[str, float]:
    try:
        df = pd.read_csv(path, names=BINANCE_SPOT_COLUMNS, header=None, usecols=["close", "quote_volume", "taker_buy_quote_volume"])
    except Exception:
        return {}
    if df.empty:
        return {}
    return {
        "close": float(df["close"].iloc[-1]),
        "turnover": float(df["quote_volume"].sum()),
        "taker_buy_quote": float(df["taker_buy_quote_volume"].sum()),
    }


def read_binance_premium(path: Path) -> float:
    try:
        df = pd.read_csv(path, usecols=["close"])
    except Exception:
        return np.nan
    if df.empty:
        return np.nan
    return float(df["close"].mean())


def read_binance_metrics(path: Path) -> dict[str, float]:
    usecols = [
        "sum_open_interest_value",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return {}
    if df.empty:
        return {}
    return {
        "oi_usd": float(df["sum_open_interest_value"].iloc[-1]),
        "ls_ratio": float(df["count_long_short_ratio"].iloc[-1]),
        "taker_ls_ratio": float(df["sum_taker_long_short_vol_ratio"].mean()),
    }


def pick_path(base: Path, date_str: str, suffix: str) -> Path | None:
    path = base / f"{date_str}{suffix}"
    return path if path.exists() else None


def build_day_paths(symbol: str, date_str: str) -> DayPaths:
    bybit_base = BYBIT_DIR / symbol
    binance_base = BINANCE_DIR / symbol
    return DayPaths(
        bybit_fut=pick_path(bybit_base, date_str, "_kline_1m.csv"),
        bybit_spot=pick_path(bybit_base, date_str, "_kline_1m_spot.csv"),
        bybit_premium=pick_path(bybit_base, date_str, "_premium_index_kline_1m.csv"),
        bybit_funding=pick_path(bybit_base, date_str, "_funding_rate.csv"),
        bybit_oi=pick_path(bybit_base, date_str, "_open_interest_5min.csv"),
        bybit_ls=pick_path(bybit_base, date_str, "_long_short_ratio_5min.csv"),
        binance_fut=pick_path(binance_base, date_str, "_kline_1m.csv"),
        binance_spot=pick_path(binance_base, date_str, "_kline_1m_spot.csv"),
        binance_premium=pick_path(binance_base, date_str, "_premium_index_kline_1m.csv"),
        binance_metrics=pick_path(binance_base, date_str, "_metrics.csv"),
    )


def row_from_paths(symbol: str, date_str: str, paths: DayPaths) -> dict[str, object] | None:
    if paths.bybit_fut is None or paths.binance_fut is None:
        return None

    row: dict[str, object] = {"symbol": symbol, "date": date_str}

    bybit_fut = read_bybit_kline(paths.bybit_fut)
    binance_fut = read_binance_fut_kline(paths.binance_fut)
    if not bybit_fut or not binance_fut:
        return None
    if bybit_fut["close"] <= 0 or binance_fut["close"] <= 0:
        return None

    row["bybit_fut_close"] = bybit_fut["close"]
    row["bybit_fut_turnover"] = bybit_fut["turnover"]
    row["binance_fut_close"] = binance_fut["close"]
    row["binance_fut_turnover"] = binance_fut["turnover"]
    row["binance_fut_taker_buy_quote"] = binance_fut["taker_buy_quote"]

    if paths.bybit_spot:
        bybit_spot = read_bybit_kline(paths.bybit_spot)
        row["bybit_spot_close"] = bybit_spot.get("close", np.nan)
        row["bybit_spot_turnover"] = bybit_spot.get("turnover", np.nan)
    else:
        row["bybit_spot_close"] = np.nan
        row["bybit_spot_turnover"] = np.nan

    if paths.binance_spot:
        binance_spot = read_binance_spot_kline(paths.binance_spot)
        row["binance_spot_close"] = binance_spot.get("close", np.nan)
        row["binance_spot_turnover"] = binance_spot.get("turnover", np.nan)
        row["binance_spot_taker_buy_quote"] = binance_spot.get("taker_buy_quote", np.nan)
    else:
        row["binance_spot_close"] = np.nan
        row["binance_spot_turnover"] = np.nan
        row["binance_spot_taker_buy_quote"] = np.nan

    row["bybit_premium_mean"] = read_bybit_premium(paths.bybit_premium) if paths.bybit_premium else np.nan
    row["binance_premium_mean"] = read_binance_premium(paths.binance_premium) if paths.binance_premium else np.nan
    row["bybit_funding_sum"] = read_bybit_funding(paths.bybit_funding) if paths.bybit_funding else np.nan
    row["bybit_oi_usd"] = read_bybit_oi(paths.bybit_oi, row["bybit_fut_close"]) if paths.bybit_oi else np.nan
    row["bybit_ls_buy"] = read_bybit_ls(paths.bybit_ls) if paths.bybit_ls else np.nan

    if paths.binance_metrics:
        binance_metrics = read_binance_metrics(paths.binance_metrics)
        row["binance_oi_usd"] = binance_metrics.get("oi_usd", np.nan)
        row["binance_ls_ratio"] = binance_metrics.get("ls_ratio", np.nan)
        row["binance_taker_ls_ratio"] = binance_metrics.get("taker_ls_ratio", np.nan)
    else:
        row["binance_oi_usd"] = np.nan
        row["binance_ls_ratio"] = np.nan
        row["binance_taker_ls_ratio"] = np.nan

    row["has_bybit_spot"] = int(paths.bybit_spot is not None)
    row["has_binance_spot"] = int(paths.binance_spot is not None)
    row["has_full_spot_pair"] = int(paths.bybit_spot is not None and paths.binance_spot is not None)
    row["has_bybit_oi"] = int(paths.bybit_oi is not None)
    row["has_binance_metrics"] = int(paths.binance_metrics is not None)
    row["futures_pair_gap_bps"] = (row["bybit_fut_close"] / row["binance_fut_close"] - 1.0) * 10000.0

    row["total_fut_turnover"] = float(row["bybit_fut_turnover"]) + float(row["binance_fut_turnover"])
    row["total_spot_turnover"] = float(np.nan_to_num(row["bybit_spot_turnover"])) + float(np.nan_to_num(row["binance_spot_turnover"]))
    row["bybit_total_turnover"] = float(row["bybit_fut_turnover"]) + float(np.nan_to_num(row["bybit_spot_turnover"]))
    row["binance_total_turnover"] = float(row["binance_fut_turnover"]) + float(np.nan_to_num(row["binance_spot_turnover"]))
    row["total_oi_usd"] = float(np.nan_to_num(row["bybit_oi_usd"])) + float(np.nan_to_num(row["binance_oi_usd"]))

    return row


def symbol_dates(symbol: str, start_date: str, end_date: str) -> list[str]:
    bybit_base = BYBIT_DIR / symbol
    binance_base = BINANCE_DIR / symbol
    common = date_set(bybit_base, "_kline_1m.csv") & date_set(binance_base, "_kline_1m.csv")
    return sorted(d for d in common if start_date <= d <= end_date)


def build_dataset(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        dates = symbol_dates(symbol, start_date, end_date)
        print(f"[{idx:03d}/{len(symbols):03d}] {symbol}: {len(dates)} futures-common days", flush=True)
        for date_str in dates:
            row = row_from_paths(symbol, date_str, build_day_paths(symbol, date_str))
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def build_summary(df: pd.DataFrame, symbols: list[str], start_date: str, end_date: str) -> dict[str, object]:
    if df.empty:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "symbols_requested": len(symbols),
            "rows": 0,
        }

    full_spot_symbols = df.groupby("symbol")["has_full_spot_pair"].max()
    return {
        "start_date": start_date,
        "end_date": end_date,
        "symbols_requested": len(symbols),
        "symbols_with_futures_rows": int(df["symbol"].nunique()),
        "symbols_with_four_way_spot": int((full_spot_symbols > 0).sum()),
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "mean_symbols_per_day": float(df.groupby("date")["symbol"].nunique().mean()),
        "mean_full_spot_symbols_per_day": float(df.groupby("date")["has_full_spot_pair"].sum().mean()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    symbols = all_symbols()
    if args.symbols:
        requested = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols = [s for s in symbols if s in requested]
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]

    df = build_dataset(symbols, args.start_date, args.end_date)
    df.to_csv(OUT_CSV, index=False)

    summary = build_summary(df, symbols, args.start_date, args.end_date)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
