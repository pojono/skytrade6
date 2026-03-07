#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_CSV = OUT_DIR / "dislocation_panel.csv.gz"
OUT_SUMMARY = OUT_DIR / "dislocation_panel_summary.json"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]

BINANCE_KLINE_COLUMNS = [
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

BINANCE_PREMIUM_COLUMNS = [
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
class DayFrame:
    frame: pd.DataFrame
    source_rows: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a joined minute-level Bybit/Binance dislocation panel.")
    parser.add_argument("--start-date", default="2025-07-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output", default=str(OUT_CSV))
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def parse_symbols(raw: str) -> list[str]:
    symbols = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return symbols if symbols else list(DEFAULT_SYMBOLS)


def day_set(exchange: str, symbol: str, suffix: str) -> set[str]:
    base = DATALAKE / exchange / symbol
    if not base.exists():
        return set()
    out: set[str] = set()
    for path in base.glob(f"*{suffix}"):
        day = path.name[:10]
        if len(day) == 10:
            out.add(day)
    return out


def common_days(symbol: str, start_date: str, end_date: str) -> list[str]:
    bn = day_set("binance", symbol, "_kline_1m.csv")
    bb = day_set("bybit", symbol, "_kline_1m.csv")
    return [day for day in sorted(bn & bb) if start_date <= day <= end_date]


def binance_has_header(path: Path) -> bool:
    with path.open("r", encoding="ascii") as fh:
        first = fh.readline().strip().split(",", 1)[0]
    return first == "open_time"


def read_binance_kline_like(path: Path, usecols: list[str]) -> pd.DataFrame:
    if binance_has_header(path):
        return pd.read_csv(path, usecols=usecols)
    return pd.read_csv(path, names=BINANCE_KLINE_COLUMNS, header=None, usecols=usecols)


def load_binance_day(symbol: str, day: str) -> tuple[pd.DataFrame, dict[str, int]] | None:
    base = DATALAKE / "binance" / symbol
    kline_path = base / f"{day}_kline_1m.csv"
    premium_path = base / f"{day}_premium_index_kline_1m.csv"
    metrics_path = base / f"{day}_metrics.csv"
    if not kline_path.exists():
        return None

    k = read_binance_kline_like(kline_path, ["open_time", "close", "quote_volume", "taker_buy_quote_volume"])
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.rename(
        columns={
            "close": "bn_close",
            "quote_volume": "bn_quote_volume",
            "taker_buy_quote_volume": "bn_taker_buy_quote_volume",
        }
    )
    for col in ["bn_close", "bn_quote_volume", "bn_taker_buy_quote_volume"]:
        k[col] = pd.to_numeric(k[col], errors="coerce")
    k = k[["ts", "bn_close", "bn_quote_volume", "bn_taker_buy_quote_volume"]].sort_values("ts")

    premium_rows = 0
    if premium_path.exists():
        premium = read_binance_kline_like(premium_path, ["open_time", "close"])
        premium["ts"] = pd.to_datetime(premium["open_time"], unit="ms", utc=True)
        premium = premium.rename(columns={"close": "bn_premium"})
        premium["bn_premium"] = pd.to_numeric(premium["bn_premium"], errors="coerce")
        premium = premium[["ts", "bn_premium"]].sort_values("ts")
        premium_rows = len(premium)
        k = k.merge(premium, on="ts", how="left")
    else:
        k["bn_premium"] = pd.NA

    metric_rows = 0
    if metrics_path.exists():
        m = pd.read_csv(
            metrics_path,
            usecols=[
                "create_time",
                "sum_open_interest_value",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ],
        )
        m["ts"] = pd.to_datetime(m["create_time"], utc=True)
        m = m.rename(
            columns={
                "sum_open_interest_value": "bn_oi_value",
                "count_long_short_ratio": "bn_ls_ratio",
                "sum_taker_long_short_vol_ratio": "bn_taker_ls_ratio",
            }
        )
        for col in ["bn_oi_value", "bn_ls_ratio", "bn_taker_ls_ratio"]:
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m = m[["ts", "bn_oi_value", "bn_ls_ratio", "bn_taker_ls_ratio"]].sort_values("ts")
        metric_rows = len(m)
        k = pd.merge_asof(
            k,
            m,
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
    else:
        k["bn_oi_value"] = pd.NA
        k["bn_ls_ratio"] = pd.NA
        k["bn_taker_ls_ratio"] = pd.NA

    return k, {"bn_kline_rows": len(k), "bn_premium_rows": premium_rows, "bn_metric_rows": metric_rows}


def load_bybit_day(symbol: str, day: str) -> tuple[pd.DataFrame, dict[str, int]] | None:
    base = DATALAKE / "bybit" / symbol
    kline_path = base / f"{day}_kline_1m.csv"
    premium_path = base / f"{day}_premium_index_kline_1m.csv"
    funding_path = base / f"{day}_funding_rate.csv"
    oi_path = base / f"{day}_open_interest_5min.csv"
    ls_path = base / f"{day}_long_short_ratio_5min.csv"
    if not kline_path.exists():
        return None

    k = pd.read_csv(kline_path, usecols=["startTime", "close", "turnover"])
    k["ts"] = pd.to_datetime(k["startTime"], unit="ms", utc=True)
    k = k.rename(columns={"close": "bb_close", "turnover": "bb_turnover"})
    for col in ["bb_close", "bb_turnover"]:
        k[col] = pd.to_numeric(k[col], errors="coerce")
    k = k[["ts", "bb_close", "bb_turnover"]].sort_values("ts")

    premium_rows = 0
    if premium_path.exists():
        premium = pd.read_csv(premium_path, usecols=["startTime", "close"])
        premium["ts"] = pd.to_datetime(premium["startTime"], unit="ms", utc=True)
        premium = premium.rename(columns={"close": "bb_premium"})
        premium["bb_premium"] = pd.to_numeric(premium["bb_premium"], errors="coerce")
        premium = premium[["ts", "bb_premium"]].sort_values("ts")
        premium_rows = len(premium)
        k = k.merge(premium, on="ts", how="left")
    else:
        k["bb_premium"] = pd.NA

    funding_rows = 0
    if funding_path.exists():
        funding = pd.read_csv(funding_path, usecols=["timestamp", "fundingRate"])
        funding["ts"] = pd.to_datetime(funding["timestamp"], unit="ms", utc=True)
        funding = funding.rename(columns={"fundingRate": "bb_funding_rate"})
        funding["bb_funding_rate"] = pd.to_numeric(funding["bb_funding_rate"], errors="coerce")
        funding = funding[["ts", "bb_funding_rate"]].sort_values("ts")
        funding_rows = len(funding)
        k = pd.merge_asof(
            k,
            funding,
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("8h"),
        )
    else:
        k["bb_funding_rate"] = pd.NA

    oi_rows = 0
    if oi_path.exists():
        oi = pd.read_csv(oi_path, usecols=["timestamp", "openInterest"])
        oi["ts"] = pd.to_datetime(oi["timestamp"], unit="ms", utc=True)
        oi = oi.rename(columns={"openInterest": "bb_open_interest"})
        oi["bb_open_interest"] = pd.to_numeric(oi["bb_open_interest"], errors="coerce")
        oi = oi[["ts", "bb_open_interest"]].sort_values("ts")
        oi_rows = len(oi)
        k = pd.merge_asof(
            k,
            oi,
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
    else:
        k["bb_open_interest"] = pd.NA

    ls_rows = 0
    if ls_path.exists():
        ls = pd.read_csv(ls_path, usecols=["timestamp", "buyRatio", "sellRatio"])
        ls["ts"] = pd.to_datetime(ls["timestamp"], unit="ms", utc=True)
        ls = ls.rename(columns={"buyRatio": "bb_buy_ratio", "sellRatio": "bb_sell_ratio"})
        for col in ["bb_buy_ratio", "bb_sell_ratio"]:
            ls[col] = pd.to_numeric(ls[col], errors="coerce")
        ls = ls[["ts", "bb_buy_ratio", "bb_sell_ratio"]].sort_values("ts")
        ls_rows = len(ls)
        k = pd.merge_asof(
            k,
            ls,
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
    else:
        k["bb_buy_ratio"] = pd.NA
        k["bb_sell_ratio"] = pd.NA

    return k, {"bb_kline_rows": len(k), "bb_premium_rows": premium_rows, "bb_funding_rows": funding_rows, "bb_oi_rows": oi_rows, "bb_ls_rows": ls_rows}


def load_symbol_day(symbol: str, day: str) -> DayFrame | None:
    bn_loaded = load_binance_day(symbol, day)
    bb_loaded = load_bybit_day(symbol, day)
    if bn_loaded is None or bb_loaded is None:
        return None
    bn, bn_counts = bn_loaded
    bb, bb_counts = bb_loaded
    frame = bn.merge(bb, on="ts", how="inner")
    if frame.empty:
        return None
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "date", day)
    frame = frame.sort_values("ts").reset_index(drop=True)
    return DayFrame(frame=frame, source_rows={**bn_counts, **bb_counts, "joined_rows": len(frame)})


def build_symbol_panel(symbol: str, start_date: str, end_date: str) -> tuple[list[pd.DataFrame], dict[str, object]]:
    frames: list[pd.DataFrame] = []
    days = common_days(symbol, start_date, end_date)
    symbol_rows = 0
    source_row_totals: dict[str, int] = {}

    for day in days:
        loaded = load_symbol_day(symbol, day)
        if loaded is None:
            continue
        frames.append(loaded.frame)
        symbol_rows += len(loaded.frame)
        for key, value in loaded.source_rows.items():
            source_row_totals[key] = source_row_totals.get(key, 0) + int(value)

    return frames, {
        "symbol": symbol,
        "day_count": len(days),
        "row_count": symbol_rows,
        "source_row_totals": source_row_totals,
    }


def build_panel(
    symbols: list[str],
    start_date: str,
    end_date: str,
    workers: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    metadata: dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
        "symbols": symbols,
        "symbol_day_counts": {},
        "symbol_row_counts": {},
        "source_row_totals": {},
    }
    source_row_totals: dict[str, int] = {}

    if workers <= 1:
        results = [build_symbol_panel(symbol, start_date, end_date) for symbol in symbols]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(build_symbol_panel, symbols, [start_date] * len(symbols), [end_date] * len(symbols)))

    for symbol_frames, symbol_meta in results:
        symbol = str(symbol_meta["symbol"])
        metadata["symbol_day_counts"][symbol] = int(symbol_meta["day_count"])
        metadata["symbol_row_counts"][symbol] = int(symbol_meta["row_count"])
        frames.extend(symbol_frames)
        for key, value in dict(symbol_meta["source_row_totals"]).items():
            source_row_totals[key] = source_row_totals.get(key, 0) + int(value)

    metadata["source_row_totals"] = source_row_totals
    if not frames:
        return pd.DataFrame(), metadata
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    metadata["total_rows"] = int(len(panel))
    metadata["symbols_with_rows"] = int((panel.groupby("symbol").size() > 0).sum())
    metadata["ts_min"] = panel["ts"].min().isoformat()
    metadata["ts_max"] = panel["ts"].max().isoformat()
    return panel, metadata


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    symbols = parse_symbols(args.symbols)
    panel, metadata = build_panel(symbols, args.start_date, args.end_date, workers=args.workers)
    if panel.empty:
        raise SystemExit("no overlapping rows found for requested symbols/date range")
    panel.to_csv(output_path, index=False, compression="gzip")
    with OUT_SUMMARY.open("w", encoding="ascii") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    print(f"wrote {output_path}")
    print(f"rows={len(panel):,} symbols={panel['symbol'].nunique()} range={metadata['ts_min']} -> {metadata['ts_max']}")


if __name__ == "__main__":
    main()
