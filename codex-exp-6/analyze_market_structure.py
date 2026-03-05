#!/usr/bin/env python3
"""Measure cross-sectional market structure and test fee-aware strategy families."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"
CACHE_VERSION = "v1"


@dataclass
class HorizonSummary:
    horizon_min: int
    universe_size: int
    rows: int
    avg_pairwise_corr_1m: float
    median_pairwise_corr_1m: float
    breadth_abs_mean: float
    breadth_q10: float
    breadth_q90: float
    strong_sync_share: float
    very_sync_share: float
    breadth_pred_corr: float
    top_breadth_forward_bps: float
    bottom_breadth_forward_bps: float


@dataclass
class StrategySummary:
    strategy: str
    horizon_min: int
    sample_count: int
    gross_bps: float
    maker_net_bps: float
    taker_net_bps: float
    maker_survives: bool
    taker_survives: bool
    win_rate: float


def collect_dates(symbol_dir: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    if not symbol_dir.exists():
        return dates
    for path in symbol_dir.glob(f"*_{suffix}"):
        if "_" not in path.name:
            continue
        dates.add(path.name.split("_", 1)[0])
    return dates


def eligible_symbols(min_overlap_days: int) -> list[str]:
    binance_symbols = {path.name for path in BINANCE.iterdir() if path.is_dir()}
    bybit_symbols = {path.name for path in BYBIT.iterdir() if path.is_dir()}
    common = sorted(binance_symbols & bybit_symbols)
    rows: list[tuple[int, str]] = []
    for symbol in common:
        overlap = len(
            collect_dates(BINANCE / symbol, "kline_1m.csv")
            & collect_dates(BYBIT / symbol, "kline_1m.csv")
        )
        if overlap >= min_overlap_days:
            rows.append((overlap, symbol))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in rows]


def parse_horizons(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    horizons = sorted({int(part) for part in parts})
    return [h for h in horizons if h > 0]


def cache_key(prefix: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    filename = f"{prefix}_{symbol}_{start_s}_{end_s}_{CACHE_VERSION}.pkl"
    return CACHE_DIR / filename


def pick_date_range(
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    if start is None and end is None:
        end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=2)
        start = end - pd.Timedelta(days=max(1, lookback_days) - 1)
    elif start is None and end is not None:
        start = end - pd.Timedelta(days=max(1, lookback_days) - 1)
    elif start is not None and end is None:
        end = start + pd.Timedelta(days=max(1, lookback_days) - 1)
    assert start is not None
    assert end is not None
    start = start.normalize()
    end = end.normalize()
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    return start, end


def load_symbol_close_series(
    exchange_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> pd.Series | None:
    symbol_dir = exchange_root / symbol
    if not symbol_dir.exists():
        return None
    cache_prefix = f"{exchange_root.name}_close"
    cache_path = cache_key(cache_prefix, symbol, start, end)
    if use_cache and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, pd.Series):
            return cached
    rows: list[pd.DataFrame] = []
    for path in sorted(symbol_dir.glob("*_kline_1m.csv")):
        day = pd.Timestamp(path.name[:10])
        if day < start or day > end:
            continue
        try:
            frame = pd.read_csv(path, usecols=["open_time", "close"])
        except ValueError:
            continue
        if frame.empty:
            continue
        rows.append(frame)
    if not rows:
        return None
    frame = pd.concat(rows, ignore_index=True)
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    idx = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(close.to_numpy(dtype=float), index=idx, name=symbol)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series.to_pickle(cache_path)
    return series


def load_symbol_metrics(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    symbol_dir = BINANCE / symbol
    if not symbol_dir.exists():
        return None
    cache_path = cache_key("binance_metrics", symbol, start, end)
    if use_cache and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, pd.DataFrame):
            return cached
    rows: list[pd.DataFrame] = []
    for path in sorted(symbol_dir.glob("*_metrics.csv")):
        day = pd.Timestamp(path.name[:10])
        if day < start or day > end:
            continue
        try:
            frame = pd.read_csv(
                path,
                usecols=[
                    "create_time",
                    "sum_open_interest",
                    "sum_taker_long_short_vol_ratio",
                    "sum_toptrader_long_short_ratio",
                    "count_long_short_ratio",
                ],
            )
        except ValueError:
            continue
        if frame.empty:
            continue
        rows.append(frame)
    if not rows:
        return None
    frame = pd.concat(rows, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    frame["sum_open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="coerce")
    frame["sum_taker_long_short_vol_ratio"] = pd.to_numeric(
        frame["sum_taker_long_short_vol_ratio"],
        errors="coerce",
    )
    frame["sum_toptrader_long_short_ratio"] = pd.to_numeric(
        frame["sum_toptrader_long_short_ratio"],
        errors="coerce",
    )
    frame["count_long_short_ratio"] = pd.to_numeric(
        frame["count_long_short_ratio"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    if frame.empty:
        return None
    frame = frame[
        [
            "timestamp",
            "sum_open_interest",
            "sum_taker_long_short_vol_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
        ]
    ]
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
    return frame


def build_price_matrix(
    exchange: str,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    coverage_ratio: float,
    fill_limit: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    exchange_root = BINANCE if exchange == "binance" else BYBIT
    series: list[pd.Series] = []
    for symbol in symbols:
        loaded = load_symbol_close_series(exchange_root, symbol, start, end, use_cache=use_cache)
        if loaded is not None:
            series.append(loaded)
    if not series:
        return pd.DataFrame()
    prices = pd.concat(series, axis=1).sort_index()
    if fill_limit > 0:
        prices = prices.ffill(limit=fill_limit)
    min_rows = max(1, int(math.ceil(prices.shape[1] * coverage_ratio)))
    prices = prices.dropna(thresh=min_rows)
    min_valid = max(1, int(math.ceil(len(prices) * coverage_ratio)))
    prices = prices.dropna(axis=1, thresh=min_valid)
    return prices


def build_metrics_matrices(
    symbols: list[str],
    index: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oi_series: list[pd.Series] = []
    taker_series: list[pd.Series] = []
    top_trader_series: list[pd.Series] = []
    account_ratio_series: list[pd.Series] = []
    for symbol in symbols:
        frame = load_symbol_metrics(symbol, start, end, use_cache=use_cache)
        if frame is None:
            continue
        metric_index = pd.DatetimeIndex(frame["timestamp"])
        oi = pd.Series(frame["sum_open_interest"].to_numpy(dtype=float), index=metric_index, name=symbol)
        taker = pd.Series(
            frame["sum_taker_long_short_vol_ratio"].to_numpy(dtype=float),
            index=metric_index,
            name=symbol,
        )
        top_trader = pd.Series(
            frame["sum_toptrader_long_short_ratio"].to_numpy(dtype=float),
            index=metric_index,
            name=symbol,
        )
        account_ratio = pd.Series(
            frame["count_long_short_ratio"].to_numpy(dtype=float),
            index=metric_index,
            name=symbol,
        )
        oi_series.append(oi)
        taker_series.append(taker)
        top_trader_series.append(top_trader)
        account_ratio_series.append(account_ratio)

    if not oi_series:
        empty = pd.DataFrame(index=index)
        return empty, empty, empty, empty

    oi_df = pd.concat(oi_series, axis=1).sort_index()
    taker_df = pd.concat(taker_series, axis=1).sort_index()
    top_trader_df = pd.concat(top_trader_series, axis=1).sort_index()
    account_ratio_df = pd.concat(account_ratio_series, axis=1).sort_index()

    union_index = index.union(oi_df.index).sort_values()
    oi_df = oi_df.reindex(union_index).ffill(limit=5).reindex(index)
    taker_df = taker_df.reindex(union_index).ffill(limit=5).reindex(index)
    top_trader_df = top_trader_df.reindex(union_index).ffill(limit=5).reindex(index)
    account_ratio_df = account_ratio_df.reindex(union_index).ffill(limit=5).reindex(index)
    return oi_df, taker_df, top_trader_df, account_ratio_df


def pairwise_corr_stats(prices: pd.DataFrame) -> tuple[float, float]:
    if prices.shape[1] < 2:
        return float("nan"), float("nan")
    ret1 = np.log(prices).diff()
    corr = ret1.corr(min_periods=max(100, len(ret1) // 20))
    matrix = corr.to_numpy(dtype=float)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[mask]
    off_diag = off_diag[np.isfinite(off_diag)]
    if len(off_diag) == 0:
        return float("nan"), float("nan")
    return float(off_diag.mean()), float(np.median(off_diag))


def safe_float(value: float) -> float:
    return float(value) if pd.notna(value) else float("nan")


def summarize_horizon(
    prices: pd.DataFrame,
    horizon: int,
    avg_corr: float,
    med_corr: float,
) -> HorizonSummary | None:
    past = prices.pct_change(horizon)
    breadth = (past > 0).sum(axis=1) / past.notna().sum(axis=1)
    breadth = breadth.replace([np.inf, -np.inf], np.nan).dropna()
    if breadth.empty:
        return None

    forward = prices.shift(-horizon).div(prices).sub(1.0)
    market_forward = forward.mean(axis=1)
    aligned = pd.concat([breadth.rename("breadth"), market_forward.rename("market_forward")], axis=1)
    aligned = aligned.dropna()
    if aligned.empty:
        return None

    top_q = aligned["breadth"].quantile(0.9)
    bottom_q = aligned["breadth"].quantile(0.1)
    top_mask = aligned["breadth"] >= top_q
    bottom_mask = aligned["breadth"] <= bottom_q

    centered = aligned["breadth"] - 0.5
    return HorizonSummary(
        horizon_min=horizon,
        universe_size=prices.shape[1],
        rows=len(aligned),
        avg_pairwise_corr_1m=avg_corr,
        median_pairwise_corr_1m=med_corr,
        breadth_abs_mean=safe_float(centered.abs().mean()),
        breadth_q10=safe_float(bottom_q),
        breadth_q90=safe_float(top_q),
        strong_sync_share=safe_float(((aligned["breadth"] >= 0.7) | (aligned["breadth"] <= 0.3)).mean()),
        very_sync_share=safe_float(((aligned["breadth"] >= 0.8) | (aligned["breadth"] <= 0.2)).mean()),
        breadth_pred_corr=safe_float(centered.corr(aligned["market_forward"])),
        top_breadth_forward_bps=safe_float(aligned.loc[top_mask, "market_forward"].mean() * 1e4),
        bottom_breadth_forward_bps=safe_float(aligned.loc[bottom_mask, "market_forward"].mean() * 1e4),
    )


def summarize_breadth_trend(
    prices: pd.DataFrame,
    horizon: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> StrategySummary | None:
    past = prices.pct_change(horizon)
    breadth = (past > 0).sum(axis=1) / past.notna().sum(axis=1)
    forward = prices.shift(-horizon).div(prices).sub(1.0)
    market_forward = forward.mean(axis=1)
    aligned = pd.concat([breadth.rename("breadth"), market_forward.rename("market_forward")], axis=1).dropna()
    if aligned.empty:
        return None
    high = aligned["breadth"].quantile(0.9)
    low = aligned["breadth"].quantile(0.1)
    selected = aligned[(aligned["breadth"] >= high) | (aligned["breadth"] <= low)].copy()
    if selected.empty:
        return None
    selected["signal"] = np.where(selected["breadth"] >= high, 1.0, -1.0)
    selected["gross_bps"] = selected["signal"] * selected["market_forward"] * 1e4
    gross = safe_float(selected["gross_bps"].mean())
    return StrategySummary(
        strategy="breadth_trend",
        horizon_min=horizon,
        sample_count=len(selected),
        gross_bps=gross,
        maker_net_bps=gross - 2.0 * maker_fee_bps,
        taker_net_bps=gross - 2.0 * taker_fee_bps,
        maker_survives=(gross - 2.0 * maker_fee_bps) > 0.0,
        taker_survives=(gross - 2.0 * taker_fee_bps) > 0.0,
        win_rate=safe_float((selected["gross_bps"] > 0.0).mean()),
    )


def cross_sectional_returns(
    prices: pd.DataFrame,
    horizon: int,
    top_quantile: float,
    bottom_quantile: float,
    long_filter: pd.DataFrame | None = None,
    short_filter: pd.DataFrame | None = None,
) -> pd.Series:
    past = prices.pct_change(horizon)
    forward = prices.shift(-horizon).div(prices).sub(1.0)
    q_high = past.quantile(top_quantile, axis=1)
    q_low = past.quantile(bottom_quantile, axis=1)
    long_mask = past.ge(q_high, axis=0)
    short_mask = past.le(q_low, axis=0)
    if long_filter is not None and not long_filter.empty:
        long_mask = long_mask & long_filter.reindex_like(long_mask).fillna(False)
    if short_filter is not None and not short_filter.empty:
        short_mask = short_mask & short_filter.reindex_like(short_mask).fillna(False)
    long_ret = forward.where(long_mask).mean(axis=1)
    short_ret = -forward.where(short_mask).mean(axis=1)
    return (long_ret + short_ret) * 1e4


def summarize_cross_sectional(
    prices: pd.DataFrame,
    horizon: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    top_quantile: float,
    bottom_quantile: float,
    strategy_name: str = "cross_sectional_momentum",
    long_filter: pd.DataFrame | None = None,
    short_filter: pd.DataFrame | None = None,
) -> StrategySummary | None:
    gross_series = cross_sectional_returns(
        prices,
        horizon,
        top_quantile,
        bottom_quantile,
        long_filter=long_filter,
        short_filter=short_filter,
    ).dropna()
    if gross_series.empty:
        return None
    gross = safe_float(gross_series.mean())
    return StrategySummary(
        strategy=strategy_name,
        horizon_min=horizon,
        sample_count=len(gross_series),
        gross_bps=gross,
        maker_net_bps=gross - 4.0 * maker_fee_bps,
        taker_net_bps=gross - 4.0 * taker_fee_bps,
        maker_survives=(gross - 4.0 * maker_fee_bps) > 0.0,
        taker_survives=(gross - 4.0 * taker_fee_bps) > 0.0,
        win_rate=safe_float((gross_series > 0.0).mean()),
    )


def summarize_gated_cross_sectional(
    prices: pd.DataFrame,
    horizon: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    top_quantile: float,
    bottom_quantile: float,
    breadth_gate: float,
    strategy_name: str = "breadth_gated_cross_sectional",
    long_filter: pd.DataFrame | None = None,
    short_filter: pd.DataFrame | None = None,
) -> StrategySummary | None:
    gross_series = cross_sectional_returns(
        prices,
        horizon,
        top_quantile,
        bottom_quantile,
        long_filter=long_filter,
        short_filter=short_filter,
    )
    past = prices.pct_change(horizon)
    breadth = (past > 0).sum(axis=1) / past.notna().sum(axis=1)
    gate = (breadth >= breadth_gate) | (breadth <= (1.0 - breadth_gate))
    gated = gross_series[gate].dropna()
    if gated.empty:
        return None
    gross = safe_float(gated.mean())
    return StrategySummary(
        strategy=strategy_name,
        horizon_min=horizon,
        sample_count=len(gated),
        gross_bps=gross,
        maker_net_bps=gross - 4.0 * maker_fee_bps,
        taker_net_bps=gross - 4.0 * taker_fee_bps,
        maker_survives=(gross - 4.0 * maker_fee_bps) > 0.0,
        taker_survives=(gross - 4.0 * taker_fee_bps) > 0.0,
        win_rate=safe_float((gated > 0.0).mean()),
    )


def compute_strategy_rows(
    prices: pd.DataFrame,
    horizons: list[int],
    maker_fee_bps: float,
    taker_fee_bps: float,
    top_quantile: float,
    bottom_quantile: float,
    breadth_gate: float,
    filtered_long_mask: pd.DataFrame | None = None,
    filtered_short_mask: pd.DataFrame | None = None,
) -> list[StrategySummary]:
    rows: list[StrategySummary] = []
    for horizon in horizons:
        breadth_summary = summarize_breadth_trend(
            prices, horizon, maker_fee_bps, taker_fee_bps
        )
        if breadth_summary is not None:
            rows.append(breadth_summary)

        cross_summary = summarize_cross_sectional(
            prices,
            horizon,
            maker_fee_bps,
            taker_fee_bps,
            top_quantile,
            bottom_quantile,
        )
        if cross_summary is not None:
            rows.append(cross_summary)

        gated_summary = summarize_gated_cross_sectional(
            prices,
            horizon,
            maker_fee_bps,
            taker_fee_bps,
            top_quantile,
            bottom_quantile,
            breadth_gate,
        )
        if gated_summary is not None:
            rows.append(gated_summary)

        if filtered_long_mask is not None and filtered_short_mask is not None:
            filtered_cross_summary = summarize_cross_sectional(
                prices,
                horizon,
                maker_fee_bps,
                taker_fee_bps,
                top_quantile,
                bottom_quantile,
                strategy_name="filtered_cross_sectional_momentum",
                long_filter=filtered_long_mask,
                short_filter=filtered_short_mask,
            )
            if filtered_cross_summary is not None:
                rows.append(filtered_cross_summary)

            filtered_gated_summary = summarize_gated_cross_sectional(
                prices,
                horizon,
                maker_fee_bps,
                taker_fee_bps,
                top_quantile,
                bottom_quantile,
                breadth_gate,
                strategy_name="breadth_gated_filtered_cross_sectional",
                long_filter=filtered_long_mask,
                short_filter=filtered_short_mask,
            )
            if filtered_gated_summary is not None:
                rows.append(filtered_gated_summary)
    return rows


def build_walkforward_rows(
    prices: pd.DataFrame,
    horizons: list[int],
    maker_fee_bps: float,
    taker_fee_bps: float,
    top_quantile: float,
    bottom_quantile: float,
    breadth_gate: float,
    splits: int,
    filtered_long_mask: pd.DataFrame | None = None,
    filtered_short_mask: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    if splits <= 1:
        return []

    rows: list[dict[str, object]] = []
    index_chunks = [chunk for chunk in np.array_split(prices.index.to_numpy(), splits) if len(chunk)]
    max_horizon = max(horizons) if horizons else 0
    for split_id, raw_chunk in enumerate(index_chunks, start=1):
        chunk_index = pd.DatetimeIndex(raw_chunk)
        split_prices = prices.loc[chunk_index]
        if len(split_prices) <= max_horizon + 1:
            continue
        split_long = filtered_long_mask.loc[chunk_index] if filtered_long_mask is not None else None
        split_short = filtered_short_mask.loc[chunk_index] if filtered_short_mask is not None else None
        split_rows = compute_strategy_rows(
            split_prices,
            horizons,
            maker_fee_bps,
            taker_fee_bps,
            top_quantile,
            bottom_quantile,
            breadth_gate,
            filtered_long_mask=split_long,
            filtered_short_mask=split_short,
        )
        for row in split_rows:
            rows.append(
                {
                    "split_id": split_id,
                    "split_start": split_prices.index.min().isoformat(),
                    "split_end": split_prices.index.max().isoformat(),
                    **row.__dict__,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    exchange: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: list[str],
    horizon_rows: list[HorizonSummary],
    strategy_rows: list[StrategySummary],
    maker_fee_bps: float,
    taker_fee_bps: float,
    walkforward_rows: list[dict[str, object]] | None = None,
) -> None:
    best_maker = sorted(
        strategy_rows,
        key=lambda row: (row.maker_net_bps, row.gross_bps),
        reverse=True,
    )[:5]
    best_taker = sorted(
        strategy_rows,
        key=lambda row: (row.taker_net_bps, row.gross_bps),
        reverse=True,
    )[:5]

    lines = [
        "# Market Structure Report",
        "",
        f"- Exchange: `{exchange}`",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{len(symbols)}`",
        f"- Maker fee assumption: `{maker_fee_bps:.2f}` bps per side",
        f"- Taker fee assumption: `{taker_fee_bps:.2f}` bps per side",
        "",
        "## Horizon Summary",
        "",
    ]
    for row in horizon_rows:
        lines.append(
            f"- {row.horizon_min}m: avg corr `{row.avg_pairwise_corr_1m:.4f}`, "
            f"strong sync `{row.strong_sync_share:.3f}`, very sync `{row.very_sync_share:.3f}`, "
            f"top breadth fwd `{row.top_breadth_forward_bps:.2f}` bps, bottom breadth fwd "
            f"`{row.bottom_breadth_forward_bps:.2f}` bps"
        )

    lines.extend(["", "## Best Maker Candidates", ""])
    for row in best_maker:
        lines.append(
            f"- {row.strategy} {row.horizon_min}m: gross `{row.gross_bps:.2f}` bps, "
            f"maker net `{row.maker_net_bps:.2f}` bps, samples `{row.sample_count}`"
        )

    lines.extend(["", "## Best Taker Candidates", ""])
    for row in best_taker:
        lines.append(
            f"- {row.strategy} {row.horizon_min}m: gross `{row.gross_bps:.2f}` bps, "
            f"taker net `{row.taker_net_bps:.2f}` bps, samples `{row.sample_count}`"
        )

    if walkforward_rows:
        walk_df = pd.DataFrame(walkforward_rows)
        if not walk_df.empty:
            grouped = (
                walk_df.groupby(["strategy", "horizon_min"], as_index=False)
                .agg(
                    splits=("split_id", "nunique"),
                    avg_maker_net_bps=("maker_net_bps", "mean"),
                    min_maker_net_bps=("maker_net_bps", "min"),
                    avg_taker_net_bps=("taker_net_bps", "mean"),
                    min_taker_net_bps=("taker_net_bps", "min"),
                )
                .sort_values(["avg_maker_net_bps", "min_maker_net_bps"], ascending=False)
            )
            lines.extend(["", "## Walk-Forward Stability", ""])
            for _, row in grouped.head(5).iterrows():
                lines.append(
                    f"- {row['strategy']} {int(row['horizon_min'])}m: splits `{int(row['splits'])}`, "
                    f"avg maker `{row['avg_maker_net_bps']:.2f}` bps, worst maker "
                    f"`{row['min_maker_net_bps']:.2f}` bps, avg taker "
                    f"`{row['avg_taker_net_bps']:.2f}` bps"
                )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", choices=["binance", "bybit"], default="binance")
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-symbols", type=int, default=60)
    parser.add_argument("--min-symbols", type=int, default=20)
    parser.add_argument("--horizons", type=str, default="1,5,15,60,240")
    parser.add_argument("--coverage-ratio", type=float, default=0.8)
    parser.add_argument("--fill-limit", type=int, default=2)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--taker-fee-bps", type=float, default=10.0)
    parser.add_argument("--top-quantile", type=float, default=0.8)
    parser.add_argument("--bottom-quantile", type=float, default=0.2)
    parser.add_argument("--breadth-gate", type=float, default=0.7)
    parser.add_argument("--use-metrics-filters", action="store_true")
    parser.add_argument("--metrics-lookback-bars", type=int, default=3)
    parser.add_argument("--min-oi-change", type=float, default=0.0)
    parser.add_argument("--taker-ratio-threshold", type=float, default=1.05)
    parser.add_argument("--top-trader-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--account-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--walkforward-splits", type=int, default=0)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.bottom_quantile < args.top_quantile < 1.0:
        raise SystemExit("Expected 0 < bottom-quantile < top-quantile < 1")
    if not 0.5 <= args.breadth_gate < 1.0:
        raise SystemExit("Expected breadth-gate in [0.5, 1)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end = pick_date_range(args.start_date, args.end_date, args.lookback_days)
    symbols = eligible_symbols(args.min_overlap_days)[: max(1, args.max_symbols)]
    if not symbols:
        raise SystemExit("No eligible symbols found. Run universe_scan.py or lower --min-overlap-days.")

    prices = build_price_matrix(
        exchange=args.exchange,
        symbols=symbols,
        start=start,
        end=end,
        coverage_ratio=args.coverage_ratio,
        fill_limit=args.fill_limit,
        use_cache=not args.no_cache,
    )
    if prices.empty or prices.shape[1] < args.min_symbols:
        raise SystemExit(
            f"Insufficient aligned data. Loaded {prices.shape[1]} symbols, need at least {args.min_symbols}."
        )

    active_symbols = list(prices.columns)
    avg_corr, med_corr = pairwise_corr_stats(prices)
    horizons = parse_horizons(args.horizons)

    oi_df = pd.DataFrame(index=prices.index)
    taker_df = pd.DataFrame(index=prices.index)
    top_trader_df = pd.DataFrame(index=prices.index)
    account_ratio_df = pd.DataFrame(index=prices.index)
    if args.use_metrics_filters:
        if args.exchange != "binance":
            raise SystemExit("Metrics filters are currently supported only for --exchange binance.")
        oi_df, taker_df, top_trader_df, account_ratio_df = build_metrics_matrices(
            active_symbols,
            prices.index,
            start,
            end,
            use_cache=not args.no_cache,
        )

    filtered_long_mask = None
    filtered_short_mask = None
    if args.use_metrics_filters and not oi_df.empty and not taker_df.empty:
        oi_change = oi_df.div(oi_df.shift(max(1, args.metrics_lookback_bars))).sub(1.0)
        long_taker_ok = taker_df >= args.taker_ratio_threshold
        short_taker_ok = taker_df <= (1.0 / args.taker_ratio_threshold)
        oi_ok = oi_change >= args.min_oi_change
        long_top_ok = (
            top_trader_df >= args.top_trader_ratio_threshold
            if not top_trader_df.empty
            else pd.DataFrame(True, index=prices.index, columns=prices.columns)
        )
        short_top_ok = (
            top_trader_df <= (1.0 / args.top_trader_ratio_threshold)
            if not top_trader_df.empty
            else pd.DataFrame(True, index=prices.index, columns=prices.columns)
        )
        long_account_ok = (
            account_ratio_df >= args.account_ratio_threshold
            if not account_ratio_df.empty
            else pd.DataFrame(True, index=prices.index, columns=prices.columns)
        )
        short_account_ok = (
            account_ratio_df <= (1.0 / args.account_ratio_threshold)
            if not account_ratio_df.empty
            else pd.DataFrame(True, index=prices.index, columns=prices.columns)
        )
        filtered_long_mask = oi_ok & long_taker_ok & long_top_ok & long_account_ok
        filtered_short_mask = oi_ok & short_taker_ok & short_top_ok & short_account_ok

    horizon_rows: list[HorizonSummary] = []
    for horizon in horizons:
        horizon_summary = summarize_horizon(prices, horizon, avg_corr, med_corr)
        if horizon_summary is not None:
            horizon_rows.append(horizon_summary)

    strategy_rows = compute_strategy_rows(
        prices,
        horizons,
        args.maker_fee_bps,
        args.taker_fee_bps,
        args.top_quantile,
        args.bottom_quantile,
        args.breadth_gate,
        filtered_long_mask=filtered_long_mask,
        filtered_short_mask=filtered_short_mask,
    )

    walkforward_rows = build_walkforward_rows(
        prices,
        horizons,
        args.maker_fee_bps,
        args.taker_fee_bps,
        args.top_quantile,
        args.bottom_quantile,
        args.breadth_gate,
        args.walkforward_splits,
        filtered_long_mask=filtered_long_mask,
        filtered_short_mask=filtered_short_mask,
    )

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    horizon_path = OUT_DIR / f"market_structure_summary{suffix}.csv"
    strategy_path = OUT_DIR / f"strategy_summary{suffix}.csv"
    report_path = OUT_DIR / f"market_structure_report{suffix}.md"
    walkforward_path = OUT_DIR / f"walkforward_summary{suffix}.csv"

    write_csv(
        horizon_path,
        [row.__dict__ for row in horizon_rows],
        [
            "horizon_min",
            "universe_size",
            "rows",
            "avg_pairwise_corr_1m",
            "median_pairwise_corr_1m",
            "breadth_abs_mean",
            "breadth_q10",
            "breadth_q90",
            "strong_sync_share",
            "very_sync_share",
            "breadth_pred_corr",
            "top_breadth_forward_bps",
            "bottom_breadth_forward_bps",
        ],
    )
    write_csv(
        strategy_path,
        [row.__dict__ for row in strategy_rows],
        [
            "strategy",
            "horizon_min",
            "sample_count",
            "gross_bps",
            "maker_net_bps",
            "taker_net_bps",
            "maker_survives",
            "taker_survives",
            "win_rate",
        ],
    )
    write_report(
        report_path,
        exchange=args.exchange,
        start=start,
        end=end,
        symbols=active_symbols,
        horizon_rows=horizon_rows,
        strategy_rows=strategy_rows,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        walkforward_rows=walkforward_rows,
    )
    if walkforward_rows:
        write_csv(
            walkforward_path,
            walkforward_rows,
            [
                "split_id",
                "split_start",
                "split_end",
                "strategy",
                "horizon_min",
                "sample_count",
                "gross_bps",
                "maker_net_bps",
                "taker_net_bps",
                "maker_survives",
                "taker_survives",
                "win_rate",
            ],
        )

    print(f"Loaded {len(active_symbols)} symbols on {args.exchange}")
    print(f"Date range: {start.date()} to {end.date()}")
    print(f"Cache: {'disabled' if args.no_cache else CACHE_DIR}")
    print(f"Wrote {horizon_path}")
    print(f"Wrote {strategy_path}")
    print(f"Wrote {report_path}")
    if walkforward_rows:
        print(f"Wrote {walkforward_path}")

    if strategy_rows:
        best = max(strategy_rows, key=lambda row: row.taker_net_bps)
        print(
            f"Best taker candidate: {best.strategy} {best.horizon_min}m "
            f"gross={best.gross_bps:.2f}bps taker_net={best.taker_net_bps:.2f}bps"
        )


if __name__ == "__main__":
    main()
