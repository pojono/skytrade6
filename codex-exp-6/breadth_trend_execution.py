#!/usr/bin/env python3
"""Fast execution-realism study for the 240m breadth trend candidate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).resolve().parent / "analyze_market_structure.py"
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_exp6_module():
    spec = importlib.util.spec_from_file_location("codex_exp_6_analyzer", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ExecutionSummary:
    sample_count: int
    gross_bps: float
    expected_net_bps: float
    expected_fill_rate: float
    gross_hit_rate: float
    expected_net_positive_rate: float


def compute_execution_summary(
    prices: pd.DataFrame,
    horizon: int,
    maker_fee_bps: float,
    queue_miss_rate: float,
    partial_fill_rate: float,
    adverse_selection_bps: float,
) -> tuple[ExecutionSummary | None, pd.DataFrame]:
    past = prices.pct_change(horizon)
    breadth = (past > 0).sum(axis=1) / past.notna().sum(axis=1)
    forward = prices.shift(-horizon).div(prices).sub(1.0)
    market_forward = forward.mean(axis=1)
    aligned = pd.concat([breadth.rename("breadth"), market_forward.rename("market_forward")], axis=1).dropna()
    if aligned.empty:
        return None, aligned

    high = aligned["breadth"].quantile(0.9)
    low = aligned["breadth"].quantile(0.1)
    selected = aligned[(aligned["breadth"] >= high) | (aligned["breadth"] <= low)].copy()
    if selected.empty:
        return None, selected

    selected["signal"] = np.where(selected["breadth"] >= high, 1.0, -1.0)
    selected["gross_bps"] = selected["signal"] * selected["market_forward"] * 1e4

    fill_rate = max(0.0, min(1.0, (1.0 - queue_miss_rate) * partial_fill_rate))
    per_fill_cost_bps = 2.0 * maker_fee_bps + adverse_selection_bps
    selected["expected_net_bps"] = fill_rate * (selected["gross_bps"] - per_fill_cost_bps)

    summary = ExecutionSummary(
        sample_count=len(selected),
        gross_bps=float(selected["gross_bps"].mean()),
        expected_net_bps=float(selected["expected_net_bps"].mean()),
        expected_fill_rate=fill_rate,
        gross_hit_rate=float((selected["gross_bps"] > 0.0).mean()),
        expected_net_positive_rate=float((selected["expected_net_bps"] > 0.0).mean()),
    )
    return summary, selected


def build_walkforward(
    prices: pd.DataFrame,
    horizon: int,
    maker_fee_bps: float,
    queue_miss_rate: float,
    partial_fill_rate: float,
    adverse_selection_bps: float,
    splits: int,
) -> list[dict[str, object]]:
    if splits <= 1:
        return []
    rows: list[dict[str, object]] = []
    chunks = [chunk for chunk in np.array_split(prices.index.to_numpy(), splits) if len(chunk)]
    for split_id, raw_chunk in enumerate(chunks, start=1):
        chunk_index = pd.DatetimeIndex(raw_chunk)
        split_prices = prices.loc[chunk_index]
        if len(split_prices) <= horizon + 1:
            continue
        summary, _ = compute_execution_summary(
            split_prices,
            horizon,
            maker_fee_bps,
            queue_miss_rate,
            partial_fill_rate,
            adverse_selection_bps,
        )
        if summary is None:
            continue
        rows.append(
            {
                "split_id": split_id,
                "split_start": split_prices.index.min().isoformat(),
                "split_end": split_prices.index.max().isoformat(),
                "sample_count": summary.sample_count,
                "gross_bps": summary.gross_bps,
                "expected_net_bps": summary.expected_net_bps,
                "expected_fill_rate": summary.expected_fill_rate,
                "gross_hit_rate": summary.gross_hit_rate,
                "expected_net_positive_rate": summary.expected_net_positive_rate,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--min-symbols", type=int, default=20)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--queue-miss-rate", type=float, default=0.35)
    parser.add_argument("--partial-fill-rate", type=float, default=0.75)
    parser.add_argument("--adverse-selection-bps", type=float, default=1.0)
    parser.add_argument("--walkforward-splits", type=int, default=3)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    exp6 = load_exp6_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end = exp6.pick_date_range(args.start_date, args.end_date, args.lookback_days)
    symbols = exp6.eligible_symbols(args.min_overlap_days)[: max(1, args.max_symbols)]
    prices = exp6.build_price_matrix(
        exchange="binance",
        symbols=symbols,
        start=start,
        end=end,
        coverage_ratio=0.8,
        fill_limit=2,
        use_cache=not args.no_cache,
    )
    if prices.empty or prices.shape[1] < args.min_symbols:
        raise SystemExit(
            f"Insufficient aligned data. Loaded {prices.shape[1]} symbols, need at least {args.min_symbols}."
        )

    summary, selected = compute_execution_summary(
        prices,
        args.horizon,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
    )
    if summary is None:
        raise SystemExit("No signals produced for the selected configuration.")

    walkforward_rows = build_walkforward(
        prices,
        args.horizon,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
        args.walkforward_splits,
    )

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    summary_path = OUT_DIR / f"breadth_trend_execution_summary{suffix}.csv"
    walkforward_path = OUT_DIR / f"breadth_trend_execution_walkforward{suffix}.csv"
    report_path = OUT_DIR / f"breadth_trend_execution_report{suffix}.md"

    write_csv(
        summary_path,
        [
            {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "universe_size": prices.shape[1],
                "horizon_min": args.horizon,
                "sample_count": summary.sample_count,
                "gross_bps": summary.gross_bps,
                "expected_net_bps": summary.expected_net_bps,
                "expected_fill_rate": summary.expected_fill_rate,
                "gross_hit_rate": summary.gross_hit_rate,
                "expected_net_positive_rate": summary.expected_net_positive_rate,
                "queue_miss_rate": args.queue_miss_rate,
                "partial_fill_rate": args.partial_fill_rate,
                "adverse_selection_bps": args.adverse_selection_bps,
            }
        ],
        [
            "start_date",
            "end_date",
            "universe_size",
            "horizon_min",
            "sample_count",
            "gross_bps",
            "expected_net_bps",
            "expected_fill_rate",
            "gross_hit_rate",
            "expected_net_positive_rate",
            "queue_miss_rate",
            "partial_fill_rate",
            "adverse_selection_bps",
        ],
    )
    if walkforward_rows:
        write_csv(
            walkforward_path,
            walkforward_rows,
            [
                "split_id",
                "split_start",
                "split_end",
                "sample_count",
                "gross_bps",
                "expected_net_bps",
                "expected_fill_rate",
                "gross_hit_rate",
                "expected_net_positive_rate",
            ],
        )

    lines = [
        "# Breadth Trend Execution Report",
        "",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{prices.shape[1]}`",
        f"- Horizon: `{args.horizon}` minutes",
        f"- Queue miss rate: `{args.queue_miss_rate:.2f}`",
        f"- Partial fill rate: `{args.partial_fill_rate:.2f}`",
        f"- Adverse selection: `{args.adverse_selection_bps:.2f}` bps",
        "",
        "## Aggregate",
        "",
        f"- Samples: `{summary.sample_count}`",
        f"- Gross edge: `{summary.gross_bps:.2f}` bps",
        f"- Expected net edge: `{summary.expected_net_bps:.2f}` bps",
        f"- Expected fill rate: `{summary.expected_fill_rate:.3f}`",
        f"- Gross hit rate: `{summary.gross_hit_rate:.3f}`",
        f"- Expected net positive rate: `{summary.expected_net_positive_rate:.3f}`",
    ]
    if walkforward_rows:
        walk_df = pd.DataFrame(walkforward_rows)
        lines.extend(["", "## Walk-Forward", ""])
        lines.append(
            f"- Avg expected net: `{walk_df['expected_net_bps'].mean():.2f}` bps, "
            f"worst split: `{walk_df['expected_net_bps'].min():.2f}` bps"
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Loaded {prices.shape[1]} symbols")
    print(f"Signals: {summary.sample_count}")
    print(f"Gross edge: {summary.gross_bps:.2f}bps")
    print(f"Expected net edge: {summary.expected_net_bps:.2f}bps")
    print(f"Wrote {summary_path}")
    if walkforward_rows:
        print(f"Wrote {walkforward_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
