#!/usr/bin/env python3
"""Identify trade/no-trade regime conditions for the 240m breadth trend setup."""

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
EXP6_PATH = Path(__file__).resolve().parent / "analyze_market_structure.py"
EXEC_PATH = Path(__file__).resolve().parent / "breadth_trend_execution.py"
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Candidate:
    rule: str
    sample_count: int
    trade_share: float
    gross_bps: float
    expected_net_bps: float
    worst_split_net_bps: float
    split_pos_rate: float
    split_1_net_bps: float
    split_2_net_bps: float
    split_3_net_bps: float


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_features(prices: pd.DataFrame, selected: pd.DataFrame, horizon: int) -> pd.DataFrame:
    market_ret_1m = np.log(prices).diff().mean(axis=1) * 1e4
    market_ret_60 = prices.pct_change(60).mean(axis=1) * 1e4
    market_ret_240 = prices.pct_change(240).mean(axis=1) * 1e4
    rv_60 = market_ret_1m.rolling(60).std()
    rv_240 = market_ret_1m.rolling(240).std()
    disp_240 = prices.pct_change(240).std(axis=1) * 1e4
    abs_breadth = (selected["breadth"] - 0.5).abs()

    features = pd.DataFrame(index=selected.index)
    features["abs_breadth"] = abs_breadth
    features["rv_60"] = rv_60.reindex(features.index)
    features["rv_240"] = rv_240.reindex(features.index)
    features["disp_240"] = disp_240.reindex(features.index)
    features["signal_align_60"] = selected["signal"] * market_ret_60.reindex(features.index)
    features["signal_align_240"] = selected["signal"] * market_ret_240.reindex(features.index)
    features["signal_align_h"] = selected["signal"] * (
        prices.pct_change(horizon).mean(axis=1) * 1e4
    ).reindex(features.index)
    return features.replace([np.inf, -np.inf], np.nan)


def assign_splits(index: pd.DatetimeIndex, splits: int) -> pd.Series:
    split_series = pd.Series(index=index, dtype="Int64")
    chunks = [chunk for chunk in np.array_split(index.to_numpy(), splits) if len(chunk)]
    for split_id, raw_chunk in enumerate(chunks, start=1):
        chunk_index = pd.DatetimeIndex(raw_chunk)
        split_series.loc[chunk_index] = split_id
    return split_series


def evaluate_rule(
    rule: str,
    mask: pd.Series,
    selected: pd.DataFrame,
    split_id: pd.Series,
    min_samples: int,
    total_count: int,
    splits: int,
) -> Candidate | None:
    mask = mask.fillna(False)
    filtered = selected[mask]
    if len(filtered) < min_samples:
        return None
    row = filtered.copy()
    row["split_id"] = split_id.reindex(row.index).fillna(-1).astype(int)

    split_nets: list[float] = []
    pos = 0
    for i in range(1, splits + 1):
        s = row[row["split_id"] == i]
        if s.empty:
            split_nets.append(float("nan"))
            continue
        val = float(s["expected_net_bps"].mean())
        split_nets.append(val)
        if val > 0:
            pos += 1
    while len(split_nets) < 3:
        split_nets.append(float("nan"))

    finite = [x for x in split_nets if pd.notna(x)]
    worst = min(finite) if finite else float("nan")
    split_pos_rate = pos / max(1, sum(pd.notna(split_nets)))

    return Candidate(
        rule=rule,
        sample_count=len(row),
        trade_share=len(row) / max(1, total_count),
        gross_bps=float(row["gross_bps"].mean()),
        expected_net_bps=float(row["expected_net_bps"].mean()),
        worst_split_net_bps=float(worst),
        split_pos_rate=float(split_pos_rate),
        split_1_net_bps=float(split_nets[0]),
        split_2_net_bps=float(split_nets[1]),
        split_3_net_bps=float(split_nets[2]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--max-symbols", type=int, default=100)
    parser.add_argument("--min-symbols", type=int, default=60)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--queue-miss-rate", type=float, default=0.35)
    parser.add_argument("--partial-fill-rate", type=float, default=0.75)
    parser.add_argument("--adverse-selection-bps", type=float, default=1.0)
    parser.add_argument("--walkforward-splits", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    exp6 = load_module(EXP6_PATH, "exp6_regime_scan")
    exec_mod = load_module(EXEC_PATH, "exec_regime_scan")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end = exp6.pick_date_range(None, None, args.lookback_days)
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

    summary, selected = exec_mod.compute_execution_summary(
        prices,
        args.horizon,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
    )
    if summary is None or selected.empty:
        raise SystemExit("No candidate signals found.")

    features = build_features(prices, selected, args.horizon)
    joined = selected.join(features, how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise SystemExit("No valid feature rows after joining signals and features.")

    split_id = assign_splits(prices.index, max(2, args.walkforward_splits))
    total_count = len(joined)

    q_abs_60 = joined["abs_breadth"].quantile(0.6)
    q_abs_75 = joined["abs_breadth"].quantile(0.75)
    q_rv60_50 = joined["rv_60"].quantile(0.5)
    q_rv60_65 = joined["rv_60"].quantile(0.65)
    q_rv240_50 = joined["rv_240"].quantile(0.5)
    q_disp_50 = joined["disp_240"].quantile(0.5)
    q_disp_65 = joined["disp_240"].quantile(0.65)

    rule_masks: list[tuple[str, pd.Series]] = [
        ("all_signals", pd.Series(True, index=joined.index)),
        (f"abs_breadth >= q60 ({q_abs_60:.4f})", joined["abs_breadth"] >= q_abs_60),
        (f"abs_breadth >= q75 ({q_abs_75:.4f})", joined["abs_breadth"] >= q_abs_75),
        (f"rv_60 <= q50 ({q_rv60_50:.4f})", joined["rv_60"] <= q_rv60_50),
        (f"rv_60 <= q65 ({q_rv60_65:.4f})", joined["rv_60"] <= q_rv60_65),
        (f"rv_240 <= q50 ({q_rv240_50:.4f})", joined["rv_240"] <= q_rv240_50),
        (f"disp_240 <= q50 ({q_disp_50:.4f})", joined["disp_240"] <= q_disp_50),
        (f"disp_240 <= q65 ({q_disp_65:.4f})", joined["disp_240"] <= q_disp_65),
        ("signal_align_60 > 0", joined["signal_align_60"] > 0),
        ("signal_align_240 > 0", joined["signal_align_240"] > 0),
        ("signal_align_h > 0", joined["signal_align_h"] > 0),
    ]

    # Pairwise combinations focused on practical filters.
    rule_masks.extend(
        [
            (
                "signal_align_240 > 0 AND rv_60 <= q50",
                (joined["signal_align_240"] > 0) & (joined["rv_60"] <= q_rv60_50),
            ),
            (
                "signal_align_240 > 0 AND disp_240 <= q50",
                (joined["signal_align_240"] > 0) & (joined["disp_240"] <= q_disp_50),
            ),
            (
                "signal_align_240 > 0 AND abs_breadth >= q60",
                (joined["signal_align_240"] > 0) & (joined["abs_breadth"] >= q_abs_60),
            ),
            (
                "signal_align_60 > 0 AND rv_60 <= q50",
                (joined["signal_align_60"] > 0) & (joined["rv_60"] <= q_rv60_50),
            ),
            (
                "signal_align_60 > 0 AND disp_240 <= q50",
                (joined["signal_align_60"] > 0) & (joined["disp_240"] <= q_disp_50),
            ),
            (
                "abs_breadth >= q60 AND rv_60 <= q50",
                (joined["abs_breadth"] >= q_abs_60) & (joined["rv_60"] <= q_rv60_50),
            ),
            (
                "abs_breadth >= q60 AND disp_240 <= q50",
                (joined["abs_breadth"] >= q_abs_60) & (joined["disp_240"] <= q_disp_50),
            ),
            (
                "signal_align_240 > 0 AND rv_60 <= q50 AND disp_240 <= q50",
                (joined["signal_align_240"] > 0)
                & (joined["rv_60"] <= q_rv60_50)
                & (joined["disp_240"] <= q_disp_50),
            ),
        ]
    )

    candidates: list[Candidate] = []
    for rule, mask in rule_masks:
        item = evaluate_rule(
            rule=rule,
            mask=mask,
            selected=joined,
            split_id=split_id,
            min_samples=args.min_samples,
            total_count=total_count,
            splits=max(2, args.walkforward_splits),
        )
        if item is not None:
            candidates.append(item)

    candidates.sort(
        key=lambda row: (
            row.worst_split_net_bps,
            row.expected_net_bps,
            row.sample_count,
        ),
        reverse=True,
    )

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    summary_path = OUT_DIR / f"regime_filter_summary{suffix}.csv"
    report_path = OUT_DIR / f"regime_filter_report{suffix}.md"

    write_csv(
        summary_path,
        [row.__dict__ for row in candidates],
        [
            "rule",
            "sample_count",
            "trade_share",
            "gross_bps",
            "expected_net_bps",
            "worst_split_net_bps",
            "split_pos_rate",
            "split_1_net_bps",
            "split_2_net_bps",
            "split_3_net_bps",
        ],
    )

    lines = [
        "# Regime Filter Report",
        "",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{prices.shape[1]}`",
        f"- Horizon: `{args.horizon}` minutes",
        f"- Base expected net (all signals): `{joined['expected_net_bps'].mean():.2f}` bps",
        f"- Minimum samples per rule: `{args.min_samples}`",
        "",
        "## Top Rules",
        "",
    ]
    for row in candidates[:10]:
        lines.append(
            f"- {row.rule}: expected `{row.expected_net_bps:.2f}` bps, "
            f"worst split `{row.worst_split_net_bps:.2f}` bps, "
            f"trade share `{row.trade_share:.2%}`, samples `{row.sample_count}`"
        )

    lines.extend(
        [
            "",
            "## Practical Decision",
            "",
            "Treat a rule as `trade` only if it keeps meaningful sample size and improves "
            "worst-split net expectancy versus `all_signals`.",
            "Everything else is `no-trade` by default.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Universe symbols: {prices.shape[1]}")
    print(f"Base signals: {len(joined)}")
    print(f"Rules tested: {len(candidates)}")
    if candidates:
        top = candidates[0]
        print(
            f"Top rule: {top.rule} expected_net={top.expected_net_bps:.2f}bps "
            f"worst_split={top.worst_split_net_bps:.2f}bps trade_share={top.trade_share:.2%}"
        )
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
