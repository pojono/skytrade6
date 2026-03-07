#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "universe_screen_micro_events.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily rolling universe selection with no lookahead.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-days", type=int, default=5)
    parser.add_argument("--max-active-symbols", type=int, default=8)
    parser.add_argument("--entry-rank", type=int, default=8)
    parser.add_argument("--keep-rank", type=int, default=12)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-mean-edge-bps", type=float, default=2.0)
    parser.add_argument("--min-positive-day-share", type=float, default=0.6)
    parser.add_argument("--output-prefix", default="rolling_universe")
    return parser.parse_args()


def symbol_metrics(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, part in train.groupby("symbol", sort=True):
        daily = part.groupby("date").agg(
            daily_events=("pair_net_15m_bps", "size"),
            daily_mean_edge=("pair_net_15m_bps", "mean"),
        )
        rows.append(
            {
                "symbol": symbol,
                "train_events": float(len(part)),
                "mean_edge_bps": float(part["pair_net_15m_bps"].mean()),
                "median_edge_bps": float(part["pair_net_15m_bps"].median()),
                "win_rate": float(part["pair_win_15m"].mean()),
                "mean_gap_bps": float(part["gap_bps"].mean()),
                "positive_day_share": float((daily["daily_mean_edge"] > 0).mean()),
                "daily_edge_std": float(daily["daily_mean_edge"].std(ddof=0)) if len(daily) > 1 else 0.0,
                "active_days": float(len(daily)),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    # Reward stable positive edge with event depth and penalize unstable daily swings.
    metrics["selection_score"] = (
        metrics["mean_edge_bps"].clip(lower=0.0)
        * np.log1p(metrics["train_events"])
        * metrics["positive_day_share"].clip(lower=0.0)
        / (1.0 + metrics["daily_edge_std"].clip(lower=0.0))
    )
    metrics = metrics.sort_values(
        ["selection_score", "mean_edge_bps", "train_events"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics["rank"] = np.arange(1, len(metrics) + 1)
    return metrics


def passes_entry(row: pd.Series, args: argparse.Namespace) -> bool:
    return (
        row["train_events"] >= args.min_train_events
        and row["mean_edge_bps"] >= args.min_mean_edge_bps
        and row["positive_day_share"] >= args.min_positive_day_share
        and row["rank"] <= args.entry_rank
    )


def passes_keep(row: pd.Series, args: argparse.Namespace) -> bool:
    return (
        row["train_events"] >= args.min_train_events
        and row["mean_edge_bps"] >= 0.5 * args.min_mean_edge_bps
        and row["positive_day_share"] >= max(0.5, args.min_positive_day_share - 0.1)
        and row["rank"] <= args.keep_rank
    )


def summarize_day(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "events": 0.0,
            "pair_net_15m_bps": np.nan,
            "pair_net_30m_bps": np.nan,
            "pair_win_15m": np.nan,
            "mean_gap_bps": np.nan,
        }
    return {
        "events": float(len(frame)),
        "pair_net_15m_bps": float(frame["pair_net_15m_bps"].mean()),
        "pair_net_30m_bps": float(frame["pair_net_30m_bps"].mean()),
        "pair_win_15m": float(frame["pair_win_15m"].mean()),
        "mean_gap_bps": float(frame["gap_bps"].mean()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.input, parse_dates=["ts"]).replace([np.inf, -np.inf], np.nan)
    if events.empty:
        raise SystemExit("input events file is empty")
    days = sorted(events["date"].unique().tolist())
    if len(days) <= args.train_days:
        raise SystemExit("not enough distinct days for requested rolling window")

    previous_active: list[str] = []
    active_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []

    for idx in range(args.train_days, len(days)):
        test_day = days[idx]
        train_days = days[idx - args.train_days : idx]
        train = events[events["date"].isin(train_days)].copy()
        test = events[events["date"] == test_day].copy()
        metrics = symbol_metrics(train)
        if metrics.empty:
            continue

        rank_lookup = {row.symbol: row for row in metrics.itertuples(index=False)}
        active: list[str] = []

        for symbol in previous_active:
            row = rank_lookup.get(symbol)
            if row is None:
                continue
            row_series = pd.Series(row._asdict())
            if passes_keep(row_series, args):
                active.append(symbol)

        for row in metrics.itertuples(index=False):
            if len(active) >= args.max_active_symbols:
                break
            if row.symbol in active:
                continue
            row_series = pd.Series(row._asdict())
            if passes_entry(row_series, args):
                active.append(row.symbol)

        for symbol in active:
            row = rank_lookup[symbol]
            active_rows.append(
                {
                    "test_day": test_day,
                    "train_start_day": train_days[0],
                    "train_end_day": train_days[-1],
                    "symbol": symbol,
                    "rank": int(row.rank),
                    "train_events": float(row.train_events),
                    "mean_edge_bps": float(row.mean_edge_bps),
                    "positive_day_share": float(row.positive_day_share),
                    "selection_score": float(row.selection_score),
                    "was_active_prev_day": symbol in previous_active,
                }
            )

        baseline = summarize_day(test)
        active_test = summarize_day(test[test["symbol"].isin(active)])
        perf_rows.append(
            {
                "test_day": test_day,
                "active_symbols": ",".join(active),
                "active_symbol_count": len(active),
                "baseline_events": baseline["events"],
                "baseline_pair_net_15m_bps": baseline["pair_net_15m_bps"],
                "baseline_pair_net_30m_bps": baseline["pair_net_30m_bps"],
                "baseline_pair_win_15m": baseline["pair_win_15m"],
                "active_events": active_test["events"],
                "active_pair_net_15m_bps": active_test["pair_net_15m_bps"],
                "active_pair_net_30m_bps": active_test["pair_net_30m_bps"],
                "active_pair_win_15m": active_test["pair_win_15m"],
            }
        )

        previous_active = active

    active_df = pd.DataFrame(active_rows)
    perf_df = pd.DataFrame(perf_rows)
    if perf_df.empty:
        raise SystemExit("rolling selector produced no test folds")

    summary = pd.DataFrame(
        [
            {
                "variant": "baseline_all_symbols",
                "folds": int(len(perf_df)),
                "mean_events": float(perf_df["baseline_events"].mean()),
                "mean_pair_net_15m_bps": float(perf_df["baseline_pair_net_15m_bps"].mean()),
                "mean_pair_net_30m_bps": float(perf_df["baseline_pair_net_30m_bps"].mean()),
                "mean_pair_win_15m": float(perf_df["baseline_pair_win_15m"].mean()),
                "positive_folds": int((perf_df["baseline_pair_net_15m_bps"] > 0).sum()),
            },
            {
                "variant": "rolling_active_universe",
                "folds": int(len(perf_df)),
                "mean_events": float(perf_df["active_events"].mean()),
                "mean_pair_net_15m_bps": float(perf_df["active_pair_net_15m_bps"].mean()),
                "mean_pair_net_30m_bps": float(perf_df["active_pair_net_30m_bps"].mean()),
                "mean_pair_win_15m": float(perf_df["active_pair_win_15m"].mean()),
                "positive_folds": int((perf_df["active_pair_net_15m_bps"] > 0).sum()),
            },
        ]
    )

    churn = (
        active_df.groupby("test_day")["symbol"].apply(list).reset_index(name="symbols")
        if not active_df.empty
        else pd.DataFrame(columns=["test_day", "symbols"])
    )

    report_lines = [
        "# Rolling Universe Selector",
        "",
        f"- Input: `{args.input}`",
        f"- Train window: `{args.train_days}` days",
        f"- Max active symbols: `{args.max_active_symbols}`",
        f"- Entry rank: `{args.entry_rank}`",
        f"- Keep rank: `{args.keep_rank}`",
        f"- Min train events: `{args.min_train_events}`",
        f"- Min mean edge: `{args.min_mean_edge_bps:.2f}` bps",
        f"- Min positive-day share: `{args.min_positive_day_share:.2f}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.variant}`: folds={row.folds}, mean_events={row.mean_events:.1f}, "
            f"mean_pair_net_15m={row.mean_pair_net_15m_bps:+.2f} bps, "
            f"mean_pair_net_30m={row.mean_pair_net_30m_bps:+.2f} bps, "
            f"mean_pair_win_15m={row.mean_pair_win_15m * 100:.1f}%, positive_folds={row.positive_folds}/{row.folds}"
        )
    report_lines.extend(["", "## Daily Active Sets", ""])
    for row in perf_df.itertuples(index=False):
        report_lines.append(
            f"- `{row.test_day}`: active={row.active_symbol_count} [{row.active_symbols}], "
            f"active15={row.active_pair_net_15m_bps:+.2f} bps vs baseline15={row.baseline_pair_net_15m_bps:+.2f} bps"
        )

    prefix = args.output_prefix
    active_df.to_csv(OUT_DIR / f"{prefix}_active_symbols.csv", index=False)
    perf_df.to_csv(OUT_DIR / f"{prefix}_daily_performance.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    churn.to_csv(OUT_DIR / f"{prefix}_daily_sets.csv", index=False)
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    print(summary.to_string(index=False))
    print()
    print(perf_df.to_string(index=False))


if __name__ == "__main__":
    main()
