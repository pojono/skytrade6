#!/usr/bin/env python3
"""Replay microstructure-gated variants on the overlapping 30-day high-resolution window."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "microstructure_window_analysis_30d.csv"


def passes(row: dict[str, str], cfg: dict[str, float]) -> bool:
    return (
        float(row["bybit_trade_count_5s"]) >= cfg["min_bybit_trade_count_5s"]
        and float(row["binance_trade_count_5s"]) >= cfg["min_binance_trade_count_5s"]
        and float(row["bybit_book_spread_bps"]) <= cfg["max_bybit_book_spread_bps"]
        and float(row["bybit_trade_imbalance_5s"]) <= cfg["max_bybit_trade_imbalance_5s"]
    )


def write_subset(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_replay(input_path: Path, report_path: Path, fills_path: Path, monthly_path: Path) -> None:
    cmd = [
        "python3",
        "codex-exp-1/paper_trade_candidate.py",
        "--input", str(input_path),
        "--starting-capital", "100000",
        "--per-trade-allocation", "0.25",
        "--max-open-positions", "1",
        "--max-open-per-symbol", "1",
        "--max-symbol-allocation", "0.25",
        "--daily-cap-per-symbol", "3",
        "--selector-mode", "spread",
        "--daily-loss-stop-pct", "0.01",
        "--monthly-loss-stop-pct", "0.03",
        "--min-signal-bps", "10",
        "--fee-bps-roundtrip", "6",
        "--extra-slippage-bps", "1",
        "--spread-slip-coeff", "0.10",
        "--velocity-slip-coeff", "0.05",
        "--size-slip-coeff", "1.5",
        "--base-allocation-ref", "0.10",
        "--output-fills", str(fills_path),
        "--output-monthly", str(monthly_path),
        "--output-report", str(report_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "microstructure_replay_comparison_30d.md")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open()))

    train_selected = {
        "min_bybit_trade_count_5s": 2.0,
        "min_binance_trade_count_5s": 0.0,
        "max_bybit_book_spread_bps": 4.5,
        "max_bybit_trade_imbalance_5s": 1.0,
    }
    hypothesis = {
        "min_bybit_trade_count_5s": 4.0,
        "min_binance_trade_count_5s": 0.0,
        "max_bybit_book_spread_bps": 4.5,
        "max_bybit_trade_imbalance_5s": 1.0,
    }

    base_csv = OUT_DIR / "microstructure_replay_base_30d.csv"
    train_csv = OUT_DIR / "microstructure_replay_train_gate_30d.csv"
    hyp_csv = OUT_DIR / "microstructure_replay_hyp_gate_30d.csv"
    write_subset(base_csv, rows)
    write_subset(train_csv, [row for row in rows if passes(row, train_selected)])
    write_subset(hyp_csv, [row for row in rows if passes(row, hypothesis)])

    run_replay(
        base_csv,
        OUT_DIR / "paper_report_micro_base_30d.md",
        OUT_DIR / "paper_fills_micro_base_30d.csv",
        OUT_DIR / "paper_monthly_micro_base_30d.csv",
    )
    run_replay(
        train_csv,
        OUT_DIR / "paper_report_micro_train_gate_30d.md",
        OUT_DIR / "paper_fills_micro_train_gate_30d.csv",
        OUT_DIR / "paper_monthly_micro_train_gate_30d.csv",
    )
    run_replay(
        hyp_csv,
        OUT_DIR / "paper_report_micro_hyp_gate_30d.md",
        OUT_DIR / "paper_fills_micro_hyp_gate_30d.csv",
        OUT_DIR / "paper_monthly_micro_hyp_gate_30d.csv",
    )

    def read_results(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in path.read_text().splitlines():
            if line.startswith("- Filled trades:"):
                values["fills"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Final capital:"):
                values["final_capital"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Total PnL:"):
                values["pnl"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Average net edge per fill:"):
                values["avg_net"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Win rate:"):
                values["win_rate"] = line.split(":", 1)[1].strip()
        return values

    base_res = read_results(OUT_DIR / "paper_report_micro_base_30d.md")
    train_res = read_results(OUT_DIR / "paper_report_micro_train_gate_30d.md")
    hyp_res = read_results(OUT_DIR / "paper_report_micro_hyp_gate_30d.md")

    lines = [
        "# Microstructure Replay Comparison (30d Window)",
        "",
        "- All three variants use the same frozen 25% live-style replay assumptions.",
        "- Only the trade eligibility differs via the microstructure gate.",
        "",
        "| Variant | Filled Trades | Total PnL | Avg Net | Win Rate | Final Capital |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Base 30d window | {base_res['fills']} | {base_res['pnl']} | {base_res['avg_net']} | {base_res['win_rate']} | {base_res['final_capital']} |",
        f"| Train-selected gate | {train_res['fills']} | {train_res['pnl']} | {train_res['avg_net']} | {train_res['win_rate']} | {train_res['final_capital']} |",
        f"| Hypothesis gate | {hyp_res['fills']} | {hyp_res['pnl']} | {hyp_res['avg_net']} | {hyp_res['win_rate']} | {hyp_res['final_capital']} |",
        "",
    ]
    args.output_md.write_text("\n".join(lines))

    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
