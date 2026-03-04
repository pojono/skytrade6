#!/usr/bin/env python3
"""Apply the selected meta-filter, export filtered trades, and replay the paper strategy."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3.csv"
DEFAULT_CFG = OUT_DIR / "meta_filter_config.json"


def passes(row: dict[str, str], cfg: dict[str, float]) -> bool:
    score = float(row["score"])
    velocity = float(row["entry_spread_velocity_bps"])
    spread_abs = float(row["entry_spread_abs_bps"])
    need_score = cfg["min_score"] + (cfg["sei_score_extra"] if row["symbol"] == "SEIUSDT" else 0.0)
    return (
        score >= need_score
        and velocity <= cfg["max_velocity"]
        and spread_abs >= cfg["min_spread_abs"]
        and float(row["ls_diff_signed"]) >= cfg["min_ls"]
        and float(row["oi_diff_signed_bps"]) >= cfg["min_oi"]
        and float(row["carry_diff_signed_bps"]) >= cfg["min_carry"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--filtered-output", type=Path, default=OUT_DIR / "candidate_trades_v3_meta.csv")
    parser.add_argument("--report-output", type=Path, default=OUT_DIR / "paper_report_v3_meta.md")
    parser.add_argument("--fills-output", type=Path, default=OUT_DIR / "paper_fills_v3_meta.csv")
    parser.add_argument("--monthly-output", type=Path, default=OUT_DIR / "paper_monthly_v3_meta.csv")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    rows = list(csv.DictReader(args.input.open()))
    keep = [row for row in rows if passes(row, cfg)]

    with args.filtered_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(keep)

    cmd = [
        "python3",
        "codex-exp-1/paper_trade_candidate.py",
        "--input", str(args.filtered_output),
        "--starting-capital", "100000",
        "--per-trade-allocation", "0.10",
        "--max-open-positions", "1",
        "--max-open-per-symbol", "1",
        "--max-symbol-allocation", "0.10",
        "--daily-cap-per-symbol", "3",
        "--selector-mode", "spread",
        "--daily-loss-stop-pct", "0.01",
        "--monthly-loss-stop-pct", "0.03",
        "--min-signal-bps", "10",
        "--fee-bps-roundtrip", "6",
        "--extra-slippage-bps", "1",
        "--spread-slip-coeff", "0.10",
        "--velocity-slip-coeff", "0.05",
        "--output-fills", str(args.fills_output),
        "--output-monthly", str(args.monthly_output),
        "--output-report", str(args.report_output),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"Filtered trades kept: {len(keep)} / {len(rows)}")
    print(f"Wrote {args.filtered_output}")
    print(cp.stdout.strip())


if __name__ == "__main__":
    main()
