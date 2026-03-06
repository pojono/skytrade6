from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "codex-exp-3" / "revalidated_exp2_symbol_trades.csv"
BINANCE_SCRIPT = ROOT / "datalake" / "download_binance_data.py"
BYBIT_SCRIPT = ROOT / "datalake" / "download_bybit_data.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand historical data for strategy symbols.")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-symbols", type=int, default=0, help="Top symbols by signal frequency (0 = all).")
    p.add_argument("--min-trades", type=int, default=0, help="Minimum historical selected-trade count per symbol.")
    return p.parse_args()


def symbol_list(max_symbols: int, min_trades: int) -> list[str]:
    df = pd.read_csv(TRADES_CSV)
    counts = df["symbol"].value_counts().reset_index()
    counts.columns = ["symbol", "n"]
    if min_trades > 0:
        counts = counts.loc[counts["n"] >= min_trades].copy()
    if max_symbols > 0:
        counts = counts.head(max_symbols).copy()
    return counts["symbol"].tolist()


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def batched(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> None:
    args = parse_args()
    syms = symbol_list(args.max_symbols, args.min_trades)
    print(f"Total symbols: {len(syms)}")
    for i, batch in enumerate(batched(syms, args.batch_size), start=1):
        joined = ",".join(batch)
        print(f"\n=== Batch {i} | {len(batch)} symbols ===")
        run_cmd(
            [
                sys.executable,
                str(BINANCE_SCRIPT),
                joined,
                args.start,
                args.end,
                "-t",
                "MetricsLinear",
                "-c",
                str(args.concurrency),
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(BYBIT_SCRIPT),
                joined,
                args.start,
                args.end,
                "-t",
                "MetricsLinear",
                "-c",
                str(args.concurrency),
            ]
        )


if __name__ == "__main__":
    main()
