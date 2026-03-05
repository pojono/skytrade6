from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRADES_INPUT = ROOT / "codex-exp-3" / "revalidated_exp2_symbol_trades.csv"
BINANCE_SCRIPT = ROOT / "datalake" / "download_binance_data.py"
BYBIT_SCRIPT = ROOT / "datalake" / "download_bybit_data.py"


def build_windows() -> pd.DataFrame:
    trades = pd.read_csv(TRADES_INPUT, parse_dates=["ts"])
    test = trades.loc[trades["ts"] >= pd.Timestamp("2026-01-01 00:00:00+00:00")].copy()
    if test.empty:
        raise RuntimeError("No 2026 test trades found.")
    test["date"] = test["ts"].dt.strftime("%Y-%m-%d")
    windows = (
        test.groupby("symbol", as_index=False)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            signal_days=("date", "nunique"),
            signals=("symbol", "count"),
        )
        .sort_values(["signals", "signal_days", "symbol"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return windows


def run_download(script: Path, symbol: str, start_date: str, end_date: str) -> None:
    cmd = [
        sys.executable,
        str(script),
        symbol,
        start_date,
        end_date,
        "-t",
        "trades",
        "-c",
        "3",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    windows = build_windows()
    print(f"Symbols to process: {len(windows)}")
    for _, row in windows.iterrows():
        symbol = row["symbol"]
        start_date = row["start_date"]
        end_date = row["end_date"]
        print(
            f"\n=== {symbol} | signals={int(row['signals'])} | days={int(row['signal_days'])} "
            f"| {start_date} -> {end_date} ==="
        )
        run_download(BINANCE_SCRIPT, symbol, start_date, end_date)
        run_download(BYBIT_SCRIPT, symbol, start_date, end_date)


if __name__ == "__main__":
    main()
