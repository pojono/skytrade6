#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dislocation_panel import build_panel


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen the recent shared-symbol universe for dislocation-event economics.")
    parser.add_argument("--start-date", default="2026-02-24")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--min-days", type=int, default=9)
    parser.add_argument("--require-micro", action="store_true")
    parser.add_argument("--min-gap-bps", type=float, default=10.0)
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--pair-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--train-days", type=int, default=5)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-test-events", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-prefix", default="universe_screen")
    return parser.parse_args()


def date_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def eligible_symbols(days: list[str], require_micro: bool, min_days: int) -> list[str]:
    bybit = {p.name for p in (DATALAKE / "bybit").iterdir() if p.is_dir()}
    binance = {p.name for p in (DATALAKE / "binance").iterdir() if p.is_dir()}
    symbols = []
    for sym in sorted(bybit & binance):
        ok = 0
        for day in days:
            minute_ok = (
                (DATALAKE / "bybit" / sym / f"{day}_kline_1m.csv").exists()
                and (DATALAKE / "binance" / sym / f"{day}_kline_1m.csv").exists()
            )
            if not minute_ok:
                continue
            if require_micro:
                micro_ok = (
                    (DATALAKE / "bybit" / sym / f"{day}_trades.csv.gz").exists()
                    and (DATALAKE / "bybit" / sym / f"{day}_orderbook.jsonl.gz").exists()
                    and (DATALAKE / "binance" / sym / f"{day}_trades.csv.gz").exists()
                    and (DATALAKE / "binance" / sym / f"{day}_bookDepth.csv.gz").exists()
                )
                if not micro_ok:
                    continue
            ok += 1
        if ok >= min_days:
            symbols.append(sym)
    return symbols


def prepare_panel(panel: pd.DataFrame, pair_fee_bps: float) -> pd.DataFrame:
    df = panel.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    group = df.groupby("symbol", sort=False)
    df["gap_bps"] = 10000.0 * (pd.to_numeric(df["bn_close"], errors="coerce") / pd.to_numeric(df["bb_close"], errors="coerce") - 1.0)
    for horizon in [5, 15, 30]:
        df[f"future_gap_bps_{horizon}m"] = group["gap_bps"].shift(-horizon)
        df[f"gap_close_{horizon}m_bps"] = df["gap_bps"] - df[f"future_gap_bps_{horizon}m"]
    df["pair_net_15m_bps"] = df["gap_close_15m_bps"] - pair_fee_bps
    df["pair_net_30m_bps"] = df["gap_close_30m_bps"] - pair_fee_bps
    df["pair_win_15m"] = (df["pair_net_15m_bps"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def select_events(df: pd.DataFrame, min_gap_bps: float, cooldown_minutes: int) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_minutes)
    for _, part in df.groupby("symbol", sort=False):
        last_ts: pd.Timestamp | None = None
        for row in part.itertuples(index=False):
            if not np.isfinite(row.gap_bps) or row.gap_bps < min_gap_bps:
                continue
            if not np.isfinite(row.pair_net_15m_bps):
                continue
            if last_ts is None or row.ts - last_ts >= cooldown:
                rows.append(
                    {
                        "symbol": row.symbol,
                        "ts": row.ts,
                        "date": row.date,
                        "gap_bps": row.gap_bps,
                        "gap_close_15m_bps": row.gap_close_15m_bps,
                        "gap_close_30m_bps": row.gap_close_30m_bps,
                        "pair_net_15m_bps": row.pair_net_15m_bps,
                        "pair_net_30m_bps": row.pair_net_30m_bps,
                        "pair_win_15m": row.pair_win_15m,
                    }
                )
                last_ts = row.ts
    return pd.DataFrame(rows)


def split_days(days: list[str], train_days: int) -> tuple[list[str], list[str]]:
    if len(days) <= train_days:
        raise ValueError("not enough distinct days for requested train window")
    return days[:train_days], days[train_days:]


def summarize(part: pd.DataFrame) -> dict[str, float]:
    if part.empty:
        return {
            "events": 0.0,
            "mean_pair_net_15m_bps": np.nan,
            "median_pair_net_15m_bps": np.nan,
            "mean_pair_net_30m_bps": np.nan,
            "win_rate_15m": np.nan,
        }
    return {
        "events": float(len(part)),
        "mean_pair_net_15m_bps": float(part["pair_net_15m_bps"].mean()),
        "median_pair_net_15m_bps": float(part["pair_net_15m_bps"].median()),
        "mean_pair_net_30m_bps": float(part["pair_net_30m_bps"].mean()),
        "win_rate_15m": float(part["pair_win_15m"].mean()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = date_range(args.start_date, args.end_date)
    symbols = eligible_symbols(days, require_micro=args.require_micro, min_days=args.min_days)
    if not symbols:
        raise SystemExit("no eligible symbols found")

    panel, metadata = build_panel(symbols, args.start_date, args.end_date, workers=args.workers)
    if panel.empty:
        raise SystemExit("panel build returned no rows")
    panel = prepare_panel(panel, pair_fee_bps=args.pair_fee_bps_roundtrip)
    events = select_events(panel, min_gap_bps=args.min_gap_bps, cooldown_minutes=args.cooldown_minutes)
    if events.empty:
        raise SystemExit("no events survived the screen filters")

    ordered_days = sorted(events["date"].unique().tolist())
    train_days, test_days = split_days(ordered_days, args.train_days)

    rows = []
    for symbol, part in events.groupby("symbol", sort=True):
        train = part[part["date"].isin(train_days)].copy()
        test = part[part["date"].isin(test_days)].copy()
        train_stats = summarize(train)
        test_stats = summarize(test)
        row = {
            "symbol": symbol,
            "train_days": len(train_days),
            "test_days": len(test_days),
            "train_events": train_stats["events"],
            "train_pair_net_15m_bps": train_stats["mean_pair_net_15m_bps"],
            "train_pair_net_30m_bps": train_stats["mean_pair_net_30m_bps"],
            "train_win_rate_15m": train_stats["win_rate_15m"],
            "test_events": test_stats["events"],
            "test_pair_net_15m_bps": test_stats["mean_pair_net_15m_bps"],
            "test_pair_net_30m_bps": test_stats["mean_pair_net_30m_bps"],
            "test_win_rate_15m": test_stats["win_rate_15m"],
            "all_events": float(len(part)),
            "all_pair_net_15m_bps": float(part["pair_net_15m_bps"].mean()),
            "all_pair_net_30m_bps": float(part["pair_net_30m_bps"].mean()),
            "mean_gap_bps": float(part["gap_bps"].mean()),
        }
        row["selection_score"] = (
            np.nan_to_num(row["test_pair_net_15m_bps"], nan=-1e9)
            * np.log1p(max(row["test_events"], 0.0))
        )
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(
        ["test_pair_net_15m_bps", "test_events", "train_pair_net_15m_bps"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    filtered = leaderboard[
        (leaderboard["train_events"] >= args.min_train_events)
        & (leaderboard["test_events"] >= args.min_test_events)
    ].copy()
    top = filtered.head(args.top_k) if not filtered.empty else leaderboard.head(args.top_k)

    prefix = args.output_prefix
    leaderboard.to_csv(OUT_DIR / f"{prefix}_leaderboard.csv", index=False)
    top.to_csv(OUT_DIR / f"{prefix}_top_candidates.csv", index=False)
    events.to_csv(OUT_DIR / f"{prefix}_events.csv", index=False)

    summary_lines = [
        "# Universe Screen",
        "",
        f"- Start date: `{args.start_date}`",
        f"- End date: `{args.end_date}`",
        f"- Eligible symbols: `{len(symbols)}`",
        f"- Event threshold: `{args.min_gap_bps:.1f}` bps",
        f"- Cooldown: `{args.cooldown_minutes}` minutes",
        f"- Pair fee assumption: `{args.pair_fee_bps_roundtrip:.1f}` bps",
        f"- Train days: `{train_days}`",
        f"- Test days: `{test_days}`",
        f"- Total events: `{len(events):,}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in top.head(15).itertuples(index=False):
        summary_lines.append(
            f"- `{row.symbol}`: test15={row.test_pair_net_15m_bps:+.2f} bps on {int(row.test_events)} events, "
            f"train15={row.train_pair_net_15m_bps:+.2f} bps on {int(row.train_events)} events, "
            f"all15={row.all_pair_net_15m_bps:+.2f} bps"
        )
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(summary_lines) + "\n", encoding="ascii")

    print(f"eligible_symbols={len(symbols)} total_events={len(events):,}")
    print(top.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
