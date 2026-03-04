#!/usr/bin/env python3
"""Run anti-overfit follow-up analyses on the frozen 3-symbol candidate trades."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3.csv"
DEFAULT_META_CFG = OUT_DIR / "meta_filter_config.json"


@dataclass(frozen=True)
class Trade:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    gross_pnl_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float
    score: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float


@dataclass(frozen=True)
class ReplayConfig:
    starting_capital: float = 100000.0
    per_trade_allocation: float = 0.10
    max_open_positions: int = 1
    max_open_per_symbol: int = 1
    max_symbol_allocation: float = 0.10
    daily_cap_per_symbol: int = 3
    min_signal_bps: float = 10.0
    fee_bps_roundtrip: float = 6.0
    extra_slippage_bps: float = 1.0
    spread_slip_coeff: float = 0.10
    velocity_slip_coeff: float = 0.05
    daily_loss_stop_pct: float = 0.01
    monthly_loss_stop_pct: float = 0.03


def load_trades(path: Path) -> list[Trade]:
    rows: list[Trade] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                Trade(
                    symbol=row["symbol"],
                    day=row["day"],
                    month=row["month"],
                    entry_ts_ms=int(row["entry_ts_ms"]),
                    exit_ts_ms=int(row["exit_ts_ms"]),
                    gross_pnl_bps=float(row["gross_pnl_bps"]),
                    entry_spread_abs_bps=float(row["entry_spread_abs_bps"]),
                    entry_spread_velocity_bps=float(row["entry_spread_velocity_bps"]),
                    score=float(row["score"]),
                    ls_diff_signed=float(row["ls_diff_signed"]),
                    oi_diff_signed_bps=float(row["oi_diff_signed_bps"]),
                    carry_diff_signed_bps=float(row["carry_diff_signed_bps"]),
                )
            )
    rows.sort(key=lambda row: (row.entry_ts_ms, -row.score, row.symbol))
    return rows


def trade_slippage(trade: Trade, cfg: ReplayConfig) -> float:
    stretch = max(0.0, trade.entry_spread_abs_bps - cfg.min_signal_bps)
    return (
        cfg.extra_slippage_bps
        + cfg.spread_slip_coeff * stretch
        + cfg.velocity_slip_coeff * trade.entry_spread_velocity_bps
    )


def net_bps(trade: Trade, cfg: ReplayConfig) -> float:
    return trade.gross_pnl_bps - cfg.fee_bps_roundtrip - trade_slippage(trade, cfg)


def selector_key(trade: Trade) -> tuple[float, float, str]:
    return (-trade.entry_spread_abs_bps, -trade.score, trade.symbol)


def passes_meta_filter(trade: Trade, cfg: dict[str, float]) -> bool:
    need_score = cfg["min_score"] + (cfg["sei_score_extra"] if trade.symbol == "SEIUSDT" else 0.0)
    return (
        trade.score >= need_score
        and trade.entry_spread_velocity_bps <= cfg["max_velocity"]
        and trade.entry_spread_abs_bps >= cfg["min_spread_abs"]
        and trade.ls_diff_signed >= cfg["min_ls"]
        and trade.oi_diff_signed_bps >= cfg["min_oi"]
        and trade.carry_diff_signed_bps >= cfg["min_carry"]
    )


def should_block_new_trade(
    trade: Trade,
    day_pnl: dict[str, float],
    month_pnl: dict[str, float],
    day_start_equity: dict[str, float],
    month_start_equity: dict[str, float],
    cfg: ReplayConfig,
) -> bool:
    if cfg.daily_loss_stop_pct > 0:
        start = day_start_equity.get(trade.day)
        if start is not None and day_pnl.get(trade.day, 0.0) <= -(start * cfg.daily_loss_stop_pct):
            return True
    if cfg.monthly_loss_stop_pct > 0:
        start = month_start_equity.get(trade.month)
        if start is not None and month_pnl.get(trade.month, 0.0) <= -(start * cfg.monthly_loss_stop_pct):
            return True
    return False


def replay(
    trades: list[Trade],
    cfg: ReplayConfig,
) -> dict[str, object]:
    balance = cfg.starting_capital
    open_positions: list[tuple[Trade, float, float]] = []
    day_pnl: dict[str, float] = {}
    month_pnl: dict[str, float] = {}
    day_start_equity: dict[str, float] = {}
    month_start_equity: dict[str, float] = {}
    daily_counts: dict[tuple[str, str], int] = {}

    filled = 0
    win_count = 0
    net_bps_sum = 0.0
    symbol_pnl: dict[str, float] = {}
    month_rows: list[tuple[str, int, float, float, float]] = []
    month_fill_counts: dict[str, int] = {}
    month_net_bps_sum: dict[str, float] = {}
    month_pnl_sum: dict[str, float] = {}

    idx = 0
    while idx < len(trades):
        entry_ts = trades[idx].entry_ts_ms

        still_open: list[tuple[Trade, float, float]] = []
        for trade, alloc, pnl in open_positions:
            if trade.exit_ts_ms <= entry_ts:
                balance += pnl
                day_pnl[trade.day] = day_pnl.get(trade.day, 0.0) + pnl
                month_pnl[trade.month] = month_pnl.get(trade.month, 0.0) + pnl
                month_pnl_sum[trade.month] = month_pnl_sum.get(trade.month, 0.0) + pnl
            else:
                still_open.append((trade, alloc, pnl))
        open_positions = still_open

        batch: list[Trade] = []
        while idx < len(trades) and trades[idx].entry_ts_ms == entry_ts:
            batch.append(trades[idx])
            idx += 1

        available_slots = max(0, cfg.max_open_positions - len(open_positions))
        if available_slots == 0:
            continue

        batch.sort(key=selector_key)
        accepted = 0
        for trade in batch:
            if accepted >= available_slots:
                break
            if trade.day not in day_start_equity:
                day_start_equity[trade.day] = balance
            if trade.month not in month_start_equity:
                month_start_equity[trade.month] = balance
            if should_block_new_trade(trade, day_pnl, month_pnl, day_start_equity, month_start_equity, cfg):
                continue

            day_key = (trade.symbol, trade.day)
            if daily_counts.get(day_key, 0) >= cfg.daily_cap_per_symbol:
                continue

            open_same_symbol = [row for row in open_positions if row[0].symbol == trade.symbol]
            if len(open_same_symbol) >= cfg.max_open_per_symbol:
                continue

            current_symbol_alloc = sum(row[1] for row in open_same_symbol)
            alloc = balance * cfg.per_trade_allocation
            max_symbol_dollars = balance * cfg.max_symbol_allocation
            if current_symbol_alloc + alloc > max_symbol_dollars + 1e-9:
                continue

            trade_net_bps = net_bps(trade, cfg)
            pnl = alloc * (trade_net_bps / 10000.0)
            open_positions.append((trade, alloc, pnl))
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
            filled += 1
            net_bps_sum += trade_net_bps
            if trade_net_bps > 0:
                win_count += 1
            symbol_pnl[trade.symbol] = symbol_pnl.get(trade.symbol, 0.0) + pnl
            month_fill_counts[trade.month] = month_fill_counts.get(trade.month, 0) + 1
            month_net_bps_sum[trade.month] = month_net_bps_sum.get(trade.month, 0.0) + trade_net_bps
            accepted += 1

    for trade, alloc, pnl in open_positions:
        balance += pnl
        day_pnl[trade.day] = day_pnl.get(trade.day, 0.0) + pnl
        month_pnl[trade.month] = month_pnl.get(trade.month, 0.0) + pnl
        month_pnl_sum[trade.month] = month_pnl_sum.get(trade.month, 0.0) + pnl

    for month in sorted(month_fill_counts):
        fills = month_fill_counts[month]
        avg_net = month_net_bps_sum[month] / fills if fills else math.nan
        month_rows.append((month, fills, avg_net, month_pnl_sum.get(month, 0.0), balance))

    avg_net_bps = net_bps_sum / filled if filled else math.nan
    win_rate = win_count / filled if filled else math.nan
    return {
        "filled_trades": filled,
        "win_rate": win_rate,
        "avg_net_bps": avg_net_bps,
        "total_pnl_dollars": balance - cfg.starting_capital,
        "final_capital": balance,
        "symbol_pnl": symbol_pnl,
        "months": month_rows,
    }


def train_test_split(trades: list[Trade], test_months: int) -> tuple[list[Trade], list[Trade], list[str], list[str]]:
    months = sorted({trade.month for trade in trades})
    test = months[-test_months:]
    train = months[:-test_months]
    train_set = set(train)
    test_set = set(test)
    train_rows = [trade for trade in trades if trade.month in train_set]
    test_rows = [trade for trade in trades if trade.month in test_set]
    return train_rows, test_rows, train, test


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_ranker(train_trades: list[Trade], replay_cfg: ReplayConfig) -> dict[str, object]:
    train_net = [net_bps(trade, replay_cfg) for trade in train_trades]
    global_mean = mean(train_net)

    score_bins = [8.0, 10.0, 12.0, 14.0]
    spread_bins = [10.0, 12.0, 16.0, 20.0]
    velocity_bins = [8.0, 16.0, 24.0]

    def score_bucket(value: float, cuts: list[float]) -> str:
        for cut in cuts:
            if value < cut:
                return f"<{cut:g}"
        return f">={cuts[-1]:g}"

    def bucket_avg(values: list[tuple[str, float]]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, value in values:
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
        return {key: sums[key] / counts[key] for key in sums}

    symbol_avg = bucket_avg([(trade.symbol, net_bps(trade, replay_cfg)) for trade in train_trades])
    score_avg = bucket_avg(
        [(score_bucket(trade.score, score_bins), net_bps(trade, replay_cfg)) for trade in train_trades]
    )
    spread_avg = bucket_avg(
        [
            (score_bucket(trade.entry_spread_abs_bps, spread_bins), net_bps(trade, replay_cfg))
            for trade in train_trades
        ]
    )
    velocity_avg = bucket_avg(
        [
            (score_bucket(trade.entry_spread_velocity_bps, velocity_bins), net_bps(trade, replay_cfg))
            for trade in train_trades
        ]
    )

    def predict(trade: Trade) -> float:
        pred = global_mean
        pred += symbol_avg.get(trade.symbol, global_mean) - global_mean
        pred += score_avg.get(score_bucket(trade.score, score_bins), global_mean) - global_mean
        pred += spread_avg.get(score_bucket(trade.entry_spread_abs_bps, spread_bins), global_mean) - global_mean
        pred += velocity_avg.get(score_bucket(trade.entry_spread_velocity_bps, velocity_bins), global_mean) - global_mean
        return pred

    return {
        "global_mean": global_mean,
        "score_bins": score_bins,
        "spread_bins": spread_bins,
        "velocity_bins": velocity_bins,
        "symbol_avg": symbol_avg,
        "score_avg": score_avg,
        "spread_avg": spread_avg,
        "velocity_avg": velocity_avg,
        "predict": predict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--meta-config", type=Path, default=DEFAULT_META_CFG)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--report-output", type=Path, default=OUT_DIR / "anti_overfit_followups.md")
    parser.add_argument("--sleeve-csv", type=Path, default=OUT_DIR / "sleeve_comparison.csv")
    parser.add_argument("--ranker-json", type=Path, default=OUT_DIR / "prob_ranker_config.json")
    parser.add_argument("--replay-json", type=Path, default=OUT_DIR / "replay_filter_config.json")
    args = parser.parse_args()

    replay_cfg = ReplayConfig()
    meta_cfg = json.loads(args.meta_config.read_text())
    all_trades = load_trades(args.input)
    train_rows, test_rows, train_months, test_months = train_test_split(all_trades, args.test_months)

    baseline_test = replay(test_rows, replay_cfg)
    filtered_train_rows = [trade for trade in train_rows if passes_meta_filter(trade, meta_cfg)]
    filtered_test_rows = [trade for trade in test_rows if passes_meta_filter(trade, meta_cfg)]
    filtered_train = replay(filtered_train_rows, replay_cfg)
    filtered_test = replay(filtered_test_rows, replay_cfg)

    train_positive_symbols = sorted(
        symbol for symbol, pnl in filtered_train["symbol_pnl"].items() if pnl > 0
    )
    if not train_positive_symbols:
        train_positive_symbols = sorted({trade.symbol for trade in filtered_train_rows})
    reduced_test_rows = [trade for trade in filtered_test_rows if trade.symbol in set(train_positive_symbols)]
    reduced_test = replay(reduced_test_rows, replay_cfg)

    sleeve_rows = [
        [
            "baseline_3sym_unfiltered",
            ",".join(sorted({trade.symbol for trade in test_rows})),
            baseline_test["filled_trades"],
            f"{baseline_test['win_rate']:.6f}",
            f"{baseline_test['avg_net_bps']:.6f}",
            f"{baseline_test['total_pnl_dollars']:.2f}",
        ],
        [
            "filtered_3sym",
            ",".join(sorted({trade.symbol for trade in filtered_test_rows})),
            filtered_test["filled_trades"],
            f"{filtered_test['win_rate']:.6f}",
            f"{filtered_test['avg_net_bps']:.6f}",
            f"{filtered_test['total_pnl_dollars']:.2f}",
        ],
        [
            "filtered_train_selected_sleeve",
            ",".join(train_positive_symbols),
            reduced_test["filled_trades"],
            f"{reduced_test['win_rate']:.6f}",
            f"{reduced_test['avg_net_bps']:.6f}",
            f"{reduced_test['total_pnl_dollars']:.2f}",
        ],
    ]
    write_csv(
        args.sleeve_csv,
        ["variant", "symbols", "filled_trades", "win_rate", "avg_net_bps", "total_pnl_dollars"],
        sleeve_rows,
    )

    ranker = build_ranker(train_rows, replay_cfg)
    predict = ranker["predict"]
    train_scored = sorted(((predict(trade), trade) for trade in train_rows), key=lambda row: row[0], reverse=True)
    train_scores = [row[0] for row in train_scored]
    cutoff_specs = []
    for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        keep_count = max(1, min(len(train_scores), int(math.ceil(frac * len(train_scores)))))
        idx = keep_count - 1
        cutoff_specs.append((frac, train_scores[idx]))

    best_rank_cut = None
    best_rank_train = None
    best_rank_score = float("-inf")
    rank_rows: list[list[object]] = []
    for top_frac, cutoff in cutoff_specs:
        keep_train = [trade for trade in train_rows if predict(trade) >= cutoff]
        keep_test = [trade for trade in test_rows if predict(trade) >= cutoff]
        train_replay = replay(keep_train, replay_cfg)
        test_replay = replay(keep_test, replay_cfg)
        rank_rows.append(
            [
                f"top_{int(top_frac * 100)}pct",
                f"{cutoff:.6f}",
                train_replay["filled_trades"],
                f"{train_replay['win_rate']:.6f}",
                f"{train_replay['avg_net_bps']:.6f}",
                f"{train_replay['total_pnl_dollars']:.2f}",
                test_replay["filled_trades"],
                f"{test_replay['win_rate']:.6f}",
                f"{test_replay['avg_net_bps']:.6f}",
                f"{test_replay['total_pnl_dollars']:.2f}",
            ]
        )
        if train_replay["filled_trades"] < 500:
            continue
        score = float(train_replay["total_pnl_dollars"])
        if score > best_rank_score:
            best_rank_score = score
            best_rank_cut = cutoff
            best_rank_train = train_replay

    if best_rank_cut is None:
        best_rank_cut = cutoff_specs[0][1]
        best_rank_train = replay(train_rows, replay_cfg)
    ranker_test = replay([trade for trade in test_rows if predict(trade) >= best_rank_cut], replay_cfg)

    ranker_out = {
        "selected_cutoff": best_rank_cut,
        "train_months": train_months,
        "test_months": test_months,
        "global_mean": ranker["global_mean"],
        "symbol_avg": ranker["symbol_avg"],
        "score_avg": ranker["score_avg"],
        "spread_avg": ranker["spread_avg"],
        "velocity_avg": ranker["velocity_avg"],
    }
    args.ranker_json.write_text(json.dumps(ranker_out, indent=2, sort_keys=True))

    # Small local search to avoid overfitting while still optimizing for replay dollars.
    search_space = {
        "min_score": [6.0, 8.0, 10.0],
        "sei_score_extra": [8.0, 10.0, 12.0],
        "max_velocity": [10.0, 12.0, 14.0],
        "min_spread_abs": [10.0, 12.0, 14.0],
    }
    fixed_tail = {"min_ls": 0.15, "min_oi": 5.0, "min_carry": 2.0}

    best_replay_cfg = None
    best_replay_train = None
    best_replay_score = float("-inf")
    replay_rows: list[list[object]] = []
    for min_score, sei_score_extra, max_velocity, min_spread_abs in product(
        search_space["min_score"],
        search_space["sei_score_extra"],
        search_space["max_velocity"],
        search_space["min_spread_abs"],
    ):
        threshold_cfg = {
            "min_score": min_score,
            "sei_score_extra": sei_score_extra,
            "max_velocity": max_velocity,
            "min_spread_abs": min_spread_abs,
            **fixed_tail,
        }
        keep_train = [trade for trade in train_rows if passes_meta_filter(trade, threshold_cfg)]
        keep_test = [trade for trade in test_rows if passes_meta_filter(trade, threshold_cfg)]
        train_replay = replay(keep_train, replay_cfg)
        if train_replay["filled_trades"] < 500:
            continue
        test_replay = replay(keep_test, replay_cfg)
        replay_rows.append(
            [
                f"{min_score:.1f}",
                f"{sei_score_extra:.1f}",
                f"{max_velocity:.1f}",
                f"{min_spread_abs:.1f}",
                train_replay["filled_trades"],
                f"{train_replay['win_rate']:.6f}",
                f"{train_replay['avg_net_bps']:.6f}",
                f"{train_replay['total_pnl_dollars']:.2f}",
                test_replay["filled_trades"],
                f"{test_replay['win_rate']:.6f}",
                f"{test_replay['avg_net_bps']:.6f}",
                f"{test_replay['total_pnl_dollars']:.2f}",
            ]
        )
        score = float(train_replay["total_pnl_dollars"])
        if score > best_replay_score:
            best_replay_score = score
            best_replay_cfg = threshold_cfg
            best_replay_train = train_replay

    if best_replay_cfg is None:
        best_replay_cfg = dict(meta_cfg)
        best_replay_train = replay(filtered_train_rows, replay_cfg)
    replay_test = replay([trade for trade in test_rows if passes_meta_filter(trade, best_replay_cfg)], replay_cfg)
    args.replay_json.write_text(json.dumps(best_replay_cfg, indent=2, sort_keys=True))

    write_csv(
        OUT_DIR / "ranker_cutoff_search.csv",
        [
            "cutoff_label",
            "predicted_cutoff",
            "train_fills",
            "train_win_rate",
            "train_avg_net_bps",
            "train_total_pnl_dollars",
            "test_fills",
            "test_win_rate",
            "test_avg_net_bps",
            "test_total_pnl_dollars",
        ],
        rank_rows,
    )
    write_csv(
        OUT_DIR / "replay_filter_search.csv",
        [
            "min_score",
            "sei_score_extra",
            "max_velocity",
            "min_spread_abs",
            "train_fills",
            "train_win_rate",
            "train_avg_net_bps",
            "train_total_pnl_dollars",
            "test_fills",
            "test_win_rate",
            "test_avg_net_bps",
            "test_total_pnl_dollars",
        ],
        replay_rows,
    )

    lines = [
        "# Anti-Overfit Follow-Ups",
        "",
        f"- Train months: {', '.join(train_months)}",
        f"- Test months: {', '.join(test_months)}",
        "",
        "## 1. Train-Selected Symbol Sleeve",
        "",
        f"- Symbols kept from filtered train replay: {', '.join(train_positive_symbols)}",
        f"- Baseline 3-symbol unfiltered test: {baseline_test['filled_trades']} fills, {baseline_test['win_rate']:.2%} win rate, {baseline_test['avg_net_bps']:.4f} bps, ${baseline_test['total_pnl_dollars']:.2f}",
        f"- Filtered 3-symbol test: {filtered_test['filled_trades']} fills, {filtered_test['win_rate']:.2%} win rate, {filtered_test['avg_net_bps']:.4f} bps, ${filtered_test['total_pnl_dollars']:.2f}",
        f"- Train-selected sleeve test: {reduced_test['filled_trades']} fills, {reduced_test['win_rate']:.2%} win rate, {reduced_test['avg_net_bps']:.4f} bps, ${reduced_test['total_pnl_dollars']:.2f}",
        "",
        "## 2. Simple Probability Ranker",
        "",
        f"- Selected predicted cutoff: {best_rank_cut:.4f}",
        f"- Train replay: {best_rank_train['filled_trades']} fills, {best_rank_train['win_rate']:.2%} win rate, {best_rank_train['avg_net_bps']:.4f} bps, ${best_rank_train['total_pnl_dollars']:.2f}",
        f"- Test replay: {ranker_test['filled_trades']} fills, {ranker_test['win_rate']:.2%} win rate, {ranker_test['avg_net_bps']:.4f} bps, ${ranker_test['total_pnl_dollars']:.2f}",
        "",
        "## 3. Replay-PnL Threshold Search",
        "",
        "Compact search space to reduce overfitting:",
        "",
        "- `min_score` in {6, 8, 10}",
        "- `sei_score_extra` in {8, 10, 12}",
        "- `max_velocity` in {10, 12, 14}",
        "- `min_spread_abs` in {10, 12, 14}",
        "- `min_ls = 0.15`, `min_oi = 5`, `min_carry = 2` fixed",
        "",
        "Selected replay-optimized config (train only):",
        "",
        "```json",
        json.dumps(best_replay_cfg, indent=2, sort_keys=True),
        "```",
        "",
        f"- Train replay: {best_replay_train['filled_trades']} fills, {best_replay_train['win_rate']:.2%} win rate, {best_replay_train['avg_net_bps']:.4f} bps, ${best_replay_train['total_pnl_dollars']:.2f}",
        f"- Test replay: {replay_test['filled_trades']} fills, {replay_test['win_rate']:.2%} win rate, {replay_test['avg_net_bps']:.4f} bps, ${replay_test['total_pnl_dollars']:.2f}",
        "",
    ]
    args.report_output.write_text("\n".join(lines))
    print(f"Wrote {args.sleeve_csv}")
    print(f"Wrote {args.ranker_json}")
    print(f"Wrote {args.replay_json}")
    print(f"Wrote {OUT_DIR / 'ranker_cutoff_search.csv'}")
    print(f"Wrote {OUT_DIR / 'replay_filter_search.csv'}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
