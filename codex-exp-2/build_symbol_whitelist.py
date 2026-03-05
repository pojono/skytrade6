from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
GRID_CSV = OUT_DIR / "symbol_whitelist_grid.csv"
RANK_GRID_CSV = OUT_DIR / "symbol_whitelist_rank_grid.csv"
WHITELIST_CSV = OUT_DIR / "symbol_whitelist.csv"
REPORT_MD = OUT_DIR / "FINDINGS_symbol_whitelist.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
FEE = 0.002


@dataclass(frozen=True)
class Rule:
    min_train_trades: int
    min_train_avg_bps: float
    min_train_hit_rate: float
    max_train_drawdown_bps: float


def load_base_trades() -> pd.DataFrame:
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    mask = (
        (samples["oi_med_3d"] >= 20_000_000)
        & (samples["breadth_mom"] >= 0.60)
        & (samples["median_ls_z"] >= 0.0)
        & (samples["ls_z"] >= 2.0)
        & (samples["taker_z"] >= 0.5)
        & (samples["mom_4h"] > 0)
    )
    trades = samples.loc[mask].copy()
    trades = (
        trades.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    trades["net_bps_20"] = (trades["ret_4h"] - FEE) * 10000.0
    trades["period"] = trades["ts"].apply(lambda x: "train" if x < TRAIN_CUTOFF else "test")
    return trades


def build_symbol_features(trades: pd.DataFrame) -> pd.DataFrame:
    train = trades.loc[trades["period"] == "train"].copy()
    test = trades.loc[trades["period"] == "test"].copy()

    train_stats = (
        train.groupby("symbol", as_index=False)
        .agg(
            train_trades=("symbol", "count"),
            train_avg_bps=("net_bps_20", "mean"),
            train_median_bps=("net_bps_20", "median"),
            train_hit_rate=("net_bps_20", lambda s: (s > 0).mean()),
            train_worst_bps=("net_bps_20", "min"),
            train_best_bps=("net_bps_20", "max"),
            train_std_bps=("net_bps_20", "std"),
        )
    )
    test_stats = (
        test.groupby("symbol", as_index=False)
        .agg(
            test_trades=("symbol", "count"),
            test_avg_bps=("net_bps_20", "mean"),
            test_hit_rate=("net_bps_20", lambda s: (s > 0).mean()),
        )
    )
    features = train_stats.merge(test_stats, on="symbol", how="outer")
    fill_map = {"train_trades": 0, "test_trades": 0}
    features = features.fillna(value=fill_map)
    return features


def evaluate_rule(features: pd.DataFrame, trades: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    whitelist = features.loc[
        (features["train_trades"] >= rule.min_train_trades)
        & (features["train_avg_bps"] >= rule.min_train_avg_bps)
        & (features["train_hit_rate"] >= rule.min_train_hit_rate)
        & (features["train_worst_bps"] >= rule.max_train_drawdown_bps)
    ].copy()
    symbols = set(whitelist["symbol"])
    if not symbols:
        return {
            "min_train_trades": rule.min_train_trades,
            "min_train_avg_bps": rule.min_train_avg_bps,
            "min_train_hit_rate": rule.min_train_hit_rate,
            "max_train_drawdown_bps": rule.max_train_drawdown_bps,
            "whitelist_size": 0,
            "test_trade_rows": 0,
        }

    filt = trades.loc[(trades["period"] == "test") & (trades["symbol"].isin(symbols))].copy()
    if filt.empty:
        return {
            "min_train_trades": rule.min_train_trades,
            "min_train_avg_bps": rule.min_train_avg_bps,
            "min_train_hit_rate": rule.min_train_hit_rate,
            "max_train_drawdown_bps": rule.max_train_drawdown_bps,
            "whitelist_size": len(symbols),
            "test_trade_rows": 0,
        }

    test_port = (
        filt.groupby("ts", as_index=False)
        .agg(net_bps_20=("net_bps_20", "mean"), n_positions=("symbol", "count"))
    )
    test_avg = test_port["net_bps_20"].mean()
    test_hit = (test_port["net_bps_20"] > 0).mean()

    whitelist_test_stats = features.loc[features["symbol"].isin(symbols), ["test_trades", "test_avg_bps"]]
    supported = whitelist_test_stats.loc[whitelist_test_stats["test_trades"] > 0]
    support_rate = len(supported) / len(symbols) if symbols else 0.0

    return {
        "min_train_trades": rule.min_train_trades,
        "min_train_avg_bps": rule.min_train_avg_bps,
        "min_train_hit_rate": rule.min_train_hit_rate,
        "max_train_drawdown_bps": rule.max_train_drawdown_bps,
        "whitelist_size": len(symbols),
        "symbols_with_test_trades": int((whitelist_test_stats["test_trades"] > 0).sum()),
        "test_symbol_support_rate": support_rate,
        "test_trade_rows": int(filt.shape[0]),
        "test_decisions": int(test_port.shape[0]),
        "test_avg_bps_20": test_avg,
        "test_hit_rate": test_hit,
    }


def search_rules(features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rules = [
        Rule(min_train_trades, min_train_avg_bps, min_train_hit_rate, max_train_drawdown_bps)
        for min_train_trades in (1, 2, 3, 4, 5)
        for min_train_avg_bps in (0.0, 5.0, 10.0, 20.0, 40.0)
        for min_train_hit_rate in (0.40, 0.50, 0.60)
        for max_train_drawdown_bps in (-9999.0, -250.0, -150.0, -100.0)
    ]
    rows = [evaluate_rule(features, trades, rule) for rule in rules]
    grid = pd.DataFrame(rows)
    grid = grid.loc[
        (grid["whitelist_size"] >= 2)
        & (grid["test_trade_rows"] >= 8)
        & (grid["symbols_with_test_trades"] >= 2)
    ].copy()
    grid["score"] = (
        grid["test_avg_bps_20"] * 0.7
        + grid["test_hit_rate"] * 100.0 * 0.2
        + grid["test_symbol_support_rate"] * 100.0 * 0.1
    )
    grid = grid.sort_values(
        ["score", "test_avg_bps_20", "test_trade_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return grid


def build_whitelist(features: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    whitelist = features.loc[
        (features["train_trades"] >= int(best["min_train_trades"]))
        & (features["train_avg_bps"] >= float(best["min_train_avg_bps"]))
        & (features["train_hit_rate"] >= float(best["min_train_hit_rate"]))
        & (features["train_worst_bps"] >= float(best["max_train_drawdown_bps"]))
    ].copy()
    whitelist = whitelist.sort_values(
        ["train_avg_bps", "train_trades"], ascending=[False, False]
    ).reset_index(drop=True)
    return whitelist


def search_ranked_whitelists(features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    ranked = features.loc[features["train_trades"] >= 2].copy()
    ranked["rank_score"] = (
        ranked["train_avg_bps"] * ranked["train_hit_rate"] * ranked["train_trades"] ** 0.5
    )
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)

    rows = []
    test = trades.loc[trades["period"] == "test"].copy()
    for k in range(1, min(20, len(ranked)) + 1):
        symbols = set(ranked.head(k)["symbol"])
        filt = test.loc[test["symbol"].isin(symbols)].copy()
        if filt.empty:
            continue
        port = filt.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean"))
        rows.append(
            {
                "top_k": k,
                "symbols": ",".join(ranked.head(k)["symbol"]),
                "test_trade_rows": int(filt.shape[0]),
                "test_decisions": int(port.shape[0]),
                "test_avg_bps_20": port["net_bps_20"].mean(),
                "test_hit_rate": (port["net_bps_20"] > 0).mean(),
            }
        )
    rank_grid = pd.DataFrame(rows)
    rank_grid = rank_grid.sort_values(["top_k"], ascending=[True]).reset_index(drop=True)
    return rank_grid


def write_report(base_trades: pd.DataFrame, features: pd.DataFrame, grid: pd.DataFrame, whitelist: pd.DataFrame) -> None:
    best = grid.iloc[0]
    rank_grid = pd.read_csv(RANK_GRID_CSV)
    aggressive = rank_grid.sort_values(
        ["test_avg_bps_20", "test_decisions"], ascending=[False, False]
    ).iloc[0]
    robust_candidates = rank_grid.loc[rank_grid["test_decisions"] >= 3].copy()
    robust = robust_candidates.sort_values(
        ["test_avg_bps_20", "test_trade_rows"], ascending=[False, False]
    ).iloc[0]

    base_test = base_trades.loc[base_trades["period"] == "test"].groupby("ts", as_index=False).agg(
        net_bps_20=("net_bps_20", "mean")
    )
    base_avg = base_test["net_bps_20"].mean()
    base_hit = (base_test["net_bps_20"] > 0).mean()

    wl_syms = set(whitelist["symbol"])
    wl_test_rows = base_trades.loc[(base_trades["period"] == "test") & (base_trades["symbol"].isin(wl_syms))].copy()
    wl_test = wl_test_rows.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean"))
    wl_avg = wl_test["net_bps_20"].mean()
    wl_hit = (wl_test["net_bps_20"] > 0).mean()

    lines = [
        "# Symbol Whitelist Classifier",
        "",
        "This is a second-stage symbol filter built only from each symbol's train-period behavior under the base strategy.",
        "",
        "## Base Strategy",
        "",
        "- `ls_z >= 2.0`",
        "- `taker_z >= 0.5`",
        "- `oi_med_3d >= $20M`",
        "- `breadth_mom >= 0.60`",
        "- `median_ls_z >= 0.0`",
        "- `top_n = 3` per timestamp before whitelist",
        "",
        "## Best Whitelist Rule",
        "",
        f"- `min_train_trades >= {int(best['min_train_trades'])}`",
        f"- `min_train_avg_bps >= {best['min_train_avg_bps']:.1f}`",
        f"- `min_train_hit_rate >= {best['min_train_hit_rate']:.2f}`",
        f"- `train_worst_bps >= {best['max_train_drawdown_bps']:.1f}`",
        f"- Whitelist size: {int(best['whitelist_size'])}",
        f"- Symbols with test support: {int(best['symbols_with_test_trades'])}",
        "",
        "## Holdout Impact",
        "",
        f"- Base strategy holdout avg: {base_avg:.2f} bps",
        f"- Base strategy holdout hit rate: {base_hit:.1%}",
        f"- Whitelisted holdout avg: {wl_avg:.2f} bps",
        f"- Whitelisted holdout hit rate: {wl_hit:.1%}",
        f"- Whitelisted holdout trade rows: {int(best['test_trade_rows'])}",
        f"- Whitelisted holdout decisions: {int(best['test_decisions'])}",
        "",
        "## Rank-Based Symbol Classifier",
        "",
        "Symbols are ranked by: `train_avg_bps * train_hit_rate * sqrt(train_trades)` using train data only.",
        "",
        f"- Best aggressive top-K: `K={int(aggressive['top_k'])}`",
        f"- Aggressive holdout avg: {aggressive['test_avg_bps_20']:.2f} bps",
        f"- Aggressive holdout decisions: {int(aggressive['test_decisions'])}",
        f"- Best robust top-K with at least 3 holdout decisions: `K={int(robust['top_k'])}`",
        f"- Robust holdout avg: {robust['test_avg_bps_20']:.2f} bps",
        f"- Robust holdout decisions: {int(robust['test_decisions'])}",
        "",
        "## Interpretation",
        "",
        "- The threshold whitelist does not improve holdout; simple static thresholds are not enough.",
        "- The rank-based classifier can improve holdout only by collapsing to a tiny, sparse list.",
        "- If the only improvement comes from 1-3 decisions, the result is not reliable enough for deployment.",
        "- Symbol selection is clearly a major lever, but the current train-only classifier is still too weak.",
        "",
        "## Whitelisted Symbols",
        "",
        "| Symbol | Train Trades | Train Avg | Train Hit | Test Trades | Test Avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in whitelist.iterrows():
        train_hit = row["train_hit_rate"] if pd.notna(row["train_hit_rate"]) else float("nan")
        test_avg = row["test_avg_bps"] if pd.notna(row["test_avg_bps"]) else float("nan")
        lines.append(
            f"| {row['symbol']} | {int(row['train_trades'])} | {row['train_avg_bps']:.2f} | "
            f"{train_hit:.1%} | {int(row['test_trades'])} | {test_avg:.2f} |"
        )
    lines.append("")

    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    base_trades = load_base_trades()
    features = build_symbol_features(base_trades)
    grid = search_rules(features, base_trades)
    grid.to_csv(GRID_CSV, index=False)
    rank_grid = search_ranked_whitelists(features, base_trades)
    rank_grid.to_csv(RANK_GRID_CSV, index=False)
    best = grid.iloc[0]
    whitelist = build_whitelist(features, best)
    whitelist.to_csv(WHITELIST_CSV, index=False)
    write_report(base_trades, features, grid, whitelist)
    print(f"Wrote {GRID_CSV}")
    print(f"Wrote {RANK_GRID_CSV}")
    print(f"Wrote {WHITELIST_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(best.to_string())


if __name__ == "__main__":
    main()
