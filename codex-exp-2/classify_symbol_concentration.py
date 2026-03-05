from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
CLASS_CSV = OUT_DIR / "symbol_classification.csv"
REPORT_MD = OUT_DIR / "FINDINGS_symbol_classification.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
FEE = 0.002


def load_selected_trades() -> pd.DataFrame:
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    mask = (
        (samples["oi_med_3d"] >= 20_000_000)
        & (samples["breadth_mom"] >= 0.60)
        & (samples["median_ls_z"] >= 0.0)
        & (samples["ls_z"] >= 2.0)
        & (samples["taker_z"] >= 0.5)
        & (samples["mom_4h"] > 0)
    )
    selected = samples.loc[mask].copy()
    selected = (
        selected.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    selected["net_bps_20"] = (selected["ret_4h"] - FEE) * 10000.0
    return selected


def classify_symbol(symbol: str, group: pd.DataFrame) -> dict[str, object]:
    train = group.loc[group["ts"] < TRAIN_CUTOFF, "net_bps_20"]
    test = group.loc[group["ts"] >= TRAIN_CUTOFF, "net_bps_20"]
    train_avg = train.mean() if not train.empty else float("nan")
    test_avg = test.mean() if not test.empty else float("nan")
    total_bps = group["net_bps_20"].sum()

    if len(group) >= 4 and len(train) >= 2 and len(test) >= 1 and train_avg > 0 and test_avg > 0:
        label = "robust_positive"
    elif (pd.notna(train_avg) and train_avg > 0) or (pd.notna(test_avg) and test_avg > 0):
        label = "mixed_or_sparse_positive"
    else:
        label = "negative"

    return {
        "symbol": symbol,
        "trades": len(group),
        "train_trades": int(train.shape[0]),
        "test_trades": int(test.shape[0]),
        "train_avg_bps_20": train_avg,
        "test_avg_bps_20": test_avg,
        "all_avg_bps_20": group["net_bps_20"].mean(),
        "total_bps_20": total_bps,
        "classification": label,
    }


def build_classification(selected: pd.DataFrame) -> pd.DataFrame:
    rows = [classify_symbol(symbol, group) for symbol, group in selected.groupby("symbol")]
    classified = pd.DataFrame(rows)
    classified = classified.sort_values(
        ["classification", "total_bps_20"], ascending=[True, False]
    ).reset_index(drop=True)
    return classified


def write_report(selected: pd.DataFrame, classified: pd.DataFrame) -> None:
    ordered = classified.sort_values("total_bps_20", ascending=False).reset_index(drop=True)
    net_total = ordered["total_bps_20"].sum()
    positive_total = ordered.loc[ordered["total_bps_20"] > 0, "total_bps_20"].sum()
    negative_total = -ordered.loc[ordered["total_bps_20"] < 0, "total_bps_20"].sum()

    pos_weights = ordered.loc[ordered["total_bps_20"] > 0, "total_bps_20"] / positive_total
    hhi = float((pos_weights**2).sum()) if positive_total > 0 else float("nan")

    def share(n: int) -> tuple[float, float]:
        top = ordered.head(n)["total_bps_20"].sum()
        share_net = top / net_total if net_total else float("nan")
        share_pos = top / positive_total if positive_total else float("nan")
        return share_net, share_pos

    top3_net, top3_pos = share(3)
    top5_net, top5_pos = share(5)

    robust = classified.loc[classified["classification"] == "robust_positive"].copy()
    mixed = classified.loc[classified["classification"] == "mixed_or_sparse_positive"].copy()
    negative = classified.loc[classified["classification"] == "negative"].copy()

    lines = [
        "# Symbol Concentration And Classification",
        "",
        "This report evaluates whether the currently best fee-aware strategy is broad or concentrated by symbol.",
        "",
        "## Configuration Tested",
        "",
        "- `ls_z >= 2.0`",
        "- `taker_z >= 0.5`",
        "- `oi_med_3d >= $20M`",
        "- `breadth_mom >= 0.60`",
        "- `median_ls_z >= 0.0`",
        "- `top_n = 3` per timestamp",
        "- fee assumption: `20 bps` round-trip all taker",
        "",
        "## Aggregate",
        "",
        f"- Selected trade rows: {len(selected)}",
        f"- Unique symbols selected: {selected['symbol'].nunique()}",
        f"- Net symbol contribution total: {net_total:.2f} bps",
        f"- Positive contribution pool: {positive_total:.2f} bps",
        f"- Negative contribution pool: {negative_total:.2f} bps",
        f"- Top 3 symbols = {top3_net:.1%} of final net and {top3_pos:.1%} of gross positive contribution",
        f"- Top 5 symbols = {top5_net:.1%} of final net and {top5_pos:.1%} of gross positive contribution",
        f"- Positive-contributor HHI: {hhi:.4f}",
        "",
        "## Interpretation",
        "",
        "- Concentration is real. A few winners contribute more than the final net because many other symbols give back gains.",
        "- This means the current edge is not broad enough to trust blindly across all triggered names.",
        "- Symbol selection quality matters almost as much as the base signal.",
        "",
        "## Class Counts",
        "",
        f"- `robust_positive`: {len(robust)}",
        f"- `mixed_or_sparse_positive`: {len(mixed)}",
        f"- `negative`: {len(negative)}",
        "",
    ]

    if not robust.empty:
        lines.extend(
            [
                "## Robust Positive Symbols",
                "",
                "These had at least 4 total trades, at least 2 train trades, at least 1 test trade, and were positive in both train and test.",
                "",
                "| Symbol | Trades | Train | Test | Train Avg | Test Avg | Total |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in robust.sort_values("total_bps_20", ascending=False).iterrows():
            lines.append(
                f"| {row['symbol']} | {int(row['trades'])} | {int(row['train_trades'])} | "
                f"{int(row['test_trades'])} | {row['train_avg_bps_20']:.2f} | "
                f"{row['test_avg_bps_20']:.2f} | {row['total_bps_20']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Top Positive Contributors",
            "",
            "| Symbol | Trades | Avg | Total | Class |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for _, row in ordered.head(10).iterrows():
        lines.append(
            f"| {row['symbol']} | {int(row['trades'])} | {row['all_avg_bps_20']:.2f} | "
            f"{row['total_bps_20']:.2f} | {row['classification']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Worst Contributors",
            "",
            "| Symbol | Trades | Avg | Total | Class |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for _, row in ordered.tail(10).sort_values("total_bps_20").iterrows():
        lines.append(
            f"| {row['symbol']} | {int(row['trades'])} | {row['all_avg_bps_20']:.2f} | "
            f"{row['total_bps_20']:.2f} | {row['classification']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Bottom Line",
            "",
            "- The strategy is promising but concentrated.",
            "- Only a very small subset currently looks robust across both train and holdout.",
            "- A production version should probably add a second-stage symbol classifier or whitelist rather than trade every triggered name.",
        ]
    )

    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    selected = load_selected_trades()
    classified = build_classification(selected)
    classified.to_csv(CLASS_CSV, index=False)
    write_report(selected, classified)
    print(f"Wrote {CLASS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(classified['classification'].value_counts().to_string())


if __name__ == "__main__":
    main()
