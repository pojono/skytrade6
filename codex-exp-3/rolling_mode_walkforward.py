from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
CANDIDATES_CSV = OUT_DIR / "candidate_quality_dataset.csv"

MONTHLY_CSV = OUT_DIR / "rolling_mode_monthly.csv"
SUMMARY_CSV = OUT_DIR / "rolling_mode_summary.csv"
SUMMARY_MD = OUT_DIR / "ROLLING_MODE_FINDINGS.md"

TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
TOP_N = 3


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES_CSV, parse_dates=["ts"])
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)


def high_conviction_thresholds(train: pd.DataFrame) -> dict[str, float]:
    return {
        "score_cut": float(train["execution_adjusted_score"].quantile(0.60)),
        "ls_cut": float(max(2.5, train["ls_z"].quantile(0.60))),
        "breadth_cut": 0.65,
        "hybrid_switch_breadth": 0.75,
    }


def select_default_soft(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_high_conviction(frame: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    filtered = frame.loc[
        (frame["execution_adjusted_score"] >= cfg["score_cut"])
        & (frame["ls_z"] >= cfg["ls_cut"])
        & (frame["breadth_mom"] >= cfg["breadth_cut"])
    ].copy()
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_hybrid_regime(frame: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    strong = frame["breadth_mom"] >= cfg["hybrid_switch_breadth"]
    hc = frame.loc[
        strong
        & (frame["execution_adjusted_score"] >= cfg["score_cut"])
        & (frame["ls_z"] >= cfg["ls_cut"])
        & (frame["breadth_mom"] >= cfg["breadth_cut"])
    ].copy()
    soft = frame.loc[~strong].copy()
    combined = pd.concat([hc, soft], ignore_index=True)
    if combined.empty:
        return combined
    return (
        combined.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def summarize_selection(selected: pd.DataFrame) -> dict[str, float | int]:
    if selected.empty:
        return {
            "rows": 0,
            "timestamps": 0,
            "avg_bps": float("nan"),
            "win_rate": float("nan"),
            "symbol_count": 0,
        }
    port = (
        selected.groupby("ts", as_index=False)
        .agg(port_ret=("net_ret_after_costs", "mean"))
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return {
        "rows": int(selected.shape[0]),
        "timestamps": int(port.shape[0]),
        "avg_bps": float(port["port_ret"].mean() * 10000.0),
        "win_rate": float((port["port_ret"] > 0).mean()),
        "symbol_count": int(selected["symbol"].nunique()),
    }


def run_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = df.loc[df["ts"] >= TEST_START].copy()
    months = (
        test_df["ts"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    monthly_rows = []
    for month in months:
        month_start = pd.Timestamp(f"{month}-01 00:00:00", tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)

        train = df.loc[df["ts"] < month_start].copy()
        test = df.loc[(df["ts"] >= month_start) & (df["ts"] <= month_end)].copy()
        if train.empty or test.empty:
            continue

        cfg = high_conviction_thresholds(train)
        selections = {
            "default_soft": select_default_soft(test),
            "high_conviction": select_high_conviction(test, cfg),
            "hybrid_regime": select_hybrid_regime(test, cfg),
        }
        for mode, selected in selections.items():
            s = summarize_selection(selected)
            monthly_rows.append(
                {
                    "month": month,
                    "mode": mode,
                    "rows": s["rows"],
                    "timestamps": s["timestamps"],
                    "avg_bps": s["avg_bps"],
                    "win_rate": s["win_rate"],
                    "symbol_count": s["symbol_count"],
                    "score_cut": cfg["score_cut"],
                    "ls_cut": cfg["ls_cut"],
                    "breadth_cut": cfg["breadth_cut"],
                    "hybrid_switch_breadth": cfg["hybrid_switch_breadth"],
                }
            )

    monthly = pd.DataFrame(monthly_rows)
    summary = (
        monthly.groupby("mode", as_index=False)
        .agg(
            months=("month", "nunique"),
            total_timestamps=("timestamps", "sum"),
            avg_monthly_bps=("avg_bps", "mean"),
            median_monthly_bps=("avg_bps", "median"),
            std_monthly_bps=("avg_bps", "std"),
            min_monthly_bps=("avg_bps", "min"),
            positive_months=("avg_bps", lambda s: int((s > 0).sum())),
            avg_monthly_win_rate=("win_rate", "mean"),
        )
        .sort_values("avg_monthly_bps", ascending=False)
        .reset_index(drop=True)
    )
    return monthly, summary


def write_summary(monthly: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Rolling Walk-Forward Mode Comparison",
        "",
        "Monthly walk-forward setup:",
        "",
        "- For each test month, thresholds are fit only on data before that month.",
        "- Comparison modes: `default_soft`, `high_conviction`, `hybrid_regime`.",
        "- Metric is `net_ret_after_costs` from candidate dataset (already fee + entry-drag adjusted), aggregated as equal-weight portfolio per timestamp.",
        "",
        "## Summary",
        "",
        "| Mode | Months | Timestamps | Avg Monthly bps | Median bps | Std bps | Min bps | Positive Months | Avg Win Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['mode']} | {int(row['months'])} | {int(row['total_timestamps'])} | "
            f"{row['avg_monthly_bps']:.2f} | {row['median_monthly_bps']:.2f} | "
            f"{(row['std_monthly_bps'] if pd.notna(row['std_monthly_bps']) else 0.0):.2f} | "
            f"{row['min_monthly_bps']:.2f} | {int(row['positive_months'])} | {row['avg_monthly_win_rate']:.1%} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Monthly Detail",
            "",
            "| Month | Mode | Timestamps | Avg bps | Win Rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in monthly.sort_values(["month", "mode"]).iterrows():
        lines.append(
            f"| {row['month']} | {row['mode']} | {int(row['timestamps'])} | {row['avg_bps']:.2f} | {row['win_rate']:.1%} |"
        )
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    df = load_candidates()
    monthly, summary = run_walkforward(df)
    monthly.to_csv(MONTHLY_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_summary(monthly, summary)
    print(f"Wrote {MONTHLY_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
