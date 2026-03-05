from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
CANDIDATES_CSV = OUT_DIR / "candidate_quality_dataset.csv"
TRADES_CSV = OUT_DIR / "high_conviction_trades.csv"
PORTFOLIO_CSV = OUT_DIR / "high_conviction_portfolio.csv"
SUMMARY_MD = OUT_DIR / "HIGH_CONVICTION_FINDINGS.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
TOP_N = 3


def load_candidates() -> pd.DataFrame:
    return pd.read_csv(CANDIDATES_CSV, parse_dates=["ts"])


def select_high_conviction(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    train = df.loc[df["ts"] <= TRAIN_END].copy()
    score_cut = float(train["execution_adjusted_score"].quantile(0.60))
    ls_cut = float(max(2.5, train["ls_z"].quantile(0.60)))
    breadth_cut = 0.65

    filtered = df.loc[
        (df["execution_adjusted_score"] >= score_cut)
        & (df["ls_z"] >= ls_cut)
        & (df["breadth_mom"] >= breadth_cut)
    ].copy()
    selected = (
        filtered.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    cfg = {"score_cut": score_cut, "ls_cut": ls_cut, "breadth_cut": breadth_cut}
    return selected, cfg


def build_portfolio(selected: pd.DataFrame) -> pd.DataFrame:
    return (
        selected.groupby("ts", as_index=False)
        .agg(
            n_positions=("symbol", "count"),
            port_ret=("net_ret_after_costs", "mean"),
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )


def summarize_slice(portfolio: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict[str, float | int]:
    part = portfolio.copy()
    if start is not None:
        part = part.loc[part["ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        part = part.loc[part["ts"] <= pd.Timestamp(end, tz="UTC")]
    if part.empty:
        return {"timestamps": 0, "avg_bps": float("nan"), "win_rate": float("nan")}
    return {
        "timestamps": int(part.shape[0]),
        "avg_bps": float(part["port_ret"].mean() * 10000.0),
        "win_rate": float((part["port_ret"] > 0).mean()),
    }


def monthly_breakdown(portfolio: pd.DataFrame) -> pd.DataFrame:
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    test["month"] = test["ts"].dt.strftime("%Y-%m")
    return (
        test.groupby("month", as_index=False)
        .agg(
            timestamps=("port_ret", "count"),
            avg_bps=("port_ret", lambda s: s.mean() * 10000.0),
            win_rate=("port_ret", lambda s: (s > 0).mean()),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def write_summary(selected: pd.DataFrame, portfolio: pd.DataFrame, cfg: dict[str, float]) -> None:
    train_stats = summarize_slice(portfolio, end="2025-12-31 23:59:59")
    test_stats = summarize_slice(portfolio, start="2026-01-01 00:00:00")
    jan_stats = summarize_slice(portfolio, start="2026-01-01 00:00:00", end="2026-01-31 23:59:59")
    feb_stats = summarize_slice(portfolio, start="2026-02-01 00:00:00", end="2026-02-28 23:59:59")
    monthly = monthly_breakdown(portfolio)

    lines = [
        "# High-Conviction Mode",
        "",
        "This is the strict, lower-frequency version of the current signal.",
        "",
        "## Rule",
        "",
        f"- Require `execution_adjusted_score >= {cfg['score_cut']:.3f}`",
        f"- Require `ls_z >= {cfg['ls_cut']:.3f}`",
        f"- Require `breadth_mom >= {cfg['breadth_cut']:.2f}`",
        f"- Rank remaining names by `execution_adjusted_score` and take up to `{TOP_N}` per timestamp",
        "",
        "## Aggregate",
        "",
        f"- Selected symbol rows: {len(selected)}",
        f"- Selected timestamps: {len(portfolio)}",
        f"- Unique traded symbols: {selected['symbol'].nunique()}",
        "",
        "## Train/Test",
        "",
        f"- Train: {train_stats['timestamps']} timestamps, {train_stats['avg_bps']:.2f} bps, win rate {train_stats['win_rate']:.1%}",
        f"- Test: {test_stats['timestamps']} timestamps, {test_stats['avg_bps']:.2f} bps, win rate {test_stats['win_rate']:.1%}",
        "",
        "## Tougher OOS Slices",
        "",
        f"- 2026-01: {jan_stats['timestamps']} timestamps, {jan_stats['avg_bps']:.2f} bps, win rate {jan_stats['win_rate']:.1%}",
        f"- 2026-02: {feb_stats['timestamps']} timestamps, {feb_stats['avg_bps']:.2f} bps, win rate {feb_stats['win_rate']:.1%}",
        "",
    ]

    if not monthly.empty:
        lines.extend(
            [
                "## Monthly Breakdown",
                "",
                "| Month | Timestamps | Avg bps | Win Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in monthly.iterrows():
            lines.append(
                f"| {row['month']} | {int(row['timestamps'])} | {row['avg_bps']:.2f} | {row['win_rate']:.1%} |"
            )
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    candidates = load_candidates()
    selected, cfg = select_high_conviction(candidates)
    portfolio = build_portfolio(selected)
    selected.to_csv(TRADES_CSV, index=False)
    portfolio.to_csv(PORTFOLIO_CSV, index=False)
    write_summary(selected, portfolio, cfg)
    print(f"Wrote {TRADES_CSV}")
    print(f"Wrote {PORTFOLIO_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(portfolio.to_string(index=False))


if __name__ == "__main__":
    main()
