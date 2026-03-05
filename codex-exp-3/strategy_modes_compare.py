from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
CANDIDATES_CSV = OUT_DIR / "candidate_quality_dataset.csv"

MODE_SUMMARY_CSV = OUT_DIR / "strategy_mode_comparison.csv"
MODE_SUMMARY_MD = OUT_DIR / "STRATEGY_MODE_FINDINGS.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
TOP_N = 3


def load_candidates() -> pd.DataFrame:
    return pd.read_csv(CANDIDATES_CSV, parse_dates=["ts"])


def select_default_soft(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def high_conviction_thresholds(train: pd.DataFrame) -> dict[str, float]:
    return {
        "score_cut": float(train["execution_adjusted_score"].quantile(0.60)),
        "ls_cut": float(max(2.5, train["ls_z"].quantile(0.60))),
        "breadth_cut": 0.65,
    }


def select_high_conviction(df: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    filtered = df.loc[
        (df["execution_adjusted_score"] >= cfg["score_cut"])
        & (df["ls_z"] >= cfg["ls_cut"])
        & (df["breadth_mom"] >= cfg["breadth_cut"])
    ].copy()
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_hybrid_regime(df: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    # Use the high-conviction filter only in stronger broad risk-on conditions.
    strong_regime = df["breadth_mom"] >= 0.75
    hc = df.loc[
        strong_regime
        & (df["execution_adjusted_score"] >= cfg["score_cut"])
        & (df["ls_z"] >= cfg["ls_cut"])
        & (df["breadth_mom"] >= cfg["breadth_cut"])
    ].copy()
    hc["mode_used"] = "high_conviction"

    soft = df.loc[~strong_regime].copy()
    soft["mode_used"] = "default_soft"

    combined = pd.concat([hc, soft], ignore_index=True)
    if combined.empty:
        return combined
    return (
        combined.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def summarize_mode(name: str, selected: pd.DataFrame) -> dict[str, float | int | str]:
    test = selected.loc[selected["ts"] >= TEST_START].copy()
    if test.empty:
        return {
            "mode": name,
            "test_rows": 0,
            "test_timestamps": 0,
            "test_avg_bps": float("nan"),
            "test_win_rate": float("nan"),
            "test_symbol_count": 0,
            "jan_bps": float("nan"),
            "feb_bps": float("nan"),
        }
    port = (
        test.groupby("ts", as_index=False)
        .agg(port_ret=("net_ret_after_costs", "mean"), n_positions=("symbol", "count"))
        .sort_values("ts")
        .reset_index(drop=True)
    )
    jan = port.loc[
        (port["ts"] >= pd.Timestamp("2026-01-01 00:00:00", tz="UTC"))
        & (port["ts"] <= pd.Timestamp("2026-01-31 23:59:59", tz="UTC"))
    ]
    feb = port.loc[
        (port["ts"] >= pd.Timestamp("2026-02-01 00:00:00", tz="UTC"))
        & (port["ts"] <= pd.Timestamp("2026-02-28 23:59:59", tz="UTC"))
    ]
    return {
        "mode": name,
        "test_rows": int(test.shape[0]),
        "test_timestamps": int(port.shape[0]),
        "test_avg_bps": float(port["port_ret"].mean() * 10000.0),
        "test_win_rate": float((port["port_ret"] > 0).mean()),
        "test_symbol_count": int(test["symbol"].nunique()),
        "jan_bps": float(jan["port_ret"].mean() * 10000.0) if not jan.empty else float("nan"),
        "feb_bps": float(feb["port_ret"].mean() * 10000.0) if not feb.empty else float("nan"),
    }


def write_summary(results: pd.DataFrame, cfg: dict[str, float]) -> None:
    lines = [
        "# Strategy Mode Comparison",
        "",
        "This compares three live-usable variants built on the same candidate set.",
        "",
        "Modes:",
        "",
        "- `default_soft`: current soft execution-penalty ranking",
        "- `high_conviction`: stricter thresholds before ranking",
        "- `hybrid_regime`: use high-conviction only when breadth is very strong (`breadth_mom >= 0.75`), otherwise use default_soft",
        "",
        "High-conviction thresholds:",
        "",
        f"- `execution_adjusted_score >= {cfg['score_cut']:.3f}`",
        f"- `ls_z >= {cfg['ls_cut']:.3f}`",
        f"- `breadth_mom >= {cfg['breadth_cut']:.2f}`",
        "",
        "## Test Results",
        "",
        "| Mode | Rows | Timestamps | Avg bps | Win Rate | Symbols | 2026-01 | 2026-02 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['mode']} | {int(row['test_rows'])} | {int(row['test_timestamps'])} | "
            f"{row['test_avg_bps']:.2f} | {row['test_win_rate']:.1%} | {int(row['test_symbol_count'])} | "
            f"{row['jan_bps']:.2f} | {row['feb_bps']:.2f} |"
        )
    lines.append("")
    MODE_SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    candidates = load_candidates()
    train = candidates.loc[candidates["ts"] <= TRAIN_END].copy()
    cfg = high_conviction_thresholds(train)

    selections = {
        "default_soft": select_default_soft(candidates),
        "high_conviction": select_high_conviction(candidates, cfg),
        "hybrid_regime": select_hybrid_regime(candidates, cfg),
    }

    results = pd.DataFrame([summarize_mode(name, frame) for name, frame in selections.items()])
    results = results.sort_values("test_avg_bps", ascending=False).reset_index(drop=True)
    results.to_csv(MODE_SUMMARY_CSV, index=False)
    write_summary(results, cfg)
    print(f"Wrote {MODE_SUMMARY_CSV}")
    print(f"Wrote {MODE_SUMMARY_MD}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
