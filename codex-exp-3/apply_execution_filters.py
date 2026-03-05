from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

SYMBOL_TRADES_CSV = OUT_DIR / "revalidated_exp2_symbol_trades.csv"
ENTRY_SUMMARY_CSV = OUT_DIR / "trade_entry_feasibility_summary_full.csv"

ADJUSTED_TRADES_CSV = OUT_DIR / "execution_adjusted_symbol_trades.csv"
COMPARISON_CSV = OUT_DIR / "execution_adjusted_comparison.csv"
SUMMARY_MD = OUT_DIR / "EXECUTION_FILTER_FINDINGS.md"

TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
FEE_ALL_MAKER = 0.0008

# Exclude names that either failed the simple Bybit maker-fill proxy
# or have severe positive first-minute chase risk.
DRAG_BLACKLIST_THRESHOLD_BPS = 8.0


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(SYMBOL_TRADES_CSV, parse_dates=["ts"])
    entry = pd.read_csv(ENTRY_SUMMARY_CSV)
    return trades, entry


def build_symbol_penalties(entry: pd.DataFrame) -> pd.DataFrame:
    penalties = entry.copy()
    penalties["bn_positive_drag_bps"] = penalties["bn_vwap_60s_bps"].clip(lower=0.0)
    penalties["bb_positive_drag_bps"] = penalties["bb_vwap_60s_bps"].clip(lower=0.0)
    penalties["avg_positive_drag_bps"] = (
        penalties["bn_positive_drag_bps"] + penalties["bb_positive_drag_bps"]
    ) / 2.0
    penalties["blacklist"] = (
        (penalties["bb_maker_fill_rate"] < 1.0)
        | (penalties["avg_positive_drag_bps"] >= DRAG_BLACKLIST_THRESHOLD_BPS)
    )
    return penalties[
        [
            "symbol",
            "signals",
            "bb_maker_fill_rate",
            "bn_vwap_60s_bps",
            "bb_vwap_60s_bps",
            "avg_positive_drag_bps",
            "blacklist",
        ]
    ].copy()


def apply_penalties(trades: pd.DataFrame, penalties: pd.DataFrame) -> pd.DataFrame:
    adjusted = trades.merge(penalties, on="symbol", how="left")
    adjusted["avg_positive_drag_bps"] = adjusted["avg_positive_drag_bps"].fillna(0.0)
    adjusted["blacklist"] = adjusted["blacklist"].eq(True)
    adjusted["execution_drag"] = adjusted["avg_positive_drag_bps"] / 10000.0
    adjusted["avg_ret_after_entry_drag"] = adjusted["avg_ret_4h_funding_adj"] - adjusted["execution_drag"]
    return adjusted


def _portfolio_from_symbol_trades(
    trades: pd.DataFrame,
    ret_col: str,
    use_filtered: bool = False,
) -> pd.DataFrame:
    working = trades.loc[~trades["blacklist"]].copy() if use_filtered else trades.copy()
    if working.empty:
        return pd.DataFrame(columns=["ts", "n_positions", "raw_ret"])
    return (
        working.groupby("ts", as_index=False)
        .agg(n_positions=("symbol", "count"), raw_ret=(ret_col, "mean"))
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _stats(portfolio: pd.DataFrame) -> dict[str, float | int]:
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    if test.empty:
        return {"test_timestamps": 0, "test_avg_bps": float("nan"), "test_win_rate": float("nan")}
    net = test["raw_ret"] - FEE_ALL_MAKER
    return {
        "test_timestamps": int(test.shape[0]),
        "test_avg_bps": net.mean() * 10000.0,
        "test_win_rate": (net > 0).mean(),
    }


def build_comparison(adjusted: pd.DataFrame) -> pd.DataFrame:
    variants = []

    base_port = _portfolio_from_symbol_trades(adjusted, "avg_ret_4h_funding_adj", use_filtered=False)
    base_stats = _stats(base_port)
    variants.append({"variant": "baseline_funding_adj", **base_stats})

    drag_port = _portfolio_from_symbol_trades(adjusted, "avg_ret_after_entry_drag", use_filtered=False)
    drag_stats = _stats(drag_port)
    variants.append({"variant": "entry_drag_applied", **drag_stats})

    filt_base_port = _portfolio_from_symbol_trades(adjusted, "avg_ret_4h_funding_adj", use_filtered=True)
    filt_base_stats = _stats(filt_base_port)
    variants.append({"variant": "blacklist_only", **filt_base_stats})

    filt_drag_port = _portfolio_from_symbol_trades(adjusted, "avg_ret_after_entry_drag", use_filtered=True)
    filt_drag_stats = _stats(filt_drag_port)
    variants.append({"variant": "blacklist_plus_drag", **filt_drag_stats})

    return pd.DataFrame(variants)


def write_summary(adjusted: pd.DataFrame, comparison: pd.DataFrame) -> None:
    blacklisted = (
        adjusted.loc[adjusted["blacklist"], ["symbol", "avg_positive_drag_bps", "bb_maker_fill_rate"]]
        .drop_duplicates()
        .sort_values(["avg_positive_drag_bps", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )
    base = comparison.loc[comparison["variant"] == "baseline_funding_adj"].iloc[0]
    drag = comparison.loc[comparison["variant"] == "entry_drag_applied"].iloc[0]
    filt = comparison.loc[comparison["variant"] == "blacklist_only"].iloc[0]
    filt_drag = comparison.loc[comparison["variant"] == "blacklist_plus_drag"].iloc[0]

    lines = [
        "# Execution Filter Findings",
        "",
        "This note folds the observed symbol-level 60-second entry drift back into the strategy and compares a filtered basket.",
        "",
        "## Rule",
        "",
        f"- Per-symbol entry drag = average of positive 60-second VWAP drift on Binance and Bybit.",
        f"- Blacklist any symbol with Bybit maker-fill rate < 100% or average positive drag >= {DRAG_BLACKLIST_THRESHOLD_BPS:.1f} bps.",
        "",
        "## Test Comparison (after 8 bps maker fees)",
        "",
        f"- Baseline funding-adjusted: {base['test_avg_bps']:.2f} bps on {int(base['test_timestamps'])} timestamps, win rate {base['test_win_rate']:.1%}",
        f"- With entry drag applied: {drag['test_avg_bps']:.2f} bps on {int(drag['test_timestamps'])} timestamps, win rate {drag['test_win_rate']:.1%}",
        f"- Blacklist only: {filt['test_avg_bps']:.2f} bps on {int(filt['test_timestamps'])} timestamps, win rate {filt['test_win_rate']:.1%}",
        f"- Blacklist plus drag: {filt_drag['test_avg_bps']:.2f} bps on {int(filt_drag['test_timestamps'])} timestamps, win rate {filt_drag['test_win_rate']:.1%}",
        "",
        "## Blacklisted Symbols",
        "",
        "| Symbol | Avg Positive Drag | Bybit Fill Rate |",
        "|---|---:|---:|",
    ]

    for _, row in blacklisted.iterrows():
        lines.append(
            f"| {row['symbol']} | {row['avg_positive_drag_bps']:.2f} bps | {row['bb_maker_fill_rate']:.0%} |"
        )
    lines.append("")
    lines.append("Files:")
    lines.append("")
    lines.append("- `execution_adjusted_symbol_trades.csv`")
    lines.append("- `execution_adjusted_comparison.csv`")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    trades, entry = load_inputs()
    penalties = build_symbol_penalties(entry)
    adjusted = apply_penalties(trades, penalties)
    comparison = build_comparison(adjusted)
    adjusted.to_csv(ADJUSTED_TRADES_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    write_summary(adjusted, comparison)
    print(f"Wrote {ADJUSTED_TRADES_CSV}")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
