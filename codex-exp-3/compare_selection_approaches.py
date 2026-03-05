from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

BINANCE_SAMPLES_CSV = OUT_DIR / "rebuilt_binance_samples_4h.csv"
ENTRY_SUMMARY_CSV = OUT_DIR / "trade_entry_feasibility_summary_full.csv"

CANDIDATES_CSV = OUT_DIR / "candidate_quality_dataset.csv"
COMPARISON_CSV = OUT_DIR / "selection_approach_comparison.csv"
SUMMARY_MD = OUT_DIR / "SELECTION_APPROACH_FINDINGS.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
HOLD_DELTA = pd.Timedelta(hours=4)
FEE_ALL_MAKER = 0.0008
TOP_N = 3


def load_bybit_close(symbol: str, cache: dict[str, pd.Series]) -> pd.Series:
    if symbol in cache:
        return cache[symbol]
    files = sorted((ROOT / "datalake" / "bybit" / symbol).glob("*_kline_1m.csv"))
    parts = []
    for path in files:
        parts.append(pd.read_csv(path, usecols=["startTime", "close"]))
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["startTime"], unit="ms", utc=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    series = (
        out[["ts", "close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")["close"]
        .resample("5min", label="right", closed="right")
        .last()
    )
    cache[symbol] = series
    return series


def load_bybit_funding(symbol: str, cache: dict[str, pd.Series]) -> pd.Series:
    if symbol in cache:
        return cache[symbol]
    files = sorted((ROOT / "datalake" / "bybit" / symbol).glob("*_funding_rate.csv"))
    if not files:
        cache[symbol] = pd.Series(dtype="float64")
        return cache[symbol]
    parts = []
    for path in files:
        parts.append(pd.read_csv(path, usecols=["timestamp", "fundingRate"]))
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["fundingRate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    series = (
        out[["ts", "fundingRate"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")["fundingRate"]
    )
    cache[symbol] = series
    return series


def build_candidates() -> pd.DataFrame:
    samples = pd.read_csv(BINANCE_SAMPLES_CSV, parse_dates=["ts"])
    base = samples.loc[
        (samples["oi_med_3d"] >= 20_000_000.0)
        & (samples["breadth_mom"] >= 0.60)
        & (samples["median_ls_z"] >= 0.0)
        & (samples["ls_z"] >= 2.0)
        & (samples["taker_z"] >= 0.5)
        & (samples["mom_4h"] > 0)
    ].copy()

    entry = pd.read_csv(ENTRY_SUMMARY_CSV)
    entry["bn_positive_drag_bps"] = entry["bn_vwap_60s_bps"].clip(lower=0.0)
    entry["bb_positive_drag_bps"] = entry["bb_vwap_60s_bps"].clip(lower=0.0)
    entry["avg_positive_drag_bps"] = (
        entry["bn_positive_drag_bps"] + entry["bb_positive_drag_bps"]
    ) / 2.0
    entry["score_penalty"] = (
        entry["avg_positive_drag_bps"] + (1.0 - entry["bb_maker_fill_rate"].clip(lower=0.0, upper=1.0)) * 8.0
    ) / 4.0
    base = base.merge(entry[["symbol", "avg_positive_drag_bps", "score_penalty"]], on="symbol", how="left")
    base["avg_positive_drag_bps"] = base["avg_positive_drag_bps"].fillna(0.0)
    base["score_penalty"] = base["score_penalty"].fillna(0.0)
    base["execution_adjusted_score"] = base["score_abs"] - base["score_penalty"]

    close_cache: dict[str, pd.Series] = {}
    funding_cache: dict[str, pd.Series] = {}
    parts = []
    for symbol, frame in base.groupby("symbol"):
        close = load_bybit_close(symbol, close_cache)
        funding = load_bybit_funding(symbol, funding_cache)
        working = frame.copy().set_index("ts")
        working["bybit_entry"] = close.reindex(working.index)
        working["bybit_exit"] = close.reindex(working.index + HOLD_DELTA).values
        working["bybit_ret_4h"] = working["bybit_exit"] / working["bybit_entry"] - 1.0
        funding_sums = []
        for ts in working.index:
            funding_sums.append(funding.loc[(funding.index > ts) & (funding.index <= ts + HOLD_DELTA)].sum())
        working["bybit_funding_4h"] = funding_sums
        working["bybit_ret_4h_funding_adj"] = working["bybit_ret_4h"] - working["bybit_funding_4h"]
        parts.append(working.reset_index())

    full = pd.concat(parts, ignore_index=True)
    full = full.dropna(subset=["bybit_ret_4h"]).copy()
    full["avg_ret_4h_funding_adj"] = (full["ret_4h"] + full["bybit_ret_4h_funding_adj"]) / 2.0
    full["entry_drag"] = full["avg_positive_drag_bps"] / 10000.0
    full["net_ret_after_costs"] = full["avg_ret_4h_funding_adj"] - FEE_ALL_MAKER - full["entry_drag"]
    full["winner"] = (full["net_ret_after_costs"] > 0).astype(int)
    return full.sort_values(["ts", "symbol"]).reset_index(drop=True)


def select_baseline_soft(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_strict_threshold(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    score_cut = train_df["execution_adjusted_score"].quantile(0.60)
    ls_cut = max(2.5, train_df["ls_z"].quantile(0.60))
    filtered = df.loc[
        (df["execution_adjusted_score"] >= score_cut)
        & (df["ls_z"] >= ls_cut)
        & (df["breadth_mom"] >= 0.65)
    ].copy()
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values(["ts", "execution_adjusted_score", "score_abs"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_blended_rank(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["rank_ls"] = working.groupby("ts")["ls_z"].rank(pct=True)
    working["rank_taker"] = working.groupby("ts")["taker_z"].rank(pct=True)
    working["rank_mom"] = working.groupby("ts")["mom_4h"].rank(pct=True)
    working["rank_penalty"] = working.groupby("ts")["score_penalty"].rank(pct=True)
    working["blend_score"] = (
        1.0 * working["rank_ls"]
        + 0.6 * working["rank_taker"]
        + 0.6 * working["rank_mom"]
        - 0.7 * working["rank_penalty"]
    )
    return (
        working.sort_values(["ts", "blend_score", "execution_adjusted_score"], ascending=[True, False, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def select_empirical_winrate(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    # Fixed score buckets from train to avoid look-ahead.
    q1 = train_df["score_abs"].quantile(1 / 3)
    q2 = train_df["score_abs"].quantile(2 / 3)

    def score_bucket(s: pd.Series) -> pd.Series:
        return pd.cut(
            s,
            bins=[-float("inf"), q1, q2, float("inf")],
            labels=["low", "mid", "high"],
            ordered=True,
        )

    working = df.copy()
    working["bucket"] = score_bucket(working["score_abs"])
    train_prior = working.loc[working["ts"] <= TRAIN_END].copy()
    test_rows = []
    for ts, frame in working.loc[working["ts"] >= TEST_START].groupby("ts"):
        prior = train_prior.loc[train_prior["ts"] < ts].copy()
        if prior.empty:
            continue
        global_p = prior["winner"].mean()
        symbol_stats = prior.groupby("symbol").agg(sym_n=("winner", "count"), sym_p=("winner", "mean"))
        bucket_stats = prior.groupby("bucket", observed=False).agg(bucket_p=("winner", "mean"))

        current = frame.copy()
        current = current.merge(symbol_stats, on="symbol", how="left")
        current = current.merge(bucket_stats, on="bucket", how="left")
        current["sym_n"] = current["sym_n"].fillna(0.0)
        current["sym_p"] = current["sym_p"].fillna(global_p)
        current["bucket_p"] = current["bucket_p"].fillna(global_p)
        # Shrink symbol win-rate toward global when history is thin.
        current["sym_weight"] = (current["sym_n"] / (current["sym_n"] + 3.0)).clip(0.0, 1.0)
        current["shrunk_sym_p"] = current["sym_weight"] * current["sym_p"] + (1.0 - current["sym_weight"]) * global_p
        current["quality_prob"] = (
            0.45 * current["shrunk_sym_p"]
            + 0.35 * current["bucket_p"]
            + 0.20 * global_p
        )
        chosen = (
            current.loc[current["quality_prob"] >= global_p]
            .sort_values(["quality_prob", "execution_adjusted_score"], ascending=[False, False])
            .head(TOP_N)
        )
        if chosen.empty:
            chosen = current.sort_values(["quality_prob", "execution_adjusted_score"], ascending=[False, False]).head(1)
        test_rows.append(chosen)
    if not test_rows:
        return pd.DataFrame(columns=working.columns)
    return pd.concat(test_rows, ignore_index=True)


def summarize(name: str, selected: pd.DataFrame) -> dict[str, float | int | str]:
    test = selected.loc[selected["ts"] >= TEST_START].copy()
    if test.empty:
        return {
            "approach": name,
            "test_rows": 0,
            "test_timestamps": 0,
            "test_avg_bps": float("nan"),
            "test_win_rate": float("nan"),
            "test_symbol_count": 0,
        }
    port = (
        test.groupby("ts", as_index=False)
        .agg(port_ret=("net_ret_after_costs", "mean"), n_positions=("symbol", "count"))
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return {
        "approach": name,
        "test_rows": int(test.shape[0]),
        "test_timestamps": int(port.shape[0]),
        "test_avg_bps": port["port_ret"].mean() * 10000.0,
        "test_win_rate": (port["port_ret"] > 0).mean(),
        "test_symbol_count": int(test["symbol"].nunique()),
    }


def write_summary(results: pd.DataFrame) -> None:
    lines = [
        "# Selection Approach Comparison",
        "",
        "This compares several ways to choose among the same base signal candidates.",
        "",
        "Target metric:",
        "",
        "- 4h cross-venue return",
        "- minus 8 bps maker fees",
        "- minus the symbol-level average entry-drag estimate",
        "",
        "## Test Results",
        "",
        "| Approach | Rows | Timestamps | Avg bps | Win Rate | Symbols |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['approach']} | {int(row['test_rows'])} | {int(row['test_timestamps'])} | "
            f"{row['test_avg_bps']:.2f} | {row['test_win_rate']:.1%} | {int(row['test_symbol_count'])} |"
        )
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    candidates = build_candidates()
    candidates.to_csv(CANDIDATES_CSV, index=False)
    train_df = candidates.loc[candidates["ts"] <= TRAIN_END].copy()

    selections = {
        "baseline_soft": select_baseline_soft(candidates),
        "strict_threshold": select_strict_threshold(candidates, train_df),
        "blended_rank": select_blended_rank(candidates),
        "empirical_winrate": select_empirical_winrate(candidates, train_df),
    }

    rows = [summarize(name, frame) for name, frame in selections.items()]
    results = pd.DataFrame(rows).sort_values("test_avg_bps", ascending=False).reset_index(drop=True)
    results.to_csv(COMPARISON_CSV, index=False)
    write_summary(results)
    print(f"Wrote {CANDIDATES_CSV}")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
