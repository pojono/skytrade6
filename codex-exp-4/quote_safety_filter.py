#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"


FEATURE_COLS = [
    "bb_spread_bps",
    "bb_top5_imbalance",
    "bb_top20_imbalance",
    "bb_top5_pull_pressure_5s",
    "bb_top5_pull_pressure_15s",
    "bb_mid_gap_bps",
    "combo_flow_30s",
    "combo_flow_60s",
    "bn_last_price",
    "bb_last_price",
]


def load_day(symbol: str, day: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return pd.read_csv(path)


def add_features(df: pd.DataFrame, danger_horizon_s: int, danger_bps: float) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        mean = df[col].rolling(300).mean()
        std = df[col].rolling(300).std()
        df[f"{col}_z"] = (df[col] - mean) / std

    df["rel_px_gap_bps"] = (df["bn_last_price"] - df["bb_last_price"]) / df["mid_px"] * 10000.0
    df["rel_px_gap_bps_z"] = (df["rel_px_gap_bps"] - df["rel_px_gap_bps"].rolling(300).mean()) / df["rel_px_gap_bps"].rolling(300).std()
    df["flow_over_spread"] = df["combo_flow_60s"] / df["bb_spread_bps"].replace(0, np.nan)
    df["flow_over_spread_z"] = (df["flow_over_spread"] - df["flow_over_spread"].rolling(300).mean()) / df["flow_over_spread"].rolling(300).std()
    df["future_abs_move_bps"] = df[f"future_ret_{danger_horizon_s}s"].abs() * 10000.0
    df["unsafe_quote"] = (df["future_abs_move_bps"] >= danger_bps).astype(int)

    cols = [f"{c}_z" for c in FEATURE_COLS] + ["rel_px_gap_bps_z", "flow_over_spread_z", "unsafe_quote", "future_abs_move_bps"]
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).reset_index(drop=True)


MODEL_FEATURES = [f"{c}_z" for c in FEATURE_COLS] + ["rel_px_gap_bps_z", "flow_over_spread_z"]


def evaluate_threshold(
    probs: pd.Series,
    labels: pd.Series,
    threshold: float,
) -> dict[str, float]:
    block_mask = probs >= threshold
    keep_mask = ~block_mask
    unsafe_rate_all = float(labels.mean())
    unsafe_rate_keep = float(labels[keep_mask].mean()) if keep_mask.sum() else np.nan
    recall_blocked = float(labels[block_mask].sum() / labels.sum()) if labels.sum() else 0.0
    return {
        "blocked_share": float(block_mask.mean()),
        "unsafe_rate_all": unsafe_rate_all,
        "unsafe_rate_keep": unsafe_rate_keep,
        "unsafe_reduction": float((unsafe_rate_all - unsafe_rate_keep) / unsafe_rate_all) if unsafe_rate_all > 0 else 0.0,
        "unsafe_recall_blocked": recall_blocked,
        "kept_share": float(keep_mask.mean()),
    }


def choose_block_threshold(train_probs: pd.Series, train_labels: pd.Series) -> tuple[float, dict[str, float]]:
    candidates: list[tuple[float, dict[str, float]]] = []
    for q in [0.70, 0.80, 0.85, 0.90, 0.95]:
        thr = float(train_probs.quantile(q))
        stats = evaluate_threshold(train_probs, train_labels, thr)
        candidates.append((thr, stats))
    candidates.sort(
        key=lambda item: (item[1]["unsafe_reduction"], item[1]["unsafe_recall_blocked"], -item[1]["blocked_share"]),
        reverse=True,
    )
    return candidates[0]


def walkforward_symbol(
    symbol: str,
    days: list[str],
    train_window_days: int,
    danger_horizon_s: int,
    danger_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = {day: add_features(load_day(symbol, day), danger_horizon_s, danger_bps) for day in days}
    rows: list[dict[str, float | str]] = []

    for idx in range(train_window_days, len(days)):
        train_days = days[idx - train_window_days : idx]
        test_day = days[idx]
        train = pd.concat([prepared[d] for d in train_days], ignore_index=True)
        test = prepared[test_day]

        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(train[MODEL_FEATURES], train["unsafe_quote"])
        train_probs = pd.Series(model.predict_proba(train[MODEL_FEATURES])[:, 1], index=train.index)
        test_probs = pd.Series(model.predict_proba(test[MODEL_FEATURES])[:, 1], index=test.index)

        threshold, train_stats = choose_block_threshold(train_probs, train["unsafe_quote"])
        test_stats = evaluate_threshold(test_probs, test["unsafe_quote"], threshold)
        rows.append(
            {
                "symbol": symbol,
                "train_days": ",".join(train_days),
                "test_day": test_day,
                "prob_threshold": threshold,
                "train_unsafe_rate": train_stats["unsafe_rate_all"],
                "train_unsafe_rate_keep": train_stats["unsafe_rate_keep"],
                "train_unsafe_reduction": train_stats["unsafe_reduction"],
                "train_unsafe_recall_blocked": train_stats["unsafe_recall_blocked"],
                "train_blocked_share": train_stats["blocked_share"],
                "test_unsafe_rate": test_stats["unsafe_rate_all"],
                "test_unsafe_rate_keep": test_stats["unsafe_rate_keep"],
                "test_unsafe_reduction": test_stats["unsafe_reduction"],
                "test_unsafe_recall_blocked": test_stats["unsafe_recall_blocked"],
                "test_blocked_share": test_stats["blocked_share"],
            }
        )

    folds = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "folds": int(len(folds)),
                "mean_test_unsafe_rate": float(folds["test_unsafe_rate"].mean()),
                "mean_test_unsafe_rate_keep": float(folds["test_unsafe_rate_keep"].mean()),
                "mean_test_unsafe_reduction": float(folds["test_unsafe_reduction"].mean()),
                "mean_test_unsafe_recall_blocked": float(folds["test_unsafe_recall_blocked"].mean()),
                "mean_test_blocked_share": float(folds["test_blocked_share"].mean()),
            }
        ]
    )
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward quote safety filter using microstructure features.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT"])
    parser.add_argument(
        "--days",
        nargs="*",
        default=[
            "2026-02-24",
            "2026-02-25",
            "2026-02-26",
            "2026-02-27",
            "2026-02-28",
            "2026-03-01",
            "2026-03-02",
            "2026-03-03",
        ],
    )
    parser.add_argument("--train-window-days", type=int, default=3)
    parser.add_argument("--danger-horizon-s", type=int, default=60, choices=[30, 60, 120])
    parser.add_argument("--danger-bps", type=float, default=8.0)
    args = parser.parse_args()

    all_folds = []
    all_summaries = []
    for symbol in args.symbols:
        folds, summary = walkforward_symbol(
            symbol,
            days=args.days,
            train_window_days=args.train_window_days,
            danger_horizon_s=args.danger_horizon_s,
            danger_bps=args.danger_bps,
        )
        all_folds.append(folds)
        all_summaries.append(summary)
        s = summary.iloc[0]
        print(
            f"{symbol}: mean_unsafe={s['mean_test_unsafe_rate']:.3f} "
            f"mean_unsafe_keep={s['mean_test_unsafe_rate_keep']:.3f} "
            f"reduction={s['mean_test_unsafe_reduction']:.2%} "
            f"blocked={s['mean_test_blocked_share']:.2%}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    folds_df.to_csv(OUT_DIR / "quote_safety_filter_folds.csv", index=False)
    summary_df.to_csv(OUT_DIR / "quote_safety_filter_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'quote_safety_filter_folds.csv'}")
    print(f"wrote {OUT_DIR / 'quote_safety_filter_summary.csv'}")


if __name__ == "__main__":
    main()
