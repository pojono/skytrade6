#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from analyze_event_microstructure import BASE_FEATURES, MICRO_FEATURES


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "event_microstructure_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for event microstructure ranking.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-days", type=int, default=5)
    parser.add_argument("--selection-quantile", type=float, default=0.6)
    parser.add_argument("--output-prefix", default="event_microstructure_walkforward")
    return parser.parse_args()


def add_symbol_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sym = pd.get_dummies(df["symbol"], prefix="sym", dtype=float)
    out = pd.concat([df.reset_index(drop=True), sym.reset_index(drop=True)], axis=1)
    return out, list(sym.columns)


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "events": 0.0,
            "pair_net_15m_bps": np.nan,
            "pair_net_max_15m_bps": np.nan,
            "pair_win_15m": np.nan,
            "pair_win_max_15m": np.nan,
            "gap_close_15m_bps": np.nan,
        }
    return {
        "events": float(len(frame)),
        "pair_net_15m_bps": float(frame["pair_net_15m_bps"].mean()),
        "pair_net_max_15m_bps": float(frame["pair_net_max_15m_bps"].mean()),
        "pair_win_15m": float(frame["pair_win_15m"].mean()),
        "pair_win_max_15m": float(frame["pair_win_max_15m"].mean()),
        "gap_close_15m_bps": float(frame["gap_close_15m_bps"].mean()),
    }


def summarize_variant(rows: list[dict[str, float | str]], variant: str) -> dict[str, float | str]:
    subset = [row for row in rows if row["variant"] == variant]
    frame = pd.DataFrame(subset)
    return {
        "variant": variant,
        "folds": int(len(frame)),
        "mean_events": float(frame["events"].mean()),
        "median_events": float(frame["events"].median()),
        "mean_pair_net_15m_bps": float(frame["pair_net_15m_bps"].mean()),
        "median_pair_net_15m_bps": float(frame["pair_net_15m_bps"].median()),
        "mean_pair_net_max_15m_bps": float(frame["pair_net_max_15m_bps"].mean()),
        "mean_pair_win_15m": float(frame["pair_win_15m"].mean()),
        "positive_folds": int((frame["pair_net_15m_bps"] > 0).sum()),
        "nonempty_folds": int((frame["events"] > 0).sum()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["ts"]).replace([np.inf, -np.inf], np.nan)
    df, dummy_cols = add_symbol_dummies(df)
    feature_cols = BASE_FEATURES + MICRO_FEATURES + dummy_cols
    state_cols = ["gap_z_240", "premium_gap_z_240", "crowding_gap_z_240", "oi_gap_30m_z_240"]

    days = sorted(df["date"].unique().tolist())
    if len(days) <= args.train_days:
        raise SystemExit("not enough distinct days for requested walk-forward window")

    fold_rows: list[dict[str, float | str]] = []
    importance_rows: list[dict[str, float | str]] = []

    for idx in range(args.train_days, len(days)):
        test_day = days[idx]
        train_days = days[idx - args.train_days : idx]
        train = df[df["date"].isin(train_days)].copy()
        test = df[df["date"] == test_day].copy()
        if train.empty or test.empty:
            continue

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(train[feature_cols], train["pair_net_15m_bps"])
        train["model_score"] = model.predict(train[feature_cols])
        test["model_score"] = model.predict(test[feature_cols])
        train["state_score"] = train[state_cols].fillna(0.0).mean(axis=1)
        test["state_score"] = test[state_cols].fillna(0.0).mean(axis=1)

        model_threshold = train["model_score"].quantile(args.selection_quantile)
        state_threshold = train["state_score"].quantile(args.selection_quantile)

        variants = {
            "baseline": test,
            "state_top_quantile": test[test["state_score"] >= state_threshold],
            "model_top_quantile": test[test["model_score"] >= model_threshold],
        }
        for variant, subset in variants.items():
            stats = evaluate(subset)
            stats["test_day"] = test_day
            stats["train_start_day"] = train_days[0]
            stats["train_end_day"] = train_days[-1]
            stats["variant"] = variant
            fold_rows.append(stats)

        importances = model.named_steps["rf"].feature_importances_
        for feature, importance in zip(feature_cols, importances):
            importance_rows.append({"test_day": test_day, "feature": feature, "importance": float(importance)})

    folds = pd.DataFrame(fold_rows)[
        [
            "test_day",
            "train_start_day",
            "train_end_day",
            "variant",
            "events",
            "pair_net_15m_bps",
            "pair_net_max_15m_bps",
            "pair_win_15m",
            "pair_win_max_15m",
            "gap_close_15m_bps",
        ]
    ]
    summary = pd.DataFrame(
        [summarize_variant(fold_rows, variant) for variant in ["baseline", "state_top_quantile", "model_top_quantile"]]
    )
    importance = (
        pd.DataFrame(importance_rows)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    report_lines = [
        "# Event Microstructure Walk-Forward",
        "",
        f"- Input: `{args.input}`",
        f"- Train window: `{args.train_days}` days",
        f"- Selection quantile: `{args.selection_quantile:.2f}`",
        f"- Distinct days: `{len(days)}`",
        f"- Walk-forward folds: `{folds['test_day'].nunique()}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.variant}`: folds={row.folds}, mean_events={row.mean_events:.1f}, "
            f"mean_pair_net_15m={row.mean_pair_net_15m_bps:+.2f} bps, "
            f"mean_pair_net_max_15m={row.mean_pair_net_max_15m_bps:+.2f} bps, "
            f"mean_pair_win_15m={row.mean_pair_win_15m * 100:.1f}%, positive_folds={row.positive_folds}/{row.folds}"
        )
    report_lines.extend(["", "## Top Mean Feature Importances", ""])
    for row in importance.head(15).itertuples(index=False):
        report_lines.append(f"- `{row.feature}`: `{row.importance:.4f}`")

    prefix = args.output_prefix
    folds.to_csv(OUT_DIR / f"{prefix}_folds.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    importance.to_csv(OUT_DIR / f"{prefix}_feature_importance.csv", index=False)
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
