#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "event_microstructure_dataset.csv"

BASE_FEATURES = [
    "gap_bps",
    "gap_z_60",
    "gap_z_240",
    "premium_gap_bps",
    "premium_gap_z_240",
    "crowding_gap",
    "crowding_gap_z_240",
    "bn_taker_imbalance",
    "rel_ret_5m_bps",
    "rel_ret_15m_bps",
    "bb_realized_vol_15m_bps",
    "bn_quote_volume_z_60",
    "bb_turnover_z_60",
    "oi_gap_30m",
    "oi_gap_30m_z_240",
    "tod_sin",
    "tod_cos",
]

MICRO_FEATURES = [
    "bn_depth_imbalance_0.2",
    "bn_depth_imbalance_1",
    "bn_depth_imbalance_5",
    "bn_depth_pressure",
    "bb_top5_imbalance",
    "bb_top20_imbalance",
    "bb_best_sz_imbalance",
    "bb_spread_bps_ob",
    "bb_top5_pull_pressure",
    "bb_top5_pull_pressure_5s",
    "bb_top5_pull_pressure_15s",
    "bb_top20_imbalance_chg_5s",
    "bb_mid_gap_vs_trades_bps",
    "ob_gap_bps",
    "ob_gap_change_5s",
    "ob_gap_change_15s",
    "sec_gap_bps",
    "sec_gap_change_5s",
    "sec_gap_change_15s",
    "bn_signed_notional_15s",
    "bn_signed_notional_60s",
    "bb_signed_notional_15s",
    "bb_signed_notional_60s",
    "combo_signed_notional_15s",
    "combo_signed_notional_60s",
    "bn_flow_ratio_15s",
    "bn_flow_ratio_60s",
    "bb_flow_ratio_15s",
    "bb_flow_ratio_60s",
    "flow_divergence_15s",
    "flow_divergence_60s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank dislocation events with microstructure features.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--selection-quantile", type=float, default=0.8)
    parser.add_argument("--output-prefix", default="event_microstructure")
    return parser.parse_args()


def add_symbol_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sym = pd.get_dummies(df["symbol"], prefix="sym", dtype=float)
    out = pd.concat([df.reset_index(drop=True), sym.reset_index(drop=True)], axis=1)
    return out, list(sym.columns)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    days = sorted(df["date"].unique().tolist())
    split_idx = min(max(int(len(days) * 0.7), 1), len(days) - 1)
    cutoff = days[split_idx]
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test, cutoff


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


def bucket_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    tmp = df[[score_col, "pair_net_15m_bps", "pair_net_max_15m_bps", "pair_win_15m"]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()
    tmp["bucket"] = pd.qcut(tmp[score_col], q=10, labels=False, duplicates="drop")
    grouped = tmp.groupby("bucket").agg(
        events=(score_col, "size"),
        mean_score=(score_col, "mean"),
        pair_net_15m_bps=("pair_net_15m_bps", "mean"),
        pair_net_max_15m_bps=("pair_net_max_15m_bps", "mean"),
        pair_win_15m=("pair_win_15m", "mean"),
    ).reset_index()
    return grouped.sort_values("bucket", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["ts"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df, dummy_cols = add_symbol_dummies(df)
    feature_cols = BASE_FEATURES + MICRO_FEATURES + dummy_cols

    train, test, cutoff = split_train_test(df)
    if train.empty or test.empty:
        raise SystemExit("train/test split produced an empty partition")

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

    state_cols = ["gap_z_240", "premium_gap_z_240", "crowding_gap_z_240", "oi_gap_30m_z_240"]
    train["state_score"] = train[state_cols].fillna(0.0).mean(axis=1)
    test["state_score"] = test[state_cols].fillna(0.0).mean(axis=1)

    model_threshold = train["model_score"].quantile(args.selection_quantile)
    state_threshold = train["state_score"].quantile(args.selection_quantile)

    rows = []
    for split_name, frame in [("train", train), ("test", test)]:
        variants = {
            "baseline": frame,
            "state_top_quantile": frame[frame["state_score"] >= state_threshold],
            "model_top_quantile": frame[frame["model_score"] >= model_threshold],
        }
        for variant, subset in variants.items():
            stats = evaluate(subset)
            stats["split"] = split_name
            stats["variant"] = variant
            rows.append(stats)
    summary = pd.DataFrame(rows)[
        [
            "split",
            "variant",
            "events",
            "pair_net_15m_bps",
            "pair_net_max_15m_bps",
            "pair_win_15m",
            "pair_win_max_15m",
            "gap_close_15m_bps",
        ]
    ]

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.named_steps["rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    buckets = bucket_table(test, "model_score")
    monthly = (
        test[test["model_score"] >= model_threshold]
        .assign(month=lambda x: x["ts"].dt.tz_localize(None).dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            events=("pair_net_15m_bps", "size"),
            pair_net_15m_bps=("pair_net_15m_bps", "mean"),
            pair_net_max_15m_bps=("pair_net_max_15m_bps", "mean"),
            pair_win_15m=("pair_win_15m", "mean"),
        )
        .reset_index()
    )

    report_lines = [
        "# Event Microstructure Report",
        "",
        f"- Input: `{args.input}`",
        f"- Train/Test cutoff day: `{cutoff}`",
        f"- Selection quantile: `{args.selection_quantile:.2f}`",
        "",
        "## Dataset",
        "",
        f"- Events: `{len(df):,}`",
        f"- Symbols: `{df['symbol'].nunique()}`",
        f"- Train events: `{len(train):,}`",
        f"- Test events: `{len(test):,}`",
        f"- Baseline test pair net 15m: `{test['pair_net_15m_bps'].mean():+.2f} bps`",
        "",
        "## Test Summary",
        "",
    ]
    for row in summary[summary["split"] == "test"].itertuples(index=False):
        report_lines.append(
            f"- `{row.variant}`: events={int(row.events):,}, pair_net_15m={row.pair_net_15m_bps:+.2f} bps, "
            f"pair_net_max_15m={row.pair_net_max_15m_bps:+.2f} bps, pair_win_15m={row.pair_win_15m * 100:.1f}%, "
            f"pair_win_max_15m={row.pair_win_max_15m * 100:.1f}%, gap_close_15m={row.gap_close_15m_bps:+.2f} bps"
        )
    report_lines.extend(["", "## Top Feature Importances", ""])
    for row in importance.head(15).itertuples(index=False):
        report_lines.append(f"- `{row.feature}`: `{row.importance:.4f}`")

    prefix = args.output_prefix
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    importance.to_csv(OUT_DIR / f"{prefix}_feature_importance.csv", index=False)
    buckets.to_csv(OUT_DIR / f"{prefix}_score_buckets.csv", index=False)
    monthly.to_csv(OUT_DIR / f"{prefix}_monthly.csv", index=False)
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
