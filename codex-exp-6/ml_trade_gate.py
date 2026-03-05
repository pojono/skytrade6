#!/usr/bin/env python3
"""ML-based trade/no-trade gate for the 240m breadth trend candidate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EXP6_PATH = Path(__file__).resolve().parent / "analyze_market_structure.py"
EXEC_PATH = Path(__file__).resolve().parent / "breadth_trend_execution.py"
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assign_splits(index: pd.DatetimeIndex, splits: int) -> pd.Series:
    split_series = pd.Series(index=index, dtype="Int64")
    chunks = [chunk for chunk in np.array_split(index.to_numpy(), splits) if len(chunk)]
    for split_id, raw_chunk in enumerate(chunks, start=1):
        chunk_index = pd.DatetimeIndex(raw_chunk)
        split_series.loc[chunk_index] = split_id
    return split_series


def build_features(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    horizon: int,
    exp6_module,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics_lookback_bars: int,
    use_cache: bool,
) -> tuple[pd.DataFrame, list[str]]:
    market_ret_1m = np.log(prices).diff().mean(axis=1) * 1e4
    market_ret_60 = prices.pct_change(60).mean(axis=1) * 1e4
    market_ret_240 = prices.pct_change(240).mean(axis=1) * 1e4
    rv_60 = market_ret_1m.rolling(60).std()
    rv_240 = market_ret_1m.rolling(240).std()
    disp_240 = prices.pct_change(240).std(axis=1) * 1e4

    features = pd.DataFrame(index=selected.index)
    features["abs_breadth"] = (selected["breadth"] - 0.5).abs()
    features["rv_60"] = rv_60.reindex(features.index)
    features["rv_240"] = rv_240.reindex(features.index)
    features["disp_240"] = disp_240.reindex(features.index)
    features["signal_align_60"] = selected["signal"] * market_ret_60.reindex(features.index)
    features["signal_align_240"] = selected["signal"] * market_ret_240.reindex(features.index)
    features["signal_align_h"] = selected["signal"] * (
        prices.pct_change(horizon).mean(axis=1) * 1e4
    ).reindex(features.index)

    oi_df, taker_df, top_df, account_df = exp6_module.build_metrics_matrices(
        list(prices.columns),
        prices.index,
        start,
        end,
        use_cache=use_cache,
    )
    if not oi_df.empty:
        oi_change = oi_df.div(oi_df.shift(max(1, metrics_lookback_bars))).sub(1.0)
        features["oi_change"] = oi_change.mean(axis=1).reindex(features.index)
        features["taker_ratio"] = taker_df.mean(axis=1).reindex(features.index)
        features["top_trader_ratio"] = top_df.mean(axis=1).reindex(features.index)
        features["account_ratio"] = account_df.mean(axis=1).reindex(features.index)

    features = features.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(features.columns)
    return features, feature_cols


def summarize_fold(
    split_id: int,
    test_df: pd.DataFrame,
    trade_mask: pd.Series,
) -> dict[str, object]:
    traded = test_df[trade_mask]
    return {
        "split_id": split_id,
        "rows_test": len(test_df),
        "rows_traded": len(traded),
        "trade_share": len(traded) / max(1, len(test_df)),
        "base_expected_net_bps": float(test_df["expected_net_bps"].mean()),
        "traded_expected_net_bps": float(traded["expected_net_bps"].mean()) if len(traded) else float("nan"),
        "uplift_bps": (
            float(traded["expected_net_bps"].mean()) - float(test_df["expected_net_bps"].mean())
            if len(traded)
            else float("nan")
        ),
        "base_positive_rate": float((test_df["expected_net_bps"] > 0).mean()),
        "traded_positive_rate": float((traded["expected_net_bps"] > 0).mean()) if len(traded) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--max-symbols", type=int, default=100)
    parser.add_argument("--min-symbols", type=int, default=60)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--queue-miss-rate", type=float, default=0.35)
    parser.add_argument("--partial-fill-rate", type=float, default=0.75)
    parser.add_argument("--adverse-selection-bps", type=float, default=1.0)
    parser.add_argument("--walkforward-splits", type=int, default=3)
    parser.add_argument("--metrics-lookback-bars", type=int, default=3)
    parser.add_argument("--prob-threshold", type=float, default=0.58)
    parser.add_argument("--min-train-rows", type=int, default=3000)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.prob_threshold < 1.0:
        raise SystemExit("Expected --prob-threshold in (0,1)")

    exp6 = load_module(EXP6_PATH, "exp6_ml_gate")
    exec_mod = load_module(EXEC_PATH, "exec_ml_gate")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end = exp6.pick_date_range(None, None, args.lookback_days)
    symbols = exp6.eligible_symbols(args.min_overlap_days)[: max(1, args.max_symbols)]
    prices = exp6.build_price_matrix(
        exchange="binance",
        symbols=symbols,
        start=start,
        end=end,
        coverage_ratio=0.8,
        fill_limit=2,
        use_cache=not args.no_cache,
    )
    if prices.empty or prices.shape[1] < args.min_symbols:
        raise SystemExit(
            f"Insufficient aligned data. Loaded {prices.shape[1]} symbols, need at least {args.min_symbols}."
        )

    summary, selected = exec_mod.compute_execution_summary(
        prices,
        args.horizon,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
    )
    if summary is None or selected.empty:
        raise SystemExit("No candidate signals found.")

    features, feature_cols = build_features(
        prices,
        selected,
        args.horizon,
        exp6,
        start,
        end,
        args.metrics_lookback_bars,
        use_cache=not args.no_cache,
    )

    data = selected.join(features, how="inner")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise SystemExit("No feature rows available after joining.")

    data["label"] = (data["expected_net_bps"] > 0).astype(int)
    data["split_id"] = assign_splits(prices.index, max(2, args.walkforward_splits)).reindex(data.index)
    data = data.dropna(subset=["split_id"]).copy()
    data["split_id"] = data["split_id"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=None)),
        ]
    )

    fold_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    max_split = int(data["split_id"].max())
    for test_split in range(2, max_split + 1):
        train_df = data[data["split_id"] < test_split]
        test_df = data[data["split_id"] == test_split]
        if len(train_df) < args.min_train_rows or test_df.empty:
            continue

        x_train = train_df[feature_cols]
        y_train = train_df["label"]
        x_test = test_df[feature_cols]
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        trade_mask = pd.Series(prob >= args.prob_threshold, index=test_df.index)

        fold_rows.append(summarize_fold(test_split, test_df, trade_mask))
        for idx, p in zip(test_df.index, prob):
            pred_rows.append(
                {
                    "timestamp": idx.isoformat(),
                    "split_id": test_split,
                    "prob_trade_positive": float(p),
                    "expected_net_bps": float(test_df.loc[idx, "expected_net_bps"]),
                }
            )

    if not fold_rows:
        raise SystemExit("No valid walk-forward folds were produced.")

    pred_df = pd.DataFrame(pred_rows)
    thresholds = np.arange(0.50, 0.81, 0.02)
    sweep_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        by_split: list[float] = []
        traded_count = 0
        for split_id, split_df in pred_df.groupby("split_id"):
            traded = split_df[split_df["prob_trade_positive"] >= threshold]
            if traded.empty:
                continue
            traded_count += len(traded)
            by_split.append(float(traded["expected_net_bps"].mean()))
        if not by_split:
            continue
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "rows_traded": int(traded_count),
                "avg_expected_net_bps": float(np.mean(by_split)),
                "worst_split_expected_net_bps": float(np.min(by_split)),
                "split_count": int(len(by_split)),
            }
        )

    # Fit once on all rows for interpretability.
    model.fit(data[feature_cols], data["label"])
    coefs = model.named_steps["clf"].coef_[0]
    coef_rows = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    ]

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    fold_path = OUT_DIR / f"ml_trade_gate_folds{suffix}.csv"
    sweep_path = OUT_DIR / f"ml_trade_gate_threshold_sweep{suffix}.csv"
    coef_path = OUT_DIR / f"ml_trade_gate_coefficients{suffix}.csv"
    report_path = OUT_DIR / f"ml_trade_gate_report{suffix}.md"

    write_csv(
        fold_path,
        fold_rows,
        [
            "split_id",
            "rows_test",
            "rows_traded",
            "trade_share",
            "base_expected_net_bps",
            "traded_expected_net_bps",
            "uplift_bps",
            "base_positive_rate",
            "traded_positive_rate",
        ],
    )
    write_csv(
        sweep_path,
        sweep_rows,
        [
            "threshold",
            "rows_traded",
            "avg_expected_net_bps",
            "worst_split_expected_net_bps",
            "split_count",
        ],
    )
    write_csv(coef_path, coef_rows, ["feature", "coefficient"])

    fold_df = pd.DataFrame(fold_rows)
    best_threshold_row = max(
        sweep_rows,
        key=lambda row: (row["worst_split_expected_net_bps"], row["avg_expected_net_bps"]),
    )
    lines = [
        "# ML Trade Gate Report",
        "",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{prices.shape[1]}`",
        f"- Signals with features: `{len(data)}`",
        f"- Prob threshold (run): `{args.prob_threshold:.2f}`",
        "",
        "## Walk-Forward Summary",
        "",
        f"- Avg base expected net: `{fold_df['base_expected_net_bps'].mean():.2f}` bps",
        f"- Avg traded expected net: `{fold_df['traded_expected_net_bps'].mean():.2f}` bps",
        f"- Avg uplift: `{fold_df['uplift_bps'].mean():.2f}` bps",
        f"- Worst traded split expected net: `{fold_df['traded_expected_net_bps'].min():.2f}` bps",
        f"- Avg trade share: `{fold_df['trade_share'].mean():.2%}`",
        "",
        "## Best Threshold By Worst-Split",
        "",
        f"- Threshold `{best_threshold_row['threshold']:.2f}`: "
        f"avg `{best_threshold_row['avg_expected_net_bps']:.2f}` bps, "
        f"worst split `{best_threshold_row['worst_split_expected_net_bps']:.2f}` bps, "
        f"rows traded `{best_threshold_row['rows_traded']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Loaded {prices.shape[1]} symbols")
    print(f"Rows with features: {len(data)}")
    print(f"Wrote {fold_path}")
    print(f"Wrote {sweep_path}")
    print(f"Wrote {coef_path}")
    print(f"Wrote {report_path}")
    print(
        "Run threshold summary: "
        f"avg uplift={fold_df['uplift_bps'].mean():.2f}bps "
        f"worst_traded_split={fold_df['traded_expected_net_bps'].min():.2f}bps"
    )


if __name__ == "__main__":
    main()
