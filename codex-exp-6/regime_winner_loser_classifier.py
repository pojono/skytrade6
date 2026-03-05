#!/usr/bin/env python3
"""Reverse engineer winner/loser regimes for breadth-trend execution using ML."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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


def build_signal_features(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    exp6_module,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics_lookback_bars: int,
    use_cache: bool,
) -> pd.DataFrame:
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
        prices.pct_change(240).mean(axis=1) * 1e4
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
    return features


def aggregate_daily(signal_df: pd.DataFrame, min_signals_per_day: int) -> pd.DataFrame:
    signal_df = signal_df.copy()
    signal_df["date"] = signal_df.index.tz_convert("UTC").date
    grouped = signal_df.groupby("date")

    daily = grouped.agg(
        signal_count=("expected_net_bps", "size"),
        avg_expected_net_bps=("expected_net_bps", "mean"),
        median_expected_net_bps=("expected_net_bps", "median"),
        avg_gross_bps=("gross_bps", "mean"),
        abs_breadth_mean=("abs_breadth", "mean"),
        rv_60_mean=("rv_60", "mean"),
        rv_240_mean=("rv_240", "mean"),
        disp_240_mean=("disp_240", "mean"),
        signal_align_60_mean=("signal_align_60", "mean"),
        signal_align_240_mean=("signal_align_240", "mean"),
        oi_change_mean=("oi_change", "mean"),
        taker_ratio_mean=("taker_ratio", "mean"),
        top_trader_ratio_mean=("top_trader_ratio", "mean"),
        account_ratio_mean=("account_ratio", "mean"),
    ).reset_index()

    daily = daily[daily["signal_count"] >= min_signals_per_day].copy()
    daily["winner"] = (daily["avg_expected_net_bps"] > 0).astype(int)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=220)
    parser.add_argument("--max-symbols", type=int, default=80)
    parser.add_argument("--min-symbols", type=int, default=50)
    parser.add_argument("--min-overlap-days", type=int, default=180)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--queue-miss-rate", type=float, default=0.35)
    parser.add_argument("--partial-fill-rate", type=float, default=0.75)
    parser.add_argument("--adverse-selection-bps", type=float, default=1.0)
    parser.add_argument("--metrics-lookback-bars", type=int, default=3)
    parser.add_argument("--min-signals-per-day", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not 0.5 <= args.train_ratio < 0.95:
        raise SystemExit("Expected --train-ratio in [0.5, 0.95)")

    exp6 = load_module(EXP6_PATH, "exp6_wl_cls")
    exec_mod = load_module(EXEC_PATH, "exec_wl_cls")
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
        240,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
    )
    if summary is None or selected.empty:
        raise SystemExit("No candidate signals found.")

    features = build_signal_features(
        prices,
        selected,
        exp6,
        start,
        end,
        args.metrics_lookback_bars,
        use_cache=not args.no_cache,
    )
    signal_df = selected.join(features, how="inner")
    signal_df = signal_df.replace([np.inf, -np.inf], np.nan).dropna()
    if signal_df.empty:
        raise SystemExit("No rows left after feature join.")

    daily = aggregate_daily(signal_df, args.min_signals_per_day)
    if len(daily) < 40:
        raise SystemExit(f"Not enough daily regime rows ({len(daily)}) after filtering.")

    raw_feature_cols = [
        "signal_count",
        "abs_breadth_mean",
        "rv_60_mean",
        "rv_240_mean",
        "disp_240_mean",
        "signal_align_60_mean",
        "signal_align_240_mean",
        "oi_change_mean",
        "taker_ratio_mean",
        "top_trader_ratio_mean",
        "account_ratio_mean",
    ]
    # Leakage-safe setup: predict day t winner/loser from day t-1 features only.
    for col in raw_feature_cols:
        daily[f"lag1_{col}"] = daily[col].shift(1)

    feature_cols = [f"lag1_{col}" for col in raw_feature_cols]
    daily = daily.dropna(subset=feature_cols + ["winner"]).copy()
    split_idx = int(len(daily) * args.train_ratio)
    train = daily.iloc[:split_idx].copy()
    test = daily.iloc[split_idx:].copy()
    if len(test) < 15:
        raise SystemExit("Test split too small. Increase lookback or adjust train-ratio.")

    x_train = train[feature_cols]
    y_train = train["winner"].astype(int).to_numpy()
    x_test = test[feature_cols]
    y_test = test["winner"].astype(int).to_numpy()

    logit = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    logit.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    logit_prob = logit.predict_proba(x_test)[:, 1]
    rf_prob = rf.predict_proba(x_test)[:, 1]
    ensemble_prob = 0.5 * logit_prob + 0.5 * rf_prob
    metrics = metric_bundle(y_test, ensemble_prob, args.prob_threshold)

    test = test.copy()
    test["prob_winner"] = ensemble_prob
    test["trade_ml"] = (test["prob_winner"] >= args.prob_threshold).astype(int)
    test["regime_pred"] = np.where(test["trade_ml"] == 1, "winner", "loser")

    traded = test[test["trade_ml"] == 1]
    no_trade = test[test["trade_ml"] == 0]
    trade_avg = float(traded["avg_expected_net_bps"].mean()) if len(traded) else float("nan")
    no_trade_avg = float(no_trade["avg_expected_net_bps"].mean()) if len(no_trade) else float("nan")

    # Winner vs loser feature deltas on train set for interpretability.
    train_w = train[train["winner"] == 1]
    train_l = train[train["winner"] == 0]
    compare_rows: list[dict[str, object]] = []
    for col in feature_cols:
        w_mean = float(train_w[col].mean()) if len(train_w) else float("nan")
        l_mean = float(train_l[col].mean()) if len(train_l) else float("nan")
        compare_rows.append(
            {
                "feature": col,
                "winner_mean": w_mean,
                "loser_mean": l_mean,
                "delta_w_minus_l": w_mean - l_mean,
            }
        )
    compare_rows.sort(key=lambda row: abs(row["delta_w_minus_l"]), reverse=True)

    logit_coef = logit.named_steps["clf"].coef_[0]
    coef_rows = [
        {"feature": feature, "logit_coef": float(coef), "rf_importance": float(imp)}
        for feature, coef, imp in sorted(
            zip(feature_cols, logit_coef, rf.feature_importances_),
            key=lambda row: abs(row[1]),
            reverse=True,
        )
    ]

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    pred_path = OUT_DIR / f"winner_loser_predictions{suffix}.csv"
    compare_path = OUT_DIR / f"winner_loser_feature_compare{suffix}.csv"
    coef_path = OUT_DIR / f"winner_loser_model_importance{suffix}.csv"
    report_path = OUT_DIR / f"winner_loser_report{suffix}.md"

    write_csv(
        pred_path,
        test[
            [
                "date",
                "signal_count",
                "avg_expected_net_bps",
                "winner",
                "prob_winner",
                "trade_ml",
                "regime_pred",
            ]
        ].to_dict("records"),
        [
            "date",
            "signal_count",
            "avg_expected_net_bps",
            "winner",
            "prob_winner",
            "trade_ml",
            "regime_pred",
        ],
    )
    write_csv(compare_path, compare_rows, ["feature", "winner_mean", "loser_mean", "delta_w_minus_l"])
    write_csv(coef_path, coef_rows, ["feature", "logit_coef", "rf_importance"])

    lines = [
        "# Winner/Loser Regime Report",
        "",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{prices.shape[1]}`",
        f"- Daily rows (train/test): `{len(train)}` / `{len(test)}`",
        f"- Probability threshold: `{args.prob_threshold:.2f}`",
        "",
        "## Classifier Quality (Test)",
        "",
        f"- Accuracy: `{metrics['accuracy']:.3f}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- F1: `{metrics['f1']:.3f}`",
        f"- AUC: `{metrics['auc']:.3f}`",
        "",
        "## Trade/No-Trade Outcome (Test)",
        "",
        f"- Trade days: `{len(traded)}` of `{len(test)}` ({len(traded)/max(1,len(test)):.2%})",
        f"- Avg expected net on trade days: `{trade_avg:.2f}` bps",
        f"- Avg expected net on no-trade days: `{no_trade_avg:.2f}` bps",
        "",
        "## Top Winner-vs-Loser Differences (Train)",
        "",
    ]
    for row in compare_rows[:8]:
        lines.append(
            f"- {row['feature']}: winner `{row['winner_mean']:.3f}`, "
            f"loser `{row['loser_mean']:.3f}`, delta `{row['delta_w_minus_l']:.3f}`"
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Loaded {prices.shape[1]} symbols")
    print(f"Daily rows train/test: {len(train)}/{len(test)}")
    print(
        "Test metrics: "
        f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={metrics['auc']:.3f}"
    )
    print(
        "Trade filter outcome: "
        f"trade_days={len(traded)} trade_avg={trade_avg:.2f}bps no_trade_avg={no_trade_avg:.2f}bps"
    )
    print(f"Wrote {pred_path}")
    print(f"Wrote {compare_path}")
    print(f"Wrote {coef_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
