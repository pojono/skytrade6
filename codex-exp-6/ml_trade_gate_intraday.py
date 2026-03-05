#!/usr/bin/env python3
"""Intraday causal ML gate for 240m breadth-trend trade/no-trade decisions."""

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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EXP6_PATH = Path(__file__).resolve().parent / "analyze_market_structure.py"
EXEC_PATH = Path(__file__).resolve().parent / "breadth_trend_execution.py"
OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache_intraday"


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
    out = pd.Series(index=index, dtype="Int64")
    chunks = [chunk for chunk in np.array_split(index.to_numpy(), splits) if len(chunk)]
    for split_id, raw_chunk in enumerate(chunks, start=1):
        out.loc[pd.DatetimeIndex(raw_chunk)] = split_id
    return out


def build_cache_id(args: argparse.Namespace) -> str:
    return (
        f"lb{args.lookback_days}_mx{args.max_symbols}_mn{args.min_symbols}_"
        f"ov{args.min_overlap_days}_h{args.horizon}_"
        f"mk{args.maker_fee_bps}_qm{args.queue_miss_rate}_pf{args.partial_fill_rate}_"
        f"as{args.adverse_selection_bps}_mlb{args.metrics_lookback_bars}_"
        f"cmp{int(args.compact_features)}"
    ).replace(".", "p")


def load_or_build_dataset(
    args: argparse.Namespace,
    exp6,
    exec_mod,
) -> tuple[pd.DataFrame, list[str], int, pd.Timestamp, pd.Timestamp]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_id = args.cache_id if args.cache_id else build_cache_id(args)
    cache_path = CACHE_DIR / f"intraday_dataset_{cache_id}.pkl"

    if not args.rebuild_dataset_cache and cache_path.exists():
        payload = pd.read_pickle(cache_path)
        if isinstance(payload, dict) and {"data", "feat_cols", "universe_size", "start", "end"} <= payload.keys():
            data = payload["data"]
            feat_cols = payload["feat_cols"]
            universe_size = int(payload["universe_size"])
            start = pd.Timestamp(payload["start"])
            end = pd.Timestamp(payload["end"])
            return data, feat_cols, universe_size, start, end

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

    _, selected = exec_mod.compute_execution_summary(
        prices,
        args.horizon,
        args.maker_fee_bps,
        args.queue_miss_rate,
        args.partial_fill_rate,
        args.adverse_selection_bps,
    )
    if selected.empty:
        raise SystemExit("No candidate signals found.")

    feats, feat_cols = build_intraday_features(
        prices,
        selected,
        exp6,
        start,
        end,
        args.metrics_lookback_bars,
        use_cache=not args.no_cache,
        compact=args.compact_features,
    )
    data = selected.join(feats, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise SystemExit("No rows left after joining features.")

    payload = {
        "data": data,
        "feat_cols": feat_cols,
        "universe_size": int(prices.shape[1]),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    pd.to_pickle(payload, cache_path)
    return data, feat_cols, int(prices.shape[1]), start, end


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=max(5, window // 5)).mean()
    sd = series.rolling(window, min_periods=max(5, window // 5)).std()
    return (series - mu) / sd.replace(0.0, np.nan)


def build_intraday_features(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    exp6_module,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics_lookback_bars: int,
    use_cache: bool,
    compact: bool,
) -> tuple[pd.DataFrame, list[str]]:
    # Market-wide causal series at signal timestamp.
    mret_1m = np.log(prices).diff().mean(axis=1) * 1e4
    mret_15 = prices.pct_change(15).mean(axis=1) * 1e4
    mret_60 = prices.pct_change(60).mean(axis=1) * 1e4
    mret_240 = prices.pct_change(240).mean(axis=1) * 1e4
    mret_1440 = prices.pct_change(1440).mean(axis=1) * 1e4

    rv_60 = mret_1m.rolling(60, min_periods=20).std()
    rv_240 = mret_1m.rolling(240, min_periods=60).std()
    rv_1440 = mret_1m.rolling(1440, min_periods=240).std()

    disp_60 = prices.pct_change(60).std(axis=1) * 1e4
    disp_240 = prices.pct_change(240).std(axis=1) * 1e4
    breadth_240 = selected["breadth"]
    abs_breadth = (breadth_240 - 0.5).abs()

    feats = pd.DataFrame(index=selected.index)
    feats["abs_breadth"] = abs_breadth
    feats["breadth_240"] = breadth_240
    feats["breadth_240_z_1d"] = rolling_zscore(breadth_240, 1440).reindex(feats.index)
    feats["abs_breadth_z_1d"] = rolling_zscore(abs_breadth, 1440).reindex(feats.index)

    feats["signal_align_15"] = selected["signal"] * mret_15.reindex(feats.index)
    feats["signal_align_60"] = selected["signal"] * mret_60.reindex(feats.index)
    feats["signal_align_240"] = selected["signal"] * mret_240.reindex(feats.index)
    feats["signal_align_1d"] = selected["signal"] * mret_1440.reindex(feats.index)

    feats["rv_60"] = rv_60.reindex(feats.index)
    feats["rv_240"] = rv_240.reindex(feats.index)
    if not compact:
        feats["rv_1440"] = rv_1440.reindex(feats.index)
        feats["rv_ratio_60_240"] = (rv_60 / rv_240).reindex(feats.index)
        feats["rv_ratio_240_1440"] = (rv_240 / rv_1440).reindex(feats.index)
        feats["rv_60_z_1d"] = rolling_zscore(rv_60, 1440).reindex(feats.index)
        feats["rv_240_z_1d"] = rolling_zscore(rv_240, 1440).reindex(feats.index)

    feats["disp_60"] = disp_60.reindex(feats.index)
    feats["disp_240"] = disp_240.reindex(feats.index)
    if not compact:
        feats["disp_ratio_60_240"] = (disp_60 / disp_240).reindex(feats.index)
        feats["disp_240_z_1d"] = rolling_zscore(disp_240, 1440).reindex(feats.index)

    oi_df, taker_df, top_df, account_df = exp6_module.build_metrics_matrices(
        list(prices.columns),
        prices.index,
        start,
        end,
        use_cache=use_cache,
    )
    if not oi_df.empty:
        oi_change = oi_df.div(oi_df.shift(max(1, metrics_lookback_bars))).sub(1.0)
        oi_x = oi_change.mean(axis=1)
        taker_x = taker_df.mean(axis=1)
        top_x = top_df.mean(axis=1)
        account_x = account_df.mean(axis=1)
        feats["oi_change"] = oi_x.reindex(feats.index)
        if not compact:
            feats["oi_change_z_1d"] = rolling_zscore(oi_x, 288).reindex(feats.index)  # 5m series ~1 day
        feats["taker_ratio"] = taker_x.reindex(feats.index)
        feats["top_trader_ratio"] = top_x.reindex(feats.index)
        feats["account_ratio"] = account_x.reindex(feats.index)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats, list(feats.columns)


def fold_eval(test_df: pd.DataFrame, trade_mask: pd.Series) -> dict[str, object]:
    traded = test_df[trade_mask]
    base_net = float(test_df["expected_net_bps"].mean())
    traded_net = float(traded["expected_net_bps"].mean()) if len(traded) else float("nan")
    return {
        "rows_test": len(test_df),
        "rows_traded": len(traded),
        "trade_share": len(traded) / max(1, len(test_df)),
        "base_expected_net_bps": base_net,
        "traded_expected_net_bps": traded_net,
        "uplift_bps": traded_net - base_net if len(traded) else float("nan"),
        "base_positive_rate": float((test_df["expected_net_bps"] > 0).mean()),
        "traded_positive_rate": float((traded["expected_net_bps"] > 0).mean()) if len(traded) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=220)
    parser.add_argument("--max-symbols", type=int, default=80)
    parser.add_argument("--min-symbols", type=int, default=50)
    parser.add_argument("--min-overlap-days", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--maker-fee-bps", type=float, default=4.0)
    parser.add_argument("--queue-miss-rate", type=float, default=0.35)
    parser.add_argument("--partial-fill-rate", type=float, default=0.75)
    parser.add_argument("--adverse-selection-bps", type=float, default=1.0)
    parser.add_argument("--metrics-lookback-bars", type=int, default=3)
    parser.add_argument("--walkforward-splits", type=int, default=6)
    parser.add_argument("--min-train-rows", type=int, default=8000)
    parser.add_argument("--prob-threshold", type=float, default=0.60)
    parser.add_argument("--min-trades-per-fold", type=int, default=500)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--compact-features", action="store_true")
    parser.add_argument("--cache-id", type=str, default="")
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    exp6 = load_module(EXP6_PATH, "exp6_intraday_gate")
    exec_mod = load_module(EXEC_PATH, "exec_intraday_gate")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, feat_cols, universe_size, start, end = load_or_build_dataset(args, exp6, exec_mod)

    data["label"] = (data["expected_net_bps"] > 0).astype(int)
    data["split_id"] = assign_splits(pd.DatetimeIndex(data.index), max(2, args.walkforward_splits))
    data = data.dropna(subset=["split_id"]).copy()
    data["split_id"] = data["split_id"].astype(int)

    logit = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    fold_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    max_split = int(data["split_id"].max())
    for test_split in range(2, max_split + 1):
        train_df = data[data["split_id"] < test_split]
        test_df = data[data["split_id"] == test_split]
        if len(train_df) < args.min_train_rows or test_df.empty:
            continue

        x_train = train_df[feat_cols]
        y_train = train_df["label"].to_numpy()
        x_test = test_df[feat_cols]
        y_test = test_df["label"].to_numpy()

        logit.fit(x_train, y_train)
        rf.fit(x_train, y_train)
        prob = 0.5 * logit.predict_proba(x_test)[:, 1] + 0.5 * rf.predict_proba(x_test)[:, 1]
        trade_mask = pd.Series(prob >= args.prob_threshold, index=test_df.index)

        row = {"split_id": test_split}
        row.update(fold_eval(test_df, trade_mask))
        row["auc"] = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan")
        row["f1"] = float(f1_score(y_test, (prob >= args.prob_threshold).astype(int), zero_division=0))
        fold_rows.append(row)

        for idx, p in zip(test_df.index, prob):
            pred_rows.append(
                {
                    "timestamp": idx.isoformat(),
                    "split_id": test_split,
                    "prob_winner": float(p),
                    "expected_net_bps": float(test_df.loc[idx, "expected_net_bps"]),
                }
            )

    if not fold_rows:
        raise SystemExit("No valid walk-forward folds produced.")

    pred_df = pd.DataFrame(pred_rows)
    sweep_rows: list[dict[str, object]] = []
    if not args.skip_sweep:
        for threshold in np.arange(0.50, 0.81, 0.02):
            split_nets: list[float] = []
            split_trades: list[int] = []
            for split_id, split_df in pred_df.groupby("split_id"):
                traded = split_df[split_df["prob_winner"] >= threshold]
                split_trades.append(len(traded))
                if len(traded) == 0:
                    split_nets.append(float("nan"))
                else:
                    split_nets.append(float(traded["expected_net_bps"].mean()))
            finite_nets = [x for x in split_nets if pd.notna(x)]
            if not finite_nets:
                continue
            valid_coverage = sum(int(x >= args.min_trades_per_fold) for x in split_trades)
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "rows_traded": int(sum(split_trades)),
                    "avg_expected_net_bps": float(np.mean(finite_nets)),
                    "worst_split_expected_net_bps": float(np.min(finite_nets)),
                    "split_count": int(len(finite_nets)),
                    "splits_meeting_min_trades": int(valid_coverage),
                    "min_trades_per_fold": int(args.min_trades_per_fold),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    valid_sweep = pd.DataFrame(sweep_rows)
    if not valid_sweep.empty:
        best = valid_sweep.sort_values(
            ["splits_meeting_min_trades", "worst_split_expected_net_bps", "avg_expected_net_bps"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best = None

    # Feature importances from final fit on full sample.
    logit.fit(data[feat_cols], data["label"].to_numpy())
    rf.fit(data[feat_cols], data["label"].to_numpy())
    logit_coef = logit.named_steps["clf"].coef_[0]
    imp_rows = [
        {"feature": f, "logit_coef": float(c), "rf_importance": float(i)}
        for f, c, i in sorted(
            zip(feat_cols, logit_coef, rf.feature_importances_),
            key=lambda row: abs(row[1]),
            reverse=True,
        )
    ]

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    fold_path = OUT_DIR / f"ml_intraday_gate_folds{suffix}.csv"
    sweep_path = OUT_DIR / f"ml_intraday_gate_sweep{suffix}.csv"
    imp_path = OUT_DIR / f"ml_intraday_gate_importance{suffix}.csv"
    report_path = OUT_DIR / f"ml_intraday_gate_report{suffix}.md"

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
            "auc",
            "f1",
        ],
    )
    if sweep_rows:
        write_csv(
            sweep_path,
            sweep_rows,
            [
                "threshold",
                "rows_traded",
                "avg_expected_net_bps",
                "worst_split_expected_net_bps",
                "split_count",
                "splits_meeting_min_trades",
                "min_trades_per_fold",
            ],
        )
    write_csv(imp_path, imp_rows, ["feature", "logit_coef", "rf_importance"])

    lines = [
        "# ML Intraday Gate Report",
        "",
        f"- Date range: `{start.date()}` to `{end.date()}`",
        f"- Universe size: `{universe_size}`",
        f"- Rows with features: `{len(data)}`",
        f"- Run threshold: `{args.prob_threshold:.2f}`",
        f"- Min trades per fold (sweep): `{args.min_trades_per_fold}`",
        "",
        "## Walk-Forward (Run Threshold)",
        "",
        f"- Avg base expected net: `{fold_df['base_expected_net_bps'].mean():.2f}` bps",
        f"- Avg traded expected net: `{fold_df['traded_expected_net_bps'].mean():.2f}` bps",
        f"- Avg uplift: `{fold_df['uplift_bps'].mean():.2f}` bps",
        f"- Worst traded split: `{fold_df['traded_expected_net_bps'].min():.2f}` bps",
        f"- Avg trade share: `{fold_df['trade_share'].mean():.2%}`",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Sweep Threshold (Coverage-Aware)",
                "",
                f"- Threshold `{best['threshold']:.2f}`",
                f"- Avg expected net `{best['avg_expected_net_bps']:.2f}` bps",
                f"- Worst split expected net `{best['worst_split_expected_net_bps']:.2f}` bps",
                f"- Splits meeting min trades: `{int(best['splits_meeting_min_trades'])}`",
                f"- Rows traded: `{int(best['rows_traded'])}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Loaded {universe_size} symbols")
    print(f"Rows with features: {len(data)}")
    print(
        "Run summary: "
        f"avg_uplift={fold_df['uplift_bps'].mean():.2f}bps "
        f"worst_split={fold_df['traded_expected_net_bps'].min():.2f}bps"
    )
    if best is not None:
        print(
            "Best threshold: "
            f"{best['threshold']:.2f} avg={best['avg_expected_net_bps']:.2f}bps "
            f"worst={best['worst_split_expected_net_bps']:.2f}bps"
        )
    print(f"Wrote {fold_path}")
    if sweep_rows:
        print(f"Wrote {sweep_path}")
    print(f"Wrote {imp_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
