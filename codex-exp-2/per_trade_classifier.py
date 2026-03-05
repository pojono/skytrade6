from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
OOF_CSV = OUT_DIR / "per_trade_classifier_oof.csv"
GRID_CSV = OUT_DIR / "per_trade_classifier_grid.csv"
TEST_CSV = OUT_DIR / "per_trade_classifier_test_scored.csv"
DIAG_CSV = OUT_DIR / "per_trade_classifier_holdout_diagnostic.csv"
REPORT_MD = OUT_DIR / "FINDINGS_per_trade_classifier.md"

TRAIN_CUTOFF = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
FEE = 0.002

BASE_MASK = {
    "oi_med_3d": 20_000_000.0,
    "breadth_mom": 0.60,
    "median_ls_z": 0.0,
    "ls_z": 2.0,
    "taker_z": 0.5,
}

NUMERIC_FEATURES = [
    "ls_z",
    "taker_z",
    "mom_4h",
    "score_abs",
    "breadth_mom",
    "median_ls_z",
    "median_taker_z",
    "log_oi_med_3d",
    "symbol_train_avg_bps",
    "symbol_train_hit_rate",
    "symbol_train_trades",
    "symbol_seen_train",
]


@dataclass(frozen=True)
class Config:
    model_name: str
    threshold: float
    max_positions: int


def load_base_trades() -> pd.DataFrame:
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    mask = (
        (samples["oi_med_3d"] >= BASE_MASK["oi_med_3d"])
        & (samples["breadth_mom"] >= BASE_MASK["breadth_mom"])
        & (samples["median_ls_z"] >= BASE_MASK["median_ls_z"])
        & (samples["ls_z"] >= BASE_MASK["ls_z"])
        & (samples["taker_z"] >= BASE_MASK["taker_z"])
        & (samples["mom_4h"] > 0)
    )
    base = samples.loc[mask].copy()
    base = (
        base.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    base["net_bps_20"] = (base["ret_4h"] - FEE) * 10000.0
    base["target"] = (base["net_bps_20"] > 0).astype(int)
    base["log_oi_med_3d"] = np.log(base["oi_med_3d"].clip(lower=1.0))
    return base


def add_symbol_priors(base: pd.DataFrame) -> pd.DataFrame:
    train = base.loc[base["ts"] < TRAIN_CUTOFF].copy()
    priors = (
        train.groupby("symbol", as_index=False)
        .agg(
            symbol_train_avg_bps=("net_bps_20", "mean"),
            symbol_train_hit_rate=("target", "mean"),
            symbol_train_trades=("symbol", "count"),
        )
    )
    enriched = base.merge(priors, on="symbol", how="left")
    enriched["symbol_seen_train"] = enriched["symbol_train_trades"].notna().astype(int)
    enriched["symbol_train_avg_bps"] = enriched["symbol_train_avg_bps"].fillna(0.0)
    enriched["symbol_train_hit_rate"] = enriched["symbol_train_hit_rate"].fillna(0.5)
    enriched["symbol_train_trades"] = enriched["symbol_train_trades"].fillna(0.0)
    return enriched


def build_models() -> dict[str, Pipeline]:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            )
        ]
    )
    logistic = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    hgb = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
                    ]
                ),
            ),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)),
        ]
    )
    return {"logistic": logistic, "hgb": hgb}


def _time_folds(train: pd.DataFrame, n_folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    timestamps = np.array(sorted(train["ts"].unique()))
    if len(timestamps) < 12:
        raise RuntimeError("Not enough timestamps for CV.")
    usable = timestamps[3:]
    chunks = np.array_split(usable, n_folds)
    folds = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        cutoff = chunk[0]
        train_idx = train.index[train["ts"] < cutoff].to_numpy()
        val_idx = train.index[train["ts"].isin(chunk)].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def make_oof_predictions(train: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    folds = _time_folds(train, n_folds=3)
    all_rows = []
    for model_name, model in models.items():
        pred_rows = []
        for train_idx, val_idx in folds:
            fit = train.loc[train_idx]
            val = train.loc[val_idx]
            model.fit(fit[NUMERIC_FEATURES], fit["target"])
            probs = model.predict_proba(val[NUMERIC_FEATURES])[:, 1]
            scored = val[["ts", "symbol", "net_bps_20", "target"]].copy()
            scored["model_name"] = model_name
            scored["prob"] = probs
            pred_rows.append(scored)
        all_rows.append(pd.concat(pred_rows, ignore_index=True))
    oof = pd.concat(all_rows, ignore_index=True)
    return oof


def evaluate_filtered(scored: pd.DataFrame, threshold: float, max_positions: int) -> dict[str, float | int]:
    taken = scored.loc[scored["prob"] >= threshold].copy()
    if taken.empty:
        return {
            "trade_rows": 0,
            "decisions": 0,
            "avg_bps_20": np.nan,
            "hit_rate": np.nan,
        }
    taken = (
        taken.sort_values(["ts", "prob"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(max_positions)
    )
    port = taken.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean"))
    return {
        "trade_rows": int(taken.shape[0]),
        "decisions": int(port.shape[0]),
        "avg_bps_20": float(port["net_bps_20"].mean()),
        "hit_rate": float((port["net_bps_20"] > 0).mean()),
    }


def search_configs(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sorted(oof["model_name"].unique()):
        subset = oof.loc[oof["model_name"] == model_name].copy()
        for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            for max_positions in (1, 2, 3):
                stats = evaluate_filtered(subset, threshold, max_positions)
                rows.append(
                    {
                        "model_name": model_name,
                        "threshold": threshold,
                        "max_positions": max_positions,
                        **stats,
                    }
                )
    grid = pd.DataFrame(rows)
    grid = grid.loc[(grid["decisions"] >= 8) & (grid["trade_rows"] >= 8)].copy()
    grid["score"] = grid["avg_bps_20"] + grid["hit_rate"] * 10.0
    grid = grid.sort_values(
        ["score", "avg_bps_20", "decisions"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return grid


def fit_full_and_score(train: pd.DataFrame, test: pd.DataFrame, config: Config) -> pd.DataFrame:
    models = build_models()
    model = models[config.model_name]
    model.fit(train[NUMERIC_FEATURES], train["target"])
    scored = test.copy()
    scored["prob"] = model.predict_proba(test[NUMERIC_FEATURES])[:, 1]
    scored = scored.sort_values(["ts", "prob"], ascending=[True, False]).reset_index(drop=True)
    scored["rank_in_ts"] = scored.groupby("ts").cumcount()
    scored["take"] = (
        (scored["prob"] >= config.threshold) & (scored["rank_in_ts"] < config.max_positions)
    ).astype(int)
    return scored


def diagnose_top_configs(train: pd.DataFrame, test: pd.DataFrame, grid: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    rows = []
    for _, row in grid.head(limit).iterrows():
        cfg = Config(
            model_name=str(row["model_name"]),
            threshold=float(row["threshold"]),
            max_positions=int(row["max_positions"]),
        )
        scored = fit_full_and_score(train, test, cfg)
        taken = scored.loc[scored["take"] == 1].copy()
        if taken.empty:
            continue
        port = taken.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean"))
        rows.append(
            {
                "model_name": cfg.model_name,
                "threshold": cfg.threshold,
                "max_positions": cfg.max_positions,
                "cv_avg_bps_20": float(row["avg_bps_20"]),
                "cv_decisions": int(row["decisions"]),
                "test_avg_bps_20": float(port["net_bps_20"].mean()),
                "test_decisions": int(port.shape[0]),
                "test_hit_rate": float((port["net_bps_20"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    base: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    grid: pd.DataFrame,
    scored_test: pd.DataFrame,
    diag: pd.DataFrame,
) -> None:
    best = grid.iloc[0]

    base_test_port = test.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean"))
    base_avg = base_test_port["net_bps_20"].mean()
    base_hit = (base_test_port["net_bps_20"] > 0).mean()

    taken = scored_test.loc[scored_test["take"] == 1].copy()
    clf_port = taken.groupby("ts", as_index=False).agg(net_bps_20=("net_bps_20", "mean")) if not taken.empty else pd.DataFrame(columns=["ts", "net_bps_20"])
    clf_avg = clf_port["net_bps_20"].mean() if not clf_port.empty else np.nan
    clf_hit = (clf_port["net_bps_20"] > 0).mean() if not clf_port.empty else np.nan
    diag_best = diag.sort_values(["test_avg_bps_20", "test_decisions"], ascending=[False, False]).iloc[0]

    train_pos = train["target"].mean()
    test_pos = test["target"].mean()

    feature_note = ", ".join(NUMERIC_FEATURES)
    lines = [
        "# Per-Trade Classifier",
        "",
        "This is a second-stage take/skip classifier on top of the current base strategy.",
        "",
        "## Setup",
        "",
        "- Base input is the current best signal instances only (the base top-3 strategy rows).",
        "- Target = whether a trade is positive after `20 bps` round-trip cost.",
        "- Train period: before `2026-01-01`.",
        "- Holdout test period: `2026-01-01` onward.",
        "- Model selection uses expanding time-order validation on the train period.",
        f"- Features: `{feature_note}`",
        "",
        "## Class Balance",
        "",
        f"- Train positive rate: {train_pos:.1%}",
        f"- Test positive rate: {test_pos:.1%}",
        "",
        "## Best CV Configuration",
        "",
        f"- Model: `{best['model_name']}`",
        f"- Probability threshold: `{best['threshold']:.2f}`",
        f"- Max positions per timestamp: `{int(best['max_positions'])}`",
        f"- CV decisions: {int(best['decisions'])}",
        f"- CV avg: {best['avg_bps_20']:.2f} bps",
        f"- CV hit rate: {best['hit_rate']:.1%}",
        "",
        "## Holdout Comparison",
        "",
        f"- Base holdout decisions: {len(base_test_port)}",
        f"- Base holdout avg: {base_avg:.2f} bps",
        f"- Base holdout hit rate: {base_hit:.1%}",
        f"- Classifier holdout decisions: {len(clf_port)}",
        f"- Classifier holdout trade rows: {int(taken.shape[0])}",
        f"- Classifier holdout avg: {clf_avg:.2f} bps",
        f"- Classifier holdout hit rate: {clf_hit:.1%}",
        "",
        "## Holdout Diagnostic Across Top CV Candidates",
        "",
        f"- Best test result among top CV candidates: `{diag_best['model_name']}` threshold `{diag_best['threshold']:.2f}` max-pos `{int(diag_best['max_positions'])}`",
        f"- That candidate holdout avg: {diag_best['test_avg_bps_20']:.2f} bps",
        f"- That candidate holdout decisions: {int(diag_best['test_decisions'])}",
        "",
        "## Interpretation",
        "",
        "- If holdout avg improves while retaining enough decisions, the classifier is useful.",
        "- If it only improves by collapsing to almost no trades, it is too sparse to trust.",
        "- If the best holdout result among strong CV candidates is still below the base strategy, the classifier is not adding value yet.",
        "- This still uses only explicit fees; execution realism is not added here.",
        "",
        "## Top Holdout Trades Kept",
        "",
        "| Timestamp | Symbol | Prob | Net Bps |",
        "|---|---|---:|---:|",
    ]

    if not taken.empty:
        top_rows = taken.sort_values(["prob", "net_bps_20"], ascending=[False, False]).head(10)
        for _, row in top_rows.iterrows():
            lines.append(
                f"| {row['ts']} | {row['symbol']} | {row['prob']:.3f} | {row['net_bps_20']:.2f} |"
            )
    lines.append("")

    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    base = add_symbol_priors(load_base_trades())
    train = base.loc[base["ts"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    test = base.loc[base["ts"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    models = build_models()
    oof = make_oof_predictions(train, models)
    oof.to_csv(OOF_CSV, index=False)

    grid = search_configs(oof)
    grid.to_csv(GRID_CSV, index=False)
    best = grid.iloc[0]
    config = Config(
        model_name=str(best["model_name"]),
        threshold=float(best["threshold"]),
        max_positions=int(best["max_positions"]),
    )
    scored_test = fit_full_and_score(train, test, config)
    scored_test.to_csv(TEST_CSV, index=False)
    diag = diagnose_top_configs(train, test, grid, limit=15)
    diag.to_csv(DIAG_CSV, index=False)
    write_report(base, train, test, grid, scored_test, diag)

    print(f"Wrote {OOF_CSV}")
    print(f"Wrote {GRID_CSV}")
    print(f"Wrote {TEST_CSV}")
    print(f"Wrote {DIAG_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(best.to_string())


if __name__ == "__main__":
    main()
