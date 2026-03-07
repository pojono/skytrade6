#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "dislocation_panel.csv.gz"

FEATURE_COLUMNS = [
    "bybit_discount_bps",
    "discount_z_60",
    "discount_z_240",
    "premium_gap_bps",
    "premium_gap_z_240",
    "crowding_gap",
    "crowding_gap_z_240",
    "crowding_align_z_240",
    "oi_gap_5m",
    "oi_gap_30m",
    "oi_gap_30m_z_240",
    "bn_taker_imbalance",
    "rel_ret_5m_bps",
    "rel_ret_15m_bps",
    "realized_vol_15m_bps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze state-conditioned Bybit dislocation reversion.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--hold-minutes", type=int, default=15)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--min-discount-bps", type=float, default=6.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--selection-quantile", type=float, default=0.8)
    parser.add_argument("--output-prefix", default="state_reversion")
    return parser.parse_args()


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(window // 4, 20)).mean()
    std = series.rolling(window, min_periods=max(window // 4, 20)).std()
    return (series - mean) / std.replace(0, np.nan)


def add_features(raw: pd.DataFrame, hold_minutes: int) -> pd.DataFrame:
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    group = df.groupby("symbol", sort=False)

    df["bb_oi_value"] = pd.to_numeric(df["bb_open_interest"], errors="coerce") * pd.to_numeric(df["bb_close"], errors="coerce")
    df["bb_crowding"] = 2.0 * (pd.to_numeric(df["bb_buy_ratio"], errors="coerce") - 0.5)
    df["bn_crowding"] = pd.to_numeric(df["bn_ls_ratio"], errors="coerce") - 1.0
    df["bn_taker_imbalance"] = 2.0 * (
        pd.to_numeric(df["bn_taker_buy_quote_volume"], errors="coerce")
        / pd.to_numeric(df["bn_quote_volume"], errors="coerce").replace(0, np.nan)
    ) - 1.0

    df["bybit_discount_bps"] = 10000.0 * (pd.to_numeric(df["bn_close"], errors="coerce") / pd.to_numeric(df["bb_close"], errors="coerce") - 1.0)
    df["premium_gap_bps"] = 10000.0 * (
        pd.to_numeric(df["bn_premium"], errors="coerce") - pd.to_numeric(df["bb_premium"], errors="coerce")
    )
    df["crowding_gap"] = df["bn_crowding"] - df["bb_crowding"]
    df["crowding_align"] = df["bybit_discount_bps"] * df["crowding_gap"]

    bb_close = pd.to_numeric(df["bb_close"], errors="coerce")
    bn_close = pd.to_numeric(df["bn_close"], errors="coerce")
    df["future_bb_ret_bps"] = 10000.0 * (group["bb_close"].shift(-hold_minutes) / bb_close - 1.0)
    df["future_bn_ret_bps"] = 10000.0 * (group["bn_close"].shift(-hold_minutes) / bn_close - 1.0)
    df["future_discount_bps"] = 10000.0 * (group["bn_close"].shift(-hold_minutes) / group["bb_close"].shift(-hold_minutes) - 1.0)
    df["future_gap_close_bps"] = df["bybit_discount_bps"] - df["future_discount_bps"]
    df["net_trade_bps"] = df["future_bb_ret_bps"]
    df["target_positive"] = (df["net_trade_bps"] > 0).astype(int)

    for lag in [1, 5, 15, 30]:
        df[f"bb_ret_{lag}m_bps"] = 10000.0 * group["bb_close"].pct_change(lag)
        df[f"bn_ret_{lag}m_bps"] = 10000.0 * group["bn_close"].pct_change(lag)
        if lag in (5, 15):
            df[f"rel_ret_{lag}m_bps"] = df[f"bn_ret_{lag}m_bps"] - df[f"bb_ret_{lag}m_bps"]
        df[f"bb_oi_value_chg_{lag}m"] = group["bb_oi_value"].pct_change(lag, fill_method=None)
        df[f"bn_oi_value_chg_{lag}m"] = group["bn_oi_value"].pct_change(lag, fill_method=None)
        if lag in (5, 30):
            df[f"oi_gap_{lag}m"] = df[f"bn_oi_value_chg_{lag}m"] - df[f"bb_oi_value_chg_{lag}m"]

    df["realized_vol_15m_bps"] = group["bb_close"].transform(
        lambda s: 10000.0 * s.pct_change().rolling(15, min_periods=10).std()
    )

    for col, window in [
        ("bybit_discount_bps", 60),
        ("bybit_discount_bps", 240),
        ("premium_gap_bps", 240),
        ("crowding_gap", 240),
        ("crowding_align", 240),
        ("oi_gap_30m", 240),
    ]:
        zcol = {
            ("bybit_discount_bps", 60): "discount_z_60",
            ("bybit_discount_bps", 240): "discount_z_240",
            ("premium_gap_bps", 240): "premium_gap_z_240",
            ("crowding_gap", 240): "crowding_gap_z_240",
            ("crowding_align", 240): "crowding_align_z_240",
            ("oi_gap_30m", 240): "oi_gap_30m_z_240",
        }[(col, window)]
        df[zcol] = group[col].transform(lambda s, w=window: rolling_z(s, w))

    minute_of_day = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["tod_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    df["tod_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    return df


def split_train_test(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_ts = sorted(df["ts"].unique().tolist())
    split_idx = min(max(int(len(unique_ts) * train_frac), 1), len(unique_ts) - 1)
    cutoff = pd.Timestamp(unique_ts[split_idx])
    train = df[df["ts"] < cutoff].copy()
    test = df[df["ts"] >= cutoff].copy()
    return train, test, cutoff


def evaluate_slice(frame: pd.DataFrame, fee_bps: float) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0.0,
            "gross_bps_per_trade": np.nan,
            "net_bps_per_trade": np.nan,
            "win_rate": np.nan,
            "after_fee_hit_rate": np.nan,
            "mean_gap_close_bps": np.nan,
            "total_net_bps": np.nan,
        }
    gross = frame["net_trade_bps"]
    net = gross - fee_bps
    return {
        "trades": float(len(frame)),
        "gross_bps_per_trade": float(gross.mean()),
        "net_bps_per_trade": float(net.mean()),
        "win_rate": float((gross > 0).mean()),
        "after_fee_hit_rate": float((net > 0).mean()),
        "mean_gap_close_bps": float(frame["future_gap_close_bps"].mean()),
        "total_net_bps": float(net.sum()),
    }


def fit_model(train: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train["target_after_fee"])
    return model


def build_state_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.35 * df["discount_z_240"].fillna(0.0)
        + 0.20 * df["premium_gap_z_240"].fillna(0.0)
        + 0.20 * df["crowding_gap_z_240"].fillna(0.0)
        + 0.15 * df["oi_gap_30m_z_240"].fillna(0.0)
        + 0.10 * df["rel_ret_5m_bps"].fillna(0.0) / 10.0
    )


def score_buckets(df: pd.DataFrame, score_col: str, fee_bps: float) -> pd.DataFrame:
    buckets = df[[score_col, "net_trade_bps", "future_gap_close_bps"]].copy()
    buckets = buckets.replace([np.inf, -np.inf], np.nan).dropna()
    if buckets.empty:
        return pd.DataFrame()
    buckets["bucket"] = pd.qcut(buckets[score_col], q=10, labels=False, duplicates="drop")
    if buckets["bucket"].nunique() == 0:
        return pd.DataFrame()
    grouped = buckets.groupby("bucket", dropna=False)
    out = grouped.agg(
        trades=(score_col, "size"),
        mean_score=(score_col, "mean"),
        gross_bps_per_trade=("net_trade_bps", "mean"),
        mean_gap_close_bps=("future_gap_close_bps", "mean"),
    ).reset_index()
    out["net_bps_per_trade"] = out["gross_bps_per_trade"] - fee_bps
    return out.sort_values("bucket", ascending=False).reset_index(drop=True)


def month_summary(df: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "trades", "gross_bps_per_trade", "net_bps_per_trade", "total_net_bps"])
    tmp = df.copy()
    tmp["month"] = tmp["ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
    out = tmp.groupby("month").agg(
        trades=("net_trade_bps", "size"),
        gross_bps_per_trade=("net_trade_bps", "mean"),
    ).reset_index()
    out["net_bps_per_trade"] = out["gross_bps_per_trade"] - fee_bps
    out["total_net_bps"] = out["net_bps_per_trade"] * out["trades"]
    return out


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    raw = pd.read_csv(args.input, parse_dates=["ts"])
    df = add_features(raw, hold_minutes=args.hold_minutes)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["target_after_fee"] = (df["net_trade_bps"] > args.fee_bps_roundtrip).astype(int)
    candidate_mask = (
        df["bybit_discount_bps"] >= args.min_discount_bps
    ) & df[FEATURE_COLUMNS + ["net_trade_bps", "future_gap_close_bps", "target_after_fee"]].notna().all(axis=1)
    candidates = df.loc[candidate_mask].copy()
    if candidates.empty:
        raise SystemExit("no candidate rows after feature engineering and discount filter")

    train, test, cutoff = split_train_test(candidates, args.train_frac)
    if train.empty or test.empty:
        raise SystemExit("train/test split produced an empty partition")

    train["state_score"] = build_state_score(train)
    test["state_score"] = build_state_score(test)

    model = fit_model(train)
    train["model_score"] = model.predict_proba(train[FEATURE_COLUMNS])[:, 1]
    test["model_score"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    state_threshold = train["state_score"].quantile(args.selection_quantile)
    model_threshold = train["model_score"].quantile(args.selection_quantile)

    summary_rows = []
    for split_name, frame in [("train", train), ("test", test)]:
        slices = {
            "baseline": frame,
            "state_top_quantile": frame[frame["state_score"] >= state_threshold],
            "model_top_quantile": frame[frame["model_score"] >= model_threshold],
        }
        for variant, subset in slices.items():
            stats = evaluate_slice(subset, fee_bps=args.fee_bps_roundtrip)
            stats["split"] = split_name
            stats["variant"] = variant
            summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows)[
        [
            "split",
            "variant",
            "trades",
            "gross_bps_per_trade",
            "net_bps_per_trade",
            "win_rate",
            "after_fee_hit_rate",
            "mean_gap_close_bps",
            "total_net_bps",
        ]
    ]

    bucket_df = score_buckets(test, "model_score", fee_bps=args.fee_bps_roundtrip)
    monthly = month_summary(test[test["model_score"] >= model_threshold], fee_bps=args.fee_bps_roundtrip)

    coefs = model.named_steps["clf"].coef_[0]
    weights = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "weight": coefs}
    ).sort_values("weight", ascending=False).reset_index(drop=True)

    train_auc = roc_auc_score(train["target_after_fee"], train["model_score"]) if train["target_after_fee"].nunique() > 1 else np.nan
    test_auc = roc_auc_score(test["target_after_fee"], test["model_score"]) if test["target_after_fee"].nunique() > 1 else np.nan

    report_lines = [
        "# State-Conditioned Reversion Report",
        "",
        f"- Input: `{args.input}`",
        f"- Hold horizon: `{args.hold_minutes}` minutes",
        f"- Fee assumption: `{args.fee_bps_roundtrip:.1f}` bps round trip",
        f"- Candidate filter: `bybit_discount_bps >= {args.min_discount_bps:.1f}`",
        f"- Train/Test cutoff: `{cutoff.isoformat()}`",
        "",
        "## Dataset",
        "",
        f"- Raw rows: `{len(df):,}`",
        f"- Candidate rows: `{len(candidates):,}`",
        f"- Train candidates: `{len(train):,}`",
        f"- Test candidates: `{len(test):,}`",
        f"- Symbols: `{candidates['symbol'].nunique()}`",
        f"- Train positive-after-fee rate: `{train['target_after_fee'].mean() * 100:.2f}%`",
        f"- Test positive-after-fee rate: `{test['target_after_fee'].mean() * 100:.2f}%`",
        "",
        "## Model",
        "",
        f"- Train ROC AUC: `{train_auc:.4f}`",
        f"- Test ROC AUC: `{test_auc:.4f}`",
        f"- State sleeve threshold: `{state_threshold:.4f}`",
        f"- Model sleeve threshold: `{model_threshold:.4f}`",
        "",
        "## Test Summary",
        "",
    ]
    for row in summary[summary["split"] == "test"].itertuples(index=False):
        report_lines.append(
            f"- `{row.variant}`: trades={int(row.trades):,}, net/trade={row.net_bps_per_trade:+.2f} bps, "
            f"win_rate={row.win_rate * 100:.1f}%, after_fee_hit={row.after_fee_hit_rate * 100:.1f}%, "
            f"gap_close={row.mean_gap_close_bps:+.2f} bps, total_net={row.total_net_bps:+.1f} bps"
        )
    report_lines.extend(
        [
            "",
            "## Top Positive Feature Weights",
            "",
        ]
    )
    for row in weights.head(8).itertuples(index=False):
        report_lines.append(f"- `{row.feature}`: `{row.weight:+.4f}`")
    report_lines.extend(
        [
            "",
            "## Top Negative Feature Weights",
            "",
        ]
    )
    for row in weights.tail(8).sort_values("weight").itertuples(index=False):
        report_lines.append(f"- `{row.feature}`: `{row.weight:+.4f}`")

    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    bucket_df.to_csv(OUT_DIR / f"{prefix}_score_buckets.csv", index=False)
    weights.to_csv(OUT_DIR / f"{prefix}_feature_weights.csv", index=False)
    monthly.to_csv(OUT_DIR / f"{prefix}_monthly_test.csv", index=False)
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    print(f"candidate_rows={len(candidates):,} train={len(train):,} test={len(test):,} cutoff={cutoff.isoformat()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
