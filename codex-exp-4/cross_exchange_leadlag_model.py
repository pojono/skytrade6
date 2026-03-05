#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"


def load_day(symbol: str, day: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for window in [1, 3, 5, 10]:
        df[f"bn_ret_{window}s"] = df["bn_last_price"].pct_change(window)
        df[f"bb_ret_{window}s"] = df["bb_last_price"].pct_change(window)
        df[f"rel_ret_{window}s"] = df[f"bn_ret_{window}s"] - df[f"bb_ret_{window}s"]

    df["cross_gap_bps"] = (df["bn_last_price"] - df["bb_last_price"]) / df["mid_px"] * 10000.0
    df["gap_chg_bps_1s"] = df["cross_gap_bps"].diff(1)
    df["gap_chg_bps_3s"] = df["cross_gap_bps"].diff(3)
    df["lead_pressure_short"] = df["rel_ret_1s"] + 0.5 * df["rel_ret_3s"]
    df["lead_pressure_med"] = df["rel_ret_3s"] + 0.5 * df["rel_ret_5s"]
    df["bn_flow_ratio_30s"] = df["combo_flow_30s"] / (df["bn_notional"] + df["bb_notional"]).rolling(30).sum()
    df["bn_flow_ratio_60s"] = df["combo_flow_60s"] / (df["bn_notional"] + df["bb_notional"]).rolling(60).sum()
    df["bb_spread_norm"] = df["bb_spread_bps"] / df["bb_spread_bps"].rolling(300).mean()
    df["target_30s"] = df["future_ret_30s"]
    df["target_60s"] = df["future_ret_60s"]
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


FEATURE_COLS = [
    "bn_ret_1s",
    "bb_ret_1s",
    "bn_ret_3s",
    "bb_ret_3s",
    "bn_ret_5s",
    "bb_ret_5s",
    "bn_ret_10s",
    "bb_ret_10s",
    "rel_ret_1s",
    "rel_ret_3s",
    "rel_ret_5s",
    "rel_ret_10s",
    "cross_gap_bps",
    "gap_chg_bps_1s",
    "gap_chg_bps_3s",
    "lead_pressure_short",
    "lead_pressure_med",
    "bn_flow_ratio_30s",
    "bn_flow_ratio_60s",
    "bb_spread_norm",
]


def choose_threshold(train_pred: pd.Series, train_target: pd.Series, fee_bps: float) -> tuple[float, int, dict[str, float]] | None:
    candidates: list[tuple[float, int, dict[str, float]]] = []
    abs_pred = train_pred.abs()
    for q in [0.90, 0.95, 0.97, 0.98, 0.99]:
        threshold = float(abs_pred.quantile(q))
        mask = abs_pred >= threshold
        if mask.sum() < 30:
            continue
        signal = np.where(train_pred[mask] >= 0, 1.0, -1.0)
        signed = train_target[mask] * signal
        gross_bps = float(signed.mean() * 10000.0)
        stats = {
            "count": float(mask.sum()),
            "gross_bps": gross_bps,
            "net_bps": gross_bps - fee_bps,
            "win_rate": float((signed > 0).mean()),
        }
        candidates.append((threshold, 0, stats))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[2]["net_bps"], x[2]["count"]), reverse=True)
    best = candidates[0]
    return best


def score_predictions(pred: pd.Series, target: pd.Series, threshold: float, fee_bps: float) -> dict[str, float] | None:
    mask = pred.abs() >= threshold
    if mask.sum() < 30:
        return None
    signal = np.where(pred[mask] >= 0, 1.0, -1.0)
    signed = target[mask] * signal
    gross_bps = float(signed.mean() * 10000.0)
    return {
        "count": float(mask.sum()),
        "gross_bps": gross_bps,
        "net_bps": gross_bps - fee_bps,
        "win_rate": float((signed > 0).mean()),
    }


def walkforward_symbol(symbol: str, days: list[str], train_window_days: int, fee_bps: float, horizon_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = {day: add_features(load_day(symbol, day)) for day in days}
    target_col = f"target_{horizon_s}s"
    fold_rows: list[dict[str, float | str]] = []

    for idx in range(train_window_days, len(days)):
        train_days = days[idx - train_window_days : idx]
        test_day = days[idx]
        train = pd.concat([prepared[d] for d in train_days], ignore_index=True)
        test = prepared[test_day]

        model = Ridge(alpha=1.0)
        model.fit(train[FEATURE_COLS], train[target_col])
        train_pred = pd.Series(model.predict(train[FEATURE_COLS]), index=train.index)
        test_pred = pd.Series(model.predict(test[FEATURE_COLS]), index=test.index)

        threshold_pick = choose_threshold(train_pred, train[target_col], fee_bps)
        if threshold_pick is None:
            fold_rows.append(
                {
                    "symbol": symbol,
                    "train_days": ",".join(train_days),
                    "test_day": test_day,
                    "threshold_abs_pred": None,
                    "train_count": 0,
                    "train_net_bps": None,
                    "test_count": 0,
                    "test_net_bps": None,
                }
            )
            continue

        threshold, _, train_stats = threshold_pick
        test_stats = score_predictions(test_pred, test[target_col], threshold, fee_bps)
        fold_rows.append(
            {
                "symbol": symbol,
                "train_days": ",".join(train_days),
                "test_day": test_day,
                "threshold_abs_pred": threshold,
                "train_count": int(train_stats["count"]),
                "train_net_bps": float(train_stats["net_bps"]),
                "train_win_rate": float(train_stats["win_rate"]),
                "test_count": int(test_stats["count"]) if test_stats else 0,
                "test_net_bps": float(test_stats["net_bps"]) if test_stats else None,
                "test_win_rate": float(test_stats["win_rate"]) if test_stats else None,
            }
        )

    folds = pd.DataFrame(fold_rows)
    usable = folds.dropna(subset=["test_net_bps"]).copy()
    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "horizon_s": horizon_s,
                "folds": int(len(usable)),
                "mean_test_net_bps": float(usable["test_net_bps"].mean()) if not usable.empty else None,
                "median_test_net_bps": float(usable["test_net_bps"].median()) if not usable.empty else None,
                "positive_test_folds": int((usable["test_net_bps"] > 0).sum()) if not usable.empty else 0,
                "positive_test_rate": float((usable["test_net_bps"] > 0).mean()) if not usable.empty else None,
                "mean_test_trades": float(usable["test_count"].mean()) if not usable.empty else None,
            }
        ]
    )
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward cross-exchange lead/lag model.")
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
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--horizon-s", type=int, default=30, choices=[30, 60])
    args = parser.parse_args()

    all_folds = []
    all_summaries = []
    for symbol in args.symbols:
        folds, summary = walkforward_symbol(
            symbol,
            days=args.days,
            train_window_days=args.train_window_days,
            fee_bps=args.fee_bps_roundtrip,
            horizon_s=args.horizon_s,
        )
        all_folds.append(folds)
        all_summaries.append(summary)
        s = summary.iloc[0]
        print(
            f"{symbol}: folds={int(s['folds'])} mean_test_net={s['mean_test_net_bps']:.2f}bps "
            f"median_test_net={s['median_test_net_bps']:.2f}bps positive_rate={s['positive_test_rate']:.2%}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    folds_df.to_csv(OUT_DIR / "cross_exchange_leadlag_folds.csv", index=False)
    summary_df.to_csv(OUT_DIR / "cross_exchange_leadlag_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'cross_exchange_leadlag_folds.csv'}")
    print(f"wrote {OUT_DIR / 'cross_exchange_leadlag_summary.csv'}")


if __name__ == "__main__":
    main()
