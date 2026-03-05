#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"


BASE_COLS = [
    "bb_spread_bps",
    "bb_top5_pull_pressure_5s",
    "bb_top5_pull_pressure_15s",
    "bb_top20_imbalance",
    "bb_mid_gap_bps",
    "combo_flow_30s",
    "combo_flow_60s",
]


EVENT_FEATURES = [
    "bb_spread_bps_z",
    "bb_top5_pull_pressure_5s_z",
    "bb_top5_pull_pressure_15s_z",
    "bb_top20_imbalance_z",
    "bb_mid_gap_bps_z",
    "combo_flow_30s_z",
    "combo_flow_60s_z",
    "abs_spread_z",
    "abs_pull5_z",
    "abs_pull15_z",
    "abs_gap_z",
    "abs_flow30_z",
    "abs_flow60_z",
    "shock_score",
]


def load_joined(symbol: str, day: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return pd.read_csv(path)


def normalize(df: pd.DataFrame, lookback: int = 300) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_COLS:
        mean = df[col].rolling(lookback).mean()
        std = df[col].rolling(lookback).std()
        df[f"{col}_z"] = (df[col] - mean) / std

    df["abs_spread_z"] = df["bb_spread_bps_z"].abs()
    df["abs_pull5_z"] = df["bb_top5_pull_pressure_5s_z"].abs()
    df["abs_pull15_z"] = df["bb_top5_pull_pressure_15s_z"].abs()
    df["abs_gap_z"] = df["bb_mid_gap_bps_z"].abs()
    df["abs_flow30_z"] = df["combo_flow_30s_z"].abs()
    df["abs_flow60_z"] = df["combo_flow_60s_z"].abs()
    df["shock_score"] = df[
        ["abs_spread_z", "abs_pull5_z", "abs_gap_z", "abs_flow60_z"]
    ].max(axis=1)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def extract_events(
    df: pd.DataFrame,
    symbol: str,
    day: str,
    danger_horizon_s: int,
    danger_bps: float,
    entry_z: float,
    cooldown_s: int,
) -> pd.DataFrame:
    shock_mask = (
        (df["abs_spread_z"] >= entry_z)
        | (df["abs_pull5_z"] >= entry_z)
        | (df["abs_gap_z"] >= entry_z)
        | (df["abs_flow60_z"] >= entry_z)
    )
    idx = np.flatnonzero(shock_mask.to_numpy())
    keep: list[int] = []
    last_sec = -10**12
    secs = df["sec"].to_numpy()
    for i in idx:
        sec = int(secs[i])
        if sec - last_sec >= cooldown_s:
            keep.append(int(i))
            last_sec = sec
    if not keep:
        return pd.DataFrame()

    sample = df.iloc[keep].copy()
    sample["symbol"] = symbol
    sample["day"] = day
    sample["future_abs_move_bps"] = sample[f"future_ret_{danger_horizon_s}s"].abs() * 10000.0
    sample["danger"] = (sample["future_abs_move_bps"] >= danger_bps).astype(int)
    cols = ["symbol", "day", "sec", "future_abs_move_bps", "danger", *EVENT_FEATURES]
    return sample[cols].reset_index(drop=True)


def build_event_dataset(
    symbols: list[str],
    days: list[str],
    danger_horizon_s: int,
    danger_bps: float,
    entry_z: float,
    cooldown_s: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for day in days:
            df = normalize(load_joined(symbol, day))
            events = extract_events(
                df,
                symbol=symbol,
                day=day,
                danger_horizon_s=danger_horizon_s,
                danger_bps=danger_bps,
                entry_z=entry_z,
                cooldown_s=cooldown_s,
            )
            if not events.empty:
                frames.append(events)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def score_threshold(probs: pd.Series, labels: pd.Series, threshold: float) -> dict[str, float] | None:
    flag = probs >= threshold
    flagged = int(flag.sum())
    positives = int(labels.sum())
    if flagged < 10 or positives == 0:
        return None
    flagged_pos = int(labels[flag].sum())
    precision = flagged_pos / flagged if flagged else 0.0
    recall = flagged_pos / positives if positives else 0.0
    return {
        "flagged": float(flagged),
        "precision": float(precision),
        "recall": float(recall),
        "base_rate": float(labels.mean()),
        "lift": float(precision / labels.mean()) if labels.mean() > 0 else np.nan,
    }


def choose_threshold(train_probs: pd.Series, train_labels: pd.Series) -> tuple[float, dict[str, float]] | None:
    candidates: list[tuple[float, dict[str, float]]] = []
    for q in [0.80, 0.90, 0.95, 0.97, 0.99]:
        thr = float(train_probs.quantile(q))
        stats = score_threshold(train_probs, train_labels, thr)
        if stats is None:
            continue
        if stats["recall"] < 0.10:
            continue
        candidates.append((thr, stats))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[1]["precision"], item[1]["lift"], item[1]["recall"]),
        reverse=True,
    )
    return candidates[0]


def walkforward(
    events: pd.DataFrame,
    days: list[str],
    train_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for idx in range(train_window_days, len(days)):
        train_days = days[idx - train_window_days : idx]
        test_day = days[idx]
        train = events[events["day"].isin(train_days)].reset_index(drop=True)
        test = events[events["day"] == test_day].reset_index(drop=True)
        if len(train) < 50 or len(test) < 20 or train["danger"].sum() < 20:
            continue

        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(train[EVENT_FEATURES], train["danger"])
        train_probs = pd.Series(model.predict_proba(train[EVENT_FEATURES])[:, 1], index=train.index)
        test_probs = pd.Series(model.predict_proba(test[EVENT_FEATURES])[:, 1], index=test.index)

        pick = choose_threshold(train_probs, train["danger"])
        if pick is None:
            continue
        threshold, train_stats = pick
        test_stats = score_threshold(test_probs, test["danger"], threshold)
        if test_stats is None:
            continue
        rows.append(
            {
                "train_days": ",".join(train_days),
                "test_day": test_day,
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "prob_threshold": threshold,
                "train_base_rate": train_stats["base_rate"],
                "train_precision": train_stats["precision"],
                "train_recall": train_stats["recall"],
                "train_lift": train_stats["lift"],
                "train_flagged": int(train_stats["flagged"]),
                "test_base_rate": test_stats["base_rate"],
                "test_precision": test_stats["precision"],
                "test_recall": test_stats["recall"],
                "test_lift": test_stats["lift"],
                "test_flagged": int(test_stats["flagged"]),
            }
        )

    folds = pd.DataFrame(rows)
    if folds.empty:
        summary = pd.DataFrame(
            [
                {
                    "folds": 0,
                    "mean_test_base_rate": None,
                    "mean_test_precision": None,
                    "mean_test_lift": None,
                    "mean_test_recall": None,
                    "mean_test_flagged": None,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "folds": int(len(folds)),
                    "mean_test_base_rate": float(folds["test_base_rate"].mean()),
                    "mean_test_precision": float(folds["test_precision"].mean()),
                    "mean_test_lift": float(folds["test_lift"].mean()),
                    "mean_test_recall": float(folds["test_recall"].mean()),
                    "mean_test_flagged": float(folds["test_flagged"].mean()),
                }
            ]
        )
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision-first rare shock event classifier.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT", "1000PEPEUSDT", "GUNUSDT"])
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
    parser.add_argument("--danger-bps", type=float, default=12.0)
    parser.add_argument("--entry-z", type=float, default=3.0)
    parser.add_argument("--cooldown-s", type=int, default=60)
    args = parser.parse_args()

    events = build_event_dataset(
        symbols=args.symbols,
        days=args.days,
        danger_horizon_s=args.danger_horizon_s,
        danger_bps=args.danger_bps,
        entry_z=args.entry_z,
        cooldown_s=args.cooldown_s,
    )
    if events.empty:
        raise SystemExit("No shock events extracted.")

    folds, summary = walkforward(events, args.days, args.train_window_days)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "rare_event_dataset.csv", index=False)
    folds.to_csv(OUT_DIR / "rare_event_precision_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "rare_event_precision_summary.csv", index=False)

    base_rate = float(events["danger"].mean())
    print(
        f"events={len(events)} base_rate={base_rate:.3f} "
        f"mean_test_precision={summary.iloc[0]['mean_test_precision']:.3f} "
        f"mean_test_lift={summary.iloc[0]['mean_test_lift']:.2f}x"
    )
    print(f"wrote {OUT_DIR / 'rare_event_dataset.csv'}")
    print(f"wrote {OUT_DIR / 'rare_event_precision_folds.csv'}")
    print(f"wrote {OUT_DIR / 'rare_event_precision_summary.csv'}")


if __name__ == "__main__":
    main()
