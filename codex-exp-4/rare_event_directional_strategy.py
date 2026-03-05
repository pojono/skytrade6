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

FEATURES = [
    "bb_spread_bps_z",
    "bb_top5_pull_pressure_5s_z",
    "bb_top5_pull_pressure_15s_z",
    "bb_top20_imbalance_z",
    "bb_mid_gap_bps_z",
    "combo_flow_30s_z",
    "combo_flow_60s_z",
    "abs_spread_z",
    "abs_pull5_z",
    "abs_gap_z",
    "abs_flow60_z",
    "shock_score",
]


def load_joined(symbol: str, day: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return pd.read_csv(path)


def prepare(df: pd.DataFrame, horizon_s: int) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_COLS:
        mean = df[col].rolling(300).mean()
        std = df[col].rolling(300).std()
        df[f"{col}_z"] = (df[col] - mean) / std
    df["abs_spread_z"] = df["bb_spread_bps_z"].abs()
    df["abs_pull5_z"] = df["bb_top5_pull_pressure_5s_z"].abs()
    df["abs_gap_z"] = df["bb_mid_gap_bps_z"].abs()
    df["abs_flow60_z"] = df["combo_flow_60s_z"].abs()
    df["shock_score"] = df[["abs_spread_z", "abs_pull5_z", "abs_gap_z", "abs_flow60_z"]].max(axis=1)

    df["bb_buy_taker_notional"] = (df["bb_notional"] + df["bb_signed_notional"]) / 2.0
    df["bb_sell_taker_notional"] = (df["bb_notional"] - df["bb_signed_notional"]) / 2.0

    future_buy = sum(df["bb_buy_taker_notional"].shift(-k) for k in range(1, 6))
    future_sell = sum(df["bb_sell_taker_notional"].shift(-k) for k in range(1, 6))
    df["future_buy_5s"] = future_buy
    df["future_sell_5s"] = future_sell

    df["bid_queue_notional"] = df["bb_best_bid_px"] * df["bb_best_bid_sz"]
    df["ask_queue_notional"] = df["bb_best_ask_px"] * df["bb_best_ask_sz"]
    df["future_mid_px"] = df["mid_px"] * (1.0 + df[f"future_ret_{horizon_s}s"])
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def extract_events(
    df: pd.DataFrame,
    symbol: str,
    day: str,
    entry_z: float,
    cooldown_s: int,
    horizon_s: int,
    move_bps: float,
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
    ret_bps = sample[f"future_ret_{horizon_s}s"] * 10000.0
    sample["target_dir"] = np.where(ret_bps >= move_bps, 1, np.where(ret_bps <= -move_bps, -1, 0))
    sample["symbol"] = symbol
    sample["day"] = day
    cols = [
        "symbol",
        "day",
        "sec",
        "mid_px",
        "bb_best_bid_px",
        "bb_best_ask_px",
        "future_mid_px",
        "future_buy_5s",
        "future_sell_5s",
        "bid_queue_notional",
        "ask_queue_notional",
        "target_dir",
        *FEATURES,
    ]
    return sample[cols].reset_index(drop=True)


def build_dataset(
    symbols: list[str],
    days: list[str],
    entry_z: float,
    cooldown_s: int,
    horizon_s: int,
    move_bps: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for day in days:
            df = prepare(load_joined(symbol, day), horizon_s=horizon_s)
            ev = extract_events(
                df,
                symbol=symbol,
                day=day,
                entry_z=entry_z,
                cooldown_s=cooldown_s,
                horizon_s=horizon_s,
                move_bps=move_bps,
            )
            if not ev.empty:
                frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fill_prob(
    row: pd.Series,
    side: int,
    queue_scale: float,
    taker_capture: float,
    max_fill_prob: float,
) -> float:
    if side > 0:
        queue = max(float(row["bid_queue_notional"]), 1e-9)
        pressure = max(float(row["future_sell_5s"]), 0.0)
    else:
        queue = max(float(row["ask_queue_notional"]), 1e-9)
        pressure = max(float(row["future_buy_5s"]), 0.0)
    return float(min(max_fill_prob, (pressure * taker_capture) / (queue * queue_scale)))


def realized_pnl_bps(
    row: pd.Series,
    side: int,
    fee_bps: float,
    entry_slip_bps: float,
    adverse_drift_bps: float,
) -> float:
    mid = float(row["mid_px"])
    fut_mid = float(row["future_mid_px"])
    if side > 0:
        entry = float(row["bb_best_bid_px"])
        gross = (fut_mid - entry) / mid * 10000.0
    else:
        entry = float(row["bb_best_ask_px"])
        gross = (entry - fut_mid) / mid * 10000.0
    return gross - fee_bps - entry_slip_bps - adverse_drift_bps


def score_selection(
    df: pd.DataFrame,
    pred_class: np.ndarray,
    pred_prob: pd.Series,
    threshold: float,
    fee_bps: float,
    queue_scale: float,
    taker_capture: float,
    max_fill_prob: float,
    entry_slip_bps: float,
    adverse_drift_bps: float,
) -> dict[str, float] | None:
    mask = (pred_class != 0) & (pred_prob >= threshold)
    if mask.sum() < 10:
        return None
    chosen = df.loc[mask].copy()
    sides = pred_class[mask]
    exp_pnl = []
    hit = 0
    fill_probs = []
    for (_, row), side in zip(chosen.iterrows(), sides):
        fp = fill_prob(
            row,
            int(side),
            queue_scale=queue_scale,
            taker_capture=taker_capture,
            max_fill_prob=max_fill_prob,
        )
        pnl = realized_pnl_bps(
            row,
            int(side),
            fee_bps,
            entry_slip_bps=entry_slip_bps,
            adverse_drift_bps=adverse_drift_bps,
        )
        exp_pnl.append(fp * pnl)
        fill_probs.append(fp)
        if int(row["target_dir"]) == int(side):
            hit += 1
    exp_pnl_s = pd.Series(exp_pnl)
    return {
        "trades": float(len(chosen)),
        "exp_net_bps_per_signal": float(exp_pnl_s.mean()),
        "precision": float(hit / len(chosen)),
        "base_directional_rate": float((chosen["target_dir"] != 0).mean()),
        "fill_weighted_total_bps": float(exp_pnl_s.sum()),
        "avg_fill_prob": float(pd.Series(fill_probs).mean()),
    }


def choose_threshold(
    train: pd.DataFrame,
    pred_class: np.ndarray,
    pred_prob: pd.Series,
    fee_bps: float,
    min_precision: float,
    min_train_exp_net_bps: float,
    queue_scale: float,
    taker_capture: float,
    max_fill_prob: float,
    entry_slip_bps: float,
    adverse_drift_bps: float,
    min_train_trades: int,
) -> tuple[float, dict[str, float]] | None:
    candidates: list[tuple[float, dict[str, float]]] = []
    for q in [0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995]:
        thr = float(pred_prob.quantile(q))
        stats = score_selection(
            train,
            pred_class,
            pred_prob,
            thr,
            fee_bps,
            queue_scale=queue_scale,
            taker_capture=taker_capture,
            max_fill_prob=max_fill_prob,
            entry_slip_bps=entry_slip_bps,
            adverse_drift_bps=adverse_drift_bps,
        )
        if stats is None:
            continue
        if stats["trades"] < min_train_trades:
            continue
        if stats["precision"] < min_precision:
            continue
        if stats["exp_net_bps_per_signal"] < min_train_exp_net_bps:
            continue
        candidates.append((thr, stats))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[1]["exp_net_bps_per_signal"], item[1]["precision"], item[1]["trades"]),
        reverse=True,
    )
    return candidates[0]


def walkforward(
    data: pd.DataFrame,
    days: list[str],
    train_window_days: int,
    fee_bps: float,
    min_precision: float,
    min_train_exp_net_bps: float,
    queue_scale: float,
    taker_capture: float,
    max_fill_prob: float,
    entry_slip_bps: float,
    adverse_drift_bps: float,
    min_train_trades: int,
    min_test_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds: list[dict[str, float | str]] = []
    for idx in range(train_window_days, len(days)):
        train_days = days[idx - train_window_days : idx]
        test_day = days[idx]
        train = data[data["day"].isin(train_days)].reset_index(drop=True)
        test = data[data["day"] == test_day].reset_index(drop=True)
        if len(train) < 100 or len(test) < 20:
            continue

        y_train = train["target_dir"].replace({-1: 0, 0: 1, 1: 2})
        model = LogisticRegression(max_iter=300, multi_class="multinomial", solver="lbfgs")
        model.fit(train[FEATURES], y_train)

        train_proba = model.predict_proba(train[FEATURES])
        test_proba = model.predict_proba(test[FEATURES])

        # Map classes 0,1,2 back to -1,0,1 by strongest direction excluding neutral.
        train_up = pd.Series(train_proba[:, 2], index=train.index)
        train_down = pd.Series(train_proba[:, 0], index=train.index)
        test_up = pd.Series(test_proba[:, 2], index=test.index)
        test_down = pd.Series(test_proba[:, 0], index=test.index)

        train_pred_class = np.where(train_up >= train_down, 1, -1)
        train_pred_prob = pd.Series(np.maximum(train_up, train_down), index=train.index)
        test_pred_class = np.where(test_up >= test_down, 1, -1)
        test_pred_prob = pd.Series(np.maximum(test_up, test_down), index=test.index)

        pick = choose_threshold(
            train,
            train_pred_class,
            train_pred_prob,
            fee_bps,
            min_precision=min_precision,
            min_train_exp_net_bps=min_train_exp_net_bps,
            queue_scale=queue_scale,
            taker_capture=taker_capture,
            max_fill_prob=max_fill_prob,
            entry_slip_bps=entry_slip_bps,
            adverse_drift_bps=adverse_drift_bps,
            min_train_trades=min_train_trades,
        )
        if pick is None:
            continue
        threshold, train_stats = pick
        test_stats = score_selection(
            test,
            test_pred_class,
            test_pred_prob,
            threshold,
            fee_bps,
            queue_scale=queue_scale,
            taker_capture=taker_capture,
            max_fill_prob=max_fill_prob,
            entry_slip_bps=entry_slip_bps,
            adverse_drift_bps=adverse_drift_bps,
        )
        if test_stats is None:
            continue
        if test_stats["trades"] < min_test_trades:
            continue
        folds.append(
            {
                "train_days": ",".join(train_days),
                "test_day": test_day,
                "prob_threshold": threshold,
                "train_trades": int(train_stats["trades"]),
                "train_exp_net_bps_per_signal": train_stats["exp_net_bps_per_signal"],
                "train_precision": train_stats["precision"],
                "train_avg_fill_prob": train_stats["avg_fill_prob"],
                "test_trades": int(test_stats["trades"]),
                "test_exp_net_bps_per_signal": test_stats["exp_net_bps_per_signal"],
                "test_precision": test_stats["precision"],
                "test_avg_fill_prob": test_stats["avg_fill_prob"],
                "test_base_directional_rate": test_stats["base_directional_rate"],
            }
        )

    fold_df = pd.DataFrame(folds)
    if fold_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "folds": 0,
                    "mean_test_exp_net_bps_per_signal": None,
                    "median_test_exp_net_bps_per_signal": None,
                    "positive_test_folds": 0,
                    "positive_test_rate": None,
                    "mean_test_precision": None,
                    "mean_test_trades": None,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "folds": int(len(fold_df)),
                    "mean_test_exp_net_bps_per_signal": float(fold_df["test_exp_net_bps_per_signal"].mean()),
                    "median_test_exp_net_bps_per_signal": float(fold_df["test_exp_net_bps_per_signal"].median()),
                    "positive_test_folds": int((fold_df["test_exp_net_bps_per_signal"] > 0).sum()),
                    "positive_test_rate": float((fold_df["test_exp_net_bps_per_signal"] > 0).mean()),
                    "mean_test_precision": float(fold_df["test_precision"].mean()),
                    "mean_test_trades": float(fold_df["test_trades"].mean()),
                }
            ]
        )
    return fold_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rare-event directional passive-entry strategy simulator.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT", "GUNUSDT"])
    parser.add_argument(
        "--days",
        nargs="*",
        default=["2026-02-22", "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-03"],
    )
    parser.add_argument("--train-window-days", type=int, default=3)
    parser.add_argument("--horizon-s", type=int, default=60, choices=[30, 60, 120])
    parser.add_argument("--move-bps", type=float, default=12.0)
    parser.add_argument("--entry-z", type=float, default=3.0)
    parser.add_argument("--cooldown-s", type=int, default=60)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--min-precision", type=float, default=0.33)
    parser.add_argument("--min-train-exp-net-bps", type=float, default=-999.0)
    parser.add_argument("--queue-scale", type=float, default=1.0)
    parser.add_argument("--taker-capture", type=float, default=1.0)
    parser.add_argument("--max-fill-prob", type=float, default=1.0)
    parser.add_argument("--entry-slip-bps", type=float, default=0.0)
    parser.add_argument("--adverse-drift-bps", type=float, default=0.0)
    parser.add_argument("--min-train-trades", type=int, default=10)
    parser.add_argument("--min-test-trades", type=int, default=10)
    args = parser.parse_args()

    data = build_dataset(
        symbols=args.symbols,
        days=args.days,
        entry_z=args.entry_z,
        cooldown_s=args.cooldown_s,
        horizon_s=args.horizon_s,
        move_bps=args.move_bps,
    )
    if data.empty:
        raise SystemExit("No rare events found.")

    folds, summary = walkforward(
        data,
        days=args.days,
        train_window_days=args.train_window_days,
        fee_bps=args.fee_bps_roundtrip,
        min_precision=args.min_precision,
        min_train_exp_net_bps=args.min_train_exp_net_bps,
        queue_scale=args.queue_scale,
        taker_capture=args.taker_capture,
        max_fill_prob=args.max_fill_prob,
        entry_slip_bps=args.entry_slip_bps,
        adverse_drift_bps=args.adverse_drift_bps,
        min_train_trades=args.min_train_trades,
        min_test_trades=args.min_test_trades,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DIR / "rare_event_directional_dataset.csv", index=False)
    folds.to_csv(OUT_DIR / "rare_event_directional_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "rare_event_directional_summary.csv", index=False)

    s = summary.iloc[0]
    if int(s["folds"]) == 0:
        print(f"events={len(data)} folds=0 no usable walk-forward folds")
    else:
        print(
            f"events={len(data)} folds={int(s['folds'])} "
            f"mean_test_exp_net={s['mean_test_exp_net_bps_per_signal']:.2f}bps "
            f"positive_rate={s['positive_test_rate']:.2%} "
            f"mean_precision={s['mean_test_precision']:.3f}"
        )
    print(f"wrote {OUT_DIR / 'rare_event_directional_dataset.csv'}")
    print(f"wrote {OUT_DIR / 'rare_event_directional_folds.csv'}")
    print(f"wrote {OUT_DIR / 'rare_event_directional_summary.csv'}")


if __name__ == "__main__":
    main()
