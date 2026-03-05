#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bybit_orderbook_edge_scan import build_joined_sec
from event_regime_edge_scan import (
    BASE_FEATURES,
    add_normalized_features,
    build_regime_mask,
    cooldown_indices,
    event_stats,
)


OUT_DIR = Path(__file__).resolve().parent / "out"


def ensure_day(symbol: str, day: str) -> pd.DataFrame:
    return add_normalized_features(build_joined_sec(symbol, day, force_rebuild=False))


def scan_train_vs_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fee_bps: float,
    cooldown_s: int,
) -> pd.DataFrame:
    regimes = ["all", "high_abs_flow", "wide_spread", "extreme_pull", "high_abs_gap"]
    quantiles = [("high", 0.95), ("high", 0.98), ("high", 0.99), ("low", 0.05), ("low", 0.02), ("low", 0.01)]

    rows: list[dict[str, float | str]] = []
    for feature in BASE_FEATURES:
        z_feature = f"{feature}_z"
        train_series = train[z_feature]
        for tail, q in quantiles:
            threshold = float(train_series.quantile(q))
            train_trigger = train_series >= threshold if tail == "high" else train_series <= threshold
            test_trigger = test[z_feature] >= threshold if tail == "high" else test[z_feature] <= threshold
            for regime in regimes:
                train_mask = train_trigger & build_regime_mask(train, train, regime)
                test_mask = test_trigger & build_regime_mask(test, train, regime)
                idx_train = cooldown_indices(train, train_mask, cooldown_s)
                idx_test = cooldown_indices(test, test_mask, cooldown_s)
                if len(idx_train) < 30 or len(idx_test) < 30:
                    continue
                for horizon_s in [60, 120]:
                    signal = 1 if train.iloc[idx_train][f"future_ret_{horizon_s}s"].mean() >= 0 else -1
                    tr = event_stats(train, idx_train, horizon_s, signal, fee_bps)
                    te = event_stats(test, idx_test, horizon_s, signal, fee_bps)
                    if tr is None or te is None:
                        continue
                    rows.append(
                        {
                            "feature": feature,
                            "tail": tail,
                            "quantile": q,
                            "regime": regime,
                            "threshold_z": threshold,
                            "signal": "long" if signal > 0 else "short",
                            "horizon_s": horizon_s,
                            "train_count": tr["count"],
                            "train_net_bps": tr["net_bps"],
                            "train_win_rate": tr["win_rate"],
                            "test_count": te["count"],
                            "test_net_bps": te["net_bps"],
                            "test_win_rate": te["win_rate"],
                        }
                    )
    return pd.DataFrame(rows)


def walkforward_symbol(
    symbol: str,
    days: list[str],
    train_window_days: int,
    fee_bps: float,
    cooldown_s: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = {day: ensure_day(symbol, day) for day in days}
    fold_rows: list[dict[str, float | str]] = []

    for idx in range(train_window_days, len(days)):
        train_days = days[idx - train_window_days : idx]
        test_day = days[idx]
        train = pd.concat([cache[d] for d in train_days], ignore_index=True)
        test = cache[test_day]
        leaderboard = scan_train_vs_test(train, test, fee_bps=fee_bps, cooldown_s=cooldown_s)
        if leaderboard.empty:
            fold_rows.append(
                {
                    "symbol": symbol,
                    "train_days": ",".join(train_days),
                    "test_day": test_day,
                    "selected_feature": "",
                    "selected_regime": "",
                    "selected_signal": "",
                    "selected_horizon_s": 0,
                    "train_count": 0,
                    "train_net_bps": None,
                    "test_count": 0,
                    "test_net_bps": None,
                }
            )
            continue

        positive_train = leaderboard[leaderboard["train_net_bps"] > 0]
        pick_source = positive_train if not positive_train.empty else leaderboard
        chosen = pick_source.sort_values(
            ["train_net_bps", "test_net_bps", "train_count"],
            ascending=[False, False, False],
        ).iloc[0]
        fold_rows.append(
            {
                "symbol": symbol,
                "train_days": ",".join(train_days),
                "test_day": test_day,
                "selected_feature": chosen["feature"],
                "selected_tail": chosen["tail"],
                "selected_quantile": chosen["quantile"],
                "selected_regime": chosen["regime"],
                "selected_signal": chosen["signal"],
                "selected_horizon_s": int(chosen["horizon_s"]),
                "train_count": int(chosen["train_count"]),
                "train_net_bps": float(chosen["train_net_bps"]),
                "test_count": int(chosen["test_count"]),
                "test_net_bps": float(chosen["test_net_bps"]),
                "test_win_rate": float(chosen["test_win_rate"]),
            }
        )

    folds = pd.DataFrame(fold_rows)
    usable = folds.dropna(subset=["test_net_bps"]).copy()
    if usable.empty:
        summary = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "folds": 0,
                    "mean_test_net_bps": None,
                    "median_test_net_bps": None,
                    "positive_test_folds": 0,
                    "positive_test_rate": None,
                    "mean_test_trades": None,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "folds": int(len(usable)),
                    "mean_test_net_bps": float(usable["test_net_bps"].mean()),
                    "median_test_net_bps": float(usable["test_net_bps"].median()),
                    "positive_test_folds": int((usable["test_net_bps"] > 0).sum()),
                    "positive_test_rate": float((usable["test_net_bps"] > 0).mean()),
                    "mean_test_trades": float(usable["test_count"].mean()),
                }
            ]
        )
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for event/regime microstructure rules.")
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
    parser.add_argument("--cooldown-s", type=int, default=90)
    args = parser.parse_args()

    all_folds = []
    all_summaries = []
    for symbol in args.symbols:
        folds, summary = walkforward_symbol(
            symbol,
            days=args.days,
            train_window_days=args.train_window_days,
            fee_bps=args.fee_bps_roundtrip,
            cooldown_s=args.cooldown_s,
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
    folds_df.to_csv(OUT_DIR / "walkforward_event_regime_folds.csv", index=False)
    summary_df.to_csv(OUT_DIR / "walkforward_event_regime_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'walkforward_event_regime_folds.csv'}")
    print(f"wrote {OUT_DIR / 'walkforward_event_regime_summary.csv'}")


if __name__ == "__main__":
    main()
