#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"


BASE_FEATURES = [
    "bb_spread_bps",
    "bb_top5_pull_pressure_5s",
    "bb_top5_pull_pressure_15s",
    "combo_flow_60s",
    "bb_top20_imbalance",
    "bb_mid_gap_bps",
]


def load_joined(symbol: str, day: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached microstructure file: {path}")
    return pd.read_csv(path)


def add_normalized_features(df: pd.DataFrame, lookback: int = 600) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES:
        mean = df[col].rolling(lookback).mean()
        std = df[col].rolling(lookback).std()
        df[f"{col}_z"] = (df[col] - mean) / std
    df["abs_flow_z"] = df["combo_flow_60s_z"].abs()
    df["abs_gap_z"] = df["bb_mid_gap_bps_z"].abs()
    df["abs_pull_z"] = df["bb_top5_pull_pressure_5s_z"].abs()
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_regime_mask(df: pd.DataFrame, train_ref: pd.DataFrame, regime: str) -> pd.Series:
    if regime == "all":
        return pd.Series(True, index=df.index)
    if regime == "high_abs_flow":
        threshold = float(train_ref["abs_flow_z"].quantile(0.70))
        return df["abs_flow_z"] >= threshold
    if regime == "wide_spread":
        threshold = float(train_ref["bb_spread_bps_z"].quantile(0.70))
        return df["bb_spread_bps_z"] >= threshold
    if regime == "extreme_pull":
        threshold = float(train_ref["abs_pull_z"].quantile(0.70))
        return df["abs_pull_z"] >= threshold
    if regime == "high_abs_gap":
        threshold = float(train_ref["abs_gap_z"].quantile(0.70))
        return df["abs_gap_z"] >= threshold
    raise ValueError(f"Unknown regime: {regime}")


def cooldown_indices(df: pd.DataFrame, mask: pd.Series, cooldown_s: int) -> list[int]:
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return []
    secs = df["sec"].to_numpy()
    kept: list[int] = []
    last_sec = -10**12
    for i in idx:
        sec = int(secs[i])
        if sec - last_sec >= cooldown_s:
            kept.append(int(i))
            last_sec = sec
    return kept


def event_stats(df: pd.DataFrame, event_idx: list[int], horizon_s: int, signal: int, fee_bps: float) -> dict[str, float] | None:
    if len(event_idx) < 30:
        return None
    sample = df.iloc[event_idx]
    signed = sample[f"future_ret_{horizon_s}s"] * signal
    gross = signed.mean() * 10000.0
    return {
        "count": float(len(sample)),
        "gross_bps": float(gross),
        "net_bps": float(gross - fee_bps),
        "win_rate": float((signed > 0).mean()),
    }


def scan_symbol(symbol: str, train_day: str, test_day: str, fee_bps: float, cooldown_s: int) -> pd.DataFrame:
    train = add_normalized_features(load_joined(symbol, train_day))
    test = add_normalized_features(load_joined(symbol, test_day))

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
                    # Train decides direction; test must use the same event and direction.
                    signal = 1 if train.iloc[idx_train][f"future_ret_{horizon_s}s"].mean() >= 0 else -1
                    tr = event_stats(train, idx_train, horizon_s, signal, fee_bps)
                    te = event_stats(test, idx_test, horizon_s, signal, fee_bps)
                    if tr is None or te is None:
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "feature": feature,
                            "tail": tail,
                            "quantile": q,
                            "regime": regime,
                            "threshold_z": threshold,
                            "signal": "long" if signal > 0 else "short",
                            "horizon_s": horizon_s,
                            "cooldown_s": cooldown_s,
                            "train_count": tr["count"],
                            "train_gross_bps": tr["gross_bps"],
                            "train_net_bps": tr["net_bps"],
                            "train_win_rate": tr["win_rate"],
                            "test_count": te["count"],
                            "test_gross_bps": te["gross_bps"],
                            "test_net_bps": te["net_bps"],
                            "test_win_rate": te["win_rate"],
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan event-based regime-conditioned microstructure edges.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT"])
    parser.add_argument("--train-day", default="2026-03-02")
    parser.add_argument("--test-day", default="2026-03-03")
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--cooldown-s", type=int, default=90)
    args = parser.parse_args()

    frames = []
    for symbol in args.symbols:
        try:
            res = scan_symbol(symbol, args.train_day, args.test_day, args.fee_bps_roundtrip, args.cooldown_s)
        except FileNotFoundError as exc:
            print(f"skip {symbol}: {exc}")
            continue
        if res.empty:
            print(f"skip {symbol}: no event definitions met minimum sample counts")
            continue
        frames.append(res)
        top = res.sort_values(["test_net_bps", "train_net_bps"], ascending=[False, False]).iloc[0]
        print(
            f"{symbol}: best={top['feature']} {top['tail']} regime={top['regime']} "
            f"{top['signal']} h={int(top['horizon_s'])}s test_net={top['test_net_bps']:.2f}bps "
            f"train_net={top['train_net_bps']:.2f}bps"
        )

    if not frames:
        raise SystemExit("No results.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.concat(frames, ignore_index=True)
    leaderboard = leaderboard.sort_values(
        ["test_net_bps", "train_net_bps", "test_win_rate", "test_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(OUT_DIR / "event_regime_rule_leaderboard.csv", index=False)
    robust = leaderboard[
        (leaderboard["train_net_bps"] > 0)
        & (leaderboard["test_net_bps"] > 0)
        & (leaderboard["train_count"] >= 30)
        & (leaderboard["test_count"] >= 30)
    ]
    (robust if not robust.empty else leaderboard.head(20)).to_csv(
        OUT_DIR / "event_regime_top_candidates.csv", index=False
    )
    print(f"wrote {OUT_DIR / 'event_regime_rule_leaderboard.csv'}")
    print(f"wrote {OUT_DIR / 'event_regime_top_candidates.csv'}")


if __name__ == "__main__":
    main()
