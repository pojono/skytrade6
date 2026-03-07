#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from analyze_event_microstructure import BASE_FEATURES, MICRO_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"

DEFAULT_EVENTS = OUT_DIR / "event_microstructure_active_union_2025sep_2026mar.csv"
DEFAULT_ACTIVE = OUT_DIR / "rolling_universe_micro_2025sep_2026mar_active_symbols.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast trade-based execution rolling backtest.")
    parser.add_argument("--events-input", default=str(DEFAULT_EVENTS))
    parser.add_argument("--active-input", default=str(DEFAULT_ACTIVE))
    parser.add_argument("--selection-quantile", type=float, default=0.6)
    parser.add_argument("--hold-minutes", type=int, default=15)
    parser.add_argument("--entry-delay-ms", type=int, default=50)
    parser.add_argument("--base-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--bybit-spread-cross-mult", type=float, default=1.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=4)
    parser.add_argument("--max-daily-trades", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-prefix", default="fast_trade_execution_rolling_backtest")
    return parser.parse_args()


def add_symbol_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sym = pd.get_dummies(df["symbol"], prefix="sym", dtype=float)
    out = pd.concat([df.reset_index(drop=True), sym.reset_index(drop=True)], axis=1)
    return out, list(sym.columns)


def forward_lookup(source: pd.DataFrame, query_ms: pd.Series, price_col: str, ts_col: str) -> pd.DataFrame:
    query = pd.DataFrame({"query_ms": query_ms.astype("int64")}).sort_values("query_ms")
    merged = pd.merge_asof(
        query,
        source[[ts_col, price_col]].sort_values(ts_col),
        left_on="query_ms",
        right_on=ts_col,
        direction="forward",
    )
    return merged.sort_values("query_ms").reset_index(drop=True)


def load_bybit_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, usecols=["timestamp", "price"])
    trades["bb_trade_ms"] = np.floor(pd.to_numeric(trades["timestamp"], errors="coerce") * 1000.0).astype("int64")
    trades["bb_trade_price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades = trades[["bb_trade_ms", "bb_trade_price"]].dropna().sort_values("bb_trade_ms").reset_index(drop=True)
    return trades


def load_binance_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, usecols=["time", "price"])
    trades["bn_trade_ms"] = pd.to_numeric(trades["time"], errors="coerce").astype("int64")
    trades["bn_trade_price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades = trades[["bn_trade_ms", "bn_trade_price"]].dropna().sort_values("bn_trade_ms").reset_index(drop=True)
    return trades


def build_symbol_day_fast_fills(
    symbol: str,
    day: str,
    subset: pd.DataFrame,
    hold_minutes: int,
    entry_delay_ms: int,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{day}_fast_trade_exec.csv.gz"
    event_ms = (subset["ts"].astype("int64") // 10**6).astype("int64")

    if cache_path.exists() and not force_rebuild:
        cached = pd.read_csv(cache_path, parse_dates=["ts"])
        cached = cached[cached["ts"].isin(subset["ts"])].copy()
        if len(cached) == len(subset):
            return cached

    bb_trade_path = DATALAKE / "bybit" / symbol / f"{day}_trades.csv.gz"
    bn_trade_path = DATALAKE / "binance" / symbol / f"{day}_trades.csv.gz"
    if not (bb_trade_path.exists() and bn_trade_path.exists()):
        raise FileNotFoundError(f"missing trade files for {symbol} {day}")

    bb = load_bybit_trades(bb_trade_path)
    bn = load_binance_trades(bn_trade_path)

    hold_ms = hold_minutes * 60 * 1000
    entry_query_ms = event_ms + entry_delay_ms
    exit_query_ms = event_ms + hold_ms + entry_delay_ms

    bb_entry = forward_lookup(bb, entry_query_ms, "bb_trade_price", "bb_trade_ms")
    bb_exit = forward_lookup(bb, exit_query_ms, "bb_trade_price", "bb_trade_ms")
    bn_entry = forward_lookup(bn, entry_query_ms, "bn_trade_price", "bn_trade_ms")
    bn_exit = forward_lookup(bn, exit_query_ms, "bn_trade_price", "bn_trade_ms")

    out = subset[["symbol", "ts", "date"]].copy().reset_index(drop=True)
    out["entry_query_ms"] = entry_query_ms.reset_index(drop=True)
    out["exit_query_ms"] = exit_query_ms.reset_index(drop=True)
    out["bb_entry_ms"] = bb_entry["bb_trade_ms"]
    out["bb_entry_price"] = bb_entry["bb_trade_price"]
    out["bn_entry_ms"] = bn_entry["bn_trade_ms"]
    out["bn_entry_price"] = bn_entry["bn_trade_price"]
    out["bb_exit_ms"] = bb_exit["bb_trade_ms"]
    out["bb_exit_price"] = bb_exit["bb_trade_price"]
    out["bn_exit_ms"] = bn_exit["bn_trade_ms"]
    out["bn_exit_price"] = bn_exit["bn_trade_price"]
    out["entry_fill_lag_ms"] = np.maximum(out["bb_entry_ms"], out["bn_entry_ms"]) - out["entry_query_ms"]
    out["exit_fill_lag_ms"] = np.maximum(out["bb_exit_ms"], out["bn_exit_ms"]) - out["exit_query_ms"]
    out["fast_entry_gap_bps"] = 10000.0 * (out["bn_entry_price"] / out["bb_entry_price"] - 1.0)
    out["fast_exit_gap_bps"] = 10000.0 * (out["bn_exit_price"] / out["bb_exit_price"] - 1.0)
    out["fast_gap_close_15m_bps"] = out["fast_entry_gap_bps"] - out["fast_exit_gap_bps"]
    out.to_csv(cache_path, index=False, compression="gzip")
    return out


def enrich_fast_execution(events: pd.DataFrame, hold_minutes: int, entry_delay_ms: int, workers: int) -> pd.DataFrame:
    groups = [subset.copy() for _, subset in events.groupby(["symbol", "date"], sort=True)]
    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for subset in groups:
            frames.append(
                build_symbol_day_fast_fills(
                    str(subset["symbol"].iloc[0]),
                    str(subset["date"].iloc[0]),
                    subset,
                    hold_minutes=hold_minutes,
                    entry_delay_ms=entry_delay_ms,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    build_symbol_day_fast_fills,
                    str(subset["symbol"].iloc[0]),
                    str(subset["date"].iloc[0]),
                    subset,
                    hold_minutes,
                    entry_delay_ms,
                )
                for subset in groups
            ]
            for future in futures:
                frames.append(future.result())
    fills = pd.concat(frames, ignore_index=True)
    return events.merge(fills, on=["symbol", "ts", "date"], how="left")


def add_execution_columns(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = events.copy()
    df["fast_pair_net_15m_bps"] = (
        df["fast_gap_close_15m_bps"]
        - args.base_fee_bps_roundtrip
        - args.bybit_spread_cross_mult * df["bb_spread_bps_ob"].clip(lower=0.0).fillna(0.0)
    )
    df["fast_pair_win_15m"] = (df["fast_pair_net_15m_bps"] > 0).astype(int)
    df["fast_entry_decay_bps"] = df["gap_bps"] - df["fast_entry_gap_bps"]
    return df.replace([np.inf, -np.inf], np.nan)


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "events": 0.0,
            "fast_pair_net_15m_bps": np.nan,
            "fast_gap_close_15m_bps": np.nan,
            "fast_pair_win_15m": np.nan,
            "fast_entry_decay_bps": np.nan,
            "bb_spread_bps_ob": np.nan,
            "entry_fill_lag_ms": np.nan,
            "exit_fill_lag_ms": np.nan,
        }
    return {
        "events": float(len(frame)),
        "fast_pair_net_15m_bps": float(frame["fast_pair_net_15m_bps"].mean()),
        "fast_gap_close_15m_bps": float(frame["fast_gap_close_15m_bps"].mean()),
        "fast_pair_win_15m": float(frame["fast_pair_win_15m"].mean()),
        "fast_entry_decay_bps": float(frame["fast_entry_decay_bps"].mean()),
        "bb_spread_bps_ob": float(frame["bb_spread_bps_ob"].mean()),
        "entry_fill_lag_ms": float(frame["entry_fill_lag_ms"].mean()),
        "exit_fill_lag_ms": float(frame["exit_fill_lag_ms"].mean()),
    }


def apply_portfolio_constraints(
    frame: pd.DataFrame,
    score_col: str,
    hold_minutes: int,
    entry_delay_ms: int,
    max_concurrent_positions: int,
    max_daily_trades: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(["ts", score_col, "gap_bps"], ascending=[True, False, False]).reset_index(drop=True)
    active_positions: list[dict[str, object]] = []
    accepted_rows: list[int] = []
    trades_taken = 0
    hold_delta = pd.Timedelta(minutes=hold_minutes)
    entry_delay = pd.Timedelta(milliseconds=entry_delay_ms)

    for row in ordered.itertuples():
        entry_ts = row.ts + entry_delay
        exit_ts = entry_ts + hold_delta
        active_positions = [pos for pos in active_positions if pos["exit_ts"] > entry_ts]
        if trades_taken >= max_daily_trades:
            continue
        if len(active_positions) >= max_concurrent_positions:
            continue
        if any(pos["symbol"] == row.symbol for pos in active_positions):
            continue
        accepted_rows.append(row.Index)
        active_positions.append({"symbol": row.symbol, "exit_ts": exit_ts})
        trades_taken += 1
    return ordered.loc[accepted_rows].copy()


def summarize_variant(rows: list[dict[str, float | str]], variant: str) -> dict[str, float | str]:
    frame = pd.DataFrame([row for row in rows if row["variant"] == variant])
    return {
        "variant": variant,
        "folds": int(len(frame)),
        "mean_events": float(frame["events"].mean()),
        "median_events": float(frame["events"].median()),
        "mean_fast_pair_net_15m_bps": float(frame["fast_pair_net_15m_bps"].mean()),
        "median_fast_pair_net_15m_bps": float(frame["fast_pair_net_15m_bps"].median()),
        "mean_fast_gap_close_15m_bps": float(frame["fast_gap_close_15m_bps"].mean()),
        "mean_fast_pair_win_15m": float(frame["fast_pair_win_15m"].mean()),
        "mean_fast_entry_decay_bps": float(frame["fast_entry_decay_bps"].mean()),
        "mean_entry_fill_lag_ms": float(frame["entry_fill_lag_ms"].mean()),
        "positive_folds": int((frame["fast_pair_net_15m_bps"] > 0).sum()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_input, parse_dates=["ts"]).replace([np.inf, -np.inf], np.nan)
    active = pd.read_csv(args.active_input)
    if events.empty or active.empty:
        raise SystemExit("missing events or active universe input")

    events = enrich_fast_execution(events, hold_minutes=args.hold_minutes, entry_delay_ms=args.entry_delay_ms, workers=args.workers)
    events = add_execution_columns(events, args)
    events = events.dropna(subset=["fast_pair_net_15m_bps"]).copy()
    if events.empty:
        raise SystemExit("no events survived fast execution target construction")

    events, dummy_cols = add_symbol_dummies(events)
    feature_cols = BASE_FEATURES + MICRO_FEATURES + dummy_cols
    state_cols = ["gap_z_240", "premium_gap_z_240", "crowding_gap_z_240", "oi_gap_30m_z_240"]

    fold_rows: list[dict[str, float | str]] = []
    importance_rows: list[dict[str, float | str]] = []

    for test_day, active_part in active.groupby("test_day", sort=True):
        active_symbols = active_part["symbol"].tolist()
        train_start_day = active_part["train_start_day"].iloc[0]
        train_end_day = active_part["train_end_day"].iloc[0]
        train_days = pd.date_range(train_start_day, train_end_day, freq="D").strftime("%Y-%m-%d").tolist()

        train = events[(events["symbol"].isin(active_symbols)) & (events["date"].isin(train_days))].copy()
        test = events[(events["symbol"].isin(active_symbols)) & (events["date"] == test_day)].copy()
        if train.empty or test.empty:
            continue

        usable_feature_cols = [col for col in feature_cols if train[col].notna().any()]
        if not usable_feature_cols:
            continue

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(train[usable_feature_cols], train["fast_pair_net_15m_bps"])
        train["model_score"] = model.predict(train[usable_feature_cols])
        test["model_score"] = model.predict(test[usable_feature_cols])
        train["state_score"] = train[state_cols].fillna(0.0).mean(axis=1)
        test["state_score"] = test[state_cols].fillna(0.0).mean(axis=1)

        model_threshold = train["model_score"].quantile(args.selection_quantile)
        state_threshold = train["state_score"].quantile(args.selection_quantile)

        variants = {
            "active_fast_exec_baseline": apply_portfolio_constraints(
                test,
                score_col="gap_bps",
                hold_minutes=args.hold_minutes,
                entry_delay_ms=args.entry_delay_ms,
                max_concurrent_positions=args.max_concurrent_positions,
                max_daily_trades=args.max_daily_trades,
            ),
            "active_fast_exec_state_top_quantile": apply_portfolio_constraints(
                test[test["state_score"] >= state_threshold],
                score_col="state_score",
                hold_minutes=args.hold_minutes,
                entry_delay_ms=args.entry_delay_ms,
                max_concurrent_positions=args.max_concurrent_positions,
                max_daily_trades=args.max_daily_trades,
            ),
            "active_fast_exec_model_top_quantile": apply_portfolio_constraints(
                test[test["model_score"] >= model_threshold],
                score_col="model_score",
                hold_minutes=args.hold_minutes,
                entry_delay_ms=args.entry_delay_ms,
                max_concurrent_positions=args.max_concurrent_positions,
                max_daily_trades=args.max_daily_trades,
            ),
        }

        for variant, subset in variants.items():
            stats = evaluate(subset)
            stats["test_day"] = test_day
            stats["train_start_day"] = train_start_day
            stats["train_end_day"] = train_end_day
            stats["variant"] = variant
            fold_rows.append(stats)

        for feature, importance in zip(usable_feature_cols, model.named_steps["rf"].feature_importances_):
            importance_rows.append({"test_day": test_day, "feature": feature, "importance": float(importance)})

    folds = pd.DataFrame(fold_rows)[
        [
            "test_day",
            "train_start_day",
            "train_end_day",
            "variant",
            "events",
            "fast_pair_net_15m_bps",
            "fast_gap_close_15m_bps",
            "fast_pair_win_15m",
            "fast_entry_decay_bps",
            "bb_spread_bps_ob",
            "entry_fill_lag_ms",
            "exit_fill_lag_ms",
        ]
    ]
    summary = pd.DataFrame(
        [
            summarize_variant(fold_rows, "active_fast_exec_baseline"),
            summarize_variant(fold_rows, "active_fast_exec_state_top_quantile"),
            summarize_variant(fold_rows, "active_fast_exec_model_top_quantile"),
        ]
    )
    importance = (
        pd.DataFrame(importance_rows)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    monthly = (
        folds.assign(month=lambda x: pd.to_datetime(x["test_day"]).dt.to_period("M").astype(str))
        .groupby(["month", "variant"], as_index=False)
        .agg(
            folds=("test_day", "size"),
            mean_events=("events", "mean"),
            mean_fast_pair_net_15m_bps=("fast_pair_net_15m_bps", "mean"),
            mean_fast_pair_win_15m=("fast_pair_win_15m", "mean"),
        )
    )

    report_lines = [
        "# Fast Trade Execution Rolling Backtest",
        "",
        f"- Events input: `{args.events_input}`",
        f"- Active set input: `{args.active_input}`",
        f"- Selection quantile: `{args.selection_quantile:.2f}`",
        f"- Hold minutes: `{args.hold_minutes}`",
        f"- Entry delay ms: `{args.entry_delay_ms}`",
        f"- Base fee assumption: `{args.base_fee_bps_roundtrip:.2f}` bps",
        f"- Bybit spread cross multiplier: `{args.bybit_spread_cross_mult:.2f}`",
        f"- Max concurrent positions: `{args.max_concurrent_positions}`",
        f"- Max daily trades: `{args.max_daily_trades}`",
        f"- Folds: `{folds['test_day'].nunique()}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.variant}`: folds={row.folds}, mean_events={row.mean_events:.1f}, "
            f"mean_fast_net_15m={row.mean_fast_pair_net_15m_bps:+.2f} bps, "
            f"mean_fast_gap_close_15m={row.mean_fast_gap_close_15m_bps:+.2f} bps, "
            f"mean_fast_win_15m={row.mean_fast_pair_win_15m * 100:.1f}%, "
            f"mean_entry_decay={row.mean_fast_entry_decay_bps:.2f} bps, "
            f"mean_entry_fill_lag={row.mean_entry_fill_lag_ms:.1f} ms, "
            f"positive_folds={row.positive_folds}/{row.folds}"
        )
    report_lines.extend(["", "## Top Mean Feature Importances", ""])
    for row in importance.head(15).itertuples(index=False):
        report_lines.append(f"- `{row.feature}`: `{row.importance:.4f}`")

    prefix = args.output_prefix
    folds.to_csv(OUT_DIR / f"{prefix}_folds.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    monthly.to_csv(OUT_DIR / f"{prefix}_monthly.csv", index=False)
    importance.to_csv(OUT_DIR / f"{prefix}_feature_importance.csv", index=False)
    (OUT_DIR / f"{prefix}_report.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    print(summary.to_string(index=False))
    print()
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
