#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from analyze_event_microstructure import BASE_FEATURES, MICRO_FEATURES


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_EVENTS = OUT_DIR / "event_microstructure_active_union_2025sep_2026mar.csv"
DEFAULT_ACTIVE = OUT_DIR / "rolling_universe_micro_2025sep_2026mar_active_symbols.csv"
DEFAULT_PANEL = OUT_DIR / "dislocation_panel_active_union_2025sep_2026mar.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution-aware no-lookahead rolling event backtest.")
    parser.add_argument("--events-input", default=str(DEFAULT_EVENTS))
    parser.add_argument("--active-input", default=str(DEFAULT_ACTIVE))
    parser.add_argument("--panel-input", default=str(DEFAULT_PANEL))
    parser.add_argument("--selection-quantile", type=float, default=0.6)
    parser.add_argument("--hold-minutes", type=int, default=15)
    parser.add_argument("--entry-delay-minutes", type=int, default=1)
    parser.add_argument("--base-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--spread-cross-mult", type=float, default=0.5)
    parser.add_argument("--stale-book-mult", type=float, default=0.25)
    parser.add_argument("--adverse-imbalance-mult", type=float, default=0.5)
    parser.add_argument("--max-concurrent-positions", type=int, default=4)
    parser.add_argument("--max-daily-trades", type=int, default=12)
    parser.add_argument("--output-prefix", default="execution_aware_rolling_backtest")
    return parser.parse_args()


def add_symbol_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sym = pd.get_dummies(df["symbol"], prefix="sym", dtype=float)
    out = pd.concat([df.reset_index(drop=True), sym.reset_index(drop=True)], axis=1)
    return out, list(sym.columns)


def load_panel_targets(panel_input: str, hold_minutes: int, entry_delay_minutes: int, base_fee_bps_roundtrip: float) -> pd.DataFrame:
    panel = pd.read_csv(panel_input, usecols=["symbol", "ts", "bn_close", "bb_close"], parse_dates=["ts"])
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["gap_bps"] = 10000.0 * (pd.to_numeric(panel["bn_close"], errors="coerce") / pd.to_numeric(panel["bb_close"], errors="coerce") - 1.0)
    group = panel.groupby("symbol", sort=False)
    panel["entry_gap_bps_delay"] = group["gap_bps"].shift(-entry_delay_minutes)
    panel["exit_gap_bps_delay"] = group["gap_bps"].shift(-(entry_delay_minutes + hold_minutes))
    panel["delayed_gap_close_15m_bps"] = panel["entry_gap_bps_delay"] - panel["exit_gap_bps_delay"]
    panel["delayed_pair_net_15m_bps"] = panel["delayed_gap_close_15m_bps"] - base_fee_bps_roundtrip
    panel["entry_decay_bps"] = panel["gap_bps"] - panel["entry_gap_bps_delay"]
    return panel[["symbol", "ts", "entry_gap_bps_delay", "delayed_gap_close_15m_bps", "delayed_pair_net_15m_bps", "entry_decay_bps"]]


def add_execution_columns(events: pd.DataFrame, panel_targets: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = events.merge(panel_targets, on=["symbol", "ts"], how="left")
    df["spread_cross_penalty_bps"] = args.spread_cross_mult * df["bb_spread_bps_ob"].clip(lower=0.0).fillna(0.0)
    df["stale_book_penalty_bps"] = args.stale_book_mult * df["bb_mid_gap_vs_trades_bps"].abs().fillna(0.0)
    df["adverse_imbalance_penalty_bps"] = (
        args.adverse_imbalance_mult
        * df["bb_spread_bps_ob"].clip(lower=0.0).fillna(0.0)
        * (-df["bb_top20_imbalance"].fillna(0.0)).clip(lower=0.0)
    )
    df["execution_extra_cost_bps"] = (
        df["spread_cross_penalty_bps"] + df["stale_book_penalty_bps"] + df["adverse_imbalance_penalty_bps"]
    )
    df["execution_pair_net_15m_bps"] = df["delayed_pair_net_15m_bps"] - df["execution_extra_cost_bps"]
    df["execution_pair_win_15m"] = (df["execution_pair_net_15m_bps"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "events": 0.0,
            "execution_pair_net_15m_bps": np.nan,
            "delayed_pair_net_15m_bps": np.nan,
            "execution_pair_win_15m": np.nan,
            "execution_extra_cost_bps": np.nan,
            "entry_decay_bps": np.nan,
        }
    return {
        "events": float(len(frame)),
        "execution_pair_net_15m_bps": float(frame["execution_pair_net_15m_bps"].mean()),
        "delayed_pair_net_15m_bps": float(frame["delayed_pair_net_15m_bps"].mean()),
        "execution_pair_win_15m": float(frame["execution_pair_win_15m"].mean()),
        "execution_extra_cost_bps": float(frame["execution_extra_cost_bps"].mean()),
        "entry_decay_bps": float(frame["entry_decay_bps"].mean()),
    }


def apply_portfolio_constraints(
    frame: pd.DataFrame,
    score_col: str,
    hold_minutes: int,
    entry_delay_minutes: int,
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
    entry_delay = pd.Timedelta(minutes=entry_delay_minutes)

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
        "mean_execution_pair_net_15m_bps": float(frame["execution_pair_net_15m_bps"].mean()),
        "median_execution_pair_net_15m_bps": float(frame["execution_pair_net_15m_bps"].median()),
        "mean_delayed_pair_net_15m_bps": float(frame["delayed_pair_net_15m_bps"].mean()),
        "mean_execution_pair_win_15m": float(frame["execution_pair_win_15m"].mean()),
        "mean_execution_extra_cost_bps": float(frame["execution_extra_cost_bps"].mean()),
        "mean_entry_decay_bps": float(frame["entry_decay_bps"].mean()),
        "positive_folds": int((frame["execution_pair_net_15m_bps"] > 0).sum()),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_input, parse_dates=["ts"]).replace([np.inf, -np.inf], np.nan)
    active = pd.read_csv(args.active_input)
    if events.empty or active.empty:
        raise SystemExit("missing events or active universe input")

    panel_targets = load_panel_targets(
        args.panel_input,
        hold_minutes=args.hold_minutes,
        entry_delay_minutes=args.entry_delay_minutes,
        base_fee_bps_roundtrip=args.base_fee_bps_roundtrip,
    )
    events = add_execution_columns(events, panel_targets, args)
    events = events.dropna(subset=["execution_pair_net_15m_bps"]).copy()
    if events.empty:
        raise SystemExit("no events survived execution target construction")

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
        model.fit(train[usable_feature_cols], train["execution_pair_net_15m_bps"])
        train["model_score"] = model.predict(train[usable_feature_cols])
        test["model_score"] = model.predict(test[usable_feature_cols])
        train["state_score"] = train[state_cols].fillna(0.0).mean(axis=1)
        test["state_score"] = test[state_cols].fillna(0.0).mean(axis=1)

        model_threshold = train["model_score"].quantile(args.selection_quantile)
        state_threshold = train["state_score"].quantile(args.selection_quantile)

        variants = {
            "active_exec_baseline": apply_portfolio_constraints(
                test,
                score_col="gap_bps",
                hold_minutes=args.hold_minutes,
                entry_delay_minutes=args.entry_delay_minutes,
                max_concurrent_positions=args.max_concurrent_positions,
                max_daily_trades=args.max_daily_trades,
            ),
            "active_exec_state_top_quantile": apply_portfolio_constraints(
                test[test["state_score"] >= state_threshold],
                score_col="state_score",
                hold_minutes=args.hold_minutes,
                entry_delay_minutes=args.entry_delay_minutes,
                max_concurrent_positions=args.max_concurrent_positions,
                max_daily_trades=args.max_daily_trades,
            ),
            "active_exec_model_top_quantile": apply_portfolio_constraints(
                test[test["model_score"] >= model_threshold],
                score_col="model_score",
                hold_minutes=args.hold_minutes,
                entry_delay_minutes=args.entry_delay_minutes,
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
            "execution_pair_net_15m_bps",
            "delayed_pair_net_15m_bps",
            "execution_pair_win_15m",
            "execution_extra_cost_bps",
            "entry_decay_bps",
        ]
    ]
    summary = pd.DataFrame(
        [
            summarize_variant(fold_rows, "active_exec_baseline"),
            summarize_variant(fold_rows, "active_exec_state_top_quantile"),
            summarize_variant(fold_rows, "active_exec_model_top_quantile"),
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
            mean_execution_pair_net_15m_bps=("execution_pair_net_15m_bps", "mean"),
            mean_execution_pair_win_15m=("execution_pair_win_15m", "mean"),
        )
    )

    report_lines = [
        "# Execution Aware Rolling Backtest",
        "",
        f"- Events input: `{args.events_input}`",
        f"- Active set input: `{args.active_input}`",
        f"- Panel input: `{args.panel_input}`",
        f"- Selection quantile: `{args.selection_quantile:.2f}`",
        f"- Hold minutes: `{args.hold_minutes}`",
        f"- Entry delay minutes: `{args.entry_delay_minutes}`",
        f"- Base fee assumption: `{args.base_fee_bps_roundtrip:.2f}` bps",
        f"- Spread cross multiplier: `{args.spread_cross_mult:.2f}`",
        f"- Stale book multiplier: `{args.stale_book_mult:.2f}`",
        f"- Adverse imbalance multiplier: `{args.adverse_imbalance_mult:.2f}`",
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
            f"mean_exec_net_15m={row.mean_execution_pair_net_15m_bps:+.2f} bps, "
            f"mean_delay_net_15m={row.mean_delayed_pair_net_15m_bps:+.2f} bps, "
            f"mean_exec_win_15m={row.mean_execution_pair_win_15m * 100:.1f}%, "
            f"mean_extra_cost={row.mean_execution_extra_cost_bps:.2f} bps, "
            f"mean_entry_decay={row.mean_entry_decay_bps:.2f} bps, positive_folds={row.positive_folds}/{row.folds}"
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
