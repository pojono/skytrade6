from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
FEATURES_CSV = OUT_DIR / "delayed_confirmation_features.csv"
GRID_CSV = OUT_DIR / "delayed_confirmation_grid.csv"
REPORT_MD = OUT_DIR / "FINDINGS_delayed_confirmation_case_study.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"


def _has_local_agg(symbol: str) -> bool:
    return (PARQUET / symbol / "binance" / "agg_trades_futures").exists()


def _local_symbols() -> set[str]:
    if not PARQUET.exists():
        return set()
    return {
        p.name
        for p in PARQUET.iterdir()
        if p.is_dir() and _has_local_agg(p.name)
    }


SYMS = _local_symbols()
START_TS = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2026-01-31 23:59:59", tz="UTC")


@dataclass(frozen=True)
class Rule:
    delay_s: int
    min_ret_delay_bps: float
    min_ret_5m_bps: float
    min_buy_share_5m: float


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    common = (
        s["symbol"].isin(SYMS)
        & (s["ts"] >= START_TS)
        & (s["ts"] <= END_TS)
        & (s["oi_med_3d"] >= 20_000_000)
        & (s["breadth_mom"] >= 0.60)
        & (s["median_ls_z"] >= 0.0)
        & (s["mom_4h"] > 0)
    )
    broad = s.loc[common & (s["ls_z"] >= 1.0) & (s["taker_z"] >= 0.0)].copy()
    strict = s.loc[common & (s["ls_z"] >= 2.0) & (s["taker_z"] >= 0.5)].copy()
    for df in (broad, strict):
        df["base_net_bps_20"] = df["ret_4h"] * 10000.0 - 20.0
    return broad, strict


@lru_cache(maxsize=None)
def load_agg(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "agg_trades_futures" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


def _last_price(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df.iloc[-1]["price"])


def build_features(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        ts = row["ts"]
        symbol = row["symbol"]
        day = ts.strftime("%Y-%m-%d")
        agg = load_agg(symbol, day)
        if agg is None:
            continue

        pre = agg.loc[agg["ts"] <= ts].copy()
        post_30s = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("30s"))].copy()
        post_60s = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("60s"))].copy()
        post_5m = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("5m"))].copy()
        if pre.empty or post_30s.empty or post_60s.empty or post_5m.empty:
            continue

        signal_price = _last_price(pre)
        p30 = _last_price(post_30s)
        p60 = _last_price(post_60s)
        p5m = _last_price(post_5m)
        if signal_price is None or p30 is None or p60 is None or p5m is None:
            continue

        exit_price_4h = signal_price * (1.0 + float(row["ret_4h"]))
        quote_5m = post_5m["price"] * post_5m["quantity"]
        buy_share_5m = (
            float(quote_5m.loc[~post_5m["is_buyer_maker"]].sum() / quote_5m.sum())
            if quote_5m.sum() > 0
            else 0.5
        )

        def to_bps(px: float) -> float:
            return (px / signal_price - 1.0) * 10000.0

        delayed_30_net = (exit_price_4h / p30 - 1.0) * 10000.0 - 20.0
        delayed_60_net = (exit_price_4h / p60 - 1.0) * 10000.0 - 20.0

        out.append(
            {
                "ts": ts,
                "symbol": symbol,
                "base_net_bps_20": row["base_net_bps_20"],
                "ret_30s_bps": to_bps(p30),
                "ret_60s_bps": to_bps(p60),
                "ret_5m_bps": to_bps(p5m),
                "buy_share_5m": buy_share_5m,
                "delayed_30_net_bps_20": delayed_30_net,
                "delayed_60_net_bps_20": delayed_60_net,
            }
        )

    df = pd.DataFrame(out).sort_values(["ts", "symbol"]).reset_index(drop=True)
    if df.empty:
        return df
    unique_ts = sorted(df["ts"].unique())
    split_idx = max(1, len(unique_ts) // 2)
    cutoff = unique_ts[split_idx]
    df["study_period"] = np.where(df["ts"] < cutoff, "train", "test")
    return df


def apply_rule(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    delay_col = "ret_30s_bps" if rule.delay_s == 30 else "ret_60s_bps"
    net_col = "delayed_30_net_bps_20" if rule.delay_s == 30 else "delayed_60_net_bps_20"
    taken = df.loc[
        (df[delay_col] >= rule.min_ret_delay_bps)
        & (df["ret_5m_bps"] >= rule.min_ret_5m_bps)
        & (df["buy_share_5m"] >= rule.min_buy_share_5m)
    ].copy()
    if taken.empty:
        return taken
    taken["strategy_net_bps_20"] = taken[net_col]
    return taken


def eval_rule(train: pd.DataFrame, test: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    train_f = apply_rule(train, rule)
    test_f = apply_rule(test, rule)
    return {
        "delay_s": rule.delay_s,
        "min_ret_delay_bps": rule.min_ret_delay_bps,
        "min_ret_5m_bps": rule.min_ret_5m_bps,
        "min_buy_share_5m": rule.min_buy_share_5m,
        "train_rows": int(train_f.shape[0]),
        "test_rows": int(test_f.shape[0]),
        "train_avg_bps": float(train_f["strategy_net_bps_20"].mean()) if not train_f.empty else np.nan,
        "test_avg_bps": float(test_f["strategy_net_bps_20"].mean()) if not test_f.empty else np.nan,
        "train_hit": float((train_f["strategy_net_bps_20"] > 0).mean()) if not train_f.empty else np.nan,
        "test_hit": float((test_f["strategy_net_bps_20"] > 0).mean()) if not test_f.empty else np.nan,
    }


def run_grid(features: pd.DataFrame) -> pd.DataFrame:
    train = features.loc[features["study_period"] == "train"].copy()
    test = features.loc[features["study_period"] == "test"].copy()
    base_train = float(train["base_net_bps_20"].mean())
    base_test = float(test["base_net_bps_20"].mean())

    q = train.quantile([0.25, 0.5, 0.75], numeric_only=True)
    rules = []
    for delay_s in (30, 60):
        delay_col = "ret_30s_bps" if delay_s == 30 else "ret_60s_bps"
        for min_ret_delay_bps in (0.0, float(q.loc[0.5, delay_col]), float(q.loc[0.75, delay_col])):
            for min_ret_5m_bps in (0.0, float(q.loc[0.5, "ret_5m_bps"]), float(q.loc[0.75, "ret_5m_bps"])):
                for min_buy_share_5m in (0.5, float(q.loc[0.5, "buy_share_5m"])):
                    rules.append(Rule(delay_s, min_ret_delay_bps, min_ret_5m_bps, min_buy_share_5m))

    rows = [eval_rule(train, test, rule) for rule in rules]
    grid = pd.DataFrame(rows).drop_duplicates(
        subset=["delay_s", "min_ret_delay_bps", "min_ret_5m_bps", "min_buy_share_5m"]
    )
    grid = grid.loc[(grid["train_rows"] >= 3) & (grid["test_rows"] >= 3)].copy()
    grid["train_improve_bps"] = grid["train_avg_bps"] - base_train
    grid["test_improve_bps"] = grid["test_avg_bps"] - base_test
    grid = grid.sort_values(
        ["test_improve_bps", "train_improve_bps", "test_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return grid


def write_report(broad_features: pd.DataFrame, strict_features: pd.DataFrame, grid: pd.DataFrame) -> None:
    if grid.empty:
        REPORT_MD.write_text("# Delayed Confirmation Case Study\n\nNo valid delayed-entry rule met the minimum train/test row thresholds.")
        return

    broad_train = broad_features.loc[broad_features["study_period"] == "train"]
    broad_test = broad_features.loc[broad_features["study_period"] == "test"]
    best = grid.iloc[0]
    rule = Rule(
        int(best["delay_s"]),
        float(best["min_ret_delay_bps"]),
        float(best["min_ret_5m_bps"]),
        float(best["min_buy_share_5m"]),
    )
    strict_test = strict_features.loc[strict_features["study_period"] == "test"]
    strict_test_f = apply_rule(strict_test, rule)
    positive_abs = grid.loc[grid["test_avg_bps"] > 0].copy()

    lines = [
        "# Delayed Confirmation Strategy Case Study",
        "",
        "This converts short-horizon follow-through into an executable strategy variant: wait, confirm, then enter.",
        "",
        "## Scope",
        "",
        f"- Covered symbols: {', '.join(sorted(SYMS))}",
        "- Entry is delayed by 30s or 60s after the original signal.",
        "- Delayed entry price = last trade observed by that delayed timestamp.",
        "- Exit remains at the original 4-hour horizon endpoint implied by the base signal.",
        "- This is a covered-subset case study, not a full-universe result.",
        "",
        "## Broad Research Set",
        "",
        f"- Train rows: {len(broad_train)}",
        f"- Test rows: {len(broad_test)}",
        f"- Base broad test avg: {broad_test['base_net_bps_20'].mean():.2f} bps",
        "",
        "## Best Delayed-Entry Rule",
        "",
        f"- Delay: `{rule.delay_s}s`",
        f"- `ret_delay_bps >= {rule.min_ret_delay_bps:.2f}`",
        f"- `ret_5m_bps >= {rule.min_ret_5m_bps:.2f}`",
        f"- `buy_share_5m >= {rule.min_buy_share_5m:.3f}`",
        f"- Broad train avg after delayed confirmation: {best['train_avg_bps']:.2f} bps on {int(best['train_rows'])} rows",
        f"- Broad test avg after delayed confirmation: {best['test_avg_bps']:.2f} bps on {int(best['test_rows'])} rows",
        f"- Broad test improvement vs base: {best['test_improve_bps']:.2f} bps",
        "- Absolute broad test result remains negative." if best["test_avg_bps"] <= 0 else "- Absolute broad test result is positive.",
        "",
        "## Apply Same Rule To Strict Strategy Subset",
        "",
        f"- Strict covered test rows before: {len(strict_test)}",
        f"- Strict covered test avg before: {strict_test['base_net_bps_20'].mean():.2f} bps",
        f"- Strict covered test rows after: {len(strict_test_f)}",
        f"- Strict covered test avg after: {strict_test_f['strategy_net_bps_20'].mean():.2f} bps" if len(strict_test_f) else "- Strict covered test avg after: no rows kept",
        (
            f"- Strict kept row: {strict_test_f.iloc[0]['symbol']} at {strict_test_f.iloc[0]['ts']}"
            if len(strict_test_f) == 1
            else ""
        ),
        "",
        "## Interpretation",
        "",
        "- Delayed confirmation still helps relative to immediate entry on the covered broad set, which supports the 'wait for follow-through' idea.",
        "- But no tested delayed-entry rule makes the expanded broad covered test positive in absolute terms.",
        "- The strict-subset improvement is not broad evidence if it is driven by one surviving trade.",
    ]
    if not positive_abs.empty:
        best_abs = positive_abs.sort_values(
            ["test_avg_bps", "train_avg_bps", "test_rows"],
            ascending=[False, False, False],
        ).iloc[0]
        lines.extend(
            [
                "",
                "## Positive Broad Rule",
                "",
                f"- Best rule with positive broad test avg: {best_abs['test_avg_bps']:.2f} bps on {int(best_abs['test_rows'])} rows",
            ]
        )
    REPORT_MD.write_text("\n".join(line for line in lines if line))


def main() -> None:
    broad, strict = load_sets()
    broad_features = build_features(broad)
    strict_features = build_features(strict)
    broad_features.to_csv(FEATURES_CSV, index=False)
    grid = run_grid(broad_features)
    grid.to_csv(GRID_CSV, index=False)
    write_report(broad_features, strict_features, grid)
    print(f"Wrote {FEATURES_CSV}")
    print(f"Wrote {GRID_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(grid.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
