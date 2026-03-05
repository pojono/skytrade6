from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = OUT_DIR / "delayed_confirmation_features.csv"
CLASS_CSV = OUT_DIR / "symbol_classification.csv"
OUT_FEATURES_CSV = OUT_DIR / "shorter_hold_features.csv"
OUT_RESULTS_CSV = OUT_DIR / "shorter_hold_results.csv"
REPORT_MD = OUT_DIR / "FINDINGS_shorter_hold_case_study.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"
HOLDS_MIN = (30, 60, 90)


@dataclass(frozen=True)
class DelayRule:
    delay_s: int
    min_ret_delay_bps: float
    min_ret_5m_bps: float
    min_buy_share_5m: float


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


def enrich_holds() -> pd.DataFrame:
    base = pd.read_csv(FEATURES_CSV, parse_dates=["ts"])
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        ts = row["ts"]
        symbol = row["symbol"]
        agg = load_agg(symbol, ts.strftime("%Y-%m-%d"))
        if agg is None:
            continue

        pre = agg.loc[agg["ts"] <= ts]
        post_30s = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("30s"))]
        post_60s = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("60s"))]
        if pre.empty or post_30s.empty or post_60s.empty:
            continue

        signal_px = _last_price(pre)
        p30 = _last_price(post_30s)
        p60 = _last_price(post_60s)
        if signal_px is None or p30 is None or p60 is None:
            continue

        out = row.to_dict()
        ok = True
        for hold_min in HOLDS_MIN:
            post_hold = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta(minutes=hold_min))]
            if post_hold.empty:
                ok = False
                break
            px_hold = _last_price(post_hold)
            if px_hold is None:
                ok = False
                break
            out[f"base_net_{hold_min}m_bps_20"] = (px_hold / signal_px - 1.0) * 10000.0 - 20.0
            out[f"delayed_30_net_{hold_min}m_bps_20"] = (px_hold / p30 - 1.0) * 10000.0 - 20.0
            out[f"delayed_60_net_{hold_min}m_bps_20"] = (px_hold / p60 - 1.0) * 10000.0 - 20.0
        if ok:
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["ts", "symbol"]).reset_index(drop=True)


def build_cohorts(df: pd.DataFrame) -> dict[str, list[str]]:
    classes = pd.read_csv(CLASS_CSV)
    covered = set(df["symbol"].unique())
    c = classes.loc[classes["symbol"].isin(covered)].copy()

    robust = sorted(c.loc[c["classification"] == "robust_positive", "symbol"])
    train_positive_2plus = sorted(
        c.loc[
            (c["classification"] != "negative")
            & (c["train_trades"] >= 2)
            & (c["train_avg_bps_20"] > 0),
            "symbol",
        ]
    )
    total_positive = sorted(
        c.loc[
            (c["classification"] != "negative")
            & (c["total_bps_20"] > 0),
            "symbol",
        ]
    )

    cohorts = {
        "covered_all": sorted(covered),
        "train_positive_2plus": train_positive_2plus,
        "total_positive": total_positive,
    }
    if robust:
        cohorts["robust_plus_sol"] = sorted(set(robust) | {"SOLUSDT"})
    return {name: syms for name, syms in cohorts.items() if syms}


def apply_delay_rule(df: pd.DataFrame, hold_min: int, rule: DelayRule) -> pd.DataFrame:
    delay_col = "ret_30s_bps" if rule.delay_s == 30 else "ret_60s_bps"
    net_col = f"delayed_{rule.delay_s}_net_{hold_min}m_bps_20"
    taken = df.loc[
        (df[delay_col] >= rule.min_ret_delay_bps)
        & (df["ret_5m_bps"] >= rule.min_ret_5m_bps)
        & (df["buy_share_5m"] >= rule.min_buy_share_5m)
    ].copy()
    if taken.empty:
        return taken
    taken["variant_net_bps"] = taken[net_col]
    return taken


def choose_delay_rule(train: pd.DataFrame, hold_min: int) -> tuple[DelayRule, pd.DataFrame]:
    if train.empty:
        return DelayRule(60, 0.0, 0.0, 0.5), train
    q = train.quantile([0.5, 0.75], numeric_only=True)
    best_rule: DelayRule | None = None
    best_taken = pd.DataFrame()
    best_score = (-np.inf, -np.inf)
    for delay_s in (30, 60):
        delay_col = "ret_30s_bps" if delay_s == 30 else "ret_60s_bps"
        for min_ret_delay_bps in (0.0, float(q.loc[0.5, delay_col]), float(q.loc[0.75, delay_col])):
            for min_ret_5m_bps in (0.0, float(q.loc[0.5, "ret_5m_bps"]), float(q.loc[0.75, "ret_5m_bps"])):
                for min_buy_share_5m in (0.5, float(q.loc[0.5, "buy_share_5m"])):
                    rule = DelayRule(delay_s, min_ret_delay_bps, min_ret_5m_bps, min_buy_share_5m)
                    taken = apply_delay_rule(train, hold_min, rule)
                    if len(taken) < 5:
                        continue
                    score = (float(taken["variant_net_bps"].mean()), len(taken))
                    if score > best_score:
                        best_score = score
                        best_rule = rule
                        best_taken = taken
    if best_rule is None:
        best_rule = DelayRule(60, 0.0, 0.0, 0.5)
        best_taken = apply_delay_rule(train, hold_min, best_rule)
    return best_rule, best_taken


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for cohort_name, syms in build_cohorts(df).items():
        cohort = df.loc[df["symbol"].isin(syms)].copy()
        train = cohort.loc[cohort["study_period"] == "train"].copy()
        test = cohort.loc[cohort["study_period"] == "test"].copy()
        for hold_min in HOLDS_MIN:
            base_col = f"base_net_{hold_min}m_bps_20"
            base_test_avg = float(test[base_col].mean()) if not test.empty else np.nan
            rule, train_taken = choose_delay_rule(train, hold_min)
            test_taken = apply_delay_rule(test, hold_min, rule)
            out_rows.append(
                {
                    "cohort": cohort_name,
                    "n_symbols": len(syms),
                    "hold_min": hold_min,
                    "base_test_avg_bps": base_test_avg,
                    "delay_s": rule.delay_s,
                    "min_ret_delay_bps": rule.min_ret_delay_bps,
                    "min_ret_5m_bps": rule.min_ret_5m_bps,
                    "min_buy_share_5m": rule.min_buy_share_5m,
                    "train_delayed_rows": int(len(train_taken)),
                    "train_delayed_avg_bps": float(train_taken["variant_net_bps"].mean()) if not train_taken.empty else np.nan,
                    "test_delayed_rows": int(len(test_taken)),
                    "test_delayed_avg_bps": float(test_taken["variant_net_bps"].mean()) if not test_taken.empty else np.nan,
                    "test_improve_bps": (float(test_taken["variant_net_bps"].mean()) - base_test_avg) if not test_taken.empty else np.nan,
                    "test_delayed_symbols": ",".join(sorted(test_taken["symbol"].unique())) if not test_taken.empty else "",
                }
            )
    return pd.DataFrame(out_rows).sort_values(["cohort", "hold_min"]).reset_index(drop=True)


def write_report(results: pd.DataFrame) -> None:
    best_abs = results.loc[results["test_delayed_avg_bps"].notna()].sort_values(
        ["test_delayed_avg_bps", "test_delayed_rows"],
        ascending=[False, False],
    )
    best_realistic = results.loc[
        results["test_delayed_avg_bps"].notna() & (results["test_delayed_rows"] >= 5)
    ].sort_values(["test_delayed_avg_bps", "test_delayed_rows"], ascending=[False, False])

    lines = [
        "# Shorter Hold Case Study",
        "",
        "This tests whether the covered-universe signal behaves better with 30m, 60m, or 90m exits instead of the original 4-hour hold.",
        "For each cohort and hold, the delayed-confirmation rule is chosen on the train split only, then applied to the test split.",
        "",
    ]

    for cohort_name, group in results.groupby("cohort", sort=False):
        lines.extend([f"## {cohort_name}", ""])
        for _, row in group.sort_values("hold_min").iterrows():
            delayed_desc = (
                f"{row['test_delayed_avg_bps']:.2f} bps on {int(row['test_delayed_rows'])} rows"
                if pd.notna(row["test_delayed_avg_bps"])
                else "no kept rows"
            )
            improve_desc = f"{row['test_improve_bps']:.2f} bps" if pd.notna(row["test_improve_bps"]) else "n/a"
            lines.extend(
                [
                    f"- `{int(row['hold_min'])}m` hold: base test `{row['base_test_avg_bps']:.2f} bps`, delayed test `{delayed_desc}`, improvement `{improve_desc}`",
                ]
            )
        lines.append("")

    lines.extend(["## Bottom Line", ""])
    if not best_abs.empty:
        r = best_abs.iloc[0]
        lines.append(
            f"- Best absolute delayed result is `{r['cohort']}` with a `{int(r['hold_min'])}m` hold: {r['test_delayed_avg_bps']:.2f} bps on {int(r['test_delayed_rows'])} rows."
        )
    if not best_realistic.empty:
        r = best_realistic.iloc[0]
        lines.append(
            f"- Best result with at least 5 delayed test rows is `{r['cohort']}` with a `{int(r['hold_min'])}m` hold: {r['test_delayed_avg_bps']:.2f} bps on {int(r['test_delayed_rows'])} rows."
        )
    else:
        lines.append("- No cohort/hold combination stayed positive with at least 5 delayed test rows.")

    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    features = enrich_holds()
    features.to_csv(OUT_FEATURES_CSV, index=False)
    results = evaluate(features)
    results.to_csv(OUT_RESULTS_CSV, index=False)
    write_report(results)
    print(f"Wrote {OUT_FEATURES_CSV}")
    print(f"Wrote {OUT_RESULTS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
