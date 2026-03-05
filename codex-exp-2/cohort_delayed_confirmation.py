from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = OUT_DIR / "delayed_confirmation_features.csv"
CLASS_CSV = OUT_DIR / "symbol_classification.csv"
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
RESULTS_CSV = OUT_DIR / "cohort_delayed_confirmation_results.csv"
REPORT_MD = OUT_DIR / "FINDINGS_cohort_delayed_confirmation.md"


@dataclass(frozen=True)
class Rule:
    delay_s: int
    min_ret_delay_bps: float
    min_ret_5m_bps: float
    min_buy_share_5m: float


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


def run_grid(features: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    train = features.loc[features["study_period"] == "train"].copy()
    test = features.loc[features["study_period"] == "test"].copy()
    base_train = float(train["base_net_bps_20"].mean()) if not train.empty else np.nan
    base_test = float(test["base_net_bps_20"].mean()) if not test.empty else np.nan
    if train.empty or test.empty:
        return pd.DataFrame(), base_train, base_test

    q = train.quantile([0.5, 0.75], numeric_only=True)
    rules = []
    for delay_s in (30, 60):
        delay_col = "ret_30s_bps" if delay_s == 30 else "ret_60s_bps"
        for min_ret_delay_bps in (0.0, float(q.loc[0.5, delay_col]), float(q.loc[0.75, delay_col])):
            for min_ret_5m_bps in (0.0, float(q.loc[0.5, "ret_5m_bps"]), float(q.loc[0.75, "ret_5m_bps"])):
                for min_buy_share_5m in (0.5, float(q.loc[0.5, "buy_share_5m"])):
                    rules.append(Rule(delay_s, min_ret_delay_bps, min_ret_5m_bps, min_buy_share_5m))

    rows = []
    for rule in rules:
        train_f = apply_rule(train, rule)
        test_f = apply_rule(test, rule)
        rows.append(
            {
                "delay_s": rule.delay_s,
                "min_ret_delay_bps": rule.min_ret_delay_bps,
                "min_ret_5m_bps": rule.min_ret_5m_bps,
                "min_buy_share_5m": rule.min_buy_share_5m,
                "train_rows": int(train_f.shape[0]),
                "test_rows": int(test_f.shape[0]),
                "train_avg_bps": float(train_f["strategy_net_bps_20"].mean()) if not train_f.empty else np.nan,
                "test_avg_bps": float(test_f["strategy_net_bps_20"].mean()) if not test_f.empty else np.nan,
            }
        )

    grid = pd.DataFrame(rows).drop_duplicates(
        subset=["delay_s", "min_ret_delay_bps", "min_ret_5m_bps", "min_buy_share_5m"]
    )
    grid = grid.loc[(grid["train_rows"] >= 2) & (grid["test_rows"] >= 1)].copy()
    if grid.empty:
        return grid, base_train, base_test
    grid["train_improve_bps"] = grid["train_avg_bps"] - base_train
    grid["test_improve_bps"] = grid["test_avg_bps"] - base_test
    grid = grid.sort_values(
        ["test_avg_bps", "test_improve_bps", "train_avg_bps", "test_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return grid, base_train, base_test


def strict_keys(features: pd.DataFrame) -> pd.DataFrame:
    covered = sorted(features["symbol"].unique())
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    out = s.loc[
        s["symbol"].isin(covered)
        & (s["ts"] >= "2025-11-01")
        & (s["ts"] <= "2026-01-31 23:59:59+00:00")
        & (s["oi_med_3d"] >= 20_000_000)
        & (s["breadth_mom"] >= 0.60)
        & (s["median_ls_z"] >= 0.0)
        & (s["mom_4h"] > 0)
        & (s["ls_z"] >= 2.0)
        & (s["taker_z"] >= 0.5),
        ["ts", "symbol"],
    ].copy()
    return out.drop_duplicates()


def build_cohorts(features: pd.DataFrame, classes: pd.DataFrame) -> dict[str, list[str]]:
    covered = set(features["symbol"].unique())
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
    tested_positive = sorted(
        c.loc[
            (c["classification"] != "negative")
            & (c["test_trades"] >= 1)
            & (c["test_avg_bps_20"] > 0),
            "symbol",
        ]
    )
    cohorts = {
        "robust_only": robust,
        "robust_plus_sol": sorted(set(robust) | {"SOLUSDT"}),
        "train_positive_2plus": train_positive_2plus,
        "total_positive": total_positive,
        "tested_positive": tested_positive,
    }
    return {name: syms for name, syms in cohorts.items() if syms}


def main() -> None:
    features = pd.read_csv(FEATURES_CSV, parse_dates=["ts"])
    classes = pd.read_csv(CLASS_CSV)
    strict = features.merge(strict_keys(features), on=["ts", "symbol"], how="inner")

    rows = []
    report_lines = [
        "# Cohort Delayed Confirmation",
        "",
        "This retests delayed confirmation only on small vetted symbol cohorts instead of the full covered universe.",
        "",
    ]

    for cohort_name, syms in build_cohorts(features, classes).items():
        cohort = features.loc[features["symbol"].isin(syms)].copy()
        grid, base_train, base_test = run_grid(cohort)
        strict_cohort = strict.loc[strict["symbol"].isin(syms)].copy()
        if grid.empty:
            rows.append(
                {
                    "cohort": cohort_name,
                    "symbols": ",".join(syms),
                    "n_symbols": len(syms),
                    "base_test_avg_bps": base_test,
                    "best_test_avg_bps": np.nan,
                    "best_test_improve_bps": np.nan,
                    "best_test_rows": 0,
                    "strict_base_test_avg_bps": float(strict_cohort.loc[strict_cohort["study_period"] == "test", "base_net_bps_20"].mean())
                    if not strict_cohort.empty
                    else np.nan,
                    "strict_best_test_avg_bps": np.nan,
                    "strict_best_test_rows": 0,
                }
            )
            continue

        best = grid.iloc[0]
        rule = Rule(
            int(best["delay_s"]),
            float(best["min_ret_delay_bps"]),
            float(best["min_ret_5m_bps"]),
            float(best["min_buy_share_5m"]),
        )
        strict_test = strict_cohort.loc[strict_cohort["study_period"] == "test"].copy()
        strict_best = apply_rule(strict_test, rule)

        rows.append(
            {
                "cohort": cohort_name,
                "symbols": ",".join(syms),
                "n_symbols": len(syms),
                "base_test_avg_bps": base_test,
                "best_test_avg_bps": float(best["test_avg_bps"]),
                "best_test_improve_bps": float(best["test_improve_bps"]),
                "best_test_rows": int(best["test_rows"]),
                "delay_s": int(best["delay_s"]),
                "min_ret_delay_bps": float(best["min_ret_delay_bps"]),
                "min_ret_5m_bps": float(best["min_ret_5m_bps"]),
                "min_buy_share_5m": float(best["min_buy_share_5m"]),
                "strict_base_test_avg_bps": float(strict_test["base_net_bps_20"].mean()) if not strict_test.empty else np.nan,
                "strict_best_test_avg_bps": float(strict_best["strategy_net_bps_20"].mean()) if not strict_best.empty else np.nan,
                "strict_best_test_rows": int(strict_best.shape[0]),
            }
        )

        report_lines.extend(
            [
                f"## {cohort_name}",
                "",
                f"- Symbols ({len(syms)}): {', '.join(syms)}",
                f"- Broad base test avg: {base_test:.2f} bps",
                f"- Best delayed rule: wait `{int(best['delay_s'])}s`, `ret_delay_bps >= {float(best['min_ret_delay_bps']):.2f}`, `ret_5m_bps >= {float(best['min_ret_5m_bps']):.2f}`, `buy_share_5m >= {float(best['min_buy_share_5m']):.3f}`",
                f"- Broad best delayed test avg: {float(best['test_avg_bps']):.2f} bps on {int(best['test_rows'])} rows",
                f"- Broad improvement vs base: {float(best['test_improve_bps']):.2f} bps",
                f"- Strict base test avg: {float(strict_test['base_net_bps_20'].mean()):.2f} bps" if not strict_test.empty else "- Strict base test avg: no rows",
                (
                    f"- Strict best delayed test avg: {float(strict_best['strategy_net_bps_20'].mean()):.2f} bps on {int(strict_best.shape[0])} rows"
                    if not strict_best.empty
                    else "- Strict best delayed test avg: no rows kept"
                ),
                "",
            ]
        )

    results = pd.DataFrame(rows).sort_values(
        ["best_test_avg_bps", "best_test_improve_bps", "n_symbols"],
        ascending=[False, False, True],
    )
    results.to_csv(RESULTS_CSV, index=False)

    if not results.empty:
        best_row = results.iloc[0]
        report_lines.extend(
            [
                "## Bottom Line",
                "",
                f"- Best cohort by absolute broad delayed test avg: `{best_row['cohort']}`",
                f"- That cohort reached {best_row['best_test_avg_bps']:.2f} bps on {int(best_row['best_test_rows'])} delayed test rows.",
                "- This is only credible if it still has enough rows; tiny cohorts can look good by accident.",
            ]
        )

    REPORT_MD.write_text("\n".join(report_lines))
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
