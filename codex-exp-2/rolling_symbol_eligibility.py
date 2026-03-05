from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = OUT_DIR / "delayed_confirmation_features.csv"
RESULTS_CSV = OUT_DIR / "rolling_symbol_eligibility_results.csv"
REPORT_MD = OUT_DIR / "FINDINGS_rolling_symbol_eligibility.md"

HOLD_LAG = pd.Timedelta("4h")


@dataclass(frozen=True)
class DelayRule:
    delay_s: int
    min_ret_delay_bps: float
    min_ret_5m_bps: float
    min_buy_share_5m: float


@dataclass(frozen=True)
class EligibilityRule:
    min_hist: int
    min_avg_bps: float


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["ts"])
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)


def apply_delay_rule(df: pd.DataFrame, rule: DelayRule) -> pd.DataFrame:
    delay_col = "ret_30s_bps" if rule.delay_s == 30 else "ret_60s_bps"
    net_col = "delayed_30_net_bps_20" if rule.delay_s == 30 else "delayed_60_net_bps_20"
    taken = df.loc[
        (df[delay_col] >= rule.min_ret_delay_bps)
        & (df["ret_5m_bps"] >= rule.min_ret_5m_bps)
        & (df["buy_share_5m"] >= rule.min_buy_share_5m)
    ].copy()
    if taken.empty:
        return taken
    taken["variant_net_bps"] = taken[net_col]
    return taken


def choose_delay_rule(df: pd.DataFrame) -> DelayRule:
    train = df.loc[df["study_period"] == "train"].copy()
    q = train.quantile([0.5, 0.75], numeric_only=True)
    candidates: list[tuple[float, int, DelayRule]] = []
    for delay_s in (30, 60):
        delay_col = "ret_30s_bps" if delay_s == 30 else "ret_60s_bps"
        for min_ret_delay_bps in (0.0, float(q.loc[0.5, delay_col]), float(q.loc[0.75, delay_col])):
            for min_ret_5m_bps in (0.0, float(q.loc[0.5, "ret_5m_bps"]), float(q.loc[0.75, "ret_5m_bps"])):
                for min_buy_share_5m in (0.5, float(q.loc[0.5, "buy_share_5m"])):
                    rule = DelayRule(delay_s, min_ret_delay_bps, min_ret_5m_bps, min_buy_share_5m)
                    taken = apply_delay_rule(train, rule)
                    if len(taken) < 8:
                        continue
                    candidates.append((float(taken["variant_net_bps"].mean()), len(taken), rule))
    if not candidates:
        return DelayRule(60, 0.0, 0.0, 0.5)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def simulate_eligibility(df: pd.DataFrame, phase: str, rule: EligibilityRule) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    phase_rows = df.loc[df["study_period"] == phase].copy()
    if phase_rows.empty:
        return pd.DataFrame(columns=df.columns)

    for ts, group in phase_rows.groupby("ts", sort=True):
        cutoff = ts - HOLD_LAG
        history = df.loc[df["ts"] <= cutoff].copy()
        if history.empty:
            continue
        stats = (
            history.groupby("symbol", sort=False)["variant_net_bps"]
            .agg(["count", "mean"])
            .rename(columns={"count": "hist_count", "mean": "hist_avg"})
            .reset_index()
        )
        eligible = stats.loc[
            (stats["hist_count"] >= rule.min_hist) & (stats["hist_avg"] >= rule.min_avg_bps),
            "symbol",
        ]
        if eligible.empty:
            continue
        kept = group.loc[group["symbol"].isin(set(eligible))].copy()
        if kept.empty:
            continue
        chosen.extend([row for _, row in kept.iterrows()])

    if not chosen:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(chosen).sort_values(["ts", "symbol"]).reset_index(drop=True)


def choose_eligibility_rule(df: pd.DataFrame) -> tuple[EligibilityRule, pd.DataFrame]:
    candidates = [
        EligibilityRule(min_hist=min_hist, min_avg_bps=min_avg_bps)
        for min_hist in (1, 2, 3, 4)
        for min_avg_bps in (0.0, 25.0, 50.0, 100.0)
    ]
    scored: list[tuple[float, int, EligibilityRule, pd.DataFrame]] = []
    for rule in candidates:
        kept = simulate_eligibility(df, "train", rule)
        if len(kept) < 5:
            continue
        scored.append((float(kept["variant_net_bps"].mean()), len(kept), rule, kept))
    if not scored:
        fallback = EligibilityRule(1, 0.0)
        return fallback, simulate_eligibility(df, "train", fallback)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0]
    return best[2], best[3]


def evaluate_variant(df: pd.DataFrame, variant_name: str) -> dict[str, object]:
    base_test = float(df.loc[df["study_period"] == "test", "variant_net_bps"].mean())
    elig_rule, train_kept = choose_eligibility_rule(df)
    test_kept = simulate_eligibility(df, "test", elig_rule)
    grid_rows = []
    for min_hist in (1, 2, 3, 4):
        for min_avg_bps in (0.0, 25.0, 50.0, 100.0):
            probe = EligibilityRule(min_hist, min_avg_bps)
            train_probe = simulate_eligibility(df, "train", probe)
            test_probe = simulate_eligibility(df, "test", probe)
            grid_rows.append(
                {
                    "min_hist": min_hist,
                    "min_avg_bps": min_avg_bps,
                    "train_rows": len(train_probe),
                    "test_rows": len(test_probe),
                    "test_avg_bps": float(test_probe["variant_net_bps"].mean()) if not test_probe.empty else np.nan,
                }
            )
    grid = pd.DataFrame(grid_rows)
    return {
        "variant": variant_name,
        "base_test_avg_bps": base_test,
        "elig_min_hist": elig_rule.min_hist,
        "elig_min_avg_bps": elig_rule.min_avg_bps,
        "train_kept_rows": int(len(train_kept)),
        "train_kept_avg_bps": float(train_kept["variant_net_bps"].mean()) if not train_kept.empty else np.nan,
        "test_kept_rows": int(len(test_kept)),
        "test_kept_avg_bps": float(test_kept["variant_net_bps"].mean()) if not test_kept.empty else np.nan,
        "test_improve_bps": (float(test_kept["variant_net_bps"].mean()) - base_test) if not test_kept.empty else np.nan,
        "test_kept_symbols": ",".join(sorted(test_kept["symbol"].unique())) if not test_kept.empty else "",
        "test_kept_trades": test_kept[["ts", "symbol", "variant_net_bps"]].to_dict("records"),
        "any_positive_test_config": bool((grid["test_avg_bps"] > 0).fillna(False).any()),
        "best_grid_test_avg_bps": float(grid["test_avg_bps"].max()) if grid["test_avg_bps"].notna().any() else np.nan,
    }


def main() -> None:
    features = load_features()

    base = features.copy()
    base["variant_net_bps"] = base["base_net_bps_20"]

    delay_rule = choose_delay_rule(features)
    delayed = apply_delay_rule(features, delay_rule)

    rows = [
        evaluate_variant(base, "base"),
        evaluate_variant(delayed, "delayed"),
    ]
    results = pd.DataFrame(
        [
            {
                k: v
                for k, v in row.items()
                if k not in {"test_kept_trades"}
            }
            for row in rows
        ]
    )
    results.to_csv(RESULTS_CSV, index=False)

    delayed_row = next(row for row in rows if row["variant"] == "delayed")
    base_row = next(row for row in rows if row["variant"] == "base")

    lines = [
        "# Rolling Symbol Eligibility",
        "",
        "This simulates a live symbol-eligibility gate with an expanding window.",
        "A symbol only becomes eligible after enough prior 4-hour outcomes are known, so history is only counted once the prior trade is at least 4 hours old.",
        "",
        "## Delayed Rule Used",
        "",
        f"- Delay: `{delay_rule.delay_s}s`",
        f"- `ret_delay_bps >= {delay_rule.min_ret_delay_bps:.2f}`",
        f"- `ret_5m_bps >= {delay_rule.min_ret_5m_bps:.2f}`",
        f"- `buy_share_5m >= {delay_rule.min_buy_share_5m:.3f}`",
        "",
        "## Base Variant",
        "",
        f"- Unfiltered test avg: {base_row['base_test_avg_bps']:.2f} bps",
        f"- Chosen eligibility: `min_hist >= {int(base_row['elig_min_hist'])}`, `hist_avg >= {base_row['elig_min_avg_bps']:.2f} bps`",
        f"- Filtered test avg: {base_row['test_kept_avg_bps']:.2f} bps on {int(base_row['test_kept_rows'])} rows" if pd.notna(base_row["test_kept_avg_bps"]) else f"- Filtered test avg: no rows on {int(base_row['test_kept_rows'])} rows",
        f"- Improvement vs unfiltered: {base_row['test_improve_bps']:.2f} bps" if pd.notna(base_row["test_improve_bps"]) else "- Improvement vs unfiltered: n/a",
        f"- Test symbols kept: {base_row['test_kept_symbols'] or 'none'}",
        "",
        "## Delayed Variant",
        "",
        f"- Unfiltered delayed test avg: {delayed_row['base_test_avg_bps']:.2f} bps",
        f"- Chosen eligibility: `min_hist >= {int(delayed_row['elig_min_hist'])}`, `hist_avg >= {delayed_row['elig_min_avg_bps']:.2f} bps`",
        f"- Filtered delayed test avg: {delayed_row['test_kept_avg_bps']:.2f} bps on {int(delayed_row['test_kept_rows'])} rows" if pd.notna(delayed_row["test_kept_avg_bps"]) else f"- Filtered delayed test avg: no rows on {int(delayed_row['test_kept_rows'])} rows",
        f"- Improvement vs unfiltered: {delayed_row['test_improve_bps']:.2f} bps" if pd.notna(delayed_row["test_improve_bps"]) else "- Improvement vs unfiltered: n/a",
        f"- Test symbols kept: {delayed_row['test_kept_symbols'] or 'none'}",
        "",
        "## Bottom Line",
        "",
    ]

    base_positive = pd.notna(base_row["test_kept_avg_bps"]) and base_row["test_kept_avg_bps"] > 0
    delayed_positive = pd.notna(delayed_row["test_kept_avg_bps"]) and delayed_row["test_kept_avg_bps"] > 0
    if base_positive or delayed_positive:
        lines.append("- A rolling earned-eligibility filter can produce a positive result on the covered set, but row count determines whether it is meaningful.")
    else:
        lines.append("- Rolling earned-eligibility improves selectivity, but it still does not make the covered test positive.")

    if not bool(base_row["any_positive_test_config"]):
        lines.append(f"- For the base variant, every tested eligibility setting stayed negative; the best observed test average was {base_row['best_grid_test_avg_bps']:.2f} bps.")
    if not bool(delayed_row["any_positive_test_config"]):
        lines.append("- For the delayed variant, the filtered sample is too sparse to establish a usable earned-eligibility rule.")

    if int(base_row["test_kept_rows"]) <= 3 or int(delayed_row["test_kept_rows"]) <= 3:
        lines.append("- Any apparent success here is fragile because the filter keeps very few test trades.")

    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
