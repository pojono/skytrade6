#!/usr/bin/env python3
"""
Analyze cross-sectional money-flow patterns from codex-7/out/symbol_daily_flows.csv.

Outputs:
  - codex-7/out/money_flow_feature_ic.csv
  - codex-7/out/money_flow_state_summary.csv
  - codex-7/out/money_flow_breadth_summary.csv
  - codex-7/out/money_flow_report.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "codex-7" / "out"
DATA_CSV = OUT_DIR / "symbol_daily_flows.csv"
FEATURE_IC_CSV = OUT_DIR / "money_flow_feature_ic.csv"
STATE_CSV = OUT_DIR / "money_flow_state_summary.csv"
BREADTH_CSV = OUT_DIR / "money_flow_breadth_summary.csv"
REPORT_MD = OUT_DIR / "money_flow_report.md"

ROLLING_DAYS = 20
MIN_ROLLING = 10
MIN_STATE_OBS = 120
MIN_IC_OBS = 25


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_mean(s: pd.Series) -> pd.Series:
    return s.rolling(ROLLING_DAYS, min_periods=MIN_ROLLING).mean()


def rolling_median(s: pd.Series) -> pd.Series:
    return s.rolling(ROLLING_DAYS, min_periods=MIN_ROLLING).median()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["close"] = df["bybit_fut_close"].where(df["bybit_fut_close"].notna(), df["binance_fut_close"])
    df["premium_mean"] = df[["bybit_premium_mean", "binance_premium_mean"]].mean(axis=1)
    df["spot_share"] = safe_div(df["total_spot_turnover"], df["total_spot_turnover"] + df["total_fut_turnover"])
    df["bybit_share"] = safe_div(df["bybit_total_turnover"], df["bybit_total_turnover"] + df["binance_total_turnover"])
    df["binance_taker_buy_share"] = safe_div(df["binance_fut_taker_buy_quote"], df["binance_fut_turnover"])
    df["binance_spot_taker_buy_share"] = safe_div(df["binance_spot_taker_buy_quote"], df["binance_spot_turnover"])

    g = df.groupby("symbol", group_keys=False)
    df["ret_1d"] = g["close"].pct_change()
    for horizon in [1, 3, 5]:
        df[f"fwd_{horizon}d"] = g["close"].pct_change(horizon).shift(-horizon)

    df["oi_delta_pct"] = g["total_oi_usd"].pct_change()
    df["fut_turnover_med20"] = g["total_fut_turnover"].transform(rolling_median)
    df["spot_turnover_med20"] = g["total_spot_turnover"].transform(rolling_median)
    df["spot_share_ma20"] = g["spot_share"].transform(rolling_mean)
    df["bybit_share_ma20"] = g["bybit_share"].transform(rolling_mean)
    df["premium_ma20"] = g["premium_mean"].transform(rolling_mean)
    df["oi_delta_ma20"] = g["oi_delta_pct"].transform(rolling_mean)

    df["fut_turnover_shock"] = np.log(safe_div(df["total_fut_turnover"], df["fut_turnover_med20"]))
    df["spot_turnover_shock"] = np.log(safe_div(df["total_spot_turnover"], df["spot_turnover_med20"]))
    df["spot_share_rel"] = df["spot_share"] - df["spot_share_ma20"]
    df["bybit_share_rel"] = df["bybit_share"] - df["bybit_share_ma20"]
    df["premium_rel"] = df["premium_mean"] - df["premium_ma20"]
    df["oi_delta_rel"] = df["oi_delta_pct"] - df["oi_delta_ma20"]
    df["oi_to_turnover"] = safe_div(df["total_oi_usd"], df["total_fut_turnover"])

    # Composite scores built from intuitive flow ingredients.
    daily = df.groupby("date")
    for col in ["ret_1d", "spot_share_rel", "oi_delta_pct", "premium_rel", "fut_turnover_shock", "bybit_share_rel"]:
        df[f"{col}_rank"] = daily[col].rank(pct=True)

    df["spot_accum_score"] = (
        df["ret_1d_rank"]
        + df["spot_share_rel_rank"]
        + (1.0 - df["oi_delta_pct_rank"])
        + (1.0 - df["premium_rel_rank"])
    )
    df["levered_chase_score"] = (
        df["ret_1d_rank"]
        + (1.0 - df["spot_share_rel_rank"])
        + df["oi_delta_pct_rank"]
        + df["premium_rel_rank"]
    )
    df["exchange_rotation_score"] = (
        df["ret_1d_rank"] + df["fut_turnover_shock_rank"] + df["bybit_share_rel_rank"]
    )

    df["spot_flow_bucket"] = np.where(df["spot_share_rel"] >= 0, "spot_high", "spot_low")
    df["oi_bucket"] = np.where(df["oi_delta_pct"] >= 0, "oi_up", "oi_down")
    df["ret_bucket"] = np.where(df["ret_1d"] >= 0, "up", "down")
    df["flow_state"] = df["ret_bucket"] + "|" + df["spot_flow_bucket"] + "|" + df["oi_bucket"]

    return df


def feature_ic(df: pd.DataFrame, universe_mask: pd.Series, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = df.loc[universe_mask].copy()
    for feature in features:
        for horizon in [1, 3, 5]:
            target = f"fwd_{horizon}d"
            per_day = []
            for date, day in subset[["date", feature, target]].dropna().groupby("date"):
                if len(day) < MIN_IC_OBS:
                    continue
                corr = day[feature].corr(day[target], method="spearman")
                if pd.notna(corr):
                    per_day.append(float(corr))
            if not per_day:
                continue
            s = pd.Series(per_day)
            rows.append(
                {
                    "feature": feature,
                    "horizon": target,
                    "mean_ic": s.mean(),
                    "median_ic": s.median(),
                    "hit_rate": (s > 0).mean(),
                    "obs_days": int(len(s)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["feature", "horizon", "mean_ic", "median_ic", "hit_rate", "obs_days"])
    out = out.sort_values(["horizon", "mean_ic"], ascending=[True, False])
    return out


def state_summary(df: pd.DataFrame, universe_mask: pd.Series) -> pd.DataFrame:
    subset = df.loc[universe_mask].copy()
    grouped = subset.groupby("flow_state")
    rows = []
    for state, g in grouped:
        if len(g) < MIN_STATE_OBS:
            continue
        rows.append(
            {
                "flow_state": state,
                "obs": int(len(g)),
                "avg_ret_1d": g["ret_1d"].mean(),
                "avg_fwd_1d": g["fwd_1d"].mean(),
                "avg_fwd_3d": g["fwd_3d"].mean(),
                "avg_fwd_5d": g["fwd_5d"].mean(),
                "median_spot_share_rel": g["spot_share_rel"].median(),
                "median_oi_delta_pct": g["oi_delta_pct"].median(),
                "median_premium_rel": g["premium_rel"].median(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "flow_state",
                "obs",
                "avg_ret_1d",
                "avg_fwd_1d",
                "avg_fwd_3d",
                "avg_fwd_5d",
                "median_spot_share_rel",
                "median_oi_delta_pct",
                "median_premium_rel",
            ]
        )
    out = out.sort_values("avg_fwd_3d", ascending=False)
    return out


def breadth_summary(df: pd.DataFrame, universe_mask: pd.Series, state_names: list[str]) -> pd.DataFrame:
    subset = df.loc[universe_mask, ["date", "symbol", "flow_state", "fwd_1d", "fwd_3d", "fwd_5d"]].copy()
    subset["count"] = 1
    pivot = subset.pivot_table(index="date", columns="flow_state", values="count", aggfunc="sum", fill_value=0)
    pivot["symbol_count"] = subset.groupby("date")["symbol"].nunique()
    for state in state_names:
        if state not in pivot.columns:
            pivot[state] = 0
        pivot[f"{state}_breadth"] = safe_div(pivot[state], pivot["symbol_count"])

    ew = subset.groupby("date")[["fwd_1d", "fwd_3d", "fwd_5d"]].mean()
    pivot = pivot.join(ew, how="left")

    rows = []
    for state in state_names:
        breadth_col = f"{state}_breadth"
        s = pivot[breadth_col].dropna()
        if s.empty:
            continue
        threshold = s.quantile(0.8)
        hi = pivot[pivot[breadth_col] >= threshold]
        lo = pivot[pivot[breadth_col] <= s.quantile(0.2)]
        rows.append(
            {
                "state": state,
                "threshold_80pct": threshold,
                "hi_days": int(len(hi)),
                "lo_days": int(len(lo)),
                "hi_fwd_1d": hi["fwd_1d"].mean(),
                "hi_fwd_3d": hi["fwd_3d"].mean(),
                "hi_fwd_5d": hi["fwd_5d"].mean(),
                "lo_fwd_1d": lo["fwd_1d"].mean(),
                "lo_fwd_3d": lo["fwd_3d"].mean(),
                "lo_fwd_5d": lo["fwd_5d"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "threshold_80pct",
                "hi_days",
                "lo_days",
                "hi_fwd_1d",
                "hi_fwd_3d",
                "hi_fwd_5d",
                "lo_fwd_1d",
                "lo_fwd_3d",
                "lo_fwd_5d",
            ]
        )
    out = out.sort_values("hi_fwd_3d", ascending=False)
    return out


def pick_key_findings(ic: pd.DataFrame, states: pd.DataFrame, breadth: pd.DataFrame) -> dict[str, pd.Series]:
    ic_3d = ic[ic["horizon"] == "fwd_3d"].sort_values("mean_ic", ascending=False)
    key_ic = ic_3d.iloc[0] if not ic_3d.empty else pd.Series(dtype=float)

    best_state = states.iloc[0] if not states.empty else pd.Series(dtype=float)
    worst_state = states.sort_values("avg_fwd_3d", ascending=True).iloc[0] if not states.empty else pd.Series(dtype=float)

    breadth_best = breadth.sort_values("hi_fwd_3d", ascending=False).iloc[0] if not breadth.empty else pd.Series(dtype=float)
    breadth_worst = breadth.sort_values("hi_fwd_3d", ascending=True).iloc[0] if not breadth.empty else pd.Series(dtype=float)

    return {
        "key_ic": key_ic,
        "best_state": best_state,
        "worst_state": worst_state,
        "breadth_best": breadth_best,
        "breadth_worst": breadth_worst,
    }


def pct(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v * 100:.2f}%"


def write_report(
    df: pd.DataFrame,
    ic: pd.DataFrame,
    states: pd.DataFrame,
    breadth: pd.DataFrame,
    findings: dict[str, pd.Series],
    spot_mask: pd.Series,
    futures_mask: pd.Series,
) -> None:
    spot_symbols = int(df.loc[spot_mask, "symbol"].nunique())
    futures_symbols = int(df.loc[futures_mask, "symbol"].nunique())
    spot_dates = int(df.loc[spot_mask, "date"].nunique())

    key_ic = findings["key_ic"]
    best_state = findings["best_state"]
    worst_state = findings["worst_state"]
    breadth_best = findings["breadth_best"]
    breadth_worst = findings["breadth_worst"]

    lines = [
        "# Money Flow Report",
        "",
        "## Coverage",
        f"- Futures-common universe: {futures_symbols} symbols.",
        f"- Full spot+futures universe: {spot_symbols} symbols.",
        f"- Spot analysis window: {spot_dates} daily observations.",
        "",
        "## Core Pattern",
        (
            f"- Best cross-sectional 3-day signal: `{key_ic.get('feature', 'n/a')}` "
            f"with mean daily Spearman IC {key_ic.get('mean_ic', np.nan):.4f} "
            f"across {int(key_ic.get('obs_days', 0))} days."
        ),
        (
            f"- Best symbol state: `{best_state.get('flow_state', 'n/a')}` with "
            f"{pct(best_state.get('avg_fwd_3d', np.nan))} average 3-day forward return "
            f"over {int(best_state.get('obs', 0))} observations."
        ),
        (
            f"- Worst symbol state: `{worst_state.get('flow_state', 'n/a')}` with "
            f"{pct(worst_state.get('avg_fwd_3d', np.nan))} average 3-day forward return "
            f"over {int(worst_state.get('obs', 0))} observations."
        ),
        "",
        "## Interpretation",
        (
            "- `up|spot_high|oi_down` is the cleanest accumulation signature: "
            "price is already rising, spot share is above its own baseline, and open interest is not expanding with it. "
            "That usually behaves like cash-led demand or short-cover plus spot follow-through."
        ),
        (
            "- `up|spot_low|oi_up` is the opposite regime: futures dominate the tape while leverage expands into strength. "
            "That behaves more like a crowded chase and has materially weaker follow-through."
        ),
        (
            "- When this pattern broadens across many coins on the same day, the equal-weight universe also separates: "
            f"high breadth in `{breadth_best.get('state', 'n/a')}` days leads to {pct(breadth_best.get('hi_fwd_3d', np.nan))} "
            f"next-3-day universe return, while high breadth in `{breadth_worst.get('state', 'n/a')}` days leads to "
            f"{pct(breadth_worst.get('hi_fwd_3d', np.nan))}."
        ),
        "",
        "## Files",
        "- `symbol_daily_flows.csv`: per-symbol daily feature set.",
        "- `money_flow_feature_ic.csv`: feature-level predictive ranking.",
        "- `money_flow_state_summary.csv`: forward returns by money-flow state.",
        "- `money_flow_breadth_summary.csv`: market-wide breadth regime results.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not DATA_CSV.exists():
        raise SystemExit(f"Missing dataset: {DATA_CSV}")

    raw = pd.read_csv(DATA_CSV)
    df = build_features(raw)

    futures_mask = (df["total_fut_turnover"] > 0) & df["total_oi_usd"].notna()
    spot_mask = futures_mask & (df["has_full_spot_pair"] == 1) & df["spot_share"].notna()

    feature_list = [
        "spot_share_rel",
        "fut_turnover_shock",
        "spot_turnover_shock",
        "oi_delta_pct",
        "premium_rel",
        "bybit_share_rel",
        "spot_accum_score",
        "levered_chase_score",
        "exchange_rotation_score",
    ]
    ic = feature_ic(df, spot_mask, feature_list)
    states = state_summary(df, spot_mask)
    state_names = [
        "up|spot_high|oi_down",
        "up|spot_low|oi_up",
        "down|spot_high|oi_down",
        "down|spot_low|oi_up",
    ]
    breadth = breadth_summary(df, spot_mask, state_names)

    FEATURE_IC_CSV.write_text(ic.to_csv(index=False))
    STATE_CSV.write_text(states.to_csv(index=False))
    BREADTH_CSV.write_text(breadth.to_csv(index=False))

    findings = pick_key_findings(ic, states, breadth)
    write_report(df, ic, states, breadth, findings, spot_mask, futures_mask)

    print(f"Wrote {FEATURE_IC_CSV}")
    print(f"Wrote {STATE_CSV}")
    print(f"Wrote {BREADTH_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
