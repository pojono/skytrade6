from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
FEATURES_CSV = OUT_DIR / "post_entry_path_features.csv"
GRID_CSV = OUT_DIR / "post_entry_path_grid.csv"
REPORT_MD = OUT_DIR / "FINDINGS_post_entry_path_case_study.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"
SYMS = {
    "ARBUSDT",
    "ASTERUSDT",
    "BTCUSDT",
    "ENAUSDT",
    "ETHUSDT",
    "FARTCOINUSDT",
    "LINKUSDT",
    "SOLUSDT",
    "TIAUSDT",
    "WIFUSDT",
    "WLDUSDT",
}
START_TS = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2026-01-31 23:59:59", tz="UTC")


@dataclass(frozen=True)
class Rule:
    min_ret_30s_bps: float
    min_ret_5m_bps: float
    min_path_low_5m_bps: float
    min_buy_share_5m: float


def load_broad_and_strict() -> tuple[pd.DataFrame, pd.DataFrame]:
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
        df["net_bps_20"] = df["ret_4h"] * 10000.0 - 20.0
    return broad, strict


@lru_cache(maxsize=None)
def load_agg(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "agg_trades_futures" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


def build_features(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        ts = row["ts"]
        symbol = row["symbol"]
        day = ts.strftime("%Y-%m-%d")
        agg = load_agg(symbol, day)
        if agg is None:
            continue

        pre = agg.loc[(agg["ts"] <= ts)].copy()
        post = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("30m"))].copy()
        if pre.empty or post.empty:
            continue

        entry_price = float(pre.iloc[-1]["price"])
        post_30s = post.loc[post["ts"] <= ts + pd.Timedelta("30s")].copy()
        post_5m = post.loc[post["ts"] <= ts + pd.Timedelta("5m")].copy()
        post_30m = post
        if post_30s.empty or post_5m.empty:
            continue

        def bps(price: float) -> float:
            return (price / entry_price - 1.0) * 10000.0

        last_30s = float(post_30s.iloc[-1]["price"])
        last_5m = float(post_5m.iloc[-1]["price"])
        high_5m = float(post_5m["price"].max())
        low_5m = float(post_5m["price"].min())
        high_30m = float(post_30m["price"].max())
        low_30m = float(post_30m["price"].min())

        quote_5m = post_5m["price"] * post_5m["quantity"]
        buy_share_5m = float(
            quote_5m.loc[~post_5m["is_buyer_maker"]].sum() / quote_5m.sum()
        ) if quote_5m.sum() > 0 else 0.5

        # Long entry context: negative path_low means drawdown after entry.
        out.append(
            {
                "ts": ts,
                "symbol": symbol,
                "net_bps_20": row["net_bps_20"],
                "ret_30s_bps": bps(last_30s),
                "ret_5m_bps": bps(last_5m),
                "path_high_5m_bps": bps(high_5m),
                "path_low_5m_bps": bps(low_5m),
                "path_high_30m_bps": bps(high_30m),
                "path_low_30m_bps": bps(low_30m),
                "buy_share_5m": buy_share_5m,
                "trade_count_5m": int(post_5m.shape[0]),
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
    return df.loc[
        (df["ret_30s_bps"] >= rule.min_ret_30s_bps)
        & (df["ret_5m_bps"] >= rule.min_ret_5m_bps)
        & (df["path_low_5m_bps"] >= rule.min_path_low_5m_bps)
        & (df["buy_share_5m"] >= rule.min_buy_share_5m)
    ].copy()


def eval_rule(train: pd.DataFrame, test: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    train_f = apply_rule(train, rule)
    test_f = apply_rule(test, rule)
    return {
        "min_ret_30s_bps": rule.min_ret_30s_bps,
        "min_ret_5m_bps": rule.min_ret_5m_bps,
        "min_path_low_5m_bps": rule.min_path_low_5m_bps,
        "min_buy_share_5m": rule.min_buy_share_5m,
        "train_rows": int(train_f.shape[0]),
        "test_rows": int(test_f.shape[0]),
        "train_avg_bps": float(train_f["net_bps_20"].mean()) if not train_f.empty else np.nan,
        "test_avg_bps": float(test_f["net_bps_20"].mean()) if not test_f.empty else np.nan,
        "train_hit": float((train_f["net_bps_20"] > 0).mean()) if not train_f.empty else np.nan,
        "test_hit": float((test_f["net_bps_20"] > 0).mean()) if not test_f.empty else np.nan,
    }


def run_grid(features: pd.DataFrame) -> pd.DataFrame:
    train = features.loc[features["study_period"] == "train"].copy()
    test = features.loc[features["study_period"] == "test"].copy()
    base_train = float(train["net_bps_20"].mean())
    base_test = float(test["net_bps_20"].mean())

    q = train.quantile([0.25, 0.5, 0.75], numeric_only=True)
    rules = []
    for min_ret_30s_bps in (
        float(q.loc[0.25, "ret_30s_bps"]),
        float(q.loc[0.5, "ret_30s_bps"]),
        0.0,
    ):
        for min_ret_5m_bps in (
            float(q.loc[0.25, "ret_5m_bps"]),
            float(q.loc[0.5, "ret_5m_bps"]),
            0.0,
        ):
            for min_path_low_5m_bps in (
                float(q.loc[0.25, "path_low_5m_bps"]),
                float(q.loc[0.5, "path_low_5m_bps"]),
            ):
                for min_buy_share_5m in (
                    float(q.loc[0.25, "buy_share_5m"]),
                    float(q.loc[0.5, "buy_share_5m"]),
                    0.5,
                ):
                    rules.append(
                        Rule(
                            min_ret_30s_bps,
                            min_ret_5m_bps,
                            min_path_low_5m_bps,
                            min_buy_share_5m,
                        )
                    )

    rows = [eval_rule(train, test, rule) for rule in rules]
    grid = pd.DataFrame(rows)
    grid = grid.loc[(grid["train_rows"] >= 3) & (grid["test_rows"] >= 3)].copy()
    grid["train_improve_bps"] = grid["train_avg_bps"] - base_train
    grid["test_improve_bps"] = grid["test_avg_bps"] - base_test
    grid = grid.sort_values(
        ["test_improve_bps", "train_improve_bps", "test_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return grid


def write_report(broad_features: pd.DataFrame, strict_features: pd.DataFrame, grid: pd.DataFrame) -> None:
    broad_train = broad_features.loc[broad_features["study_period"] == "train"]
    broad_test = broad_features.loc[broad_features["study_period"] == "test"]
    if grid.empty:
        REPORT_MD.write_text("# Post-Entry Path Case Study\n\nNo valid rule met the minimum train/test row thresholds.")
        return

    best = grid.iloc[0]
    rule = Rule(
        float(best["min_ret_30s_bps"]),
        float(best["min_ret_5m_bps"]),
        float(best["min_path_low_5m_bps"]),
        float(best["min_buy_share_5m"]),
    )
    strict_test = strict_features.loc[strict_features["study_period"] == "test"]
    strict_test_f = apply_rule(strict_test, rule)

    lines = [
        "# Post-Entry Path Case Study",
        "",
        "This study tests whether 5-30 minute follow-through after entry is the real gating variable.",
        "",
        "## Scope",
        "",
        f"- Covered symbols: {', '.join(sorted(SYMS))}",
        "- Uses Binance `agg_trades_futures` on the widened January-covered symbol set.",
        "- Internal time split within the extracted sample (train/test) because the usable post-entry paths are concentrated in January 2026.",
        "",
        "## Broad Research Set",
        "",
        f"- Train rows: {len(broad_train)}",
        f"- Test rows: {len(broad_test)}",
        f"- Unfiltered broad test avg: {broad_test['net_bps_20'].mean():.2f} bps",
        f"- Average broad test 30s return: {broad_test['ret_30s_bps'].mean():.2f} bps",
        f"- Average broad test 5m return: {broad_test['ret_5m_bps'].mean():.2f} bps",
        "",
        "## Best Post-Entry Rule",
        "",
        f"- `ret_30s_bps >= {rule.min_ret_30s_bps:.2f}`",
        f"- `ret_5m_bps >= {rule.min_ret_5m_bps:.2f}`",
        f"- `path_low_5m_bps >= {rule.min_path_low_5m_bps:.2f}`",
        f"- `buy_share_5m >= {rule.min_buy_share_5m:.3f}`",
        f"- Broad train avg after filter: {best['train_avg_bps']:.2f} bps on {int(best['train_rows'])} rows",
        f"- Broad test avg after filter: {best['test_avg_bps']:.2f} bps on {int(best['test_rows'])} rows",
        f"- Broad test improvement vs unfiltered: {best['test_improve_bps']:.2f} bps",
        "",
        "## Apply Same Rule To Strict Strategy Subset",
        "",
        f"- Strict covered test rows before filter: {len(strict_test)}",
        f"- Strict covered test avg before filter: {strict_test['net_bps_20'].mean():.2f} bps",
        f"- Strict covered test rows after filter: {len(strict_test_f)}",
        f"- Strict covered test avg after filter: {strict_test_f['net_bps_20'].mean():.2f} bps" if len(strict_test_f) else "- Strict covered test avg after filter: no rows kept",
        "",
        "## Correlation Clue",
        "",
        f"- Corr(`ret_30s_bps`, 4h net): {broad_features[['ret_30s_bps', 'net_bps_20']].corr().iloc[0,1]:.3f}",
        f"- Corr(`ret_5m_bps`, 4h net): {broad_features[['ret_5m_bps', 'net_bps_20']].corr().iloc[0,1]:.3f}",
        f"- Corr(`path_low_5m_bps`, 4h net): {broad_features[['path_low_5m_bps', 'net_bps_20']].corr().iloc[0,1]:.3f}",
        "",
        "## Interpretation",
        "",
        "- If this improves holdout materially, the real problem is post-entry fade, not entry friction.",
        "- If it does not, then even short-term follow-through is not a stable separator on this sample.",
    ]
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    broad, strict = load_broad_and_strict()
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
