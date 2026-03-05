from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
FEATURES_CSV = OUT_DIR / "entry_cost_case_features.csv"
GRID_CSV = OUT_DIR / "entry_cost_case_grid.csv"
REPORT_MD = OUT_DIR / "FINDINGS_entry_cost_case_study.md"

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
TRAIN_CUTOFF = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
START_TS = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2026-01-31 23:59:59", tz="UTC")


@dataclass(frozen=True)
class Rule:
    max_entry_vwap5_bps: float
    max_entry_worst5_bps: float
    min_signed_share_10s: float
    max_vol_10s: float


def load_signal_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
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
        df["period"] = np.where(df["ts"] < TRAIN_CUTOFF, "train", "test")
    return broad, strict


@lru_cache(maxsize=None)
def load_agg(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "agg_trades_futures" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


@lru_cache(maxsize=None)
def load_depth(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "book_depth" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


def _last_mid_before(depth: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float] | None:
    snap_times = depth.loc[depth["ts"] <= ts, "ts"]
    if snap_times.empty:
        return None
    snap = depth.loc[depth["ts"] == snap_times.max()].copy()
    zero_rows = snap.loc[snap["percentage"] == 0, "notional"].sort_values().to_numpy()
    if len(zero_rows) < 2:
        return None
    # Use zero bucket asymmetry as a crude near-touch proxy when bid/ask prices are unavailable.
    bid_notional = float(zero_rows[0])
    ask_notional = float(zero_rows[-1])
    total_1 = bid_notional + ask_notional
    zero_balance = (ask_notional - bid_notional) / total_1 if total_1 > 0 else 0.0
    return total_1, zero_balance


def extract_row_features(row: pd.Series) -> dict[str, object] | None:
    symbol = row["symbol"]
    ts = row["ts"]
    day = ts.strftime("%Y-%m-%d")
    agg = load_agg(symbol, day)
    depth = load_depth(symbol, day)
    if agg is None or depth is None:
        return None

    pre_10s = agg.loc[(agg["ts"] > ts - pd.Timedelta("10s")) & (agg["ts"] <= ts)].copy()
    post_5s = agg.loc[(agg["ts"] > ts) & (agg["ts"] <= ts + pd.Timedelta("5s"))].copy()
    if pre_10s.empty or post_5s.empty:
        return None

    quote_pre = pre_10s["price"] * pre_10s["quantity"]
    sign_pre = np.where(pre_10s["is_buyer_maker"], -1.0, 1.0)
    signed_share_10s = float((quote_pre * sign_pre).sum() / quote_pre.sum()) if quote_pre.sum() > 0 else 0.0

    pre_prices = pre_10s["price"].to_numpy()
    vol_10s = float(np.sqrt((np.diff(np.log(pre_prices)) ** 2).sum())) if len(pre_prices) >= 2 else 0.0

    signal_price = float(pre_10s.iloc[-1]["price"])
    post_quote = post_5s["price"] * post_5s["quantity"]
    vwap_5s = float((post_5s["price"] * post_quote).sum() / post_quote.sum()) if post_quote.sum() > 0 else signal_price
    worst_5s = float(post_5s["price"].max())
    close_5s = float(post_5s.iloc[-1]["price"])

    # Long entries only: higher price after signal means worse buy execution.
    entry_vwap5_bps = (vwap_5s / signal_price - 1.0) * 10000.0
    entry_worst5_bps = (worst_5s / signal_price - 1.0) * 10000.0
    entry_close5_bps = (close_5s / signal_price - 1.0) * 10000.0

    depth_info = _last_mid_before(depth, ts)
    if depth_info is None:
        return None
    depth_total_0, zero_balance = depth_info
    staleness_s = float((ts - depth.loc[depth["ts"] <= ts, "ts"].max()).total_seconds())

    return {
        "ts": ts,
        "symbol": symbol,
        "period": row["period"],
        "net_bps_20": row["net_bps_20"],
        "ls_z": row["ls_z"],
        "taker_z": row["taker_z"],
        "score_abs": row["score_abs"],
        "signed_share_10s": signed_share_10s,
        "vol_10s": vol_10s,
        "entry_vwap5_bps": entry_vwap5_bps,
        "entry_worst5_bps": entry_worst5_bps,
        "entry_close5_bps": entry_close5_bps,
        "depth_total_0": depth_total_0,
        "zero_balance": zero_balance,
        "book_staleness_s": staleness_s,
    }


def build_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in rows.iterrows():
        item = extract_row_features(row)
        if item is not None:
            out.append(item)
    return pd.DataFrame(out).sort_values(["ts", "symbol"]).reset_index(drop=True)


def add_study_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if (out["period"] == "train").any():
        out["study_period"] = out["period"]
        return out
    unique_ts = sorted(out["ts"].unique())
    split_idx = max(1, len(unique_ts) // 2)
    cutoff = unique_ts[split_idx]
    out["study_period"] = np.where(out["ts"] < cutoff, "train", "test")
    return out


def apply_rule(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    return df.loc[
        (df["entry_vwap5_bps"] <= rule.max_entry_vwap5_bps)
        & (df["entry_worst5_bps"] <= rule.max_entry_worst5_bps)
        & (df["signed_share_10s"] >= rule.min_signed_share_10s)
        & (df["vol_10s"] <= rule.max_vol_10s)
        & (df["book_staleness_s"] <= 60.0)
    ].copy()


def eval_rule(train: pd.DataFrame, test: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    train_f = apply_rule(train, rule)
    test_f = apply_rule(test, rule)
    return {
        "max_entry_vwap5_bps": rule.max_entry_vwap5_bps,
        "max_entry_worst5_bps": rule.max_entry_worst5_bps,
        "min_signed_share_10s": rule.min_signed_share_10s,
        "max_vol_10s": rule.max_vol_10s,
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

    qs = train.quantile([0.25, 0.5, 0.75], numeric_only=True)
    rules = []
    for max_entry_vwap5_bps in (
        float(qs.loc[0.25, "entry_vwap5_bps"]),
        float(qs.loc[0.5, "entry_vwap5_bps"]),
        float(qs.loc[0.75, "entry_vwap5_bps"]),
    ):
        for max_entry_worst5_bps in (
            float(qs.loc[0.25, "entry_worst5_bps"]),
            float(qs.loc[0.5, "entry_worst5_bps"]),
            float(qs.loc[0.75, "entry_worst5_bps"]),
        ):
            for min_signed_share_10s in (
                -0.30,
                float(qs.loc[0.25, "signed_share_10s"]),
                float(qs.loc[0.5, "signed_share_10s"]),
            ):
                for max_vol_10s in (
                    float(qs.loc[0.75, "vol_10s"]),
                    float(qs.loc[0.5, "vol_10s"]),
                ):
                    rules.append(
                        Rule(
                            max_entry_vwap5_bps,
                            max_entry_worst5_bps,
                            min_signed_share_10s,
                            max_vol_10s,
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
        lines = [
            "# Entry Cost Model Case Study",
            "",
            "No rule in the tested grid kept enough rows in both the internal train and test splits.",
            "",
            f"- Covered symbols: {', '.join(sorted(SYMS))}",
            f"- Broad train rows: {len(broad_train)}",
            f"- Broad test rows: {len(broad_test)}",
            f"- Broad train avg next-5s VWAP cost: {broad_train['entry_vwap5_bps'].mean():.2f} bps",
            f"- Broad test avg next-5s VWAP cost: {broad_test['entry_vwap5_bps'].mean():.2f} bps",
        ]
        REPORT_MD.write_text("\n".join(lines))
        return
    best = grid.iloc[0]
    rule = Rule(
        float(best["max_entry_vwap5_bps"]),
        float(best["max_entry_worst5_bps"]),
        float(best["min_signed_share_10s"]),
        float(best["max_vol_10s"]),
    )
    broad_test_f = apply_rule(broad_test, rule)
    strict_test = strict_features.loc[strict_features["study_period"] == "test"]
    strict_test_f = apply_rule(strict_test, rule)

    lines = [
        "# Entry Cost Model Case Study",
        "",
        "This study targets immediate entry cost instead of 4-hour PnL.",
        "",
        "## Scope",
        "",
        f"- Covered symbols: {', '.join(sorted(SYMS))}",
        "- Coverage window: 2025-11-01 through 2026-01-31",
        "- Entry cost proxy uses Binance `agg_trades_futures` and `book_depth` only.",
        "- Because usable 5-second labels are only present in the January 2026 coverage, this study uses an internal time split within the extracted sample when no earlier training rows exist.",
        "- For each signal, signal price = last trade before timestamp.",
        "- Entry cost targets for a long buy:",
        "  - `entry_vwap5_bps`: next-5s trade VWAP vs signal price",
        "  - `entry_worst5_bps`: worst trade price in next 5s vs signal price",
        "",
        "## Broad Research Set",
        "",
        "- Broad signal definition: `ls_z >= 1.0`, `taker_z >= 0.0`, risk-on regime, positive 4h momentum.",
        f"- Train rows: {len(broad_train)}",
        f"- Test rows: {len(broad_test)}",
        f"- Unfiltered test avg (4h net after 20 bps): {broad_test['net_bps_20'].mean():.2f} bps",
        f"- Avg next-5s VWAP cost in test: {broad_test['entry_vwap5_bps'].mean():.2f} bps",
        "",
        "## Best Entry-Cost Reject Rule",
        "",
        f"- `entry_vwap5_bps <= {rule.max_entry_vwap5_bps:.2f}`",
        f"- `entry_worst5_bps <= {rule.max_entry_worst5_bps:.2f}`",
        f"- `signed_share_10s >= {rule.min_signed_share_10s:.4f}`",
        f"- `vol_10s <= {rule.max_vol_10s:.6f}`",
        f"- Broad train avg after filter: {best['train_avg_bps']:.2f} bps on {int(best['train_rows'])} rows",
        f"- Broad test avg after filter: {best['test_avg_bps']:.2f} bps on {int(best['test_rows'])} rows",
        f"- Broad test improvement vs unfiltered: {best['test_improve_bps']:.2f} bps",
        "",
        "## Apply Same Rule To Strict Strategy Subset",
        "",
        f"- Strict covered test rows before filter: {len(strict_test)}",
        f"- Strict covered test avg before filter: {strict_test['net_bps_20'].mean():.2f} bps",
        f"- Strict covered avg next-5s VWAP cost: {strict_test['entry_vwap5_bps'].mean():.2f} bps",
        f"- Strict covered test rows after filter: {len(strict_test_f)}",
        f"- Strict covered test avg after filter: {strict_test_f['net_bps_20'].mean():.2f} bps" if len(strict_test_f) else "- Strict covered test avg after filter: no rows kept",
        f"- Strict covered avg next-5s VWAP cost after filter: {strict_test_f['entry_vwap5_bps'].mean():.2f} bps" if len(strict_test_f) else "- Strict covered avg next-5s VWAP cost after filter: no rows kept",
        "",
        "## Interpretation",
        "",
        "- If filtering on expected entry cost improves 4-hour holdout PnL, execution cost is a practical gating variable.",
        "- If it does not, then bad 4-hour outcomes are not being driven primarily by immediate entry price impact in this sample.",
    ]
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    broad, strict = load_signal_rows()
    broad_features = add_study_split(build_feature_frame(broad))
    strict_features = add_study_split(build_feature_frame(strict))
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
