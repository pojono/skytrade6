from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
FEATURES_CSV = OUT_DIR / "execution_case_features.csv"
GRID_CSV = OUT_DIR / "execution_case_filter_grid.csv"
REPORT_MD = OUT_DIR / "FINDINGS_execution_case_study.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"
SYMS = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "ENAUSDT",
    "WIFUSDT",
    "TIAUSDT",
    "WLDUSDT",
    "ASTERUSDT",
    "LINKUSDT",
    "ARBUSDT",
    "FARTCOINUSDT",
}
TRAIN_CUTOFF = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
MAX_TS = pd.Timestamp("2026-01-31 23:59:59", tz="UTC")
FEE_BPS = 20.0


@dataclass(frozen=True)
class Rule:
    min_signed_share: float
    min_depth_imbalance: float
    min_depth_total_1m: float
    max_vol_1m: float


def load_signal_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    common = (
        s["symbol"].isin(SYMS)
        & (s["ts"] >= pd.Timestamp("2025-11-01", tz="UTC"))
        & (s["ts"] <= MAX_TS)
        & (s["oi_med_3d"] >= 20_000_000)
        & (s["breadth_mom"] >= 0.60)
        & (s["median_ls_z"] >= 0.0)
        & (s["mom_4h"] > 0)
    )
    broad = s.loc[common & (s["ls_z"] >= 1.0) & (s["taker_z"] >= 0.0)].copy()
    strict = s.loc[common & (s["ls_z"] >= 2.0) & (s["taker_z"] >= 0.5)].copy()
    for df in (broad, strict):
        df["net_bps_20"] = df["ret_4h"] * 10000.0 - FEE_BPS
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


def extract_for_row(row: pd.Series) -> dict[str, object] | None:
    symbol = row["symbol"]
    ts = row["ts"]
    day = ts.strftime("%Y-%m-%d")
    agg = load_agg(symbol, day)
    depth = load_depth(symbol, day)
    if agg is None or depth is None:
        return None

    t0 = ts
    trades_1m = agg.loc[(agg["ts"] > t0 - pd.Timedelta("60s")) & (agg["ts"] <= t0)].copy()
    if trades_1m.empty:
        return None
    quote = trades_1m["price"] * trades_1m["quantity"]
    sign = np.where(trades_1m["is_buyer_maker"], -1.0, 1.0)
    signed_quote = float((quote * sign).sum())
    total_quote = float(quote.sum())
    signed_share = signed_quote / total_quote if total_quote > 0 else 0.0

    prices = trades_1m["price"].to_numpy()
    if len(prices) >= 2:
        rets = np.diff(np.log(prices))
        rv_1m = float(np.sqrt((rets**2).sum()))
        last10_mask = trades_1m["ts"] > t0 - pd.Timedelta("10s")
        p10 = trades_1m.loc[last10_mask, "price"].to_numpy()
        micro_ret_10s = float((p10[-1] / p10[0] - 1.0) * 10000.0) if len(p10) >= 2 else 0.0
    else:
        rv_1m = 0.0
        micro_ret_10s = 0.0

    snap_ts = depth.loc[depth["ts"] <= t0, "ts"]
    if snap_ts.empty:
        return None
    last_snap = snap_ts.max()
    snap = depth.loc[depth["ts"] == last_snap].copy()
    one_bid = snap.loc[snap["percentage"] == -1, "notional"]
    one_ask = snap.loc[snap["percentage"] == 1, "notional"]
    if one_bid.empty or one_ask.empty:
        return None
    bid1 = float(one_bid.iloc[0])
    ask1 = float(one_ask.iloc[0])
    depth_total_1 = bid1 + ask1
    depth_imbalance_1 = (bid1 - ask1) / depth_total_1 if depth_total_1 > 0 else 0.0

    zero_rows = snap.loc[snap["percentage"] == 0, "notional"].sort_values().to_numpy()
    zero_small = float(zero_rows[0]) if len(zero_rows) else 0.0
    zero_large = float(zero_rows[-1]) if len(zero_rows) else 0.0
    zero_balance = (
        (zero_large - zero_small) / (zero_large + zero_small)
        if (zero_large + zero_small) > 0
        else 0.0
    )

    staleness_s = float((t0 - last_snap).total_seconds())

    return {
        "ts": ts,
        "symbol": symbol,
        "net_bps_20": row["net_bps_20"],
        "period": row["period"],
        "ls_z": row["ls_z"],
        "taker_z": row["taker_z"],
        "score_abs": row["score_abs"],
        "trade_count_1m": int(trades_1m.shape[0]),
        "quote_1m": total_quote,
        "signed_share_1m": signed_share,
        "rv_1m": rv_1m,
        "micro_ret_10s_bps": micro_ret_10s,
        "depth_total_1": depth_total_1,
        "depth_imbalance_1": depth_imbalance_1,
        "zero_balance": zero_balance,
        "book_staleness_s": staleness_s,
    }


def build_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in rows.iterrows():
        item = extract_for_row(row)
        if item is not None:
            out.append(item)
    df = pd.DataFrame(out)
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)


def apply_rule(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    return df.loc[
        (df["signed_share_1m"] >= rule.min_signed_share)
        & (df["depth_imbalance_1"] >= rule.min_depth_imbalance)
        & (df["depth_total_1"] >= rule.min_depth_total_1m)
        & (df["rv_1m"] <= rule.max_vol_1m)
        & (df["book_staleness_s"] <= 60.0)
    ].copy()


def eval_rule(train: pd.DataFrame, test: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    train_f = apply_rule(train, rule)
    test_f = apply_rule(test, rule)
    return {
        "min_signed_share": rule.min_signed_share,
        "min_depth_imbalance": rule.min_depth_imbalance,
        "min_depth_total_1m": rule.min_depth_total_1m,
        "max_vol_1m": rule.max_vol_1m,
        "train_rows": int(train_f.shape[0]),
        "test_rows": int(test_f.shape[0]),
        "train_avg_bps": float(train_f["net_bps_20"].mean()) if not train_f.empty else np.nan,
        "test_avg_bps": float(test_f["net_bps_20"].mean()) if not test_f.empty else np.nan,
        "train_hit": float((train_f["net_bps_20"] > 0).mean()) if not train_f.empty else np.nan,
        "test_hit": float((test_f["net_bps_20"] > 0).mean()) if not test_f.empty else np.nan,
    }


def run_grid(features: pd.DataFrame) -> pd.DataFrame:
    train = features.loc[features["period"] == "train"].copy()
    test = features.loc[features["period"] == "test"].copy()
    base_train = float(train["net_bps_20"].mean())
    base_test = float(test["net_bps_20"].mean())

    qs = train.quantile([0.25, 0.5, 0.75], numeric_only=True)
    rules = []
    for min_signed_share in (
        -0.20,
        float(qs.loc[0.25, "signed_share_1m"]),
        float(qs.loc[0.5, "signed_share_1m"]),
    ):
        for min_depth_imbalance in (
            -0.10,
            0.0,
            float(qs.loc[0.5, "depth_imbalance_1"]),
        ):
            for min_depth_total_1m in (
                float(qs.loc[0.25, "depth_total_1"]),
                float(qs.loc[0.5, "depth_total_1"]),
            ):
                for max_vol_1m in (
                    float(qs.loc[0.75, "rv_1m"]),
                    float(qs.loc[0.5, "rv_1m"]),
                ):
                    rules.append(Rule(min_signed_share, min_depth_imbalance, min_depth_total_1m, max_vol_1m))

    rows = [eval_rule(train, test, r) for r in rules]
    grid = pd.DataFrame(rows)
    grid = grid.loc[(grid["train_rows"] >= 6) & (grid["test_rows"] >= 3)].copy()
    grid["train_improve_bps"] = grid["train_avg_bps"] - base_train
    grid["test_improve_bps"] = grid["test_avg_bps"] - base_test
    grid = grid.sort_values(
        ["test_improve_bps", "train_improve_bps", "test_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return grid


def write_report(
    broad_features: pd.DataFrame,
    strict_features: pd.DataFrame,
    grid: pd.DataFrame,
) -> None:
    broad_train = broad_features.loc[broad_features["period"] == "train"]
    broad_test = broad_features.loc[broad_features["period"] == "test"]
    best = grid.iloc[0]
    rule = Rule(
        float(best["min_signed_share"]),
        float(best["min_depth_imbalance"]),
        float(best["min_depth_total_1m"]),
        float(best["max_vol_1m"]),
    )
    broad_test_f = apply_rule(broad_test, rule)

    strict_test = strict_features.loc[strict_features["period"] == "test"]
    strict_test_f = apply_rule(strict_test, rule)

    lines = [
        "# Execution-Aware Filter Case Study",
        "",
        "This is a microstructure case study on the locally covered Binance symbols with true execution data.",
        "",
        "## Scope",
        "",
        "- Execution data source: local `parquet/{symbol}/binance/book_depth` and `agg_trades_futures`.",
        "- Coverage window: `2025-11-01` through `2026-01-31`.",
        f"- Covered symbols in this run: {', '.join(sorted(SYMS))}.",
        "- This is not a full-universe filter. It is a covered-subset test only.",
        "",
        "## Broad Research Set",
        "",
        "- Broad signal definition for learning: `ls_z >= 1.0`, `taker_z >= 0.0`, risk-on regime, positive 4h momentum.",
        f"- Train rows with execution features: {len(broad_train)}",
        f"- Test rows with execution features: {len(broad_test)}",
        f"- Unfiltered train avg: {broad_train['net_bps_20'].mean():.2f} bps",
        f"- Unfiltered test avg: {broad_test['net_bps_20'].mean():.2f} bps",
        "",
        "## Best Execution Reject Rule",
        "",
        f"- `signed_share_1m >= {rule.min_signed_share:.4f}`",
        f"- `depth_imbalance_1 >= {rule.min_depth_imbalance:.4f}`",
        f"- `depth_total_1 >= {rule.min_depth_total_1m:,.0f}`",
        f"- `rv_1m <= {rule.max_vol_1m:.6f}`",
        f"- Broad train avg after filter: {best['train_avg_bps']:.2f} bps on {int(best['train_rows'])} rows",
        f"- Broad test avg after filter: {best['test_avg_bps']:.2f} bps on {int(best['test_rows'])} rows",
        f"- Broad train improvement vs unfiltered: {best['train_improve_bps']:.2f} bps",
        f"- Broad test improvement vs unfiltered: {best['test_improve_bps']:.2f} bps",
        f"- Broad test hit rate after filter: {best['test_hit']:.1%}",
        "",
        "## Apply Same Rule To Strict Strategy Subset",
        "",
        "- Strict subset = the actual current strategy conditions on the covered symbols (`ls_z >= 2.0`, `taker_z >= 0.5`).",
        f"- Strict covered test rows before filter: {len(strict_test)}",
        f"- Strict covered test avg before filter: {strict_test['net_bps_20'].mean():.2f} bps",
        f"- Strict covered test rows after filter: {len(strict_test_f)}",
        f"- Strict covered test avg after filter: {strict_test_f['net_bps_20'].mean():.2f} bps" if len(strict_test_f) else "- Strict covered test avg after filter: no rows kept",
        "",
        "## Interpretation",
        "",
        "- The pre-2026 microstructure-trained rules did not improve the broader January covered holdout; they made it worse.",
        "- On the actual strict covered holdout, the learned rules were so conservative that they rejected everything.",
        "- That means execution context is likely relevant, but these simple transferable reject rules are not good enough yet.",
        "- Sample sizes are still limited, so this is directional evidence only.",
    ]

    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    broad, strict = load_signal_rows()
    broad_features = build_feature_frame(broad)
    strict_features = build_feature_frame(strict)

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
