from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake" / "binance"
OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
RESULTS_CSV = OUT_DIR / "grid_results.csv"
SUMMARY_MD = OUT_DIR / "FINDINGS_codex_exp_2.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

ROLL_BARS = 14 * 24 * 12
HOLD_BARS = 48
MOM_BARS = 48
LIQUIDITY_BARS = 3 * 24 * 12

FEE_ALL_TAKER = 0.0020
FEE_TAKER_MAKER = 0.0014
FEE_ALL_MAKER = 0.0008
FEE_STRESS = 0.0024


@dataclass(frozen=True)
class Config:
    ls_threshold: float
    taker_threshold: float
    min_oi_value: float
    top_n: int
    breadth_threshold: float
    median_ls_threshold: float


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean().shift(1)
    std = series.rolling(window, min_periods=window).std(ddof=0).shift(1)
    z = (series - mean) / std.replace(0.0, pd.NA)
    return z.astype("float64")


def _load_symbol(symbol_dir: Path) -> pd.DataFrame | None:
    metrics_files = sorted(symbol_dir.glob("*_metrics.csv"))
    kline_files = sorted(
        p for p in symbol_dir.glob("*_kline_1m.csv")
        if "mark_price" not in p.name
        and "premium_index" not in p.name
        and "index_price" not in p.name
    )
    if not metrics_files or not kline_files:
        return None

    metrics_parts = []
    for path in metrics_files:
        df = pd.read_csv(
            path,
            usecols=[
                "create_time",
                "sum_open_interest_value",
                "count_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ],
        )
        metrics_parts.append(df)
    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics["ts"] = pd.to_datetime(metrics["create_time"], utc=True)
    metrics = metrics.rename(
        columns={
            "sum_open_interest_value": "oi_value",
            "count_toptrader_long_short_ratio": "ls_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
        }
    )
    metrics = metrics[["ts", "oi_value", "ls_ratio", "taker_ratio"]]
    metrics = metrics.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    kline_parts = []
    for path in kline_files:
        df = pd.read_csv(path, usecols=["open_time", "close"])
        kline_parts.append(df)
    kline = pd.concat(kline_parts, ignore_index=True)
    kline["ts"] = pd.to_datetime(kline["open_time"], unit="ms", utc=True)
    kline["close"] = pd.to_numeric(kline["close"], errors="coerce")
    kline = (
        kline[["ts", "close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )
    close_5m = kline["close"].resample("5min", label="right", closed="right").last()
    close_5m = close_5m.rename("close").reset_index()

    merged = metrics.merge(close_5m, on="ts", how="inner")
    if merged.empty:
        return None

    merged["ret_4h"] = merged["close"].shift(-HOLD_BARS) / merged["close"] - 1.0
    merged["mom_4h"] = merged["close"] / merged["close"].shift(MOM_BARS) - 1.0
    merged["ls_z"] = _safe_zscore(merged["ls_ratio"], ROLL_BARS)
    merged["taker_z"] = _safe_zscore(merged["taker_ratio"], ROLL_BARS)
    merged["oi_med_3d"] = (
        merged["oi_value"].rolling(LIQUIDITY_BARS, min_periods=LIQUIDITY_BARS).median().shift(1)
    )
    merged["hour"] = merged["ts"].dt.hour
    merged["minute"] = merged["ts"].dt.minute

    sample = merged.loc[
        (merged["minute"] == 5)
        & (merged["hour"] % 4 == 0)
        & merged["ret_4h"].notna()
        & merged["mom_4h"].notna()
        & merged["ls_z"].notna()
        & merged["taker_z"].notna()
        & merged["oi_med_3d"].notna(),
        ["ts", "ret_4h", "mom_4h", "ls_z", "taker_z", "oi_med_3d"],
    ].copy()
    if sample.empty:
        return None
    sample["symbol"] = symbol_dir.name
    sample["score_abs"] = sample["ls_z"].abs() + 0.35 * sample["taker_z"].abs()
    return sample


def build_samples(max_workers: int = 8) -> pd.DataFrame:
    symbol_dirs = sorted(p for p in DATALAKE.iterdir() if p.is_dir())
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            parts = list(pool.map(_load_symbol, symbol_dirs))
    except (PermissionError, OSError):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parts = list(pool.map(_load_symbol, symbol_dirs))
    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        raise RuntimeError("No symbol samples were built.")
    samples = pd.concat(frames, ignore_index=True)
    samples = samples.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return samples


def _summarize_side(frame: pd.DataFrame, sign: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts", "raw_ret", "n_positions"])
    working = frame.copy()
    if sign > 0:
        working["raw_ret"] = working["ret_4h"]
    else:
        working["raw_ret"] = -working["ret_4h"]
    grouped = (
        working.groupby("ts", as_index=False)
        .agg(raw_ret=("raw_ret", "mean"), n_positions=("symbol", "count"))
    )
    return grouped


def _trade_stats(rets: pd.Series, fee: float) -> tuple[float, float, float]:
    if rets.empty:
        return (math.nan, math.nan, math.nan)
    net = rets - fee
    avg_bps = net.mean() * 10000.0
    win_rate = (net > 0).mean()
    std = net.std(ddof=0)
    sharpe_like = math.nan
    if std and std > 0:
        sharpe_like = (net.mean() / std) * math.sqrt(len(net))
    return (avg_bps, win_rate, sharpe_like)


def add_regime_features(samples: pd.DataFrame) -> pd.DataFrame:
    liquid = samples.loc[samples["oi_med_3d"] >= 50_000_000.0].copy()
    regime = (
        liquid.groupby("ts", as_index=False)
        .agg(
            breadth_mom=("mom_4h", lambda s: (s > 0).mean()),
            median_ls_z=("ls_z", "median"),
            median_taker_z=("taker_z", "median"),
        )
    )
    return samples.merge(regime, on="ts", how="left")


def backtest_config(samples: pd.DataFrame, cfg: Config) -> dict[str, float | int]:
    eligible = samples.loc[
        (samples["oi_med_3d"] >= cfg.min_oi_value)
        & (samples["breadth_mom"] >= cfg.breadth_threshold)
        & (samples["median_ls_z"] >= cfg.median_ls_threshold)
    ].copy()
    if eligible.empty:
        return {
            "ls_threshold": cfg.ls_threshold,
            "taker_threshold": cfg.taker_threshold,
            "min_oi_value": cfg.min_oi_value,
            "top_n": cfg.top_n,
            "breadth_threshold": cfg.breadth_threshold,
            "median_ls_threshold": cfg.median_ls_threshold,
            "train_trades": 0,
            "test_trades": 0,
        }

    longs = (
        eligible.loc[
            (eligible["ls_z"] >= cfg.ls_threshold)
            & (eligible["taker_z"] >= cfg.taker_threshold)
            & (eligible["mom_4h"] > 0)
        ]
        .sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(cfg.top_n)
    )

    long_trades = _summarize_side(longs, sign=1)
    trades = pd.concat([long_trades], ignore_index=True)
    if trades.empty:
        return {
            "ls_threshold": cfg.ls_threshold,
            "taker_threshold": cfg.taker_threshold,
            "min_oi_value": cfg.min_oi_value,
            "top_n": cfg.top_n,
            "breadth_threshold": cfg.breadth_threshold,
            "median_ls_threshold": cfg.median_ls_threshold,
            "train_trades": 0,
            "test_trades": 0,
        }

    train = trades.loc[trades["ts"] <= TRAIN_END, "raw_ret"]
    test = trades.loc[trades["ts"] >= TEST_START, "raw_ret"]

    train_avg_taker, train_win_taker, train_sharpe_taker = _trade_stats(train, FEE_ALL_TAKER)
    test_avg_taker, test_win_taker, test_sharpe_taker = _trade_stats(test, FEE_ALL_TAKER)
    test_avg_taker_maker, _, _ = _trade_stats(test, FEE_TAKER_MAKER)
    test_avg_all_maker, _, _ = _trade_stats(test, FEE_ALL_MAKER)
    test_avg_stress, _, _ = _trade_stats(test, FEE_STRESS)

    return {
        "ls_threshold": cfg.ls_threshold,
        "taker_threshold": cfg.taker_threshold,
        "min_oi_value": int(cfg.min_oi_value),
        "top_n": cfg.top_n,
        "breadth_threshold": cfg.breadth_threshold,
        "median_ls_threshold": cfg.median_ls_threshold,
        "train_trades": int(train.shape[0]),
        "test_trades": int(test.shape[0]),
        "train_avg_bps_all_taker": train_avg_taker,
        "train_win_rate_all_taker": train_win_taker,
        "train_sharpe_like_all_taker": train_sharpe_taker,
        "test_avg_bps_all_taker": test_avg_taker,
        "test_win_rate_all_taker": test_win_taker,
        "test_sharpe_like_all_taker": test_sharpe_taker,
        "test_avg_bps_taker_maker": test_avg_taker_maker,
        "test_avg_bps_all_maker": test_avg_all_maker,
        "test_avg_bps_stress_24bps": test_avg_stress,
        "train_months_active": int(
            trades.loc[trades["ts"] <= TRAIN_END, "ts"].dt.strftime("%Y-%m").nunique()
        ),
        "test_months_active": int(
            trades.loc[trades["ts"] >= TEST_START, "ts"].dt.strftime("%Y-%m").nunique()
        ),
    }


def run_grid(samples: pd.DataFrame) -> pd.DataFrame:
    configs = [
        Config(ls_threshold, taker_threshold, min_oi, top_n, breadth, median_ls)
        for ls_threshold in (1.0, 1.5, 2.0)
        for taker_threshold in (0.0, 0.5, 1.0)
        for min_oi in (20_000_000.0, 50_000_000.0, 100_000_000.0)
        for top_n in (1, 2, 3)
        for breadth in (0.50, 0.55, 0.60, 0.65)
        for median_ls in (0.0, 0.25, 0.5)
    ]
    rows = [backtest_config(samples, cfg) for cfg in configs]
    results = pd.DataFrame(rows)
    results = results.loc[
        (results["train_trades"] >= 25)
        & (results["test_trades"] >= 25)
        & (results["train_avg_bps_all_taker"].notna())
        & (results["train_avg_bps_all_taker"] > 0)
        & (results["test_avg_bps_all_taker"] > 0)
    ].copy()
    results["consistency_bps"] = results[
        ["train_avg_bps_all_taker", "test_avg_bps_all_taker"]
    ].min(axis=1)
    results = results.sort_values(
        [
            "consistency_bps",
            "test_avg_bps_all_taker",
            "train_avg_bps_all_taker",
            "test_trades",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return results


def monthly_breakdown(samples: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    eligible = samples.loc[
        (samples["oi_med_3d"] >= cfg.min_oi_value)
        & (samples["breadth_mom"] >= cfg.breadth_threshold)
        & (samples["median_ls_z"] >= cfg.median_ls_threshold)
    ].copy()
    longs = (
        eligible.loc[
            (eligible["ls_z"] >= cfg.ls_threshold)
            & (eligible["taker_z"] >= cfg.taker_threshold)
            & (eligible["mom_4h"] > 0)
        ]
        .sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(cfg.top_n)
    )
    trades = pd.concat([_summarize_side(longs, sign=1)], ignore_index=True)
    if trades.empty:
        return pd.DataFrame()
    trades["month"] = trades["ts"].dt.strftime("%Y-%m")
    monthly = (
        trades.groupby("month", as_index=False)
        .agg(
            trades=("raw_ret", "count"),
            avg_bps_all_taker=("raw_ret", lambda s: (s.mean() - FEE_ALL_TAKER) * 10000.0),
            avg_bps_taker_maker=("raw_ret", lambda s: (s.mean() - FEE_TAKER_MAKER) * 10000.0),
            avg_bps_stress_24bps=("raw_ret", lambda s: (s.mean() - FEE_STRESS) * 10000.0),
            hit_rate_all_taker=("raw_ret", lambda s: ((s - FEE_ALL_TAKER) > 0).mean()),
        )
    )
    return monthly


def write_summary(samples: pd.DataFrame, results: pd.DataFrame) -> None:
    best = results.iloc[0]
    cfg = Config(
        ls_threshold=float(best["ls_threshold"]),
        taker_threshold=float(best["taker_threshold"]),
        min_oi_value=float(best["min_oi_value"]),
        top_n=int(best["top_n"]),
        breadth_threshold=float(best["breadth_threshold"]),
        median_ls_threshold=float(best["median_ls_threshold"]),
    )
    monthly = monthly_breakdown(samples, cfg)

    eligible = samples.loc[samples["oi_med_3d"] >= cfg.min_oi_value].copy()
    train_syms = eligible.loc[eligible["ts"] <= TRAIN_END, "symbol"].nunique()
    test_syms = eligible.loc[eligible["ts"] >= TEST_START, "symbol"].nunique()
    total_rows = len(samples)

    lines = [
        "# Codex Exp 2: Fee-Aware LS Momentum",
        "",
        "Independent research from raw `datalake/binance` daily CSVs only.",
        "",
        "## Method",
        "",
        "- Universe: Binance futures symbols with local `metrics` and `kline_1m` files.",
        "- Features per symbol: 14-day rolling z-scores of top-trader long/short ratio and taker long/short volume ratio, plus 4h price momentum.",
        "- Market regime filter: only trade when the liquid-universe breadth is risk-on (`share of positive 4h momentum >= threshold`) and median LS z-score is not bearish.",
        "- Signal cadence: every 4 hours at `HH:05 UTC`.",
        "- Long-only rule: `ls_z >= threshold`, `taker_z >= threshold`, and 4h momentum > 0.",
        "- Selection: top `N` symbols by `abs(ls_z) + 0.35 * abs(taker_z)`.",
        "- Hold: fixed 4 hours, no stop, no compounding.",
        "- Fees tested: all taker 20 bps, taker entry + maker exit 14 bps, all maker 8 bps, and a 24 bps stress case.",
        "",
        "## Data Coverage",
        "",
        f"- Sample rows: {total_rows:,}",
        f"- Eligible symbols after liquidity filter in train: {train_syms}",
        f"- Eligible symbols after liquidity filter in test: {test_syms}",
        f"- Train cutoff: through {TRAIN_END.date()}",
        f"- Test period: from {TEST_START.date()}",
        "",
        "## Best Configuration (positive in train and test, ranked by the weaker period)",
        "",
        f"- `ls_threshold={cfg.ls_threshold}`",
        f"- `taker_threshold={cfg.taker_threshold}`",
        f"- `min_oi_value={int(cfg.min_oi_value):,}`",
        f"- `top_n={cfg.top_n}`",
        f"- `breadth_threshold={cfg.breadth_threshold}`",
        f"- `median_ls_threshold={cfg.median_ls_threshold}`",
        f"- Train trades: {int(best['train_trades'])}",
        f"- Test trades: {int(best['test_trades'])}",
        f"- Train avg/trade after 20 bps: {best['train_avg_bps_all_taker']:.2f} bps",
        f"- Test avg/trade after 20 bps: {best['test_avg_bps_all_taker']:.2f} bps",
        f"- Consistency score (min of train/test): {best['consistency_bps']:.2f} bps",
        f"- Test avg/trade after 14 bps: {best['test_avg_bps_taker_maker']:.2f} bps",
        f"- Test avg/trade after 8 bps: {best['test_avg_bps_all_maker']:.2f} bps",
        f"- Test avg/trade after 24 bps stress: {best['test_avg_bps_stress_24bps']:.2f} bps",
        f"- Test win rate after 20 bps: {best['test_win_rate_all_taker']:.1%}",
        "",
        "## Notes",
        "",
        "- This only prices explicit fees. It does not model slippage, queue position, partial fills, borrow/funding transfers, or capital constraints.",
        "- Because the signal is slow (4h horizon), maker exits are at least operationally plausible; all-taker is the stricter benchmark.",
        "- If all-taker stays positive, the signal is harder to dismiss as fee leakage.",
        "",
    ]

    if not monthly.empty:
        lines.extend(
            [
                "## Monthly Test Breakdown",
                "",
                "| Month | Trades | Avg 20bps | Avg 14bps | Avg 24bps | Hit Rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in monthly.loc[monthly["month"] >= "2026-01"].iterrows():
            lines.append(
                f"| {row['month']} | {int(row['trades'])} | "
                f"{row['avg_bps_all_taker']:.2f} | "
                f"{row['avg_bps_taker_maker']:.2f} | "
                f"{row['avg_bps_stress_24bps']:.2f} | "
                f"{row['hit_rate_all_taker']:.1%} |"
            )
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    samples = build_samples()
    samples = add_regime_features(samples)
    samples.to_csv(SAMPLES_CSV, index=False)
    results = run_grid(samples)
    results.to_csv(RESULTS_CSV, index=False)
    write_summary(samples, results)
    print(f"Wrote {SAMPLES_CSV}")
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
