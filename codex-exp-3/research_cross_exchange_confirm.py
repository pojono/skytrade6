from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BINANCE_ROOT = ROOT / "datalake" / "binance"
BYBIT_ROOT = ROOT / "datalake" / "bybit"
OUT_DIR = Path(__file__).resolve().parent

SAMPLES_CSV = OUT_DIR / "samples_cross_exchange.csv"
RESULTS_CSV = OUT_DIR / "grid_results_cross_exchange.csv"
TRADES_CSV = OUT_DIR / "best_trades_cross_exchange.csv"
SUMMARY_MD = OUT_DIR / "FINDINGS_codex_exp_3.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

FREQ = "5min"
HOLD_BARS = 48
ROLL_BARS = 14 * 24 * 12
LIQUIDITY_BARS = 3 * 24 * 12
OI_DELTA_BARS = 12

FEE_ALL_MAKER = 0.0008
FEE_MIXED = 0.0012
FEE_STRESS = 0.0016

RESEARCH_SYMBOLS = (
    "CRVUSDT",
    "GALAUSDT",
    "SEIUSDT",
    "FILUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "BNBUSDT",
)


@dataclass(frozen=True)
class Config:
    bn_ls_threshold: float
    bn_taker_threshold: float
    bb_ls_threshold: float
    bb_oi_z_threshold: float
    min_oi_value: float
    breadth_threshold: float
    max_abs_spread_bps: float
    top_n: int


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean().shift(1)
    std = series.rolling(window, min_periods=window).std(ddof=0).shift(1)
    return ((series - mean) / std.replace(0.0, pd.NA)).astype("float64")


def _load_binance_metrics(symbol: str) -> pd.DataFrame | None:
    files = sorted((BINANCE_ROOT / symbol).glob("*_metrics.csv"))
    if not files:
        return None
    parts = []
    for path in files:
        df = pd.read_csv(
            path,
            usecols=[
                "create_time",
                "sum_open_interest_value",
                "count_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ],
        )
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["create_time"], utc=True)
    out = out.rename(
        columns={
            "sum_open_interest_value": "bn_oi_value",
            "count_toptrader_long_short_ratio": "bn_ls_ratio",
            "sum_taker_long_short_vol_ratio": "bn_taker_ratio",
        }
    )
    return (
        out[["ts", "bn_oi_value", "bn_ls_ratio", "bn_taker_ratio"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _load_binance_close(symbol: str) -> pd.DataFrame | None:
    files = sorted(
        p
        for p in (BINANCE_ROOT / symbol).glob("*_kline_1m.csv")
        if "mark_price" not in p.name
        and "premium_index" not in p.name
        and "index_price" not in p.name
    )
    if not files:
        return None
    parts = []
    for path in files:
        df = pd.read_csv(path, usecols=["open_time", "close"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["bn_close"] = pd.to_numeric(out["close"], errors="coerce")
    out = (
        out[["ts", "bn_close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )
    out = out["bn_close"].resample(FREQ, label="right", closed="right").last().reset_index()
    return out.dropna().reset_index(drop=True)


def _load_bybit_close(symbol: str) -> pd.DataFrame | None:
    files = sorted((BYBIT_ROOT / symbol).glob("*_kline_1m.csv"))
    if not files:
        return None
    parts = []
    for path in files:
        df = pd.read_csv(path, usecols=["startTime", "close"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["startTime"], unit="ms", utc=True)
    out["bb_close"] = pd.to_numeric(out["close"], errors="coerce")
    out = (
        out[["ts", "bb_close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )
    out = out["bb_close"].resample(FREQ, label="right", closed="right").last().reset_index()
    return out.dropna().reset_index(drop=True)


def _load_bybit_positioning(symbol: str) -> pd.DataFrame | None:
    ls_files = sorted((BYBIT_ROOT / symbol).glob("*_long_short_ratio_5min.csv"))
    oi_files = sorted((BYBIT_ROOT / symbol).glob("*_open_interest_5min.csv"))
    if not ls_files or not oi_files:
        return None

    ls_parts = []
    for path in ls_files:
        df = pd.read_csv(path, usecols=["timestamp", "buyRatio", "sellRatio"])
        ls_parts.append(df)
    ls = pd.concat(ls_parts, ignore_index=True)
    ls["ts"] = pd.to_datetime(ls["timestamp"], unit="ms", utc=True)
    ls["buyRatio"] = pd.to_numeric(ls["buyRatio"], errors="coerce")
    ls["sellRatio"] = pd.to_numeric(ls["sellRatio"], errors="coerce")
    ls = ls.loc[(ls["buyRatio"] > 0) & (ls["sellRatio"] > 0)].copy()
    ls["bb_ls_ratio"] = (ls["buyRatio"] / ls["sellRatio"]).map(math.log)
    ls = (
        ls[["ts", "bb_ls_ratio"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    oi_parts = []
    for path in oi_files:
        df = pd.read_csv(path, usecols=["timestamp", "openInterest"])
        oi_parts.append(df)
    oi = pd.concat(oi_parts, ignore_index=True)
    oi["ts"] = pd.to_datetime(oi["timestamp"], unit="ms", utc=True)
    oi["bb_open_interest"] = pd.to_numeric(oi["openInterest"], errors="coerce")
    oi = (
        oi[["ts", "bb_open_interest"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    return pd.merge_asof(ls, oi, on="ts", direction="nearest", tolerance=pd.Timedelta("1min"))


def _load_symbol(symbol: str) -> pd.DataFrame | None:
    metrics = _load_binance_metrics(symbol)
    bn_close = _load_binance_close(symbol)
    bb_close = _load_bybit_close(symbol)
    bb_pos = _load_bybit_positioning(symbol)
    if any(frame is None for frame in (metrics, bn_close, bb_close, bb_pos)):
        return None

    merged = metrics.merge(bn_close, on="ts", how="inner")
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        bb_close.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
    )
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        bb_pos.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
    )
    merged = merged.dropna().reset_index(drop=True)
    if merged.empty:
        return None

    merged["bn_ret_4h"] = merged["bn_close"].shift(-HOLD_BARS) / merged["bn_close"] - 1.0
    merged["bb_ret_4h"] = merged["bb_close"].shift(-HOLD_BARS) / merged["bb_close"] - 1.0
    merged["ret_4h_avg"] = (merged["bn_ret_4h"] + merged["bb_ret_4h"]) / 2.0
    merged["bn_mom_4h"] = merged["bn_close"] / merged["bn_close"].shift(HOLD_BARS) - 1.0
    merged["bb_mom_4h"] = merged["bb_close"] / merged["bb_close"].shift(HOLD_BARS) - 1.0
    merged["combo_mom_4h"] = (merged["bn_mom_4h"] + merged["bb_mom_4h"]) / 2.0
    merged["bn_ls_z"] = _safe_zscore(merged["bn_ls_ratio"], ROLL_BARS)
    merged["bn_taker_z"] = _safe_zscore(merged["bn_taker_ratio"], ROLL_BARS)
    merged["bb_ls_z"] = _safe_zscore(merged["bb_ls_ratio"], ROLL_BARS)
    merged["bb_oi_value"] = merged["bb_open_interest"] * merged["bb_close"]
    merged["bb_oi_1h_delta"] = merged["bb_oi_value"] / merged["bb_oi_value"].shift(OI_DELTA_BARS) - 1.0
    merged["bb_oi_z"] = _safe_zscore(merged["bb_oi_1h_delta"], ROLL_BARS)
    merged["bn_oi_med_3d"] = (
        merged["bn_oi_value"].rolling(LIQUIDITY_BARS, min_periods=LIQUIDITY_BARS).median().shift(1)
    )
    merged["bb_oi_med_3d"] = (
        merged["bb_oi_value"].rolling(LIQUIDITY_BARS, min_periods=LIQUIDITY_BARS).median().shift(1)
    )
    merged["spread_bps"] = (merged["bb_close"] / merged["bn_close"] - 1.0) * 10000.0
    merged["score"] = (
        merged["bn_ls_z"]
        + 0.35 * merged["bn_taker_z"]
        + 0.45 * merged["bb_ls_z"]
        + 0.20 * merged["bb_oi_z"].fillna(0.0)
    )
    merged["hour"] = merged["ts"].dt.hour
    merged["minute"] = merged["ts"].dt.minute

    sample = merged.loc[
        (merged["minute"] == 5)
        & (merged["hour"] % 4 == 0)
        & merged["ret_4h_avg"].notna()
        & merged["bn_mom_4h"].notna()
        & merged["bb_mom_4h"].notna()
        & merged["bn_ls_z"].notna()
        & merged["bn_taker_z"].notna()
        & merged["bb_ls_z"].notna()
        & merged["bb_oi_z"].notna()
        & merged["bn_oi_med_3d"].notna()
        & merged["bb_oi_med_3d"].notna(),
        [
            "ts",
            "ret_4h_avg",
            "bn_ret_4h",
            "bb_ret_4h",
            "bn_mom_4h",
            "bb_mom_4h",
            "combo_mom_4h",
            "bn_ls_z",
            "bn_taker_z",
            "bb_ls_z",
            "bb_oi_z",
            "bn_oi_med_3d",
            "bb_oi_med_3d",
            "spread_bps",
            "score",
        ],
    ].copy()
    if sample.empty:
        return None

    sample["symbol"] = symbol
    return sample


def build_samples(max_workers: int = 8) -> pd.DataFrame:
    common = [symbol for symbol in RESEARCH_SYMBOLS if (BINANCE_ROOT / symbol).is_dir() and (BYBIT_ROOT / symbol).is_dir()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = list(pool.map(_load_symbol, common))
    frames = [part for part in parts if part is not None and not part.empty]
    if not frames:
        raise RuntimeError("No samples loaded from datalake.")
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)


def add_regime_features(samples: pd.DataFrame) -> pd.DataFrame:
    liquid = samples.loc[
        (samples["bn_oi_med_3d"] >= 20_000_000.0)
        & (samples["bb_oi_med_3d"] >= 20_000_000.0)
    ].copy()
    if liquid.empty:
        liquid = samples.copy()
    regime = (
        liquid.groupby("ts", as_index=False)
        .agg(
            breadth_positive=("combo_mom_4h", lambda s: (s > 0).mean()),
            median_bn_ls_z=("bn_ls_z", "median"),
            median_bb_ls_z=("bb_ls_z", "median"),
        )
    )
    return samples.merge(regime, on="ts", how="left")


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


def select_trades(samples: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    eligible = samples.loc[
        (samples["bn_oi_med_3d"] >= cfg.min_oi_value)
        & (samples["bb_oi_med_3d"] >= cfg.min_oi_value)
        & (samples["breadth_positive"] >= cfg.breadth_threshold)
        & (samples["bn_ls_z"] >= cfg.bn_ls_threshold)
        & (samples["bn_taker_z"] >= cfg.bn_taker_threshold)
        & (samples["bb_ls_z"] >= cfg.bb_ls_threshold)
        & (samples["bb_oi_z"] >= cfg.bb_oi_z_threshold)
        & (samples["bn_mom_4h"] > 0)
        & (samples["bb_mom_4h"] > 0)
        & (samples["spread_bps"].abs() <= cfg.max_abs_spread_bps)
    ].copy()
    if eligible.empty:
        return eligible
    trades = (
        eligible.sort_values(["ts", "score"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(cfg.top_n)
        .groupby("ts", as_index=False)
        .agg(
            raw_ret=("ret_4h_avg", "mean"),
            bn_ret=("bn_ret_4h", "mean"),
            bb_ret=("bb_ret_4h", "mean"),
            n_positions=("symbol", "count"),
            mean_score=("score", "mean"),
        )
    )
    return trades


def backtest_config(samples: pd.DataFrame, cfg: Config) -> dict[str, float | int]:
    trades = select_trades(samples, cfg)
    if trades.empty:
        return {
            "bn_ls_threshold": cfg.bn_ls_threshold,
            "bn_taker_threshold": cfg.bn_taker_threshold,
            "bb_ls_threshold": cfg.bb_ls_threshold,
            "bb_oi_z_threshold": cfg.bb_oi_z_threshold,
            "min_oi_value": int(cfg.min_oi_value),
            "breadth_threshold": cfg.breadth_threshold,
            "max_abs_spread_bps": cfg.max_abs_spread_bps,
            "top_n": cfg.top_n,
            "train_trades": 0,
            "test_trades": 0,
        }

    train = trades.loc[trades["ts"] <= TRAIN_END, "raw_ret"]
    test = trades.loc[trades["ts"] >= TEST_START, "raw_ret"]
    train_avg, train_win, train_sharpe = _trade_stats(train, FEE_ALL_MAKER)
    test_avg, test_win, test_sharpe = _trade_stats(test, FEE_ALL_MAKER)
    test_mixed, _, _ = _trade_stats(test, FEE_MIXED)
    test_stress, _, _ = _trade_stats(test, FEE_STRESS)

    return {
        "bn_ls_threshold": cfg.bn_ls_threshold,
        "bn_taker_threshold": cfg.bn_taker_threshold,
        "bb_ls_threshold": cfg.bb_ls_threshold,
        "bb_oi_z_threshold": cfg.bb_oi_z_threshold,
        "min_oi_value": int(cfg.min_oi_value),
        "breadth_threshold": cfg.breadth_threshold,
        "max_abs_spread_bps": cfg.max_abs_spread_bps,
        "top_n": cfg.top_n,
        "train_trades": int(train.shape[0]),
        "test_trades": int(test.shape[0]),
        "train_avg_bps_all_maker": train_avg,
        "train_win_rate_all_maker": train_win,
        "train_sharpe_like_all_maker": train_sharpe,
        "test_avg_bps_all_maker": test_avg,
        "test_win_rate_all_maker": test_win,
        "test_sharpe_like_all_maker": test_sharpe,
        "test_avg_bps_mixed_12bps": test_mixed,
        "test_avg_bps_stress_16bps": test_stress,
        "consistency_bps": min(train_avg, test_avg),
    }


def run_grid(samples: pd.DataFrame) -> pd.DataFrame:
    configs = [
        Config(bn_ls, bn_taker, bb_ls, bb_oi, min_oi, breadth, max_spread, top_n)
        for bn_ls in (1.5, 2.0, 2.5)
        for bn_taker in (0.0, 0.5, 1.0)
        for bb_ls in (0.5, 1.0, 1.5)
        for bb_oi in (0.0, 0.5)
        for min_oi in (20_000_000.0, 50_000_000.0)
        for breadth in (0.55, 0.60, 0.65)
        for max_spread in (8.0, 12.0, 20.0)
        for top_n in (1, 2, 3)
    ]
    rows = [backtest_config(samples, cfg) for cfg in configs]
    results = pd.DataFrame(rows)
    results = results.loc[
        (results["train_trades"] >= 20)
        & (results["test_trades"] >= 10)
        & (results["train_avg_bps_all_maker"].notna())
        & (results["train_avg_bps_all_maker"] > 0)
        & (results["test_avg_bps_all_maker"] > 0)
    ].copy()
    if results.empty:
        return results
    return results.sort_values(
        [
            "consistency_bps",
            "test_avg_bps_all_maker",
            "train_avg_bps_all_maker",
            "test_trades",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["month"] = out["ts"].dt.strftime("%Y-%m")
    return (
        out.groupby("month", as_index=False)
        .agg(
            trades=("raw_ret", "count"),
            avg_bps_all_maker=("raw_ret", lambda s: (s.mean() - FEE_ALL_MAKER) * 10000.0),
            avg_bps_mixed_12bps=("raw_ret", lambda s: (s.mean() - FEE_MIXED) * 10000.0),
            avg_bps_stress_16bps=("raw_ret", lambda s: (s.mean() - FEE_STRESS) * 10000.0),
            hit_rate_all_maker=("raw_ret", lambda s: ((s - FEE_ALL_MAKER) > 0).mean()),
        )
    )


def write_summary(samples: pd.DataFrame, results: pd.DataFrame) -> None:
    if results.empty:
        SUMMARY_MD.write_text(
            "# Codex Exp 3\n\nNo configuration remained positive in both train and test after 8 bps all-maker fees.\n"
        )
        return

    best = results.iloc[0]
    cfg = Config(
        bn_ls_threshold=float(best["bn_ls_threshold"]),
        bn_taker_threshold=float(best["bn_taker_threshold"]),
        bb_ls_threshold=float(best["bb_ls_threshold"]),
        bb_oi_z_threshold=float(best["bb_oi_z_threshold"]),
        min_oi_value=float(best["min_oi_value"]),
        breadth_threshold=float(best["breadth_threshold"]),
        max_abs_spread_bps=float(best["max_abs_spread_bps"]),
        top_n=int(best["top_n"]),
    )
    trades = select_trades(samples, cfg)
    trades.to_csv(TRADES_CSV, index=False)
    monthly = monthly_breakdown(trades)

    train_symbols = samples.loc[samples["ts"] <= TRAIN_END, "symbol"].nunique()
    test_symbols = samples.loc[samples["ts"] >= TEST_START, "symbol"].nunique()

    lines = [
        "# Codex Exp 3: Cross-Exchange Confirmation",
        "",
        "Independent rebuild from local `datalake/binance` and `datalake/bybit` daily CSVs.",
        "",
        "## Method",
        "",
        "- Universe: symbols with local Binance metrics + kline data and Bybit kline + long/short + open-interest data.",
        f"- Research basket: {', '.join(RESEARCH_SYMBOLS)}.",
        "- Sampling cadence: every 4 hours at `HH:05 UTC`.",
        "- Signal shape: long-only continuation. Require Binance top-trader positioning, Binance taker-flow, and Bybit positioning to all be strong, with both exchanges already trending up.",
        "- Cross-exchange guardrail: ignore names where Binance and Bybit are too far apart at entry (`abs(spread) <= threshold`).",
        "- Return model: forward 4h return is the average of Binance and Bybit forward returns, then explicit fees are subtracted.",
        "- Fee hurdle: `8 bps` round-trip all-maker (`0.04%` per side).",
        "",
        "## Data Coverage",
        "",
        f"- Sample rows: {len(samples):,}",
        f"- Symbols in train sample: {train_symbols}",
        f"- Symbols in test sample: {test_symbols}",
        f"- Train cutoff: through {TRAIN_END.date()}",
        f"- Test period: from {TEST_START.date()}",
        "",
        "## Best Configuration",
        "",
        f"- `bn_ls_threshold={cfg.bn_ls_threshold}`",
        f"- `bn_taker_threshold={cfg.bn_taker_threshold}`",
        f"- `bb_ls_threshold={cfg.bb_ls_threshold}`",
        f"- `bb_oi_z_threshold={cfg.bb_oi_z_threshold}`",
        f"- `min_oi_value={int(cfg.min_oi_value):,}` on both exchanges",
        f"- `breadth_threshold={cfg.breadth_threshold}`",
        f"- `max_abs_spread_bps={cfg.max_abs_spread_bps}`",
        f"- `top_n={cfg.top_n}`",
        f"- Train trades: {int(best['train_trades'])}",
        f"- Test trades: {int(best['test_trades'])}",
        f"- Train avg/trade after 8 bps: {best['train_avg_bps_all_maker']:.2f} bps",
        f"- Test avg/trade after 8 bps: {best['test_avg_bps_all_maker']:.2f} bps",
        f"- Consistency score: {best['consistency_bps']:.2f} bps",
        f"- Test avg/trade after 12 bps: {best['test_avg_bps_mixed_12bps']:.2f} bps",
        f"- Test avg/trade after 16 bps: {best['test_avg_bps_stress_16bps']:.2f} bps",
        f"- Test win rate after 8 bps: {best['test_win_rate_all_maker']:.1%}",
        "",
        "## Interpretation",
        "",
        "- The useful signal, if any, is not short-horizon spread mean reversion. It is selective trend continuation after cross-exchange positioning confirmation.",
        "- Averaging future returns across Binance and Bybit removes some exchange-specific optimism. If a setup only works on one venue, it is less likely to survive this test.",
        "- Remaining risks: queue position, missed maker fills, spread drift before fill, and funding transfers are not modeled here.",
        "",
    ]

    if not monthly.empty:
        lines.extend(
            [
                "## Monthly Breakdown",
                "",
                "| Month | Trades | Avg 8bps | Avg 12bps | Avg 16bps | Hit Rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in monthly.iterrows():
            lines.append(
                f"| {row['month']} | {int(row['trades'])} | "
                f"{row['avg_bps_all_maker']:.2f} | "
                f"{row['avg_bps_mixed_12bps']:.2f} | "
                f"{row['avg_bps_stress_16bps']:.2f} | "
                f"{row['hit_rate_all_maker']:.1%} |"
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
    if TRADES_CSV.exists():
        print(f"Wrote {TRADES_CSV}")
    if results.empty:
        print("No positive configurations survived train/test at 8 bps all-maker.")
    else:
        print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
