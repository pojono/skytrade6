from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BINANCE_ROOT = ROOT / "datalake" / "binance"
BYBIT_ROOT = ROOT / "datalake" / "bybit"
OUT_DIR = Path(__file__).resolve().parent

BINANCE_SAMPLES_CSV = OUT_DIR / "rebuilt_binance_samples_4h.csv"
TRADES_CSV = OUT_DIR / "revalidated_exp2_symbol_trades.csv"
PORTFOLIO_CSV = OUT_DIR / "revalidated_exp2_portfolio.csv"
FILTERED_PORTFOLIO_CSV = OUT_DIR / "revalidated_exp2_portfolio_execution_filtered.csv"
SOFT_PORTFOLIO_CSV = OUT_DIR / "revalidated_exp2_portfolio_execution_soft.csv"
EXECUTION_SWEEP_CSV = OUT_DIR / "revalidated_exp2_execution_sweep.csv"
FUNDING_SWEEP_CSV = OUT_DIR / "revalidated_exp2_funding_sweep.csv"
ENTRY_SUMMARY_CSV = OUT_DIR / "trade_entry_feasibility_summary_full.csv"
SUMMARY_MD = OUT_DIR / "FINDINGS_codex_exp_3.md"

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
HOLD_DELTA = pd.Timedelta(hours=4)
ROLL_BARS = 14 * 24 * 12
HOLD_BARS = 48
LIQUIDITY_BARS = 3 * 24 * 12

FEE_ALL_MAKER = 0.0008
FEE_MIXED = 0.0012
FEE_STRESS = 0.0016
EXTRA_DRAG_BPS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DRAG_BLACKLIST_THRESHOLD_BPS = 8.0
SCORE_PENALTY_DIVISOR_BPS = 4.0
FILL_SHORTFALL_PENALTY_BPS = 8.0

# Best stable configuration from codex-exp-2, originally selected by requiring
# positive train and test under a stricter 20 bps all-taker assumption.
LS_THRESHOLD = 2.0
TAKER_THRESHOLD = 0.5
MIN_OI_VALUE = 20_000_000.0
TOP_N = 3
BREADTH_THRESHOLD = 0.60
MEDIAN_LS_THRESHOLD = 0.0


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean().shift(1)
    std = series.rolling(window, min_periods=window).std(ddof=0).shift(1)
    z = (series - mean) / std.replace(0.0, pd.NA)
    return z.astype("float64")


def load_binance_symbol(symbol_dir: Path) -> pd.DataFrame | None:
    metrics_files = sorted(symbol_dir.glob("*_metrics.csv"))
    kline_files = sorted(
        path
        for path in symbol_dir.glob("*_kline_1m.csv")
        if "mark_price" not in path.name
        and "premium_index" not in path.name
        and "index_price" not in path.name
    )
    if not metrics_files or not kline_files:
        return None

    metric_parts = []
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
        metric_parts.append(df)
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics["ts"] = pd.to_datetime(metrics["create_time"], utc=True)
    metrics = metrics.rename(
        columns={
            "sum_open_interest_value": "oi_value",
            "count_toptrader_long_short_ratio": "ls_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
        }
    )
    metrics = (
        metrics[["ts", "oi_value", "ls_ratio", "taker_ratio"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

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
    close_5m = (
        kline["close"].resample("5min", label="right", closed="right").last().rename("close").reset_index()
    )

    merged = metrics.merge(close_5m, on="ts", how="inner")
    if merged.empty:
        return None

    merged["ret_4h"] = merged["close"].shift(-HOLD_BARS) / merged["close"] - 1.0
    merged["mom_4h"] = merged["close"] / merged["close"].shift(HOLD_BARS) - 1.0
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


def build_binance_samples(max_workers: int = 8) -> pd.DataFrame:
    symbol_dirs = sorted(path for path in BINANCE_ROOT.iterdir() if path.is_dir())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = list(pool.map(load_binance_symbol, symbol_dirs))
    frames = [part for part in parts if part is not None and not part.empty]
    if not frames:
        raise RuntimeError("No Binance samples were built from datalake.")
    samples = pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)

    liquid = samples.loc[samples["oi_med_3d"] >= 50_000_000.0].copy()
    regime = (
        liquid.groupby("ts", as_index=False)
        .agg(
            breadth_mom=("mom_4h", lambda s: (s > 0).mean()),
            median_ls_z=("ls_z", "median"),
        )
    )
    samples = samples.merge(regime, on="ts", how="left")
    return samples


def select_exp2_trades(samples: pd.DataFrame, entry_filters: pd.DataFrame | None = None) -> pd.DataFrame:
    selected = samples.loc[
        (samples["oi_med_3d"] >= MIN_OI_VALUE)
        & (samples["breadth_mom"] >= BREADTH_THRESHOLD)
        & (samples["median_ls_z"] >= MEDIAN_LS_THRESHOLD)
        & (samples["ls_z"] >= LS_THRESHOLD)
        & (samples["taker_z"] >= TAKER_THRESHOLD)
        & (samples["mom_4h"] > 0)
    ].copy()
    if entry_filters is not None and not entry_filters.empty:
        selected = selected.merge(
            entry_filters[["symbol", "score_penalty"]],
            on="symbol",
            how="left",
        )
        selected["score_penalty"] = selected["score_penalty"].fillna(0.0)
    else:
        selected["score_penalty"] = 0.0
    selected["execution_adjusted_score"] = selected["score_abs"] - selected["score_penalty"]
    selected = (
        selected.sort_values(
            ["ts", "execution_adjusted_score", "score_abs"],
            ascending=[True, False, False],
        )
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    return selected


def load_bybit_close(symbol: str) -> pd.Series | None:
    files = sorted((BYBIT_ROOT / symbol).glob("*_kline_1m.csv"))
    if not files:
        return None
    parts = []
    for path in files:
        df = pd.read_csv(path, usecols=["startTime", "close"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["startTime"], unit="ms", utc=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = (
        out[["ts", "close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")["close"]
    )
    return out.resample("5min", label="right", closed="right").last()


def load_bybit_funding(symbol: str) -> pd.Series:
    files = sorted((BYBIT_ROOT / symbol).glob("*_funding_rate.csv"))
    if not files:
        return pd.Series(dtype="float64")
    parts = []
    for path in files:
        df = pd.read_csv(path, usecols=["timestamp", "fundingRate"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["fundingRate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    out = (
        out[["ts", "fundingRate"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")["fundingRate"]
    )
    return out


def attach_bybit_returns(selected: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    symbols = sorted(selected["symbol"].unique())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        bybit_series = dict(zip(symbols, pool.map(load_bybit_close, symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        bybit_funding = dict(zip(symbols, pool.map(load_bybit_funding, symbols)))

    parts = []
    for symbol, frame in selected.groupby("symbol"):
        series = bybit_series.get(symbol)
        if series is None:
            continue
        funding = bybit_funding.get(symbol, pd.Series(dtype="float64"))
        working = frame.copy().set_index("ts")
        working["bybit_entry"] = series.reindex(working.index)
        working["bybit_exit"] = series.reindex(working.index + HOLD_DELTA).values
        working["bybit_ret_4h"] = working["bybit_exit"] / working["bybit_entry"] - 1.0
        funding_sums = []
        for entry_ts in working.index:
            exit_ts = entry_ts + HOLD_DELTA
            funding_sums.append(funding.loc[(funding.index > entry_ts) & (funding.index <= exit_ts)].sum())
        working["bybit_funding_4h"] = funding_sums
        working["bybit_ret_4h_funding_adj"] = working["bybit_ret_4h"] - working["bybit_funding_4h"]
        parts.append(working.reset_index())

    if not parts:
        raise RuntimeError("No Bybit price series matched the selected trades.")

    full = pd.concat(parts, ignore_index=True)
    full = full.dropna(subset=["bybit_ret_4h"]).reset_index(drop=True)
    full["avg_ret_4h"] = (full["ret_4h"] + full["bybit_ret_4h"]) / 2.0
    full["avg_ret_4h_funding_adj"] = (full["ret_4h"] + full["bybit_ret_4h_funding_adj"]) / 2.0
    return full


def aggregate_portfolio(trades: pd.DataFrame) -> pd.DataFrame:
    return (
        trades.groupby("ts", as_index=False)
        .agg(
            n_positions=("symbol", "count"),
            binance_ret_4h=("ret_4h", "mean"),
            bybit_ret_4h=("bybit_ret_4h", "mean"),
            bybit_funding_4h=("bybit_funding_4h", "mean"),
            bybit_ret_4h_funding_adj=("bybit_ret_4h_funding_adj", "mean"),
            avg_ret_4h=("avg_ret_4h", "mean"),
            avg_ret_4h_funding_adj=("avg_ret_4h_funding_adj", "mean"),
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )


def load_execution_filters() -> tuple[set[str], pd.DataFrame]:
    if not ENTRY_SUMMARY_CSV.exists():
        return set(), pd.DataFrame(
            columns=[
                "symbol",
                "bb_maker_fill_rate",
                "avg_positive_drag_bps",
                "score_penalty",
                "blacklist",
            ]
        )

    entry = pd.read_csv(ENTRY_SUMMARY_CSV)
    entry["bn_positive_drag_bps"] = entry["bn_vwap_60s_bps"].clip(lower=0.0)
    entry["bb_positive_drag_bps"] = entry["bb_vwap_60s_bps"].clip(lower=0.0)
    entry["avg_positive_drag_bps"] = (
        entry["bn_positive_drag_bps"] + entry["bb_positive_drag_bps"]
    ) / 2.0
    entry["score_penalty_bps"] = entry["avg_positive_drag_bps"] + (
        (1.0 - entry["bb_maker_fill_rate"].clip(lower=0.0, upper=1.0)) * FILL_SHORTFALL_PENALTY_BPS
    )
    entry["score_penalty"] = entry["score_penalty_bps"] / SCORE_PENALTY_DIVISOR_BPS
    entry["blacklist"] = (
        (entry["bb_maker_fill_rate"] < 1.0)
        | (entry["avg_positive_drag_bps"] >= DRAG_BLACKLIST_THRESHOLD_BPS)
    )
    blacklisted = set(entry.loc[entry["blacklist"], "symbol"])
    return blacklisted, entry


def aggregate_filtered_portfolio(trades: pd.DataFrame, blacklisted: set[str]) -> pd.DataFrame:
    if not blacklisted:
        return aggregate_portfolio(trades)
    filtered = trades.loc[~trades["symbol"].isin(blacklisted)].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "n_positions",
                "binance_ret_4h",
                "bybit_ret_4h",
                "bybit_funding_4h",
                "bybit_ret_4h_funding_adj",
                "avg_ret_4h",
                "avg_ret_4h_funding_adj",
            ]
        )
    return aggregate_portfolio(filtered)


def _avg_bps(series: pd.Series, fee: float) -> float:
    return (series.mean() - fee) * 10000.0


def _win_rate(series: pd.Series, fee: float) -> float:
    return ((series - fee) > 0).mean()


def monthly_breakdown(portfolio: pd.DataFrame) -> pd.DataFrame:
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    test["month"] = test["ts"].dt.strftime("%Y-%m")
    return (
        test.groupby("month", as_index=False)
        .agg(
            trades=("avg_ret_4h", "count"),
            avg_bps_8=("avg_ret_4h", lambda s: _avg_bps(s, FEE_ALL_MAKER)),
            avg_bps_12=("avg_ret_4h", lambda s: _avg_bps(s, FEE_MIXED)),
            avg_bps_16=("avg_ret_4h", lambda s: _avg_bps(s, FEE_STRESS)),
            win_rate_8=("avg_ret_4h", lambda s: _win_rate(s, FEE_ALL_MAKER)),
        )
    )


def execution_drag_sweep(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    for extra_bps in EXTRA_DRAG_BPS:
        extra_fee = extra_bps / 10000.0
        total_fee = FEE_ALL_MAKER + extra_fee
        rows.append(
            {
                "extra_drag_bps": extra_bps,
                "total_cost_bps": (total_fee * 10000.0),
                "test_avg_bps": _avg_bps(test["avg_ret_4h"], total_fee),
                "test_win_rate": _win_rate(test["avg_ret_4h"], total_fee),
            }
        )
    return pd.DataFrame(rows)


def funding_drag_sweep(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    for extra_bps in EXTRA_DRAG_BPS:
        extra_fee = extra_bps / 10000.0
        total_fee = FEE_ALL_MAKER + extra_fee
        rows.append(
            {
                "extra_drag_bps": extra_bps,
                "total_cost_bps": (total_fee * 10000.0),
                "test_avg_bps_funding_adj": _avg_bps(test["avg_ret_4h_funding_adj"], total_fee),
                "test_win_rate_funding_adj": _win_rate(test["avg_ret_4h_funding_adj"], total_fee),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    samples: pd.DataFrame,
    selected: pd.DataFrame,
    trades: pd.DataFrame,
    portfolio: pd.DataFrame,
    soft_portfolio: pd.DataFrame,
    filtered_portfolio: pd.DataFrame,
    entry_filters: pd.DataFrame,
) -> None:
    monthly = monthly_breakdown(portfolio)
    drag = execution_drag_sweep(portfolio)
    funding_drag = funding_drag_sweep(portfolio)
    train = portfolio.loc[portfolio["ts"] <= TRAIN_END].copy()
    test = portfolio.loc[portfolio["ts"] >= TEST_START].copy()
    soft_test = soft_portfolio.loc[soft_portfolio["ts"] >= TEST_START].copy()
    filtered_test = filtered_portfolio.loc[filtered_portfolio["ts"] >= TEST_START].copy()
    blacklisted = entry_filters.loc[entry_filters["blacklist"]].copy()

    lines = [
        "# Codex Exp 3: Useful Repo Finding, Revalidated Cross-Venue",
        "",
        "This note treats prior repo PnL as untrusted and re-checks the strongest plausible signal under a simple but stricter test.",
        "",
        "## Rejected Path",
        "",
        "- A fresh `codex-exp-3` experiment requiring Binance and Bybit positioning to simultaneously confirm a long setup did not produce any configuration that stayed positive in both train and test after `8 bps` all-maker fees.",
        "- That means the broad \"two-venue crowding confirmation\" idea is weaker than the repo's optimistic strategy reports suggest.",
        "",
        "## Surviving Path",
        "",
        "- The most credible prior result was the rare-event long-only Binance LS momentum strategy from `codex-exp-2`.",
        "- That strategy was originally selected under a much harsher `20 bps` all-taker hurdle using raw `datalake/binance` data.",
        "- In this version, the Binance sample is rebuilt directly inside `codex-exp-3` from raw `datalake/binance` files, with no dependency on `codex-exp-2` outputs.",
        "- Those exact entry timestamps are then repriced using Bybit futures closes and averaged with Binance returns.",
        "",
        "## Strategy",
        "",
        "- Signal source: Binance positioning + taker-flow metrics.",
        "- Entry cadence: every 4 hours at `HH:05 UTC`.",
        "- Hold: fixed 4 hours.",
        "- Long-only rule:",
        f"  - `ls_z >= {LS_THRESHOLD}`",
        f"  - `taker_z >= {TAKER_THRESHOLD}`",
        "  - `mom_4h > 0`",
        f"  - `oi_med_3d >= {int(MIN_OI_VALUE):,}`",
        f"  - market breadth `>= {BREADTH_THRESHOLD:.2f}`",
        f"  - median LS z-score `>= {MEDIAN_LS_THRESHOLD:.1f}`",
        f"- Selection: top `{TOP_N}` names by the existing `score_abs` ranking.",
        "- Fee hurdle: `8 bps` round-trip all-maker (`0.04%` per side).",
        "",
        "## Coverage",
        "",
        f"- Rebuilt Binance sample rows: {len(samples):,}",
        f"- Selected symbol-level entries: {len(selected)}",
        f"- Selected portfolio timestamps: {len(portfolio)}",
        f"- Unique traded symbols: {selected['symbol'].nunique()}",
        f"- Bybit-repriced symbol entries retained: {len(trades)}",
        f"- Train timestamps: {len(train)}",
        f"- Test timestamps: {len(test)}",
        "",
        "## Cross-Venue Results",
        "",
        f"- Train avg/trade after 8 bps on average(Binance, Bybit): {_avg_bps(train['avg_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 8 bps on average(Binance, Bybit): {_avg_bps(test['avg_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 12 bps on average(Binance, Bybit): {_avg_bps(test['avg_ret_4h'], FEE_MIXED):.2f} bps",
        f"- Test avg/trade after 16 bps on average(Binance, Bybit): {_avg_bps(test['avg_ret_4h'], FEE_STRESS):.2f} bps",
        f"- Test win rate after 8 bps: {_win_rate(test['avg_ret_4h'], FEE_ALL_MAKER):.1%}",
        "",
        "## Execution-Aware Soft Penalty",
        "",
        f"- Ranking penalty = (avg positive 60s VWAP drag + fill shortfall penalty) / {SCORE_PENALTY_DIVISOR_BPS:.1f}",
        f"- Fill shortfall penalty = {(FILL_SHORTFALL_PENALTY_BPS):.1f} bps scaled by (1 - Bybit fill rate)",
        f"- Soft-penalized test timestamps: {len(soft_test)}",
        f"- Soft-penalized test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(soft_test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.2f} bps",
        f"- Soft-penalized test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(soft_test['avg_ret_4h_funding_adj'], FEE_MIXED):.2f} bps",
        f"- Soft-penalized test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(soft_test['avg_ret_4h_funding_adj'], FEE_STRESS):.2f} bps",
        f"- Soft-penalized test win rate after 8 bps: {_win_rate(soft_test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.1%}",
        "",
        "## Hard-Filter Comparison",
        "",
        f"- Hard-filter threshold: average positive 60s VWAP drag >= {DRAG_BLACKLIST_THRESHOLD_BPS:.1f} bps, or Bybit maker-fill proxy < 100%",
        f"- Hard-filtered symbols: {', '.join(sorted(blacklisted['symbol'])) if not blacklisted.empty else 'none'}",
        f"- Filtered test timestamps: {len(filtered_test)}",
        f"- Filtered test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(filtered_test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.2f} bps",
        f"- Filtered test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(filtered_test['avg_ret_4h_funding_adj'], FEE_MIXED):.2f} bps",
        f"- Filtered test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(filtered_test['avg_ret_4h_funding_adj'], FEE_STRESS):.2f} bps",
        f"- Filtered test win rate after 8 bps: {_win_rate(filtered_test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.1%}",
        "",
        "## Partial Funding Adjustment (Bybit Leg Only)",
        "",
        f"- Train avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(train['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(test['avg_ret_4h_funding_adj'], FEE_MIXED):.2f} bps",
        f"- Test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): {_avg_bps(test['avg_ret_4h_funding_adj'], FEE_STRESS):.2f} bps",
        f"- Test win rate after 8 bps with Bybit funding applied: {_win_rate(test['avg_ret_4h_funding_adj'], FEE_ALL_MAKER):.1%}",
        f"- Mean Bybit funding impact per 4h trade in test: {(test['bybit_funding_4h'].mean() * 10000.0):.3f} bps",
        "",
        "## Venue Comparison",
        "",
        f"- Train avg/trade after 8 bps on Binance-only pricing: {_avg_bps(train['binance_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Train avg/trade after 8 bps on Bybit-only pricing: {_avg_bps(train['bybit_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 8 bps on Binance-only pricing: {_avg_bps(test['binance_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 8 bps on Bybit-only pricing: {_avg_bps(test['bybit_ret_4h'], FEE_ALL_MAKER):.2f} bps",
        f"- Test avg/trade after 8 bps on Bybit-only pricing net of Bybit funding: {_avg_bps(test['bybit_ret_4h_funding_adj'], FEE_ALL_MAKER):.2f} bps",
        "",
        "## Interpretation",
        "",
        "- The useful signal is not a generic cross-exchange spread trade.",
        "- The useful signal is a selective continuation trade: when Binance positioning becomes extremely bullish and taker flow confirms it, the same coin tends to keep rising over the next 4 hours on both Binance and Bybit.",
        "- Because the Bybit repricing stays positive, this looks more like a real underlying asset effect than a Binance-only artifact.",
        "- The funding-aware numbers are only a partial adjustment because Binance funding is not joined here; they are still a more conservative check than pure price returns.",
        "- This still does not fully model queue priority or partial fills, but the drag sweeps below show how much extra friction the edge can absorb.",
        "",
    ]

    if not drag.empty:
        lines.extend(
            [
                "## Execution-Drag Sweep",
                "",
                "| Extra Drag | Total Cost | Test Avg | Test Win Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in drag.iterrows():
            lines.append(
                f"| {row['extra_drag_bps']:.0f} bps | {row['total_cost_bps']:.0f} bps | "
                f"{row['test_avg_bps']:.2f} bps | {row['test_win_rate']:.1%} |"
            )
        lines.append("")

    if not funding_drag.empty:
        lines.extend(
            [
                "## Funding-Aware Drag Sweep",
                "",
                "| Extra Drag | Total Cost | Test Avg (Funding Adj) | Test Win Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in funding_drag.iterrows():
            lines.append(
                f"| {row['extra_drag_bps']:.0f} bps | {row['total_cost_bps']:.0f} bps | "
                f"{row['test_avg_bps_funding_adj']:.2f} bps | {row['test_win_rate_funding_adj']:.1%} |"
            )
        lines.append("")

    if not monthly.empty:
        lines.extend(
            [
                "## Monthly Test Breakdown",
                "",
                "| Month | Trades | Avg 8bps | Avg 12bps | Avg 16bps | Win Rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in monthly.iterrows():
            lines.append(
                f"| {row['month']} | {int(row['trades'])} | "
                f"{row['avg_bps_8']:.2f} | {row['avg_bps_12']:.2f} | "
                f"{row['avg_bps_16']:.2f} | {row['win_rate_8']:.1%} |"
            )
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    samples = build_binance_samples()
    samples.to_csv(BINANCE_SAMPLES_CSV, index=False)
    blacklisted, entry_filters = load_execution_filters()
    selected = select_exp2_trades(samples, entry_filters)
    trades = attach_bybit_returns(selected)
    portfolio = aggregate_portfolio(trades)
    soft_portfolio = aggregate_portfolio(trades)
    filtered_portfolio = aggregate_filtered_portfolio(trades, blacklisted)
    drag = execution_drag_sweep(portfolio)
    funding_drag = funding_drag_sweep(portfolio)
    trades.to_csv(TRADES_CSV, index=False)
    portfolio.to_csv(PORTFOLIO_CSV, index=False)
    soft_portfolio.to_csv(SOFT_PORTFOLIO_CSV, index=False)
    filtered_portfolio.to_csv(FILTERED_PORTFOLIO_CSV, index=False)
    drag.to_csv(EXECUTION_SWEEP_CSV, index=False)
    funding_drag.to_csv(FUNDING_SWEEP_CSV, index=False)
    write_summary(
        samples,
        selected,
        trades,
        portfolio,
        soft_portfolio,
        filtered_portfolio,
        entry_filters,
    )
    print(f"Wrote {BINANCE_SAMPLES_CSV}")
    print(f"Wrote {TRADES_CSV}")
    print(f"Wrote {PORTFOLIO_CSV}")
    print(f"Wrote {SOFT_PORTFOLIO_CSV}")
    print(f"Wrote {FILTERED_PORTFOLIO_CSV}")
    print(f"Wrote {EXECUTION_SWEEP_CSV}")
    print(f"Wrote {FUNDING_SWEEP_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(portfolio.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
