from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SIGNALS_CSV = OUT_DIR / "delayed_confirmation_features.csv"
OUT_FEATURES_CSV = OUT_DIR / "orderbook_walkforward_30s_60m_features.csv"
OUT_RESULTS_CSV = OUT_DIR / "orderbook_walkforward_30s_60m_results.csv"
REPORT_MD = OUT_DIR / "FINDINGS_orderbook_walkforward_30s_60m.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"
ORDER_NOTIONALS_USD = (10_000.0, 50_000.0, 100_000.0)
ROUND_TRIP_FEE_BPS = 20.0
MAX_DEPTH_STALENESS_S = 40.0
ENTRY_DELAY = pd.Timedelta("30s")
EXIT_HOLD = pd.Timedelta("60m")


@dataclass(frozen=True)
class GateRule:
    min_ret_30s_bps: float
    min_buy_share_30s: float


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


def _last_price(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df.iloc[-1]["price"])


def _buy_share(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.5
    quote = df["price"] * df["quantity"]
    total = float(quote.sum())
    if total <= 0:
        return 0.5
    buy_quote = float(quote.loc[~df["is_buyer_maker"]].sum())
    return buy_quote / total


def _cumulative_curve(snapshot: pd.DataFrame, side: str) -> list[tuple[int, float]]:
    zero_rows = snapshot.loc[snapshot["percentage"] == 0, "notional"].tolist()
    if len(zero_rows) < 2:
        return []
    if side == "buy":
        points = [(0, float(zero_rows[-1]))]
        for pct in range(1, 6):
            vals = snapshot.loc[snapshot["percentage"] == pct, "notional"]
            if vals.empty:
                return []
            points.append((pct, float(vals.iloc[0])))
        return points
    points = [(0, float(zero_rows[0]))]
    for pct in range(1, 6):
        vals = snapshot.loc[snapshot["percentage"] == -pct, "notional"]
        if vals.empty:
            return []
        points.append((pct, float(vals.iloc[0])))
    return points


def _avg_impact_pct(curve: list[tuple[int, float]], order_notional: float) -> float | None:
    if not curve or order_notional <= 0:
        return None
    max_notional = curve[-1][1]
    if max_notional < order_notional:
        return None

    prev_cum = 0.0
    prev_pct = 0.0
    weighted_pct_notional = 0.0
    remaining = order_notional

    for pct, cum in curve:
        band_cap = max(0.0, cum - prev_cum)
        if band_cap <= 0:
            prev_cum = cum
            prev_pct = float(pct)
            continue
        take = min(remaining, band_cap)
        used_end_pct = prev_pct + (float(pct) - prev_pct) * (take / band_cap)
        avg_band_pct = (prev_pct + used_end_pct) / 2.0
        weighted_pct_notional += take * avg_band_pct
        remaining -= take
        prev_cum = cum
        prev_pct = float(pct)
        if remaining <= 1e-9:
            break

    if remaining > 1e-9:
        return None
    return weighted_pct_notional / order_notional


def _snapshot(depth: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.DataFrame, float] | tuple[None, None]:
    snaps = depth.loc[depth["ts"] <= ts, "ts"]
    if snaps.empty:
        return None, None
    snap_ts = snaps.max()
    staleness_s = float((ts - snap_ts).total_seconds())
    if staleness_s > MAX_DEPTH_STALENESS_S:
        return None, None
    snap = depth.loc[depth["ts"] == snap_ts].copy()
    return snap, staleness_s


def build_execution_features() -> pd.DataFrame:
    signals = pd.read_csv(SIGNALS_CSV, parse_dates=["ts"])
    out: list[dict[str, object]] = []

    for _, row in signals.iterrows():
        symbol = row["symbol"]
        ts0 = row["ts"]
        day = ts0.strftime("%Y-%m-%d")
        agg = load_agg(symbol, day)
        depth = load_depth(symbol, day)
        if agg is None or depth is None:
            continue

        t_entry = ts0 + ENTRY_DELAY
        t_exit = t_entry + EXIT_HOLD

        pre_signal = agg.loc[agg["ts"] <= ts0]
        until_entry = agg.loc[(agg["ts"] > ts0) & (agg["ts"] <= t_entry)]
        pre_entry = agg.loc[agg["ts"] <= t_entry]
        pre_exit = agg.loc[agg["ts"] <= t_exit]
        if pre_signal.empty or until_entry.empty or pre_entry.empty or pre_exit.empty:
            continue

        signal_px = _last_price(pre_signal)
        ref_entry_px = _last_price(pre_entry)
        ref_exit_px = _last_price(pre_exit)
        if signal_px is None or ref_entry_px is None or ref_exit_px is None:
            continue

        entry_snap, entry_stale = _snapshot(depth, t_entry)
        exit_snap, exit_stale = _snapshot(depth, t_exit)
        if entry_snap is None or exit_snap is None:
            continue

        entry_curve = _cumulative_curve(entry_snap, "buy")
        exit_curve = _cumulative_curve(exit_snap, "sell")
        if not entry_curve or not exit_curve:
            continue

        for order_notional_usd in ORDER_NOTIONALS_USD:
            entry_impact_pct = _avg_impact_pct(entry_curve, order_notional_usd)
            exit_impact_pct = _avg_impact_pct(exit_curve, order_notional_usd)
            if entry_impact_pct is None or exit_impact_pct is None:
                continue

            fill_entry_px = ref_entry_px * (1.0 + entry_impact_pct / 100.0)
            fill_exit_px = ref_exit_px * (1.0 - exit_impact_pct / 100.0)
            net_bps = (fill_exit_px / fill_entry_px - 1.0) * 10000.0 - ROUND_TRIP_FEE_BPS

            out.append(
                {
                    "ts": ts0,
                    "symbol": symbol,
                    "source_study_period": row["study_period"],
                    "order_notional_usd": order_notional_usd,
                    "ret_30s_bps": (ref_entry_px / signal_px - 1.0) * 10000.0,
                    "buy_share_30s": _buy_share(until_entry),
                    "entry_depth_staleness_s": entry_stale,
                    "exit_depth_staleness_s": exit_stale,
                    "entry_impact_bps": entry_impact_pct * 100.0,
                    "exit_impact_bps": exit_impact_pct * 100.0,
                    "gross_ref_60m_bps": (ref_exit_px / ref_entry_px - 1.0) * 10000.0,
                    "net_exec_60m_bps_20": net_bps,
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


def apply_gate(df: pd.DataFrame, rule: GateRule) -> pd.DataFrame:
    return df.loc[
        (df["ret_30s_bps"] >= rule.min_ret_30s_bps)
        & (df["buy_share_30s"] >= rule.min_buy_share_30s)
    ].copy()


def choose_gate(train: pd.DataFrame) -> tuple[GateRule, pd.DataFrame]:
    q = train.quantile([0.5, 0.75], numeric_only=True)
    candidates = []
    for min_ret in (0.0, float(q.loc[0.5, "ret_30s_bps"]), float(q.loc[0.75, "ret_30s_bps"])):
        for min_buy in (0.5, float(q.loc[0.5, "buy_share_30s"]), float(q.loc[0.75, "buy_share_30s"])):
            rule = GateRule(min_ret, min_buy)
            taken = apply_gate(train, rule)
            if len(taken) < 8:
                continue
            candidates.append((float(taken["net_exec_60m_bps_20"].mean()), len(taken), rule, taken))
    if not candidates:
        rule = GateRule(0.0, 0.5)
        taken = apply_gate(train, rule)
        return rule, taken
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, rule, taken = candidates[0]
    return rule, taken


def main() -> None:
    features = build_execution_features()
    features.to_csv(OUT_FEATURES_CSV, index=False)

    result_rows = []
    details: dict[float, pd.DataFrame] = {}
    for order_notional_usd, bucket in features.groupby("order_notional_usd", sort=True):
        train = bucket.loc[bucket["study_period"] == "train"].copy()
        test = bucket.loc[bucket["study_period"] == "test"].copy()
        base_test_avg = float(test["net_exec_60m_bps_20"].mean()) if not test.empty else np.nan
        gate, train_taken = choose_gate(train)
        test_taken = apply_gate(test, gate)
        details[order_notional_usd] = test_taken
        result_rows.append(
            {
                "order_notional_usd": order_notional_usd,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "base_test_avg_bps": base_test_avg,
                "gate_min_ret_30s_bps": gate.min_ret_30s_bps,
                "gate_min_buy_share_30s": gate.min_buy_share_30s,
                "train_kept_rows": int(len(train_taken)),
                "train_kept_avg_bps": float(train_taken["net_exec_60m_bps_20"].mean()) if not train_taken.empty else np.nan,
                "test_kept_rows": int(len(test_taken)),
                "test_kept_avg_bps": float(test_taken["net_exec_60m_bps_20"].mean()) if not test_taken.empty else np.nan,
                "test_improve_bps": (float(test_taken["net_exec_60m_bps_20"].mean()) - base_test_avg) if not test_taken.empty else np.nan,
                "test_kept_symbols": ",".join(sorted(test_taken["symbol"].unique())) if not test_taken.empty else "",
                "avg_entry_impact_bps_test": float(test["entry_impact_bps"].mean()) if not test.empty else np.nan,
                "avg_exit_impact_bps_test": float(test["exit_impact_bps"].mean()) if not test.empty else np.nan,
            }
        )
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_RESULTS_CSV, index=False)

    base_lines = [
        "# Orderbook Walk-Forward: 30s Entry, 60m Exit",
        "",
        "This is a causal, execution-aware walk-forward on the covered raw signal stream.",
        "- Entry decision time: signal + 30s",
        "- Exit time: entry + 60m",
        f"- Fees: {ROUND_TRIP_FEE_BPS:.0f} bps round-trip",
        "- Entry and exit fills are simulated from actual `book_depth` snapshots using cumulative depth buckets (0% to 5%) and a piecewise-linear fill model.",
        "- Reference price uses the last observed `agg_trade` before each execution timestamp.",
        "",
    ]
    lines = base_lines + ["## Size Sweep", ""]
    for _, row in results.sort_values("order_notional_usd").iterrows():
        lines.extend(
            [
                f"### ${row['order_notional_usd']:,.0f}",
                "",
                f"- Train rows with full execution data: {int(row['train_rows'])}",
                f"- Test rows with full execution data: {int(row['test_rows'])}",
                f"- Mean test entry impact: {row['avg_entry_impact_bps_test']:.2f} bps",
                f"- Mean test exit impact: {row['avg_exit_impact_bps_test']:.2f} bps",
                f"- Unfiltered test avg after execution + fees: {row['base_test_avg_bps']:.2f} bps",
                f"- Train-chosen gate: `ret_30s_bps >= {row['gate_min_ret_30s_bps']:.2f}`, `buy_share_30s >= {row['gate_min_buy_share_30s']:.3f}`",
                f"- Filtered test avg: {row['test_kept_avg_bps']:.2f} bps on {int(row['test_kept_rows'])} rows" if pd.notna(row["test_kept_avg_bps"]) else f"- Filtered test avg: no rows on {int(row['test_kept_rows'])} rows",
                f"- Improvement vs unfiltered: {row['test_improve_bps']:.2f} bps" if pd.notna(row["test_improve_bps"]) else "- Improvement vs unfiltered: n/a",
                "",
            ]
        )

    lines.extend(
        [
            "## Bottom Line",
            "",
        ]
    )

    positive = results.loc[results["test_kept_avg_bps"] > 0].copy()
    if not positive.empty:
        best = positive.sort_values(["order_notional_usd"], ascending=[True]).iloc[0]
        lines.append(
            f"- A causal 30-second gate combined with orderbook-based fills stays positive on the covered test sample at ${best['order_notional_usd']:,.0f} notional."
        )
    else:
        lines.append("- Once fills are simulated against orderbook depth, the causal 30-second variant does not stay positive on the covered test sample.")

    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {OUT_FEATURES_CSV}")
    print(f"Wrote {OUT_RESULTS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(results.to_string(index=False))
    for order_notional_usd, test_taken in details.items():
        if not test_taken.empty:
            print(f"\n$ {order_notional_usd:,.0f} kept trades")
            cols = ["ts", "symbol", "ret_30s_bps", "buy_share_30s", "entry_impact_bps", "exit_impact_bps", "net_exec_60m_bps_20"]
            print(test_taken[cols].to_string(index=False))


if __name__ == "__main__":
    main()
