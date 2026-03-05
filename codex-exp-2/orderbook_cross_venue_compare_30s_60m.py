from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SIGNALS_CSV = OUT_DIR / "delayed_confirmation_features.csv"
OUT_FEATURES_CSV = OUT_DIR / "orderbook_cross_venue_compare_30s_60m_features.csv"
OUT_RESULTS_CSV = OUT_DIR / "orderbook_cross_venue_compare_30s_60m_results.csv"
REPORT_MD = OUT_DIR / "FINDINGS_orderbook_cross_venue_compare_30s_60m.md"

PARQUET = Path(__file__).resolve().parents[1] / "parquet"
ORDER_NOTIONALS_USD = (10_000.0, 50_000.0, 100_000.0)
ROUND_TRIP_FEE_BPS = 20.0
MAX_DEPTH_STALENESS_S = 40.0
ENTRY_DELAY = pd.Timedelta("30s")
EXIT_HOLD = pd.Timedelta("60m")

# Reuse the current best gate from the latest Binance walk-forward output.
GATE_MIN_RET_30S_BPS = 1.471836
GATE_MIN_BUY_SHARE_30S = 0.700624


@dataclass(frozen=True)
class GateRule:
    min_ret_30s_bps: float
    min_buy_share_30s: float


GATE = GateRule(GATE_MIN_RET_30S_BPS, GATE_MIN_BUY_SHARE_30S)


@lru_cache(maxsize=None)
def load_agg(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "agg_trades_futures" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


@lru_cache(maxsize=None)
def load_binance_depth(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "binance" / "book_depth" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df


@lru_cache(maxsize=None)
def load_bybit_depth(symbol: str, day: str) -> pd.DataFrame | None:
    path = PARQUET / symbol / "orderbook" / "bybit_futures" / f"{day}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "timestamp_us" not in df.columns:
        return None
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


def _snapshot_binance(depth: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.DataFrame, float] | tuple[None, None]:
    snaps = depth.loc[depth["ts"] <= ts, "ts"]
    if snaps.empty:
        return None, None
    snap_ts = snaps.max()
    staleness_s = float((ts - snap_ts).total_seconds())
    if staleness_s > MAX_DEPTH_STALENESS_S:
        return None, None
    snap = depth.loc[depth["ts"] == snap_ts].copy()
    return snap, staleness_s


def _snapshot_bybit(depth: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Series, float] | tuple[None, None]:
    snaps = depth.loc[depth["ts"] <= ts, "ts"]
    if snaps.empty:
        return None, None
    snap_ts = snaps.max()
    staleness_s = float((ts - snap_ts).total_seconds())
    if staleness_s > MAX_DEPTH_STALENESS_S:
        return None, None
    row = depth.loc[depth["ts"] == snap_ts].iloc[-1]
    return row, staleness_s


def _avg_impact_bps_from_curve(curve: list[tuple[float, float]], order_notional: float) -> float | None:
    if len(curve) < 2 or order_notional <= 0:
        return None
    max_notional = curve[-1][1]
    if max_notional < order_notional:
        return None

    prev_cum = 0.0
    prev_bps = 0.0
    weighted_bps_notional = 0.0
    remaining = order_notional

    for bps, cum in curve:
        band_cap = max(0.0, cum - prev_cum)
        if band_cap <= 0:
            prev_cum = cum
            prev_bps = float(bps)
            continue
        take = min(remaining, band_cap)
        used_end_bps = prev_bps + (float(bps) - prev_bps) * (take / band_cap)
        avg_band_bps = (prev_bps + used_end_bps) / 2.0
        weighted_bps_notional += take * avg_band_bps
        remaining -= take
        prev_cum = cum
        prev_bps = float(bps)
        if remaining <= 1e-9:
            break

    if remaining > 1e-9:
        return None
    return weighted_bps_notional / order_notional


def _binance_curve(snapshot: pd.DataFrame, side: str) -> list[tuple[float, float]]:
    zero_rows = snapshot.loc[snapshot["percentage"] == 0, "notional"].tolist()
    # Some symbols/dates have no explicit 0 bucket; start from 0 notional in that case.
    if side == "buy":
        zero_notional = float(zero_rows[-1]) if zero_rows else 0.0
        points = [(0.0, zero_notional)]
        for pct in range(1, 6):
            vals = snapshot.loc[snapshot["percentage"] == pct, "notional"]
            if vals.empty:
                return []
            # In Binance depth files these are "percentage" buckets.
            # Preserve existing project convention: pct -> pct*100 bps.
            points.append((float(pct * 100), float(vals.iloc[0])))
        return points
    zero_notional = float(zero_rows[0]) if zero_rows else 0.0
    points = [(0.0, zero_notional)]
    for pct in range(1, 6):
        vals = snapshot.loc[snapshot["percentage"] == -pct, "notional"]
        if vals.empty:
            return []
        points.append((float(pct * 100), float(vals.iloc[0])))
    return points


def _bybit_curve(snapshot: pd.Series, side: str) -> list[tuple[float, float]]:
    mid = float(snapshot.get("mid_price", np.nan))
    if not np.isfinite(mid) or mid <= 0:
        return []
    if side == "buy":
        keys = [
            ("ask_depth_0.5bps", 0.5),
            ("ask_depth_1bps", 1.0),
            ("ask_depth_2bps", 2.0),
            ("ask_depth_3bps", 3.0),
            ("ask_depth_5bps", 5.0),
        ]
    else:
        keys = [
            ("bid_depth_0.5bps", 0.5),
            ("bid_depth_1bps", 1.0),
            ("bid_depth_2bps", 2.0),
            ("bid_depth_3bps", 3.0),
            ("bid_depth_5bps", 5.0),
        ]
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    prev = 0.0
    for col, bps in keys:
        qty = float(snapshot.get(col, np.nan))
        if not np.isfinite(qty):
            return []
        notional = max(0.0, qty * mid)
        if notional < prev:
            notional = prev
        points.append((bps, notional))
        prev = notional
    return points


def _net_exec_bps(ref_entry_px: float, ref_exit_px: float, entry_impact_bps: float, exit_impact_bps: float) -> float:
    fill_entry_px = ref_entry_px * (1.0 + entry_impact_bps / 10000.0)
    fill_exit_px = ref_exit_px * (1.0 - exit_impact_bps / 10000.0)
    return (fill_exit_px / fill_entry_px - 1.0) * 10000.0 - ROUND_TRIP_FEE_BPS


def _gate(df: pd.DataFrame, gate: GateRule) -> pd.DataFrame:
    return df.loc[
        (df["ret_30s_bps"] >= gate.min_ret_30s_bps)
        & (df["buy_share_30s"] >= gate.min_buy_share_30s)
    ].copy()


def build_features() -> pd.DataFrame:
    signals = pd.read_csv(SIGNALS_CSV, parse_dates=["ts"])
    out: list[dict[str, object]] = []

    for _, row in signals.iterrows():
        symbol = row["symbol"]
        ts0 = row["ts"]
        day = ts0.strftime("%Y-%m-%d")

        agg = load_agg(symbol, day)
        binance_depth = load_binance_depth(symbol, day)
        bybit_depth = load_bybit_depth(symbol, day)
        if agg is None or binance_depth is None or bybit_depth is None:
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

        bin_entry_snap, bin_entry_stale = _snapshot_binance(binance_depth, t_entry)
        bin_exit_snap, bin_exit_stale = _snapshot_binance(binance_depth, t_exit)
        by_entry_snap, by_entry_stale = _snapshot_bybit(bybit_depth, t_entry)
        by_exit_snap, by_exit_stale = _snapshot_bybit(bybit_depth, t_exit)
        if (
            bin_entry_snap is None
            or bin_exit_snap is None
            or by_entry_snap is None
            or by_exit_snap is None
        ):
            continue

        bin_buy_curve = _binance_curve(bin_entry_snap, "buy")
        bin_sell_curve = _binance_curve(bin_exit_snap, "sell")
        by_buy_curve = _bybit_curve(by_entry_snap, "buy")
        by_sell_curve = _bybit_curve(by_exit_snap, "sell")
        if not (bin_buy_curve and bin_sell_curve and by_buy_curve and by_sell_curve):
            continue

        ret_30s_bps = (ref_entry_px / signal_px - 1.0) * 10000.0
        buy_share_30s = _buy_share(until_entry)
        gross_ref_60m_bps = (ref_exit_px / ref_entry_px - 1.0) * 10000.0

        for notional in ORDER_NOTIONALS_USD:
            bin_entry_bps = _avg_impact_bps_from_curve(bin_buy_curve, notional)
            bin_exit_bps = _avg_impact_bps_from_curve(bin_sell_curve, notional)
            by_entry_bps = _avg_impact_bps_from_curve(by_buy_curve, notional)
            by_exit_bps = _avg_impact_bps_from_curve(by_sell_curve, notional)
            if (
                bin_entry_bps is None
                or bin_exit_bps is None
                or by_entry_bps is None
                or by_exit_bps is None
            ):
                continue

            out.append(
                {
                    "ts": ts0,
                    "symbol": symbol,
                    "source_study_period": row["study_period"],
                    "order_notional_usd": notional,
                    "ret_30s_bps": ret_30s_bps,
                    "buy_share_30s": buy_share_30s,
                    "gross_ref_60m_bps": gross_ref_60m_bps,
                    "binance_entry_impact_bps": bin_entry_bps,
                    "binance_exit_impact_bps": bin_exit_bps,
                    "binance_entry_depth_staleness_s": bin_entry_stale,
                    "binance_exit_depth_staleness_s": bin_exit_stale,
                    "binance_net_exec_60m_bps_20": _net_exec_bps(ref_entry_px, ref_exit_px, bin_entry_bps, bin_exit_bps),
                    "bybit_entry_impact_bps": by_entry_bps,
                    "bybit_exit_impact_bps": by_exit_bps,
                    "bybit_entry_depth_staleness_s": by_entry_stale,
                    "bybit_exit_depth_staleness_s": by_exit_stale,
                    "bybit_net_exec_60m_bps_20": _net_exec_bps(ref_entry_px, ref_exit_px, by_entry_bps, by_exit_bps),
                }
            )

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_values(["ts", "symbol", "order_notional_usd"]).reset_index(drop=True)
    unique_ts = sorted(df["ts"].unique())
    split_idx = max(1, len(unique_ts) // 2)
    cutoff = unique_ts[split_idx]
    df["study_period"] = np.where(df["ts"] < cutoff, "train", "test")
    return df


def main() -> None:
    feats = build_features()
    feats.to_csv(OUT_FEATURES_CSV, index=False)

    if feats.empty:
        OUT_RESULTS_CSV.write_text("")
        REPORT_MD.write_text(
            "# Cross-Venue Orderbook Comparison (Binance vs Bybit)\n\n"
            "No overlap rows were available with both Binance and Bybit depth at required entry/exit timestamps.\n"
        )
        print("No overlap rows available.")
        return

    rows: list[dict[str, object]] = []
    for notional, bucket in feats.groupby("order_notional_usd", sort=True):
        test = bucket.loc[bucket["study_period"] == "test"].copy()
        if test.empty:
            continue
        test_g = _gate(test, GATE)
        row = {
            "order_notional_usd": float(notional),
            "test_rows_overlap": int(len(test)),
            "test_gated_rows_overlap": int(len(test_g)),
            "binance_test_avg_bps": float(test["binance_net_exec_60m_bps_20"].mean()),
            "bybit_test_avg_bps": float(test["bybit_net_exec_60m_bps_20"].mean()),
            "test_diff_bybit_minus_binance_bps": float(
                test["bybit_net_exec_60m_bps_20"].mean() - test["binance_net_exec_60m_bps_20"].mean()
            ),
            "binance_test_gated_avg_bps": float(test_g["binance_net_exec_60m_bps_20"].mean()) if not test_g.empty else np.nan,
            "bybit_test_gated_avg_bps": float(test_g["bybit_net_exec_60m_bps_20"].mean()) if not test_g.empty else np.nan,
            "test_gated_diff_bybit_minus_binance_bps": (
                float(test_g["bybit_net_exec_60m_bps_20"].mean() - test_g["binance_net_exec_60m_bps_20"].mean())
                if not test_g.empty
                else np.nan
            ),
            "symbols_overlap_test": ",".join(sorted(test["symbol"].unique().tolist())),
        }
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("order_notional_usd")
    res.to_csv(OUT_RESULTS_CSV, index=False)

    base_overlap_syms = sorted(feats["symbol"].unique().tolist())
    lines = [
        "# Cross-Venue Orderbook Comparison: Binance vs Bybit",
        "",
        "## Setup",
        "- Same signal timestamps and same Binance agg-trade reference prices for entry/exit.",
        "- Only execution depth source is changed: Binance `book_depth` vs Bybit `orderbook/bybit_futures`.",
        "- Fees: 20 bps round-trip.",
        f"- Gate fixed from current Binance walk-forward: `ret_30s_bps >= {GATE.min_ret_30s_bps:.6f}` and `buy_share_30s >= {GATE.min_buy_share_30s:.6f}`.",
        f"- Overlap symbols in this run: `{','.join(base_overlap_syms)}`",
        "",
        "## Results (Test, overlap subset only)",
        "",
        res.to_string(index=False),
        "",
        "## Notes",
        "- This is an overlap-only comparison, not a full-universe backtest.",
        "- Current Bybit orderbook coverage is sparse vs Binance, so results are directionally useful but not final.",
    ]
    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {OUT_FEATURES_CSV}")
    print(f"Wrote {OUT_RESULTS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
