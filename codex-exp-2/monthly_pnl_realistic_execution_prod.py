from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BINANCE = ROOT / "datalake" / "binance"
OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"

TRADES_OUT = OUT_DIR / "realistic_prod_symbol_trades.csv"
TS_OUT = OUT_DIR / "realistic_prod_timestamp_returns.csv"
MONTHLY_OUT = OUT_DIR / "realistic_prod_monthly_breakdown.csv"
REPORT_MD = OUT_DIR / "FINDINGS_realistic_prod_monthly.md"

# Frozen production config
LS_THRESHOLD = 2.0
TAKER_THRESHOLD = 0.5
MIN_OI = 20_000_000.0
TOP_N = 5
BREADTH = 0.60
MEDIAN_LS = 0.0

# Production portfolio
INITIAL_EQUITY = 1000.0
ALLOCATION = 1.0
LEVERAGE = 3.0

# Execution modeling
FEE_MAKER_BPS_RT = 8.0
FEE_MIXED_BPS_RT = 14.0
FEE_TAKER_BPS_RT = 20.0
SPREAD_BPS_RT = 2.0
MAX_STALENESS_S = 120.0
FALLBACK_COST_BPS = 35.0
HOLD_HOURS = 4


@lru_cache(maxsize=None)
def load_depth_day(symbol: str, day: str) -> pd.DataFrame | None:
    p = BINANCE / symbol / f"{day}_bookDepth.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["timestamp", "percentage", "notional"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    return df.dropna(subset=["ts", "percentage", "notional"])


@lru_cache(maxsize=None)
def load_trades_day(symbol: str, day: str) -> pd.DataFrame | None:
    p = BINANCE / symbol / f"{day}_trades.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["time", "quote_qty", "price", "qty"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["quote_qty"] = pd.to_numeric(df["quote_qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["quote"] = df["quote_qty"].fillna(df["price"] * df["qty"]).fillna(0.0)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    return df.dropna(subset=["ts", "quote"]).sort_values("ts")


def snapshot(depth: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.DataFrame | None, float | None]:
    snaps = depth.loc[depth["ts"] <= ts, "ts"]
    if snaps.empty:
        return None, None
    t = snaps.max()
    stale = float((ts - t).total_seconds())
    if stale > MAX_STALENESS_S:
        return None, None
    return depth.loc[depth["ts"] == t].copy(), stale


def build_curve(snap: pd.DataFrame, side: str) -> list[tuple[float, float]]:
    if side == "buy":
        pcts = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
    else:
        pcts = [-0.2, -1.0, -2.0, -3.0, -4.0, -5.0]
    out = []
    for p in pcts:
        vals = snap.loc[snap["percentage"] == p, "notional"]
        if vals.empty:
            return []
        out.append((abs(float(p)), float(vals.iloc[0])))
    out.sort(key=lambda x: x[0])
    return out


def impact_bps(curve: list[tuple[float, float]], order_notional: float) -> float | None:
    if not curve or order_notional <= 0:
        return None
    if curve[-1][1] < order_notional:
        return None
    prev_cum = 0.0
    prev_pct = 0.0
    acc = 0.0
    rem = order_notional
    for pct, cum in curve:
        band = max(0.0, cum - prev_cum)
        if band <= 0:
            prev_cum = cum
            prev_pct = pct
            continue
        take = min(rem, band)
        end_pct = prev_pct + (pct - prev_pct) * (take / band)
        acc += take * (prev_pct + end_pct) / 2.0
        rem -= take
        prev_cum = cum
        prev_pct = pct
        if rem <= 1e-9:
            break
    if rem > 1e-9:
        return None
    return acc / order_notional * 100.0


def flow_30m(symbol: str, ts: pd.Timestamp) -> float | None:
    day = ts.strftime("%Y-%m-%d")
    t = load_trades_day(symbol, day)
    if t is None or t.empty:
        return None
    left = ts - pd.Timedelta(minutes=30)
    x = t.loc[(t["ts"] > left) & (t["ts"] <= ts), "quote"]
    return float(x.sum()) if not x.empty else 0.0


def execution_cost_bps(symbol: str, ts_entry: pd.Timestamp, order_notional: float) -> tuple[float, str]:
    ts_exit = ts_entry + pd.Timedelta(hours=HOLD_HOURS)
    day_e = ts_entry.strftime("%Y-%m-%d")
    day_x = ts_exit.strftime("%Y-%m-%d")
    de = load_depth_day(symbol, day_e)
    dx = load_depth_day(symbol, day_x)
    if de is None or dx is None:
        return FALLBACK_COST_BPS, "fallback_no_depth"

    se, _ = snapshot(de, ts_entry)
    sx, _ = snapshot(dx, ts_exit)
    if se is None or sx is None:
        return FALLBACK_COST_BPS, "fallback_stale"

    buy = build_curve(se, "buy")
    sell = build_curve(sx, "sell")
    ie = impact_bps(buy, order_notional)
    ix = impact_bps(sell, order_notional)
    if ie is None or ix is None:
        return FALLBACK_COST_BPS, "fallback_no_liquidity"

    d02e = se.loc[se["percentage"] == 0.2, "notional"]
    d02x = sx.loc[sx["percentage"] == -0.2, "notional"]
    fe = flow_30m(symbol, ts_entry)
    fx = flow_30m(symbol, ts_exit)
    maker_e = bool((not d02e.empty) and float(d02e.iloc[0]) >= order_notional and (fe is not None) and fe >= 5 * order_notional)
    maker_x = bool((not d02x.empty) and float(d02x.iloc[0]) >= order_notional and (fx is not None) and fx >= 5 * order_notional)

    if maker_e and maker_x:
        fee = FEE_MAKER_BPS_RT
        mode = "maker_maker"
    elif maker_e or maker_x:
        fee = FEE_MIXED_BPS_RT
        mode = "maker_taker"
    else:
        fee = FEE_TAKER_BPS_RT
        mode = "taker_taker"

    return float(ie + ix + SPREAD_BPS_RT + fee), mode


def select_positions(samples: pd.DataFrame) -> pd.DataFrame:
    f = samples.loc[
        (samples["oi_med_3d"] >= MIN_OI)
        & (samples["breadth_mom"] >= BREADTH)
        & (samples["median_ls_z"] >= MEDIAN_LS)
        & (samples["ls_z"] >= LS_THRESHOLD)
        & (samples["taker_z"] >= TAKER_THRESHOLD)
        & (samples["mom_4h"] > 0)
    ].copy()
    f = (
        f.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    return f


def main() -> None:
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    pos = select_positions(s)
    pos["period"] = np.where(pos["ts"] >= pd.Timestamp("2026-01-01 00:00:00+00:00"), "test", "train")

    symbol_rows = []
    ts_rows = []

    equity = INITIAL_EQUITY
    for ts, g in pos.groupby("ts", sort=True):
        n = len(g)
        if n == 0:
            continue
        per_pos_notional = max(1.0, equity * ALLOCATION * LEVERAGE / n)

        net_rets = []
        for r in g.itertuples(index=False):
            cost_bps, mode = execution_cost_bps(r.symbol, r.ts, per_pos_notional)
            ret_net_1x = float(r.ret_4h - cost_bps / 10000.0)
            net_rets.append(ret_net_1x)
            symbol_rows.append(
                {
                    "ts": r.ts,
                    "symbol": r.symbol,
                    "period": "test" if r.ts >= pd.Timestamp("2026-01-01 00:00:00+00:00") else "train",
                    "raw_ret_1x": float(r.ret_4h),
                    "cost_bps": float(cost_bps),
                    "net_ret_1x": ret_net_1x,
                    "mode": mode,
                    "order_notional_usd": per_pos_notional,
                }
            )

        port_ret_1x = float(np.mean(net_rets))
        sleeve_mult = max(0.0, 1.0 + LEVERAGE * port_ret_1x)
        step_mult = (1.0 - ALLOCATION) + ALLOCATION * sleeve_mult
        equity = equity * step_mult

        ts_rows.append(
            {
                "ts": ts,
                "period": "test" if ts >= pd.Timestamp("2026-01-01 00:00:00+00:00") else "train",
                "n_positions": n,
                "order_notional_usd": per_pos_notional,
                "port_ret_1x_net": port_ret_1x,
                "port_ret_equity": step_mult - 1.0,
                "equity": equity,
            }
        )

    sym_df = pd.DataFrame(symbol_rows).sort_values(["ts", "symbol"]).reset_index(drop=True)
    ts_df = pd.DataFrame(ts_rows).sort_values("ts").reset_index(drop=True)

    if ts_df.empty:
        raise RuntimeError("No timestamp returns produced.")

    m = ts_df.copy()
    m["month"] = m["ts"].dt.strftime("%Y-%m")
    monthly = m.groupby("month", as_index=False).agg(
        trades=("ts", "count"),
        avg_ret_bps=("port_ret_equity", lambda x: float(x.mean() * 10000.0)),
        compounded_ret_pct=("port_ret_equity", lambda x: float((1.0 + x).prod() - 1.0) * 100.0),
        end_equity=("equity", "last"),
    )

    sym_df.to_csv(TRADES_OUT, index=False)
    ts_df.to_csv(TS_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)

    mode_share = sym_df["mode"].value_counts(normalize=True).mul(100).round(1).to_dict()
    lines = [
        "# Realistic Production Monthly PnL",
        "",
        "- Strategy filters: frozen production config (`ls>=2.0`, `taker>=0.5`, `oi>=20M`, `breadth>=0.60`, `top_n=5`).",
        f"- Portfolio: allocation={ALLOCATION:.2f}, leverage={LEVERAGE:.1f}x, initial equity=${INITIAL_EQUITY:.0f}.",
        "- Execution model per symbol-trade: depth-based slippage (entry+exit), spread, maker/taker fee regime.",
        f"- Fee schedule: maker/maker={FEE_MAKER_BPS_RT:.1f} bps RT, mixed={FEE_MIXED_BPS_RT:.1f} bps RT, taker/taker={FEE_TAKER_BPS_RT:.1f} bps RT.",
        f"- Fallback cost for missing/stale depth: {FALLBACK_COST_BPS:.1f} bps RT.",
        "",
        "## Coverage",
        f"- Symbol-trades modeled: {len(sym_df)}",
        f"- Timestamp decisions modeled: {len(ts_df)}",
        f"- Execution mode mix (%): {mode_share}",
        "",
        "## Result",
        f"- Final equity: ${ts_df['equity'].iloc[-1]:,.2f}",
        f"- Total return: {(ts_df['equity'].iloc[-1]/INITIAL_EQUITY-1.0)*100.0:.1f}%",
        "",
        "## Files",
        f"- `{TRADES_OUT.name}`",
        f"- `{TS_OUT.name}`",
        f"- `{MONTHLY_OUT.name}`",
    ]
    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {TRADES_OUT}")
    print(f"Wrote {TS_OUT}")
    print(f"Wrote {MONTHLY_OUT}")
    print(f"Wrote {REPORT_MD}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
