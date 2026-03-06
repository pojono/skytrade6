from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake" / "binance"
OUT_DIR = Path(__file__).resolve().parent

SAMPLES_CSV = OUT_DIR / "samples_4h.csv"
TRADES_OUT = OUT_DIR / "improved_vol_exec_trades.csv"
MONTHLY_OUT = OUT_DIR / "improved_vol_exec_monthly.csv"
SUMMARY_OUT = OUT_DIR / "FINDINGS_improved_vol_exec.md"

INITIAL_EQUITY = 1000.0
FEE_MIXED = 0.0014
FEE_ALL_TAKER = 0.0020

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

TOP_K_CANDIDATES = 5
MAX_STALENESS_S = 40.0
VOL_WINDOW_5M = 24 * 12  # 24h of 5m bars
MAX_WEIGHT = 0.55

BASE_CFG = {
    "ls_threshold": 2.0,
    "taker_threshold": 0.5,
    "min_oi_value": 20_000_000.0,
    "breadth_threshold": 0.60,
    "median_ls_threshold": 0.0,
}


@dataclass(frozen=True)
class GateRule:
    depth_q: float
    flow_q: float
    depth_min: float
    flow_min: float


def _period(ts: pd.Series) -> pd.Series:
    out = pd.Series(np.where(ts >= TEST_START, "test", "train"), index=ts.index)
    return out


def load_samples() -> pd.DataFrame:
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    s = s.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return s


def select_candidates(samples: pd.DataFrame) -> pd.DataFrame:
    f = samples.loc[
        (samples["oi_med_3d"] >= BASE_CFG["min_oi_value"])
        & (samples["breadth_mom"] >= BASE_CFG["breadth_threshold"])
        & (samples["median_ls_z"] >= BASE_CFG["median_ls_threshold"])
        & (samples["ls_z"] >= BASE_CFG["ls_threshold"])
        & (samples["taker_z"] >= BASE_CFG["taker_threshold"])
        & (samples["mom_4h"] > 0)
    ].copy()
    f = (
        f.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(TOP_K_CANDIDATES)
        .reset_index(drop=True)
    )
    f["period"] = _period(f["ts"])
    return f


def _load_symbol_vol(symbol: str) -> pd.DataFrame | None:
    files = sorted(
        p
        for p in (DATALAKE / symbol).glob("*_kline_1m.csv")
        if "mark_price" not in p.name
        and "premium_index" not in p.name
        and "index_price" not in p.name
    )
    if not files:
        return None
    parts = []
    for p in files:
        df = pd.read_csv(p, usecols=["open_time", "close"])
        parts.append(df)
    k = pd.concat(parts, ignore_index=True)
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k["close"] = pd.to_numeric(k["close"], errors="coerce")
    k = (
        k[["ts", "close"]]
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )
    k5 = k["close"].resample("5min", label="right", closed="right").last()
    ret = k5.pct_change(fill_method=None)
    vol = ret.rolling(VOL_WINDOW_5M, min_periods=VOL_WINDOW_5M).std(ddof=0).shift(1)
    out = vol.rename("vol_24h").reset_index()
    out["symbol"] = symbol
    return out


def add_vol_feature(candidates: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(candidates["symbol"].unique())
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(_load_symbol_vol, symbols))
    vol_frames = [p for p in parts if p is not None and not p.empty]
    vol = pd.concat(vol_frames, ignore_index=True)
    out = candidates.merge(vol, on=["symbol", "ts"], how="left")
    return out


@lru_cache(maxsize=None)
def _load_depth_1pct(symbol: str, day: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = DATALAKE / symbol / f"{day}_bookDepth.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["timestamp", "percentage", "notional"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df = df.loc[df["percentage"].isin([-1.0, 1.0])].dropna(subset=["ts", "notional"])
    if df.empty:
        return None
    g = df.groupby("ts")["notional"].apply(lambda s: float(np.abs(s).sum())).reset_index()
    g = g.rename(columns={"notional": "depth_1pct"}).sort_values("ts")
    ts_ns = g["ts"].astype("int64").to_numpy()
    val = g["depth_1pct"].to_numpy(dtype=float)
    return ts_ns, val


@lru_cache(maxsize=None)
def _load_trade_cumsum(symbol: str, day: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = DATALAKE / symbol / f"{day}_trades.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["time", "price", "qty", "quote_qty"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["quote_qty"] = pd.to_numeric(df["quote_qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    q = df["quote_qty"].fillna(df["price"] * df["qty"]).fillna(0.0)
    ts_ns = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce").astype("int64").to_numpy()
    valid = np.isfinite(ts_ns) & np.isfinite(q.to_numpy())
    ts_ns = ts_ns[valid]
    qv = q.to_numpy(dtype=float)[valid]
    if ts_ns.size == 0:
        return None
    order = np.argsort(ts_ns)
    ts_ns = ts_ns[order]
    qv = qv[order]
    csum = np.cumsum(qv)
    return ts_ns, csum


def _flow_30m(symbol: str, day: str, ts: pd.Timestamp) -> float | None:
    payload = _load_trade_cumsum(symbol, day)
    if payload is None:
        return None
    t_ns, csum = payload
    cur = int(ts.value)
    left = int((ts - pd.Timedelta(minutes=30)).value)
    i2 = np.searchsorted(t_ns, cur, side="right")
    i1 = np.searchsorted(t_ns, left, side="right")
    if i2 <= i1:
        return 0.0
    right = csum[i2 - 1]
    leftv = csum[i1 - 1] if i1 > 0 else 0.0
    return float(right - leftv)


def _depth_at(symbol: str, day: str, ts: pd.Timestamp) -> tuple[float | None, float | None]:
    payload = _load_depth_1pct(symbol, day)
    if payload is None:
        return None, None
    t_ns, val = payload
    cur = int(ts.value)
    i = np.searchsorted(t_ns, cur, side="right") - 1
    if i < 0:
        return None, None
    snap_ns = int(t_ns[i])
    stale = (cur - snap_ns) / 1_000_000_000.0
    return float(val[i]), float(stale)


def add_execution_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    days = out["ts"].dt.strftime("%Y-%m-%d")
    depth = []
    stale = []
    flow = []
    for sym, day, ts in zip(out["symbol"], days, out["ts"]):
        d, s = _depth_at(sym, day, ts)
        f = _flow_30m(sym, day, ts)
        depth.append(d)
        stale.append(s)
        flow.append(f)
    out["depth_1pct_usd"] = depth
    out["depth_staleness_s"] = stale
    out["flow_30m_usd"] = flow
    return out


def _normalize_capped(weights: pd.Series, cap: float = MAX_WEIGHT) -> pd.Series:
    w = weights.clip(lower=0.0).astype(float)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(3):
        over = w > cap
        if not over.any():
            break
        fixed = w.where(over, 0.0).clip(upper=cap)
        rem = 1.0 - fixed.sum()
        if rem <= 0:
            w = fixed / fixed.sum()
            break
        free = (~over)
        if w[free].sum() <= 0:
            w = fixed
            break
        w = fixed + (w.where(free, 0.0) / w[free].sum()) * rem
    return w / w.sum()


def _build_portfolio(trades: pd.DataFrame, use_vol_weight: bool) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["ts", "raw_ret", "n_names"])
    rows = []
    for ts, g in trades.groupby("ts", sort=True):
        if use_vol_weight:
            gv = g.dropna(subset=["vol_24h"]).copy()
            gv = gv.loc[gv["vol_24h"] > 0].copy()
            if gv.empty:
                continue
            score = gv["score_abs"] / gv["vol_24h"].clip(lower=1e-9)
            w = _normalize_capped(score, cap=MAX_WEIGHT)
            raw = float((gv["ret_4h"] * w).sum())
            n = int(gv.shape[0])
        else:
            raw = float(g["ret_4h"].mean())
            n = int(g.shape[0])
        rows.append({"ts": ts, "raw_ret": raw, "n_names": n})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def _score_train(port: pd.DataFrame) -> float:
    t = port.loc[port["ts"] <= TRAIN_END]
    if t.empty:
        return -1e9
    return float((t["raw_ret"] - FEE_MIXED).mean() * 10000.0)


def choose_gate_rule(candidates: pd.DataFrame) -> GateRule:
    train = candidates.loc[candidates["period"] == "train"].copy()
    train = train.dropna(subset=["depth_1pct_usd", "flow_30m_usd", "depth_staleness_s", "vol_24h"])
    train = train.loc[train["depth_staleness_s"] <= MAX_STALENESS_S].copy()
    if train.empty:
        return GateRule(0.0, 0.0, 0.0, 0.0)

    best_rule = GateRule(0.3, 0.3, 0.0, 0.0)
    best_score = -1e9
    for dq in (0.2, 0.3, 0.4, 0.5, 0.6):
        for fq in (0.2, 0.3, 0.4, 0.5, 0.6):
            dmin = float(train["depth_1pct_usd"].quantile(dq))
            fmin = float(train["flow_30m_usd"].quantile(fq))
            gated = train.loc[
                (train["depth_1pct_usd"] >= dmin)
                & (train["flow_30m_usd"] >= fmin)
            ].copy()
            if gated["ts"].nunique() < 60:
                continue
            p = _build_portfolio(gated, use_vol_weight=True)
            score = _score_train(p)
            if score > best_score:
                best_score = score
                best_rule = GateRule(dq, fq, dmin, fmin)
    return best_rule


def apply_gate(candidates: pd.DataFrame, rule: GateRule) -> pd.DataFrame:
    c = candidates.copy()
    c = c.dropna(subset=["depth_1pct_usd", "flow_30m_usd", "depth_staleness_s"])
    c = c.loc[
        (c["depth_staleness_s"] <= MAX_STALENESS_S)
        & (c["depth_1pct_usd"] >= rule.depth_min)
        & (c["flow_30m_usd"] >= rule.flow_min)
    ].copy()
    return c


def _add_equity(port: pd.DataFrame, fee: float) -> pd.Series:
    if port.empty:
        return pd.Series(dtype=float)
    return INITIAL_EQUITY * (1.0 + (port["raw_ret"] - fee)).cumprod()


def monthly(port: pd.DataFrame, label: str) -> pd.DataFrame:
    if port.empty:
        return pd.DataFrame(columns=["month", "strategy", "trades", "avg_raw_bps", "avg_net_bps_mixed", "comp_ret_mixed"])
    x = port.copy()
    x["month"] = x["ts"].dt.strftime("%Y-%m")
    m = (
        x.groupby("month", as_index=False)
        .agg(
            trades=("ts", "count"),
            avg_raw_bps=("raw_ret", lambda s: float(s.mean() * 10000.0)),
            avg_net_bps_mixed=("raw_ret", lambda s: float((s.mean() - FEE_MIXED) * 10000.0)),
            comp_ret_mixed=("raw_ret", lambda s: float((1.0 + (s - FEE_MIXED)).prod() - 1.0)),
        )
    )
    m["strategy"] = label
    return m


def summarize_md(
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    rule: GateRule,
    candidates: pd.DataFrame,
) -> None:
    b_mixed = float(_add_equity(baseline, FEE_MIXED).iloc[-1]) if not baseline.empty else INITIAL_EQUITY
    i_mixed = float(_add_equity(improved, FEE_MIXED).iloc[-1]) if not improved.empty else INITIAL_EQUITY
    b_taker = float(_add_equity(baseline, FEE_ALL_TAKER).iloc[-1]) if not baseline.empty else INITIAL_EQUITY
    i_taker = float(_add_equity(improved, FEE_ALL_TAKER).iloc[-1]) if not improved.empty else INITIAL_EQUITY

    b_test = baseline.loc[baseline["ts"] >= TEST_START]
    i_test = improved.loc[improved["ts"] >= TEST_START]

    lines = [
        "# Improved Strategy: Vol-Target + Execution Gate",
        "",
        "## Scope",
        "- Data source: raw `datalake/binance` only",
        "- Window: `2025-01-01` to `2026-03-04` available; strategy warmup starts trading later",
        f"- Candidate rows (top-{TOP_K_CANDIDATES} pre-gate): `{len(candidates)}`",
        "",
        "## Gate (chosen on train only)",
        f"- Max staleness: `{MAX_STALENESS_S:.0f}s`",
        f"- Depth quantile: `{rule.depth_q}` -> min depth_1pct `{rule.depth_min:,.0f}` USD",
        f"- Flow quantile: `{rule.flow_q}` -> min flow_30m `{rule.flow_min:,.0f}` USD",
        "",
        "## Full-Period Equity from $1,000",
        f"- Baseline (equal-weight top-3), mixed 14 bps: `${b_mixed:,.2f}`",
        f"- Improved (vol-target + exec gate), mixed 14 bps: `${i_mixed:,.2f}`",
        f"- Baseline all-taker 20 bps: `${b_taker:,.2f}`",
        f"- Improved all-taker 20 bps: `${i_taker:,.2f}`",
        "",
        "## Test-Only (2026+)",
        f"- Baseline test timestamps: `{len(b_test)}`",
        f"- Improved test timestamps: `{len(i_test)}`",
    ]

    if not b_test.empty:
        lines.append(f"- Baseline test avg net bps (mixed): `{((b_test['raw_ret']-FEE_MIXED).mean()*10000):.2f}`")
    if not i_test.empty:
        lines.append(f"- Improved test avg net bps (mixed): `{((i_test['raw_ret']-FEE_MIXED).mean()*10000):.2f}`")

    lines.extend(
        [
            "",
            "## Files",
            "- `improved_vol_exec_trades.csv`",
            "- `improved_vol_exec_monthly.csv`",
        ]
    )
    SUMMARY_OUT.write_text("\n".join(lines))


def main() -> None:
    samples = load_samples()
    candidates = select_candidates(samples)
    candidates = add_vol_feature(candidates)
    candidates = add_execution_features(candidates)

    # Baseline portfolio for fair comparison: equal-weight top-3 (existing approach).
    baseline_candidates = (
        candidates.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(3)
        .copy()
    )
    baseline_port = _build_portfolio(baseline_candidates, use_vol_weight=False)

    rule = choose_gate_rule(candidates)
    gated = apply_gate(candidates, rule)
    improved_port = _build_portfolio(gated, use_vol_weight=True)

    baseline_port["strategy"] = "baseline_eqw_top3"
    improved_port["strategy"] = "improved_vol_exec"
    both = pd.concat([baseline_port, improved_port], ignore_index=True).sort_values(["strategy", "ts"])
    both["net_ret_mixed"] = both["raw_ret"] - FEE_MIXED
    both["net_ret_all_taker"] = both["raw_ret"] - FEE_ALL_TAKER
    both["period"] = _period(both["ts"])

    eq_parts = []
    for name, g in both.groupby("strategy", sort=False):
        g = g.sort_values("ts").copy()
        g["equity_mixed"] = _add_equity(g, FEE_MIXED).values
        g["equity_all_taker"] = _add_equity(g, FEE_ALL_TAKER).values
        eq_parts.append(g)
    both = pd.concat(eq_parts, ignore_index=True).sort_values(["strategy", "ts"])

    month = pd.concat(
        [monthly(baseline_port, "baseline_eqw_top3"), monthly(improved_port, "improved_vol_exec")],
        ignore_index=True,
    ).sort_values(["month", "strategy"])

    both.to_csv(TRADES_OUT, index=False)
    month.to_csv(MONTHLY_OUT, index=False)
    summarize_md(baseline_port, improved_port, rule, candidates)

    print(f"Wrote {TRADES_OUT}")
    print(f"Wrote {MONTHLY_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print("\nFinal mixed-fee equity:")
    fin = both.groupby("strategy", as_index=False).tail(1)[["strategy", "equity_mixed", "equity_all_taker", "ts"]]
    print(fin.to_string(index=False))


if __name__ == "__main__":
    main()
