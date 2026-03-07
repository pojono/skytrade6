#!/usr/bin/env python3
"""
XS-9 — Combined Strategy: Vol Dip-Buying + Fragility Overlay + Compression Gate

Combines:
  1. Vol Dip-Buying (Tier 1): rvol_z + mr_4h signal, 4h hold, 1h candles
  2. Fragility Overlay (Tier 2): crowd_oi + pca_var1 from XS-8, position sizing
  3. Compression Gate (Tier 2): S07 signal (rv_low + oi_high), entry timing

Test variants:
  A) Baseline: Vol dip-buying standalone (reproduce known results)
  B) + Fragility sizing: Scale position 50-100% based on fragility quintile
  C) + Compression gate: Only enter when market is compressed (or widen threshold)
  D) + Both overlays combined
  E) + Asymmetric: Heavier downside protection in fragile regimes

Data: datalake/bybit 1m klines → 1h bars, XS-8c features (5m grid)
Period: 2025-07-01 → 2026-02-28 (8 months, matching XS-8 range)
Symbols: 9 Tier-A from vol dip-buying portfolio
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATALAKE = Path(__file__).resolve().parent.parent / "datalake" / "bybit"
XS8_PATH = Path(__file__).resolve().parent / "output" / "xs8c" / "xs8c_extended.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

SYMBOLS = ["ONDOUSDT", "TAOUSDT", "SOLUSDT", "HBARUSDT", "SEIUSDT",
           "ADAUSDT", "BNBUSDT", "XRPUSDT", "AAVEUSDT"]

# Vol dip-buying params (fixed, DO NOT CHANGE)
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4
RT_FEE_BPS = 4.0

# Fragility overlay params
FRAG_REDUCE_Q5 = 0.50    # reduce to 50% size in Q5 (most fragile)
FRAG_REDUCE_Q4 = 0.75    # reduce to 75% in Q4
FRAG_BOOST_Q1 = 1.00     # no boost in Q1 (conservative)

# Compression gate params (from XS cross-sectional research)
# rv_6h <= P20 expanding AND oi_z >= 1.5 → compressed → big move likely
COMPRESS_RV_PCTL = 0.20
COMPRESS_OI_Z = 1.5

# Asymmetric protection: extra reduction for short trades in fragile regime
ASYM_SHORT_EXTRA_REDUCE = 0.30  # additional 30% cut for shorts in Q5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_1h_from_datalake(symbol):
    """Load 1m klines from datalake and resample to 1h."""
    sym_dir = DATALAKE / symbol
    if not sym_dir.exists():
        return pd.DataFrame()

    # Find kline_1m files (not mark_price, not premium_index)
    files = sorted([f for f in sym_dir.glob("*_kline_1m.csv")
                    if "mark_price" not in f.name and "premium_index" not in f.name])
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["startTime"].astype(int), unit="ms", utc=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    raw = raw[(raw["ts"] >= START) & (raw["ts"] <= END)]

    if len(raw) == 0:
        return pd.DataFrame()

    # Resample to 1h OHLCV
    raw = raw.set_index("ts")
    ohlcv = raw.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    return ohlcv


def load_fragility_features():
    """Load XS-8c features and compute fragility quintiles."""
    if not XS8_PATH.exists():
        print(f"  WARNING: {XS8_PATH} not found!")
        return None

    df = pd.read_parquet(XS8_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Compute fragility score = -crowd_oi + (-pca_var1) [higher = more fragile]
    # Use expanding quantiles for walk-forward correctness
    df = df.dropna(subset=["crowd_oi", "pca_var1"]).copy()

    # Simple logistic model: fragility_score = -4.62 * crowd_oi - 0.95 * pca_var1
    # (coefficients from XS-8 LogReg on 12×ATR ≥10% target)
    df["frag_score"] = -4.62 * df["crowd_oi"] - 0.95 * df["pca_var1"]

    # Expanding quintile (no lookahead)
    df["frag_quintile"] = np.nan
    min_warmup = 2000  # need at least ~1 week of 5m data
    for i in range(min_warmup, len(df)):
        past = df["frag_score"].iloc[:i+1]
        val = df["frag_score"].iloc[i]
        pctl = (past <= val).mean()
        if pctl <= 0.20:
            df.iloc[i, df.columns.get_loc("frag_quintile")] = 1  # Q1 = least fragile
        elif pctl <= 0.40:
            df.iloc[i, df.columns.get_loc("frag_quintile")] = 2
        elif pctl <= 0.60:
            df.iloc[i, df.columns.get_loc("frag_quintile")] = 3
        elif pctl <= 0.80:
            df.iloc[i, df.columns.get_loc("frag_quintile")] = 4
        else:
            df.iloc[i, df.columns.get_loc("frag_quintile")] = 5  # Q5 = most fragile

    return df[["ts", "crowd_oi", "pca_var1", "frag_score", "frag_quintile"]].set_index("ts")


def load_compression_features():
    """Load OI and RV features for compression detection from XS-8c."""
    if not XS8_PATH.exists():
        return None

    df = pd.read_parquet(XS8_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna(subset=["crowd_oi"]).copy()

    # breadth_extreme is our proxy for realized vol level
    # crowd_oi is our OI crowding measure
    # Compression = low breadth (calm market) + high crowd_oi (positions building)
    # This is the inverse of fragility — compressed markets are about to move

    # Expanding percentile for breadth_extreme (walk-forward)
    df["bx_pctl"] = df["breadth_extreme"].expanding(min_periods=2000).rank(pct=True)

    # Compressed = breadth_extreme <= P20 AND crowd_oi >= P80 (expanding)
    df["oi_pctl"] = df["crowd_oi"].expanding(min_periods=2000).rank(pct=True)
    df["compressed"] = ((df["bx_pctl"] <= COMPRESS_RV_PCTL) &
                         (df["oi_pctl"] >= 0.80)).astype(float)

    return df[["ts", "bx_pctl", "oi_pctl", "compressed"]].set_index("ts")


# ---------------------------------------------------------------------------
# Vol dip-buying signal (exact replication)
# ---------------------------------------------------------------------------

def compute_vdb_signals(ohlcv):
    """Compute vol dip-buying signals on 1h data."""
    c = ohlcv["close"].values.astype(np.float64)
    n = len(c)

    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=ohlcv.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    rvol_z = (rvol - rvol_mean) / rvol_std

    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    mr_4h = -((r4 - r4_mean) / r4_std)

    ohlcv["combined"] = (rvol_z.values + mr_4h.values) / 2
    return ohlcv


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(sym_data, frag_df, comp_df, variant="baseline"):
    """
    Run combined backtest across all symbols.

    Variants:
      baseline: pure vol dip-buying, no overlay
      fragility: scale position by fragility quintile
      compression: only enter when compressed (or lower threshold)
      combined: fragility + compression
      asymmetric: combined + extra short protection in fragile
    """
    all_trades = []

    for symbol, ohlcv in sym_data.items():
        if len(ohlcv) < 200:
            continue

        sig = ohlcv["combined"].values
        c = ohlcv["close"].values.astype(np.float64)
        idx = ohlcv.index
        n = len(c)

        last_exit = 0

        for i in range(168, n):  # warmup for rolling windows
            if i < last_exit + COOLDOWN_BARS:
                continue
            if i + HOLD_BARS >= n:
                continue
            if np.isnan(sig[i]) or abs(sig[i]) < THRESHOLD:
                continue

            ts_entry = idx[i]
            trade_dir = "long" if sig[i] > 0 else "short"

            # ── Get fragility state at entry time ──
            frag_q = np.nan
            size_mult = 1.0

            if frag_df is not None and variant in ("fragility", "combined", "asymmetric"):
                # Find nearest 5m fragility reading before entry
                mask = frag_df.index <= ts_entry
                if mask.any():
                    nearest = frag_df.loc[mask].iloc[-1]
                    frag_q = nearest["frag_quintile"]
                    if frag_q == 5:
                        size_mult = FRAG_REDUCE_Q5
                    elif frag_q == 4:
                        size_mult = FRAG_REDUCE_Q4
                    # Q1-Q3: full size

                    # Asymmetric: extra reduce for shorts in fragile regime
                    if variant == "asymmetric" and trade_dir == "short" and frag_q == 5:
                        size_mult *= (1.0 - ASYM_SHORT_EXTRA_REDUCE)

            # ── Check compression gate ──
            is_compressed = False
            if comp_df is not None and variant in ("compression", "combined", "asymmetric"):
                mask = comp_df.index <= ts_entry
                if mask.any():
                    nearest = comp_df.loc[mask].iloc[-1]
                    is_compressed = nearest["compressed"] > 0.5

                if variant == "compression" and not is_compressed:
                    # In pure compression mode, skip non-compressed entries
                    # Actually: don't skip, but boost size when compressed
                    pass  # We'll use compression as a boost, not a gate

                # Compression boost: 20% more size when compressed
                if is_compressed:
                    size_mult *= 1.20

            # ── Execute trade ──
            entry = c[i]
            exit_p = c[i + HOLD_BARS]
            raw_bps = ((exit_p - entry) / entry * 10000) if trade_dir == "long" else \
                      ((entry - exit_p) / entry * 10000)
            net_bps = raw_bps - RT_FEE_BPS

            # Size-adjusted PnL (for portfolio)
            sized_bps = net_bps * size_mult

            all_trades.append({
                "symbol": symbol,
                "ts_entry": ts_entry,
                "ts_exit": idx[i + HOLD_BARS],
                "direction": trade_dir,
                "signal": sig[i],
                "entry": entry,
                "exit": exit_p,
                "raw_bps": raw_bps,
                "net_bps": net_bps,
                "size_mult": size_mult,
                "sized_bps": sized_bps,
                "frag_quintile": frag_q,
                "compressed": is_compressed,
                "variant": variant,
            })

            last_exit = i + HOLD_BARS

    return pd.DataFrame(all_trades)


def analyze_trades(trades_df, label):
    """Analyze trade results."""
    if len(trades_df) == 0:
        print(f"  {label}: NO TRADES")
        return None

    n = len(trades_df)
    net = trades_df["net_bps"]
    sized = trades_df["sized_bps"]

    # Monthly PnL
    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["ts_entry"].dt.to_period("M")
    monthly = trades_df.groupby("month")["sized_bps"].sum() / 100  # to %
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)

    # Portfolio-level stats (equal weight across symbols)
    n_syms = trades_df["symbol"].nunique()

    cum = monthly.cumsum()
    peak = cum.cummax()
    maxdd = (peak - cum).max()

    sharpe = monthly.mean() / max(monthly.std(), 0.001) * np.sqrt(12)
    ann_ret = monthly.sum() / max(total_months / 12, 0.5)

    # Win rate
    wr = (sized > 0).mean() * 100

    # By direction
    longs = trades_df[trades_df["direction"] == "long"]
    shorts = trades_df[trades_df["direction"] == "short"]

    result = {
        "label": label,
        "n_trades": n,
        "n_symbols": n_syms,
        "avg_net_bps": net.mean(),
        "avg_sized_bps": sized.mean(),
        "total_pnl_pct": monthly.sum(),
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "calmar": ann_ret / max(maxdd, 0.1),
        "wr": wr,
        "pos_months": pos_months,
        "total_months": total_months,
        "long_avg": longs["sized_bps"].mean() if len(longs) > 0 else 0,
        "short_avg": shorts["sized_bps"].mean() if len(shorts) > 0 else 0,
        "long_n": len(longs),
        "short_n": len(shorts),
    }

    print(f"\n  ── {label} ──")
    print(f"  Trades: {n} ({n_syms} symbols), WR: {wr:.1f}%")
    print(f"  Avg net: {net.mean():+.1f} bps, Avg sized: {sized.mean():+.1f} bps")
    print(f"  Total PnL: {monthly.sum():+.1f}%, Ann: {ann_ret:+.1f}%/yr")
    print(f"  mSharpe: {sharpe:+.2f}, MaxDD: {maxdd:.1f}%, Calmar: {result['calmar']:.2f}")
    print(f"  Months: {pos_months}/{total_months} positive ({pos_months/total_months*100:.0f}%)")
    print(f"  Long: {len(longs)} trades, avg {longs['sized_bps'].mean():+.1f} bps" if len(longs) > 0 else "  Long: 0")
    print(f"  Short: {len(shorts)} trades, avg {shorts['sized_bps'].mean():+.1f} bps" if len(shorts) > 0 else "  Short: 0")

    # Monthly detail
    print(f"\n  Monthly PnL:")
    for m, pnl in monthly.items():
        n_m = len(trades_df[trades_df["month"] == m])
        print(f"    {m}: {pnl:+.2f}% ({n_m} trades)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 80)
    print("XS-9 — COMBINED STRATEGY BACKTEST")
    print(f"Period: {START.date()} → {END.date()}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Variants: baseline, fragility, compression, combined, asymmetric")
    print("=" * 80, flush=True)

    # ── Phase 1: Load 1h data ──
    print(f"\n{'─'*70}")
    print("PHASE 1: Loading 1h data from datalake")
    print(f"{'─'*70}", flush=True)

    sym_data = {}
    for sym in SYMBOLS:
        ohlcv = load_1h_from_datalake(sym)
        if len(ohlcv) == 0:
            print(f"  {sym}: NO DATA")
            continue
        ohlcv = compute_vdb_signals(ohlcv)
        sym_data[sym] = ohlcv
        n_sigs = (ohlcv["combined"].abs() > THRESHOLD).sum()
        print(f"  {sym}: {len(ohlcv)} bars, {n_sigs} signals "
              f"({ohlcv.index[0].strftime('%Y-%m-%d')} → {ohlcv.index[-1].strftime('%Y-%m-%d')})",
              flush=True)

    # ── Phase 2: Load fragility + compression features ──
    print(f"\n{'─'*70}")
    print("PHASE 2: Loading fragility & compression overlays")
    print(f"{'─'*70}", flush=True)

    print("  Loading fragility features...", flush=True)
    frag_df = load_fragility_features()
    if frag_df is not None:
        valid = frag_df["frag_quintile"].dropna()
        print(f"  Fragility: {len(valid):,} valid readings, "
              f"Q1={( valid == 1).sum():,}, Q5={(valid == 5).sum():,}")

    print("  Loading compression features...", flush=True)
    comp_df = load_compression_features()
    if comp_df is not None:
        compressed = comp_df["compressed"].sum()
        print(f"  Compression: {int(compressed):,} compressed readings "
              f"out of {len(comp_df):,} ({compressed/len(comp_df)*100:.1f}%)")

    # ── Phase 3: Run all variants ──
    print(f"\n{'─'*70}")
    print("PHASE 3: Running backtests")
    print(f"{'─'*70}", flush=True)

    variants = [
        ("A) Baseline (pure VDB)", "baseline"),
        ("B) + Fragility sizing", "fragility"),
        ("C) + Compression boost", "compression"),
        ("D) + Both (frag + comp)", "combined"),
        ("E) + Asymmetric downside", "asymmetric"),
    ]

    all_results = []
    all_trades = {}

    for label, variant in variants:
        print(f"\n  Running: {label}...", flush=True)
        trades = run_backtest(sym_data, frag_df, comp_df, variant=variant)
        r = analyze_trades(trades, label)
        if r:
            all_results.append(r)
        all_trades[variant] = trades

    # ── Phase 4: Comparison summary ──
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}", flush=True)

    if all_results:
        print(f"\n  {'Variant':35s}  {'Trades':>6s}  {'Avg':>7s}  {'Total':>7s}  "
              f"{'Ann':>7s}  {'Sharpe':>7s}  {'DD':>6s}  {'WR':>5s}  {'Pos Mo':>7s}")
        print("  " + "-" * 100)

        baseline = all_results[0] if all_results else None

        for r in all_results:
            delta = ""
            if baseline and r["label"] != baseline["label"]:
                d_sharpe = r["sharpe"] - baseline["sharpe"]
                d_dd = r["maxdd"] - baseline["maxdd"]
                delta = f"  ΔS={d_sharpe:+.2f} ΔDD={d_dd:+.1f}%"

            print(f"  {r['label']:35s}  {r['n_trades']:>6d}  {r['avg_sized_bps']:>+6.1f}  "
                  f"{r['total_pnl_pct']:>+6.1f}%  {r['ann_return']:>+6.1f}%  "
                  f"{r['sharpe']:>+6.2f}  {r['maxdd']:>5.1f}%  {r['wr']:>4.1f}%  "
                  f"{r['pos_months']}/{r['total_months']}{delta}")

    # ── Phase 5: Fragility quintile analysis ──
    print(f"\n{'─'*70}")
    print("PHASE 5: Trade performance by fragility quintile")
    print(f"{'─'*70}", flush=True)

    baseline_trades = all_trades.get("baseline", pd.DataFrame())
    if len(baseline_trades) > 0 and frag_df is not None:
        # Assign fragility quintile to baseline trades
        bt = baseline_trades.copy()
        frag_qs = []
        for _, row in bt.iterrows():
            ts = row["ts_entry"]
            mask = frag_df.index <= ts
            if mask.any():
                frag_qs.append(frag_df.loc[mask, "frag_quintile"].iloc[-1])
            else:
                frag_qs.append(np.nan)
        bt["fq"] = frag_qs

        print(f"\n  Baseline VDB trades split by fragility quintile:")
        print(f"  {'Quintile':>10s}  {'N':>5s}  {'Avg bps':>8s}  {'WR':>6s}  {'Long avg':>9s}  {'Short avg':>10s}")

        for q in [1, 2, 3, 4, 5]:
            qd = bt[bt["fq"] == q]
            if len(qd) == 0:
                continue
            avg = qd["net_bps"].mean()
            wr = (qd["net_bps"] > 0).mean() * 100
            l_avg = qd[qd["direction"] == "long"]["net_bps"].mean()
            s_avg = qd[qd["direction"] == "short"]["net_bps"].mean()
            label_q = {1: "Q1 (safe)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (fragile)"}
            print(f"  {label_q.get(q, f'Q{q}'):>10s}  {len(qd):>5d}  {avg:>+7.1f}  "
                  f"{wr:>5.1f}%  {l_avg:>+8.1f}  {s_avg:>+9.1f}")

        # By direction × fragility
        print(f"\n  Long trades by fragility:")
        for q in [1, 2, 3, 4, 5]:
            qd = bt[(bt["fq"] == q) & (bt["direction"] == "long")]
            if len(qd) < 3:
                continue
            print(f"    Q{q}: N={len(qd)}, avg={qd['net_bps'].mean():+.1f} bps, "
                  f"WR={(qd['net_bps']>0).mean()*100:.0f}%")

        print(f"\n  Short trades by fragility:")
        for q in [1, 2, 3, 4, 5]:
            qd = bt[(bt["fq"] == q) & (bt["direction"] == "short")]
            if len(qd) < 3:
                continue
            print(f"    Q{q}: N={len(qd)}, avg={qd['net_bps'].mean():+.1f} bps, "
                  f"WR={(qd['net_bps']>0).mean()*100:.0f}%")

    # ── Phase 6: Compression analysis ──
    print(f"\n{'─'*70}")
    print("PHASE 6: Trade performance by compression state")
    print(f"{'─'*70}", flush=True)

    if len(baseline_trades) > 0 and comp_df is not None:
        bt = baseline_trades.copy()
        comp_states = []
        for _, row in bt.iterrows():
            ts = row["ts_entry"]
            mask = comp_df.index <= ts
            if mask.any():
                comp_states.append(comp_df.loc[mask, "compressed"].iloc[-1] > 0.5)
            else:
                comp_states.append(False)
        bt["comp"] = comp_states

        for state, label in [(True, "Compressed"), (False, "Not compressed")]:
            sub = bt[bt["comp"] == state]
            if len(sub) < 3:
                continue
            avg = sub["net_bps"].mean()
            wr = (sub["net_bps"] > 0).mean() * 100
            print(f"  {label:20s}: N={len(sub)}, avg={avg:+.1f} bps, WR={wr:.1f}%")

    # ── Save ──
    for variant, trades in all_trades.items():
        if len(trades) > 0:
            trades.to_csv(OUTPUT_DIR / f"xs9_trades_{variant}.csv", index=False)

    if all_results:
        pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "xs9_summary.csv", index=False)

    # ── Final verdict ──
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}", flush=True)

    if len(all_results) >= 2:
        base = all_results[0]
        best = max(all_results, key=lambda r: r["sharpe"])

        print(f"\n  Baseline: Sharpe={base['sharpe']:+.2f}, Ann={base['ann_return']:+.1f}%, DD={base['maxdd']:.1f}%")
        print(f"  Best:     Sharpe={best['sharpe']:+.2f}, Ann={best['ann_return']:+.1f}%, DD={best['maxdd']:.1f}% ({best['label']})")

        d_sharpe = best["sharpe"] - base["sharpe"]
        d_dd = best["maxdd"] - base["maxdd"]

        if d_sharpe > 0.2 and d_dd < 0:
            print(f"\n  → Overlay IMPROVES both return and risk ✅ (ΔSharpe={d_sharpe:+.2f}, ΔDD={d_dd:+.1f}%)")
        elif d_sharpe > 0.1:
            print(f"\n  → Overlay provides MARGINAL improvement ⚠️ (ΔSharpe={d_sharpe:+.2f})")
        elif d_dd < -0.5:
            print(f"\n  → Overlay reduces DRAWDOWN without hurting return ✅ (ΔDD={d_dd:+.1f}%)")
        else:
            print(f"\n  → Overlay provides NO meaningful improvement ❌")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
