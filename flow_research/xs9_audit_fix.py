#!/usr/bin/env python3
"""
XS-9 Audit Fix — Address all gaps from pre-production audit

Tests:
  1. VDB parameter stability on FULL datalake history (not just 8-month window)
     - Threshold sweep 1.5-3.0
     - Hold sweep 2-8h
     - Per-symbol and portfolio level
     - Monthly Sharpe, MaxDD, trade count

  2. PnL concentration on full history
     - Is top-5 = 106% of PnL a window artifact or structural?
     - t-test on avg trade
     - Bootstrap confidence interval on Sharpe

  3. Proper temporal split for fragility overlay
     - Train fragility on Jul-Oct 2025 ONLY
     - Freeze coefficients and quintile thresholds
     - Test on Nov 2025-Feb 2026 ONLY (true OOS)
     - Compare VDB alone vs VDB+fragility on OOS only

  4. Honest combined assessment
     - What's the real expected edge after costs?

Data: datalake/bybit 1m klines → 1h bars
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DATALAKE = Path(__file__).resolve().parent.parent / "datalake" / "bybit"
XS8_PATH = Path(__file__).resolve().parent / "output" / "xs8c" / "xs8c_extended.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs9_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["ONDOUSDT", "TAOUSDT", "SOLUSDT", "HBARUSDT", "SEIUSDT",
           "ADAUSDT", "BNBUSDT", "XRPUSDT", "AAVEUSDT"]

# Long-history symbols (have data since 2021-2022)
LONG_SYMBOLS = ["SOLUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT", "AAVEUSDT", "HBARUSDT"]

RT_FEE_BPS = 4.0
WARMUP_HOURS = 480  # 20 days


# ── Data loading ──

def load_1h_full(symbol):
    """Load ALL available 1m klines from datalake and resample to 1h."""
    sym_dir = DATALAKE / symbol
    if not sym_dir.exists():
        return pd.DataFrame()

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
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.sort_values("ts").drop_duplicates("ts").set_index("ts")

    ohlcv = raw.resample("1h").agg({"close": "last"}).dropna()
    return ohlcv


def compute_signal(ohlcv):
    """Compute VDB combined signal."""
    c = ohlcv["close"].values.astype(np.float64)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=ohlcv.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_z = (rvol - rvol.rolling(168, min_periods=48).mean()) / \
             rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)

    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    mr_4h = -((r4 - r4_mean) / r4_std)

    ohlcv["combined"] = (rvol_z.values + mr_4h.values) / 2
    return ohlcv


def simulate_trades(ohlcv, threshold, hold_bars, cooldown_bars, fee_bps):
    """Run walk-forward sim with 6-month warmup, return trade list."""
    sig = ohlcv["combined"].values
    c = ohlcv["close"].values.astype(np.float64)
    idx = ohlcv.index
    n = len(c)

    trades = []
    last_exit = 0

    for i in range(WARMUP_HOURS, n):
        if i < last_exit + cooldown_bars:
            continue
        if i + hold_bars >= n:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < threshold:
            continue

        d = "long" if sig[i] > 0 else "short"
        entry = c[i]
        exit_p = c[i + hold_bars]
        raw_bps = ((exit_p - entry) / entry * 10000) if d == "long" else \
                  ((entry - exit_p) / entry * 10000)
        net_bps = raw_bps - fee_bps

        trades.append({
            "ts": idx[i],
            "dir": d,
            "net_bps": net_bps,
        })
        last_exit = i + hold_bars

    return trades


def monthly_stats(trades):
    """Compute monthly PnL series and stats from trade list."""
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["month"] = df["ts"].dt.to_period("M")
    monthly = df.groupby("month")["net_bps"].sum() / 100  # to %
    return monthly


def portfolio_stats(sym_monthly_dict):
    """Combine per-symbol monthly PnL into portfolio stats."""
    # Equal weight: sum monthly PnL across symbols, divide by N symbols
    all_months = set()
    for m in sym_monthly_dict.values():
        if m is not None:
            all_months.update(m.index)
    if not all_months:
        return None

    all_months = sorted(all_months)
    n_syms = len(sym_monthly_dict)
    port_monthly = pd.Series(0.0, index=all_months)

    for sym, m in sym_monthly_dict.items():
        if m is None:
            continue
        for month in all_months:
            if month in m.index:
                port_monthly[month] += m[month] / n_syms

    total = port_monthly.sum()
    n_months = len(port_monthly)
    pos = (port_monthly > 0).sum()
    sharpe = port_monthly.mean() / max(port_monthly.std(), 0.001) * np.sqrt(12)
    cum = port_monthly.cumsum()
    maxdd = (cum.cummax() - cum).max()
    ann = total / max(n_months / 12, 0.5)

    return {
        "n_months": n_months,
        "total_pct": total,
        "ann": ann,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "pos_months": pos,
        "pos_pct": pos / n_months * 100,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 90)
    print("XS-9 AUDIT FIX — Addressing all pre-production gaps")
    print("=" * 90, flush=True)

    # ── Load data ──
    print(f"\n{'─'*70}")
    print("Loading full-history 1h data from datalake")
    print(f"{'─'*70}", flush=True)

    sym_data = {}
    for sym in SYMBOLS:
        ohlcv = load_1h_full(sym)
        if len(ohlcv) < 2000:
            print(f"  {sym}: TOO SHORT ({len(ohlcv)} bars)")
            continue
        ohlcv = compute_signal(ohlcv)
        sym_data[sym] = ohlcv
        print(f"  {sym}: {len(ohlcv):,} bars "
              f"({ohlcv.index[0].strftime('%Y-%m-%d')} → {ohlcv.index[-1].strftime('%Y-%m-%d')})",
              flush=True)

    # ══════════════════════════════════════════════════════════
    # TEST 1: PARAMETER STABILITY ON FULL HISTORY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TEST 1: PARAMETER STABILITY — FULL HISTORY")
    print(f"{'='*90}", flush=True)

    # 1a: Threshold sweep (hold=4, full portfolio)
    print(f"\n  ── Threshold sweep (hold=4h, cooldown=4h, fee=4bps) ──")
    print(f"  {'Thresh':>7s}  {'N':>5s}  {'Avg bps':>8s}  {'Med bps':>8s}  {'WR':>5s}  "
          f"{'Total%':>8s}  {'mSharpe':>8s}  {'MaxDD':>6s}  {'Pos Mo':>7s}  {'t-stat':>7s}")
    print(f"  {'-'*85}")

    for thresh in [1.0, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
        sym_monthly = {}
        all_trades = []
        for sym, ohlcv in sym_data.items():
            trades = simulate_trades(ohlcv, thresh, 4, 4, RT_FEE_BPS)
            all_trades.extend([t["net_bps"] for t in trades])
            sym_monthly[sym] = monthly_stats(trades)

        ps = portfolio_stats(sym_monthly)
        arr = np.array(all_trades) if all_trades else np.array([0])
        tstat = arr.mean() / (arr.std() / np.sqrt(len(arr))) if len(arr) > 1 and arr.std() > 0 else 0

        if ps:
            print(f"  {thresh:>7.1f}  {len(arr):>5d}  {arr.mean():>+7.1f}  {np.median(arr):>+7.1f}  "
                  f"{(arr>0).mean()*100:>4.1f}%  {ps['total_pct']:>+7.1f}%  {ps['sharpe']:>+7.2f}  "
                  f"{ps['maxdd']:>5.1f}%  {ps['pos_months']}/{ps['n_months']}  {tstat:>+6.2f}")

    # 1b: Hold sweep (threshold=2.0, full portfolio)
    print(f"\n  ── Hold sweep (threshold=2.0, cooldown=hold, fee=4bps) ──")
    print(f"  {'Hold':>7s}  {'N':>5s}  {'Avg bps':>8s}  {'Med bps':>8s}  {'WR':>5s}  "
          f"{'Total%':>8s}  {'mSharpe':>8s}  {'MaxDD':>6s}  {'t-stat':>7s}")
    print(f"  {'-'*75}")

    for hold in [2, 3, 4, 5, 6, 8, 12]:
        sym_monthly = {}
        all_trades = []
        for sym, ohlcv in sym_data.items():
            trades = simulate_trades(ohlcv, 2.0, hold, hold, RT_FEE_BPS)
            all_trades.extend([t["net_bps"] for t in trades])
            sym_monthly[sym] = monthly_stats(trades)

        ps = portfolio_stats(sym_monthly)
        arr = np.array(all_trades) if all_trades else np.array([0])
        tstat = arr.mean() / (arr.std() / np.sqrt(len(arr))) if len(arr) > 1 and arr.std() > 0 else 0

        if ps:
            print(f"  {hold:>7d}  {len(arr):>5d}  {arr.mean():>+7.1f}  {np.median(arr):>+7.1f}  "
                  f"{(arr>0).mean()*100:>4.1f}%  {ps['total_pct']:>+7.1f}%  {ps['sharpe']:>+7.2f}  "
                  f"{ps['maxdd']:>5.1f}%  {tstat:>+6.2f}")

    # 1c: Per-symbol stability at threshold=2.0, hold=4h
    print(f"\n  ── Per-symbol (threshold=2.0, hold=4h) ──")
    print(f"  {'Symbol':>12s}  {'Months':>7s}  {'N':>5s}  {'Avg bps':>8s}  {'WR':>5s}  "
          f"{'Total%':>8s}  {'mSharpe':>8s}  {'MaxDD':>6s}  {'t-stat':>7s}")
    print(f"  {'-'*80}")

    full_all_trades = []
    full_sym_monthly = {}
    for sym, ohlcv in sym_data.items():
        trades = simulate_trades(ohlcv, 2.0, 4, 4, RT_FEE_BPS)
        full_all_trades.extend(trades)
        m = monthly_stats(trades)
        full_sym_monthly[sym] = m
        arr = np.array([t["net_bps"] for t in trades])
        tstat = arr.mean() / (arr.std() / np.sqrt(len(arr))) if len(arr) > 1 and arr.std() > 0 else 0
        n_months = len(m) if m is not None else 0
        pos = (m > 0).sum() if m is not None else 0
        total = m.sum() if m is not None else 0
        sharpe = m.mean() / max(m.std(), 0.001) * np.sqrt(12) if m is not None and len(m) > 1 else 0
        maxdd_s = (m.cumsum().cummax() - m.cumsum()).max() if m is not None and len(m) > 1 else 0
        print(f"  {sym:>12s}  {pos}/{n_months:>2d}  {len(arr):>5d}  {arr.mean():>+7.1f}  "
              f"{(arr>0).mean()*100:>4.1f}%  {total:>+7.1f}%  {sharpe:>+7.2f}  "
              f"{maxdd_s:>5.1f}%  {tstat:>+6.2f}")

    # ══════════════════════════════════════════════════════════
    # TEST 2: PNL CONCENTRATION ON FULL HISTORY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TEST 2: PNL CONCENTRATION — FULL HISTORY")
    print(f"{'='*90}", flush=True)

    all_bps = np.array([t["net_bps"] for t in full_all_trades])
    total_bps = all_bps.sum()
    sorted_bps = np.sort(all_bps)[::-1]
    n = len(all_bps)

    print(f"\n  Total trades: {n}")
    print(f"  Total PnL: {total_bps:+.0f} bps ({total_bps/100:+.1f}%)")
    print(f"  Avg: {all_bps.mean():+.1f} bps, Median: {np.median(all_bps):+.1f} bps, Std: {all_bps.std():.1f} bps")

    tstat = all_bps.mean() / (all_bps.std() / np.sqrt(n))
    pval = 1 - stats.t.cdf(tstat, df=n-1)
    print(f"  t-stat: {tstat:+.3f}, p-value (one-sided): {pval:.4f}")

    print(f"\n  Trade concentration:")
    for topn in [1, 3, 5, 10, 20, 50]:
        if topn > n:
            continue
        top_sum = sorted_bps[:topn].sum()
        pct = top_sum / total_bps * 100 if total_bps != 0 else float("inf")
        print(f"    Top-{topn:3d}: {top_sum:+.0f} bps = {pct:.0f}% of total")

    print(f"\n  Without top-N trades:")
    for rm in [1, 3, 5, 10, 20]:
        if rm >= n:
            continue
        remaining = sorted_bps[rm:]
        r_mean = remaining.mean()
        r_total = remaining.sum()
        print(f"    Without top-{rm:3d}: total={r_total:+.0f} bps ({r_total/100:+.1f}%), avg={r_mean:+.1f} bps")

    # Bootstrap Sharpe CI
    print(f"\n  Bootstrap 95% CI for monthly Sharpe (1000 resamples):")
    ps_full = portfolio_stats(full_sym_monthly)
    if ps_full:
        # Bootstrap on trade-level returns
        rng = np.random.RandomState(42)
        boot_sharpes = []
        for _ in range(1000):
            idx = rng.choice(n, size=n, replace=True)
            boot_arr = all_bps[idx]
            boot_mean = boot_arr.mean()
            boot_std = boot_arr.std()
            if boot_std > 0:
                # approximate monthly sharpe from trade-level
                boot_sharpes.append(boot_mean / boot_std * np.sqrt(12))
        lo, hi = np.percentile(boot_sharpes, [2.5, 97.5])
        print(f"    Point estimate: {ps_full['sharpe']:+.2f}")
        print(f"    95% CI: [{lo:+.2f}, {hi:+.2f}]")
        if lo < 0:
            print(f"    ⚠️ CI includes zero — Sharpe is NOT significantly positive")
        else:
            print(f"    ✅ CI excludes zero — Sharpe IS significantly positive")

    # ══════════════════════════════════════════════════════════
    # TEST 3: YEARLY BREAKDOWN
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TEST 3: YEARLY PERFORMANCE BREAKDOWN")
    print(f"{'='*90}", flush=True)

    trades_df = pd.DataFrame(full_all_trades)
    trades_df["year"] = trades_df["ts"].dt.year
    trades_df["month"] = trades_df["ts"].dt.to_period("M")

    print(f"\n  {'Year':>6s}  {'N':>5s}  {'Avg bps':>8s}  {'WR':>5s}  {'Total%':>8s}  {'t-stat':>7s}")
    print(f"  {'-'*50}")
    for yr in sorted(trades_df["year"].unique()):
        ydf = trades_df[trades_df["year"] == yr]
        arr = ydf["net_bps"].values
        tstat_y = arr.mean() / (arr.std() / np.sqrt(len(arr))) if len(arr) > 1 and arr.std() > 0 else 0
        print(f"  {yr:>6d}  {len(arr):>5d}  {arr.mean():>+7.1f}  {(arr>0).mean()*100:>4.1f}%  "
              f"{arr.sum()/100:>+7.1f}%  {tstat_y:>+6.2f}")

    # ══════════════════════════════════════════════════════════
    # TEST 4: PROPER TEMPORAL SPLIT FOR FRAGILITY OVERLAY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TEST 4: HONEST OOS FRAGILITY OVERLAY")
    print("  Train: Jul-Oct 2025 (fit coefficients + quintile thresholds)")
    print("  Test:  Nov 2025-Feb 2026 (frozen — true OOS)")
    print(f"{'='*90}", flush=True)

    if not XS8_PATH.exists():
        print("  ⚠️ XS-8c parquet not found — skipping fragility test")
    else:
        frag_df = pd.read_parquet(XS8_PATH)
        frag_df["ts"] = pd.to_datetime(frag_df["ts"], utc=True)
        frag_df = frag_df.dropna(subset=["crowd_oi", "pca_var1"]).copy()

        # ── TRAIN: Jul-Oct 2025 ──
        train_mask = (frag_df["ts"] >= "2025-07-01") & (frag_df["ts"] < "2025-11-01")
        test_mask = (frag_df["ts"] >= "2025-11-01") & (frag_df["ts"] < "2026-03-01")

        train_frag = frag_df[train_mask].copy()
        test_frag = frag_df[test_mask].copy()

        print(f"\n  Train fragility: {len(train_frag):,} rows ({train_frag['ts'].min().date()} → {train_frag['ts'].max().date()})")
        print(f"  Test fragility:  {len(test_frag):,} rows ({test_frag['ts'].min().date()} → {test_frag['ts'].max().date()})")

        # Fit fragility score on train data
        # Use simple: frag_score = -crowd_oi (crowd_oi is the dominant feature)
        # Fit LogReg on train to get coefficients
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        FEATURE_COLS = ["crowd_oi", "pca_var1"]
        TARGET_COL = "tail_any_60m"

        if TARGET_COL in train_frag.columns:
            # Compute binary target
            train_frag["target"] = (train_frag[TARGET_COL] >= 0.10).astype(float)
            train_valid = train_frag.dropna(subset=FEATURE_COLS + ["target"])

            X_train = train_valid[FEATURE_COLS].values
            y_train = train_valid["target"].values

            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)

            coef_oi = lr.coef_[0][0]
            coef_pca = lr.coef_[0][1]
            print(f"\n  Train LogReg coefficients: crowd_oi={coef_oi:+.3f}, pca_var1={coef_pca:+.3f}")

            # Compute fragility score with TRAIN coefficients
            train_frag["frag_score"] = coef_oi * train_frag["crowd_oi"] + coef_pca * train_frag["pca_var1"]
            test_frag["frag_score"] = coef_oi * test_frag["crowd_oi"] + coef_pca * test_frag["pca_var1"]

            # Compute quintile thresholds on TRAIN data ONLY
            q_thresholds = train_frag["frag_score"].quantile([0.20, 0.40, 0.60, 0.80]).values
            print(f"  Train quintile thresholds: {q_thresholds}")

            # Apply FROZEN thresholds to test data
            def assign_quintile(val, thresholds):
                if val <= thresholds[0]: return 1
                elif val <= thresholds[1]: return 2
                elif val <= thresholds[2]: return 3
                elif val <= thresholds[3]: return 4
                else: return 5

            test_frag["frag_quintile"] = test_frag["frag_score"].apply(
                lambda x: assign_quintile(x, q_thresholds))

            test_frag_idx = test_frag.set_index("ts")

            # ── Run VDB on test period with and without overlay ──
            TEST_START = pd.Timestamp("2025-11-01", tz="UTC")
            TEST_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

            baseline_trades = []
            overlay_trades = []

            for sym, ohlcv in sym_data.items():
                test_ohlcv = ohlcv[(ohlcv.index >= TEST_START) & (ohlcv.index <= TEST_END)]
                if len(test_ohlcv) < 100:
                    continue

                sig = test_ohlcv["combined"].values
                c = test_ohlcv["close"].values.astype(np.float64)
                idx = test_ohlcv.index
                n_t = len(c)
                last_exit = 0

                for i in range(0, n_t):
                    if i < last_exit + 4:
                        continue
                    if i + 4 >= n_t:
                        continue
                    if np.isnan(sig[i]) or abs(sig[i]) < 2.0:
                        continue

                    d = "long" if sig[i] > 0 else "short"
                    entry = c[i]
                    exit_p = c[i + 4]
                    raw_bps = ((exit_p - entry) / entry * 10000) if d == "long" else \
                              ((entry - exit_p) / entry * 10000)
                    net_bps = raw_bps - RT_FEE_BPS

                    # Get fragility quintile (from frozen test labels)
                    ts_entry = idx[i]
                    fq = np.nan
                    size_mult = 1.0
                    fmask = test_frag_idx.index <= ts_entry
                    if fmask.any():
                        fq = test_frag_idx.loc[fmask, "frag_quintile"].iloc[-1]
                        if fq == 5:
                            size_mult = 0.50
                        elif fq == 4:
                            size_mult = 0.75

                    baseline_trades.append({
                        "sym": sym, "ts": ts_entry, "net_bps": net_bps,
                        "fq": fq, "dir": d,
                    })
                    overlay_trades.append({
                        "sym": sym, "ts": ts_entry,
                        "net_bps": net_bps,
                        "sized_bps": net_bps * size_mult,
                        "fq": fq, "size_mult": size_mult, "dir": d,
                    })
                    last_exit = i + 4

            bdf = pd.DataFrame(baseline_trades)
            odf = pd.DataFrame(overlay_trades)

            if len(bdf) > 0:
                bdf["month"] = bdf["ts"].dt.to_period("M")
                odf["month"] = odf["ts"].dt.to_period("M")

                # Baseline stats (OOS only)
                b_monthly = bdf.groupby("month")["net_bps"].sum() / 100
                b_sharpe = b_monthly.mean() / max(b_monthly.std(), 0.001) * np.sqrt(12)
                b_total = b_monthly.sum()
                b_dd = (b_monthly.cumsum().cummax() - b_monthly.cumsum()).max()

                # Overlay stats (OOS only)
                o_monthly = odf.groupby("month")["sized_bps"].sum() / 100
                o_sharpe = o_monthly.mean() / max(o_monthly.std(), 0.001) * np.sqrt(12)
                o_total = o_monthly.sum()
                o_dd = (o_monthly.cumsum().cummax() - o_monthly.cumsum()).max()

                print(f"\n  ── TRUE OOS RESULTS (Nov 2025-Feb 2026) ──")
                print(f"  {'Variant':30s}  {'N':>5s}  {'Avg bps':>8s}  {'Total%':>8s}  "
                      f"{'mSharpe':>8s}  {'MaxDD':>6s}")
                print(f"  {'-'*75}")
                print(f"  {'VDB baseline (OOS)':30s}  {len(bdf):>5d}  {bdf['net_bps'].mean():>+7.1f}  "
                      f"{b_total:>+7.1f}%  {b_sharpe:>+7.2f}  {b_dd:>5.1f}%")
                print(f"  {'VDB + fragility (OOS)':30s}  {len(odf):>5d}  {odf['sized_bps'].mean():>+7.1f}  "
                      f"{o_total:>+7.1f}%  {o_sharpe:>+7.2f}  {o_dd:>5.1f}%")

                delta_sharpe = o_sharpe - b_sharpe
                delta_dd = o_dd - b_dd
                print(f"\n  ΔSharpe = {delta_sharpe:+.2f}, ΔMaxDD = {delta_dd:+.1f}%")

                if delta_sharpe > 0.1 and delta_dd < 0:
                    print(f"  → Fragility overlay HELPS on OOS ✅")
                elif delta_sharpe > 0:
                    print(f"  → Fragility overlay marginal on OOS ⚠️")
                else:
                    print(f"  → Fragility overlay DOES NOT HELP on OOS ❌")

                # Quintile breakdown on OOS
                print(f"\n  ── OOS trade performance by fragility quintile ──")
                print(f"  {'Q':>4s}  {'N':>5s}  {'Avg bps':>8s}  {'WR':>5s}")
                for q in [1, 2, 3, 4, 5]:
                    qd = bdf[bdf["fq"] == q]
                    if len(qd) < 2:
                        continue
                    print(f"  Q{q:>3d}  {len(qd):>5d}  {qd['net_bps'].mean():>+7.1f}  "
                          f"{(qd['net_bps']>0).mean()*100:>4.1f}%")

                # Monthly detail
                print(f"\n  ── OOS monthly detail ──")
                print(f"  {'Month':>10s}  {'Base%':>8s}  {'Overlay%':>9s}  {'Δ':>7s}  {'N':>4s}")
                for m in sorted(b_monthly.index):
                    bv = b_monthly.get(m, 0)
                    ov = o_monthly.get(m, 0)
                    n_m = len(bdf[bdf["month"] == m])
                    print(f"  {str(m):>10s}  {bv:>+7.2f}%  {ov:>+8.2f}%  {ov-bv:>+6.2f}%  {n_m:>4d}")

    # ══════════════════════════════════════════════════════════
    # TEST 5: FEE SENSITIVITY ON FULL HISTORY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TEST 5: FEE SENSITIVITY — FULL HISTORY")
    print(f"{'='*90}", flush=True)

    print(f"\n  {'Fee (bps)':>10s}  {'Scenario':20s}  {'N':>5s}  {'Avg bps':>8s}  {'WR':>5s}  "
          f"{'Total%':>8s}  {'mSharpe':>8s}  {'t-stat':>7s}")
    print(f"  {'-'*80}")

    for fee, label in [(0, "Zero (gross)"), (4, "Maker+maker"), (8, "Maker+taker"),
                        (12, "Taker+maker+slip"), (16, "Taker+taker+slip"), (20, "Taker+taker")]:
        sym_monthly = {}
        all_trades = []
        for sym, ohlcv in sym_data.items():
            trades = simulate_trades(ohlcv, 2.0, 4, 4, fee)
            all_trades.extend([t["net_bps"] for t in trades])
            sym_monthly[sym] = monthly_stats(trades)

        ps = portfolio_stats(sym_monthly)
        arr = np.array(all_trades) if all_trades else np.array([0])
        tstat = arr.mean() / (arr.std() / np.sqrt(len(arr))) if len(arr) > 1 and arr.std() > 0 else 0

        if ps:
            print(f"  {fee:>10d}  {label:20s}  {len(arr):>5d}  {arr.mean():>+7.1f}  "
                  f"{(arr>0).mean()*100:>4.1f}%  {ps['total_pct']:>+7.1f}%  "
                  f"{ps['sharpe']:>+7.2f}  {tstat:>+6.2f}")

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("FINAL VERDICT")
    print(f"{'='*90}", flush=True)

    # Full history stats
    arr_full = np.array([t["net_bps"] for t in full_all_trades])
    tstat_full = arr_full.mean() / (arr_full.std() / np.sqrt(len(arr_full)))
    pval_full = 1 - stats.t.cdf(tstat_full, df=len(arr_full)-1)

    print(f"\n  VDB (thresh=2.0, hold=4h) on full history:")
    print(f"    Trades: {len(arr_full)}")
    print(f"    Avg: {arr_full.mean():+.1f} bps, Median: {np.median(arr_full):+.1f} bps")
    print(f"    t-stat: {tstat_full:+.3f}, p-value: {pval_full:.4f}")
    if pval_full < 0.05:
        print(f"    ✅ Statistically significant at p<0.05")
    elif pval_full < 0.10:
        print(f"    ⚠️ Marginal significance (p<0.10)")
    else:
        print(f"    ❌ NOT statistically significant")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
