#!/usr/bin/env python3
"""
Cross-Sectional Relative Edge Research (50 coins)

Tests whether regime conditions produce excess returns (vs market beta)
that are unexploitable in absolute terms but visible in relative terms.

Phases:
  1) Load 1m bars, OI 5m, FR for all symbols → unified 5m panel
  2) Build market proxies (EW, VW), rolling beta
  3) Compute excess returns at 15/30/60m horizons
  4) Engineer features + cross-sectional ranks, define regimes R1/R2/R3
  5) Conditional tests on excess returns per regime
  6) Cross-sectional portfolio test (long top-K / short bottom-K)
  7) Statistical validation (bootstrap CI, permutation, FDR)

Output:
  output/xs/xs_dataset.parquet
  output/xs/xs_coin_regime_report.csv
  output/xs/xs_portfolio_report.csv
  output/xs/xs_weekly_stability.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_TS = pd.Timestamp("2026-01-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# Minimum days of data to include a symbol
MIN_DAYS = 50

# Beta window (minutes of 1m data)
W_BETA = 3 * 24 * 60  # 3 days = 4320 minutes
BETA_CLIP = 3.0

# Horizons (minutes)
HORIZONS = [15, 30, 60]

# Regime thresholds
R1_OI_RANK = 0.90
R1_FUND_Z = 1.0
R1_VOL_RANK = 0.70

R2_FUND_Z = 2.0

R3_DISP_PCTL = 0.80
R3_VOL_RANK = 0.80

# Portfolio
PORT_K_VALUES = [5, 10]

# Stats
N_BOOTSTRAP = 2000
N_PERMUTATION = 1000
SEED = 42


# ---------------------------------------------------------------------------
# §1: Data loading
# ---------------------------------------------------------------------------


def discover_symbols() -> list[str]:
    """Find symbols with enough data (mark 1m + OI + FR)."""
    syms = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        nmark = len(list(d.glob("*_mark_price_kline_1m.csv")))
        noi = len(list(d.glob("*_open_interest_5min.csv")))
        nfr = len(list(d.glob("*_funding_rate.csv")))
        if nmark >= MIN_DAYS and noi >= MIN_DAYS and nfr >= MIN_DAYS:
            syms.append(d.name)
    return syms


def load_mark_1m(symbol: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_mark_price_kline_1m.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)]
    return df[["ts", "open", "high", "low", "close"]].reset_index(drop=True)


def load_kline_1m(symbol: str) -> pd.DataFrame:
    """Load kline 1m for volume/turnover."""
    sym_dir = DATA_DIR / symbol
    files = sorted(
        f for f in sym_dir.glob("*_kline_1m.csv")
        if "mark_price" not in f.name and "premium_index" not in f.name
    )
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
    df["turnover"] = df["turnover"].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)]
    return df[["ts", "turnover"]].reset_index(drop=True)


def load_oi_5m(symbol: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_open_interest_5min.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"] = df["openInterest"].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)]
    return df[["ts", "oi"]].reset_index(drop=True)


def load_funding(symbol: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_funding_rate.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["fr"] = df["fundingRate"].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)]
    return df[["ts", "fr"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# §2-3: Build unified 5m panel with market proxies and betas
# ---------------------------------------------------------------------------


def build_5m_panel(symbols: list[str]) -> pd.DataFrame:
    """Build the main 5m panel: one row per (symbol, ts_5m)."""
    t0 = time.monotonic()

    # Step 1: Load all 1m data per symbol, compute 1m log returns
    sym_1m = {}  # symbol -> DataFrame with ts, close, log_ret, turnover
    for i, sym in enumerate(symbols, 1):
        mark = load_mark_1m(sym)
        kline = load_kline_1m(sym)
        if len(mark) < 1000:
            continue

        df = mark[["ts", "close"]].copy()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        if len(kline) > 0:
            kline = kline.set_index("ts")
            df = df.set_index("ts")
            df["turnover"] = kline["turnover"].reindex(df.index).fillna(0)
            df = df.reset_index()
        else:
            df["turnover"] = 0.0

        sym_1m[sym] = df
        if i % 10 == 0:
            print(f"  Loaded 1m data: {i}/{len(symbols)}")

    valid_symbols = sorted(sym_1m.keys())
    print(f"  Valid symbols with 1m data: {len(valid_symbols)}")

    # Step 2: Build common 5m grid
    all_ts = set()
    for df in sym_1m.values():
        all_ts.update(df["ts"].values)
    ts_grid_1m = np.sort(np.array(list(all_ts)))

    # 5m grid: every 5th minute
    ts_5m_set = set()
    for t in ts_grid_1m:
        t_pd = pd.Timestamp(t, tz="UTC")
        if t_pd.minute % 5 == 0:
            ts_5m_set.add(t)
    ts_5m = np.sort(np.array(list(ts_5m_set)))
    print(f"  5m grid points: {len(ts_5m):,}")

    # Step 3: Compute 1m market returns (EW and VW)
    # For each 1m timestamp, compute cross-sectional EW and VW return
    # Build a wide DataFrame of 1m returns
    ret_wide = pd.DataFrame(index=pd.DatetimeIndex(ts_grid_1m, tz="UTC"))
    ret_wide.index.name = "ts"
    turn_wide = pd.DataFrame(index=ret_wide.index)

    for sym, df in sym_1m.items():
        s = df.set_index("ts")
        ret_wide[sym] = s["log_ret"].reindex(ret_wide.index)
        turn_wide[sym] = s["turnover"].reindex(ret_wide.index).fillna(0)

    # EW market return: mean of available returns
    ret_mkt_ew_1m = ret_wide.mean(axis=1)

    # VW market return: turnover-weighted
    turn_60m = turn_wide.rolling(60, min_periods=10).sum()
    weights = turn_60m.div(turn_60m.sum(axis=1), axis=0)
    ret_mkt_vw_1m = (ret_wide * weights).sum(axis=1)
    # Fill NaN VW with EW
    ret_mkt_vw_1m = ret_mkt_vw_1m.fillna(ret_mkt_ew_1m)

    # Cumulative market log price for computing H-bar returns
    mkt_cum_ew = ret_mkt_ew_1m.fillna(0).cumsum()
    mkt_cum_vw = ret_mkt_vw_1m.fillna(0).cumsum()

    print(f"  Market index computed. EW/VW 1m returns: {len(ret_mkt_ew_1m):,}")

    # Step 4: For each symbol, build 5m rows
    # Pre-compute cumulative log price per symbol for fast H-bar returns
    cum_log = {}
    for sym, df in sym_1m.items():
        s = df.set_index("ts")["close"]
        s = np.log(s)
        cum_log[sym] = s

    # Build panel rows
    panel_rows = []
    ts_5m_idx = pd.DatetimeIndex(ts_5m, tz="UTC")

    # Pre-index 1m data for fast lookup
    mkt_ew_arr = ret_mkt_ew_1m.reindex(pd.DatetimeIndex(ts_grid_1m, tz="UTC"))
    mkt_vw_arr = ret_mkt_vw_1m.reindex(pd.DatetimeIndex(ts_grid_1m, tz="UTC"))
    mkt_cum_ew_s = mkt_cum_ew.reindex(pd.DatetimeIndex(ts_grid_1m, tz="UTC"))
    mkt_cum_vw_s = mkt_cum_vw.reindex(pd.DatetimeIndex(ts_grid_1m, tz="UTC"))

    # Load OI and FR per symbol
    sym_oi = {}
    sym_fr = {}
    for sym in valid_symbols:
        oi = load_oi_5m(sym)
        if len(oi) > 0:
            sym_oi[sym] = oi.set_index("ts")["oi"]
        fr = load_funding(sym)
        if len(fr) > 0:
            sym_fr[sym] = fr.set_index("ts")["fr"]

    print(f"  OI loaded for {len(sym_oi)} symbols, FR for {len(sym_fr)} symbols")
    print(f"  Building 5m panel rows...")

    # For beta: we need rolling cov/var on 1m returns
    # Pre-compute per-symbol 1m return series aligned to common index
    sym_ret_1m = {}
    for sym, df in sym_1m.items():
        s = df.set_index("ts")["log_ret"]
        sym_ret_1m[sym] = s.reindex(pd.DatetimeIndex(ts_grid_1m, tz="UTC"))

    elapsed = time.monotonic() - t0
    print(f"  Data prep done in {elapsed:.1f}s. Computing per-symbol features...")

    # Process each symbol
    sym_count = 0
    for sym in valid_symbols:
        sym_count += 1
        if sym_count % 10 == 0:
            print(f"  Processing symbol {sym_count}/{len(valid_symbols)}: {sym}")

        s_ret = sym_ret_1m[sym]
        s_cum = cum_log.get(sym)
        if s_cum is None or len(s_cum) < W_BETA:
            continue

        # Rolling beta (W_BETA window on 1m)
        # cov(ret_i, ret_mkt) / var(ret_mkt) over [t-W, t)
        r_sym = s_ret.values
        r_mkt_ew = mkt_ew_arr.values
        r_mkt_vw = mkt_vw_arr.values

        # Use pandas rolling for efficiency
        df_beta = pd.DataFrame({
            "r": s_ret,
            "m_ew": mkt_ew_arr,
            "m_vw": mkt_vw_arr,
        })
        # Only compute at 5m points for speed
        # First compute rolling stats on full 1m series, then sample at 5m
        rm_prod_ew = (df_beta["r"] * df_beta["m_ew"])
        rm_prod_vw = (df_beta["r"] * df_beta["m_vw"])

        roll_cov_ew = rm_prod_ew.rolling(W_BETA, min_periods=W_BETA // 2).mean() - \
                      df_beta["r"].rolling(W_BETA, min_periods=W_BETA // 2).mean() * \
                      df_beta["m_ew"].rolling(W_BETA, min_periods=W_BETA // 2).mean()
        roll_var_ew = df_beta["m_ew"].rolling(W_BETA, min_periods=W_BETA // 2).var()

        roll_cov_vw = rm_prod_vw.rolling(W_BETA, min_periods=W_BETA // 2).mean() - \
                      df_beta["r"].rolling(W_BETA, min_periods=W_BETA // 2).mean() * \
                      df_beta["m_vw"].rolling(W_BETA, min_periods=W_BETA // 2).mean()
        roll_var_vw = df_beta["m_vw"].rolling(W_BETA, min_periods=W_BETA // 2).var()

        beta_ew = (roll_cov_ew / roll_var_ew).clip(-BETA_CLIP, BETA_CLIP)
        beta_vw = (roll_cov_vw / roll_var_vw).clip(-BETA_CLIP, BETA_CLIP)

        # OI features
        oi_s = sym_oi.get(sym)
        # FR features
        fr_s = sym_fr.get(sym)

        # Per-symbol 1m features (will sample at 5m)
        # rv_past_60: realized vol over past 60 1m bars
        rv_60 = s_ret.rolling(60, min_periods=30).std() * np.sqrt(60)
        # ret_past_60: cumulative return over past 60m
        ret_60 = s_ret.rolling(60, min_periods=30).sum()
        # turnover_60m
        turn_s = turn_wide[sym] if sym in turn_wide.columns else pd.Series(0, index=mkt_ew_arr.index)
        turn_60 = turn_s.rolling(60, min_periods=30).sum()

        # Sample at 5m grid
        for t5 in ts_5m_idx:
            if t5 not in s_cum.index:
                continue

            # Current price
            try:
                price_t = np.exp(s_cum.loc[t5])
            except (KeyError, ValueError):
                continue

            # Beta
            b_ew = beta_ew.get(t5, np.nan) if t5 in beta_ew.index else np.nan
            b_vw = beta_vw.get(t5, np.nan) if t5 in beta_vw.index else np.nan
            if pd.isna(b_ew):
                continue

            # Forward returns at each horizon
            row = {
                "symbol": sym,
                "ts": t5,
                "price_t": price_t,
                "beta_ew": b_ew,
                "beta_vw": b_vw if not pd.isna(b_vw) else b_ew,
            }

            # Forward absolute returns
            for H in HORIZONS:
                t_fwd = t5 + pd.Timedelta(minutes=H)
                if t_fwd in s_cum.index and t_fwd in mkt_cum_ew_s.index:
                    ret_i = (s_cum.loc[t_fwd] - s_cum.loc[t5]) * 10000  # bp
                    ret_m_ew = (mkt_cum_ew_s.loc[t_fwd] - mkt_cum_ew_s.loc[t5]) * 10000
                    ret_m_vw = (mkt_cum_vw_s.loc[t_fwd] - mkt_cum_vw_s.loc[t5]) * 10000

                    excess_ew = ret_i - b_ew * ret_m_ew
                    excess_vw = ret_i - row["beta_vw"] * ret_m_vw

                    row[f"ret_{H}"] = ret_i
                    row[f"ret_mkt_ew_{H}"] = ret_m_ew
                    row[f"ret_mkt_vw_{H}"] = ret_m_vw
                    row[f"excess_ew_{H}"] = excess_ew
                    row[f"excess_vw_{H}"] = excess_vw
                else:
                    row[f"ret_{H}"] = np.nan
                    row[f"ret_mkt_ew_{H}"] = np.nan
                    row[f"ret_mkt_vw_{H}"] = np.nan
                    row[f"excess_ew_{H}"] = np.nan
                    row[f"excess_vw_{H}"] = np.nan

            # Per-coin features
            row["rv_60"] = rv_60.get(t5, np.nan) if t5 in rv_60.index else np.nan
            row["ret_past_60"] = ret_60.get(t5, np.nan) if t5 in ret_60.index else np.nan
            row["turnover_60"] = turn_60.get(t5, np.nan) if t5 in turn_60.index else np.nan

            # OI
            if oi_s is not None and t5 in oi_s.index:
                oi_now = oi_s.loc[t5]
                # Find OI 60m ago (12 5m bars)
                t_back = t5 - pd.Timedelta(minutes=60)
                oi_back = oi_s.get(t_back, np.nan) if t_back in oi_s.index else np.nan
                if not pd.isna(oi_back) and oi_back > 0:
                    row["oi_chg_60"] = (oi_now - oi_back) / oi_back
                else:
                    row["oi_chg_60"] = np.nan
                row["oi_val"] = oi_now
            else:
                row["oi_chg_60"] = np.nan
                row["oi_val"] = np.nan

            # FR
            if fr_s is not None:
                # Find latest FR <= t5
                fr_before = fr_s[fr_s.index <= t5]
                if len(fr_before) > 0:
                    row["fr_val"] = fr_before.iloc[-1]
                else:
                    row["fr_val"] = np.nan
            else:
                row["fr_val"] = np.nan

            panel_rows.append(row)

    panel = pd.DataFrame(panel_rows)
    elapsed = time.monotonic() - t0
    print(f"  Panel built: {len(panel):,} rows, {panel['symbol'].nunique()} symbols in {elapsed:.1f}s")
    return panel


# ---------------------------------------------------------------------------
# §6: Cross-sectional features + regimes
# ---------------------------------------------------------------------------


def add_xs_features_and_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional ranks, z-scores, dispersion, and regime flags."""
    print("  Computing cross-sectional features...")

    # Z-scores per symbol (rolling)
    # funding_z: z-score of FR across recent history
    for col, zcol in [("fr_val", "funding_z"), ("rv_60", "vol_z")]:
        panel[zcol] = panel.groupby("symbol")[col].transform(
            lambda x: (x - x.rolling(288, min_periods=50).mean()) /
                       x.rolling(288, min_periods=50).std().clip(lower=1e-12)
        )

    # Cross-sectional ranks at each timestamp
    for col, rankcol in [
        ("oi_chg_60", "rank_oi_chg_60"),
        ("funding_z", "rank_funding_z"),
        ("rv_60", "rank_rv_60"),
        ("vol_z", "rank_vol_z"),
    ]:
        panel[rankcol] = panel.groupby("ts")[col].rank(pct=True)

    # Dispersion: stdev of ret_past_60 across coins at each t
    disp = panel.groupby("ts")["ret_past_60"].std().rename("dispersion_60")
    panel = panel.merge(disp, on="ts", how="left")

    # Dispersion percentile (rolling over time, across all timestamps)
    disp_ts = panel.drop_duplicates("ts")[["ts", "dispersion_60"]].sort_values("ts")
    disp_ts["disp_pctl"] = disp_ts["dispersion_60"].rank(pct=True)
    panel = panel.merge(disp_ts[["ts", "disp_pctl"]], on="ts", how="left")

    # Trend strength
    panel["trend_strength"] = panel["ret_past_60"].abs()

    # Rolling correlation to market (using ret_past_60 as proxy — simplified)
    # True corr would need 1m series; we approximate with rank correlation at 5m
    panel["corr_to_mkt_approx"] = panel.groupby("ts")["ret_past_60"].rank(pct=True)

    # §7: Regime definitions
    # R1: Crowd build-up
    panel["R1"] = (
        (panel["rank_oi_chg_60"] >= R1_OI_RANK) &
        (panel["funding_z"].abs() >= R1_FUND_Z) &
        (panel["rank_vol_z"] >= R1_VOL_RANK)
    ).astype(int)

    # R2: Funding divergence
    panel["R2"] = (panel["funding_z"].abs() >= R2_FUND_Z).astype(int)

    # R3: Market dispersion regime
    panel["R3"] = (
        (panel["disp_pctl"] >= R3_DISP_PCTL) &
        (panel["rank_vol_z"] >= R3_VOL_RANK)
    ).astype(int)

    # Quality flag
    panel["is_valid"] = panel[[f"excess_ew_{H}" for H in HORIZONS]].notna().all(axis=1)

    print(f"  Regimes: R1={panel['R1'].sum():,} ({panel['R1'].mean():.1%}), "
          f"R2={panel['R2'].sum():,} ({panel['R2'].mean():.1%}), "
          f"R3={panel['R3'].sum():,} ({panel['R3'].mean():.1%})")

    return panel


# ---------------------------------------------------------------------------
# §8: Conditional tests
# ---------------------------------------------------------------------------


def conditional_tests(panel: pd.DataFrame) -> pd.DataFrame:
    """Test excess returns under each regime vs unconditional."""
    valid = panel[panel["is_valid"]].copy()
    rows = []

    for regime in ["R1", "R2", "R3"]:
        for sym in sorted(valid["symbol"].unique()):
            sym_data = valid[valid["symbol"] == sym]
            regime_data = sym_data[sym_data[regime] == 1]
            base_data = sym_data[sym_data[regime] == 0]
            n_r = len(regime_data)
            n_b = len(base_data)
            if n_r < 5:
                continue

            for H in HORIZONS:
                col = f"excess_ew_{H}"
                r_vals = regime_data[col].dropna().values
                b_vals = base_data[col].dropna().values
                if len(r_vals) < 5:
                    continue

                med_r = np.median(r_vals)
                mean_r = np.mean(r_vals)
                wr_r = (r_vals > 0).mean()
                med_b = np.median(b_vals) if len(b_vals) > 0 else 0
                mean_b = np.mean(b_vals) if len(b_vals) > 0 else 0

                # Bootstrap CI for median
                rng = np.random.default_rng(SEED)
                boot_meds = [np.median(rng.choice(r_vals, len(r_vals), replace=True))
                             for _ in range(N_BOOTSTRAP)]
                ci_lo = np.percentile(boot_meds, 5)
                ci_hi = np.percentile(boot_meds, 95)

                # Permutation test: is regime median different from baseline?
                if len(b_vals) >= 5:
                    combined = np.concatenate([r_vals, b_vals])
                    obs_diff = np.abs(med_r - med_b)
                    count = 0
                    for _ in range(N_PERMUTATION):
                        perm = rng.permutation(combined)
                        d = np.abs(np.median(perm[:len(r_vals)]) - np.median(perm[len(r_vals):]))
                        if d >= obs_diff:
                            count += 1
                    p_perm = count / N_PERMUTATION
                else:
                    p_perm = 1.0

                rows.append({
                    "symbol": sym,
                    "regime": regime,
                    "horizon": H,
                    "n_regime": len(r_vals),
                    "n_base": len(b_vals),
                    "regime_rate": len(r_vals) / (len(r_vals) + len(b_vals)),
                    "median_excess_regime": med_r,
                    "mean_excess_regime": mean_r,
                    "median_excess_base": med_b,
                    "wr_regime": wr_r,
                    "wr_base": (b_vals > 0).mean() if len(b_vals) > 0 else 0.5,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_perm": p_perm,
                    "p5": np.percentile(r_vals, 5),
                    "p95": np.percentile(r_vals, 95),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # FDR correction
    pvals = df["p_perm"].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    q = np.empty(n)
    q[sorted_idx] = pvals[sorted_idx] * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    df["q_fdr"] = np.clip(q, 0, 1)

    return df


# ---------------------------------------------------------------------------
# §8.3: Cross-sectional portfolio test
# ---------------------------------------------------------------------------


def portfolio_test(panel: pd.DataFrame) -> pd.DataFrame:
    """At each 5m point, form long top-K / short bottom-K portfolio by regime signal."""
    valid = panel[panel["is_valid"]].copy()
    rows = []

    rng = np.random.default_rng(SEED)

    for regime in ["R1", "R2", "R3"]:
        # For R1/R2: rank by the primary signal (oi_chg_60 for R1, funding_z for R2)
        # For R3: rank by vol_z
        if regime == "R1":
            rank_col = "rank_oi_chg_60"
        elif regime == "R2":
            rank_col = "rank_funding_z"
        else:
            rank_col = "rank_vol_z"

        for H in HORIZONS:
            ret_col = f"excess_ew_{H}"

            for K in PORT_K_VALUES:
                port_rets = []
                port_ts = []

                for t5, grp in valid.groupby("ts"):
                    if grp[regime].sum() < K:
                        continue
                    regime_grp = grp[grp[regime] == 1].dropna(subset=[ret_col, rank_col])
                    if len(regime_grp) < 2 * K:
                        continue

                    # Long top-K by rank, short bottom-K
                    sorted_grp = regime_grp.sort_values(rank_col, ascending=False)
                    top_k = sorted_grp.head(K)
                    bot_k = sorted_grp.tail(K)

                    long_ret = top_k[ret_col].mean()
                    short_ret = bot_k[ret_col].mean()
                    port_ret = (long_ret - short_ret) / 2  # dollar-neutral

                    port_rets.append(port_ret)
                    port_ts.append(t5)

                if len(port_rets) < 20:
                    continue

                pr = np.array(port_rets)

                # Bootstrap CI
                boot_means = [np.mean(rng.choice(pr, len(pr), replace=True))
                              for _ in range(N_BOOTSTRAP)]
                ci_lo = np.percentile(boot_means, 5)
                ci_hi = np.percentile(boot_means, 95)

                # t-test
                t_stat, p_val = sp_stats.ttest_1samp(pr, 0)

                # Weekly stability
                ts_arr = pd.DatetimeIndex(port_ts)
                weeks = ts_arr.isocalendar().week.values
                unique_weeks = np.unique(weeks)
                week_means = [pr[weeks == w].mean() for w in unique_weeks]
                weeks_pos = sum(1 for m in week_means if m > 0)

                rows.append({
                    "regime": regime,
                    "horizon": H,
                    "K": K,
                    "n_rebalances": len(pr),
                    "mean_ret_bp": pr.mean(),
                    "median_ret_bp": np.median(pr),
                    "std_ret_bp": pr.std(),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "sharpe_ann": pr.mean() / pr.std() * np.sqrt(252 * 288 / (H / 5)) if pr.std() > 0 else 0,
                    "weeks_pos": weeks_pos,
                    "weeks_total": len(unique_weeks),
                    "weeks_pos_rate": weeks_pos / len(unique_weeks) if len(unique_weeks) > 0 else 0,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §11: Weekly stability
# ---------------------------------------------------------------------------


def weekly_stability(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel[panel["is_valid"]].copy()
    valid["week"] = valid["ts"].dt.isocalendar().week.astype(int)
    rows = []

    for regime in ["R1", "R2", "R3"]:
        for H in HORIZONS:
            col = f"excess_ew_{H}"
            regime_data = valid[valid[regime] == 1]
            if len(regime_data) == 0:
                continue
            for week, wgrp in regime_data.groupby("week"):
                vals = wgrp[col].dropna()
                if len(vals) == 0:
                    continue
                rows.append({
                    "regime": regime,
                    "horizon": H,
                    "week": week,
                    "n": len(vals),
                    "mean_excess": vals.mean(),
                    "median_excess": vals.median(),
                    "wr": (vals > 0).mean(),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("CROSS-SECTIONAL RELATIVE EDGE RESEARCH")
    print(f"Period: {START_TS.date()} → {END_TS.date()}")
    print("=" * 80)

    # Discover symbols
    symbols = discover_symbols()
    print(f"\nSymbols with sufficient data: {len(symbols)}")

    # Build panel
    print(f"\n{'─'*70}")
    print("PHASE 1: Building 5m panel")
    print(f"{'─'*70}")
    panel = build_5m_panel(symbols)

    if len(panel) == 0:
        print("ERROR: Empty panel. Aborting.")
        return

    # Add XS features and regimes
    print(f"\n{'─'*70}")
    print("PHASE 2: Cross-sectional features + regimes")
    print(f"{'─'*70}")
    panel = add_xs_features_and_regimes(panel)

    # Save dataset
    panel.to_parquet(OUTPUT_DIR / "xs_dataset.parquet", index=False)
    print(f"  Saved xs_dataset.parquet ({len(panel):,} rows)")

    # Baseline sanity check
    print(f"\n{'─'*70}")
    print("BASELINE CHECK (unconditional excess ~ 0?)")
    print(f"{'─'*70}")
    valid = panel[panel["is_valid"]]
    for H in HORIZONS:
        col = f"excess_ew_{H}"
        vals = valid[col].dropna()
        print(f"  H={H:>2}m: mean={vals.mean():+.2f}bp  med={vals.median():+.2f}bp  "
              f"std={vals.std():.1f}bp  WR={(vals>0).mean():.1%}  N={len(vals):,}")

    # Conditional tests
    print(f"\n{'─'*70}")
    print("PHASE 3: Conditional tests (regime vs baseline)")
    print(f"{'─'*70}")
    cond_df = conditional_tests(panel)

    if len(cond_df) > 0:
        cond_df.to_csv(OUTPUT_DIR / "xs_coin_regime_report.csv", index=False)
        print(f"  Saved xs_coin_regime_report.csv ({len(cond_df)} rows)")

        # Print top results
        sig = cond_df[(cond_df["q_fdr"] < 0.20) & (cond_df["n_regime"] >= 10)]
        print(f"\n  Results with q_fdr < 0.20 and N≥10: {len(sig)}")
        if len(sig) > 0:
            sig = sig.sort_values("q_fdr")
            print(f"\n  {'Sym':<16} {'Reg':>3} {'H':>3} | {'N':>4} {'Rate':>5} | "
                  f"{'MedExc':>7} {'MeanExc':>8} {'WR':>5} | {'CI':>14} {'q':>6}")
            for _, r in sig.head(30).iterrows():
                print(f"  {r['symbol']:<16} {r['regime']:>3} {r['horizon']:>3} | "
                      f"{r['n_regime']:>4.0f} {r['regime_rate']:>5.1%} | "
                      f"{r['median_excess_regime']:>+7.1f} {r['mean_excess_regime']:>+8.1f} "
                      f"{r['wr_regime']:>5.1%} | "
                      f"[{r['ci_lo']:>+.1f},{r['ci_hi']:>+.1f}] {r['q_fdr']:>6.3f}")

        # Summary by regime
        print(f"\n  Summary by regime (mean across symbols, H=60m):")
        for regime in ["R1", "R2", "R3"]:
            sub = cond_df[(cond_df["regime"] == regime) & (cond_df["horizon"] == 60)]
            if len(sub) == 0:
                print(f"    {regime}: no data")
                continue
            print(f"    {regime}: N_symbols={len(sub)}, "
                  f"avg_med_excess={sub['median_excess_regime'].mean():+.1f}bp, "
                  f"avg_WR={sub['wr_regime'].mean():.1%}, "
                  f"sig(q<0.10)={sum(sub['q_fdr'] < 0.10)}")
    else:
        print("  No conditional test results.")

    # Portfolio test
    print(f"\n{'─'*70}")
    print("PHASE 4: Cross-sectional portfolio test")
    print(f"{'─'*70}")
    port_df = portfolio_test(panel)

    if len(port_df) > 0:
        port_df.to_csv(OUTPUT_DIR / "xs_portfolio_report.csv", index=False)
        print(f"  Saved xs_portfolio_report.csv ({len(port_df)} rows)")

        print(f"\n  {'Reg':>3} {'H':>3} {'K':>3} | {'N_reb':>6} {'Mean':>7} {'Med':>7} "
              f"{'Std':>6} | {'CI':>14} {'t':>6} {'p':>6} | {'Wk+':>4}/{'':<4} {'Sh_ann':>7}")
        for _, r in port_df.sort_values(["regime", "horizon", "K"]).iterrows():
            sig_str = ""
            if r["p_value"] < 0.01: sig_str = "***"
            elif r["p_value"] < 0.05: sig_str = "**"
            elif r["p_value"] < 0.10: sig_str = "*"
            print(f"  {r['regime']:>3} {r['horizon']:>3} {r['K']:>3} | "
                  f"{r['n_rebalances']:>6.0f} {r['mean_ret_bp']:>+7.2f} {r['median_ret_bp']:>+7.2f} "
                  f"{r['std_ret_bp']:>6.1f} | "
                  f"[{r['ci_lo']:>+.2f},{r['ci_hi']:>+.2f}] "
                  f"{r['t_stat']:>6.2f} {r['p_value']:>6.3f}{sig_str:>3} | "
                  f"{r['weeks_pos']:>2.0f}/{r['weeks_total']:<2.0f}  {r['sharpe_ann']:>7.2f}")
    else:
        print("  No portfolio results.")

    # Weekly stability
    print(f"\n{'─'*70}")
    print("PHASE 5: Weekly stability")
    print(f"{'─'*70}")
    weekly_df = weekly_stability(panel)
    if len(weekly_df) > 0:
        weekly_df.to_csv(OUTPUT_DIR / "xs_weekly_stability.csv", index=False)
        print(f"  Saved xs_weekly_stability.csv ({len(weekly_df)} rows)")

        for regime in ["R1", "R2", "R3"]:
            sub = weekly_df[(weekly_df["regime"] == regime) & (weekly_df["horizon"] == 60)]
            if len(sub) == 0:
                continue
            weeks_pos = (sub["mean_excess"] > 0).sum()
            print(f"  {regime} H=60m: {len(sub)} weeks, {weeks_pos} positive ({weeks_pos/len(sub):.0%}), "
                  f"avg={sub['mean_excess'].mean():+.1f}bp")

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print("=" * 80)

    if len(port_df) > 0:
        best = port_df.loc[port_df["mean_ret_bp"].abs().idxmax()]
        sig_ports = port_df[port_df["p_value"] < 0.10]
        print(f"\n  Portfolio tests significant (p<0.10): {len(sig_ports)}/{len(port_df)}")
        if len(sig_ports) > 0:
            print(f"  Best: {best['regime']} H={best['horizon']:.0f}m K={best['K']:.0f}: "
                  f"mean={best['mean_ret_bp']:+.2f}bp, t={best['t_stat']:.2f}, p={best['p_value']:.4f}")

    if len(cond_df) > 0:
        sig_coins = cond_df[cond_df["q_fdr"] < 0.10]
        print(f"  Coin-level tests significant (q<0.10): {len(sig_coins)}/{len(cond_df)}")

    elapsed = time.monotonic() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
