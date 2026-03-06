"""
Phase 2: Cross-sectional portfolio backtest with walk-forward validation.

Signals (from Phase 1 IC analysis):
  - premium_z   IC=0.116  ICIR=0.96  (dominant)
  - mom_24h     IC=0.067  ICIR=0.42
  - funding     IC=0.061  ICIR=0.53

Portfolio mechanics:
  - Long top N / short bottom N coins by composite signal
  - Rebalance every 8h (aligned to funding settlements: 00:00, 08:00, 16:00 UTC)
  - Equal-weight within each leg
  - Turnover-adjusted fee drag

Fee scenarios:
  - Taker: 10 bps/side = 20 bps RT
  - Maker: 4 bps/side  =  8 bps RT
  - Mixed: 4 enter / 10 exit = 14 bps RT

Walk-forward:
  - 6-month train (IC stability check) / 6-month OOS, sliding by 3 months
  - Full-sample backtest + per-OOS-window stats

Outputs:
  results/phase2_summary.csv       — per-strategy stats
  results/phase2_equity.csv        — equity curves (8h bars)
  results/phase2_walkforward.csv   — per-window OOS stats
  results/phase2_signal_corr.csv   — pairwise signal correlation
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import time

warnings.filterwarnings("ignore")

SIGNALS_DIR  = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR  = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"

# Signals to include (Phase 1 winners with positive IC)
SIGNAL_COLS  = ["prem_z", "funding", "mom_24h"]

# Portfolio parameters
N_LONG       = 10   # coins per leg
N_SHORT      = 10
REBAL_FREQ   = "8h"  # rebalance every 8 hours

# Fee scenarios (bps per side, round-trip = 2x)
FEE_TAKER    = 10   # bps per side
FEE_MAKER    = 4
FEE_MIXED    = 7    # average (4 enter + 10 exit) / 2

# Walk-forward windows
TRAIN_MONTHS = 6
OOS_MONTHS   = 3

# Annualisation: 8h periods per year
PERIODS_PER_YEAR = 365 * 3  # 3 × 8h per day


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panels(signals_dir, cols):
    """Load signal parquets → wide DataFrames {col: (timestamp × symbol)}."""
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
    print(f"Loading {len(files)} parquet files for cols: {cols}")

    data = {c: {} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet", "")
        try:
            df = pd.read_parquet(fpath, columns=cols)
            for c in cols:
                if c in df.columns:
                    data[c][sym] = df[c]
        except Exception as e:
            pass

    panels = {}
    for c in cols:
        if data[c]:
            p = pd.DataFrame(data[c])
            p.index = pd.to_datetime(p.index, utc=True)
            panels[c] = p.sort_index()
    print(f"  Loaded: {list(panels.keys())}")
    return panels


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def cs_zscore(panel, min_valid=20):
    """
    Cross-sectional z-score at each bar.
    Clips at ±3 to reduce outlier impact.
    """
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0)
    z[n < min_valid] = np.nan
    return z.clip(-3, 3)


def build_composite(panels, signal_cols, ic_weights=None):
    """
    Equal-weight (or IC-weighted) composite of cross-sectionally z-scored signals.
    Returns a panel of composite scores.
    """
    zscored = {}
    for c in signal_cols:
        if c in panels:
            zscored[c] = cs_zscore(panels[c])

    if not zscored:
        raise ValueError("No signals available")

    if ic_weights is None:
        # Equal weight
        ic_weights = {c: 1.0 for c in zscored}

    total_w = sum(ic_weights[c] for c in zscored)

    # Align all to common index
    idx = zscored[list(zscored.keys())[0]].index
    composite = None
    for c, z in zscored.items():
        w = ic_weights.get(c, 1.0) / total_w
        aligned = z.reindex(idx)
        if composite is None:
            composite = aligned * w
        else:
            composite = composite.add(aligned * w, fill_value=0)

    return composite


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def resample_to_rebal(panel, freq="8h"):
    """
    Resample a 1h signal panel to rebalancing frequency.
    Uses the value at the START of each window (decision time) — no lookahead.
    For 8h rebal at 00:00/08:00/16:00, takes the 1h bar value AT those timestamps.
    """
    return panel.resample(freq, closed="left", label="left").first()


def align_fwd_to_rebal(fwd_panel, rebal_index):
    """
    Get forward return values aligned exactly to rebalancing timestamps.
    fwd_Nh[t] = (close[t+N] - close[t]) / close[t] — already computed.
    We just reindex to the rebal timestamps (no aggregation needed).
    Also clips extreme values (|ret| > 3.0) that indicate data corruption.
    """
    aligned = fwd_panel.reindex(rebal_index)
    # Clip corrupt returns: >300% in 8h is data error (zeroed-out prices etc.)
    aligned = aligned.clip(-0.99, 3.0)
    # Any remaining inf/nan from division by zero → NaN
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    return aligned


def sim_portfolio(signal_panel, fwd_panel, n_long=10, n_short=10,
                  fee_side_bps=10, min_universe=20):
    """
    Vectorised long/short portfolio simulation.

    Args:
        signal_panel : (timestamps × symbols) — composite signal at rebal bars
        fwd_panel    : (timestamps × symbols) — forward return for holding period
        n_long       : number of coins in long leg
        n_short      : number of coins in short leg
        fee_side_bps : fee per side in bps (round-trip = 2×)
        min_universe : min valid coins required to trade

    Returns:
        df with columns: gross_ret, turnover, fee_drag, net_ret, n_long, n_short
    """
    fee_rt = fee_side_bps * 2 / 10000  # round-trip fee as decimal

    records = []
    prev_longs  = set()
    prev_shorts = set()

    timestamps = signal_panel.index.intersection(fwd_panel.index)

    for ts in timestamps:
        sig = signal_panel.loc[ts].dropna()
        fwd = fwd_panel.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe:
            records.append({"timestamp": ts, "gross_ret": np.nan,
                            "turnover": np.nan, "fee_drag": np.nan,
                            "net_ret": np.nan, "n_active": len(sig)})
            prev_longs, prev_shorts = set(), set()
            continue

        ranked = sig.rank(ascending=True)
        n = len(ranked)
        top_thresh = ranked.quantile((n - n_long) / n)
        bot_thresh = ranked.quantile(n_short / n)

        longs  = set(sig[ranked >= top_thresh].index[-n_long:])
        shorts = set(sig[ranked <= bot_thresh].index[:n_short])

        # Returns
        long_ret  = fwd.loc[list(longs)].mean()  if longs  else np.nan
        short_ret = fwd.loc[list(shorts)].mean() if shorts else np.nan

        if np.isnan(long_ret) or np.isnan(short_ret):
            gross = np.nan
        else:
            gross = long_ret - short_ret  # long minus short

        # Turnover: fraction of portfolio changed
        all_prev = prev_longs | prev_shorts
        all_curr = longs | shorts
        if all_prev:
            changed  = len((longs - prev_longs) | (shorts - prev_shorts))
            total    = len(longs) + len(shorts)
            turnover = changed / total if total > 0 else 0
        else:
            turnover = 1.0  # first bar: full portfolio construction

        fee_drag = turnover * fee_rt

        records.append({
            "timestamp": ts,
            "gross_ret": gross,
            "turnover":  turnover,
            "fee_drag":  fee_drag,
            "net_ret":   gross - fee_drag if not np.isnan(gross) else np.nan,
            "n_active":  len(sig),
        })

        prev_longs  = longs
        prev_shorts = shorts

    df = pd.DataFrame(records).set_index("timestamp")
    return df


def portfolio_stats(pnl_series, periods_per_year=PERIODS_PER_YEAR, label=""):
    """Compute annualised performance statistics."""
    s = pnl_series.dropna()
    if len(s) < 10:
        return {}

    mean_ret  = s.mean()
    std_ret   = s.std()
    ann_ret   = mean_ret * periods_per_year
    ann_vol   = std_ret  * np.sqrt(periods_per_year)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside  = s[s < 0].std() * np.sqrt(periods_per_year)
    sortino   = ann_ret / downside if downside > 0 else np.nan

    cum       = (1 + s).cumprod()
    rolling_max = cum.cummax()
    dd        = (cum / rolling_max - 1)
    max_dd    = dd.min()

    win_rate  = (s > 0).mean()
    n_bars    = len(s)

    # t-stat on mean return
    t_stat    = mean_ret / (std_ret / np.sqrt(n_bars)) if std_ret > 0 else np.nan

    return {
        "label":        label,
        "n_bars":       n_bars,
        "mean_ret_bps": round(mean_ret * 10000, 2),
        "ann_ret_pct":  round(ann_ret * 100, 2),
        "ann_vol_pct":  round(ann_vol * 100, 2),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_dd_pct":   round(max_dd * 100, 2),
        "win_rate":     round(win_rate, 3),
        "t_stat":       round(t_stat, 2),
    }


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward(signal_panel, fwd_panel, train_months=6, oos_months=3,
                 n_long=10, n_short=10, fee_side_bps=10):
    """
    Rolling walk-forward: for each OOS window, run sim_portfolio and collect stats.
    Train window is used only to confirm IC stability (not for parameter fitting).
    """
    start = signal_panel.index.min().tz_localize(None) if signal_panel.index.tz else signal_panel.index.min()
    end   = signal_panel.index.max().tz_localize(None) if signal_panel.index.tz else signal_panel.index.max()

    # Use tz-aware throughout
    start = signal_panel.index.min()
    end   = signal_panel.index.max()

    windows = []
    t = start + pd.DateOffset(months=train_months)
    while t + pd.DateOffset(months=oos_months) <= end + pd.Timedelta(days=1):
        train_start = t - pd.DateOffset(months=train_months)
        train_end   = t
        oos_start   = t
        oos_end     = t + pd.DateOffset(months=oos_months)
        windows.append((train_start, train_end, oos_start, oos_end))
        t += pd.DateOffset(months=oos_months)

    all_oos_pnl = []
    wf_stats    = []

    for train_s, train_e, oos_s, oos_e in windows:
        # Train: IC check (informational)
        sig_train = signal_panel.loc[train_s:train_e]
        fwd_train = fwd_panel.loc[train_s:train_e]
        common_ts_train = sig_train.index.intersection(fwd_train.index)
        if len(common_ts_train) < 30:
            continue

        # Compute train IC
        S = sig_train.loc[common_ts_train]
        F = fwd_train.loc[common_ts_train]
        valid = S.notna() & F.notna()
        n_valid = valid.sum(axis=1)
        good = common_ts_train[n_valid >= 20]
        if len(good) > 0:
            S_g = S.loc[good]; F_g = F.loc[good]
            S_r = S_g.rank(axis=1); F_r = F_g.rank(axis=1)
            S_dm = S_r.sub(S_r.mean(axis=1), axis=0)
            F_dm = F_r.sub(F_r.mean(axis=1), axis=0)
            num  = (S_dm * F_dm).sum(axis=1)
            den  = np.sqrt((S_dm**2).sum(axis=1) * (F_dm**2).sum(axis=1))
            ic_series = (num / den.replace(0, np.nan)).dropna()
            train_ic  = ic_series.mean()
        else:
            train_ic = np.nan

        # OOS: full portfolio simulation
        sig_oos = signal_panel.loc[oos_s:oos_e]
        fwd_oos = fwd_panel.reindex(sig_oos.index).clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
        if sig_oos.empty or fwd_oos.empty:
            continue

        pnl = sim_portfolio(sig_oos, fwd_oos, n_long=n_long, n_short=n_short,
                            fee_side_bps=fee_side_bps)
        net = pnl["net_ret"].dropna()
        gross = pnl["gross_ret"].dropna()

        if len(net) < 5:
            continue

        all_oos_pnl.append(pnl)

        st = portfolio_stats(net, label=f"OOS {oos_s.date()}–{oos_e.date()}")
        st["train_ic"]    = round(train_ic, 4) if not np.isnan(train_ic) else np.nan
        st["gross_ret_bps"] = round(gross.mean() * 10000, 2)
        st["avg_turnover"]  = round(pnl["turnover"].mean(), 3)
        wf_stats.append(st)

    combined_pnl = pd.concat(all_oos_pnl) if all_oos_pnl else pd.DataFrame()
    return combined_pnl, wf_stats


# ---------------------------------------------------------------------------
# Signal correlation analysis
# ---------------------------------------------------------------------------

def signal_correlation(panels, signal_cols, resample_freq="8h"):
    """Cross-sectional rank correlation between signals (avg over time)."""
    resampled = {}
    for c in signal_cols:
        if c in panels:
            resampled[c] = resample_to_rebal(panels[c], freq=resample_freq)

    results = {}
    cols = list(resampled.keys())
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            p1 = resampled[c1]
            p2 = resampled[c2]
            common_ts = p1.index.intersection(p2.index)
            common_sym = p1.columns.intersection(p2.columns)
            if len(common_ts) == 0 or len(common_sym) == 0:
                results[(c1, c2)] = np.nan
                continue
            r1 = p1.loc[common_ts, common_sym].rank(axis=1)
            r2 = p2.loc[common_ts, common_sym].rank(axis=1)
            r1_dm = r1.sub(r1.mean(axis=1), axis=0)
            r2_dm = r2.sub(r2.mean(axis=1), axis=0)
            num = (r1_dm * r2_dm).sum(axis=1)
            den = np.sqrt((r1_dm**2).sum(axis=1) * (r2_dm**2).sum(axis=1))
            corr_ts = (num / den.replace(0, np.nan)).dropna()
            results[(c1, c2)] = round(corr_ts.mean(), 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load all needed columns
    all_cols = SIGNAL_COLS + [f"fwd_{h}h" for h in [1, 4, 8, 24]]
    panels   = load_panels(SIGNALS_DIR, all_cols)

    # IC weights from Phase 1 (for IC-weighted composite)
    IC_WEIGHTS = {"prem_z": 0.116, "funding": 0.061, "mom_24h": 0.067}

    # ------------------------------------------------------------------
    # Signal correlation analysis
    # ------------------------------------------------------------------
    print("\n--- Signal Pairwise Cross-Sectional Rank Correlation (avg over time) ---")
    corr = signal_correlation(panels, SIGNAL_COLS)
    for (c1, c2), r in corr.items():
        print(f"  {c1} vs {c2}: {r:+.4f}")
    corr_rows = [{"sig1": c1, "sig2": c2, "corr": r} for (c1, c2), r in corr.items()]
    pd.DataFrame(corr_rows).to_csv(os.path.join(RESULTS_DIR, "phase2_signal_corr.csv"), index=False)

    # ------------------------------------------------------------------
    # Build composite signal and resample to 8h rebalance bars
    # ------------------------------------------------------------------
    print(f"\n--- Building composite signal (resampled to {REBAL_FREQ}) ---")

    # Equal-weight composite
    comp_eq  = build_composite(panels, SIGNAL_COLS, ic_weights=None)
    # IC-weighted composite
    comp_ic  = build_composite(panels, SIGNAL_COLS, ic_weights=IC_WEIGHTS)
    # Individual signals (cross-sectionally z-scored)
    ind_sigs = {c: cs_zscore(panels[c]) for c in SIGNAL_COLS if c in panels}

    # Resample signal to 8h (use last value in each window — no lookahead)
    comp_eq_8h = resample_to_rebal(comp_eq,  REBAL_FREQ)
    comp_ic_8h = resample_to_rebal(comp_ic,  REBAL_FREQ)
    ind_8h     = {c: resample_to_rebal(s, REBAL_FREQ) for c, s in ind_sigs.items()}

    # Rebalance-bar forward returns: use exact 1h values at rebal timestamps
    # (avoids lookahead that resample.last() would introduce)
    rebal_index  = comp_eq_8h.index
    fwd_8h_panel = align_fwd_to_rebal(panels["fwd_8h"], rebal_index)

    # Check coverage
    common = comp_eq_8h.index.intersection(fwd_8h_panel.index)
    print(f"  8h rebal bars: {len(common)}  ({common[0].date()} → {common[-1].date()})")
    print(f"  Universe: {comp_eq_8h.shape[1]} symbols\n")

    # ------------------------------------------------------------------
    # Full-sample backtest — all fee scenarios
    # ------------------------------------------------------------------
    strategies = [
        ("Composite EW",   comp_eq_8h),
        ("Composite IC-wt",comp_ic_8h),
        ("prem_z only",    ind_8h.get("prem_z")),
        ("funding only",   ind_8h.get("funding")),
        ("mom_24h only",   ind_8h.get("mom_24h")),
    ]

    fee_scenarios = [
        ("Taker", FEE_TAKER),
        ("Maker", FEE_MAKER),
        ("Mixed", FEE_MIXED),
    ]

    summary_rows = []
    equity_dfs   = {}

    print("=" * 105)
    print(f"{'Strategy':<20} {'Fees':<8} {'Mean(bps)':>10} {'Ann Ret%':>9} {'Ann Vol%':>9} "
          f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'WinRate':>8} {'t-stat':>7}")
    print("=" * 105)

    for strat_name, sig_panel in strategies:
        if sig_panel is None:
            continue
        print(f"\n  {strat_name}")
        for fee_name, fee_bps in fee_scenarios:
            pnl_df = sim_portfolio(sig_panel, fwd_8h_panel,
                                   n_long=N_LONG, n_short=N_SHORT,
                                   fee_side_bps=fee_bps)
            st = portfolio_stats(pnl_df["net_ret"],
                                 label=f"{strat_name} / {fee_name}")

            row = {**st, "strategy": strat_name, "fee_scenario": fee_name,
                   "fee_bps_side": fee_bps,
                   "avg_turnover": round(pnl_df["turnover"].mean(), 3),
                   "gross_ret_bps": round(pnl_df["gross_ret"].mean() * 10000, 2),
                   "avg_fee_drag_bps": round(pnl_df["fee_drag"].mean() * 10000, 2)}
            summary_rows.append(row)

            print(f"    {fee_name:<8} "
                  f"{st.get('mean_ret_bps',0):>10.2f} "
                  f"{st.get('ann_ret_pct',0):>9.2f} "
                  f"{st.get('ann_vol_pct',0):>9.2f} "
                  f"{st.get('sharpe',0):>7.3f} "
                  f"{st.get('sortino',0):>8.3f} "
                  f"{st.get('max_dd_pct',0):>8.2f} "
                  f"{st.get('win_rate',0):>8.3f} "
                  f"{st.get('t_stat',0):>7.2f}")

            # Save equity curve for best scenario only (maker)
            if fee_name == "Maker":
                eq = (1 + pnl_df["net_ret"].dropna()).cumprod()
                equity_dfs[strat_name] = eq

    print("=" * 105)

    # ------------------------------------------------------------------
    # Walk-forward OOS validation (composite EW, maker fees)
    # ------------------------------------------------------------------
    print(f"\n--- Walk-forward OOS validation (Composite EW, Maker fees) ---")
    print(f"  Train={TRAIN_MONTHS}mo / OOS={OOS_MONTHS}mo, rolling\n")

    wf_pnl, wf_stats = walk_forward(
        comp_eq_8h, fwd_8h_panel,
        train_months=TRAIN_MONTHS, oos_months=OOS_MONTHS,
        n_long=N_LONG, n_short=N_SHORT, fee_side_bps=FEE_MAKER
    )

    if wf_stats:
        print(f"  {'Window':<30} {'TrainIC':>8} {'GrossRet':>9} {'NetRet':>9} "
              f"{'Sharpe':>7} {'MaxDD%':>8} {'Turnover':>9}")
        print(f"  {'-'*90}")
        for st in wf_stats:
            print(f"  {st['label']:<30} {st.get('train_ic', 0):>8.4f} "
                  f"{st.get('gross_ret_bps',0):>9.2f} "
                  f"{st.get('mean_ret_bps',0):>9.2f} "
                  f"{st.get('sharpe',0):>7.3f} "
                  f"{st.get('max_dd_pct',0):>8.2f} "
                  f"{st.get('avg_turnover',0):>9.3f}")

        # Combined OOS stats
        if not wf_pnl.empty:
            combined_st = portfolio_stats(wf_pnl["net_ret"],
                                         label="Combined OOS (all windows)")
            print(f"\n  Combined OOS: Sharpe={combined_st.get('sharpe'):.3f}  "
                  f"Ann={combined_st.get('ann_ret_pct'):.2f}%  "
                  f"MaxDD={combined_st.get('max_dd_pct'):.2f}%  "
                  f"t-stat={combined_st.get('t_stat'):.2f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "phase2_summary.csv"), index=False)

    if equity_dfs:
        df_equity = pd.DataFrame(equity_dfs)
        df_equity.to_csv(os.path.join(RESULTS_DIR, "phase2_equity.csv"))

    if wf_stats:
        pd.DataFrame(wf_stats).to_csv(
            os.path.join(RESULTS_DIR, "phase2_walkforward.csv"), index=False)

    if not wf_pnl.empty:
        wf_pnl.to_csv(os.path.join(RESULTS_DIR, "phase2_wf_pnl.csv"))

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
