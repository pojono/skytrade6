#!/usr/bin/env python3
"""
Walk-Forward Momentum Backtest (v22)

Clean out-of-sample test of: "Does momentum work in volatile regimes?"

Rules — NO LOOKAHEAD:
1. Warmup: first 30 days used to fit initial HMM + compute feature stats
2. HMM is refit every 7 days on expanding window (all data up to that point)
3. Feature z-scores use expanding mean/std (only past data)
4. Online regime detection: HMM forward filter + 3-bar confirmation
5. Trade: when detected volatile AND |momentum_z| > threshold → enter
6. Hold for 4h (48 bars), non-overlapping trades only
7. Fee: 7 bps round-trip deducted from each trade

Comparison:
- GMM online detection (single-bar posterior, refit same schedule)
- HMM online detection (forward filter + 3-bar confirm)
- Unconditional (ignore regime, trade momentum everywhere)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

from regime_detection import load_bars, compute_regime_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = ["ETHUSDT", "BTCUSDT"]
START = "2025-01-01"
END = "2026-01-31"

WARMUP_DAYS = 30          # days before we start trading
REFIT_EVERY = 14 * 288    # refit HMM/GMM every 14 days (in bars)
MAX_TRAIN_BARS = 90 * 288 # cap training window at 90 days (rolling) to keep HMM fast
HOLD_BARS = 48            # 4h hold period
FEE_BPS = 7               # round-trip fee
MOM_THRESHOLD = 1.0       # z-score threshold for entry
CONFIRM_BARS = 3          # bars of confirmation for regime detection

CLUSTER_FEATURES = [
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "parkvol_1h", "parkvol_4h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h", "vol_ratio_bar",
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    "trade_intensity_ratio",
    "bar_eff_1h", "bar_eff_4h",
    "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "sign_persist_1h", "sign_persist_2h",
    "vol_sma_24h",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def fit_hmm(X_arr, n_init=3):
    """Fit HMM with multiple inits, return best model."""
    best, best_score = None, -np.inf
    for i in range(n_init):
        try:
            m = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=200, tol=1e-4, random_state=42 + i, verbose=False)
            m.fit(X_arr)
            s = m.score(X_arr)
            if s > best_score:
                best_score, best = s, m
        except:
            pass
    return best


def fit_gmm(X_arr):
    """Fit GMM K=2."""
    m = GaussianMixture(n_components=2, covariance_type="diag",
                        n_init=3, random_state=42, max_iter=200)
    m.fit(X_arr)
    return m


def align_model_states(model, X_arr, rvol_vals, model_type="hmm"):
    """Figure out which state index = volatile (higher rvol).
    Returns the index of the volatile state."""
    if model_type == "hmm":
        labels = model.predict(X_arr)
    else:
        labels = model.predict(X_arr)
    r0_vol = rvol_vals[labels == 0].mean()
    r1_vol = rvol_vals[labels == 1].mean()
    return 1 if r1_vol > r0_vol else 0


def hmm_emission_loglik(hmm, X):
    """Compute log emission probabilities for each observation."""
    n, k = len(X), hmm.n_components
    log_lik = np.zeros((n, k))
    for j in range(k):
        mean = hmm.means_[j]
        if hmm.covars_.ndim == 2:
            var = hmm.covars_[j]
        else:
            var = np.diag(hmm.covars_[j])
        diff = X - mean
        log_lik[:, j] = (-0.5 * np.sum(diff**2 / var, axis=1)
                         - 0.5 * np.sum(np.log(var))
                         - 0.5 * X.shape[1] * np.log(2 * np.pi))
    return log_lik


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def run_walkforward(symbol, start, end):
    """Run walk-forward backtest for one symbol."""
    print(f"\n{'#'*70}")
    print(f"  WALK-FORWARD MOMENTUM BACKTEST — {symbol}")
    print(f"  Period: {start} → {end}")
    print(f"  Warmup: {WARMUP_DAYS} days, Refit: every {REFIT_EVERY//288} days")
    print(f"  Hold: {HOLD_BARS} bars ({HOLD_BARS*5}min), Fee: {FEE_BPS}bps")
    print(f"  Momentum threshold: {MOM_THRESHOLD} z-score")
    print(f"  Regime confirmation: {CONFIRM_BARS} bars")
    print(f"{'#'*70}")

    # Load data
    print(f"\n  Loading bars...")
    df = load_bars(symbol, start, end)
    if df.empty:
        print(f"  No data for {symbol}")
        return None
    print(f"  Computing features...")
    df = compute_regime_features(df)

    # Get feature columns
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_all = df[cols].copy()
    valid_mask = X_all.notna().all(axis=1)
    X_all = X_all[valid_mask]
    idx = X_all.index
    df_valid = df.loc[idx].copy()

    n_total = len(df_valid)
    warmup_bars = WARMUP_DAYS * 288
    trade_start = warmup_bars

    if n_total < warmup_bars + 288:
        print(f"  Not enough data: {n_total} bars, need {warmup_bars + 288}")
        return None

    print(f"  Total valid bars: {n_total}")
    print(f"  Warmup: {warmup_bars} bars ({WARMUP_DAYS} days)")
    print(f"  Trading period: bar {trade_start} to {n_total} ({(n_total - trade_start) / 288:.0f} days)")

    # Pre-extract arrays
    returns = df_valid["returns"].values
    rvol_1h = df_valid["rvol_1h"].values
    momentum_4h = df_valid["momentum_4h"].values
    X_raw = X_all.values  # unscaled features

    # ---------------------------------------------------------------------------
    # State: models, scaling params, online detection state
    # ---------------------------------------------------------------------------
    hmm_model = None
    gmm_model = None
    hmm_volatile_idx = 1
    gmm_volatile_idx = 1
    last_fit_bar = 0

    # Expanding stats for z-scoring momentum
    mom_sum = 0.0
    mom_sq_sum = 0.0
    mom_count = 0

    # HMM forward filter state
    log_alpha_prev = None  # previous forward variable
    hmm_log_transmat = None
    hmm_log_lik_cache = None

    # Online regime detection state (with confirmation)
    hmm_p_volatile_history = []
    gmm_p_volatile_history = []
    hmm_detected_regime = 0  # start assuming quiet
    gmm_detected_regime = 0
    hmm_confirm_count = 0
    gmm_confirm_count = 0

    # Trade tracking
    trades_hmm = []       # (bar, direction, entry_ret, pnl_bps)
    trades_gmm = []
    trades_uncond = []    # unconditional momentum
    trades_quiet_hmm = [] # momentum in quiet regime (should fail)

    in_trade_hmm = False
    in_trade_gmm = False
    in_trade_uncond = False
    in_trade_quiet = False
    trade_entry_bar_hmm = -999
    trade_entry_bar_gmm = -999
    trade_entry_bar_uncond = -999
    trade_entry_bar_quiet = -999

    # ---------------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------------
    t0 = time.time()

    for t in range(n_total):
        # --- Update expanding momentum stats ---
        if not np.isnan(momentum_4h[t]):
            mom_sum += momentum_4h[t]
            mom_sq_sum += momentum_4h[t] ** 2
            mom_count += 1

        # --- Fit/refit models ---
        if t >= warmup_bars and (t == warmup_bars or t - last_fit_bar >= REFIT_EVERY):
            # Standardize using expanding stats up to bar t
            X_past = X_raw[:t+1]
            past_means = np.nanmean(X_past, axis=0)
            past_stds = np.nanstd(X_past, axis=0)
            past_stds = np.clip(past_stds, 1e-10, None)

            # Cap training window for model fitting (keep it fast)
            train_start = max(0, t + 1 - MAX_TRAIN_BARS)
            X_train = X_raw[train_start:t+1]
            X_train_scaled = (X_train - past_means) / past_stds

            # Fit HMM/GMM on capped window
            hmm_model = fit_hmm(X_train_scaled)
            gmm_model = fit_gmm(X_train_scaled)

            if hmm_model is not None:
                hmm_volatile_idx = align_model_states(
                    hmm_model, X_train_scaled, rvol_1h[train_start:t+1], "hmm")
                hmm_log_transmat = np.log(hmm_model.transmat_ + 1e-300)

                # Reset forward filter with new model
                log_startprob = np.log(hmm_model.startprob_ + 1e-300)
                # Re-run forward filter on recent history to warm up state
                # Use last 288 bars (1 day) to initialize
                recent_start = max(0, t - 288)
                X_recent_scaled = (X_raw[recent_start:t+1] - past_means) / past_stds
                recent_log_lik = hmm_emission_loglik(hmm_model, X_recent_scaled)

                log_alpha = np.zeros(2)
                log_alpha = log_startprob + recent_log_lik[0]
                for tt in range(1, len(X_recent_scaled)):
                    new_log_alpha = np.zeros(2)
                    for j in range(2):
                        new_log_alpha[j] = recent_log_lik[tt, j] + logsumexp(
                            log_alpha + hmm_log_transmat[:, j])
                    log_alpha = new_log_alpha
                log_alpha_prev = log_alpha

            if gmm_model is not None:
                gmm_volatile_idx = align_model_states(
                    gmm_model, X_train_scaled, rvol_1h[train_start:t+1], "gmm")

            last_fit_bar = t
            # Store current scaling params
            current_means = past_means
            current_stds = past_stds

            if t == warmup_bars:
                print(f"  Initial fit done at bar {t}")
            else:
                elapsed = time.time() - t0
                days_done = (t - warmup_bars) / 288
                print(f"  Refit at bar {t} ({days_done:.0f} days in), "
                      f"HMM trades={len(trades_hmm)}, GMM trades={len(trades_gmm)}, "
                      f"Uncond trades={len(trades_uncond)}, elapsed={elapsed:.0f}s")

        # --- Skip if still in warmup ---
        if t < trade_start:
            continue

        # --- Online feature scaling (using expanding stats from last fit) ---
        x_scaled = (X_raw[t] - current_means) / current_stds

        # --- HMM forward filter: update P(state_t | obs_1..t) ---
        hmm_p_volatile = 0.5  # default
        if hmm_model is not None and log_alpha_prev is not None:
            x_log_lik = hmm_emission_loglik(hmm_model, x_scaled.reshape(1, -1))[0]
            new_log_alpha = np.zeros(2)
            for j in range(2):
                new_log_alpha[j] = x_log_lik[j] + logsumexp(
                    log_alpha_prev + hmm_log_transmat[:, j])
            log_alpha_prev = new_log_alpha

            # Posterior
            log_norm = logsumexp(new_log_alpha)
            posterior = np.exp(new_log_alpha - log_norm)
            hmm_p_volatile = posterior[hmm_volatile_idx]

        hmm_p_volatile_history.append(hmm_p_volatile)

        # --- GMM online detection ---
        gmm_p_volatile = 0.5
        if gmm_model is not None:
            probs = gmm_model.predict_proba(x_scaled.reshape(1, -1))[0]
            gmm_p_volatile = probs[gmm_volatile_idx]
        gmm_p_volatile_history.append(gmm_p_volatile)

        # --- Apply 3-bar confirmation for regime detection ---
        # HMM
        raw_hmm = 1 if hmm_p_volatile >= 0.5 else 0
        if raw_hmm != hmm_detected_regime:
            hmm_confirm_count += 1
            if hmm_confirm_count >= CONFIRM_BARS:
                hmm_detected_regime = raw_hmm
                hmm_confirm_count = 0
        else:
            hmm_confirm_count = 0

        # GMM
        raw_gmm = 1 if gmm_p_volatile >= 0.5 else 0
        if raw_gmm != gmm_detected_regime:
            gmm_confirm_count += 1
            if gmm_confirm_count >= CONFIRM_BARS:
                gmm_detected_regime = raw_gmm
                gmm_confirm_count = 0
        else:
            gmm_confirm_count = 0

        # --- Compute momentum z-score (expanding) ---
        mom_z = np.nan
        if mom_count > 100 and not np.isnan(momentum_4h[t]):
            mom_mean = mom_sum / mom_count
            mom_var = mom_sq_sum / mom_count - mom_mean ** 2
            mom_std = max(np.sqrt(max(mom_var, 0)), 1e-10)
            mom_z = (momentum_4h[t] - mom_mean) / mom_std

        # --- Check if we can compute forward return ---
        can_compute_fwd = (t + HOLD_BARS < n_total)
        if can_compute_fwd:
            fwd_ret = np.sum(returns[t+1:t+1+HOLD_BARS])
        else:
            fwd_ret = np.nan

        # --- Trade logic: HMM volatile + momentum ---
        if not in_trade_hmm and not np.isnan(mom_z) and can_compute_fwd:
            if hmm_detected_regime == 1 and abs(mom_z) > MOM_THRESHOLD:
                direction = 1 if mom_z > 0 else -1
                pnl_bps = direction * fwd_ret * 10000 - FEE_BPS
                trades_hmm.append({
                    "bar": t, "direction": direction, "mom_z": mom_z,
                    "fwd_ret": fwd_ret, "pnl_bps": pnl_bps,
                    "p_volatile": hmm_p_volatile
                })
                in_trade_hmm = True
                trade_entry_bar_hmm = t

        if in_trade_hmm and t - trade_entry_bar_hmm >= HOLD_BARS:
            in_trade_hmm = False

        # --- Trade logic: GMM volatile + momentum ---
        if not in_trade_gmm and not np.isnan(mom_z) and can_compute_fwd:
            if gmm_detected_regime == 1 and abs(mom_z) > MOM_THRESHOLD:
                direction = 1 if mom_z > 0 else -1
                pnl_bps = direction * fwd_ret * 10000 - FEE_BPS
                trades_gmm.append({
                    "bar": t, "direction": direction, "mom_z": mom_z,
                    "fwd_ret": fwd_ret, "pnl_bps": pnl_bps,
                    "p_volatile": gmm_p_volatile
                })
                in_trade_gmm = True
                trade_entry_bar_gmm = t

        if in_trade_gmm and t - trade_entry_bar_gmm >= HOLD_BARS:
            in_trade_gmm = False

        # --- Trade logic: Unconditional momentum (ignore regime) ---
        if not in_trade_uncond and not np.isnan(mom_z) and can_compute_fwd:
            if abs(mom_z) > MOM_THRESHOLD:
                direction = 1 if mom_z > 0 else -1
                pnl_bps = direction * fwd_ret * 10000 - FEE_BPS
                trades_uncond.append({
                    "bar": t, "direction": direction, "mom_z": mom_z,
                    "fwd_ret": fwd_ret, "pnl_bps": pnl_bps,
                })
                in_trade_uncond = True
                trade_entry_bar_uncond = t

        if in_trade_uncond and t - trade_entry_bar_uncond >= HOLD_BARS:
            in_trade_uncond = False

        # --- Trade logic: HMM quiet + momentum (control — should fail) ---
        if not in_trade_quiet and not np.isnan(mom_z) and can_compute_fwd:
            if hmm_detected_regime == 0 and abs(mom_z) > MOM_THRESHOLD:
                direction = 1 if mom_z > 0 else -1
                pnl_bps = direction * fwd_ret * 10000 - FEE_BPS
                trades_quiet_hmm.append({
                    "bar": t, "direction": direction, "mom_z": mom_z,
                    "fwd_ret": fwd_ret, "pnl_bps": pnl_bps,
                })
                in_trade_quiet = True
                trade_entry_bar_quiet = t

        if in_trade_quiet and t - trade_entry_bar_quiet >= HOLD_BARS:
            in_trade_quiet = False

        # Progress
        if t > 0 and t % (30 * 288) == 0:
            days = (t - trade_start) / 288
            elapsed = time.time() - t0
            print(f"  Bar {t}/{n_total} ({days:.0f} trading days), "
                  f"HMM={len(trades_hmm)} GMM={len(trades_gmm)} Uncond={len(trades_uncond)} "
                  f"Quiet={len(trades_quiet_hmm)} trades, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    trading_days = (n_total - trade_start) / 288

    print(f"\n{'='*70}")
    print(f"  RESULTS — {symbol}")
    print(f"  Trading period: {trading_days:.0f} days")
    print(f"{'='*70}")

    strategies = [
        ("HMM volatile + momentum", trades_hmm),
        ("GMM volatile + momentum", trades_gmm),
        ("Unconditional momentum", trades_uncond),
        ("HMM quiet + momentum (control)", trades_quiet_hmm),
    ]

    results = {}
    for name, trades in strategies:
        print(f"\n  --- {name} ---")
        if len(trades) == 0:
            print(f"  No trades")
            results[name] = {"n_trades": 0}
            continue

        tdf = pd.DataFrame(trades)
        pnls = tdf["pnl_bps"].values
        n_trades = len(pnls)
        trades_per_day = n_trades / trading_days

        avg_pnl = pnls.mean()
        med_pnl = np.median(pnls)
        std_pnl = pnls.std()
        win_rate = (pnls > 0).mean()
        total_pnl = pnls.sum()

        # Sharpe-like: avg / std * sqrt(trades_per_year)
        if std_pnl > 0:
            sharpe_per_trade = avg_pnl / std_pnl
            annual_trades = trades_per_day * 365
            sharpe_annual = sharpe_per_trade * np.sqrt(annual_trades)
        else:
            sharpe_annual = 0

        # Long vs short breakdown
        longs = tdf[tdf["direction"] == 1]["pnl_bps"]
        shorts = tdf[tdf["direction"] == -1]["pnl_bps"]

        # Monthly breakdown
        bar_to_month = {}
        timestamps = df_valid["timestamp_us"].values
        for _, row in tdf.iterrows():
            bar_idx = int(row["bar"])
            if bar_idx < len(timestamps):
                ts = pd.Timestamp(timestamps[bar_idx], unit="us", tz="UTC")
                month = ts.strftime("%Y-%m")
                bar_to_month[bar_idx] = month

        tdf["month"] = tdf["bar"].map(bar_to_month)
        monthly = tdf.groupby("month")["pnl_bps"].agg(["mean", "count", "sum"])

        print(f"  Trades: {n_trades} ({trades_per_day:.1f}/day)")
        print(f"  Avg PnL: {avg_pnl:+.2f} bps")
        print(f"  Median PnL: {med_pnl:+.2f} bps")
        print(f"  Std PnL: {std_pnl:.2f} bps")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Total PnL: {total_pnl:+.1f} bps")
        print(f"  Sharpe (annualized): {sharpe_annual:.2f}")
        print(f"  Longs: {len(longs)} trades, avg={longs.mean():+.2f} bps" if len(longs) > 0 else "  Longs: 0")
        print(f"  Shorts: {len(shorts)} trades, avg={shorts.mean():+.2f} bps" if len(shorts) > 0 else "  Shorts: 0")

        print(f"\n  Monthly breakdown:")
        print(f"  {'Month':>8s}  {'Trades':>6s}  {'Avg PnL':>8s}  {'Total':>8s}")
        for month, row in monthly.iterrows():
            print(f"  {month:>8s}  {int(row['count']):>6d}  {row['mean']:>+8.2f}  {row['sum']:>+8.1f}")

        # Positive/negative months
        pos_months = (monthly["mean"] > 0).sum()
        neg_months = (monthly["mean"] <= 0).sum()
        print(f"  Positive months: {pos_months}/{pos_months + neg_months}")

        results[name] = {
            "n_trades": n_trades, "trades_per_day": trades_per_day,
            "avg_pnl": avg_pnl, "med_pnl": med_pnl, "std_pnl": std_pnl,
            "win_rate": win_rate, "total_pnl": total_pnl,
            "sharpe_annual": sharpe_annual,
            "pos_months": pos_months, "neg_months": neg_months,
        }

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY COMPARISON — {symbol}")
    print(f"{'='*70}")
    print(f"  {'Strategy':>35s}  {'Trades':>6s}  {'Avg PnL':>8s}  {'Win%':>5s}  "
          f"{'Sharpe':>6s}  {'Total':>8s}  {'Pos Mo':>6s}")
    print(f"  {'-'*80}")
    for name, r in results.items():
        if r["n_trades"] == 0:
            print(f"  {name:>35s}  {'0':>6s}  {'N/A':>8s}")
            continue
        print(f"  {name:>35s}  {r['n_trades']:>6d}  {r['avg_pnl']:>+8.2f}  "
              f"{r['win_rate']:>5.1%}  {r['sharpe_annual']:>6.2f}  "
              f"{r['total_pnl']:>+8.1f}  {r['pos_months']}/{r['pos_months']+r['neg_months']}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"{'#'*70}")
    print(f"  WALK-FORWARD MOMENTUM BACKTEST (v22)")
    print(f"  NO LOOKAHEAD — online regime detection only")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Period: {START} → {END}")
    print(f"  Warmup: {WARMUP_DAYS} days")
    print(f"  Refit: every {REFIT_EVERY // 288} days")
    print(f"{'#'*70}")

    all_results = {}
    for sym in SYMBOLS:
        r = run_walkforward(sym, START, END)
        if r is not None:
            all_results[sym] = r

    # Cross-symbol summary
    if len(all_results) > 1:
        print(f"\n\n{'#'*70}")
        print(f"  CROSS-SYMBOL SUMMARY")
        print(f"{'#'*70}")
        for strat in ["HMM volatile + momentum", "GMM volatile + momentum",
                       "Unconditional momentum", "HMM quiet + momentum (control)"]:
            print(f"\n  {strat}:")
            for sym, results in all_results.items():
                r = results.get(strat, {})
                if r.get("n_trades", 0) == 0:
                    print(f"    {sym}: no trades")
                else:
                    print(f"    {sym}: {r['n_trades']} trades, avg={r['avg_pnl']:+.2f}bps, "
                          f"win={r['win_rate']:.1%}, sharpe={r['sharpe_annual']:.2f}")

    print(f"\n\nDone.")


if __name__ == "__main__":
    main()
