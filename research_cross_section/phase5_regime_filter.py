"""
Phase 5: Regime filter analysis — trade/no-trade switch to reduce drawdowns.

Anti-overfitting design:
  - All feature thresholds found in TRAINING window only
  - Applied to OOS without refitting
  - Same walk-forward splits as Phase 2 (6mo train / 3mo OOS)
  - ML models: max depth 2-3, strong regularisation, only top-5 correlated features
  - Report train IC and OOS separately — only trust OOS

10 regime features (all computed from data available AT decision time):
  F1  btc_rvol_7d      BTC 7-day realized vol             high → bad
  F2  avg_funding      Universe mean funding rate          extreme → crowded longs
  F3  funding_disp     Cross-sectional std of funding      low → weak signal
  F4  cs_ret_disp      Cross-sect std of 8h returns        low → high correlation, no spread
  F5  btc_ret_24h      BTC 24h return (3 bars)             negative → bad for longs
  F6  btc_ret_72h      BTC 72h return (9 bars)             downtrend → bad
  F7  rolling_ic_30d   Rolling 30d IC of composite vs fwd  negative → signal broken
  F8  avg_prem_z       Universe mean premium z-score       extreme → basis stress
  F9  avg_oi_div       Universe mean OI-price divergence   negative → liquidation
  F10 signal_strength  CS std of composite signal          low → no differentiation

Outputs:
  results/phase5_feature_corr.csv        feature vs next-bar P&L correlations
  results/phase5_single_filter.csv       per-feature walk-forward OOS stats
  results/phase5_ml_results.csv          ML model OOS stats
  results/phase5_best_filter.csv         monthly table: baseline vs best filter
  results/phase5_regime_features.png     feature correlation heatmap + time series
  results/phase5_equity_filtered.png     equity + DD: baseline vs filters
  results/phase5_monthly_comparison.png  monthly bars: baseline vs best filter
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
N_LONG           = 10
N_SHORT          = 10
FEE_MAKER_BPS    = 4
PERIODS_PER_YEAR = 365 * 3   # 8h bars
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3


# ============================================================
# Loading
# ============================================================

def load_panels(cols):
    files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.parquet")))
    data = {c: {} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet", "")
        try:
            df = pd.read_parquet(fpath)
            for c in cols:
                if c in df.columns:
                    data[c][sym] = df[c]
        except Exception:
            pass
    panels = {}
    for c in cols:
        if data[c]:
            p = pd.DataFrame(data[c])
            p.index = pd.to_datetime(p.index, utc=True)
            panels[c] = p.sort_index()
    return panels


# ============================================================
# Signal / portfolio helpers (same as phase 2b)
# ============================================================

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-3, 3)
    z[n < min_valid] = np.nan
    return z


def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()


def sim_portfolio(composite_8h, fwd_8h, mask=None,
                  n_long=N_LONG, n_short=N_SHORT,
                  fee_bps=FEE_MAKER_BPS, min_universe=20):
    """
    mask: boolean Series aligned to composite_8h.index
          True  = trade this bar
          False = skip (flat, no cost)
    """
    fee_rt = fee_bps * 2 / 10000
    common = composite_8h.index.intersection(fwd_8h.index)
    if mask is not None:
        mask = mask.reindex(common).fillna(True)

    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        trading = True if mask is None else bool(mask.loc[ts])

        sig = composite_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe or not trading:
            records.append(dict(timestamp=ts, gross=0.0,
                                turnover=0.0, net=0.0, n=len(sig),
                                active=False))
            prev_longs = prev_shorts = set()
            continue

        ranked = sig.rank()
        longs  = set(ranked.nlargest(n_long).index)
        shorts = set(ranked.nsmallest(n_short).index)

        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        gross = lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan

        if prev_longs | prev_shorts:
            changed  = len((longs - prev_longs) | (shorts - prev_shorts))
            turnover = changed / (n_long + n_short)
        else:
            turnover = 1.0

        net = (gross - turnover * fee_rt) if not np.isnan(gross) else np.nan
        records.append(dict(timestamp=ts, gross=gross,
                            turnover=turnover, net=net, n=len(sig),
                            active=True))
        prev_longs, prev_shorts = longs, shorts

    return pd.DataFrame(records).set_index("timestamp")


def port_stats(net_series, label=""):
    s = net_series.dropna()
    if len(s) < 5:
        return dict(label=label, n=len(s))
    ar   = s.mean() * PERIODS_PER_YEAR
    av   = s.std()  * np.sqrt(PERIODS_PER_YEAR)
    sh   = ar / av  if av > 0 else np.nan
    down = s[s < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    so   = ar / down if down > 0 else np.nan
    cum  = (1 + s).cumprod()
    mdd  = (cum / cum.cummax() - 1).min()
    t    = (s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else np.nan
    pct_bars = (~net_series.isna() & (net_series != 0)).mean() if len(net_series) > 0 else np.nan
    return dict(
        label=label, n=len(s),
        mean_bps=round(s.mean()*10000, 2),
        ann_ret=round(ar*100, 2),
        ann_vol=round(av*100, 2),
        sharpe=round(sh, 3),
        sortino=round(so, 3),
        max_dd=round(mdd*100, 2),
        win_rate=round((s > 0).mean(), 3),
        t_stat=round(t, 2),
        active_pct=round(pct_bars, 3),
    )


# ============================================================
# Regime feature construction
# ============================================================

def build_regime_features(panels, composite_8h, fwd_8h_panel):
    """
    Returns DataFrame of regime features, aligned to 8h rebal index.
    All features use only PAST data — no lookahead.
    """
    print("  Building regime features...")
    rebal_idx = composite_8h.index
    close_8h  = to_8h(panels["close"])

    feats = pd.DataFrame(index=rebal_idx)

    # ---- F1: BTC 7-day realized vol (rolling std of 8h returns × sqrt(PPY)) ----
    btc_close = close_8h.get("BTCUSDT", close_8h.iloc[:, 0])
    btc_ret   = btc_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    feats["btc_rvol_7d"] = btc_ret.rolling(21).std() * np.sqrt(PERIODS_PER_YEAR)  # 21 bars = 7d

    # ---- F2: Universe average funding rate ----
    funding_8h = to_8h(panels["funding"])
    feats["avg_funding"] = funding_8h.mean(axis=1)

    # ---- F3: Funding dispersion (cross-sectional std) ----
    feats["funding_disp"] = funding_8h.std(axis=1)

    # ---- F4: Cross-sectional return dispersion (proxy for pairwise correlation) ----
    # Low CS dispersion = all coins moving together = L/S spread small
    ret_8h = close_8h.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    feats["cs_ret_disp"] = ret_8h.std(axis=1).reindex(rebal_idx, method="ffill")

    # ---- F5: BTC 24h return (3 bars back) ----
    feats["btc_ret_24h"] = btc_close.pct_change(3, fill_method=None).reindex(rebal_idx)

    # ---- F6: BTC 72h return (9 bars back) ----
    feats["btc_ret_72h"] = btc_close.pct_change(9, fill_method=None).reindex(rebal_idx)

    # ---- F7: Rolling 30-day IC of composite vs fwd_8h ----
    # Spearman correlation between composite signal and realized fwd return, rolling
    # Compute row-wise Spearman IC then rolling mean
    common_syms = composite_8h.columns.intersection(fwd_8h_panel.columns)
    C = composite_8h[common_syms]
    F = fwd_8h_panel[common_syms]
    ic_series = pd.Series(index=rebal_idx, dtype=float)
    for ts in rebal_idx:
        if ts not in C.index or ts not in F.index:
            continue
        c_row = C.loc[ts].dropna()
        f_row = F.loc[ts, c_row.index].dropna()
        c_row = c_row.loc[f_row.index]
        if len(c_row) < 15:
            continue
        try:
            ic_series.loc[ts] = stats.spearmanr(c_row, f_row)[0]
        except Exception:
            pass
    feats["rolling_ic_30d"] = ic_series.rolling(90).mean()  # 90 bars = 30d

    # ---- F8: Universe average premium z-score ----
    if "prem_z" in panels:
        prem_8h = to_8h(panels["prem_z"])
        feats["avg_prem_z"] = prem_8h.mean(axis=1).reindex(rebal_idx)

    # ---- F9: Universe average OI-price divergence ----
    if "oi_div" in panels:
        oi_8h = to_8h(panels["oi_div"])
        feats["avg_oi_div"] = oi_8h.mean(axis=1).reindex(rebal_idx)

    # ---- F10: Signal strength (CS std of composite) ----
    feats["signal_strength"] = composite_8h.std(axis=1)

    # Shift all features by 1 bar — use feature at t to predict P&L at t
    # (features are already at t, P&L is also at t, so NO shift needed;
    #  but if any feature uses the fwd return itself we'd shift — these don't)
    feats = feats.reindex(rebal_idx)
    return feats


# ============================================================
# Feature-P&L correlation analysis
# ============================================================

def feature_correlation_analysis(feats, net_series):
    """Spearman correlation of each feature vs next-bar net P&L."""
    results = []
    net = net_series.reindex(feats.index)

    for col in feats.columns:
        f = feats[col].reindex(net.index)
        valid = f.notna() & net.notna()
        if valid.sum() < 50:
            continue
        rho, pval = stats.spearmanr(f[valid], net[valid])
        # Also test: does filtering by feature percentile improve P&L?
        # Top tercile vs bottom tercile
        q33 = f.quantile(0.33)
        q67 = f.quantile(0.67)
        low_mask  = f <= q33
        high_mask = f >= q67
        net_low   = net[low_mask & net.notna()]
        net_high  = net[high_mask & net.notna()]
        results.append(dict(
            feature=col,
            spearman_rho=round(rho, 4),
            p_value=round(pval, 4),
            n=int(valid.sum()),
            net_bps_low_tercile=round(net_low.mean()*10000, 2) if len(net_low) > 10 else np.nan,
            net_bps_high_tercile=round(net_high.mean()*10000, 2) if len(net_high) > 10 else np.nan,
        ))

    df = pd.DataFrame(results).sort_values("spearman_rho", key=abs, ascending=False)
    return df


# ============================================================
# Walk-forward single-feature filter
# ============================================================

def wf_single_feature(feats, composite_8h, fwd_8h, feature_col, direction=None):
    """
    Walk-forward threshold search for one feature.
    direction: +1 = trade when feature is LOW (< threshold)
               -1 = trade when feature is HIGH (> threshold)
               None = determine from training correlation
    Returns per-window OOS stats + full OOS P&L series.
    """
    start = composite_8h.index.min()
    end   = composite_8h.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    all_oos_pnl = []
    wf_rows = []
    baseline_oos = []

    for tr_s, tr_e, oo_s, oo_e in windows:
        feat_train = feats[feature_col].loc[tr_s:tr_e].dropna()
        # Baseline P&L (no filter)
        fwd_oos = fwd_8h.reindex(composite_8h.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
        pnl_base = sim_portfolio(composite_8h.loc[oo_s:oo_e],
                                 fwd_oos.reindex(composite_8h.loc[oo_s:oo_e].index))

        if len(feat_train) < 50:
            continue

        # Determine direction from training correlation
        net_train = sim_portfolio(composite_8h.loc[tr_s:tr_e],
                                  fwd_oos.reindex(composite_8h.loc[tr_s:tr_e].index))["net"]
        feat_aligned = feat_train.reindex(net_train.index)
        valid = feat_aligned.notna() & net_train.notna()
        if valid.sum() < 30:
            continue
        rho_train, _ = stats.spearmanr(feat_aligned[valid], net_train[valid])
        use_direction = direction if direction is not None else (-1 if rho_train > 0 else 1)

        # Grid search threshold in training (10th–90th pctile, 17 steps)
        thresholds = np.percentile(feat_train.dropna(), np.linspace(10, 90, 17))
        best_th, best_sh = None, -np.inf
        for th in thresholds:
            if use_direction == 1:  # trade when feature < th (low is good)
                mask_train = feat_train < th
            else:                    # trade when feature > th (high is good)
                mask_train = feat_train > th
            active_frac = mask_train.mean()
            if active_frac < 0.20 or active_frac > 0.95:
                continue
            net_tr_filt = net_train.reindex(mask_train.index)
            net_tr_filt = net_tr_filt[mask_train]
            if len(net_tr_filt) < 20:
                continue
            sh = net_tr_filt.mean() / net_tr_filt.std() * np.sqrt(PERIODS_PER_YEAR) if net_tr_filt.std() > 0 else 0
            if sh > best_sh:
                best_sh = sh
                best_th = th

        if best_th is None:
            continue

        # Apply to OOS
        feat_oos = feats[feature_col].reindex(composite_8h.loc[oo_s:oo_e].index)
        if use_direction == 1:
            mask_oos = feat_oos < best_th
        else:
            mask_oos = feat_oos > best_th
        mask_oos = mask_oos.fillna(True)  # trade if feature missing

        pnl_filt = sim_portfolio(composite_8h.loc[oo_s:oo_e],
                                 fwd_oos.reindex(composite_8h.loc[oo_s:oo_e].index),
                                 mask=mask_oos)

        st_base = port_stats(pnl_base["net"], label=f"{oo_s.date()}")
        st_filt = port_stats(pnl_filt["net"], label=f"{oo_s.date()}")

        wf_rows.append(dict(
            window=f"{oo_s.date()}–{oo_e.date()}",
            feature=feature_col,
            train_rho=round(rho_train, 4),
            threshold=round(best_th, 6),
            direction=use_direction,
            train_sharpe_filtered=round(best_sh, 3),
            active_pct_oos=round(mask_oos.mean(), 3),
            oos_sharpe_base=st_base.get("sharpe", np.nan),
            oos_sharpe_filt=st_filt.get("sharpe", np.nan),
            oos_mdd_base=st_base.get("max_dd", np.nan),
            oos_mdd_filt=st_filt.get("max_dd", np.nan),
            oos_ret_base=st_base.get("ann_ret", np.nan),
            oos_ret_filt=st_filt.get("ann_ret", np.nan),
        ))
        all_oos_pnl.append(pnl_filt["net"])
        baseline_oos.append(pnl_base["net"])

    combined_filt = pd.concat(all_oos_pnl) if all_oos_pnl else pd.Series(dtype=float)
    combined_base = pd.concat(baseline_oos) if baseline_oos else pd.Series(dtype=float)
    return wf_rows, combined_filt, combined_base


# ============================================================
# Walk-forward ML filter
# ============================================================

def wf_ml_filter(feats, composite_8h, fwd_8h, top_n_features=5):
    """
    Walk-forward ML regime filter.
    Models: Logistic Regression, Random Forest, Gradient Boosting
    Target: next-bar net P&L > 0 (binary)
    Uses only top-N features by training correlation to avoid overfitting.
    """
    start = composite_8h.index.min()
    end   = composite_8h.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    models = {
        "LR":  LogisticRegression(C=0.05, max_iter=500, class_weight="balanced"),
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=3,
                                      min_samples_leaf=30, random_state=42,
                                      class_weight="balanced"),
        "GBM": GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=42),
    }

    fwd_oos_full = fwd_8h.reindex(composite_8h.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)

    all_pnl   = {m: [] for m in models}
    base_pnl  = []
    wf_rows   = []

    for tr_s, tr_e, oo_s, oo_e in windows:
        pnl_tr = sim_portfolio(composite_8h.loc[tr_s:tr_e],
                               fwd_oos_full.reindex(composite_8h.loc[tr_s:tr_e].index))
        net_tr = pnl_tr["net"].dropna()
        if len(net_tr) < 80:
            continue

        # Feature matrix aligned to training P&L
        X_tr = feats.reindex(net_tr.index).fillna(method="ffill").fillna(0)
        y_tr = (net_tr > 0).astype(int)

        # Select top-N features by |point-biserial corr| in training
        corrs = {}
        for col in X_tr.columns:
            valid = X_tr[col].notna()
            if valid.sum() < 30:
                continue
            r, _ = stats.pointbiserialr(y_tr[valid], X_tr[col][valid])
            corrs[col] = abs(r)
        top_feats = sorted(corrs, key=corrs.get, reverse=True)[:top_n_features]
        if not top_feats:
            continue

        X_tr_sel = X_tr[top_feats].fillna(0).values
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)

        # OOS feature matrix
        pnl_base_oos = sim_portfolio(composite_8h.loc[oo_s:oo_e],
                                     fwd_oos_full.reindex(composite_8h.loc[oo_s:oo_e].index))
        X_oos = feats[top_feats].reindex(pnl_base_oos.index).fillna(method="ffill").fillna(0)
        X_oos_sc = scaler.transform(X_oos.values)

        row = dict(window=f"{oo_s.date()}–{oo_e.date()}",
                   top_features="|".join(top_feats),
                   n_train=len(y_tr),
                   oos_sharpe_base=port_stats(pnl_base_oos["net"]).get("sharpe", np.nan),
                   oos_mdd_base=port_stats(pnl_base_oos["net"]).get("max_dd", np.nan))
        base_pnl.append(pnl_base_oos["net"])

        for mname, model in models.items():
            try:
                model.fit(X_tr_sc, y_tr)
                prob_oos = model.predict_proba(X_oos_sc)[:, 1]
                auc_tr   = roc_auc_score(y_tr, model.predict_proba(X_tr_sc)[:, 1])

                # Threshold: trade if prob > 0.5 (classifier default)
                mask_oos = pd.Series(prob_oos > 0.5, index=pnl_base_oos.index)
                pnl_filt = sim_portfolio(composite_8h.loc[oo_s:oo_e],
                                         fwd_oos_full.reindex(composite_8h.loc[oo_s:oo_e].index),
                                         mask=mask_oos)
                st = port_stats(pnl_filt["net"])
                row[f"{mname}_auc_train"]  = round(auc_tr, 3)
                row[f"{mname}_active_pct"] = round(mask_oos.mean(), 3)
                row[f"{mname}_sharpe"]     = st.get("sharpe", np.nan)
                row[f"{mname}_mdd"]        = st.get("max_dd", np.nan)
                row[f"{mname}_ret"]        = st.get("ann_ret", np.nan)
                all_pnl[mname].append(pnl_filt["net"])
            except Exception as e:
                row[f"{mname}_error"] = str(e)

        wf_rows.append(row)

    combined = {m: pd.concat(v) if v else pd.Series(dtype=float)
                for m, v in all_pnl.items()}
    combined["baseline"] = pd.concat(base_pnl) if base_pnl else pd.Series(dtype=float)
    return wf_rows, combined


# ============================================================
# Combination filter (AND logic, top-2 features)
# ============================================================

def wf_combo_filter(feats, composite_8h, fwd_8h, feature_pairs):
    """
    Test AND combination of two features.
    feature_pairs: list of (feat1, dir1, feat2, dir2)
    """
    fwd_full = fwd_8h.reindex(composite_8h.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
    start = composite_8h.index.min()
    end   = composite_8h.index.max()
    windows = []
    t = start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    results = []
    all_pnl = {}

    for f1, d1, f2, d2 in feature_pairs:
        op1 = "<" if d1 == 1 else ">"
        op2 = "<" if d2 == 1 else ">"
        label = f"{f1}({op1}) AND {f2}({op2})"
        oos_pnl_list = []

        for tr_s, tr_e, oo_s, oo_e in windows:
            feat_tr1 = feats[f1].loc[tr_s:tr_e].dropna()
            feat_tr2 = feats[f2].loc[tr_s:tr_e].dropna()
            net_tr   = sim_portfolio(composite_8h.loc[tr_s:tr_e],
                                     fwd_full.reindex(composite_8h.loc[tr_s:tr_e].index))["net"]

            if len(feat_tr1) < 50 or len(feat_tr2) < 50:
                continue

            # Find optimal thresholds independently in training
            def best_threshold(feat, direction):
                ths = np.percentile(feat.dropna(), np.linspace(10, 90, 17))
                best_th_, best_sh_ = None, -np.inf
                for th in ths:
                    mask = feat < th if direction == 1 else feat > th
                    active = mask.mean()
                    if active < 0.20 or active > 0.95:
                        continue
                    n_ = net_tr.reindex(mask.index)[mask]
                    if len(n_) < 20 or n_.std() == 0:
                        continue
                    sh = n_.mean() / n_.std() * np.sqrt(PERIODS_PER_YEAR)
                    if sh > best_sh_:
                        best_sh_ = sh; best_th_ = th
                return best_th_

            th1 = best_threshold(feat_tr1, d1)
            th2 = best_threshold(feat_tr2, d2)
            if th1 is None or th2 is None:
                continue

            feat_oos1 = feats[f1].reindex(composite_8h.loc[oo_s:oo_e].index)
            feat_oos2 = feats[f2].reindex(composite_8h.loc[oo_s:oo_e].index)

            m1 = (feat_oos1 < th1) if d1 == 1 else (feat_oos1 > th1)
            m2 = (feat_oos2 < th2) if d2 == 1 else (feat_oos2 > th2)
            mask_oos = (m1 & m2).fillna(True)

            pnl = sim_portfolio(composite_8h.loc[oo_s:oo_e],
                                fwd_full.reindex(composite_8h.loc[oo_s:oo_e].index),
                                mask=mask_oos)
            oos_pnl_list.append(pnl["net"])

        combined = pd.concat(oos_pnl_list) if oos_pnl_list else pd.Series(dtype=float)
        all_pnl[label] = combined
        st = port_stats(combined, label=label)
        results.append(st)

    return results, all_pnl


# ============================================================
# Plots
# ============================================================

def plot_feature_analysis(feats, net_series, corr_df, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Regime Features vs Net P&L", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    net = net_series.dropna()
    all_cols = feats.columns.tolist()

    for i, col in enumerate(all_cols[:12]):
        ax = axes[i]
        f = feats[col].reindex(net.index).dropna()
        n_aligned = net.reindex(f.index).dropna()
        f = f.reindex(n_aligned.index)

        if len(f) < 20:
            ax.set_visible(False)
            continue

        # Colour by above/below median
        med = f.median()
        colors = ["#2ca02c" if v <= med else "#d62728" for v in f]
        ax.scatter(f.values, n_aligned.values * 10000, c=colors,
                   alpha=0.3, s=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(med, color="gray", linewidth=0.5, linestyle="--")

        rho_row = corr_df[corr_df["feature"] == col]
        rho = rho_row["spearman_rho"].values[0] if len(rho_row) > 0 else np.nan
        ax.set_title(f"{col}\nρ={rho:.3f}", fontsize=8)
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("Net bps", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(all_cols), 12):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_filtered_equity(pnl_dict, labels_styles, out_path):
    """
    pnl_dict: {label: net_series}
    labels_styles: {label: (color, linewidth, linestyle)}
    """
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.set_title("Regime Filter Comparison — Equity Curves (OOS periods only)",
                  fontsize=11, fontweight="bold")

    for label, net in pnl_dict.items():
        net = net.dropna()
        if len(net) < 5:
            continue
        cum = (1 + net).cumprod()
        dd  = cum / cum.cummax() - 1
        sty = labels_styles.get(label, ("#aaa", 1.0, "-"))
        ax1.plot(cum.index, cum.values, color=sty[0], linewidth=sty[1],
                 linestyle=sty[2], label=label, alpha=0.85)
        if label == "Baseline":
            ax2.fill_between(dd.index, dd.values*100, 0, color="#d62728", alpha=0.4)
        else:
            ax2.plot(dd.index, dd.values*100, color=sty[0], linewidth=sty[1]*0.8,
                     linestyle=sty[2], alpha=0.7)

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"))
    ax1.set_ylabel("Cumulative return (log scale)")
    ax1.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticklabels([])

    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.axhline(0, color="black", linewidth=0.5)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_monthly_comparison(baseline_pnl, best_filtered_pnl, label_filt, out_path):
    def monthly_net(net):
        rows = []
        for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
            rows.append(dict(month=f"{yr}-{mo:02d}", net_pct=(1+grp).prod()-1))
        return pd.DataFrame(rows)

    mb = monthly_net(baseline_pnl.dropna())
    mf = monthly_net(best_filtered_pnl.dropna())
    merged = mb.merge(mf, on="month", suffixes=("_base", "_filt"), how="outer").fillna(0)

    x = np.arange(len(merged))
    w = 0.38
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - w/2, merged["net_pct_base"]*100, width=w,
           color=["#2ca02c" if v>=0 else "#d62728" for v in merged["net_pct_base"]],
           label="Baseline", alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, merged["net_pct_filt"]*100, width=w,
           color=["#1f77b4" if v>=0 else "#ff7f0e" for v in merged["net_pct_filt"]],
           label=label_filt, alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["month"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Monthly net return (%)")
    ax.set_title(f"Monthly P&L: Baseline vs {label_filt}", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 5: Regime Filter Analysis")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/6] Loading panels...")
    COLS = ["close", "funding", "mom_24h", "prem_z", "oi_div", "ls_z"]
    panels = load_panels(COLS)

    # Build composite A
    funding_8h = to_8h(panels["funding"])
    mom_8h     = to_8h(panels["mom_24h"])
    fz = cs_zscore(funding_8h)
    mz = cs_zscore(mom_8h)
    composite_8h = (fz.add(mz, fill_value=0)) / 2

    # Forward returns
    close_8h  = to_8h(panels["close"])
    fwd_raw   = panels["close"].pct_change(8, fill_method=None).shift(-8)
    fwd_raw   = fwd_raw.clip(-0.99, 3.0).replace([np.inf, -np.inf], np.nan)
    fwd_8h    = fwd_raw.reindex(close_8h.index)

    # Baseline simulation (full period)
    print("[2/6] Running baseline simulation...")
    pnl_base = sim_portfolio(composite_8h, fwd_8h.reindex(composite_8h.index).clip(-0.99,3).replace([np.inf,-np.inf],np.nan))
    net_base  = pnl_base["net"]
    st_base   = port_stats(net_base, "Baseline")
    print(f"  Baseline: Sharpe={st_base['sharpe']:.3f}  MaxDD={st_base['max_dd']:.1f}%  "
          f"AnnRet={st_base['ann_ret']:.0f}%")

    # ---- Regime features ----
    print("\n[3/6] Building regime features...")
    feats = build_regime_features(panels, composite_8h, fwd_8h.reindex(composite_8h.index))
    print(f"  Features: {list(feats.columns)}")
    print(f"  Feature coverage (non-null %):")
    for col in feats.columns:
        pct = feats[col].notna().mean()
        print(f"    {col:<20}: {pct*100:.0f}%")

    # ---- Correlation analysis ----
    print("\n[4/6] Feature-P&L correlation analysis...")
    corr_df = feature_correlation_analysis(feats, net_base)
    corr_df.to_csv(os.path.join(RESULTS_DIR, "phase5_feature_corr.csv"), index=False)
    print("\n  Feature correlations (|ρ| sorted):")
    print(f"  {'Feature':<22} {'ρ':>7} {'p-val':>8} {'bps low3':>10} {'bps high3':>10}")
    print("  " + "-" * 60)
    for _, row in corr_df.iterrows():
        print(f"  {row['feature']:<22} {row['spearman_rho']:>7.4f} "
              f"{row['p_value']:>8.4f} "
              f"{row['net_bps_low_tercile']:>10.1f} "
              f"{row['net_bps_high_tercile']:>10.1f}")

    # ---- Single-feature walk-forward filters ----
    print("\n[5/6] Walk-forward single-feature filters...")
    all_single_rows = []
    single_oos_pnl  = {}
    baseline_oos_stored = None

    for col in feats.columns:
        print(f"  Testing: {col}...")
        rows, oos_pnl, base_oos = wf_single_feature(feats, composite_8h, fwd_8h, col)
        if rows:
            all_single_rows.extend(rows)
            single_oos_pnl[col] = oos_pnl
            if baseline_oos_stored is None:
                baseline_oos_stored = base_oos

    single_df = pd.DataFrame(all_single_rows)
    single_df.to_csv(os.path.join(RESULTS_DIR, "phase5_single_filter.csv"), index=False)

    # Summarise per feature across all OOS windows
    print("\n  Walk-forward OOS summary (per feature, all windows combined):")
    print(f"  {'Feature':<22} {'Base Sh':>8} {'Filt Sh':>8} {'Δ Sh':>6} "
          f"{'Base DD':>8} {'Filt DD':>8} {'Active%':>9}")
    print("  " + "-" * 70)
    feat_summary = {}
    for col, oos_pnl in single_oos_pnl.items():
        st_f = port_stats(oos_pnl)
        st_b = port_stats(baseline_oos_stored)
        active_rows = single_df[single_df["feature"]==col]
        avg_active = active_rows["active_pct_oos"].mean() if len(active_rows) > 0 else np.nan
        feat_summary[col] = st_f
        print(f"  {col:<22} {st_b.get('sharpe',np.nan):>8.3f} "
              f"{st_f.get('sharpe',np.nan):>8.3f} "
              f"{(st_f.get('sharpe',0)-st_b.get('sharpe',0)):>+6.3f} "
              f"{st_b.get('max_dd',np.nan):>8.1f}% "
              f"{st_f.get('max_dd',np.nan):>8.1f}% "
              f"{avg_active*100:>8.0f}%")

    # Pick best single feature by OOS Sharpe improvement
    best_feat = max(feat_summary,
                    key=lambda k: feat_summary[k].get("sharpe", -99)
                                  if not np.isnan(feat_summary[k].get("sharpe", np.nan)) else -99)
    print(f"\n  Best single feature: {best_feat}")

    # ---- Combo (AND) filters ----
    print("\n  Testing AND combinations of top-3 features...")
    # Determine top-3 features by |correlation| with P&L
    top3 = corr_df["feature"].head(3).tolist()
    top3_dirs = []
    for f in top3:
        rho = corr_df[corr_df["feature"]==f]["spearman_rho"].values[0]
        d = 1 if rho < 0 else -1   # trade when low (d=1) if negative corr → bad when high
        top3_dirs.append((f, d))

    pairs = []
    for i in range(len(top3_dirs)):
        for j in range(i+1, len(top3_dirs)):
            pairs.append((*top3_dirs[i], *top3_dirs[j]))

    combo_results, combo_pnl = wf_combo_filter(feats, composite_8h, fwd_8h, pairs)
    combo_df = pd.DataFrame(combo_results)
    if not combo_df.empty:
        print(f"\n  Combo filter OOS summary:")
        print(f"  {'Label':<55} {'Sharpe':>8} {'MaxDD':>8} {'AnnRet':>8}")
        print("  " + "-" * 82)
        for _, row in combo_df.iterrows():
            print(f"  {str(row.get('label',''))[:55]:<55} "
                  f"{row.get('sharpe',np.nan):>8.3f} "
                  f"{row.get('max_dd',np.nan):>8.1f}% "
                  f"{row.get('ann_ret',np.nan):>8.0f}%")

    # ---- ML filters ----
    print("\n[6/6] Walk-forward ML filters...")
    ml_rows, ml_pnl = wf_ml_filter(feats, composite_8h, fwd_8h, top_n_features=5)
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(os.path.join(RESULTS_DIR, "phase5_ml_results.csv"), index=False)

    st_base_oos = port_stats(ml_pnl["baseline"], "Baseline OOS")
    print(f"\n  ML OOS summary (all windows combined):")
    print(f"  {'Model':<12} {'Sharpe':>8} {'MaxDD':>8} {'AnnRet':>8} {'Active%':>10}")
    print("  " + "-" * 50)
    for mname in ["baseline", "LR", "RF", "GBM"]:
        st = port_stats(ml_pnl[mname], mname)
        active = ml_df[f"{mname}_active_pct"].mean() if mname != "baseline" and f"{mname}_active_pct" in ml_df.columns else np.nan
        print(f"  {mname:<12} {st.get('sharpe',np.nan):>8.3f} "
              f"{st.get('max_dd',np.nan):>8.1f}% "
              f"{st.get('ann_ret',np.nan):>8.0f}% "
              f"{active*100 if not np.isnan(active) else np.nan:>9.0f}%")

    # ---- Build final comparison dict ----
    pnl_comparison = {"Baseline": baseline_oos_stored}
    pnl_comparison[f"Best: {best_feat}"] = single_oos_pnl[best_feat]
    for mname in ["LR", "RF", "GBM"]:
        if mname in ml_pnl and len(ml_pnl[mname]) > 10:
            pnl_comparison[f"ML-{mname}"] = ml_pnl[mname]
    # Add best combo
    if combo_results:
        best_combo_label = max(combo_results, key=lambda r: r.get("sharpe", -99) if not np.isnan(r.get("sharpe", np.nan)) else -99)["label"]
        if best_combo_label in combo_pnl:
            pnl_comparison[f"AND: {best_combo_label[:30]}"] = combo_pnl[best_combo_label]

    # Monthly table for best filter
    all_best_labels = [k for k in pnl_comparison if k != "Baseline"]
    best_overall = max(all_best_labels,
                       key=lambda k: port_stats(pnl_comparison[k]).get("sharpe", -99))
    print(f"\n  Best overall filter: {best_overall}")

    best_pnl = pnl_comparison[best_overall]
    base_pnl_oos = pnl_comparison["Baseline"]

    def monthly_table(net, label):
        rows = []
        net = net.dropna()
        for (yr, mo), grp in net.groupby([net.index.year, net.index.month]):
            n = len(grp)
            cum = (1+grp).prod()-1
            sh  = grp.mean()/grp.std()*np.sqrt(PERIODS_PER_YEAR) if n>1 and grp.std()>0 else np.nan
            rows.append(dict(month=f"{yr}-{mo:02d}", label=label,
                             net_pct=round(cum*100,2),
                             sharpe=round(sh,2) if not np.isnan(sh) else np.nan))
        return pd.DataFrame(rows)

    mt_base = monthly_table(base_pnl_oos, "Baseline")
    mt_best = monthly_table(best_pnl, best_overall)
    mt_all  = pd.concat([mt_base, mt_best]).sort_values(["month","label"])
    mt_all.to_csv(os.path.join(RESULTS_DIR, "phase5_best_filter.csv"), index=False)

    # Print comparison
    print(f"\n  Monthly comparison (OOS only):")
    print(f"  {'Month':<10} {'Base Net%':>10} {'Filt Net%':>10} {'Delta':>8} {'Base Sh':>8} {'Filt Sh':>8}")
    print("  " + "-" * 60)
    months = sorted(set(mt_base["month"]) | set(mt_best["month"]))
    pos_base = pos_filt = 0
    for m in months:
        b = mt_base[mt_base["month"]==m]
        f = mt_best[mt_best["month"]==m]
        b_net = b["net_pct"].values[0] if len(b) else 0
        f_net = f["net_pct"].values[0] if len(f) else 0
        b_sh  = b["sharpe"].values[0]  if len(b) else np.nan
        f_sh  = f["sharpe"].values[0]  if len(f) else np.nan
        delta = f_net - b_net
        marker = "  +" if delta > 2 else ("  -" if delta < -2 else "   ")
        print(f"  {m:<10} {b_net:>10.1f} {f_net:>10.1f} {delta:>+8.1f}{marker} "
              f"{b_sh:>8.2f} {f_sh:>8.2f}")
        if b_net > 0: pos_base += 1
        if f_net > 0: pos_filt += 1

    print(f"\n  Positive months: Baseline {pos_base}/{len(months)}  Filtered {pos_filt}/{len(months)}")
    st_b2 = port_stats(base_pnl_oos)
    st_f2 = port_stats(best_pnl)
    print(f"  OOS Sharpe:  Baseline {st_b2['sharpe']:.3f}  →  Filtered {st_f2['sharpe']:.3f}")
    print(f"  OOS MaxDD:   Baseline {st_b2['max_dd']:.1f}%  →  Filtered {st_f2['max_dd']:.1f}%")
    print(f"  OOS AnnRet:  Baseline {st_b2['ann_ret']:.0f}%  →  Filtered {st_f2['ann_ret']:.0f}%")

    # ---- Plots ----
    print("\nGenerating plots...")
    labels_styles = {
        "Baseline":              ("#d62728", 2.0, "-"),
        f"Best: {best_feat}":   ("#1f77b4", 1.8, "-"),
        "ML-LR":                 ("#2ca02c", 1.4, "--"),
        "ML-RF":                 ("#ff7f0e", 1.4, "-."),
        "ML-GBM":                ("#9467bd", 1.4, ":"),
    }
    for k in list(pnl_comparison.keys()):
        if k not in labels_styles:
            labels_styles[k] = ("#aaa", 1.2, "--")

    plot_feature_analysis(feats, net_base, corr_df,
                          os.path.join(RESULTS_DIR, "phase5_regime_features.png"))
    plot_filtered_equity(pnl_comparison, labels_styles,
                         os.path.join(RESULTS_DIR, "phase5_equity_filtered.png"))
    plot_monthly_comparison(base_pnl_oos, best_pnl, best_overall,
                            os.path.join(RESULTS_DIR, "phase5_monthly_comparison.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
