#!/usr/bin/env python3
"""
Long Leg ML — Entry Decision + Exit Optimization
=================================================
Two models:
  A. ENTRY DECISION: At the moment the short exit ML fires, should we go 2x (long)
     or 1x (short-only)? Uses features observable at that exact moment.
  B. EXIT TIMING: Once long, when should we sell? Same tick-level approach as
     short exit ML but trained on recovery phase data.

Usage:
    python3 research_long_leg_ml.py
"""

import json
import time as _time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOCAL_DATA_DIR = Path("charts_settlement")
TICK_MS = 100
MAX_POST_MS = 60000
TAKER_FEE_BPS = 10
MAKER_FEE_BPS = 4
LONG_HOLD_FIXED_MS = 20000  # baseline fixed hold for comparison

# ═══════════════════════════════════════════════════════════════════════
# PART A: LONG ENTRY DECISION — features at short-exit moment
# ═══════════════════════════════════════════════════════════════════════

def build_entry_decision_dataset():
    """For each settlement, extract features at the bottom tick and label
    whether the long leg would have been profitable."""
    from research_position_sizing import parse_last_ob_before_settlement, compute_slippage_bps

    jsonl_files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    print(f"Building entry decision dataset from {len(jsonl_files)} files...")
    t0 = _time.time()

    rows = []
    for fi, fp in enumerate(jsonl_files):
        ob = parse_last_ob_before_settlement(fp)
        if ob is None:
            continue

        mid = ob['mid_price']
        bids = ob['bids']
        asks = ob['asks']
        spread_bps = ob['spread_bps']
        fr_bps = ob.get('fr_abs_bps', 0)
        symbol = ob['symbol']
        depth_20 = sum(p * q for p, q in bids if (mid - p) / mid * 10000 <= 20)

        if depth_20 < 2000 or spread_bps > 8:
            continue

        # Parse trades + OB
        trades, ob1_data, tickers = [], [], []
        with open(fp) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except:
                    continue
                t_ms = m.get('_t_ms', 0)
                topic = m.get('topic', '')
                data = m.get('data', {})

                if 'publicTrade' in topic:
                    for tr in (data if isinstance(data, list) else [data]):
                        p = float(tr.get('p', 0))
                        q = float(tr.get('v', 0))
                        s = tr.get('S', '')
                        trades.append((t_ms, p, q, s, p * q))
                elif topic.startswith('orderbook.1.'):
                    b = data.get('b', [])
                    a = data.get('a', [])
                    if b and a:
                        ob1_data.append((t_ms, float(b[0][0]), float(b[0][1]),
                                         float(a[0][0]), float(a[0][1])))
                elif 'tickers' in topic:
                    fr_val = float(data.get('fundingRate', 0))
                    tickers.append((t_ms, fr_val))

        pre_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t < 0]
        post_trades = sorted([(t, p, q, s, n) for t, p, q, s, n in trades if t >= 0],
                             key=lambda x: x[0])

        if not pre_trades or len(post_trades) < 10:
            continue

        ref_price = pre_trades[-1][1]
        bps = lambda p: (p / ref_price - 1) * 10000

        # Pre-settlement stats
        pre_10s = [(t, p, q, s, n) for t, p, q, s, n in pre_trades if t >= -10000]
        pre_vol_rate = sum(n for _, _, _, _, n in pre_10s) / 10.0 if pre_10s else 1.0
        pre_trade_rate = len(pre_10s) / 10.0

        # Build 100ms price bins
        post_times = np.array([t for t, _, _, _, _ in post_trades])
        post_prices_bps = np.array([bps(p) for _, p, _, _, _ in post_trades])
        post_sides = np.array([1 if s == 'Sell' else 0 for _, _, _, s, _ in post_trades])
        post_notionals = np.array([n for _, _, _, _, n in post_trades])
        post_sizes = np.array([q for _, _, q, _, _ in post_trades])

        # Find bottom (T+1s to T+30s)
        price_bins = {}
        for t_ms, p, _, _, _ in post_trades:
            if 0 <= t_ms <= 60000:
                bk = int(t_ms / 100) * 100
                price_bins[bk] = bps(p)

        bottom_bps = None
        bottom_t = None
        for t_ms in sorted(price_bins.keys()):
            if t_ms < 1000 or t_ms > 30000:
                continue
            if bottom_bps is None or price_bins[t_ms] < bottom_bps:
                bottom_bps = price_bins[t_ms]
                bottom_t = t_ms

        if bottom_bps is None or bottom_t is None:
            continue

        drop_bps = -bottom_bps  # positive = how much it dropped

        # === FEATURES AT THE BOTTOM MOMENT ===
        # These are what we'd observe when the ML says "bottom reached"
        feat = {}
        feat['symbol'] = symbol
        feat['settle_id'] = fp.stem

        # Timing
        feat['bottom_t_s'] = bottom_t / 1000.0
        feat['bottom_t_ms'] = bottom_t
        feat['pct_of_window_elapsed'] = bottom_t / MAX_POST_MS

        # Drop characteristics
        feat['drop_bps'] = drop_bps
        feat['drop_rate_bps_per_s'] = drop_bps / (bottom_t / 1000) if bottom_t > 0 else 0

        # FR
        pre_tk = [tk for tk in tickers if tk[0] < 0]
        feat['fr_bps'] = pre_tk[-1][1] * 10000 if pre_tk else 0
        feat['fr_abs_bps'] = abs(feat['fr_bps'])

        # Orderbook at settlement
        feat['depth_20'] = depth_20
        feat['spread_bps'] = spread_bps
        feat['ob_bid_depth'] = sum(p * q for p, q in bids[:5])
        feat['ob_ask_depth'] = sum(p * q for p, q in asks[:5])

        # Price action in the 2s window AROUND the bottom
        mask_around = (post_times >= bottom_t - 2000) & (post_times <= bottom_t)
        if mask_around.sum() >= 2:
            window_prices = post_prices_bps[mask_around]
            feat['velocity_2s_at_bottom'] = window_prices[-1] - window_prices[0]
            feat['price_range_2s_at_bottom'] = window_prices.max() - window_prices.min()
            feat['price_std_2s_at_bottom'] = window_prices.std()
        else:
            feat['velocity_2s_at_bottom'] = 0
            feat['price_range_2s_at_bottom'] = 0
            feat['price_std_2s_at_bottom'] = 0

        # Price action in the 1s window AFTER the bottom (early recovery signal)
        mask_after_1s = (post_times > bottom_t) & (post_times <= bottom_t + 1000)
        if mask_after_1s.sum() >= 2:
            after_prices = post_prices_bps[mask_after_1s]
            feat['velocity_1s_after_bottom'] = after_prices[-1] - after_prices[0]
            feat['max_recovery_1s'] = after_prices.max() - bottom_bps
        else:
            feat['velocity_1s_after_bottom'] = 0
            feat['max_recovery_1s'] = 0

        # Trade flow at bottom (sell exhaustion signals)
        mask_bottom_2s = (post_times >= bottom_t - 2000) & (post_times <= bottom_t)
        if mask_bottom_2s.sum() > 0:
            bottom_sides = post_sides[mask_bottom_2s]
            bottom_notionals = post_notionals[mask_bottom_2s]
            bottom_sizes = post_sizes[mask_bottom_2s]
            total_vol = bottom_notionals.sum()
            sell_vol = bottom_notionals[bottom_sides == 1].sum()
            feat['sell_ratio_at_bottom'] = sell_vol / total_vol if total_vol > 0 else 0.5
            feat['trade_count_at_bottom'] = mask_bottom_2s.sum()
            feat['vol_rate_at_bottom'] = total_vol / 2.0
            feat['avg_size_at_bottom'] = bottom_sizes.mean()
            # Sell exhaustion: is sell ratio dropping?
            mask_b1 = (post_times >= bottom_t - 2000) & (post_times <= bottom_t - 1000)
            mask_b2 = (post_times >= bottom_t - 1000) & (post_times <= bottom_t)
            if mask_b1.sum() > 0 and mask_b2.sum() > 0:
                sr1 = post_notionals[mask_b1 & (post_sides == 1)].sum() / max(post_notionals[mask_b1].sum(), 1)
                sr2 = post_notionals[mask_b2 & (post_sides == 1)].sum() / max(post_notionals[mask_b2].sum(), 1)
                feat['sell_ratio_change'] = sr2 - sr1  # negative = sell pressure decreasing
            else:
                feat['sell_ratio_change'] = 0
        else:
            feat['sell_ratio_at_bottom'] = 0.5
            feat['trade_count_at_bottom'] = 0
            feat['vol_rate_at_bottom'] = 0
            feat['avg_size_at_bottom'] = 0
            feat['sell_ratio_change'] = 0

        # Volume surge vs pre-settlement
        feat['vol_surge'] = feat['vol_rate_at_bottom'] / pre_vol_rate if pre_vol_rate > 0 else 1
        feat['trade_surge'] = feat['trade_count_at_bottom'] / (2 * pre_trade_rate) if pre_trade_rate > 0 else 1

        # OB state at bottom time
        ob1_post = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1_data if t >= 0]
        if ob1_post:
            ob1_times = np.array([t for t, _, _, _, _ in ob1_post])
            ob_mask = ob1_times <= bottom_t
            if ob_mask.sum() > 0:
                idx = ob_mask.sum() - 1
                _, bp, bq, ap, aq = ob1_post[idx]
                feat['spread_at_bottom'] = (ap - bp) / ref_price * 10000
                feat['ob1_imbalance_at_bottom'] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0
                feat['ob1_bid_qty_at_bottom'] = bq
                feat['ob1_ask_qty_at_bottom'] = aq
            else:
                feat['spread_at_bottom'] = spread_bps
                feat['ob1_imbalance_at_bottom'] = 0
                feat['ob1_bid_qty_at_bottom'] = 0
                feat['ob1_ask_qty_at_bottom'] = 0
        else:
            feat['spread_at_bottom'] = spread_bps
            feat['ob1_imbalance_at_bottom'] = 0
            feat['ob1_bid_qty_at_bottom'] = 0
            feat['ob1_ask_qty_at_bottom'] = 0

        # Bounce pattern before bottom
        bounce_count = 0
        running_min = 0
        for t_ms in sorted(price_bins.keys()):
            if t_ms > bottom_t:
                break
            p_bps = price_bins[t_ms]
            if p_bps < running_min - 0.5:
                running_min = p_bps
            elif p_bps > running_min + 3:
                bounce_count += 1
                running_min = p_bps  # reset after bounce counted
        feat['bounce_count_to_bottom'] = bounce_count

        # New lows: how many 100ms bins set a new low on the way down
        new_low_count = 0
        running_min = 0
        for t_ms in sorted(price_bins.keys()):
            if t_ms > bottom_t:
                break
            if t_ms < 1000:
                continue
            p_bps = price_bins[t_ms]
            if p_bps < running_min - 0.5:
                running_min = p_bps
                new_low_count += 1
        feat['new_low_count'] = new_low_count

        # === TARGETS ===
        # What happens in the recovery? Measure at multiple horizons
        limit_fill = 0.54
        rt_fee = 2 * (limit_fill * MAKER_FEE_BPS + (1 - limit_fill) * TAKER_FEE_BPS)
        slip_factor = 0.4
        entry_s = compute_slippage_bps(bids, 500, 'sell', mid_price=mid)
        exit_s = compute_slippage_bps(asks, 500, 'buy', mid_price=mid)
        total_slip = (entry_s['slippage_bps'] + exit_s['slippage_bps']) * slip_factor

        for hold_ms in [5000, 10000, 15000, 20000, 30000]:
            target_t = bottom_t + hold_ms
            rec_p = None
            for t_ms in price_bins:
                if target_t - 500 <= t_ms <= target_t + 500:
                    rec_p = price_bins[t_ms]

            if rec_p is not None:
                recovery_bps = rec_p - bottom_bps
                net_bps = recovery_bps - rt_fee - total_slip
                feat[f'recovery_{hold_ms//1000}s_bps'] = recovery_bps
                feat[f'net_{hold_ms//1000}s_bps'] = net_bps
                feat[f'profitable_{hold_ms//1000}s'] = 1 if net_bps > 0 else 0
            else:
                feat[f'recovery_{hold_ms//1000}s_bps'] = np.nan
                feat[f'net_{hold_ms//1000}s_bps'] = np.nan
                feat[f'profitable_{hold_ms//1000}s'] = np.nan

        # Max recovery in 30s window after bottom
        mask_recovery = (post_times > bottom_t) & (post_times <= bottom_t + 30000)
        if mask_recovery.sum() > 0:
            recovery_prices = post_prices_bps[mask_recovery]
            feat['max_recovery_bps'] = recovery_prices.max() - bottom_bps
            feat['max_recovery_t_ms'] = int(post_times[mask_recovery][np.argmax(recovery_prices)] - bottom_t)
        else:
            feat['max_recovery_bps'] = 0
            feat['max_recovery_t_ms'] = 0

        rows.append(feat)

        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{len(jsonl_files)}] {_time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    print(f"  Built {len(df)} entries in {_time.time()-t0:.1f}s")
    return df


def train_entry_decision(df):
    """Train ML model to predict: should we go long?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    import copy

    print(f"\n{'='*70}")
    print("PART A: LONG ENTRY DECISION ML")
    print(f"{'='*70}")

    target_col = 'profitable_20s'  # primary target: profitable at +20s hold
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"  N={len(df_clean)}, positive rate={df_clean[target_col].mean():.1%}")

    # Feature columns: everything observable at bottom time
    skip_cols = {'symbol', 'settle_id', 'bottom_t_ms'}
    target_cols = {c for c in df_clean.columns if c.startswith(('recovery_', 'net_', 'profitable_', 'max_recovery'))}
    feature_cols = [c for c in df_clean.columns if c not in skip_cols | target_cols]

    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(int)
    symbols = df_clean['symbol'].values
    settle_ids = df_clean['settle_id'].values

    print(f"  Features: {len(feature_cols)}")
    for c in sorted(feature_cols):
        print(f"    {c}")

    # 70/30 temporal split
    unique_settle = df_clean['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = df_clean['settle_id'].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    print(f"  Split: train={train_mask.sum()}, test={test_mask.sum()}")
    print(f"  Train pos rate: {y_tr.mean():.1%}, Test pos rate: {y_te.mean():.1%}")

    models = {
        'LogReg': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        'HGBC': HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, min_samples_leaf=10,
            learning_rate=0.05, l2_regularization=2.0, random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        y_prob_tr = m.predict_proba(X_tr)[:, 1]
        y_prob_te = m.predict_proba(X_te)[:, 1]

        auc_tr = roc_auc_score(y_tr, y_prob_tr) if len(set(y_tr)) > 1 else 0
        auc_te = roc_auc_score(y_te, y_prob_te) if len(set(y_te)) > 1 else 0

        print(f"\n  {name}: Train AUC={auc_tr:.4f}  Test AUC={auc_te:.4f}  Gap={auc_tr-auc_te:.3f}")

        # Try different thresholds
        print(f"  Threshold sweep:")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred = (y_prob_te >= thresh).astype(int)
            n_long = y_pred.sum()
            if n_long == 0:
                print(f"    p>={thresh:.1f}: N=0 (skip all)")
                continue
            tp = ((y_pred == 1) & (y_te == 1)).sum()
            fp = ((y_pred == 1) & (y_te == 0)).sum()
            wr = tp / n_long * 100
            # Compute $ impact
            test_df = df_clean[test_mask].copy()
            test_df['pred'] = y_pred
            long_taken = test_df[test_df['pred'] == 1]
            avg_net = long_taken['net_20s_bps'].mean()
            total_net = long_taken['net_20s_bps'].sum()
            print(f"    p>={thresh:.1f}: N={n_long:3d}/{len(y_te)} "
                  f"WR={wr:5.1f}%  avg_net={avg_net:+.1f}bps  "
                  f"total_net={total_net:+.1f}bps  tp={tp} fp={fp}")

        results[name] = {'auc_train': auc_tr, 'auc_test': auc_te, 'model': m}

    # LOSO
    print(f"\n--- LOSO (symbol) ---")
    best_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, min_samples_leaf=10,
        learning_rate=0.05, l2_regularization=2.0, random_state=42,
    )
    logo = LeaveOneGroupOut()
    try:
        y_pred_loso = cross_val_predict(best_model, X, y, cv=logo, groups=symbols,
                                        method='predict_proba')[:, 1]
        auc_loso = roc_auc_score(y, y_pred_loso)
        print(f"  LOSO AUC: {auc_loso:.4f}")

        # Honest threshold sweep on LOSO predictions
        print(f"\n  LOSO threshold sweep (honest):")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred = (y_pred_loso >= thresh).astype(int)
            n_long = y_pred.sum()
            if n_long == 0:
                continue
            tp = ((y_pred == 1) & (y == 1)).sum()
            wr = tp / n_long * 100
            loso_df = df_clean.copy()
            loso_df['pred'] = y_pred
            taken = loso_df[loso_df['pred'] == 1]
            avg_net = taken['net_20s_bps'].mean()
            total_net = taken['net_20s_bps'].sum()
            print(f"    p>={thresh:.1f}: N={n_long:3d}/{len(y)}  "
                  f"WR={wr:5.1f}%  avg_net={avg_net:+.1f}bps  total_net={total_net:+.1f}bps")
    except Exception as e:
        print(f"  LOSO failed: {e}")

    # Feature importance
    print(f"\n--- Feature Importance (permutation, HGBC) ---")
    from sklearn.inspection import permutation_importance
    m_imp = copy.deepcopy(models['HGBC'])
    m_imp.fit(X_tr, y_tr)
    perm = permutation_importance(m_imp, X_te, y_te, n_repeats=10,
                                  random_state=42, scoring='roc_auc')
    sorted_idx = np.argsort(-perm.importances_mean)
    for i in sorted_idx[:15]:
        imp = perm.importances_mean[i]
        std = perm.importances_std[i]
        print(f"  {feature_cols[i]:35s}: {imp:+.4f} +/- {std:.4f}")

    # Simple rule-based comparison
    print(f"\n--- Rule-Based Comparison ---")
    rules = [
        ('Always long', lambda r: True),
        ('bottom_t <= 10s', lambda r: r['bottom_t_s'] <= 10),
        ('bottom_t <= 15s', lambda r: r['bottom_t_s'] <= 15),
        ('drop >= 40bps', lambda r: r['drop_bps'] >= 40),
        ('drop >= 40 & t<=15s', lambda r: r['drop_bps'] >= 40 and r['bottom_t_s'] <= 15),
        ('FR >= 25 & t<=15s', lambda r: r['fr_abs_bps'] >= 25 and r['bottom_t_s'] <= 15),
        ('velocity_1s_after > 5', lambda r: r['velocity_1s_after_bottom'] > 5),
        ('velocity_1s_after > 10', lambda r: r['velocity_1s_after_bottom'] > 10),
        ('sell_ratio_change < 0', lambda r: r['sell_ratio_change'] < 0),
    ]

    print(f"  {'Rule':40s}  {'N':>4s}  {'WR':>5s}  {'Avg net':>8s}  {'Total net':>9s}")
    print(f"  {'-'*40}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*9}")
    for name, rule in rules:
        mask = df_clean.apply(rule, axis=1)
        sub = df_clean[mask]
        if len(sub) == 0:
            continue
        wr = sub['profitable_20s'].mean() * 100
        avg_net = sub['net_20s_bps'].mean()
        total_net = sub['net_20s_bps'].sum()
        print(f"  {name:40s}  {len(sub):4d}  {wr:4.0f}%  {avg_net:+7.1f}  {total_net:+8.1f}")

    return results, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# PART B: LONG EXIT ML — tick-level recovery features
# ═══════════════════════════════════════════════════════════════════════

def build_long_exit_dataset():
    """Build tick-level features during the RECOVERY phase (after bottom).
    Target: is this the optimal sell point?"""

    jsonl_files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    print(f"\nBuilding long exit dataset from {len(jsonl_files)} files...")
    t0 = _time.time()

    all_rows = []
    for fi, fp in enumerate(jsonl_files):
        rows = _build_recovery_ticks(fp)
        if rows:
            all_rows.extend(rows)

        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{len(jsonl_files)}] {len(all_rows)} ticks, {_time.time()-t0:.1f}s")

    if not all_rows:
        print("  No recovery ticks")
        return None

    df = pd.DataFrame(all_rows)
    print(f"  Built {len(df)} recovery ticks from {df['settle_id'].nunique()} settlements "
          f"in {_time.time()-t0:.1f}s")
    return df


def _build_recovery_ticks(fp):
    """Build tick-level features for the recovery phase of one settlement."""
    trades, ob1_data, tickers = [], [], []
    with open(fp) as f:
        for line in f:
            try:
                m = json.loads(line)
            except:
                continue
            t_ms = m.get('_t_ms', 0)
            topic = m.get('topic', '')
            data = m.get('data', {})

            if 'publicTrade' in topic:
                for tr in (data if isinstance(data, list) else [data]):
                    p = float(tr.get('p', 0))
                    q = float(tr.get('v', 0))
                    s = tr.get('S', '')
                    trades.append((t_ms, p, q, s, p * q))
            elif topic.startswith('orderbook.1.'):
                b = data.get('b', [])
                a = data.get('a', [])
                if b and a:
                    ob1_data.append((t_ms, float(b[0][0]), float(b[0][1]),
                                     float(a[0][0]), float(a[0][1])))
            elif 'tickers' in topic:
                fr_val = float(data.get('fundingRate', 0))
                tickers.append((t_ms, fr_val))

    pre_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t < 0]
    post_trades = sorted([(t, p, q, s, n) for t, p, q, s, n in trades if t >= 0],
                         key=lambda x: x[0])

    if not pre_trades or len(post_trades) < 10:
        return None

    ref_price = pre_trades[-1][1]
    bps = lambda p: (p / ref_price - 1) * 10000

    stem = fp.stem
    symbol = stem.split('_')[0]

    # FR
    pre_tk = [tk for tk in tickers if tk[0] < 0]
    fr_bps = pre_tk[-1][1] * 10000 if pre_tk else 0

    # Pre-settlement stats
    pre_10s = [(t, p, q, s, n) for t, p, q, s, n in pre_trades if t >= -10000]
    pre_vol_rate = sum(n for _, _, _, _, n in pre_10s) / 10.0 if pre_10s else 1.0

    # Arrays
    post_times = np.array([t for t, _, _, _, _ in post_trades])
    post_prices_bps = np.array([bps(p) for _, p, _, _, _ in post_trades])
    post_sides = np.array([1 if s == 'Sell' else 0 for _, _, _, s, _ in post_trades])
    post_notionals = np.array([n for _, _, _, _, n in post_trades])
    post_sizes = np.array([q for _, _, q, _, _ in post_trades])

    # OB
    ob1_post = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1_data if t >= 0]
    ob1_times = np.array([t for t, _, _, _, _ in ob1_post]) if ob1_post else np.array([])

    # Find bottom (same logic)
    price_bins = {}
    for t_ms, p, _, _, _ in post_trades:
        if 0 <= t_ms <= 60000:
            bk = int(t_ms / 100) * 100
            price_bins[bk] = bps(p)

    bottom_bps = None
    bottom_t = None
    for t_ms in sorted(price_bins.keys()):
        if t_ms < 1000 or t_ms > 30000:
            continue
        if bottom_bps is None or price_bins[t_ms] < bottom_bps:
            bottom_bps = price_bins[t_ms]
            bottom_t = t_ms

    if bottom_bps is None:
        return None

    drop_bps = -bottom_bps

    # Recovery phase: tick every 100ms from bottom to bottom+30s
    max_recovery_t = min(bottom_t + 30000, 60000)

    # Find the PEAK recovery (for target computation)
    recovery_peak_bps = bottom_bps
    recovery_peak_t = bottom_t
    for t_ms in sorted(price_bins.keys()):
        if t_ms <= bottom_t or t_ms > max_recovery_t:
            continue
        if price_bins[t_ms] > recovery_peak_bps:
            recovery_peak_bps = price_bins[t_ms]
            recovery_peak_t = t_ms

    rows = []
    running_max_bps = bottom_bps  # track max during recovery
    last_new_high_t = bottom_t

    for tick_t in range(bottom_t + TICK_MS, int(max_recovery_t), TICK_MS):
        mask_up_to = post_times <= tick_t
        if mask_up_to.sum() < 2:
            continue

        # Current state
        mask_recovery = (post_times > bottom_t) & (post_times <= tick_t)
        if mask_recovery.sum() < 1:
            continue

        current_price = post_prices_bps[mask_up_to][-1]
        recovery_so_far = current_price - bottom_bps  # positive = recovered
        time_since_bottom = tick_t - bottom_t

        # Running max during recovery
        recovery_prices = post_prices_bps[mask_recovery]
        new_max = recovery_prices.max()
        if new_max > running_max_bps:
            running_max_bps = new_max
            last_new_high_t = tick_t

        feat = {}
        feat['t_ms'] = tick_t
        feat['time_since_bottom_ms'] = time_since_bottom
        feat['pct_recovery_elapsed'] = time_since_bottom / 30000.0

        # Recovery metrics
        feat['recovery_bps'] = recovery_so_far
        feat['recovery_pct_of_drop'] = recovery_so_far / drop_bps * 100 if drop_bps > 0 else 0
        feat['running_max_bps'] = running_max_bps - bottom_bps
        feat['distance_from_high_bps'] = running_max_bps - current_price  # drawdown from peak
        feat['time_since_new_high_ms'] = tick_t - last_new_high_t

        # Price velocity during recovery
        for window, label in [(500, '500ms'), (1000, '1s'), (2000, '2s')]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_prices = post_prices_bps[w_mask]
            if len(w_prices) >= 2:
                feat[f'velocity_{label}'] = w_prices[-1] - w_prices[0]
            else:
                feat[f'velocity_{label}'] = 0

        # Acceleration
        mask_early = (post_times > tick_t - 1000) & (post_times <= tick_t - 500)
        mask_late = (post_times > tick_t - 500) & (post_times <= tick_t)
        if mask_early.sum() >= 1 and mask_late.sum() >= 1:
            v_early = post_prices_bps[mask_early][-1] - post_prices_bps[mask_early][0] if mask_early.sum() > 1 else 0
            v_late = post_prices_bps[mask_late][-1] - post_prices_bps[mask_late][0] if mask_late.sum() > 1 else 0
            feat['acceleration'] = v_late - v_early
        else:
            feat['acceleration'] = 0

        # Trade flow
        for window, label in [(500, '500ms'), (1000, '1s'), (2000, '2s')]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_sides = post_sides[w_mask]
            w_notionals = post_notionals[w_mask]
            total_vol = w_notionals.sum()
            buy_vol = w_notionals[w_sides == 0].sum()
            feat[f'buy_ratio_{label}'] = buy_vol / total_vol if total_vol > 0 else 0.5
            feat[f'vol_rate_{label}'] = total_vol / (window / 1000)
            feat[f'trade_count_{label}'] = w_mask.sum()

        # Buy pressure momentum
        mask_1s_early = (post_times > tick_t - 2000) & (post_times <= tick_t - 1000)
        mask_1s_late = (post_times > tick_t - 1000) & (post_times <= tick_t)
        if mask_1s_early.sum() > 0 and mask_1s_late.sum() > 0:
            br_early = post_notionals[mask_1s_early & (post_sides == 0)].sum() / max(post_notionals[mask_1s_early].sum(), 1)
            br_late = post_notionals[mask_1s_late & (post_sides == 0)].sum() / max(post_notionals[mask_1s_late].sum(), 1)
            feat['buy_ratio_change'] = br_late - br_early
        else:
            feat['buy_ratio_change'] = 0

        # Orderbook
        if len(ob1_times) > 0:
            ob_mask = ob1_times <= tick_t
            if ob_mask.sum() > 0:
                idx = ob_mask.sum() - 1
                _, bp, bq, ap, aq = ob1_post[idx]
                feat['spread_bps'] = (ap - bp) / ref_price * 10000
                feat['ob1_imbalance'] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0
            else:
                feat['spread_bps'] = 0
                feat['ob1_imbalance'] = 0
        else:
            feat['spread_bps'] = 0
            feat['ob1_imbalance'] = 0

        # Static context
        feat['drop_bps'] = drop_bps
        feat['bottom_t_s'] = bottom_t / 1000.0
        feat['fr_abs_bps'] = abs(fr_bps)

        # Price range and std during recovery
        if len(recovery_prices) >= 2:
            feat['price_range_recovery'] = recovery_prices.max() - recovery_prices.min()
            feat['price_std_recovery'] = recovery_prices.std()
        else:
            feat['price_range_recovery'] = 0
            feat['price_std_recovery'] = 0

        # === TARGETS ===
        # How much MORE recovery remains from this tick?
        future_mask = (post_times > tick_t) & (post_times <= max_recovery_t)
        if future_mask.sum() > 0:
            future_prices = post_prices_bps[future_mask]
            future_max = future_prices.max()
            # Upside remaining
            feat['target_upside_remaining'] = future_max - current_price
            # Is this near the peak? (within 5 bps of eventual max)
            feat['target_near_peak_5'] = 1 if feat['target_upside_remaining'] < 5 else 0
            feat['target_near_peak_10'] = 1 if feat['target_upside_remaining'] < 10 else 0
            # Will price be LOWER in 5s? (exit signal)
            mask_5s = (post_times > tick_t) & (post_times <= tick_t + 5000)
            if mask_5s.sum() > 0:
                future_5s_min = post_prices_bps[mask_5s].min()
                feat['target_drops_5bps_in_5s'] = 1 if (current_price - future_5s_min) > 5 else 0
            else:
                feat['target_drops_5bps_in_5s'] = 0
        else:
            feat['target_upside_remaining'] = 0
            feat['target_near_peak_5'] = 1
            feat['target_near_peak_10'] = 1
            feat['target_drops_5bps_in_5s'] = 0

        feat['symbol'] = symbol
        feat['settle_id'] = stem

        rows.append(feat)

    return rows if rows else None


def train_long_exit_ml(df):
    """Train ML model to predict optimal long exit point."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    import copy

    print(f"\n{'='*70}")
    print("PART B: LONG EXIT ML (recovery phase)")
    print(f"{'='*70}")

    # Target: near_peak_10 — are we within 10bps of the recovery peak?
    target_col = 'target_near_peak_10'

    skip_cols = {'symbol', 'settle_id', 't_ms'}
    target_cols = {c for c in df.columns if c.startswith('target_')}
    feature_cols = [c for c in df.columns if c not in skip_cols | target_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(int)
    symbols = df['symbol'].values

    print(f"  {len(df)} ticks, {df['settle_id'].nunique()} settlements, {len(feature_cols)} features")
    print(f"  Positive rate (near_peak_10): {y.mean():.1%}")

    # 70/30 temporal split
    unique_settle = df['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = df['settle_id'].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    print(f"  Split: train={train_mask.sum()} ({len(train_settles)} settle), "
          f"test={test_mask.sum()} ({len(unique_settle)-n_train} settle)")

    models = {
        'LogReg': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        'HGBC': HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        y_prob_tr = m.predict_proba(X_tr)[:, 1]
        y_prob_te = m.predict_proba(X_te)[:, 1]

        auc_tr = roc_auc_score(y_tr, y_prob_tr)
        auc_te = roc_auc_score(y_te, y_prob_te)

        print(f"\n  {name}: Train AUC={auc_tr:.4f}  Test AUC={auc_te:.4f}  Gap={auc_tr-auc_te:.3f}")
        results[name] = {'auc_train': auc_tr, 'auc_test': auc_te}

    # Also test "near_peak_5" and "drops_5bps_in_5s"
    for alt_target in ['target_near_peak_5', 'target_drops_5bps_in_5s']:
        y_alt = df[alt_target].values.astype(int)
        y_tr_a, y_te_a = y_alt[train_mask], y_alt[test_mask]
        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        )
        m.fit(X_tr, y_tr_a)
        auc_tr = roc_auc_score(y_tr_a, m.predict_proba(X_tr)[:, 1])
        auc_te = roc_auc_score(y_te_a, m.predict_proba(X_te)[:, 1])
        print(f"\n  HGBC ({alt_target}): Train AUC={auc_tr:.4f}  Test AUC={auc_te:.4f}")

    # LOSO
    print(f"\n--- LOSO (symbol) for near_peak_10 ---")
    best_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    logo = LeaveOneGroupOut()
    try:
        y_pred_loso = cross_val_predict(best_model, X, y, cv=logo, groups=symbols,
                                        method='predict_proba')[:, 1]
        auc_loso = roc_auc_score(y, y_pred_loso)
        print(f"  LOSO AUC: {auc_loso:.4f}")
    except Exception as e:
        print(f"  LOSO failed: {e}")

    # Feature importance
    print(f"\n--- Feature Importance (permutation, HGBC near_peak_10) ---")
    from sklearn.inspection import permutation_importance
    m_imp = copy.deepcopy(models['HGBC'])
    m_imp.fit(X_tr, y_tr)
    perm = permutation_importance(m_imp, X_te, y_te, n_repeats=5,
                                  random_state=42, scoring='roc_auc')
    sorted_idx = np.argsort(-perm.importances_mean)
    for i in sorted_idx[:15]:
        imp = perm.importances_mean[i]
        std = perm.importances_std[i]
        print(f"  {feature_cols[i]:35s}: {imp:+.4f} +/- {std:.4f}")

    # === BACKTEST: ML exit vs fixed hold ===
    print(f"\n{'='*70}")
    print("BACKTEST: ML EXIT vs FIXED HOLD")
    print(f"{'='*70}")

    # Train on all data for backtest comparison
    m_full = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    m_full.fit(X, y)
    df['pred_near_peak'] = m_full.predict_proba(X)[:, 1]

    # Per-settlement: when does ML first say "near peak" (p > threshold)?
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        exit_times = []
        exit_recoveries = []
        fixed_recoveries = []

        for settle_id in df['settle_id'].unique():
            sdf = df[df['settle_id'] == settle_id].sort_values('t_ms')
            if sdf.empty:
                continue

            bottom_t = sdf['t_ms'].iloc[0] - sdf['time_since_bottom_ms'].iloc[0]

            # ML exit: first tick where p > threshold
            ml_mask = sdf['pred_near_peak'] >= thresh
            if ml_mask.any():
                ml_exit_row = sdf[ml_mask].iloc[0]
                ml_exit_recovery = ml_exit_row['recovery_bps']
                ml_exit_time = ml_exit_row['time_since_bottom_ms']
            else:
                # Never triggers — use last tick
                ml_exit_row = sdf.iloc[-1]
                ml_exit_recovery = ml_exit_row['recovery_bps']
                ml_exit_time = ml_exit_row['time_since_bottom_ms']

            exit_times.append(ml_exit_time)
            exit_recoveries.append(ml_exit_recovery)

            # Fixed hold comparison
            fixed_mask = sdf['time_since_bottom_ms'] >= LONG_HOLD_FIXED_MS
            if fixed_mask.any():
                fixed_recoveries.append(sdf[fixed_mask].iloc[0]['recovery_bps'])
            else:
                fixed_recoveries.append(sdf.iloc[-1]['recovery_bps'])

        n = len(exit_recoveries)
        ml_mean = np.mean(exit_recoveries)
        fixed_mean = np.mean(fixed_recoveries)
        ml_median_t = np.median(exit_times) / 1000

        print(f"\n  Threshold p>={thresh:.1f}: N={n}")
        print(f"    ML exit: avg recovery={ml_mean:+.1f}bps, median exit time={ml_median_t:.1f}s")
        print(f"    Fixed +{LONG_HOLD_FIXED_MS//1000}s: avg recovery={fixed_mean:+.1f}bps")
        print(f"    Delta: {ml_mean - fixed_mean:+.1f}bps ({'+' if ml_mean > fixed_mean else ''}{(ml_mean-fixed_mean)/abs(fixed_mean)*100:.0f}%)")

    return results, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("LONG LEG ML — ENTRY DECISION + EXIT OPTIMIZATION")
    print("=" * 70)

    # Part A: Entry decision
    entry_df = build_entry_decision_dataset()
    if entry_df is not None and len(entry_df) > 20:
        entry_results, entry_features = train_entry_decision(entry_df)
        entry_df.to_csv('results/long_entry_decision.csv', index=False)
        print(f"\n  Saved entry decision data: results/long_entry_decision.csv")

    # Part B: Exit ML
    exit_df = build_long_exit_dataset()
    if exit_df is not None and len(exit_df) > 100:
        exit_results, exit_features = train_long_exit_ml(exit_df)
        exit_df.to_parquet('results/long_exit_ticks.parquet', index=False)
        print(f"\n  Saved exit tick data: results/long_exit_ticks.parquet")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
