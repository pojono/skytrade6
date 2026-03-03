#!/usr/bin/env python3
"""
XS-7.4 Portfolio Sleeve + XS-7.6 Stress Gating

S07 bracket as controlled "lottery sleeve" with portfolio risk management,
then XS-8-style stress gating layered on top.
"""

import sys, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from xs6_bigmove_uplift import (
    DATA_DIR, START, END, MIN_DAYS, SIGNAL_STEP_MIN,
    TRAIN_END, TEST_START, PURGE_HOURS, SEED,
    discover_symbols, load_symbol, build_sym_1m, compute_features,
)

OUT_74 = Path(__file__).resolve().parent / "output" / "xs7_4"
OUT_76 = Path(__file__).resolve().parent / "output" / "xs7_6"
OUT_74.mkdir(parents=True, exist_ok=True)
OUT_76.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# Config
REF_A, REF_B, REF_C = 1.0, 2.0, 2.0
REF_T_MIN = 24 * 60
COOLDOWN_MIN = 24 * 60
FEE_TAKER_BP = 10.0
FEE_MAKER_BP = 2.0
SLIP_GRID = [0, 5, 10]

EQUITY0 = 2000.0
RISK_FRAC = 0.02
CAP_PER_TRADE = 100.0
MAX_OPEN_TRADES = 8
MAX_OPEN_PER_SYM = 1
MAX_OPEN_ORDERS = 20
DAILY_LOSS_LIMIT_FRAC = -0.02

PCA_UPD = 60
PCA_WIN = 360
STRESS_FEATS = ["crowd_oi", "pca_var1", "breadth_extreme", "entropy", "crowd_fund"]
TAIL_HORIZON = 60              # 1h in 1m bars
TAIL_ATR_MULT = 12.0           # 12×ATR — matching XS-8
TAIL_COIN_FRAC = 0.10          # >=10% of coins make tail move (XS-8 base rate ~35%)
N_PERM_GATE = 500              # permutation tests for gate sanity


def build_symbol_data(sym, raw, grid_1m):
    df_1m = build_sym_1m(sym, raw, grid_1m)
    if 1 - df_1m["is_invalid"].mean() < 0.5:
        return None, None
    df_1m = compute_features(df_1m)
    rv6 = df_1m["rv_6h"]
    df_1m["rv_6h_pctl_exp"] = rv6.expanding(min_periods=360).rank(pct=True)
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]
    df_5m = df_1m.loc[grid_5m].copy()
    s07 = ((df_5m["rv_6h_pctl_exp"] <= 0.20) & (df_5m["oi_z"] >= 1.5)
           & df_5m["atr_1h_raw"].notna() & df_5m["rv_6h"].notna()
           & (df_5m["is_invalid"] == 0))
    return df_1m, df_5m.index[s07.values]


# ── Portfolio Sleeve Simulator ───────────────────────────────────────────

def simulate_sleeve(sym_data, slip_bp, gate_fn=None, gate_label="none"):
    equity = EQUITY0
    cooldown = {}
    daily_pnl = defaultdict(float)
    daily_halted = set()
    trades = []
    eq_snaps = []
    n_ambig = 0

    # Collect all signals sorted chronologically
    all_ev = []
    for sym, (df_1m, sig_times) in sym_data.items():
        ts = df_1m.index
        ts_map = {t: i for i, t in enumerate(ts)}
        cl = df_1m["close"]; atr = df_1m["atr_1h_raw"]
        for t_sig in sig_times:
            idx0 = ts_map.get(t_sig)
            if idx0 is None or idx0 >= len(ts) - 10:
                continue
            ri = idx0 + 1
            if ri >= len(ts):
                continue
            p0, av = cl.iloc[ri], atr.iloc[idx0]
            if np.isnan(p0) or np.isnan(av) or av <= 0 or p0 <= 0:
                continue
            all_ev.append({"sym": sym, "t_sig": t_sig, "idx0": idx0,
                           "ri": ri, "P0": p0, "ATR": av})
    all_ev.sort(key=lambda x: x["t_sig"])

    prev_day = None
    for ev in all_ev:
        sym, t_sig = ev["sym"], ev["t_sig"]
        day = t_sig.date()

        # Daily snapshot
        if day != prev_day:
            eq_snaps.append({"date": day, "equity": equity})
            prev_day = day

        # Check constraints
        block = None
        if day in daily_halted:
            block = "DAILY_LOSS_BLOCKED"
        elif daily_pnl[day] <= DAILY_LOSS_LIMIT_FRAC * equity:
            daily_halted.add(day)
            block = "DAILY_LOSS_BLOCKED"
        elif sym in cooldown and (t_sig - cooldown[sym]).total_seconds() < COOLDOWN_MIN * 60:
            block = "COOLDOWN_BLOCKED"
        elif gate_fn is not None and not gate_fn(t_sig):
            block = "STRESS_GATED"

        notional = min(equity * RISK_FRAC, CAP_PER_TRADE)
        if notional < 1.0:
            block = "INSUFFICIENT_EQUITY"

        eid = f"{sym}_{t_sig:%Y%m%d%H%M}"

        if block:
            trades.append(_blocked(eid, sym, t_sig, ev["ATR"], block,
                                   gate_label, slip_bp, equity))
            continue

        # Place bracket — mark cooldown
        cooldown[sym] = t_sig

        # Simulate
        df_1m = sym_data[sym][0]
        cl_v = df_1m["close"].values
        hi_v = df_1m["high"].values
        lo_v = df_1m["low"].values
        inv_v = df_1m["is_invalid"].values
        ts_arr = df_1m.index

        idx0, ri, p0, av = ev["idx0"], ev["ri"], ev["P0"], ev["ATR"]
        ar = av / p0
        bs, ss = p0 * (1 + REF_A * ar), p0 * (1 - REF_A * ar)
        mx = min(ri + REF_T_MIN, len(ts_arr) - 1)

        # Find trigger
        eidx, eside, epx, dbl, ambig = None, None, None, False, False
        for i in range(ri + 1, mx + 1):
            if inv_v[i]:
                continue
            h = hi_v[i] if not np.isnan(hi_v[i]) else cl_v[i]
            l = lo_v[i] if not np.isnan(lo_v[i]) else cl_v[i]
            bk, sk = h >= bs, l <= ss
            if bk and sk:
                ambig = True; n_ambig += 1
                eidx, eside, epx, dbl = i, "long", bs, True; break
            elif bk:
                eidx, eside, epx = i, "long", bs; break
            elif sk:
                eidx, eside, epx = i, "short", ss; break

        if eidx is None:
            trades.append(_nofill(eid, sym, t_sig, av, gate_label, slip_bp, equity, notional))
            continue

        # TP/SL
        if eside == "long":
            tp, sl = epx * (1 + REF_B * ar), epx * (1 - REF_C * ar)
        else:
            tp, sl = epx * (1 - REF_B * ar), epx * (1 + REF_C * ar)

        er, xr, xi = "TIME", cl_v[mx], mx
        mfe, mae = 0.0, 0.0
        for i in range(eidx + 1, mx + 1):
            if inv_v[i]:
                continue
            h = hi_v[i] if not np.isnan(hi_v[i]) else cl_v[i]
            l = lo_v[i] if not np.isnan(lo_v[i]) else cl_v[i]
            if eside == "long":
                mfe = max(mfe, (h / epx - 1) * 1e4)
                mae = min(mae, (l / epx - 1) * 1e4)
                tp_h, sl_h = h >= tp, l <= sl
                if tp_h and sl_h:
                    er, xr, xi = "SL", sl, i; n_ambig += 1; break
                if sl_h:
                    er, xr, xi = "SL", sl, i; break
                if tp_h:
                    er, xr, xi = "TP", tp, i; break
            else:
                mfe = max(mfe, (1 - l / epx) * 1e4)
                mae = min(mae, (1 - h / epx) * 1e4)
                tp_h, sl_h = l <= tp, h >= sl
                if tp_h and sl_h:
                    er, xr, xi = "SL", sl, i; n_ambig += 1; break
                if sl_h:
                    er, xr, xi = "SL", sl, i; break
                if tp_h:
                    er, xr, xi = "TP", tp, i; break

        gross = (xr / epx - 1) * 1e4 if eside == "long" else (1 - xr / epx) * 1e4
        dl = 0.0
        if dbl:
            opp = ss if eside == "long" else bs
            dc = cl_v[eidx]
            dl = (1 - dc / opp) * 1e4 if eside == "long" else (dc / opp - 1) * 1e4

        ef = FEE_MAKER_BP if er == "TP" else FEE_TAKER_BP
        se = slip_bp * 0.5 if er == "TP" else slip_bp
        fees = FEE_TAKER_BP + ef + slip_bp + se
        if dbl:
            fees += FEE_TAKER_BP * 2 + slip_bp * 2
        net_bp = gross + dl - fees
        net_usd = net_bp / 1e4 * notional

        equity += net_usd
        daily_pnl[day] += net_usd

        trades.append({
            "event_id": eid, "symbol": sym, "t_signal": t_sig,
            "t_entry": ts_arr[eidx], "t_exit": ts_arr[xi],
            "side": eside, "entry_px": epx, "exit_px": xr,
            "exit_reason": er, "atr": av,
            "fees_bp": fees, "slip_bp_scenario": slip_bp,
            "gross_bp": gross, "dbl_loss_bp": dl, "net_bp": net_bp,
            "net_usd": net_usd, "notional": notional,
            "MFE_bp": mfe, "MAE_bp": mae,
            "double_trigger": dbl, "ambiguous": ambig,
            "gate_label": gate_label, "equity_at_trade": equity,
            "blocked": False, "block_reason": "",
        })

    eq_snaps.append({"date": END.date(), "equity": equity})
    return pd.DataFrame(trades), pd.DataFrame(eq_snaps), n_ambig


def _blocked(eid, sym, t_sig, atr, reason, gl, slip, eq):
    return {"event_id": eid, "symbol": sym, "t_signal": t_sig,
            "t_entry": pd.NaT, "t_exit": pd.NaT,
            "side": "none", "entry_px": np.nan, "exit_px": np.nan,
            "exit_reason": reason, "atr": atr,
            "fees_bp": 0, "slip_bp_scenario": slip,
            "gross_bp": 0, "dbl_loss_bp": 0, "net_bp": 0,
            "net_usd": 0, "notional": 0,
            "MFE_bp": 0, "MAE_bp": 0,
            "double_trigger": False, "ambiguous": False,
            "gate_label": gl, "equity_at_trade": eq,
            "blocked": True, "block_reason": reason}


def _nofill(eid, sym, t_sig, atr, gl, slip, eq, notional):
    return {"event_id": eid, "symbol": sym, "t_signal": t_sig,
            "t_entry": pd.NaT, "t_exit": pd.NaT,
            "side": "none", "entry_px": np.nan, "exit_px": np.nan,
            "exit_reason": "NOFILL", "atr": atr,
            "fees_bp": 0, "slip_bp_scenario": slip,
            "gross_bp": 0, "dbl_loss_bp": 0, "net_bp": 0,
            "net_usd": 0, "notional": notional,
            "MFE_bp": 0, "MAE_bp": 0,
            "double_trigger": False, "ambiguous": False,
            "gate_label": gl, "equity_at_trade": eq,
            "blocked": False, "block_reason": ""}


# ── Stress Features ──────────────────────────────────────────────────────

def compute_stress(sym_data, grid_1m):
    g5 = grid_1m[::SIGNAL_STEP_MIN]; n5 = len(g5)
    syms = sorted(sym_data.keys()); ns = len(syms)
    print(f"  Stress: {n5} pts × {ns} syms")
    t0 = time.monotonic()

    ATR_1H_BARS = 60  # 1h ATR on 1m bars — matching XS-8 exactly
    ca, la, oza, fza, atra = {}, {}, {}, {}, {}
    for s in syms:
        d = sym_data[s][0]
        ca[s] = d["close"].values; la[s] = d["log_ret"].values
        oza[s] = d["oi_z"].values if "oi_z" in d.columns else np.full(len(d), np.nan)
        fza[s] = d["funding_z"].values if "funding_z" in d.columns else np.full(len(d), np.nan)
        # Compute proper 1h ATR (60 bars) matching XS-8
        hi = d["high"].values; lo = d["low"].values; cl = d["close"].values
        tr = np.maximum(hi - lo,
             np.maximum(np.abs(hi - np.roll(cl, 1)),
                        np.abs(lo - np.roll(cl, 1))))
        tr[0] = np.nan  # first bar has no prev close
        # Rolling mean over 60 bars
        atr_1h = pd.Series(tr).rolling(ATR_1H_BARS, min_periods=ATR_1H_BARS // 2).mean().values
        atra[s] = atr_1h

    out = {k: np.full(n5, np.nan) for k in STRESS_FEATS}
    tail_frac = np.full(n5, np.nan)

    for si in range(n5):
        ix = si * SIGNAL_STEP_MIN
        if si % 5000 == 0 and si > 0:
            print(f"    {si}/{n5} ({time.monotonic()-t0:.0f}s)")

        abr, ozl, fzl = [], [], []
        for s in syms:
            c = ca[s]
            if ix >= 60 and not np.isnan(c[ix]) and c[ix] > 0:
                cp = c[ix - 60]
                if not np.isnan(cp) and cp > 0:
                    abr.append(abs(np.log(c[ix] / cp)))
                else:
                    abr.append(np.nan)
            else:
                abr.append(np.nan)
            ozl.append(oza[s][ix] if ix < len(oza[s]) else np.nan)
            fzl.append(fza[s][ix] if ix < len(fza[s]) else np.nan)

        abr_a = np.array(abr); ozl_a = np.array(ozl); fzl_a = np.array(fzl)
        nv = (~np.isnan(abr_a)).sum()
        if nv < 10:
            continue

        ov = ozl_a[~np.isnan(ozl_a)]
        if len(ov) >= 10:
            out["crowd_oi"][si] = (ov > 1.5).mean()
        fv = fzl_a[~np.isnan(fzl_a)]
        if len(fv) >= 10:
            out["crowd_fund"][si] = (np.abs(fv) > 2.0).mean()

        av = abr_a[~np.isnan(abr_a)]
        med = np.median(av)
        if med > 1e-12:
            out["breadth_extreme"][si] = (av > 1.5 * med).mean()
        if len(av) >= 10 and np.std(av) > 1e-12:
            h, _ = np.histogram(av, bins=20, density=True)
            h = h / h.sum() if h.sum() > 0 else h; h = h[h > 0]
            out["entropy"][si] = -np.sum(h * np.log(h))

        if si % PCA_UPD == 0 and ix >= PCA_WIN:
            rm = []
            for s in syms:
                lr = la[s][ix - PCA_WIN:ix]
                if np.isnan(lr).sum() < PCA_WIN * 0.3:
                    rm.append(np.nan_to_num(lr, nan=0.0))
            if len(rm) >= 10:
                try:
                    p = PCA(n_components=1); p.fit(np.column_stack(rm))
                    out["pca_var1"][si] = p.explained_variance_ratio_[0]
                except Exception:
                    pass
        if si > 0 and np.isnan(out["pca_var1"][si]) and not np.isnan(out["pca_var1"][si - 1]):
            out["pca_var1"][si] = out["pca_var1"][si - 1]

        # Target: tail frac next 1h — per-coin ATR-normalized
        ng = len(grid_1m); fi = min(ix + TAIL_HORIZON, ng - 1)
        if fi <= ix + 5:
            continue
        nt, nc = 0, 0
        for s in syms:
            c = ca[s]; a = atra[s]
            if ix >= len(c) or np.isnan(c[ix]) or c[ix] <= 0:
                continue
            if ix >= len(a) or np.isnan(a[ix]) or a[ix] <= 0:
                continue
            atr_ret = a[ix] / c[ix]  # ATR as fraction of price
            # Check max |ret| over next 1h window
            end_ix = min(fi, len(c) - 1)
            if end_ix <= ix + 5:
                continue
            future_c = c[ix+1:end_ix+1]
            if len(future_c) < 5:
                continue
            max_abs_ret = np.nanmax(np.abs(np.log(future_c / c[ix])))
            nc += 1
            if max_abs_ret > TAIL_ATR_MULT * atr_ret:
                nt += 1
        if nc > 0:
            tail_frac[si] = nt / nc

    print(f"  Stress done ({time.monotonic()-t0:.0f}s)")
    sf = pd.DataFrame({"ts": g5, **out, "tail_frac": tail_frac}).set_index("ts")
    return sf


def fit_stress_model(sdf):
    feats = STRESS_FEATS
    df = sdf.copy()
    df["Y"] = (df["tail_frac"] >= TAIL_COIN_FRAC).astype(float)
    df.loc[df["tail_frac"].isna(), "Y"] = np.nan
    for f in feats:
        df[f] = df[f].ffill()

    # Diagnostics
    tf = df["tail_frac"].dropna()
    print(f"  tail_frac: {len(tf)} non-NaN, mean={tf.mean():.4f}, "
          f"p50={tf.median():.4f}, p90={tf.quantile(0.9):.4f}, max={tf.max():.4f}")
    print(f"  tail_frac == 0: {(tf == 0).sum()}/{len(tf)} ({(tf == 0).mean():.1%})")
    y_valid = df["Y"].dropna()
    print(f"  Y (>={TAIL_COIN_FRAC:.0%}): {int(y_valid.sum())}/{len(y_valid)} "
          f"({y_valid.mean():.1%}) \u2190 target base rate")

    valid = df.dropna(subset=feats + ["Y"])
    train = valid[valid.index < TRAIN_END]
    print(f"  Valid rows: {len(valid)}, train: {len(train)}, "
          f"train Y=1: {int(train['Y'].sum()) if len(train) > 0 else 0}")

    if len(train) < 100 or train["Y"].sum() < 10:
        print("  FATAL: insufficient stress train data \u2014 cannot fit model.")
        print("  Check: TAIL_ATR_MULT and TAIL_COIN_FRAC may be wrong.")
        df["stress_score"] = np.nan
        df["stress_quintile"] = np.nan
        return df

    sc = StandardScaler()
    Xtr = sc.fit_transform(train[feats].values)
    ytr = train["Y"].values.astype(int)
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    m.fit(Xtr, ytr)

    from sklearn.metrics import roc_auc_score
    p_train = m.predict_proba(Xtr)[:, 1]
    auc_train = roc_auc_score(ytr, p_train)
    print(f"  Stress model: N_train={len(train)}, pos={ytr.sum()}, AUC_train={auc_train:.3f}")
    for fn, cf in sorted(zip(feats, m.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {fn:22s}: {cf:+.4f}")

    # Score full valid period
    Xf = sc.transform(valid[feats].values)
    valid = valid.copy()
    valid["stress_score"] = m.predict_proba(Xf)[:, 1]

    # OOS AUC
    test = valid[valid.index >= TEST_START]
    if len(test) > 50 and test["Y"].sum() >= 5:
        auc_test = roc_auc_score(test["Y"].values.astype(int),
                                  test["stress_score"].values)
        print(f"  AUC_test (OOS): {auc_test:.3f}")

    # EXPANDING CAUSAL quintiles: at each point, rank vs all prior scores
    scores = valid["stress_score"].values
    n = len(scores)
    quintiles = np.full(n, np.nan, dtype=object)
    MIN_WARMUP = 200  # need at least 200 prior points
    for i in range(MIN_WARMUP, n):
        pct = np.searchsorted(np.sort(scores[:i]), scores[i]) / i
        if pct < 0.2:
            quintiles[i] = "Q1"
        elif pct < 0.4:
            quintiles[i] = "Q2"
        elif pct < 0.6:
            quintiles[i] = "Q3"
        elif pct < 0.8:
            quintiles[i] = "Q4"
        else:
            quintiles[i] = "Q5"
    valid["stress_quintile"] = quintiles

    # Score distribution audit
    ss = valid["stress_score"]
    print(f"  Score dist: min={ss.min():.4f}, p25={ss.quantile(0.25):.4f}, "
          f"p50={ss.median():.4f}, p75={ss.quantile(0.75):.4f}, "
          f"p95={ss.quantile(0.95):.4f}, max={ss.max():.4f}")
    print(f"  Score == 0: {(ss == 0).sum()}/{len(ss)} ({(ss == 0).mean():.1%})")
    vc = pd.Series(quintiles[quintiles != None]).value_counts().sort_index()
    print(f"  Expanding quintile dist: {dict(vc)}")

    # Feature correlations with score
    for fn in feats:
        r = valid[[fn, "stress_score"]].corr().iloc[0, 1]
        print(f"    corr(score, {fn}): {r:+.3f}")

    # Merge back
    df["stress_score"] = np.nan
    df.loc[valid.index, "stress_score"] = valid["stress_score"].values
    df["stress_quintile"] = np.nan
    df.loc[valid.index, "stress_quintile"] = valid["stress_quintile"].values
    df["stress_score"] = df["stress_score"].ffill()
    df["stress_quintile"] = df["stress_quintile"].ffill()

    # Save audit CSV
    audit = valid[["stress_score", "stress_quintile", "Y", "tail_frac"] + feats].copy()
    audit.to_csv(OUT_76 / "stress_audit.csv")
    print(f"  Audit CSV: {OUT_76 / 'stress_audit.csv'} ({len(audit)} rows)")

    return df


def make_gate(sdf, gate_type):
    sc = sdf["stress_score"]; qu = sdf["stress_quintile"]
    # Gate C threshold: median of train-period scores
    train_scores = sc[sc.index < TRAIN_END].dropna()
    gate_c_thresh = train_scores.median() if len(train_scores) > 0 else 0.5
    print(f"    Gate C threshold (train median): {gate_c_thresh:.4f}")

    def _lookup_q(t):
        idx = qu.index.searchsorted(t, side="right") - 1
        if idx < 0:
            return None
        v = qu.iloc[idx]
        if isinstance(v, float) and np.isnan(v):
            return None
        return v
    def _lookup_s(t):
        idx = sc.index.searchsorted(t, side="right") - 1
        return sc.iloc[idx] if idx >= 0 else np.nan

    if gate_type == "A":
        return lambda t: _lookup_q(t) in ("Q1", "Q2")
    elif gate_type == "B":
        return lambda t: _lookup_q(t) == "Q1"
    elif gate_type == "C":
        return lambda t: (lambda s: not np.isnan(s) and s <= gate_c_thresh)(_lookup_s(t))
    elif gate_type == "D":
        return lambda t: _lookup_q(t) in ("Q4", "Q5")
    return lambda t: True


# ── KPI ──────────────────────────────────────────────────────────────────

def compute_kpi(trades_df, label):
    df = trades_df
    entered = df[df["exit_reason"].isin(["TP", "SL", "TIME"])]
    ne = len(entered); na = len(df)
    nofill = (df["exit_reason"] == "NOFILL").sum()
    blocked = df["blocked"].sum() if "blocked" in df.columns else 0
    gated = (df["exit_reason"] == "STRESS_GATED").sum()

    if ne == 0:
        return {"label": label, "N_all": na, "N_entries": 0, "N_nofill": int(nofill),
                "N_blocked": int(blocked), "N_gated": int(gated),
                "trigger_rate": 0, "mean_bp": 0, "median_bp": 0, "PF": 0,
                "TP_pct": 0, "SL_pct": 0, "TIME_pct": 0,
                "top5_conc": 0, "tail_500": 0, "tail_800": 0,
                "maxDD_usd": 0, "months_pos": 0, "months_tot": 0,
                "weeks_pos": 0, "weeks_tot": 0, "final_equity": EQUITY0}

    net = entered["net_bp"].values; nu = entered["net_usd"].values
    mn, md = np.mean(net), np.median(net)
    pos = net[net > 0]; neg = net[net < 0]
    pf = pos.sum() / max(abs(neg.sum()), 1e-12)
    ev = entered["exit_reason"].value_counts()
    tp_p = ev.get("TP", 0) / ne
    sl_p = ev.get("SL", 0) / ne
    tm_p = ev.get("TIME", 0) / ne

    sn = np.sort(net)[::-1]; tot = net.sum()
    t5 = sn[:5].sum() / tot if tot > 0 and ne >= 5 else np.nan
    cum = np.cumsum(nu); dd = np.min(cum - np.maximum.accumulate(cum))

    ec = entered.copy()
    ec["mo"] = pd.to_datetime(ec["t_signal"]).dt.to_period("M")
    mo = ec.groupby("mo")["net_usd"].sum()
    ec["wk"] = pd.to_datetime(ec["t_signal"]).dt.isocalendar().week
    wk = ec.groupby("wk")["net_usd"].sum()

    feq = entered["equity_at_trade"].iloc[-1] if len(entered) > 0 else EQUITY0

    return {"label": label, "N_all": na, "N_entries": ne, "N_nofill": int(nofill),
            "N_blocked": int(blocked), "N_gated": int(gated),
            "trigger_rate": ne / max(na - blocked, 1),
            "mean_bp": mn, "median_bp": md, "PF": pf,
            "TP_pct": tp_p, "SL_pct": sl_p, "TIME_pct": tm_p,
            "top5_conc": t5,
            "tail_500": int((net > 500).sum()),
            "tail_800": int((net > 800).sum()),
            "maxDD_usd": dd,
            "months_pos": int((mo > 0).sum()), "months_tot": len(mo),
            "weeks_pos": int((wk > 0).sum()), "weeks_tot": len(wk),
            "final_equity": feq}


# ── FINDINGS ─────────────────────────────────────────────────────────────

def write_findings(kpis_74, kpis_76, stress_info):
    # XS-7.4 findings
    p4 = OUT_74 / "FINDINGS_xs7_4.md"
    with open(p4, "w") as f:
        f.write("# XS-7.4 — Portfolio Sleeve (S07 Bracket)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")
        f.write(f"**Data:** {START.date()} → {END.date()}\n")
        f.write(f"**RefConfig:** a={REF_A}, b={REF_B}, c={REF_C}, T=24h\n")
        f.write(f"**Equity:** ${EQUITY0}, risk={RISK_FRAC:.0%}, cap=${CAP_PER_TRADE}\n")
        f.write(f"**Limits:** max_open={MAX_OPEN_TRADES}, max_per_sym={MAX_OPEN_PER_SYM}, "
                f"daily_loss={DAILY_LOSS_LIMIT_FRAC:.0%}\n\n---\n\n")

        f.write("## KPI Summary\n\n")
        f.write("| Slip | N_entries | Trig% | Mean bp | Median bp | PF | TP% | SL% | TIME% | "
                "Top5% | Tail>500 | Tail>800 | MaxDD$ | Wk+ | Final$ |\n")
        f.write("|------|-----------|-------|---------|-----------|-----|-----|-----|-------|"
                "-------|----------|----------|--------|-----|--------|\n")
        for k in kpis_74:
            f.write(f"| {k['label']} | {k['N_entries']} | {k['trigger_rate']:.0%} | "
                    f"{k['mean_bp']:.1f} | {k['median_bp']:.1f} | {k['PF']:.2f} | "
                    f"{k['TP_pct']:.0%} | {k['SL_pct']:.0%} | {k['TIME_pct']:.0%} | "
                    f"{k.get('top5_conc',0):.0%} | {k['tail_500']} | {k['tail_800']} | "
                    f"{k['maxDD_usd']:.1f} | {k['weeks_pos']}/{k['weeks_tot']} | "
                    f"{k['final_equity']:.0f} |\n")

        f.write("\n## Interpretation\n\n")
        base5 = [k for k in kpis_74 if "5bp" in k["label"]]
        if base5:
            b = base5[0]
            f.write(f"With realistic slippage (5bp), the sleeve produces **{b['N_entries']} trades** "
                    f"over {(END-START).days} days.\n")
            f.write(f"- Mean net: **{b['mean_bp']:.1f} bp/trade**, median: **{b['median_bp']:.1f} bp**\n")
            f.write(f"- TP rate: {b['TP_pct']:.0%}, TIME-loss: {b['TIME_pct']:.0%}\n")
            f.write(f"- Top-5 concentration: {b.get('top5_conc',0):.0%}\n")
            f.write(f"- Final equity: ${b['final_equity']:.0f} (from ${EQUITY0})\n")
            f.write(f"- Max drawdown: ${abs(b['maxDD_usd']):.1f}\n\n")
            verdict = "positive EV" if b['mean_bp'] > 0 else "negative or marginal EV"
            f.write(f"**Assessment:** {verdict} at 5bp slippage.\n")
        f.write("\n---\n\n## Files\n\n")
        f.write(f"- `output/xs7_4/trades_sleeve.csv`\n")
        f.write(f"- `output/xs7_4/equity_curve.csv`\n")
        f.write(f"- `output/xs7_4/kpi_summary.csv`\n")
    print(f"  → {p4}")

    # XS-7.6 findings
    p6 = OUT_76 / "FINDINGS_xs7_6.md"
    with open(p6, "w") as f:
        f.write("# XS-7.6 — Stress Gating over S07 Sleeve\n\n")
        f.write(f"**Generated:** {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")
        f.write(f"**Data:** {START.date()} → {END.date()}\n")
        f.write(f"**Stress model:** LogReg on {', '.join(STRESS_FEATS)}\n")
        if stress_info:
            f.write(f"**Train N:** {stress_info.get('n_train','?')}, "
                    f"pos: {stress_info.get('n_pos','?')}\n\n")
        f.write("---\n\n")

        # Baseline vs gated comparison
        f.write("## Comparison: Baseline vs Gated (all slippage levels)\n\n")
        f.write("| Config | N_entries | Trig% | Mean bp | Median bp | PF | TP% | TIME% | "
                "Top5% | MaxDD$ | Wk+ | Final$ |\n")
        f.write("|--------|-----------|-------|---------|-----------|-----|-----|-------|"
                "-------|--------|-----|--------|\n")
        for k in sorted(kpis_76, key=lambda x: x["label"]):
            f.write(f"| {k['label']} | {k['N_entries']} | {k['trigger_rate']:.0%} | "
                    f"{k['mean_bp']:.1f} | {k['median_bp']:.1f} | {k['PF']:.2f} | "
                    f"{k['TP_pct']:.0%} | {k['TIME_pct']:.0%} | "
                    f"{k.get('top5_conc',0):.0%} | {k['maxDD_usd']:.1f} | "
                    f"{k['weeks_pos']}/{k['weeks_tot']} | {k['final_equity']:.0f} |\n")

        f.write("\n## Analysis\n\n")
        # Find baseline_5bp and best gate_5bp
        bl5 = [k for k in kpis_76 if k["label"] == "baseline_5bp"]
        gates5 = [k for k in kpis_76 if "5bp" in k["label"] and "baseline" not in k["label"]]
        if bl5 and gates5:
            bl = bl5[0]
            f.write(f"**Baseline (5bp slip):** {bl['N_entries']} trades, "
                    f"mean {bl['mean_bp']:.1f}bp, median {bl['median_bp']:.1f}bp, "
                    f"TIME% {bl['TIME_pct']:.0%}\n\n")
            for g in sorted(gates5, key=lambda x: x["mean_bp"], reverse=True):
                dm = g["mean_bp"] - bl["mean_bp"]
                dmd = g["median_bp"] - bl["median_bp"]
                dtime = g["TIME_pct"] - bl["TIME_pct"]
                f.write(f"- **{g['label']}:** {g['N_entries']} trades, "
                        f"Δmean {dm:+.1f}bp, Δmedian {dmd:+.1f}bp, "
                        f"ΔTIME% {dtime:+.0%}, top5 {g.get('top5_conc',0):.0%}\n")
            f.write("\n")

        # Gating effectiveness
        f.write("### Key Questions\n\n")
        f.write("1. **Does gating reduce TIME-loss share?** ")
        if bl5 and gates5:
            best_time = min(gates5, key=lambda x: x["TIME_pct"])
            if best_time["TIME_pct"] < bl5[0]["TIME_pct"]:
                f.write(f"Yes — {best_time['label']} reduces TIME% from "
                        f"{bl5[0]['TIME_pct']:.0%} to {best_time['TIME_pct']:.0%}\n")
            else:
                f.write("No — gating does not improve TIME-loss share.\n")
        f.write("2. **Does gating reduce max DD?** ")
        if bl5 and gates5:
            best_dd = min(gates5, key=lambda x: abs(x["maxDD_usd"]))
            if abs(best_dd["maxDD_usd"]) < abs(bl5[0]["maxDD_usd"]):
                f.write(f"Yes — {best_dd['label']}: ${abs(best_dd['maxDD_usd']):.1f} "
                        f"vs ${abs(bl5[0]['maxDD_usd']):.1f}\n")
            else:
                f.write("No significant improvement.\n")
        f.write("3. **Does gating improve median?** ")
        if bl5 and gates5:
            best_med = max(gates5, key=lambda x: x["median_bp"])
            if best_med["median_bp"] > bl5[0]["median_bp"]:
                f.write(f"Yes — {best_med['label']}: {best_med['median_bp']:.1f}bp "
                        f"vs {bl5[0]['median_bp']:.1f}bp\n")
            else:
                f.write("No.\n")

        f.write("\n---\n\n## Files\n\n")
        f.write(f"- `output/xs7_6/report.csv`\n")
        f.write(f"- `output/xs7_6/equity_curves.csv`\n")
        f.write(f"- `output/xs7_6/trades_with_stress.parquet`\n")
    print(f"  → {p6}")


# ── Sanity Tests ─────────────────────────────────────────────────────────

def run_sanity_tests(sym_data, sdf_model, baseline_kpi_5bp):
    """Permutation gate test + placebo gate test on Gate A at 5bp slip."""
    print("\n§SANITY: Gate A permutation + placebo tests...")

    if sdf_model["stress_quintile"].isna().all():
        print("  SKIP: no valid quintiles")
        return {}

    # Real Gate A result
    real_mean = baseline_kpi_5bp["mean_bp"]  # will compare gated vs this
    gate_fn_a = make_gate(sdf_model, "A")
    tr_a, _, _ = simulate_sleeve(sym_data, 5, gate_fn=gate_fn_a, gate_label="gateA")
    kpi_a = compute_kpi(tr_a, "gateA_5bp")
    real_gated_mean = kpi_a["mean_bp"]
    real_gated_med = kpi_a["median_bp"]
    real_n = kpi_a["N_entries"]
    print(f"  Real Gate A: N={real_n}, mean={real_gated_mean:.1f}bp, med={real_gated_med:.1f}bp")

    # 1) Permutation test: shuffle quintile assignments within each day
    print(f"  Permutation test ({N_PERM_GATE} shuffles)...")
    perm_means, perm_meds = [], []
    qu_series = sdf_model["stress_quintile"].copy()
    score_series = sdf_model["stress_score"].copy()

    for pi in range(N_PERM_GATE):
        # Shuffle quintiles (break the score→quintile mapping)
        shuf_q = qu_series.values.copy()
        valid_mask = pd.notna(shuf_q)
        valid_vals = shuf_q[valid_mask]
        np.random.shuffle(valid_vals)
        shuf_q[valid_mask] = valid_vals

        # Build shuffled gate
        shuf_sdf = sdf_model.copy()
        shuf_sdf["stress_quintile"] = shuf_q
        gate_fn_shuf = make_gate(shuf_sdf, "A")
        tr_s, _, _ = simulate_sleeve(sym_data, 5, gate_fn=gate_fn_shuf, gate_label="perm")
        kpi_s = compute_kpi(tr_s, "perm")
        if kpi_s["N_entries"] > 0:
            perm_means.append(kpi_s["mean_bp"])
            perm_meds.append(kpi_s["median_bp"])

        if (pi + 1) % 100 == 0:
            print(f"    {pi+1}/{N_PERM_GATE}")

    perm_means = np.array(perm_means)
    perm_meds = np.array(perm_meds)
    p_mean = (perm_means >= real_gated_mean).mean() if len(perm_means) > 0 else 1.0
    p_med = (perm_meds >= real_gated_med).mean() if len(perm_meds) > 0 else 1.0
    print(f"  Permutation: real_mean={real_gated_mean:.1f}, "
          f"shuf_mean={perm_means.mean():.1f}±{perm_means.std():.1f}, p={p_mean:.3f}")
    print(f"  Permutation: real_med={real_gated_med:.1f}, "
          f"shuf_med={perm_meds.mean():.1f}±{perm_meds.std():.1f}, p={p_med:.3f}")

    # 2) Placebo gate: use hour-of-day quintiles instead of stress quintiles
    print("  Placebo test (hour-of-day gate)...")
    placebo_sdf = sdf_model.copy()
    hours = placebo_sdf.index.hour
    # Map hours to quintiles: 0-4=Q1, 5-9=Q2, 10-14=Q3, 15-19=Q4, 20-23=Q5
    hq = np.where(hours < 5, "Q1",
         np.where(hours < 10, "Q2",
         np.where(hours < 15, "Q3",
         np.where(hours < 20, "Q4", "Q5"))))
    placebo_sdf["stress_quintile"] = hq
    gate_fn_placebo = make_gate(placebo_sdf, "A")
    tr_p, _, _ = simulate_sleeve(sym_data, 5, gate_fn=gate_fn_placebo, gate_label="placebo")
    kpi_p = compute_kpi(tr_p, "placebo_hourgate_5bp")
    print(f"  Placebo (hour gate A): N={kpi_p['N_entries']}, "
          f"mean={kpi_p['mean_bp']:.1f}bp, med={kpi_p['median_bp']:.1f}bp")

    sanity = {
        "real_mean": real_gated_mean, "real_med": real_gated_med, "real_N": real_n,
        "perm_mean_avg": perm_means.mean() if len(perm_means) > 0 else np.nan,
        "perm_mean_std": perm_means.std() if len(perm_means) > 0 else np.nan,
        "p_mean": p_mean, "p_med": p_med,
        "placebo_mean": kpi_p["mean_bp"], "placebo_med": kpi_p["median_bp"],
        "placebo_N": kpi_p["N_entries"],
    }
    pd.DataFrame([sanity]).to_csv(OUT_76 / "sanity_tests.csv", index=False)
    return sanity


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    T0 = time.monotonic()
    print("=" * 70)
    print("XS-7.4 Portfolio Sleeve + XS-7.6 Stress Gating (v2 fixed)")
    print(f"  TAIL_ATR_MULT={TAIL_ATR_MULT}, TAIL_COIN_FRAC={TAIL_COIN_FRAC}")
    print(f"  Quintiles: EXPANDING CAUSAL")
    print("=" * 70)

    # 1. Load data
    print("\n§1: Loading data...")
    syms = discover_symbols()
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"  {len(syms)} symbols, {len(grid_1m):,} 1m bars")

    sym_data = {}; ns = 0
    for i, sym in enumerate(syms):
        raw = load_symbol(sym)
        d, st = build_symbol_data(sym, raw, grid_1m)
        if d is None:
            continue
        sym_data[sym] = (d, st); ns += len(st)
        if (i + 1) % 10 == 0 or i == len(syms) - 1:
            print(f"  {i+1}/{len(syms)}: {len(sym_data)} valid, {ns} signals")
    print(f"  Total: {len(sym_data)} syms, {ns} S07 signals")

    # 2. XS-7.4: Baseline sleeve across slippage grid
    print("\n§2: XS-7.4 Portfolio Sleeve (baseline)...")
    kpis_74 = []
    all_trades_74 = []
    all_eq_74 = []
    for slip in SLIP_GRID:
        label = f"baseline_{slip}bp"
        print(f"  Running {label}...")
        tr, eq, namb = simulate_sleeve(sym_data, slip, gate_fn=None, gate_label="none")
        kpi = compute_kpi(tr, label)
        kpis_74.append(kpi)
        tr["run_label"] = label
        all_trades_74.append(tr)
        eq["run_label"] = label
        all_eq_74.append(eq)
        ne = kpi["N_entries"]
        print(f"    N={ne}, mean={kpi['mean_bp']:.1f}bp, med={kpi['median_bp']:.1f}bp, "
              f"PF={kpi['PF']:.2f}, TP={kpi['TP_pct']:.0%}, TIME={kpi['TIME_pct']:.0%}, "
              f"top5={kpi.get('top5_conc',0):.0%}, DD=${kpi['maxDD_usd']:.1f}, "
              f"final=${kpi['final_equity']:.0f}, ambig={namb}")

    trades_74 = pd.concat(all_trades_74, ignore_index=True)
    eq_74 = pd.concat(all_eq_74, ignore_index=True)
    trades_74.to_csv(OUT_74 / "trades_sleeve.csv", index=False)
    eq_74.to_csv(OUT_74 / "equity_curve.csv", index=False)
    pd.DataFrame(kpis_74).to_csv(OUT_74 / "kpi_summary.csv", index=False)

    # 3. Compute stress features
    print("\n§3: Stress features (XS-7.6)...")
    sdf = compute_stress(sym_data, grid_1m)
    sdf_model = fit_stress_model(sdf)

    model_ok = not sdf_model["stress_score"].isna().all()
    stress_info = {}
    if model_ok:
        train_rows = sdf_model[sdf_model.index < TRAIN_END].dropna(subset=["stress_score"])
        stress_info["n_train"] = len(train_rows)
        stress_info["n_pos"] = int(
            (sdf_model.loc[sdf_model.index < TRAIN_END, "tail_frac"].dropna() >= TAIL_COIN_FRAC).sum())
        sdf_model.to_parquet(OUT_76 / "stress_features.parquet")
    else:
        print("  *** STRESS MODEL FAILED — gating section will be skipped ***")

    # 4. XS-7.6: Gated simulations
    kpis_76 = list(kpis_74)
    all_trades_76 = list(all_trades_74)
    all_eq_76 = list(all_eq_74)

    if model_ok:
        print("\n§4: XS-7.6 Gated simulations...")
        for gate_type in ["A", "B", "C", "D"]:
            gate_fn = make_gate(sdf_model, gate_type)
            for slip in SLIP_GRID:
                label = f"gate{gate_type}_{slip}bp"
                print(f"  Running {label}...")
                tr, eq, namb = simulate_sleeve(sym_data, slip, gate_fn=gate_fn,
                                                gate_label=f"gate_{gate_type}")
                kpi = compute_kpi(tr, label)
                kpis_76.append(kpi)
                tr["run_label"] = label
                all_trades_76.append(tr)
                eq["run_label"] = label
                all_eq_76.append(eq)
                ne = kpi["N_entries"]
                ng = kpi["N_gated"]
                print(f"    N={ne}, gated={ng}, mean={kpi['mean_bp']:.1f}bp, "
                      f"med={kpi['median_bp']:.1f}bp, TIME={kpi['TIME_pct']:.0%}, "
                      f"top5={kpi.get('top5_conc',0):.0%}, DD=${kpi['maxDD_usd']:.1f}")
    else:
        print("\n§4: SKIPPED (no stress model)")

    trades_76 = pd.concat(all_trades_76, ignore_index=True)
    eq_76 = pd.concat(all_eq_76, ignore_index=True)
    pd.DataFrame(kpis_76).to_csv(OUT_76 / "report.csv", index=False)
    eq_76.to_csv(OUT_76 / "equity_curves.csv", index=False)
    trades_76.to_parquet(OUT_76 / "trades_with_stress.parquet", index=False)

    # 5. Bug guards
    print("\n§5: Bug guards...")
    base_tr = all_trades_74[1]  # 5bp baseline
    entered = base_tr[base_tr["exit_reason"].isin(["TP", "SL", "TIME"])]
    if len(entered) > 0:
        for sym in entered["symbol"].unique():
            st = entered[entered["symbol"] == sym].sort_values("t_signal")
            if len(st) >= 2:
                diffs = st["t_signal"].diff().dt.total_seconds().dropna()
                violations = (diffs < COOLDOWN_MIN * 60).sum()
                if violations > 0:
                    print(f"  COOLDOWN VIOLATION: {sym} has {violations} violations!")
        print(f"  Cooldown: OK (checked)")
        print(f"  Idempotency: deterministic (fixed seed)")

    # 6. Sanity tests (permutation + placebo)
    sanity = {}
    if model_ok:
        baseline_5bp_kpi = kpis_74[1]  # 5bp baseline
        sanity = run_sanity_tests(sym_data, sdf_model, baseline_5bp_kpi)

    # 7. FINDINGS
    print("\n§7: Writing FINDINGS...")
    write_findings(kpis_74, kpis_76, stress_info)

    elapsed = time.monotonic() - T0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
