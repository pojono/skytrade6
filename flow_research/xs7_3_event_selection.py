#!/usr/bin/env python3
"""
XS-7.3 — Event Selection over S07 (reduce tail-dependency)

Pipeline:
  §1  Data loading & S07 signal (reuse xs6)
  §2  Event generation with 24h cooldown
  §3  Feature computation (local + cross-sectional + relative)
  §4  RefConfig bracket simulation
  §5  Target labels (Y_tail, Y_pnl, hit_TP)
  §6  Walk-forward splits
  §7  Metrics computation
  §8  Quantile selectors
  §9  Model selector (LogReg)
  §10 Shuffle sanity
  §11 FINDINGS
  §12 Main
"""

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from xs6_bigmove_uplift import (
    DATA_DIR, START, END, MIN_DAYS, SIGNAL_STEP_MIN,
    TRAIN_END, TEST_START, PURGE_HOURS, SEED,
    discover_symbols, load_symbol, build_sym_1m, compute_features,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs7_3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# Config
FEE_TAKER_BP, FEE_MAKER_BP, SLIP_BP_BASE = 10.0, 2.0, 5.0
COOLDOWN_MIN = 24 * 60
PURGE_TD = pd.Timedelta(hours=PURGE_HOURS)
REF_A, REF_B, REF_C, REF_T_MIN = 1.0, 2.0, 2.0, 24 * 60
TAIL_K = 3.0
N_SHUFFLES = 500
PCA_UPDATE_INTERVAL = 60
PCA_WINDOW_1M = 360


# ── §1: Build 1m data with S07 ──────────────────────────────────────────

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


# ── §2: Events with cooldown ────────────────────────────────────────────

def generate_events(sym_data):
    rows = []
    for sym, (df_1m, signal_times) in sym_data.items():
        last = None
        close = df_1m["close"]; atr = df_1m["atr_1h_raw"]; ts = df_1m.index
        ts_map = {t: i for i, t in enumerate(ts)}
        for t_sig in signal_times:
            if last and (t_sig - last).total_seconds() < COOLDOWN_MIN * 60:
                continue
            idx0 = ts_map.get(t_sig)
            if idx0 is None or idx0 >= len(ts) - 10:
                continue
            ri = idx0 + 1
            if ri >= len(ts): continue
            p0, av = close.iloc[ri], atr.iloc[idx0]
            if np.isnan(p0) or np.isnan(av) or av <= 0 or p0 <= 0:
                continue
            last = t_sig
            rows.append({"event_id": f"{sym}_{t_sig:%Y%m%d%H%M}", "symbol": sym,
                         "t_signal": t_sig, "t_entry_ref": ts[ri],
                         "P0": p0, "ATR_1h": av, "idx0": idx0, "ref_idx": ri})
    events = pd.DataFrame(rows)
    print(f"  Events after cooldown: {len(events)} ({events['symbol'].nunique()} symbols)")
    return events


# ── §3: Features ────────────────────────────────────────────────────────

def _rstd(arr, idx, w):
    s = max(0, idx - w + 1)
    return np.nanstd(arr[s:idx+1]) if idx - s >= w // 2 else np.nan

def _rsum(arr, idx, w):
    s = max(0, idx - w + 1)
    return np.nansum(arr[s:idx+1]) if idx - s >= w // 2 else np.nan

def _rmean_agg(arr, idx, lb, aw):
    s = max(0, idx - lb + 1)
    if idx - s < lb // 4: return np.nan
    ch = arr[s:idx+1]; n = len(ch) // aw
    if n < 2: return np.nan
    return np.nanmean([np.nansum(ch[i*aw:(i+1)*aw]) for i in range(n)])

def _rstd_agg(arr, idx, lb, aw):
    s = max(0, idx - lb + 1)
    if idx - s < lb // 4: return np.nan
    ch = arr[s:idx+1]; n = len(ch) // aw
    if n < 2: return np.nan
    return np.nanstd([np.nansum(ch[i*aw:(i+1)*aw]) for i in range(n)])


def compute_event_features_local(events, sym_data):
    rows = []
    for _, ev in events.iterrows():
        sym = ev["symbol"]; df = sym_data[sym][0]; idx0 = ev["idx0"]
        p0, av = ev["P0"], ev["ATR_1h"]
        lr = df["log_ret"].values; cl = df["close"].values

        rv_2h = _rstd(lr, idx0, 120)
        rv_6h = _rstd(lr, idx0, 360)
        rv_24h = _rstd(lr, idx0, 1440)
        rv_1h = _rstd(lr, idx0, 60)

        vol = df["turnover"].values if "turnover" in df.columns else None
        if vol is not None:
            v1 = _rsum(vol, idx0, 60); v6 = _rsum(vol, idx0, 360)
            m1 = _rmean_agg(vol, idx0, 7*1440, 60); s1 = _rstd_agg(vol, idx0, 7*1440, 60)
            m6 = _rmean_agg(vol, idx0, 7*1440, 360); s6 = _rstd_agg(vol, idx0, 7*1440, 360)
            vz1 = (v1 - m1) / max(s1, 1e-12) if not np.isnan(m1) else np.nan
            vz6 = (v6 - m6) / max(s6, 1e-12) if not np.isnan(m6) else np.nan
        else:
            vz1, vz6, v1 = np.nan, np.nan, np.nan

        oi_z = df["oi_z"].values[idx0] if "oi_z" in df.columns else np.nan
        oi = df["oi"].values if "oi" in df.columns else None
        dOI_1h = (oi[idx0]-oi[idx0-60])/max(abs(oi[idx0-60]),1e-12) if oi is not None and idx0>=60 and not np.isnan(oi[idx0]) and not np.isnan(oi[idx0-60]) else np.nan
        dOI_6h = (oi[idx0]-oi[idx0-360])/max(abs(oi[idx0-360]),1e-12) if oi is not None and idx0>=360 and not np.isnan(oi[idx0]) and not np.isnan(oi[idx0-360]) else np.nan

        fz = df["funding_z"].values[idx0] if "funding_z" in df.columns else np.nan
        fl = df["fr"].values[idx0] if "fr" in df.columns else np.nan
        trend = np.log(cl[idx0]/cl[idx0-360]) if idx0>=360 and cl[idx0-360]>0 else np.nan

        if idx0 >= 360 and "high" in df.columns:
            rng = (np.nanmax(df["high"].values[idx0-360:idx0+1]) -
                   np.nanmin(df["low"].values[idx0-360:idx0+1])) / p0
            comp = rng / (av/p0) if av/p0 > 1e-12 else np.nan
        else:
            comp = np.nan

        liq = v1 / max(rv_1h, 1e-12) if vol is not None and not np.isnan(rv_1h) and not np.isnan(v1) else np.nan

        rows.append({"event_id": ev["event_id"],
            "rv_2h": rv_2h, "rv_6h": rv_6h, "rv_24h": rv_24h,
            "atr_1h": av, "vol_1h_z": vz1, "vol_6h_z": vz6,
            "oi_z": oi_z, "dOI_1h": dOI_1h, "dOI_6h": dOI_6h,
            "fund_z": fz, "fund_level": fl, "trend_6h": trend,
            "compression": comp, "liquidity_proxy": liq})
    return pd.DataFrame(rows)


def compute_cross_sectional_features(sym_data, grid_1m):
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]; n5 = len(grid_5m)
    syms = sorted(sym_data.keys()); ns = len(syms)
    print(f"  XS features: {n5} timepoints × {ns} syms")
    t0 = time.monotonic()

    ca, la, oza, fza, rva6, rva2 = {}, {}, {}, {}, {}, {}
    for s in syms:
        d = sym_data[s][0]
        ca[s] = d["close"].values; la[s] = d["log_ret"].values
        oza[s] = d["oi_z"].values if "oi_z" in d.columns else np.full(len(d), np.nan)
        fza[s] = d["funding_z"].values if "funding_z" in d.columns else np.full(len(d), np.nan)
        rva6[s] = d["rv_6h"].values if "rv_6h" in d.columns else np.full(len(d), np.nan)
        rva2[s] = d["rv_2h"].values if "rv_2h" in d.columns else np.full(len(d), np.nan)

    out = {k: np.full(n5, np.nan) for k in [
        "pca_var1","dispersion_1h","breadth_extreme","crowd_oi","crowd_fund",
        "entropy","median_rv_6h","median_rv_2h","median_oi_z","median_fund_z"]}

    for si in range(n5):
        ix = si * SIGNAL_STEP_MIN
        if si % 5000 == 0 and si > 0:
            print(f"    {si}/{n5} ({time.monotonic()-t0:.0f}s)")

        r1h, ozl, fzl, r6l, r2l, abr = [], [], [], [], [], []
        for s in syms:
            c = ca[s]
            if ix >= 60 and not np.isnan(c[ix]) and c[ix] > 0:
                cp = c[ix-60]
                if not np.isnan(cp) and cp > 0:
                    r = np.log(c[ix]/cp); r1h.append(r); abr.append(abs(r))
                else: r1h.append(np.nan); abr.append(np.nan)
            else: r1h.append(np.nan); abr.append(np.nan)
            ozl.append(oza[s][ix] if ix < len(oza[s]) else np.nan)
            fzl.append(fza[s][ix] if ix < len(fza[s]) else np.nan)
            r6l.append(rva6[s][ix] if ix < len(rva6[s]) else np.nan)
            r2l.append(rva2[s][ix] if ix < len(rva2[s]) else np.nan)

        r1h = np.array(r1h); abr = np.array(abr)
        ozl = np.array(ozl); fzl = np.array(fzl)
        r6l = np.array(r6l); r2l = np.array(r2l)
        nv = (~np.isnan(r1h)).sum()
        if nv < 10: continue

        out["dispersion_1h"][si] = np.nanstd(r1h)
        med_abs = np.nanmedian(abr)
        if med_abs > 1e-12:
            out["breadth_extreme"][si] = np.nanmean(abr > 1.5 * med_abs)

        ov = ozl[~np.isnan(ozl)]
        if len(ov) >= 10:
            out["crowd_oi"][si] = (ov > 1.5).mean()
            out["median_oi_z"][si] = np.median(ov)
        fv = fzl[~np.isnan(fzl)]
        if len(fv) >= 10:
            out["crowd_fund"][si] = (np.abs(fv) > 2.0).mean()
            out["median_fund_z"][si] = np.median(fv)
        rv = r6l[~np.isnan(r6l)]
        if len(rv) >= 10: out["median_rv_6h"][si] = np.median(rv)
        rv2 = r2l[~np.isnan(r2l)]
        if len(rv2) >= 10: out["median_rv_2h"][si] = np.median(rv2)

        av = abr[~np.isnan(abr)]
        if len(av) >= 10 and np.std(av) > 1e-12:
            h, _ = np.histogram(av, bins=20, density=True)
            h = h / h.sum() if h.sum() > 0 else h; h = h[h > 0]
            out["entropy"][si] = -np.sum(h * np.log(h))

        if si % PCA_UPDATE_INTERVAL == 0 and ix >= PCA_WINDOW_1M:
            rm = []
            for s in syms:
                lr = la[s][ix-PCA_WINDOW_1M:ix]
                if np.isnan(lr).sum() < PCA_WINDOW_1M * 0.3:
                    rm.append(np.nan_to_num(lr, nan=0.0))
            if len(rm) >= 10:
                try:
                    p = PCA(n_components=1); p.fit(np.column_stack(rm))
                    out["pca_var1"][si] = p.explained_variance_ratio_[0]
                except Exception: pass
        if si > 0 and np.isnan(out["pca_var1"][si]) and not np.isnan(out["pca_var1"][si-1]):
            out["pca_var1"][si] = out["pca_var1"][si-1]

    print(f"  XS features done ({time.monotonic()-t0:.0f}s)")
    mf = pd.DataFrame({"ts": grid_5m, **out}).set_index("ts")
    return mf


def attach_features(events, local_feats, mkt_feats, sym_data):
    ev = events.merge(local_feats, on="event_id", how="left")
    mkt = mkt_feats.copy(); mkt.index.name = "ts"
    mc = [c for c in mkt.columns if c not in ev.columns]
    ev = ev.sort_values("t_signal")
    ev = pd.merge_asof(ev, mkt[mc].reset_index().rename(columns={"ts": "t_signal"}),
                        on="t_signal", direction="backward")
    ev["rv_6h_rel"] = ev["rv_6h"] / ev["median_rv_6h"].clip(lower=1e-12)
    ev["oi_z_rel"] = ev["oi_z"] - ev["median_oi_z"]
    ev["fund_z_rel"] = ev["fund_z"] - ev["median_fund_z"]
    return ev


# ── §4: RefConfig simulation ────────────────────────────────────────────

def simulate_ref_trades(events, sym_data):
    a, b, c, tsm = REF_A, REF_B, REF_C, REF_T_MIN
    trades = []
    for _, ev in events.iterrows():
        sym = ev["symbol"]; df = sym_data[sym][0]
        cl = df["close"].values; hi = df["high"].values; lo = df["low"].values
        inv = df["is_invalid"].values; ts = df.index
        idx0, ri, p0, av = ev["idx0"], ev["ref_idx"], ev["P0"], ev["ATR_1h"]
        ar = av / p0
        bs, ss = p0*(1+a*ar), p0*(1-a*ar)
        mx = min(ri + tsm, len(ts) - 1)

        eidx, eside, epx, dbl = None, None, None, False
        for i in range(ri+1, mx+1):
            if inv[i]: continue
            h = hi[i] if not np.isnan(hi[i]) else cl[i]
            l = lo[i] if not np.isnan(lo[i]) else cl[i]
            if h >= bs and l <= ss:
                eidx, eside, epx, dbl = i, "long", bs, True; break
            elif h >= bs:
                eidx, eside, epx = i, "long", bs; break
            elif l <= ss:
                eidx, eside, epx = i, "short", ss; break

        if eidx is None:
            trades.append({"event_id": ev["event_id"], "symbol": sym,
                "t_signal": ev["t_signal"], "entry_side": "none",
                "entry_time": pd.NaT, "entry_px": np.nan,
                "exit_type": "NO_TRIGGER", "exit_time": pd.NaT, "exit_px": np.nan,
                "pnl_gross_bp": 0, "pnl_net_bp": 0, "MFE_bp": 0, "MAE_bp": 0,
                "time_to_entry": np.nan, "double_trigger": False, "fees_bp": 0})
            continue

        if eside == "long":
            tp, sl = epx*(1+b*ar), epx*(1-c*ar)
        else:
            tp, sl = epx*(1-b*ar), epx*(1+c*ar)

        er, xr = "TIME", cl[mx]; xi = mx; mfe, mae = 0.0, 0.0
        for i in range(eidx+1, mx+1):
            if inv[i]: continue
            h = hi[i] if not np.isnan(hi[i]) else cl[i]
            l = lo[i] if not np.isnan(lo[i]) else cl[i]
            if eside == "long":
                mfe = max(mfe, (h/epx-1)*1e4); mae = min(mae, (l/epx-1)*1e4)
                if l <= sl: er, xr, xi = "SL", sl, i; break
                if h >= tp: er, xr, xi = "TP", tp, i; break
            else:
                mfe = max(mfe, (1-l/epx)*1e4); mae = min(mae, (1-h/epx)*1e4)
                if h >= sl: er, xr, xi = "SL", sl, i; break
                if l <= tp: er, xr, xi = "TP", tp, i; break

        gross = (xr/epx-1)*1e4 if eside == "long" else (1-xr/epx)*1e4
        dl = 0.0
        if dbl:
            opp = ss if eside == "long" else bs; dc = cl[eidx]
            dl = (1-dc/opp)*1e4 if eside == "long" else (dc/opp-1)*1e4

        ef = FEE_MAKER_BP if er == "TP" else FEE_TAKER_BP
        se = SLIP_BP_BASE * 0.5 if er == "TP" else SLIP_BP_BASE
        fees = FEE_TAKER_BP + ef + SLIP_BP_BASE + se
        if dbl: fees += FEE_TAKER_BP * 2 + SLIP_BP_BASE * 2

        trades.append({"event_id": ev["event_id"], "symbol": sym,
            "t_signal": ev["t_signal"], "entry_side": eside,
            "entry_time": ts[eidx], "entry_px": epx,
            "exit_type": er, "exit_time": ts[xi], "exit_px": xr,
            "pnl_gross_bp": gross, "pnl_net_bp": gross + dl - fees,
            "MFE_bp": mfe, "MAE_bp": mae,
            "time_to_entry": (ts[eidx]-ev["t_signal"]).total_seconds()/60,
            "double_trigger": dbl, "fees_bp": fees})
    return pd.DataFrame(trades)


# ── §5: Targets ─────────────────────────────────────────────────────────

def compute_targets(events, sym_data):
    labs = []
    for _, ev in events.iterrows():
        df = sym_data[ev["symbol"]][0]; cl = df["close"].values
        ri, p0, av = ev["ref_idx"], ev["P0"], ev["ATR_1h"]
        ar = av / p0; mx = min(ri + REF_T_MIN, len(cl) - 1)
        fut = cl[ri+1:mx+1]
        if len(fut) < 10:
            labs.append({"event_id": ev["event_id"], "Y_tail": np.nan}); continue
        mr = np.nanmax(np.abs(np.log(fut / p0)))
        labs.append({"event_id": ev["event_id"], "Y_tail": int(mr >= TAIL_K * ar)})
    return pd.DataFrame(labs)


# ── §6: Splits ──────────────────────────────────────────────────────────

def label_splits(df):
    ts = df["t_signal"]
    df["split_fwd"] = "purge"
    df.loc[ts < TRAIN_END - PURGE_TD, "split_fwd"] = "train"
    df.loc[ts > TEST_START + PURGE_TD, "split_fwd"] = "test"
    df["split_rev"] = "purge"
    df.loc[ts > TEST_START + PURGE_TD, "split_rev"] = "train"
    df.loc[ts < TRAIN_END - PURGE_TD, "split_rev"] = "test"
    return df


# ── §7: Metrics ─────────────────────────────────────────────────────────

def compute_selector_metrics(trades, selector_id, method, sel_rate, split):
    act = trades[trades["exit_type"] != "NO_TRIGGER"]
    n = len(act)
    if n == 0:
        return {"selector_id": selector_id, "method": method,
                "selection_rate": sel_rate, "split": split, "N": 0,
                "mean": 0, "median": 0, "PF": 0, "hit_TP": 0, "tail_rate": 0,
                "top1_conc": 0, "top2_conc": 0, "top5_conc": 0,
                "maxDD": 0, "weeks_pos": 0, "weeks_total": 0}
    net = act["pnl_net_bp"].values
    mn, md = np.mean(net), np.median(net)
    pf = np.sum(net[net>0]) / max(abs(np.sum(net[net<0])), 1e-12)
    tp_r = (act["exit_type"] == "TP").mean()
    tail_r = act["Y_tail"].mean() if "Y_tail" in act.columns else np.nan
    sn = np.sort(net)[::-1]; tot = np.sum(net)
    t1 = sn[0]/tot if tot > 0 else np.nan
    t2 = sn[:2].sum()/tot if tot > 0 and n >= 2 else np.nan
    t5 = sn[:5].sum()/tot if tot > 0 and n >= 5 else np.nan
    cum = np.cumsum(net); dd = np.min(cum - np.maximum.accumulate(cum))
    ac = act.copy(); ac["wk"] = pd.to_datetime(ac["t_signal"]).dt.isocalendar().week
    wk = ac.groupby("wk")["pnl_net_bp"].sum()
    return {"selector_id": selector_id, "method": method,
            "selection_rate": sel_rate, "split": split, "N": n,
            "mean": mn, "median": md, "PF": pf, "hit_TP": tp_r,
            "tail_rate": tail_r, "top1_conc": t1, "top2_conc": t2,
            "top5_conc": t5, "maxDD": dd, "weeks_pos": int((wk>0).sum()),
            "weeks_total": len(wk)}


# ── §8: Quantile selectors ──────────────────────────────────────────────

Q_FEATS = ["rv_6h","rv_24h","vol_1h_z","vol_6h_z","oi_z","dOI_1h","dOI_6h",
           "fund_z","fund_level","trend_6h","compression","liquidity_proxy",
           "pca_var1","dispersion_1h","breadth_extreme","crowd_oi","crowd_fund",
           "entropy","rv_6h_rel","oi_z_rel","fund_z_rel"]

COMBOS = [
    ("crowd_oi_neg", {"crowd_oi": -1.0}),
    ("low_rv_high_oi", {"rv_6h": -1.0, "oi_z": 1.0}),
    ("compress_stress", {"compression": -1.0, "crowd_oi": -1.0, "dispersion_1h": -1.0}),
    ("oi_fund_crowd", {"crowd_oi": -1.0, "crowd_fund": -1.0}),
    ("liq_compress", {"liquidity_proxy": -1.0, "compression": -1.0}),
    ("vol_oi_interact", {"vol_6h_z": -1.0, "oi_z": 1.0, "crowd_oi": -1.0}),
]

def run_quantile_selectors(merged, split_col, split_label):
    test = merged[merged[split_col] == "test"].copy()
    if len(test) < 20: return []
    results = []
    for feat in Q_FEATS:
        if feat not in test.columns or test[feat].dropna().nunique() < 5: continue
        v = test.dropna(subset=[feat])
        if len(v) < 20: continue
        try:
            v["Q"] = pd.qcut(v[feat], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
        except ValueError: continue
        for ql in ["Q1", "Q5"]:
            sub = v[v["Q"] == ql]
            if len(sub) < 5: continue
            results.append(compute_selector_metrics(
                sub, f"q_{feat}_{ql}", "quantile", len(sub)/len(v), split_label))
    for cn, ws in COMBOS:
        cols = [c for c in ws if c in test.columns]
        if len(cols) < len(ws): continue
        v = test.dropna(subset=cols)
        if len(v) < 20: continue
        v["_sc"] = sum(w * v[c] for c, w in ws.items())
        try:
            v["Q"] = pd.qcut(v["_sc"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
        except ValueError: continue
        for ql in ["Q1", "Q5"]:
            sub = v[v["Q"] == ql]
            if len(sub) < 5: continue
            results.append(compute_selector_metrics(
                sub, f"combo_{cn}_{ql}", "quantile_combo", len(sub)/len(v), split_label))
    return results


# ── §9: Model selector ──────────────────────────────────────────────────

M_FEATS = ["rv_6h","rv_24h","vol_1h_z","vol_6h_z","oi_z","dOI_1h","dOI_6h",
           "fund_z","trend_6h","compression","liquidity_proxy",
           "pca_var1","dispersion_1h","breadth_extreme","crowd_oi","crowd_fund",
           "entropy","rv_6h_rel","oi_z_rel","fund_z_rel"]

def run_model_selector(merged, split_col, split_label, target="Y_tail"):
    tr = merged[merged[split_col] == "train"]; te = merged[merged[split_col] == "test"]
    if len(tr) < 30 or len(te) < 20: return []
    fa = [f for f in M_FEATS if f in merged.columns]
    tv = tr.dropna(subset=fa+[target]); tev = te.dropna(subset=fa+[target])
    if len(tv) < 30 or len(tev) < 20: return []
    Xtr, ytr = tv[fa].values, tv[target].values.astype(int)
    Xte, yte = tev[fa].values, tev[target].values.astype(int)
    if ytr.sum() < 5 or (len(ytr)-ytr.sum()) < 5: return []
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED); m.fit(Xtr_s, ytr)
    pr = m.predict_proba(Xte_s)[:, 1]
    tev = tev.copy(); tev["model_score"] = pr

    print(f"\n  Model coefficients ({split_label}):")
    for fn, cf in sorted(zip(fa, m.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:8]:
        print(f"    {fn:22s}: {cf:+.4f}")

    results = []
    try: auc = roc_auc_score(yte, pr)
    except: auc = np.nan
    for pl, pp in [("top20", 0.80), ("top30", 0.70), ("top50", 0.50)]:
        th = np.percentile(pr, pp * 100)
        sel = tev[tev["model_score"] >= th]
        if len(sel) < 5: continue
        rm = compute_selector_metrics(sel, f"model_logreg_{pl}", "model_logreg",
                                       len(sel)/len(tev), split_label)
        rm["AUC"] = auc
        results.append(rm)
    return results


# ── §10: Shuffle sanity ─────────────────────────────────────────────────

def shuffle_sanity(merged, real_metrics, n_shuffles=N_SHUFFLES):
    rows = []; rng = np.random.default_rng(SEED)
    if not real_metrics: return pd.DataFrame()
    bl = [m for m in real_metrics if m["selector_id"] == "baseline"]
    if not bl: return pd.DataFrame()
    bm = bl[0]["mean"]
    imp = sorted([(m["selector_id"], m["split"], m["mean"]-bm)
                   for m in real_metrics if m["mean"] > bm and m["selector_id"] != "baseline" and m["N"] >= 15],
                  key=lambda x: x[2], reverse=True)
    seen = set(); tsel = []
    for sid, sp, d in imp:
        if sid not in seen and len(tsel) < 5: tsel.append((sid, sp, d)); seen.add(sid)
    if not tsel:
        print("  No improved selectors to shuffle-test."); return pd.DataFrame()

    print(f"\n  Shuffle sanity: {len(tsel)} selectors × {n_shuffles} shuffles")
    t0 = time.monotonic()
    for sid, sp, rd in tsel:
        rm = [m for m in real_metrics if m["selector_id"] == sid and m["split"] == sp]
        if not rm: continue
        rm = rm[0]; ns = rm["N"]
        sc = "split_fwd" if "fwd" in sp else "split_rev"
        at = merged[(merged[sc] == "test") & (merged["exit_type"] != "NO_TRIGGER")]
        if len(at) < ns + 5: continue
        na = at["pnl_net_bp"].values
        sm, smd, st5 = [], [], []
        for _ in range(n_shuffles):
            ix = rng.choice(len(na), size=ns, replace=False); sn = na[ix]
            sm.append(np.mean(sn)); smd.append(np.median(sn))
            ss = np.sort(sn)[::-1]; st = np.sum(sn)
            st5.append(ss[:5].sum()/st if st > 0 and len(ss) >= 5 else np.nan)
        sm, smd, st5 = np.array(sm), np.array(smd), np.array(st5)
        for mn, rv, sa, lo in [
            ("mean_net_bp", rm["mean"], sm, False),
            ("median_net_bp", rm["median"], smd, False),
            ("top5_conc", rm["top5_conc"], st5, True)]:
            pv = np.mean(sa <= rv) if lo else np.mean(sa >= rv)
            rows.append({"selector_id": sid, "split": sp, "metric_name": mn,
                "real_value": rv, "shuffle_mean": np.nanmean(sa),
                "shuffle_p95": np.nanpercentile(sa, 5 if lo else 95),
                "shuffle_p99": np.nanpercentile(sa, 1 if lo else 99), "p_value": pv})
    print(f"  Shuffle done ({time.monotonic()-t0:.0f}s)")
    return pd.DataFrame(rows)


# ── §11: FINDINGS ────────────────────────────────────────────────────────

def generate_findings(bl_m, all_m, shuf, events, trades):
    p = OUTPUT_DIR / "FINDINGS_xs7_3.md"
    bf = [m for m in bl_m if m["split"] == "test_fwd"]
    br = [m for m in bl_m if m["split"] == "test_rev"]

    with open(p, "w") as f:
        f.write("# XS-7.3 — Event Selection over S07\n\n")
        f.write(f"**Generated:** {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")
        f.write(f"**Data:** {START.date()} → {END.date()}\n")
        f.write(f"**RefConfig:** a={REF_A}, b={REF_B}, c={REF_C}, T=24h, slip={SLIP_BP_BASE}bp\n")
        f.write(f"**Events:** {len(events)} ({events['symbol'].nunique()} syms)\n")
        f.write(f"**Triggered:** {(trades['exit_type']!='NO_TRIGGER').sum()}\n\n---\n\n")

        f.write("## Baseline (no filter)\n\n")
        f.write("| Split | N | Mean | Median | PF | TP% | Tail% | Top5% | MaxDD | Wk+ |\n")
        f.write("|-------|---|------|--------|----|-----|-------|-------|-------|-----|\n")
        for bl in [bf, br]:
            if bl:
                b = bl[0]
                f.write(f"| {b['split']} | {b['N']} | {b['mean']:.1f} | {b['median']:.1f} | "
                    f"{b['PF']:.2f} | {b['hit_TP']:.0%} | {b.get('tail_rate',0):.0%} | "
                    f"{b.get('top5_conc',0):.0%} | {b['maxDD']:.0f} | "
                    f"{b['weeks_pos']}/{b['weeks_total']} |\n")
        f.write("\n")

        # Improvements
        imps = []
        for m in all_m:
            if m["selector_id"] == "baseline" or m["N"] < 10: continue
            br_ref = bf[0] if "fwd" in m["split"] else (br[0] if br else None)
            if br_ref is None: continue
            imps.append({**m, "delta_mean": m["mean"]-br_ref["mean"],
                         "delta_median": m["median"]-br_ref["median"],
                         "delta_top5": m.get("top5_conc",9)-br_ref.get("top5_conc",9)})
        imps.sort(key=lambda x: x["delta_mean"], reverse=True)

        f.write("## Top Selectors (Δmean over baseline)\n\n")
        f.write("| Selector | Split | N | Mean | ΔMean | Median | ΔMed | Top5% | ΔTop5 | Tail% | Wk+ |\n")
        f.write("|----------|-------|---|------|-------|--------|------|-------|-------|-------|-----|\n")
        for m in imps[:20]:
            f.write(f"| {m['selector_id']} | {m['split']} | {m['N']} | "
                f"{m['mean']:.1f} | {m['delta_mean']:+.1f} | "
                f"{m['median']:.1f} | {m['delta_median']:+.1f} | "
                f"{m.get('top5_conc',0):.0%} | {m['delta_top5']:+.0%} | "
                f"{m.get('tail_rate',0):.0%} | {m['weeks_pos']}/{m['weeks_total']} |\n")
        f.write("\n")

        if len(shuf) > 0:
            f.write("## Shuffle Sanity\n\n")
            f.write(f"N={N_SHUFFLES}\n\n")
            f.write("| Selector | Metric | Real | ShufMean | p |\n")
            f.write("|----------|--------|------|----------|---|\n")
            for _, r in shuf.iterrows():
                f.write(f"| {r['selector_id']} | {r['metric_name']} | "
                    f"{r['real_value']:.1f} | {r['shuffle_mean']:.1f} | "
                    f"{r['p_value']:.3f} |\n")
            f.write("\n")

        # GO/NO-GO
        f.write("## GO/NO-GO\n\n")
        n_go = 0
        for m in imps[:10]:
            ok = (m["delta_mean"] >= 10 and m["delta_median"] >= 15
                  and (m.get("top5_conc",9) <= 1.40 or m["delta_top5"] <= -0.80)
                  and m["weeks_pos"] >= 3 and m["N"] >= 60)
            if ok: n_go += 1
        f.write(f"**Full GO configs: {n_go}**\n\n")
        if n_go > 0:
            f.write("**Verdict: CONDITIONAL GO ✅**\n")
        else:
            f.write("**Verdict: NO-GO ❌** — no selector passes all criteria.\n")

        f.write("\n---\n\n## Files\n\n")
        for fn in ["events.parquet","trades_ref.csv","selector_report.csv",
                    "selector_weekly.csv","shuffle_sanity.csv"]:
            f.write(f"- `flow_research/output/xs7_3/{fn}`\n")

    print(f"\nFindings → {p}")


# ── §12: Main ────────────────────────────────────────────────────────────

def main():
    T0 = time.monotonic()
    print("=" * 70)
    print("XS-7.3 — Event Selection over S07")
    print("=" * 70)
    print(f"Period: {START.date()} → {END.date()}")
    print(f"RefConfig: a={REF_A}, b={REF_B}, c={REF_C}, T=24h, slip={SLIP_BP_BASE}bp")
    print(f"Tail: |ret| >= {TAIL_K}×ATR, Shuffles: {N_SHUFFLES}\n")

    # 1. Load
    print("§1: Loading data...")
    syms = discover_symbols()
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"  {len(syms)} symbols, {len(grid_1m):,} 1m bars")

    sym_data = {}; ns = 0; excl = []
    for i, sym in enumerate(syms):
        raw = load_symbol(sym)
        d, st = build_symbol_data(sym, raw, grid_1m)
        if d is None: excl.append(sym); continue
        sym_data[sym] = (d, st); ns += len(st)
        if (i+1) % 10 == 0 or i == len(syms)-1:
            print(f"  {i+1}/{len(syms)}: {len(sym_data)} valid, {ns} signals")
    print(f"\nTotal: {len(sym_data)} syms, {ns} signals")
    if excl: print(f"Excluded ({len(excl)}): {', '.join(excl[:10])}")

    # 2. Events
    print("\n§2: Events...")
    events = generate_events(sym_data)

    # 3. Features
    print("\n§3: Features...")
    t3 = time.monotonic()
    print("  §3.1 Local...")
    lf = compute_event_features_local(events, sym_data)
    print(f"  {len(lf)} × {len(lf.columns)-1} feats ({time.monotonic()-t3:.0f}s)")
    print("  §3.2 Cross-sectional...")
    mf = compute_cross_sectional_features(sym_data, grid_1m)
    print("  §3.3 Merge...")
    events = attach_features(events, lf, mf, sym_data)
    fc = len([c for c in events.columns if c not in ["event_id","symbol","t_signal","t_entry_ref","P0","ATR_1h","idx0","ref_idx"]])
    print(f"  {fc} features per event")

    # 4. Simulate
    print("\n§4: RefConfig simulation...")
    t4 = time.monotonic()
    trades = simulate_ref_trades(events, sym_data)
    nt = (trades["exit_type"] != "NO_TRIGGER").sum()
    print(f"  {len(trades)} events → {nt} triggered ({time.monotonic()-t4:.0f}s)")

    # 5. Targets
    print("\n§5: Targets...")
    tgt = compute_targets(events, sym_data)
    trades = trades.merge(tgt, on="event_id", how="left")
    events = events.merge(tgt, on="event_id", how="left")
    tr_act = trades[trades["exit_type"] != "NO_TRIGGER"]
    print(f"  Tail rate: {tr_act['Y_tail'].mean():.1%}")

    # 6. Merge & splits
    print("\n§6: Merge & splits...")
    merged = trades.merge(events.drop(columns=["symbol","t_signal"], errors="ignore"),
                           on="event_id", how="left", suffixes=("","_ev"))
    merged = label_splits(merged)

    events.to_parquet(OUTPUT_DIR / "events.parquet", index=False)
    trades.to_csv(OUTPUT_DIR / "trades_ref.csv", index=False)

    # 7. Baseline
    print("\n§7: Baseline...")
    bl_m = []
    for sc, sl in [("split_fwd","test_fwd"), ("split_rev","test_rev")]:
        tt = merged[merged[sc] == "test"]
        m = compute_selector_metrics(tt, "baseline", "none", 1.0, sl)
        bl_m.append(m)
        print(f"  {sl}: N={m['N']}, mean={m['mean']:.1f}, med={m['median']:.1f}, "
              f"PF={m['PF']:.2f}, TP={m['hit_TP']:.0%}, tail={m.get('tail_rate',0):.0%}, "
              f"top5={m.get('top5_conc',0):.0%}, wk={m['weeks_pos']}/{m['weeks_total']}")

    # 8. Quantile
    print("\n§8: Quantile selectors...")
    all_m = list(bl_m); all_wk = []
    for sc, sl in [("split_fwd","test_fwd"), ("split_rev","test_rev")]:
        qm = run_quantile_selectors(merged, sc, sl)
        all_m.extend(qm)
        print(f"  {sl}: {len(qm)} selector-quintile combos")

    # 9. Model
    print("\n§9: Model selectors...")
    for sc, sl in [("split_fwd","test_fwd"), ("split_rev","test_rev")]:
        mm = run_model_selector(merged, sc, sl)
        all_m.extend(mm)
        print(f"  {sl}: {len(mm)} model configs")

    # Save report
    pd.DataFrame(all_m).to_csv(OUTPUT_DIR / "selector_report.csv", index=False)

    # 10. Shuffle
    print("\n§10: Shuffle sanity...")
    shuf = shuffle_sanity(merged, all_m)
    shuf.to_csv(OUTPUT_DIR / "shuffle_sanity.csv", index=False)

    # 11. Findings
    print("\n§11: Generating FINDINGS...")
    generate_findings(bl_m, all_m, shuf, events, trades)

    elapsed = time.monotonic() - T0
    print(f"\nXS-7.3 done in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
