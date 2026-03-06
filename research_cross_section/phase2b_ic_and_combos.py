"""
Phase 2b: Correct IC analysis + multi-combo portfolio comparison.

Fixes vs Phase 1:
  - Clean data (post-2025-01-01 corruption fix)
  - IC computed at 8h rebal frequency (non-overlapping returns) — same as portfolio
  - Includes -prem_z (flipped direction) as a candidate

Combos tested:
  A: funding + mom_24h          (hypothesis: best clean pair)
  B: funding + mom_24h - prem_z (flipped premium = mean-reversion)
  C: funding only
  D: mom_24h only
  E: funding + mom_24h + prem_z (original EW)
  F: funding + mom_48h          (longer momentum lookback)
  G: funding + mom_8h + mom_24h (momentum combo)

Walk-forward: 6mo train / 3mo OOS, rolling (maker fees = 4 bps/side)

Outputs:
  results/phase2b_ic_8h.csv        — IC at 8h frequency per signal
  results/phase2b_rolling_ic.csv   — rolling 90d IC per signal
  results/phase2b_combos.csv       — full-sample stats per combo
  results/phase2b_walkforward.csv  — per-window OOS stats per combo
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import time

warnings.filterwarnings("ignore")

SIGNALS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"

REBAL_FREQ       = "8h"
N_LONG           = 10
N_SHORT          = 10
FEE_MAKER_BPS    = 4    # per side
FEE_TAKER_BPS    = 10
PERIODS_PER_YEAR = 365 * 3   # 8h periods
TRAIN_MONTHS     = 6
OOS_MONTHS       = 3
ROLLING_IC_DAYS  = 90


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panels(signals_dir, cols):
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
    data = {c: {} for c in cols}
    for fpath in files:
        sym = os.path.basename(fpath).replace(".parquet", "")
        try:
            df = pd.read_parquet(fpath, columns=[c for c in cols if c in
                                  pd.read_parquet(fpath, columns=[]).columns
                                  or True])
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


def load_panels_fast(signals_dir, cols):
    """Load without double-read."""
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
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


# ---------------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------------

def cs_zscore(panel, min_valid=15):
    """Cross-sectional z-score, clipped at ±3."""
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-3, 3)
    z[n < min_valid] = np.nan
    return z


def to_8h_signal(panel):
    """Resample 1h signal to 8h, using value AT start of each window (no lookahead)."""
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()


def to_8h_fwd(panel, rebal_idx):
    """Align forward-return panel to rebal timestamps (exact reindex, no lookahead)."""
    aligned = panel.reindex(rebal_idx).clip(-0.99, 3.0)
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    return aligned


# ---------------------------------------------------------------------------
# IC at 8h frequency
# ---------------------------------------------------------------------------

def compute_ic_series(sig_8h, fwd_8h, min_valid=15):
    """Spearman IC at each 8h bar. Returns Series."""
    common_ts   = sig_8h.index.intersection(fwd_8h.index)
    common_syms = sig_8h.columns.intersection(fwd_8h.columns)
    S = sig_8h.loc[common_ts, common_syms]
    F = fwd_8h.loc[common_ts, common_syms]
    valid = S.notna() & F.notna()
    good  = valid.sum(axis=1) >= min_valid

    S = S.loc[good]; F = F.loc[good]
    if S.empty:
        return pd.Series(dtype=float)

    S_r  = S.rank(axis=1); F_r  = F.rank(axis=1)
    S_dm = S_r.sub(S_r.mean(axis=1), axis=0)
    F_dm = F_r.sub(F_r.mean(axis=1), axis=0)
    num  = (S_dm * F_dm).sum(axis=1)
    den  = np.sqrt((S_dm**2).sum(axis=1) * (F_dm**2).sum(axis=1))
    return (num / den.replace(0, np.nan)).dropna()


def ic_stats(ic_series, label=""):
    n    = len(ic_series)
    if n < 5:
        return {}
    mu   = ic_series.mean()
    sd   = ic_series.std()
    icir = mu / sd if sd > 0 else 0
    t    = icir * np.sqrt(n)
    pct_pos = (ic_series > 0).mean()
    return dict(label=label, n=n,
                ic_mean=round(mu,5), ic_std=round(sd,5),
                icir=round(icir,4), t_stat=round(t,3),
                pct_pos=round(pct_pos,3))


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def sim_portfolio(sig_8h, fwd_8h, n_long=N_LONG, n_short=N_SHORT,
                  fee_bps=FEE_MAKER_BPS, min_universe=20):
    fee_rt = fee_bps * 2 / 10000
    common = sig_8h.index.intersection(fwd_8h.index)
    records = []
    prev_longs, prev_shorts = set(), set()

    for ts in common:
        sig = sig_8h.loc[ts].dropna()
        fwd = fwd_8h.loc[ts, sig.index].dropna()
        sig = sig.loc[fwd.index]

        if len(sig) < min_universe:
            records.append(dict(timestamp=ts, gross=np.nan,
                                turnover=np.nan, net=np.nan, n=len(sig)))
            prev_longs = prev_shorts = set()
            continue

        ranked   = sig.rank()
        n        = len(ranked)
        longs    = set(ranked.nlargest(n_long).index)
        shorts   = set(ranked.nsmallest(n_short).index)

        lr = fwd.loc[list(longs)].mean()
        sr = fwd.loc[list(shorts)].mean()
        gross = lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan

        all_prev = prev_longs | prev_shorts
        if all_prev:
            changed  = len((longs - prev_longs) | (shorts - prev_shorts))
            turnover = changed / (n_long + n_short)
        else:
            turnover = 1.0

        net = (gross - turnover * fee_rt) if not np.isnan(gross) else np.nan
        records.append(dict(timestamp=ts, gross=gross,
                            turnover=turnover, net=net, n=len(sig)))
        prev_longs, prev_shorts = longs, shorts

    df = pd.DataFrame(records).set_index("timestamp")
    return df


def port_stats(net_series, label=""):
    s = net_series.dropna()
    if len(s) < 5:
        return dict(label=label)
    ar   = s.mean() * PERIODS_PER_YEAR
    av   = s.std()  * np.sqrt(PERIODS_PER_YEAR)
    sh   = ar / av  if av > 0 else np.nan
    down = s[s < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    so   = ar / down if down > 0 else np.nan
    cum  = (1 + s).cumprod()
    mdd  = (cum / cum.cummax() - 1).min()
    t    = (s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else np.nan
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
    )


def walk_forward(sig_8h, fwd_8h, fee_bps=FEE_MAKER_BPS):
    start, end = sig_8h.index.min(), sig_8h.index.max()
    windows, t = [], start + pd.DateOffset(months=TRAIN_MONTHS)
    while t + pd.DateOffset(months=OOS_MONTHS) <= end + pd.Timedelta(days=1):
        windows.append((t - pd.DateOffset(months=TRAIN_MONTHS), t,
                        t, t + pd.DateOffset(months=OOS_MONTHS)))
        t += pd.DateOffset(months=OOS_MONTHS)

    all_pnl, wf_rows = [], []
    for tr_s, tr_e, oo_s, oo_e in windows:
        # Train IC
        ic_tr = compute_ic_series(sig_8h.loc[tr_s:tr_e], fwd_8h.loc[tr_s:tr_e])
        train_ic = ic_tr.mean() if len(ic_tr) > 0 else np.nan

        # OOS sim
        sig_oos = sig_8h.loc[oo_s:oo_e]
        fwd_oos = fwd_8h.reindex(sig_oos.index).clip(-0.99, 3.0).replace([np.inf,-np.inf], np.nan)
        if sig_oos.empty:
            continue
        pnl = sim_portfolio(sig_oos, fwd_oos, fee_bps=fee_bps)
        st  = port_stats(pnl["net"], label=f"{oo_s.date()}–{oo_e.date()}")
        st["train_ic"] = round(train_ic, 4) if not np.isnan(train_ic) else np.nan
        st["gross_bps"] = round(pnl["gross"].mean()*10000, 2)
        st["turnover"]  = round(pnl["turnover"].mean(), 3)
        wf_rows.append(st)
        all_pnl.append(pnl)

    combined = pd.concat(all_pnl) if all_pnl else pd.DataFrame()
    return combined, wf_rows


# ---------------------------------------------------------------------------
# Build composite from a dict of {name: (panel, weight)}
# ---------------------------------------------------------------------------

def build_composite_from_spec(panels_z, spec):
    """
    spec = list of (col_name, direction, weight)
      direction: +1 or -1 (to flip signal)
    panels_z: dict of cs-zscored 8h panels
    """
    composite = None
    total_w = sum(abs(w) for _, _, w in spec)
    for col, direction, weight in spec:
        if col not in panels_z:
            continue
        z = panels_z[col] * direction * (weight / total_w)
        if composite is None:
            composite = z
        else:
            composite = composite.add(z, fill_value=0)
    return composite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    needed = ["prem_z", "funding", "mom_8h", "mom_24h", "mom_48h",
              "ls_z", "oi_div",
              "fwd_1h", "fwd_4h", "fwd_8h", "fwd_24h"]

    print("Loading signal parquets...")
    panels = load_panels_fast(SIGNALS_DIR, needed)
    print(f"  Loaded: {list(panels.keys())}\n")

    # ----------------------------------------------------------------
    # Step 1: IC at 8h frequency on clean data (non-overlapping)
    # ----------------------------------------------------------------
    print("=" * 80)
    print("STEP 1 — IC at 8h rebalancing frequency (non-overlapping, clean data)")
    print("=" * 80)

    signal_list = ["prem_z", "funding", "mom_8h", "mom_24h", "mom_48h", "ls_z", "oi_div"]
    fwd_labels  = ["fwd_1h", "fwd_4h", "fwd_8h", "fwd_24h"]

    # Build 8h signal panels
    panels_8h = {}
    for col in signal_list:
        if col in panels:
            panels_8h[col] = to_8h_signal(panels[col])

    # Rebal index from most complete signal
    rebal_idx = panels_8h.get("mom_24h", next(iter(panels_8h.values()))).index

    # Build 8h fwd panels
    fwds_8h = {}
    for fwd in fwd_labels:
        if fwd in panels:
            fwds_8h[fwd] = to_8h_fwd(panels[fwd], rebal_idx)

    ic_rows = []
    print(f"\n{'Signal':<20} {'Fwd':>7}  {'IC':>8}  {'ICIR':>7}  {'t-stat':>8}  {'%pos':>6}  {'N':>6}")
    print("-" * 70)

    for col in signal_list:
        if col not in panels_8h:
            continue
        sp = panels_8h[col]
        for fwd in fwd_labels:
            if fwd not in fwds_8h:
                continue
            ic_s = compute_ic_series(sp, fwds_8h[fwd])
            st   = ic_stats(ic_s, label=f"{col}|{fwd}")
            if not st:
                continue
            ic_rows.append({**st, "signal": col, "fwd": fwd})
            flag = " <<<" if abs(st["icir"]) > 0.3 else (" <<" if abs(st["icir"]) > 0.15 else "")
            print(f"  {col:<18} {fwd:>7}  {st['ic_mean']:>+8.4f}  "
                  f"{st['icir']:>7.3f}  {st['t_stat']:>8.2f}  "
                  f"{st['pct_pos']:>6.3f}  {st['n']:>6}{flag}")
        print()

    pd.DataFrame(ic_rows).to_csv(os.path.join(RESULTS_DIR, "phase2b_ic_8h.csv"), index=False)

    # Rolling IC for top signals
    print("\nRolling IC (trailing 90d) for key signals vs fwd_8h:")
    rolling_rows = []
    for col in ["funding", "mom_24h", "prem_z"]:
        if col not in panels_8h:
            continue
        ic_s = compute_ic_series(panels_8h[col], fwds_8h.get("fwd_8h", pd.DataFrame()))
        if ic_s.empty:
            continue
        roll = ic_s.rolling(ROLLING_IC_DAYS * 3, min_periods=30).mean()  # 90d * 3 bars/day
        for ts, v in roll.items():
            rolling_rows.append({"timestamp": ts, "signal": col, "rolling_ic": v})
        print(f"  {col:<20}: mean={ic_s.mean():+.4f}  min(90d)={roll.min():+.4f}  max(90d)={roll.max():+.4f}")
    pd.DataFrame(rolling_rows).to_csv(os.path.join(RESULTS_DIR, "phase2b_rolling_ic.csv"), index=False)

    # ----------------------------------------------------------------
    # Step 2: Build z-scored 8h signal panels for portfolio combos
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 — Portfolio combo comparison (Maker fees, walk-forward OOS)")
    print("=" * 80)

    panels_z = {col: cs_zscore(panels_8h[col]) for col in panels_8h}
    fwd_port = fwds_8h.get("fwd_8h", pd.DataFrame())

    # Define combos: (name, spec)
    # spec = list of (col, direction, weight)
    combos = [
        ("A: funding + mom_24h",
         [("funding", +1, 1), ("mom_24h", +1, 1)]),

        ("B: funding + mom_24h - prem_z",
         [("funding", +1, 1), ("mom_24h", +1, 1), ("prem_z", -1, 1)]),

        ("C: funding only",
         [("funding", +1, 1)]),

        ("D: mom_24h only",
         [("mom_24h", +1, 1)]),

        ("E: funding + mom_24h + prem_z",
         [("funding", +1, 1), ("mom_24h", +1, 1), ("prem_z", +1, 1)]),

        ("F: funding + mom_48h",
         [("funding", +1, 1), ("mom_48h", +1, 1)]),

        ("G: funding + mom_8h + mom_24h",
         [("funding", +1, 1), ("mom_8h", +1, 1), ("mom_24h", +1, 1)]),

        ("H: funding + mom_24h + mom_48h",
         [("funding", +1, 1), ("mom_24h", +1, 1), ("mom_48h", +1, 1)]),
    ]

    all_combo_stats = []
    all_wf_rows     = []

    # --- Full-sample header ---
    print(f"\n{'Combo':<36} {'Mean bps':>9} {'Ann%':>7} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'WinR':>6} {'t':>6}")
    print("-" * 90)

    for name, spec in combos:
        comp = build_composite_from_spec(panels_z, spec)
        if comp is None or comp.empty:
            continue

        # Full-sample (maker)
        pnl  = sim_portfolio(comp, fwd_port, fee_bps=FEE_MAKER_BPS)
        fs   = port_stats(pnl["net"], label=name)
        fs["combo"] = name
        fs["avg_turnover"] = round(pnl["turnover"].mean(), 3)
        all_combo_stats.append(fs)

        print(f"  {name:<34} {fs.get('mean_bps',0):>9.2f} {fs.get('ann_ret',0):>7.1f} "
              f"{fs.get('sharpe',0):>7.3f} {fs.get('max_dd',0):>8.1f} "
              f"{fs.get('win_rate',0):>6.3f} {fs.get('t_stat',0):>6.2f}")

        # Walk-forward (maker)
        wf_pnl, wf_rows = walk_forward(comp, fwd_port, fee_bps=FEE_MAKER_BPS)
        for r in wf_rows:
            r["combo"] = name
        all_wf_rows.extend(wf_rows)

    # --- Walk-forward summary per combo ---
    print("\n--- Walk-forward OOS (Maker) ---")
    wf_df = pd.DataFrame(all_wf_rows)
    if not wf_df.empty:
        for name, _ in combos:
            sub = wf_df[wf_df["combo"] == name]
            if sub.empty:
                continue
            # Compute combined OOS stats
            oos_nets = []
            for _, row in sub.iterrows():
                pass  # already aggregated per window

            mean_sh  = sub["sharpe"].mean()
            min_sh   = sub["sharpe"].min()
            n_pos    = (sub["sharpe"] > 0).sum()
            n_total  = len(sub)
            mean_net = sub["mean_bps"].mean()
            mean_tr  = sub["turnover"].mean()
            print(f"  {name:<34} avg_Sharpe={mean_sh:+.3f}  min={min_sh:+.3f}  "
                  f"pos={n_pos}/{n_total}  avg_net={mean_net:.1f}bps  turn={mean_tr:.3f}")

    # Also print per-window for best combo (A)
    best_name = "A: funding + mom_24h"
    print(f"\n  Per-window detail for '{best_name}':")
    sub = wf_df[wf_df["combo"] == best_name] if not wf_df.empty else pd.DataFrame()
    if not sub.empty:
        print(f"  {'Window':<24} {'TrainIC':>8} {'Gross':>8} {'Net':>7} {'Sharpe':>7} {'MaxDD':>7} {'Turn':>6}")
        for _, r in sub.iterrows():
            print(f"  {r['label']:<24} {r.get('train_ic',0):>8.4f} "
                  f"{r.get('gross_bps',0):>8.2f} {r.get('mean_bps',0):>7.2f} "
                  f"{r.get('sharpe',0):>7.3f} {r.get('max_dd',0):>7.1f} "
                  f"{r.get('turnover',0):>6.3f}")

    # Fee sensitivity for best combo (A)
    print(f"\n--- Fee sensitivity for '{best_name}' ---")
    comp_a = build_composite_from_spec(panels_z,
                 [("funding", +1, 1), ("mom_24h", +1, 1)])
    for fee_label, fee_bps in [("Taker(10)", FEE_TAKER_BPS), ("Mixed(7)", 7),
                                ("Maker(4)", FEE_MAKER_BPS), ("Maker(2)", 2)]:
        pnl = sim_portfolio(comp_a, fwd_port, fee_bps=fee_bps)
        st  = port_stats(pnl["net"])
        avg_turn = pnl["turnover"].mean()
        eff_cost = avg_turn * fee_bps * 2
        print(f"  {fee_label:<12} Sharpe={st.get('sharpe',0):>6.3f}  "
              f"Net={st.get('mean_bps',0):>7.2f}bps  "
              f"Gross={pnl['gross'].mean()*10000:>7.2f}bps  "
              f"EffCost={eff_cost:.1f}bps/bar")

    # Save
    pd.DataFrame(all_combo_stats).to_csv(
        os.path.join(RESULTS_DIR, "phase2b_combos.csv"), index=False)
    if not wf_df.empty:
        wf_df.to_csv(os.path.join(RESULTS_DIR, "phase2b_walkforward.csv"), index=False)

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
