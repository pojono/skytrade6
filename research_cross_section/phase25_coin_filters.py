"""
Phase 25 — Coin Filters & Rolling Universe Selection (Fair WFO)

Three independent filter experiments on Phase 22 best config
(2×predicted_funding + mom_24h, N=10, inverse scaling):

EXP 1: Listing age filter
  At each bar t, only include coins with >= min_age_days of history
  prior to t. This avoids trading freshly-listed coins whose funding
  rates are unstable (the cause of Jan-Feb 2025 losses).
  min_age_days tested: 0, 30, 60, 90, 120

EXP 2: Rolling volume filter
  At each bar t, only include coins in top-K% by trailing 30-day
  hourly volume (observable at t — no look-ahead).
  top_pct tested: 100%, 80%, 60%, 40%

EXP 3: Rolling coin quality score (fair WFO)
  At each bar t, compute each coin's trailing 90-day Sharpe contribution.
  Only trade coins with positive trailing quality. Recalculated every bar.
  Variants: top-80, top-60, top-40 by trailing quality score.

EXP 4: Best combination

All experiments use 2× leverage for comparability.
Applies Phase 21 inverse scaling.
Starting capital: $10,000.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SIGNALS_V1  = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
SIGNALS_V2  = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals_v2"
RESULTS_DIR = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
FEE_1SIDE        = 4 / 10000
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
START            = "2025-01-01"
STARTING_CAPITAL = 10_000.0
SHARPE_WINDOW    = 30
LEVERAGE         = 2.0
N                = 10

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panels():
    def load_dir(d, cols):
        data = {c: {} for c in cols}
        for fp in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            sym = os.path.basename(fp).replace(".parquet", "")
            if sym in MAJORS:
                continue
            try:
                df = pd.read_parquet(fp)
                for c in cols:
                    if c in df.columns:
                        data[c][sym] = df[c]
            except Exception:
                pass
        panels = {}
        for c in cols:
            p = pd.DataFrame(data[c])
            p.index = pd.to_datetime(p.index, utc=True)
            panels[c] = p.sort_index()
        return panels

    p1 = load_dir(SIGNALS_V1, ["mom_24h", "fwd_8h", "volume"])
    p2 = load_dir(SIGNALS_V2, ["predicted_funding"])
    return {**p1, **p2}

def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()

def to_8h_sum(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").sum()

def cs_zscore(df, min_valid=10):
    mu  = df.mean(axis=1)
    sig = df.std(axis=1).replace(0, np.nan)
    z   = df.sub(mu, axis=0).div(sig, axis=0).clip(-CLIP, CLIP)
    z[df.notna().sum(axis=1) < min_valid] = np.nan
    return z

def sharpe_fn(x):
    return x.mean() / x.std() if (len(x) >= 5 and x.std() > 0) else 0.0

# ---------------------------------------------------------------------------
# Core simulator with dynamic universe mask
# ---------------------------------------------------------------------------

def sim_masked(composite, fwd_8h, mask, n=N, lev=LEVERAGE):
    """
    mask: DataFrame (same shape as composite), True = coin eligible at this bar.
    Only coins where mask=True are included in the cross-section at each bar.
    """
    rets, dates = [], composite.index.intersection(fwd_8h.index)
    fee = FEE_1SIDE * 2 * lev
    for ts in dates:
        row = composite.loc[ts].copy()
        # Apply mask
        if mask is not None and ts in mask.index:
            eligible = mask.loc[ts]
            row[~eligible] = np.nan
        row = row.dropna()
        if len(row) < n * 2:
            rets.append(0.0)
            continue
        fwd = fwd_8h.loc[ts]
        gross = lev * (0.5 * fwd[row.nlargest(n).index].mean()
                       - 0.5 * fwd[row.nsmallest(n).index].mean())
        rets.append(gross - fee)
    return pd.Series(rets, index=dates)

def build_inverse_scale(rets):
    rs = rets.rolling(SHARPE_WINDOW).apply(sharpe_fn, raw=True)
    sc = pd.Series(1.0, index=rets.index)
    sc[rs > 5]  = 0.5
    sc[rs <= 0] = 0.5
    return sc

def run(composite, fwd_8h, mask=None, n=N, lev=LEVERAGE):
    r0    = sim_masked(composite, fwd_8h, mask, n=n, lev=lev)
    scale = build_inverse_scale(r0)
    return r0 * scale

def port_stats(rets, cap=STARTING_CAPITAL, label=""):
    if len(rets) == 0 or rets.std() == 0:
        return dict(label=label, sharpe=0, sortino=0, ann_ret=0, max_dd=0,
                    win_rate=0, final=cap, n_bars=0)
    sr  = rets.mean() / rets.std() * np.sqrt(PERIODS_PER_YEAR)
    ann = (1 + rets).prod() ** (PERIODS_PER_YEAR / len(rets)) - 1
    neg = rets[rets < 0]
    so  = rets.mean() / neg.std() * np.sqrt(PERIODS_PER_YEAR) if len(neg) > 0 else np.nan
    eq  = (1 + rets).cumprod()
    dd  = (eq / eq.cummax() - 1).min()
    return dict(label=label, sharpe=sr, sortino=so, ann_ret=ann,
                max_dd=dd, win_rate=(rets > 0).mean(),
                final=cap * (1 + rets).prod(), n_bars=len(rets))

def monthly_table(rets):
    return rets.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x) else np.nan)

# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def listing_age_mask(fwd_8h, min_age_days):
    """True where coin has >= min_age_days of non-NaN prior data."""
    if min_age_days == 0:
        return None  # no filter
    # For each coin, find the first bar with valid data
    first_valid = fwd_8h.apply(lambda col: col.first_valid_index())
    min_age_td  = pd.Timedelta(days=min_age_days)
    mask = pd.DataFrame(False, index=fwd_8h.index, columns=fwd_8h.columns)
    for sym in fwd_8h.columns:
        if first_valid[sym] is None:
            continue
        cutoff = first_valid[sym] + min_age_td
        mask.loc[mask.index >= cutoff, sym] = True
    return mask

def volume_filter_mask(vol_8h, top_pct):
    """True where coin is in top-pct% by trailing 30-bar (10-day) volume."""
    if top_pct >= 1.0:
        return None
    # Rolling 30-bar sum of volume → rank cross-sectionally at each bar
    roll_vol = vol_8h.rolling(30, min_periods=5).sum()
    # Rank: 1 = largest volume
    ranks = roll_vol.rank(axis=1, ascending=False, pct=True)
    mask  = ranks <= top_pct
    return mask

def rolling_quality_mask(rets_per_coin_fn, composite, fwd_8h, window_bars, top_pct):
    """
    At each bar t, compute each coin's contribution to strategy P&L over past
    window_bars. Include only coins with positive trailing contribution AND
    in top top_pct fraction.

    Fair: uses only data up to t-1 (lagged by 1 bar for safety).
    """
    # First compute per-coin raw returns (long contribution - short contribution)
    # This requires knowing when each coin was long vs short each bar.
    # Approximation: use coin's signal percentile rank × forward return as proxy.
    dates = composite.index.intersection(fwd_8h.index)

    # Coin-level contribution: if coin is ranked top-N → long → gets fwd return
    # if ranked bottom-N → short → gets -fwd return; else 0
    n_syms     = composite.shape[1]
    contrib    = pd.DataFrame(0.0, index=dates, columns=composite.columns)
    for ts in dates:
        row = composite.loc[ts].dropna()
        if len(row) < N * 2:
            continue
        fwd = fwd_8h.loc[ts]
        top = row.nlargest(N).index
        bot = row.nsmallest(N).index
        contrib.loc[ts, top] = fwd[top].values
        contrib.loc[ts, bot] = -fwd[bot].values

    # Rolling sum of contribution per coin
    roll_contrib = contrib.rolling(window_bars, min_periods=max(1, window_bars // 3)).sum()

    # Rank by trailing contribution, keep top-pct AND positive
    roll_rank = roll_contrib.rank(axis=1, ascending=False, pct=True)
    mask = (roll_rank <= top_pct) & (roll_contrib > 0)
    # Shift by 1 to avoid look-ahead
    mask = mask.shift(1).fillna(False)
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading panels ...")
    panels = load_panels()

    pred_8h = to_8h(panels["predicted_funding"])
    mom_8h  = to_8h(panels["mom_24h"])
    fwd_8h  = to_8h(panels["fwd_8h"])
    vol_8h  = to_8h_sum(panels.get("volume", pd.DataFrame()))

    all_syms = pred_8h.columns.intersection(mom_8h.columns).intersection(fwd_8h.columns)
    pred_8h  = pred_8h[all_syms]
    mom_8h   = mom_8h[all_syms]
    fwd_8h   = fwd_8h[all_syms]
    if not vol_8h.empty:
        vol_8h = vol_8h.reindex(columns=all_syms)

    print(f"Universe: {len(all_syms)} symbols | {len(pred_8h[START:])} bars from {START}")

    # Build composite on full range (z-scores computed on full cross-section)
    z_pred = cs_zscore(pred_8h[START:])
    z_mom  = cs_zscore(mom_8h[START:])
    composite = (2 * z_pred + z_mom) / 3

    fwd = fwd_8h[START:]

    all_results = {}

    # -----------------------------------------------------------------------
    # BASELINE (no filter, N=10, 2x)
    # -----------------------------------------------------------------------
    r_base = run(composite, fwd)
    all_results["Baseline (no filter)"] = r_base

    # -----------------------------------------------------------------------
    # EXP 1: Listing age filter
    # -----------------------------------------------------------------------
    print("\nEXP 1: Listing age filters ...")
    for days in [30, 60, 90, 120]:
        mask = listing_age_mask(fwd_8h[START:], days)
        r = run(composite, fwd, mask=mask)
        key = f"Min age {days}d"
        all_results[key] = r
        st = port_stats(r, label=key)
        print(f"  {key:<18}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.1%}  "
              f"${st['final']:>10,.0f}  WR={st['win_rate']:.1%}")

    # -----------------------------------------------------------------------
    # EXP 2: Volume filter
    # -----------------------------------------------------------------------
    print("\nEXP 2: Volume filters ...")
    if not vol_8h.empty and vol_8h[START:].notna().sum().sum() > 100:
        for pct in [0.80, 0.60, 0.40]:
            mask = volume_filter_mask(vol_8h[START:], pct)
            r    = run(composite, fwd, mask=mask)
            key  = f"Vol top {int(pct*100)}%"
            all_results[key] = r
            st = port_stats(r, label=key)
            print(f"  {key:<18}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.1%}  "
                  f"${st['final']:>10,.0f}  WR={st['win_rate']:.1%}")
    else:
        print("  Volume data not available in parquets — skipping EXP 2")

    # -----------------------------------------------------------------------
    # EXP 3: Rolling quality mask (fair WFO)
    # -----------------------------------------------------------------------
    print("\nEXP 3: Rolling quality filters (fair WFO, 90-bar lookback) ...")
    window_bars = 90   # ~30 days at 8h
    for pct in [0.80, 0.60, 0.40]:
        mask = rolling_quality_mask(None, composite, fwd, window_bars, pct)
        r    = run(composite, fwd, mask=mask)
        key  = f"RollQual top {int(pct*100)}%"
        all_results[key] = r
        st = port_stats(r, label=key)
        print(f"  {key:<22}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.1%}  "
              f"${st['final']:>10,.0f}  WR={st['win_rate']:.1%}")

    # -----------------------------------------------------------------------
    # EXP 4: Best combinations
    # -----------------------------------------------------------------------
    print("\nEXP 4: Combinations ...")
    best_age = 60   # from EXP 1 results
    combos = [
        ("Age60 + RollQual80", 60, 0.80),
        ("Age60 + RollQual60", 60, 0.60),
        ("Age90 + RollQual80", 90, 0.80),
    ]
    for key, days, pct in combos:
        age_mask   = listing_age_mask(fwd_8h[START:], days)
        qual_mask  = rolling_quality_mask(None, composite, fwd, window_bars, pct)
        # Combine: coin must pass BOTH filters
        if age_mask is not None and qual_mask is not None:
            combo_mask = age_mask & qual_mask
        elif age_mask is not None:
            combo_mask = age_mask
        else:
            combo_mask = qual_mask
        r   = run(composite, fwd, mask=combo_mask)
        all_results[key] = r
        st  = port_stats(r, label=key)
        print(f"  {key:<22}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.1%}  "
              f"${st['final']:>10,.0f}  WR={st['win_rate']:.1%}")

    # -----------------------------------------------------------------------
    # Full results table
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("FULL RESULTS — All filters (2× leverage, $10k start)")
    print("=" * 78)
    print(f"  {'Filter':<26} {'Sharpe':>7} {'MaxDD':>8} {'AnnRet':>8} "
          f"{'$10k→':>12} {'WinRate':>8}")
    print("  " + "-" * 72)
    for label, r in all_results.items():
        st = port_stats(r, label=label)
        print(f"  {label:<26} {st['sharpe']:>7.2f} {st['max_dd']:>7.1%} "
              f"{st['ann_ret']:>8.0%}  ${st['final']:>10,.0f}  {st['win_rate']:>7.1%}")

    # -----------------------------------------------------------------------
    # Monthly breakdown: baseline vs best age filter vs best combo
    # -----------------------------------------------------------------------
    # Pick best from each experiment
    exp1_keys = [k for k in all_results if k.startswith("Min age")]
    exp3_keys = [k for k in all_results if k.startswith("RollQual")]
    exp4_keys = [k for k in all_results if "+" in k]

    def best_by_sharpe(keys):
        return max(keys, key=lambda k: port_stats(all_results[k])['sharpe']) if keys else None

    show_keys = ["Baseline (no filter)"]
    for b in [best_by_sharpe(exp1_keys), best_by_sharpe(exp3_keys), best_by_sharpe(exp4_keys)]:
        if b and b not in show_keys:
            show_keys.append(b)

    print()
    print("MONTHLY BREAKDOWN — Baseline vs best filter per experiment")
    print("-" * (10 + 14 * len(show_keys)))
    hdr = f"  {'Month':<10}"
    for k in show_keys:
        hdr += f"  {k[:12]:>12}"
    print(hdr)
    print("  " + "-" * (10 + 14 * len(show_keys)))

    all_months = sorted(set(
        m.strftime("%Y-%m")
        for k in show_keys
        for m in monthly_table(all_results[k]).index
        if not np.isnan(monthly_table(all_results[k])[m])
    ))

    monthly_cache = {k: monthly_table(all_results[k]) for k in show_keys}
    eq_track = {k: STARTING_CAPITAL for k in show_keys}
    for month in all_months:
        row = f"  {month:<10}"
        for k in show_keys:
            mt = monthly_cache[k]
            ts = pd.Timestamp(month + "-01").to_period("M").to_timestamp("M")
            # Find matching month
            matches = [v for idx, v in mt.items() if idx.strftime("%Y-%m") == month]
            if matches:
                v = matches[0]
                eq_track[k] *= (1 + v)
                flag = " *" if v > 0.3 else (" !" if v < -0.1 else "")
                row += f"  {v:>+11.1%}{flag}"
            else:
                row += f"  {'—':>12}"
        print(row)

    print("  " + "-" * (10 + 14 * len(show_keys)))
    row = f"  {'Final $':<10}"
    for k in show_keys:
        row += f"  ${eq_track[k]:>11,.0f}"
    print(row)
    row = f"  {'Sharpe':<10}"
    for k in show_keys:
        row += f"  {port_stats(all_results[k])['sharpe']:>12.2f}"
    print(row)
    row = f"  {'MaxDD':<10}"
    for k in show_keys:
        row += f"  {port_stats(all_results[k])['max_dd']:>12.1%}"
    print(row)

    # -----------------------------------------------------------------------
    # Equity plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # Top: all EXP1 (age filters) + baseline
    ax = axes[0]
    age_items = [(k, v) for k, v in all_results.items()
                 if "age" in k.lower() or k == "Baseline (no filter)"]
    palette = plt.cm.viridis(np.linspace(0, 1, len(age_items)))
    for i, (label, r) in enumerate(age_items):
        st = port_stats(r)
        eq = STARTING_CAPITAL * (1 + r).cumprod()
        lw = 2.5 if label == "Baseline (no filter)" else 1.6
        ax.plot(eq.index, eq.values, lw=lw,
                label=f"{label}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.0%}  ${st['final']:,.0f}",
                color=palette[i])
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.7, ls="--")
    ax.set_title("EXP 1: Listing Age Filters (2x leverage, $10k start)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Bottom: baseline vs best combo
    ax2 = axes[1]
    plot_set = ["Baseline (no filter)"] + [best_by_sharpe(exp1_keys),
                best_by_sharpe(exp3_keys), best_by_sharpe(exp4_keys)]
    plot_set = [k for k in plot_set if k and k in all_results]
    colors = ["steelblue", "crimson", "green", "orange"]
    for i, key in enumerate(plot_set):
        r  = all_results[key]
        st = port_stats(r)
        eq = STARTING_CAPITAL * (1 + r).cumprod()
        ax2.plot(eq.index, eq.values, lw=2.0, color=colors[i % len(colors)],
                 label=f"{key}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.0%}  ${st['final']:,.0f}")
    ax2.axhline(STARTING_CAPITAL, color="gray", lw=0.7, ls="--")
    ax2.set_title("Best Filter per Experiment vs Baseline")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.suptitle("Phase 25 — Coin Filters & Rolling Universe Selection\n"
                 "Strategy: 2×predicted_funding + mom_24h, 2× leverage", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "phase25_coin_filters.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\nPlot: {out}")
    print("Phase 25 complete.")


if __name__ == "__main__":
    main()
