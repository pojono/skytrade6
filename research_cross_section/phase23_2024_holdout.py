"""
Phase 23 — 2024 Holdout (True OOS)

The existing signals/ parquets only cover 2025+. This script rebuilds
the three signals needed for Phase 21 config directly from raw kline/funding
files — no cached parquets needed.

Signals built inline (per symbol, streaming):
  close      : 1h resample of kline_1m
  funding    : forward-filled settlement rate
  funding_trend: diff(funding, 24h)
  mom_24h    : close.pct_change(24)
  fwd_8h     : close.pct_change(8).shift(-8)

Strategy config (frozen Phase 21):
  Composite  : (2×z_funding + z_mom24h + z_funding_trend) / 4
  Universe   : no-Majors (~113 coins)
  N          : 10 long + 10 short, equal-weight
  Fees       : 8 bps round-trip (maker)
  Scaling    : inverse (0.5× when rolling Sharpe > 5 or < 0)

Tests:
  T1: 2024 full year  — strict holdout, never touched
  T2: 2025–2026       — in-sample reference
  T3: combined        — full picture
"""

import os, glob, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DATALAKE         = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
CACHE_DIR        = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals_2024"
RESULTS_DIR      = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
UNIVERSE_FILE    = "/home/ubuntu/Projects/skytrade6/research_cross_section/universe.txt"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
FEE_RT           = 4 * 2 / 10000
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
SHARPE_WINDOW    = 30
STARTING_CAPITAL = 1000.0

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

# ---------------------------------------------------------------------------
# Per-symbol signal builder (streaming from raw files)
# ---------------------------------------------------------------------------

def build_symbol(symbol):
    """Build close/funding/mom_24h/funding_trend/fwd_8h for one symbol."""
    sym_dir = os.path.join(DATALAKE, symbol)

    # --- close: 1h resample of all kline_1m files ---
    kfiles = sorted([
        f for f in glob.glob(os.path.join(sym_dir, "*_kline_1m.csv"))
        if "premium" not in f and "mark" not in f
    ])
    if not kfiles:
        return None
    chunks = []
    for f in kfiles:
        try:
            d = pd.read_csv(f, usecols=["startTime", "close"])
            d["startTime"] = pd.to_datetime(d["startTime"], unit="ms", utc=True)
            d = d.set_index("startTime").sort_index()
            h = d["close"].resample("1h").last().dropna()
            chunks.append(h)
        except Exception:
            pass
    if not chunks:
        return None
    close = pd.concat(chunks).sort_index()
    close = close[~close.index.duplicated(keep="last")]

    # Data quality: zero / corrupted prices
    med = close.median()
    bad = (close <= 0) | (close < med * 0.01)
    if bad.sum() > 0:
        close[bad] = np.nan
        close = close.interpolate(method="time", limit=12)
    close = close.dropna()
    if len(close) < 500:
        return None

    h_index = close.index

    # --- funding: settlement forward-filled ---
    ffiles = sorted(glob.glob(os.path.join(sym_dir, "*_funding_rate.csv")))
    rows = []
    for f in ffiles:
        try:
            d = pd.read_csv(f, usecols=["timestamp", "fundingRate"])
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return None
    fdf = pd.concat(rows, ignore_index=True)
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], unit="ms", utc=True)
    fdf = fdf.set_index("timestamp").sort_index()["fundingRate"]
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    funding = fdf.reindex(h_index, method="ffill")

    # --- signals ---
    mom_24h       = close.pct_change(24, fill_method=None)
    funding_trend = funding.diff(24)
    fwd_8h        = close.pct_change(8, fill_method=None).shift(-8)

    df = pd.DataFrame({
        "close":          close,
        "funding":        funding,
        "mom_24h":        mom_24h,
        "funding_trend":  funding_trend,
        "fwd_8h":         fwd_8h,
    })
    return df


def load_or_build(symbol):
    cache = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = build_symbol(symbol)
    if df is not None:
        df.to_parquet(cache)
    return df


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------

def load_panels(universe):
    cols = ["funding", "funding_trend", "mom_24h", "fwd_8h"]
    data = {c: {} for c in cols}
    built = 0
    t0 = time.time()
    for i, sym in enumerate(universe):
        df = load_or_build(sym)
        if df is None:
            continue
        for c in cols:
            if c in df.columns:
                data[c][sym] = df[c]
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(universe)}] {sym:<22}  built={built}  "
                  f"{time.time()-t0:.0f}s")
    panels = {}
    for c in cols:
        p = pd.DataFrame(data[c])
        p.index = pd.to_datetime(p.index, utc=True)
        panels[c] = p.sort_index()
    return panels


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------

def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    n   = panel.notna().sum(axis=1)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-CLIP, CLIP)
    z[n < min_valid] = np.nan
    return z

def sharpe_fn(x):
    return x.mean() / x.std() if (len(x) >= 5 and x.std() > 0) else 0.0

def sim(composite, fwd_8h, n=10, scale_series=None):
    rets, dates = [], composite.index.intersection(fwd_8h.index)
    for ts in dates:
        row = composite.loc[ts].dropna()
        if len(row) < n * 2:
            rets.append(0.0)
            continue
        fwd    = fwd_8h.loc[ts]
        gross  = (0.5 * fwd[row.nlargest(n).index].mean()
                  - 0.5 * fwd[row.nsmallest(n).index].mean())
        rets.append(gross - FEE_RT)
    s = pd.Series(rets, index=dates)
    if scale_series is not None:
        s = s * scale_series.reindex(s.index).fillna(1.0)
    return s

def build_inverse_scale(rets):
    rs = rets.rolling(SHARPE_WINDOW).apply(sharpe_fn, raw=True)
    sc = pd.Series(1.0, index=rets.index)
    sc[rs > 5]  = 0.5
    sc[rs <= 0] = 0.5
    return sc

def port_stats(rets, label=""):
    if len(rets) == 0 or rets.std() == 0:
        return dict(label=label, sharpe=0, sortino=0, ann_ret=0,
                    max_dd=0, win_rate=0, n_bars=len(rets), final=STARTING_CAPITAL)
    sr  = rets.mean() / rets.std() * np.sqrt(PERIODS_PER_YEAR)
    ann = (1 + rets).prod() ** (PERIODS_PER_YEAR / max(len(rets), 1)) - 1
    neg = rets[rets < 0]
    so  = rets.mean() / neg.std() * np.sqrt(PERIODS_PER_YEAR) if len(neg) > 0 else np.nan
    eq  = (1 + rets).cumprod()
    dd  = (eq / eq.cummax() - 1).min()
    final = STARTING_CAPITAL * (1 + rets).prod()
    return dict(label=label, sharpe=sr, sortino=so, ann_ret=ann,
                max_dd=dd, win_rate=(rets > 0).mean(), n_bars=len(rets), final=final)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(UNIVERSE_FILE) as f:
        universe = [l.strip() for l in f if l.strip() and l.strip() not in MAJORS]

    print(f"Universe: {len(universe)} symbols (Majors excluded)")
    print("Building / loading signals from raw files ...")
    panels = load_panels(universe)

    fund_8h = to_8h(panels["funding"])
    ft_8h   = to_8h(panels["funding_trend"])
    mom_8h  = to_8h(panels["mom_24h"])
    fwd_8h  = to_8h(panels["fwd_8h"])

    all_syms = (fund_8h.columns.intersection(ft_8h.columns)
                               .intersection(mom_8h.columns)
                               .intersection(fwd_8h.columns))
    fund_8h = fund_8h[all_syms]
    ft_8h   = ft_8h[all_syms]
    mom_8h  = mom_8h[all_syms]
    fwd_8h  = fwd_8h[all_syms]

    print(f"\nFinal universe : {len(all_syms)} symbols")
    print(f"Data range     : {fund_8h.index[0].date()} → {fund_8h.index[-1].date()}")
    print(f"2024 bars      : {len(fund_8h['2024-01-01':'2024-12-31'])}")
    print(f"2025+ bars     : {len(fund_8h['2025-01-01':])}")
    print()

    # Composite (Phase 21 frozen config: 2×funding + mom_24h + funding_trend, 2:1:1)
    # NaN-safe: if funding_trend is flat cross-sectionally (as in 2024 when funding
    # was uniformly capped), we fill its z-score with 0 (neutral) so the other two
    # signals still drive the composite. This is correct — a missing signal should
    # contribute nothing, not poison the whole composite.
    z_fund = cs_zscore(fund_8h)
    z_ft   = cs_zscore(ft_8h).fillna(0)   # 0 = neutral when cross-section is flat
    z_mom  = cs_zscore(mom_8h).fillna(0)
    # Mask bars where the primary signal (funding) is also unavailable
    composite = (2 * z_fund + z_mom + z_ft) / 4
    composite[z_fund.isna()] = np.nan

    # How many coins are valid at each period?
    valid_2024 = (fund_8h["2024-01-01":"2024-12-31"].notna().sum(axis=1) > 0).sum()
    n_coins_2024 = fund_8h["2024-01-01":"2024-12-31"].notna().any().sum()
    n_coins_2025 = fund_8h["2025-01-01":].notna().any().sum()
    print(f"Coins with ANY 2024 data : {n_coins_2024}")
    print(f"Coins with ANY 2025 data : {n_coins_2025}")
    print(f"NOTE: Most alpha coins (meme/AI/infra) launched in 2025 — 2024 universe is much smaller.")
    print()

    # -----------------------------------------------------------------------
    # Run three time periods
    # -----------------------------------------------------------------------
    # 2024: use n=8 (need 16 valid rows; we have ~20 coins available)
    # 2025+: use n=10 as before
    results = {}
    for s, e, label, n in [
        ("2024-01-01", "2024-12-31", "2024 holdout (OOS)", 8),
        ("2025-01-01", "2026-03-06", "2025–2026 (in-sample ref)", 10),
        ("2024-01-01", "2026-03-06", "2024+2025 combined", 10),
    ]:
        comp_p = composite[s:e]
        fwd_p  = fwd_8h[s:e]
        r0     = sim(comp_p, fwd_p, n=n)
        scale  = build_inverse_scale(r0)
        r      = sim(comp_p, fwd_p, n=n, scale_series=scale)
        results[label] = (r, port_stats(r, label))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("PHASE 23 — 2024 TRUE OOS HOLDOUT")
    print("=" * 72)
    print(f"  {'Period':<28} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} "
          f"{'AnnRet':>8} {'$1k→':>8} {'WinRate':>8}")
    print("  " + "-" * 68)
    for label, (r, st) in results.items():
        print(f"  {st['label']:<28} {st['sharpe']:>7.3f} {st['sortino']:>8.3f} "
              f"{st['max_dd']:>7.1%} {st['ann_ret']:>8.0%}  ${st['final']:>7.0f}  "
              f"{st['win_rate']:>7.1%}")
    print()

    # -----------------------------------------------------------------------
    # Monthly 2024
    # -----------------------------------------------------------------------
    r_2024 = results["2024 holdout (OOS)"][0]
    print("MONTHLY BREAKDOWN — 2024 holdout")
    print("  " + "-" * 50)
    monthly = r_2024.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x) else np.nan)
    eq = STARTING_CAPITAL
    print(f"  {'Month':<10} {'Return':>9}   {'Equity':>9}")
    for m, ret in monthly.items():
        if np.isnan(ret):
            continue
        eq *= (1 + ret)
        flag = "  <-- BAD" if ret < -0.05 else ("  ***" if ret > 0.15 else "")
        print(f"  {m.strftime('%Y-%m'):<10} {ret:>+8.1%}   ${eq:>8.0f}{flag}")
    print(f"\n  2024 full year: $1k → ${results['2024 holdout (OOS)'][1]['final']:.0f}  "
          f"({results['2024 holdout (OOS)'][1]['ann_ret']:+.0%} ann)")

    # Monthly 2025
    r_2025 = results["2025–2026 (in-sample ref)"][0]
    print("\nMONTHLY BREAKDOWN — 2025–2026 (reference)")
    print("  " + "-" * 50)
    monthly25 = r_2025.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x) else np.nan)
    eq = STARTING_CAPITAL
    for m, ret in monthly25.items():
        if np.isnan(ret):
            continue
        eq *= (1 + ret)
        flag = "  <-- BAD" if ret < -0.05 else ("  ***" if ret > 0.15 else "")
        print(f"  {m.strftime('%Y-%m'):<10} {ret:>+8.1%}   ${eq:>8.0f}{flag}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    st24 = results["2024 holdout (OOS)"][1]
    st25 = results["2025–2026 (in-sample ref)"][1]
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    pass_sharpe  = st24["sharpe"] > 0.5
    pass_pos_ret = st24["final"]  > STARTING_CAPITAL
    pass_consist = abs(st24["sharpe"] - st25["sharpe"]) < 2.5
    print(f"  2024 Sharpe > 0.5          : {'PASS' if pass_sharpe  else 'FAIL'}  ({st24['sharpe']:.2f})")
    print(f"  2024 positive return       : {'PASS' if pass_pos_ret else 'FAIL'}  (${st24['final']:.0f})")
    print(f"  2024 vs 2025 consistent    : {'PASS' if pass_consist else 'FAIL'}  "
          f"(diff={abs(st24['sharpe']-st25['sharpe']):.2f})")
    if pass_sharpe and pass_pos_ret:
        print("\n  >>> EDGE CONFIRMED ON 2024 HOLDOUT. Structural alpha spans 2+ years.")
    elif pass_pos_ret:
        print("\n  >>> Positive return but low Sharpe in 2024. Edge present, weaker regime.")
    else:
        print("\n  >>> FAIL. 2024 holdout negative. Edge may be 2025-specific.")

    # -----------------------------------------------------------------------
    # Equity plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    colors = {"2024 holdout (OOS)": "steelblue",
              "2025–2026 (in-sample ref)": "crimson",
              "2024+2025 combined": "green"}
    for label, (r, st) in results.items():
        eq = STARTING_CAPITAL * (1 + r).cumprod()
        lw = 2.5 if "combined" in label else 1.8
        ax.plot(eq.index, eq.values, label=f"{label}  Sh={st['sharpe']:.2f}  DD={st['max_dd']:.1%}  ${st['final']:.0f}",
                color=colors[label], lw=lw)
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, linestyle="--")
    ax.set_title("Phase 23 — 2024 OOS Holdout vs 2025 In-Sample\n"
                 "Strategy: 2×funding + mom24h + funding_trend, N=10, no-Majors, inverse scaling")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Monthly bar chart — 2024 and 2025 side by side
    ax2 = axes[1]
    m24 = r_2024.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x) else np.nan)
    m25 = r_2025.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x) else np.nan)
    x24 = np.arange(len(m24))
    x25 = np.arange(len(m25)) + len(m24) + 1
    c24 = ["green" if v >= 0 else "red" for v in m24.values]
    c25 = ["darkgreen" if v >= 0 else "darkred" for v in m25.values]
    ax2.bar(x24, m24.values * 100, color=c24, alpha=0.8, label="2024 OOS")
    ax2.bar(x25, m25.values * 100, color=c25, alpha=0.8, label="2025+ in-sample")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axvline(len(m24) - 0.5 + 0.5, color="gray", lw=1.5, linestyle="--", label="OOS boundary")
    xticks = list(x24) + list(x25)
    xlabels = [m.strftime("%b") for m in m24.index] + [m.strftime("%b %y") for m in m25.index]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax2.set_ylabel("Monthly Return (%)")
    ax2.set_title("Monthly Returns — 2024 OOS vs 2025+ In-Sample")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "phase23_2024_holdout.png")
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"\nPlot saved: {out_png}")
    print("Phase 23 complete.")


if __name__ == "__main__":
    main()
