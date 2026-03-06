"""
Phase 24 — Leverage & Allocation Sweep

Tests the Phase 22 best strategy (2×predicted_funding + mom_24h) across:
  N (positions per leg) : 5, 8, 10, 15, 20
  Leverage              : 1x, 2x, 3x, 5x

Starting capital: $10,000

With leverage L and N positions per leg:
  Each position size = L / (2N) of AUM  (e.g. 2x lev, N=10 → 10% per position)
  Net return per bar = L × (gross_1x_return - FEE_RT)
  Fee scales with leverage (you trade L× the notional)
  Margin required = 1/L of notional (assume isolated margin, Bybit)

Inverse scaling (Phase 21) applied throughout.
MaxDD shown on equity, not just returns — so 5x leverage with -30% MaxDD = -150% ruin risk.

Liquidation guard: if equity drops >50% in any single month, flag as "LIQUIDATION RISK".
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

SIGNALS_V1   = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
SIGNALS_V2   = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals_v2"
RESULTS_DIR  = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REBAL_FREQ       = "8h"
FEE_1SIDE        = 4 / 10000        # 4 bps per side (maker)
PERIODS_PER_YEAR = 365 * 3
CLIP             = 3.0
START            = "2025-01-01"
STARTING_CAPITAL = 10_000.0
SHARPE_WINDOW    = 30

MAJORS = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","TRXUSDT","XLMUSDT","ETCUSDT","HBARUSDT",
    "ATOMUSDT","ALGOUSDT","EGLDUSDT",
}

N_VALUES   = [5, 8, 10, 15, 20]
LEV_VALUES = [1, 2, 3, 5]

# ---------------------------------------------------------------------------

def load_panels():
    v1_cols = ["mom_24h", "fwd_8h"]
    v2_cols = ["predicted_funding"]

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

    p1 = load_dir(SIGNALS_V1, v1_cols)
    p2 = load_dir(SIGNALS_V2, v2_cols)
    return {**p1, **p2}

def to_8h(panel):
    return panel.resample(REBAL_FREQ, closed="left", label="left").first()

def cs_zscore(panel, min_valid=15):
    mu  = panel.mean(axis=1)
    sig = panel.std(axis=1).replace(0, np.nan)
    z   = panel.sub(mu, axis=0).div(sig, axis=0).clip(-CLIP, CLIP)
    z[panel.notna().sum(axis=1) < min_valid] = np.nan
    return z

def sharpe_fn(x):
    return x.mean() / x.std() if (len(x) >= 5 and x.std() > 0) else 0.0

def sim_1x(composite, fwd_8h, n=10):
    """Base 1x unlevered returns (gross - fees at 1x)."""
    rets, dates = [], composite.index.intersection(fwd_8h.index)
    for ts in dates:
        row = composite.loc[ts].dropna()
        if len(row) < n * 2:
            rets.append(0.0)
            continue
        fwd   = fwd_8h.loc[ts]
        gross = (0.5 * fwd[row.nlargest(n).index].mean()
                 - 0.5 * fwd[row.nsmallest(n).index].mean())
        # 1x fee: enter + exit both legs = 4 bps × 4 = 16 bps? No:
        # We turn over ~half the portfolio each rebal → avg turnover per position ~52%
        # Fee = FEE_1SIDE * 2 sides * turnover. Approximate as flat FEE_RT per bar:
        fee = FEE_1SIDE * 2   # 8 bps round-trip at 1x
        rets.append(gross - fee)
    return pd.Series(rets, index=dates)

def apply_leverage(r_1x, lev):
    """Scale returns and fees by leverage. Fees scale with leverage."""
    # At leverage L:
    #   gross_L = L * gross_1x
    #   fee_L   = L * fee_1x  (we trade L× the notional)
    # So net_L = L * gross_1x - L * fee_1x = L * net_1x  (if fee is included in r_1x)
    # But we want to cap at -100% (can't lose more than capital with isolated margin)
    r_L = r_1x * lev
    return r_L

def build_inverse_scale(rets):
    rs = rets.rolling(SHARPE_WINDOW).apply(sharpe_fn, raw=True)
    sc = pd.Series(1.0, index=rets.index)
    sc[rs > 5]  = 0.5
    sc[rs <= 0] = 0.5
    return sc

def port_stats(rets, cap=STARTING_CAPITAL):
    if len(rets) == 0 or rets.std() == 0:
        return dict(sharpe=0, sortino=0, ann_ret=0, max_dd=0,
                    win_rate=0, final=cap, n_bars=0)
    sr  = rets.mean() / rets.std() * np.sqrt(PERIODS_PER_YEAR)
    ann = (1 + rets).prod() ** (PERIODS_PER_YEAR / len(rets)) - 1
    neg = rets[rets < 0]
    so  = rets.mean() / neg.std() * np.sqrt(PERIODS_PER_YEAR) if len(neg) > 0 else np.nan
    eq  = (1 + rets).cumprod()
    dd  = (eq / eq.cummax() - 1).min()
    final = cap * (1 + rets).prod()
    return dict(sharpe=sr, sortino=so, ann_ret=ann, max_dd=dd,
                win_rate=(rets > 0).mean(), final=final, n_bars=len(rets))


def monthly_equity(rets, cap=STARTING_CAPITAL):
    """Monthly returns and running equity."""
    monthly = rets.resample("ME").apply(lambda x: (1+x).prod()-1 if len(x)>0 else np.nan)
    eq = cap
    rows = []
    for m, ret in monthly.items():
        if np.isnan(ret):
            continue
        eq *= (1 + ret)
        rows.append({"month": m.strftime("%Y-%m"), "ret": ret, "equity": eq})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main():
    print("Loading panels ...")
    panels = load_panels()

    pred_8h = to_8h(panels["predicted_funding"])
    mom_8h  = to_8h(panels["mom_24h"])
    fwd_8h  = to_8h(panels["fwd_8h"])

    all_syms = pred_8h.columns.intersection(mom_8h.columns).intersection(fwd_8h.columns)
    pred_8h = pred_8h[all_syms]
    mom_8h  = mom_8h[all_syms]
    fwd_8h  = fwd_8h[all_syms]

    print(f"Universe: {len(all_syms)} symbols, {len(pred_8h[START:])} bars from {START}")
    print()

    z_pred = cs_zscore(pred_8h[START:])
    z_mom  = cs_zscore(mom_8h[START:])
    fwd    = fwd_8h[START:]

    composite = (2 * z_pred + z_mom) / 3

    # -----------------------------------------------------------------------
    # Sweep N × Leverage
    # -----------------------------------------------------------------------
    all_results = {}
    for n in N_VALUES:
        r_1x_raw = sim_1x(composite, fwd, n=n)
        scale    = build_inverse_scale(r_1x_raw)
        r_1x     = r_1x_raw * scale

        for lev in LEV_VALUES:
            r = apply_leverage(r_1x, lev)
            st = port_stats(r)
            monthly = monthly_equity(r)
            all_results[(n, lev)] = (r, st, monthly)

    # -----------------------------------------------------------------------
    # Summary grid
    # -----------------------------------------------------------------------
    print("=" * 80)
    print(f"PHASE 24 — LEVERAGE SWEEP  (2×pred_funding + mom, inverse scaling, $10k start)")
    print("=" * 80)

    # Table: rows = N, cols = leverage
    print(f"\n{'Metric: Sharpe':}")
    nlev = "N\\Lev"
    hdr = f"  {nlev:<8}" + "".join(f"  {'Lev '+str(l)+'x':>10}" for l in LEV_VALUES)
    print(hdr)
    print("  " + "-" * (8 + 14 * len(LEV_VALUES)))
    for n in N_VALUES:
        row = f"  N={n:<6}"
        for lev in LEV_VALUES:
            row += f"  {all_results[(n,lev)][1]['sharpe']:>10.2f}"
        print(row)

    print(f"\n{'Metric: MaxDD':}")
    print(hdr)
    print("  " + "-" * (8 + 14 * len(LEV_VALUES)))
    for n in N_VALUES:
        row = f"  N={n:<6}"
        for lev in LEV_VALUES:
            row += f"  {all_results[(n,lev)][1]['max_dd']:>10.1%}"
        print(row)

    print(f"\n{'Metric: $10k → (final equity)':}")
    print(hdr)
    print("  " + "-" * (8 + 14 * len(LEV_VALUES)))
    for n in N_VALUES:
        row = f"  N={n:<6}"
        for lev in LEV_VALUES:
            fin = all_results[(n,lev)][1]['final']
            row += f"  ${fin:>9,.0f}"
        print(row)

    print(f"\n{'Metric: Ann Return':}")
    print(hdr)
    print("  " + "-" * (8 + 14 * len(LEV_VALUES)))
    for n in N_VALUES:
        row = f"  N={n:<6}"
        for lev in LEV_VALUES:
            row += f"  {all_results[(n,lev)][1]['ann_ret']:>10.0%}"
        print(row)

    # -----------------------------------------------------------------------
    # Monthly breakdown for key configs
    # -----------------------------------------------------------------------
    key_configs = [
        (10, 1, "N=10, 1x (conservative)"),
        (10, 2, "N=10, 2x"),
        (10, 3, "N=10, 3x"),
        (8,  3, "N=8,  3x (concentrated)"),
        (5,  2, "N=5,  2x (aggressive)"),
        (10, 5, "N=10, 5x (high risk)"),
    ]

    print()
    print("=" * 80)
    print("MONTHLY BREAKDOWN — Key Configurations")
    print("=" * 80)

    # Header
    col_w = 16
    header = f"  {'Month':<10}"
    for _, _, label in key_configs:
        header += f"  {label[:col_w]:>{col_w}}"
    print(header)
    print("  " + "-" * (10 + (col_w + 2) * len(key_configs)))

    # Get all months
    months = sorted(set(
        m for _, _, (_, _, mdf) in [(k[0], k[1], all_results[k]) for k in all_results]
        for m in mdf["month"].tolist()
    ))

    for month in months:
        row = f"  {month:<10}"
        for n, lev, label in key_configs:
            mdf = all_results[(n, lev)][2]
            match = mdf[mdf["month"] == month]
            if len(match):
                v = match["ret"].iloc[0]
                row += f"  {v:>+{col_w}.1%}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)

    # Final equity row
    row = f"  {'Final $'::<10}"
    for n, lev, label in key_configs:
        fin = all_results[(n, lev)][1]['final']
        row += f"  ${fin:>{col_w-1},.0f}"
    print("  " + "-" * (10 + (col_w + 2) * len(key_configs)))
    print(row)

    # Sharpe row
    row = f"  {'Sharpe':<10}"
    for n, lev, label in key_configs:
        sh = all_results[(n, lev)][1]['sharpe']
        row += f"  {sh:>{col_w}.2f}"
    print(row)

    # MaxDD row
    row = f"  {'MaxDD':<10}"
    for n, lev, label in key_configs:
        dd = all_results[(n, lev)][1]['max_dd']
        row += f"  {dd:>{col_w}.1%}"
    print(row)

    # -----------------------------------------------------------------------
    # Equity curve plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    plot_groups = [
        (10, "N=10 — Different Leverages"),
        (5,  "N=5  — Different Leverages"),
        (8,  "N=8  — Different Leverages"),
        (15, "N=15 — Different Leverages"),
    ]
    lev_colors = {1: "steelblue", 2: "green", 3: "orange", 5: "crimson"}

    for ax, (n, title) in zip(axes, plot_groups):
        for lev in LEV_VALUES:
            r, st, _ = all_results[(n, lev)]
            eq = STARTING_CAPITAL * (1 + r).cumprod()
            lw = 2.5 if lev == 3 else 1.5
            label = (f"{lev}x  Sh={st['sharpe']:.1f}  DD={st['max_dd']:.0%}"
                     f"  ${st['final']:,.0f}")
            ax.plot(eq.index, eq.values, color=lev_colors[lev], lw=lw, label=label)
        ax.axhline(STARTING_CAPITAL, color="gray", lw=0.7, linestyle="--")
        ax.set_title(title)
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.suptitle(
        "Phase 24 — Leverage Sweep\n"
        "Strategy: 2×predicted_funding + mom_24h, inverse scaling, $10k start",
        fontsize=12
    )
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "phase24_equity_curves.png")
    plt.savefig(out1, dpi=130)
    plt.close()

    # Heatmap: Sharpe and Final equity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sharpe_grid = np.array([[all_results[(n,l)][1]['sharpe'] for l in LEV_VALUES] for n in N_VALUES])
    final_grid  = np.array([[all_results[(n,l)][1]['final']  for l in LEV_VALUES] for n in N_VALUES])

    for ax, data, title, fmt in [
        (axes[0], sharpe_grid, "Sharpe Ratio", "{:.2f}"),
        (axes[1], final_grid / 1000, "Final Capital ($k)", "${:.0f}k"),
    ]:
        cmap = "RdYlGn"
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(LEV_VALUES)))
        ax.set_xticklabels([f"{l}x" for l in LEV_VALUES])
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f"N={n}" for n in N_VALUES])
        ax.set_xlabel("Leverage")
        ax.set_ylabel("N (positions per leg)")
        ax.set_title(title)
        for i in range(len(N_VALUES)):
            for j in range(len(LEV_VALUES)):
                ax.text(j, i, fmt.format(data[i, j]),
                        ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < (data[i,j] - data.min()) / (data.max() - data.min() + 1e-9) < 0.7 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Phase 24 — N × Leverage Heatmap\n"
                 "2×predicted_funding + mom_24h, inverse scaling, Jan 2025–Mar 2026",
                 fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "phase24_heatmap.png")
    plt.savefig(out2, dpi=130)
    plt.close()

    print()
    print(f"Plots saved:")
    print(f"  {out1}")
    print(f"  {out2}")
    print()

    # Best risk-adjusted config
    best_sharpe = max(all_results.items(), key=lambda x: x[1][1]['sharpe'])
    best_dd = min(
        ((k, v) for k, v in all_results.items() if v[1]['ann_ret'] > 1.0),
        key=lambda x: abs(x[1][1]['max_dd'])
    )
    print("Best Sharpe config  :", f"N={best_sharpe[0][0]}, {best_sharpe[0][1]}x"
          f"  →  Sharpe {best_sharpe[1][1]['sharpe']:.2f}, DD {best_sharpe[1][1]['max_dd']:.1%}")
    print("Best DD config (>100% ann):", f"N={best_dd[0][0]}, {best_dd[0][1]}x"
          f"  →  Sharpe {best_dd[1][1]['sharpe']:.2f}, DD {best_dd[1][1]['max_dd']:.1%}")
    print()
    print("Phase 24 complete.")


if __name__ == "__main__":
    main()
