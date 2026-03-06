from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
TRADES_CSV = OUT_DIR / "best_config_trades.csv"

SUMMARY_CSV = OUT_DIR / "allocation_leverage_summary.csv"
MONTHLY_CSV = OUT_DIR / "allocation_leverage_monthly.csv"
EQUITY_CSV = OUT_DIR / "allocation_leverage_equity.csv"
REPORT_MD = OUT_DIR / "FINDINGS_allocation_leverage_monthly.md"

INITIAL_EQUITY = 1000.0
ALLOCATIONS = [0.10, 0.25, 0.50, 0.75, 1.00]  # fraction of equity allocated per signal timestamp
LEVERAGES = [1, 2, 3, 5, 8, 10]
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def run_path(trades: pd.DataFrame, alloc: float, lev: float) -> pd.DataFrame:
    x = trades[["ts", "period", "ret_net_mixed"]].copy()
    # Isolated sleeve: allocated slice can go to zero, non-allocated stays in cash.
    sleeve_mult = (1.0 + lev * x["ret_net_mixed"]).clip(lower=0.0)
    x["step_mult"] = (1.0 - alloc) + alloc * sleeve_mult
    x["ret_port"] = x["step_mult"] - 1.0
    x["equity"] = INITIAL_EQUITY * x["step_mult"].cumprod()
    x["allocation"] = alloc
    x["leverage"] = lev
    x["sleeve_wipe"] = sleeve_mult <= 0.0
    return x


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if not equity.empty else np.nan


def main() -> None:
    trades = pd.read_csv(TRADES_CSV, parse_dates=["ts"])
    if "ret_net_mixed" not in trades.columns:
        raise RuntimeError("best_config_trades.csv must contain ret_net_mixed")

    paths = []
    summary_rows = []
    monthly_rows = []

    for alloc in ALLOCATIONS:
        for lev in LEVERAGES:
            p = run_path(trades, alloc=alloc, lev=lev)
            paths.append(p[["ts", "period", "allocation", "leverage", "ret_port", "equity", "sleeve_wipe"]])

            test = p.loc[p["ts"] >= TEST_START]
            summary_rows.append(
                {
                    "allocation": alloc,
                    "leverage": lev,
                    "final_equity": float(p["equity"].iloc[-1]),
                    "total_return_pct": float(p["equity"].iloc[-1] / INITIAL_EQUITY - 1.0) * 100.0,
                    "max_drawdown_pct": max_drawdown(p["equity"]) * 100.0,
                    "test_avg_ret_bps": float(test["ret_port"].mean() * 10000.0) if not test.empty else np.nan,
                    "test_comp_ret_pct": float((test["step_mult"].prod() - 1.0) * 100.0) if not test.empty else np.nan,
                    "sleeve_wipe_events": int(p["sleeve_wipe"].sum()),
                }
            )

            m = p.copy()
            m["month"] = m["ts"].dt.strftime("%Y-%m")
            g = m.groupby("month", as_index=False).agg(
                trades=("ts", "count"),
                avg_ret_bps=("ret_port", lambda s: float(s.mean() * 10000.0)),
                compounded_ret_pct=("step_mult", lambda s: float((s.prod() - 1.0) * 100.0)),
                end_equity=("equity", "last"),
                sleeve_wipe_events=("sleeve_wipe", "sum"),
            )
            g["allocation"] = alloc
            g["leverage"] = lev
            monthly_rows.append(g)

    summary = pd.DataFrame(summary_rows).sort_values(["allocation", "leverage"]).reset_index(drop=True)
    monthly = pd.concat(monthly_rows, ignore_index=True).sort_values(["allocation", "leverage", "month"])
    equity = pd.concat(paths, ignore_index=True).sort_values(["allocation", "leverage", "ts"])

    summary.to_csv(SUMMARY_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    equity.to_csv(EQUITY_CSV, index=False)

    best = summary.sort_values("final_equity", ascending=False).head(8)
    lines = [
        "# Monthly Breakdown: Allocation % and Leverage Sweep",
        "",
        f"- Grid allocations: `{ALLOCATIONS}`",
        f"- Grid leverage: `{LEVERAGES}`",
        "- Return source: `best_config_trades.csv` (`ret_net_mixed`, i.e. after mixed fees).",
        "- Portfolio model: only `allocation` part of capital is deployed each signal; undeployed capital stays in cash.",
        "- If leveraged sleeve return <= -100%, allocated sleeve is fully lost for that step (sleeve wipe).",
        "",
        "## Top Configurations by Final Equity",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"- alloc={r['allocation']:.2f}, lev={int(r['leverage'])}x: "
            f"final=${r['final_equity']:.2f}, return={r['total_return_pct']:.1f}%, "
            f"maxDD={r['max_drawdown_pct']:.1f}%, test_comp={r['test_comp_ret_pct']:.1f}%, "
            f"sleeve_wipes={int(r['sleeve_wipe_events'])}"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- `{SUMMARY_CSV.name}`",
            f"- `{MONTHLY_CSV.name}`",
            f"- `{EQUITY_CSV.name}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {MONTHLY_CSV}")
    print(f"Wrote {EQUITY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
