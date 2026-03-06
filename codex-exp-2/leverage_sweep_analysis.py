from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
TRADES_CSV = OUT_DIR / "best_config_trades.csv"

SUMMARY_CSV = OUT_DIR / "leverage_sweep_summary.csv"
MONTHLY_CSV = OUT_DIR / "leverage_sweep_monthly.csv"
EQUITY_CSV = OUT_DIR / "leverage_sweep_equity.csv"
REPORT_MD = OUT_DIR / "FINDINGS_leverage_sweep.md"

INITIAL_EQUITY = 1000.0
LEVERAGES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

# Scenario A: close-to-close only (optimistic)
# Scenario B: stress intraperiod tails by amplifying negative returns
STRESS_NEG_MULT = 1.8

TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _apply_leverage(ret_1x: pd.Series, lev: float, neg_mult: float) -> pd.Series:
    stressed = ret_1x.copy()
    stressed = np.where(stressed < 0, stressed * neg_mult, stressed)
    lev_ret = lev * stressed
    lev_ret = pd.Series(lev_ret, index=ret_1x.index, dtype="float64")
    return lev_ret


def _equity_with_liq(ret_lev: pd.Series) -> tuple[pd.Series, bool, int | None]:
    eq = []
    cur = INITIAL_EQUITY
    liq = False
    liq_idx = None
    for i, r in enumerate(ret_lev):
        if r <= -1.0:
            cur = 0.0
            liq = True
            liq_idx = i
            eq.append(cur)
            break
        cur = cur * (1.0 + float(r))
        eq.append(cur)
    if liq:
        if liq_idx is not None and liq_idx + 1 < len(ret_lev):
            eq.extend([0.0] * (len(ret_lev) - (liq_idx + 1)))
    else:
        if len(eq) < len(ret_lev):
            eq.extend([cur] * (len(ret_lev) - len(eq)))
    return pd.Series(eq, index=ret_lev.index, dtype="float64"), liq, liq_idx


def run_scenario(trades: pd.DataFrame, scenario: str, neg_mult: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    monthly_rows = []
    equity_rows = []

    for lev in LEVERAGES:
        lev_ret = _apply_leverage(trades["ret_net_mixed"], lev=lev, neg_mult=neg_mult)
        eq, liq, liq_idx = _equity_with_liq(lev_ret)

        tmp = trades[["ts", "period"]].copy()
        tmp["scenario"] = scenario
        tmp["leverage"] = lev
        tmp["ret_lev"] = lev_ret.values
        tmp["equity"] = eq.values
        tmp["month"] = tmp["ts"].dt.strftime("%Y-%m")
        equity_rows.append(tmp[["ts", "scenario", "leverage", "ret_lev", "equity", "period"]])

        test = tmp.loc[tmp["ts"] >= TEST_START]
        summary_rows.append(
            {
                "scenario": scenario,
                "leverage": lev,
                "final_equity": float(tmp["equity"].iloc[-1]),
                "total_return_pct": float(tmp["equity"].iloc[-1] / INITIAL_EQUITY - 1.0) * 100.0,
                "max_drawdown_pct": float((tmp["equity"] / tmp["equity"].cummax() - 1.0).min() * 100.0),
                "liq_hit": bool(liq),
                "liq_ts": str(tmp.iloc[liq_idx]["ts"]) if liq_idx is not None else "",
                "test_avg_ret_pct_per_trade": float(test["ret_lev"].mean() * 100.0) if not test.empty else np.nan,
                "test_trades": int(len(test)),
            }
        )

        m = tmp.groupby("month", as_index=False).agg(
            trades=("ts", "count"),
            avg_ret_pct=("ret_lev", lambda s: float(s.mean() * 100.0)),
            compounded_ret_pct=("ret_lev", lambda s: float(((1.0 + s).prod() - 1.0) * 100.0)),
            end_equity=("equity", "last"),
        )
        m["scenario"] = scenario
        m["leverage"] = lev
        monthly_rows.append(m)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(monthly_rows, ignore_index=True),
        pd.concat(equity_rows, ignore_index=True),
    )


def write_report(summary: pd.DataFrame) -> None:
    s = summary.sort_values(["scenario", "leverage"]).reset_index(drop=True)
    best = s.loc[~s["liq_hit"]].sort_values("final_equity", ascending=False).head(5)

    lines = [
        "# Leverage Sweep (Aggressive Variants)",
        "",
        "## Setup",
        "- Base strategy: current `best_config_trades.csv` (top-5 equal, mixed-fee net returns at 1x).",
        f"- Leverage grid: `{LEVERAGES}`",
        "- Liquidation rule: if per-trade leveraged return <= -100%, equity is set to zero from that point onward.",
        f"- Stress scenario amplifies negative trade returns by `{STRESS_NEG_MULT}x` before leverage.",
        "",
        "## Best Non-Liquidating Variants",
    ]
    if best.empty:
        lines.append("- None")
    else:
        for _, r in best.iterrows():
            lines.append(
                f"- `{r['scenario']}` {int(r['leverage'])}x: final `${r['final_equity']:.2f}`, "
                f"return `{r['total_return_pct']:.1f}%`, maxDD `{r['max_drawdown_pct']:.1f}%`"
            )

    lines.extend(
        [
            "",
            "## Files",
            "- `leverage_sweep_summary.csv`",
            "- `leverage_sweep_monthly.csv`",
            "- `leverage_sweep_equity.csv`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    trades = pd.read_csv(TRADES_CSV, parse_dates=["ts"])
    if "ret_net_mixed" not in trades.columns:
        raise RuntimeError("Expected `ret_net_mixed` in best_config_trades.csv")

    s1, m1, e1 = run_scenario(trades, scenario="close_to_close", neg_mult=1.0)
    s2, m2, e2 = run_scenario(trades, scenario="stress_intraperiod", neg_mult=STRESS_NEG_MULT)

    summary = pd.concat([s1, s2], ignore_index=True).sort_values(["scenario", "leverage"])
    monthly = pd.concat([m1, m2], ignore_index=True).sort_values(["scenario", "leverage", "month"])
    equity = pd.concat([e1, e2], ignore_index=True).sort_values(["scenario", "leverage", "ts"])

    summary.to_csv(SUMMARY_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    equity.to_csv(EQUITY_CSV, index=False)
    write_report(summary)

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {MONTHLY_CSV}")
    print(f"Wrote {EQUITY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
