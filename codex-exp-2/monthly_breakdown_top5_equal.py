from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"

TRADES_CSV = OUT_DIR / "top5_equal_trades.csv"
MONTHLY_CSV = OUT_DIR / "top5_equal_monthly_breakdown.csv"
EQUITY_CSV = OUT_DIR / "top5_equal_equity_curve.csv"
REPORT_MD = OUT_DIR / "FINDINGS_top5_equal_monthly_equity.md"

INITIAL_EQUITY = 1000.0
FEE_MIXED = 0.0014
FEE_ALL_TAKER = 0.0020

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def main() -> None:
    s = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    f = s.loc[
        (s["oi_med_3d"] >= 20_000_000.0)
        & (s["breadth_mom"] >= 0.60)
        & (s["median_ls_z"] >= 0.0)
        & (s["ls_z"] >= 2.0)
        & (s["taker_z"] >= 0.5)
        & (s["mom_4h"] > 0)
    ].copy()

    picked = (
        f.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(5)
        .copy()
    )
    trades = (
        picked.groupby("ts", as_index=False)
        .agg(raw_ret=("ret_4h", "mean"), n_positions=("symbol", "count"))
        .sort_values("ts")
        .reset_index(drop=True)
    )
    trades["period"] = "train"
    trades.loc[trades["ts"] >= TEST_START, "period"] = "test"
    trades["ret_net_mixed"] = trades["raw_ret"] - FEE_MIXED
    trades["ret_net_all_taker"] = trades["raw_ret"] - FEE_ALL_TAKER
    trades["equity_mixed"] = INITIAL_EQUITY * (1.0 + trades["ret_net_mixed"]).cumprod()
    trades["equity_all_taker"] = INITIAL_EQUITY * (1.0 + trades["ret_net_all_taker"]).cumprod()

    m = trades.copy()
    m["month"] = m["ts"].dt.strftime("%Y-%m")
    monthly = m.groupby("month", as_index=False).agg(
        trades=("ts", "count"),
        avg_raw_bps=("raw_ret", lambda x: float(x.mean() * 10000.0)),
        avg_net_bps_mixed=("ret_net_mixed", lambda x: float(x.mean() * 10000.0)),
        avg_net_bps_all_taker=("ret_net_all_taker", lambda x: float(x.mean() * 10000.0)),
        compounded_return_mixed=("ret_net_mixed", lambda x: float((1.0 + x).prod() - 1.0)),
        compounded_return_all_taker=("ret_net_all_taker", lambda x: float((1.0 + x).prod() - 1.0)),
    )

    trades.to_csv(TRADES_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    trades[["ts", "equity_mixed", "equity_all_taker", "period"]].to_csv(EQUITY_CSV, index=False)

    test = trades.loc[trades["ts"] >= TEST_START]
    lines = [
        "# Top-5 Equal Monthly Breakdown and Equity",
        "",
        "- Strategy: same thresholds as baseline, but `top_n=5` equal-weight",
        "- Fees: mixed 14 bps and all-taker 20 bps round-trip",
        "- Data: full available `samples_4h.csv` window",
        "",
        f"- Final equity mixed: `${trades['equity_mixed'].iloc[-1]:,.2f}`",
        f"- Final equity all-taker: `${trades['equity_all_taker'].iloc[-1]:,.2f}`",
        f"- Total timestamps: `{len(trades)}`",
        f"- Test timestamps: `{len(test)}`",
        f"- Test avg net bps (mixed): `{test['ret_net_mixed'].mean()*10000:.2f}`",
    ]
    REPORT_MD.write_text("\n".join(lines))

    print(f"Wrote {TRADES_CSV}")
    print(f"Wrote {MONTHLY_CSV}")
    print(f"Wrote {EQUITY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
