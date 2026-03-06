from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = OUT_DIR / "samples_4h.csv"

TRADES_CSV = OUT_DIR / "best_config_trades.csv"
MONTHLY_CSV = OUT_DIR / "best_config_monthly_breakdown.csv"
EQUITY_CSV = OUT_DIR / "best_config_equity_curve.csv"
REPORT_MD = OUT_DIR / "FINDINGS_monthly_breakdown_equity.md"

INITIAL_EQUITY = 1000.0
FEE_MIXED = 0.0014  # 14 bps round-trip
FEE_ALL_TAKER = 0.0020  # 20 bps round-trip

# Default production baseline (updated): top-5 equal-weight.
CFG = {
    "ls_threshold": 2.0,
    "taker_threshold": 0.5,
    "min_oi_value": 20_000_000.0,
    "top_n": 5,
    "breadth_threshold": 0.60,
    "median_ls_threshold": 0.0,
}

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def select_trades(samples: pd.DataFrame) -> pd.DataFrame:
    eligible = samples.loc[
        (samples["oi_med_3d"] >= CFG["min_oi_value"])
        & (samples["breadth_mom"] >= CFG["breadth_threshold"])
        & (samples["median_ls_z"] >= CFG["median_ls_threshold"])
        & (samples["ls_z"] >= CFG["ls_threshold"])
        & (samples["taker_z"] >= CFG["taker_threshold"])
        & (samples["mom_4h"] > 0)
    ].copy()

    picked = (
        eligible.sort_values(["ts", "score_abs"], ascending=[True, False])
        .groupby("ts", group_keys=False)
        .head(int(CFG["top_n"]))
        .copy()
    )
    if picked.empty:
        return pd.DataFrame(columns=["ts", "raw_ret", "n_positions"])

    trades = (
        picked.groupby("ts", as_index=False)
        .agg(raw_ret=("ret_4h", "mean"), n_positions=("symbol", "count"))
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return trades


def add_fee_and_equity(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["period"] = "train"
    out.loc[out["ts"] >= TEST_START, "period"] = "test"

    out["ret_net_mixed"] = out["raw_ret"] - FEE_MIXED
    out["ret_net_all_taker"] = out["raw_ret"] - FEE_ALL_TAKER

    out["equity_mixed"] = INITIAL_EQUITY * (1.0 + out["ret_net_mixed"]).cumprod()
    out["equity_all_taker"] = INITIAL_EQUITY * (1.0 + out["ret_net_all_taker"]).cumprod()
    return out


def monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    m = trades.copy()
    m["month"] = m["ts"].dt.strftime("%Y-%m")
    grouped = m.groupby("month", as_index=False).agg(
        trades=("ts", "count"),
        avg_raw_bps=("raw_ret", lambda s: float(s.mean() * 10000.0)),
        avg_net_bps_mixed=("ret_net_mixed", lambda s: float(s.mean() * 10000.0)),
        avg_net_bps_all_taker=("ret_net_all_taker", lambda s: float(s.mean() * 10000.0)),
        compounded_return_mixed=("ret_net_mixed", lambda s: float((1.0 + s).prod() - 1.0)),
        compounded_return_all_taker=("ret_net_all_taker", lambda s: float((1.0 + s).prod() - 1.0)),
    )
    return grouped


def write_report(trades: pd.DataFrame, monthly: pd.DataFrame) -> None:
    train = trades.loc[trades["ts"] <= TRAIN_END]
    test = trades.loc[trades["ts"] >= TEST_START]

    final_mixed = float(trades["equity_mixed"].iloc[-1]) if not trades.empty else INITIAL_EQUITY
    final_taker = float(trades["equity_all_taker"].iloc[-1]) if not trades.empty else INITIAL_EQUITY

    lines = [
        "# Monthly Breakdown and Equity Curve (Best Config)",
        "",
        "## Setup",
        f"- Initial equity: `${INITIAL_EQUITY:,.2f}`",
        f"- Mixed fee assumption (production): `{FEE_MIXED*10000:.1f} bps` round-trip",
        f"- All-taker reference: `{FEE_ALL_TAKER*10000:.1f} bps` round-trip",
        f"- Config: `ls_z>={CFG['ls_threshold']}`, `taker_z>={CFG['taker_threshold']}`, "
        f"`oi_med_3d>={int(CFG['min_oi_value'])}`, `top_n={CFG['top_n']}`, "
        f"`breadth>={CFG['breadth_threshold']}`, `median_ls>={CFG['median_ls_threshold']}`",
        "",
        "## Trade Counts",
        f"- Total decision timestamps: `{len(trades)}`",
        f"- Train (<= 2025-12-31): `{len(train)}`",
        f"- Test (>= 2026-01-01): `{len(test)}`",
        "",
        "## Equity Results",
        f"- Final equity (mixed 14 bps): `${final_mixed:,.2f}`",
        f"- Final equity (all-taker 20 bps): `${final_taker:,.2f}`",
        "",
        "## Test-Only Snapshot",
    ]
    if not test.empty:
        test_ret_mixed = float((1.0 + test["ret_net_mixed"]).prod() - 1.0)
        test_ret_taker = float((1.0 + test["ret_net_all_taker"]).prod() - 1.0)
        lines.extend(
            [
                f"- Test compounded return (mixed): `{test_ret_mixed*100:.2f}%`",
                f"- Test compounded return (all-taker): `{test_ret_taker*100:.2f}%`",
                f"- Test avg net bps/trade (mixed): `{test['ret_net_mixed'].mean()*10000:.2f}`",
                f"- Test avg net bps/trade (all-taker): `{test['ret_net_all_taker'].mean()*10000:.2f}`",
            ]
        )
    else:
        lines.append("- No test trades found.")

    lines.extend(
        [
            "",
            "## Files",
            "- `best_config_trades.csv`",
            "- `best_config_monthly_breakdown.csv`",
            "- `best_config_equity_curve.csv`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["ts"])
    trades = select_trades(samples)
    trades = add_fee_and_equity(trades)
    monthly = monthly_breakdown(trades)

    trades.to_csv(TRADES_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    trades[["ts", "equity_mixed", "equity_all_taker", "period"]].to_csv(EQUITY_CSV, index=False)
    write_report(trades, monthly)

    print(f"Wrote {TRADES_CSV}")
    print(f"Wrote {MONTHLY_CSV}")
    print(f"Wrote {EQUITY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
