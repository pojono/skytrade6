"""Centralized configuration for the settlement trading pipeline."""

import os
from pathlib import Path

# ── Remote data source ────────────────────────────────────────────────
REMOTE_HOST = "ubuntu@13.251.79.76"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_remote")
REMOTE_DATA_DIR = "~/skytrade7/logs/market_data"

# ── Local paths ───────────────────────────────────────────────────────
LOCAL_DATA_DIR = Path("charts_settlement")
FEATURES_CSV = Path("settlement_features_v2.csv")
REPORT_FILE = Path("REPORT_ml_settlement.md")
EXIT_ML_TICKS = Path("exit_ml_ticks.parquet")
MODEL_DIR = Path("models")

# ── Fee structure (Bybit) ────────────────────────────────────────────
TAKER_FEE_BPS = 10          # per-leg taker fee (0.10%)
MAKER_FEE_BPS = 4           # per-leg maker fee (0.04%)
RT_TAKER_FEE_BPS = 20       # round-trip taker (10 × 2)

# ── Short leg parameters ─────────────────────────────────────────────
ENTRY_DELAY_MS = 20          # T+20ms BB fill (escapes FR)
GROSS_PNL_BPS = 23.6         # ML LOSO honest short edge
LIMIT_FILL_RATE = 0.54       # limit exit fill rate
LIMIT_RESCUE_TIMEOUT_MS = 1000

# ── Position sizing ──────────────────────────────────────────────────
NOTIONAL_TIERS = [500, 1000, 2000, 3000]
CAP_PCT = 0.15               # 15% of depth_20
MIN_DEPTH_20 = 2000           # minimum $2K bid depth within 20bps
MAX_SPREAD_BPS = 8            # maximum spread

# ── Long leg parameters ──────────────────────────────────────────────
LONG_ENTRY_MAX_T_S = 15.0    # only go long if bottom at T ≤ 15s
LONG_HOLD_FIXED_MS = 20000   # fixed hold baseline (for comparison)
LONG_NOTIONAL_MAX = 1000     # max long notional (conservative)
LONG_SLIP_FACTOR = 0.4       # slippage discount vs T-0 OB
LONG_EXIT_ML_THRESHOLD = 0.6 # LogReg p(near_peak) threshold

# ── Tick-level ML ─────────────────────────────────────────────────────
TICK_MS = 100                 # 100ms tick resolution
MAX_POST_MS = 60000           # 60s post-settlement window

# ── Settlement prediction features ───────────────────────────────────
PRODUCTION_FEATURES = [
    "fr_bps", "fr_abs_bps", "fr_squared",
    "total_depth_usd", "total_depth_imb_mean",
    "ask_concentration", "thin_side_depth", "depth_within_50bps",
    "oi_change_60s",
]


def mixed_fee_bps(fill_rate=None):
    """Compute blended fee for limit+rescue strategy."""
    fr = fill_rate if fill_rate is not None else LIMIT_FILL_RATE
    return fr * MAKER_FEE_BPS + (1 - fr) * TAKER_FEE_BPS


def compute_notional(depth_20, cap_pct=None, tiers=None):
    """Adaptive notional sizing based on orderbook depth."""
    cap = cap_pct if cap_pct is not None else CAP_PCT
    ts = tiers if tiers is not None else NOTIONAL_TIERS
    notional = ts[0]
    for n in ts:
        if n <= depth_20 * cap:
            notional = n
    return notional
