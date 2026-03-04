#!/usr/bin/env python3
"""
Cross-exchange feature engineering.

Takes a merged 5m DataFrame (from load_data.py) and computes cross-exchange
features that may predict future price direction.

Features are designed around the thesis that dislocations between exchanges
create predictable short-term price movements.
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-exchange features on a merged 5m DataFrame.

    All features use ONLY past data (no lookahead).
    Returns the input DataFrame with new feature columns added.
    """
    f = df.copy()

    # --- Midpoint price (average of both exchanges) ---
    f["mid"] = (f["bb_close"] + f["bn_close"]) / 2

    # --- Returns (5m) ---
    f["bb_ret"] = f["bb_close"].pct_change()
    f["bn_ret"] = f["bn_close"].pct_change()
    f["mid_ret"] = f["mid"].pct_change()

    # ========================================================================
    # TIER 1: Price-based cross-exchange signals
    # ========================================================================

    # 1. Price divergence: (Bybit - Binance) / midpoint in bps
    f["price_div_bps"] = (f["bb_close"] - f["bn_close"]) / f["mid"] * 10000

    # 2. Price divergence moving averages (smoothed)
    for w in [6, 12, 36, 72]:  # 30m, 1h, 3h, 6h
        f[f"price_div_ma{w}"] = f["price_div_bps"].rolling(w).mean()

    # 3. Price divergence z-score (how extreme vs recent history)
    for w in [72, 288]:  # 6h, 24h lookback
        mu = f["price_div_bps"].rolling(w).mean()
        sigma = f["price_div_bps"].rolling(w).std()
        f[f"price_div_z{w}"] = (f["price_div_bps"] - mu) / sigma.replace(0, np.nan)

    # 4. Return lead/lag: does Bybit return predict next Binance return (and vice versa)?
    #    We compute rolling correlation and also the raw difference
    f["ret_diff"] = f["bb_ret"] - f["bn_ret"]  # who moved more this bar
    for w in [6, 12]:
        f[f"ret_diff_sum{w}"] = f["ret_diff"].rolling(w).sum() * 10000  # cumulative in bps

    # 5. Bybit leads: cumulative excess return of BB over BN over N bars
    for w in [3, 6, 12]:
        f[f"bb_lead_{w}"] = (f["bb_ret"].rolling(w).sum() - f["bn_ret"].rolling(w).sum()) * 10000

    # ========================================================================
    # TIER 2: Premium-based signals
    # ========================================================================

    if "bb_premium" in f.columns and "bn_premium" in f.columns:
        # 6. Premium spread: BB premium - BN premium (both are futures vs index)
        f["premium_spread"] = (f["bb_premium"] - f["bn_premium"]) * 10000  # in bps

        for w in [6, 12, 36]:
            f[f"premium_spread_ma{w}"] = f["premium_spread"].rolling(w).mean()

        # 7. Premium z-score
        for w in [72, 288]:
            mu = f["premium_spread"].rolling(w).mean()
            sigma = f["premium_spread"].rolling(w).std()
            f[f"premium_z{w}"] = (f["premium_spread"] - mu) / sigma.replace(0, np.nan)

        # 8. Individual premium change (acceleration)
        f["bb_prem_chg"] = f["bb_premium"].diff() * 10000
        f["bn_prem_chg"] = f["bn_premium"].diff() * 10000
        f["prem_chg_div"] = f["bb_prem_chg"] - f["bn_prem_chg"]

    # ========================================================================
    # TIER 3: Volume / flow signals
    # ========================================================================

    # 9. Volume ratio: BB turnover / BN turnover
    f["vol_ratio"] = f["bb_turnover"] / f["bn_turnover"].replace(0, np.nan)
    f["vol_ratio_log"] = np.log(f["vol_ratio"].replace(0, np.nan))

    for w in [12, 36, 72]:
        f[f"vol_ratio_ma{w}"] = f["vol_ratio_log"].rolling(w).mean()

    # 10. Volume ratio z-score (sudden shift in where volume is happening)
    for w in [72, 288]:
        mu = f["vol_ratio_log"].rolling(w).mean()
        sigma = f["vol_ratio_log"].rolling(w).std()
        f[f"vol_ratio_z{w}"] = (f["vol_ratio_log"] - mu) / sigma.replace(0, np.nan)

    # 11. Binance taker buy ratio (aggressive buyers vs sellers)
    if "bn_taker_buy_turnover" in f.columns:
        f["bn_taker_buy_pct"] = f["bn_taker_buy_turnover"] / f["bn_turnover"].replace(0, np.nan)
        f["bn_taker_imbalance"] = (f["bn_taker_buy_pct"] - 0.5) * 2  # -1 to +1

        for w in [6, 12, 36]:
            f[f"bn_taker_imb_ma{w}"] = f["bn_taker_imbalance"].rolling(w).mean()

    # 12. Volume spike detection: current bar vs rolling mean
    for prefix in ["bb", "bn"]:
        turn_col = f"{prefix}_turnover"
        mu = f[turn_col].rolling(72).mean()  # 6h rolling avg
        f[f"{prefix}_vol_spike"] = f[turn_col] / mu.replace(0, np.nan)

    f["vol_spike_ratio"] = f["bb_vol_spike"] / f["bn_vol_spike"].replace(0, np.nan)

    # ========================================================================
    # TIER 4: OI / positioning signals
    # ========================================================================

    if "bb_oi" in f.columns and "bn_oi" in f.columns:
        # 13. OI divergence (rate of change difference)
        f["bb_oi_chg"] = f["bb_oi"].pct_change(6)  # 30m OI change
        f["bn_oi_chg"] = f["bn_oi"].pct_change(6)
        f["oi_div"] = (f["bb_oi_chg"] - f["bn_oi_chg"]) * 10000

        for w in [12, 36]:
            f[f"oi_div_ma{w}"] = f["oi_div"].rolling(w).mean()

    if "bb_ls_ratio" in f.columns and "bn_ls_ratio" in f.columns:
        # 14. LS ratio divergence
        f["ls_div"] = f["bb_ls_ratio"] - f["bn_ls_ratio"]

        for w in [12, 36]:
            f[f"ls_div_ma{w}"] = f["ls_div"].rolling(w).mean()

    # 15. Funding rate features
    if "bb_fr" in f.columns:
        f["bb_fr_bps"] = f["bb_fr"] * 10000
        f["bb_fr_extreme"] = f["bb_fr_bps"].abs()

    # ========================================================================
    # TIER 5: Composite / interaction signals
    # ========================================================================

    # 16. Volume-weighted price divergence momentum
    #     If BB has more volume AND price is higher → strong buy signal for BN
    if "vol_ratio_z72" in f.columns:
        f["flow_momentum"] = f["price_div_bps"] * f["vol_ratio_z72"]

    # 17. OI acceleration + premium divergence
    if "oi_div" in f.columns and "premium_spread" in f.columns:
        f["oi_prem_composite"] = f["oi_div"] * np.sign(f["premium_spread"])

    # ========================================================================
    # TARGET VARIABLES (future returns at various horizons)
    # ========================================================================

    # Forward returns on midpoint (what we're predicting)
    for h in [1, 2, 3, 6, 12, 24]:  # 5m, 10m, 15m, 30m, 1h, 2h
        f[f"fwd_ret_{h}"] = f["mid"].pct_change(h).shift(-h) * 10000  # in bps

    # Forward returns on individual exchanges (for directional trading)
    for h in [3, 6, 12]:
        f[f"bb_fwd_ret_{h}"] = f["bb_close"].pct_change(h).shift(-h) * 10000
        f[f"bn_fwd_ret_{h}"] = f["bn_close"].pct_change(h).shift(-h) * 10000

    # Large move indicator (>50 bps within 30m = 6 bars)
    f["fwd_big_up_6"] = (f["fwd_ret_6"] > 50).astype(int)
    f["fwd_big_dn_6"] = (f["fwd_ret_6"] < -50).astype(int)
    f["fwd_big_up_12"] = (f["fwd_ret_12"] > 50).astype(int)
    f["fwd_big_dn_12"] = (f["fwd_ret_12"] < -50).astype(int)

    return f


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (excludes targets and raw price data)."""
    exclude_prefixes = ("fwd_", "bb_open", "bb_high", "bb_low", "bb_close",
                        "bn_open", "bn_high", "bn_low", "bn_close",
                        "bb_volume", "bn_volume", "bb_turnover", "bn_turnover",
                        "bb_premium", "bn_premium", "bb_fr", "bb_oi", "bn_oi",
                        "bb_ls_ratio", "bn_ls_ratio", "bn_oi_value",
                        "bn_trades", "bn_taker_buy_vol", "bn_taker_buy_turnover",
                        "bn_taker_ls_ratio", "mid")
    features = [c for c in df.columns if not c.startswith(exclude_prefixes)]
    return features


if __name__ == "__main__":
    import time
    from load_data import load_symbol

    t0 = time.time()
    df = load_symbol("SOLUSDT")
    print(f"Loaded SOLUSDT: {len(df)} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    feat = compute_features(df)
    print(f"Features computed in {time.time()-t0:.1f}s")

    fcols = get_feature_columns(feat)
    print(f"\n{len(fcols)} feature columns:")
    for c in fcols:
        print(f"  {c}: {feat[c].notna().sum()} non-null, "
              f"mean={feat[c].mean():.4f}, std={feat[c].std():.4f}")

    # Quick correlation with forward returns
    print(f"\n--- Correlation with fwd_ret_6 (30m forward return) ---")
    target = "fwd_ret_6"
    valid = feat.dropna(subset=[target])
    corrs = {}
    for c in fcols:
        if valid[c].notna().sum() > 1000:
            corrs[c] = valid[c].corr(valid[target])
    corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in corrs[:20]:
        print(f"  {name:30s}  r = {corr:+.4f}")
