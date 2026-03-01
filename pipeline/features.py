"""Feature extraction for settlement, short exit, and long exit ML.

All feature builders work on SettlementData objects (parsed once, reused everywhere).
"""

import numpy as np

from pipeline.config import TICK_MS, MAX_POST_MS


# ═══════════════════════════════════════════════════════════════════════
# SHORT EXIT FEATURES — tick-level during crash phase
# ═══════════════════════════════════════════════════════════════════════

def build_short_exit_ticks(sd):
    """Build 100ms tick features for short exit ML from a SettlementData.

    Target: is this tick near the bottom? (< 10 bps of eventual min)
    Returns list of dicts (one per tick), or None.
    """
    post_times = sd.post_times
    post_prices_bps = sd.post_prices_bps
    post_sides = sd.post_sides
    post_notionals = sd.post_notionals
    post_sizes = sd.post_sizes
    ob1_post = sd.ob1_bids  # [(t, bp, bq, ap, aq), ...]
    ob1_times = sd.ob1_times

    if len(post_times) < 10:
        return None

    max_t = min(float(post_times[-1]), MAX_POST_MS)
    global_min_bps = post_prices_bps.min()

    # Entry price (for PnL targets)
    entry_mask = post_times <= 20  # ENTRY_DELAY_MS
    entry_price_bps = float(post_prices_bps[entry_mask][-1]) if entry_mask.sum() > 0 else 0.0

    rows = []
    running_min_bps = 0.0
    last_new_low_t = 0
    bounce_count = 0
    consecutive_new_lows = 0
    prev_was_new_low = False

    for tick_t in range(0, int(max_t), TICK_MS):
        mask_up_to = post_times <= tick_t
        if mask_up_to.sum() < 2:
            continue

        current_prices = post_prices_bps[mask_up_to]
        current_times = post_times[mask_up_to]
        current_price = current_prices[-1]

        # Running minimum
        new_min = current_prices.min()
        is_new_low = new_min < running_min_bps - 0.5
        if new_min < running_min_bps:
            running_min_bps = new_min
            last_new_low_t = tick_t
        elif tick_t == 0:
            running_min_bps = current_price

        # Sequence tracking
        if is_new_low:
            if not prev_was_new_low:
                consecutive_new_lows = 1
            else:
                consecutive_new_lows += 1
            prev_was_new_low = True
        else:
            if prev_was_new_low and current_price > running_min_bps + 3:
                bounce_count += 1
            consecutive_new_lows = 0
            prev_was_new_low = False

        feat = {}
        feat["t_ms"] = tick_t
        feat["price_bps"] = current_price
        feat["entry_price_bps"] = entry_price_bps
        feat["running_min_bps"] = running_min_bps
        feat["distance_from_low_bps"] = current_price - running_min_bps
        feat["new_low"] = 1 if current_price <= running_min_bps + 0.5 else 0
        feat["time_since_new_low_ms"] = tick_t - last_new_low_t
        feat["pct_of_window_elapsed"] = tick_t / MAX_POST_MS

        # Price velocity
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s")]:
            w_mask = (current_times > tick_t - window) & (current_times <= tick_t)
            w_prices = post_prices_bps[mask_up_to][-w_mask.sum():] if w_mask.sum() > 0 else np.array([])
            feat[f"price_velocity_{label}"] = (w_prices[-1] - w_prices[0]) if len(w_prices) >= 2 else 0

        # Acceleration
        mask_early = (current_times > tick_t - 1000) & (current_times <= tick_t - 500)
        mask_late = (current_times > tick_t - 500) & (current_times <= tick_t)
        if mask_early.sum() >= 1 and mask_late.sum() >= 1:
            early_p = post_prices_bps[mask_up_to][-mask_late.sum() - mask_early.sum():-mask_late.sum()]
            late_p = post_prices_bps[mask_up_to][-mask_late.sum():]
            v_early = early_p[-1] - early_p[0] if len(early_p) > 1 else 0
            v_late = late_p[-1] - late_p[0] if len(late_p) > 1 else 0
            feat["price_accel"] = v_late - v_early
        else:
            feat["price_accel"] = 0

        # Drop rate
        feat["drop_rate_bps_per_s"] = running_min_bps / (tick_t / 1000) if tick_t > 0 else 0

        # Trade flow
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s"), (5000, "5s")]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_sides = post_sides[w_mask]
            w_notionals = post_notionals[w_mask]
            w_sizes = post_sizes[w_mask]
            n_trades = w_mask.sum()
            total_vol = w_notionals.sum()
            sell_vol = w_notionals[w_sides == 1].sum()

            feat[f"sell_ratio_{label}"] = sell_vol / total_vol if total_vol > 0 else 0.5
            feat[f"trade_rate_{label}"] = n_trades / (window / 1000)
            feat[f"vol_rate_{label}"] = total_vol / (window / 1000)
            feat[f"avg_size_{label}"] = w_sizes.mean() if n_trades > 0 else 0
            feat[f"large_trade_pct_{label}"] = (
                (w_sizes > 2 * np.median(w_sizes)).mean() if n_trades > 2 else 0
            )

        # Surge
        feat["vol_surge"] = feat["vol_rate_1s"] / sd.pre_vol_rate if sd.pre_vol_rate > 0 else 1
        feat["trade_surge"] = feat["trade_rate_1s"] / sd.pre_trade_rate if sd.pre_trade_rate > 0 else 1

        # Buy/sell ratio
        mask_1s = (post_times > tick_t - 1000) & (post_times <= tick_t)
        buy_v = post_notionals[mask_1s & (post_sides == 0)].sum()
        sell_v = post_notionals[mask_1s & (post_sides == 1)].sum()
        feat["buy_sell_ratio_1s"] = buy_v / sell_v if sell_v > 0 else 1.0

        # Trade rate accel
        if tick_t >= 2000:
            r_early = ((post_times > tick_t - 2000) & (post_times <= tick_t - 1000)).sum()
            r_late = ((post_times > tick_t - 1000) & (post_times <= tick_t)).sum()
            feat["trade_rate_accel"] = r_late - r_early
        else:
            feat["trade_rate_accel"] = 0

        # Orderbook
        if len(ob1_times) > 0:
            ob_mask = ob1_times <= tick_t
            if ob_mask.sum() > 0:
                idx = ob_mask.sum() - 1
                _, bp, bq, ap, aq = ob1_post[idx]
                spread = (ap - bp) / sd.ref_price * 10000
                feat["spread_bps"] = spread
                feat["spread_change"] = spread - sd.pre_spread_bps
                feat["ob1_bid_qty"] = bq
                feat["ob1_ask_qty"] = aq
                feat["ob1_imbalance"] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0
                if idx > 0:
                    _, _, bq2, _, aq2 = ob1_post[idx - 1]
                    feat["bid_qty_change"] = bq - bq2
                    feat["ask_qty_change"] = aq - aq2
                else:
                    feat["bid_qty_change"] = feat["ask_qty_change"] = 0
            else:
                for k in ["spread_bps", "spread_change", "ob1_bid_qty", "ob1_ask_qty",
                           "ob1_imbalance", "bid_qty_change", "ask_qty_change"]:
                    feat[k] = 0
        else:
            for k in ["spread_bps", "spread_change", "ob1_bid_qty", "ob1_ask_qty",
                       "ob1_imbalance", "bid_qty_change", "ask_qty_change"]:
                feat[k] = 0

        # Sequence features (v3)
        feat["bounce_count"] = bounce_count
        feat["consecutive_new_lows"] = consecutive_new_lows

        for window, label in [(2000, "2s"), (5000, "5s")]:
            w_mask = (current_times > tick_t - window) & (current_times <= tick_t)
            w_prices = post_prices_bps[mask_up_to][-(w_mask.sum()):]
            if len(w_prices) >= 2:
                feat[f"price_range_{label}"] = w_prices.max() - w_prices.min()
                feat[f"price_std_{label}"] = w_prices.std()
            else:
                feat[f"price_range_{label}"] = feat[f"price_std_{label}"] = 0

        recent_times = current_times[(current_times > tick_t - 1000) & (current_times <= tick_t)]
        if len(recent_times) >= 2:
            diffs = np.diff(recent_times)
            feat["avg_inter_trade_ms"] = diffs.mean()
            feat["max_inter_trade_ms"] = diffs.max()
        else:
            feat["avg_inter_trade_ms"] = feat["max_inter_trade_ms"] = 1000

        w_mask = (current_times > tick_t - 2000) & (current_times <= tick_t)
        w_prices = post_prices_bps[mask_up_to][-(w_mask.sum()):]
        if len(w_prices) >= 3:
            diffs = np.diff(w_prices)
            signs = np.sign(diffs)
            signs = signs[signs != 0]
            feat["reversals_2s"] = (np.diff(signs) != 0).sum() if len(signs) >= 2 else 0
        else:
            feat["reversals_2s"] = 0

        # Static
        feat["fr_bps"] = sd.fr_bps
        feat["fr_abs_bps"] = sd.fr_abs_bps
        feat["t_seconds"] = tick_t / 1000.0
        feat["log_t"] = np.log1p(tick_t)
        feat["phase"] = (0 if tick_t < 1000 else 1 if tick_t < 5000 else
                         2 if tick_t < 10000 else 3 if tick_t < 30000 else 4)

        # Targets
        future_mask = post_times > tick_t
        if future_mask.sum() > 0:
            future_prices = post_prices_bps[future_mask]
            future_min = future_prices.min()
            feat["target_drop_remaining"] = current_price - future_min
            feat["target_near_bottom_5"] = 1 if feat["target_drop_remaining"] < 5 else 0
            feat["target_near_bottom_10"] = 1 if feat["target_drop_remaining"] < 10 else 0
            feat["target_near_bottom_15"] = 1 if feat["target_drop_remaining"] < 15 else 0
            feat["target_bottom_passed"] = 1 if running_min_bps <= global_min_bps + 0.5 else 0
        else:
            feat["target_drop_remaining"] = 0
            for t in ["target_near_bottom_5", "target_near_bottom_10",
                       "target_near_bottom_15", "target_bottom_passed"]:
                feat[t] = 1

        feat["symbol"] = sd.symbol
        feat["settle_id"] = sd.settle_id
        rows.append(feat)

    return rows if rows else None


# ═══════════════════════════════════════════════════════════════════════
# LONG EXIT FEATURES — tick-level during recovery phase
# ═══════════════════════════════════════════════════════════════════════

def find_bottom(sd, t_min_ms=1000, t_max_ms=30000):
    """Find price bottom between t_min and t_max in price_bins.
    Returns (bottom_bps, bottom_t_ms) or (None, None).
    """
    bottom_bps = None
    bottom_t = None
    for t_ms in sorted(sd.price_bins.keys()):
        if t_ms < t_min_ms or t_ms > t_max_ms:
            continue
        p = sd.price_bins[t_ms]
        if bottom_bps is None or p < bottom_bps:
            bottom_bps = p
            bottom_t = t_ms
    return bottom_bps, bottom_t


def build_long_exit_ticks(sd, bottom_bps=None, bottom_t=None):
    """Build 100ms tick features for long exit ML during recovery phase.

    Target: is this tick near the recovery peak?
    Returns list of dicts, or None.
    """
    if bottom_bps is None or bottom_t is None:
        bottom_bps, bottom_t = find_bottom(sd)
    if bottom_bps is None:
        return None

    drop_bps = -bottom_bps  # positive = how much it dropped
    if drop_bps <= 0:
        return None

    post_times = sd.post_times
    post_prices_bps = sd.post_prices_bps
    post_sides = sd.post_sides
    post_notionals = sd.post_notionals
    ob1_post = sd.ob1_bids
    ob1_times = sd.ob1_times

    max_recovery_t = min(bottom_t + 30000, 60000)

    rows = []
    running_max_bps = bottom_bps
    last_new_high_t = bottom_t

    for tick_t in range(bottom_t + TICK_MS, int(max_recovery_t), TICK_MS):
        mask_up_to = post_times <= tick_t
        if mask_up_to.sum() < 2:
            continue

        mask_recovery = (post_times > bottom_t) & (post_times <= tick_t)
        if mask_recovery.sum() < 1:
            continue

        current_price = post_prices_bps[mask_up_to][-1]
        recovery_so_far = current_price - bottom_bps
        time_since_bottom = tick_t - bottom_t

        recovery_prices = post_prices_bps[mask_recovery]
        new_max = recovery_prices.max()
        if new_max > running_max_bps:
            running_max_bps = new_max
            last_new_high_t = tick_t

        feat = {}
        feat['t_ms'] = tick_t
        feat['time_since_bottom_ms'] = time_since_bottom
        feat['pct_recovery_elapsed'] = time_since_bottom / 30000.0

        feat['recovery_bps'] = recovery_so_far
        feat['recovery_pct_of_drop'] = recovery_so_far / drop_bps * 100 if drop_bps > 0 else 0
        feat['running_max_bps'] = running_max_bps - bottom_bps
        feat['distance_from_high_bps'] = running_max_bps - current_price
        feat['time_since_new_high_ms'] = tick_t - last_new_high_t

        # Velocity
        for window, label in [(500, '500ms'), (1000, '1s'), (2000, '2s')]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_prices = post_prices_bps[w_mask]
            feat[f'velocity_{label}'] = (w_prices[-1] - w_prices[0]) if len(w_prices) >= 2 else 0

        # Acceleration
        mask_early = (post_times > tick_t - 1000) & (post_times <= tick_t - 500)
        mask_late = (post_times > tick_t - 500) & (post_times <= tick_t)
        if mask_early.sum() >= 1 and mask_late.sum() >= 1:
            v_early = post_prices_bps[mask_early][-1] - post_prices_bps[mask_early][0] if mask_early.sum() > 1 else 0
            v_late = post_prices_bps[mask_late][-1] - post_prices_bps[mask_late][0] if mask_late.sum() > 1 else 0
            feat['acceleration'] = v_late - v_early
        else:
            feat['acceleration'] = 0

        # Trade flow
        for window, label in [(500, '500ms'), (1000, '1s'), (2000, '2s')]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_sides = post_sides[w_mask]
            w_notionals = post_notionals[w_mask]
            total_vol = w_notionals.sum()
            buy_vol = w_notionals[w_sides == 0].sum()
            feat[f'buy_ratio_{label}'] = buy_vol / total_vol if total_vol > 0 else 0.5
            feat[f'vol_rate_{label}'] = total_vol / (window / 1000)
            feat[f'trade_count_{label}'] = w_mask.sum()

        # Buy pressure momentum
        mask_1s_early = (post_times > tick_t - 2000) & (post_times <= tick_t - 1000)
        mask_1s_late = (post_times > tick_t - 1000) & (post_times <= tick_t)
        if mask_1s_early.sum() > 0 and mask_1s_late.sum() > 0:
            br_early = post_notionals[mask_1s_early & (post_sides == 0)].sum() / max(post_notionals[mask_1s_early].sum(), 1)
            br_late = post_notionals[mask_1s_late & (post_sides == 0)].sum() / max(post_notionals[mask_1s_late].sum(), 1)
            feat['buy_ratio_change'] = br_late - br_early
        else:
            feat['buy_ratio_change'] = 0

        # OB
        if len(ob1_times) > 0:
            ob_mask = ob1_times <= tick_t
            if ob_mask.sum() > 0:
                idx = ob_mask.sum() - 1
                _, bp, bq, ap, aq = ob1_post[idx]
                feat['spread_bps'] = (ap - bp) / sd.ref_price * 10000
                feat['ob1_imbalance'] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0
            else:
                feat['spread_bps'] = feat['ob1_imbalance'] = 0
        else:
            feat['spread_bps'] = feat['ob1_imbalance'] = 0

        # Static context
        feat['drop_bps'] = drop_bps
        feat['bottom_t_s'] = bottom_t / 1000.0
        feat['fr_abs_bps'] = sd.fr_abs_bps

        # Recovery range/std
        if len(recovery_prices) >= 2:
            feat['price_range_recovery'] = recovery_prices.max() - recovery_prices.min()
            feat['price_std_recovery'] = recovery_prices.std()
        else:
            feat['price_range_recovery'] = feat['price_std_recovery'] = 0

        # Targets
        future_mask = (post_times > tick_t) & (post_times <= max_recovery_t)
        if future_mask.sum() > 0:
            future_prices = post_prices_bps[future_mask]
            future_max = future_prices.max()
            feat['target_upside_remaining'] = future_max - current_price
            feat['target_near_peak_5'] = 1 if feat['target_upside_remaining'] < 5 else 0
            feat['target_near_peak_10'] = 1 if feat['target_upside_remaining'] < 10 else 0
            mask_5s = (post_times > tick_t) & (post_times <= tick_t + 5000)
            if mask_5s.sum() > 0:
                feat['target_drops_5bps_in_5s'] = 1 if (current_price - post_prices_bps[mask_5s].min()) > 5 else 0
            else:
                feat['target_drops_5bps_in_5s'] = 0
        else:
            feat['target_upside_remaining'] = 0
            feat['target_near_peak_5'] = feat['target_near_peak_10'] = 1
            feat['target_drops_5bps_in_5s'] = 0

        feat['symbol'] = sd.symbol
        feat['settle_id'] = sd.settle_id
        rows.append(feat)

    return rows if rows else None
