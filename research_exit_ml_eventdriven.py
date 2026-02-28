#!/usr/bin/env python3
"""
Event-Driven Exit ML — Process every WS event, evaluate on triggers
====================================================================

Architecture:
  1. StreamingState: O(1) incremental updates per event
  2. Triggers: evaluate model only on meaningful events
  3. Feature extraction: from streaming state (no recomputation)
  4. Simulator: replays JSONL, compares event-driven vs 100ms polling

Trigger types:
  - NEW_LOW: price made a new running minimum
  - BOUNCE: price bounced >3 bps off running minimum
  - COOLDOWN: 100ms since last evaluation (fallback)
  - BIG_TRADE: single trade > 2x rolling median size

Usage:
    python3 research_exit_ml_eventdriven.py
"""

import json
import sys
import time as _time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

LOCAL_DATA_DIR = Path("charts_settlement")
MAX_POST_MS = 60000

# ═══════════════════════════════════════════════════════════════════════
# Streaming State — O(1) updates per event
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StreamingState:
    """Maintains all feature state incrementally. No recomputation."""
    ref_price: float = 0.0
    fr_bps: float = 0.0

    # Pre-settlement baselines
    pre_vol_rate: float = 1.0
    pre_trade_rate: float = 1.0
    pre_spread_bps: float = 0.0

    # Running price stats
    running_min_bps: float = 0.0
    current_price_bps: float = 0.0
    last_new_low_t: float = 0.0

    # Sequence tracking
    bounce_count: int = 0
    consecutive_new_lows: int = 0
    prev_was_new_low: bool = False

    # OB state
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    prev_bid_qty: float = 0.0
    prev_ask_qty: float = 0.0

    # Rolling windows — store recent events in deques
    # trades: (t_ms, price_bps, qty, is_sell, notional)
    trades_500ms: deque = field(default_factory=deque)
    trades_1s: deque = field(default_factory=deque)
    trades_2s: deque = field(default_factory=deque)
    trades_5s: deque = field(default_factory=deque)

    # Aggregates maintained incrementally
    sell_vol_500ms: float = 0.0
    total_vol_500ms: float = 0.0
    sell_vol_1s: float = 0.0
    total_vol_1s: float = 0.0
    sell_vol_2s: float = 0.0
    total_vol_2s: float = 0.0
    sell_vol_5s: float = 0.0
    total_vol_5s: float = 0.0
    n_trades_500ms: int = 0
    n_trades_1s: int = 0
    n_trades_2s: int = 0
    n_trades_5s: int = 0
    sum_qty_1s: float = 0.0
    sum_qty_500ms: float = 0.0
    sum_qty_2s: float = 0.0
    sum_qty_5s: float = 0.0
    buy_vol_1s: float = 0.0

    # For large trade detection
    recent_sizes: deque = field(default_factory=lambda: deque(maxlen=50))
    median_size: float = 0.0

    # Price history for velocity/acceleration/range
    price_history: deque = field(default_factory=lambda: deque(maxlen=500))

    # Trigger state
    last_eval_t: float = -1000.0
    total_trades: int = 0

    def bps(self, price):
        if self.ref_price == 0:
            return 0.0
        return (price / self.ref_price - 1) * 10000

    def _expire_window(self, window_ms, t_ms):
        """Remove old trades from rolling windows and update aggregates."""
        cutoff = t_ms - window_ms

        for window, attr_suffix in [
            (500, "500ms"), (1000, "1s"), (2000, "2s"), (5000, "5s")
        ]:
            dq = getattr(self, f"trades_{attr_suffix}")
            while dq and dq[0][0] <= t_ms - window:
                old_t, old_p, old_q, old_sell, old_n = dq.popleft()
                setattr(self, f"total_vol_{attr_suffix}",
                        getattr(self, f"total_vol_{attr_suffix}") - old_n)
                if old_sell:
                    setattr(self, f"sell_vol_{attr_suffix}",
                            getattr(self, f"sell_vol_{attr_suffix}") - old_n)
                setattr(self, f"n_trades_{attr_suffix}",
                        getattr(self, f"n_trades_{attr_suffix}") - 1)
                setattr(self, f"sum_qty_{attr_suffix}",
                        getattr(self, f"sum_qty_{attr_suffix}") - old_q)
                if attr_suffix == "1s" and not old_sell:
                    self.buy_vol_1s -= old_n

    def on_trade(self, t_ms, price, qty, side, notional):
        """Process a single trade event. O(1) amortized."""
        self._expire_window(5000, t_ms)

        price_bps = self.bps(price)
        is_sell = 1 if side == "Sell" else 0
        entry = (t_ms, price_bps, qty, is_sell, notional)

        # Add to all windows
        for suffix in ["500ms", "1s", "2s", "5s"]:
            getattr(self, f"trades_{suffix}").append(entry)
            setattr(self, f"total_vol_{suffix}",
                    getattr(self, f"total_vol_{suffix}") + notional)
            if is_sell:
                setattr(self, f"sell_vol_{suffix}",
                        getattr(self, f"sell_vol_{suffix}") + notional)
            setattr(self, f"n_trades_{suffix}",
                    getattr(self, f"n_trades_{suffix}") + 1)
            setattr(self, f"sum_qty_{suffix}",
                    getattr(self, f"sum_qty_{suffix}") + qty)
        if not is_sell:
            self.buy_vol_1s += notional

        self.current_price_bps = price_bps
        self.price_history.append((t_ms, price_bps))
        self.total_trades += 1

        # Running minimum
        is_new_low = price_bps < self.running_min_bps - 0.5
        if price_bps < self.running_min_bps:
            self.running_min_bps = price_bps
            self.last_new_low_t = t_ms

        # Sequence tracking
        if is_new_low:
            if not self.prev_was_new_low:
                self.consecutive_new_lows = 1
            else:
                self.consecutive_new_lows += 1
            self.prev_was_new_low = True
        else:
            if self.prev_was_new_low and price_bps > self.running_min_bps + 3:
                self.bounce_count += 1
            self.consecutive_new_lows = 0
            self.prev_was_new_low = False

        # Median trade size tracking
        self.recent_sizes.append(qty)
        if len(self.recent_sizes) >= 5:
            self.median_size = float(np.median(list(self.recent_sizes)))

        # Return trigger type
        trigger = None
        if is_new_low:
            trigger = "NEW_LOW"
        elif price_bps > self.running_min_bps + 3 and not is_new_low:
            trigger = "BOUNCE"
        elif self.median_size > 0 and qty > 2 * self.median_size:
            trigger = "BIG_TRADE"
        return trigger

    def on_ob1(self, t_ms, bid_price, bid_qty, ask_price, ask_qty):
        """Process orderbook L1 update."""
        self.prev_bid_qty = self.bid_qty
        self.prev_ask_qty = self.ask_qty
        self.best_bid = bid_price
        self.best_ask = ask_price
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty

    def compute_features(self, t_ms):
        """Extract feature dict from current streaming state. O(1)."""
        feat = {}
        feat["t_ms"] = t_ms
        feat["price_bps"] = self.current_price_bps
        feat["running_min_bps"] = self.running_min_bps
        feat["distance_from_low_bps"] = self.current_price_bps - self.running_min_bps
        feat["new_low"] = 1 if self.current_price_bps <= self.running_min_bps + 0.5 else 0
        feat["time_since_new_low_ms"] = t_ms - self.last_new_low_t
        feat["pct_of_window_elapsed"] = t_ms / MAX_POST_MS

        # Price velocity from history
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s")]:
            cutoff = t_ms - window
            prices_in_window = [(pt, pp) for pt, pp in self.price_history if pt > cutoff]
            if len(prices_in_window) >= 2:
                feat[f"price_velocity_{label}"] = prices_in_window[-1][1] - prices_in_window[0][1]
            else:
                feat[f"price_velocity_{label}"] = 0

        # Price acceleration
        prices_early = [(pt, pp) for pt, pp in self.price_history if t_ms - 1000 < pt <= t_ms - 500]
        prices_late = [(pt, pp) for pt, pp in self.price_history if t_ms - 500 < pt <= t_ms]
        if prices_early and prices_late:
            v_early = prices_early[-1][1] - prices_early[0][1] if len(prices_early) > 1 else 0
            v_late = prices_late[-1][1] - prices_late[0][1] if len(prices_late) > 1 else 0
            feat["price_accel"] = v_late - v_early
        else:
            feat["price_accel"] = 0

        # Drop rate
        feat["drop_rate_bps_per_s"] = self.running_min_bps / (t_ms / 1000) if t_ms > 0 else 0

        # Trade flow from maintained aggregates
        for suffix, window_s in [("500ms", 0.5), ("1s", 1.0), ("2s", 2.0), ("5s", 5.0)]:
            tv = getattr(self, f"total_vol_{suffix}")
            sv = getattr(self, f"sell_vol_{suffix}")
            nt = getattr(self, f"n_trades_{suffix}")
            sq = getattr(self, f"sum_qty_{suffix}")

            feat[f"sell_ratio_{suffix}"] = sv / tv if tv > 0 else 0.5
            feat[f"trade_rate_{suffix}"] = nt / window_s
            feat[f"vol_rate_{suffix}"] = tv / window_s
            feat[f"avg_size_{suffix}"] = sq / nt if nt > 0 else 0
            feat[f"large_trade_pct_{suffix}"] = 0  # simplified for streaming

        # Surges
        feat["vol_surge"] = feat["vol_rate_1s"] / self.pre_vol_rate if self.pre_vol_rate > 0 else 1
        feat["trade_surge"] = feat["trade_rate_1s"] / self.pre_trade_rate if self.pre_trade_rate > 0 else 1

        # Buy/sell ratio
        sv1 = self.sell_vol_1s
        feat["buy_sell_ratio_1s"] = self.buy_vol_1s / sv1 if sv1 > 0 else 1.0

        # Trade rate accel
        if t_ms >= 2000:
            early_trades = sum(1 for pt, _ in self.price_history if t_ms - 2000 < pt <= t_ms - 1000)
            late_trades = sum(1 for pt, _ in self.price_history if t_ms - 1000 < pt <= t_ms)
            feat["trade_rate_accel"] = late_trades - early_trades
        else:
            feat["trade_rate_accel"] = 0

        # Orderbook
        if self.best_bid > 0 and self.best_ask > 0:
            spread = (self.best_ask - self.best_bid) / self.ref_price * 10000
            feat["spread_bps"] = spread
            feat["spread_change"] = spread - self.pre_spread_bps
            feat["ob1_bid_qty"] = self.bid_qty
            feat["ob1_ask_qty"] = self.ask_qty
            feat["ob1_imbalance"] = ((self.bid_qty - self.ask_qty) /
                                      (self.bid_qty + self.ask_qty)
                                      if (self.bid_qty + self.ask_qty) > 0 else 0)
            feat["bid_qty_change"] = self.bid_qty - self.prev_bid_qty
            feat["ask_qty_change"] = self.ask_qty - self.prev_ask_qty
        else:
            for k in ["spread_bps", "spread_change", "ob1_bid_qty", "ob1_ask_qty",
                       "ob1_imbalance", "bid_qty_change", "ask_qty_change"]:
                feat[k] = 0

        # Sequence features (v3)
        feat["bounce_count"] = self.bounce_count
        feat["consecutive_new_lows"] = self.consecutive_new_lows

        # Price range/std from history
        for window, label in [(2000, "2s"), (5000, "5s")]:
            prices_w = [pp for pt, pp in self.price_history if pt > t_ms - window]
            if len(prices_w) >= 2:
                feat[f"price_range_{label}"] = max(prices_w) - min(prices_w)
                feat[f"price_std_{label}"] = float(np.std(prices_w))
            else:
                feat[f"price_range_{label}"] = 0
                feat[f"price_std_{label}"] = 0

        # Inter-trade time
        recent_t = [pt for pt, _ in self.price_history if pt > t_ms - 1000]
        if len(recent_t) >= 2:
            diffs = np.diff(recent_t)
            feat["avg_inter_trade_ms"] = diffs.mean()
            feat["max_inter_trade_ms"] = diffs.max()
        else:
            feat["avg_inter_trade_ms"] = 1000
            feat["max_inter_trade_ms"] = 1000

        # Reversals
        prices_2s = [pp for pt, pp in self.price_history if pt > t_ms - 2000]
        if len(prices_2s) >= 3:
            diffs = np.diff(prices_2s)
            signs = np.sign(diffs)
            signs = signs[signs != 0]
            feat["reversals_2s"] = int((np.diff(signs) != 0).sum()) if len(signs) >= 2 else 0
        else:
            feat["reversals_2s"] = 0

        # Static
        feat["fr_bps"] = self.fr_bps
        feat["fr_abs_bps"] = abs(self.fr_bps)
        feat["t_seconds"] = t_ms / 1000.0
        feat["log_t"] = np.log1p(t_ms)
        feat["phase"] = (0 if t_ms < 1000 else 1 if t_ms < 5000 else
                         2 if t_ms < 10000 else 3 if t_ms < 30000 else 4)

        return feat


# ═══════════════════════════════════════════════════════════════════════
# Simulator: replay JSONL event-by-event
# ═══════════════════════════════════════════════════════════════════════

def simulate_settlement(fp, model, feature_cols, threshold=0.5,
                        min_exit_ms=1000, mode="event_driven",
                        cooldown_ms=100):
    """
    Replay a JSONL file and find exit point.

    Modes:
      - "polling_100ms": evaluate every 100ms (current approach)
      - "event_driven":  evaluate on triggers + 100ms cooldown
      - "every_trade":   evaluate on every single trade (upper bound)

    Returns: (exit_t_ms, exit_price_bps, n_evals, trigger_type)
    """
    # Parse all events
    events = []  # (t_ms, type, data)
    with open(fp) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except:
                continue
            t = msg.get("_t_ms", 0)
            topic = msg.get("topic", "")
            data = msg.get("data", {})

            if "publicTrade" in topic:
                for tr in (data if isinstance(data, list) else [data]):
                    p = float(tr.get("p", 0))
                    q = float(tr.get("v", 0))
                    s = tr.get("S", "")
                    events.append((t, "trade", p, q, s, p * q))
            elif topic.startswith("orderbook.1."):
                b = data.get("b", [])
                a = data.get("a", [])
                if b and a:
                    events.append((t, "ob1", float(b[0][0]), float(b[0][1]),
                                   float(a[0][0]), float(a[0][1])))
            elif "tickers" in topic:
                fr = float(data.get("fundingRate", 0))
                events.append((t, "ticker", fr))

    events.sort(key=lambda x: x[0])

    # Split pre/post
    pre_trades = [(e[0], e[2]) for e in events if e[1] == "trade" and e[0] < 0]
    if not pre_trades:
        return None

    state = StreamingState()
    state.ref_price = pre_trades[-1][1]

    # Pre-settlement stats
    pre_10s_trades = [e for e in events if e[1] == "trade" and -10000 <= e[0] < 0]
    if pre_10s_trades:
        state.pre_vol_rate = sum(e[5] for e in pre_10s_trades) / 10.0
        state.pre_trade_rate = len(pre_10s_trades) / 10.0

    pre_ob = [e for e in events if e[1] == "ob1" and -5000 <= e[0] < 0]
    if pre_ob:
        _, _, bp, bq, ap, aq = pre_ob[-1]
        state.pre_spread_bps = (ap - bp) / state.ref_price * 10000

    pre_tickers = [e for e in events if e[1] == "ticker" and e[0] < 0]
    if pre_tickers:
        state.fr_bps = pre_tickers[-1][2] * 10000

    # Post-settlement events only
    post_events = [e for e in events if e[0] >= 0]
    if len([e for e in post_events if e[1] == "trade"]) < 10:
        return None

    # Oracle: find actual minimum
    post_trade_prices = [state.bps(e[2]) for e in post_events if e[1] == "trade"]
    oracle_min_bps = min(post_trade_prices) if post_trade_prices else 0

    # Replay
    n_evals = 0
    exit_t = None
    exit_price = None
    exit_trigger = None
    next_poll_t = cooldown_ms  # for polling mode

    for evt in post_events:
        t_ms = evt[0]
        if t_ms > MAX_POST_MS:
            break

        if evt[1] == "trade":
            _, _, price, qty, side, notional = evt
            trigger = state.on_trade(t_ms, price, qty, side, notional)

            should_eval = False

            if mode == "polling_100ms":
                if t_ms >= next_poll_t:
                    should_eval = True
                    next_poll_t = t_ms + 100
            elif mode == "every_trade":
                should_eval = True
            elif mode == "event_driven":
                if trigger in ("NEW_LOW", "BOUNCE", "BIG_TRADE"):
                    should_eval = True
                elif t_ms - state.last_eval_t >= cooldown_ms:
                    should_eval = True  # cooldown fallback

            if should_eval and t_ms >= min_exit_ms and state.total_trades >= 5:
                feat = state.compute_features(t_ms)
                feat_arr = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
                prob = model.predict_proba(feat_arr)[0, 1]
                n_evals += 1
                state.last_eval_t = t_ms

                if prob > threshold:
                    exit_t = t_ms
                    exit_price = state.current_price_bps
                    exit_trigger = trigger or "COOLDOWN"
                    break

        elif evt[1] == "ob1":
            _, _, bp, bq, ap, aq = evt
            state.on_ob1(t_ms, bp, bq, ap, aq)

    # If no exit triggered, use last known price
    if exit_t is None:
        last_trade = [e for e in post_events if e[1] == "trade"]
        if last_trade:
            exit_t = last_trade[-1][0]
            exit_price = state.bps(last_trade[-1][2])
            exit_trigger = "TIMEOUT"

    return {
        "exit_t": exit_t,
        "exit_price_bps": exit_price,
        "n_evals": n_evals,
        "trigger": exit_trigger,
        "oracle_min_bps": oracle_min_bps,
        "pnl_bps": -exit_price - 20 if exit_price is not None else 0,
        "oracle_pnl_bps": -oracle_min_bps - 20,
    }


# ═══════════════════════════════════════════════════════════════════════
# Train model on 100ms ticks, then test both modes
# ═══════════════════════════════════════════════════════════════════════

def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import gc

    print("=" * 70)
    print("EVENT-DRIVEN vs POLLING EXIT ML")
    print("=" * 70)

    # STEP 1: Train model on 100ms tick data (same as v3)
    print("\n[1] Training model on 100ms tick data...")
    import research_exit_ml_v3 as v3
    v3.TICK_MS = 100

    files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    dfs = []
    for fp in files:
        df = v3.build_tick_features(fp)
        if df is not None:
            dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    print(f"    {len(df_all)} ticks, {df_all['settle_id'].nunique()} settlements")

    SKIP = {"target_drop_remaining", "target_near_bottom_5", "target_near_bottom_10",
            "target_near_bottom_15", "target_bottom_passed", "symbol", "settle_id", "t_ms", "phase"}
    feature_cols = [c for c in df_all.columns if c not in SKIP]

    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["target_near_bottom_10"].values

    # Train on ALL data (we're testing inference mode, not generalization)
    print(f"    Training LogReg on {len(feature_cols)} features...")
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.1, max_iter=5000))
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    print(f"    LogReg AUC (train): {auc:.4f}")

    print(f"    Training HGBC on {len(feature_cols)} features...")
    hgbc = HistGradientBoostingClassifier(max_iter=300, max_depth=6, min_samples_leaf=30,
                                           learning_rate=0.05, l2_regularization=1.0, random_state=42)
    hgbc.fit(X, y)
    auc_hg = roc_auc_score(y, hgbc.predict_proba(X)[:, 1])
    print(f"    HGBC AUC (train): {auc_hg:.4f}")
    del X, y, df_all; gc.collect()

    # STEP 2: Simulate all settlements with both modes
    print(f"\n[2] Simulating {len(files)} files with 3 modes...")
    modes = ["polling_100ms", "event_driven", "every_trade"]
    # Test with LogReg (production candidate)
    model = lr

    results = {m: [] for m in modes}
    for i, fp in enumerate(files, 1):
        for mode in modes:
            r = simulate_settlement(fp, model, feature_cols, threshold=0.5,
                                    min_exit_ms=1000, mode=mode, cooldown_ms=100)
            if r is not None:
                r["file"] = fp.stem
                r["mode"] = mode
                results[mode].append(r)
        if i % 30 == 0:
            print(f"    [{i}/{len(files)}]")

    # STEP 3: Compare
    print(f"\n{'='*70}")
    print(f"RESULTS COMPARISON (LogReg, threshold=0.5)")
    print(f"{'='*70}")

    print(f"\n{'Mode':<20s} {'N':>4s} {'Avg PnL':>8s} {'Med PnL':>8s} {'WR':>6s} "
          f"{'Avg Exit':>9s} {'Evals/settle':>13s} {'Oracle%':>8s}")
    print(f"{'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*13} {'-'*8}")

    for mode in modes:
        data = results[mode]
        if not data:
            continue
        pnls = [r["pnl_bps"] for r in data]
        exits = [r["exit_t"] / 1000 for r in data]
        evals = [r["n_evals"] for r in data]
        oracles = [r["oracle_pnl_bps"] for r in data]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        oracle_pct = np.mean(pnls) / np.mean(oracles) * 100 if np.mean(oracles) > 0 else 0

        print(f"{mode:<20s} {len(data):4d} {np.mean(pnls):+8.1f} {np.median(pnls):+8.1f} "
              f"{wr:5.0f}% {np.mean(exits):8.1f}s {np.mean(evals):12.0f} {oracle_pct:7.0f}%")

    # Detailed exit timing comparison
    print(f"\n{'='*70}")
    print(f"EXIT TIMING ANALYSIS")
    print(f"{'='*70}")

    # Match settlements across modes for paired comparison
    poll_map = {r["file"]: r for r in results["polling_100ms"]}
    ed_map = {r["file"]: r for r in results["event_driven"]}
    et_map = {r["file"]: r for r in results["every_trade"]}

    common = set(poll_map) & set(ed_map) & set(et_map)
    if common:
        earlier = 0
        later = 0
        same = 0
        pnl_diff = []
        time_diff = []

        for f in common:
            p = poll_map[f]
            e = ed_map[f]
            if e["exit_t"] < p["exit_t"]:
                earlier += 1
            elif e["exit_t"] > p["exit_t"]:
                later += 1
            else:
                same += 1
            pnl_diff.append(e["pnl_bps"] - p["pnl_bps"])
            time_diff.append(e["exit_t"] - p["exit_t"])

        print(f"\nEvent-driven vs Polling (N={len(common)} matched settlements):")
        print(f"  Exits EARLIER: {earlier} ({earlier/len(common)*100:.0f}%)")
        print(f"  Exits LATER:   {later} ({later/len(common)*100:.0f}%)")
        print(f"  Exits SAME:    {same} ({same/len(common)*100:.0f}%)")
        print(f"  Avg time diff: {np.mean(time_diff):+.0f}ms (negative = earlier)")
        print(f"  Avg PnL diff:  {np.mean(pnl_diff):+.2f} bps (positive = better)")
        print(f"  Med PnL diff:  {np.median(pnl_diff):+.2f} bps")

    # Trigger distribution
    print(f"\n{'='*70}")
    print(f"TRIGGER DISTRIBUTION (event_driven mode)")
    print(f"{'='*70}")
    from collections import Counter
    triggers = Counter(r["trigger"] for r in results["event_driven"])
    for trig, count in triggers.most_common():
        pnls_t = [r["pnl_bps"] for r in results["event_driven"] if r["trigger"] == trig]
        wr_t = sum(1 for p in pnls_t if p > 0) / len(pnls_t) * 100 if pnls_t else 0
        print(f"  {trig:<12s} {count:3d} ({count/len(results['event_driven'])*100:4.0f}%)  "
              f"Avg PnL={np.mean(pnls_t):+.1f}  WR={wr_t:.0f}%")

    # Also test with HGBC
    print(f"\n{'='*70}")
    print(f"HGBC COMPARISON (same simulation)")
    print(f"{'='*70}")
    for mode in modes:
        data_hg = []
        for fp in files:
            r = simulate_settlement(fp, hgbc, feature_cols, threshold=0.5,
                                    min_exit_ms=1000, mode=mode, cooldown_ms=100)
            if r is not None:
                data_hg.append(r)
        pnls = [r["pnl_bps"] for r in data_hg]
        evals = [r["n_evals"] for r in data_hg]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
        print(f"  {mode:<20s} N={len(data_hg):3d}  PnL={np.mean(pnls):+.1f}  "
              f"WR={wr:.0f}%  Evals={np.mean(evals):.0f}")


if __name__ == "__main__":
    t0 = _time.time()
    main()
    print(f"\nTotal: {_time.time()-t0:.1f}s")
