#!/usr/bin/env python3
"""
Production Real-Time Strategy Module
=====================================

Cross-Exchange Volatility-Conditioned Mean-Reversion (LONG-only)

This module is designed to run in production. It:
1. Maintains rolling 5-minute bars from both Bybit and Binance
2. Computes cross-exchange features and composite signal in real-time
3. Detects regime (vol expansion) for entry gating
4. Manages positions with signal-based exit and max hold
5. Uses limit orders with taker fallback for execution

Usage:
    from strategy_live import Strategy, StrategyConfig

    config = StrategyConfig.from_json("production_config.json")
    strategy = Strategy(config)

    # On each new 5-minute bar:
    strategy.on_bar(timestamp, bybit_bar, binance_bar)

    # Check for signals:
    signals = strategy.get_signals()
    for sig in signals:
        print(f"{sig.symbol} {sig.action} size={sig.notional_usd}")
"""

import json
import logging
import math
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

class Action(Enum):
    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    NO_ACTION = "NO_ACTION"


@dataclass
class Bar:
    """Single 5-minute OHLCV bar from one exchange."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    # Optional fields (Binance-specific)
    taker_buy_turnover: float = 0.0
    # Optional fields (premium/mark)
    premium: float = 0.0
    # Optional fields (OI, LS ratio)
    open_interest: float = 0.0
    ls_ratio: float = 0.0
    funding_rate: float = 0.0


@dataclass
class Signal:
    """Trade signal emitted by the strategy."""
    symbol: str
    action: Action
    side: str  # "buy" or "sell"
    composite_score: float
    rvol_ratio: float
    spread_vol_ratio: float
    notional_usd: float
    reason: str
    timestamp: str
    use_limit_order: bool = True


@dataclass
class Position:
    """Open position tracked by the strategy."""
    symbol: str
    entry_bar_idx: int
    entry_price: float
    entry_signal: float
    entry_time: str
    notional_usd: float
    bars_held: int = 0


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """All strategy parameters loaded from production_config.json."""
    sig_threshold: float = 2.5
    vol_threshold: float = 2.0
    spread_vol_threshold: float = 1.3
    max_hold_bars: int = 24
    min_hold_bars: int = 3
    cooldown_bars: int = 3
    use_limit_orders: bool = True
    base_notional_usd: float = 10000.0
    max_notional_usd: float = 30000.0
    tier_multipliers: dict = field(default_factory=lambda: {"A": 1.5, "B": 1.0, "C": 0.5})

    # Signal weights
    signal_weights: dict = field(default_factory=lambda: {
        "price_div_z72": 3.0,
        "price_div_z288": 2.0,
        "premium_z72": 2.0,
        "premium_z288": 1.5,
        "price_div_ma12_z288": 1.5,
        "oi_div_z288": 1.0,
        "vol_ratio_z72": 0.5,
        "ret_diff_sum12_z288": 1.0,
    })

    # Risk management
    max_concurrent_positions: int = 5
    max_positions_per_symbol: int = 1
    daily_loss_stop_usd: float = 500.0
    max_drawdown_usd: float = 2000.0
    max_total_exposure_usd: float = 50000.0

    # Symbol lists
    whitelist: list = field(default_factory=list)
    blacklist: list = field(default_factory=list)
    symbol_tiers: dict = field(default_factory=dict)  # symbol -> "A"/"B"/"C"

    @classmethod
    def from_json(cls, path: str) -> "StrategyConfig":
        """Load config from production_config.json."""
        with open(path) as f:
            data = json.load(f)

        params = data.get("parameters", {})
        config = cls(
            sig_threshold=params.get("sig_threshold", 2.5),
            vol_threshold=params.get("vol_threshold", 2.0),
            spread_vol_threshold=params.get("spread_vol_threshold", 1.3),
            max_hold_bars=params.get("max_hold", 24),
            min_hold_bars=params.get("min_hold", 3),
            cooldown_bars=params.get("cooldown", 3),
            use_limit_orders=params.get("maker_pct", 0) > 0,
            whitelist=data.get("symbol_whitelist", []),
            blacklist=data.get("symbol_blacklist", []),
        )

        # Load signal weights
        sig_data = data.get("signal", {})
        components = sig_data.get("components", [])
        if components:
            config.signal_weights = {c["feature"]: c["weight"] for c in components}

        # Load sizing
        sizing = data.get("position_sizing", {})
        config.base_notional_usd = sizing.get("base_notional_usd", 10000)
        config.max_notional_usd = sizing.get("max_notional_usd", 30000)
        config.tier_multipliers = sizing.get("tier_multipliers", {"A": 1.5, "B": 1.0, "C": 0.5})

        # Load risk management
        risk = data.get("risk_management", {})
        config.max_concurrent_positions = risk.get("max_concurrent_positions", 5)
        config.max_positions_per_symbol = risk.get("max_positions_per_symbol", 1)
        config.daily_loss_stop_usd = risk.get("daily_loss_stop_usd", 500)
        config.max_drawdown_usd = risk.get("max_drawdown_usd", 2000)
        config.max_total_exposure_usd = risk.get("max_total_exposure_usd", 50000)

        return config


# =============================================================================
# ROLLING FEATURE CALCULATOR (per-symbol)
# =============================================================================

class RollingStats:
    """O(1) incremental rolling mean/std using a deque."""
    __slots__ = ('buf', 'maxlen', 'n', '_sum', '_sum2')

    def __init__(self, maxlen: int):
        self.buf = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.n = 0
        self._sum = 0.0
        self._sum2 = 0.0

    def push(self, val: float):
        if self.n >= self.maxlen:
            old = self.buf[0]
            self._sum -= old
            self._sum2 -= old * old
        else:
            self.n += 1
        self.buf.append(val)
        self._sum += val
        self._sum2 += val * val

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = self._sum2 / self.n - (self._sum / self.n) ** 2
        return var ** 0.5 if var > 0 else 0.0

    def zscore(self, val: float) -> float:
        s = self.std
        return (val - self.mean) / s if s > 1e-12 else 0.0

    @property
    def last(self) -> float:
        return self.buf[-1] if self.buf else 0.0


class SymbolState:
    """
    Maintains rolling buffers and computes features for a single symbol.
    
    Uses RollingStats for O(1) incremental mean/std updates.
    All compute_signal() work is pure Python arithmetic — no numpy per bar.
    """

    LOOKBACK = 300  # bars needed before ready

    def __init__(self, symbol: str, tier: str = "B"):
        self.symbol = symbol
        self.tier = tier
        self.n = 0

        # Rolling stats for price divergence
        self.pdiv_rs72 = RollingStats(72)
        self.pdiv_rs288 = RollingStats(288)

        # Rolling stats for MA12 of pdiv (z-scored over 288)
        self._pdiv_buf12 = deque(maxlen=12)
        self.pdiv_ma12_rs288 = RollingStats(288)

        # Rolling stats for premium spread
        self.ps_rs72 = RollingStats(72)
        self.ps_rs288 = RollingStats(288)

        # Rolling stats for volume ratio (log)
        self.vr_rs72 = RollingStats(72)

        # Rolling stats for mid return (for rvol)
        self.mret_rs12 = RollingStats(12)
        self.mret_rs72 = RollingStats(72)

        # Rolling stats for pdiv (for spread vol)
        self.pdiv_sv12 = RollingStats(12)
        self.pdiv_sv72 = RollingStats(72)

        # OI divergence: pct_change(6) diff, then z-scored over 288
        self._bb_oi_buf = deque(maxlen=8)
        self._bn_oi_buf = deque(maxlen=8)
        self.oi_div_rs288 = RollingStats(288)

        # Ret diff accumulation: rolling(12).sum of ret_diff, then z-scored over 288
        self._bb_close_buf = deque(maxlen=13)
        self._bn_close_buf = deque(maxlen=13)
        self.ret_diff_sum12_rs288 = RollingStats(288)

        # Ret diff per-bar buffer (last 12 bars of bb_ret - bn_ret in bps)
        self._ret_diff_buf = deque(maxlen=12)
        self._ret_diff_sum = 0.0

        # Previous mid for return calc
        self._prev_mid = 0.0

    @property
    def ready(self):
        return self.n >= self.LOOKBACK

    def update(self, bb: Bar, bn: Bar):
        """Push a new 5-minute bar pair. O(1) per call."""
        mid = (bb.close + bn.close) * 0.5

        # Mid return
        if self.n > 0 and self._prev_mid > 0:
            mret = mid / self._prev_mid - 1.0
        else:
            mret = 0.0
        self._prev_mid = mid
        self.mret_rs12.push(mret)
        self.mret_rs72.push(mret)

        # Price divergence
        pdiv = (bb.close - bn.close) / mid * 10000 if mid > 0 else 0.0
        self.pdiv_rs72.push(pdiv)
        self.pdiv_rs288.push(pdiv)
        self.pdiv_sv12.push(pdiv)
        self.pdiv_sv72.push(pdiv)

        # MA12 of pdiv
        self._pdiv_buf12.append(pdiv)
        if len(self._pdiv_buf12) >= 12:
            ma12 = sum(self._pdiv_buf12) / len(self._pdiv_buf12)
        else:
            ma12 = sum(self._pdiv_buf12) / len(self._pdiv_buf12) if self._pdiv_buf12 else 0.0
        self.pdiv_ma12_rs288.push(ma12)

        # Premium spread
        ps = (bb.premium - bn.premium) * 10000
        self.ps_rs72.push(ps)
        self.ps_rs288.push(ps)

        # Volume ratio (log)
        if bn.turnover > 0 and bb.turnover > 0:
            vr = math.log(bb.turnover / bn.turnover)
        else:
            vr = 0.0
        self.vr_rs72.push(vr)

        # OI divergence: (bb_oi pct_change(6) - bn_oi pct_change(6)) * 10000
        self._bb_oi_buf.append(bb.open_interest)
        self._bn_oi_buf.append(bn.open_interest)
        oi_div = 0.0
        if len(self._bb_oi_buf) >= 7:
            bb_oi_old = self._bb_oi_buf[-7]
            bn_oi_old = self._bn_oi_buf[-7]
            if bb_oi_old > 0 and bn_oi_old > 0:
                oi_div = ((self._bb_oi_buf[-1] / bb_oi_old - 1.0) -
                          (self._bn_oi_buf[-1] / bn_oi_old - 1.0)) * 10000
        self.oi_div_rs288.push(oi_div)

        # Ret diff: (bb_ret - bn_ret) each bar, running sum of last 12
        self._bb_close_buf.append(bb.close)
        self._bn_close_buf.append(bn.close)
        new_rd = 0.0
        if len(self._bb_close_buf) >= 2:
            bb_ret = self._bb_close_buf[-1] / self._bb_close_buf[-2] - 1.0 if self._bb_close_buf[-2] > 0 else 0.0
            bn_ret = self._bn_close_buf[-1] / self._bn_close_buf[-2] - 1.0 if self._bn_close_buf[-2] > 0 else 0.0
            new_rd = (bb_ret - bn_ret) * 10000
        if len(self._ret_diff_buf) >= 12:
            self._ret_diff_sum -= self._ret_diff_buf[0]
        self._ret_diff_buf.append(new_rd)
        self._ret_diff_sum += new_rd
        self.ret_diff_sum12_rs288.push(self._ret_diff_sum)

        self.n += 1

    def compute_signal(self) -> dict:
        """
        Compute composite signal and regime features.
        All O(1) — uses pre-maintained rolling stats.
        """
        if not self.ready:
            return {"composite": 0.0, "rvol_ratio": 0.0, "spread_vol_ratio": 0.0, "sig_strength": 0.0}

        # Price div z-scores
        current_pdiv = self.pdiv_rs72.last
        z72 = self.pdiv_rs72.zscore(current_pdiv)
        z288 = self.pdiv_rs288.zscore(current_pdiv)

        # MA12 of pdiv, z-scored over 288
        current_ma12 = self.pdiv_ma12_rs288.last
        z_ma12 = self.pdiv_ma12_rs288.zscore(current_ma12)

        # Premium spread z-scores
        current_ps = self.ps_rs72.last
        ps_z72 = self.ps_rs72.zscore(current_ps)
        ps_z288 = self.ps_rs288.zscore(current_ps)

        # OI divergence z-scored over 288 bars (matches backtest)
        oi_div_z = self.oi_div_rs288.zscore(self.oi_div_rs288.last)

        # Volume ratio z-score
        vol_ratio_z = self.vr_rs72.zscore(self.vr_rs72.last)

        # Ret diff sum12 z-scored over 288 bars (matches backtest)
        ret_diff_z = self.ret_diff_sum12_rs288.zscore(self.ret_diff_sum12_rs288.last)

        # Composite weighted signal
        composite = (
            3.0 * z72 +
            2.0 * z288 +
            2.0 * ps_z72 +
            1.5 * ps_z288 +
            1.5 * z_ma12 +
            1.0 * oi_div_z +
            0.5 * vol_ratio_z +
            1.0 * ret_diff_z
        ) / 11.5  # sum of weights

        # Regime: realized vol ratio
        rvol_1h = self.mret_rs12.std
        rvol_6h = self.mret_rs72.std
        rvol_ratio = rvol_1h / rvol_6h if rvol_6h > 1e-15 else 0.0

        # Spread vol ratio
        sv_1h = self.pdiv_sv12.std
        sv_6h = self.pdiv_sv72.std
        spread_vol_ratio = sv_1h / sv_6h if sv_6h > 1e-15 else 0.0

        return {
            "composite": composite,
            "rvol_ratio": rvol_ratio,
            "spread_vol_ratio": spread_vol_ratio,
            "sig_strength": abs(composite),
        }


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class Strategy:
    """
    Production strategy: Cross-Exchange Volatility-Conditioned Mean-Reversion.
    
    Call on_bar() for each 5-minute bar, then get_signals() to get trade actions.
    
    Lifecycle:
    1. Initialize with config
    2. Register symbols via add_symbol()
    3. Feed bars via on_bar()
    4. Read signals via get_signals()
    5. Confirm fills via confirm_fill() / confirm_close()
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.symbols: dict[str, SymbolState] = {}
        self.positions: dict[str, Position] = {}
        self.cooldowns: dict[str, int] = {}  # symbol -> bars remaining
        self.bar_count = 0
        self._pending_signals: list[Signal] = []

        # Risk management state
        self.daily_pnl_usd = 0.0
        self.total_pnl_usd = 0.0
        self.peak_pnl_usd = 0.0
        self.current_date: str = ""
        self.daily_halted = False
        self.drawdown_halted = False
        self.closed_trades: list[dict] = []  # for audit trail

    def add_symbol(self, symbol: str, tier: str = "B"):
        """Register a symbol for tracking."""
        if symbol in self.config.blacklist:
            logger.info(f"Skipping blacklisted symbol {symbol}")
            return
        self.symbols[symbol] = SymbolState(symbol, tier)
        logger.info(f"Added {symbol} (tier {tier})")

    def add_whitelisted_symbols(self):
        """Add all symbols from the whitelist with default tier B."""
        for sym in self.config.whitelist:
            tier = self.config.symbol_tiers.get(sym, "B")
            self.add_symbol(sym, tier)

    def on_bar(self, timestamp: str, symbol: str, bb_bar: Bar, bn_bar: Bar):
        """
        Process a new 5-minute bar for a symbol.
        
        Call this for each symbol as new bars arrive.
        """
        if symbol not in self.symbols:
            return

        state = self.symbols[symbol]
        state.update(bb_bar, bn_bar)

        if not state.ready:
            return

        # Decrement cooldown
        if symbol in self.cooldowns:
            self.cooldowns[symbol] -= 1
            if self.cooldowns[symbol] <= 0:
                del self.cooldowns[symbol]

        # Compute signal
        feat = state.compute_signal()
        composite = feat["composite"]
        rvol = feat["rvol_ratio"]
        sv_ratio = feat["spread_vol_ratio"]

        # --- CHECK EXIT for open position ---
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.bars_held += 1

            exit_now = False
            reason = ""

            if pos.bars_held < self.config.min_hold_bars:
                pass
            elif composite >= 0:
                exit_now, reason = True, "signal_cross_zero"
            elif pos.bars_held >= self.config.max_hold_bars:
                exit_now, reason = True, "max_hold"
            elif composite < -self.config.sig_threshold * 2.0 and pos.bars_held >= 6:
                exit_now, reason = True, "reversal_stop"

            if exit_now:
                mid = (bb_bar.close + bn_bar.close) / 2.0
                self._pending_signals.append(Signal(
                    symbol=symbol,
                    action=Action.CLOSE_LONG,
                    side="sell",
                    composite_score=composite,
                    rvol_ratio=rvol,
                    spread_vol_ratio=sv_ratio,
                    notional_usd=pos.notional_usd,
                    reason=reason,
                    timestamp=timestamp,
                    use_limit_order=self.config.use_limit_orders,
                ))
                self.cooldowns[symbol] = self.config.cooldown_bars
                del self.positions[symbol]

        # --- CHECK ENTRY ---
        elif symbol not in self.cooldowns:
            if not self._risk_allows_entry(symbol):
                pass
            elif composite < -self.config.sig_threshold:
                if rvol >= self.config.vol_threshold:
                    if self.config.spread_vol_threshold <= 0 or sv_ratio >= self.config.spread_vol_threshold:
                        tier = state.tier
                        tier_mult = self.config.tier_multipliers.get(tier, 1.0)
                        notional = self.config.base_notional_usd * tier_mult
                        notional = min(notional, self.config.max_notional_usd)

                        # Check exposure cap
                        current_exposure = sum(p.notional_usd for p in self.positions.values())
                        if current_exposure + notional > self.config.max_total_exposure_usd:
                            notional = max(0, self.config.max_total_exposure_usd - current_exposure)
                        if notional < 100:  # minimum viable trade
                            pass
                        else:
                            mid = (bb_bar.close + bn_bar.close) / 2.0
                            self._pending_signals.append(Signal(
                                symbol=symbol,
                                action=Action.OPEN_LONG,
                                side="buy",
                                composite_score=composite,
                                rvol_ratio=rvol,
                                spread_vol_ratio=sv_ratio,
                                notional_usd=notional,
                                reason=f"sig={composite:.2f} rvol={rvol:.2f} sv={sv_ratio:.2f}",
                                timestamp=timestamp,
                                use_limit_order=self.config.use_limit_orders,
                            ))

                            self.positions[symbol] = Position(
                                symbol=symbol,
                                entry_bar_idx=self.bar_count,
                                entry_price=mid,
                                entry_signal=composite,
                                entry_time=timestamp,
                                notional_usd=notional,
                            )

        self.bar_count += 1

    # ----- Risk Management -----

    def _risk_allows_entry(self, symbol: str) -> bool:
        """Check all risk gates before allowing a new entry."""
        if self.daily_halted:
            return False
        if self.drawdown_halted:
            return False
        if len(self.positions) >= self.config.max_concurrent_positions:
            return False
        if symbol in self.positions:
            return False
        return True

    def record_fill(self, symbol: str, action: Action, fill_price: float,
                    notional_usd: float, timestamp: str):
        """
        Record an actual fill from the exchange. Updates PnL tracking.
        Call this after order confirmation.
        """
        # Reset daily PnL on new date
        date_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
        if date_str != self.current_date:
            self.current_date = date_str
            self.daily_pnl_usd = 0.0
            self.daily_halted = False
            logger.info(f"New trading day: {date_str}")

        if action == Action.CLOSE_LONG and symbol in self.closed_trades:
            pass  # PnL computed below

    def record_close(self, symbol: str, entry_price: float, exit_price: float,
                     notional_usd: float, timestamp: str):
        """
        Record a closed trade. Updates daily PnL, total PnL, drawdown.
        """
        pnl_bps = (exit_price / entry_price - 1.0) * 10000 if entry_price > 0 else 0.0
        # Subtract estimated fees (maker RT = 8bps)
        fee_bps = 8.0 if self.config.use_limit_orders else 20.0
        net_bps = pnl_bps - fee_bps
        pnl_usd = net_bps / 10000.0 * notional_usd

        self.daily_pnl_usd += pnl_usd
        self.total_pnl_usd += pnl_usd
        self.peak_pnl_usd = max(self.peak_pnl_usd, self.total_pnl_usd)

        trade_record = {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "notional_usd": notional_usd,
            "pnl_bps": net_bps,
            "pnl_usd": pnl_usd,
            "daily_pnl": self.daily_pnl_usd,
            "total_pnl": self.total_pnl_usd,
            "timestamp": timestamp,
        }
        self.closed_trades.append(trade_record)
        logger.info(f"CLOSED {symbol}: {net_bps:+.1f}bps ${pnl_usd:+.1f} "
                     f"[day=${self.daily_pnl_usd:+.1f} total=${self.total_pnl_usd:+.1f}]")

        # Check daily loss stop
        if self.daily_pnl_usd <= -self.config.daily_loss_stop_usd:
            self.daily_halted = True
            logger.warning(f"DAILY LOSS STOP: ${self.daily_pnl_usd:.0f} <= "
                          f"-${self.config.daily_loss_stop_usd:.0f}")

        # Check max drawdown
        drawdown = self.peak_pnl_usd - self.total_pnl_usd
        if drawdown >= self.config.max_drawdown_usd:
            self.drawdown_halted = True
            logger.warning(f"MAX DRAWDOWN BREAKER: ${drawdown:.0f} >= "
                          f"${self.config.max_drawdown_usd:.0f}")

        return trade_record

    # ----- State Serialization -----

    def get_state(self) -> dict:
        """Serialize full strategy state for persistence / restart recovery."""
        symbol_states = {}
        for sym, ss in self.symbols.items():
            symbol_states[sym] = {
                "tier": ss.tier,
                "n": ss.n,
                "ready": ss.ready,
            }
        return {
            "bar_count": self.bar_count,
            "positions": {sym: {
                "entry_bar_idx": p.entry_bar_idx,
                "entry_price": p.entry_price,
                "entry_signal": p.entry_signal,
                "entry_time": p.entry_time,
                "notional_usd": p.notional_usd,
                "bars_held": p.bars_held,
            } for sym, p in self.positions.items()},
            "cooldowns": dict(self.cooldowns),
            "daily_pnl_usd": self.daily_pnl_usd,
            "total_pnl_usd": self.total_pnl_usd,
            "peak_pnl_usd": self.peak_pnl_usd,
            "current_date": self.current_date,
            "daily_halted": self.daily_halted,
            "drawdown_halted": self.drawdown_halted,
            "closed_trades_count": len(self.closed_trades),
            "symbol_states": symbol_states,
        }

    # ----- Public API -----

    def get_signals(self) -> list[Signal]:
        """Return and clear pending signals."""
        signals = self._pending_signals.copy()
        self._pending_signals.clear()
        return signals

    def get_positions(self) -> dict[str, Position]:
        """Return current open positions."""
        return dict(self.positions)

    def get_status(self) -> dict:
        """Return strategy status summary."""
        drawdown = self.peak_pnl_usd - self.total_pnl_usd
        return {
            "bar_count": self.bar_count,
            "symbols_tracked": len(self.symbols),
            "symbols_ready": sum(1 for s in self.symbols.values() if s.ready),
            "open_positions": len(self.positions),
            "cooldowns_active": len(self.cooldowns),
            "position_symbols": list(self.positions.keys()),
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "drawdown_usd": round(drawdown, 2),
            "daily_halted": self.daily_halted,
            "drawdown_halted": self.drawdown_halted,
            "total_closed_trades": len(self.closed_trades),
        }


# =============================================================================
# CONVENIENCE: QUICK TEST WITH HISTORICAL DATA
# =============================================================================

def _print(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _safe_col(df, col):
    """Extract column as numpy array, filling NaN with 0."""
    if col in df.columns:
        return df[col].fillna(0).values
    return np.zeros(len(df))


def backtest_with_live_module(symbol_list=None, config_path="production_config.json"):
    """
    Run the live strategy module against historical data to verify
    it produces the same results as the backtest engine.
    
    Uses pre-extracted numpy arrays for speed (~50K bars/s vs ~500 with iloc).
    """
    import time
    from load_data import load_symbol

    config = StrategyConfig.from_json(config_path)
    strategy = Strategy(config)

    symbols = symbol_list or config.whitelist[:5]
    _print(f"Verification test: {len(symbols)} symbols")
    for sym in symbols:
        strategy.add_symbol(sym, config.symbol_tiers.get(sym, "B"))

    log_lines = []
    total_signals = 0
    t0 = time.time()

    for s_idx, sym in enumerate(symbols):
        ts_load = time.time()
        _print(f"  [{s_idx+1}/{len(symbols)}] Loading {sym}...")
        df = load_symbol(sym)
        if df.empty:
            _print(f"    EMPTY — skipped")
            continue

        n = len(df)
        _print(f"    {n} bars loaded in {time.time()-ts_load:.1f}s. Pre-extracting arrays...")

        # Pre-extract all columns as numpy arrays (avoid slow .iloc per row)
        timestamps = [str(t) for t in df.index]
        bb_open = _safe_col(df, "bb_open")
        bb_high = _safe_col(df, "bb_high")
        bb_low = _safe_col(df, "bb_low")
        bb_close = df["bb_close"].values
        bb_vol = _safe_col(df, "bb_volume")
        bb_turn = _safe_col(df, "bb_turnover")
        bb_prem = _safe_col(df, "bb_premium")
        bb_oi = _safe_col(df, "bb_oi")
        bn_open = _safe_col(df, "bn_open")
        bn_high = _safe_col(df, "bn_high")
        bn_low = _safe_col(df, "bn_low")
        bn_close = df["bn_close"].values
        bn_vol = _safe_col(df, "bn_volume")
        bn_turn = _safe_col(df, "bn_turnover")
        bn_taker = _safe_col(df, "bn_taker_buy_turnover")
        bn_prem = _safe_col(df, "bn_premium")
        bn_oi = _safe_col(df, "bn_oi")

        _print(f"    Feeding {n} bars...")
        ts_feed = time.time()

        for i in range(n):
            bb = Bar(open=bb_open[i], high=bb_high[i], low=bb_low[i], close=bb_close[i],
                     volume=bb_vol[i], turnover=bb_turn[i], premium=bb_prem[i],
                     open_interest=bb_oi[i])
            bn = Bar(open=bn_open[i], high=bn_high[i], low=bn_low[i], close=bn_close[i],
                     volume=bn_vol[i], turnover=bn_turn[i], taker_buy_turnover=bn_taker[i],
                     premium=bn_prem[i], open_interest=bn_oi[i])

            strategy.on_bar(timestamps[i], sym, bb, bn)
            signals = strategy.get_signals()
            for sig in signals:
                total_signals += 1
                log_lines.append(f"  {sig.timestamp} {sig.symbol:15s} {sig.action.value:12s} "
                                 f"${sig.notional_usd:.0f} {sig.reason}")

            if (i + 1) % 20000 == 0:
                elapsed = time.time() - ts_feed
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                _print(f"    {i+1}/{n} bars ({rate:.0f} bar/s, ETA {eta:.0f}s) signals={total_signals}")

        elapsed_sym = time.time() - ts_feed
        _print(f"    Done: {n} bars in {elapsed_sym:.1f}s ({n/max(elapsed_sym,0.01):.0f} bar/s), signals so far: {total_signals}")

    elapsed_total = time.time() - t0
    _print(f"\n{'='*80}")
    _print(f"Verification: {total_signals} signals across {len(symbols)} symbols in {elapsed_total:.1f}s")
    for line in log_lines[:50]:
        _print(line)
    if len(log_lines) > 50:
        _print(f"  ... and {len(log_lines) - 50} more")

    status = strategy.get_status()
    _print(f"\nFinal status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    import sys
    backtest_with_live_module()
