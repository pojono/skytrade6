#!/usr/bin/env python3
"""
Execution Layer — Order placement, fill tracking, limit+taker fallback.

Supports two modes:
- PAPER: simulates fills from WS price data (no real orders)
- LIVE: places real orders via REST API

Usage:
    executor = Executor(mode="paper")
    await executor.execute_signal(signal, current_price)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import aiohttp

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class OrderResult:
    """Result of an order execution attempt."""
    success: bool
    order_id: str
    symbol: str
    side: str  # "Buy" or "Sell"
    fill_price: float
    fill_qty: float
    notional_usd: float
    fee_usd: float
    is_maker: bool
    timestamp: str
    error: str = ""


@dataclass
class ExecutorConfig:
    """Execution layer configuration."""
    mode: ExecutionMode = ExecutionMode.PAPER
    exchange: str = "bybit"  # which exchange to trade on
    # Bybit API credentials (live mode only)
    api_key: str = ""
    api_secret: str = ""
    # Order parameters
    limit_order_offset_bps: float = 1.0  # place limit X bps better than mid
    limit_order_timeout_s: float = 10.0  # seconds to wait for fill
    taker_fallback: bool = True  # if limit doesn't fill, use market
    # Paper mode assumptions
    paper_slippage_bps: float = 2.0  # simulated slippage
    paper_maker_fill_rate: float = 0.7  # probability limit order fills as maker


class Executor:
    """
    Handles order execution for the strategy.
    
    In PAPER mode: simulates fills with configurable slippage.
    In LIVE mode: places orders on Bybit via REST API.
    """

    def __init__(self, config: ExecutorConfig,
                 on_fill: Optional[Callable] = None):
        self.config = config
        self.on_fill = on_fill  # callback(OrderResult)
        self._session: Optional[aiohttp.ClientSession] = None
        self._pending_orders: dict[str, dict] = {}  # order_id -> order info
        self._order_counter = 0
        self._stats = {
            "orders_placed": 0,
            "fills": 0,
            "maker_fills": 0,
            "taker_fills": 0,
            "errors": 0,
            "total_fee_usd": 0.0,
        }

    async def start(self):
        self._session = aiohttp.ClientSession()
        logger.info(f"Executor started in {self.config.mode.value} mode "
                    f"on {self.config.exchange}")

    async def stop(self):
        if self._session:
            await self._session.close()
        logger.info(f"Executor stopped. Stats: {self._stats}")

    async def execute_signal(self, symbol: str, side: str,
                             notional_usd: float, current_price: float,
                             use_limit: bool = True,
                             timestamp: str = "") -> Optional[OrderResult]:
        """
        Execute a trade signal.
        
        Args:
            symbol: e.g. "SOLUSDT"
            side: "buy" or "sell"
            notional_usd: trade size in USD
            current_price: current mid price
            use_limit: True to try limit order first
            timestamp: ISO timestamp string
        
        Returns:
            OrderResult on fill, None on failure
        """
        if self.config.mode == ExecutionMode.PAPER:
            return await self._paper_fill(symbol, side, notional_usd,
                                          current_price, use_limit, timestamp)
        else:
            return await self._live_fill(symbol, side, notional_usd,
                                         current_price, use_limit, timestamp)

    # ---- Paper Mode ----

    async def _paper_fill(self, symbol: str, side: str, notional_usd: float,
                          current_price: float, use_limit: bool,
                          timestamp: str) -> OrderResult:
        """Simulate a fill with configurable slippage and maker probability."""
        import random

        self._order_counter += 1
        order_id = f"paper_{self._order_counter:06d}"

        # Determine if maker or taker
        is_maker = use_limit and random.random() < self.config.paper_maker_fill_rate

        # Slippage
        if is_maker:
            # Maker: fill at limit price (slight improvement)
            slip_bps = -self.config.limit_order_offset_bps
            fee_rate = 0.0002  # 2 bps maker
        else:
            # Taker: fill with slippage
            slip_bps = self.config.paper_slippage_bps
            fee_rate = 0.0005  # 5 bps taker (one leg)

        # Apply slippage directionally
        if side == "buy":
            fill_price = current_price * (1 + slip_bps / 10000)
        else:
            fill_price = current_price * (1 - slip_bps / 10000)

        qty = notional_usd / fill_price
        fee_usd = notional_usd * fee_rate

        result = OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            fill_price=fill_price,
            fill_qty=qty,
            notional_usd=notional_usd,
            fee_usd=fee_usd,
            is_maker=is_maker,
            timestamp=timestamp,
        )

        self._stats["orders_placed"] += 1
        self._stats["fills"] += 1
        if is_maker:
            self._stats["maker_fills"] += 1
        else:
            self._stats["taker_fills"] += 1
        self._stats["total_fee_usd"] += fee_usd

        logger.info(f"PAPER FILL: {side} {symbol} ${notional_usd:.0f} "
                    f"@ {fill_price:.4f} ({'maker' if is_maker else 'taker'}) "
                    f"fee=${fee_usd:.2f}")

        if self.on_fill:
            self.on_fill(result)

        return result

    # ---- Live Mode (Bybit) ----

    async def _live_fill(self, symbol: str, side: str, notional_usd: float,
                         current_price: float, use_limit: bool,
                         timestamp: str) -> Optional[OrderResult]:
        """Place a real order on Bybit."""
        if not self.config.api_key or not self.config.api_secret:
            logger.error("Live mode requires API credentials")
            return None

        qty = round(notional_usd / current_price, 3)
        bybit_side = "Buy" if side == "buy" else "Sell"

        if use_limit:
            # Try limit order first
            if side == "buy":
                limit_price = current_price * (1 - self.config.limit_order_offset_bps / 10000)
            else:
                limit_price = current_price * (1 + self.config.limit_order_offset_bps / 10000)

            result = await self._bybit_place_order(
                symbol=symbol,
                side=bybit_side,
                qty=qty,
                order_type="Limit",
                price=limit_price,
            )

            if result and result.success:
                return result

            # Wait for fill
            if result:
                filled = await self._wait_for_fill(result.order_id,
                                                   timeout=self.config.limit_order_timeout_s)
                if filled:
                    return filled

                # Cancel unfilled limit and fall back to taker
                if self.config.taker_fallback:
                    await self._bybit_cancel_order(symbol, result.order_id)
                    logger.info(f"Limit unfilled, falling back to market for {symbol}")
                else:
                    await self._bybit_cancel_order(symbol, result.order_id)
                    return None

        # Market order (taker)
        result = await self._bybit_place_order(
            symbol=symbol,
            side=bybit_side,
            qty=qty,
            order_type="Market",
        )

        return result

    async def _bybit_place_order(self, symbol: str, side: str, qty: float,
                                  order_type: str,
                                  price: float = 0) -> Optional[OrderResult]:
        """Place an order on Bybit V5 API."""
        endpoint = "/v5/order/create"
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "PostOnly" if order_type == "Limit" else "GTC",
        }
        if order_type == "Limit" and price > 0:
            params["price"] = str(round(price, 4))

        try:
            resp = await self._bybit_signed_request("POST", endpoint, params)
            if resp and resp.get("retCode") == 0:
                order_id = resp.get("result", {}).get("orderId", "")
                self._stats["orders_placed"] += 1
                logger.info(f"Order placed: {side} {symbol} qty={qty} "
                           f"type={order_type} id={order_id}")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    symbol=symbol,
                    side=side.lower(),
                    fill_price=price if order_type == "Limit" else 0,
                    fill_qty=qty,
                    notional_usd=qty * price if price > 0 else 0,
                    fee_usd=0,
                    is_maker=order_type == "Limit",
                    timestamp=str(time.time()),
                )
            else:
                err = resp.get("retMsg", "unknown") if resp else "no response"
                logger.error(f"Order failed: {err}")
                self._stats["errors"] += 1
                return OrderResult(success=False, order_id="", symbol=symbol,
                                  side=side.lower(), fill_price=0, fill_qty=0,
                                  notional_usd=0, fee_usd=0, is_maker=False,
                                  timestamp="", error=err)
        except Exception as e:
            logger.error(f"Order exception: {e}")
            self._stats["errors"] += 1
            return None

    async def _bybit_cancel_order(self, symbol: str, order_id: str):
        """Cancel an order on Bybit."""
        endpoint = "/v5/order/cancel"
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        try:
            resp = await self._bybit_signed_request("POST", endpoint, params)
            if resp and resp.get("retCode") == 0:
                logger.info(f"Order cancelled: {order_id}")
            else:
                logger.error(f"Cancel failed: {resp}")
        except Exception as e:
            logger.error(f"Cancel exception: {e}")

    async def _wait_for_fill(self, order_id: str, timeout: float) -> Optional[OrderResult]:
        """Poll order status until filled or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(1.0)
            # Check order status
            endpoint = "/v5/order/realtime"
            params = {
                "category": "linear",
                "orderId": order_id,
            }
            try:
                resp = await self._bybit_signed_request("GET", endpoint, params)
                if resp and resp.get("retCode") == 0:
                    orders = resp.get("result", {}).get("list", [])
                    if orders:
                        order = orders[0]
                        if order.get("orderStatus") == "Filled":
                            fill_price = float(order.get("avgPrice", 0))
                            fill_qty = float(order.get("cumExecQty", 0))
                            fee = float(order.get("cumExecFee", 0))
                            self._stats["fills"] += 1
                            self._stats["maker_fills"] += 1
                            self._stats["total_fee_usd"] += abs(fee)
                            return OrderResult(
                                success=True,
                                order_id=order_id,
                                symbol=order.get("symbol", ""),
                                side=order.get("side", "").lower(),
                                fill_price=fill_price,
                                fill_qty=fill_qty,
                                notional_usd=fill_price * fill_qty,
                                fee_usd=abs(fee),
                                is_maker=True,
                                timestamp=str(time.time()),
                            )
            except Exception:
                pass
        return None

    async def _bybit_signed_request(self, method: str, endpoint: str,
                                     params: dict) -> Optional[dict]:
        """Make a signed request to Bybit V5 API."""
        ts = str(int(time.time() * 1000))
        recv_window = "5000"

        if method == "GET":
            query = urllib.parse.urlencode(params)
            sign_str = f"{ts}{self.config.api_key}{recv_window}{query}"
            url = f"https://api.bybit.com{endpoint}?{query}"
            body = None
        else:
            body = json.dumps(params)
            sign_str = f"{ts}{self.config.api_key}{recv_window}{body}"
            url = f"https://api.bybit.com{endpoint}"

        signature = hmac.new(
            self.config.api_secret.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": self.config.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

        async with self._session.request(method, url, headers=headers,
                                          data=body) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                logger.error(f"Bybit API {resp.status}: {text[:200]}")
                return None

    def get_stats(self) -> dict:
        return dict(self._stats)
