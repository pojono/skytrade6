#!/usr/bin/env python3
"""
WebSocket Data Feed — Real-time kline + OI + premium streams from Bybit & Binance.

Aggregates 1m klines into 5m bars and delivers them to the strategy engine.

Usage:
    feed = DataFeed(symbols=["SOLUSDT", "BTCUSDT"], on_bar=my_callback)
    await feed.start()
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Bybit WS: wss://stream.bybit.com/v5/public/linear
# Binance WS: wss://fstream.binance.com/ws

BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
BINANCE_WS = "wss://fstream.binance.com/ws"
BINANCE_REST = "https://fapi.binance.com"
BYBIT_REST = "https://api.bybit.com"


@dataclass
class Kline1m:
    """Single 1-minute kline from one exchange."""
    timestamp_ms: int  # bar open time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirmed: bool  # True if bar is closed


@dataclass
class BarAccumulator:
    """Accumulates 1m klines into a 5m bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    taker_buy_turnover: float = 0.0
    count: int = 0
    bar_open_ts: int = 0  # 5m bar open timestamp ms

    def reset(self, ts_ms: int):
        self.open = 0.0
        self.high = 0.0
        self.low = float('inf')
        self.close = 0.0
        self.volume = 0.0
        self.turnover = 0.0
        self.taker_buy_turnover = 0.0
        self.count = 0
        self.bar_open_ts = ts_ms

    def add(self, k: Kline1m, taker_buy_turnover: float = 0.0):
        if self.count == 0:
            self.open = k.open
        self.high = max(self.high, k.high) if self.high != float('inf') else k.high
        self.low = min(self.low, k.low)
        self.close = k.close
        self.volume += k.volume
        self.turnover += k.turnover
        self.taker_buy_turnover += taker_buy_turnover
        self.count += 1

    @property
    def ready(self) -> bool:
        return self.count >= 5


@dataclass
class SymbolFeedState:
    """Per-symbol state for bar aggregation."""
    bb_acc: BarAccumulator = field(default_factory=BarAccumulator)
    bn_acc: BarAccumulator = field(default_factory=BarAccumulator)
    bb_premium: float = 0.0
    bn_premium: float = 0.0
    bb_oi: float = 0.0
    bn_oi: float = 0.0
    last_bar_ts: int = 0


# Type for the callback: on_bar(symbol, timestamp_iso, bb_bar, bn_bar)
BarCallback = Callable


def _align_5m(ts_ms: int) -> int:
    """Align a timestamp to the 5-minute boundary (floor)."""
    return ts_ms - (ts_ms % (5 * 60 * 1000))


class DataFeed:
    """
    Manages WebSocket connections to Bybit and Binance.
    Aggregates 1m klines into 5m bars and delivers via callback.
    """

    def __init__(self, symbols: list[str], on_bar: BarCallback,
                 paper_mode: bool = True):
        self.symbols = symbols
        self.on_bar = on_bar
        self.paper_mode = paper_mode
        self._states: dict[str, SymbolFeedState] = {s: SymbolFeedState() for s in symbols}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            "bb_messages": 0,
            "bn_messages": 0,
            "bars_emitted": 0,
            "errors": 0,
            "last_bb_msg": 0.0,
            "last_bn_msg": 0.0,
        }

    async def start(self):
        """Start all WebSocket connections."""
        self._running = True
        self._session = aiohttp.ClientSession()
        logger.info(f"DataFeed starting for {len(self.symbols)} symbols")

        # Fetch initial OI + premium via REST
        await self._fetch_initial_state()

        # Start WS tasks
        self._tasks = [
            asyncio.create_task(self._run_bybit_ws()),
            asyncio.create_task(self._run_binance_ws()),
            asyncio.create_task(self._run_bybit_oi_poll()),
            asyncio.create_task(self._run_binance_oi_poll()),
            asyncio.create_task(self._heartbeat()),
        ]
        logger.info("DataFeed all tasks started")

    async def stop(self):
        """Gracefully stop all connections."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._session:
            await self._session.close()
        logger.info("DataFeed stopped")

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ---- Initial state fetch via REST ----

    async def _fetch_initial_state(self):
        """Fetch current OI and premium for all symbols via REST."""
        try:
            # Bybit OI
            for sym in self.symbols:
                url = f"{BYBIT_REST}/v5/market/open-interest?category=linear&symbol={sym}&intervalTime=5min&limit=1"
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("result", {}).get("list", [])
                        if items:
                            self._states[sym].bb_oi = float(items[0].get("openInterest", 0))
                await asyncio.sleep(0.05)  # rate limit

            # Binance OI
            for sym in self.symbols:
                url = f"{BINANCE_REST}/fapi/v1/openInterest?symbol={sym}"
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._states[sym].bn_oi = float(data.get("openInterest", 0))
                await asyncio.sleep(0.05)

            logger.info("Initial OI fetched for all symbols")
        except Exception as e:
            logger.error(f"Failed to fetch initial state: {e}")

    # ---- Bybit WebSocket ----

    async def _run_bybit_ws(self):
        """Connect to Bybit and subscribe to kline + premium streams."""
        while self._running:
            try:
                async with self._session.ws_connect(BYBIT_WS, heartbeat=20) as ws:
                    # Subscribe to 1m klines
                    kline_topics = [f"kline.1.{sym}" for sym in self.symbols]
                    # Subscribe to premium index klines
                    premium_topics = [f"kline.1.{sym}" for sym in self.symbols]

                    # Bybit limits 10 topics per subscribe message
                    for i in range(0, len(kline_topics), 10):
                        batch = kline_topics[i:i+10]
                        await ws.send_json({"op": "subscribe", "args": batch})

                    logger.info(f"Bybit WS subscribed to {len(kline_topics)} kline topics")

                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_bybit_msg(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

            except asyncio.CancelledError:
                return
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Bybit WS error: {e}")
                await asyncio.sleep(5)

    def _handle_bybit_msg(self, data: dict):
        """Parse Bybit kline message and update accumulators."""
        topic = data.get("topic", "")
        if not topic.startswith("kline.1."):
            return

        symbol = topic.split(".")[-1]
        if symbol not in self._states:
            return

        kline_data = data.get("data", [])
        if not kline_data:
            return

        self._stats["bb_messages"] += 1
        self._stats["last_bb_msg"] = time.time()

        for kd in kline_data:
            k = Kline1m(
                timestamp_ms=int(kd["start"]),
                open=float(kd["open"]),
                high=float(kd["high"]),
                low=float(kd["low"]),
                close=float(kd["close"]),
                volume=float(kd["volume"]),
                turnover=float(kd["turnover"]),
                confirmed=kd.get("confirm", False),
            )

            if k.confirmed:
                self._add_bybit_kline(symbol, k)

    def _add_bybit_kline(self, symbol: str, k: Kline1m):
        """Add a confirmed 1m kline to the Bybit accumulator."""
        state = self._states[symbol]
        bar_ts = _align_5m(k.timestamp_ms)

        if state.bb_acc.bar_open_ts != bar_ts:
            state.bb_acc.reset(bar_ts)

        state.bb_acc.add(k)
        self._try_emit_bar(symbol, bar_ts)

    # ---- Binance WebSocket ----

    async def _run_binance_ws(self):
        """Connect to Binance and subscribe to kline streams."""
        while self._running:
            try:
                # Binance combined stream
                streams = "/".join(f"{sym.lower()}@kline_1m" for sym in self.symbols)
                url = f"{BINANCE_WS}/{streams}"

                async with self._session.ws_connect(url, heartbeat=20) as ws:
                    logger.info(f"Binance WS connected ({len(self.symbols)} streams)")

                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_binance_msg(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

            except asyncio.CancelledError:
                return
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Binance WS error: {e}")
                await asyncio.sleep(5)

    def _handle_binance_msg(self, data: dict):
        """Parse Binance kline message and update accumulators."""
        if "e" not in data or data["e"] != "kline":
            # Combined stream wraps in {"stream": ..., "data": {...}}
            data = data.get("data", data)
            if data.get("e") != "kline":
                return

        kline = data.get("k", {})
        symbol = kline.get("s", "")
        if symbol not in self._states:
            return

        self._stats["bn_messages"] += 1
        self._stats["last_bn_msg"] = time.time()

        is_closed = kline.get("x", False)
        if not is_closed:
            return

        k = Kline1m(
            timestamp_ms=int(kline["t"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            turnover=float(kline["q"]),
            confirmed=True,
        )
        taker_buy_turnover = float(kline.get("Q", 0))

        state = self._states[symbol]
        bar_ts = _align_5m(k.timestamp_ms)

        if state.bn_acc.bar_open_ts != bar_ts:
            state.bn_acc.reset(bar_ts)

        state.bn_acc.add(k, taker_buy_turnover)
        self._try_emit_bar(symbol, bar_ts)

    # ---- Bar emission ----

    def _try_emit_bar(self, symbol: str, bar_ts: int):
        """If both exchanges have 5 confirmed 1m klines, emit a 5m bar."""
        state = self._states[symbol]

        if not (state.bb_acc.ready and state.bn_acc.ready):
            return
        if state.bb_acc.bar_open_ts != bar_ts or state.bn_acc.bar_open_ts != bar_ts:
            return
        if bar_ts <= state.last_bar_ts:
            return  # already emitted

        state.last_bar_ts = bar_ts

        from strategy_live import Bar
        import datetime

        ts_iso = datetime.datetime.utcfromtimestamp(bar_ts / 1000).strftime("%Y-%m-%d %H:%M:%S")

        bb_bar = Bar(
            open=state.bb_acc.open,
            high=state.bb_acc.high,
            low=state.bb_acc.low,
            close=state.bb_acc.close,
            volume=state.bb_acc.volume,
            turnover=state.bb_acc.turnover,
            premium=state.bb_premium,
            open_interest=state.bb_oi,
        )
        bn_bar = Bar(
            open=state.bn_acc.open,
            high=state.bn_acc.high,
            low=state.bn_acc.low,
            close=state.bn_acc.close,
            volume=state.bn_acc.volume,
            turnover=state.bn_acc.turnover,
            taker_buy_turnover=state.bn_acc.taker_buy_turnover,
            premium=state.bn_premium,
            open_interest=state.bn_oi,
        )

        self._stats["bars_emitted"] += 1
        self.on_bar(symbol, ts_iso, bb_bar, bn_bar)

    # ---- OI polling (REST, every 5 minutes) ----

    async def _run_bybit_oi_poll(self):
        """Poll Bybit OI every 5 minutes."""
        while self._running:
            try:
                for sym in self.symbols:
                    url = f"{BYBIT_REST}/v5/market/open-interest?category=linear&symbol={sym}&intervalTime=5min&limit=1"
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            items = data.get("result", {}).get("list", [])
                            if items:
                                self._states[sym].bb_oi = float(items[0].get("openInterest", 0))
                    await asyncio.sleep(0.1)  # rate limit
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Bybit OI poll error: {e}")

            await asyncio.sleep(300)  # 5 minutes

    async def _run_binance_oi_poll(self):
        """Poll Binance OI every 5 minutes."""
        while self._running:
            try:
                for sym in self.symbols:
                    url = f"{BINANCE_REST}/fapi/v1/openInterest?symbol={sym}"
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._states[sym].bn_oi = float(data.get("openInterest", 0))
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Binance OI poll error: {e}")

            await asyncio.sleep(300)

    # ---- Premium polling (REST, every minute) ----
    # Premium index is not available via WS on all exchanges, so we poll

    async def _run_premium_poll(self):
        """Poll premium index every minute."""
        while self._running:
            try:
                # Bybit premium
                for sym in self.symbols:
                    url = f"{BYBIT_REST}/v5/market/premium-index-price-kline?category=linear&symbol={sym}&interval=1&limit=1"
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            items = data.get("result", {}).get("list", [])
                            if items:
                                self._states[sym].bb_premium = float(items[0][4])  # close
                    await asyncio.sleep(0.05)

                # Binance premium
                for sym in self.symbols:
                    url = f"{BINANCE_REST}/fapi/v1/premiumIndex?symbol={sym}"
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # markPrice - indexPrice gives premium
                            mark = float(data.get("markPrice", 0))
                            index = float(data.get("indexPrice", 0))
                            if index > 0:
                                self._states[sym].bn_premium = mark / index - 1.0
                    await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Premium poll error: {e}")

            await asyncio.sleep(60)

    # ---- Heartbeat ----

    async def _heartbeat(self):
        """Log connection health every 60 seconds."""
        while self._running:
            await asyncio.sleep(60)
            stats = self._stats
            bb_age = time.time() - stats["last_bb_msg"] if stats["last_bb_msg"] > 0 else -1
            bn_age = time.time() - stats["last_bn_msg"] if stats["last_bn_msg"] > 0 else -1
            logger.info(f"HEARTBEAT: bars={stats['bars_emitted']} "
                        f"bb_msgs={stats['bb_messages']} bn_msgs={stats['bn_messages']} "
                        f"bb_age={bb_age:.0f}s bn_age={bn_age:.0f}s "
                        f"errors={stats['errors']}")

            # Alert if no messages for 2 minutes
            if bb_age > 120:
                logger.warning(f"No Bybit messages for {bb_age:.0f}s!")
            if bn_age > 120:
                logger.warning(f"No Binance messages for {bn_age:.0f}s!")
