"""Shared JSONL parser and data downloader.

Parses each JSONL file ONCE into a SettlementData object that all modules reuse.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pipeline.config import (
    LOCAL_DATA_DIR, REMOTE_HOST, SSH_KEY, REMOTE_DATA_DIR,
    MIN_DEPTH_20, MAX_SPREAD_BPS,
)


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class SettlementData:
    """Parsed data from one JSONL settlement recording."""
    file_path: Path
    symbol: str
    settle_id: str           # e.g. "BTCUSDT_20260228_080000"

    # Pre-settlement
    ref_price: float         # last pre-settlement trade price
    fr_bps: float            # funding rate in bps (signed)
    fr_abs_bps: float
    pre_vol_rate: float      # $/s in last 10s pre-settlement
    pre_trade_rate: float    # trades/s in last 10s
    pre_spread_bps: float

    # Orderbook at settlement
    mid_price: float
    bids: list               # [(price, qty), ...] descending
    asks: list               # [(price, qty), ...] ascending
    spread_bps: float
    depth_20: float          # $ within 20bps of mid on bid side

    # Post-settlement trades (sorted by time)
    post_times: np.ndarray       # ms relative to settlement
    post_prices: np.ndarray      # raw prices
    post_prices_bps: np.ndarray  # bps relative to ref_price
    post_sides: np.ndarray       # 1=Sell, 0=Buy
    post_notionals: np.ndarray   # price * qty
    post_sizes: np.ndarray       # qty

    # Post-settlement OB.1 updates
    ob1_times: np.ndarray
    ob1_bids: list           # [(t, bp, bq, ap, aq), ...]
    ob1_asks: list

    # Tickers
    tickers: list            # [(t_ms, fr_value), ...]

    # Derived
    passes_filters: bool     # depth >= MIN and spread <= MAX

    # Fields with defaults must come last
    ob200_deltas: list = field(default_factory=list)  # [(t_ms, bids_delta, asks_delta), ...]
    price_bins: dict = field(default_factory=dict)  # 100ms -> bps


# ── OB reconstruction ────────────────────────────────────────────────

def reconstruct_ob_at(sd, t_ms):
    """Reconstruct full orderbook at a given post-settlement time.

    Starts from the T-0 snapshot (sd.bids, sd.asks) and applies all
    ob200 delta updates up to t_ms.

    Returns:
        dict with 'bids' [(price, qty) desc], 'asks' [(price, qty) asc],
        'mid_price', 'spread_bps', 'depth_20'
        or None if no data available.
    """
    # Start from T-0 snapshot as dicts {price: qty}
    bid_book = {p: q for p, q in sd.bids if q > 0}
    ask_book = {p: q for p, q in sd.asks if q > 0}

    # Apply all ob200 deltas up to t_ms
    for dt, b_deltas, a_deltas in sd.ob200_deltas:
        if dt > t_ms:
            break
        for price, qty in b_deltas:
            if qty <= 0:
                bid_book.pop(price, None)
            else:
                bid_book[price] = qty
        for price, qty in a_deltas:
            if qty <= 0:
                ask_book.pop(price, None)
            else:
                ask_book[price] = qty

    if not bid_book or not ask_book:
        return None

    # Sort: bids descending, asks ascending
    bids = sorted(bid_book.items(), key=lambda x: -x[0])
    asks = sorted(ask_book.items(), key=lambda x: x[0])

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 10000 if mid > 0 else 0

    depth_20 = sum(p * q for p, q in bids if (mid - p) / mid * 10000 <= 20)

    return {
        'bids': bids,
        'asks': asks,
        'mid_price': mid,
        'spread_bps': spread_bps,
        'depth_20': depth_20,
    }


# ── JSONL parsing ─────────────────────────────────────────────────────

def parse_settlement(fp: Path) -> Optional[SettlementData]:
    """Parse a single JSONL file into a SettlementData object."""
    trades_pre, trades_post = [], []
    ob1_data = []
    ob200_data = []
    tickers = []
    ob_snapshot_bids = []
    ob_snapshot_asks = []

    with open(fp) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except Exception:
                continue

            t_ms = msg.get('_t_ms', 0)
            topic = msg.get('topic', '')
            data = msg.get('data', {})

            if 'publicTrade' in topic:
                items = data if isinstance(data, list) else [data]
                for tr in items:
                    p = float(tr.get('p', 0))
                    q = float(tr.get('v', 0))
                    s = tr.get('S', '')
                    rec = (t_ms, p, q, s, p * q)
                    if t_ms < 0:
                        trades_pre.append(rec)
                    else:
                        trades_post.append(rec)

            elif topic.startswith('orderbook.1.'):
                b = data.get('b', [])
                a = data.get('a', [])
                if b and a:
                    ob1_data.append((t_ms, float(b[0][0]), float(b[0][1]),
                                     float(a[0][0]), float(a[0][1])))

            elif 'orderbook.200' in topic:
                # OB.200 delta updates (for reconstructing full book at any time)
                b_deltas = [(float(lv[0]), float(lv[1])) for lv in data.get('b', [])]
                a_deltas = [(float(lv[0]), float(lv[1])) for lv in data.get('a', [])]
                if b_deltas or a_deltas:
                    ob200_data.append((t_ms, b_deltas, a_deltas))

            elif 'orderbook' in topic and 'snapshot' in str(data.get('type', '')):
                # Full OB snapshot (for depth computation)
                for level in data.get('b', []):
                    ob_snapshot_bids.append((float(level[0]), float(level[1])))
                for level in data.get('a', []):
                    ob_snapshot_asks.append((float(level[0]), float(level[1])))

            elif 'tickers' in topic:
                fr_val = float(data.get('fundingRate', 0))
                tickers.append((t_ms, fr_val))

    if not trades_pre or len(trades_post) < 10:
        return None

    trades_pre.sort(key=lambda x: x[0])
    trades_post.sort(key=lambda x: x[0])

    # Reference price
    ref_price = trades_pre[-1][1]
    if ref_price <= 0:
        return None

    bps_fn = lambda p: (p / ref_price - 1) * 10000

    # Symbol and ID
    stem = fp.stem
    parts = stem.split('_')
    symbol = parts[0]

    # FR
    pre_tickers = [tk for tk in tickers if tk[0] < 0]
    fr_bps = pre_tickers[-1][1] * 10000 if pre_tickers else 0.0

    # Pre-settlement stats
    pre_10s = [t for t in trades_pre if t[0] >= -10000]
    pre_vol_rate = sum(n for _, _, _, _, n in pre_10s) / 10.0 if pre_10s else 1.0
    pre_trade_rate = len(pre_10s) / 10.0

    # OB at settlement
    pre_ob1 = sorted([o for o in ob1_data if o[0] < 0], key=lambda x: x[0])
    pre_spread_bps = 0.0
    mid_price = ref_price
    if pre_ob1:
        _, bp, bq, ap, aq = pre_ob1[-1]
        mid_price = (bp + ap) / 2
        pre_spread_bps = (ap - bp) / mid_price * 10000

    # Build bids/asks from OB snapshot or pre_ob1
    # Use the research_position_sizing approach
    from research_position_sizing import parse_last_ob_before_settlement
    ob = parse_last_ob_before_settlement(fp)
    if ob is None:
        return None

    bids = ob['bids']
    asks = ob['asks']
    mid_price = ob['mid_price']
    spread_bps = ob['spread_bps']
    depth_20 = sum(p * q for p, q in bids if (mid_price - p) / mid_price * 10000 <= 20)

    # Post-trade arrays
    post_times = np.array([t for t, _, _, _, _ in trades_post])
    post_prices = np.array([p for _, p, _, _, _ in trades_post])
    post_prices_bps = np.array([bps_fn(p) for _, p, _, _, _ in trades_post])
    post_sides = np.array([1 if s == 'Sell' else 0 for _, _, _, s, _ in trades_post])
    post_notionals = np.array([n for _, _, _, _, n in trades_post])
    post_sizes = np.array([q for _, _, q, _, _ in trades_post])

    # OB.1 post arrays
    ob1_post = sorted([(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1_data if t >= 0],
                       key=lambda x: x[0])
    ob1_times = np.array([t for t, _, _, _, _ in ob1_post]) if ob1_post else np.array([])

    # OB.200 deltas sorted by time
    ob200_sorted = sorted(ob200_data, key=lambda x: x[0])

    # Price bins (100ms resolution)
    price_bins = {}
    for t_ms, p, _, _, _ in trades_post:
        if 0 <= t_ms <= 60000:
            bk = int(t_ms / 100) * 100
            price_bins[bk] = bps_fn(p)

    passes = depth_20 >= MIN_DEPTH_20 and spread_bps <= MAX_SPREAD_BPS

    return SettlementData(
        file_path=fp,
        symbol=symbol,
        settle_id=stem,
        ref_price=ref_price,
        fr_bps=fr_bps,
        fr_abs_bps=abs(fr_bps),
        pre_vol_rate=pre_vol_rate,
        pre_trade_rate=pre_trade_rate,
        pre_spread_bps=pre_spread_bps,
        mid_price=mid_price,
        bids=bids,
        asks=asks,
        spread_bps=spread_bps,
        depth_20=depth_20,
        post_times=post_times,
        post_prices=post_prices,
        post_prices_bps=post_prices_bps,
        post_sides=post_sides,
        post_notionals=post_notionals,
        post_sizes=post_sizes,
        ob1_times=ob1_times,
        ob1_bids=ob1_post,
        ob1_asks=ob1_post,
        ob200_deltas=ob200_sorted,
        tickers=tickers,
        passes_filters=passes,
        price_bins=price_bins,
    )


def load_all_settlements(data_dir: Path = None) -> List[SettlementData]:
    """Parse all JSONL files into SettlementData objects (one pass)."""
    d = data_dir or LOCAL_DATA_DIR
    jsonl_files = sorted(d.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ✗ No JSONL files in {d}")
        return []

    print(f"  Parsing {len(jsonl_files)} JSONL files...")
    t0 = time.time()
    settlements = []
    skipped = 0

    for i, fp in enumerate(jsonl_files):
        sd = parse_settlement(fp)
        if sd is not None:
            settlements.append(sd)
        else:
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(jsonl_files)}] {len(settlements)} valid, {time.time()-t0:.1f}s")

    print(f"  ✅ {len(settlements)} settlements parsed ({skipped} skipped) [{time.time()-t0:.1f}s]")
    return settlements


# ── Data download ─────────────────────────────────────────────────────

def _ssh(cmd, timeout=30):
    """Run SSH command on remote server."""
    return subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10", REMOTE_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )


def _scp(remote_path, local_path, timeout=120):
    """SCP file from remote."""
    return subprocess.run(
        ["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
         f"{REMOTE_HOST}:{remote_path}", str(local_path)],
        capture_output=True, text=True, timeout=timeout
    )


def download_new_data(data_dir: Path = None) -> int:
    """Download new JSONL files from remote server. Returns count of new files."""
    d = data_dir or LOCAL_DATA_DIR
    d.mkdir(exist_ok=True)

    print(f"\n  Syncing from {REMOTE_HOST}:{REMOTE_DATA_DIR}...")

    # List remote files
    result = _ssh(f"ls {REMOTE_DATA_DIR}/*.jsonl 2>/dev/null | xargs -I{{}} basename {{}}")
    if result.returncode != 0:
        print(f"  ✗ SSH failed: {result.stderr}")
        return 0

    remote_files = set(result.stdout.strip().split('\n'))
    remote_files.discard('')
    local_files = {f.name for f in d.glob("*.jsonl")}
    new_files = remote_files - local_files

    if not new_files:
        print(f"  ✓ Already up to date ({len(local_files)} files)")
        return 0

    print(f"  Downloading {len(new_files)} new files...")
    downloaded = 0
    for fn in sorted(new_files):
        r = _scp(f"{REMOTE_DATA_DIR}/{fn}", d / fn)
        if r.returncode == 0:
            downloaded += 1

    print(f"  ✅ Downloaded {downloaded} new files (total: {len(local_files) + downloaded})")
    return downloaded
