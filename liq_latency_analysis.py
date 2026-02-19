#!/usr/bin/env python3
"""
LATENCY ANALYSIS — Millisecond-precision timing for the liquidation cascade strategy.

Uses WebSocket ticker data (~100ms resolution) for sub-second price tracking.

Questions answered:
1. WS delivery latency: how fast does Bybit push liquidation events?
2. Inter-liquidation timing during cascades
3. Price reaction speed: how fast does price move after P95 liq event?
4. Fill window: how many ms until our limit price is touched?
5. Strategy PnL at various order placement delays
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


def load_liquidations_ms(symbol, data_dir='data'):
    """Load liquidation events with all ms timestamps."""
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    capture_ts = data['ts']
                    ws_ts = data['result'].get('ts', capture_ts)
                    for ev in data['result']['data']:
                        records.append({
                            'event_ts_ms': int(ev['T']),
                            'ws_ts_ms': int(ws_ts),
                            'capture_ts_ms': int(capture_ts),
                            'side': ev['S'],
                            'volume': float(ev['v']),
                            'price': float(ev['p']),
                        })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    df = pd.DataFrame(records)
    df['notional'] = df['volume'] * df['price']
    df = df.sort_values('event_ts_ms').reset_index(drop=True)
    return df


def load_ws_ticker_ms(symbol, data_dir='data'):
    """Load preprocessed WS ticker CSV with ms timestamps."""
    csv_path = Path(data_dir) / symbol / "ticker_prices.csv.gz"
    if not csv_path.exists():
        print(f"  No ticker_prices.csv.gz found!")
        return pd.DataFrame()
    print(f"  Loading ticker CSV...", end='', flush=True)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'ts': 'ts_ms'})
    df['price'] = df['price'].astype(float)
    df = df.sort_values('ts_ms').reset_index(drop=True)
    print(f" done ({len(df):,})")
    return df


def pct_table(arr, label, pcts=[1, 5, 10, 25, 50, 75, 90, 95, 99]):
    vals = np.percentile(arr, pcts)
    print(f"     {label} (n={len(arr):,}):")
    for p, v in zip(pcts, vals):
        bar = '█' * max(1, int(v / max(vals) * 40))
        print(f"       P{p:>2d}: {v:>10.0f}ms  {bar}")
    print(f"       Mean: {np.mean(arr):>8.0f}ms  Std: {np.std(arr):>8.0f}ms")


def main():
    t0 = time.time()
    print("=" * 100)
    print("  LATENCY ANALYSIS — Millisecond Precision (WebSocket Data)")
    print("=" * 100)

    # Aggregate across symbols
    agg = {
        'ws_delay': [], 'capture_delay': [],
        'inter_liq_all': [], 'inter_liq_cascade': [],
        'price_at_delay': {d: [] for d in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]},
        'time_to_bps': {b: [] for b in [5, 10, 15, 20, 30]},
        'fill_by_delay': {},
    }

    delays_to_test = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]
    for d in delays_to_test:
        agg['fill_by_delay'][d] = {'fills': 0, 'signals': 0, 'tp': 0, 'pnl': []}

    for symbol in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  {symbol}")
        print(f"{'─'*80}")

        liq_df = load_liquidations_ms(symbol)
        ws_tick = load_ws_ticker_ms(symbol)

        if len(liq_df) < 100 or len(ws_tick) < 1000:
            print(f"  Insufficient data, skipping")
            continue

        # ================================================================
        # 1. WS DELIVERY LATENCY
        # ================================================================
        ws_delay = (liq_df['ws_ts_ms'] - liq_df['event_ts_ms']).values
        capture_delay = (liq_df['capture_ts_ms'] - liq_df['event_ts_ms']).values
        ws_delay = ws_delay[(ws_delay >= 0) & (ws_delay < 60000)]
        capture_delay = capture_delay[(capture_delay >= 0) & (capture_delay < 60000)]

        print(f"\n  1. WS DELIVERY LATENCY")
        pct_table(ws_delay, "Event → WS server push")
        pct_table(capture_delay, "Event → Dataminer capture")
        agg['ws_delay'].extend(ws_delay.tolist())
        agg['capture_delay'].extend(capture_delay.tolist())

        # ================================================================
        # 2. INTER-LIQUIDATION TIMING
        # ================================================================
        event_times = liq_df['event_ts_ms'].values
        inter_ms = np.diff(event_times)
        inter_ms_valid = inter_ms[(inter_ms >= 0) & (inter_ms < 3600000)]

        print(f"\n  2. INTER-LIQUIDATION TIMING (all consecutive events)")
        pct_table(inter_ms_valid, "Gap between consecutive liquidations")
        agg['inter_liq_all'].extend(inter_ms_valid.tolist())

        # P95 events within cascades
        p95_thresh = liq_df['notional'].quantile(0.95)
        large = liq_df[liq_df['notional'] >= p95_thresh].copy()
        large_times = large['event_ts_ms'].values
        cascade_gaps = []
        for i in range(1, len(large_times)):
            gap = large_times[i] - large_times[i-1]
            if 0 < gap <= 60000:
                cascade_gaps.append(gap)

        if cascade_gaps:
            print(f"\n     P95 events within 60s of each other (cascade internal):")
            pct_table(np.array(cascade_gaps), "Cascade internal gap")
            agg['inter_liq_cascade'].extend(cascade_gaps)

        # ================================================================
        # 3. PRICE REACTION SPEED (using WS ticker — sub-second)
        # ================================================================
        print(f"\n  3. PRICE REACTION AFTER P95 EVENT (WS ticker, sub-second)")
        tick_ts = ws_tick['ts_ms'].values
        tick_price = ws_tick['price'].values

        n_analyzed = 0
        for _, row in large.iterrows():
            evt_ms = row['event_ts_ms']
            idx = np.searchsorted(tick_ts, evt_ms)
            if idx >= len(tick_ts) - 100 or idx < 1:
                continue
            base_price = tick_price[idx - 1]
            if base_price <= 0:
                continue

            # Price move at various delays
            for delay_ms in agg['price_at_delay']:
                target_ts = evt_ms + delay_ms
                tidx = np.searchsorted(tick_ts, target_ts)
                if tidx < len(tick_ts):
                    move_bps = abs(tick_price[tidx] - base_price) / base_price * 10000
                    agg['price_at_delay'][delay_ms].append(move_bps)

            # Time to reach bps thresholds
            found_bps = set()
            for j in range(idx, min(idx + 5000, len(tick_ts))):
                elapsed_ms = tick_ts[j] - evt_ms
                if elapsed_ms > 120000:
                    break
                move_bps = abs(tick_price[j] - base_price) / base_price * 10000
                for b in agg['time_to_bps']:
                    if b not in found_bps and move_bps >= b:
                        agg['time_to_bps'][b].append(elapsed_ms)
                        found_bps.add(b)
                if len(found_bps) == len(agg['time_to_bps']):
                    break

            n_analyzed += 1

        print(f"     Analyzed {n_analyzed:,} P95 events")
        print(f"\n     Price move (bps) at various delays after P95 event:")
        print(f"     {'Delay':>10s}  {'Median':>8s}  {'Mean':>8s}  {'P75':>8s}  {'P90':>8s}  {'P95':>8s}  {'n':>6s}")
        for delay_ms in sorted(agg['price_at_delay']):
            moves = agg['price_at_delay'][delay_ms]
            if moves:
                arr = np.array(moves)
                print(f"     {delay_ms:>8d}ms  {np.median(arr):7.1f}  {np.mean(arr):7.1f}  "
                      f"{np.percentile(arr,75):7.1f}  {np.percentile(arr,90):7.1f}  "
                      f"{np.percentile(arr,95):7.1f}  {len(arr):>6d}")

        # ================================================================
        # 4. FILL WINDOW — simulate limit order at various delays (ms precision)
        # ================================================================
        print(f"\n  4. FILL WINDOW ANALYSIS (ms precision, using WS ticker)")

        entry_offset_pct = 0.15
        tp_pct = 0.12
        maker_fee = 0.02
        taker_fee = 0.055
        max_hold_ms = 60 * 60000  # 60 min

        n_signals = 0
        for _, row in large.iterrows():
            evt_ms = row['event_ts_ms']
            idx = np.searchsorted(tick_ts, evt_ms)
            if idx >= len(tick_ts) - 100 or idx < 10:
                continue

            # Check displacement (use ~1min lookback)
            lookback_ts = evt_ms - 60000
            lb_idx = np.searchsorted(tick_ts, lookback_ts)
            if lb_idx >= len(tick_ts):
                continue
            pre_price = tick_price[lb_idx]
            current_price = tick_price[idx]
            if pre_price <= 0:
                continue
            disp_bps = abs(current_price - pre_price) / pre_price * 10000
            if disp_bps < 10:
                continue

            is_buy_liq = row['side'] == 'Buy'
            if is_buy_liq:
                limit_price = current_price * (1 - entry_offset_pct / 100)
                tp_price = limit_price * (1 + tp_pct / 100)
            else:
                limit_price = current_price * (1 + entry_offset_pct / 100)
                tp_price = limit_price * (1 - tp_pct / 100)

            n_signals += 1

            for delay_ms in delays_to_test:
                agg['fill_by_delay'][delay_ms]['signals'] += 1
                start_ts = evt_ms + delay_ms
                start_idx = np.searchsorted(tick_ts, start_ts)
                end_ts = evt_ms + max_hold_ms
                end_idx = min(np.searchsorted(tick_ts, end_ts), len(tick_ts) - 1)

                if start_idx >= end_idx:
                    continue

                # Check fill
                filled = False
                fill_idx = None
                for j in range(start_idx, end_idx):
                    p = tick_price[j]
                    if is_buy_liq and p <= limit_price:
                        filled = True; fill_idx = j; break
                    elif not is_buy_liq and p >= limit_price:
                        filled = True; fill_idx = j; break

                if not filled:
                    continue

                agg['fill_by_delay'][delay_ms]['fills'] += 1
                fill_ts = tick_ts[fill_idx]

                # Check TP
                exit_price = None; exit_reason = 'timeout'
                for k in range(fill_idx, end_idx):
                    p = tick_price[k]
                    if is_buy_liq and p >= tp_price:
                        exit_price = tp_price; exit_reason = 'take_profit'; break
                    elif not is_buy_liq and p <= tp_price:
                        exit_price = tp_price; exit_reason = 'take_profit'; break

                if exit_price is None:
                    exit_price = tick_price[end_idx]

                if is_buy_liq:
                    raw_pnl = (exit_price - limit_price) / limit_price * 100
                else:
                    raw_pnl = (limit_price - exit_price) / limit_price * 100

                entry_fee = maker_fee
                exit_fee = maker_fee if exit_reason == 'take_profit' else taker_fee
                net_pnl = raw_pnl - entry_fee - exit_fee

                agg['fill_by_delay'][delay_ms]['pnl'].append(net_pnl)
                if exit_reason == 'take_profit':
                    agg['fill_by_delay'][delay_ms]['tp'] += 1

        print(f"     Signals with disp≥10bps: {n_signals}")

    # ================================================================
    # AGGREGATE SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print(f"  AGGREGATE SUMMARY (all {len(SYMBOLS)} symbols)")
    print(f"{'='*100}")

    print(f"\n  1. WS DELIVERY LATENCY (how fast you receive the liq event)")
    if agg['ws_delay']:
        pct_table(np.array(agg['ws_delay']), "Event → WS server")
    if agg['capture_delay']:
        pct_table(np.array(agg['capture_delay']), "Event → Dataminer capture (your bot)")

    print(f"\n  2. INTER-LIQUIDATION TIMING")
    if agg['inter_liq_all']:
        pct_table(np.array(agg['inter_liq_all']), "All consecutive liquidations")
    if agg['inter_liq_cascade']:
        pct_table(np.array(agg['inter_liq_cascade']), "P95 cascade internal gaps")

    print(f"\n  3. PRICE REACTION SPEED (bps move after P95 event)")
    print(f"     {'Delay':>10s}  {'Median':>8s}  {'Mean':>8s}  {'P75':>8s}  {'P90':>8s}  {'P95':>8s}  {'n':>6s}")
    print(f"     {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    for delay_ms in sorted(agg['price_at_delay']):
        moves = agg['price_at_delay'][delay_ms]
        if moves:
            arr = np.array(moves)
            print(f"     {delay_ms:>8d}ms  {np.median(arr):7.1f}  {np.mean(arr):7.1f}  "
                  f"{np.percentile(arr,75):7.1f}  {np.percentile(arr,90):7.1f}  "
                  f"{np.percentile(arr,95):7.1f}  {len(arr):>6d}")

    print(f"\n  4. TIME TO REACH BPS THRESHOLDS (ms after P95 event)")
    print(f"     {'Target':>8s}  {'Median':>8s}  {'Mean':>10s}  {'P25':>8s}  {'P75':>8s}  {'P90':>8s}  {'Hit%':>6s}")
    print(f"     {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    total_p95 = len(agg['price_at_delay'].get(50, []))
    for bps in sorted(agg['time_to_bps']):
        times = agg['time_to_bps'][bps]
        if times:
            arr = np.array(times)
            hit_pct = len(arr) / total_p95 * 100 if total_p95 > 0 else 0
            print(f"     {bps:>6d}bps  {np.median(arr):7.0f}  {np.mean(arr):9.0f}  "
                  f"{np.percentile(arr,25):7.0f}  {np.percentile(arr,75):7.0f}  "
                  f"{np.percentile(arr,90):7.0f}  {hit_pct:5.1f}%")

    print(f"\n  5. STRATEGY PnL BY ORDER PLACEMENT DELAY")
    print(f"     {'Delay':>10s}  {'Signals':>8s}  {'Fills':>6s}  {'Fill%':>6s}  {'TP%':>6s}  "
          f"{'AvgPnL':>8s}  {'TotPnL':>8s}  {'vs 0ms':>8s}")
    print(f"     {'─'*10}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    base_total = None
    for delay_ms in delays_to_test:
        r = agg['fill_by_delay'][delay_ms]
        if r['pnl']:
            pnl = np.array(r['pnl'])
            fill_pct = r['fills'] / r['signals'] * 100 if r['signals'] > 0 else 0
            tp_pct = r['tp'] / r['fills'] * 100 if r['fills'] > 0 else 0
            total = pnl.sum()
            if base_total is None:
                base_total = total
            vs_base = total / base_total * 100 if base_total > 0 else 0
            flag = '✅' if total > 0 else '❌'
            print(f"  {flag} {delay_ms:>8d}ms  {r['signals']:>8d}  {r['fills']:>6d}  {fill_pct:5.1f}%  "
                  f"{tp_pct:5.1f}%  {pnl.mean():+7.4f}%  {total:+7.2f}%  {vs_base:6.1f}%")

    # ================================================================
    # LATENCY BUDGET CONCLUSION
    # ================================================================
    print(f"\n{'='*100}")
    print(f"  LATENCY BUDGET CONCLUSION")
    print(f"{'='*100}")
    print(f"""
  Your total latency budget = time from P95 liq event to limit order on exchange.

  Pipeline:
    1. Bybit WS pushes liq event          (see WS delivery latency above)
    2. Your bot receives it                (network latency to your server)
    3. Your bot computes signal            (CPU time — negligible if optimized)
    4. Your bot sends limit order to Bybit (network latency to exchange)
    5. Bybit acknowledges order            (exchange processing)

  The fill window analysis above shows exactly how much PnL you lose
  at each delay level. Use this to set your latency target.
""")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
