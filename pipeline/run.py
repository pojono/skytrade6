#!/usr/bin/env python3
"""Settlement Trading Pipeline v4 — Modular Architecture.

End-to-end pipeline:
  1. Download latest JSONL data from remote server
  2. Parse all settlements (once, shared data layer)
  3. Train short exit ML (predict bottom of crash)
  4. Train long exit ML (predict peak of recovery)
  5. Run combined backtest (short + conditional long, ML exits)
  6. Compare strategies and generate report

Usage:
    python3 -m pipeline.run                    # full pipeline
    python3 -m pipeline.run --skip-download    # skip download step
    python3 -m pipeline.run --skip-training    # use saved models
    python3 -m pipeline.run --backtest-only    # skip training, just backtest
"""

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipeline.config import GROSS_PNL_BPS


def main():
    parser = argparse.ArgumentParser(description="Settlement Trading Pipeline v4")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-training", action="store_true", help="Use saved models (skip training)")
    parser.add_argument("--backtest-only", action="store_true", help="Only run backtest + report")
    args = parser.parse_args()

    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       SETTLEMENT TRADING PIPELINE v4 — MODULAR                  ║")
    print("║  Parse → Train (short+long ML) → Backtest → Report             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    from pipeline.data import download_new_data, load_all_settlements
    from pipeline.features import build_short_exit_ticks, build_long_exit_ticks, find_bottom
    from pipeline.models import (
        train_short_exit, train_long_exit,
        load_model, save_model,
    )
    from pipeline.backtest import run_backtest, compare_strategies
    from pipeline.report import generate_report

    # ── Step 1: Download ──────────────────────────────────────────────
    if not args.skip_download and not args.backtest_only:
        print(f"\n{'='*70}")
        print("STEP 1: DOWNLOAD DATA")
        print(f"{'='*70}")
        n_new = download_new_data()
    else:
        print("  (Download skipped)")

    # ── Step 2: Parse all settlements (single pass) ───────────────────
    print(f"\n{'='*70}")
    print("STEP 2: PARSE SETTLEMENTS")
    print(f"{'='*70}")
    settlements = load_all_settlements()
    if not settlements:
        print("  ✗ No valid settlements. Aborting.")
        sys.exit(1)

    n_pass = sum(1 for s in settlements if s.passes_filters)
    n_fail = len(settlements) - n_pass
    print(f"  Passes filters: {n_pass} / {len(settlements)} (skipped {n_fail})")

    # ── Step 3: Train short exit ML ───────────────────────────────────
    short_exit_results = None
    if not args.skip_training and not args.backtest_only:
        print(f"\n{'='*70}")
        print("STEP 3: TRAIN SHORT EXIT ML")
        print(f"{'='*70}")

        print("  Building short exit tick features...")
        t1 = time.time()
        all_ticks = []
        for i, sd in enumerate(settlements):
            ticks = build_short_exit_ticks(sd)
            if ticks:
                all_ticks.extend(ticks)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(settlements)}] {len(all_ticks)} ticks, {time.time()-t1:.1f}s")

        if all_ticks:
            tick_df = pd.DataFrame(all_ticks)
            print(f"  ✅ {len(tick_df)} ticks from {tick_df['settle_id'].nunique()} settlements")
            short_exit_results = train_short_exit(tick_df)
        else:
            print("  ✗ No tick data for short exit")

    # ── Step 4: Train long exit ML ────────────────────────────────────
    long_exit_results = None
    long_exit_model = None
    long_exit_features = None

    if not args.skip_training and not args.backtest_only:
        print(f"\n{'='*70}")
        print("STEP 4: TRAIN LONG EXIT ML")
        print(f"{'='*70}")

        print("  Building long exit tick features...")
        t1 = time.time()
        all_recovery_ticks = []
        for i, sd in enumerate(settlements):
            if not sd.passes_filters:
                continue
            bottom_bps, bottom_t = find_bottom(sd)
            if bottom_bps is None:
                continue
            ticks = build_long_exit_ticks(sd, bottom_bps, bottom_t)
            if ticks:
                all_recovery_ticks.extend(ticks)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(settlements)}] {len(all_recovery_ticks)} ticks, {time.time()-t1:.1f}s")

        if all_recovery_ticks:
            recovery_df = pd.DataFrame(all_recovery_ticks)
            print(f"  ✅ {len(recovery_df)} recovery ticks from {recovery_df['settle_id'].nunique()} settlements")
            long_exit_results = train_long_exit(recovery_df)
            long_exit_model = long_exit_results.get('model_lr')
            long_exit_features = long_exit_results.get('feature_cols')
        else:
            print("  ✗ No recovery tick data")

    # Load saved models if skipping training
    short_exit_model = None
    short_exit_features_list = None

    if args.skip_training or args.backtest_only:
        print("\n  Loading saved models...")

        short_exit_model, short_exit_features_list = load_model('short_exit_logreg')
        if short_exit_model is not None:
            print(f"    ✅ Loaded short_exit_logreg ({len(short_exit_features_list)} features)")
        else:
            print(f"    ✗ No saved short exit model — will use constant {GROSS_PNL_BPS} bps")

        long_exit_model, long_exit_features = load_model('long_exit_logreg')
        if long_exit_model is not None:
            print(f"    ✅ Loaded long_exit_logreg ({len(long_exit_features)} features)")
        else:
            print(f"    ✗ No saved long exit model found")
    else:
        # After training, use the just-trained models
        if short_exit_results:
            short_exit_model = short_exit_results.get('model_lr')
            short_exit_features_list = short_exit_results.get('feature_cols')

    # ── Step 5: Combined backtest ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 5: COMBINED BACKTEST")
    print(f"{'='*70}")

    strategies = compare_strategies(
        settlements,
        short_exit_model=short_exit_model,
        short_exit_features=short_exit_features_list,
        long_exit_model=long_exit_model,
        long_exit_features=long_exit_features,
    )

    # ── Step 6: Generate report ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 6: GENERATE REPORT")
    print(f"{'='*70}")

    generate_report(
        strategies,
        short_exit_results=short_exit_results,
        long_exit_results=long_exit_results,
    )

    # ── Done ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PIPELINE v4 COMPLETE  [{elapsed:.1f}s]")
    print(f"{'='*70}")

    best_name = max(strategies.keys(),
                    key=lambda k: strategies[k][1]['combined_per_day'])
    best = strategies[best_name][1]
    print(f"  Best: {best_name} → ${best['combined_per_day']:.1f}/day")
    print(f"  Settlements: {best['n_settlements']} total, {best['n_traded']} traded")
    print(f"  Short: ${best['short_per_day']:.1f}/day ({best['short_wr']:.0f}% WR)")
    print(f"  Long:  ${best['long_per_day']:.1f}/day ({best['long_wr']:.0f}% WR)")


if __name__ == "__main__":
    main()
