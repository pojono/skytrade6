# Fibonacci Microstructure Execution Alpha

## The Concept

Standard technical analysis uses "Structural Alpha"—buying at levels like the Fibonacci 0.618 retracement, assuming the level will hold. Our large-scale study over millions of klines showed that this naked approach catches "falling knives", resulting in a negative Sharpe ratio.

We transitioned to **Execution Alpha**. Instead of blindly placing limit orders at the structural level, we define a "Zone of Interest" (ZOI) from `0.58` to `0.65` retracement. We then monitor the **tick-level orderflow** specifically when price enters this zone.

We look for **Absorption**: When price forms the absolute high/low of the pullback inside the ZOI, who is trading?
- If the price is making a local low, but 60%+ of the market volume is aggressive **selling** (Sell Market Orders), yet the price refuses to drop further, it means a massive passive buyer (absorption) is defending the level.

## The Results

We ran a study over 859 true ZOI extremes across BTC, ETH, SOL, XRP, and DOGE in Q1 2025.

### Baseline (Structural Alpha Only)
- If you simply buy when price hits the extreme in the 0.618 zone, the success rate of the swing reversing to a new macro high is **74.5%**.

### Execution Alpha (Microstructure Confirmation)
- If we filter those events to only include times where `absorption_ratio > 0.6` (meaning >60% of the volume at the very bottom was aggressive selling into passive limit buyers):
- The success rate jumps to **85.7%**.

### What this means
By requiring tick-level confirmation of absorption, we dramatically reduce adverse selection. The market makers are signaling their defense of the structural level.

This translates to a massive improvement in expected value. A strategy trading structural alpha alone will slowly bleed to death via adverse selection on the failed 25.5% of trades (which tend to be catastrophic liquidations slicing straight through the level). The execution alpha reduces that failure rate to just 14%, filtering out the most dangerous "falling knife" scenarios.
