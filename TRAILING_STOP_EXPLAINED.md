# Trailing Stop — Explained Simply

---

## The Dog on a Leash Analogy

Imagine you're walking a dog on a retractable leash in a park.

**You** = the trailing stop level (the exit trigger)  
**The dog** = the price  
**The leash** = the trail width (5 bps in our case)

### How it works:

1. **The dog starts walking forward** (price moves in your favor after you enter a trade).

2. **You follow the dog**, always staying exactly one leash-length behind. As the dog moves forward, you move forward too. You never stand still when the dog is moving ahead — you always keep up.

3. **The dog stops and turns around** (price starts reversing). Now the dog is walking back toward you. But **you don't move backward** — you stay planted where you are.

4. **When the dog reaches you** (price drops back to the trail level), you grab the dog and go home. **Trade closed.**

### The key rules:
- **You only move forward, never backward** (trail level only tightens, never loosens)
- **The leash length is fixed** (always 5 bps behind the best price)
- **The moment the dog reaches you, you're done** (exit immediately)

---

## What This Looks Like With Real Numbers

You buy ETH at **$2,500.00**. Your leash is **5 bps = $1.25**.

```
Second 0:   ETH = $2,500.00   You stand at $2,498.75   (dog just started walking)
Second 2:   ETH = $2,501.00   You move to  $2,499.75   (dog moved, you follow)
Second 5:   ETH = $2,503.00   You move to  $2,501.75   (dog keeps going, you keep following)
Second 8:   ETH = $2,502.00   You stay at  $2,501.75   (dog turned around, you DON'T move back)
Second 10:  ETH = $2,501.50   You stay at  $2,501.75   (dog getting closer...)
Second 11:  ETH = $2,501.75   DOG REACHED YOU → SELL!   (trade done)

You bought at $2,500.00, sold at $2,501.75 → profit: +$1.75 per ETH
```

The dog went as far as $2,503.00 but you didn't sell there — you can't predict the top. Instead, you let the dog run and only acted when it **clearly turned around** and came back to you.

---

## Why Not Just Use a Fixed Target?

The old approach was like telling the dog: "I'll only leave the park when you reach that specific tree 12 meters away."

**Problem:** Sometimes the dog walks 10 meters toward the tree, then turns around and runs the other direction for 100 meters. You're stuck in the park holding a leash attached to a dog that's now very far away in the wrong direction. After 60 minutes you give up and go home (timeout) with a big loss.

**With trailing stop:** You would have captured those 10 meters of forward progress. The moment the dog turned around and walked 5 bps back, you'd leave the park with a small win instead of waiting for a big loss.

---

## The Numbers Tell the Story

In our backtest, **every single timeout trade** (the ones that lost money) **was profitable at some point**. The dog always walked forward at least a little. But we were waiting for the specific tree, so we missed the chance to leave with a profit.

| Approach | What happens | Result |
|----------|-------------|--------|
| **Fixed target (old)** | Wait for dog to reach the tree (12 bps) | 98% of the time it works. 2% of the time the dog turns around and you lose big. |
| **Trailing stop (new)** | Follow the dog, leave when it turns around | 88% of the time you profit. 12% small losses. **No big losses ever.** |

The trailing stop makes less per winning trade, but it **completely eliminates the catastrophic losses** that happen when the dog runs the wrong way for an hour.

---

## One More Way to Think About It

Think of it like selling a stock that's going up:

- **Fixed target** = "I'll sell at exactly $105." If it goes to $104.90 and crashes to $80, you missed your chance.
- **Trailing stop** = "I'll sell whenever it drops $0.50 from its highest point." If it peaks at $104.90, you sell at $104.40. Not perfect, but you locked in most of the gain.

The trailing stop says: **"I don't know where the top is, but I know when the move is over."**

---

*This is the exit logic used in the liquidation cascade strategy. See `TRADING_GUIDE_v3.md` for the full step-by-step trading guide.*
