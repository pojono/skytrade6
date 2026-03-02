# Flow Shock Research - Quick Start Guide

## 🚀 Getting Started

### 1. Download Trade Data

```bash
cd /home/ubuntu/Projects/skytrade6/flow_shock_research

# Download SOLUSDT trade data (1.8 GB, ~3 minutes)
python3 download_trade_data.py --symbol SOLUSDT --start-date 2025-05-11 --end-date 2025-08-10
```

**Output:** `data_bybit/SOLUSDT/trade/` (1,488 hourly files)

---

### 2. Run Quick Analysis (Sample)

```bash
# Fast 7-day sample analysis (~4 minutes)
python3 research_flow_shock_quick.py --sample-days 7
```

**Output:**
- Console: Event statistics by window size
- `results/flow_shock_sample_*.csv` - Detected events
- `results/FINDINGS_flow_shock_quick.md` - Summary report

---

### 3. Scan Z-Thresholds

```bash
# Test multiple thresholds to find optimal (~4 minutes)
python3 research_flow_shock_threshold_scan.py
```

**Output:**
- Console: Threshold scan table
- `results/threshold_scan_results.csv` - Full scan data

**Result:** Optimal threshold = z > 30 (2.3 events/day)

---

### 4. Full Dataset Analysis (Optional)

```bash
# Process entire 92-day dataset (~2-3 hours)
python3 research_flow_shock_detector.py --start-date 2025-05-11 --end-date 2025-08-10
```

**Use this for:** Final validation after parameter tuning

---

## 📊 Key Results

### Optimal Configuration

```python
WINDOW_SECONDS = 30
FLOW_Z_THRESHOLD = 30.0
```

**Expected:** 2.3 events/day (210 events over 92 days)

### Why z > 30?

Crypto markets are extremely volatile:
- z > 3: 11,105 events/day ❌
- z > 10: 407 events/day ❌
- z > 20: 18 events/day ❌
- z > 30: 2.3 events/day ✅

---

## 📁 File Structure

```
flow_shock_research/
├── README.md                              # Full documentation
├── FINDINGS.md                            # Detailed research findings
├── QUICKSTART.md                          # This file
├── download_trade_data.py                 # Data downloader
├── research_flow_shock_quick.py           # Quick sample analysis
├── research_flow_shock_threshold_scan.py  # Threshold scanner
├── research_flow_shock_detector.py        # Full dataset processor
└── results/
    ├── threshold_scan_results.csv
    ├── flow_shock_sample_30s.csv
    └── FINDINGS_flow_shock_quick.md
```

---

## 🎯 Next Steps

### Phase 2: Price Behavior Analysis

**Goal:** Analyze forward returns around flow shock events

**Script to create:**
```python
# research_flow_shock_price_behavior.py
- Load detected events (z > 30)
- Extract price data around each event (T-30s to T+5m)
- Calculate forward returns at multiple horizons
- Analyze overshoot/mean reversion patterns
- Generate return distribution statistics
```

**Key questions:**
- Do prices overshoot after volume shocks?
- How long does disequilibrium last?
- What's the optimal entry/exit timing?
- Is there a tradeable edge?

---

## 💡 Tips

### Fast Iteration
- Use sample analysis (7 days) for parameter tuning
- Only run full dataset for final validation
- 50x speedup with minimal accuracy loss

### Memory Management
- Scripts use streaming architecture
- No full dataset loaded into RAM
- Safe to run on machines with limited memory

### Progress Tracking
- All scripts show real-time progress
- ETA calculations included
- Clear console output

---

## 🔧 Troubleshooting

### "No trade data files found"
- Check data directory: `data_bybit/SOLUSDT/trade/`
- Run download script first
- Verify SSH key exists: `~/.ssh/id_ed25519_remote`

### "Script is slow"
- Use sample analysis for iteration
- Full dataset takes 2-3 hours (expected)
- Check progress bar for ETA

### "Out of memory"
- Scripts use streaming (shouldn't happen)
- Reduce sample size if needed
- Check for memory leaks in custom code

---

## 📚 Documentation

- **README.md** - Complete research documentation
- **FINDINGS.md** - Detailed analysis and results
- **QUICKSTART.md** - This quick reference

---

**Last Updated:** March 2, 2026  
**Status:** Phase 1 Complete, Phase 2 Next
