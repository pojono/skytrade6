# Flow Shock Research - Document Index

## 📚 Documentation

### 1. [README.md](README.md) - Start Here
**Purpose:** Complete research documentation  
**Contents:**
- Research objectives and methodology
- Data infrastructure and processing
- Technical implementation details
- Results and optimal configuration
- Next steps and production considerations

**Read this for:** Full understanding of the research project

---

### 2. [FINDINGS.md](FINDINGS.md) - Detailed Analysis
**Purpose:** Comprehensive research findings  
**Contents:**
- Executive summary
- Data infrastructure performance
- Complete threshold scan results
- Critical insights and discoveries
- Validation methodology
- Comparison with other strategies
- Production considerations

**Read this for:** Deep dive into analysis and results

---

### 3. [QUICKSTART.md](QUICKSTART.md) - Quick Reference
**Purpose:** Fast setup and usage guide  
**Contents:**
- Step-by-step commands
- Expected outputs
- Key results summary
- Troubleshooting tips

**Read this for:** Getting started quickly

---

## 🔧 Scripts

### Data Collection

**[download_trade_data.py](download_trade_data.py)**
- Downloads WebSocket trade data from remote dataminer
- Optimized rsync with fast cipher (9.6 MB/s)
- Usage: `python3 download_trade_data.py --symbol SOLUSDT`

---

### Analysis Scripts

**[research_flow_shock_quick.py](research_flow_shock_quick.py)**
- **Runtime:** 3-4 minutes
- **Purpose:** Fast sample-based analysis (7 days)
- **Use for:** Parameter exploration and iteration
- **Output:** Event statistics, CSV files, summary report

**[research_flow_shock_threshold_scan.py](research_flow_shock_threshold_scan.py)**
- **Runtime:** 3-4 minutes
- **Purpose:** Multi-threshold scanner (z = 3 to 30)
- **Use for:** Finding optimal z-threshold
- **Output:** Threshold comparison table, CSV results

**[research_flow_shock_detector.py](research_flow_shock_detector.py)**
- **Runtime:** 2-3 hours
- **Purpose:** Full dataset processor (92 days)
- **Use for:** Final validation after parameter tuning
- **Output:** Complete event dataset, detailed statistics

---

## 📊 Results

### CSV Files

**[results/threshold_scan_results.csv](results/threshold_scan_results.csv)**
- Complete threshold scan data
- Columns: threshold, sample_events, est_per_day, est_total_92d, in_target, max_z, mean_z

**[results/flow_shock_sample_30s.csv](results/flow_shock_sample_30s.csv)**
- Detected events using optimal configuration (z > 30, 30s window)
- Columns: timestamp, datetime, flow_z, volume, price, mean_vol, std_vol, window_trades

**[results/flow_shock_sample_*.csv](results/)**
- Events for different window sizes (5s, 10s, 15s, 30s)
- Use for comparing window size effects

---

## 🎯 Key Results Summary

### Optimal Configuration
```python
WINDOW_SECONDS = 30
FLOW_Z_THRESHOLD = 30.0
```

### Performance
- **Events per day:** 2.3 (target: 1-5) ✅
- **Total events (92d):** 210
- **Flow Z range:** 30.0 - 60.1
- **Mean flow Z:** 35.6

### Critical Discovery
Crypto markets require **z > 30** (vs z > 3 in traditional finance) to filter noise and identify genuine forced flow events.

---

## 🗺️ Research Roadmap

### ✅ Phase 1: Event Detection (COMPLETE)
- Optimal threshold found: z > 30
- Expected frequency: 2.3 events/day
- Scripts validated and documented

### 🔄 Phase 2: Price Behavior (NEXT)
- Analyze forward returns around events
- Measure overshoot/mean reversion patterns
- Validate trading opportunity

### ⏳ Phase 3: Strategy Development (PENDING)
- Add directional filters (buy vs sell aggression)
- Define entry/exit rules
- Build backtest framework

### ⏳ Phase 4: Production (PENDING)
- Real-time detector implementation
- Execution system
- Risk management

---

## 📁 File Structure

```
flow_shock_research/
├── INDEX.md                               # This file
├── README.md                              # Complete documentation
├── FINDINGS.md                            # Detailed analysis
├── QUICKSTART.md                          # Quick reference
│
├── download_trade_data.py                 # Data downloader
├── research_flow_shock_quick.py           # Quick sample analysis
├── research_flow_shock_threshold_scan.py  # Threshold scanner
├── research_flow_shock_detector.py        # Full dataset processor
│
└── results/
    ├── threshold_scan_results.csv         # Threshold scan data
    ├── flow_shock_sample_5s.csv           # Events (5s window)
    ├── flow_shock_sample_10s.csv          # Events (10s window)
    ├── flow_shock_sample_15s.csv          # Events (15s window)
    ├── flow_shock_sample_30s.csv          # Events (30s window)
    └── FINDINGS_flow_shock_quick.md       # Initial findings
```

---

## 🚀 Quick Commands

```bash
# Navigate to research folder
cd /home/ubuntu/Projects/skytrade6/flow_shock_research

# Download data (if not already done)
python3 download_trade_data.py --symbol SOLUSDT

# Run quick analysis
python3 research_flow_shock_quick.py --sample-days 7

# Scan thresholds
python3 research_flow_shock_threshold_scan.py

# Full dataset (optional)
python3 research_flow_shock_detector.py
```

---

## 📞 Contact & Version

**Research Team:** Systematic Trading Research  
**Date:** March 2, 2026  
**Version:** 1.0  
**Status:** Phase 1 Complete

---

## 🔗 Related Research

- **FR Flash Scalp** - Settlement-based strategy (funding rate arbitrage)
- **Liquidation Cascade** - Microstructure strategy (liquidation events)
- **Market Making** - Order book imbalance strategies

---

**Last Updated:** March 2, 2026
