# XS-6 Deep Analysis — Time-to-move, Skew, Sanity

Generated: 2026-03-03 15:14 UTC

## Q1: Time-to-move Distribution

### S06_compress_vol
- Signals (test): 1300
- Breach rate (ATR k=3.0, 24h): 0.0%

### S07_compress_oi
- Signals (test): 1873
- Breach rate (ATR k=3.0, 24h): 3.1%
- Median time to breach: 610min (10.2h)
- P25/P75: 558min / 1274min
  - 0-1h: 0.0%
  - 1-3h: 0.0%
  - 3-6h: 1.7%
  - 6-12h: 53.4%
  - 12-24h: 44.8%
- Direction at breach: 100% up, 0% down

### S01_fund_high
- Signals (test): 4893
- Breach rate (ATR k=3.0, 24h): 3.2%
- Median time to breach: 976min (16.3h)
- P25/P75: 633min / 1392min
  - 0-1h: 0.0%
  - 1-3h: 0.0%
  - 3-6h: 0.0%
  - 6-12h: 30.4%
  - 12-24h: 69.0%
- Direction at breach: 84% up, 16% down

### S03_oi_surge
- Signals (test): 12517
- Breach rate (ATR k=3.0, 24h): 1.9%
- Median time to breach: 948min (15.8h)
- P25/P75: 757min / 1374min
  - 0-1h: 0.8%
  - 1-3h: 0.0%
  - 3-6h: 8.5%
  - 6-12h: 15.3%
  - 12-24h: 73.7%
- Direction at breach: 96% up, 4% down

### S09_stall_fund
- Signals (test): 4815
- Breach rate (ATR k=3.0, 24h): 0.9%
- Median time to breach: 548min (9.1h)
- P25/P75: 135min / 1387min
  - 0-1h: 0.0%
  - 1-3h: 35.6%
  - 3-6h: 6.7%
  - 6-12h: 13.3%
  - 12-24h: 44.4%
- Direction at breach: 89% up, 11% down

### S10_thin_move
- Signals (test): 4544
- Breach rate (ATR k=3.0, 24h): 0.9%
- Median time to breach: 990min (16.5h)
- P25/P75: 639min / 1068min
  - 0-1h: 0.0%
  - 1-3h: 0.0%
  - 3-6h: 2.4%
  - 6-12h: 28.6%
  - 12-24h: 69.0%
- Direction at breach: 100% up, 0% down

## Q2: Direction Skew

### S06_compress_vol
  - 12h: P(up>200bp)=0.120, P(down>200bp)=0.381, skew_ratio=28571.93, uplift_up=0.51x, uplift_down=1.38x
  - 24h: P(up>200bp)=0.251, P(down>200bp)=0.402, skew_ratio=67815.38, uplift_up=0.74x, uplift_down=1.13x

### S07_compress_oi
  - 12h: P(up>200bp)=0.238, P(down>200bp)=0.344, skew_ratio=89895.46, uplift_up=0.96x, uplift_down=1.06x
  - 24h: P(up>200bp)=0.313, P(down>200bp)=0.398, skew_ratio=102304.70, uplift_up=1.09x, uplift_down=1.00x

### S01_fund_high
  - 12h: P(up>200bp)=0.254, P(down>200bp)=0.407, skew_ratio=62502.25, uplift_up=0.79x, uplift_down=1.33x
  - 24h: P(up>200bp)=0.286, P(down>200bp)=0.506, skew_ratio=117187.88, uplift_up=0.74x, uplift_down=1.34x

### S03_oi_surge
  - 12h: P(up>200bp)=0.311, P(down>200bp)=0.342, skew_ratio=1.32, uplift_up=1.25x, uplift_down=1.10x
  - 24h: P(up>200bp)=0.339, P(down>200bp)=0.410, skew_ratio=1.95, uplift_up=1.16x, uplift_down=1.07x

### S09_stall_fund
  - 12h: P(up>200bp)=0.216, P(down>200bp)=0.347, skew_ratio=20734.17, uplift_up=0.83x, uplift_down=1.13x
  - 24h: P(up>200bp)=0.244, P(down>200bp)=0.480, skew_ratio=1.18, uplift_up=0.80x, uplift_down=1.25x

### S10_thin_move
  - 12h: P(up>200bp)=0.222, P(down>200bp)=0.272, skew_ratio=1.04, uplift_up=0.90x, uplift_down=0.89x
  - 24h: P(up>200bp)=0.295, P(down>200bp)=0.335, skew_ratio=1.18, uplift_up=1.03x, uplift_down=0.88x

## Q3: Shuffle Sanity Test

If real uplift is genuine, it should exceed p99 of shuffled distribution for most symbols.

- **S06_compress_vol** / big_A_k3.0_24h: real_uplift=0.00, shuffle_mean=0.02, shuffle_p99=0.10, exceeds_p99=0% of symbols
- **S06_compress_vol** / big_B_24h: real_uplift=1.11, shuffle_mean=0.91, shuffle_p99=1.36, exceeds_p99=29% of symbols
- **S07_compress_oi** / big_A_k3.0_24h: real_uplift=2.50, shuffle_mean=0.85, shuffle_p99=1.81, exceeds_p99=9% of symbols
- **S07_compress_oi** / big_B_24h: real_uplift=0.89, shuffle_mean=0.94, shuffle_p99=1.41, exceeds_p99=24% of symbols
- **S01_fund_high** / big_A_k3.0_24h: real_uplift=3.51, shuffle_mean=1.27, shuffle_p99=3.23, exceeds_p99=12% of symbols
- **S01_fund_high** / big_B_24h: real_uplift=1.50, shuffle_mean=1.53, shuffle_p99=1.75, exceeds_p99=50% of symbols
- **S03_oi_surge** / big_A_k3.0_24h: real_uplift=0.58, shuffle_mean=0.93, shuffle_p99=1.66, exceeds_p99=2% of symbols
- **S03_oi_surge** / big_B_24h: real_uplift=1.39, shuffle_mean=1.29, shuffle_p99=1.49, exceeds_p99=27% of symbols
- **S09_stall_fund** / big_A_k3.0_24h: real_uplift=0.34, shuffle_mean=0.30, shuffle_p99=0.95, exceeds_p99=2% of symbols
- **S09_stall_fund** / big_B_24h: real_uplift=1.10, shuffle_mean=1.04, shuffle_p99=1.34, exceeds_p99=27% of symbols
- **S10_thin_move** / big_A_k3.0_24h: real_uplift=0.41, shuffle_mean=0.40, shuffle_p99=1.41, exceeds_p99=2% of symbols
- **S10_thin_move** / big_B_24h: real_uplift=0.75, shuffle_mean=0.78, shuffle_p99=1.12, exceeds_p99=6% of symbols

## Q4: OI Leakage Trap

If OI is causal, leaking future OI should boost uplift significantly.

- **S03_oi_surge** / big_A_k3.0_24h: normal=0.58, leaked=0.62, boost=1.07x
- **S03_oi_surge** / big_B_24h: normal=1.39, leaked=1.40, boost=1.00x
- **S07_compress_oi** / big_A_k3.0_24h: normal=2.50, leaked=2.56, boost=1.02x
- **S07_compress_oi** / big_B_24h: normal=0.89, leaked=0.90, boost=1.01x
- **S08_stall_oi** / big_A_k3.0_24h: normal=0.04, leaked=0.01, boost=0.36x
- **S08_stall_oi** / big_B_24h: normal=1.27, leaked=1.25, boost=0.98x
