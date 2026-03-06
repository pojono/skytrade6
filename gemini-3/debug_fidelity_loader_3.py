import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

symbol = 'SUIUSDT'
start_date = "2025-01-01"

try:
    from full_universe_scan_fidelity import load_data
    h, m = load_data('SUIUSDT')
    print("Hourly size:", len(h) if h is not None else None)
except Exception as e:
    import traceback
    traceback.print_exc()

