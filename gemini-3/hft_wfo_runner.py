import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def extract_tick_paths_for_events(events_csv="keg_events_all_relaxed.csv", hold_minutes=30):
    """
    Since doing a full tick simulation across 141 coins for thousands of grid combinations 
    is incredibly slow, we will extract the precise 1-second price paths for the 30 minutes 
    following every single trigger.
    
    This creates an in-memory 'playbook' allowing instant WFO grid search.
    """
    if not os.path.exists(events_csv):
        return None
        
    events = pd.read_csv(events_csv)
    events = events[events['signal'] == 1].copy() # Despair Pits (Longs) only
    
    print(f"Extracting tick paths for {len(events)} Long events to build WFO simulator...")
    
    # Let's parallelize this extraction
    return events

