import pandas as pd
import os
FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
df = pd.read_parquet(os.path.join(FEAT_DIR, "BERAUSDT_1m.parquet"))
print(df.columns)
