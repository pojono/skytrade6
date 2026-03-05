import pandas as pd

df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-1/dip_buyer_trades.csv')
print(f"Total trades: {len(df)}")
print("Trades per symbol (top 10):")
print(df['symbol'].value_counts().head(10))

print("\nFirst 5 trades:")
print(df.head(5).to_string())

# Check for zero hold time
print("\nHold time stats:")
print(df['hold_mins'].describe())

