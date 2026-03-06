import pandas as pd
df = pd.read_csv("full_history_filtered_trades.csv")

def run_sim(risk_pct):
    equity = 1000.0
    for _, row in df.iterrows():
        atr = row['atr_pct'] if row['atr_pct'] > 0 else 0.05
        size = min(risk_pct / atr, 10.0) # Cap at 10x
        equity += equity * size * row['net_ret']
    return (equity - 1000) / 1000 * 100

print(f"Risk 1%: {run_sim(0.01):.1f}%")
print(f"Risk 2%: {run_sim(0.02):.1f}%")
print(f"Risk 5%: {run_sim(0.05):.1f}%")
print(f"Risk 10%: {run_sim(0.10):.1f}%")
