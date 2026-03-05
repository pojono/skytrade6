import pandas as pd

# Mock data to verify rolling logic for forward returns and forward funding
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=10, freq='D'),
    'close': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'fundingRate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
})

print("Original logic from v38:")
# df['fwd_ret_5d'] = df['close'].shift(-5) / df['close'] - 1
df['fwd_funding_original'] = df['fundingRate'].shift(-1).rolling(5).sum()

print("Correct forward logic (i+1 to i+5):")
# Rolling looks backward. So a rolling(5) at i+5 covers i+1 to i+5. 
# Therefore, we want rolling(5).sum() shifted backwards by 5, which means shifted by -5.
df['fwd_funding_correct'] = df['fundingRate'].rolling(5).sum().shift(-5)

print(df[['timestamp', 'fundingRate', 'fwd_funding_original', 'fwd_funding_correct']])

print("\n--- Explanation of the bug in v38 ---")
print("If we execute at the close of Jan 1 (index 0), we hold for 5 days and exit at the close of Jan 6 (index 5).")
print("The funding we collect should be the funding on Jan 2, 3, 4, 5, 6.")
print(f"Sum of Jan 2 to Jan 6 funding: {0.02 + 0.03 + 0.04 + 0.05 + 0.06:.2f}")
print(f"Original logic gave: {df.loc[0, 'fwd_funding_original']} (which is NaN because rolling needs 5 prior days, including days before Jan 1)")
print(f"Correct logic gives: {df.loc[0, 'fwd_funding_correct']}")

