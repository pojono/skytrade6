import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("ml_dataset.csv")

# We want to predict if a trade will be highly profitable.
# Let's say our target is a trade that yields > 2% (200 bps) return over 24 hours.

# Target 1: Will it pump > 2%? (For long trades)
df['target_long'] = (df['fwd_ret_24h'] > 0.02).astype(int)

# Target 2: Will it dump > 2%? (For short trades)
df['target_short'] = (df['fwd_ret_24h'] < -0.02).astype(int)

# Features we engineered
features = ['oi_z', 'fr_z', 'count_z', 'vol_z', 'taker_z', 'prem_z', 'mom_4h', 'mom_24h']

# Train Model for Longs
print("--- Training Long Classifier ---")
X = df[features]
y = df['target_long']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Time series split

rf_long = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf_long.fit(X_train, y_train)

y_pred_long = rf_long.predict(X_test)
print(classification_report(y_test, y_pred_long))

print("Feature Importances (Longs):")
for feat, imp in zip(features, rf_long.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# Train Model for Shorts
print("\n--- Training Short Classifier ---")
y_short = df['target_short']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_short, test_size=0.2, random_state=42, shuffle=False)

rf_short = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf_short.fit(X_train_s, y_train_s)

y_pred_short = rf_short.predict(X_test_s)
print(classification_report(y_test_s, y_pred_short))

print("Feature Importances (Shorts):")
for feat, imp in zip(features, rf_short.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

