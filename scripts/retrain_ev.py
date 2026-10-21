"""
Retrain the EV grid model from the feature CSV.
"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'day_of_week', 'month', 'week_of_year',
    'active_chargers', 'unique_vehicles',
]

df = pd.read_csv('data/raw/ev_grid_features.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[FEATURES].values
y_train = train['total_energy_kwh'].values
X_test = test[FEATURES].values
y_test = test['total_energy_kwh'].values

print(f"Train: {len(train)} rows | Test: {len(test)} rows")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))

print(f"Train MAE: {train_mae:.2f} kWh")
print(f"Test MAE:  {test_mae:.2f} kWh")

with open('models/ev_grid_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Saved models/ev_grid_rf.pkl")

# Sanity: reload and predict
with open('models/ev_grid_rf.pkl', 'rb') as f:
    reloaded = pickle.load(f)
preds = reloaded.predict(X_test[:5])
print(f"Reload test — first 5 predictions: {preds.round(2)}")