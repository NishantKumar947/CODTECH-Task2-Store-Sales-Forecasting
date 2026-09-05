import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
processed_path = os.path.join("data", "processed")
models_path = os.path.join("models")
os.makedirs(models_path, exist_ok=True)

input_file = os.path.join(processed_path, "featured_sales_data.csv")

print("1. Loading featured data...")
df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# 2. Time-based Train-Validation Split (No data leakage)
# Walmart train set spans 2010-02-05 to 2012-10-26
# Train: 2010 to mid-2012 | Validation: last ~4 months
split_date = "2012-05-01"
train_df = df[df["Date"] < split_date].copy()
val_df = df[df["Date"] >= split_date].copy()

features = [
    "Store", "Dept", "Type", "Size", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Year", "Month", "Week", "DayOfWeek"
]
target = "Weekly_Sales"

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]

print(f"   - Training samples: {len(X_train)}")
print(f"   - Validation samples: {len(X_val)}")

# 3. Train Random Forest Model
print("\n2. Training Random Forest Regressor (this may take 1-2 mins)...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 4. Predictions & Evaluation
print("\n3. Evaluating Model Performance...")
preds = rf.predict(X_val)

# Official Walmart Competition Metric: WMAE (Weighted MAE)
# Holiday weeks carry weight of 5, normal weeks carry weight of 1
weights = val_df["IsHoliday"].apply(lambda x: 5 if x == 1 else 1).values
wmae = np.sum(weights * np.abs(y_val.values - preds)) / np.sum(weights)
mae = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)

print("\n" + "="*35)
print("       EVALUATION METRICS       ")
print("="*35)
print(f"  WMAE (Kaggle Metric) : {wmae:.2f}")
print(f"  MAE                  : {mae:.2f}")
print(f"  RMSE                 : {rmse:.2f}")
print(f"  R2 Score             : {r2:.4f}")
print("="*35)

# 5. Save Model and Artifacts
model_file = os.path.join(models_path, "rf_walmart_model.pkl")
joblib.dump(rf, model_file)
print(f"\n✓ Trained model saved to: {model_file}")