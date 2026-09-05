import os
import pandas as pd

raw_path = os.path.join("data", "raw")
processed_path = os.path.join("data", "processed")
os.makedirs(processed_path, exist_ok=True)

print("1. Loading raw tables...")
stores = pd.read_csv(os.path.join(raw_path, "stores.csv"))
features = pd.read_csv(os.path.join(raw_path, "features.csv"))
train = pd.read_csv(os.path.join(raw_path, "train.csv"))

print(f"   - stores: {stores.shape}")
print(f"   - features: {features.shape}")
print(f"   - train: {train.shape}")

# Convert Date to datetime
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

print("\n2. Merging datasets...")
# Step 1: Merge train with stores on Store
df = pd.merge(train, stores, on="Store", how="left")

# Step 2: Merge with features on Store and Date
df = pd.merge(df, features, on=["Store", "Date"], how="left")

# Fix duplicate column name generated from merge (IsHoliday_x, IsHoliday_y)
if "IsHoliday_y" in df.columns:
    df.drop(columns=["IsHoliday_y"], inplace=True)
    df.rename(columns={"IsHoliday_x": "IsHoliday"}, inplace=True)

print(f"   - Merged dataset shape: {df.shape}")

# Save the unified dataset
output_file = os.path.join(processed_path, "merged_sales_data.csv")
df.to_csv(output_file, index=False)
print(f"\n✓ Processed dataset successfully saved at: {output_file}")