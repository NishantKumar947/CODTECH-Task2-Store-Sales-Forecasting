import os
import pandas as pd
import numpy as np

processed_path = os.path.join("data", "processed")
input_file = os.path.join(processed_path, "merged_sales_data.csv")

print("1. Loading merged dataset...")
df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

print("\n2. Extracting Date & Calendar features...")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

print("3. Handling Missing Values...")
# MarkDown1-5 me NaN ka matlab promotion run nahi hua (replace with 0)
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
for col in markdown_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# CPI aur Unemployment me agar minor missing ho to forward fill
df["CPI"] = df["CPI"].ffill()
df["Unemployment"] = df["Unemployment"].ffill()

print("4. Encoding Categorical Features...")
# Store Type (A, B, C) mapped to ordinal numeric
type_map = {"A": 3, "B": 2, "C": 1}
df["Type"] = df["Type"].map(type_map)

# IsHoliday boolean to integer (0 or 1)
df["IsHoliday"] = df["IsHoliday"].astype(int)

# Sort strictly by Store, Dept, and Date for time-series consistency
df = df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

output_file = os.path.join(processed_path, "featured_sales_data.csv")
df.to_csv(output_file, index=False)

print(f"\n✓ Feature Engineering Complete! Saved at: {output_file}")
print(f"Shape: {df.shape}")
print("Sample Columns:", list(df.columns))