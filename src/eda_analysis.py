import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("1. Loading merged dataset for EDA...")
df = pd.read_csv(os.path.join("data", "processed", "merged_sales_data.csv"))
df["Date"] = pd.to_datetime(df["Date"])

sns.set_theme(style="whitegrid")

# 1. Plot: Weekly Sales Distribution
print("2. Generating Weekly Sales Distribution Plot...")
plt.figure(figsize=(10, 5))
sns.histplot(df["Weekly_Sales"], bins=50, kde=True, color="#1f77b4")
plt.title("Distribution of Weekly Sales", fontsize=14, fontweight="bold")
plt.xlabel("Weekly Sales ($)")
plt.ylabel("Frequency")
plt.xlim(0, 100000)
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "eda_sales_distribution.png"), dpi=300)
plt.close()

# 2. Plot: Holiday vs Non-Holiday Sales
print("3. Generating Holiday Impact Analysis Plot...")
plt.figure(figsize=(8, 5))
sns.barplot(x="IsHoliday", y="Weekly_Sales", data=df, palette="coolwarm", estimator="mean")
plt.title("Average Weekly Sales: Non-Holiday (0) vs Holiday (1)", fontsize=14, fontweight="bold")
plt.xlabel("Is Holiday Week")
plt.ylabel("Average Sales ($)")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "eda_holiday_impact.png"), dpi=300)
plt.close()

# 3. Plot: Correlation Heatmap
print("4. Generating Correlation Heatmap...")
plt.figure(figsize=(12, 8))
numeric_cols = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", cbar=True, square=True)
plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "eda_correlation_matrix.png"), dpi=300)
plt.close()

# 4. Plot: Sales by Store Type
print("5. Generating Store Type Comparison Plot...")
plt.figure(figsize=(8, 5))
sns.boxplot(x="Type", y="Weekly_Sales", data=df, palette="Set2", showfliers=False)
plt.title("Weekly Sales by Store Type (A, B, C)", fontsize=14, fontweight="bold")
plt.xlabel("Store Type")
plt.ylabel("Weekly Sales ($)")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "eda_store_types.png"), dpi=300)
plt.close()

print("\n✓ SUCCESS: All 4 EDA charts generated and saved to reports/ folder!")