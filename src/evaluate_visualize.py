import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory setup
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# 1. Load Model and Validation Data
print("1. Loading model and validation dataset...")
model = joblib.load(os.path.join("models", "rf_walmart_model.pkl"))
df = pd.read_csv(os.path.join("data", "processed", "featured_sales_data.csv"))
df["Date"] = pd.to_datetime(df["Date"])

val_df = df[df["Date"] >= "2012-05-01"].copy()

features = [
    "Store", "Dept", "Type", "Size", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Year", "Month", "Week", "DayOfWeek"
]

print("2. Generating validation predictions...")
val_df["Predicted_Sales"] = model.predict(val_df[features])

# Save comparison dataset
pred_output = os.path.join("data", "processed", "validation_predictions.csv")
val_df[["Store", "Dept", "Date", "Weekly_Sales", "Predicted_Sales"]].to_csv(pred_output, index=False)
print(f"✓ Predictions saved to: {pred_output}")

# 3. Chart 1: Actual vs Predicted Over Time (Aggregated)
print("3. Generating Actual vs Predicted Trend Chart...")
weekly_agg = val_df.groupby("Date")[["Weekly_Sales", "Predicted_Sales"]].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(weekly_agg["Date"], weekly_agg["Weekly_Sales"], label="Actual Sales", marker="o", color="#1f77b4", linewidth=2)
plt.plot(weekly_agg["Date"], weekly_agg["Predicted_Sales"], label="Predicted Sales", marker="s", color="#ff7f0e", linestyle="--", linewidth=2)
plt.title("Total Weekly Sales: Actual vs Predicted (Validation Period)", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Sales ($)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

chart1_path = os.path.join(reports_dir, "actual_vs_predicted.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"✓ Saved: {chart1_path}")

# 4. Chart 2: Feature Importance
print("4. Generating Feature Importance Chart...")
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Random Forest: Feature Importance", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()

chart2_path = os.path.join(reports_dir, "feature_importance.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"✓ Saved: {chart2_path}")

print("\n✓ All evaluation artifacts generated successfully!")