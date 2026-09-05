# 🛒 Walmart Store Sales Forecasting & Analytics System

An end-to-end Machine Learning pipeline and web application designed to forecast weekly departmental sales across 45 Walmart stores using historical retail data, markdown promotions, and macroeconomic factors.

---

## 📌 Project Overview

- **Domain**: Retail Analytics & Time-Series Forecasting
- **Core Model**: Random Forest Regressor
- **Target Metric**: WMAE (Weighted Mean Absolute Error - Official Kaggle Benchmark)
- **Dashboard Stack**: Streamlit, Plotly, SQLite3 Authentication

---

## 📊 Key Results & Evaluation

- **R² Score**: `0.9733` (Explains ~97.3% of sales variance)
- **WMAE (Kaggle)**: `1779.28` (5x penalty on holiday weeks)
- **MAE**: `1753.04`
- **RMSE**: `3596.99`

---

## 🔍 Exploratory Data Analysis (EDA) Highlights

1. **Sales Skewness**: Right-skewed distribution caused by massive spikes during festive weeks.
2. **Holiday Impact**: Holiday weeks consistently produce significantly higher sales volumes compared to standard operational weeks.
3. **Store Categories**: Type A stores dominate volume due to larger physical square footage, followed by Type B and Type C.
4. **Key Feature Drivers**: Department ID and Store Size emerged as the most critical predictors of sales performance.

---

## 🛠️ Architecture & Pipeline

1. `src/data_preprocessing.py`: Joins raw stores, economic features, and train data into a unified dataset.
2. `src/feature_engineering.py`: Temporal feature extraction (Week, Month, Year), markdown zero-fill imputation, and ordinal encoding.
3. `src/train_model.py`: Time-based holdout validation split (prevents lookahead bias) and Random Forest serialization.
4. `src/eda_analysis.py`: Generates all statistical audits and visual plots into `reports/`.
5. `src/evaluate_visualize.py`: Evaluates holdout validation set and tracks actual vs predicted trends.
6. `app.py`: Authenticated multi-tab Streamlit dashboard with KPI tracking, time-series graphs, and interactive What-If scenario simulator.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/NishantKumar947/CODTECH-Task2-Store-Sales-Forecasting.git](https://github.com/NishantKumar947/CODTECH-Task2-Store-Sales-Forecasting.git)
cd CODTECH-Task2-Store-Sales-Forecasting

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch Dashboard
streamlit run app.py
```
