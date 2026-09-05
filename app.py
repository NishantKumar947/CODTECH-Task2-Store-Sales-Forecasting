import os
import sqlite3
import hashlib
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# ----------------- DATABASE AUTH SETUP -----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    if data and data[0] == hash_password(password):
        return True
    return False

init_db()

# Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# ----------------- AUTHENTICATION INTERFACE -----------------
if not st.session_state["logged_in"]:
    st.markdown("<h2 style='text-align: center;'>🛒 Walmart Forecasting System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please login or register to access the forecasting dashboard.</p>", unsafe_allow_html=True)

    auth_choice = st.radio("Select Action", ["Login", "Sign Up"], horizontal=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if auth_choice == "Login":
            st.subheader("Account Login")
            login_user = st.text_input("Username")
            login_pass = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                if authenticate_user(login_user, login_pass):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user
                    st.success(f"Welcome back, {login_user}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        elif auth_choice == "Sign Up":
            st.subheader("Create New Account")
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")

            if st.button("Sign Up", use_container_width=True):
                if not new_user or not new_pass:
                    st.warning("Please fill in all fields.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    if add_user(new_user, new_pass):
                        st.success("Account created successfully! Please proceed to Login.")
                    else:
                        st.error("Username already exists. Please choose a different one.")
    st.stop()

# ----------------- MAIN FORECASTING DASHBOARD -----------------
# Logout button in sidebar
st.sidebar.markdown(f"👤 **Logged in as:** `{st.session_state['username']}`")
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

st.sidebar.markdown("---")

# Data & Model Loader
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join("data", "processed", "validation_predictions.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load(os.path.join("models", "rf_walmart_model.pkl"))

pred_df = load_data()
model = load_model()

# Navigation Tabs
tab_forecast, tab_eda = st.tabs(["📊 Sales Forecast & Simulation", "🔍 Exploratory Data Analysis (EDA)"])

# ----------------- TAB 1: FORECASTING -----------------
with tab_forecast:
    st.title("🛒 Walmart Store Sales Forecasting Dashboard")
    st.markdown("Interactive forecasting & performance tracking across stores and departments.")

    # Filters
    st.sidebar.header("🔍 Filter Options")
    stores = sorted(pred_df["Store"].unique())
    selected_store = st.sidebar.selectbox("Select Store Number", stores, index=0)

    depts = sorted(pred_df[pred_df["Store"] == selected_store]["Dept"].unique())
    selected_dept = st.sidebar.selectbox("Select Department Number", depts, index=0)

    store_dept_data = pred_df[(pred_df["Store"] == selected_store) & (pred_df["Dept"] == selected_dept)].sort_values("Date")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_actual = store_dept_data["Weekly_Sales"].sum()
    total_pred = store_dept_data["Predicted_Sales"].sum()
    dept_mae = np.mean(np.abs(store_dept_data["Weekly_Sales"] - store_dept_data["Predicted_Sales"]))
    pct_error = (abs(total_actual - total_pred) / total_actual) * 100 if total_actual != 0 else 0

    with col1:
        st.metric("Total Actual Sales", f"${total_actual:,.2f}")
    with col2:
        st.metric("Total Predicted Sales", f"${total_pred:,.2f}")
    with col3:
        st.metric("Department MAE", f"${dept_mae:,.2f}")
    with col4:
        st.metric("Aggregate Error Rate", f"{pct_error:.2f}%")

    st.markdown("---")

    # Chart
    st.subheader(f"📈 Weekly Sales Trend: Store {selected_store} - Dept {selected_dept}")
    if not store_dept_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=store_dept_data["Date"], 
            y=store_dept_data["Weekly_Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#1f77b4", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=store_dept_data["Date"], 
            y=store_dept_data["Predicted_Sales"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color="#ff7f0e", width=3, dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # What-If Simulator
    st.subheader("🔮 What-If Scenario Predictor")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p_type = st.selectbox("Store Type", options=[3, 2, 1], format_func=lambda x: {3:"Type A", 2:"Type B", 1:"Type C"}[x])
            p_size = st.number_input("Store Size (sq ft)", value=150000, step=5000)
        with c2:
            p_temp = st.slider("Temperature (°F)", min_value=-10.0, max_value=110.0, value=65.0)
            p_fuel = st.slider("Fuel Price ($)", min_value=2.0, max_value=5.0, value=3.5)
        with c3:
            p_cpi = st.number_input("CPI", value=211.0, step=0.5)
            p_unemp = st.slider("Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=7.5)
        with c4:
            p_holiday = st.selectbox("Is Holiday Week?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            p_week = st.slider("Week of Year", min_value=1, max_value=52, value=45)

        submit_btn = st.form_submit_button("Predict Weekly Sales")

        if submit_btn:
            input_data = pd.DataFrame([{
                "Store": selected_store, "Dept": selected_dept, "Type": p_type, "Size": p_size,
                "IsHoliday": p_holiday, "Temperature": p_temp, "Fuel_Price": p_fuel,
                "CPI": p_cpi, "Unemployment": p_unemp,
                "MarkDown1": 0.0, "MarkDown2": 0.0, "MarkDown3": 0.0, "MarkDown4": 0.0, "MarkDown5": 0.0,
                "Year": 2012, "Month": int(p_week / 4.3) + 1, "Week": p_week, "DayOfWeek": 4
            }])
            
            feature_order = [
                "Store", "Dept", "Type", "Size", "IsHoliday",
                "Temperature", "Fuel_Price", "CPI", "Unemployment",
                "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
                "Year", "Month", "Week", "DayOfWeek"
            ]
            predicted_val = model.predict(input_data[feature_order])[0]
            st.success(f"🎯 Estimated Weekly Sales: **${predicted_val:,.2f}**")

# ----------------- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) -----------------
with tab_eda:
    st.title("🔍 Exploratory Data Analysis & Feature Insights")
    st.markdown("Detailed graphical view of historical sales patterns, environmental factors, and store behaviors.")

    col_a, col_b = st.columns(2)

    with col_a:
        if os.path.exists("reports/eda_sales_distribution.png"):
            st.subheader("1. Sales Distribution & Skewness")
            st.image("reports/eda_sales_distribution.png", use_container_width=True)
            st.caption("Sales follow an extreme right-skewed distribution driven by peak festive weeks.")

        if os.path.exists("reports/eda_store_types.png"):
            st.subheader("3. Store Types & Size Analysis")
            st.image("reports/eda_store_types.png", use_container_width=True)
            st.caption("Type A stores dominate sales volume due to larger physical square footage.")

        if os.path.exists("reports/feature_importance.png"):
            st.subheader("5. Model Feature Importance")
            st.image("reports/feature_importance.png", use_container_width=True)
            st.caption("Department ID, Store Size, and Store number drive the majority of prediction weight.")

    with col_b:
        if os.path.exists("reports/eda_holiday_impact.png"):
            st.subheader("2. Holiday vs Regular Weeks Impact")
            st.image("reports/eda_holiday_impact.png", use_container_width=True)
            st.caption("Significant sales spikes observed across key holidays (Thanksgiving, Christmas).")

        if os.path.exists("reports/eda_correlation_matrix.png"):
            st.subheader("4. Feature Correlation Matrix")
            st.image("reports/eda_correlation_matrix.png", use_container_width=True)
            st.caption("Correlation between economic indicators (CPI, Fuel Price, Unemployment) and Sales.")

        if os.path.exists("reports/actual_vs_predicted.png"):
            st.subheader("6. Aggregate Actual vs Predicted Trend")
            st.image("reports/actual_vs_predicted.png", use_container_width=True)
            st.caption("Holdout validation tracking confirms high model generalization with minimal drift.")