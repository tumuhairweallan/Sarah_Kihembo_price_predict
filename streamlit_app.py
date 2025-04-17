import pandas as pd
import streamlit as st
import numpy as np
import joblib
import os
import time
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# --- Page Configuration ---
st.set_page_config(page_title="🌾 Food Price Prediction", layout="centered")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #00796B;
    }
    .stButton>button {
        background-color: #00796B;
        color: white;
        font-size: 18px;
        padding: 10px;
        width: 100%;
        transition: 0.3s;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #004D40;
        transform: scale(1.05);
    }
    .stTitle {
        text-align: center;
        color: #004D40;
        font-size: 28px;
    }
    .stCaption {
        text-align: center;
        color: gray;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌾 Food Price Prediction App")
st.markdown("This app predicts the future prices of farm produce (per 50Kg bag or per tuber). It provides farmers with price estimates to help them plan and budget more effectively.")

# --- Load Dataset ---
file_path = "world_bank_cleaned.csv"
df = pd.read_csv(file_path)

# Preprocess Dataset
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df['Avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
df = df.drop(columns=['Country', 'Region', 'Market', 'Open', 'High', 'Low', 'Close', 'Currency'])
df = df.iloc[1:].reset_index(drop=True)

# Encode Categorical Columns
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Train and Load Model ---
@st.cache_resource
def train_model():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }

    best_model = None
    best_r2 = float('-inf')

    X = df.drop(columns=["Date", "Avg_price"])
    y = df["Avg_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    joblib.dump(best_model, "food_price_model.pkl")
    return best_model

best_model = train_model()

# --- User Input Section ---
st.subheader("🎯 Predict Food Price")

product_list = label_encoders['Product'].classes_
selected_product = st.selectbox("Select Product", product_list)

# Dropdowns for Year, Month, Day
year = st.selectbox("Select Year", options=range(int(df["Year"].min()), int(df["Year"].max()) + 1), index=18)
month = st.selectbox("Select Month", options=list(range(1, 13)), index=3)
day = st.selectbox("Select Day", options=list(range(1, 32)), index=14)

encoded_product = label_encoders['Product'].transform([selected_product])[0]
user_input = np.array([[encoded_product, year, month, day]])

# --- Prediction Button ---
if st.button("Predict Food Price"):
    with st.spinner("⚙️ Predicting... Please wait!"):
        time.sleep(2)  # Simulated delay for smooth UX
        prediction = best_model.predict(user_input)[0]

    st.success(f"💰 Predicted Price for **{selected_product}** on **{year}-{month:02}-{day:02}** is **${prediction:.2f}**")

    # --- Display Insights ---
    st.subheader("📈 Insights from Data")

    # Product Price Distribution
    st.markdown("### Distribution of Prices")
    fig1 = px.histogram(df, x='Avg_price', title="Price Distribution")
    st.plotly_chart(fig1)

    # Product Trend Over Time
    st.markdown(f"### {selected_product} Price Trend Over Time")
    product_encoded = label_encoders['Product'].transform([selected_product])[0]
    filtered_df = df[df['Product'] == product_encoded]
    fig2 = px.line(filtered_df, x='Date', y='Avg_price', title=f"{selected_product} Price Trend")
    st.plotly_chart(fig2)

# --- Footer ---
st.markdown("---")
st.caption("NOTICE: The predictions are based on historical data and may not reflect real-world prices. Hence may not be 100% accurate.")
st.caption("Developed by KIHEMBO SARAH")
