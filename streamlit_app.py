import pandas as pd
import streamlit as st
import numpy as np
import joblib
import warnings
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import time
from streamlit_lottie import st_lottie
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(page_title="🌾 Food Price Predictor", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #D4422B;
        transform: scale(1.05);
    }
    .stTitle {
        text-align: center;
        color: #4CAF50;
    }
    .stCaption {
        text-align: center;
        color: gray;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌾 Food Price Predicting App")
st.markdown("Predict the future prices of farm produces. **Note:** Prediction is not 100% accurate.")

# Load Lottie Animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottie("prediction_animation.json")
st_lottie(lottie_animation, speed=1, height=200, key="header_animation")

# Load Dataset
file_path = "world_bank_cleaned.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df['Avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Clean up and drop unnecessary columns
df = df.drop(columns=['Country', 'Region', 'Market', 'Open', 'High', 'Low', 'Close', 'Currency'])
df = df.iloc[1:].reset_index(drop=True)

# Encode categorical
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop(columns=["Date", "Avg_price"])
y = df["Avg_price"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor()
}

best_model = None
best_r2 = float('-inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

joblib.dump(best_model, "food_price_model.pkl")

# User Input
st.subheader("🎯 Predict Food Price")
product_list = label_encoders['Product'].classes_
selected_product = st.selectbox("Select Product", product_list)
year = st.slider("Year", 2007, 2030, 2025)
month = st.slider("Month", 1, 12, 4)
day = st.slider("Day", 1, 31, 15)

encoded_product = label_encoders['Product'].transform([selected_product])[0]
user_input = np.array([[encoded_product, year, month, day]])

# Prediction Button
if st.button("Predict Food Price"):
    with st.spinner("⚙️ Predicting... Please wait!"):
        time.sleep(2)  # Simulated delay
        prediction = best_model.predict(user_input)[0]
    
    st.success(f"💰 Predicted Price for **{selected_product}** on **{year}-{month:02}-{day:02}** is **{prediction:.2f}**")

    # Show Charts after Prediction
    st.subheader("📈 Price Trends & Insights")

    # Distribution Chart
    st.markdown("### Product Price Distribution")
    fig1 = px.histogram(df, x='Avg_price', title="Price Distribution")
    st.plotly_chart(fig1)

    # Price Trend Over Time
    st.markdown(f"### 📊 Price Trend for {selected_product}")
    product_encoded = label_encoders['Product'].transform([selected_product])[0]
    filtered_df = df[df['Product'] == product_encoded]
    fig2 = px.line(filtered_df, x='Date', y='Avg_price', title=f"{selected_product} Price Over Time")
    st.plotly_chart(fig2)

# Footer
st.markdown("---")
st.caption("Made with ❤️ · Powered by Streamlit & Machine Learning")