import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --------------------------------
# Page Config & Title
# --------------------------------
st.set_page_config(page_title="🌾 Food Price Prediction", layout="wide")
st.title("🌾 Food Price Prediction App")
st.markdown("Analyze and forecast agricultural product prices using Machine Learning.")

# --------------------------------
# Load Dataset
# --------------------------------
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

# --------------------------------
# Train Models
# --------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor()
}

results = {}
best_model = None
best_model_name = None
best_r2 = float('-inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "R2 Score": r2}

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results).T.sort_values(by="R2 Score", ascending=False)

# Save best model
joblib.dump(best_model, "food_price_model.pkl")

# --------------------------------
# TABS LAYOUT
# --------------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Model Results", "🎯 Prediction"])

# --------------------------------
# Tab 1: EDA
# --------------------------------
with tab1:
    st.subheader("📌 Dataset Overview")
    st.dataframe(df.head())

    st.markdown("### Product Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Avg_price'], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.markdown("### Box Plot (Outliers)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Avg_price'], ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 📈 Price Trend of 'Rice'")
    rice_df = df[df['Product'] == label_encoders['Product'].transform(['rice'])[0]]
    fig3 = px.line(rice_df, x='Date', y='Avg_price', title='Rice Price Over Time')
    st.plotly_chart(fig3)

# --------------------------------
# Tab 2: Model Results
# --------------------------------
with tab2:
    st.subheader("🧠 Model Performance")
    st.dataframe(results_df)

    st.markdown("### Feature Importance (Top Model)")
    importances = best_model.feature_importances_
    feat_importances = pd.Series(importances, index=X_train.columns).sort_values()

    fig4, ax4 = plt.subplots()
    sns.barplot(x=feat_importances, y=feat_importances.index, ax=ax4)
    st.pyplot(fig4)

    st.markdown("### Actual vs. Predicted")
    y_pred = best_model.predict(X_test)
    fig5, ax5 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax5)
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax5.set_xlabel('Actual')
    ax5.set_ylabel('Predicted')
    st.pyplot(fig5)

# --------------------------------
# Tab 3: Prediction
# --------------------------------
with tab3:
    st.subheader("🎯 Predict Average Food Price")

    product_list = label_encoders['Product'].classes_
    selected_product = st.selectbox("Select Product", product_list)
    year = st.slider("Year", 2007, 2030, 2025)
    month = st.slider("Month", 1, 12, 4)
    day = st.slider("Day", 1, 31, 15)

    encoded_product = label_encoders['Product'].transform([selected_product])[0]
    user_input = np.array([[encoded_product, year, month, day]])

    prediction = best_model.predict(user_input)[0]
    st.success(f"💰 Predicted Price for **{selected_product}** on **{year}-{month:02}-{day:02}** is **{prediction:.2f}**")

    # Optional download
    with open("food_price_model.pkl", "rb") as f:
        st.download_button("📥 Download Trained Model", f, "food_price_model.pkl")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Made with ❤️ by SARAH KIHEMBO · Powered by Streamlit & ML")
