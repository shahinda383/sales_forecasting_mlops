
# ================================================================
# 🌟 Stage 5: Streamlit Business Forecasting Dashboard
# ================================================================
# Author: Shahinda | Project: Sales Forecasting & Optimization
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
from io import BytesIO

# ===============================
# ⚙ Page Configuration
# ===============================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ===============================
# 🎨 Custom CSS for Aesthetics
# ===============================
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stDownloadButton>button {
        background-color: #00b894;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .metric-container {
        text-align: center;
        background-color: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# 🚀 Load Model and Scaler
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ===============================
# 📂 Upload Section
# ===============================
st.title("📊 Sales Forecasting & Business Insights Dashboard")
st.write("Upload your *Business Impact CSV* to generate forecast and insights automatically.")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

# ===============================
# 📈 Process Data if Uploaded
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(inplace=True)

    # Feature Engineering
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year

    # Select required features
    features = ['Revenue_Current', 'Revenue_Change', 'Growth_%', 'Month', 'Quarter', 'Year']
    X = df[features]

    # Scale data
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df['Predicted_Revenue'] = predictions
    df['Forecast_Error'] = np.abs(df['Revenue_Optimized'] - df['Predicted_Revenue'])
    df['Error_%'] = np.round((df['Forecast_Error'] / df['Revenue_Optimized']) * 100, 2)

    st.success("✅ Forecast generated successfully!")

    # ===============================
    # 📊 KPIs
    # ===============================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Average Actual Revenue", f"${df['Revenue_Optimized'].mean():,.2f}")
    with col2:
        st.metric("💰 Average Predicted Revenue", f"${df['Predicted_Revenue'].mean():,.2f}")
    with col3:
        st.metric("🎯 Average Error %", f"{df['Error_%'].mean():.2f}%")

    # ===============================
    # 📆 Trend Visualization
    # ===============================
    st.subheader("📉 Revenue Forecast Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Revenue_Optimized'], label='Actual', color='blue')
    ax.plot(df['Date'], df['Predicted_Revenue'], label='Predicted', color='orange', linestyle='--')
    plt.title("Actual vs Predicted Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    st.pyplot(fig)

    # ===============================
    # 🔍 Feature Importance (from SHAP)
    # ===============================
    st.subheader("🧩 Feature Impact Analysis (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(bbox_inches="tight", dpi=300)

    # ===============================
    # 🔥 Error Distribution
    # ===============================
    st.subheader("⚖ Forecast Error Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Error_%'], bins=30, kde=True, color="#00b894", ax=ax2)
    plt.title("Distribution of Forecast Error (%)")
    st.pyplot(fig2)

    # ===============================
    # 💡 Correlation Insights
    # ===============================
    st.subheader("🔗 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[features + ['Revenue_Optimized']].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # ===============================
    # 💾 Download Results
    # ===============================
    st.subheader("💾 Download Forecast Results")
    output = BytesIO()
    df.to_csv(output, index=False)
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=output.getvalue(),
        file_name="Business_Forecast_Results.csv",
        mime="text/csv"
    )

    st.balloons()
else:
    st.info("👆 Please upload your dataset to continue.")

# ===============================
# ✅ Footer
# ===============================
st.markdown("---")
st.caption("🚀 Developed by *Data Gems Team* | MLOps Project – Stage 5")