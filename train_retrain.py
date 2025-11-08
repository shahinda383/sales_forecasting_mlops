
# ================================================================
# 🚀 Automated Retraining Script
# Author: Shahinda | Stage 6 – CI/CD MLOps
# ================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime
import os

# ===============================
# 📂 Load Data
# ===============================
df = pd.read_csv("Business_Impact_Model.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)

# Feature Engineering
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

features = ['Revenue_Current', 'Revenue_Change', 'Growth_%', 'Month', 'Quarter', 'Year']
target = 'Revenue_Optimized'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 🔁 Drift Detection (Simple)
# ===============================
if os.path.exists("previous_stats.csv"):
    prev = pd.read_csv("previous_stats.csv")
    drift = abs(prev['mean_Revenue_Current'][0] - df['Revenue_Current'].mean()) > 0.05 * prev['mean_Revenue_Current'][0]
else:
    drift = True

if drift:
    print("⚠ Data drift detected → Retraining model...")

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    joblib.dump(model, "best_model.pkl")
    print(f"✅ Model retrained successfully! R² = {r2:.3f}, MAE = {mae:.3f}")

    # Save new data stats
    pd.DataFrame({
        "mean_Revenue_Current": [df['Revenue_Current'].mean()],
        "std_Revenue_Current": [df['Revenue_Current'].std()],
        "Last_Retrained": [datetime.now()]
    }).to_csv("previous_stats.csv", index=False)

    # Log deployment metrics
    pd.DataFrame({
        "Timestamp": [datetime.now()],
        "R2_Score": [r2],
        "MAE": [mae],
        "Data_Drift": [True],
        "Model_Status": ["Retrained"]
    }).to_csv("deployment_report.csv", index=False)
else:
    print("✅ No data drift detected. Using existing model.")
    pd.DataFrame({
        "Timestamp": [datetime.now()],
        "R2_Score": [None],
        "MAE": [None],
        "Data_Drift": [False],
        "Model_Status": ["No Change"]
    }).to_csv("deployment_report.csv", index=False)