import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import torch

# Add root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.feature_builder import FeatureBuilder
from src.data.database import DatabaseManager
from src.models.dl_models import TabularAutoencoder, TabularMLP
from src.models.failure_modes import FailureModeClassifier
from src.evaluation.explainability import SHAPExplainerManager

st.set_page_config(
    page_title="AI4I Predictive Maintenance & Failure Prevention",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Industrial Predictive Maintenance & Failure Prevention Platform")
st.markdown("""
**TechTrek Advanced Data Science & AI Graduation Project (Project 5)**  
*AI4I 2020 Predictive Maintenance Intelligence & Decision-Support System*
""")

st.divider()

# Sidebar Input Controls for Real-time Simulator
st.sidebar.header("🛠️ Real-Time Sensor Input Controls")

product_type = st.sidebar.selectbox("Product Type Quality Tier", ["L", "M", "H"], index=0)
air_temp = st.sidebar.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=298.1, step=0.1)
process_temp = st.sidebar.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=308.6, step=0.1)
rpm = st.sidebar.slider("Rotational Speed [rpm]", min_value=1100, max_value=3000, value=1551, step=5)
torque = st.sidebar.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=42.8, step=0.5)
tool_wear = st.sidebar.slider("Tool Wear [min]", min_value=0, max_value=300, value=120, step=1)

# Raw input dataframe
input_df = pd.DataFrame([{
    'UDI': 99999,
    'Product ID': f"{product_type}99999",
    'Type': product_type,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rpm,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear
}])

fb = FeatureBuilder()
df_transformed = fb.transform(input_df)
feature_cols = fb.get_feature_names()

# Main Page Layout: 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Sensor Features & Engineered Physical Metrics")
    temp_diff = process_temp - air_temp
    power_w = torque * rpm * (2 * np.pi / 60)
    strain = tool_wear * torque

    m1, m2, m3 = st.columns(3)
    m1.metric("Temp Diff (ΔT)", f"{temp_diff:.1f} K", delta="Normal" if temp_diff >= 8.6 else "Critical Low HDF")
    m2.metric("Mechanical Power", f"{power_w:.0f} W", delta="Normal" if 3500 <= power_w <= 9000 else "Critical PWF")
    m3.metric("Tool Wear Strain", f"{strain:.0f}", delta="High Strain" if strain > 10000 else "Normal")

    st.dataframe(df_transformed[feature_cols].T, use_container_width=True)

with col2:
    st.subheader("🎯 Real-Time AI Prediction & Risk Assessment")

    # Load Scaler & Best Classical ML Model
    scaler_path = "models/scaler.joblib"
    model_path = "models/random_forest.joblib"

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        calibrated_model = joblib.load(model_path)

        X_scaled = scaler.transform(df_transformed[feature_cols])
        failure_prob = calibrated_model.predict_proba(X_scaled)[0, 1]

        # Determine Risk Tier
        if failure_prob < 0.25:
            risk_tier = "LOW RISK"
            color = "green"
        elif failure_prob < 0.60:
            risk_tier = "MEDIUM RISK"
            color = "orange"
        else:
            risk_tier = "HIGH RISK (CRITICAL)"
            color = "red"

        st.markdown(f"### Machine Failure Probability: **:{color}[{failure_prob * 100:.2f}%]**")
        st.markdown(f"### Risk Tier: **:{color}[{risk_tier}]**")
        st.progress(float(failure_prob))

        # Predict specific failure modes
        fm_clf = FailureModeClassifier()
        fm_res = fm_clf.predict_failure_modes(X_scaled)[0]

        st.info(f"**Detected Specific Failure Modes:** {', '.join(fm_res['detected_modes'])}")
        st.success(f"**Recommended Maintenance Action:** {fm_res['recommendation']}")

        # Log prediction to database button
        if st.button("💾 Log Prediction to SQLite Database"):
            db = DatabaseManager()
            db.log_prediction(99999, f"{product_type}99999", int(failure_prob > 0.35), float(failure_prob), 0.0, risk_tier)
            st.toast("Prediction successfully logged to database!", icon="✅")
    else:
        st.warning("Models are currently training. Please wait a few seconds and refresh.")

st.divider()

# SHAP Local Explanation Section
st.subheader("🔍 Local SHAP Feature Contribution Breakdown")
if os.path.exists(scaler_path) and os.path.exists(model_path):
    explainer_mgr = SHAPExplainerManager()
    df_shap_local = explainer_mgr.get_local_explanation(df_transformed)
    st.dataframe(df_shap_local, use_container_width=True)
