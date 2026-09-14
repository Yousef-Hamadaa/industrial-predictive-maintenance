import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.feature_builder import FeatureBuilder
from src.models.failure_modes import FailureModeClassifier

st.set_page_config(page_title="Batch CSV Scoring", page_icon="📁", layout="wide")

st.title("📁 Batch Machine CSV Scoring & Maintenance Schedule")
st.markdown("Upload raw sensor dataset CSV files to perform automated batch inference, failure risk ranking, and action recommendations.")

uploaded_file = st.file_uploader("Upload Sensor Records CSV (e.g. ai4i2020.csv)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write(f"**Loaded Dataset Shape:** {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    scaler_path = "models/scaler.joblib"
    model_path = "models/random_forest.joblib"

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        fb = FeatureBuilder()
        df_transformed = fb.transform(df_raw)
        feature_cols = fb.get_feature_names()

        scaler = joblib.load(scaler_path)
        calibrated_model = joblib.load(model_path)

        X_scaled = scaler.transform(df_transformed[feature_cols])
        probs = calibrated_model.predict_proba(X_scaled)[:, 1]

        df_results = df_raw.copy()
        df_results['Failure_Probability'] = np.round(probs, 4)

        # Risk Tier Assignment
        conditions = [
            (df_results['Failure_Probability'] < 0.25),
            (df_results['Failure_Probability'] >= 0.25) & (df_results['Failure_Probability'] < 0.60),
            (df_results['Failure_Probability'] >= 0.60)
        ]
        choices = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK (CRITICAL)']
        df_results['Risk_Tier'] = np.select(conditions, choices, default='UNKNOWN')

        # Failure mode recommendations
        fm_clf = FailureModeClassifier()
        mode_results = fm_clf.predict_failure_modes(X_scaled)
        df_results['Detected_Failure_Modes'] = [", ".join(res['detected_modes']) for res in mode_results]
        df_results['Recommended_Action'] = [res['recommendation'] for res in mode_results]

        # Summary KPIs
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Inspected Machines", len(df_results))
        k2.metric("High Risk Machines", (df_results['Risk_Tier'] == 'HIGH RISK (CRITICAL)').sum())
        k3.metric("Medium Risk Machines", (df_results['Risk_Tier'] == 'MEDIUM RISK').sum())
        k4.metric("Low Risk Machines", (df_results['Risk_Tier'] == 'LOW RISK').sum())

        st.subheader("🚨 Prioritized Maintenance Queue (High Risk First)")
        df_sorted = df_results.sort_values(by='Failure_Probability', ascending=False)
        st.dataframe(df_sorted, use_container_width=True)

        # Download CSV
        csv_bytes = df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Risk Report (CSV)",
            data=csv_bytes,
            file_name="predictive_maintenance_risk_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("Trained models not found. Please train models first.")
