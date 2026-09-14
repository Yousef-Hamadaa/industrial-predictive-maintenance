import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.drift import DriftSimulator

st.set_page_config(page_title="Drift Monitoring", page_icon="📡", layout="wide")

st.title("📡 MLOps Data & Feature Drift Monitoring Simulator")
st.markdown("Simulate sensor degradation over time (temperature drift, tool wear acceleration, bearing speed decay) and monitor standardized mean difference shifts.")

drift_intensity = st.slider("Select Simulated Drift Intensity", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

if os.path.exists("ai4i2020.csv"):
    df_orig = pd.read_csv("ai4i2020.csv")
    ds = DriftSimulator()
    df_drifted = ds.simulate_drift(df_orig, drift_intensity=drift_intensity)

    df_report = ds.compute_drift_metrics(df_orig, df_drifted)

    st.subheader("📊 Feature Drift Shift Report")
    st.dataframe(df_report, use_container_width=True)

    sig_shifts = (df_report['Drift Status'] == 'Significant Drift').sum()
    if sig_shifts > 0:
        st.warning(f"⚠️ Warning: {sig_shifts} features are displaying significant distribution drift! Model retraining recommended.")
    else:
        st.success("✅ Feature distributions remain within stable operational boundaries.")
else:
    st.warning("Original dataset not found.")
