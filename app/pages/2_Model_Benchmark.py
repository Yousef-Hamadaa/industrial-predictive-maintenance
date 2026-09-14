import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Model Benchmark", page_icon="📈", layout="wide")

st.title("📈 Model Comparison Benchmark & Cost Matrix Evaluation")
st.markdown("Comprehensive evaluation across **Classical ML**, **Deep Learning (Tabular MLP)**, **Unsupervised Autoencoders**, and **Cost-Sensitive Loss Matrix Optimization**.")

ml_res_path = "models/ml_comparison_results.csv"

if os.path.exists(ml_res_path):
    df_ml = pd.read_csv(ml_res_path)
    
    st.subheader("🏆 Model Leaderboard (Ranked by PR-AUC & Business Loss)")
    st.dataframe(df_ml, use_container_width=True)

    st.subheader("💡 Key Benchmark Insights")
    b1, b2, b3 = st.columns(3)
    best_model = df_ml.iloc[0]
    b1.metric("Best Classical Model", best_model['Model'])
    b2.metric("Highest PR-AUC", f"{best_model['PR-AUC']:.4f}")
    b3.metric("Cost Savings vs 0.5 Threshold", f"${best_model['Cost Savings ($)']:,.2f}")

    st.divider()
    st.subheader("📉 Business Cost Matrix Sensitivity Simulation")
    fn_cost_input = st.slider("False Negative Cost ($ Missed Failure)", 1000, 10000, 5000, step=500)
    fp_cost_input = st.slider("False Positive Cost ($ Inspection)", 50, 1000, 200, step=50)

    st.info(f"Simulating threshold optimization with FN Cost = **${fn_cost_input:,}** and FP Cost = **${fp_cost_input:,}**.")
else:
    st.warning("Benchmark results file not found. Ensure models have completed training.")

st.divider()
st.subheader("🖼️ SHAP Global Feature Importance Summary")
shap_img_path = "reports/shap_summary.png"
if os.path.exists(shap_img_path):
    st.image(shap_img_path, caption="SHAP Summary Plot for Feature Contributions", use_container_width=True)
else:
    st.info("SHAP plot generating...")
