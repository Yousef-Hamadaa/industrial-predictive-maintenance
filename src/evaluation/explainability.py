import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.features.feature_builder import FeatureBuilder

class SHAPExplainerManager:
    """
    SHAP Explainability Manager for Global and Local Model Interpretability.
    Generates summary plots, waterfall charts, and top feature contribution lists.
    """

    def __init__(self, artifact_dir: str = "models", reports_dir: str = "reports"):
        self.artifact_dir = artifact_dir
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self.feature_builder = FeatureBuilder()

    def generate_shap_explanations(self, df_raw: pd.DataFrame, sample_size: int = 200):
        df_transformed = self.feature_builder.transform(df_raw)
        feature_cols = self.feature_builder.get_feature_names()

        X = df_transformed[feature_cols]
        scaler = joblib.load(os.path.join(self.artifact_dir, "scaler.joblib"))
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

        # Load best tree model (e.g. Random Forest or LightGBM)
        model_path = os.path.join(self.artifact_dir, "random_forest.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.artifact_dir, "xgboost.joblib")
            
        calibrated_model = joblib.load(model_path)
        base_estimator = calibrated_model.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(base_estimator)
        sample_data = X_scaled.sample(n=min(sample_size, len(X_scaled)), random_state=42)
        shap_values = explainer.shap_values(sample_data)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # positive class
        else:
            shap_vals = shap_values

        # Generate and save SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, sample_data, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "shap_summary.png"), dpi=300)
        plt.close()

        print("SHAP Global Summary Plot generated and saved.")
        return explainer, sample_data, shap_vals

    def get_local_explanation(self, input_features_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        model_path = os.path.join(self.artifact_dir, "random_forest.joblib")
        scaler = joblib.load(os.path.join(self.artifact_dir, "scaler.joblib"))
        
        calibrated_model = joblib.load(model_path)
        base_estimator = calibrated_model.calibrated_classifiers_[0].estimator
        feature_cols = self.feature_builder.get_feature_names()

        X_scaled = pd.DataFrame(scaler.transform(input_features_df[feature_cols]), columns=feature_cols)
        explainer = shap.TreeExplainer(base_estimator)
        shap_vals = explainer.shap_values(X_scaled)

        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        else:
            vals = shap_vals

        vals_flat = np.array(vals).flatten()
        feat_vals_flat = input_features_df[feature_cols].iloc[0].values.flatten()

        df_local = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP Value (Impact)': vals_flat[:len(feature_cols)],
            'Feature Value': feat_vals_flat[:len(feature_cols)]
        }).sort_values(by='SHAP Value (Impact)', key=abs, ascending=False).head(top_k)

        return df_local

if __name__ == "__main__":
    df_raw = pd.read_csv("ai4i2020.csv")
    expl = SHAPExplainerManager()
    expl.generate_shap_explanations(df_raw)
