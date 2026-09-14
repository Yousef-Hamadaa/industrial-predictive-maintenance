import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.features.feature_builder import FeatureBuilder

class FailureModeClassifier:
    """
    Multi-Label Classifier for predicting specific machine failure modes:
    - TWF: Tool Wear Failure
    - HDF: Heat Dissipation Failure
    - PWF: Power Failure
    - OSF: Overstrain Failure
    - RNF: Random Failure
    """

    FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    RECOMMENDED_ACTIONS = {
        'TWF': 'Replace tool bit immediately; check cumulative tool wear and cutting force.',
        'HDF': 'Inspect cooling system, heat exchanger, and thermal sensor calibration; reduce process load.',
        'PWF': 'Check motor power supply, electrical connections, and torque limits; verify voltage stability.',
        'OSF': 'Inspect mechanical structural load, reduce torque/tool wear combination; balance feed rate.',
        'RNF': 'Perform routine diagnostic scan; verify sensor noise levels and grounding.',
        'NORMAL': 'Machine operating within normal parameters. Routine maintenance scheduled.'
    }

    def __init__(self, artifact_dir: str = "models"):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.feature_builder = FeatureBuilder()
        self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))

    def train(self, df_raw: pd.DataFrame):
        df_transformed = self.feature_builder.transform(df_raw)
        feature_cols = self.feature_builder.get_feature_names()

        X = df_transformed[feature_cols].values
        Y = df_transformed[self.FAILURE_MODES].values

        scaler = joblib.load(os.path.join(self.artifact_dir, "scaler.joblib")) if os.path.exists(os.path.join(self.artifact_dir, "scaler.joblib")) else None
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        self.model.fit(X_scaled, Y)
        joblib.dump(self.model, os.path.join(self.artifact_dir, "failure_modes_model.joblib"))
        print("Multi-label Failure Modes Model Trained & Serialized.")

    def predict_failure_modes(self, X_input: np.ndarray) -> list:
        if not os.path.exists(os.path.join(self.artifact_dir, "failure_modes_model.joblib")):
            return [{'detected_modes': ['NORMAL'], 'recommendation': self.RECOMMENDED_ACTIONS['NORMAL']}]

        model = joblib.load(os.path.join(self.artifact_dir, "failure_modes_model.joblib"))
        preds = model.predict(X_input)

        results = []
        for row in preds:
            active_modes = [self.FAILURE_MODES[i] for i, val in enumerate(row) if val == 1]
            if not active_modes:
                results.append({
                    'detected_modes': ['No Active Specific Failure Mode'],
                    'recommendation': self.RECOMMENDED_ACTIONS['NORMAL']
                })
            else:
                recs = [self.RECOMMENDED_ACTIONS[m] for m in active_modes]
                results.append({
                    'detected_modes': active_modes,
                    'recommendation': " | ".join(recs)
                })
        return results

if __name__ == "__main__":
    df_raw = pd.read_csv("ai4i2020.csv")
    fm = FailureModeClassifier()
    fm.train(df_raw)
