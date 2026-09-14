import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from src.features.feature_builder import FeatureBuilder
from src.evaluation.cost_optimizer import CostMatrixOptimizer

class MLPipeline:
    """
    Classical Machine Learning Pipeline for Predictive Maintenance.
    Handles data splitting, scaling, stratified CV, training 5 models:
    1. Logistic Regression (Baseline)
    2. Decision Tree
    3. Random Forest
    4. XGBoost Classifier
    5. LightGBM Classifier
    """

    def __init__(self, artifact_dir: str = "models"):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.feature_builder = FeatureBuilder()
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []

    def prepare_data(self, df_raw: pd.DataFrame):
        df_transformed = self.feature_builder.transform(df_raw)
        
        feature_cols = self.feature_builder.get_feature_names()
        target_col = self.feature_builder.FAILURE_TARGET

        X = df_transformed[feature_cols]
        y = df_transformed[target_col].values

        self.feature_names = feature_cols
        return X, y, feature_cols

    def train_and_evaluate_all(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        X, y, feature_cols = self.prepare_data(df_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.artifact_dir, "scaler.joblib"))

        models = {
            "Logistic Regression (Baseline)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=8, random_state=42),
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), max_depth=5, learning_rate=0.05, n_estimators=150, random_state=42, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), max_depth=5, learning_rate=0.05, n_estimators=150, random_state=42, verbose=-1)
        }

        optimizer = CostMatrixOptimizer(fn_cost=5000, fp_cost=200)
        results = []
        best_pr_auc = -1

        for name, model in models.items():
            # Calibrate probabilities using Platt Scaling / Sigmoid
            calibrated_model = CalibratedClassifierCV(estimator=model, cv=3, method='sigmoid')
            calibrated_model.fit(X_train_scaled, y_train)

            y_probs = calibrated_model.predict_proba(X_test_scaled)[:, 1]

            metrics = optimizer.evaluate_model_comprehensive(y_test, y_probs, model_name=name)
            results.append(metrics)

            # Save model artifact
            model_file = os.path.join(self.artifact_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
            joblib.dump(calibrated_model, model_file)

            if metrics['PR-AUC'] > best_pr_auc:
                best_pr_auc = metrics['PR-AUC']
                self.best_model = calibrated_model
                self.best_model_name = name

        df_results = pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)
        df_results.to_csv(os.path.join(self.artifact_dir, "ml_comparison_results.csv"), index=False)
        print(f"Classical ML Training Complete. Best Model: {self.best_model_name} (PR-AUC: {best_pr_auc:.4f})")
        return df_results

if __name__ == "__main__":
    df_raw = pd.read_csv("ai4i2020.csv")
    pipeline = MLPipeline()
    res = pipeline.train_and_evaluate_all(df_raw)
    print(res[['Model', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Total Loss ($)']])
