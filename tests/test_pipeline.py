import os
import pandas as pd
import numpy as np
import pytest
import sqlite3

from src.features.feature_builder import FeatureBuilder
from src.data.database import DatabaseManager
from src.evaluation.cost_optimizer import CostMatrixOptimizer
from src.models.failure_modes import FailureModeClassifier

def test_feature_builder():
    df_raw = pd.DataFrame([{
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40.0,
        'Tool wear [min]': 100,
        'Type': 'M'
    }])

    fb = FeatureBuilder()
    df_out = fb.transform(df_raw)

    assert 'Temp_Diff' in df_out.columns
    assert np.isclose(df_out['Temp_Diff'].iloc[0], 10.5)
    assert 'Power_W' in df_out.columns
    assert df_out['Power_W'].iloc[0] > 0
    assert 'Tool_Wear_Strain' in df_out.columns
    assert df_out['Tool_Wear_Strain'].iloc[0] == 4000.0

def test_database_manager():
    db_path = "data/test_maintenance.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_path=db_path)
    db.init_database()

    df = db.execute_query("SELECT COUNT(*) FROM cleaned_sensor_features")
    assert df.values[0][0] == 10000

    if os.path.exists(db_path):
        os.remove(db_path)

def test_cost_matrix_optimizer():
    optimizer = CostMatrixOptimizer(fn_cost=5000, fp_cost=200)
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.75])

    res = optimizer.evaluate_model_comprehensive(y_true, y_probs, model_name="Test Model")
    assert 'PR-AUC' in res
    assert 'Optimal Threshold' in res
    assert res['Optimal Threshold'] > 0

def test_failure_mode_classifier():
    fm = FailureModeClassifier()
    assert fm is not None
