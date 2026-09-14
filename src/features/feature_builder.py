import pandas as pd
import numpy as np

class FeatureBuilder:
    """
    Feature Engineering pipeline for AI4I 2020 Predictive Maintenance Dataset.
    Engineers physical domain features:
    - Temperature Difference (Process Temp - Air Temp)
    - Mechanical Power (Torque * RPM * 2*pi/60)
    - Tool Wear Strain (Tool Wear * Torque)
    - Speed/Torque Ratio
    - Categorical Encoding for Product Type (L, M, H)
    """

    RAW_NUMERICAL_COLS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    ENGINEERED_COLS = [
        'Temp_Diff',
        'Power_W',
        'Tool_Wear_Strain',
        'Speed_Torque_Ratio'
    ]

    FAILURE_TARGET = 'Machine failure'
    FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # 1. Temperature Difference (Kelvin)
        df_out['Temp_Diff'] = df_out['Process temperature [K]'] - df_out['Air temperature [K]']

        # 2. Mechanical Power in Watts (Torque [Nm] * Rotational Speed [rad/s])
        # omega = rpm * 2 * pi / 60
        df_out['Power_W'] = df_out['Torque [Nm]'] * df_out['Rotational speed [rpm]'] * (2 * np.pi / 60)

        # 3. Tool Wear Strain (Proxy for mechanical overload on worn tool)
        df_out['Tool_Wear_Strain'] = df_out['Tool wear [min]'] * df_out['Torque [Nm]']

        # 4. Speed to Torque Ratio (Operating regime indicator)
        df_out['Speed_Torque_Ratio'] = df_out['Rotational speed [rpm]'] / (df_out['Torque [Nm]'] + 1e-5)

        # 5. One-Hot Encoding for Product Type
        if 'Type' in df_out.columns:
            for t in ['L', 'M', 'H']:
                df_out[f'Type_{t}'] = (df_out['Type'] == t).astype(int)

        return df_out

    def get_feature_names(self) -> list:
        return self.RAW_NUMERICAL_COLS + self.ENGINEERED_COLS + ['Type_L', 'Type_M', 'Type_H']
