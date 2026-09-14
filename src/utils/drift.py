import numpy as np
import pandas as pd

class DriftSimulator:
    """
    Simulates feature distribution drift (e.g. sensor degradation over time,
    temperature spikes, bearing wear) to evaluate model robustness under drift.
    """

    def simulate_drift(self, df_raw: pd.DataFrame, drift_intensity: float = 0.2) -> pd.DataFrame:
        df_drifted = df_raw.copy()

        # 1. Temperature drift (Global warming / cooling system degradation)
        df_drifted['Air temperature [K]'] += np.random.normal(2.0 * drift_intensity, 1.0, len(df_drifted))
        df_drifted['Process temperature [K]'] += np.random.normal(3.5 * drift_intensity, 1.2, len(df_drifted))

        # 2. Motor wear drift (Lower RPM, higher Torque variance)
        df_drifted['Rotational speed [rpm]'] -= np.abs(np.random.normal(50 * drift_intensity, 20, len(df_drifted)))
        df_drifted['Torque [Nm]'] += np.abs(np.random.normal(5 * drift_intensity, 2, len(df_drifted)))

        # 3. Accelerated Tool Wear
        df_drifted['Tool wear [min]'] = (df_drifted['Tool wear [min]'] * (1 + 0.3 * drift_intensity)).clip(upper=300)

        return df_drifted

    def compute_drift_metrics(self, df_original: pd.DataFrame, df_drifted: pd.DataFrame) -> pd.DataFrame:
        num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        drift_report = []
        for col in num_cols:
            orig_mean = df_original[col].mean()
            drift_mean = df_drifted[col].mean()
            orig_std = df_original[col].std()
            
            # Simple Normalized Shift Metric (Wasserstein/Standardized Mean Difference)
            smd = abs(drift_mean - orig_mean) / (orig_std + 1e-5)
            status = "Significant Drift" if smd > 0.3 else "Minor Drift" if smd > 0.1 else "Stable"

            drift_report.append({
                'Feature': col,
                'Original Mean': round(orig_mean, 2),
                'Drifted Mean': round(drift_mean, 2),
                'Standardized Shift': round(smd, 3),
                'Drift Status': status
            })

        return pd.DataFrame(drift_report)

if __name__ == "__main__":
    df = pd.read_csv("ai4i2020.csv")
    ds = DriftSimulator()
    df_d = ds.simulate_drift(df, 0.3)
    print(ds.compute_drift_metrics(df, df_d))
