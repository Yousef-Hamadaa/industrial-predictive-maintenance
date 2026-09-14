import os
import sqlite3
import pandas as pd
from src.features.feature_builder import FeatureBuilder

class DatabaseManager:
    """
    Database Manager for SQLite analytical data store.
    Handles data ingestion, cleaning, feature transformation, table population,
    and analytical query execution.
    """

    def __init__(self, db_path: str = "data/predictive_maintenance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_database(self, csv_path: str = "ai4i2020.csv"):
        """Reads raw CSV, applies feature builder, and populates SQLite tables."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df_raw = pd.read_csv(csv_path)

        # 1. Standardize column names for SQL friendly storage
        df_sql = df_raw.copy()
        df_sql.columns = [
            c.replace(' ', '_').replace('[', '').replace(']', '') 
            for c in df_sql.columns
        ]

        # 2. Apply Feature Engineering
        fb = FeatureBuilder()
        df_engineered = fb.transform(df_raw)
        df_engineered_sql = df_engineered.copy()
        df_engineered_sql.columns = [
            c.replace(' ', '_').replace('[', '').replace(']', '') 
            for c in df_engineered_sql.columns
        ]

        conn = self.get_connection()
        
        # Save raw table
        df_sql.to_sql("raw_sensor_data", conn, if_exists="replace", index=False)
        
        # Save cleaned & engineered table
        df_engineered_sql.to_sql("cleaned_sensor_features", conn, if_exists="replace", index=False)

        # Create maintenance logs schema if not exists
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                Prediction_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                UDI INTEGER,
                Product_ID TEXT,
                Predicted_Class INTEGER,
                Failure_Probability REAL,
                Anomaly_Score REAL,
                Risk_Tier TEXT,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print(f"Database successfully initialized at {self.db_path}")

    def execute_query(self, sql_statement: str) -> pd.DataFrame:
        """Executes a SQL query and returns results as pandas DataFrame."""
        conn = self.get_connection()
        df = pd.read_sql_query(sql_statement, conn)
        conn.close()
        return df

    def log_prediction(self, udi: int, product_id: str, pred_class: int, proba: float, anomaly_score: float, risk_tier: str):
        """Logs a single model inference into SQLite database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO maintenance_logs (UDI, Product_ID, Predicted_Class, Failure_Probability, Anomaly_Score, Risk_Tier)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (udi, product_id, pred_class, proba, anomaly_score, risk_tier))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    db = DatabaseManager()
    db.init_database()
