import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib

from src.features.feature_builder import FeatureBuilder
from src.evaluation.cost_optimizer import CostMatrixOptimizer

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TabularMLP(nn.Module):
    """Deep Learning Multi-Layer Perceptron for Supervised Failure Prediction."""
    def __init__(self, input_dim: int):
        super(TabularMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class TabularAutoencoder(nn.Module):
    """PyTorch Autoencoder for Unsupervised Anomaly Detection trained on Normal Data."""
    def __init__(self, input_dim: int):
        super(TabularAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8)  # Bottleneck latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class DLManager:
    """Manages PyTorch MLP classifier and PyTorch Autoencoder training & evaluation."""

    def __init__(self, artifact_dir: str = "models"):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_builder = FeatureBuilder()
        self.mlp_model = None
        self.autoencoder = None

    def train_dl_models(self, df_raw: pd.DataFrame) -> dict:
        df_transformed = self.feature_builder.transform(df_raw)
        feature_cols = self.feature_builder.get_feature_names()
        X = df_transformed[feature_cols].values
        y = df_transformed[self.feature_builder.FAILURE_TARGET].values

        scaler = joblib.load(os.path.join(self.artifact_dir, "scaler.joblib")) if os.path.exists(os.path.join(self.artifact_dir, "scaler.joblib")) else None
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # -------------------------------------------------------------
        # 1. Train Tabular MLP Classifier
        # -------------------------------------------------------------
        input_dim = X_train.shape[1]
        self.mlp_model = TabularMLP(input_dim).to(self.device)

        # Handle class weight in BCEWithLogitsLoss
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(self.device)
        criterion_mlp = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer_mlp = optim.Adam(self.mlp_model.parameters(), lr=0.003, weight_decay=1e-4)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        self.mlp_model.train()
        for epoch in range(40):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer_mlp.zero_grad()
                out = self.mlp_model(batch_x)
                loss = criterion_mlp(out, batch_y)
                loss.backward()
                optimizer_mlp.step()

        # Evaluate MLP
        self.mlp_model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            mlp_logits = self.mlp_model(test_tensor)
            mlp_probs = torch.sigmoid(mlp_logits).cpu().numpy().flatten()

        cost_opt = CostMatrixOptimizer(fn_cost=5000, fp_cost=200)
        mlp_metrics = cost_opt.evaluate_model_comprehensive(y_test, mlp_probs, model_name="Deep Learning (Tabular MLP)")
        torch.save(self.mlp_model.state_dict(), os.path.join(self.artifact_dir, "mlp_model.pt"))

        # -------------------------------------------------------------
        # 2. Train PyTorch Autoencoder (Unsupervised Anomaly Detector)
        # -------------------------------------------------------------
        X_train_normal = X_train[y_train == 0]
        self.autoencoder = TabularAutoencoder(input_dim).to(self.device)
        criterion_ae = nn.MSELoss()
        optimizer_ae = optim.Adam(self.autoencoder.parameters(), lr=0.002)

        ae_dataset = TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32))
        ae_loader = DataLoader(ae_dataset, batch_size=64, shuffle=True)

        self.autoencoder.train()
        for epoch in range(50):
            for batch_x, in ae_loader:
                batch_x = batch_x.to(self.device)
                optimizer_ae.zero_grad()
                reconstructed = self.autoencoder(batch_x)
                loss = criterion_ae(reconstructed, batch_x)
                loss.backward()
                optimizer_ae.step()

        # Compute Anomaly Score (Reconstruction Loss) on Test Set
        self.autoencoder.eval()
        with torch.no_grad():
            recon_test = self.autoencoder(test_tensor)
            mse_loss = torch.mean((test_tensor - recon_test) ** 2, dim=1).cpu().numpy()

        # Min-max scale anomaly score into 0-1 probability proxy
        anomaly_scores = (mse_loss - mse_loss.min()) / (mse_loss.max() - mse_loss.min() + 1e-8)
        ae_metrics = cost_opt.evaluate_model_comprehensive(y_test, anomaly_scores, model_name="Autoencoder Anomaly Score")
        torch.save(self.autoencoder.state_dict(), os.path.join(self.artifact_dir, "autoencoder_model.pt"))

        # -------------------------------------------------------------
        # 3. Hybrid Risk Score (Supervised MLP + Autoencoder Anomaly)
        # -------------------------------------------------------------
        hybrid_probs = 0.6 * mlp_probs + 0.4 * anomaly_scores
        hybrid_metrics = cost_opt.evaluate_model_comprehensive(y_test, hybrid_probs, model_name="Hybrid Risk (MLP + Autoencoder)")

        print("Deep Learning Models Training Complete.")
        return {
            'mlp_metrics': mlp_metrics,
            'ae_metrics': ae_metrics,
            'hybrid_metrics': hybrid_metrics
        }

if __name__ == "__main__":
    df_raw = pd.read_csv("ai4i2020.csv")
    dl_mgr = DLManager()
    results = dl_mgr.train_dl_models(df_raw)
    print(pd.DataFrame([results['mlp_metrics'], results['ae_metrics'], results['hybrid_metrics']]))
