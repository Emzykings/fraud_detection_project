import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderFraudDetector:
    def __init__(self, threshold=None):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.input_dim = None

    def fit(self, df: pd.DataFrame, feature_cols: list, epochs=20, batch_size=64, learning_rate=0.001):
        logger.info("Fitting AutoEncoder on training data...")

        X = df[feature_cols].copy().fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X_scaled.shape[1]

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = AutoEncoder(self.input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Set anomaly threshold based on training reconstruction error
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            recon = self.model(X_tensor)
            loss = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
            self.threshold = np.percentile(loss, 99.5)
            logger.info(f"AutoEncoder threshold (99.5th percentile): {self.threshold:.4f}")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        logger.info("Scoring fraud with AutoEncoder...")

        df = df.copy()
        X = df[feature_cols].copy().fillna(0)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            loss = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
            df['recon_error'] = loss
            df['is_fraud'] = (loss > self.threshold).astype(int)

        return df

    def get_model(self):
        return self.model