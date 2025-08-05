import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class UnsupervisedFraudDetector:
    def __init__(self, method='isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.supported_methods = ['isolation_forest', 'lof']

    def fit(self, df: pd.DataFrame, feature_cols: list):
        logger.info(f"Training unsupervised model: {self.method}")

        if self.method not in self.supported_methods:
            raise ValueError(f"Unsupported method '{self.method}'. Choose from {self.supported_methods}")

        X = df[feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X)

        if self.method == 'isolation_forest':
            self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            self.model.fit(X_scaled)

        elif self.method == 'lof':
            # Note: LOF is unsupervised only at fit_predict time
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
            self.model.fit(X_scaled)

        logger.info("Model training complete.")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        logger.info(f"Applying {self.method} to detect anomalies...")

        df = df.copy()
        X = df[feature_cols].copy()
        X_scaled = self.scaler.transform(X)

        if self.method == 'isolation_forest':
            df['anomaly_score'] = self.model.decision_function(X_scaled)
            df['is_fraud'] = (self.model.predict(X_scaled) == -1).astype(int)

        elif self.method == 'lof':
            df['anomaly_score'] = self.model.decision_function(X_scaled)
            df['is_fraud'] = (self.model.predict(X_scaled) == -1).astype(int)

        return df

    def get_model(self):
        return self.model