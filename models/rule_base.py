import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RuleBasedFraudDetector:
    def __init__(self):
        self.rules = [
            self.rule_large_transaction,
            self.rule_odd_hour,
            self.rule_location_jump,
            self.rule_new_device,
            self.rule_currency_blank,
            self.rule_high_txn_frequency,
        ]

    def fit(self, df: pd.DataFrame):
        logger.info("No training required for rule-based detector.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying rule-based fraud detection rules...")

        df = df.copy()
        df['fraud_score'] = 0
        df['fraud_reason'] = ''

        for rule in self.rules:
            try:
                rule(df)
            except Exception as e:
                logger.warning(f"Rule {rule.__name__} failed: {e}")

        df['is_fraud'] = (df['fraud_score'] > 0).astype(int)
        return df

    # RULES
    def rule_large_transaction(self, df):
        threshold = 1000000
        mask = df['amount'] > threshold
        df.loc[mask, 'fraud_score'] += 1
        df.loc[mask, 'fraud_reason'] += 'HighAmount;'

    def rule_odd_hour(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = df['timestamp'].dt.hour.isin([0, 1, 2, 3, 4])
        df.loc[mask, 'fraud_score'] += 1
        df.loc[mask, 'fraud_reason'] += 'OddHour;'

    def rule_location_jump(self, df):
        if 'location_jump_km' in df.columns:
            mask = df['location_jump_km'] > 100
            df.loc[mask, 'fraud_score'] += 1
            df.loc[mask, 'fraud_reason'] += 'LocationJump;'

    def rule_new_device(self, df):
        df['is_new_device'] = df.groupby('user_id')['device'].transform(lambda x: x != x.shift(1))
        mask = df['is_new_device']
        df.loc[mask, 'fraud_score'] += 1
        df.loc[mask, 'fraud_reason'] += 'NewDevice;'

    def rule_currency_blank(self, df):
        mask = df['currency'].isna() | (df['currency'].str.strip() == '')
        df.loc[mask, 'fraud_score'] += 1
        df.loc[mask, 'fraud_reason'] += 'NoCurrency;'

    def rule_high_txn_frequency(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['user_id', 'timestamp'])
        df['prev_time'] = df.groupby('user_id')['timestamp'].shift(1)
        df['time_diff'] = (df['timestamp'] - df['prev_time']).dt.total_seconds().fillna(np.inf)
        mask = df['time_diff'] < 60  # less than 1 minute apart
        df.loc[mask, 'fraud_score'] += 1
        df.loc[mask, 'fraud_reason'] += 'RapidTxn;'