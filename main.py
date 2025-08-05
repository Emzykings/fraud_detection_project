import pandas as pd
import numpy as np
import logging
from models.rule_base import RuleBasedFraudDetector
from models.unsupervised import UnsupervisedFraudDetector
from models.autoencoder import AutoEncoderFraudDetector
import matplotlib.pyplot as plt


# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load Data
train_path = "data/train_data/raw_logs_train.xlsx"
test_path = "data/test_data/raw_logs_test.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

#   Feature Engineering Funcs
def engineer_features(df):
    """
    Feature engineering pipeline:
    - Time difference from previous txn per user
    - Location and device change detection
    - Handling missing values
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)

    df['prev_time'] = df.groupby('user_id')['timestamp'].shift(1)
    df['time_diff'] = (df['timestamp'] - df['prev_time']).dt.total_seconds().fillna(999999)

    df['prev_location'] = df.groupby('user_id')['location'].shift(1)
    df['is_location_jump'] = (df['location'] != df['prev_location']).astype(int)

    df['prev_device'] = df.groupby('user_id')['device'].shift(1)
    df['is_new_device'] = (df['device'] != df['prev_device']).astype(int)

    df.fillna({
        'location_jump_km': 0,
        'amount': 0,
        'time_diff': 999999,
        'is_location_jump': 0,
        'is_new_device': 0
    }, inplace=True)

    return df


# Apply Feature Engineering 
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Define numeric features for anomaly models
feature_cols = ['amount', 'location_jump_km', 'time_diff', 'is_new_device', 'is_location_jump']


# Initialize Models
rule_model = RuleBasedFraudDetector()
unsup_model = UnsupervisedFraudDetector()
autoencoder_model = AutoEncoderFraudDetector()


# Train Models 
logger.info("Training unsupervised and autoencoder models...")
unsup_model.fit(train_df, feature_cols)
autoencoder_model.fit(train_df, feature_cols)


# Predict Test Set 
rule_result = rule_model.predict(test_df)
unsup_result = unsup_model.predict(test_df, feature_cols)
auto_result = autoencoder_model.predict(test_df, feature_cols)


# Evaluation Block
logger.info("Evaluation Results:")
print("Rule-Based:\n", rule_result['is_fraud'].value_counts())
print("Unsupervised:\n", unsup_result['is_fraud'].value_counts())
print("AutoEncoder:\n", auto_result['is_fraud'].value_counts())


# Fraud Count Plot 
methods = ['Rule-Based', 'Unsupervised', 'AutoEncoder']
fraud_counts = [
    rule_result['is_fraud'].sum(),
    unsup_result['is_fraud'].sum(),
    auto_result['is_fraud'].sum()
]

plt.figure(figsize=(6, 4))
plt.bar(methods, fraud_counts, color=['red', 'blue', 'green'])
plt.ylabel("Fraudulent Transactions")
plt.title("Fraud Detection Comparison")
plt.tight_layout()
plt.savefig("data/test_data/fraud_count_comparison.png")

# Ensemble Voting Logic 
# Fraud if flagged by at least 2 out of 3 models
final_output = test_df.copy()
final_output['fraud_votes'] = (
    rule_result['is_fraud'] +
    unsup_result['is_fraud'] +
    auto_result['is_fraud']
)
final_output['is_fraud_final'] = (final_output['fraud_votes'] >= 2).astype(int)


# Save Model Outputs
rule_result.to_excel("data/test_data/rule_based_output.xlsx", index=False)
unsup_result.to_excel("data/test_data/unsupervised_output.xlsx", index=False)
auto_result.to_excel("data/test_data/autoencoder_output.xlsx", index=False)
final_output.to_excel("data/test_data/ensemble_output.xlsx", index=False)

logger.info("All outputs saved. Pipeline complete.")