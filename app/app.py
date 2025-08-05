import sys
import os

# Adding the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from models.rule_base import RuleBasedFraudDetector
from models.unsupervised import UnsupervisedFraudDetector
from models.autoencoder import AutoEncoderFraudDetector
import matplotlib.pyplot as plt
import io


# Feature Engineering
def engineer_features(df):
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


# Fraud Detection App

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("🕵️‍♀️ Fraud Detection System (Rule-Based + ML Models)")

st.markdown("""
Upload a cleaned transaction log (Excel). The system will run 3 fraud models and return predictions.
""")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Feature engineering
    df = engineer_features(df)
    feature_cols = ['amount', 'location_jump_km', 'time_diff', 'is_new_device', 'is_location_jump']

    # Load Models
    rule_model = RuleBasedFraudDetector()
    unsup_model = UnsupervisedFraudDetector()
    auto_model = AutoEncoderFraudDetector()

    # Fit unsupervised and autoencoder on current data (if needed, use train set)
    unsup_model.fit(df, feature_cols)
    auto_model.fit(df, feature_cols)

    # Predict
    rule_pred = rule_model.predict(df)
    unsup_pred = unsup_model.predict(df, feature_cols)
    auto_pred = auto_model.predict(df, feature_cols)

    # Ensemble Voting: fraud if at least 2 models agree
    df['fraud_votes'] = (
        rule_pred['is_fraud'] +
        unsup_pred['is_fraud'] +
        auto_pred['is_fraud']
    )
    df['is_fraud_final'] = (df['fraud_votes'] >= 2).astype(int)

    st.subheader("Fraud Prediction Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rule-Based Fraud", int(rule_pred['is_fraud'].sum()))
    col2.metric("Unsupervised Fraud", int(unsup_pred['is_fraud'].sum()))
    col3.metric("AutoEncoder Fraud", int(auto_pred['is_fraud'].sum()))
    st.metric("Ensemble Flagged Fraud", int(df['is_fraud_final'].sum()))

    # Fraud Count Chart
    fig, ax = plt.subplots()
    models = ['Rule-Based', 'Unsupervised', 'AutoEncoder']
    frauds = [
        rule_pred['is_fraud'].sum(),
        unsup_pred['is_fraud'].sum(),
        auto_pred['is_fraud'].sum()
    ]
    ax.bar(models, frauds, color=['red', 'blue', 'green'])
    ax.set_ylabel("Fraudulent Transactions")
    ax.set_title("Fraud Detection Comparison")
    st.pyplot(fig)

    # Show final frauds
    st.subheader("🧾 Flagged Fraud Transactions (Ensemble)")
    fraud_df = df[df['is_fraud_final'] == 1]
    st.dataframe(fraud_df[['user_id', 'amount', 'timestamp', 'location', 'device', 'is_fraud_final']])

    # Download results
    to_download = df.copy()
    output = io.BytesIO()
    to_download.to_excel(output, index=False)
    st.download_button(
        label="Download Full Results (Excel)",
        data=output.getvalue(),
        file_name="fraud_detection_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )