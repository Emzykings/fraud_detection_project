# 🕵️ Fraud Detection System (Rule-Based + ML)

## 🛡️ Overview

This project is a hybrid **Fraud Detection System** that combines rule-based heuristics, unsupervised machine learning (Isolation Forest), and deep learning (AutoEncoder) models to detect fraudulent financial transactions.

The solution features:
- A **modular pipeline** for inference and evaluation
- A **Streamlit UI** for real-time predictions
- Cleaned and structured datasets
- **End-to-end anomaly detection pipeline**

---

## 🧠 Detection Models

| Model         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Rule-Based** | Detects fraud using business logic (e.g., abnormal amount, odd hour)        |
| **Unsupervised** | Isolation Forest to identify anomalies in unlabeled data                   |
| **AutoEncoder**  | Deep neural network trained on normal behavior and flags reconstruction errors |

Each model outputs:
- `is_fraud`: 1 (fraudulent) or 0 (clean)
- `fraud_score`: Confidence or anomaly score
- `fraud_reason`: Rule explanation (for rule-based only)

---

## ✅ Features

- 🔍 **Rule-based detection** using custom heuristics (odd transaction times, missing fields, etc.)
- 🤖 **Unsupervised model** (Isolation Forest) for anomaly detection
- 🔐 **AutoEncoder** to learn and flag behavior deviations
- 📊 **Streamlit UI** for real-time predictions and insights
- 🧼 **Log Parser** for converting noisy logs into structured Excel format
- 📁 **Processed log tracking** to avoid redundant parsing

---

## 📂 Project Structure

```
fraud_detection_project/
│
├── app.py                         # Streamlit UI for real-time predictions
├── main.py                        # Core inference script for running all models
├── README.md                      # Project documentation
│
├── data/
│   ├── raw_log_data/              # Original unprocessed logs
│   │   └── raw_logs.xlsx
│   ├── parsed_data/               # Structured, cleaned logs
│   │   └── raw_logs_parsed.xlsx
│   ├── MALFORMED_data/            # Logs with parsing issues
│   │   └── raw_logs_malformed.xlsx
│   ├── train_data/                # Training datasets
│   │   └── raw_logs_train.xlsx
│   ├── test_data/                 # Test datasets
│   │   └── raw_logs_test.xlsx
│   └── .processed_files.txt       # Tracker for previously parsed files
│
├── models/
│   ├── rule_based.py              # Rule-based detection logic
│   ├── unsupervised.py            # Isolation Forest model
│   └── autoencoder.py             # Deep learning model
│
├── parser/
│   ├── log_parser.py              # Script to clean and structure logs
│   └── parsing_errors.log         # Log file for failed parsing attempts
│
├── notebooks/
│   └── Data_Exploration_and_Feature_Engineering.ipynb  # EDA and feature engineering
│
└── requirements.txt               # List of dependencies
```

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites
- Python 3.9+
- pip

### 📥 Installation Steps

```bash
# Clone the repository
git clone https://github.com/emzykings/fraud_detection_project.git
cd fraud_detection_project

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Run the Detection Pipeline

```bash
python main.py
```

This will:
- Load training and test datasets
- Train and apply all three models
- Save results to:

```bash
data/test_data/rule_based_output.xlsx
data/test_data/unsupervised_output.xlsx
data/test_data/autoencoder_output.xlsx
```

---

### Step 2: Launch Streamlit Dashboard

```bash
streamlit run app.py
```

- Upload `.xlsx` transaction logs
- View results from all three detection models
- Evaluate flagged transactions in real time

---

## 📊 Evaluation Strategy

- No ground-truth fraud labels are available
- Evaluation is done by:
  - Comparing model predictions
  - Distribution analysis of fraud scores
  - Business interpretability of rule-based flags

---

## 💡 Future Enhancements

- 📧 Email alerts for high-confidence fraud
- ⚖️ Confidence-weighted ensemble voting system
- 📈 Time-series visualizations
- 🔄 Feedback loop integration for model retraining

---

## 👨‍💻 Author

**Emmanuel Adeitan**  
Senior Data Scientist | ML Engineer | AI Engineer  
📧 Email: [adeitanemmanuel086@gmail.com](mailto:adeitanemmanuel086@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/emmanuel-adeitan](https://www.linkedin.com/in/emmanuel-adeitan)

> ⚠️ This project is for assesment purposes and not production-grade financial fraud prevention.# fraud_detection_project
# fraud_detection_project
