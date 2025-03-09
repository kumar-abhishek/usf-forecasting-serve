import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
from scipy.stats import ks_2samp
from datetime import datetime
from api.app import model  # Load the trained model
from sklearn.metrics import mean_absolute_error

# Load historical training data for reference
TRAIN_DATA_PATH = "data/train.csv"
train_df = pd.read_csv(TRAIN_DATA_PATH)

# Features used in the model
FEATURES = ["store", "item", "month", "day", "year"]

# Load recent data for monitoring
def load_recent_data(filepath):
    """Load recent data to compare against training distribution."""
    return pd.read_csv(filepath)

# Compute data drift using KS Test
def detect_data_drift(reference_data, new_data, threshold=0.05):
    """Detect data drift using the Kolmogorov-Smirnov (KS) test."""
    drift_results = {}
    for col in FEATURES:
        stat, p_value = ks_2samp(reference_data[col], new_data[col])
        drift_results[col] = {"KS Statistic": stat, "p-value": p_value, "Drift Detected": p_value < threshold}
    return drift_results

# Compute prediction drift using MAE shift
def detect_prediction_drift(reference_preds, new_preds, threshold=0.1):
    """Detect prediction drift based on Mean Absolute Error (MAE) shift."""
    mae_ref = np.mean(np.abs(reference_preds - train_df["sales"][:len(reference_preds)]))
    mae_new = np.mean(np.abs(new_preds - train_df["sales"][:len(new_preds)]))
    drift_score = abs(mae_ref - mae_new) / mae_ref
    return {"MAE Shift": drift_score, "Drift Detected": drift_score > threshold}

# Feature Engineering Function
def add_time_features(df):
    """Add time-based features (month, day, year) from the date column."""
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year
    return df.drop(columns=["date"])  # Drop date after extracting features

# Load historical training data
train_df = pd.read_csv(TRAIN_DATA_PATH)
train_df = add_time_features(train_df)  # Apply feature engineering

DRIFT_REPORT_PATH = "monitoring/drift_report.json"

# Convert booleans to JSON-compatible format
def convert_booleans(obj):
    """Recursively convert boolean values to string for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_booleans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_booleans(i) for i in obj]
    elif isinstance(obj, bool):
        return str(obj)  # Convert True/False to "True"/"False"
    else:
        return obj


def convert_numpy(obj):
    """Recursively convert NumPy data types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):  # Convert np.float64, np.bool_ to Python types
        return obj.item()
    else:
        return obj
    
# Run model drift checks
def monitor_drift(recent_data_path):
    print("🔍 Running Model Drift Monitoring...")

    new_data = pd.read_csv(recent_data_path)
    new_data = add_time_features(new_data)  # Ensure time-based features exist

    # Ensure feature consistency
    new_data = new_data[FEATURES]
    train_features = train_df[FEATURES]  # Ensure train data has the same features

    # Detect data drift
    data_drift_results = detect_data_drift(train_features, new_data)

    # Get predictions
    new_preds = model.predict(new_data)
    reference_preds = model.predict(train_features[:len(new_data)])

    # Detect prediction drift
    prediction_drift_results = detect_prediction_drift(reference_preds, new_preds)

    # Prepare drift report
    drift_report = {
        "data_drift": data_drift_results,
        "prediction_drift": prediction_drift_results,
        "timestamp": str(datetime.now())
    }
    print('drift_report: ', drift_report)
    drift_report = convert_numpy(drift_report)


    # Save report
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(drift_report, f, indent=4)
    
    # Check if the file was saved
    if os.path.exists(DRIFT_REPORT_PATH):
        print(f"✅ Drift Report Saved: {DRIFT_REPORT_PATH}")
    else:
        print("❌ ERROR: Drift report file was NOT saved!")

    # If drift detected, trigger retraining
    if any([res["Drift Detected"] for res in data_drift_results.values()]) or prediction_drift_results["Drift Detected"]:
        print("🚨 Drift Detected! Retraining Needed.")
        retrain_model()
    else:
        print("✅ No Significant Drift Detected.")


# Placeholder function for retraining
def retrain_model():
    """Placeholder function for triggering retraining."""
    print("🔄 Triggering model retraining...")
    # Here you can call `training/train.py` or an automated pipeline

# Run monitoring (for example, using `data/recent_sales.csv`)
if __name__ == "__main__":
    monitor_drift("data/recent_sales.csv")