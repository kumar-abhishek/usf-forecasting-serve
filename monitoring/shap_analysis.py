# pip install shap matplotlib joblib


import os
import shap
import joblib
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# Ensure SHAP plots directory exists
PLOTS_DIR = "shap_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the trained model
MODEL_PATH = "training/lgb_model.txt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print(f"✅ Loading model from {MODEL_PATH}")
# # Load the model
model = lgb.Booster(model_file=MODEL_PATH)
print("Model loaded successfully!")

# Load test data
TEST_DATA_PATH = "data/test.csv"
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

print(f"✅ Loading test data from {TEST_DATA_PATH}")
test_df = pd.read_csv(TEST_DATA_PATH)

# Ensure 'date' column exists
if "date" not in test_df.columns:
    raise KeyError("❌ 'date' column missing in test.csv. Check your data!")

# 🔹 Feature Engineering: Extract `month`, `day`, `year` from `date`
test_df["date"] = pd.to_datetime(test_df["date"])  # Convert to datetime
test_df["month"] = test_df["date"].dt.month
test_df["day"] = test_df["date"].dt.day
test_df["year"] = test_df["date"].dt.year

# 🔹 Get feature names from trained model
model_features = model.feature_name()

# 🔹 Ensure test data contains the same features
if not set(model_features).issubset(set(test_df.columns)):
    missing_features = set(model_features) - set(test_df.columns)
    raise KeyError(f"❌ Missing features in test data: {missing_features}")

# Select only the features the model was trained on
X_test = test_df[model_features]

# Initialize SHAP explainer
print("🚀 Computing SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# 🔹 1️⃣ SHAP Summary Plot (Global Feature Importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
summary_plot_path = os.path.join(PLOTS_DIR, "shap_summary_plot.png")
plt.savefig(summary_plot_path, bbox_inches="tight")
print(f"📊 SHAP Summary Plot saved: {summary_plot_path}")

# 🔹 2️⃣ SHAP Bar Plot (Mean Absolute Impact)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
bar_plot_path = os.path.join(PLOTS_DIR, "shap_bar_plot.png")
plt.savefig(bar_plot_path, bbox_inches="tight")
print(f"📊 SHAP Bar Plot saved: {bar_plot_path}")

# 🔹 3️⃣ SHAP Dependence Plot (Example: 'month' feature)
if "month" in X_test.columns:
    plt.figure(figsize=(8, 5))
    # shap.dependence_plot("month", shap_values, X_test, show=False)
    shap.dependence_plot("month", shap_values.values, X_test, show=False)  # Convert to NumPy
    dependence_plot_path = os.path.join(PLOTS_DIR, "shap_dependence_plot.png")
    plt.savefig(dependence_plot_path, bbox_inches="tight")
    print(f"📊 SHAP Dependence Plot saved: {dependence_plot_path}")

print("✅ SHAP analysis completed. Check the `shap_plots/` folder for visualizations!")