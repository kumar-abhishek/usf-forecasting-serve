# 📊 Model Drift Detection & Monitoring

To ensure our forecasting model remains accurate over time, we have implemented drift detection to monitor changes in data distribution and model predictions.

## 🚀 How Drift Monitoring Works

	1.	Loads the trained model and historical sales data (train.csv).

	2.	Loads recent sales data (recent_sales.csv) for comparison.

	3.	Detects data drift using statistical tests (e.g., Kolmogorov-Smirnov test) to compare feature distributions.

	4.	Detects prediction drift by checking if model outputs have significantly changed.

	5.	Saves a drift report (monitoring/drift_report.json) summarizing detected drift.

	6.	Triggers an alert if drift is detected, suggesting model retraining.

## 🛠 Running Drift Detection Manually

To manually check for drift, run:

PYTHONPATH=. python monitoring/drift_monitor.py

The script will generate a drift report and indicate whether retraining is needed.

## 🔄 Automating Drift Detection via GitHub Actions

Drift detection is automatically run daily at midnight UTC via GitHub Actions.

	•	If drift is detected, the workflow fails and suggests retraining.

	•	If no drift is detected, the workflow passes ✅.

To manually trigger the workflow:
	1.	Go to GitHub → Actions → Drift Detection Workflow.
	2.	Click “Run workflow”.

### 📊 Example Output (Drift Report)
```
{
    "data_drift": {
        "month": {"KS Statistic": 0.91, "p-value": 0.00, "Drift Detected": true},
        "day": {"KS Statistic": 0.03, "p-value": 0.00, "Drift Detected": true}
    },
    "prediction_drift": {
        "MAE Shift": 5.06,
        "Drift Detected": true
    },
    "timestamp": "2025-03-09 00:47:59"
}
```

🚨 Drift detected! Consider retraining the model.

## 🔔 Next Steps
	•	If drift is detected, review the drift report and retrain the model.
	•	If no drift is detected, the model remains stable.

For more details, see monitoring/drift_monitor.py. 🚀