# 📊 Demand Forecasting with LightGBM: Project Report

## **1️⃣ Introduction**
- **Goal**: Build a machine learning model to forecast sales.
- **Dataset**: Historical store-item sales data.
- **Tools Used**: Python, FastAPI, LightGBM, SHAP, AWS, Docker, CI/CD.

## **2️⃣ Approach**
### 📌 **Data Preprocessing & Feature Engineering**
- **Extracted time-based features**: `month`, `day`, `year`.
- **Handled missing values and categorical encoding**.
- **Train-Test Split**: 80% training, 20% validation.

### 📌 **Model Selection & Training**
- Used **LightGBM** for fast and accurate forecasting.
- **Tuned hyperparameters** using Bayesian Optimization.
- Evaluated alternative approaches (**ARIMA, Prophet, XGBoost**).

## **3️⃣ Model Evaluation & Results**
- **Evaluation Metrics**: MAPE (Mean Absolute Percentage Error).
- **Best Model Performance**:
  - LightGBM (MAPE: **0.1327** with `regression_l1` loss).
  - Compared with `huber` loss (MAPE: **0.1429**, worse performance).

## **4️⃣ SHAP Model Explainability**
- **Feature Importance Insights**:
  - `item` has the highest impact on predictions.
  - `month` confirms seasonality in sales.
- **SHAP Analysis**:
  - **Summary Plot**: Shows feature impact distribution.
  - **Bar Plot**: Ranks features by importance.
  - **Dependence Plot**: Highlights seasonal dependencies.

## **5️⃣ Deployment & MLOps**
- **API Deployment**:
  - Built a **FastAPI-based inference server**.
  - Dockerized the application.
  - Configured AWS for cloud deployment.

- **CI/CD Pipeline**:
  - GitHub Actions for automated testing and deployment.

## **6️⃣ Recommendations & Next Steps**
### 🔹 Improvements:
- Use **time-series cross-validation** for better evaluation.
- Implement **model retraining on live data**.
- Experiment with **deep learning-based forecasting (LSTMs, Transformers)**.

---
🔗 **For additional details, see [SHAP Analysis](./shap_explanation.md).**