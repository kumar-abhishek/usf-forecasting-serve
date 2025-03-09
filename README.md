# USF Forecasting Serve

## 📌 Project Overview
This project provides an API for sales forecasting using LightGBM. The API is built using FastAPI and is containerized with Docker.

## 🛠 Setting Up a Virtual Environment (Mac/Linux)

To ensure a clean and isolated environment, we use Python's built-in `venv` module.

### 1️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```
✅ Your terminal should now show `(venv)` before the command prompt.

### 2️⃣ Upgrade `pip` and Install Dependencies
```bash
pip install --upgrade pip
pip install -r api/requirements.txt
```

### 3️⃣ Install `pytest` for Testing
```bash
pip install pytest
```

### 4️⃣ Run Tests
```bash
PYTHONPATH=. pytest -v tests/
```
This should run the tests:
```
PYTHONPATH=. pytest -v tests/
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.9.6, pytest-8.3.5, pluggy-1.5.0 -- /Users/kumarabhishek/Documents/usf-forecasting-serve/venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/kumarabhishek/Documents/usf-forecasting-serve
plugins: anyio-4.8.0
collected 6 items

tests/test_api.py::test_status PASSED                                                                                                                                                                    [ 16%]
tests/test_api.py::test_valid_prediction PASSED                                                                                                                                                          [ 33%]
tests/test_api.py::test_invalid_date PASSED                                                                                                                                                              [ 50%]
tests/test_api.py::test_missing_parameters PASSED                                                                                                                                                        [ 66%]
tests/test_model.py::test_model_file_exists[/Users/kumarabhishek/Documents/usf-forecasting-serve/training/lgb_model.txt] PASSED                                                                          [ 83%]
tests/test_model.py::test_model_loading PASSED                                                                                                                                                           [100%]

============================================================================================== 6 passed in 0.42s ===============================================================================================
```


### 5️⃣ Run the API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
✅ The API will now be available at `http://localhost:8000`.

### 6️⃣ Deactivate the Virtual Environment (When Done)
```bash
deactivate
```

---

## 🏗 Running the API with Docker

### 1️⃣ Build the Docker Image
```bash
docker build --no-cache -t forecast-api .
```
➡ **Use `--no-cache` to force fresh dependency installation.**  
🚀 **For faster builds, remove `--no-cache`:**
```bash
docker build -t forecast-api .
```

### 2️⃣ Run the Docker Container
```bash
docker run -p 8000:8000 forecast-api
```
✅ The API will now be available at `http://localhost:8000`.

### 3️⃣ Verify API is Running
```bash
curl -X GET "http://localhost:8000/status"
```
Expected Response:
```json
{"status": "API is running"}
```

### 4️⃣ Send a Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"date":"2013-01-01", "store":1, "item":1}'
```
Expected Output:
```json
{"sales_prediction": 42.5}
```

### 5️⃣ Debugging & Logs
To view container logs in real-time:
```bash
docker logs -f $(docker ps -q)
```

To run an interactive shell inside the container:
```bash
docker run -it --entrypoint /bin/sh forecast-api
```

### 6️⃣ Stop & Cleanup
```bash
docker ps -a  # View all running/stopped containers
docker rm $(docker ps -aq)  # Remove stopped containers
docker rmi forecast-api  # Remove the image
```

## Drift monitoring
```
PYTHONPATH=. python monitoring/drift_monitor.py
```

## 🔍 SHAP Model Explainability

To understand how our trained LightGBM model makes predictions, we use **SHAP (SHapley Additive exPlanations)**. SHAP provides insights into **which features drive the model's predictions**, their importance, and their interaction effects.

### 📊 SHAP Analysis Includes:
1. **SHAP Summary Plot:** Shows how each feature impacts the prediction (positive or negative).  
2. **SHAP Bar Plot:** Ranks feature importance based on average absolute SHAP values.  
3. **SHAP Dependence Plot:** Examines interactions between features and their effect on predictions.

### 📁 Outputs:
All SHAP plots are saved inside the `shap_plots/` directory:
- `shap_summary_plot.png`
- `shap_bar_plot.png`
- `shap_dependence_plot.png`

### 📜 Detailed Explanation:
For a full explanation of these plots and how to interpret them, check out the **[SHAP Explanation Guide](./shap_plots/shap_explanation.md)**.

### 📌 Running SHAP Analysis:
To generate SHAP plots, run:
```bash
python monitoring/shap_analysis.py
```

## 🚀 Improvements & Next Steps
- Implement LSTM-based forecasting.
- Deploy API using AWS Lambda for serverless inference.
- Improve feature engineering with external datasets (weather, holidays).

## 📜 License
This project is licensed under the MIT License.


## Slides
https://docs.google.com/presentation/d/1rxKUrLI-I_UOhm_Bx5lyZ4EvQqqCuHd2/edit#slide=id.p1