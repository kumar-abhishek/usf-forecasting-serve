
# USF Forecasting Serve

## 📌 Project Overview
This project provides an API for sales forecasting using LightGBM. The API is built using FastAPI and is containerized with Docker.

## 🛠 Running the API with Docker

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
{"sales_prediction": 8.428806020645183}
```

### 5️⃣ Debugging & Logs
To view container logs in real-time:
```bash
docker logs -f <container_id>
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

## 🏗 Model Training

### 1️⃣ Install Dependencies
```bash
pip install -r api/requirements.txt
```

### 2️⃣ Train the Model
```bash
python training/train.py
```
This will generate `lgb_model.txt` inside the `training/` folder.

## 🚀 Improvements & Next Steps
- Implement LSTM-based forecasting.
- Deploy API using AWS Lambda for serverless inference.
- Improve feature engineering with external datasets (weather, holidays).

## 📜 License
This project is licensed under the MIT License.