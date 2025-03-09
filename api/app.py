from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import os 
import lightgbm as lgb
from datetime import datetime


import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow overriding model path via environment variable
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "lgb_model.txt")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)  # Use environment variable if set

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"✅ Model to be loaded from: {MODEL_PATH}")

# Verify if the file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the model
model = lgb.Booster(model_file=MODEL_PATH)
print("Model loaded successfully!")

# Initialize API
app = FastAPI()

# Define request schema
class PredictionRequest(BaseModel):
    date: str # Incoming date as a string
    store: int
    item: int

@app.get("/status")
def status():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert date string to datetime
    try:
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    # Create a DataFrame matching the model input format
    input_data = pd.DataFrame([{
        "store": request.store,
        "item": request.item,
        "month": date_obj.month,
        "day": date_obj.day,
        "year": date_obj.year
    }])

    # Make prediction
    prediction = model.predict(input_data)
    
    return {"sales_prediction": float(prediction[0])}