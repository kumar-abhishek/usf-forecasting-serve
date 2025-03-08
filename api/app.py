from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import os 
import lightgbm as lgb
from datetime import datetime

# Correct the model path
model_path = os.path.join(os.path.dirname(__file__), "training/lgb_model.txt")

# Verify if the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# # Load the model
model = lgb.Booster(model_file=model_path)
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