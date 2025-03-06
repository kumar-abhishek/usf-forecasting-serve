from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import os 

# Correct the model path
model_path = os.path.join(os.path.dirname(__file__), "training/lgb_model.pkl")

# Verify if the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Initialize API
app = FastAPI()

# Define request schema
class PredictionRequest(BaseModel):
    date: str
    store: int
    item: int

@app.get("/status")
def status():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Prepare input data
    input_data = pd.DataFrame([request.dict()])
    prediction = model.predict(input_data)
    return {"sales_prediction": prediction[0]}