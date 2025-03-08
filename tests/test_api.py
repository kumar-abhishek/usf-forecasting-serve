import pytest
from unittest.mock import patch
import os

# Set the environment variable BEFORE importing app.py
os.environ["MODEL_PATH"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "lgb_model.txt")

from fastapi.testclient import TestClient
from api.app import app  # Import the FastAPI app

client = TestClient(app)

# ✅ Test API Status
def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

# ✅ Test Prediction Endpoint with Valid Data
def test_valid_prediction():
    response = client.post("/predict", json={
        "date": "2013-01-01",
        "store": 1,
        "item": 1
    })
    assert response.status_code == 200
    assert "sales_prediction" in response.json()
    assert isinstance(response.json()["sales_prediction"], float)  # Ensure response is a number

# ✅ Test Prediction Endpoint with Invalid Date Format
def test_invalid_date():
    response = client.post("/predict", json={
        "date": "01-01-2013",  # Wrong format
        "store": 1,
        "item": 1
    })
    assert response.status_code == 200  # FastAPI does not auto-reject wrong formats
    assert "error" in response.json()

# ✅ Test Missing Parameters
def test_missing_parameters():
    response = client.post("/predict", json={
        "store": 1,
        "item": 1
    })
    assert response.status_code == 422  # Missing `date` should trigger validation error