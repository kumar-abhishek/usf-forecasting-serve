import pytest
import pickle
import pandas as pd
import os
import lightgbm as lgb
# from training.preprocess import preprocess_data  # Import preprocessing function


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is tests/
PROJECT_DIR = os.path.dirname(BASE_DIR)  # Go up to project root

# Path inside the container (matches Dockerfile)
MODEL_PATH = os.path.join(PROJECT_DIR, "training", "lgb_model.txt")

@pytest.mark.parametrize("model_file", [MODEL_PATH])
def test_model_file_exists(model_file):
    """Ensure the model file exists in the correct directory."""
    assert os.path.exists(model_file), f"Model file not found at {model_file}"


# ✅ Test Model Loading
def test_model_loading():
    try:
        model = lgb.Booster(model_file=MODEL_PATH)  # Load the model from a text file
        assert model is not None  # Ensure model is loaded
    except FileNotFoundError:
        pytest.fail(f"Model file not found at {MODEL_PATH}")
    except lgb.basic.LightGBMError as e:
        pytest.fail(f"LightGBM failed to load model: {e}")
