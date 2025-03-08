import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# ✅ Ensure `training/` directory exists
os.makedirs("training", exist_ok=True)

# ✅ Load datasets
def load_data(datapath):
    data = pd.read_csv(datapath)
    print(f"✅ Loaded {datapath}: Shape = {data.shape}")
    return data

# Adjust paths if needed
train_df = load_data("data/train.csv")
test_df = load_data("data/test.csv")
sample_df = load_data("data/sample_submission.csv")


# ✅ Preprocess Data
def preprocess_data(train_data, test_data):
    train_data["date"] = pd.to_datetime(train_data["date"])
    test_data["date"] = pd.to_datetime(test_data["date"])

    for df in [train_data, test_data]:
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.dayofweek
        df["year"] = df["date"].dt.year

    features = [col for col in test_data.columns if col not in ["date", "id"]]
    target = "sales"

    train_x, test_x, train_y, test_y = train_test_split(
        train_data[features], train_data[target], test_size=0.2, random_state=2018
    )

    return train_x, test_x, train_y, test_y, features


train_x, test_x, train_y, test_y, features = preprocess_data(train_df, test_df)


# ✅ Train LightGBM Model
def train_lgbm(train_x, train_y, test_x, test_y, features):
    params = {
        "nthread": 10,
        "max_depth": 5,
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "huber", #"regression_l1",
        "metric": "mape",
        "num_leaves": 64,
        "learning_rate": 0.2,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 3.097758978478437,
        "lambda_l2": 2.9482537987198496,
        "verbose": 1,
        "min_child_weight": 6.996211413900573,
        "min_split_gain": 0.037310344962162616,
    }

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(test_x, test_y)

    model = lgb.train(
        params,
        lgb_train,
        3000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    return model


print("🚀 Training LightGBM Model...")
model = train_lgbm(train_x, train_y, test_x, test_y, features)


# ✅ Make Predictions
y_test = model.predict(test_df[features])
sample_df["sales"] = y_test

# ✅ Save predictions
sample_df.to_csv("training/lgb_predictions.csv", index=False)
print("✅ Predictions saved to `training/lgb_predictions.csv`")


# ✅ Save the Trained Model
model.save_model('training/lgb_model.txt', num_iteration=model.best_iteration) 
# model.save_model("training/lgb_model.pkl")
print("✅ Model saved to `training/lgb_model.txt`")


if __name__ == "__main__":
    print("🎯 Training Complete!")