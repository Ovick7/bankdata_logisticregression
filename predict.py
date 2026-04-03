import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "../models/logistic_regression_model.pkl"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def predict(input_data: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    pipeline = load_model(model_path)
    processed = preprocess_input(input_data)
    predictions = pipeline.predict(processed)
    probabilities = pipeline.predict_proba(processed)[:, 1]

    result = input_data.copy()
    result["predicted_subscription"] = ["yes" if p == 1 else "no" for p in predictions]
    result["subscription_probability"] = probabilities.round(4)
    return result


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 2500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 300,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }])

    result = predict(sample)
    print("\nPrediction Result:")
    print(result[["predicted_subscription", "subscription_probability"]].to_string(index=False))
