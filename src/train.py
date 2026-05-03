from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parents[0].parent / "models" / "housing_price_model.joblib"


def load_data():
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame
    return frame, dataset.feature_names, dataset.target.name


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=12,
                ),
            ),
        ]
    )


def train():
    df, feature_names, target_name = load_data()
    X = df[feature_names]
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": feature_names}, MODEL_PATH)

    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"Evaluation RMSE: {rmse:.3f}")
    print(f"Evaluation R2: {r2:.3f}")

    return MODEL_PATH


if __name__ == "__main__":
    train()
