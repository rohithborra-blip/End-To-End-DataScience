from pathlib import Path
from typing import List

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[0].parent / "models" / "housing_price_model.joblib"


class HousingPriceModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.features = list(artifact["features"])

    def predict(self, values: List[float]) -> float:
        row = pd.DataFrame([values], columns=self.features)
        prediction = self.pipeline.predict(row)
        return float(prediction[0])
