from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import HousingPriceModel
from .schemas import HouseFeatures, feature_names

app = FastAPI(
    title="California Housing Price Predictor",
    description="Predict median housing prices using a trained RandomForest model.",
    version="1.0.0",
)

model = HousingPriceModel()


class PredictionResponse(BaseModel):
    prediction: float
    unit: str = "median_house_value"


@app.get("/")
def root():
    return {
        "title": app.title,
        "description": app.description,
        "predict_endpoint": "/predict",
        "features": feature_names,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HouseFeatures):
    values = [getattr(payload, name) for name in feature_names]
    try:
        prediction = model.predict(values)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"prediction": prediction, "unit": "median_house_value"}
