# End-To-End Data Science Project

This repository contains an end-to-end data science pipeline that:

- collects and preprocesses data from the California Housing dataset,
- trains a regression model using a scikit-learn pipeline,
- saves the trained model artifact,
- exposes a FastAPI service for live prediction.

## Project Structure

- `src/train.py` - trains and saves the model pipeline
- `src/model.py` - loads the saved pipeline and runs predictions
- `src/app.py` - FastAPI application exposing `/predict`
- `src/schemas.py` - request schema for prediction inputs
- `requirements.txt` - Python dependencies
- `models/` - output directory for saved model artifact

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train the model:

```bash
python -m src.train
```

3. Run the API:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## Prediction API

POST `http://localhost:8000/predict`

Example request body:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Example response:

```json
{
  "prediction": 4.235,
  "unit": "median_house_value"
}
```

## Docker Deployment

Build the container image:

```bash
docker build -t housing-price-api .
```

Run the container:

```bash
docker run -p 8000:8000 housing-price-api
```

## Notes

- The dataset is sourced from `sklearn.datasets.fetch_california_housing`.
- The API model returns median house values in normalized units used by the dataset.
