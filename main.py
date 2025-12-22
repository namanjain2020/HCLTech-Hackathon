from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# -------------------------
# App init
# -------------------------
app = FastAPI(title="Retail CLV Prediction API")

# -------------------------
# Load model safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_clv_model.pkl")

model = joblib.load(MODEL_PATH)

# -------------------------
# Expected feature order
# -------------------------
FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "total_quantity",
    "total_spend",
    "avg_order_value",
    "unique_products",
    "country"
]

# -------------------------
# Input schemas
# -------------------------
class CLVInput(BaseModel):
    recency: int
    frequency: int
    total_quantity: float
    total_spend: float
    avg_order_value: float
    unique_products: int
    country: str


class BatchCLVInput(BaseModel):
    records: list[CLVInput]

# -------------------------
# Health
# -------------------------
@app.get("/")
def health():
    return {"status": "API running"}

# -------------------------
# Predict (single)
# -------------------------
@app.post("/predict")
def predict_clv(data: CLVInput):

    df = pd.DataFrame([{
        "recency": data.recency,
        "frequency": data.frequency,
        "total_quantity": data.total_quantity,
        "total_spend": data.total_spend,
        "avg_order_value": data.avg_order_value,
        "unique_products": data.unique_products,
        "country": data.country
    }])

    pred = model.predict(df)[0]

    return {
        "predicted_30d_clv": round(float(pred), 2)
    }

# -------------------------
# Batch prediction
# -------------------------
@app.post("/batch_predict")
def batch_predict(data: BatchCLVInput):

    df = pd.DataFrame([r.dict() for r in data.records])
    preds = model.predict(df)

    return {
        "predictions": [round(float(p), 2) for p in preds]
    }

# -------------------------
# Model info
# -------------------------
@app.get("/model_info")
def model_info():
    return {
        "model_type": str(model.named_steps["model"]),
        "features": FEATURE_COLUMNS,
        "preprocessing": "StandardScaler + OneHotEncoder",
        "target": "30-day future spend"
    }

# -------------------------
# Metrics (offline evaluation summary)
# -------------------------
@app.get("/metrics")
def metrics():
    return {
        "metric_note": "Metrics computed during training on train window",
        "primary_metric": "MAE",
        "handling_zero_clv": "Included",
        "log_target": "Not applied in this version"
    }
