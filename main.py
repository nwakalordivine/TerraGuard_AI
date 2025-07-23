from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = FastAPI()
model = joblib.load("flood_model.pkl")


class FloodInput(BaseModel):
    location: str
    rainfall_7_days: list
    soil_moisture: float
    elevation: float


@app.get("/")
def home():
    return "Welcome to TerraGuard AI"


@app.post("/predict")
def predict(data: FloodInput):
    features = np.array(data.rainfall_7_days + [data.soil_moisture, data.elevation]).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]
    prediction = "flood" if prob >= 0.6 else "no_flood"
    flood_date = (datetime.now() + timedelta(days=np.argmax(data.rainfall_7_days) + 1)).strftime("%Y-%m-%d")

    return {
        "location": data.location,
        "likelihood": round(prob * 100, 2),
        "flood_expected": prediction == "flood",
        "date": flood_date
    }


app = app
