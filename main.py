# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.losses import MeanAbsoluteError
import numpy as np

# Load model
model = load_model("model/price_predictor_lstm.h5", custom_objects={"mae": MeanAbsoluteError()})

# App
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "⚡️ Welcome to the ChargeGenie Price Predictor API!"}

class PriceInput(BaseModel):
    price_history: list

@app.get("/")
def home():
    return {"message": "API is live"}

@app.post("/predict")
def predict(input_data: PriceInput):
    if len(input_data.price_history) != 24:
        raise HTTPException(status_code=400, detail="Provide exactly 24 values.")
    
    X = np.array(input_data.price_history).reshape(1, 24, 1)
    prediction = model.predict(X)[0][0]
    return {"predicted_price_kWh": round(float(prediction), 5)}
