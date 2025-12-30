from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import os

app = FastAPI(title="Energy Consumption Prediction API")

MODEL_PATH = "models/energy_lstm_model.h5"
SCALER_PATH = "models/scaler.joblib"

# Load model and scaler at startup
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

class PredictionRequest(BaseModel):
    data: list # A list of 100 floats representing consumption over time

@app.get("/")
def read_root():
    return {"message": "Energy Consumption Prediction API is running. Use /predict endpoint."}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded. Run training first.")
    
    if len(request.data) < 100:
        raise HTTPException(status_code=400, detail="Input data must contain at least 100 time steps.")

    # Prepare input
    # 1. Scale input
    input_data = np.array(request.data[-100:]).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    
    # 2. Reshape [1, 100, 1]
    final_input = scaled_input.reshape(1, 100, 1)
    
    # 3. Predict
    prediction_scaled = model.predict(final_input)
    
    # 4. Inverse scale
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return {
        "prediction": float(prediction[0][0]),
        "unit": "MWh"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
