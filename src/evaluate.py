import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import joblib
from data_processing import load_data, preprocess_data, prepare_ml_data
import math

MODEL_PATH = "models/energy_lstm_model.h5"
SCALER_PATH = "models/scaler.joblib"
DATA_URL = "https://raw.githubusercontent.com/MohamadNach/Machine-Learning-to-Predict-Energy-Consumption/master/events.csv"

def evaluate():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found. Please run train.py first.")
        return

    print("Loading model and scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Loading and preparing data...")
    df_raw = load_data(DATA_URL)
    df_processed = preprocess_data(df_raw)
    _, _, (X_test, y_test), _ = prepare_ml_data(df_processed)

    print("Generating predictions...")
    test_predict = model.predict(X_test)
    
    # Inverse transform
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test_inv, test_predict_inv)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, test_predict_inv)
    r2 = r2_score(y_test_inv, test_predict_inv)

    print("\nEvaluation Results:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 :  {r2:.2f}")

    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv, label='Actual Consumption')
    plt.plot(test_predict_inv, label='Predicted Consumption')
    plt.title('Energy Consumption Prediction - Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Consumption (MWh)')
    plt.legend()
    plt.savefig('notebooks/evaluation_plot.png')
    print("Evaluation plot saved to notebooks/evaluation_plot.png")

if __name__ == "__main__":
    import os
    evaluate()
