import os
import argparse
from data_processing import load_data, preprocess_data, prepare_ml_data
from model import create_lstm_model
import tensorflow as tf

# Constants
DATA_URL = "https://raw.githubusercontent.com/MohamadNach/Machine-Learning-to-Predict-Energy-Consumption/master/events.csv"
MODEL_SAVE_PATH = "models/energy_lstm_model.h5"

def train(model_type='lstm'):
    print("Loading data...")
    df_raw = load_data(DATA_URL)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df_raw)
    
    print("Preparing data for ML...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_ml_data(df_processed)
    
    print(f"X_train shape: {X_train.shape}")
    
    if model_type == 'lstm':
        print("Building LSTM model...")
        model = create_lstm_model(input_shape=(X_train.shape[1], 1))
        
        print("Starting training...")
        # Using 60 epochs as per original notebook
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=20,
            verbose=1
        )
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        model.save(MODEL_SAVE_PATH)
    else:
        from model import create_baseline_model
        import joblib
        print("Building Baseline (Random Forest) model...")
        # Flatten input for RF
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = create_baseline_model()
        model.fit(X_train_flat, y_train)
        print("Saving baseline model...")
        joblib.dump(model, "models/baseline_rf_model.joblib")
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'baseline'])
    args = parser.parse_args()
    train(model_type=args.model)
