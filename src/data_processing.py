import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_data(url):
    """
    Load data from a given URL.
    """
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    """
    Perform initial preprocessing: column dropping, datetime conversion, 
    and feature extraction.
    """
    # Drop irrelevant columns (based on original notebook analysis)
    # The notebook dropped specific indices, but it's safer to use names if known.
    # From the notebook: dataset.drop(dataset.columns[[0, 1, 2, 4]], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 4]], axis=1, inplace=True)
    
    # Rename columns for clarity if needed
    df.columns = ['DateTime', 'Consumption']
    
    # Convert to datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Extract time-based features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Date'] = df.index.day
    df['Time'] = df.index.hour
    df['Week'] = df.index.isocalendar().week
    df['Day'] = df.index.dayofweek
    
    # Handling potential outliers or specific range as done in notebook
    # Original: dataset = dataset[71:-121]
    df = df.iloc[71:-121]
    
    return df

def get_sequences(data, time_step=100):
    """
    Create sequences for LSTM training.
    """
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        x.append(a)
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

def scale_data(data, feature_range=(0, 1), save_path=None):
    """
    Normalize data using MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    if save_path:
        import joblib
        joblib.dump(scaler, save_path)
        
    return scaled_data, scaler

def prepare_ml_data(df, time_step=100, train_split=0.8, val_split=0.2):
    """
    Complete pipeline to prepare data for the model.
    """
    target = df['Consumption'].values
    
    # Scaling
    scaled_target, scaler = scale_data(target, save_path='models/scaler.joblib')
    
    # Split
    training_size = int(len(scaled_target) * train_split)
    test_size = len(scaled_target) - training_size
    val_size = int(training_size * val_split)
    
    train_data = scaled_target[0:training_size - val_size]
    test_data = scaled_target[training_size:len(scaled_target)]
    val_data = scaled_target[len(scaled_target) - test_size - val_size : len(scaled_target) - test_size]
    
    X_train, y_train = get_sequences(train_data, time_step)
    X_test, y_test = get_sequences(test_data, time_step)
    X_val, y_val = get_sequences(val_data, time_step)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
