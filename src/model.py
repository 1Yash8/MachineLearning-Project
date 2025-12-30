from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_lstm_model(input_shape=(100, 1)):
    """
    Construct the LSTM model as defined in the original notebook.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

from sklearn.ensemble import RandomForestRegressor

def create_baseline_model():
    """
    Construct a simple Random Forest regressor as a baseline.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model
