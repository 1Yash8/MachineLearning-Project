import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import preprocess_data, scale_data, get_sequences

def test_preprocess_data():
    # Create dummy data
    data = {
        'col1': [1]*1000,
        'col2': [2]*1000,
        'col3': [3]*1000,
        'DateTime': pd.date_range(start='1/1/2020', periods=1000, freq='H'),
        'col4': [4]*1000,
        'Consumption': np.random.rand(1000)
    }
    df = pd.DataFrame(data)
    
    processed_df = preprocess_data(df)
    
    assert 'Year' in processed_df.columns
    assert 'Month' in processed_df.columns
    assert 'Consumption' in processed_df.columns
    assert processed_df.index.name == 'DateTime'
    # Check slicing (71:-121) -> 1000 - 71 - 121 = 808
    assert len(processed_df) == 808

def test_scale_data():
    data = np.array([10, 20, 30, 40, 50])
    scaled_data, scaler = scale_data(data)
    
    assert np.max(scaled_data) == 1.0
    assert np.min(scaled_data) == 0.0

def test_get_sequences():
    data = np.array([[1], [2], [3], [4], [5]])
    time_step = 2
    x, y = get_sequences(data, time_step)
    
    # data[0:2, 0] -> [1, 2], data[2, 0] -> 3
    # data[1:3, 0] -> [2, 3], data[3, 0] -> 4
    # len(data) - time_step - 1 = 5 - 2 - 1 = 2
    assert len(x) == 2
    assert np.array_equal(x[0], [1, 2])
    assert y[0] == 3
