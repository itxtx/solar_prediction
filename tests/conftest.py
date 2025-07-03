"""
Shared pytest fixtures for the solar prediction test suite.
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Add the project root to the path so we can import our modules
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def sample_data_path():
    """Path to the sample CSV file."""
    return project_root / "SolarPrediction_sample.csv"

@pytest.fixture(scope="session")
def sample_data(sample_data_path):
    """Load the sample solar prediction dataset."""
    if not sample_data_path.exists():
        pytest.skip(f"Sample data file not found at {sample_data_path}")
    
    df = pd.read_csv(sample_data_path)
    return df

@pytest.fixture(scope="session")
def small_sample_data(sample_data):
    """A smaller subset of sample data for quick tests."""
    return sample_data.head(50).copy()

@pytest.fixture
def numpy_arrays_2d():
    """Small numpy arrays for testing purposes."""
    np.random.seed(42)
    X = np.random.randn(20, 5, 3)  # 20 samples, 5 timesteps, 3 features
    y = np.random.randn(20, 1)     # 20 samples, 1 target
    return X, y

@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    
    # Create minimal test data
    data = {
        'Timestamp': ['2023-01-01 08:00:00', '2023-01-01 09:00:00', '2023-01-01 10:00:00'],
        'Radiation': [100.0, 200.0, 300.0],
        'Temperature': [15.0, 16.0, 17.0],
        'Pressure': [1013.0, 1014.0, 1015.0],
        'Humidity': [60.0, 65.0, 70.0],
        'WindSpeed': [2.0, 2.5, 3.0]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def memory_budget():
    """Memory budget for performance tests (in MB)."""
    return 500  # 500MB budget

@pytest.fixture
def time_budget():
    """Time budget for performance tests (in seconds)."""
    return {
        'data_preparation': 2.0,
        'model_training': 10.0,
        'inference': 1.0,
        'memory_test': 5.0
    }
