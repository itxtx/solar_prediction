"""
Test utilities: scaling inverse, time parsing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import the modules we want to test
from solar_prediction.data_prep import _parse_time_to_minutes, _parse_time_to_minutes_vectorized


class TestTimeParsingUtils:
    """Test time parsing utility functions."""
    
    def test_parse_time_to_minutes_basic(self):
        """Test basic time string parsing."""
        # Test valid time strings
        assert _parse_time_to_minutes("08:30:00") == 8 * 60 + 30
        assert _parse_time_to_minutes("12:00:00") == 12 * 60
        assert _parse_time_to_minutes("23:59:59") == 23 * 60 + 59 + 59/60
        assert _parse_time_to_minutes("00:00:00") == 0
        
    def test_parse_time_to_minutes_invalid(self):
        """Test parsing invalid time strings."""
        # Test invalid inputs
        assert pd.isna(_parse_time_to_minutes("25:00:00"))  # Invalid hour
        assert pd.isna(_parse_time_to_minutes("12:60:00"))  # Invalid minute
        assert pd.isna(_parse_time_to_minutes("12:30:61"))  # Invalid second
        assert pd.isna(_parse_time_to_minutes("invalid"))
        assert pd.isna(_parse_time_to_minutes(None))
        assert pd.isna(_parse_time_to_minutes(np.nan))
        
    def test_parse_time_to_minutes_timestamps(self):
        """Test parsing pandas Timestamp objects."""
        ts = pd.Timestamp("2023-01-01 14:30:45")
        expected = 14 * 60 + 30 + 45/60
        assert abs(_parse_time_to_minutes(ts) - expected) < 0.01
        
    def test_parse_time_vectorized(self):
        """Test vectorized time parsing function."""
        time_series = pd.Series([
            "08:30:00",
            "12:00:00", 
            "23:59:59",
            "invalid",
            np.nan
        ])
        
        result = _parse_time_to_minutes_vectorized(time_series)
        
        # Check valid times
        assert abs(result[0] - (8 * 60 + 30)) < 0.01
        assert abs(result[1] - (12 * 60)) < 0.01
        assert abs(result[2] - (23 * 60 + 59 + 59/60)) < 0.01
        
        # Check invalid times are NaN
        assert np.isnan(result[3])
        assert np.isnan(result[4])
        
    def test_parse_time_vectorized_empty_series(self):
        """Test vectorized parsing with empty series."""
        empty_series = pd.Series(dtype=object)
        result = _parse_time_to_minutes_vectorized(empty_series)
        assert len(result) == 0
        
    def test_parse_time_vectorized_all_nan(self):
        """Test vectorized parsing with all NaN values."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        result = _parse_time_to_minutes_vectorized(nan_series)
        assert all(np.isnan(result))


class TestScalingInverse:
    """Test scaling and inverse scaling operations."""
    
    def test_standard_scaler_inverse(self):
        """Test StandardScaler fit, transform, and inverse_transform."""
        # Create test data
        data = np.array([[100, 200, 300], [150, 250, 350], [200, 300, 400]], dtype=float)
        
        scaler = StandardScaler()
        
        # Fit and transform
        scaled_data = scaler.fit_transform(data)
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_data.std(axis=0, ddof=0), 1, atol=1e-10)
        
        # Test inverse transform
        recovered_data = scaler.inverse_transform(scaled_data)
        assert np.allclose(data, recovered_data, rtol=1e-10)
        
    def test_minmax_scaler_inverse(self):
        """Test MinMaxScaler fit, transform, and inverse_transform."""
        # Create test data
        data = np.array([[100, 200, 300], [150, 250, 350], [200, 300, 400]], dtype=float)
        
        scaler = MinMaxScaler()
        
        # Fit and transform
        scaled_data = scaler.fit_transform(data)
        
        # Check that data is scaled to [0, 1] range
        assert np.all(scaled_data >= 0)
        assert np.all(scaled_data <= 1)
        assert np.allclose(scaled_data.min(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_data.max(axis=0), 1, atol=1e-10)
        
        # Test inverse transform
        recovered_data = scaler.inverse_transform(scaled_data)
        assert np.allclose(data, recovered_data, rtol=1e-10)
        
    def test_scaler_with_single_column(self):
        """Test scaling operations with single column data."""
        data = np.array([[100], [150], [200]], dtype=float)
        
        for ScalerClass in [StandardScaler, MinMaxScaler]:
            scaler = ScalerClass()
            scaled_data = scaler.fit_transform(data)
            recovered_data = scaler.inverse_transform(scaled_data)
            assert np.allclose(data, recovered_data, rtol=1e-10)
            
    def test_scaler_with_constant_values(self):
        """Test scaling behavior with constant values."""
        # All values are the same
        data = np.array([[100], [100], [100]], dtype=float)
        
        # StandardScaler should handle this gracefully
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        recovered_data = scaler.inverse_transform(scaled_data)
        
        # Should recover original data even with zero variance
        assert np.allclose(data, recovered_data, rtol=1e-10)
        
    def test_scaler_with_nan_values(self):
        """Test scaling behavior with NaN values."""
        data = np.array([[100, np.nan], [150, 250], [200, 300]], dtype=float)
        
        # Test that scalers handle NaN appropriately 
        # (this might require preprocessing in real applications)
        scaler = StandardScaler()
        
        # StandardScaler will propagate NaN values
        scaled_data = scaler.fit_transform(data)
        assert np.isnan(scaled_data[0, 1])  # NaN should remain NaN
        
    def test_inverse_scaling_consistency(self):
        """Test that multiple fit-transform-inverse cycles are consistent."""
        np.random.seed(42)
        data = np.random.randn(50, 3) * 100 + 500  # Random data with offset and scale
        
        for ScalerClass in [StandardScaler, MinMaxScaler]:
            scaler = ScalerClass()
            
            # First cycle
            scaled1 = scaler.fit_transform(data)
            recovered1 = scaler.inverse_transform(scaled1)
            
            # Second cycle with same scaler
            scaled2 = scaler.transform(data)
            recovered2 = scaler.inverse_transform(scaled2)
            
            # Results should be identical
            assert np.allclose(scaled1, scaled2, rtol=1e-10)
            assert np.allclose(recovered1, recovered2, rtol=1e-10)
            assert np.allclose(data, recovered1, rtol=1e-10)
            assert np.allclose(data, recovered2, rtol=1e-10)


class TestUtilityEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_time_parsing_edge_cases(self):
        """Test edge cases in time parsing."""
        # Test with seconds as float
        assert abs(_parse_time_to_minutes("12:30:30.5") - (12 * 60 + 30.5)) < 0.01
        
        # Test with missing seconds
        assert abs(_parse_time_to_minutes("12:30") - (12 * 60 + 30)) < 0.01
        
    def test_scaling_edge_cases(self):
        """Test edge cases in scaling operations."""
        # Test with very large numbers
        large_data = np.array([[1e10], [2e10], [3e10]], dtype=float)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(large_data)
        recovered = scaler.inverse_transform(scaled)
        assert np.allclose(large_data, recovered, rtol=1e-6)
        
        # Test with very small numbers
        small_data = np.array([[1e-10], [2e-10], [3e-10]], dtype=float)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(small_data)
        recovered = scaler.inverse_transform(scaled)
        assert np.allclose(small_data, recovered, rtol=1e-6)
        
    def test_vectorized_vs_scalar_time_parsing(self):
        """Test that vectorized and scalar time parsing give same results."""
        test_times = ["08:30:00", "12:00:00", "23:59:59", "00:00:00"]
        
        # Scalar results
        scalar_results = [_parse_time_to_minutes(t) for t in test_times]
        
        # Vectorized results
        time_series = pd.Series(test_times)
        vectorized_results = _parse_time_to_minutes_vectorized(time_series)
        
        # Should be identical
        for scalar, vectorized in zip(scalar_results, vectorized_results):
            assert abs(scalar - vectorized) < 1e-10
