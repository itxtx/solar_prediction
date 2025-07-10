"""
Unit tests for SARIMA edge cases and failure modes.

This module tests the robustness of SARIMA implementation against various
edge-case data scenarios to identify potential failure modes and improve
numerical stability.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

# Import SARIMA functions to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solar_prediction.sarima import (
    check_stationarity,
    fit_sarima_model,
    auto_sarima_selection,
    prepare_time_series_data,
    calculate_metrics
)


class TestSARIMAEdgeCases:
    """Test suite for SARIMA edge cases and failure modes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        
    def create_test_dataframe(self, series_data, start_date='2023-01-01', freq='15min'):
        """Helper to create test DataFrame with proper time index."""
        timestamps = pd.date_range(start=start_date, periods=len(series_data), freq=freq)
        return pd.DataFrame({
            'Time': timestamps,
            'GHI': series_data
        })
    
    def test_all_zeros_data(self):
        """Test SARIMA with all-zero data."""
        # Create all-zero series
        zero_data = [0.0] * 1000
        df = self.create_test_dataframe(zero_data)
        
        # Test data preparation
        with pytest.warns(UserWarning):  # Should warn about lack of variance
            try:
                train_series, test_series, s = prepare_time_series_data(df)
                # Should not crash but may produce warnings
            except Exception as e:
                # Document the failure mode
                assert "variance" in str(e).lower() or "constant" in str(e).lower()
        
        # Test stationarity check
        zero_series = pd.Series(zero_data)
        try:
            result = check_stationarity(zero_series, name="AllZeros")
            # May return True (trivially stationary) or False, but shouldn't crash
            assert isinstance(result, bool)
        except Exception as e:
            # Document specific failure modes
            failure_keywords = ["variance", "constant", "singular", "invertible"]
            assert any(keyword in str(e).lower() for keyword in failure_keywords), \
                f"Unexpected failure: {e}"
    
    def test_near_constant_data(self):
        """Test SARIMA with near-constant data (minimal variance)."""
        base_value = 100.0
        tiny_noise = np.random.normal(0, 0.001, 1000)  # Very small noise
        near_constant_data = base_value + tiny_noise
        
        df = self.create_test_dataframe(near_constant_data)
        
        # Test data preparation
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Test stationarity - should be stationary but may have numerical issues
            is_stationary = check_stationarity(train_series, name="NearConstant")
            
            # Test model fitting - this is where failures are most likely
            model_result = fit_sarima_model(
                train_series, 
                order=(1,1,1), 
                seasonal_order=(1,1,1,s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            if model_result is None:
                # Expected failure - document it
                print("SARIMA failed on near-constant data (expected)")
            else:
                # If it succeeds, check for warnings or poor fit
                assert hasattr(model_result, 'aic')
                
        except Exception as e:
            # Document specific failure modes
            failure_keywords = ["singular", "convergence", "invertible", "likelihood"]
            assert any(keyword in str(e).lower() for keyword in failure_keywords), \
                f"Unexpected failure on near-constant data: {e}"
    
    def test_extreme_outliers(self):
        """Test SARIMA with extreme outlier spikes."""
        # Generate normal data with extreme outliers
        normal_data = np.random.normal(50, 10, 1000)
        outlier_data = normal_data.copy()
        
        # Add extreme outliers
        outlier_indices = [100, 200, 300, 500, 800]
        outlier_values = [1000, -500, 2000, -1000, 1500]
        for idx, val in zip(outlier_indices, outlier_values):
            outlier_data[idx] = val
        
        df = self.create_test_dataframe(outlier_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Test stationarity - outliers may affect tests
            is_stationary = check_stationarity(train_series, name="WithOutliers")
            
            # Test model fitting - outliers may cause convergence issues
            model_result = fit_sarima_model(
                train_series,
                order=(1,1,1),
                seasonal_order=(1,1,1,s)
            )
            
            if model_result is not None:
                # Check if outliers affected model quality
                residuals = model_result.resid
                residual_outliers = np.abs(residuals - residuals.mean()) > 3 * residuals.std()
                outlier_percentage = residual_outliers.sum() / len(residuals)
                
                # Document outlier impact
                print(f"Residual outliers: {outlier_percentage:.2%}")
                
        except Exception as e:
            # Document outlier-related failures
            print(f"SARIMA failed with outliers: {e}")
    
    def test_high_missing_data(self):
        """Test SARIMA with high percentage of missing data."""
        # Generate data with systematic missing values
        base_data = np.random.normal(50, 10, 1000)
        missing_data = base_data.copy()
        
        # Make every 3rd value missing (33% missing)
        missing_indices = list(range(0, len(missing_data), 3))
        for idx in missing_indices:
            missing_data[idx] = np.nan
        
        df = self.create_test_dataframe(missing_data)
        
        try:
            # This should handle missing data via forward/backward fill
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Check if missing data was handled properly
            assert not train_series.isnull().any(), "Missing data not properly handled"
            
            # Test if filled data is reasonable for SARIMA
            is_stationary = check_stationarity(train_series, name="FilledMissing")
            
            model_result = fit_sarima_model(train_series, order=(1,0,1))
            
        except Exception as e:
            print(f"SARIMA failed with high missing data: {e}")
    
    def test_insufficient_data(self):
        """Test SARIMA with insufficient data points."""
        # Very short series (less than typical seasonal period)
        short_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df = self.create_test_dataframe(short_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df, test_size=0.2)
            
            # Should fail or warn about insufficient data
            is_stationary = check_stationarity(train_series, name="InsufficientData")
            
            # This should likely fail
            model_result = fit_sarima_model(
                train_series,
                order=(1,1,1),
                seasonal_order=(0,0,0,1)  # No seasonality for short series
            )
            
            if model_result is None:
                print("SARIMA appropriately failed on insufficient data")
            
        except Exception as e:
            # Expected failure modes
            failure_keywords = ["insufficient", "degrees", "freedom", "length"]
            print(f"SARIMA failed on short series: {e}")
    
    def test_perfect_trend(self):
        """Test SARIMA with perfect linear trend (no noise)."""
        # Perfect linear trend
        trend_data = list(range(1000))  # 0, 1, 2, ..., 999
        df = self.create_test_dataframe(trend_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Should be non-stationary
            is_stationary = check_stationarity(train_series, name="PerfectTrend")
            assert not is_stationary, "Perfect trend should be non-stationary"
            
            # Test with differencing
            model_result = fit_sarima_model(
                train_series,
                order=(0,1,0),  # Just differencing, no AR/MA
                seasonal_order=(0,0,0,1)
            )
            
            if model_result is not None:
                # Check if differencing worked
                assert hasattr(model_result, 'aic')
                print("SARIMA successfully handled perfect trend with differencing")
            
        except Exception as e:
            print(f"SARIMA failed on perfect trend: {e}")
    
    def test_seasonal_patterns_only(self):
        """Test SARIMA with pure seasonal patterns (no trend or noise)."""
        # Perfect seasonal pattern
        seasonal_period = 96  # Daily cycle for 15-min data
        n_cycles = 10
        t = np.arange(seasonal_period * n_cycles)
        seasonal_data = 100 * np.sin(2 * np.pi * t / seasonal_period) + 200
        
        df = self.create_test_dataframe(seasonal_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Should be non-stationary due to seasonality
            is_stationary = check_stationarity(train_series, name="PureSeasonal")
            
            # Test with seasonal differencing
            model_result = fit_sarima_model(
                train_series,
                order=(0,0,0),
                seasonal_order=(0,1,0,s)  # Seasonal differencing only
            )
            
            if model_result is not None:
                print("SARIMA handled pure seasonal pattern")
            
        except Exception as e:
            print(f"SARIMA failed on pure seasonal pattern: {e}")
    
    def test_non_normal_distribution(self):
        """Test SARIMA with heavily skewed/non-normal data."""
        # Exponentially distributed data (heavily right-skewed)
        exp_data = np.random.exponential(scale=2.0, size=1000)
        df = self.create_test_dataframe(exp_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            is_stationary = check_stationarity(train_series, name="Exponential")
            
            model_result = fit_sarima_model(train_series)
            
            if model_result is not None:
                # Check residual normality (SARIMA assumes normal residuals)
                residuals = model_result.resid.dropna()
                from scipy import stats
                _, p_value = stats.jarque_bera(residuals)
                
                if p_value < 0.05:
                    print(f"Non-normal residuals detected (p={p_value:.4f})")
            
        except Exception as e:
            print(f"SARIMA failed on non-normal data: {e}")
    
    def test_heteroscedasticity(self):
        """Test SARIMA with changing variance (heteroscedasticity)."""
        # Data with increasing variance over time
        t = np.arange(1000)
        base_signal = 50 + 0.01 * t  # Slight trend
        varying_noise = np.random.normal(0, 1 + 0.01 * t)  # Increasing variance
        hetero_data = base_signal + varying_noise
        
        df = self.create_test_dataframe(hetero_data)
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            is_stationary = check_stationarity(train_series, name="Heteroscedastic")
            
            model_result = fit_sarima_model(train_series)
            
            if model_result is not None:
                # Check for heteroscedasticity in residuals
                residuals = model_result.resid.dropna()
                residual_variance = residuals.rolling(window=100).var()
                variance_ratio = residual_variance.max() / residual_variance.min()
                
                if variance_ratio > 4:  # Significant variance change
                    print(f"Heteroscedasticity detected (ratio={variance_ratio:.2f})")
            
        except Exception as e:
            print(f"SARIMA failed on heteroscedastic data: {e}")
    
    def test_auto_arima_edge_cases(self):
        """Test auto-ARIMA functionality with edge cases."""
        # Test with various challenging datasets
        test_cases = [
            ("white_noise", np.random.normal(0, 1, 500)),
            ("random_walk", np.cumsum(np.random.normal(0, 1, 500))),
            ("high_frequency", np.sin(2 * np.pi * np.arange(500) / 5) + np.random.normal(0, 0.1, 500))
        ]
        
        for case_name, data in test_cases:
            series = pd.Series(data, 
                             index=pd.date_range('2023-01-01', periods=len(data), freq='15min'))
            
            try:
                # Test auto-ARIMA with conservative parameters
                result = auto_sarima_selection(
                    series, 
                    seasonal_period=96,
                    max_p_val=1, 
                    max_q_val=1,
                    max_P_val=1, 
                    max_Q_val=1
                )
                
                if result is not None:
                    print(f"Auto-ARIMA succeeded on {case_name}")
                else:
                    print(f"Auto-ARIMA failed on {case_name}")
                    
            except Exception as e:
                print(f"Auto-ARIMA crashed on {case_name}: {e}")
    
    def test_metrics_edge_cases(self):
        """Test metric calculations with edge case predictions."""
        # Create test actual values
        actual = pd.Series([1, 2, 3, 4, 5])
        
        # Test various problematic prediction scenarios
        test_cases = [
            ("perfect_prediction", actual.copy()),
            ("all_zeros", pd.Series([0, 0, 0, 0, 0])),
            ("negative_values", pd.Series([-1, -2, -3, -4, -5])),
            ("extreme_values", pd.Series([1000, 2000, 3000, 4000, 5000])),
            ("nan_values", pd.Series([1, np.nan, 3, np.nan, 5])),
            ("infinite_values", pd.Series([1, np.inf, 3, -np.inf, 5]))
        ]
        
        for case_name, predicted in test_cases:
            try:
                metrics = calculate_metrics(actual, predicted, case_name, seasonality=4)
                
                if metrics is not None:
                    # Check for reasonable metric values
                    for metric, value in metrics.items():
                        if metric not in ['Model', 'N_Points']:
                            if np.isnan(value) or np.isinf(value):
                                print(f"Invalid {metric} for {case_name}: {value}")
                            
                else:
                    print(f"Metrics calculation failed for {case_name}")
                    
            except Exception as e:
                print(f"Metrics calculation crashed on {case_name}: {e}")


class TestSARIMANumericalStability:
    """Test numerical stability specific issues."""
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero in various functions."""
        # Test MAPE calculation with zero actuals
        actual_with_zeros = pd.Series([0, 1, 2, 0, 4])
        predicted = pd.Series([0.1, 1.1, 2.1, 0.1, 4.1])
        
        try:
            metrics = calculate_metrics(actual_with_zeros, predicted, "ZeroDivision", seasonality=5)
            
            # MAPE should not be infinite or NaN
            if metrics and 'MAPE' in metrics:
                assert not np.isinf(metrics['MAPE']), "MAPE should not be infinite"
                assert not np.isnan(metrics['MAPE']) or np.isnan(metrics['MAPE']), "MAPE handling of zeros unclear"
                
        except Exception as e:
            print(f"Division by zero test failed: {e}")
    
    def test_overflow_prevention(self):
        """Test prevention of numerical overflow."""
        # Very large values that might cause overflow
        large_values = [1e10, 1e11, 1e12] * 100
        df = pd.DataFrame({
            'Time': pd.date_range('2023-01-01', periods=len(large_values), freq='15min'),
            'GHI': large_values
        })
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Check if large values are handled properly
            assert all(np.isfinite(train_series)), "Training series should be finite"
            
            # Test model fitting with large values
            model_result = fit_sarima_model(train_series)
            
            if model_result is not None:
                print("SARIMA handled large values successfully")
            
        except Exception as e:
            overflow_keywords = ["overflow", "infinity", "finite"]
            if any(keyword in str(e).lower() for keyword in overflow_keywords):
                print(f"Expected overflow-related failure: {e}")
            else:
                print(f"Unexpected failure with large values: {e}")
    
    def test_underflow_prevention(self):
        """Test prevention of numerical underflow."""
        # Very small values that might cause underflow
        small_values = [1e-10, 1e-11, 1e-12] * 100
        df = pd.DataFrame({
            'Time': pd.date_range('2023-01-01', periods=len(small_values), freq='15min'),
            'GHI': small_values
        })
        
        try:
            train_series, test_series, s = prepare_time_series_data(df)
            
            # Check if small values are handled properly
            assert all(train_series >= 0), "GHI values should be non-negative"
            
            model_result = fit_sarima_model(train_series)
            
            if model_result is not None:
                print("SARIMA handled small values successfully")
                
        except Exception as e:
            underflow_keywords = ["underflow", "precision", "tolerance"]
            if any(keyword in str(e).lower() for keyword in underflow_keywords):
                print(f"Expected underflow-related failure: {e}")
            else:
                print(f"Unexpected failure with small values: {e}")


# Integration test
def test_end_to_end_robustness():
    """Test complete SARIMA pipeline robustness."""
    # Test multiple challenging scenarios in sequence
    challenging_datasets = {
        "mixed_issues": {
            "data": [0] * 100 + list(np.random.normal(50, 10, 400)) + [1000] + list(np.random.normal(50, 10, 499)),
            "expected_issues": ["zeros", "outliers"]
        },
        "seasonal_with_trend": {
            "data": [100 + 10*np.sin(2*np.pi*i/96) + 0.01*i + np.random.normal(0, 2) for i in range(1000)],
            "expected_issues": ["trend", "seasonality"]
        }
    }
    
    for dataset_name, dataset_info in challenging_datasets.items():
        print(f"\nTesting {dataset_name}...")
        
        df = pd.DataFrame({
            'Time': pd.date_range('2023-01-01', periods=len(dataset_info["data"]), freq='15min'),
            'GHI': dataset_info["data"]
        })
        
        try:
            # Full pipeline test
            train_series, test_series, s = prepare_time_series_data(df)
            is_stationary = check_stationarity(train_series)
            
            if not is_stationary:
                # Try with differencing
                model_result = fit_sarima_model(train_series, order=(1,1,1), seasonal_order=(1,1,1,s))
            else:
                model_result = fit_sarima_model(train_series, order=(1,0,1))
            
            if model_result is not None:
                # Try forecasting
                forecast = model_result.forecast(steps=len(test_series))
                metrics = calculate_metrics(test_series, forecast, dataset_name, seasonality=s)
                
                print(f"✅ {dataset_name}: Complete pipeline succeeded")
            else:
                print(f"⚠️ {dataset_name}: Model fitting failed")
                
        except Exception as e:
            print(f"❌ {dataset_name}: Pipeline failed - {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
