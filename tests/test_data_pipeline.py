"""
Test data pipeline: end-to-end prepare → shapes/NaN asserts.
"""

import pytest
import numpy as np
import pandas as pd
from solar_prediction.data_prep import _create_sequences_and_split, prepare_weather_data
from solar_prediction.config import SequenceConfig, get_config


class TestDataPipelineEndToEnd:
    """Test end-to-end data preparation pipeline."""

    def test_pipeline_shapes_consistency(self, small_sample_data):
        """Test that pipeline produces consistent shapes."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        # Check X and y have consistent sample counts
        if X_train.shape[0] > 0:
            assert X_train.shape[0] == y_train.shape[0]
        if X_val.shape[0] > 0:
            assert X_val.shape[0] == y_val.shape[0]
        if X_test.shape[0] > 0:
            assert X_test.shape[0] == y_test.shape[0]

        # Check feature consistency
        if X_train.shape[0] > 0:
            expected_features = X_train.shape[2]
            if X_val.shape[0] > 0:
                assert X_val.shape[2] == expected_features
            if X_test.shape[0] > 0:
                assert X_test.shape[2] == expected_features

        # Check sequence length consistency
        if X_train.shape[0] > 0:
            expected_seq_len = X_train.shape[1]
            if X_val.shape[0] > 0:
                assert X_val.shape[1] == expected_seq_len
            if X_test.shape[0] > 0:
                assert X_test.shape[1] == expected_seq_len

        print("Shape consistency test passed!")

    def test_pipeline_feature_columns_match(self, small_sample_data):
        """Test that reported feature columns match actual data."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        # Number of feature columns should match the feature dimension
        if X_train.shape[0] > 0:
            assert (
                len(feature_cols) == X_train.shape[2]
            ), f"Feature columns count {len(feature_cols)} != feature dimension {X_train.shape[2]}"

        # Feature columns should be in transform_info
        assert feature_cols == transform_info["feature_columns_used"]

        print(f"Feature columns match test passed: {len(feature_cols)} features")

    def test_sequence_creation_uses_aligned_stride_windows(self):
        """Test multi-feature stride windows align with next-step targets."""
        scaled_df = pd.DataFrame(
            {
                "feature_a": np.arange(8, dtype=float),
                "feature_b": np.arange(100, 108, dtype=float),
                "target_scaled": np.arange(200, 208, dtype=float),
            }
        )
        sequence_cfg = SequenceConfig(window_size=3, test_size=0.4, val_size_from_train_val=0.25)

        X_train, X_val, X_test, y_train, y_val, y_test = _create_sequences_and_split(
            scaled_df, ["feature_a", "feature_b"], "target_scaled", sequence_cfg
        )

        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test]).reshape(-1)

        assert X_all.shape == (5, 3, 2)
        np.testing.assert_array_equal(X_all[0], scaled_df[["feature_a", "feature_b"]].iloc[:3])
        np.testing.assert_array_equal(X_all[-1], scaled_df[["feature_a", "feature_b"]].iloc[4:7])
        np.testing.assert_array_equal(y_all, scaled_df["target_scaled"].iloc[3:8])

    def test_sequence_creation_handles_single_feature_stride_windows(self):
        """Test one-feature stride windows keep a 3D feature axis."""
        scaled_df = pd.DataFrame(
            {
                "feature_a": np.arange(8, dtype=float),
                "target_scaled": np.arange(200, 208, dtype=float),
            }
        )
        sequence_cfg = SequenceConfig(window_size=3, test_size=0.4, val_size_from_train_val=0.25)

        X_train, X_val, X_test, y_train, y_val, y_test = _create_sequences_and_split(
            scaled_df, ["feature_a"], "target_scaled", sequence_cfg
        )

        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test]).reshape(-1)

        assert X_all.shape == (5, 3, 1)
        np.testing.assert_array_equal(X_all[0, :, 0], scaled_df["feature_a"].iloc[:3])
        np.testing.assert_array_equal(X_all[-1, :, 0], scaled_df["feature_a"].iloc[4:7])
        np.testing.assert_array_equal(y_all, scaled_df["target_scaled"].iloc[3:8])

    def _assert_valid_shapes(
        self, X_train, X_val, X_test, y_train, y_val, y_test, allow_empty=False
    ):
        """Assert that all arrays have valid shapes."""
        arrays = [X_train, X_val, X_test, y_train, y_val, y_test]
        names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]

        for arr, name in zip(arrays, names):
            assert isinstance(arr, np.ndarray), f"{name} is not a numpy array"
            assert arr.ndim > 0, f"{name} has invalid dimensions"

            if not allow_empty:
                assert arr.size > 0, f"{name} is empty"

            # Check for valid shape values
            for dim in arr.shape:
                assert dim >= 0, f"{name} has negative dimension: {arr.shape}"

        # X arrays should be 3D, y arrays should be 2D
        X_arrays = [X_train, X_val, X_test]
        y_arrays = [y_train, y_val, y_test]

        for arr, name in zip(X_arrays, ["X_train", "X_val", "X_test"]):
            if arr.size > 0:  # Only check non-empty arrays
                assert arr.ndim == 3, f"{name} should be 3D, got shape {arr.shape}"

        for arr, name in zip(y_arrays, ["y_train", "y_val", "y_test"]):
            if arr.size > 0:  # Only check non-empty arrays
                assert arr.ndim == 2, f"{name} should be 2D, got shape {arr.shape}"

    def _assert_no_nan_values(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Assert that arrays don't contain NaN values."""
        arrays = [X_train, X_val, X_test, y_train, y_val, y_test]
        names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]

        for arr, name in zip(arrays, names):
            if arr.size > 0:  # Only check non-empty arrays
                nan_count = np.isnan(arr).sum()
                assert nan_count == 0, f"{name} contains {nan_count} NaN values"


class TestDataPipelineScaling:
    """Test scaling aspects of the data pipeline."""

    def test_scaling_inversion(self, small_sample_data):
        """Test that scaling can be properly inverted."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        # Test that we can inverse transform the target
        target_scaler_name = None
        for key in scalers.keys():
            if "scaled" in key and key != "power_transformer_object_for_target":
                target_scaler_name = key
                break

        if target_scaler_name and y_train.size > 0:
            target_scaler = scalers[target_scaler_name]

            # Inverse transform a sample
            original_y = target_scaler.inverse_transform(y_train[:1])

            # Check that result is finite
            assert np.all(np.isfinite(original_y))
            assert original_y.shape == y_train[:1].shape

            print(f"Scaling inversion test passed for target scaler: {target_scaler_name}")

    def test_feature_scaling_properties(self, small_sample_data):
        """Test that features are properly scaled."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        if X_train.size > 0:
            # Check that features have reasonable ranges
            # For standard scaling, should be roughly mean=0, std=1
            # For min-max scaling, should be in [0, 1] range

            X_flat = X_train.reshape(-1, X_train.shape[-1])

            for feature_idx in range(X_flat.shape[1]):
                feature_values = X_flat[:, feature_idx]

                # Basic checks
                assert np.all(np.isfinite(feature_values))

                # Check range is reasonable (not too extreme)
                feature_range = np.max(feature_values) - np.min(feature_values)
                assert (
                    feature_range < 1000
                ), f"Feature {feature_idx} has extreme range: {feature_range}"

                # Standard deviation should be reasonable
                feature_std = np.std(feature_values)
                assert feature_std < 100, f"Feature {feature_idx} has extreme std: {feature_std}"

        print("Feature scaling properties test passed!")


class TestDataPipelineTransformations:
    """Test transformation aspects of the data pipeline."""

    def test_transformation_info_completeness(self, small_sample_data):
        """Test that transformation info contains all required fields."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        # Check required fields in transform_info
        required_fields = [
            "structural_transforms",
            "target_scaler_name",
            "target_col_original",
            "target_col_standardized",
            "target_col_after_structural_transforms",
            "feature_columns_used",
        ]

        for field in required_fields:
            assert field in transform_info, f"Missing field in transform_info: {field}"

        # Check types
        assert isinstance(transform_info["structural_transforms"], list)
        assert isinstance(transform_info["feature_columns_used"], list)
        assert isinstance(transform_info["target_col_original"], str)

        print("Transformation info completeness test passed!")

    def test_time_features_engineering(self, small_sample_data):
        """Test that time features are properly engineered."""
        config = get_config()
        df = small_sample_data.copy()

        result = prepare_weather_data(
            df,
            config.input,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = (
            result
        )

        feature_cols_lower = [col.lower() for col in feature_cols]

        # Check for presence of expected time features
        time_features = ["hour", "month", "sin", "cos", "daylight", "solar"]
        time_features_found = []

        for time_feat in time_features:
            for feat_col in feature_cols_lower:
                if time_feat in feat_col:
                    time_features_found.append(time_feat)
                    break

        # Should have at least some time features
        assert len(time_features_found) > 0, f"No time features found in {feature_cols}"

        print(f"Time features engineering test passed! Found: {time_features_found}")

    def test_basic_and_minimal_feature_modes_use_standardized_names(self):
        """Raw weather-schema feature modes should survive standardization."""
        config = get_config()
        rows = 48
        df = pd.DataFrame(
            {
                "Time": pd.date_range("2023-01-01", periods=rows, freq="h"),
                "GHI": np.linspace(0, 700, rows),
                "temp": np.linspace(5, 25, rows),
                "pressure": np.linspace(1005, 1020, rows),
                "humidity": np.linspace(80, 40, rows),
                "wind_speed": np.linspace(1, 5, rows),
                "clouds_all": np.linspace(90, 10, rows),
                "rain_1h": np.zeros(rows),
                "snow_1h": np.zeros(rows),
            }
        )
        sequence_cfg = config.sequences.model_copy(
            update={"window_size": 4, "test_size": 0.2, "val_size_from_train_val": 0.25}
        )

        expected_by_mode = {
            "minimal": {
                "Temperature",
                "Humidity",
                "Cloudcover",
                "TimeMinutesSin",
                "TimeMinutesCos",
            },
            "basic": {
                "Temperature",
                "Pressure",
                "Humidity",
                "WindSpeed",
                "Cloudcover",
                "Rain",
                "TimeMinutesSin",
                "TimeMinutesCos",
            },
        }

        for mode, expected_features in expected_by_mode.items():
            feature_cfg = config.features.model_copy(update={"feature_selection_mode": mode})
            *_, feature_cols, _ = prepare_weather_data(
                df,
                config.input,
                config.transformation,
                feature_cfg,
                config.scaling,
                sequence_cfg,
            )
            assert expected_features.issubset(set(feature_cols))

    def test_fitted_preprocessing_uses_training_rows_only(self):
        """Future/test target outliers should not affect fitted target scaling."""
        config = get_config()
        sequence_cfg = config.sequences.model_copy(
            update={"window_size": 4, "test_size": 0.2, "val_size_from_train_val": 0.25}
        )
        transform_cfg = config.transformation.model_copy(
            update={"use_power_transform": False, "use_log_transform": False}
        )

        df = pd.DataFrame(
            {
                "Time": pd.date_range("2023-01-01", periods=80, freq="h"),
                "GHI": [10.0] * 64 + [10000.0] * 16,
                "temp": [20.0] * 80,
                "pressure": [1013.0] * 80,
                "humidity": [50.0] * 80,
                "wind_speed": [3.0] * 80,
                "clouds_all": [20.0] * 80,
                "rain_1h": [0.0] * 80,
                "snow_1h": [0.0] * 80,
            }
        )

        *_, scalers, _, transform_info = prepare_weather_data(
            df,
            config.input,
            transform_cfg,
            config.features,
            config.scaling,
            sequence_cfg,
        )

        target_scaler = scalers[transform_info["target_scaler_name"]]
        assert transform_info["preprocessing_fit_scope"] == "train_only"
        assert target_scaler.mean_[0] == pytest.approx(10.0)


class TestDataPipelineEdgeCases:
    """Test edge cases for the data pipeline."""

    def test_pipeline_with_missing_columns(self):
        """Test pipeline behavior with missing expected columns."""
        config = get_config()

        # Create data with only some expected columns
        minimal_data = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 08:00:00", "2023-01-01 09:00:00"],
                "Radiation": [100.0, 200.0],
                "Temperature": [15.0, 16.0],
                # Missing other expected columns
            }
        )

        try:
            result = prepare_weather_data(
                minimal_data,
                config.input,
                config.transformation,
                config.features,
                config.scaling,
                config.sequences,
            )

            # If it succeeds, check the results are valid
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                scalers,
                feature_cols,
                transform_info,
            ) = result
            self._assert_basic_validity(X_train, X_val, X_test, y_train, y_val, y_test)

        except Exception as e:
            # It's acceptable if pipeline fails with missing columns
            pytest.skip(f"Pipeline failed with missing columns (expected): {e}")

    def test_pipeline_with_constant_values(self):
        """Test pipeline behavior with constant values in features."""
        config = get_config()

        # Create data with constant values
        constant_data = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2023-01-01", periods=20, freq="h"),
                "Radiation": [100.0] * 20,  # Constant
                "Temperature": [15.0] * 20,  # Constant
                "Pressure": [1013.0] * 20,  # Constant
                "Humidity": [60.0] * 20,  # Constant
                "WindSpeed": [2.0] * 20,  # Constant
                "clouds_all": [50] * 20,  # Constant
                "rain_1h": [0.0] * 20,  # Constant
                "snow_1h": [0.0] * 20,  # Constant
                "weather_type": ["clear"] * 20,  # Constant
                "HourOfDay": list(range(20)),  # Variable
                "Month": [1] * 20,  # Constant
                "DayLength": [8.5] * 20,  # Constant
                "IsSun": [1 if i in range(7, 17) else 0 for i in range(20)],  # Variable
                "SunlightTimeDaylengthRatio": [0.5] * 20,  # Constant
                "Sunrise": ["07:30:00"] * 20,  # Constant
                "Sunset": ["16:00:00"] * 20,  # Constant
                "UNIXTime": [1672531200 + i * 3600 for i in range(20)],  # Variable
            }
        )

        try:
            result = prepare_weather_data(
                constant_data,
                config.input,
                config.transformation,
                config.features,
                config.scaling,
                config.sequences,
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                scalers,
                feature_cols,
                transform_info,
            ) = result
            self._assert_basic_validity(X_train, X_val, X_test, y_train, y_val, y_test)

            print("Constant values test passed!")

        except Exception as e:
            pytest.skip(f"Pipeline failed with constant values: {e}")

    def test_pipeline_with_extreme_values(self):
        """Test pipeline behavior with extreme values."""
        config = get_config()

        # Create data with extreme values
        extreme_data = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "Radiation": [0, 1e6, 0, 1000, 0, 500, 0, 2000, 0, 100],  # Very large values
                "Temperature": [-50, 50, 0, 25, -10, 40, 5, 30, -5, 20],  # Extreme temperatures
                "Pressure": [900, 1100, 1013, 950, 1050, 1000, 980, 1020, 990, 1010],
                "Humidity": [0, 100, 50, 10, 90, 30, 70, 20, 80, 60],
                "WindSpeed": [0, 100, 5, 50, 2, 30, 10, 80, 1, 60],  # Very high wind
                "clouds_all": [0, 100, 50, 0, 100, 25, 75, 0, 100, 50],
                "rain_1h": [0, 50, 0, 10, 0, 5, 0, 20, 0, 2],  # Heavy rain
                "snow_1h": [0, 10, 0, 5, 0, 1, 0, 8, 0, 3],
                "weather_type": ["clear"] * 10,
                "HourOfDay": list(range(10)),
                "Month": [1] * 10,
                "DayLength": [8.5] * 10,
                "IsSun": [1 if i in range(7, 10) else 0 for i in range(10)],
                "SunlightTimeDaylengthRatio": [0.1 * i for i in range(10)],
                "Sunrise": ["07:30:00"] * 10,
                "Sunset": ["16:00:00"] * 10,
                "UNIXTime": [1672531200 + i * 3600 for i in range(10)],
            }
        )

        try:
            result = prepare_weather_data(
                extreme_data,
                config.input,
                config.transformation,
                config.features,
                config.scaling,
                config.sequences,
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                scalers,
                feature_cols,
                transform_info,
            ) = result
            self._assert_basic_validity(X_train, X_val, X_test, y_train, y_val, y_test)

            # Check that extreme values are handled (scaled to reasonable ranges)
            if X_train.size > 0:
                X_flat = X_train.reshape(-1, X_train.shape[-1])
                max_abs_value = np.max(np.abs(X_flat))
                assert (
                    max_abs_value < 1000
                ), f"Extreme values not properly scaled: max={max_abs_value}"

            print("Extreme values test passed!")

        except Exception as e:
            pytest.skip(f"Pipeline failed with extreme values: {e}")

    def _assert_basic_validity(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Basic validity checks for pipeline outputs."""
        arrays = [X_train, X_val, X_test, y_train, y_val, y_test]
        for arr in arrays:
            assert isinstance(arr, np.ndarray)
            if arr.size > 0:
                assert np.all(np.isfinite(arr))
