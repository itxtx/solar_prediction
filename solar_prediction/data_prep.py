import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import math
import logging
from dataclasses import dataclass, field, replace
from typing import List, Dict, Tuple, Optional, Any
from numpy.lib.stride_tricks import sliding_window_view

# Import centralized configuration
from .config import get_config, DataInputConfig, DataTransformationConfig as TransformationConfig, FeatureEngineeringConfig, ScalingConfig, SequenceConfig
from .benchmark import benchmark, benchmark_context

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration classes are now imported from centralized config module
# Legacy dataclasses are deprecated - use centralized config instead

# --- Standardized Column Names (Internal) ---
# These are the names the rest of the functions will expect after initial mapping.
STD_RADIATION_COL = 'Radiation'
STD_TEMP_COL = 'Temperature'
STD_PRESSURE_COL = 'Pressure'
STD_HUMIDITY_COL = 'Humidity'
STD_WINDSPEED_COL = 'WindSpeed'
STD_CLOUDCOVER_COL = 'Cloudcover' # Example, map 'clouds_all' to this
STD_RAIN_COL = 'Rain' # Example, map 'rain_1h'
STD_SNOW_COL = 'Snow' # Example, map 'snow_1h'
STD_WEATHER_TYPE_COL = 'WeatherType'
# Time related standardized names
STD_TIME_COL = 'Timestamp' # Standardized name for the main time column
STD_HOUR_OF_DAY = 'HourOfDay'
STD_MONTH = 'Month'
STD_DAYLIGHT_MINUTES = 'DaylightMinutes'
STD_IS_DAYLIGHT = 'IsDaylight'
STD_DAYLIGHT_POSITION = 'DaylightPosition'
STD_SOLAR_ELEVATION = 'SolarElevation'
STD_TIME_MINUTES_SIN = 'TimeMinutesSin'
STD_TIME_MINUTES_COS = 'TimeMinutesCos'
STD_TIME_SINCE_SUNRISE = 'TimeSinceSunrise'
STD_TIME_UNTIL_SUNSET = 'TimeUntilSunset'


# --- Main Data Preparation Function ---
@benchmark(stage_name="prepare_weather_data_pipeline", track_memory=True)
def prepare_weather_data(
    df_input: pd.DataFrame,
    input_cfg: DataInputConfig,
    transform_cfg: TransformationConfig,
    feature_cfg: FeatureEngineeringConfig,
    scaling_cfg: ScalingConfig,
    sequence_cfg: SequenceConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Revised pipeline to prepare weather time series data for recurrent model training.
    Uses configuration objects and has improved structure and error handling.
    """
    if df_input.empty:
        raise ValueError("Input DataFrame is empty.")
    if input_cfg.target_col_original_name not in df_input.columns:
        raise ValueError(f"Original target column '{input_cfg.target_col_original_name}' not found in input DataFrame.")

    logging.info("Starting weather data preparation pipeline v2.")
    
    # Set chained assignment to None for in-place operations optimization
    with pd.option_context('mode.chained_assignment', None):
        # 1. Initial DataFrame Setup (with in-place operations optimization)
        df, standardized_target_col_name, column_rename_map = _initial_df_setup(df_input, input_cfg)
    
        # 2. Engineer Time-Based Features
        df = _engineer_time_features(df, feature_cfg, input_cfg) # Pass input_cfg for raw time col names

        # 3. Apply Target Variable Transformations (optimized - no copy needed with chained assignment None)
        # target_col_after_transforms is the name of the column that holds the target values
        # after all structural transformations (log, power, piecewise) but BEFORE scaling.
        df, target_col_after_transforms, applied_target_transforms_info = _apply_target_transformations(
            df, # No copy needed with chained assignment guard
            standardized_target_col_name, 
            transform_cfg
        )

    # 4. Engineer Other Features (e.g., low target indicator)
    df = _engineer_other_domain_features(df, target_col_after_transforms, feature_cfg, standardized_target_col_name)

    # 5. Select Final Set of Feature Columns (using standardized names)
    feature_cols_final = _select_final_features(df, feature_cfg, target_col_after_transforms)
    
    # 6. Scale Features and the (potentially transformed) Target
    # scaled_df contains all selected features and the target, scaled.
    # target_col_scaled is the name of the target column within scaled_df.
    scaled_df, scalers, target_col_scaled = _scale_data(
        df, feature_cols_final, target_col_after_transforms, scaling_cfg, applied_target_transforms_info
    )

    # 7. Create Sequences and Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = _create_sequences_and_split(
        scaled_df, feature_cols_final, target_col_scaled, sequence_cfg
    )
    
    # 8. Consolidate Transformation Information for saving/loading/inverting
    # This should include details about structural transforms and the final scaler for the target.
    full_transform_details = {
        'structural_transforms': applied_target_transforms_info,
        'target_scaler_name': target_col_scaled, # Name of the target column in the scalers dict
        'target_col_original': input_cfg.target_col_original_name, # The very original name
        'target_col_standardized': standardized_target_col_name, # Name after initial standardization
        'target_col_after_structural_transforms': target_col_after_transforms, # Name before scaling
        'feature_columns_used': feature_cols_final
    }
    
    logging.info("Weather data preparation pipeline v2 finished successfully.")
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols_final, full_transform_details


# --- Helper Functions for `prepare_weather_data_v2` ---

@benchmark(stage_name="initial_df_setup")
def _initial_df_setup(df: pd.DataFrame, cfg: DataInputConfig) -> Tuple[pd.DataFrame, str, Dict[str,str]]:
    """Standardizes key column names, and sorts by time (optimized with in-place operations)."""
    logging.debug("Performing initial DataFrame setup.")
    
    # Work with a copy to avoid modifying the original
    df = df.copy()
    
    # Define the mapping from common input names to standardized internal names
    # This map should be comprehensive for columns used in feature engineering or as target
    rename_map = {
        cfg.common_ghi_col: STD_RADIATION_COL,
        cfg.common_temp_col: STD_TEMP_COL,
        cfg.common_pressure_col: STD_PRESSURE_COL,
        cfg.common_humidity_col: STD_HUMIDITY_COL,
        cfg.common_wind_speed_col: STD_WINDSPEED_COL,
        'clouds_all': STD_CLOUDCOVER_COL, # From new dataset
        'rain_1h': STD_RAIN_COL,         # From new dataset
        'snow_1h': STD_SNOW_COL,         # From new dataset
        'weather_type': STD_WEATHER_TYPE_COL, # From new dataset
        cfg.time_col: STD_TIME_COL,
        # Raw time features that might be directly used or for deriving others
        cfg.hour_col_raw: cfg.hour_col_raw, # Keep original if used directly
        cfg.month_col_raw: cfg.month_col_raw,
        cfg.daylength_col_raw: cfg.daylength_col_raw,
        cfg.is_sun_col_raw: cfg.is_sun_col_raw,
        cfg.sunlight_time_daylength_ratio_raw: cfg.sunlight_time_daylength_ratio_raw,
        cfg.sunrise_col: cfg.sunrise_col, # Keep original for parsing
        cfg.sunset_col: cfg.sunset_col,   # Keep original for parsing
    }
    # Add the original target column to the rename map if it's different from standardized
    standardized_target_name = STD_RADIATION_COL if cfg.target_col_original_name == cfg.common_ghi_col else \
                              (STD_TEMP_COL if cfg.target_col_original_name == cfg.common_temp_col else cfg.target_col_original_name)
    
    if cfg.target_col_original_name not in rename_map or rename_map[cfg.target_col_original_name] != standardized_target_name :
         rename_map[cfg.target_col_original_name] = standardized_target_name
    
    # Apply renaming for columns present in the DataFrame
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=actual_rename_map, inplace=True)
    logging.info(f"Applied column renames: {actual_rename_map}")
    
    final_target_col_name = actual_rename_map.get(cfg.target_col_original_name, cfg.target_col_original_name)
    if final_target_col_name not in df.columns: # Should not happen if logic is correct
        raise ValueError(f"Target column '{final_target_col_name}' (from '{cfg.target_col_original_name}') not found after renaming.")

    # Time sorting with optimized datetime parsing
    if STD_TIME_COL in df.columns:
        try:
            # Optimized datetime parsing with caching
            df[STD_TIME_COL] = pd.to_datetime(df[STD_TIME_COL], cache=True)
            df.sort_values(STD_TIME_COL, inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Sorted DataFrame by '{STD_TIME_COL}'.")
        except Exception as e:
            logging.warning(f"Could not parse or sort by '{STD_TIME_COL}': {e}. Trying UNIXTime.")
            if cfg.unix_time_col in df.columns:
                df.sort_values(cfg.unix_time_col, inplace=True) # Assuming UNIXTime is sortable
                df.reset_index(drop=True, inplace=True)
                logging.info(f"Sorted DataFrame by '{cfg.unix_time_col}'.")
            else:
                logging.warning("No primary or fallback time column found for sorting.")
    elif cfg.unix_time_col in df.columns:
        df.sort_values(cfg.unix_time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Sorted DataFrame by '{cfg.unix_time_col}'.")
    else:
        logging.warning(f"Neither '{STD_TIME_COL}' nor '{cfg.unix_time_col}' found for sorting. Data order is preserved as is.")
        
    return df, final_target_col_name, actual_rename_map

def _parse_time_to_minutes_vectorized(time_series: pd.Series) -> np.ndarray:
    """Vectorized time parsing to minutes from midnight."""
    result = np.full(len(time_series), np.nan)
    
    # Handle NaN values
    valid_mask = time_series.notna()
    if not valid_mask.any():
        return result
    
    valid_times = time_series[valid_mask]
    
    # Try to parse as datetime first (most efficient)
    try:
        # Try common datetime formats first to avoid the warning
        parsed_times = pd.to_datetime(valid_times, format='mixed', cache=True, errors='coerce')
        datetime_mask = parsed_times.notna()
        if datetime_mask.any():
            dt_values = parsed_times[datetime_mask]
            minutes = dt_values.dt.hour * 60 + dt_values.dt.minute + dt_values.dt.second / 60.0
            result[valid_mask] = minutes.reindex(valid_times.index, fill_value=np.nan)
            return result
    except:
        pass
    
    # Fallback to individual parsing for remaining values
    for idx, time_val in valid_times.items():
        if isinstance(time_val, (pd.Timestamp, pd.Timedelta)):
            result[idx] = time_val.hour * 60 + time_val.minute + time_val.second / 60.0
        elif isinstance(time_val, str) and ':' in time_val:
            try:
                parts = time_val.split(':')
                h = int(parts[0])
                m = int(parts[1])
                s = float(parts[2]) if len(parts) > 2 else 0.0
                if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                    result[idx] = h * 60 + m + s / 60.0
            except ValueError:
                continue
    
    return result

def _parse_time_to_minutes(time_val: Any) -> Optional[float]:
    """Legacy single-value time parsing (kept for compatibility)."""
    if pd.isna(time_val): return np.nan
    if isinstance(time_val, (pd.Timestamp, pd.Timedelta)):
        return time_val.hour * 60 + time_val.minute + time_val.second / 60.0
    if isinstance(time_val, str):
        try:
            if ':' in time_val:
                parts = time_val.split(':')
                h = int(parts[0])
                m = int(parts[1])
                s = float(parts[2]) if len(parts) > 2 else 0.0
                if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                    return np.nan
                return h * 60 + m + s / 60.0
        except ValueError:
            return np.nan
    return np.nan


@benchmark(stage_name="engineer_time_features")
def _engineer_time_features(df: pd.DataFrame, feature_cfg: FeatureEngineeringConfig, input_cfg: DataInputConfig) -> pd.DataFrame:
    """Engineers time-based features like cyclical hour/month, daylight, solar elevation (vectorized)."""
    logging.debug("Engineering time features.")
    if STD_TIME_COL not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[STD_TIME_COL]):
        logging.warning(f"'{STD_TIME_COL}' not found or not datetime. Skipping detailed time feature engineering.")
        return df

    # Hour and Month (Cyclical)
    df[STD_HOUR_OF_DAY] = df[STD_TIME_COL].dt.hour
    df[STD_MONTH] = df[STD_TIME_COL].dt.month
    
    # Use centralized configuration for time constants
    config = get_config()
    minutes_in_day = config.features.minutes_in_day
    
    # Vectorized time feature engineering using NumPy operations
    time_values = df[STD_TIME_COL].dt
    current_time_minutes = time_values.hour * 60 + time_values.minute + time_values.second / 60.0
    
    # Pre-compute the scaling factor for efficiency
    time_scale_factor = 2 * np.pi / minutes_in_day
    scaled_minutes = current_time_minutes * time_scale_factor
    
    df[STD_TIME_MINUTES_SIN] = np.sin(scaled_minutes)
    df[STD_TIME_MINUTES_COS] = np.cos(scaled_minutes)

    # Sunrise/Sunset and Daylight Features (vectorized)
    if input_cfg.sunrise_col in df.columns and input_cfg.sunset_col in df.columns:
        # Use vectorized time parsing for better performance
        sunrise_minutes = pd.Series(_parse_time_to_minutes_vectorized(df[input_cfg.sunrise_col]), index=df.index)
        sunset_minutes = pd.Series(_parse_time_to_minutes_vectorized(df[input_cfg.sunset_col]), index=df.index)

        # Handle cases where parsing might fail or times are missing
        if sunrise_minutes.notna().all() and sunset_minutes.notna().all():
            df[STD_DAYLIGHT_MINUTES] = sunset_minutes - sunrise_minutes
            # Handle overnight (sunset < sunrise)
            df.loc[df[STD_DAYLIGHT_MINUTES] < 0, STD_DAYLIGHT_MINUTES] += minutes_in_day 
            
            df[STD_IS_DAYLIGHT] = ((current_time_minutes >= sunrise_minutes) & (current_time_minutes <= sunset_minutes)).astype(float)
            
            # DaylightPosition: 0 at sunrise, 1 at sunset, normalized (vectorized)
            # Avoid division by zero or by very small daylight duration
            daylight_duration_safe = df[STD_DAYLIGHT_MINUTES].replace(0, np.nan)
            time_since_sunrise = current_time_minutes - sunrise_minutes
            
            # Vectorized adjustment for negative time_since_sunrise
            negative_mask = time_since_sunrise < 0
            time_since_sunrise = np.where(negative_mask, 
                                        time_since_sunrise + minutes_in_day, 
                                        time_since_sunrise)
            
            # Vectorized daylight position calculation
            df[STD_DAYLIGHT_POSITION] = np.clip(time_since_sunrise / daylight_duration_safe, 0, 1)
            df[STD_DAYLIGHT_POSITION].fillna(0, inplace=True)

            df[STD_TIME_SINCE_SUNRISE] = time_since_sunrise
            time_until_sunset = sunset_minutes - current_time_minutes
            # Vectorized adjustment for negative time_until_sunset
            df[STD_TIME_UNTIL_SUNSET] = np.where(time_until_sunset < 0, 
                                               time_until_sunset + minutes_in_day, 
                                               time_until_sunset)


        else:
            logging.warning(f"Could not parse all sunrise/sunset times from '{input_cfg.sunrise_col}'/'{input_cfg.sunset_col}'. Daylight features might be incomplete.")
    # Fallback to raw columns if they exist and parsed ones are missing
    elif input_cfg.daylength_col_raw in df.columns and STD_DAYLIGHT_MINUTES not in df.columns:
        df[STD_DAYLIGHT_MINUTES] = pd.to_numeric(df[input_cfg.daylength_col_raw], errors='coerce') * 60 # Assuming dayLength is in hours
    
    if input_cfg.is_sun_col_raw in df.columns and STD_IS_DAYLIGHT not in df.columns:
        df[STD_IS_DAYLIGHT] = df[input_cfg.is_sun_col_raw].astype(float)
        
    if input_cfg.sunlight_time_daylength_ratio_raw in df.columns and STD_DAYLIGHT_POSITION not in df.columns:
         df[STD_DAYLIGHT_POSITION] = pd.to_numeric(df[input_cfg.sunlight_time_daylength_ratio_raw], errors='coerce').clip(0,1)


    # Solar Elevation Proxy (vectorized)
    if feature_cfg.use_solar_elevation_proxy and STD_DAYLIGHT_POSITION in df.columns:
        # Vectorized solar elevation proxy calculation
        daylight_pos = df[STD_DAYLIGHT_POSITION].values
        valid_mask = (pd.notna(daylight_pos)) & (daylight_pos >= 0) & (daylight_pos <= 1)
        
        solar_elevation = np.zeros_like(daylight_pos)
        solar_elevation[valid_mask] = np.sin(daylight_pos[valid_mask] * np.pi)
        
        df[STD_SOLAR_ELEVATION] = solar_elevation
        logging.info(f"Engineered '{STD_SOLAR_ELEVATION}' feature.")
    elif feature_cfg.use_solar_elevation_proxy:
        logging.warning(f"Cannot create '{STD_SOLAR_ELEVATION}' as '{STD_DAYLIGHT_POSITION}' is missing.")
        
    return df

@benchmark(stage_name="apply_target_transformations")
def _apply_target_transformations(
    df: pd.DataFrame, 
    target_col: str, 
    cfg: TransformationConfig
) -> Tuple[pd.DataFrame, str, List[Dict[str, Any]]]:
    """Applies configured transformations to the target column (optimized)."""
    logging.debug(f"Applying target transformations to '{target_col}'. Config: {cfg}")
    
    current_target_col = target_col
    applied_transforms_log: List[Dict[str, Any]] = []

    # 0. Initial Min Threshold
    if cfg.min_target_threshold_initial is not None:
        logging.info(f"Applying initial min threshold of {cfg.min_target_threshold_initial} to '{current_target_col}'.")
        df[current_target_col] = df[current_target_col].clip(lower=cfg.min_target_threshold_initial)
        # No specific log for this simple clip in applied_transforms_log, assumed part of pre-processing.

    # Check if target is radiation-like for specific GHI transforms
    is_radiation_target = (current_target_col == STD_RADIATION_COL)

    # 1. Piecewise Transform (typically for GHI/Radiation)
    if is_radiation_target and cfg.use_piecewise_transform_target:
        logging.info(f"Applying piecewise radiation transform to '{current_target_col}'.")
        new_col_name = f"{current_target_col}_piecewise"
        
        radiation_values = df[current_target_col].values.astype(float)
        transformed = np.zeros_like(radiation_values)
        
        night_mask = radiation_values < cfg.piecewise_night_threshold
        transformed[night_mask] = np.log1p(radiation_values[night_mask]) 
        
        moderate_mask = (radiation_values >= cfg.piecewise_night_threshold) & \
                        (radiation_values < cfg.piecewise_moderate_threshold)
        # Value at the end of night_mask (just before piecewise_night_threshold)
        val_at_night_thresh_end = np.log1p(cfg.piecewise_night_threshold - 1e-6) # Approx
        transformed[moderate_mask] = val_at_night_thresh_end + \
                                     cfg.piecewise_moderate_slope * (radiation_values[moderate_mask] - cfg.piecewise_night_threshold)
        
        high_mask = radiation_values >= cfg.piecewise_moderate_threshold
        # Value at the end of moderate_mask
        val_at_moderate_thresh_end = val_at_night_thresh_end + \
                                     cfg.piecewise_moderate_slope * (cfg.piecewise_moderate_threshold - cfg.piecewise_night_threshold -1e-6)
        transformed[high_mask] = val_at_moderate_thresh_end + \
                                 cfg.piecewise_high_slope * (radiation_values[high_mask] - cfg.piecewise_moderate_threshold)

        df[new_col_name] = transformed
        applied_transforms_log.append({'type': 'piecewise', 'original_col': current_target_col, 'new_col': new_col_name, 'applied': True,
                                       'params': {'night_thresh': cfg.piecewise_night_threshold, 
                                                  'moderate_thresh': cfg.piecewise_moderate_threshold}})
        current_target_col = new_col_name

    # 2. Power Transform (Yeo-Johnson) OR Log Transform (mutually exclusive)
    # Power transform takes precedence if both are True
    if cfg.use_power_transform:
        logging.info(f"Applying Yeo-Johnson Power Transform to '{current_target_col}'.")
        new_col_name = f"{current_target_col}_yj"
        
        # Optional clipping before power transform
        clip_bounds_orig = None
        if cfg.clip_original_target_before_power_transform and is_radiation_target: # Typically for radiation
            lower_b = np.percentile(df[current_target_col].dropna(), cfg.original_target_clip_lower_percentile)
            upper_b = np.percentile(df[current_target_col].dropna(), cfg.original_target_clip_upper_percentile)
            df[current_target_col] = df[current_target_col].clip(lower_b, upper_b)
            clip_bounds_orig = (float(lower_b), float(upper_b))
            logging.info(f"Clipped '{current_target_col}' to [{lower_b:.2f}, {upper_b:.2f}] before Yeo-Johnson.")

        if is_radiation_target and cfg.min_radiation_floor_before_power_transform > 0:
            df[current_target_col] = np.maximum(df[current_target_col].astype(float), cfg.min_radiation_floor_before_power_transform)
            logging.info(f"Applied floor of {cfg.min_radiation_floor_before_power_transform} to '{current_target_col}' before Yeo-Johnson.")

        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False) # Scaler will handle standardization later
        values_to_transform = df[current_target_col].values.reshape(-1, 1)
        
        # Handle NaNs before fitting PowerTransformer
        nan_mask = np.isnan(values_to_transform.squeeze())
        if np.all(nan_mask): 
            raise ValueError(f"All values in '{current_target_col}' are NaN before PowerTransform. Cannot proceed.")
        
        transformed_values = np.full_like(values_to_transform, np.nan)
        if not np.all(nan_mask): # Only fit if there are non-NaN values
             transformed_values[~nan_mask] = power_transformer.fit_transform(values_to_transform[~nan_mask])
        
        df[new_col_name] = transformed_values.flatten()
        
        applied_transforms_log.append({
            'type': 'yeo-johnson', 'original_col': current_target_col, 'new_col': new_col_name, 'applied': True,
            'lambda': power_transformer.lambdas_[0] if power_transformer.lambdas_ is not None else None,
            'power_transformer_object': power_transformer, # Store the fitted object
            'original_clip_bounds_before_yj': clip_bounds_orig
        })
        current_target_col = new_col_name
    
    elif cfg.use_log_transform: # Only if power transform was not applied
        logging.info(f"Applying Log Transform to '{current_target_col}'.")
        new_col_name = f"{current_target_col}_log"
        
        values_for_log = df[current_target_col].copy()
        if is_radiation_target: # Specific handling for radiation
            values_for_log = np.maximum(values_for_log, cfg.min_radiation_for_log)
        
        # log(x + offset)
        df[new_col_name] = np.log(values_for_log + cfg.log_transform_offset)
        
        clip_bounds_log = None
        if cfg.clip_log_transformed_target:
            lower_b = np.percentile(df[new_col_name].dropna(), cfg.log_clip_lower_percentile)
            upper_b = np.percentile(df[new_col_name].dropna(), cfg.log_clip_upper_percentile)
            df[new_col_name] = df[new_col_name].clip(lower_b, upper_b)
            clip_bounds_log = (float(lower_b), float(upper_b))
            logging.info(f"Clipped log-transformed target '{new_col_name}' to [{lower_b:.2f}, {upper_b:.2f}].")

        applied_transforms_log.append({
            'type': 'log', 'original_col': current_target_col, 'new_col': new_col_name, 'applied': True,
            'offset': cfg.log_transform_offset, 
            'min_val_before_log': cfg.min_radiation_for_log if is_radiation_target else None,
            'clip_bounds_after_log': clip_bounds_log
        })
        current_target_col = new_col_name
        
    return df, current_target_col, applied_transforms_log


@benchmark(stage_name="engineer_other_domain_features")
def _engineer_other_domain_features(df: pd.DataFrame, target_col_after_transforms: str, 
                                    feature_cfg: FeatureEngineeringConfig, 
                                    standardized_target_col_name: str) -> pd.DataFrame:
    """Engineers other domain-specific features, e.g., low target indicator."""
    logging.debug("Engineering other domain features.")
    
    # Low Target Indicator (based on the structurally transformed target, or original standardized if no structural transforms)
    # This helps the model identify periods of very low values, which might behave differently.
    # The reference for "low" should ideally be the target column *before* any scaling,
    # but after structural transforms if they significantly change the distribution (e.g. log).
    # If no structural transforms, use standardized_target_col_name.
    
    col_for_low_indicator_ref = target_col_after_transforms # This is after structural, before scaling.
                                                           # If no structural, it's same as standardized_target_col_name.

    if feature_cfg.create_low_target_indicator and col_for_low_indicator_ref in df.columns:
        try:
            # Ensure the reference column is numeric and has variance
            if pd.api.types.is_numeric_dtype(df[col_for_low_indicator_ref]) and df[col_for_low_indicator_ref].nunique() > 1:
                low_threshold = df[col_for_low_indicator_ref].quantile(feature_cfg.low_target_indicator_quantile)
                indicator_col_name = f"{standardized_target_col_name}_is_low" # Name based on original standardized target
                df[indicator_col_name] = (df[col_for_low_indicator_ref] < low_threshold).astype(float)
                logging.info(f"Engineered '{indicator_col_name}' feature with threshold {low_threshold:.4f} (based on '{col_for_low_indicator_ref}').")
            else:
                logging.warning(f"Cannot create low target indicator for '{col_for_low_indicator_ref}': not numeric or no variance.")
        except Exception as e:
            logging.warning(f"Failed to create low target indicator for '{col_for_low_indicator_ref}': {e}")
            
    return df


def _get_base_feature_set(df_columns: pd.Index, feature_cfg: FeatureEngineeringConfig) -> List[str]:
    """Determines the base set of features based on selection mode and availability."""
    if feature_cfg.feature_selection_mode == 'minimal':
        selected_set = feature_cfg.minimal_features
    elif feature_cfg.feature_selection_mode == 'basic':
        selected_set = feature_cfg.basic_features
    elif feature_cfg.feature_selection_mode == 'all':
        # For 'all', we can be more dynamic. Start with a broad list of known standardized cols.
        # This list should contain all potentially useful standardized column names.
        potential_all_features = [
            STD_RADIATION_COL, STD_TEMP_COL, STD_PRESSURE_COL, STD_HUMIDITY_COL, 
            STD_WINDSPEED_COL, STD_CLOUDCOVER_COL, STD_RAIN_COL, STD_SNOW_COL, STD_WEATHER_TYPE_COL,
            # Add all engineered time features that are always generated
            STD_HOUR_OF_DAY, STD_MONTH, STD_TIME_MINUTES_SIN, STD_TIME_MINUTES_COS,
            STD_DAYLIGHT_MINUTES, STD_IS_DAYLIGHT, STD_DAYLIGHT_POSITION, STD_SOLAR_ELEVATION,
            STD_TIME_SINCE_SUNRISE, STD_TIME_UNTIL_SUNSET
        ]
        selected_set = [col for col in potential_all_features if col in df_columns]
    else:
        logging.warning(f"Unknown feature_selection_mode: '{feature_cfg.feature_selection_mode}'. Defaulting to 'all'.")
        # Recursive call with 'all' or define default 'all' set here. For safety, use a defined 'all'.
        all_mode_cfg = replace(feature_cfg, feature_selection_mode='all')
        return _get_base_feature_set(df_columns, all_mode_cfg)

    # Filter the selected set by actual columns present in the DataFrame
    available_features = [col for col in selected_set if col in df_columns]
    
    # Add low target indicator if created and not already included by chance in 'all'
    # The indicator name is based on the original standardized target name.
    # We need to know that name here. This suggests a slight refactor or passing it.
    # For now, let's assume we can construct it if needed.
    # This part is tricky as standardized_target_col_name is not directly available here.
    # This indicates that feature selection might need the standardized target name.
    # Temporarily, we'll rely on it being added if it exists in df_columns.
    
    return available_features


@benchmark(stage_name="select_final_features")
def _select_final_features(df: pd.DataFrame, feature_cfg: FeatureEngineeringConfig, 
                           target_col_after_transforms: str) -> List[str]:
    """Selects the final list of feature columns to be used for modeling."""
    logging.debug("Selecting final feature columns.")
    
    base_features = _get_base_feature_set(df.columns, feature_cfg)
    
    # Add low target indicator if it was created (its name depends on original standardized target)
    # This logic is a bit indirect here. It's better if _engineer_other_domain_features returns the name.
    # Assuming it follows a pattern like f"{STD_SOME_TARGET}_is_low"
    possible_low_indicator_cols = [col for col in df.columns if col.endswith("_is_low")]
    for lic in possible_low_indicator_cols:
        if lic not in base_features:
            base_features.append(lic)
            logging.info(f"Including low target indicator '{lic}' in features.")

    # Ensure the actual target column (after transforms, before scaling) is NOT in features
    final_feature_list = [col for col in base_features if col != target_col_after_transforms]
    
    # Remove duplicates that might have crept in
    final_feature_list = sorted(list(set(final_feature_list)))

    if not final_feature_list:
        raise ValueError("No feature columns were selected or are available after processing.")
    
    logging.info(f"Final selected features for scaling ({len(final_feature_list)}): {final_feature_list}")
    return final_feature_list


@benchmark(stage_name="scale_data")
def _scale_data(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col_to_scale: str, # This is the target after structural transforms
    scaling_cfg: ScalingConfig,
    applied_target_transforms_info: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Scales features and the (potentially transformed) target column."""
    logging.debug(f"Scaling data. Features: {feature_cols}, Target to scale: {target_col_to_scale}")
    
    scalers: Dict[str, Any] = {}
    scaled_df_data = {} # To build the new DataFrame

    # Determine feature scaler type
    FeatureScalerClass = StandardScaler if scaling_cfg.standardize_features else MinMaxScaler
    logging.info(f"Using {FeatureScalerClass.__name__} for feature scaling.")

    # Scale features
    for col in feature_cols:
        if col not in df.columns:
            logging.warning(f"Feature column '{col}' not found in DataFrame for scaling. Skipping.")
            continue
        if df[col].isnull().all():
            logging.warning(f"All values in feature column '{col}' are NaN. Scaled output will be NaN.")
            scaled_df_data[col] = np.nan
            continue
        
        feature_values = df[col].ffill().bfill().values.reshape(-1, 1) # Fill NaNs before scaling
        scaler = FeatureScalerClass()
        scaled_df_data[col] = scaler.fit_transform(feature_values).flatten()
        scalers[col] = scaler

    # Scale target
    if target_col_to_scale not in df.columns:
        raise ValueError(f"Target column for scaling '{target_col_to_scale}' not found in DataFrame.")
    if df[target_col_to_scale].isnull().all():
        raise ValueError(f"All values in target column '{target_col_to_scale}' are NaN before scaling.")

    # Determine target scaler type
    # If Yeo-Johnson was applied, it's common to follow with StandardScaler.
    # The PowerTransformer itself has a 'standardize=True' option, but if False, we scale after.
    # Let's check if the last structural transform was Yeo-Johnson and if its object implies standardization.
    
    TargetScalerClass = StandardScaler # Default or if Yeo-Johnson was applied
    yj_was_applied = any(t['type'] == 'yeo-johnson' for t in applied_target_transforms_info if t['applied'])
    
    if yj_was_applied:
        logging.info(f"Yeo-Johnson was applied to target. Using StandardScaler for '{target_col_to_scale}'.")
    else: # No Yeo-Johnson, use general scaling config
        TargetScalerClass = StandardScaler if scaling_cfg.standardize_features else MinMaxScaler
        logging.info(f"Using {TargetScalerClass.__name__} for target '{target_col_to_scale}'.")

    target_values = df[target_col_to_scale].ffill().bfill().values.reshape(-1, 1)
    target_scaler_instance = TargetScalerClass()
    
    scaled_target_col_name = f"{target_col_to_scale}_scaled"
    scaled_df_data[scaled_target_col_name] = target_scaler_instance.fit_transform(target_values).flatten()
    scalers[scaled_target_col_name] = target_scaler_instance # Store under its scaled name
    
    # If Yeo-Johnson was used, its transformer object is already in applied_target_transforms_info.
    # We need to ensure it's passed to the final `scalers` dict for inverse transform.
    for transform_detail in applied_target_transforms_info:
        if transform_detail.get('type') == 'yeo-johnson' and 'power_transformer_object' in transform_detail:
            scalers['power_transformer_object_for_target'] = transform_detail['power_transformer_object']
            break
            
    logging.info(f"Target '{target_col_to_scale}' scaled to '{scaled_target_col_name}'.")
    if isinstance(target_scaler_instance, StandardScaler):
        logging.info(f"Target scaler ({scaled_target_col_name}): mean={target_scaler_instance.mean_[0]:.4f}, std={target_scaler_instance.scale_[0]:.4f}")
    elif isinstance(target_scaler_instance, MinMaxScaler):
         logging.info(f"Target scaler ({scaled_target_col_name}): min={target_scaler_instance.min_[0]:.4f}, scale={target_scaler_instance.scale_[0]:.4f} (data_min={target_scaler_instance.data_min_[0]:.4f}, data_max={target_scaler_instance.data_max_[0]:.4f})")


    final_scaled_df = pd.DataFrame(scaled_df_data, index=df.index)
    return final_scaled_df, scalers, scaled_target_col_name


@benchmark(stage_name="create_sequences_and_split")
def _create_sequences_and_split(
    scaled_df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col_scaled: str, 
    sequence_cfg: SequenceConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates sequences from scaled data and splits into train, validation, and test sets (optimized)."""
    logging.debug("Creating sequences and splitting data.")
    
    if not (feature_cols and target_col_scaled in scaled_df.columns):
        raise ValueError("Feature columns list is empty or scaled target column not in DataFrame for sequencing.")

    # Ensure no NaNs in the data used for sequences
    # ffill/bfill should have handled most, but drop any remaining full NaN rows if windowing is affected.
    data_for_sequences = scaled_df[feature_cols + [target_col_scaled]].copy()
    data_for_sequences.dropna(how='all', inplace=True) # Drop rows where ALL selected cols are NaN
    data_for_sequences.ffill(inplace=True) # Apply ffill inplace
    data_for_sequences.bfill(inplace=True) # Then apply bfill inplace

    if len(data_for_sequences) <= sequence_cfg.window_size:
        raise ValueError(f"Data length ({len(data_for_sequences)}) after NaN handling is <= window_size ({sequence_cfg.window_size}). Cannot create sequences.")

    # Optimized sequence generation using pre-allocated arrays and stride tricks
    num_features = len(feature_cols)
    sequence_length = len(data_for_sequences) - sequence_cfg.window_size
    
    if sequence_length <= 0:
        # Handle edge case where not enough data for sequences
        empty_X_shape = (0, sequence_cfg.window_size, num_features)
        empty_y_shape = (0, 1)
        return (np.empty(empty_X_shape), np.empty(empty_X_shape), np.empty(empty_X_shape),
                np.empty(empty_y_shape), np.empty(empty_y_shape), np.empty(empty_y_shape))
    
    # Pre-allocate arrays for better memory efficiency
    X_all = np.empty((sequence_length, sequence_cfg.window_size, num_features), dtype=np.float32)
    y_all = np.empty(sequence_length, dtype=np.float32)
    
    # Convert feature data to numpy array once
    feature_data = data_for_sequences[feature_cols].values.astype(np.float32)
    target_data = data_for_sequences[target_col_scaled].values.astype(np.float32)
    
    # Use sliding window view for efficient sequence creation
    try:
        # Create sliding windows using stride tricks for X data
        X_windows = sliding_window_view(feature_data, 
                                      window_shape=(sequence_cfg.window_size, num_features),
                                      axis=(0, 1))
        X_all = X_windows.squeeze(axis=-1)  # Remove extra dimension
        
        # Extract target values efficiently
        y_all = target_data[sequence_cfg.window_size:sequence_cfg.window_size + sequence_length]
        
    except Exception as e:
        logging.warning(f"Sliding window optimization failed, falling back to loop: {e}")
        # Fallback to the original method if stride tricks fail
        for i in range(sequence_length):
            X_all[i] = feature_data[i:i + sequence_cfg.window_size]
            y_all[i] = target_data[i + sequence_cfg.window_size]

    # Reshape y to column vector
    y_all = y_all.reshape(-1, 1)

    # Split: Test set first
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=sequence_cfg.test_size, shuffle=False
    )

    # Split: Validation from remaining train_val
    if len(X_train_val) > 1 and sequence_cfg.val_size_from_train_val > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=sequence_cfg.val_size_from_train_val, shuffle=False
        )
    else: # Not enough data for validation split or val_size is 0
        logging.warning("Insufficient data for validation split or val_size_from_train_val is 0. Validation set will be empty or X_train_val used as X_train.")
        X_train, y_train = X_train_val, y_train_val
        # Create empty val arrays with correct dimensions
        val_X_shape = (0, X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 and X_train.shape[0] > 0 else (0,0,0)
        val_y_shape = (0, y_train.shape[1]) if y_train.ndim == 2 and y_train.shape[0] > 0 else (0,1)
        X_val, y_val = np.empty(val_X_shape), np.empty(val_y_shape)


    logging.info(f"Data split complete: "
                 f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
                 f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
                 f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

