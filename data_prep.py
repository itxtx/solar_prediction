import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import math


def create_solar_elevation_proxy(time_str, sunrise_str, sunset_str):
    """Creates a continuous feature representing approximate solar elevation
    
    Args:
        time_str: Current time in 'HH:MM' format
        sunrise_str: Sunrise time in 'HH:MM' format 
        sunset_str: Sunset time in 'HH:MM' format
        
    Returns:
        A value between 0-1 representing solar elevation
    """
    # Convert times to minutes
    time_parts = time_str.split(':')
    current_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
    
    sunrise_parts = sunrise_str.split(':')
    sunrise_minutes = int(sunrise_parts[0]) * 60 + int(sunrise_parts[1])
    
    sunset_parts = sunset_str.split(':')
    sunset_minutes = int(sunset_parts[0]) * 60 + int(sunset_parts[1])
    
    # Handle wrap-around for night hours
    if current_minutes < sunrise_minutes and current_minutes > 0:
        # It's early morning before sunrise
        return 0
    if current_minutes > sunset_minutes and current_minutes < 24*60:
        # It's evening after sunset
        return 0
    
    # For daytime, calculate normalized position in daylight hours (0 to 1)
    daylight_duration = sunset_minutes - sunrise_minutes
    midday_point = sunrise_minutes + (daylight_duration / 2)
    
    if current_minutes <= midday_point:
        # Morning to noon
        position = (current_minutes - sunrise_minutes) / (midday_point - sunrise_minutes)
    else:
        # Noon to evening
        position = 1 - ((current_minutes - midday_point) / (sunset_minutes - midday_point))
    
    # Return value between 0-1 representing solar elevation
    return math.sin(position * math.pi/2)  # Creates sine curve peaking at 1.0 during midday


def piecewise_radiation_transform(radiation_values):
    """
    Applies different scaling to different radiation ranges.
    
    Args:
        radiation_values: NumPy array of radiation values.
        
    Returns:
        Transformed values with preserved nighttime accuracy.
    """
    transformed = np.zeros_like(radiation_values, dtype=float)
    
    # Night/low radiation (preserve accuracy for values near zero)
    night_mask = radiation_values < 10
    transformed[night_mask] = np.log1p(radiation_values[night_mask]) 
    
    # Morning/evening radiation (moderate scaling)
    moderate_mask = (radiation_values >= 10) & (radiation_values < 200)
    transformed[moderate_mask] = np.log1p(9.999) + 0.05 * (radiation_values[moderate_mask] - 10)
    
    # Midday/high radiation (reduced compression)
    high_mask = radiation_values >= 200
    moderate_max = np.log1p(9.999) + 0.05 * (199.999 - 10) 
    transformed[high_mask] = moderate_max + 0.002 * (radiation_values[high_mask] - 200)

    return transformed


def prepare_weather_data(df_input, target_col, window_size=12, test_size=0.2, val_size=0.25,
                         # --- Transformation Flags ---
                         use_log_transform=False,
                         use_power_transform=False,
                         # --- End Transformation Flags ---
                         min_radiation_for_log=0.1,
                         min_radiation_floor=0.0,
                         # --- Parameters for pre-power-transform clipping of original target ---
                         clip_original_target_before_transform=False,
                         original_clip_lower_percentile=5.0,
                         original_clip_upper_percentile=95.0,
                         # --- End of new parameters for pre-transform clipping ---
                         clip_log_target=False,
                         log_clip_lower_percentile=1.0,
                         log_clip_upper_percentile=99.0,
                         use_piecewise_transform=False, 
                         use_solar_elevation=True,
                         standardize_features=True,
                         feature_selection_mode='all',
                         min_target_threshold=None):
    """
    Prepare weather time series data for LSTM training, adapted for the new dataset.
    Includes optional clipping of the original target, Yeo-Johnson power transformation or
    log transformation for the target, flooring, and other feature engineering.

    Args:
        df_input: DataFrame with weather data. Will be copied.
        target_col: Column to predict (e.g., 'Energy delta[Wh]' or 'GHI').
        window_size: Size of the sliding window.
        test_size: Proportion for testing.
        val_size: Proportion for validation.
        use_log_transform: Whether to apply log transform (if power transform is not used).
        use_power_transform: Whether to apply Yeo-Johnson power transform to the target.
        min_radiation_for_log: Floor for radiation values before log transform (applies if target is GHI).
        min_radiation_floor: Floor for radiation values AFTER original clipping but BEFORE power transform (applies if target is GHI).
        clip_original_target_before_transform: Whether to clip the original target (e.g., GHI)
                                               before other transformations.
        original_clip_lower_percentile: Lower percentile for original target clipping.
        original_clip_upper_percentile: Upper percentile for original target clipping.
        clip_log_target: Whether to clip log-transformed target values.
        log_clip_lower_percentile: Lower percentile for clipping log-transformed target.
        log_clip_upper_percentile: Upper percentile for clipping log-transformed target.
        use_piecewise_transform: Whether to apply piecewise transform for GHI.
        use_solar_elevation: Whether to add solar elevation proxy.
        standardize_features: If True, use StandardScaler for features, else MinMaxScaler.
                              Target scaler depends on transformations applied.
        feature_selection_mode: 'all', 'basic', or 'minimal'.
        min_target_threshold: Initial minimum threshold for original target values.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
    """
    # ====================================================================
    # 1. INITIALIZE AND PREPARE DATAFRAME
    # ====================================================================
    df = df_input.copy()
    
    # Initialize transform tracking dictionaries
    piecewise_transform_details = {'applied': False, 'type': None, 'original_col': None}
    log_transform_details = {'applied': False, 'type': None, 'offset': 0, 'original_col': None, 'clip_bounds': None}
    power_transform_details = {
        'applied': False, 'type': None, 'lambda': None, 'original_col': None, 
        'scaler_used_after': None, 'power_transformer_obj': None, 'original_clip_bounds': None
    }
    
    # Standardize column names for processing
    column_map, df = _standardize_column_names(df, df_input, target_col)
    
    # Keep track of the original target column name and the processed name
    original_target_col_name = target_col
    target_col = _get_processed_target_col_name(target_col, df)
    
    # ====================================================================
    # 2. PROCESS TIME FEATURES
    # ====================================================================
    df = _process_time_features(df, df_input, use_solar_elevation, column_map)
    
    # ====================================================================
    # 3. APPLY TARGET TRANSFORMATIONS
    # ====================================================================
    # Apply initial threshold to target if specified
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying initial min_target_threshold of {min_target_threshold} to original '{target_col}'")
        df[target_col] = df[target_col].clip(lower=min_target_threshold)
    
    # Track current target column name as it goes through transformations
    target_col_processing = target_col
    
    # Check if target is radiation (renamed from GHI)
    is_radiation_target = (target_col_processing == 'Radiation')
    
    # Step 1: Optional clipping of the original target variable
    if clip_original_target_before_transform and is_radiation_target:
        target_col_processing, power_transform_details = _clip_original_target(
            df, target_col_processing, original_clip_lower_percentile, 
            original_clip_upper_percentile, power_transform_details
        )
    
    # Step 2: Apply piecewise transform if needed
    if is_radiation_target and use_piecewise_transform:
        target_col_processing, piecewise_transform_details = _apply_piecewise_transform(
            df, target_col_processing
        )
    
    # Step 3: Apply power transform OR log transform
    current_target_is_radiation_family = (
        target_col_processing == 'Radiation' or 
        target_col_processing == 'Radiation_transformed'
    )
    
    if use_power_transform and current_target_is_radiation_family:
        target_col_processing, power_transform_details = _apply_power_transform(
            df, target_col_processing, min_radiation_floor
        )
        use_log_transform = False
    elif use_log_transform:
        target_col_processing, log_transform_details = _apply_log_transform(
            df, target_col_processing, current_target_is_radiation_family,
            min_radiation_for_log, clip_log_target,
            log_clip_lower_percentile, log_clip_upper_percentile
        )
    
    # Final target column name after all transformations
    target_col_actual = target_col_processing
    
    # ====================================================================
    # 4. FEATURE ENGINEERING & SELECTION
    # ====================================================================
    # Add low threshold indicator feature if applicable
    _add_low_threshold_indicator(df, target_col)
    
    # Select features based on mode
    feature_cols = _select_features(df, df_input, feature_selection_mode, target_col, use_solar_elevation, column_map)
    
    # Ensure target isn't in feature list
    if target_col_actual in feature_cols:
        print(f"Warning: Final target column '{target_col_actual}' is also in feature_cols. Removing it from features.")
        feature_cols = [f_col for f_col in feature_cols if f_col != target_col_actual]
    
    # ====================================================================
    # 5. SCALING
    # ====================================================================
    cols_to_scale, df_for_scaling = _prepare_columns_for_scaling(df, feature_cols, target_col_actual)
    
    # Scale features and target
    scaled_data_df, scalers = _scale_features_and_target(
        df_for_scaling, feature_cols, target_col_actual, 
        standardize_features, power_transform_details
    )
    
    # ====================================================================
    # 6. CREATE SEQUENCES
    # ====================================================================
    X_train, X_val, X_test, y_train, y_val, y_test = _create_sequences(
        scaled_data_df, feature_cols, target_col_actual, window_size, test_size, val_size
    )
    
    # ====================================================================
    # 7. PREPARE RETURN VALUES
    # ====================================================================
    combined_transform_info = _prepare_transform_info(
        original_target_col_name, target_col, target_col_actual,
        piecewise_transform_details, power_transform_details, log_transform_details, scalers
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def _standardize_column_names(df, df_input, target_col):
    """Standardize column names for consistent processing."""
    # Map common column names to standardized names
    column_map = {
        'temp': 'Temperature',
        'pressure': 'Pressure',
        'humidity': 'Humidity',
        'wind_speed': 'Speed',
        'GHI': 'Radiation',  # Crucial for radiation-specific transforms
    }
    
    df = df.rename(columns=column_map, inplace=False)
    return column_map, df


def _get_processed_target_col_name(target_col, df):
    """Handle potentially problematic characters in target column name."""
    if target_col == 'Energy delta[Wh]':
        processed_name = 'Energy_delta_Wh'
        df.rename(columns={'Energy delta[Wh]': processed_name}, inplace=True)
        print(f"Renamed target column '{target_col}' to '{processed_name}'")
        return processed_name
    return target_col


def _process_time_features(df, df_input, use_solar_elevation, column_map):
    """Process time-related features in the dataframe."""
    # Parse the primary 'Time' column
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
    elif 'UNIXTime' in df.columns:  # Fallback for old format
        df = df.sort_values('UNIXTime').reset_index(drop=True)
    else:
        print("Warning: Neither 'Time' nor 'UNIXTime' column found for sorting.")
    
    # Extract time features
    _add_time_minute_features(df)
    _add_hour_and_month_features(df, df_input)
    _add_daylight_features(df, df_input)
    
    # Add solar elevation if requested and possible
    if use_solar_elevation and 'DaylightPosition' in df.columns:
        _add_solar_elevation(df)
    
    return df


def _add_time_minute_features(df):
    """Extract time minute features from datetime column."""
    if 'Time' in df.columns and isinstance(df['Time'].iloc[0], pd.Timestamp):
        df['TimeMinutes'] = df['Time'].apply(_time_to_minutes_from_datetime)
        minutes_in_day = 24 * 60
        df['TimeMinutesSin'] = np.sin(2 * np.pi * df['TimeMinutes'] / minutes_in_day)
        df['TimeMinutesCos'] = np.cos(2 * np.pi * df['TimeMinutes'] / minutes_in_day)


def _time_to_minutes_from_datetime(dt_obj):
    """Convert datetime object to minutes in day."""
    if pd.isna(dt_obj):
        return np.nan
    return dt_obj.hour * 60 + dt_obj.minute + dt_obj.second / 60


def _add_hour_and_month_features(df, df_input):
    """Add hour and month features from appropriate source, avoiding redundancy."""
    # Add hour feature - prefer original input column if available
    if 'hour' in df_input.columns:
        df['HourOfDay'] = df_input['hour']
        print("Using original 'hour' column from input data")
    elif 'TimeMinutes' in df:
        df['HourOfDay'] = df['TimeMinutes'] / 60
        print("Derived 'HourOfDay' from TimeMinutes")
    elif 'Time' in df.columns:
        df['HourOfDay'] = df['Time'].dt.hour
        print("Derived 'HourOfDay' from Time column")
    
    # Add month feature - prefer original input column if available
    if 'month' in df_input.columns:
        df['Month'] = df_input['month']
        print("Using original 'month' column from input data")
    elif 'Time' in df.columns:
        df['Month'] = df['Time'].dt.month
        print("Derived 'Month' from Time column")


def _add_daylight_features(df, df_input):
    """Add features related to daylight, avoiding redundancy."""
    # Add daylight minutes - use input column if available or calculate
    if 'dayLength' in df_input.columns:
        df['DaylightMinutes'] = df_input['dayLength']
        print("Using original 'dayLength' column from input data")
    elif 'TimeSunRise' in df.columns and 'TimeSunSet' in df.columns:
        _add_sunrise_sunset_features(df)
        print("Calculated 'DaylightMinutes' from sunrise/sunset times")
    
    # Only proceed if we have daylight minutes
    if 'DaylightMinutes' in df.columns and 'TimeMinutes' in df.columns:
        # Add daylight indicator - use input column if available
        if 'isSun' in df_input.columns:
            df['IsDaylight'] = df_input['isSun'].astype(float)
            print("Using original 'isSun' column as 'IsDaylight'")
        elif 'SunriseMinutes' in df.columns and 'SunsetMinutes' in df.columns:
            df['IsDaylight'] = (
                (df['TimeMinutes'] >= df['SunriseMinutes']) & 
                (df['TimeMinutes'] <= df['SunsetMinutes'])
            ).astype(float)
            print("Calculated 'IsDaylight' from sunrise/sunset times")
        
        # Add daylight position - use input column if available
        if 'SunlightTime/daylength' in df_input.columns:
            df['DaylightPosition'] = df_input['SunlightTime/daylength'].clip(0, 1)
            print("Using original 'SunlightTime/daylength' column as 'DaylightPosition'")
        elif 'TimeSinceSunrise' in df.columns and df['DaylightMinutes'].nunique() > 1:
            # Ensure DaylightMinutes is not zero to avoid division by zero
            df['DaylightPosition'] = (df['TimeSinceSunrise'] / df['DaylightMinutes'].replace(0, np.nan)).clip(0, 1)
            print("Calculated 'DaylightPosition' from time since sunrise and daylight minutes")
        
        # Add sunrise/sunset timing features
        _add_sunrise_sunset_timing(df)


def _add_sunrise_sunset_features(df):
    """Calculate sunrise and sunset features."""
    df['SunriseMinutes'] = df['TimeSunRise'].apply(_time_to_minutes_str_parse)
    df['SunsetMinutes'] = df['TimeSunSet'].apply(_time_to_minutes_str_parse)
    df['DaylightMinutes'] = df['SunsetMinutes'] - df['SunriseMinutes']
    df.loc[df['DaylightMinutes'] < 0, 'DaylightMinutes'] += 1440  # Handles overnight


def _time_to_minutes_str_parse(time_val):
    """Parse various time string formats to minutes."""
    if pd.isna(time_val):
        return np.nan
    if hasattr(time_val, 'hour'):
        return time_val.hour * 60 + time_val.minute + time_val.second / 60
    elif isinstance(time_val, str):
        if ' ' in time_val:
            time_part = time_val.split(' ')[-1]
        else:
            time_part = time_val
        if ':' in time_part:
            parts = time_part.split(':')
            if len(parts) >= 2:
                hours, minutes = map(int, parts[:2])
            else:
                return np.nan
            return hours * 60 + minutes
    return np.nan


def _add_sunrise_sunset_timing(df):
    """Add timing features related to sunrise and sunset."""
    # Add time since sunrise if possible
    if 'SunriseMinutes' in df.columns:
        df['TimeSinceSunrise'] = (df['TimeMinutes'] - df['SunriseMinutes'])
        df.loc[df['TimeSinceSunrise'] < 0, 'TimeSinceSunrise'] += 1440
        print("Added 'TimeSinceSunrise' feature")
    
    # Add time until sunset if possible
    if 'SunsetMinutes' in df.columns:
        df['TimeUntilSunset'] = (df['SunsetMinutes'] - df['TimeMinutes'])
        df.loc[df['TimeUntilSunset'] < 0, 'TimeUntilSunset'] += 1440
        print("Added 'TimeUntilSunset' feature")


def _add_solar_elevation(df):
    """Add solar elevation proxy feature."""
    print("Adding solar elevation proxy feature")
    df['SolarElevation'] = 0.0
    # Apply only where DaylightPosition is valid (not NaN)
    valid_idx = df['DaylightPosition'].dropna().index
    df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
        lambda x: math.sin(x * math.pi) if (x >= 0 and x <= 1) else 0
    )
    print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")


def _clip_original_target(df, target_col, lower_percentile, upper_percentile, power_transform_details):
    """Clip original target values to specified percentiles."""
    if target_col not in df.columns:
        print(f"Warning: Column '{target_col}' for original clipping not found. Skipping.")
        return target_col, power_transform_details
    
    original_target_values = df[target_col].values
    lower_bound_orig = np.percentile(original_target_values, lower_percentile)
    upper_bound_orig = np.percentile(original_target_values, upper_percentile)
    
    power_transform_details['original_clip_bounds'] = (float(lower_bound_orig), float(upper_bound_orig))
    print(f"Clipping original '{target_col}' to range [{lower_bound_orig:.4f}, {upper_bound_orig:.4f}] BEFORE other transformations.")
    
    df[target_col] = np.clip(original_target_values, lower_bound_orig, upper_bound_orig)
    return target_col, power_transform_details


def _apply_piecewise_transform(df, target_col):
    """Apply piecewise radiation transform to target column."""
    print(f"Applying piecewise radiation transform to '{target_col}' column.")
    df['Radiation_transformed'] = piecewise_radiation_transform(df[target_col].values)
    
    piecewise_details = {
        'applied': True, 
        'type': 'piecewise_radiation', 
        'original_col': target_col
    }
    
    transformed_col = 'Radiation_transformed'
    print(f"Using '{transformed_col}' as target for further processing.")
    
    return transformed_col, piecewise_details


def _apply_power_transform(df, target_col, min_radiation_floor):
    """Apply Yeo-Johnson power transform to target column."""
    print(f"Applying Yeo-Johnson Power Transform to '{target_col}'")
    
    if min_radiation_floor is not None:
        print(f"Applying floor of {min_radiation_floor} to '{target_col}' before Power Transform.")
        df[target_col] = np.maximum(df[target_col].astype(float), min_radiation_floor)
    
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    values_to_transform = df[target_col].values.reshape(-1, 1)
    transformed_values = power_transformer.fit_transform(values_to_transform)
    
    col_fed_to_power_transform = target_col
    new_target_col_name = f'{col_fed_to_power_transform}_yeo'
    df[new_target_col_name] = transformed_values.flatten()
    
    power_transform_details = {
        'applied': True, 
        'type': 'yeo-johnson',
        'lambda': power_transformer.lambdas_[0] if power_transformer.lambdas_ is not None else None,
        'original_col': col_fed_to_power_transform,
        'scaler_used_after': 'StandardScaler',
        'power_transformer_obj': power_transformer
    }
    
    print(f"Yeo-Johnson applied (lambda={power_transform_details['lambda']:.4f}). New target: '{new_target_col_name}'")
    
    return new_target_col_name, power_transform_details


def _apply_log_transform(df, target_col, is_radiation, min_radiation_for_log, 
                         clip_log_target, log_clip_lower_percentile, log_clip_upper_percentile):
    """Apply logarithmic transform to target column."""
    is_other_loggable_target = target_col in ['Temperature', 'Speed', 'Energy_delta_Wh']
    eligible_for_log = is_radiation or is_other_loggable_target
    
    if not eligible_for_log or target_col not in df.columns:
        print(f"Log transform not applied to '{target_col}' as it's not an eligible type or configuration for log.")
        return target_col, {'applied': False, 'type': None, 'offset': 0, 'original_col': None, 'clip_bounds': None}
    
    epsilon = 1e-6
    original_values_for_log = df[target_col].copy()
    log_input_col_name = target_col
    log_output_col_name = f'{log_input_col_name}_log'
    
    if is_radiation:  # Specific handling for radiation
        print(f"Applying floor of {min_radiation_for_log} to '{log_input_col_name}' before log transform.")
        floored_values = np.maximum(original_values_for_log, min_radiation_for_log)
        df[log_output_col_name] = np.log(floored_values + epsilon)
    else:  # For Temperature, Speed, Energy_delta_Wh, etc.
        if (original_values_for_log < 0).any():
            print(f"Warning: Negative values found in '{log_input_col_name}'. Log transform might produce NaNs or errors.")
            print(f"Applying np.log(np.maximum(original_values_for_log, 0) + epsilon). Review if this is appropriate.")
            df[log_output_col_name] = np.log(np.maximum(original_values_for_log, 0) + epsilon)
        else:
            df[log_output_col_name] = np.log(original_values_for_log + epsilon)
    
    log_transform_details = {
        'applied': True, 
        'type': 'log', 
        'offset': epsilon,
        'original_col': log_input_col_name,
        'clip_bounds': None
    }
    
    transformed_col = log_output_col_name
    print(f"Log-transformed '{log_transform_details['original_col']}' -> '{transformed_col}'")
    
    if clip_log_target:
        log_values = df[transformed_col].values
        lower_bound = np.percentile(log_values, log_clip_lower_percentile)
        upper_bound = np.percentile(log_values, log_clip_upper_percentile)
        log_transform_details['clip_bounds'] = (float(lower_bound), float(upper_bound))
        print(f"Clipping log-transformed '{transformed_col}' to range [{lower_bound:.4f}, {upper_bound:.4f}]")
        df[transformed_col] = np.clip(log_values, lower_bound, upper_bound)
    
    return transformed_col, log_transform_details


def _add_low_threshold_indicator(df, target_col):
    """Add indicator for low target values."""
    low_threshold_col_ref_options = ['Radiation', 'Temperature', 'Speed', 'Energy_delta_Wh']
    
    if target_col in low_threshold_col_ref_options and target_col in df.columns:
        values_for_threshold = df[target_col]
        low_threshold = values_for_threshold.quantile(0.1)
        df[f'{target_col}_is_low'] = (values_for_threshold < low_threshold).astype(float)
        print(f"Added '{target_col}_is_low' feature (threshold: {low_threshold:.4f})")


def _select_features(df, df_input, feature_selection_mode, target_col, use_solar_elevation, column_map):
    """Select features based on specified mode, removing redundancies."""
    # Define base feature column sets with potential redundancies removed
    base_feature_cols_map = {
        'minimal': ['Radiation', 'Temperature', 'Humidity', 'TimeMinutesSin', 'TimeMinutesCos', 'clouds_all'],
        'basic': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'Speed',
                 'TimeMinutesSin', 'TimeMinutesCos', 'clouds_all', 'rain_1h'],
        'all': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'Speed',
               'rain_1h', 'snow_1h', 'clouds_all', 'weather_type']
    }
    
    # Select appropriate feature columns based on mode
    if feature_selection_mode == 'all':
        base_feature_cols = _get_all_relevant_features(df, df_input, base_feature_cols_map, column_map)
    else:
        base_feature_cols = [col for col in base_feature_cols_map.get(feature_selection_mode, base_feature_cols_map['all'])
                            if col in df.columns or col in df_input.columns]
    
    # Add low indicator feature if available
    low_indicator_col = f'{target_col}_is_low'
    if low_indicator_col in df.columns and low_indicator_col not in base_feature_cols:
        base_feature_cols.append(low_indicator_col)
    
    # Add solar elevation if requested
    if 'SolarElevation' in df.columns and use_solar_elevation and 'SolarElevation' not in base_feature_cols:
        base_feature_cols.append('SolarElevation')
    
    # Add time features
    feature_cols = _add_time_feature_columns(df, base_feature_cols)
    
    # Add new dataset features explicitly
    feature_cols = _add_new_dataset_features(df, df_input, feature_cols)
    
    # Remove redundancies
    feature_cols = _remove_redundant_features(df, df_input, feature_cols, column_map)
    
    # Remove duplicates and sort
    feature_cols = sorted(list(set(feature_cols)))
    
    # Verify all selected features exist
    feature_cols = [f_col for f_col in feature_cols if f_col in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns selected or available in the DataFrame after processing.")
    
    print(f"Final selected features before scaling ({len(feature_cols)}): {feature_cols}")
    return feature_cols


def _remove_redundant_features(df, df_input, feature_cols, column_map):
    """Remove redundant features to avoid duplication."""
    # Define pairs of redundant features (original_name, derived_name)
    redundant_pairs = [
        ('dayLength', 'DaylightMinutes'),
        ('SunlightTime/daylength', 'DaylightPosition'),
        ('hour', 'HourOfDay'),
        ('month', 'Month'),
        ('isSun', 'IsDaylight'),
        # Add any other redundant pairs here
    ]
    
    # Remove redundancies
    for orig, derived in redundant_pairs:
        if orig in feature_cols and derived in feature_cols:
            print(f"Removing redundant feature '{derived}' since '{orig}' is already present")
            feature_cols.remove(derived)
        
        # Check for mapped column names as well
        mapped_orig = column_map.get(orig, orig)
        if mapped_orig in feature_cols and derived in feature_cols:
            print(f"Removing redundant feature '{derived}' since mapped '{mapped_orig}' (from '{orig}') is already present")
            feature_cols.remove(derived)
    
    return feature_cols


def _get_all_relevant_features(df, df_input, base_feature_cols_map, column_map):
    """Get all relevant features for 'all' mode."""
    # Start with mapped names from 'all' set, then add relevant columns
    current_all_selection = [col for col in base_feature_cols_map['all'] 
                            if col in df.columns or col in df_input.columns]
    
    # Add any other original columns not explicitly listed but present in the remapped df
    for orig_col in df_input.columns:
        mapped_col = column_map.get(orig_col, orig_col)  # Get mapped name if exists
        if mapped_col in df.columns and mapped_col not in current_all_selection:
            current_all_selection.append(mapped_col)
        elif (orig_col in df.columns and orig_col not in current_all_selection 
              and orig_col not in column_map.values()):
            current_all_selection.append(orig_col)
    
    return list(set(current_all_selection))  # Use set to remove duplicates


def _add_time_feature_columns(df, base_feature_cols):
    """Add time-related feature columns to the base set."""
    feature_cols = base_feature_cols.copy()
    
    time_features = [
        'TimeMinutesSin', 'TimeMinutesCos', 'HourOfDay', 'Month',
        'DaylightMinutes', 'IsDaylight', 'DaylightPosition'
    ]
    
    # Add conditional time features
    if 'TimeSinceSunrise' in df.columns:
        time_features.append('TimeSinceSunrise')
    if 'TimeUntilSunset' in df.columns:
        time_features.append('TimeUntilSunset')
    
    # Add available time features that aren't already in the feature list
    for feature in time_features:
        if feature in df.columns and feature not in feature_cols and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
    
    return feature_cols


def _add_new_dataset_features(df, df_input, feature_cols):
    """Add new dataset-specific features."""
    # Original names from new dataset
    additional_new_features = ['rain_1h', 'snow_1h', 'clouds_all', 'weather_type']
    direct_time_features = ['hour', 'month', 'isSun', 'sunlightTime', 'dayLength', 'SunlightTime/daylength']
    
    for f in additional_new_features + direct_time_features:
        # Check if it was renamed through column_map
        col_to_check = f  # Default to original name
        if col_to_check in df.columns and col_to_check not in feature_cols:
            feature_cols.append(col_to_check)
        elif f in df_input.columns and f not in feature_cols and f not in df.columns:
            # If it's in original input but not in current df and features
            df[f] = df_input[f]  # Add to working df
            feature_cols.append(f)
    
    return feature_cols


def _prepare_columns_for_scaling(df, feature_cols, target_col_actual):
    """Prepare columns for scaling and handle missing values."""
    # Combine feature columns and target column for scaling
    cols_to_scale = feature_cols.copy()
    if target_col_actual not in cols_to_scale and target_col_actual in df.columns:
        cols_to_scale.append(target_col_actual)
    
    # Verify columns exist in DataFrame
    cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    
    # Check for valid data
    if not df[cols_to_scale].isna().all().all():
        df_for_scaling = df[cols_to_scale].copy()
        
        # Handle potential non-numeric types
        for col in df_for_scaling.columns:
            if df_for_scaling[col].dtype == 'object':
                print(f"Warning: Column '{col}' is of object type. Attempting to convert to numeric.")
                df_for_scaling[col] = pd.to_numeric(df_for_scaling[col], errors='coerce')
        
        # Fill missing values
        df_for_scaling = df_for_scaling.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError(f"All values in columns selected for scaling are NaN: {cols_to_scale}")
    
    return cols_to_scale, df_for_scaling


def _scale_features_and_target(df_for_scaling, feature_cols, target_col_actual, 
                               standardize_features, power_transform_details):
    """Scale features and target to prepare for model training."""
    scalers = {}
    scaled_data_df = pd.DataFrame(index=df_for_scaling.index)
    
    # Choose appropriate scaler for features
    FeatureScalerClass = StandardScaler if standardize_features else MinMaxScaler
    print(f"Using {FeatureScalerClass.__name__} for feature scaling.")
    
    # Scale each feature
    for col in feature_cols:
        if col not in df_for_scaling.columns:
            print(f"Warning: Feature column '{col}' not found in df_for_scaling. Skipping scaling.")
            continue
        
        # Check for all NaN values
        if df_for_scaling[col].isnull().all():
            print(f"Warning: All values in feature column '{col}' are NaN before scaling. Scaled output will be NaN.")
            scaled_data_df[col] = np.nan
            scalers[col] = None  # No scaler fitted
            continue
        
        # Create and apply scaler
        feature_scaler = (FeatureScalerClass() if standardize_features 
                         else FeatureScalerClass(feature_range=(0, 1)))
        values = df_for_scaling[col].values.reshape(-1, 1)
        scaled_data_df[col] = feature_scaler.fit_transform(values).flatten()
        scalers[col] = feature_scaler
    
    # Verify target column exists
    if target_col_actual not in df_for_scaling.columns:
        raise ValueError(f"Final target column '{target_col_actual}' not found for scaling.")
    if df_for_scaling[target_col_actual].isnull().all():
        raise ValueError(f"All values in target column '{target_col_actual}' are NaN before scaling. Cannot proceed.")
    
    # Choose appropriate scaler for target
    if power_transform_details['applied']:
        TargetScalerClass = StandardScaler
        print(f"Using StandardScaler for Yeo-Johnson transformed target '{target_col_actual}'.")
    else:
        TargetScalerClass = StandardScaler if standardize_features else MinMaxScaler
        print(f"Using {TargetScalerClass.__name__} for target '{target_col_actual}'.")
    
    # Create and apply target scaler
    target_scaler = (TargetScalerClass() if TargetScalerClass == StandardScaler 
                    else TargetScalerClass(feature_range=(0, 1)))
    target_values_to_scale = df_for_scaling[target_col_actual].values.reshape(-1, 1)
    
    # Log target statistics before scaling
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' BEFORE scaling: "
          f"Mean={np.nanmean(target_values_to_scale):.4f}, Std={np.nanstd(target_values_to_scale):.4f}, "
          f"Min={np.nanmin(target_values_to_scale):.4f}, Max={np.nanmax(target_values_to_scale):.4f}")
    
    # Scale target and store scaler
    scaled_data_df[target_col_actual] = target_scaler.fit_transform(target_values_to_scale).flatten()
    scalers[target_col_actual] = target_scaler
    
    # Log target statistics after scaling
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' AFTER scaling: "
          f"Mean={np.nanmean(scaled_data_df[target_col_actual]):.4f}, Std={np.nanstd(scaled_data_df[target_col_actual]):.4f}, "
          f"Min={np.nanmin(scaled_data_df[target_col_actual]):.4f}, Max={np.nanmax(scaled_data_df[target_col_actual]):.4f}")
    
    # Log scaler parameters
    if isinstance(target_scaler, StandardScaler):
        if hasattr(target_scaler, 'mean_') and target_scaler.mean_ is not None:
            print(f"DEBUG [PREPARE DATA]: Target Scaler learned mean: {target_scaler.mean_[0]:.4f}, "
                  f"scale (std): {target_scaler.scale_[0]:.4f}")
        else:
            print(f"DEBUG [PREPARE DATA]: Target Scaler ({TargetScalerClass.__name__}) attributes "
                  f"(mean_, scale_) not found (possibly all NaN input or not fitted).")
    
    return scaled_data_df, scalers


def _create_sequences(scaled_data_df, feature_cols, target_col_actual, window_size, test_size, val_size):
    """Create sequence data for time series model training."""
    X, y_seq = [], []
    
    # Check if data is sufficient for sequences
    if len(scaled_data_df) <= window_size:
        if scaled_data_df.empty or scaled_data_df[feature_cols + [target_col_actual]].isna().all().all():
            raise ValueError(f"scaled_data_df is empty or all NaN for selected cols. "
                            f"Data length ({len(scaled_data_df)}) is <= window_size ({window_size}).")
        raise ValueError(f"Data length ({len(scaled_data_df)}) is <= window_size ({window_size}). "
                        f"Cannot create sequences.")
    
    # Remove rows with NaN values
    sequence_input_df = scaled_data_df[feature_cols + [target_col_actual]].dropna()
    if len(sequence_input_df) <= window_size:
        raise ValueError(f"Data length after dropping NaNs ({len(sequence_input_df)}) is <= window_size ({window_size}).")
    
    # Create sliding window sequences
    for i in range(len(sequence_input_df) - window_size):
        X.append(sequence_input_df.iloc[i:i+window_size][feature_cols].values)
        y_seq.append(sequence_input_df[target_col_actual].iloc[i+window_size])
    
    # Handle case with no sequences
    if not X:
        print("Warning: No sequences were created. X and y will be empty. "
              "Check data length and window size after NaN handling.")
        num_features = len(feature_cols)
        return (np.empty((0, window_size, num_features)), np.empty((0, window_size, num_features)), 
                np.empty((0, window_size, num_features)), np.empty((0,1)), np.empty((0,1)), np.empty((0,1)))
    
    # Convert to arrays
    X = np.array(X)
    y_seq = np.array(y_seq).reshape(-1, 1)
    
    # Split data into train, validation, and test sets
    return _split_train_val_test(X, y_seq, test_size, val_size, feature_cols)


def _split_train_val_test(X, y_seq, test_size, val_size, feature_cols):
    """Split sequences into training, validation, and test sets."""
    # Split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_seq, test_size=test_size, shuffle=False)
    
    # Calculate validation size relative to remaining data
    actual_val_size_for_split = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
    
    # Handle edge cases
    if len(X_temp) == 0:
        X_train, X_val, y_train, y_val = np.array([]), np.array([]), np.array([]), np.array([])
        print("Warning: No data available for training/validation after initial test split.")
    elif actual_val_size_for_split >= 1.0 or actual_val_size_for_split <= 0 or len(X_temp) < 2:
        print(f"Warning: Adjusted validation size ({actual_val_size_for_split:.2f}) is invalid or "
              f"insufficient data in X_temp ({len(X_temp)} samples).")
        
        # Try to split if possible
        if len(X_temp) >= 2 and 0 < actual_val_size_for_split < 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False
            )
        else:  # Not enough for split or val_size is 0
            X_train, y_train = X_temp, y_temp
            # Create empty validation arrays with correct dimensions
            if X_train.ndim == 3 and X_train.shape[1] > 0:
                val_shape_x = (0, X_train.shape[1], X_train.shape[2])
            elif X_train.ndim == 2 and X_train.shape[1] > 0:
                val_shape_x = (0, X_train.shape[1])
            else:
                val_shape_x = (0, 0, 0)
            
            val_shape_y = (0, y_train.shape[1]) if y_train.ndim == 2 else (0,)
            X_val, y_val = np.empty(val_shape_x), np.empty(val_shape_y)
            print("Using all remaining data for training, validation set will be empty.")
    else:
        # Normal case: split remaining data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False
        )
    
    # Log split sizes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Final features used for sequences ({len(feature_cols)}): {feature_cols}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _prepare_transform_info(original_target_col_name, target_col, target_col_actual,
                           piecewise_transform_details, power_transform_details, 
                           log_transform_details, scalers):
    """Prepare transform information for return."""
    combined_transform_info = {
        'transforms': [],
        'target_col_original': original_target_col_name,        # Very original name
        'target_col_processed_name': target_col,                # Name after potential rename
        'target_col_transformed_final': target_col_actual       # Name after transforms, before scaling
    }
    
    # Add transforms in order of application
    if piecewise_transform_details['applied']:
        combined_transform_info['transforms'].append(piecewise_transform_details)
    
    if power_transform_details['applied']:
        if 'power_transformer_obj' in power_transform_details:
            scalers['power_transformer_object_for_target'] = power_transform_details.pop('power_transformer_obj')
        combined_transform_info['transforms'].append(power_transform_details)
    elif log_transform_details['applied']:  # log and power are mutually exclusive
        combined_transform_info['transforms'].append(log_transform_details)
    
    return combined_transform_info