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



# Assume piecewise_radiation_transform is defined globally as provided
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
                         use_log_transform=False, # Original log transform flag
                         use_power_transform=False, # Flag to enable Yeo-Johnson for target
                         # --- End Transformation Flags ---
                         min_radiation_for_log=0.1, # Floor for radiation before log transform
                         min_radiation_floor=0.0, # Floor for radiation BEFORE power transform (0 is okay for Yeo-Johnson)
                         clip_log_target=False, 
                         log_clip_lower_percentile=1.0, 
                         log_clip_upper_percentile=99.0, 
                         use_piecewise_transform=False, use_solar_elevation=True,
                         standardize_features=True, # Note: This also influences target scaler if not power transformed
                         feature_selection_mode='all',
                         min_target_threshold=None):
    """
    Prepare weather time series data for LSTM training.
    Includes optional Yeo-Johnson power transformation or log transformation for the target,
    flooring, clipping, and other feature engineering.
    
    Args:
        df_input: DataFrame with weather data. Will be copied.
        target_col: Column to predict.
        window_size: Size of the sliding window.
        test_size: Proportion for testing.
        val_size: Proportion for validation.
        use_log_transform: Whether to apply log transform (if power transform is not used).
        use_power_transform: Whether to apply Yeo-Johnson power transform to the target (primarily Radiation).
        min_radiation_for_log: Floor for radiation values before log transform.
        min_radiation_floor: Floor for radiation values before power transform.
        clip_log_target: Whether to clip log-transformed target values.
        log_clip_lower_percentile: Lower percentile for clipping log-transformed target.
        log_clip_upper_percentile: Upper percentile for clipping log-transformed target.
        use_piecewise_transform: Whether to apply piecewise transform for Radiation.
        use_solar_elevation: Whether to add solar elevation proxy.
        standardize_features: If True, use StandardScaler for features (and target if not power-transformed),
                              else MinMaxScaler. If target is power-transformed, it's always standardized.
        feature_selection_mode: 'all', 'basic', or 'minimal'.
        min_target_threshold: Minimum threshold for original target values.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
    """
    df = df_input.copy() 

    if 'UNIXTime' in df.columns:
        df = df.sort_values('UNIXTime').reset_index(drop=True)
    
    # --- Time Feature Processing ---
    def time_to_minutes(time_val):
        if pd.isna(time_val): return np.nan
        if hasattr(time_val, 'hour'): 
            return time_val.hour * 60 + time_val.minute + time_val.second / 60
        elif isinstance(time_val, str): 
            if ' ' in time_val: time_part = time_val.split(' ')[-1]
            else: time_part = time_val
            if ':' in time_part:
                parts = time_part.split(':')
                if len(parts) == 3: hours, minutes, seconds = map(int, parts)
                elif len(parts) == 2: hours, minutes = map(int, parts); seconds = 0
                else: return np.nan
                return hours * 60 + minutes + seconds / 60
        return np.nan

    if 'Time' in df.columns:
        df['TimeMinutes'] = df['Time'].apply(time_to_minutes)
        minutes_in_day = 24 * 60
        df['TimeMinutesSin'] = np.sin(2 * np.pi * df['TimeMinutes'] / minutes_in_day)
        df['TimeMinutesCos'] = np.cos(2 * np.pi * df['TimeMinutes'] / minutes_in_day)
        df['HourOfDay'] = df['TimeMinutes'] / 60

    if 'TimeSunRise' in df.columns and 'TimeSunSet' in df.columns:
        df['SunriseMinutes'] = df['TimeSunRise'].apply(time_to_minutes)
        df['SunsetMinutes'] = df['TimeSunSet'].apply(time_to_minutes)
        df['DaylightMinutes'] = df['SunsetMinutes'] - df['SunriseMinutes']
        df.loc[df['DaylightMinutes'] < 0, 'DaylightMinutes'] += 1440

        if 'TimeMinutes' in df.columns:
            df['IsDaylight'] = ((df['TimeMinutes'] >= df['SunriseMinutes']) & \
                               (df['TimeMinutes'] <= df['SunsetMinutes'])).astype(float)
            df['TimeSinceSunrise'] = (df['TimeMinutes'] - df['SunriseMinutes'])
            df.loc[df['TimeSinceSunrise'] < 0, 'TimeSinceSunrise'] += 1440 
            df['TimeUntilSunset'] = (df['SunsetMinutes'] - df['TimeMinutes'])
            df.loc[df['TimeUntilSunset'] < 0, 'TimeUntilSunset'] += 1440
            df['DaylightPosition'] = (df['TimeSinceSunrise'] / df['DaylightMinutes']).clip(0, 1) # Clip to ensure 0-1
            
            if use_solar_elevation:
                print("Adding solar elevation proxy feature")
                df['SolarElevation'] = 0.0
                # Ensure DaylightMinutes is not NaN and is positive before division
                valid_idx = df[['TimeMinutes', 'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes']].dropna().index
                valid_idx = valid_idx.intersection(df[df['DaylightMinutes'] > 0].index)
                
                df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
                    lambda x: math.sin(x * math.pi) if (x >= 0 and x <= 1) else 0 # x should be in [0,1]
                )
                print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")

    # --- Target Pre-processing and Transformation ---
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying min_target_threshold of {min_target_threshold} to original '{target_col}'")
        df[target_col] = df[target_col].clip(lower=min_target_threshold)

    target_col_processing = target_col 
    
    piecewise_transform_details = {'applied': False, 'type': None, 'original_col': None}
    log_transform_details = {'applied': False, 'type': None, 'offset': 0, 'original_col': None, 'clip_bounds': None}
    power_transform_details = {'applied': False, 'type': None, 'lambda': None, 'original_col': None, 'scaler_used_after': None, 'power_transformer_obj': None}


    # 1. Piecewise Transform (if applicable, typically for Radiation)
    if target_col == 'Radiation' and use_piecewise_transform:
        print(f"Applying piecewise radiation transform to '{target_col}' column.")
        df['Radiation_transformed'] = piecewise_radiation_transform(df[target_col].values)
        piecewise_transform_details = {'applied': True, 'type': 'piecewise_radiation', 'original_col': target_col}
        target_col_processing = 'Radiation_transformed'
        print(f"Using '{target_col_processing}' as target for further processing.")

    # 2. Power Transform OR Log Transform
    # Power transform takes precedence if enabled for Radiation
    current_target_is_radiation_family = target_col_processing == 'Radiation' or \
                                       (piecewise_transform_details['applied'] and target_col_processing == 'Radiation_transformed')

    if use_power_transform and current_target_is_radiation_family:
        print(f"Applying Yeo-Johnson Power Transform to '{target_col_processing}'")
        
        if min_radiation_floor > 0: # Apply floor before power transform if specified
            print(f"Applying floor of {min_radiation_floor} to '{target_col_processing}' before Power Transform.")
            df[target_col_processing] = np.maximum(df[target_col_processing].astype(float), min_radiation_floor)

        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False) # Standardize separately later
        
        values_to_transform = df[target_col_processing].values.reshape(-1, 1)
        transformed_values = power_transformer.fit_transform(values_to_transform)
        
        log_input_col_name = target_col_processing # Save name of column fed to transform
        new_target_col_name = f'{log_input_col_name}_yeo'
        df[new_target_col_name] = transformed_values.flatten()

        power_transform_details = {
            'applied': True, 'type': 'yeo-johnson', 
            'lambda': power_transformer.lambdas_[0] if power_transformer.lambdas_ is not None else None,
            'original_col': log_input_col_name, 
            'scaler_used_after': 'StandardScaler', # Explicitly state that StandardScaler will be used
            'power_transformer_obj': power_transformer # Store the fitted transformer
        }
        target_col_processing = new_target_col_name
        print(f"Yeo-Johnson applied (lambda={power_transform_details['lambda']:.4f}). New target: '{target_col_processing}'")
        
        use_log_transform = False # Disable log transform if power transform was applied

    elif use_log_transform: # Original log transform logic
        is_other_loggable = target_col_processing in ['Temperature', 'Speed'] # Assuming not piecewise transformed to something else
        eligible_for_log = current_target_is_radiation_family or is_other_loggable

        if eligible_for_log and target_col_processing in df.columns:
            epsilon = 1e-6 
            original_values_for_log = df[target_col_processing].copy()
            log_input_col_name = target_col_processing 
            log_output_col_name = f'{log_input_col_name}_log'

            if current_target_is_radiation_family: # Specifically for Radiation or Radiation_transformed
                print(f"Applying floor of {min_radiation_for_log} to '{log_input_col_name}' before log transform.")
                floored_values = np.maximum(original_values_for_log, min_radiation_for_log)
                df[log_output_col_name] = np.log(floored_values + epsilon)
            else: # For Temperature, Speed
                df[log_output_col_name] = np.log(np.maximum(original_values_for_log, epsilon) + epsilon) 

            log_transform_details = {
                'applied': True, 'type': 'log', 'offset': epsilon, 
                'original_col': log_input_col_name, 
                'clip_bounds': None 
            }
            target_col_processing = log_output_col_name 
            print(f"Log-transformed '{log_transform_details['original_col']}' -> '{target_col_processing}'")

            if clip_log_target:
                log_values = df[target_col_processing].values
                lower_bound = np.percentile(log_values, log_clip_lower_percentile)
                upper_bound = np.percentile(log_values, log_clip_upper_percentile)
                log_transform_details['clip_bounds'] = (float(lower_bound), float(upper_bound))
                print(f"Clipping '{target_col_processing}' to range [{lower_bound:.4f}, {upper_bound:.4f}]")
                df[target_col_processing] = np.clip(log_values, lower_bound, upper_bound)
        
        elif not eligible_for_log:
             print(f"Log transform not applied to '{target_col_processing}' as it's not an eligible type or configuration.")
    
    target_col_actual = target_col_processing

    # --- Feature Engineering & Selection ---
    if target_col in ['Temperature', 'Radiation', 'Speed']: 
        low_threshold_col_ref = target_col 
        if low_threshold_col_ref in df.columns:
            low_threshold = df[low_threshold_col_ref].quantile(0.1)
            df[f'{low_threshold_col_ref}_is_low'] = (df[low_threshold_col_ref] < low_threshold).astype(float)
            print(f"Added '{low_threshold_col_ref}_is_low' feature (threshold: {low_threshold:.4f})")

    base_feature_cols_map = {
        'minimal': ['Radiation', 'Temperature', 'Humidity', 'TimeMinutesSin', 'TimeMinutesCos'],
        'basic': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'TimeMinutesSin', 'TimeMinutesCos'],
        'all': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']
    }
    base_feature_cols = [col for col in base_feature_cols_map.get(feature_selection_mode, base_feature_cols_map['all']) if col in df.columns]

    low_indicator_col = f'{target_col}_is_low'
    if low_indicator_col in df.columns and low_indicator_col not in base_feature_cols:
        base_feature_cols.append(low_indicator_col)
        
    if 'SolarElevation' in df.columns and use_solar_elevation and 'SolarElevation' not in base_feature_cols:
        base_feature_cols.append('SolarElevation')

    time_features = ['SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes', 'TimeSinceSunrise', 
                     'TimeUntilSunset', 'DaylightPosition', 'TimeMinutesSin', 'TimeMinutesCos', 
                     'HourOfDay', 'IsDaylight']
    
    feature_cols = base_feature_cols.copy()
    for feature in time_features:
        if feature in df.columns and feature not in feature_cols and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
            
    if target_col_actual in feature_cols:
        print(f"Warning: Final target column '{target_col_actual}' is also in feature_cols. Removing it from features.")
        feature_cols = [f for f in feature_cols if f != target_col_actual]

    feature_cols = [col for col in feature_cols if col in df.columns] 
    if not feature_cols:
        raise ValueError("No feature columns selected or available in the DataFrame after processing.")
        
    cols_to_scale = feature_cols + ([target_col_actual] if target_col_actual not in feature_cols else [])
    df_for_scaling = df[cols_to_scale].copy()
    df_for_scaling = df_for_scaling.fillna(method='ffill').fillna(method='bfill')

    # --- Scaling ---
    scalers = {} 
    scaled_data_df = pd.DataFrame(index=df_for_scaling.index) 

    # Scale features
    FeatureScalerClass = StandardScaler if standardize_features else MinMaxScaler
    print(f"Using {FeatureScalerClass.__name__} for feature scaling.")
    for col in feature_cols:
        feature_scaler = FeatureScalerClass() if standardize_features else FeatureScalerClass(feature_range=(0, 1))
        values = df_for_scaling[col].values.reshape(-1, 1)
        scaled_data_df[col] = feature_scaler.fit_transform(values).flatten()
        scalers[col] = feature_scaler

    # Scale target
    if target_col_actual not in df_for_scaling.columns:
        raise ValueError(f"Final target column '{target_col_actual}' not found for scaling.")

    # If power transform was applied, target is always standardized.
    # Otherwise, target scaler type depends on 'standardize_features' flag.
    if power_transform_details['applied']:
        TargetScalerClass = StandardScaler
        print(f"Using StandardScaler for Yeo-Johnson transformed target '{target_col_actual}'.")
    else:
        TargetScalerClass = StandardScaler if standardize_features else MinMaxScaler
        print(f"Using {TargetScalerClass.__name__} for target '{target_col_actual}'.")

    target_scaler = TargetScalerClass() if TargetScalerClass == StandardScaler else TargetScalerClass(feature_range=(0,1))
    target_values_to_scale = df_for_scaling[target_col_actual].values.reshape(-1, 1)
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' BEFORE scaling: "
          f"Mean={np.mean(target_values_to_scale):.4f}, Std={np.std(target_values_to_scale):.4f}, "
          f"Min={np.min(target_values_to_scale):.4f}, Max={np.max(target_values_to_scale):.4f}")
          
    scaled_data_df[target_col_actual] = target_scaler.fit_transform(target_values_to_scale).flatten()
    scalers[target_col_actual] = target_scaler 
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' AFTER scaling: "
          f"Mean={np.mean(scaled_data_df[target_col_actual]):.4f}, Std={np.std(scaled_data_df[target_col_actual]):.4f}, "
          f"Min={np.min(scaled_data_df[target_col_actual]):.4f}, Max={np.max(scaled_data_df[target_col_actual]):.4f}")
    if isinstance(target_scaler, StandardScaler):
        print(f"DEBUG [PREPARE DATA]: Target Scaler learned mean: {target_scaler.mean_[0]:.4f}, scale (std): {target_scaler.scale_[0]:.4f}")

    # --- Create Sequences ---
    X, y_seq = [], [] # Renamed y to y_seq to avoid conflict with general y variable if any
    if len(scaled_data_df) <= window_size:
        raise ValueError(f"Data length ({len(scaled_data_df)}) is <= window_size ({window_size}). Cannot create sequences.")

    for i in range(len(scaled_data_df) - window_size):
        X.append(scaled_data_df.iloc[i:i+window_size][feature_cols].values)
        y_seq.append(scaled_data_df[target_col_actual].iloc[i+window_size])
    
    X = np.array(X)
    y_seq = np.array(y_seq).reshape(-1, 1) 
    
    # --- Train/Validation/Test Split ---
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_seq, test_size=test_size, shuffle=False)
    
    # Calculate val_size for the second split, ensuring it's a proportion of the X_temp set
    actual_val_size_for_split = val_size / (1 - test_size) if (1 - test_size) > 0 else 0 
    
    if len(X_temp) == 0: # No data left for training/validation
        X_train, X_val, y_train, y_val = np.array([]), np.array([]), np.array([]), np.array([])
        print("Warning: No data available for training/validation after initial test split.")
    elif actual_val_size_for_split >= 1.0 or actual_val_size_for_split <= 0 or len(X_temp) < 2: 
        print(f"Warning: Adjusted validation size ({actual_val_size_for_split:.2f}) is invalid or insufficient data in X_temp ({len(X_temp)} samples).")
        if len(X_temp) >= 2 and actual_val_size_for_split > 0 and actual_val_size_for_split < 1:
             # Try to make a split if possible, even if small
             X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False)
        else: # Use all X_temp for training, empty for validation
             X_train, y_train = X_temp, y_temp
             # Define X_val, y_val shape based on X_train, y_train to avoid errors downstream
             val_shape_x = (0, X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (0, X_train.shape[1]) if X_train.ndim == 2 else (0,)
             val_shape_y = (0, y_train.shape[1]) if y_train.ndim == 2 else (0,)
             X_val, y_val = np.empty(val_shape_x), np.empty(val_shape_y)
             print("Using all remaining data for training, validation set will be empty.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # --- Combined Transformation Info ---
    combined_transform_info = {
        'transforms': [],
        'target_col_original': target_col, 
        'target_col_transformed_final': target_col_actual
        # power_transformer_obj is now in power_transform_details if applied
    }
    if piecewise_transform_details['applied']:
        combined_transform_info['transforms'].append(piecewise_transform_details)
    
    # Append either power or log transform details, not both for the primary target transform
    if power_transform_details['applied']:
        # Store the actual transformer object for inverse transform later
        scalers['power_transformer_object_for_target'] = power_transform_details.pop('power_transformer_obj') 
        combined_transform_info['transforms'].append(power_transform_details)
    elif log_transform_details['applied']:
        combined_transform_info['transforms'].append(log_transform_details)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
