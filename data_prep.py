import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    # log1p(x) = log(1+x)
    night_mask = radiation_values < 10
    transformed[night_mask] = np.log1p(radiation_values[night_mask]) 
    
    # Morning/evening radiation (moderate scaling)
    moderate_mask = (radiation_values >= 10) & (radiation_values < 200)
    transformed[moderate_mask] = np.log1p(9.999) + 0.05 * (radiation_values[moderate_mask] - 10) # Adjusted scaling factor
    
    # Midday/high radiation (reduced compression)
    high_mask = radiation_values >= 200
    moderate_max = np.log1p(9.999) + 0.05 * (199.999 - 10) 
    transformed[high_mask] = moderate_max + 0.002 * (radiation_values[high_mask] - 200) # Adjusted scaling factor

    return transformed

def prepare_weather_data(df_input, target_col, window_size=12, test_size=0.2, val_size=0.25, 
                         log_transform=False, 
                         min_radiation_for_log=0.1, # Floor for original radiation before log
                         # --- NEW PARAMETERS FOR CLIPPING LOGGED VALUES ---
                         clip_log_target=False, # Flag to enable/disable clipping of log_target
                         log_clip_lower_percentile=1.0, # e.g., 1st percentile
                         log_clip_upper_percentile=99.0, # e.g., 99th percentile
                         # --- End of new parameters ---
                         use_piecewise_transform=False, use_solar_elevation=True,
                         standardize_features=False, feature_selection_mode='all',
                         min_target_threshold=None): # min_target_threshold for original scale
    """
    Prepare weather time series data for LSTM training with enhanced features.
    Includes flooring for radiation before log, and optional clipping of log-transformed target.
    
    Args:
        df_input: DataFrame with weather data. Will be copied.
        target_col: Column to predict (e.g., 'Temperature', 'Humidity', etc.).
        window_size: Size of the sliding window for sequence creation.
        test_size: Proportion of data to use for testing.
        val_size: Proportion of training data to use for validation.
        log_transform: Whether to apply log transformation to the target column.
        min_radiation_for_log: Floor value for radiation before log transform if target is Radiation.
        clip_log_target: Whether to clip the log-transformed target values.
        log_clip_lower_percentile: Lower percentile for clipping log-transformed target.
        log_clip_upper_percentile: Upper percentile for clipping log-transformed target.
        use_piecewise_transform: Whether to apply piecewise transform for Radiation.
        use_solar_elevation: Whether to add solar elevation proxy feature.
        standardize_features: Whether to use StandardScaler (True) instead of MinMaxScaler (False).
        feature_selection_mode: 'all', 'basic', or 'minimal' for different feature set sizes.
        min_target_threshold: Minimum threshold for target values (applied to original target_col).
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
    """
    df = df_input.copy() # Work on a copy

    # Sort by UNIXTime (ascending) to ensure chronological order
    if 'UNIXTime' in df.columns:
        df = df.sort_values('UNIXTime').reset_index(drop=True)
    
    # --- Time Feature Processing ---
    def time_to_minutes(time_val):
        if pd.isna(time_val): return np.nan
        if hasattr(time_val, 'hour'): # datetime.time object
            return time_val.hour * 60 + time_val.minute + time_val.second / 60
        elif isinstance(time_val, str): # String time
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
            df['DaylightPosition'] = (df['TimeSinceSunrise'] / df['DaylightMinutes']).clip(0, 1)
            
            if use_solar_elevation:
                print("Adding solar elevation proxy feature")
                df['SolarElevation'] = 0.0
                valid_idx = df[['TimeMinutes', 'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes']].dropna().index
                valid_idx = valid_idx.intersection(df[df['DaylightMinutes'] > 0].index)
                df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
                    lambda x: math.sin(x * math.pi) if (x >= 0 and x <= 1) else 0
                )
                print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")

    # --- Target Pre-processing and Transformation ---
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying min_target_threshold of {min_target_threshold} to original '{target_col}'")
        below_threshold_count = (df[target_col] < min_target_threshold).sum()
        if below_threshold_count > 0:
            print(f"Found {below_threshold_count} values in '{target_col}' below threshold ({below_threshold_count/len(df)*100:.2f}% of data). Clipping.")
            df[target_col] = df[target_col].clip(lower=min_target_threshold)

    target_col_processing = target_col # This variable will track the name of the target column as it's transformed
    
    piecewise_transform_details = {'applied': False, 'type': None, 'original_col': None}
    log_transform_details = {'applied': False, 'type': None, 'offset': 0, 'original_col': None, 'clip_bounds': None} # Added clip_bounds

    # 1. Apply Piecewise Transform (if applicable)
    if target_col == 'Radiation' and use_piecewise_transform:
        print(f"Applying piecewise radiation transform to '{target_col}' column.")
        df['Radiation_transformed'] = piecewise_radiation_transform(df[target_col].values)
        piecewise_transform_details = {'applied': True, 'type': 'piecewise_radiation', 'original_col': target_col}
        target_col_processing = 'Radiation_transformed'
        print(f"Using '{target_col_processing}' as target for further processing.")

    # 2. Apply Log Transform (if applicable, to the current state of target_col_processing)
    if log_transform:
        # Determine if the current target_col_processing is eligible for log transform
        # Eligible if it's 'Radiation' or 'Radiation_transformed' (if piecewise was applied to Radiation),
        # or if it's 'Temperature' or 'Speed' (and no piecewise transform made it something else).
        eligible_for_log = False
        is_radiation_family = target_col_processing == 'Radiation' or \
                              (piecewise_transform_details['applied'] and piecewise_transform_details['original_col'] == 'Radiation' and target_col_processing == 'Radiation_transformed')
        is_other_loggable = target_col_processing in ['Temperature', 'Speed'] and not piecewise_transform_details['applied'] # only if not already piecewise transformed

        if is_radiation_family or is_other_loggable:
            eligible_for_log = True

        if eligible_for_log and target_col_processing in df.columns:
            epsilon = 1e-6 
            original_values_for_log = df[target_col_processing].copy()
            log_input_col_name = target_col_processing # The column name that is being log-transformed
            log_output_col_name = f'{log_input_col_name}_log'

            if is_radiation_family:
                print(f"Applying floor of {min_radiation_for_log} to '{log_input_col_name}' before log transform.")
                floored_values = np.maximum(original_values_for_log, min_radiation_for_log)
                df[log_output_col_name] = np.log(floored_values + epsilon) # Assumes min_radiation_for_log is positive
            else: # For Temperature, Speed
                # Ensure values are positive before log.
                df[log_output_col_name] = np.log(np.maximum(original_values_for_log, epsilon) + epsilon) 

            log_transform_details = {
                'applied': True, 'type': 'log', 'offset': epsilon, 
                'original_col': log_input_col_name, # Column that was fed into log
                'clip_bounds': None # Initialize, will be updated if clipping is applied
            }
            target_col_processing = log_output_col_name 
            print(f"Log-transformed '{log_transform_details['original_col']}' -> '{target_col_processing}'")

            # --- NEW: Clipping of log-transformed target values ---
            if clip_log_target:
                log_values = df[target_col_processing].values
                lower_bound = np.percentile(log_values, log_clip_lower_percentile)
                upper_bound = np.percentile(log_values, log_clip_upper_percentile)
                
                log_transform_details['clip_bounds'] = (float(lower_bound), float(upper_bound)) # Store clip bounds

                print(f"Clipping '{target_col_processing}' to range [{lower_bound:.4f}, {upper_bound:.4f}] "
                      f"(based on {log_clip_lower_percentile}th and {log_clip_upper_percentile}th percentiles).")
                df[target_col_processing] = np.clip(log_values, lower_bound, upper_bound)
        
        elif not eligible_for_log:
             print(f"Log transform not applied to '{target_col_processing}' as it's not an eligible type or configuration.")
        elif target_col_processing not in df.columns : # Should not happen if logic is correct
             print(f"Warning: Column '{target_col_processing}' designated for log transform not found. Skipping log transform.")
    
    target_col_actual = target_col_processing # This is the final name of the target column before scaling

    # --- Feature Engineering & Selection ---
    # (Using original target_col for the _is_low feature for consistency)
    if target_col in ['Temperature', 'Radiation', 'Speed']: 
        low_threshold_col_ref = target_col 
        if low_threshold_col_ref in df.columns:
            low_threshold = df[low_threshold_col_ref].quantile(0.1)
            df[f'{low_threshold_col_ref}_is_low'] = (df[low_threshold_col_ref] < low_threshold).astype(float)
            print(f"Added '{low_threshold_col_ref}_is_low' feature (threshold: {low_threshold:.4f} based on original '{low_threshold_col_ref}')")

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
        
    df_for_scaling = df[feature_cols + ([target_col_actual] if target_col_actual not in feature_cols else [])].copy()
    df_for_scaling = df_for_scaling.fillna(method='ffill').fillna(method='bfill')

    # --- Scaling ---
    ScalerClass = StandardScaler if standardize_features else MinMaxScaler
    scaler_name = "StandardScaler" if standardize_features else "MinMaxScaler"
    print(f"Using {scaler_name} for feature and target scaling.")
    
    scalers = {} # Use 'scalers' to match existing return structure
    scaled_data_df = pd.DataFrame(index=df_for_scaling.index) 

    for col in feature_cols:
        scaler = ScalerClass() if standardize_features else ScalerClass(feature_range=(0, 1))
        values = df_for_scaling[col].values.reshape(-1, 1)
        scaled_data_df[col] = scaler.fit_transform(values).flatten()
        scalers[col] = scaler

    if target_col_actual not in df_for_scaling.columns: # Check in df_for_scaling
        raise ValueError(f"Final target column '{target_col_actual}' not found in DataFrame subset for scaling.")

    target_scaler = ScalerClass() if standardize_features else ScalerClass(feature_range=(0, 1))
    target_values_to_scale = df_for_scaling[target_col_actual].values.reshape(-1, 1)
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' BEFORE scaling (after potential log and clip): "
          f"Mean={np.mean(target_values_to_scale):.4f}, Std={np.std(target_values_to_scale):.4f}, "
          f"Min={np.min(target_values_to_scale):.4f}, Max={np.max(target_values_to_scale):.4f}")
          
    scaled_data_df[target_col_actual] = target_scaler.fit_transform(target_values_to_scale).flatten()
    scalers[target_col_actual] = target_scaler # Store the target scaler
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' AFTER scaling: "
          f"Mean={np.mean(scaled_data_df[target_col_actual]):.4f}, Std={np.std(scaled_data_df[target_col_actual]):.4f}, "
          f"Min={np.min(scaled_data_df[target_col_actual]):.4f}, Max={np.max(scaled_data_df[target_col_actual]):.4f}")
    if isinstance(target_scaler, StandardScaler):
        print(f"DEBUG [PREPARE DATA]: Target Scaler learned mean: {target_scaler.mean_[0]:.4f}, scale (std): {target_scaler.scale_[0]:.4f}")

    # --- Create Sequences ---
    X, y = [], []
    if len(scaled_data_df) <= window_size:
        raise ValueError(f"Data length ({len(scaled_data_df)}) is less than or equal to window_size ({window_size}). Cannot create sequences.")

    for i in range(len(scaled_data_df) - window_size):
        X.append(scaled_data_df.iloc[i:i+window_size][feature_cols].values)
        y.append(scaled_data_df[target_col_actual].iloc[i+window_size])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1) 
    
    # --- Train/Validation/Test Split ---
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    actual_val_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0 
    if actual_val_size >= 1.0 or actual_val_size <=0: 
        print(f"Warning: Adjusted validation size ({actual_val_size:.2f}) is invalid or results in no validation samples. Adjusting split.")
        if len(X_temp) < 2 : 
             X_train, X_val, y_train, y_val = X_temp, np.array([]).reshape(0, X_temp.shape[1] if X_temp.ndim > 1 and X_temp.shape[1] > 0 else 0, X_temp.shape[-1] if X_temp.ndim > 2 else 0 ), y_temp, np.array([]).reshape(0,1)
             if len(X_temp) >=2 : X_val = np.array([]).reshape(0, X_temp.shape[1], X_temp.shape[2]) # ensure 3D for val if X_temp is 3D
             else: X_val = np.array([]).reshape(0,0,0) # default for safety
        else: # Split roughly, ensuring train gets at least one sample
             val_test_size_adjusted = 0.5 if len(X_temp) > 1 else 0 
             if len(X_temp) * (1-val_test_size_adjusted) < 1: val_test_size_adjusted = (len(X_temp)-1)/len(X_temp) if len(X_temp)>0 else 0

             X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_test_size_adjusted, shuffle=False) 
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=actual_val_size, shuffle=False)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # --- Combined Transformation Info ---
    combined_transform_info = {
        'transforms': [],
        'target_col_original': target_col, 
        'target_col_transformed_final': target_col_actual # Final name of target col after all transforms
    }
    if piecewise_transform_details['applied']:
        combined_transform_info['transforms'].append(piecewise_transform_details)
    if log_transform_details['applied']:
        combined_transform_info['transforms'].append(log_transform_details)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
