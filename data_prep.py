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
    # For radiation = 10, transformed = log1p(10) + 0.5 * (10 - 10) = log1p(10)
    # For radiation = 199, transformed = log1p(10) + 0.5 * (199 - 10) = log1p(10) + 0.5 * 189
    moderate_mask = (radiation_values >= 10) & (radiation_values < 200)
    transformed[moderate_mask] = np.log1p(9.999) + 0.05 * (radiation_values[moderate_mask] - 10) # Adjusted scaling factor
    
    # Midday/high radiation (reduced compression)
    # For radiation = 200
    high_mask = radiation_values >= 200
    # Max value from moderate transform (approx value at just under 200)
    moderate_max = np.log1p(9.999) + 0.05 * (199.999 - 10) 
    transformed[high_mask] = moderate_max + 0.002 * (radiation_values[high_mask] - 200) # Adjusted scaling factor

    return transformed

def prepare_weather_data(df, target_col, window_size=12, test_size=0.2, val_size=0.25, 
                         log_transform=False, min_target_threshold=None, # min_target_threshold is for original scale
                         use_piecewise_transform=False, use_solar_elevation=True,
                         standardize_features=False, feature_selection_mode='all',
                         min_radiation_for_log=0.1): # NEW PARAMETER
    """
    Prepare weather time series data for LSTM training with enhanced features.
    Includes a floor for radiation values before log transformation if target_col is 'Radiation'.
    
    Args:
        df: DataFrame with weather data.
        target_col: Column to predict (e.g., 'Temperature', 'Humidity', etc.).
        window_size: Size of the sliding window for sequence creation.
        test_size: Proportion of data to use for testing.
        val_size: Proportion of training data to use for validation.
        log_transform: Whether to apply log transformation to the target column.
        min_target_threshold: Minimum threshold for target values (applied to original target_col).
        use_piecewise_transform: Whether to apply piecewise transform for Radiation.
        use_solar_elevation: Whether to add solar elevation proxy feature.
        standardize_features: Whether to use StandardScaler (True) instead of MinMaxScaler (False).
        feature_selection_mode: 'all', 'basic', or 'minimal' for different feature set sizes.
        min_radiation_for_log: Floor value for radiation before log transform if target is Radiation.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
    """
    # Sort by UNIXTime (ascending) to ensure chronological order
    if 'UNIXTime' in df.columns:
        df = df.sort_values('UNIXTime').reset_index(drop=True) # Reset index after sorting
    
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
        df.loc[df['DaylightMinutes'] < 0, 'DaylightMinutes'] += 1440 # Adjust for next day

        if 'TimeMinutes' in df.columns:
            df['IsDaylight'] = ((df['TimeMinutes'] >= df['SunriseMinutes']) & \
                               (df['TimeMinutes'] <= df['SunsetMinutes'])).astype(float)
            df['TimeSinceSunrise'] = (df['TimeMinutes'] - df['SunriseMinutes'])
            # Adjust for times before sunrise on the same day or after sunset previous day
            df.loc[df['TimeSinceSunrise'] < 0, 'TimeSinceSunrise'] += 1440 
            
            df['TimeUntilSunset'] = (df['SunsetMinutes'] - df['TimeMinutes'])
             # Adjust for times after sunset or before sunrise next day
            df.loc[df['TimeUntilSunset'] < 0, 'TimeUntilSunset'] += 1440

            df['DaylightPosition'] = (df['TimeSinceSunrise'] / df['DaylightMinutes']).clip(0, 1)
            
            if use_solar_elevation:
                print("Adding solar elevation proxy feature")
                df['SolarElevation'] = 0.0
                valid_idx = df[['TimeMinutes', 'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes']].dropna().index
                # Ensure DaylightMinutes is not zero to avoid division by zero
                valid_idx = valid_idx.intersection(df[df['DaylightMinutes'] > 0].index)

                # Using DaylightPosition (0 to 1 from sunrise to sunset)
                # sin(pi * x) gives a 0-1-0 curve for x in 0-1.
                # sin(pi/2 * x) gives a 0-1 curve for x in 0-1 (quarter sine wave).
                # Let's use sin(pi * x) for a more representative solar arch.
                df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
                    lambda x: math.sin(x * math.pi) if (x >= 0 and x <= 1) else 0
                )
                print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")

    # --- Target Pre-processing and Transformation ---
    # Apply minimum threshold to the original target variable if specified
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying minimum threshold of {min_target_threshold} to original '{target_col}'")
        below_threshold_count = (df[target_col] < min_target_threshold).sum()
        if below_threshold_count > 0:
            print(f"Found {below_threshold_count} values in '{target_col}' below threshold ({below_threshold_count/len(df)*100:.2f}% of data). Clipping.")
            df[target_col] = df[target_col].clip(lower=min_target_threshold)

    # Initialize current name for the target column being processed
    target_col_processing = target_col
    
    # Details for transformations
    piecewise_transform_details = {'applied': False, 'type': None, 'original_col': None}
    log_transform_details = {'applied': False, 'type': None, 'offset': 0, 'original_col': None}

    # 1. Apply Piecewise Transform (if applicable)
    if target_col == 'Radiation' and use_piecewise_transform:
        print(f"Applying piecewise radiation transform to '{target_col}' column.")
        df['Radiation_transformed'] = piecewise_radiation_transform(df[target_col].values)
        piecewise_transform_details = {'applied': True, 'type': 'piecewise_radiation', 'original_col': target_col}
        target_col_processing = 'Radiation_transformed'
        print(f"Using '{target_col_processing}' as target for further processing.")

    # 2. Apply Log Transform (if applicable, to the current state of target_col_processing)
    if log_transform:
        is_radiation_type = (target_col_processing == 'Radiation' or target_col_processing == 'Radiation_transformed')
        # Check original target name for other types if target_col_processing is still target_col
        is_other_loggable_type = (target_col_processing == target_col and target_col in ['Temperature', 'Speed'])

        if (is_radiation_type or is_other_loggable_type) and target_col_processing in df.columns:
            epsilon = 1e-6 # Small constant to avoid log(0) or log of very small numbers
            original_values_for_log = df[target_col_processing].copy()
            log_applied_to_col = target_col_processing 

            if is_radiation_type:
                print(f"Applying floor of {min_radiation_for_log} to '{log_applied_to_col}' before log transform.")
                floored_values = np.maximum(original_values_for_log, min_radiation_for_log)
                # Ensure argument to log is positive
                df[f'{log_applied_to_col}_log'] = np.log(np.maximum(floored_values, epsilon / 10) + epsilon) 
            else: # For Temperature, Speed
                if (original_values_for_log <= 0).any():
                     print(f"Warning: Column '{log_applied_to_col}' contains non-positive values. Clamping to 0 before adding epsilon for log transform.")
                # Ensure argument to log is positive
                df[f'{log_applied_to_col}_log'] = np.log(np.maximum(original_values_for_log, 0) + epsilon)
            
            new_target_col_name = f'{log_applied_to_col}_log'
            log_transform_details = {
                'applied': True, 'type': 'log', 'offset': epsilon, 
                'original_col': log_applied_to_col # The column name that was fed into the log function
            }
            target_col_processing = new_target_col_name 
            print(f"Log-transformed '{log_transform_details['original_col']}' -> '{target_col_processing}'")
        elif target_col_processing not in df.columns:
             print(f"Warning: Column '{target_col_processing}' designated for log transform not found. Skipping log transform.")
        else: # Not eligible or not found
            print(f"Log transform not applied to '{target_col_processing}' as it's not an eligible type (e.g., Humidity) or configuration.")
    
    # Final name of the target column to be scaled and used for y
    target_col_actual = target_col_processing

    # --- Feature Engineering & Selection ---
    if target_col_actual in ['Temperature', 'Radiation', 'Speed', 'Radiation_transformed', f"{target_col}_log", f"Radiation_transformed_log"]: # Check if target_col_actual is one of these
        # Use the state of the target column *before* scaling for this indicator
        # If target_col_actual is already logged, this quantile might be on log scale.
        # It's often better to define 'is_low' based on the original scale or a consistent transformed scale.
        # For simplicity, using target_col_actual here. Consider defining based on 'target_col' or 'target_col_processing' before log.
        low_threshold_col_ref = target_col # Base this on the original target for clearer interpretation
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

    # Add low value indicator for the original target column if created
    low_indicator_col = f'{target_col}_is_low'
    if low_indicator_col in df.columns and low_indicator_col not in base_feature_cols:
        base_feature_cols.append(low_indicator_col)
        
    if 'SolarElevation' in df.columns and use_solar_elevation and 'SolarElevation' not in base_feature_cols:
        base_feature_cols.append('SolarElevation')
        print("Added SolarElevation to features")

    time_features = ['SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes', 'TimeSinceSunrise', 
                     'TimeUntilSunset', 'DaylightPosition', 'TimeMinutesSin', 'TimeMinutesCos', 
                     'HourOfDay', 'IsDaylight']
    
    feature_cols = base_feature_cols.copy()
    for feature in time_features:
        if feature in df.columns and feature not in feature_cols and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
            
    # Ensure target_col_actual is not in feature_cols to prevent data leakage if it shares a base name
    # with an original feature (e.g. if target_col_actual is 'Radiation' and 'Radiation' is also a feature)
    # However, if target_col_actual is 'Radiation_transformed_log', original 'Radiation' can be a feature.
    # This is usually handled by LSTM structure, but good to be mindful.
    # For now, we assume feature_cols are distinct inputs from the target_col_actual.
    if target_col_actual in feature_cols:
        print(f"Warning: Final target column '{target_col_actual}' is also in feature_cols. Removing it from features to avoid direct leakage.")
        feature_cols = [f for f in feature_cols if f != target_col_actual]


    feature_cols = [col for col in feature_cols if col in df.columns] # Final check
    if not feature_cols:
        raise ValueError("No feature columns selected or available in the DataFrame after processing.")
        
    df_for_scaling = df[feature_cols + ([target_col_actual] if target_col_actual not in feature_cols else [])].copy()
    df_for_scaling = df_for_scaling.fillna(method='ffill').fillna(method='bfill') # Fill NaNs in selected features and target

    # --- Scaling ---
    ScalerClass = StandardScaler if standardize_features else MinMaxScaler
    scaler_name = "StandardScaler" if standardize_features else "MinMaxScaler"
    print(f"Using {scaler_name} for feature and target scaling.")
    
    scalers = {}
    scaled_data_df = pd.DataFrame(index=df_for_scaling.index) # Preserve index

    for col in feature_cols:
        scaler = ScalerClass() if standardize_features else ScalerClass(feature_range=(0, 1))
        values = df_for_scaling[col].values.reshape(-1, 1)
        scaled_data_df[col] = scaler.fit_transform(values).flatten()
        scalers[col] = scaler

    if target_col_actual not in df.columns:
        raise ValueError(f"Final target column '{target_col_actual}' not found in DataFrame before scaling. Transformations might have failed or column was dropped.")

    target_scaler = ScalerClass() if standardize_features else ScalerClass(feature_range=(0, 1))
    target_values_to_scale = df_for_scaling[target_col_actual].values.reshape(-1, 1)
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' BEFORE scaling: "
          f"Mean={np.mean(target_values_to_scale):.4f}, Std={np.std(target_values_to_scale):.4f}, "
          f"Min={np.min(target_values_to_scale):.4f}, Max={np.max(target_values_to_scale):.4f}")
          
    scaled_data_df[target_col_actual] = target_scaler.fit_transform(target_values_to_scale).flatten()
    scalers[target_col_actual] = target_scaler
    
    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' AFTER scaling: "
          f"Mean={np.mean(scaled_data_df[target_col_actual]):.4f}, Std={np.std(scaled_data_df[target_col_actual]):.4f}, "
          f"Min={np.min(scaled_data_df[target_col_actual]):.4f}, Max={np.max(scaled_data_df[target_col_actual]):.4f}")

    # --- Create Sequences ---
    X, y = [], []
    # Ensure there's enough data for at least one sequence
    if len(scaled_data_df) <= window_size:
        raise ValueError(f"Data length ({len(scaled_data_df)}) is less than or equal to window_size ({window_size}). Cannot create sequences.")

    for i in range(len(scaled_data_df) - window_size):
        X.append(scaled_data_df.iloc[i:i+window_size][feature_cols].values)
        y.append(scaled_data_df[target_col_actual].iloc[i+window_size])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1) # Ensure y is 2D for scikit-learn/Keras
    
    # --- Train/Validation/Test Split ---
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    # Adjust val_size relative to the new temporary training set size
    actual_val_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0 
    if actual_val_size >= 1.0 or actual_val_size <=0: # val_size could be too large for the remaining data
        print(f"Warning: Adjusted validation size ({actual_val_size:.2f}) is invalid. Splitting temp data 50/50 if possible, or using all for training if too small.")
        if len(X_temp) < 2 : # Not enough data to split
             X_train, X_val, y_train, y_val = X_temp, np.array([]).reshape(0, X_temp.shape[1] if X_temp.ndim > 1 else 0, X_temp.shape[-1]), y_temp, np.array([]).reshape(0,1)
        else:
             X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=min(0.5, len(X_temp)-1 / len(X_temp) ), shuffle=False) # ensure at least 1 sample for train
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
        'target_col_transformed': target_col_actual # Final name of target col after all transforms
    }
    if piecewise_transform_details['applied']:
        combined_transform_info['transforms'].append(piecewise_transform_details)
    if log_transform_details['applied']:
        combined_transform_info['transforms'].append(log_transform_details)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info

