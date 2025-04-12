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
    Applies different scaling to different radiation ranges
    
    Args:
        radiation_values: NumPy array of radiation values
        
    Returns:
        Transformed values with preserved nighttime accuracy
    """
    transformed = np.zeros_like(radiation_values, dtype=float)
    
    # Night/low radiation (preserve accuracy for values near zero)
    night_mask = radiation_values < 10
    transformed[night_mask] = np.log1p(radiation_values[night_mask])
    
    # Morning/evening radiation (moderate scaling)
    moderate_mask = (radiation_values >= 10) & (radiation_values < 200)
    transformed[moderate_mask] = np.log1p(10) + 0.5 * (radiation_values[moderate_mask] - 10)
    
    # Midday/high radiation (reduced compression)
    high_mask = radiation_values >= 200
    moderate_max = np.log1p(10) + 0.5 * (200 - 10)
    transformed[high_mask] = moderate_max + 0.8 * (radiation_values[high_mask] - 200)
    
    return transformed

def prepare_weather_data(df, target_col, window_size=12, test_size=0.2, val_size=0.25, log_transform=False, 
                       min_target_threshold=None, use_piecewise_transform=False, use_solar_elevation=True,
                       standardize_features=False, feature_selection_mode='all'):
    """
    Prepare weather time series data for LSTM training with enhanced features
    
    Args:
        df: DataFrame with weather data
        target_col: Column to predict (e.g., 'Temperature', 'Humidity', etc.)
        window_size: Size of the sliding window for sequence creation (12 steps = ~1 hour)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        log_transform: Whether to apply log transformation to the target column
        min_target_threshold: Minimum threshold for target values (set very small values to this threshold)
        use_piecewise_transform: Whether to apply piecewise transform for Radiation
        use_solar_elevation: Whether to add solar elevation proxy feature
        standardize_features: Whether to use StandardScaler (True) instead of MinMaxScaler (False)
        feature_selection_mode: 'all', 'basic', or 'minimal' for different feature set sizes
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, log_transform_info
    """
    # Sort by UNIXTime (ascending) to ensure chronological order
    if 'UNIXTime' in df.columns:
        df = df.sort_values('UNIXTime')
    
    # Process time features
    
    # Function to convert time string or time object to minutes since midnight
    def time_to_minutes(time_val):
        if pd.isna(time_val):
            return np.nan
            
        # If it's a datetime.time object
        if hasattr(time_val, 'hour'):
            return time_val.hour * 60 + time_val.minute + time_val.second / 60
            
        # If it's a string
        elif isinstance(time_val, str):
            if ' ' in time_val:  # Format like "23:55:26"
                time_part = time_val.split(' ')[-1]
            else:
                time_part = time_val
            
            # Handle different time formats
            if ':' in time_part:
                parts = time_part.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, parts)
                elif len(parts) == 2:  # HH:MM
                    hours, minutes = map(int, parts)
                    seconds = 0
                else:
                    return np.nan
                
                return hours * 60 + minutes + seconds / 60
            
        # Return NaN for unexpected types
        return np.nan
    
    # Process current time
    if 'Time' in df.columns:
        # Extract time from the Time column
        df['TimeMinutes'] = df['Time'].apply(lambda x: time_to_minutes(x))
        
        # Create cyclical time features
        minutes_in_day = 24 * 60
        df['TimeMinutesSin'] = np.sin(2 * np.pi * df['TimeMinutes'] / minutes_in_day)
        df['TimeMinutesCos'] = np.cos(2 * np.pi * df['TimeMinutes'] / minutes_in_day)

        # Add hour of day feature
        df['HourOfDay'] = df['TimeMinutes'] / 60
        
        # Add indicators for day/night
        if 'SunriseMinutes' in df.columns and 'SunsetMinutes' in df.columns:
            df['IsDaylight'] = ((df['TimeMinutes'] >= df['SunriseMinutes']) & 
                               (df['TimeMinutes'] <= df['SunsetMinutes'])).astype(float)
    
    # Process sunrise and sunset times
    if 'TimeSunRise' in df.columns and 'TimeSunSet' in df.columns:
        # Convert sunrise and sunset times to minutes since midnight
        df['SunriseMinutes'] = df['TimeSunRise'].apply(time_to_minutes)
        df['SunsetMinutes'] = df['TimeSunSet'].apply(time_to_minutes)
        
        # Calculate daylight duration in minutes
        df['DaylightMinutes'] = df['SunsetMinutes'] - df['SunriseMinutes']
        
        # If time goes to next day (negative value), add 24 hours (1440 minutes)
        df.loc[df['DaylightMinutes'] < 0, 'DaylightMinutes'] += 1440
        
        # Calculate time since sunrise and until sunset
        if 'TimeMinutes' in df.columns:
            # Calculate time since sunrise (capped at 0 for before sunrise)
            df['TimeSinceSunrise'] = df['TimeMinutes'] - df['SunriseMinutes']
            df.loc[df['TimeSinceSunrise'] < 0, 'TimeSinceSunrise'] += 1440  # Adjust for overnight
            
            # Calculate time until sunset (capped at 0 for after sunset)
            df['TimeUntilSunset'] = df['SunsetMinutes'] - df['TimeMinutes']
            df.loc[df['TimeUntilSunset'] < 0, 'TimeUntilSunset'] += 1440  # Adjust for overnight
            
            # Calculate normalized position in daylight (0 to 1)
            df['DaylightPosition'] = df['TimeSinceSunrise'] / df['DaylightMinutes']
            df['DaylightPosition'] = df['DaylightPosition'].clip(0, 1)  # Clip to 0-1 range
            
            # Add solar elevation proxy feature (optional)
            if use_solar_elevation and 'Time' in df.columns and 'TimeSunRise' in df.columns and 'TimeSunSet' in df.columns:
                print("Adding solar elevation proxy feature")
                # Format times for solar elevation calculation if needed
                df['SolarElevation'] = 0.0  # Initialize with zeros
                
                # Apply function only to rows where all required columns are not NaN
                valid_idx = df[['Time', 'TimeSunRise', 'TimeSunSet']].dropna().index
                
                # Create SolarElevation from existing features instead of calculating directly
                # This is more efficient and avoids potential format issues
                df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
                    lambda x: math.sin(x * math.pi/2) if (x >= 0 and x <= 1) else 0
                )
                
                print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")
    
    # Apply minimum threshold to target variable if specified
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying minimum threshold of {min_target_threshold} to {target_col}")
        # Count values below threshold
        below_threshold_count = (df[target_col] < min_target_threshold).sum()
        if below_threshold_count > 0:
            print(f"Found {below_threshold_count} values below threshold ({below_threshold_count/len(df)*100:.2f}% of data)")
            df[target_col] = df[target_col].clip(lower=min_target_threshold)
    
    # Apply piecewise transformation to Radiation column if it's the target (optional)
    transform_info = {'applied': False, 'type': None}
    if target_col == 'Radiation' and use_piecewise_transform:
        print("Applying piecewise radiation transform to Radiation data")
        # Create a new column with transformed values
        df['Radiation_transformed'] = piecewise_radiation_transform(df['Radiation'].values)
        transform_info = {'applied': True, 'type': 'piecewise_radiation', 'original_col': 'Radiation'}
    
    # Add indicators for small values to improve prediction
    if target_col in ['Temperature', 'Radiation', 'Speed']:
        low_threshold = df[target_col].quantile(0.1)
        df[f'{target_col}_is_low'] = (df[target_col] < low_threshold).astype(float)
        print(f"Added '{target_col}_is_low' feature (threshold: {low_threshold:.4f})")
    
    # Feature selection based on mode
    if feature_selection_mode == 'minimal':
        # Only essential features
        base_feature_cols = [col for col in ['Radiation', 'Temperature', 'Humidity', 'TimeMinutesSin', 'TimeMinutesCos'] 
                            if col in df.columns]
    elif feature_selection_mode == 'basic':
        # Basic set of features
        base_feature_cols = [col for col in ['Radiation', 'Temperature', 'Pressure', 'Humidity', 
                                           'WindDirection(Degrees)', 'Speed', 'TimeMinutesSin', 'TimeMinutesCos'] 
                            if col in df.columns]
    else:  # 'all'
        # All available base features
        base_feature_cols = [col for col in ['Radiation', 'Temperature', 'Pressure', 'Humidity', 
                                           'WindDirection(Degrees)', 'Speed'] 
                            if col in df.columns]
    
    # Add the low value indicator
    if f'{target_col}_is_low' not in base_feature_cols and f'{target_col}_is_low' in df.columns:
        base_feature_cols.append(f'{target_col}_is_low')
    
    # Add solar elevation if available and enabled
    if 'SolarElevation' in df.columns and use_solar_elevation:
        base_feature_cols.append('SolarElevation')
        print("Added SolarElevation to features")
    
    # Time features to try - always include all time features for consistency with original
    time_features = [
        'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes',
        'TimeSinceSunrise', 'TimeUntilSunset', 'DaylightPosition',
        'TimeMinutesSin', 'TimeMinutesCos', 'HourOfDay', 'IsDaylight'
    ]
    
    # Start with base features
    feature_cols = base_feature_cols.copy()  # Use copy instead of set to maintain order
    
    # Only add time features if they don't have NaN values and not already in base_feature_cols
    for feature in time_features:
        if feature in df.columns and feature not in feature_cols and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
    
    # Make sure all feature columns exist in the DataFrame
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Handle NaN values in the features
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Log transform target if specified
    target_col_actual = target_col
    log_transform_info = {'applied': False, 'epsilon': 0}
    
    # If we're using piecewise transform for Radiation as target, use the transformed column
    if target_col == 'Radiation' and transform_info['applied']:
        target_col_actual = 'Radiation_transformed'
        print(f"Using transformed radiation as target: {target_col_actual}")
    elif log_transform and target_col in ['Temperature', 'Radiation', 'Speed']:
        # Add a small constant to avoid log(0)
        epsilon = 1e-6
        df[f'{target_col}_log'] = np.log(df[target_col] + epsilon)
        # Use the log-transformed column as the target
        target_col_actual = f'{target_col}_log'
        log_transform_info = {'applied': True, 'epsilon': epsilon, 'original_col': target_col}
        print(f"Log-transformed {target_col} -> {target_col_actual}")
    
    # Choose the appropriate scaler based on the standardize_features flag
    ScalerClass = StandardScaler if standardize_features else MinMaxScaler
    scaler_name = "StandardScaler" if standardize_features else "MinMaxScaler"
    print(f"Using {scaler_name} for feature scaling")
    
    # Initialize scalers dictionary
    scalers = {}
    scaled_data = pd.DataFrame()
    
    # Normalize each feature individually
    for col in feature_cols:
        if standardize_features:
            scaler = ScalerClass()
        else:
            scaler = ScalerClass(feature_range=(0, 1))
            
        # Make sure there are no NaNs before scaling
        values = df[col].values.reshape(-1, 1)
        scaled_data[col] = scaler.fit_transform(values).flatten()
        scalers[col] = scaler
    
    # Scale the target column
    if standardize_features:
        target_scaler = ScalerClass()
    else:
        target_scaler = ScalerClass(feature_range=(0, 1))
        
    target_values = df[target_col_actual].values.reshape(-1, 1)
    scaled_data[target_col_actual] = target_scaler.fit_transform(target_values).flatten()
    scalers[target_col_actual] = target_scaler
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        # Use all features for input X
        X.append(scaled_data.iloc[i:i+window_size][feature_cols].values)
        # Use only target column for output y
        y.append(scaled_data[target_col_actual].iloc[i+window_size])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split into train, validation, and test sets (maintaining temporal order)
    # First split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False
    )
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Features used: {feature_cols}")
    
    # Create a combined transform info dictionary
    combined_transform_info = {
        'transforms': [],
        'target_col_original': target_col,
        'target_col_transformed': target_col_actual
    }
    
    if transform_info['applied']:
        combined_transform_info['transforms'].append(transform_info)
        
    if log_transform_info['applied']:
        combined_transform_info['transforms'].append(log_transform_info)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info