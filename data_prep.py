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
                                 use_log_transform=False,
                                 use_power_transform=False,
                                 # --- End Transformation Flags ---
                                 min_radiation_for_log=0.1,
                                 min_radiation_floor=0.0,
                                 # --- Parameters for pre-power-transform clipping of original target ---
                                 clip_original_target_before_transform=False, # Enable this new clipping
                                 original_clip_lower_percentile=5.0, # e.g., 5th percentile of original target
                                 original_clip_upper_percentile=95.0, # e.g., 95th percentile of original target
                                 # --- End of new parameters for pre-transform clipping ---
                                 clip_log_target=False,
                                 log_clip_lower_percentile=1.0,
                                 log_clip_upper_percentile=99.0,
                                 use_piecewise_transform=False, use_solar_elevation=True,
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
    df = df_input.copy()

    # NEW DATA: Standardize column names expected by later parts of the function
    # It's often easier to map new names to expected old names if the logic is complex.
    column_map = {
        'temp': 'Temperature',
        'pressure': 'Pressure',
        'humidity': 'Humidity',
        'wind_speed': 'Speed',
        'GHI': 'Radiation', # Crucial for radiation-specific transforms
        # 'Energy delta[Wh]' will be handled by target_col directly
        # New columns like rain_1h, snow_1h, clouds_all, weather_type will be added later
    }
    df.rename(columns=column_map, inplace=True)

    # NEW DATA: Handle potentially problematic characters in target_col name
    original_target_col_name = target_col # Keep original name for records
    if target_col == 'Energy delta[Wh]':
        target_col = 'Energy_delta_Wh'
        df.rename(columns={'Energy delta[Wh]': target_col}, inplace=True)
        print(f"Renamed target column '{original_target_col_name}' to '{target_col}'")


    # NEW DATA: Parse the primary 'Time' column
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
    elif 'UNIXTime' in df.columns: # Fallback for old format
        df = df.sort_values('UNIXTime').reset_index(drop=True)
    else:
        print("Warning: Neither 'Time' nor 'UNIXTime' column found for sorting.")


    # --- Time Feature Processing ---
    # NEW DATA: The time_to_minutes function is for string times like "HH:MM" or "HH:MM:SS".
    # We will use datetime properties from the parsed 'Time' column primarily.
    def time_to_minutes_from_datetime(dt_obj):
        if pd.isna(dt_obj): return np.nan
        return dt_obj.hour * 60 + dt_obj.minute + dt_obj.second / 60

    if 'Time' in df.columns and isinstance(df['Time'].iloc[0], pd.Timestamp):
        df['TimeMinutes'] = df['Time'].apply(time_to_minutes_from_datetime)
        minutes_in_day = 24 * 60
        df['TimeMinutesSin'] = np.sin(2 * np.pi * df['TimeMinutes'] / minutes_in_day)
        df['TimeMinutesCos'] = np.cos(2 * np.pi * df['TimeMinutes'] / minutes_in_day)

        # NEW DATA: Use provided 'hour' column if available, otherwise derive
        if 'hour' in df_input.columns: # check original df_input for 'hour'
             df['HourOfDay'] = df_input['hour'] # Use original before potential rename
        elif 'TimeMinutes' in df:
             df['HourOfDay'] = df['TimeMinutes'] / 60
        else:
            df['HourOfDay'] = df['Time'].dt.hour

        # NEW DATA: Use provided 'month' column if available, otherwise derive
        if 'month' in df_input.columns: # check original df_input for 'month'
            df['Month'] = df_input['month']
        else:
            df['Month'] = df['Time'].dt.month

    # NEW DATA: Utilize provided dayLength, isSun, SunlightTime/daylength
    if 'dayLength' in df_input.columns:
        df['DaylightMinutes'] = df_input['dayLength']
    # Fallback to old logic if new columns aren't there but TimeSunRise/Set are
    elif 'TimeSunRise' in df.columns and 'TimeSunSet' in df.columns:
        # (Original time_to_minutes for TimeSunRise/Set if they are strings)
        def time_to_minutes_str_parse(time_val): # Simplified from original for this context
            if pd.isna(time_val): return np.nan
            if hasattr(time_val, 'hour'):
                return time_val.hour * 60 + time_val.minute + time_val.second / 60
            elif isinstance(time_val, str):
                if ' ' in time_val: time_part = time_val.split(' ')[-1]
                else: time_part = time_val
                if ':' in time_part:
                    parts = time_part.split(':')
                    if len(parts) >= 2: hours, minutes = map(int, parts[:2])
                    else: return np.nan
                    return hours * 60 + minutes
            return np.nan
        df['SunriseMinutes'] = df['TimeSunRise'].apply(time_to_minutes_str_parse)
        df['SunsetMinutes'] = df['TimeSunSet'].apply(time_to_minutes_str_parse)
        df['DaylightMinutes'] = df['SunsetMinutes'] - df['SunriseMinutes']
        df.loc[df['DaylightMinutes'] < 0, 'DaylightMinutes'] += 1440 # Handles overnight

    if 'DaylightMinutes' in df.columns and 'TimeMinutes' in df.columns:
        if 'isSun' in df_input.columns:
            df['IsDaylight'] = df_input['isSun'].astype(float)
        elif 'SunriseMinutes' in df.columns and 'SunsetMinutes' in df.columns: # Fallback
             df['IsDaylight'] = ((df['TimeMinutes'] >= df['SunriseMinutes']) & \
                                 (df['TimeMinutes'] <= df['SunsetMinutes'])).astype(float)

        # TimeSinceSunrise, TimeUntilSunset might be harder to get accurately without explicit rise/set times
        # If 'SunriseMinutes' and 'SunsetMinutes' were not calculated above, these might be NaN or incorrect.
        if 'SunriseMinutes' in df.columns: # Check if calculable
            df['TimeSinceSunrise'] = (df['TimeMinutes'] - df['SunriseMinutes'])
            df.loc[df['TimeSinceSunrise'] < 0, 'TimeSinceSunrise'] += 1440
        if 'SunsetMinutes' in df.columns:
            df['TimeUntilSunset'] = (df['SunsetMinutes'] - df['TimeMinutes'])
            df.loc[df['TimeUntilSunset'] < 0, 'TimeUntilSunset'] += 1440

        if 'SunlightTime/daylength' in df_input.columns:
            df['DaylightPosition'] = df_input['SunlightTime/daylength'].clip(0, 1)
        elif 'TimeSinceSunrise' in df.columns and 'DaylightMinutes' in df.columns and df['DaylightMinutes'].nunique() > 1 : # Fallback
            # Ensure DaylightMinutes is not zero to avoid division by zero
            df['DaylightPosition'] = (df['TimeSinceSunrise'] / df['DaylightMinutes'].replace(0, np.nan)).clip(0, 1)


        if use_solar_elevation and 'DaylightPosition' in df.columns:
            print("Adding solar elevation proxy feature")
            df['SolarElevation'] = 0.0
            # Apply only where DaylightPosition is valid (not NaN)
            valid_idx = df['DaylightPosition'].dropna().index
            df.loc[valid_idx, 'SolarElevation'] = df.loc[valid_idx, 'DaylightPosition'].apply(
                lambda x: math.sin(x * math.pi) if (x >= 0 and x <= 1) else 0
            )
            print(f"SolarElevation created for {len(valid_idx)} rows, {len(valid_idx)/len(df)*100:.1f}% of data")
        elif use_solar_elevation:
            print("Warning: Cannot create SolarElevation. 'DaylightPosition' is missing or could not be calculated.")

    # --- Target Pre-processing and Transformation ---
    # The target_col variable should now be the (potentially renamed) target column name.
    # e.g., 'Energy_delta_Wh' or 'Radiation' (if GHI was target and renamed)

    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying initial min_target_threshold of {min_target_threshold} to original '{target_col}'")
        df[target_col] = df[target_col].clip(lower=min_target_threshold)

    target_col_processing = target_col
    piecewise_transform_details = {'applied': False, 'type': None, 'original_col': None}
    log_transform_details = {'applied': False, 'type': None, 'offset': 0, 'original_col': None, 'clip_bounds': None}
    power_transform_details = {'applied': False, 'type': None, 'lambda': None,
                               'original_col': None, 'scaler_used_after': None,
                               'power_transformer_obj': None, 'original_clip_bounds': None}

    # Step 1: Optional clipping of the original target variable
    # NEW DATA: This logic refers to 'Radiation'. If GHI is target, it's renamed to 'Radiation'.
    # If target is 'Energy_delta_Wh', this specific block might not apply unless you adapt the condition.
    is_radiation_target = (target_col_processing == 'Radiation') # Check if current target is 'Radiation' (i.e. original GHI)

    if clip_original_target_before_transform and is_radiation_target:
        if target_col_processing not in df.columns:
            print(f"Warning: Column '{target_col_processing}' for original clipping not found. Skipping.")
        else:
            original_target_values = df[target_col_processing].values
            lower_bound_orig = np.percentile(original_target_values, original_clip_lower_percentile)
            upper_bound_orig = np.percentile(original_target_values, original_clip_upper_percentile)
            power_transform_details['original_clip_bounds'] = (float(lower_bound_orig), float(upper_bound_orig))
            print(f"Clipping original '{target_col_processing}' to range [{lower_bound_orig:.4f}, {upper_bound_orig:.4f}] BEFORE other transformations.")
            df[target_col_processing] = np.clip(original_target_values, lower_bound_orig, upper_bound_orig)

    # Step 2. Piecewise Transform
    if is_radiation_target and use_piecewise_transform:
        print(f"Applying piecewise radiation transform to '{target_col_processing}' column.")
        df['Radiation_transformed'] = piecewise_radiation_transform(df[target_col_processing].values)
        piecewise_transform_details = {'applied': True, 'type': 'piecewise_radiation', 'original_col': target_col_processing}
        target_col_processing = 'Radiation_transformed'
        print(f"Using '{target_col_processing}' as target for further processing.")

    # Step 3. Power Transform OR Log Transform
    current_target_is_radiation_family_for_transform = target_col_processing == 'Radiation' or \
                                                       target_col_processing == 'Radiation_transformed'

    if use_power_transform and current_target_is_radiation_family_for_transform: # Primarily for Radiation family
        print(f"Applying Yeo-Johnson Power Transform to '{target_col_processing}'")
        if min_radiation_floor is not None:
            print(f"Applying floor of {min_radiation_floor} to '{target_col_processing}' before Power Transform.")
            df[target_col_processing] = np.maximum(df[target_col_processing].astype(float), min_radiation_floor)

        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        values_to_transform = df[target_col_processing].values.reshape(-1, 1)
        transformed_values = power_transformer.fit_transform(values_to_transform)
        col_fed_to_power_transform = target_col_processing
        new_target_col_name = f'{col_fed_to_power_transform}_yeo'
        df[new_target_col_name] = transformed_values.flatten()
        power_transform_details.update({
            'applied': True, 'type': 'yeo-johnson',
            'lambda': power_transformer.lambdas_[0] if power_transformer.lambdas_ is not None else None,
            'original_col': col_fed_to_power_transform,
            'scaler_used_after': 'StandardScaler',
            'power_transformer_obj': power_transformer
        })
        target_col_processing = new_target_col_name
        print(f"Yeo-Johnson applied (lambda={power_transform_details['lambda']:.4f}). New target: '{target_col_processing}'")
        use_log_transform = False

    elif use_log_transform:
        # NEW DATA: Adapt for other potential targets like 'Energy_delta_Wh' or if 'Temperature' (from 'temp') is target
        is_other_loggable_target = target_col_processing in ['Temperature', 'Speed', 'Energy_delta_Wh'] # Add other suitable targets
        eligible_for_log = current_target_is_radiation_family_for_transform or is_other_loggable_target

        if eligible_for_log and target_col_processing in df.columns:
            epsilon = 1e-6
            original_values_for_log = df[target_col_processing].copy()
            log_input_col_name = target_col_processing
            log_output_col_name = f'{log_input_col_name}_log'

            if current_target_is_radiation_family_for_transform: # Specific handling for radiation
                print(f"Applying floor of {min_radiation_for_log} to '{log_input_col_name}' before log transform.")
                floored_values = np.maximum(original_values_for_log, min_radiation_for_log)
                df[log_output_col_name] = np.log(floored_values + epsilon)
            else: # For Temperature, Speed, Energy_delta_Wh, etc.
                # Ensure values are positive before log. Add small epsilon if values can be zero.
                # If values can be negative, log transform is not directly applicable without adjustment (e.g. log(x+abs(min)+epsilon))
                if (original_values_for_log < 0).any():
                    print(f"Warning: Negative values found in '{log_input_col_name}'. Log transform might produce NaNs or errors.")
                    print(f"Applying np.log(np.maximum(original_values_for_log, 0) + epsilon). Review if this is appropriate.")
                    df[log_output_col_name] = np.log(np.maximum(original_values_for_log, 0) + epsilon) # Max with 0 before log
                else:
                    df[log_output_col_name] = np.log(original_values_for_log + epsilon)


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
                print(f"Clipping log-transformed '{target_col_processing}' to range [{lower_bound:.4f}, {upper_bound:.4f}]")
                df[target_col_processing] = np.clip(log_values, lower_bound, upper_bound)
        elif not eligible_for_log:
            print(f"Log transform not applied to '{target_col_processing}' as it's not an eligible type or configuration for log.")

    target_col_actual = target_col_processing # Final name of target before scaling

    # --- Feature Engineering & Selection ---
    # NEW DATA: 'target_col' here is the original name (e.g. Energy_delta_Wh or GHI (if GHI was target, it's now 'Radiation'))
    # low_threshold_col_ref logic
    # If target_col was 'GHI', it was renamed to 'Radiation'. If 'Energy delta[Wh]', it's 'Energy_delta_Wh'.
    # The reference column for '_is_low' should be the state *before* major transformations like log/power.
    # So, we use `target_col` (which is the name after initial renaming but before _log or _yeo).
    
    low_threshold_col_ref_options = ['Radiation', 'Temperature', 'Speed', 'Energy_delta_Wh'] # Add any relevant original scale features
    if target_col in low_threshold_col_ref_options : # Check if the *current* target_col is one of these
        low_threshold_col_for_indicator = target_col # This is the column *before* log/power transforms
        if low_threshold_col_for_indicator in df.columns:
            # Ensure this column is not already transformed to a different scale if using it for thresholding
            # For instance, if target_col is 'Radiation' and it underwent clipping, use the clipped version.
            # If target_col is 'Energy_delta_Wh', it hasn't been transformed by radiation-specific logic.
            values_for_threshold = df[low_threshold_col_for_indicator]
            low_threshold = values_for_threshold.quantile(0.1) # Using the potentially clipped values for quantile
            df[f'{low_threshold_col_for_indicator}_is_low'] = (values_for_threshold < low_threshold).astype(float)
            print(f"Added '{low_threshold_col_for_indicator}_is_low' feature (threshold: {low_threshold:.4f})")


    # NEW DATA: Update base_feature_cols_map with new dataset names
    # 'Radiation' here means GHI if GHI is a feature (even if not the target)
    # 'Temperature' means 'temp', etc.
    base_feature_cols_map = {
        'minimal': ['Radiation', 'Temperature', 'Humidity', 'TimeMinutesSin', 'TimeMinutesCos', 'clouds_all'], # Added clouds_all
        'basic': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'Speed',
                  'TimeMinutesSin', 'TimeMinutesCos', 'clouds_all', 'rain_1h', 'wind_speed'], # Using original wind_speed as Speed
        'all': ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'Speed', # These are the renamed ones
                'rain_1h', 'snow_1h', 'clouds_all', 'weather_type', # NEW DATA columns
                'isSun', 'sunlightTime', 'dayLength', 'SunlightTime/daylength'] # Original names from new dataset
    }
    # Ensure we use original names from df_input for 'all' list where they haven't been mapped by `column_map`
    all_original_cols = list(df_input.columns)
    if 'all' == feature_selection_mode:
        # Start with mapped names, then add other relevant ones from original data
        current_all_selection = [col for col in base_feature_cols_map['all'] if col in df.columns or col in df_input.columns]
        # Add any other original columns not explicitly listed but present in the remapped df
        for orig_col in df_input.columns:
            mapped_col = column_map.get(orig_col, orig_col) # Get mapped name if exists
            if mapped_col in df.columns and mapped_col not in current_all_selection:
                 current_all_selection.append(mapped_col)
            elif orig_col in df.columns and orig_col not in current_all_selection and orig_col not in column_map.values(): # if it wasn't mapped but exists
                 current_all_selection.append(orig_col)
        base_feature_cols = list(set(current_all_selection)) # Use set to remove duplicates
    else:
         base_feature_cols = [col for col in base_feature_cols_map.get(feature_selection_mode, base_feature_cols_map['all'])
                             if col in df.columns or col in df_input.columns] # Check both df and df_input

    # Handle _is_low feature addition
    # low_indicator_col should use the name of the column it was derived from (e.g. 'Radiation_is_low')
    # target_col is the name of the target *before* log/power transforms
    low_indicator_col_name_base = target_col # e.g. 'Radiation' or 'Energy_delta_Wh'
    low_indicator_col = f'{low_indicator_col_name_base}_is_low'
    if low_indicator_col in df.columns and low_indicator_col not in base_feature_cols:
        base_feature_cols.append(low_indicator_col)

    if 'SolarElevation' in df.columns and use_solar_elevation and 'SolarElevation' not in base_feature_cols:
        base_feature_cols.append('SolarElevation')

    time_features = ['TimeMinutesSin', 'TimeMinutesCos', 'HourOfDay', 'Month', # Added Month
                     'DaylightMinutes', 'IsDaylight', 'DaylightPosition']
    # TimeSinceSunrise, TimeUntilSunset are conditional
    if 'TimeSinceSunrise' in df.columns: time_features.append('TimeSinceSunrise')
    if 'TimeUntilSunset' in df.columns: time_features.append('TimeUntilSunset')


    feature_cols = base_feature_cols.copy()
    for feature in time_features:
        if feature in df.columns and feature not in feature_cols and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
    
    # NEW DATA: Explicitly add new features if 'all' or 'basic' (some already handled by base_feature_cols_map)
    # These are original names from your data (unless mapped by column_map earlier, like GHI to Radiation)
    additional_new_features = ['rain_1h', 'snow_1h', 'clouds_all', 'weather_type']
    # Also add the direct time features if not already captured
    direct_time_features_from_input = ['hour', 'month', 'isSun', 'sunlightTime', 'dayLength', 'SunlightTime/daylength']

    for f in additional_new_features + direct_time_features_from_input:
        col_to_check = column_map.get(f, f) # Check if it was renamed (e.g. GHI -> Radiation)
        if col_to_check in df.columns and col_to_check not in feature_cols:
            feature_cols.append(col_to_check)
        elif f in df_input.columns and f not in feature_cols and f not in df.columns and f not in column_map:
            # If it's in original input, not mapped, not in current df, and not in feature_cols yet
            # This case might be rare if renaming is comprehensive, but as a fallback
            df[f] = df_input[f] # Ensure it's in the working df
            feature_cols.append(f)


    # Remove duplicates that might have been added
    feature_cols = sorted(list(set(feature_cols)))


    if target_col_actual in feature_cols:
        print(f"Warning: Final target column '{target_col_actual}' is also in feature_cols. Removing it from features.")
        feature_cols = [f_col for f_col in feature_cols if f_col != target_col_actual]

    # Ensure all selected features actually exist in the DataFrame at this point
    feature_cols = [f_col for f_col in feature_cols if f_col in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns selected or available in the DataFrame after processing.")
    print(f"Final selected features before scaling ({len(feature_cols)}): {feature_cols}")

    cols_to_scale = feature_cols + ([target_col_actual] if target_col_actual not in feature_cols else [])
    cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    if target_col_actual not in cols_to_scale and target_col_actual in df.columns:
        cols_to_scale.append(target_col_actual)

    if not df[cols_to_scale].isna().all().all():
        df_for_scaling = df[cols_to_scale].copy()
        # NEW DATA: Handle potential non-numeric types before fillna if new cols were added without type conversion
        for col in df_for_scaling.columns:
            if df_for_scaling[col].dtype == 'object':
                print(f"Warning: Column '{col}' is of object type. Attempting to convert to numeric.")
                df_for_scaling[col] = pd.to_numeric(df_for_scaling[col], errors='coerce')
        df_for_scaling = df_for_scaling.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError(f"All values in columns selected for scaling are NaN: {cols_to_scale}")

    # --- Scaling ---
    scalers = {}
    scaled_data_df = pd.DataFrame(index=df_for_scaling.index)

    FeatureScalerClass = StandardScaler if standardize_features else MinMaxScaler
    print(f"Using {FeatureScalerClass.__name__} for feature scaling.")
    for col in feature_cols:
        if col not in df_for_scaling.columns:
            print(f"Warning: Feature column '{col}' not found in df_for_scaling. Skipping scaling.")
            continue
        # NEW DATA: Check for all NaN after potential coercions/fills
        if df_for_scaling[col].isnull().all():
            print(f"Warning: All values in feature column '{col}' are NaN before scaling. Scaled output will be NaN.")
            scaled_data_df[col] = np.nan
            scalers[col] = None # No scaler fitted
            continue

        feature_scaler = FeatureScalerClass() if standardize_features else FeatureScalerClass(feature_range=(0, 1))
        values = df_for_scaling[col].values.reshape(-1, 1)
        scaled_data_df[col] = feature_scaler.fit_transform(values).flatten()
        scalers[col] = feature_scaler

    if target_col_actual not in df_for_scaling.columns:
        raise ValueError(f"Final target column '{target_col_actual}' not found for scaling.")
    if df_for_scaling[target_col_actual].isnull().all():
         raise ValueError(f"All values in target column '{target_col_actual}' are NaN before scaling. Cannot proceed.")


    if power_transform_details['applied']:
        TargetScalerClass = StandardScaler
        print(f"Using StandardScaler for Yeo-Johnson transformed target '{target_col_actual}'.")
    else:
        TargetScalerClass = StandardScaler if standardize_features else MinMaxScaler
        print(f"Using {TargetScalerClass.__name__} for target '{target_col_actual}'.")

    target_scaler = TargetScalerClass() if TargetScalerClass == StandardScaler else TargetScalerClass(feature_range=(0,1))
    target_values_to_scale = df_for_scaling[target_col_actual].values.reshape(-1, 1)

    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' BEFORE scaling: "
          f"Mean={np.nanmean(target_values_to_scale):.4f}, Std={np.nanstd(target_values_to_scale):.4f}, "
          f"Min={np.nanmin(target_values_to_scale):.4f}, Max={np.nanmax(target_values_to_scale):.4f}")

    scaled_data_df[target_col_actual] = target_scaler.fit_transform(target_values_to_scale).flatten()
    scalers[target_col_actual] = target_scaler

    print(f"DEBUG [PREPARE DATA]: Stats for target '{target_col_actual}' AFTER scaling: "
          f"Mean={np.nanmean(scaled_data_df[target_col_actual]):.4f}, Std={np.nanstd(scaled_data_df[target_col_actual]):.4f}, "
          f"Min={np.nanmin(scaled_data_df[target_col_actual]):.4f}, Max={np.nanmax(scaled_data_df[target_col_actual]):.4f}")
    if isinstance(target_scaler, StandardScaler):
        if hasattr(target_scaler, 'mean_') and target_scaler.mean_ is not None:
              print(f"DEBUG [PREPARE DATA]: Target Scaler learned mean: {target_scaler.mean_[0]:.4f}, scale (std): {target_scaler.scale_[0]:.4f}")
        else:
              print(f"DEBUG [PREPARE DATA]: Target Scaler ({TargetScalerClass.__name__}) attributes (mean_, scale_) not found (possibly all NaN input or not fitted).")


    # --- Create Sequences ---
    X, y_seq = [], []
    if len(scaled_data_df) <= window_size:
        if scaled_data_df.empty or scaled_data_df[feature_cols + [target_col_actual]].isna().all().all():
            raise ValueError(f"scaled_data_df is empty or all NaN for selected cols. Data length ({len(scaled_data_df)}) is <= window_size ({window_size}).")
        raise ValueError(f"Data length ({len(scaled_data_df)}) is <= window_size ({window_size}). Cannot create sequences.")

    sequence_input_df = scaled_data_df[feature_cols + [target_col_actual]].dropna()
    if len(sequence_input_df) <= window_size:
          raise ValueError(f"Data length after dropping NaNs ({len(sequence_input_df)}) is <= window_size ({window_size}).")

    for i in range(len(sequence_input_df) - window_size):
        X.append(sequence_input_df.iloc[i:i+window_size][feature_cols].values)
        y_seq.append(sequence_input_df[target_col_actual].iloc[i+window_size])

    if not X:
        print("Warning: No sequences were created. X and y will be empty. Check data length and window size after NaN handling.")
        num_features = len(feature_cols)
        X_train, X_val, X_test = np.empty((0, window_size, num_features)), np.empty((0, window_size, num_features)), np.empty((0, window_size, num_features))
        y_train, y_val, y_test = np.empty((0,1)), np.empty((0,1)), np.empty((0,1))
    else:
        X = np.array(X)
        y_seq = np.array(y_seq).reshape(-1, 1)

        X_temp, X_test, y_temp, y_test = train_test_split(X, y_seq, test_size=test_size, shuffle=False)
        actual_val_size_for_split = val_size / (1 - test_size) if (1 - test_size) > 0 else 0

        if len(X_temp) == 0:
            X_train, X_val, y_train, y_val = np.array([]), np.array([]), np.array([]), np.array([])
            print("Warning: No data available for training/validation after initial test split.")
        elif actual_val_size_for_split >= 1.0 or actual_val_size_for_split <= 0 or len(X_temp) < 2:
            print(f"Warning: Adjusted validation size ({actual_val_size_for_split:.2f}) is invalid or insufficient data in X_temp ({len(X_temp)} samples).")
            if len(X_temp) >= 2 and actual_val_size_for_split > 0 and actual_val_size_for_split < 1: # Enough for a split
                 X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False)
            else: # Not enough for split or val_size is 0
                 X_train, y_train = X_temp, y_temp
                 val_shape_x = (0, X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 and X_train.shape[1]>0 else (0,0,0)
                 if X_train.ndim == 2 : val_shape_x = (0, X_train.shape[1]) if X_train.shape[1]>0 else (0,0) # If X_train somehow became 2D
                 val_shape_y = (0, y_train.shape[1]) if y_train.ndim == 2 else (0,)
                 X_val, y_val = np.empty(val_shape_x), np.empty(val_shape_y)
                 print("Using all remaining data for training, validation set will be empty.")
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=actual_val_size_for_split, shuffle=False)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Final features used for sequences ({len(feature_cols)}): {feature_cols}")


    combined_transform_info = {
        'transforms': [],
        'target_col_original': original_target_col_name, # Use the very original name
        'target_col_processed_name': target_col, # Name after potential rename like Energy delta -> Energy_delta
        'target_col_transformed_final': target_col_actual # Name after log/power transforms, before scaling
    }
    if piecewise_transform_details['applied']:
        combined_transform_info['transforms'].append(piecewise_transform_details)
    if power_transform_details['applied']:
        if 'power_transformer_obj' in power_transform_details:
              scalers['power_transformer_object_for_target'] = power_transform_details.pop('power_transformer_obj')
        combined_transform_info['transforms'].append(power_transform_details)
    elif log_transform_details['applied']: # elif because power and log are mutually exclusive in the current logic
        combined_transform_info['transforms'].append(log_transform_details)

    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, combined_transform_info
