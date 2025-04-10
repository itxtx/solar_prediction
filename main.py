"""
TDMC Solar Prediction - Real-world Implementation Example
This script shows how to use the TDMC model with real solar data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tdmc_model import SolarTDMC  # Import the TDMC model we created

# Load and preprocess solar data
def load_solar_data(filepath):
    """
    Load and preprocess solar irradiance data.
    Expected format: CSV with timestamp, irradiance, and other weather variables
    """
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure no missing values
    df = df.dropna(subset=['irradiance', 'temperature'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Add hour of day feature
    df['hour'] = df['timestamp'].dt.hour
    
    return df

# Train TDMC model with real data
def train_solar_tdmc(data, n_states=5, n_emissions=2):
    """
    Train the TDMC model with real solar data
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame with columns: timestamp, irradiance, temperature
    n_states : int
        Number of hidden states
    n_emissions : int
        Number of emission variables
    
    Returns:
    --------
    model : SolarTDMC
        Trained TDMC model
    """
    # Extract features
    X = data[['irradiance', 'temperature']].values
    timestamps = data['timestamp'].values
    
    # Initialize and train the model
    model = SolarTDMC(n_states=n_states, n_emissions=n_emissions, time_slices=24)
    model.fit(
        X, timestamps, 
        max_iter=100, 
        tol=1e-5,
        state_names=[f'State_{i}' for i in range(n_states)]
    )
    
    return model

# Evaluate model on test data
def evaluate_model(model, test_data):
    """
    Evaluate the TDMC model on test data
    
    Parameters:
    -----------
    model : SolarTDMC
        Trained TDMC model
    test_data : DataFrame
        Test data with columns: timestamp, irradiance, temperature
    
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    # Extract features
    X_test = test_data[['irradiance', 'temperature']].values
    timestamps_test = test_data['timestamp'].values
    
    # Predict states
    predicted_states = model.predict_states(X_test, timestamps_test)
    
    # Add predicted states to the test data
    test_data_with_states = test_data.copy()
    test_data_with_states['predicted_state'] = predicted_states
    
    # Generate one-step ahead forecasts for each point in test data
    forecasts = []
    forecast_times = []
    
    for i in range(len(test_data) - 1):
        # Make forecast for next time step
        forecast, _ = model.forecast(
            X_test[i].reshape(1, -1), 
            timestamps_test[i], 
            forecast_horizon=1
        )
        forecasts.append(forecast[0])
        forecast_times.append(timestamps_test[i+1])
    
    # Convert to numpy array
    forecasts = np.array(forecasts)
    
    # Calculate metrics
    actual = X_test[1:, 0]  # Actual irradiance values
    predicted = forecasts[:, 0]  # Predicted irradiance values
    
    # Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Percentage Error (handle zeros)
    mape_data = np.abs((actual - predicted) / np.maximum(actual, 0.1))
    mape = np.mean(mape_data) * 100
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'timestamp': forecast_times,
        'actual_irradiance': actual,
        'forecast_irradiance': predicted
    })
    
    # Return metrics and forecast dataframe
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'forecast_df': forecast_df,
        'test_data_with_states': test_data_with_states
    }

# Generate solar production forecast
def generate_production_forecast(model, current_data, system_capacity_kw, forecast_days=7):
    """
    Generate solar energy production forecast
    
    Parameters:
    -----------
    model : SolarTDMC
        Trained TDMC model
    current_data : DataFrame
        Current data with latest observations
    system_capacity_kw : float
        Solar system capacity in kW
    forecast_days : int
        Number of days to forecast
    
    Returns:
    --------
    forecast_df : DataFrame
        DataFrame with production forecast
    """
    # Get latest observation
    latest_row = current_data.iloc[-1]
    latest_obs = latest_row[['irradiance', 'temperature']].values.reshape(1, -1)
    latest_time = latest_row['timestamp']
    
    # Generate forecast
    forecast_horizon = 24 * forecast_days
    irradiance_forecast, confidence = model.forecast(
        latest_obs, latest_time, forecast_horizon
    )
    
    # Convert irradiance to energy production (kWh)
    # Simple model: production = irradiance_ratio * capacity * efficiency * time
    efficiency = 0.15  # 15% panel efficiency
    hour_fraction = 1.0  # Assuming hourly data
    
    production_forecast = irradiance_forecast[:, 0] * system_capacity_kw * efficiency * hour_fraction
    
    # Convert negative values to zero (no negative production)
    production_forecast = np.maximum(production_forecast, 0)
    
    # Generate timestamp for each forecast point
    forecast_times = [latest_time + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'timestamp': forecast_times,
        'forecasted_irradiance': irradiance_forecast[:, 0],
        'forecasted_temperature': irradiance_forecast[:, 1],
        'forecasted_production_kwh': production_forecast,
        'lower_bound_production': confidence[0][:, 0] * system_capacity_kw * efficiency * hour_fraction,
        'upper_bound_production': confidence[1][:, 0] * system_capacity_kw * efficiency * hour_fraction
    })
    
    # Add date and hour columns for analysis
    forecast_df['date'] = forecast_df['timestamp'].dt.date
    forecast_df['hour'] = forecast_df['timestamp'].dt.hour
    
    return forecast_df

# Calculate daily production summary
def get_daily_production_summary(forecast_df):
    """
    Calculate daily production summary from forecast
    
    Parameters:
    -----------
    forecast_df : DataFrame
        Forecast dataframe from generate_production_forecast
    
    Returns:
    --------
    daily_summary : DataFrame
        Daily production summary
    """
    # Group by date and sum production
    daily_summary = forecast_df.groupby('date').agg({
        'forecasted_production_kwh': 'sum',
        'lower_bound_production': 'sum',
        'upper_bound_production': 'sum'
    }).reset_index()
    
    return daily_summary

# Visualize forecast
def visualize_forecast(forecast_df, daily_summary):
    """
    Visualize production forecast
    
    Parameters:
    -----------
    forecast_df : DataFrame
        Forecast dataframe
    daily_summary : DataFrame
        Daily production summary
    """
    # Hourly production forecast
    plt.figure(figsize=(14, 6))
    plt.plot(forecast_df['timestamp'], forecast_df['forecasted_production_kwh'], label='Forecasted Production')
    plt.fill_between(
        forecast_df['timestamp'],
        forecast_df['lower_bound_production'],
        forecast_df['upper_bound_production'],
        alpha=0.3, label='95% Confidence Interval'
    )
    plt.title('Hourly Solar Energy Production Forecast')
    plt.xlabel('Date/Time')
    plt.ylabel('Energy Production (kWh)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Daily production summary
    plt.figure(figsize=(10, 6))
    plt.bar(
        daily_summary['date'].astype(str),
        daily_summary['forecasted_production_kwh'],
        yerr=[
            daily_summary['forecasted_production_kwh'] - daily_summary['lower_bound_production'],
            daily_summary['upper_bound_production'] - daily_summary['forecasted_production_kwh']
        ],
        alpha=0.7,
        capsize=5
    )
    plt.title('Daily Solar Energy Production Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Production (kWh)')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Main execution
def main():
    # File paths
    data_file = "solar_data.csv"  # Replace with your data file
    model_save_path = "tdmc_solar_model.npy"
    
    # Load data
    print("Loading solar data...")
    try:
        data = load_solar_data(data_file)
        print(f"Loaded {len(data)} records from {data['timestamp'].min()} to {data['timestamp'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create synthetic data for demo if real data not available
        print("Creating synthetic data for demonstration...")
        from create_synthetic_data import create_synthetic_solar_data
        data = create_synthetic_solar_data(n_days=120, time_points_per_day=24)
    
    # Split data into train and test sets
    train_end_date = data['timestamp'].max() - pd.Timedelta(days=14)
    train_data = data[data['timestamp'] <= train_end_date]
    test_data = data[data['timestamp'] > train_end_date]
    
    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")
    
    # Train model or load existing
    try:
        print("Loading existing model...")
        model = SolarTDMC.load_model(model_save_path)
        print("Model loaded successfully!")
    except:
        print("Training new TDMC model...")
        model = train_solar_tdmc(train_data, n_states=5, n_emissions=2)
        
        # Save model
        print("Saving model...")
        model.save_model(model_save_path)
    
    # Evaluate model
    print("Evaluating model on test data...")
    evaluation = evaluate_model(model, test_data)
    
    print(f"Evaluation metrics:")
    print(f"MSE: {evaluation['mse']:.4f}")
    print(f"MAE: {evaluation['mae']:.4f}")
    print(f"MAPE: {evaluation['mape']:.2f}%")
    
    # Get state characteristics
    state_info = model.get_state_characteristics()
    print("\nState Characteristics:")
    for state, info in state_info.items():
        print(f"- {state}: Mean Irradiance = {info['mean_emissions'][0]:.2f}, "
              f"Mean Temperature = {info['mean_emissions'][1]:.2f}°C")
    
    # Generate production forecast
    print("\nGenerating solar production forecast...")
    # Use last 24 hours of data as current data
    current_data = data.iloc[-24:]
    
    # System capacity
    system_capacity_kw = 10.0  # 10 kW system
    
    # Generate forecast for next 7 days
    forecast_df = generate_production_forecast(
        model, current_data, system_capacity_kw, forecast_days=7
    )
    
    # Get daily production summary
    daily_summary = get_daily_production_summary(forecast_df)
    
    # Display daily summary
    print("\nDaily Production Forecast:")
    for _, row in daily_summary.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        production = row['forecasted_production_kwh']
        lower = row['lower_bound_production']
        upper = row['upper_bound_production']
        print(f"- {date_str}: {production:.2f} kWh (95% CI: {lower:.2f} - {upper:.2f} kWh)")
    
    # Visualize forecast
    print("\nVisualizing forecast...")
    visualize_forecast(forecast_df, daily_summary)
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()