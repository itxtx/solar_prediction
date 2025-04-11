def create_evaluation_dashboard(predictions, actuals, scalers, target_col, 
                           timestamps, figsize=(15, 15), resample_freq='1min'):
    """
    Create a comprehensive evaluation dashboard for time series predictions
    
    Args:
        predictions: Model predictions (numpy array)
        actuals: Actual values (numpy array)
        scalers: Dictionary of scalers used to normalize each feature (optional)
        target_col: Name of the target column for inverse scaling (optional)
        timestamps: Array of timestamps for x-axis if available (optional)
        figsize: Size of the figure (width, height)
        resample_freq: Frequency for resampling time series data (e.g., '1min', '1H', '1D')
                      Set to None to disable resampling
    
    Returns:
        matplotlib.figure.Figure: The dashboard figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from datetime import datetime
    
    # Ensure predictions and actuals are flattened
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    # If scalers and target_col are provided, inverse transform the data
    if scalers is not None and target_col is not None:
        if target_col in scalers:
            predictions = scalers[target_col].inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scalers[target_col].inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    errors = actuals - predictions
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    # Handle potential division by zero in MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(errors / actuals)) * 100  # Mean Absolute Percentage Error
        mape = np.nan_to_num(mape)  # Replace NaN with 0
    r2 = r2_score(actuals, predictions)
    
    # Before plotting, make sure data is sorted by timestamp
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_indices]
    sorted_actuals = actuals[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    # Create pandas DataFrame for easier handling of time series
    # Check if timestamps are Unix timestamps (numeric) and convert if necessary
    if np.issubdtype(sorted_timestamps.dtype, np.number):
        # Convert Unix timestamps to datetime objects
        dt_timestamps = pd.to_datetime(sorted_timestamps, unit='s')
    else:
        # Assuming timestamps are already in datetime format or string format that pandas can parse
        dt_timestamps = pd.to_datetime(sorted_timestamps)
    
    data_df = pd.DataFrame({
        'timestamp': dt_timestamps,
        'actual': sorted_actuals,
        'predicted': sorted_predictions
    })
    
    # Set timestamp as index
    data_df.set_index('timestamp', inplace=True)
    
    # Check for and remove duplicates in the index
    if data_df.index.duplicated().any():
        print(f"Warning: Found {data_df.index.duplicated().sum()} duplicate timestamps. Keeping the first occurrence.")
        data_df = data_df[~data_df.index.duplicated(keep='first')]
    
    # Resample if requested
    if resample_freq is not None:
        print(f"Resampling data to {resample_freq} frequency...")
        
        # For resampling, we'll use mean for aggregation
        resampled_df = data_df.resample(resample_freq).mean()
        
        # Drop NaN values that may be introduced by resampling
        resampled_df.dropna(inplace=True)
        
        print(f"Original data points: {len(data_df)}, After resampling: {len(resampled_df)}")
        
        # Update our data
        data_df = resampled_df
    
    # Extract resampled values
    plot_timestamps = data_df.index
    plot_actuals = data_df['actual'].values
    plot_predictions = data_df['predicted'].values
        
    # Create a figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    
    # 1. Actual vs Predicted Line Plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(plot_timestamps, plot_actuals, label='Actual')
    ax1.scatter(plot_timestamps, plot_predictions, label='Predicted')
    ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel(f'Value ({target_col})' if target_col else 'Value', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Format date axis if we have dates
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    fig.autofmt_xdate()
    
    # Shade the error area between actual and predicted
    #ax1.fill_between(plot_timestamps, plot_actuals, plot_predictions, color='gray', alpha=0.2, label='Error')
    
    # 2. Scatter Plot with Regression Line
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(plot_actuals, plot_predictions, alpha=0.6, edgecolor='k', s=50)
    
    # Add perfect prediction line
    min_val = min(np.min(plot_actuals), np.min(plot_predictions))
    max_val = max(np.max(plot_actuals), np.max(plot_predictions))
    # Add some margin
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(plot_actuals, plot_predictions)
    regression_line = slope * np.array([min_val, max_val]) + intercept
    ax2.plot([min_val, max_val], regression_line, 'g-', label=f'Regression Line (slope={slope:.3f})')
    
    ax2.set_title('Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual', fontsize=12)
    ax2.set_ylabel('Predicted', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add correlation coefficient
    correlation = np.corrcoef(plot_actuals, plot_predictions)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nR²: {r2:.4f}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Error Histogram
    ax3 = fig.add_subplot(gs[1, 1])
    errors_resampled = plot_actuals - plot_predictions
    n, bins, patches = ax3.hist(errors_resampled, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # Add normal distribution curve
    from scipy.stats import norm
    mu, std = norm.fit(errors_resampled)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) * len(errors_resampled) * (xmax - xmin) / 30
    ax3.plot(x, p, 'k--', linewidth=2, label=f'Normal Dist. (μ={mu:.2f}, σ={std:.2f})')
    
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Error', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. Residual Plot
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Calculate z-scores for coloring
    from scipy import stats
    z_scores = np.abs(stats.zscore(errors_resampled))
    outlier_threshold = 2
    
    # Create a colormap
    cm = plt.cm.RdYlGn_r
    scatter = ax4.scatter(plot_predictions, errors_resampled, c=z_scores, 
                         cmap=cm, alpha=0.7, edgecolor='k', s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('|Z-score| of Error', fontsize=10)
    
    # Highlight outliers
    outliers = z_scores > outlier_threshold
    if np.any(outliers):
        ax4.scatter(plot_predictions[outliers], errors_resampled[outliers], 
                   s=80, facecolors='none', edgecolors='red', 
                   label=f'Outliers (|Z| > {outlier_threshold})')
    
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted', fontsize=12)
    ax4.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    if np.any(outliers):
        ax4.legend(fontsize=10)
    
    # 5. Metrics Table with more detailed statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate additional metrics
    mse = mean_squared_error(plot_actuals, plot_predictions)
    rmse = np.sqrt(mse)
    rmse_normalized = rmse / (np.max(plot_actuals) - np.min(plot_actuals))
    
    # Calculate additional percentile-based metrics
    q_errors = np.percentile(np.abs(errors_resampled), [25, 50, 75, 90, 95, 99])
    
    metrics_text = (
        f"Model Evaluation Metrics:\n\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"Normalized RMSE: {rmse_normalized:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
        f"R-squared (R²): {r2:.4f}\n"
        f"Correlation Coefficient: {correlation:.4f}\n\n"
        f"Error Percentiles:\n"
        f"25th: {q_errors[0]:.4f}\n"
        f"50th (Median): {q_errors[1]:.4f}\n"
        f"75th: {q_errors[2]:.4f}\n"
        f"90th: {q_errors[3]:.4f}\n"
        f"95th: {q_errors[4]:.4f}\n"
        f"99th: {q_errors[5]:.4f}\n"
    )
    ax5.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    
    # Add title with target variable info
    if target_col:
        plt.suptitle(f'Prediction Evaluation for {target_col}', 
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        plt.suptitle('Time Series Prediction Evaluation', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Print summary statistics
    print(f"Evaluation Summary:")
    print(f"Number of samples (after resampling): {len(plot_actuals)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    return fig