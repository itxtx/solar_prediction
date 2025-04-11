import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Custom loss function combining MSE and MAPE
class CombinedLoss(nn.Module):
    """
    Custom loss function that combines MSE and MAPE with improved handling of small values
    """
    def __init__(self, mse_weight=0.7, mape_weight=0.3, epsilon=1e-8, clip_mape=100.0):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight
        self.epsilon = epsilon  # To avoid division by zero
        self.clip_mape = clip_mape  # Maximum value for MAPE to avoid extreme values
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE component
        mse = self.mse_loss(y_pred, y_true)
        
        # MAPE component - with safeguards against zero values and extreme percentages
        abs_percentage_error = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
        
        # Clip extremely high percentage errors to avoid instability
        abs_percentage_error = torch.clamp(abs_percentage_error, max=self.clip_mape)
        
        # Calculate MAPE
        mape = torch.mean(abs_percentage_error) * 100.0
        
        # Combined loss
        return self.mse_weight * mse + self.mape_weight * mape / 100.0

def prepare_weather_data(df, target_col, window_size=12, test_size=0.2, val_size=0.25, log_transform=False, 
                      min_target_threshold=None):
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
            hours, minutes, seconds = map(int, time_part.split(':'))
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
    
    # Apply minimum threshold to target variable if specified (for handling zero/near-zero values)
    if min_target_threshold is not None and target_col in df.columns:
        print(f"Applying minimum threshold of {min_target_threshold} to {target_col}")
        # Count values below threshold
        below_threshold_count = (df[target_col] < min_target_threshold).sum()
        if below_threshold_count > 0:
            print(f"Found {below_threshold_count} values below threshold ({below_threshold_count/len(df)*100:.2f}% of data)")
            df[target_col] = df[target_col].clip(lower=min_target_threshold)
    
    # Add indicators for small values to improve prediction
    if target_col in ['Temperature', 'Radiation', 'Speed']:
        low_threshold = df[target_col].quantile(0.1)
        df[f'{target_col}_is_low'] = (df[target_col] < low_threshold).astype(float)
        print(f"Added '{target_col}_is_low' feature (threshold: {low_threshold:.4f})")
    
    # Base numerical features
    base_feature_cols = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 
                   'WindDirection(Degrees)', 'Speed']
    
    # Add the low value indicator
    if f'{target_col}_is_low' not in base_feature_cols and f'{target_col}_is_low' in df.columns:
        base_feature_cols.append(f'{target_col}_is_low')
    
    # Time features to try
    time_features = [
        'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes',
        'TimeSinceSunrise', 'TimeUntilSunset', 'DaylightPosition',
        'TimeMinutesSin', 'TimeMinutesCos', 'HourOfDay', 'IsDaylight'
    ]
    
    # Start with base features
    feature_cols = base_feature_cols.copy()
    
    # Only add time features if they don't have NaN values
    for feature in time_features:
        if feature in df.columns and df[feature].isna().sum() == 0:
            feature_cols.append(feature)
    
    # Make sure all feature columns exist in the DataFrame
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Handle NaN values in the base features
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Log transform target if specified
    target_col_actual = target_col
    log_transform_info = {'applied': False, 'epsilon': 0}
    
    if log_transform and target_col in ['Temperature', 'Radiation', 'Speed']:
        # Add a small constant to avoid log(0)
        epsilon = 1e-6
        df[f'{target_col}_log'] = np.log(df[target_col] + epsilon)
        # Use the log-transformed column as the target
        target_col_actual = f'{target_col}_log'
        log_transform_info = {'applied': True, 'epsilon': epsilon, 'original_col': target_col}
        print(f"Log-transformed {target_col} -> {target_col_actual}")
    
    # Initialize scalers dictionary
    scalers = {}
    scaled_data = pd.DataFrame()
    
    # Normalize each feature individually - simple approach
    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Make sure there are no NaNs before scaling
        values = df[col].values.reshape(-1, 1)
        scaled_data[col] = scaler.fit_transform(values).flatten()
        scalers[col] = scaler
    
    # Scale the target column
    target_scaler = MinMaxScaler(feature_range=(0, 1))
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, log_transform_info


# Define the improved LSTM model
class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(WeatherLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        
        # LSTM layers with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Additional dropout layer after LSTM for better regularization
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Fully connected layers for better feature extraction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)  # Additional dropout between FC layers
        
        # Add a second hidden layer for more capacity
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        
        # Training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': [],
            'val_mape': [],
            'lr': []
        }
        
        # Store log transform info
        self.log_transform_info = None
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout after LSTM
        out = self.dropout1(out)
        
        # Apply dense layers for better feature extraction
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Second hidden layer
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
            learning_rate=0.001, patience=10, factor=0.5, min_lr=1e-6, device="cpu",
            scheduler_type="plateau", T_max=None, weight_decay=1e-5, clip_grad_norm=1.0,
            use_combined_loss=True, mse_weight=0.7, mape_weight=0.3):
        """
        Complete training method with validation, early stopping, and learning rate scheduling
        Now with training history tracking and regularization techniques
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            patience: Number of epochs with no improvement before early stopping
            factor: Factor by which to reduce learning rate on plateau
            min_lr: Minimum learning rate
            device: Device to train on ('cpu' or 'cuda')
            scheduler_type: Type of learning rate scheduler ('plateau' or 'cosine')
            T_max: Maximum number of iterations for CosineAnnealingLR (defaults to epochs if None)
            weight_decay: L2 regularization strength (default: 1e-5)
            clip_grad_norm: Maximum norm for gradient clipping (default: 1.0)
            use_combined_loss: Whether to use the combined MSE+MAPE loss function
            mse_weight: Weight for MSE in combined loss
            mape_weight: Weight for MAPE in combined loss
            
        Returns:
            self: The trained model
        """
        # Debug the shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Make sure the shapes match on the first dimension
        assert X_train.shape[0] == y_train.shape[0], f"Training data mismatch: X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]}"
        assert X_val.shape[0] == y_val.shape[0], f"Validation data mismatch: X_val has {X_val.shape[0]} samples but y_val has {y_val.shape[0]}"
        
        # Reset training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': [],
            'val_mape': [],
            'lr': []
        }
        
        # Prepare data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer with L2 regularization (weight decay)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Use either MSE Loss or Combined Loss depending on configuration
        if use_combined_loss:
            criterion = CombinedLoss(mse_weight=mse_weight, mape_weight=mape_weight)
            print(f"Using Combined Loss (MSE weight: {mse_weight}, MAPE weight: {mape_weight})")
        else:
            criterion = nn.MSELoss()
            print("Using MSE Loss")
        
        # Print regularization settings
        print(f"Regularization settings:")
        print(f"- Dropout probability: {self.dropout_prob}")
        print(f"- L2 regularization (weight decay): {weight_decay}")
        print(f"- Gradient clipping norm: {clip_grad_norm}")
        
        # Learning rate scheduler
        if scheduler_type.lower() == "plateau":
            print(f"Using ReduceLROnPlateau scheduler")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience//2, 
                min_lr=min_lr, verbose=True
            )
        elif scheduler_type.lower() == "cosine":
            # If T_max is not provided, use the number of epochs
            if T_max is None:
                T_max = epochs
            print(f"Using CosineAnnealingLR scheduler with T_max={T_max}")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # Move model to device
        self.to(device)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                
                # Optimize
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_outputs_all = []
            val_targets_all = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Collect predictions and targets for metrics
                    val_outputs_all.append(outputs.cpu().numpy())
                    val_targets_all.append(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            
            # Combine predictions and targets for metrics calculation
            val_predictions = np.vstack(val_outputs_all)
            val_actuals = np.vstack(val_targets_all)
            
            # Calculate metrics
            val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
            val_r2 = r2_score(val_actuals, val_predictions)
            
            # Calculate MAPE with protection against zero values
            epsilon = 1.0
            val_mape = np.mean(np.abs((val_actuals - val_predictions) / (np.abs(val_actuals) + epsilon))) * 100
            
            # Store metrics in history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2)
            self.history['val_mape'].append(val_mape)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f} - Val RMSE: {val_rmse:.6f} - Val R²: {val_r2:.6f} - Val MAPE: {val_mape:.2f}%')
            
            # Adjust learning rate
            if scheduler_type.lower() == "plateau":
                scheduler.step(val_loss)
            elif scheduler_type.lower() == "cosine":
                scheduler.step()
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Check early stopping condition
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            
        # Save the best model
        torch.save(self.state_dict(), 'best_model.pt')
        print("Training complete. Best model saved.")
        
        return self
    
    def evaluate(self, X_test, y_test, device="cpu", target_scaler=None, log_transform_info=None):
        """
        Evaluate the model on test data
        """
        self.eval()
        self.to(device)
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics on scaled data
        # Use epsilon for MAPE calculation to avoid division by zero
        epsilon = 1e-8
        
        # Calculate MSE and RMSE
        mse_scaled = np.mean((actuals - predictions) ** 2)
        rmse_scaled = np.sqrt(mse_scaled)
        
        # Calculate R²
        r2_scaled = r2_score(actuals, predictions)
        
        # Calculate MAPE with protection against zero values
        # Clip absolute percentage errors to 100% to avoid extreme values
        abs_percentage_errors = np.abs((actuals - predictions) / (np.abs(actuals) + epsilon)) * 100
        abs_percentage_errors = np.clip(abs_percentage_errors, 0, 100)
        mape_scaled = np.mean(abs_percentage_errors)
        
        print(f"Scaled Metrics:")
        print(f"Test RMSE: {rmse_scaled:.6f}")
        print(f"Test R²: {r2_scaled:.6f}")
        print(f"Test MAPE (capped at 100%): {mape_scaled:.2f}%")
        
        # If we have a scaler, calculate metrics on the original scale
        if target_scaler is not None:
            # Inverse transform the predictions and actuals
            if log_transform_info and log_transform_info['applied']:
                # For log-transformed data, first inverse scale, then inverse log
                epsilon = log_transform_info['epsilon']
                predictions_orig = np.exp(target_scaler.inverse_transform(predictions)) - epsilon
                actuals_orig = np.exp(target_scaler.inverse_transform(actuals)) - epsilon
                print(f"Applied inverse log transform with epsilon={epsilon}")
            else:
                # Just inverse scale
                predictions_orig = target_scaler.inverse_transform(predictions)
                actuals_orig = target_scaler.inverse_transform(actuals)
            
            # Calculate metrics on original scale
            mse_orig = np.mean((actuals_orig - predictions_orig) ** 2)
            rmse_orig = np.sqrt(mse_orig)
            r2_orig = r2_score(actuals_orig, predictions_orig)
            
            # Calculate MAPE with protection against zero values
            # Use epsilon and clip values to avoid extreme percentages
            epsilon = 1e-8
            abs_percentage_errors = np.abs((actuals_orig - predictions_orig) / (np.abs(actuals_orig) + epsilon)) * 100
            abs_percentage_errors = np.clip(abs_percentage_errors, 0, 100)  # Cap at 100%
            mape_orig = np.mean(abs_percentage_errors)
            
            print(f"\nOriginal Scale Metrics:")
            print(f"Test RMSE: {rmse_orig:.6f}")
            print(f"Test R²: {r2_orig:.6f}")
            print(f"Test MAPE (capped at 100%): {mape_orig:.2f}%")
            
            return predictions_orig, actuals_orig, (rmse_orig, r2_orig, mape_orig)
        
        return predictions, actuals, (rmse_scaled, r2_scaled, mape_scaled)
        

    def predict(self, X, batch_size=64, device="cpu", target_scaler=None, log_transform_info=None):
        """
        Make predictions on new data
        """
        self.eval()
        self.to(device)
        
        # Convert data to tensor
        tensor_x = torch.FloatTensor(X)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # If we have a scaler, transform predictions back to original scale
        if target_scaler is not None:
            if log_transform_info and log_transform_info['applied']:
                # For log-transformed data, first inverse scale, then inverse log
                epsilon = log_transform_info['epsilon']
                predictions = np.exp(target_scaler.inverse_transform(predictions)) - epsilon
            else:
                # Just inverse scale
                predictions = target_scaler.inverse_transform(predictions)
        
        return predictions
    
    def plot_training_history(self, figsize=(20, 15), log_scale=True):
        """
        Plot the training history metrics
        
        Args:
            figsize: Figure size as (width, height)
            log_scale: Whether to use log scale for loss plots
        
        Returns:
            matplotlib.figure.Figure: The figure containing the plots
        """
        if not self.history['epochs']:
            print("No training history available. Please train the model first.")
            return None
        
        # Create a figure with 3x2 subplots
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Plot training and validation loss
        ax = axes[0]
        ax.plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Training Loss', 
                linewidth=2, marker='o', markersize=4)
        ax.plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Validation Loss', 
                linewidth=2, marker='x', markersize=6)
        
        # Find best validation loss point
        best_val_loss_idx = np.argmin(self.history['val_loss'])
        best_val_loss_epoch = self.history['epochs'][best_val_loss_idx]
        best_val_loss = self.history['val_loss'][best_val_loss_idx]
        
        # Highlight best model
        ax.scatter(best_val_loss_epoch, best_val_loss, s=150, c='green', marker='*', 
                  label=f'Best Model (Epoch {best_val_loss_epoch}, Loss {best_val_loss:.6f})', zorder=10)
        
        # Add gray vertical line at best model
        ax.axvline(x=best_val_loss_epoch, color='gray', linestyle='--', alpha=0.5)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Use log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # 2. Plot RMSE
        ax = axes[1]
        ax.plot(self.history['epochs'], self.history['val_rmse'], 'g-', label='Validation RMSE', 
                linewidth=2, marker='s', markersize=6)
        
        # Find best RMSE point
        best_rmse_idx = np.argmin(self.history['val_rmse'])
        best_rmse_epoch = self.history['epochs'][best_rmse_idx]
        best_rmse = self.history['val_rmse'][best_rmse_idx]
        
        # Highlight best RMSE
        ax.scatter(best_rmse_epoch, best_rmse, s=150, c='purple', marker='*', 
                  label=f'Best RMSE (Epoch {best_rmse_epoch}, RMSE {best_rmse:.6f})', zorder=10)
        
        # Add gray vertical line at best RMSE
        ax.axvline(x=best_rmse_epoch, color='gray', linestyle='--', alpha=0.5)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Validation RMSE Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 3. Plot R²
        ax = axes[2]
        ax.plot(self.history['epochs'], self.history['val_r2'], 'm-', label='Validation R²', 
                linewidth=2, marker='d', markersize=6)
        
        # Find best R² point
        best_r2_idx = np.argmax(self.history['val_r2'])
        best_r2_epoch = self.history['epochs'][best_r2_idx]
        best_r2 = self.history['val_r2'][best_r2_idx]
        
        # Highlight best R²
        ax.scatter(best_r2_epoch, best_r2, s=150, c='orange', marker='*', 
                  label=f'Best R² (Epoch {best_r2_epoch}, R² {best_r2:.6f})', zorder=10)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('Validation R² Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # 4. Plot MAPE
        ax = axes[3]
        ax.plot(self.history['epochs'], self.history['val_mape'], 'r-', label='Validation MAPE (%)', 
                linewidth=2, marker='o', markersize=6)
        
        # Find best MAPE point
        best_mape_idx = np.argmin(self.history['val_mape'])
        best_mape_epoch = self.history['epochs'][best_mape_idx]
        best_mape = self.history['val_mape'][best_mape_idx]
        
        # Highlight best MAPE
        ax.scatter(best_mape_epoch, best_mape, s=150, c='brown', marker='*', 
                  label=f'Best MAPE (Epoch {best_mape_epoch}, MAPE {best_mape:.2f}%)', zorder=10)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Validation MAPE Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 5. Plot Learning Rate
        ax = axes[4]
        ax.plot(self.history['epochs'], self.history['lr'], 'c-', 
                linewidth=2, marker='d', markersize=6)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for learning rate
        ax.set_yscale('log')
        
        # 6. Plot Train vs Val Loss Ratio (to detect overfitting)
        ax = axes[5]
        loss_ratio = [v/t for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        ax.plot(self.history['epochs'], loss_ratio, 'm-', 
                linewidth=2, marker='^', markersize=6)
        
        # Add horizontal line at ratio=1
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Val Loss / Train Loss Ratio', fontsize=12)
        ax.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for interpretation
        if max(loss_ratio) > 1.5:
            ax.text(0.5, 0.9, "Ratio > 1: Potential overfitting", 
                   transform=ax.transAxes, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add a title for the entire figure
        plt.suptitle('LSTM Model Training Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        return fig

    def save(self, path):
        """
        Save model to file
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers
            },
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load(cls, path, device="cpu"):
        """
        Load model from file
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            model.history = checkpoint['history']
        
        model.to(device)
        return model
