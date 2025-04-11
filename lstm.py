import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def prepare_weather_data(df, target_col='Temperature', window_size=12, test_size=0.02, val_size=0.025):
    """
    Prepare weather time series data for LSTM training with sunrise/sunset features
    
    Args:
        df: DataFrame with weather data
        target_col: Column to predict (e.g., 'Temperature', 'Humidity', etc.)
        window_size: Size of the sliding window for sequence creation (12 steps = ~1 hour)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols
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
    
    # Base numerical features
    base_feature_cols = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 
                   'WindDirection(Degrees)', 'Speed']
    
    # Time features to try
    time_features = [
        'SunriseMinutes', 'SunsetMinutes', 'DaylightMinutes',
        'TimeSinceSunrise', 'TimeUntilSunset', 'DaylightPosition',
        'TimeMinutesSin', 'TimeMinutesCos'
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
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        # Use all features for input X
        X.append(scaled_data.iloc[i:i+window_size].values)
        # Use only target column for output y
        y.append(scaled_data[target_col].iloc[i+window_size])
    
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols


# Define the LSTM model


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
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'lr': []
        }
        
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
        out = self.fc2(out)
        
        return out

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
            learning_rate=0.001, patience=10, factor=0.5, min_lr=1e-6, device="cpu",
            scheduler_type="plateau", T_max=None, weight_decay=1e-5, clip_grad_norm=1.0):
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
            'lr': []
        }
        
        # Prepare data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer with L2 regularization (weight decay)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
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
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(val_loss)
            
            # Store metrics in history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f} - Val RMSE: {val_rmse:.6f}')
            
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
                #print(f"New best model saved with validation loss: {val_loss:.6f}")
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
    
    def evaluate(self, X_test, y_test, device="cpu"):
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
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"Test RMSE: {rmse:.6f}")
        
        return predictions, actuals
    
    def predict(self, X, batch_size=64, device="cpu"):
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
        
        return np.array(predictions)
    
    def plot_training_history(self, figsize=(20, 12), log_scale=True):
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
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
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
        
        # 3. Plot Learning Rate
        ax = axes[2]
        ax.plot(self.history['epochs'], self.history['lr'], 'c-', 
                linewidth=2, marker='d', markersize=6)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for learning rate
        ax.set_yscale('log')
        
        # 4. Plot Train vs Val Loss Ratio (to detect overfitting)
        ax = axes[3]
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
        
        # Add early stopping annotation if training stopped early
        max_epochs = max(self.history['epochs'])
        if len(self.history['epochs']) < max_epochs:  # Check if training stopped before reaching max epochs
            axes[0].annotate('Early Stopping Triggered', 
                          xy=(max_epochs, self.history['train_loss'][-1]),
                          xytext=(max_epochs-10, self.history['train_loss'][-1]*1.5),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                          fontsize=10)
        
        # Print summary statistics
        print(f"Training Summary:")
        print(f"Total Epochs: {max_epochs}")
        print(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_val_loss_epoch})")
        print(f"Best Validation RMSE: {best_rmse:.6f} (Epoch {best_rmse_epoch})")
        print(f"Final Training Loss: {self.history['train_loss'][-1]:.6f}")
        print(f"Final Validation Loss: {self.history['val_loss'][-1]:.6f}")
        print(f"Final Validation RMSE: {self.history['val_rmse'][-1]:.6f}")
        
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
    
    @classmethod
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

    def plot_predictions(self, predictions, actuals, timestamps=None, target_col=None, figsize=(15, 8)):
        """
        Plot predictions against actual values
        
        Args:
            predictions: Model predictions (numpy array)
            actuals: Actual values (numpy array)
            timestamps: Array of timestamps for x-axis if available (optional)
            target_col: Name of the target column for labeling (optional)
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create x-axis values
        if timestamps is not None:
            x_values = timestamps
            x_label = 'Time'
        else:
            x_values = np.arange(len(actuals))
            x_label = 'Data Point'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(x_values, actuals, label='Actual', color='blue', alpha=0.7, linewidth=2)
        
        # Plot predictions
        ax.plot(x_values, predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
        
        # Shade the error area between actual and predicted
        ax.fill_between(x_values, actuals, predictions, color='gray', alpha=0.2, label='Error')
        
        # Add formatting
        ax.set_title(f'Actual vs Predicted {target_col if target_col else "Values"}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f'Value ({target_col})' if target_col else 'Value', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Calculate and display metrics
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 0.1))) * 100
        
        metrics_text = (
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"MAPE: {mape:.2f}%"
        )
        
        # Add metrics to the plot
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Parameters
    file_path = 'weather_data.csv'
    df = pd.read_csv(file_path)
    target_col = 'Temperature'  # Column to predict
    window_size = 12  # Use 12 previous time steps (~1 hour) to predict the next value
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols = prepare_weather_data(
        df, target_col=target_col, window_size=window_size
    )
    
    # Model parameters
    input_dim = len(feature_cols)  # Number of features
    hidden_dim = 128  # Number of hidden units (increased from 64)
    num_layers = 2  # Number of LSTM layers
    output_dim = 1  # Dimension of output (predicting a single value)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
    
    # Train the model with advanced features
    model.fit(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              learning_rate=0.001, patience=10, device=device)
    
    # Visualize training metrics
    fig = model.plot_training_history()
    plt.show()
    
    # Evaluate on test data
    predictions, actuals = model.evaluate(X_test, y_test, device=device)
    
    # Inverse transform to get actual temperature values
    predictions_orig = scalers[target_col].inverse_transform(predictions)
    actuals_orig = scalers[target_col].inverse_transform(actuals)
    
    # Calculate RMSE in original scale
    rmse = np.sqrt(np.mean((predictions_orig - actuals_orig) ** 2))
    print(f"Test RMSE (original scale): {rmse:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_orig, label='Actual')
    plt.plot(predictions_orig, label='Predicted')
    plt.title('Temperature Prediction - Test Set')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save model
    model.save('weather_lstm_model.pt')