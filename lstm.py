import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_weather_data(df, target_col='Temperature', window_size=12, test_size=0.2, val_size=0.25):
    """
    Prepare weather time series data for LSTM training
    
    Args:
        file_path: Path to the CSV file
        target_col: Column to predict (e.g., 'Temperature', 'Humidity', etc.)
        window_size: Size of the sliding window for sequence creation (12 steps = ~1 hour)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols
    """
    # Read the data

    
    # Sort by UNIXTime (ascending) to ensure chronological order
    df = df.sort_values('UNIXTime')
    
    # Select numerical features
    feature_cols = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 
                    'WindDirection(Degrees)', 'Speed']
    
    # Handle NaN values
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Initialize scalers dictionary
    scalers = {}
    scaled_data = pd.DataFrame()
    
    # Normalize each feature individually
    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
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
    
    # No need to reshape X as it's already in [samples, time steps, features] format
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols

# Define the LSTM model
class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(WeatherLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001, patience=10):
    """Train the LSTM model with early stopping"""
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    
    # Store loss history
    train_loss_history = []
    val_loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    

    
    return model, train_loss_history, val_loss_history

# Function to make predictions
def predict(model, X_test, scaler, target_col):
    """Make predictions and inverse transform them to the original scale"""
    
    # Convert to tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    
    # Inverse transform
    predictions = scaler[target_col].inverse_transform(predictions)
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Parameters
    file_path = 'weather_data.csv'
    target_col = 'Temperature'  # Column to predict
    window_size = 12  # Use 12 previous time steps (~1 hour) to predict the next value
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols = prepare_weather_data(
        file_path, target_col=target_col, window_size=window_size
    )
    
    # Model parameters
    input_dim = len(feature_cols)  # Number of features
    hidden_dim = 64  # Number of hidden units
    num_layers = 2  # Number of LSTM layers
    output_dim = 1  # Dimension of output (predicting a single value)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
    
    # Train the model
    model = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Make predictions
    predictions = predict(model, X_test, scalers, target_col)
    
    # Inverse transform the true values
    y_test_orig = scalers[target_col].inverse_transform(y_test)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - y_test_orig) ** 2))
    print(f'Root Mean Square Error: {rmse:.4f}')