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
    
    def value_aware_combined_loss(self, y_pred, y_true, value_multiplier=0.01):
        """
        A value-aware loss that gives higher weight to larger true values.
        
        Args:
            y_pred: The predicted values
            y_true: The true values
            value_multiplier: Controls how much to scale the weighting by true values
        
        Returns:
            Weighted loss that focuses more on higher radiation values
        """
        # Basic MSE component
        squared_errors = (y_true - y_pred)**2
        
        # Create weights that increase with radiation value
        # This focuses more attention on higher values without sacrificing low values
        value_weights = 1.0 + y_true * value_multiplier
        
        # Weighted MSE
        weighted_mse = torch.mean(value_weights * squared_errors)
        
        # Standard MAPE with epsilon protection
        mape = torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))) * 100
        
        # Combined loss with original weights
        return self.mse_weight * weighted_mse + self.mape_weight * mape / 100.0

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
        self.transform_info = None
        
    def forward(self, x):
        # Use Xavier/Glorot initialization for hidden states (optional)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout after LSTM (reduced dropout rate)
        out = self.dropout1(out)
        
        # First dense layer with batch normalization
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Second hidden layer with batch normalization
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
            learning_rate=0.001, patience=10, factor=0.5, min_lr=1e-6, device="cpu",
            scheduler_type="plateau", T_max=None, weight_decay=1e-5, clip_grad_norm=1.0,
            loss_type="mse", mse_weight=0.7, mape_weight=0.3, value_multiplier=0.01):
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
            loss_type: Type of loss function to use ('mse', 'combined', or 'value_aware')
            mse_weight: Weight for MSE in combined losses
            mape_weight: Weight for MAPE in combined losses
            value_multiplier: Multiplier for value-aware weighting (default: 0.01)
            
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
        
        # Initialize loss function based on user choice
        combined_loss_instance = None
        
        if loss_type.lower() == "mse":
            criterion = nn.MSELoss()
            print("Using MSE Loss")
        elif loss_type.lower() == "combined":
            combined_loss_instance = CombinedLoss(mse_weight=mse_weight, mape_weight=mape_weight)
            criterion = combined_loss_instance
            print(f"Using Combined Loss (MSE weight: {mse_weight}, MAPE weight: {mape_weight})")
        elif loss_type.lower() == "value_aware":
            combined_loss_instance = CombinedLoss(mse_weight=mse_weight, mape_weight=mape_weight)
            print(f"Using Value-Aware Combined Loss (MSE weight: {mse_weight}, MAPE weight: {mape_weight}, value multiplier: {value_multiplier})")
            # We'll handle this case specially in the training loop
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose from 'mse', 'combined', or 'value_aware'")
        
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
                
                # Calculate loss based on loss_type
                if loss_type.lower() == "value_aware":
                    loss = combined_loss_instance.value_aware_combined_loss(outputs, targets, value_multiplier)
                else:
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
                    
                    # Calculate validation loss (use same loss type as training)
                    if loss_type.lower() == "value_aware":
                        loss = combined_loss_instance.value_aware_combined_loss(outputs, targets, value_multiplier)
                    else:
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
    
    def _inverse_transform_target(self, y, target_scaler, transform_info):
        """
        Apply inverse transformations to recover original target values
        
        Args:
            y: Transformed target values
            target_scaler: Scaler used for the target
            transform_info: Combined transform info dictionary
            
        Returns:
            Original scale target values
        """
        # Make a copy to avoid modifying the original
        y_transformed = y.copy()
        
        # Get the list of transforms in reverse order (to undo in reverse)
        transforms = transform_info.get('transforms', [])[::-1]
        
        # Apply inverse transformations in reverse order
        for transform in transforms:
            transform_type = transform.get('type')
            
            if transform_type == 'log' and transform.get('applied', False):
                # Undo log transform
                if transform.get('offset', 0) > 0:
                    # If log1p was used: exp(y) - offset
                    y_transformed = np.exp(y_transformed) - transform.get('offset')
                else:
                    # If simple log was used
                    y_transformed = np.exp(y_transformed)
                    
            elif transform_type == 'scale' and transform.get('applied', False):
                # No direct inverse needed as the scaler will handle this
                pass
        
        # Apply inverse scaling if scaler is provided
        if target_scaler is not None:
            # Create dummy array with the right shape for inverse_transform
            if len(y_transformed.shape) > 1 and y_transformed.shape[1] == 1:
                y_transformed = y_transformed.squeeze(axis=1)
                
            # Check if we need a dummy array (if y is just the target column)
            if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
                # Create dummy array with zeros except for target column
                dummy = np.zeros((y_transformed.shape[0], target_scaler.n_features_in_))
                
                # Find target column index from transform_info
                target_col = transform_info.get('target_col_original', -1)
                if isinstance(target_col, str) and hasattr(target_scaler, 'feature_names_in_'):
                    # If we have column names, find the index
                    try:
                        target_idx = np.where(target_scaler.feature_names_in_ == target_col)[0][0]
                    except:
                        # Default to last column if name not found
                        target_idx = -1
                else:
                    # Default to the provided index or last column
                    target_idx = -1 if isinstance(target_col, str) else target_col
                
                # Place values in the correct column
                dummy[:, target_idx] = y_transformed
                
                # Apply inverse transform
                transformed_dummy = target_scaler.inverse_transform(dummy)
                
                # Extract target column
                y_transformed = transformed_dummy[:, target_idx]
            else:
                # If scaler was fitted only on target, just inverse transform directly
                y_transformed = target_scaler.inverse_transform(
                    y_transformed.reshape(-1, 1)).squeeze()
        
        return y_transformed

    def evaluate(self, X_test, y_test, device="cpu", target_scaler=None, transform_info=None):
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
            # Use transform_info from instance if not provided
            if transform_info is None and hasattr(self, 'transform_info'):
                transform_info = self.transform_info
            elif transform_info is None and hasattr(self, 'log_transform_info'):
                # For backward compatibility
                log_transform_info = self.log_transform_info
                if log_transform_info and log_transform_info.get('applied', False):
                    transform_info = {
                        'transforms': [
                            {'type': 'log', 'applied': True, 'offset': log_transform_info.get('epsilon', 0)},
                            {'type': 'scale', 'applied': True}
                        ],
                        'target_col_original': -1
                    }
                else:
                    transform_info = {
                        'transforms': [
                            {'type': 'scale', 'applied': True}
                        ],
                        'target_col_original': -1
                    }
            
            # Use the new inverse transform method
            predictions_orig = self._inverse_transform_target(predictions, target_scaler, transform_info)
            actuals_orig = self._inverse_transform_target(actuals, target_scaler, transform_info)
            
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
            # Handle backward compatibility with log_transform_info
            if log_transform_info is not None:
                transform_info = {
                    'transforms': [
                        {'type': 'log', 'applied': log_transform_info.get('applied', False), 
                         'offset': log_transform_info.get('epsilon', 0)},
                        {'type': 'scale', 'applied': True}
                    ],
                    'target_col_original': -1
                }
            else:
                transform_info = {
                    'transforms': [
                        {'type': 'scale', 'applied': True}
                    ],
                    'target_col_original': -1
                }
            
            # Use the new inverse transform method
            predictions = self._inverse_transform_target(predictions, target_scaler, transform_info)
        
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
    
    def predict_with_uncertainty(self, X, mc_samples=30, device="cpu", 
                            target_scaler=None, transform_info=None,
                            return_samples=False, alpha=0.05):
        """
        Generate predictions with uncertainty estimates using MC Dropout
        
        Args:
            X: Input data of shape (batch_size, seq_length, input_dim)
            mc_samples: Number of Monte Carlo forward passes
            device: Device for computation
            target_scaler: Scaler for inverse transformation
            transform_info: Info about transformations applied to the target
            return_samples: Whether to return all MC samples
            alpha: Significance level for confidence intervals (default 0.05 for 95% CI)
            
        Returns:
            Dictionary containing:
                - mean: Mean prediction
                - std: Standard deviation of predictions
                - lower_ci: Lower confidence interval bound
                - upper_ci: Upper confidence interval bound
                - samples: All MC samples (if return_samples=True)
        """
        # Convert to tensor if not already
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Move to device
        X = X.to(device)
        
        # Set model to evaluation mode but keep dropout active
        self.eval()
        
        # Enable dropout during inference
        def enable_dropout(model):
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        
        enable_dropout(self)
        
        # Store predictions from multiple forward passes
        all_predictions = []
        
        # Run multiple forward passes
        with torch.no_grad():
            for _ in range(mc_samples):
                # Forward pass with dropout active
                outputs = self(X)
                all_predictions.append(outputs.cpu().numpy())
        
        # Stack predictions along new axis
        # Shape: (mc_samples, batch_size, output_dim)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # If we have a target scaler, apply inverse transformation to each sample
        if target_scaler is not None:
            # For each MC sample
            for i in range(mc_samples):
                # Use the inverse transform method
                all_predictions[i, :, 0] = self._inverse_transform_target(
                    all_predictions[i, :, 0], target_scaler, transform_info
                )
        
        # Calculate statistics across MC samples
        # Mean prediction for each input example
        mean_prediction = np.mean(all_predictions, axis=0)
        
        # Standard deviation for each input example
        std_prediction = np.std(all_predictions, axis=0)
        
        # Confidence intervals (using percentiles for non-parametric intervals)
        lower_percentile = alpha/2 * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_ci = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_ci = np.percentile(all_predictions, upper_percentile, axis=0)
        
        # Prepare return dictionary
        uncertainty_dict = {
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_ci': lower_ci, 
            'upper_ci': upper_ci,
        }
        
        # Add all samples if requested
        if return_samples:
            uncertainty_dict['samples'] = all_predictions
        
        return uncertainty_dict

    def plot_prediction_with_uncertainty(self, X, y_true=None, mc_samples=30, 
                                    target_scaler=None, transform_info=None,
                                    figsize=(12, 8), device="cpu", alpha=0.05,
                                    indices=None, max_samples=5):
        """
        Plot predictions with uncertainty bounds
        
        Args:
            X: Input data
            y_true: Ground truth values (optional)
            mc_samples: Number of Monte Carlo samples
            target_scaler: Scaler for inverse transformation
            transform_info: Info about transformations applied to the target
            figsize: Figure size
            device: Computation device
            alpha: Significance level for confidence intervals
            indices: Specific indices to plot (default: first max_samples)
            max_samples: Maximum number of samples to plot
            
        Returns:
            Matplotlib figure
        """
        # Get predictions with uncertainty
        uncertainty = self.predict_with_uncertainty(
            X, mc_samples=mc_samples, device=device,
            target_scaler=target_scaler, transform_info=transform_info,
            return_samples=True, alpha=alpha
        )
        
        # If indices not specified, use first max_samples
        if indices is None:
            n_samples = min(max_samples, len(X))
            indices = np.arange(n_samples)
        else:
            indices = np.array(indices)
            n_samples = len(indices)
        
        # Prepare ground truth if available
        if y_true is not None:
            if not torch.is_tensor(y_true):
                y_true = torch.tensor(y_true, dtype=torch.float32)
                
            y_true = y_true.numpy()
            
            if target_scaler is not None:
                # Use the inverse transform method for ground truth
                y_true = self._inverse_transform_target(
                    y_true.squeeze(), target_scaler, transform_info
                )
        
        # Create figure
        fig, axs = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)
        
        # For each sample to plot
        for i, idx in enumerate(indices):
            ax = axs[i, 0]
            
            # Extract predictions and bounds for this sample
            mean = uncertainty['mean'][idx, 0]
            lower = uncertainty['lower_ci'][idx, 0]
            upper = uncertainty['upper_ci'][idx, 0]
            
            # Get all MC samples for this input
            all_samples = uncertainty['samples'][:, idx, 0]
            
            # Plot MC samples as semi-transparent lines
            for j in range(min(10, mc_samples)):  # Plot up to 10 individual samples
                ax.plot([0], [all_samples[j]], 'o', alpha=0.3, color='gray')
            
            # Plot prediction and confidence interval
            ax.errorbar([0], [mean], yerr=[[mean-lower], [upper-mean]], 
                    fmt='o', color='blue', ecolor='lightblue', 
                    capsize=5, label='Prediction with 95% CI')
            
            # Plot ground truth if available
            if y_true is not None and idx < len(y_true):
                ax.plot([0], [y_true[idx]], 'ro', label='Actual')
                
                # Calculate error
                error = abs(mean - y_true[idx])
                within_ci = lower <= y_true[idx] <= upper
                
                # Add error information
                ax.text(0.02, 0.95, f"Error: {error:.2f}", transform=ax.transAxes)
                ax.text(0.02, 0.90, f"Within CI: {'Yes' if within_ci else 'No'}", 
                    transform=ax.transAxes)
            
            # Calculate prediction interval width
            interval_width = upper - lower
            ax.text(0.02, 0.85, f"Interval width: {interval_width:.2f}", transform=ax.transAxes)
            
            ax.set_title(f"Sample {idx}")
            ax.set_ylim([max(0, lower - 0.2 * interval_width), upper + 0.2 * interval_width])
            ax.set_xticks([])
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        return fig
