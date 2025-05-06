import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)  # Additional dropout between FC layers
        
        # Add a second hidden layer for more capacity
        self.fc2 = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
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
            Apply inverse transformations to recover original target values.
            Handles both MinMaxScaler and StandardScaler for the target, and log transform.

            Args:
                y: Transformed target values (numpy array).
                target_scaler: Scaler object (e.g., MinMaxScaler, StandardScaler) used for the target.
                               Should be fitted only on the target column.
                transform_info: Dictionary containing information about transformations applied,
                                including log transformation details.
                                Example:
                                {
                                    'transforms': [
                                        {'applied': True, 'type': 'log', 'offset': 1e-06, 'original_col': 'Radiation'}
                                    ],
                                    'target_col_original': 'Radiation',
                                    'target_col_transformed': 'Radiation_log'
                                }

            Returns:
                Original scale target values (numpy array).
            """
            # Make a copy to avoid modifying the original array
            y_transformed = y.copy()

            # --- 1. Apply Inverse Scaling ---
            if target_scaler is not None:
                # Ensure y_transformed is 1D if it's a column vector, for consistency
                # before reshaping for the scaler.
                if len(y_transformed.shape) > 1 and y_transformed.shape[1] == 1:
                    y_transformed = y_transformed.squeeze(axis=1)

                # The following 'if' block is for a complex case where the target_scaler
                # was fitted on multiple features including the target.
                # Given prepare_weather_data fits scalers individually, target_scaler.n_features_in_
                # for 'Radiation_log' should be 1, so the 'else' block should be taken.
                if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
                    # This part handles scalers fitted on multi-feature arrays.
                    # It reconstructs a dummy array to perform inverse_transform.
                    print("Warning: target_scaler seems to be fitted on multiple features. Ensure 'target_col_original' and feature names are correctly set in transform_info and scaler.")
                    dummy_array = np.zeros((y_transformed.shape[0], target_scaler.n_features_in_))
                    
                    target_col_name_original = transform_info.get('target_col_original', -1) # Fallback to -1 if not found
                    target_idx = -1 # Default to last column

                    if isinstance(target_col_name_original, str) and hasattr(target_scaler, 'feature_names_in_') and target_scaler.feature_names_in_ is not None:
                        try:
                            # Convert feature_names_in_ to a list for robust searching
                            feature_names = list(target_scaler.feature_names_in_)
                            target_idx = feature_names.index(target_col_name_original)
                        except ValueError:
                            print(f"Warning: Target column '{target_col_name_original}' not found in scaler's feature_names_in_: {target_scaler.feature_names_in_}. Defaulting to last column.")
                            target_idx = -1 # Default to last column
                    elif isinstance(target_col_name_original, int):
                        target_idx = target_col_name_original
                    else:
                        print(f"Warning: 'target_col_original' ('{target_col_name_original}') is not a valid string or int index. Defaulting to last column for multi-feature scaler.")
                        target_idx = -1

                    dummy_array[:, target_idx] = y_transformed
                    transformed_dummy = target_scaler.inverse_transform(dummy_array)
                    y_transformed = transformed_dummy[:, target_idx]
                else:
                    # This path is taken if the scaler was fitted only on the target variable (e.g., 'Radiation_log').
                    # This is typical for both MinMaxScaler and StandardScaler when processed individually.
                    # Scaler's inverse_transform expects a 2D array [n_samples, n_features].
                    # If y_transformed is 1D, reshape it to [n_samples, 1].
                    if len(y_transformed.shape) == 1:
                        y_to_unscale = y_transformed.reshape(-1, 1)
                    else: # Should already be 2D if not squeezed from a column vector earlier
                        y_to_unscale = y_transformed
                    
                    y_unscaled = target_scaler.inverse_transform(y_to_unscale)
                    y_transformed = y_unscaled.squeeze() # Squeeze back to 1D if it became [n_samples, 1]

            # --- 2. Apply Additional Inverse Transformations (e.g., Log) ---
            if transform_info is not None and 'transforms' in transform_info:
                # Apply transforms in reverse order (though only 'log' is typical here from transform_info['transforms'])
                for transform_details in transform_info.get('transforms', [])[::-1]:
                    transform_type = transform_details.get('type')
                    
                    if transform_type == 'log' and transform_details.get('applied', False):
                        # Get the offset value (e.g., epsilon used in log(X + offset))
                        # Your transform_info now correctly provides 'offset'
                        offset_val = transform_details.get('offset', 0)

                        # Apply inverse log transformation (np.exp)
                        # Clip input to np.exp to prevent overflow/underflow with very large/small numbers
                        y_exp = np.exp(np.clip(y_transformed, -100, 100)) # Clipping for numerical stability

                        # Subtract the offset if one was used (e.g., for log(X + epsilon))
                        if offset_val != 0: # Check if an offset was specified and is non-zero
                            y_transformed = y_exp - offset_val
                        else:
                            y_transformed = y_exp # No offset to subtract
                        
                        # Final clip to a sensible range for the original data (e.g., radiation >= 0)
                        # This range (0 to 2000) should be based on your domain knowledge for 'Radiation'.
                        y_transformed = np.clip(y_transformed, 0, 2000) 
            
            return y_transformed
        
    def evaluate(self, X_test_data, y_test_data, device="cpu", 
                    target_scaler_object=None, transform_info_dict=None):
            """
            Evaluates the model and includes diagnostic analysis in the transformed space.
            Args:
                X_test_data (np.array or torch.Tensor): Test features.
                y_test_data (np.array or torch.Tensor): True target values in standardized log space.
                device (str): Computation device ('cpu' or 'cuda').
                target_scaler_object (sklearn.preprocessing.Scaler): Fitted scaler for the target variable (e.g., StandardScaler for 'Radiation_log').
                transform_info_dict (dict): Dictionary with transformation details.
            """
            self.eval()  # Set the model to evaluation mode
            self.to(device)

            # --- 1. Prepare Actuals in Standardized Log Space ---
            # y_test_data are the true target values, already in the standardized log space.
            # Ensure it's a flattened NumPy array.
            if torch.is_tensor(y_test_data): # Convert to numpy if it's a tensor
                actuals_std_log_np = y_test_data.cpu().numpy().flatten()
            else:
                actuals_std_log_np = np.array(y_test_data).flatten()


            # --- 2. Get Model Predictions in Standardized Log Space ---
            # Convert X_test_data to a PyTorch tensor and move to the specified device.
            if isinstance(X_test_data, np.ndarray):
                inputs_tensor = torch.FloatTensor(X_test_data).to(device)
            elif torch.is_tensor(X_test_data):
                inputs_tensor = X_test_data.to(device)
            else:
                raise ValueError("X_test_data must be a NumPy array or a PyTorch tensor.")

            model_predictions_std_log_np = None
            with torch.no_grad():  # Ensure no gradients are computed during evaluation
                outputs_tensor = self(inputs_tensor)  # Get raw model outputs
                # Move predictions to CPU and convert to a flattened NumPy array.
                model_predictions_std_log_np = outputs_tensor.cpu().numpy().flatten()
            
            print(f"DEBUG [EVALUATE]: Shape of actuals_std_log_np: {actuals_std_log_np.shape}")
            print(f"DEBUG [EVALUATE]: Shape of model_predictions_std_log_np: {model_predictions_std_log_np.shape}")
            if actuals_std_log_np.shape != model_predictions_std_log_np.shape:
                print(f"WARNING [EVALUATE]: Shape mismatch between actuals ({actuals_std_log_np.shape}) and predictions ({model_predictions_std_log_np.shape}). This might cause issues.")


            # --- 3. Extract Standard Deviation from the Scaler ---
            # This is std_dev_log, used for estimating the original multiplicative factor C.
            std_dev_of_log_data = None
            if target_scaler_object is not None:
                if isinstance(target_scaler_object, StandardScaler) and hasattr(target_scaler_object, 'scale_'):
                    # For StandardScaler, .scale_ attribute holds the standard deviations.
                    # Assuming it was fitted on a single feature (e.g., 'Radiation_log'), 
                    # it will be the first (and only) element.
                    std_dev_of_log_data = target_scaler_object.scale_[0]
                    print(f"DEBUG [EVALUATE]: Extracted std_dev_log for analysis: {std_dev_of_log_data:.4f}")
                # Check for MinMaxScaler, though it doesn't directly give std_dev, it might be passed mistakenly
                elif isinstance(target_scaler_object, MinMaxScaler):
                    print(f"DEBUG [EVALUATE]: target_scaler_object is a MinMaxScaler. "
                        "Standard deviation for multiplicative factor estimation is typically derived from StandardScaler on log-transformed data.")
                else:
                    print(f"DEBUG [EVALUATE]: target_scaler_object is of type {type(target_scaler_object)}, "
                        "not a fitted StandardScaler with a 'scale_' attribute. "
                        "Cannot extract std_dev_log for C estimation.")
            else:
                print("DEBUG [EVALUATE]: target_scaler_object is None. Cannot extract std_dev_log for C estimation.")

            # --- 4. Call the Diagnostic Analysis Function (Method of this class) ---
            self.analyze_transformed_space_predictions( # Call the method using self
                actuals_std_log_np,          # Input 1: True values in standardized log space
                model_predictions_std_log_np, # Input 2: Model predictions in standardized log space
                std_dev_log=std_dev_of_log_data # Input 3: Standard deviation from the scaler (optional)
            )

            # --- 5. Proceed with Scaled Metrics Calculation (using the same arrays) ---
            print("\n--- Scaled Metrics (Calculated in Evaluate Method) ---")
            # Ensure actuals and predictions are 1D for metric calculations if they expect that
            mse_scaled = np.mean((actuals_std_log_np - model_predictions_std_log_np) ** 2)
            rmse_scaled = np.sqrt(mse_scaled)
            # r2_scaled = r2_score(actuals_std_log_np, model_predictions_std_log_np) # Requires sklearn.metrics
            
            # Calculate MAPE on scaled data (optional, but good for comparison if loss uses it)
            epsilon_mape = 1e-8 # To avoid division by zero
            mape_scaled = np.mean(np.abs((actuals_std_log_np - model_predictions_std_log_np) / (np.abs(actuals_std_log_np) + epsilon_mape))) * 100
            # Cap MAPE for stability if needed, similar to CombinedLoss
            mape_scaled_capped = np.mean(np.clip(np.abs((actuals_std_log_np - model_predictions_std_log_np) / (np.abs(actuals_std_log_np) + epsilon_mape)), 0, 100.0/100.0)) * 100 # Clip at 100%

            print(f"Test RMSE (scaled): {rmse_scaled:.6f}")
            # print(f"Test R² (scaled): {r2_scaled:.6f}") # Uncomment if r2_score is used
            print(f"Test MAPE (scaled): {mape_scaled:.2f}%")
            print(f"Test MAPE (scaled, capped at 100% error per point): {mape_scaled_capped:.2f}%")


            # --- 6. Proceed with Inverse Transformation and Original Scale Metrics ---
            predictions_original_scale = None
            actuals_original_scale = None
            original_scale_metrics_results = {'rmse': np.nan, 'mape': np.nan} # Initialize with NaN

            if target_scaler_object is not None and transform_info_dict is not None:
                print("\n--- Calculating Original Scale Metrics ---")
                # Note: _inverse_transform_target expects NumPy arrays
                predictions_original_scale = self._inverse_transform_target(
                    model_predictions_std_log_np.reshape(-1, 1), # Reshape to column vector
                    target_scaler_object, 
                    transform_info_dict
                ).flatten() # Ensure it's flat for metrics
                actuals_original_scale = self._inverse_transform_target(
                    actuals_std_log_np.reshape(-1, 1), # Reshape to column vector
                    target_scaler_object, 
                    transform_info_dict
                ).flatten() # Ensure it's flat for metrics
                
                mse_original = np.mean((actuals_original_scale - predictions_original_scale) ** 2)
                rmse_original = np.sqrt(mse_original)
                # r2_original = r2_score(actuals_original_scale, predictions_original_scale) # Uncomment if used
                
                # MAPE for original scale
                mape_original = np.mean(np.abs((actuals_original_scale - predictions_original_scale) / (np.abs(actuals_original_scale) + epsilon_mape))) * 100
                mape_original_capped = np.mean(np.clip(np.abs((actuals_original_scale - predictions_original_scale) / (np.abs(actuals_original_scale) + epsilon_mape)), 0, 1.0)) * 100


                print(f"Test RMSE (original scale): {rmse_original:.6f}")
                # print(f"Test R² (original scale): {r2_original:.6f}") # Uncomment if used
                print(f"Test MAPE (original scale): {mape_original:.2f}%")
                print(f"Test MAPE (original scale, capped at 100% error per point): {mape_original_capped:.2f}%")
                print(f"Target scaler mean: {target_scaler_object.mean_[0]:.4f}")
                print(f"Target scaler scale: {target_scaler_object.scale_[0]:.4f}")
                original_scale_metrics_results = {
                    'rmse': rmse_original, 
                    # 'r2': r2_original, # Uncomment if used
                    'mape': mape_original_capped
                }
            else:
                print("Skipping original scale metrics: target_scaler_object or transform_info_dict is None.")

            # Return values as appropriate for your application
            return (model_predictions_std_log_np, actuals_std_log_np, 
                    predictions_original_scale, actuals_original_scale,
                    {'scaled_rmse': rmse_scaled, 'scaled_mape': mape_scaled_capped, **original_scale_metrics_results})

        
    def analyze_transformed_space_predictions(self, actuals_std_log, predictions_std_log, std_dev_log=None):
        """
        Analyzes model predictions in the standardized log space.

        Args:
            actuals_std_log (np.array): True target values in the standardized log space.
            predictions_std_log (np.array): Model's predictions in the standardized log space.
            std_dev_log (float, optional): The standard deviation used to standardize the 
                                           log-transformed data. Used to estimate the original
                                           multiplicative factor.
        """
        print("\n--- Analysis in Standardized Log Space ---")

        # Ensure inputs are flat numpy arrays for consistent calculations
        actuals_std_log = np.array(actuals_std_log).flatten()
        predictions_std_log = np.array(predictions_std_log).flatten()

        # 1. Calculate Residuals
        residuals_std_log = actuals_std_log - predictions_std_log
        print(f"Number of samples: {len(actuals_std_log)}")

        # 2. Analyze Residuals
        mean_residuals_std_log = np.mean(residuals_std_log)
        std_residuals_std_log = np.std(residuals_std_log)
        
        print(f"Mean of Residuals (actuals - predictions) in Standardized Log Space (K_prime): {mean_residuals_std_log:.4f}")
        print(f"Std Dev of Residuals in Standardized Log Space: {std_residuals_std_log:.4f}")

        if std_dev_log is not None and std_dev_log != 0:
            # K_prime = -log(C) / std_dev_log  => log(C) = -K_prime * std_dev_log => C = exp(-K_prime * std_dev_log)
            # This K_prime is (actual - prediction), so if prediction = actual - K_prime,
            # then the bias K in prediction = actual + K is K = -K_prime.
            # The multiplicative factor C was derived from: Predicted_Original approx C * Actual_Original
            # Which led to: Predicted_Standardized_Log approx True_Standardized_Log + log(C) / std_dev_log
            # So, K_model_bias = log(C) / std_dev_log
            # And K_prime (mean_residuals) = True_Standardized_Log - Predicted_Standardized_Log 
            #                               = True_Standardized_Log - (True_Standardized_Log + K_model_bias)
            #                               = -K_model_bias
            # So, K_model_bias = -mean_residuals_std_log
            # log(C) = K_model_bias * std_dev_log = -mean_residuals_std_log * std_dev_log
            # C = exp(-mean_residuals_std_log * std_dev_log)
            
            estimated_multiplicative_factor_C = np.exp(-mean_residuals_std_log * std_dev_log)
            print(f"Estimated Multiplicative Factor (C) in Original Scale (from mean residual): {estimated_multiplicative_factor_C:.4f}")
            print(f"(This C is such that Predicted_Original approx C * Actual_Original)")
            # The following line was in the original function, kept for context if needed by the user.
            # print(f"The observed slope was 0.079. Does this match C? If not, there might be other non-linearities or issues.")


        # 3. Plot Histogram of Residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals_std_log, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mean_residuals_std_log, color='red', linestyle='dashed', linewidth=2, label=f'Mean Residual: {mean_residuals_std_log:.4f}')
        plt.title('Histogram of Residuals in Standardized Log Space')
        plt.xlabel('Residual (Actual_Std_Log - Predicted_Std_Log)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 4. Plot Predictions vs. Actuals in Standardized Log Space
        plt.figure(figsize=(10, 8))
        plt.scatter(actuals_std_log, predictions_std_log, alpha=0.5, label='Predicted vs. Actual')
        
        # Add y=x line (perfect prediction)
        min_val = min(np.min(actuals_std_log), np.min(predictions_std_log))
        max_val = max(np.max(actuals_std_log), np.max(predictions_std_log))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
        
        # Add y = x - K_prime line (representing the mean bias)
        # K_prime is mean_residuals_std_log
        # So, predictions_std_log = actuals_std_log - mean_residuals_std_log
        plt.plot([min_val, max_val], [min_val - mean_residuals_std_log, max_val - mean_residuals_std_log], 'g:', lw=2, label=f'Observed Trend (y = x - K\') (K\'={mean_residuals_std_log:.4f})')
        
        plt.title('Predictions vs. Actuals in Standardized Log Space')
        plt.xlabel('Actual Standardized Log Values')
        plt.ylabel('Predicted Standardized Log Values')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Ensures a 1:1 aspect ratio for easier visual assessment of slope
        plt.show()

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
            for (inputs,) in loader: # Note: DataLoader returns a tuple, so (inputs,) unpacks it
                inputs = inputs.to(device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # If we have a scaler, transform predictions back to original scale
        if target_scaler is not None:
            # Handle backward compatibility with log_transform_info
            if log_transform_info is not None: # This is the old parameter name
                # Construct the new transform_info structure
                current_transform_info = {
                    'transforms': [
                        {'type': 'log', 'applied': log_transform_info.get('applied', False), 
                         'offset': log_transform_info.get('epsilon', log_transform_info.get('offset', 0))}, # check for epsilon or offset
                        {'type': 'scale', 'applied': True} # Assume scaling was always applied if target_scaler is present
                    ],
                    'target_col_original': -1 # Default or get from log_transform_info if available
                }
            elif hasattr(self, 'transform_info') and self.transform_info is not None: # Use the class attribute
                 current_transform_info = self.transform_info
            else: # Default if no info is provided
                current_transform_info = {
                    'transforms': [
                        {'type': 'scale', 'applied': True}
                    ],
                    'target_col_original': -1
                }
            
            # Use the new inverse transform method
            predictions = self._inverse_transform_target(predictions, target_scaler, current_transform_info)
        
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
        # Ensure train_loss is not zero to avoid division by zero error
        train_loss_safe = [max(t, 1e-9) for t in self.history['train_loss']] # Add small epsilon
        loss_ratio = [v/t for t, v in zip(train_loss_safe, self.history['val_loss'])]
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
        if max(loss_ratio) > 1.5: # Check if max ratio is available and greater than 1.5
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
                'num_layers': self.num_layers,
                'dropout_prob': self.dropout_prob # Save dropout_prob as well
            },
            'history': self.history,
            'transform_info': self.transform_info # Save transform_info
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod # Use classmethod for loading
    def load(cls, path, device="cpu"): # Add cls as first argument
        """
        Load model from file
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls( # Use cls to instantiate the class
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            dropout_prob=config.get('dropout_prob', 0.3) # Load dropout_prob, default if not found
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            model.history = checkpoint['history']

        # Load transform_info if available
        if 'transform_info' in checkpoint:
            model.transform_info = checkpoint['transform_info']
        else: # For backward compatibility if old model was saved without transform_info
            model.transform_info = None 
            
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
            transform_info: Info about transformations applied to the target. 
                            If None, uses self.transform_info.
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
        def enable_dropout(model_to_set): # Renamed to avoid conflict
            for m in model_to_set.modules():
                if isinstance(m, nn.Dropout):
                    m.train() # Activates dropout
        
        enable_dropout(self) # Apply to the current model instance
        
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
        
        # Use self.transform_info if transform_info argument is None
        current_transform_info = transform_info if transform_info is not None else self.transform_info

        # If we have a target scaler, apply inverse transformation to each sample
        if target_scaler is not None:
            # For each MC sample
            for i in range(mc_samples):
                # Use the inverse transform method
                # Assuming output_dim is 1, so we access all_predictions[i, :, 0]
                all_predictions[i, :, 0] = self._inverse_transform_target(
                    all_predictions[i, :, 0], target_scaler, current_transform_info
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
                                         indices=None, max_samples_to_plot=5): # Renamed max_samples to avoid conflict
        """
        Plot predictions with uncertainty bounds
        
        Args:
            X: Input data
            y_true: Ground truth values (optional)
            mc_samples: Number of Monte Carlo samples
            target_scaler: Scaler for inverse transformation
            transform_info: Info about transformations applied to the target.
                            If None, uses self.transform_info.
            figsize: Figure size
            device: Computation device
            alpha: Significance level for confidence intervals
            indices: Specific indices to plot (default: first max_samples_to_plot)
            max_samples_to_plot: Maximum number of samples to plot
            
        Returns:
            Matplotlib figure
        """
        # Use self.transform_info if transform_info argument is None
        current_transform_info = transform_info if transform_info is not None else self.transform_info

        # Get predictions with uncertainty
        uncertainty = self.predict_with_uncertainty(
            X, mc_samples=mc_samples, device=device,
            target_scaler=target_scaler, transform_info=current_transform_info,
            return_samples=True, alpha=alpha
        )
        
        # If indices not specified, use first max_samples_to_plot
        if indices is None:
            n_plot_samples = min(max_samples_to_plot, len(X)) # n_samples was conflicting
            indices_to_plot = np.arange(n_plot_samples) # Renamed indices
        else:
            indices_to_plot = np.array(indices)
            n_plot_samples = len(indices_to_plot)
        
        # Prepare ground truth if available
        if y_true is not None:
            y_true_np = y_true.copy() # Work with a copy
            if not isinstance(y_true_np, np.ndarray): # Ensure it's a numpy array
                 y_true_np = np.array(y_true_np)
            
            if target_scaler is not None:
                # Use the inverse transform method for ground truth
                y_true_np = self._inverse_transform_target(
                    y_true_np.squeeze(), target_scaler, current_transform_info # Squeeze if it's a column vector
                )
        
        # Create figure
        fig, axs = plt.subplots(n_plot_samples, 1, figsize=figsize, squeeze=False)
        
        # For each sample to plot
        for i, idx in enumerate(indices_to_plot):
            ax = axs[i, 0]
            
            # Extract predictions and bounds for this sample
            mean_val = uncertainty['mean'][idx, 0] # Renamed mean to mean_val
            lower_val = uncertainty['lower_ci'][idx, 0] # Renamed lower to lower_val
            upper_val = uncertainty['upper_ci'][idx, 0] # Renamed upper to upper_val
            
            # Get all MC samples for this input
            all_mc_samples = uncertainty['samples'][:, idx, 0] # Renamed all_samples
            
            # Plot MC samples as semi-transparent lines/points
            # Plotting many individual lines can be slow, so plot as points or a few representative lines
            # Here, we plot individual points for a subset of MC samples
            num_mc_to_plot = min(10, mc_samples) # Plot up to 10 individual MC samples
            for j in range(num_mc_to_plot): 
                ax.plot([0], [all_mc_samples[j]], 'o', alpha=0.2, color='gray', markersize=3) # Smaller markersize
            
            # Plot prediction and confidence interval
            ax.errorbar([0], [mean_val], yerr=[[mean_val-lower_val], [upper_val-mean_val]], 
                        fmt='o', color='blue', ecolor='lightblue', 
                        capsize=5, label=f'{int((1-alpha)*100)}% CI Prediction') # More descriptive label
            
            # Plot ground truth if available
            if y_true is not None and idx < len(y_true_np):
                ax.plot([0], [y_true_np[idx]], 'ro', label='Actual')
                
                # Calculate error
                error = abs(mean_val - y_true_np[idx])
                within_ci = lower_val <= y_true_np[idx] <= upper_val
                
                # Add error information
                ax.text(0.02, 0.95, f"Error: {error:.2f}", transform=ax.transAxes, fontsize=9)
                ax.text(0.02, 0.90, f"Actual in CI: {'Yes' if within_ci else 'No'}", 
                        transform=ax.transAxes, fontsize=9,
                        color='green' if within_ci else 'red') # Color code for quick visual
            
            # Calculate prediction interval width
            interval_width = upper_val - lower_val
            ax.text(0.02, 0.85, f"Interval width: {interval_width:.2f}", transform=ax.transAxes, fontsize=9)
            
            ax.set_title(f"Sample Index: {idx}", fontsize=10)
            
            # Adjust y-limits to better fit the data, ensuring some padding
            plot_min_y = min(lower_val, y_true_np[idx] if y_true is not None and idx < len(y_true_np) else lower_val)
            plot_max_y = max(upper_val, y_true_np[idx] if y_true is not None and idx < len(y_true_np) else upper_val)
            padding = 0.2 * (plot_max_y - plot_min_y) if (plot_max_y - plot_min_y) > 1e-6 else 1.0 # Add padding, handle near-zero range
            
            ax.set_ylim([max(0, plot_min_y - padding), plot_max_y + padding]) # Ensure y_min is not negative if data is non-negative
            ax.set_xticks([]) # Remove x-ticks as they are not meaningful here
            ax.set_ylabel("Value", fontsize=9) # Add y-axis label
            
            if i == 0: # Add legend only to the first subplot
                ax.legend(fontsize=8, loc='best')
        
        plt.suptitle(f'Predictions with {int((1-alpha)*100)}% Confidence Intervals', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap with suptitle
        return fig

