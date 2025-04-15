import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
import time
import pandas as pd

class WeatherGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3, 
                 bidirectional=False):
        super(WeatherGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, 
            output_dim
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Store training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mape': [],
            'val_mape': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Get last time step output
        # Shape: (batch_size, hidden_dim) or (batch_size, hidden_dim*2) if bidirectional
        out = gru_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out
    
    def forecast(self, x, steps=24, batch_size=32, device="cpu", 
                 target_scaler=None, include_history=True):
        """
        Generate multi-step forecasts
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            steps: Number of steps to forecast
            batch_size: Batch size for processing
            device: Device for computation
            target_scaler: Scaler to inverse transform predictions
            include_history: Whether to include input data in the output sequence
            
        Returns:
            Array of forecasted values with shape (batch_size, steps)
        """
        self.eval()  # Set model to evaluation mode
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Move to device
        x = x.to(device)
        
        # Store predictions
        predictions = []
        current_input = x.clone()
        
        with torch.no_grad():
            for _ in range(steps):
                # Get prediction for next step
                output = self.forward(current_input)
                predictions.append(output.cpu().numpy())
                
                # Create next input by shifting window
                if len(current_input.shape) == 3:  # For batched input
                    # Create new last step based on prediction
                    new_step = torch.zeros((current_input.shape[0], 1, current_input.shape[2]), 
                                          device=device)
                    
                    # Place prediction in target variable position (assuming last column)
                    # This assumes the target variable is also present as an input feature
                    new_step[:, 0, -1] = output.squeeze()
                    
                    # Roll window forward
                    current_input = torch.cat([current_input[:, 1:, :], new_step], dim=1)
                else:
                    # For single sample, similar logic but without batch dimension
                    new_step = torch.zeros((1, current_input.shape[1]), device=device)
                    new_step[0, -1] = output.item()
                    current_input = torch.cat([current_input[1:, :], new_step], dim=0)
        
        # Concatenate predictions
        predictions = np.concatenate(predictions, axis=1)
        
        # Apply inverse transformation if scaler provided
        if target_scaler is not None:
            # Create dummy array with right shape for inverse_transform
            dummy = np.zeros((predictions.shape[0], target_scaler.n_features_in_))
            # Assume target is last column
            dummy[:, -1] = predictions.squeeze()
            # Inverse transform
            predictions = target_scaler.inverse_transform(dummy)[:, -1].reshape(-1, steps)
        
        return predictions
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
            learning_rate=0.001, patience=10, factor=0.5, min_lr=1e-6, device="cpu",
            scheduler_type="plateau", T_max=None, weight_decay=1e-5, clip_grad_norm=1.0,
            loss_type="mse", mse_weight=0.7, mape_weight=0.3, value_multiplier=0.01):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            patience: Patience for learning rate scheduler
            factor: Factor for reducing learning rate
            min_lr: Minimum learning rate
            device: Device for computation
            scheduler_type: Type of learning rate scheduler (plateau, cosine)
            T_max: Maximum number of iterations for cosine scheduler
            weight_decay: L2 regularization parameter
            clip_grad_norm: Max norm of gradients for clipping
            loss_type: Type of loss function (mse, mape, combined)
            mse_weight, mape_weight: Weights for combined loss
            value_multiplier: Multiplier for values to improve numerical stability
        
        Returns:
            self: Trained model
        """
        # Convert data to PyTorch tensors if not already
        if not torch.is_tensor(X_train):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Move model to device
        self.to(device)
        
        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define loss function
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mape":
            # Custom MAPE loss
            def mape_loss(pred, true):
                return torch.mean(torch.abs((true - pred) / (true + 1e-8))) * 100
            criterion = mape_loss
        elif loss_type == "combined":
            # Combined MSE and MAPE loss
            mse_criterion = nn.MSELoss()
            def combined_loss(pred, true):
                mse = mse_criterion(pred, true)
                mape = torch.mean(torch.abs((true - pred) / (true + 1e-8))) * 100
                return mse_weight * mse + mape_weight * mape
            criterion = combined_loss
        else:
            criterion = nn.MSELoss()  # Default to MSE
        
        # Define learning rate scheduler
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )
        elif scheduler_type == "cosine":
            if T_max is None:
                T_max = epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr
            )
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0
        
        # Train the model
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            train_mapes = []
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                
                # Update parameters
                optimizer.step()
                
                # Store training metrics
                train_losses.append(loss.item())
                
                # Calculate MAPE for monitoring
                with torch.no_grad():
                    mape = torch.mean(torch.abs((targets - outputs) / (targets + 1e-8))) * 100
                    train_mapes.append(mape.item())
            
            # Validation phase
            self.eval()
            val_losses = []
            val_mapes = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = self(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
                    
                    # Calculate MAPE
                    mape = torch.mean(torch.abs((targets - outputs) / (targets + 1e-8))) * 100
                    val_mapes.append(mape.item())
            
            # Calculate average losses and metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_mape = np.mean(train_mapes)
            avg_val_mape = np.mean(val_mapes)
            
            # Update learning rate based on scheduler
            if scheduler_type == "plateau":
                scheduler.step(avg_val_loss)
            else:  # cosine
                scheduler.step()
            
            # Store current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store training history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mape'].append(avg_train_mape)
            self.history['val_mape'].append(avg_val_mape)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch + 1)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | '
                      f'Train Loss: {avg_train_loss:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f} | '
                      f'Train MAPE: {avg_train_mape:.2f}% | '
                      f'Val MAPE: {avg_val_mape:.2f}% | '
                      f'LR: {current_lr:.6f}')
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience * 2:  # 2x patience for early stopping
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return self
    
    def evaluate(self, X_test, y_test, device="cpu", target_scaler=None, transform_info=None):
        """
        Evaluate the model on test data
        
        Args:
            X_test, y_test: Test data
            device: Device for computation
            target_scaler: Scaler for inverse transformation
            transform_info: Info about transformations applied to target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to tensors if not already
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # Create data loader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Set model to evaluation mode
        self.eval()
        self.to(device)
        
        all_preds = []
        all_targets = []
        
        # Make predictions
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                
                # Move to CPU for numpy conversion
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # Concatenate batches
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        # Inverse transform if scaler provided
        if target_scaler is not None:
            # Create dummy arrays with right shape for inverse_transform
            dummy_pred = np.zeros((y_pred.shape[0], target_scaler.n_features_in_))
            dummy_true = np.zeros((y_true.shape[0], target_scaler.n_features_in_))
            
            # Last column is typically the target
            dummy_pred[:, -1] = y_pred.squeeze()
            dummy_true[:, -1] = y_true.squeeze()
            
            # Inverse transform
            y_pred = target_scaler.inverse_transform(dummy_pred)[:, -1]
            y_true = target_scaler.inverse_transform(dummy_true)[:, -1]
        
        # Reverse log transform if used
        if transform_info and transform_info.get('log_transform', False):
            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'r2': r2_score(y_true, y_pred),
            'mean_pred': np.mean(y_pred),
            'mean_true': np.mean(y_true),
            'std_pred': np.std(y_pred),
            'std_true': np.std(y_true),
            'min_pred': np.min(y_pred),
            'min_true': np.min(y_true),
            'max_pred': np.max(y_pred),
            'max_true': np.max(y_true)
        }
        
        # Print main metrics
        print(f"Evaluation Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics, y_pred, y_true
    
    def predict(self, X, batch_size=64, device="cpu", target_scaler=None, log_transform_info=None):
        """
        Make predictions on new data
        
        Args:
            X: Input data
            batch_size: Batch size for processing
            device: Device for computation
            target_scaler: Scaler for inverse transformation
            log_transform_info: Info about log transformation
            
        Returns:
            Array of predictions
        """
        # Convert to tensor if not already
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Create data loader
        loader = DataLoader(X, batch_size=batch_size)
        
        # Set model to evaluation mode
        self.eval()
        self.to(device)
        
        all_preds = []
        
        # Make predictions
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                all_preds.append(outputs.cpu().numpy())
        
        # Concatenate batches
        y_pred = np.concatenate(all_preds)
        
        # Inverse transform if scaler provided
        if target_scaler is not None:
            # Create dummy array with right shape for inverse_transform
            dummy = np.zeros((y_pred.shape[0], target_scaler.n_features_in_))
            # Last column is typically the target
            dummy[:, -1] = y_pred.squeeze()
            # Inverse transform
            y_pred = target_scaler.inverse_transform(dummy)[:, -1]
        
        # Reverse log transform if used
        if log_transform_info and log_transform_info.get('applied', False):
            y_pred = np.exp(y_pred)
        
        return y_pred
    
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
        
        # 2. Plot MAPE
        ax = axes[1]
        ax.plot(self.history['epochs'], self.history['train_mape'], 'b-', label='Training MAPE', 
                linewidth=2, marker='o', markersize=4)
        ax.plot(self.history['epochs'], self.history['val_mape'], 'r-', label='Validation MAPE', 
                linewidth=2, marker='x', markersize=6)
        
        # Find best MAPE point
        best_mape_idx = np.argmin(self.history['val_mape'])
        best_mape_epoch = self.history['epochs'][best_mape_idx]
        best_mape = self.history['val_mape'][best_mape_idx]
        
        # Highlight best MAPE
        ax.scatter(best_mape_epoch, best_mape, s=150, c='purple', marker='*', 
                  label=f'Best MAPE (Epoch {best_mape_epoch}, MAPE {best_mape:.2f}%)', zorder=10)
        
        # Add gray vertical line at best MAPE
        ax.axvline(x=best_mape_epoch, color='gray', linestyle='--', alpha=0.5)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Training and Validation MAPE Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 3. Plot learning rate
        ax = axes[2]
        ax.plot(self.history['epochs'], self.history['learning_rates'], 'g-', 
                label='Learning Rate', linewidth=2, marker='s', markersize=6)
        
        # Add formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        
        # Hide the last subplot
        axes[3].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save(self, path):
        """
        Save model to file
        
        Args:
            path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'history': self.history,
            'hyperparams': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'output_dim': self.output_dim,
                'dropout_prob': self.dropout_prob,
                'bidirectional': self.bidirectional
            }
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load model from file
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        save_dict = torch.load(path, map_location=device)
        
        # Create model with saved hyperparameters
        model = cls(
            input_dim=save_dict['hyperparams']['input_dim'],
            hidden_dim=save_dict['hyperparams']['hidden_dim'],
            num_layers=save_dict['hyperparams']['num_layers'],
            output_dim=save_dict['hyperparams']['output_dim'],
            dropout_prob=save_dict['hyperparams']['dropout_prob'],
            bidirectional=save_dict['hyperparams'].get('bidirectional', False)
        )
        
        # Load state dict
        model.load_state_dict(save_dict['model_state_dict'])
        
        # Load history
        model.history = save_dict['history']
        
        return model
    
    def plot_forecast(self, X_test, y_test, forecast_steps=24, target_scaler=None,
                     figsize=(15, 8), device="cpu", plot_samples=5, offset=0):
        """
        Generate and plot multi-step forecasts
        
        Args:
            X_test: Test inputs
            y_test: Test targets (for comparison)
            forecast_steps: Number of steps to forecast
            target_scaler: Scaler for inverse transformation
            figsize: Figure size
            device: Computation device
            plot_samples: Number of samples to plot
            offset: Offset to start plotting from
            
        Returns:
            Matplotlib figure
        """
        # Convert to tensors if not already
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        
        # Get a subset of test samples
        X_subset = X_test[offset:offset+plot_samples]
        
        # Generate forecasts
        forecasts = self.forecast(
            X_subset, steps=forecast_steps, device=device, 
            target_scaler=target_scaler
        )
        
        # Create plots
        fig, axs = plt.subplots(plot_samples, 1, figsize=figsize)
        if plot_samples == 1:
            axs = [axs]
            
        # Get ground truth if available
        if y_test is not None:
            if not torch.is_tensor(y_test):
                y_test = torch.tensor(y_test, dtype=torch.float32)
                
            if target_scaler is not None:
                # Create dummy array with right shape
                dummy = np.zeros((y_test.shape[0], target_scaler.n_features_in_))
                # Last column is target
                dummy[:, -1] = y_test.numpy().squeeze()
                # Inverse transform
                y_truth = target_scaler.inverse_transform(dummy)[:, -1]
            else:
                y_truth = y_test.numpy().squeeze()
        else:
            y_truth = None
            
        # Plot each sample
        for i in range(plot_samples):
            ax = axs[i]
            
            # Plot forecast
            ax.plot(range(forecast_steps), forecasts[i], 'r-', label='Forecast')
            
            # Plot ground truth if available
            if y_truth is not None and i + offset < len(y_truth):
                # Determine how many steps of ground truth to plot
                steps_to_plot = min(forecast_steps, len(y_truth) - (i + offset))
                ax.plot(range(steps_to_plot), y_truth[i+offset:i+offset+steps_to_plot], 
                       'b-', label='Actual')
            
            ax.set_title(f'Sample {i+offset+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Solar Radiance')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        return fig
    
    def predict_with_uncertainty(self, X, mc_samples=30, device="cpu", 
                            target_scaler=None, log_transform_info=None,
                            return_samples=False, alpha=0.05):
        """
        Generate predictions with uncertainty estimates using MC Dropout
        
        Args:
            X: Input data of shape (batch_size, seq_length, input_dim)
            mc_samples: Number of Monte Carlo forward passes
            device: Device for computation
            target_scaler: Scaler for inverse transformation
            log_transform_info: Info about log transformation
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
                # Create dummy array with right shape for inverse_transform
                dummy = np.zeros((all_predictions.shape[1], target_scaler.n_features_in_))
                # Place predictions in the target column (assuming last column)
                dummy[:, -1] = all_predictions[i, :, 0]
                # Inverse transform
                all_predictions[i, :, 0] = target_scaler.inverse_transform(dummy)[:, -1]
        
        # If log transform was applied, reverse it
        if log_transform_info and log_transform_info.get('applied', False):
            all_predictions = np.exp(all_predictions)
        
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
                                    target_scaler=None, log_transform_info=None,
                                    figsize=(12, 8), device="cpu", alpha=0.05,
                                    indices=None, max_samples=5):
        """
        Plot predictions with uncertainty bounds
        
        Args:
            X: Input data
            y_true: Ground truth values (optional)
            mc_samples: Number of Monte Carlo samples
            target_scaler: Scaler for inverse transformation
            log_transform_info: Info about log transformation
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
            target_scaler=target_scaler, log_transform_info=log_transform_info,
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
                # Create dummy array with right shape
                dummy = np.zeros((y_true.shape[0], target_scaler.n_features_in_))
                # Place ground truth in target column
                dummy[:, -1] = y_true.squeeze()
                # Inverse transform
                y_true = target_scaler.inverse_transform(dummy)[:, -1]
                
            if log_transform_info and log_transform_info.get('applied', False):
                y_true = np.exp(y_true)
        
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