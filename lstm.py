import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F # For memory-efficient evaluation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer 
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
import matplotlib.pyplot as plt
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration for data processing and model constraints
class DataProcessingConfig:
    MAX_EXP_INPUT: float = 700.0  # Max input to np.exp to avoid overflow
    MIN_RADIATION_CLIP: float = 0.0
    MAX_RADIATION_CLIP: float = 2000.0 # Example, should be domain-specific

@dataclass
class ModelHyperparameters:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1 # Typically 1 for regression
    dropout_prob: float = 0.3

    def __post_init__(self):
        if not (0 <= self.dropout_prob <= 1):
            raise ValueError("dropout_prob must be between 0 and 1")

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10 # For early stopping
    factor: float = 0.5 # For LR scheduler
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    clip_grad_norm: Optional[float] = 1.0 # Max norm for gradient clipping, None to disable
    scheduler_type: str = "plateau" # "plateau" or "cosine"
    T_max_cosine: Optional[int] = None # For CosineAnnealingLR, defaults to epochs if None
    loss_type: str = "mse" # "mse", "combined", "value_aware"
    mse_weight: float = 0.7 # For combined losses
    mape_weight: float = 0.3 # For combined losses
    value_multiplier: float = 0.01 # For value_aware_combined_loss


class CombinedLoss(nn.Module):
    """
    Custom loss function that combines MSE and MAPE with improved handling of small values.
    MAPE component is effectively a "capped MAPE" as errors are clamped.
    Can operate in 'standard' or 'value_aware' mode.
    """
    def __init__(self, mse_weight: float = 0.7, mape_weight: float = 0.3, 
                 epsilon: float = 1e-8, clip_mape_percentage: float = 100.0,
                 loss_mode: str = "standard"): # Added loss_mode
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight
        self.epsilon = epsilon
        self.clip_mape_fraction = clip_mape_percentage / 100.0
        self.mse_loss = nn.MSELoss()
        self.loss_mode = loss_mode
        if self.loss_mode not in ["standard", "value_aware"]:
            raise ValueError(f"loss_mode must be 'standard' or 'value_aware', got {self.loss_mode}")
        
    def _standard_combined_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(y_pred, y_true)
        abs_percentage_error = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
        abs_percentage_error = torch.clamp(abs_percentage_error, max=self.clip_mape_fraction) 
        mape = torch.mean(abs_percentage_error) * 100.0
        return self.mse_weight * mse + self.mape_weight * (mape / 100.0)

    def _value_aware_combined_loss_calc(self, y_pred: torch.Tensor, y_true: torch.Tensor, value_multiplier: float) -> torch.Tensor:
        squared_errors = (y_true - y_pred)**2
        value_weights = 1.0 + torch.abs(y_true) * value_multiplier
        weighted_mse = torch.mean(value_weights * squared_errors)
        
        abs_percentage_error = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
        abs_percentage_error = torch.clamp(abs_percentage_error, max=self.clip_mape_fraction)
        mape = torch.mean(abs_percentage_error) * 100.0
        
        return self.mse_weight * weighted_mse + self.mape_weight * (mape / 100.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, value_multiplier: Optional[float] = 0.01) -> torch.Tensor:
        if self.loss_mode == "value_aware":
            if value_multiplier is None: # Should be passed by training loop
                logging.warning("value_multiplier not provided for value_aware loss, using default 0.01.")
                value_multiplier = 0.01 
            return self._value_aware_combined_loss_calc(y_pred, y_true, value_multiplier)
        else: # Standard combined loss
            return self._standard_combined_loss(y_pred, y_true)


class WeatherLSTM(nn.Module):
    def __init__(self, model_params: ModelHyperparameters):
        super(WeatherLSTM, self).__init__()
        self.params = model_params
        
        self.lstm = nn.LSTM(
            input_size=model_params.input_dim, 
            hidden_size=model_params.hidden_dim, 
            num_layers=model_params.num_layers,
            batch_first=True, 
            dropout=model_params.dropout_prob if model_params.num_layers > 1 else 0
        )
        self.dropout1 = nn.Dropout(model_params.dropout_prob)
        self.fc1 = nn.Linear(model_params.hidden_dim, model_params.hidden_dim // 4)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(model_params.dropout_prob)
        self.fc2 = nn.Linear(model_params.hidden_dim // 4, model_params.hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(model_params.dropout_prob)
        self.fc3 = nn.Linear(model_params.hidden_dim // 2, model_params.output_dim)
        
        self.history: Dict[str, List] = {
            'epochs': [], 'train_loss': [], 'val_loss': [],
            'val_rmse': [], 'val_r2': [], 'val_mape': [], 'val_mae': [], 'lr': []
        }
        self.transform_info: Optional[Dict] = None 
        self._mc_dropout_enabled: bool = False # For managing MC Dropout state
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Let LSTM handle hidden state initialization automatically if (h0, c0) are not provided.
        # PyTorch will create zero hidden states by default.
        out, _ = self.lstm(x) 
        
        out = out[:, -1, :] 
        out = self.dropout1(out)
        out = self.fc1(out); out = self.relu(out); out = self.dropout2(out)
        out = self.fc2(out); out = self.relu2(out); out = self.dropout3(out)
        out = self.fc3(out)
        return out

    def _get_criterion(self, config: TrainingConfig) -> nn.Module:
        if config.loss_type.lower() == "mse":
            return nn.MSELoss()
        elif config.loss_type.lower() == "combined":
            return CombinedLoss(
                mse_weight=config.mse_weight,
                mape_weight=config.mape_weight,
                loss_mode="standard" # Explicitly set mode
            )
        elif config.loss_type.lower() == "value_aware":
            return CombinedLoss(
                mse_weight=config.mse_weight,
                mape_weight=config.mape_weight,
                loss_mode="value_aware" # Explicitly set mode
            )
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")

    def _validate_fit_inputs(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must have the same number of samples.")
        if X_train.ndim != 3 or X_val.ndim != 3:
            raise ValueError("Input features (X_train, X_val) must be 3-dimensional (samples, sequence_length, num_features).")
        if y_train.ndim != 2 or y_val.ndim != 2 or y_train.shape[-1] != self.params.output_dim or y_val.shape[-1] != self.params.output_dim:
             raise ValueError(f"Target variables (y_train, y_val) must be 2-dimensional (samples, output_dim) and match model output_dim={self.params.output_dim}.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
            train_config: TrainingConfig, device: str = "cpu"):
        try:
            self._validate_fit_inputs(X_train, y_train, X_val, y_val)
        except ValueError as e:
            logging.error(f"Input validation failed for fit method: {e}")
            raise

        logging.info(f"LSTM Training started. Device: {device}, Training Config: {train_config}")
        self.history = {key: [] for key in self.history}
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        criterion = self._get_criterion(train_config)
        
        if train_config.scheduler_type.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_config.factor, patience=train_config.patience//2, min_lr=train_config.min_lr, verbose=True)
        elif train_config.scheduler_type.lower() == "cosine":
            T_max = train_config.T_max_cosine if train_config.T_max_cosine is not None else train_config.epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=train_config.min_lr)
        else: raise ValueError(f"Unknown scheduler type: {train_config.scheduler_type}")
        
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        self.to(device)
        
        for epoch in range(train_config.epochs):
            self.train()
            train_loss_epoch = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                # Pass value_multiplier if criterion is CombinedLoss and mode is value_aware
                if isinstance(criterion, CombinedLoss) and criterion.loss_mode == "value_aware":
                    loss = criterion(outputs, targets, value_multiplier=train_config.value_multiplier)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                if train_config.clip_grad_norm is not None and train_config.clip_grad_norm > 0:
                     torch.nn.utils.clip_grad_norm_(self.parameters(), train_config.clip_grad_norm)
                optimizer.step()
                train_loss_epoch += loss.item() * inputs.size(0)
            train_loss_epoch /= len(train_loader.dataset)
            
            # Validation phase
            self.eval()
            val_loss_epoch = 0.0
            val_outputs_all, val_targets_all = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    if isinstance(criterion, CombinedLoss) and criterion.loss_mode == "value_aware":
                        loss_val = criterion(outputs, targets, value_multiplier=train_config.value_multiplier)
                    else:
                        loss_val = criterion(outputs, targets)
                    val_loss_epoch += loss_val.item() * inputs.size(0)
                    val_outputs_all.append(outputs.cpu().numpy())
                    val_targets_all.append(targets.cpu().numpy())
            val_loss_epoch /= len(val_loader.dataset)
            
            val_predictions = np.vstack(val_outputs_all)
            val_actuals = np.vstack(val_targets_all)
            
            val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
            val_r2 = r2_score(val_actuals, val_predictions)
            val_mae = mean_absolute_error(val_actuals, val_predictions)
            epsilon_mape = 1e-8 
            val_mape_capped = np.mean(np.clip(np.abs((val_actuals - val_predictions) / (np.abs(val_actuals) + epsilon_mape)), 0, 1.0)) * 100 

            self.history['epochs'].append(epoch + 1); self.history['train_loss'].append(train_loss_epoch)
            self.history['val_loss'].append(val_loss_epoch); self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2); self.history['val_mape'].append(val_mape_capped)
            self.history['val_mae'].append(val_mae); self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            logging.info(f'LSTM Epoch {epoch+1}/{train_config.epochs} - TrainLoss: {train_loss_epoch:.4f} - ValLoss: {val_loss_epoch:.4f} | Scaled Metrics: ValRMSE: {val_rmse:.4f}, ValR²: {val_r2:.4f}, ValCappedMAPE: {val_mape_capped:.2f}%, ValMAE: {val_mae:.4f}')
            
            if train_config.scheduler_type.lower() == "plateau": scheduler.step(val_loss_epoch)
            else: scheduler.step()
            
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch; best_model_state = self.state_dict().copy(); early_stopping_counter = 0
            else: early_stopping_counter += 1
            if early_stopping_counter >= train_config.patience:
                logging.info(f"LSTM Early stopping at epoch {epoch+1}"); break
        
        if best_model_state: self.load_state_dict(best_model_state)
        logging.info("LSTM Training complete. Best model state loaded.")
        return self

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0: return arr.reshape(1, -1)
        elif arr.ndim == 1: return arr.reshape(-1, 1)
        return arr

    def _apply_inverse_scaling(self, y: np.ndarray, target_scaler: Any, transform_info: Dict) -> np.ndarray:
        """Handles inverse scaling part of the transformation."""
        y_processed = y # Assumes y is already 2D via _ensure_2d
        if target_scaler is not None:
            if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
                target_col_name = transform_info.get('target_col_transformed_final', transform_info.get('target_col_original'))
                target_idx = -1
                if hasattr(target_scaler, 'feature_names_in_') and target_scaler.feature_names_in_ is not None:
                    try: target_idx = list(target_scaler.feature_names_in_).index(target_col_name)
                    except ValueError: logging.warning(f"Target '{target_col_name}' not in scaler features. Defaulting to last.")
                else: logging.warning("Scaler has no feature_names_in_. Assuming target is last for multi-feature scaler.")
                
                dummy = np.zeros((y_processed.shape[0], target_scaler.n_features_in_))
                dummy[:, target_idx] = y_processed.squeeze()
                y_processed = target_scaler.inverse_transform(dummy)[:, target_idx]
            else: # Scaler was for single feature (target)
                y_processed = target_scaler.inverse_transform(y_processed).squeeze()
        return self._ensure_2d(y_processed).squeeze() # Ensure 1D for next steps

    def _apply_inverse_structural_transforms(self, y: np.ndarray, transform_info: Dict, scalers_dict: Optional[Dict] = None) -> np.ndarray:
        """Handles inverse structural transformations (log, Yeo-Johnson)."""
        y_processed = y # Assumes y is 1D from _apply_inverse_scaling
        if transform_info and 'transforms' in transform_info:
            target_col_orig_name = transform_info.get('target_col_original')
            for t_details in reversed(transform_info.get('transforms', [])):
                if t_details.get('original_col') != target_col_orig_name or not t_details.get('applied', False):
                    continue
                t_type = t_details.get('type')
                logging.info(f"Applying inverse structural transform: {t_type} for {target_col_orig_name}")
                if t_type == 'log':
                    offset = t_details.get('offset', 0)
                    exp_input = np.clip(y_processed, -DataProcessingConfig.MAX_EXP_INPUT, DataProcessingConfig.MAX_EXP_INPUT)
                    y_processed = np.exp(exp_input)
                    if offset > 0: y_processed -= offset
                elif t_type == 'yeo-johnson':
                    lambda_val = t_details.get('lambda')
                    pt_obj = scalers_dict.get('power_transformer_object_for_target') if scalers_dict else None
                    if pt_obj and lambda_val is not None:
                        y_processed = pt_obj.inverse_transform(self._ensure_2d(y_processed)).flatten()
                    elif lambda_val is not None:
                        logging.warning("Recreating PowerTransformer for inverse Yeo-Johnson.")
                        pt_inv = PowerTransformer(method='yeo-johnson', standardize=False)
                        pt_inv.lambdas_ = np.array([lambda_val])
                        y_processed = pt_inv.inverse_transform(self._ensure_2d(y_processed)).flatten()
                    else: logging.warning(f"Yeo-Johnson lambda/object not found. Skipping.")
        return y_processed

    def _apply_domain_clipping(self, y: np.ndarray, transform_info: Dict) -> np.ndarray:
        """Applies domain-specific clipping, e.g., for GHI."""
        y_processed = y
        if transform_info and transform_info.get('target_col_original') == 'Radiation':
            y_processed = np.clip(y, DataProcessingConfig.MIN_RADIATION_CLIP, DataProcessingConfig.MAX_RADIATION_CLIP)
        return y_processed

    def _inverse_transform_target(self, y: np.ndarray, target_scaler: Any, 
                                  transform_info: Dict, scalers_dict: Optional[Dict] = None) -> np.ndarray:
        """Main orchestrator for inverse transformation."""
        y_processed_2d = self._ensure_2d(y.copy())
        
        # Step 1: Inverse scaling
        y_unscaled_1d = self._apply_inverse_scaling(y_processed_2d, target_scaler, transform_info)
        
        # Step 2: Inverse structural transforms
        y_untransformed_1d = self._apply_inverse_structural_transforms(y_unscaled_1d, transform_info, scalers_dict)
        
        # Step 3: Domain-specific clipping
        y_final_1d = self._apply_domain_clipping(y_untransformed_1d, transform_info)
            
        return y_final_1d # Should be 1D array
        
    def evaluate(self, X_test_data: np.ndarray, y_test_data: np.ndarray, device: str = "cpu", 
                 target_scaler_object: Optional[Any] = None, 
                 transform_info_dict: Optional[Dict] = None, 
                 scalers_dict: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
        self.eval(); self.to(device)
        actuals_scaled_np = np.asarray(y_test_data).flatten()
        inputs_tensor = torch.FloatTensor(X_test_data).to(device)
        model_predictions_scaled_np: Optional[np.ndarray] = None
        with torch.no_grad():
            model_predictions_scaled_np = self(inputs_tensor).cpu().numpy().flatten()
        
        logging.info("\n--- LSTM Scaled Metrics ---")
        # ... (scaled metrics calculation as before)
        mse_scaled = mean_squared_error(actuals_scaled_np, model_predictions_scaled_np)
        rmse_scaled = np.sqrt(mse_scaled); r2_scaled = r2_score(actuals_scaled_np, model_predictions_scaled_np)
        mae_scaled = mean_absolute_error(actuals_scaled_np, model_predictions_scaled_np)
        epsilon_mape = 1e-8
        mape_scaled_capped = np.mean(np.clip(np.abs((actuals_scaled_np - model_predictions_scaled_np) / (np.abs(actuals_scaled_np) + epsilon_mape)), 0, 1.0)) * 100
        logging.info(f"RMSE (scaled): {rmse_scaled:.4f}, R² (scaled): {r2_scaled:.4f}, MAE (scaled): {mae_scaled:.4f}, Capped MAPE (scaled): {mape_scaled_capped:.2f}%")

        predictions_original_scale, actuals_original_scale = None, None
        original_metrics: Dict[str, float] = {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'mape_capped': np.nan}
        if target_scaler_object and transform_info_dict:
            logging.info("\n--- LSTM Original Scale Metrics ---")
            try:
                # Ensure inputs to _inverse_transform_target are 2D column vectors if they are single series
                preds_scaled_2d = model_predictions_scaled_np.reshape(-1, 1)
                actuals_scaled_2d = actuals_scaled_np.reshape(-1, 1)

                predictions_original_scale = self._inverse_transform_target(preds_scaled_2d, target_scaler_object, transform_info_dict, scalers_dict).flatten()
                actuals_original_scale = self._inverse_transform_target(actuals_scaled_2d, target_scaler_object, transform_info_dict, scalers_dict).flatten()
                
                # ... (original metrics calculation as before)
                if not (np.isnan(predictions_original_scale).any() or np.isnan(actuals_original_scale).any()):
                    if len(predictions_original_scale) > 0 and len(actuals_original_scale) > 0 and len(predictions_original_scale) == len(actuals_original_scale):
                        rmse_orig = np.sqrt(mean_squared_error(actuals_original_scale, predictions_original_scale))
                        r2_orig = r2_score(actuals_original_scale, predictions_original_scale)
                        mae_orig = mean_absolute_error(actuals_original_scale, predictions_original_scale)
                        mape_orig_capped = np.mean(np.clip(np.abs((actuals_original_scale - predictions_original_scale) / (np.abs(actuals_original_scale) + epsilon_mape)), 0, 1.0)) * 100
                        logging.info(f"RMSE (original): {rmse_orig:.4f}, R² (original): {r2_orig:.4f}, MAE (original): {mae_orig:.4f}, Capped MAPE (original): {mape_orig_capped:.2f}%")
                        original_metrics = {'rmse': rmse_orig, 'r2': r2_orig, 'mae': mae_orig, 'mape_capped': mape_orig_capped}
            except Exception as e: logging.error(f"Error in original scale metrics calculation: {e}")
        
        all_metrics = {'scaled_rmse': rmse_scaled, 'scaled_r2': r2_scaled, 'scaled_mae': mae_scaled, 'scaled_mape_capped': mape_scaled_capped, **original_metrics}
        return (model_predictions_scaled_np, actuals_scaled_np, predictions_original_scale, actuals_original_scale, all_metrics)

    def evaluate_memory_efficient(self, val_loader: DataLoader, criterion: nn.Module, 
                                  train_config: TrainingConfig, device: str = "cpu",
                                  target_scaler_object: Optional[Any] = None, 
                                  transform_info_dict: Optional[Dict] = None, 
                                  scalers_dict: Optional[Dict] = None) -> Dict[str, float]:
        self.eval(); self.to(device)
        total_loss = 0.0; all_scaled_preds_list, all_scaled_actuals_list = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                if isinstance(criterion, CombinedLoss) and criterion.loss_mode == "value_aware":
                     loss = criterion(outputs, targets, value_multiplier=train_config.value_multiplier)
                else: loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                all_scaled_preds_list.append(outputs.cpu().numpy()); all_scaled_actuals_list.append(targets.cpu().numpy())
        avg_loss = total_loss / len(val_loader.dataset)
        preds_s = np.concatenate(all_scaled_preds_list); actuals_s = np.concatenate(all_scaled_actuals_list)
        # ... (scaled metrics calculation as before) ...
        rmse_s = np.sqrt(mean_squared_error(actuals_s, preds_s)); mae_s = mean_absolute_error(actuals_s, preds_s)
        epsilon_mape = 1e-8
        mape_s_capped = np.mean(np.clip(np.abs((actuals_s - preds_s) / (np.abs(actuals_s) + epsilon_mape)), 0, 1.0)) * 100
        metrics = {'avg_val_loss_scaled': avg_loss, 'rmse_scaled': rmse_s, 'mae_scaled': mae_s, 'mape_scaled_capped': mape_s_capped}
        logging.info(f"Memory-Efficient Eval (Scaled): Loss={avg_loss:.4f}, RMSE={rmse_s:.4f}, MAE={mae_s:.4f}, CappedMAPE={mape_s_capped:.2f}%")
        if target_scaler_object and transform_info_dict:
            try:
                # Ensure inputs are 2D for _inverse_transform_target
                preds_orig = self._inverse_transform_target(preds_s.reshape(-1,1), target_scaler_object, transform_info_dict, scalers_dict)
                actuals_orig = self._inverse_transform_target(actuals_s.reshape(-1,1), target_scaler_object, transform_info_dict, scalers_dict)
                if not (np.isnan(preds_orig).any() or np.isnan(actuals_orig).any()):
                    if len(preds_orig) > 0 and len(actuals_orig) > 0 and len(preds_orig) == len(actuals_orig):
                        metrics['rmse_original'] = np.sqrt(mean_squared_error(actuals_orig, preds_orig))
                        metrics['mae_original'] = mean_absolute_error(actuals_orig, preds_orig)
                        metrics['mape_original_capped'] = np.mean(np.clip(np.abs((actuals_orig - preds_orig) / (np.abs(actuals_orig) + epsilon_mape)), 0, 1.0)) * 100
                        logging.info(f"Memory-Efficient Eval (Original): RMSE={metrics['rmse_original']:.4f}, MAE={metrics['mae_original']:.4f}, CappedMAPE={metrics['mape_original_capped']:.2f}%")
            except Exception as e: logging.error(f"Error in original scale metrics during memory-efficient eval: {e}")
        return metrics

    def _ensure_batched_input(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Ensure input is a PyTorch tensor and properly batched for prediction."""
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float() # Ensure float
        else:
            raise TypeError(f"Input X must be np.ndarray or torch.Tensor, got {type(X)}")

        if X_tensor.ndim == 1:  # Single feature vector for one timestep (features,)
            # This case is ambiguous for LSTMs expecting (seq_len, features).
            # Assuming it's a sequence of length 1.
            logging.warning("1D input to _ensure_batched_input, assuming seq_len=1.")
            return X_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, features)
        elif X_tensor.ndim == 2:  # Single sequence (seq_len, features)
            return X_tensor.unsqueeze(0)  # (1, seq_len, features)
        elif X_tensor.ndim == 3:  # Already batched (batch_size, seq_len, features)
            return X_tensor
        else:
            raise ValueError(f"Input X must be 1D, 2D, or 3D, got {X_tensor.ndim}D")

    def predict(self, X: np.ndarray, batch_size: int = 32, device: str = "cpu", 
                target_scaler: Optional[Any] = None, 
                transform_info: Optional[Dict] = None, 
                scalers_dict: Optional[Dict] = None) -> np.ndarray:
        self.eval(); self.to(device)
        
        X_batched_tensor = self._ensure_batched_input(X).to(device)
        
        dataset = TensorDataset(X_batched_tensor) # DataLoader needs a dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions_scaled_list = []
        with torch.no_grad():
            for (inputs_batch,) in loader: # inputs_batch is already on device from _ensure_batched_input
                outputs = self(inputs_batch)
                predictions_scaled_list.append(outputs.cpu().numpy())
        
        predictions_scaled_np = np.concatenate(predictions_scaled_list, axis=0)
        
        if target_scaler is not None and transform_info is not None:
            # Output from model is (N, output_dim). Ensure it's (N,1) for inverse transform if output_dim is 1.
            if predictions_scaled_np.ndim == 1: predictions_scaled_np = predictions_scaled_np.reshape(-1,1)
            return self._inverse_transform_target(predictions_scaled_np, target_scaler, transform_info, scalers_dict)
        else:
            logging.warning("target_scaler or transform_info not provided to predict method. Returning scaled predictions.")
            return predictions_scaled_np.squeeze()
    
    # plot_training_history remains largely the same, ensure it uses self.history correctly.
    def plot_training_history(self, figsize: Tuple[int, int] = (20, 18), log_scale_loss: bool = True):

        if not self.history['epochs']: 
            logging.info("No LSTM training history to plot.")
            return None
        
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        axes = axes.flatten() 

        metrics_plot_config = [
            ('Loss', ['train_loss', 'val_loss'], log_scale_loss, ['b-', 'r-']),
            ('RMSE (Scaled)', ['val_rmse'], False, ['g-']),
            ('R² (Scaled)', ['val_r2'], False, ['m-']),
            ('Capped MAPE (Scaled, %)', ['val_mape'], False, ['darkorange-']),
            ('MAE (Scaled)', ['val_mae'], False, ['darkcyan-']),
            ('Learning Rate', ['lr'], True, ['c-'])
        ]
        for i, (title, keys, ylog, line_colors) in enumerate(metrics_plot_config):
            ax = axes[i]
            for k_idx, key in enumerate(keys):
                if key in self.history and self.history[key]: 
                    label = key.replace('_', ' ').title()
                    ax.plot(self.history['epochs'], self.history[key], line_colors[k_idx], label=label)
                    if 'val' in key and self.history[key]:
                        best_idx = np.argmin(self.history[key]) if any(s in key for s in ['loss', 'rmse', 'mape', 'mae']) else np.argmax(self.history[key])
                        if best_idx < len(self.history['epochs']):
                             ax.scatter(self.history['epochs'][best_idx], self.history[key][best_idx], s=100, c='gold', marker='*', zorder=5, label=f'Best {label.split(" ")[0]}')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(keys[0].split('_')[-1].upper() if '_' in keys[0] else keys[0].upper()); ax.legend(); ax.grid(True)
            if ylog: ax.set_yscale('log')
        
        ax_ratio = axes[6] 
        if all(k in self.history and self.history[k] for k in ['train_loss', 'val_loss']) and len(self.history['train_loss']) == len(self.history['val_loss']) > 0:
            train_loss_s = np.maximum(np.array(self.history['train_loss']), 1e-9)
            ratio = np.array(self.history['val_loss']) / train_loss_s
            ax_ratio.plot(self.history['epochs'], ratio, 'slateblue', label='Val/Train Loss Ratio')
            ax_ratio.axhline(1.0, color='grey', ls='--'); 
            if len(ratio) > 0 and np.nanmax(ratio) > 1.5 : ax_ratio.text(0.5, 0.9, "Ratio > 1.5: Overfitting?", transform=ax_ratio.transAxes, ha='center', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
        else: ax_ratio.text(0.5, 0.5, "Loss data missing", transform=ax_ratio.transAxes, ha='center')
        ax_ratio.set_title('Overfitting Indicator'); ax_ratio.set_xlabel('Epoch'); ax_ratio.set_ylabel('Ratio'); ax_ratio.legend(); ax_ratio.grid(True)
        if len(axes) > 7 and fig.axes[-1] == axes[7]: fig.delaxes(axes[7])
        plt.suptitle('LSTM Model Training Metrics', fontsize=16, fontweight='bold'); plt.tight_layout(rect=[0,0,1,0.96]); return fig


    def save(self, path: str):
        # ... (save logic as in previous version, using self.params) ...
        try:
            save_content = {
                'model_state_dict': self.state_dict(),
                'model_params': self.params, 
                'history': self.history, 
                'transform_info': self.transform_info
            }
            torch.save(save_content, path)
            logging.info(f"LSTM Model saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save LSTM model to {path}: {e}")
            raise
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'WeatherLSTM':
        try:
            # MODIFIED LINE: Added weights_only=False
            checkpoint = torch.load(path, map_location=device, weights_only=False) 
            
            model_params = checkpoint['model_params']
            if not isinstance(model_params, ModelHyperparameters):
                 # If it was saved as dict from an older version, recreate dataclass
                 model_params = ModelHyperparameters(**model_params) 
            
            model = cls(model_params=model_params)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Ensure history keys exist if loading older models; provide default empty lists
            default_history = {key: [] for key in model.history_template_keys} # Assuming you add a class attr like history_template_keys = ['epochs', 'train_loss', ...]
            model.history = checkpoint.get('history', default_history) 
            model.transform_info = checkpoint.get('transform_info')
            model.to(device)
            logging.info(f"LSTM Model loaded from {path} with weights_only=False.")
            return model
        except Exception as e:
            logging.error(f"Failed to load LSTM model from {path}: {e}", exc_info=True)
            # Add the specific error message for unsupported globals if that's the case
            if "Unsupported global" in str(e):
                logging.error(
                    "This might be due to custom classes (like ModelHyperparameters) in the checkpoint. "
                    "Ensure these classes are defined in the scope where load is called. "
                    "If using PyTorch 1.13+ and weights_only=True is implicitly active, "
                    "consider using weights_only=False (if you trust the source) "
                    "or torch.serialization.add_safe_globals."
                )
            raise
    
    def enable_mc_dropout(self):
        """Enables dropout layers for Monte Carlo Dropout."""
        self._mc_dropout_enabled = True
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train() # Set dropout to train mode to activate it

    def disable_mc_dropout(self):
        """Disables dropout layers and sets model to eval mode."""
        self._mc_dropout_enabled = False
        self.eval() # Sets all modules, including dropout, to eval mode

    def predict_with_uncertainty(self, X: np.ndarray, mc_samples: int = 30, device: str = "cpu", 
                                 target_scaler: Optional[Any] = None, 
                                 transform_info: Optional[Dict] = None, 
                                 scalers_dict: Optional[Dict] = None,
                                 return_samples: bool = False, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        self.enable_mc_dropout() # Enable dropout for MC sampling
        try:
            X_batched_tensor = self._ensure_batched_input(X).to(device)
            predictions_scaled_mc_list = []
            with torch.no_grad(): # Still no grad for inference part
                for _ in range(mc_samples):
                    predictions_scaled_mc_list.append(self(X_batched_tensor).cpu().numpy())
            
            predictions_scaled_mc_np = np.stack(predictions_scaled_mc_list, axis=0)
            
            predictions_original_mc = np.zeros_like(predictions_scaled_mc_np)
            if target_scaler is not None and transform_info is not None:
                for i in range(mc_samples):
                    current_batch_scaled = predictions_scaled_mc_np[i, :, :]
                    # Ensure input to _inverse_transform_target is 2D (samples, features)
                    # Model output is (batch, output_dim). If output_dim=1, it's (batch, 1)
                    inverted_sample = self._inverse_transform_target(
                        current_batch_scaled, target_scaler, transform_info, scalers_dict
                    ) # _inverse_transform_target now returns 1D if input was effectively (N,1)
                    # Reshape to (batch, output_dim) for consistency in predictions_original_mc
                    predictions_original_mc[i, :, :] = inverted_sample.reshape(current_batch_scaled.shape)
            else:
                logging.warning("No scalers/transform_info for uncertainty. Returning scaled uncertainty.")
                predictions_original_mc = predictions_scaled_mc_np

            results = {'mean': np.mean(predictions_original_mc, axis=0), 
                       'std': np.std(predictions_original_mc, axis=0),
                       'lower_ci': np.percentile(predictions_original_mc, (alpha/2)*100, axis=0),
                       'upper_ci': np.percentile(predictions_original_mc, (1-alpha/2)*100, axis=0)}
            if return_samples: results['samples'] = predictions_original_mc
            return results
        finally:
            self.disable_mc_dropout() # Ensure dropout is disabled and model is back in eval mode

    def plot_prediction_with_uncertainty(self, X: np.ndarray, y_true: Optional[np.ndarray] = None, 
                                         mc_samples: int = 30, 
                                         target_scaler: Optional[Any] = None, 
                                         transform_info: Optional[Dict] = None, 
                                         scalers_dict: Optional[Dict] = None,
                                         figsize: Tuple[int, int] = (12, 8), device: str = "cpu", 
                                         alpha: float = 0.05,
                                         indices: Optional[List[int]] = None, 
                                         max_samples_to_plot: int = 5):
        current_transform_info = transform_info if transform_info is not None else self.transform_info
        X_for_prediction = self._ensure_batched_input(X) # Uses the new helper

        uncertainty_results = self.predict_with_uncertainty(
            X_for_prediction.cpu().numpy(), # predict_with_uncertainty expects numpy
            mc_samples, device, target_scaler, 
            current_transform_info, scalers_dict, True, alpha
        )
        # ... (rest of the plotting logic as in previous version, ensuring indices and shapes are handled correctly) ...
        num_available_samples_in_batch = X_for_prediction.shape[0]
        if indices is None:
            num_to_plot = min(max_samples_to_plot, num_available_samples_in_batch)
            plot_indices = np.arange(num_to_plot)
        else:
            plot_indices = [i for i in np.array(indices) if i < num_available_samples_in_batch]

        y_true_original_np: Optional[np.ndarray] = None
        if y_true is not None:
            y_true_arr = np.array(y_true).copy()
            if target_scaler and current_transform_info:
                # Ensure y_true_arr is 2D for _inverse_transform_target if it's a single series
                y_true_reshaped = y_true_arr.reshape(-1, self.params.output_dim)
                y_true_original_np = self._inverse_transform_target(
                    y_true_reshaped, target_scaler, current_transform_info, scalers_dict
                ).flatten()
            else: y_true_original_np = y_true_arr.flatten()

        if not plot_indices: 
            logging.info("No valid samples to plot for LSTM uncertainty.")
            return plt.figure(figsize=figsize) 

        fig, axs = plt.subplots(len(plot_indices), 1, figsize=figsize, squeeze=False)
        for i, sample_idx_in_batch in enumerate(plot_indices):
            ax = axs[i,0]
            mean_val = uncertainty_results['mean'][sample_idx_in_batch, 0] 
            lower_val = uncertainty_results['lower_ci'][sample_idx_in_batch, 0]
            upper_val = uncertainty_results['upper_ci'][sample_idx_in_batch, 0]
            all_mc_samps = uncertainty_results['samples'][:, sample_idx_in_batch, 0]
            
            for j in range(min(10, mc_samples)): ax.plot([0],[all_mc_samps[j]],'o',alpha=0.15,c='gray',ms=3)
            ax.errorbar([0],[mean_val],yerr=[[mean_val-lower_val],[upper_val-mean_val]],fmt='o',c='blue',ecolor='lightblue',capsize=5,label=f'{int((1-alpha)*100)}% CI')
            
            if y_true_original_np is not None and sample_idx_in_batch < len(y_true_original_np):
                actual_val = y_true_original_np[sample_idx_in_batch]
                ax.plot([0],[actual_val],'ro',label='Actual')
                err = abs(mean_val - actual_val); in_ci = lower_val <= actual_val <= upper_val
                ax.text(0.02,0.95,f"Err: {err:.2f}",transform=ax.transAxes,fontsize=9)
                ax.text(0.02,0.90,f"In CI: {'Y' if in_ci else 'N'}",transform=ax.transAxes,fontsize=9,color='g' if in_ci else 'r')
            
            iw = upper_val-lower_val; ax.text(0.02,0.85,f"IW: {iw:.2f}",transform=ax.transAxes,fontsize=9)
            ax.set_title(f"Pred for Sample Idx in Batch: {sample_idx_in_batch}",fontsize=10)
            
            plot_min_y_vals = [lower_val] + ([y_true_original_np[sample_idx_in_batch]] if y_true_original_np is not None and sample_idx_in_batch < len(y_true_original_np) else [])
            plot_max_y_vals = [upper_val] + ([y_true_original_np[sample_idx_in_batch]] if y_true_original_np is not None and sample_idx_in_batch < len(y_true_original_np) else [])
            plot_min_y = min(plot_min_y_vals) if plot_min_y_vals else 0
            plot_max_y = max(plot_max_y_vals) if plot_max_y_vals else 1 # Default if empty
            padding = 0.2*(plot_max_y-plot_min_y) if (plot_max_y-plot_min_y)>1e-6 else 1.0
            ax.set_ylim([max(DataProcessingConfig.MIN_RADIATION_CLIP, plot_min_y - padding), plot_max_y + padding]) 
            ax.set_xticks([]); ax.set_ylabel("GHI (Original Scale)",fontsize=9)
            if i==0: ax.legend(fontsize=8,loc='best')
        
        plt.suptitle(f'LSTM Preds w {int((1-alpha)*100)}% CI (Original Scale)',fontsize=14,fontweight='bold'); plt.tight_layout(rect=[0,0.03,1,0.95]); return fig
