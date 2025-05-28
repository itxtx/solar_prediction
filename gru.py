import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F # For memory-efficient evaluation (if used)
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

# Configuration for data processing (shared with LSTM or can be defined here if specific)
class DataProcessingConfig:
    MAX_EXP_INPUT: float = 700.0
    MIN_RADIATION_CLIP: float = 0.0
    MAX_RADIATION_CLIP: float = 2000.0

@dataclass
class GRUModelHyperparameters: # Specific to GRU
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout_prob: float = 0.3
    bidirectional: bool = False

    def __post_init__(self):
        if not (0 <= self.dropout_prob <= 1):
            raise ValueError("dropout_prob must be between 0 and 1")

@dataclass
class TrainingConfig: 
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10
    factor: float = 0.5
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    clip_grad_norm: Optional[float] = 1.0
    scheduler_type: str = "plateau"
    T_max_cosine: Optional[int] = None
    loss_type: str = "mse" # For GRU, simpler: "mse", "mae"
class WeatherGRU(nn.Module):
    def __init__(self, model_params: GRUModelHyperparameters):
        super(WeatherGRU, self).__init__()
        self.params = model_params
        
        self.gru = nn.GRU(
            input_size=model_params.input_dim,
            hidden_size=model_params.hidden_dim,
            num_layers=model_params.num_layers,
            batch_first=True,
            dropout=model_params.dropout_prob if model_params.num_layers > 1 else 0,
            bidirectional=model_params.bidirectional
        )
        
        fc_input_features = model_params.hidden_dim * 2 if model_params.bidirectional else model_params.hidden_dim
        self.fc = nn.Linear(fc_input_features, model_params.output_dim)
        # Dropout after GRU output processing before FC layer
        self.dropout_fc = nn.Dropout(model_params.dropout_prob) 
        
        self.history: Dict[str, List] = {
            'epochs': [], 'train_loss': [], 'val_loss': [],
            'val_rmse': [], 'val_r2': [], 'val_mape': [], 'val_mae': [], 'lr': []
        }
        self.transform_info: Optional[Dict] = None 
        self._mc_dropout_enabled: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Let GRU handle hidden state initialization automatically
        gru_out, _ = self.gru(x) 
        
        # Get output from the last time step
        # If bidirectional, gru_out[:, -1, :] contains concatenated forward and backward hidden states
        out = gru_out[:, -1, :] 
        
        out = self.dropout_fc(out) # Apply dropout before the final fully connected layer
        out = self.fc(out)
        return out

    def _get_criterion(self, config: TrainingConfig) -> nn.Module:
        if config.loss_type.lower() == "mse":
            return nn.MSELoss()
        elif config.loss_type.lower() == "mae":
            return nn.L1Loss() # MAE Loss

        else:
            logging.warning(f"Unknown loss type '{config.loss_type}' for GRU. Defaulting to MSE.")
            return nn.MSELoss()

    def _validate_fit_inputs(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        if X_train.shape[0] != y_train.shape[0]: raise ValueError("X_train/y_train sample mismatch.")
        if X_val.shape[0] != y_val.shape[0]: raise ValueError("X_val/y_val sample mismatch.")
        if X_train.ndim != 3 or X_val.ndim != 3: raise ValueError("Input X must be 3D.")
        if y_train.ndim != 2 or y_val.ndim != 2 or y_train.shape[-1] != self.params.output_dim or y_val.shape[-1] != self.params.output_dim:
             raise ValueError(f"Target y must be 2D and match model output_dim={self.params.output_dim}.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
            train_config: TrainingConfig, device: str = "cpu"):
        try:
            self._validate_fit_inputs(X_train, y_train, X_val, y_val)
        except ValueError as e:
            logging.error(f"Input validation failed for GRU fit: {e}")
            raise

        logging.info(f"GRU Training started. Device: {device}, Config: {train_config}")
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
                optimizer.zero_grad(); outputs = self(inputs); loss = criterion(outputs, targets)
                loss.backward()
                if train_config.clip_grad_norm is not None and train_config.clip_grad_norm > 0:
                     torch.nn.utils.clip_grad_norm_(self.parameters(), train_config.clip_grad_norm)
                optimizer.step(); train_loss_epoch += loss.item() * inputs.size(0)
            train_loss_epoch /= len(train_loader.dataset)
            
            self.eval()
            val_loss_epoch = 0.0; val_outputs_all, val_targets_all = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs); loss_val = criterion(outputs, targets)
                    val_loss_epoch += loss_val.item() * inputs.size(0)
                    val_outputs_all.append(outputs.cpu().numpy()); val_targets_all.append(targets.cpu().numpy())
            val_loss_epoch /= len(val_loader.dataset)
            
            val_preds = np.vstack(val_outputs_all); val_acts = np.vstack(val_targets_all)
            val_rmse = np.sqrt(mean_squared_error(val_acts, val_preds)); val_r2 = r2_score(val_acts, val_preds)
            val_mae = mean_absolute_error(val_acts, val_preds); epsilon_mape = 1e-8 
            val_mape_cap = np.mean(np.clip(np.abs((val_acts - val_preds) / (np.abs(val_acts) + epsilon_mape)), 0, 1.0)) * 100 

            self.history['epochs'].append(epoch + 1); self.history['train_loss'].append(train_loss_epoch)
            self.history['val_loss'].append(val_loss_epoch); self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2); self.history['val_mape'].append(val_mape_cap)
            self.history['val_mae'].append(val_mae); self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            logging.info(f'GRU Epoch {epoch+1}/{train_config.epochs} - TrainLoss: {train_loss_epoch:.4f} - ValLoss: {val_loss_epoch:.4f} | Scaled Metrics: ValRMSE: {val_rmse:.4f}, ValR²: {val_r2:.4f}, ValCappedMAPE: {val_mape_cap:.2f}%, ValMAE: {val_mae:.4f}')
            
            if train_config.scheduler_type.lower() == "plateau": scheduler.step(val_loss_epoch)
            else: scheduler.step()
            
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch; best_model_state = self.state_dict().copy(); early_stopping_counter = 0
            else: early_stopping_counter += 1
            if early_stopping_counter >= train_config.patience:
                logging.info(f"GRU Early stopping at epoch {epoch+1}"); break
        
        if best_model_state: self.load_state_dict(best_model_state)
        logging.info("GRU Training complete. Best model state loaded.")
        return self

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0: return arr.reshape(1, -1)
        elif arr.ndim == 1: return arr.reshape(-1, 1)
        return arr

    def _apply_inverse_scaling(self, y: np.ndarray, target_scaler: Any, transform_info: Dict) -> np.ndarray:
        y_proc = y 
        if target_scaler is not None:
            if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
                # ... (logic similar to LSTM's _apply_inverse_scaling) ...
                target_col_name = transform_info.get('target_col_transformed_final', transform_info.get('target_col_original'))
                target_idx = -1
                if hasattr(target_scaler, 'feature_names_in_') and target_scaler.feature_names_in_ is not None:
                    try: target_idx = list(target_scaler.feature_names_in_).index(target_col_name)
                    except ValueError: logging.warning(f"GRU Target '{target_col_name}' not in scaler features. Defaulting to last.")
                else: logging.warning("GRU Scaler has no feature_names_in_. Assuming target is last for multi-feature scaler.")
                dummy = np.zeros((y_proc.shape[0], target_scaler.n_features_in_))
                dummy[:, target_idx] = y_proc.squeeze()
                y_proc = target_scaler.inverse_transform(dummy)[:, target_idx]
            else: 
                y_proc = target_scaler.inverse_transform(y_proc).squeeze()
        return self._ensure_2d(y_proc).squeeze()

    def _apply_inverse_structural_transforms(self, y: np.ndarray, transform_info: Dict, scalers_dict: Optional[Dict] = None) -> np.ndarray:
        y_proc = y
        if transform_info and 'transforms' in transform_info:
            # ... (logic similar to LSTM's _apply_inverse_structural_transforms) ...
            target_col_orig_name = transform_info.get('target_col_original')
            for t_details in reversed(transform_info.get('transforms', [])):
                if t_details.get('original_col') != target_col_orig_name or not t_details.get('applied', False): continue
                t_type = t_details.get('type')
                logging.info(f"GRU Applying inverse structural transform: {t_type} for {target_col_orig_name}")
                if t_type == 'log':
                    offset = t_details.get('offset', 0)
                    exp_input = np.clip(y_proc, -DataProcessingConfig.MAX_EXP_INPUT, DataProcessingConfig.MAX_EXP_INPUT)
                    y_proc = np.exp(exp_input)
                    if offset > 0: y_proc -= offset
                elif t_type == 'yeo-johnson':
                    lambda_val = t_details.get('lambda')
                    pt_obj = scalers_dict.get('power_transformer_object_for_target') if scalers_dict else None
                    if pt_obj and lambda_val is not None:
                        y_proc = pt_obj.inverse_transform(self._ensure_2d(y_proc)).flatten()
                    elif lambda_val is not None:
                        pt_inv = PowerTransformer(method='yeo-johnson', standardize=False); pt_inv.lambdas_ = np.array([lambda_val])
                        y_proc = pt_inv.inverse_transform(self._ensure_2d(y_proc)).flatten()
                    else: logging.warning(f"GRU Yeo-Johnson lambda/object not found. Skipping.")
        return y_proc

    def _apply_domain_clipping(self, y: np.ndarray, transform_info: Dict) -> np.ndarray:
        y_proc = y
        if transform_info and transform_info.get('target_col_original') == 'Radiation':
            y_proc = np.clip(y, DataProcessingConfig.MIN_RADIATION_CLIP, DataProcessingConfig.MAX_RADIATION_CLIP)
        return y_proc

    def _inverse_transform_target(self, y: np.ndarray, target_scaler: Any, 
                                  transform_info: Dict, scalers_dict: Optional[Dict] = None) -> np.ndarray:
        y_2d = self._ensure_2d(y.copy())
        y_unscaled_1d = self._apply_inverse_scaling(y_2d, target_scaler, transform_info)
        y_untransformed_1d = self._apply_inverse_structural_transforms(y_unscaled_1d, transform_info, scalers_dict)
        y_final_1d = self._apply_domain_clipping(y_untransformed_1d, transform_info)
        return y_final_1d
        
    def evaluate(self, X_test_data: np.ndarray, y_test_data: np.ndarray, device: str = "cpu", 
                 target_scaler_object: Optional[Any] = None, 
                 transform_info_dict: Optional[Dict] = None, 
                 scalers_dict: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
        self.eval(); self.to(device)
        actuals_s_np = np.asarray(y_test_data).flatten()
        inputs_t = torch.FloatTensor(X_test_data).to(device)
        preds_s_np: Optional[np.ndarray] = None
        with torch.no_grad(): preds_s_np = self(inputs_t).cpu().numpy().flatten()
        
        logging.info("\n--- GRU Scaled Metrics ---")
        # ... (scaled metrics calculation as before) ...
        mse_s = mean_squared_error(actuals_s_np, preds_s_np)
        rmse_s = np.sqrt(mse_s); r2_s = r2_score(actuals_s_np, preds_s_np)
        mae_s = mean_absolute_error(actuals_s_np, preds_s_np); eps_mape = 1e-8
        mape_s_cap = np.mean(np.clip(np.abs((actuals_s_np - preds_s_np) / (np.abs(actuals_s_np) + eps_mape)), 0, 1.0)) * 100
        logging.info(f"RMSE (scaled): {rmse_s:.4f}, R² (scaled): {r2_s:.4f}, MAE (scaled): {mae_s:.4f}, Capped MAPE (scaled): {mape_s_cap:.2f}%")

        preds_orig, actuals_orig = None, None
        orig_metrics: Dict[str, float] = {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'mape_capped': np.nan}
        if target_scaler_object and transform_info_dict:
            logging.info("\n--- GRU Original Scale Metrics ---")
            try:
                preds_s_2d = preds_s_np.reshape(-1, 1); actuals_s_2d = actuals_s_np.reshape(-1, 1)
                preds_orig = self._inverse_transform_target(preds_s_2d, target_scaler_object, transform_info_dict, scalers_dict).flatten()
                actuals_orig = self._inverse_transform_target(actuals_s_2d, target_scaler_object, transform_info_dict, scalers_dict).flatten()
                # ... (original metrics calculation as before) ...
                if not (np.isnan(preds_orig).any() or np.isnan(actuals_orig).any()):
                    if len(preds_orig) > 0 and len(actuals_orig) > 0 and len(preds_orig) == len(actuals_orig):
                        rmse_o = np.sqrt(mean_squared_error(actuals_orig, preds_orig)); r2_o = r2_score(actuals_orig, preds_orig)
                        mae_o = mean_absolute_error(actuals_orig, preds_orig)
                        mape_o_cap = np.mean(np.clip(np.abs((actuals_orig - preds_orig) / (np.abs(actuals_orig) + eps_mape)), 0, 1.0)) * 100
                        logging.info(f"RMSE (original): {rmse_o:.4f}, R² (original): {r2_o:.4f}, MAE (original): {mae_o:.4f}, Capped MAPE (original): {mape_o_cap:.2f}%")
                        orig_metrics = {'rmse': rmse_o, 'r2': r2_o, 'mae': mae_o, 'mape_capped': mape_o_cap}
            except Exception as e: logging.error(f"Error in GRU original scale metrics: {e}")
        
        all_metrics = {'scaled_rmse': rmse_s, 'scaled_r2': r2_s, 'scaled_mae': mae_s, 'scaled_mape_capped': mape_s_cap, **orig_metrics}
        return (preds_s_np, actuals_s_np, preds_orig, actuals_orig, all_metrics)

    def _ensure_batched_input(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray): X_t = torch.FloatTensor(X)
        elif isinstance(X, torch.Tensor): X_t = X.float()
        else: raise TypeError(f"Input X must be np.ndarray or torch.Tensor, got {type(X)}")
        if X_t.ndim == 1: logging.warning("1D input to GRU _ensure_batched_input, assuming seq_len=1."); return X_t.unsqueeze(0).unsqueeze(0)
        elif X_t.ndim == 2: return X_t.unsqueeze(0)
        elif X_t.ndim == 3: return X_t
        else: raise ValueError(f"Input X must be 1D, 2D, or 3D, got {X_t.ndim}D")

    def predict(self, X: np.ndarray, batch_size: int = 32, device: str = "cpu", 
                target_scaler: Optional[Any] = None, 
                transform_info: Optional[Dict] = None, 
                scalers_dict: Optional[Dict] = None) -> np.ndarray:
        self.eval(); self.to(device)
        X_batch_t = self._ensure_batched_input(X).to(device)
        dataset = TensorDataset(X_batch_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_s_list = []
        with torch.no_grad():
            for (inputs_b,) in loader:
                preds_s_list.append(self(inputs_b).cpu().numpy())
        preds_s_np = np.concatenate(preds_s_list, axis=0)
        if target_scaler and transform_info:
            if preds_s_np.ndim == 1: preds_s_np = preds_s_np.reshape(-1,1) # Ensure 2D for inverse
            return self._inverse_transform_target(preds_s_np, target_scaler, transform_info, scalers_dict)
        return preds_s_np.squeeze()
    
    def plot_training_history(self, figsize: Tuple[int, int] = (20, 18), log_scale_loss: bool = True):
        if not self.history['epochs']: logging.info("No GRU training history."); return None
        # ... (Plotting logic similar to LSTM, using self.history) ...
        fig, axes = plt.subplots(4, 2, figsize=figsize); axes = axes.flatten()
        metrics_cfg = [
            ('Loss', ['train_loss', 'val_loss'], log_scale_loss, ['b-', 'r-']),
            ('RMSE (Scaled)', ['val_rmse'], False, ['g-']), ('R² (Scaled)', ['val_r2'], False, ['m-']),
            ('Capped MAPE (Scaled, %)', ['val_mape'], False, ['darkorange-']),
            ('MAE (Scaled)', ['val_mae'], False, ['darkcyan-']), ('Learning Rate', ['lr'], True, ['c-'])
        ]
        for i, (title, keys, ylog, l_colors) in enumerate(metrics_cfg):
            ax = axes[i]
            for k_idx, key in enumerate(keys):
                if key in self.history and self.history[key]:
                    lbl = key.replace('_', ' ').title()
                    ax.plot(self.history['epochs'], self.history[key], l_colors[k_idx], label=lbl)
                    #if 'val' in key and self.history[key]:
                        #best_idx = np.argmin(self.history[key]) if any(s in key for s in ['loss','rmse','mape','mae']) else np.argmax(self.history[key])
                        #if best_idx < len(self.history['epochs']):
                        #     ax.scatter(self.history['epochs'][best_idx], self.history[key][best_idx], s=100, c='gold', marker='*', zorder=5, label=f'Best {lbl.split(" ")[0]}')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(keys[0].split('_')[-1].upper() if '_' in keys[0] else keys[0].upper()); ax.legend(); ax.grid(True)
            if ylog: ax.set_yscale('log')
        ax_r = axes[6]
        if all(k in self.history and self.history[k] for k in ['train_loss','val_loss']) and len(self.history['train_loss'])==len(self.history['val_loss']) > 0:
            tr_loss_s = np.maximum(np.array(self.history['train_loss']),1e-9)
            ratio = np.array(self.history['val_loss'])/tr_loss_s
            ax_r.plot(self.history['epochs'],ratio,'slateblue',label='Val/Train Loss Ratio')
            ax_r.axhline(1.0,color='grey',ls='--');
            if len(ratio)>0 and np.nanmax(ratio)>1.5 : ax_r.text(0.5,0.9,"Ratio > 1.5: Overfitting?",transform=ax_r.transAxes,ha='center',bbox=dict(boxstyle='round',fc='wheat',alpha=0.7))
        else: ax_r.text(0.5,0.5,"Loss data missing",transform=ax_r.transAxes,ha='center')
        ax_r.set_title('Overfitting Indicator'); ax_r.set_xlabel('Epoch'); ax_r.set_ylabel('Ratio'); ax_r.legend(); ax_r.grid(True)
        if len(axes)>7 and fig.axes[-1]==axes[7]: fig.delaxes(axes[7])
        plt.suptitle('GRU Model Training Metrics',fontsize=16,fontweight='bold'); plt.tight_layout(rect=[0,0,1,0.96]); return fig

    def save(self, path: str):
        try:
            torch.save({'model_state_dict': self.state_dict(), 'model_params': self.params, 
                        'history': self.history, 'transform_info': self.transform_info}, path)
            logging.info(f"GRU Model saved to {path}")
        except Exception as e: logging.error(f"Failed to save GRU model: {e}"); raise
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'WeatherGRU':
        try:
            ckpt = torch.load(path, map_location=device); params = ckpt['model_params']
            if not isinstance(params, GRUModelHyperparameters): params = GRUModelHyperparameters(**params)
            model = cls(model_params=params)
            model.load_state_dict(ckpt['model_state_dict'])
            model.history = ckpt.get('history', {key:[] for key in model.history})
            model.transform_info = ckpt.get('transform_info')
            model.to(device); logging.info(f"GRU Model loaded from {path}"); return model
        except Exception as e: logging.error(f"Failed to load GRU model: {e}"); raise
    
    def enable_mc_dropout(self):
        self._mc_dropout_enabled = True
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.train()

    def disable_mc_dropout(self):
        self._mc_dropout_enabled = False
        self.eval()

    def predict_with_uncertainty(self, X: np.ndarray, mc_samples: int = 30, device: str = "cpu", 
                                 target_scaler: Optional[Any] = None, 
                                 transform_info: Optional[Dict] = None, 
                                 scalers_dict: Optional[Dict] = None,
                                 return_samples: bool = False, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        self.enable_mc_dropout()
        try:
            X_batch_t = self._ensure_batched_input(X).to(device)
            preds_s_mc_list = []
            with torch.no_grad():
                for _ in range(mc_samples): preds_s_mc_list.append(self(X_batch_t).cpu().numpy())
            preds_s_mc_np = np.stack(preds_s_mc_list, axis=0)
            
            preds_orig_mc = np.zeros_like(preds_s_mc_np)
            if target_scaler and transform_info:
                for i in range(mc_samples):
                    curr_batch_s = preds_s_mc_np[i,:,:]
                    inverted_s = self._inverse_transform_target(curr_batch_s, target_scaler, transform_info, scalers_dict)
                    preds_orig_mc[i,:,:] = inverted_s.reshape(curr_batch_s.shape)
            else: logging.warning("No scalers/transform for GRU uncertainty. Returning scaled."); preds_orig_mc = preds_s_mc_np
            
            res = {'mean': np.mean(preds_orig_mc, axis=0), 'std': np.std(preds_orig_mc, axis=0),
                   'lower_ci': np.percentile(preds_orig_mc, (alpha/2)*100, axis=0),
                   'upper_ci': np.percentile(preds_orig_mc, (1-alpha/2)*100, axis=0)}
            if return_samples: res['samples'] = preds_orig_mc
            return res
        finally: self.disable_mc_dropout()

    def plot_prediction_with_uncertainty(self, X: np.ndarray, y_true: Optional[np.ndarray] = None, 
                                         mc_samples: int = 30, 
                                         target_scaler: Optional[Any] = None, 
                                         transform_info: Optional[Dict] = None, 
                                         scalers_dict: Optional[Dict] = None,
                                         figsize: Tuple[int, int] = (12, 8), device: str = "cpu", 
                                         alpha: float = 0.05,
                                         indices: Optional[List[int]] = None, 
                                         max_samples_to_plot: int = 5):
        curr_tf_info = transform_info if transform_info is not None else self.transform_info
        X_for_pred = self._ensure_batched_input(X)
        uncertainty_res = self.predict_with_uncertainty(X_for_pred.cpu().numpy(), mc_samples, device, target_scaler, curr_tf_info, scalers_dict, True, alpha)
        
        num_avail_s = X_for_pred.shape[0]
        plot_idxs = np.arange(min(max_samples_to_plot,num_avail_s)) if indices is None else [i for i in np.array(indices) if i < num_avail_s]
        
        y_true_orig: Optional[np.ndarray] = None
        if y_true is not None:
            y_true_arr = np.array(y_true).copy()
            if target_scaler and curr_tf_info:
                y_true_orig = self._inverse_transform_target(y_true_arr.reshape(-1, self.params.output_dim), target_scaler, curr_tf_info, scalers_dict).flatten()
            else: y_true_orig = y_true_arr.flatten()

        if not plot_idxs: logging.info("No valid samples for GRU uncertainty plot."); return plt.figure(figsize=figsize)
        fig, axs = plt.subplots(len(plot_idxs),1,figsize=figsize,squeeze=False)
        for i, s_idx_batch in enumerate(plot_idxs):
            ax=axs[i,0]; mean=uncertainty_res['mean'][s_idx_batch,0]; low=uncertainty_res['lower_ci'][s_idx_batch,0]; upp=uncertainty_res['upper_ci'][s_idx_batch,0]
            mc_samps_item = uncertainty_res['samples'][:,s_idx_batch,0]
            for j in range(min(10,mc_samples)): ax.plot([0],[mc_samps_item[j]],'o',alpha=0.15,c='gray',ms=3)
            ax.errorbar([0],[mean],yerr=[[mean-low],[upp-mean]],fmt='o',c='darkorange',ecolor='sandybrown',capsize=5,label=f'{int((1-alpha)*100)}% CI')
            if y_true_orig is not None and s_idx_batch < len(y_true_orig):
                act=y_true_orig[s_idx_batch]; ax.plot([0],[act],'ro',label='Actual')
                err=abs(mean-act); in_ci = low <= act <= upp
                ax.text(0.02,0.95,f"Err:{err:.2f}",transform=ax.transAxes,fontsize=9)
                ax.text(0.02,0.90,f"InCI:{'Y' if in_ci else 'N'}",transform=ax.transAxes,fontsize=9,color='g' if in_ci else 'r')
            iw=upp-low; ax.text(0.02,0.85,f"IW:{iw:.2f}",transform=ax.transAxes,fontsize=9)
            ax.set_title(f"GRU Pred Sample Idx (Batch): {s_idx_batch}",fontsize=10)
            min_y_p = min([low] + ([y_true_orig[s_idx_batch]] if y_true_orig is not None and s_idx_batch < len(y_true_orig) else [])); min_y_p = min_y_p if min_y_p is not None else 0
            max_y_p = max([upp] + ([y_true_orig[s_idx_batch]] if y_true_orig is not None and s_idx_batch < len(y_true_orig) else [])); max_y_p = max_y_p if max_y_p is not None else 1
            pad=0.2*(max_y_p-min_y_p) if (max_y_p-min_y_p)>1e-6 else 1.0
            ax.set_ylim([max(DataProcessingConfig.MIN_RADIATION_CLIP,min_y_p-pad),max_y_p+pad]); ax.set_xticks([]); ax.set_ylabel("GHI (Original)",fontsize=9)
            if i==0: ax.legend(fontsize=8,loc='best')
        plt.suptitle(f'GRU Preds w {int((1-alpha)*100)}% CI (Original Scale)',fontsize=14,fontweight='bold'); plt.tight_layout(rect=[0,0.03,1,0.95]); return fig

