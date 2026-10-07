"""Shared numeric-sequence training and inference for recurrent architectures."""

from contextlib import contextmanager
from dataclasses import dataclass, fields
import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import get_config

logger = logging.getLogger(__name__)


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
    loss_type: str = "mse"
    mse_weight: float = 0.7
    mape_weight: float = 0.3
    value_multiplier: float = 0.01

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience"):
            _positive_int(getattr(self, name), name)
        if self.T_max_cosine is not None:
            _positive_int(self.T_max_cosine, "T_max_cosine")
        for name in (
            "learning_rate",
            "min_lr",
            "weight_decay",
            "mse_weight",
            "mape_weight",
            "value_multiplier",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or not 0 < self.factor < 1:
            raise ValueError("learning_rate must be positive and factor must be between 0 and 1")
        if self.clip_grad_norm is not None and (
            not np.isfinite(self.clip_grad_norm) or self.clip_grad_norm < 0
        ):
            raise ValueError("clip_grad_norm must be finite and nonnegative, or None")
        if self.scheduler_type.lower() not in {"plateau", "cosine"}:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        if self.loss_type.lower() not in {"mse", "mae", "combined", "value_aware"}:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def training_config_from_config(model_name, overrides=None):
    source = getattr(get_config().models, model_name)
    aliases = {"factor": "lr_scheduler_factor", "T_max_cosine": "t_max_cosine"}
    values = {
        field.name: getattr(source, aliases.get(field.name, field.name))
        for field in fields(TrainingConfig)
        if hasattr(source, aliases.get(field.name, field.name))
    }
    return TrainingConfig(**(values | (overrides or {})))


class TrainingHistory(dict):
    """Dictionary-compatible epoch history shared by models and plot consumers."""

    keys_template = (
        "epochs",
        "train_loss",
        "val_loss",
        "val_rmse",
        "val_r2",
        "val_mape",
        "val_mae",
        "lr",
    )

    def __init__(self, values=None):
        super().__init__({key: [] for key in self.keys_template})
        if values is not None:
            self.update(values)

    def record(self, epoch, train_loss, val_loss, metrics, learning_rate):
        values = (
            epoch,
            train_loss,
            val_loss,
            metrics["rmse"],
            metrics["r2"],
            metrics["capped_mape"],
            metrics["mae"],
            learning_rate,
        )
        for key, value in zip(self.keys_template, values):
            self[key].append(value)


class CombinedLoss(nn.Module):
    """Weighted MSE plus capped percentage error, with optional value weighting."""

    def __init__(
        self,
        mse_weight=0.7,
        mape_weight=0.3,
        epsilon=None,
        clip_mape_percentage=None,
        loss_mode="standard",
    ):
        super().__init__()
        cfg = get_config().models.lstm
        self.mse_weight, self.mape_weight = mse_weight, mape_weight
        self.epsilon = cfg.mape_epsilon if epsilon is None else epsilon
        self.clip_mape_fraction = (
            cfg.mape_clip_percentage if clip_mape_percentage is None else clip_mape_percentage
        ) / 100
        if loss_mode not in {"standard", "value_aware"}:
            raise ValueError(f"Unknown loss mode: {loss_mode}")
        self.loss_mode = loss_mode

    def forward(self, y_pred, y_true, value_multiplier=0.01):
        squared_error = (y_true - y_pred).square()
        if self.loss_mode == "value_aware":
            squared_error = squared_error * (1 + y_true.abs() * value_multiplier)
        percentage_error = ((y_true - y_pred) / (y_true.abs() + self.epsilon)).abs()
        return (
            self.mse_weight * squared_error.mean()
            + self.mape_weight * percentage_error.clamp(max=self.clip_mape_fraction).mean()
        )


def _criterion(config):
    if config.loss_type.lower() == "mse":
        return nn.MSELoss()
    if config.loss_type.lower() == "mae":
        return nn.L1Loss()
    return CombinedLoss(
        config.mse_weight,
        config.mape_weight,
        loss_mode="value_aware" if config.loss_type.lower() == "value_aware" else "standard",
    )


def _loss(criterion, outputs, targets, config):
    if isinstance(criterion, CombinedLoss):
        return criterion(outputs, targets, config.value_multiplier)
    return criterion(outputs, targets)


def _scheduler(optimizer, config):
    if config.scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.factor, patience=config.patience // 2, min_lr=config.min_lr
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max_cosine or config.epochs, eta_min=config.min_lr
    )


class RecurrentForecaster(nn.Module):
    """Own orchestration only; subclasses define the encoder, head, and forward."""

    history_template_keys = list(TrainingHistory.keys_template)

    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        self.history = TrainingHistory()
        self.transform_info = None  # Legacy model-only checkpoint metadata.
        self._mc_dropout_enabled = False
        self._manual_dropout_modes = None

    def _validate_sequence_input(self, X, name="X", context="forecast"):
        if X.ndim != 3 or 0 in X.shape:
            raise ValueError(
                f"{context} expected nonempty {name} to be 3D (samples, sequence_length, num_features)"
            )
        if X.shape[-1] != self.params.input_dim:
            raise ValueError(
                f"{context} expected {name} with {self.params.input_dim} features, got {X.shape[-1]}. "
                "The checkpoint was trained with a different feature set; use matching preprocessing."
            )

    def _cpu_features(self, X, *, allow_single=False):
        values = (
            X.detach().to(device="cpu", dtype=torch.float32)
            if torch.is_tensor(X)
            else torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        )
        if allow_single and values.ndim == 2:
            values = values.unsqueeze(0)
        self._validate_sequence_input(values)
        if not torch.isfinite(values).all():
            raise ValueError("Features must be finite")
        return values

    def _dataset(self, X, y):
        X = self._cpu_features(X)
        y = torch.as_tensor(y, dtype=torch.float32, device="cpu")
        if y.ndim != 2 or y.shape != (len(X), self.params.output_dim):
            raise ValueError(
                f"Targets must have shape (samples, output_dim={self.params.output_dim}) matching features"
            )
        if not torch.isfinite(y).all():
            raise ValueError("Targets must be finite")
        return TensorDataset(X, y)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        train_config=None,
        device="cpu",
        memory_tracker=None,
        *,
        config=None,
    ):
        if train_config is not None and config is not None:
            raise ValueError("Supply either config or the legacy train_config argument")
        config = config or train_config or TrainingConfig()
        config.__post_init__()  # Validate configs edited after construction, too.
        train = DataLoader(
            self._dataset(X_train, y_train), batch_size=config.batch_size, shuffle=True
        )
        val = DataLoader(self._dataset(X_val, y_val), batch_size=config.batch_size)
        self.to(device)
        self.history = TrainingHistory()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler, criterion = _scheduler(optimizer, config), _criterion(config)
        use_amp = torch.device(device).type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        best_loss, best_state, stale = float("inf"), None, 0
        if memory_tracker is not None:
            memory_tracker.snapshot("training_start", "before training")
        for epoch in range(config.epochs):
            train_loss = self._train_epoch(train, optimizer, criterion, scaler, config, device)
            val_loss, metrics = self._validation_epoch(val, criterion, config, device)
            self.history.record(
                epoch + 1, train_loss, val_loss, metrics, optimizer.param_groups[0]["lr"]
            )
            if config.scheduler_type.lower() == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            logger.info(
                "%s epoch %d: train_loss=%.6f val_loss=%.6f",
                type(self).__name__,
                epoch + 1,
                train_loss,
                val_loss,
            )
            if memory_tracker is not None:
                memory_tracker.snapshot(f"epoch_{epoch}_end", "after validation")
            if stale >= config.patience:
                break
        self.load_state_dict(best_state)
        self.eval()
        if memory_tracker is not None:
            memory_tracker.snapshot("training_end", "training complete")
        return self

    def _train_epoch(self, loader, optimizer, criterion, scaler, config, device):
        self.train()
        total = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                loss = _loss(criterion, self(X), y, config)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if config.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.parameters(), config.clip_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            total += loss.item() * len(X)
        return total / len(loader.dataset)

    def _validation_epoch(self, loader, criterion, config, device):
        from .evaluation import _MetricAccumulator

        self.eval()
        total, metrics = 0.0, _MetricAccumulator()
        with torch.inference_mode():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                outputs = self(X)
                loss = _loss(criterion, outputs, y, config)
                if not torch.isfinite(loss):
                    raise ValueError("Validation loss is non-finite")
                total += loss.item() * len(X)
                metrics.update(y.cpu().numpy(), outputs.cpu().numpy())
        return total / len(loader.dataset), metrics.metrics()

    @contextmanager
    def _inference_mode(self, *, dropout=False):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            if dropout:
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
            with torch.inference_mode():
                yield
        finally:
            for module, training in modes:
                module.training = training

    def _predict_batches(self, X, batch_size, device):
        return np.concatenate(
            [self(batch.to(device)).cpu().numpy() for batch in X.split(batch_size)]
        )

    def predict(
        self,
        X,
        batch_size=32,
        device="cpu",
        target_scaler=None,
        transform_info=None,
        scalers_dict=None,
    ):
        """Return (rows, outputs) in model units; legacy decoding arguments are delegated."""
        _positive_int(batch_size, "batch_size")
        X = self._cpu_features(X, allow_single=True)
        self.to(device)
        with self._inference_mode(dropout=self._mc_dropout_enabled):
            predictions = self._predict_batches(X, batch_size, device)
        if target_scaler is not None or transform_info is not None:
            from .preprocessing import decode_legacy_target

            return decode_legacy_target(predictions, target_scaler, transform_info, scalers_dict)
        return predictions

    def sample_predictions(self, X, *, mc_samples=30, batch_size=256, device="cpu"):
        """Return Monte Carlo model-unit draws of shape (draws, rows, outputs)."""
        _positive_int(mc_samples, "mc_samples")
        _positive_int(batch_size, "batch_size")
        X = self._cpu_features(X, allow_single=True)
        self.to(device)
        with self._inference_mode(dropout=True):
            return np.stack(
                [self._predict_batches(X, batch_size, device) for _ in range(mc_samples)]
            )

    # Compatibility entrypoints delegate to the module that owns each responsibility.
    def evaluate(self, *args, **kwargs):
        from .evaluation import evaluate_legacy

        return evaluate_legacy(self, *args, **kwargs)

    def _inverse_transform_target(self, y, target_scaler, transform_info, scalers_dict=None):
        from .preprocessing import decode_legacy_target

        return decode_legacy_target(y, target_scaler, transform_info, scalers_dict).ravel()

    def predict_with_uncertainty(
        self,
        X,
        mc_samples=30,
        device="cpu",
        target_scaler=None,
        transform_info=None,
        scalers_dict=None,
        return_samples=False,
        alpha=0.05,
        *,
        batch_size=256,
    ):
        from .evaluation import predict_uncertainty

        return predict_uncertainty(
            self,
            X,
            mc_samples=mc_samples,
            device=device,
            batch_size=batch_size,
            target_scaler=target_scaler,
            transform_info=transform_info,
            scalers_dict=scalers_dict,
            return_samples=return_samples,
            alpha=alpha,
        ).as_dict()

    def enable_mc_dropout(self):
        if not self._mc_dropout_enabled:
            self._manual_dropout_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        self._mc_dropout_enabled = True

    def disable_mc_dropout(self):
        if self._manual_dropout_modes is not None:
            for module, training in self._manual_dropout_modes:
                module.training = training
        self._manual_dropout_modes = None
        self._mc_dropout_enabled = False

    def save(self, path, train_cfg=None, metrics=None, use_enhanced=True, *, preprocessor=None):
        from .checkpointing import save_model

        return save_model(
            self,
            path,
            train_cfg=train_cfg,
            metrics=metrics,
            use_enhanced=use_enhanced,
            preprocessor=preprocessor,
        )

    @classmethod
    def load(cls, path, device="cpu", strict=False):
        from .checkpointing import load_model

        return load_model(path, device=device, strict=strict, expected_class=cls)

    def plot_training_history(self, figsize=(20, 18), log_scale_loss=True):
        from .plot_utils import plot_training_history

        return plot_training_history(self.history, figsize=figsize, log_scale_loss=log_scale_loss)
