"""LSTM network construction; training and inference are shared."""

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

from .config import get_config
from .recurrent_forecaster import (
    RecurrentForecaster,
    TrainingConfig,
    training_config_from_config,
    _positive_int,
    CombinedLoss as CombinedLoss,  # Preserve the historical import path.
)


def create_model_hyperparameters_from_config(
    input_dim: int, config_override: Optional[dict] = None
) -> "ModelHyperparameters":
    """Create ModelHyperparameters from centralized config with optional overrides."""
    config = get_config()
    lstm_config = config.models.lstm

    params = {
        "input_dim": input_dim,
        "hidden_dim": lstm_config.hidden_dim,
        "num_layers": lstm_config.num_layers,
        "output_dim": lstm_config.output_dim,
        "dropout_prob": lstm_config.dropout_prob,
    }

    if config_override:
        params.update(config_override)

    return ModelHyperparameters(**params)


@dataclass
class ModelHyperparameters:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1  # Typically 1 for regression
    dropout_prob: float = 0.3

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "num_layers", "output_dim"):
            _positive_int(getattr(self, name), name)
        if self.hidden_dim < 4:
            raise ValueError("LSTM hidden_dim must be >= 4 for the existing head")
        if not (0 <= self.dropout_prob <= 1):
            raise ValueError("dropout_prob must be between 0 and 1")


def create_training_config_from_config(config_override=None) -> TrainingConfig:
    """Compatibility factory for the shared training configuration."""
    return training_config_from_config("lstm", config_override)


class WeatherLSTM(RecurrentForecaster):
    def __init__(self, model_params: ModelHyperparameters):
        super().__init__(model_params)

        self.lstm = nn.LSTM(
            input_size=model_params.input_dim,
            hidden_size=model_params.hidden_dim,
            num_layers=model_params.num_layers,
            batch_first=True,
            dropout=model_params.dropout_prob if model_params.num_layers > 1 else 0,
        )
        self.dropout1 = nn.Dropout(model_params.dropout_prob)
        self.fc1 = nn.Linear(model_params.hidden_dim, model_params.hidden_dim // 4)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(model_params.dropout_prob)
        self.fc2 = nn.Linear(model_params.hidden_dim // 4, model_params.hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(model_params.dropout_prob)
        self.fc3 = nn.Linear(model_params.hidden_dim // 2, model_params.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_sequence_input(x, "x", "LSTM forward")

        # Let LSTM handle hidden state initialization automatically if (h0, c0) are not provided.
        # PyTorch will create zero hidden states by default.
        out, _ = self.lstm(x)

        out = out[:, -1, :]
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        return out
