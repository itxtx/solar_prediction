"""GRU network construction; training and inference are shared."""

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
)


def create_gru_model_hyperparameters_from_config(
    input_dim: int, config_override: Optional[dict] = None
) -> "GRUModelHyperparameters":
    """Create GRUModelHyperparameters from centralized config with optional overrides."""
    config = get_config()
    gru_config = config.models.gru

    params = {
        "input_dim": input_dim,
        "hidden_dim": gru_config.hidden_dim,
        "num_layers": gru_config.num_layers,
        "output_dim": gru_config.output_dim,
        "dropout_prob": gru_config.dropout_prob,
        "bidirectional": gru_config.bidirectional,
    }

    if config_override:
        params.update(config_override)

    return GRUModelHyperparameters(**params)


@dataclass
class GRUModelHyperparameters:  # Specific to GRU
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout_prob: float = 0.3
    bidirectional: bool = False

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "num_layers", "output_dim"):
            _positive_int(getattr(self, name), name)
        if not (0 <= self.dropout_prob <= 1):
            raise ValueError("dropout_prob must be between 0 and 1")


def create_gru_training_config_from_config(config_override=None) -> TrainingConfig:
    """Compatibility factory for the shared training configuration."""
    return training_config_from_config("gru", config_override)


class WeatherGRU(RecurrentForecaster):
    def __init__(self, model_params: GRUModelHyperparameters):
        super().__init__(model_params)

        self.gru = nn.GRU(
            input_size=model_params.input_dim,
            hidden_size=model_params.hidden_dim,
            num_layers=model_params.num_layers,
            batch_first=True,
            dropout=model_params.dropout_prob if model_params.num_layers > 1 else 0,
            bidirectional=model_params.bidirectional,
        )

        fc_input_features = (
            model_params.hidden_dim * 2 if model_params.bidirectional else model_params.hidden_dim
        )
        self.fc = nn.Linear(fc_input_features, model_params.output_dim)
        # Dropout after GRU output processing before FC layer
        self.dropout_fc = nn.Dropout(model_params.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_sequence_input(x, "x", "GRU forward")

        # Let GRU handle hidden state initialization automatically
        _, hidden = self.gru(x)

        out = (
            torch.cat((hidden[-2], hidden[-1]), dim=1) if self.params.bidirectional else hidden[-1]
        )

        out = self.dropout_fc(out)  # Apply dropout before the final fully connected layer
        out = self.fc(out)
        return out
