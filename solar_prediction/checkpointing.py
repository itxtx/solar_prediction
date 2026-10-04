"""
Enhanced checkpointing utility for solar prediction models.

This module provides standardized saving and loading functionality with comprehensive
metadata tracking for model checkpoints. It supports versioning and graceful
handling of older checkpoint formats.
"""

import torch
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

# Import centralized configuration
from .config import get_config


def save_checkpoint(
    model: Any,
    path: str,
    hp: Any,
    train_cfg: Any,
    history: Dict[str, Any],
    metrics: Dict[str, float],
    version: str = "1.1",
    preprocessor: Any = None,
) -> None:
    """
    Save model checkpoint with comprehensive metadata.

    Parameters:
    -----------
    model : Any
        Model object (LSTM or GRU)
    path : str
        Path to save the checkpoint
    hp : Any
        Model hyperparameters (dataclass or dict)
    train_cfg : Any
        Training configuration (dataclass or dict)
    history : Dict[str, Any]
        Training history dictionary
    metrics : Dict[str, float]
        Final model metrics
    version : str
        Checkpoint format version
    """
    try:
        # Ensure path has correct extension
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")

        # Convert dataclasses to dicts if needed
        hp_dict = hp.dict() if hasattr(hp, "dict") else hp if isinstance(hp, dict) else hp.__dict__
        train_cfg_dict = (
            train_cfg.dict()
            if hasattr(train_cfg, "dict")
            else train_cfg if isinstance(train_cfg, dict) else train_cfg.__dict__
        )

        state_dict = model.state_dict()

        # Create comprehensive checkpoint data
        checkpoint_data = {
            "state_dict": state_dict,
            "hyperparameters": hp_dict,
            "training_config": train_cfg_dict,
            "history": history,
            "metrics": metrics,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "model_type": _get_model_type(model),
            "pytorch_version": torch.__version__,
            "config_snapshot": _get_config_snapshot(),
        }

        if preprocessor is not None:
            preprocessor._require_fitted()
            checkpoint_data["preprocessor"] = preprocessor

        # Add model-specific metadata
        if hasattr(model, "params"):
            # LSTM/GRU models
            checkpoint_data["model_params"] = _convert_params_to_dict(model.params)
            if hasattr(model, "transform_info"):
                checkpoint_data["transform_info"] = model.transform_info

        # Save checkpoint
        torch.save(checkpoint_data, str(path))
        logging.info(f"Enhanced checkpoint saved to {path} (version {version})")

    except Exception as e:
        logging.error(f"Failed to save checkpoint to {path}: {e}")
        raise


def load_checkpoint(
    path: str, map_location: Optional[str] = None, strict: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model checkpoint with graceful handling of older versions.

    Parameters:
    -----------
    path : str
        Path to the checkpoint file
    map_location : Optional[str]
        Device to load tensors to (e.g., 'cpu', 'cuda:0')
    strict : bool
        Whether to strictly enforce checkpoint format compatibility

    Returns:
    --------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (checkpoint_data, metadata) tuple where checkpoint_data contains
        model state and metadata contains additional information
    """
    try:
        # Load checkpoint with appropriate settings
        if map_location:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        else:
            checkpoint = torch.load(path, weights_only=False)

        # Handle version compatibility
        version = checkpoint.get("version", "1.0")
        logging.info(f"Loading checkpoint version {version} from {path}")

        # Apply version-specific compatibility fixes
        checkpoint = _apply_version_compatibility(checkpoint, version, strict)

        # Extract metadata
        metadata = {
            "version": version,
            "timestamp": checkpoint.get("timestamp"),
            "model_type": checkpoint.get("model_type"),
            "pytorch_version": checkpoint.get("pytorch_version"),
            "config_snapshot": checkpoint.get("config_snapshot", {}),
            "load_time": datetime.now().isoformat(),
        }

        return checkpoint, metadata

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {path}: {e}")
        raise


def _get_model_type(model) -> str:
    """Determine model type from model object."""
    class_name = model.__class__.__name__
    if "LSTM" in class_name:
        return "LSTM"
    elif "GRU" in class_name:
        return "GRU"
    else:
        return "Unknown"


def _convert_params_to_dict(params) -> Dict[str, Any]:
    """Convert model parameters to dictionary."""
    if hasattr(params, "__dict__"):
        return params.__dict__.copy()
    elif isinstance(params, dict):
        return params.copy()
    else:
        return {}


def _get_config_snapshot() -> Dict[str, Any]:
    """Get snapshot of current configuration."""
    try:
        config = get_config()
        return {
            "data_config": config.data.__dict__ if hasattr(config.data, "__dict__") else {},
            "model_configs": {
                "lstm": (
                    config.models.lstm.__dict__ if hasattr(config.models.lstm, "__dict__") else {}
                ),
                "gru": config.models.gru.__dict__ if hasattr(config.models.gru, "__dict__") else {},
            },
            "evaluation_config": (
                config.evaluation.__dict__ if hasattr(config.evaluation, "__dict__") else {}
            ),
        }
    except Exception as e:
        logging.warning(f"Could not capture config snapshot: {e}")
        return {}


def _apply_version_compatibility(
    checkpoint: Dict[str, Any], version: str, strict: bool
) -> Dict[str, Any]:
    """Apply compatibility fixes for different checkpoint versions."""

    if version == "1.0":
        # Handle legacy checkpoints (missing keys get defaults)
        logging.info("Applying compatibility fixes for version 1.0 checkpoint")

        # Add missing keys with defaults
        defaults = {
            "version": "1.0",
            "timestamp": "unknown",
            "model_type": "unknown",
            "pytorch_version": None,
            "config_snapshot": {},
            "metrics": {},
        }

        for key, default_value in defaults.items():
            if key not in checkpoint:
                checkpoint[key] = default_value
                logging.debug(f"Added missing key '{key}' with default value")

    elif version == "1.1":
        # Current version - no fixes needed
        pass

    else:
        # Unknown version
        if strict:
            raise ValueError(f"Unknown checkpoint version {version} and strict=True")
        else:
            logging.warning(f"Unknown checkpoint version {version}, proceeding with caution")

    return checkpoint


def create_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> Any:
    """
    Factory function to create and load model from checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to the checkpoint file
    device : str
        Device to load model on

    Returns:
    --------
    Any
        Loaded model instance
    """
    checkpoint, metadata = load_checkpoint(checkpoint_path, map_location=device)
    return _model_from_checkpoint(checkpoint, device)


def _model_from_checkpoint(checkpoint, device):
    model_type = checkpoint.get("model_type", "unknown")

    if model_type == "LSTM":
        return _create_lstm_from_checkpoint(checkpoint, device)
    elif model_type == "GRU":
        return _create_gru_from_checkpoint(checkpoint, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_forecaster(checkpoint_path: str, device: str = "cpu") -> Tuple[Any, Any]:
    """Load the model and its independently fitted preprocessing state."""
    from .preprocessing import WeatherPreprocessor

    checkpoint, _ = load_checkpoint(checkpoint_path, map_location=device)
    preprocessor = checkpoint.get("preprocessor")
    if not isinstance(preprocessor, WeatherPreprocessor):
        raise ValueError(
            "Checkpoint has no fitted preprocessor. Retrain and save a new checkpoint."
        )
    preprocessor._require_fitted()
    return _model_from_checkpoint(checkpoint, device), preprocessor


def _create_lstm_from_checkpoint(checkpoint: Dict[str, Any], device: str) -> Any:
    """Create LSTM model from checkpoint."""
    from .lstm import WeatherLSTM, ModelHyperparameters

    # Create model parameters
    model_params_dict = checkpoint.get("model_params", checkpoint.get("hyperparameters", {}))
    model_params = ModelHyperparameters(**model_params_dict)

    # Create model
    model = WeatherLSTM(model_params)
    model.load_state_dict(checkpoint["state_dict"])

    # Load additional attributes
    model.history = checkpoint.get("history", {})
    model.transform_info = checkpoint.get("transform_info")

    model.to(device)
    return model


def _create_gru_from_checkpoint(checkpoint: Dict[str, Any], device: str) -> Any:
    """Create GRU model from checkpoint."""
    from .gru import WeatherGRU, GRUModelHyperparameters

    # Create model parameters
    model_params_dict = checkpoint.get("model_params", checkpoint.get("hyperparameters", {}))
    model_params = GRUModelHyperparameters(**model_params_dict)

    # Create model
    model = WeatherGRU(model_params)
    model.load_state_dict(checkpoint["state_dict"])

    # Load additional attributes
    model.history = checkpoint.get("history", {})
    model.transform_info = checkpoint.get("transform_info")

    model.to(device)
    return model
