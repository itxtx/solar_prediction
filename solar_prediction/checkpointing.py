"""Model/preprocessor persistence and explicit normalization of legacy checkpoints."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
import logging
from pathlib import Path

import torch

from .config import get_config


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected a config mapping or dataclass, got {type(value).__name__}")


def _get_model_type(model):
    from .lstm import WeatherLSTM
    from .gru import WeatherGRU

    if isinstance(model, WeatherLSTM):
        return "LSTM"
    if isinstance(model, WeatherGRU):
        return "GRU"
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def _validate_bundle(model, preprocessor):
    from .preprocessing import WeatherPreprocessor

    if not isinstance(preprocessor, WeatherPreprocessor):
        raise ValueError("Expected a fitted WeatherPreprocessor")
    preprocessor._require_fitted()
    if model.params.input_dim != len(preprocessor.feature_columns) or model.params.output_dim != 1:
        raise ValueError("Model dimensions do not match the fitted preprocessor")


def save_checkpoint(
    model,
    path,
    hp,
    train_cfg,
    history,
    metrics,
    version="1.1",
    preprocessor=None,
    metrics_scope=None,
):
    """Save weights and metadata, optionally including the fitted preprocessing bundle."""
    if preprocessor is not None:
        _validate_bundle(model, preprocessor)
    config = get_config()
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyperparameters": _as_dict(hp),
        "model_params": _as_dict(model.params),
        "training_config": _as_dict(train_cfg),
        "history": dict(history),
        "metrics": metrics or {},
        "metrics_scope": metrics_scope,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model_type": _get_model_type(model),
        "pytorch_version": str(torch.__version__),
        "config_snapshot": {
            "data_config": config.data.model_dump(),
            "model_configs": config.models.model_dump(),
            "evaluation_config": config.evaluation.model_dump(),
        },
        "transform_info": model.transform_info,
    }
    if preprocessor is not None:
        checkpoint["preprocessor"] = preprocessor
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_forecaster(
    model, preprocessor, path, *, training_config=None, metrics=None, metrics_scope=None
):
    """Persist the complete forecasting bundle used by CLI evaluation."""
    _validate_bundle(model, preprocessor)
    save_checkpoint(
        model,
        path,
        model.params,
        training_config,
        model.history,
        metrics,
        preprocessor=preprocessor,
        metrics_scope=metrics_scope,
    )


def save_model(model, path, *, train_cfg=None, metrics=None, use_enhanced=True, preprocessor=None):
    """Compatibility adapter for the old model.save API."""
    if use_enhanced:
        return save_checkpoint(
            model, path, model.params, train_cfg, model.history, metrics, preprocessor=preprocessor
        )
    if preprocessor is not None:
        raise ValueError(
            "Legacy checkpoint format cannot contain a preprocessor; use enhanced format"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": _as_dict(model.params),
            "history": dict(model.history),
            "transform_info": model.transform_info,
        },
        path,
    )


def _normalize_checkpoint(checkpoint, strict):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping")
    checkpoint = dict(checkpoint)
    if "state_dict" not in checkpoint:
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint has no model state")
        checkpoint["state_dict"] = checkpoint["model_state_dict"]
    version = checkpoint.setdefault("version", "1.0")
    if version not in {"1.0", "1.1"}:
        if strict:
            raise ValueError(f"Unknown checkpoint version {version}")
        logging.warning("Unknown checkpoint version %s", version)
    params = checkpoint.get("model_params", checkpoint.get("hyperparameters"))
    if params is None:
        raise ValueError("Checkpoint has no architecture parameters")
    checkpoint["model_params"] = _as_dict(params)
    families = [
        kind
        for kind in ("LSTM", "GRU")
        if any(key.startswith(kind.lower() + ".") for key in checkpoint["state_dict"])
    ]
    if len(families) != 1:
        raise ValueError("Cannot identify one recurrent architecture from checkpoint keys")
    declared = checkpoint.get("model_type", "unknown")
    if declared not in {None, "unknown", families[0]}:
        raise ValueError("Checkpoint model type disagrees with its state keys")
    checkpoint["model_type"] = families[0]
    return checkpoint


def load_checkpoint(path, map_location=None, strict=False):
    """Load a trusted local checkpoint; detect formats by keys, never by extension."""
    checkpoint = _normalize_checkpoint(
        torch.load(path, map_location=map_location or "cpu", weights_only=False), strict
    )
    metadata = {
        key: checkpoint.get(key)
        for key in ("version", "timestamp", "model_type", "pytorch_version", "config_snapshot")
    }
    metadata["load_time"] = datetime.now().isoformat()
    return checkpoint, metadata


def _model_from_checkpoint(checkpoint, device):
    from .lstm import WeatherLSTM, ModelHyperparameters
    from .gru import WeatherGRU, GRUModelHyperparameters
    from .recurrent_forecaster import TrainingHistory

    cls, params_cls = {
        "LSTM": (WeatherLSTM, ModelHyperparameters),
        "GRU": (WeatherGRU, GRUModelHyperparameters),
    }[checkpoint["model_type"]]
    model = cls(params_cls(**checkpoint["model_params"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.history = TrainingHistory(checkpoint.get("history"))
    model.transform_info = checkpoint.get("transform_info")
    return model.to(device).eval()


def load_model(path, *, device="cpu", strict=False, expected_class=None):
    checkpoint, _ = load_checkpoint(path, map_location=device, strict=strict)
    model = _model_from_checkpoint(checkpoint, device)
    if expected_class is not None and not isinstance(model, expected_class):
        raise ValueError(f"Expected {expected_class.__name__}, got {type(model).__name__}")
    return model


def create_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Compatibility factory for model-only loading."""
    return load_model(checkpoint_path, device=device)


def load_forecaster(checkpoint_path, device="cpu"):
    checkpoint, _ = load_checkpoint(checkpoint_path, map_location=device)
    preprocessor = checkpoint.get("preprocessor")
    if preprocessor is None:
        raise ValueError(
            "Checkpoint has no fitted preprocessor. Retrain and save a new checkpoint."
        )
    model = _model_from_checkpoint(checkpoint, device)
    _validate_bundle(model, preprocessor)
    return model, preprocessor
