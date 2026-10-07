"""Decode forecasts and score them independently of network architecture."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .preprocessing import SequenceData, WeatherPreprocessor


@dataclass
class EvaluationResult:
    """Original-unit scores and optional arrays, retaining the (rows, outputs) axes."""

    metrics: dict[str, float]
    predictions: np.ndarray | None
    actuals: np.ndarray | None
    target_indices: np.ndarray
    target_name: str
    units: str = "original"


def _targets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Targets must be finite, nonempty arrays of shape (rows, outputs).")
    return values


class _MetricAccumulator:
    """Merge batch moments without retaining forecasts or losing the output axis."""

    def __init__(self):
        self.count = 0

    def update(self, actuals: np.ndarray, predictions: np.ndarray) -> None:
        actuals, predictions = _targets(actuals), _targets(predictions)
        if actuals.shape != predictions.shape:
            raise ValueError("Actuals and predictions must have matching shapes.")
        if not self.count:
            self.mean = np.zeros(actuals.shape[1])
            self.m2 = np.zeros_like(self.mean)
            self.squared_error = np.zeros_like(self.mean)
            self.absolute_error = np.zeros_like(self.mean)
            self.percentage_error = np.zeros_like(self.mean)
        if actuals.shape[1] != len(self.mean):
            raise ValueError("Output count must be consistent across batches.")
        batch_count = len(actuals)
        batch_mean = actuals.mean(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += ((actuals - batch_mean) ** 2).sum(axis=0)
        self.m2 += delta**2 * self.count * batch_count / total
        self.mean += delta * batch_count / total
        error = np.abs(actuals - predictions)
        self.squared_error += (error**2).sum(axis=0)
        self.absolute_error += error.sum(axis=0)
        self.percentage_error += np.clip(error / (np.abs(actuals) + 1e-8), 0, 1).sum(axis=0)
        self.count = total

    def metrics(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("Cannot score an empty dataset.")
        # Uniform average over outputs; constant outputs have R2=0, as in the CLI.
        r2 = np.zeros_like(self.mean)
        variable = self.m2 > 0
        r2[variable] = 1 - self.squared_error[variable] / self.m2[variable]
        return {
            "rmse": float(np.sqrt(self.squared_error / self.count).mean()),
            "mae": float((self.absolute_error / self.count).mean()),
            "r2": float(r2.mean()),
            "capped_mape": float((self.percentage_error / self.count).mean() * 100),
        }


def regression_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Score matching arrays, averaging each metric uniformly over outputs."""
    accumulator = _MetricAccumulator()
    accumulator.update(actuals, predictions)
    return accumulator.metrics()


def baseline_predictions(observations, target_indices, *, horizon_steps, seasonal_lag):
    """Forecast the supplied target rows using only observations available at each origin."""
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
        for value in (horizon_steps, seasonal_lag)
    ):
        raise ValueError("Horizon and seasonal lag must be positive integers")
    observations = _targets(observations)
    if observations.shape[1] != 1:
        raise ValueError("Baselines require a single observed target")
    target = observations[:, 0]
    indices = np.asarray(target_indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < horizon_steps)
        or np.any(indices >= len(target))
    ):
        raise ValueError("Target indices must have available forecast origins and labels")
    persistence = target[indices - horizon_steps]
    cycles_back = (horizon_steps + seasonal_lag - 1) // seasonal_lag
    seasonal_indices = indices - cycles_back * seasonal_lag
    seasonal = persistence.copy()
    available = seasonal_indices >= 0
    seasonal[available] = target[seasonal_indices[available]]
    return target[indices], {"persistence": persistence, "seasonal_naive": seasonal}


def evaluate_forecaster(
    model: Any,
    data: SequenceData,
    preprocessor: WeatherPreprocessor,
    *,
    batch_size: int = 256,
    device: str = "cpu",
    return_predictions: bool = True,
) -> EvaluationResult:
    """Score raw observed labels against decoded predictions, one batch at a time.

    Label clipping is intentionally not inverted to reconstruct ground truth.
    Decode errors propagate, and metrics are identical whether arrays are retained.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    actuals = _targets(data.actuals)
    if len(data.X) != len(actuals) or np.shape(data.target_indices) != (len(actuals),):
        raise ValueError("Features, actuals, and target indices must have matching row counts.")
    accumulator = _MetricAccumulator()
    predictions = []
    for start in range(0, len(actuals), batch_size):
        stop = start + batch_size
        scaled = model.predict(data.X[start:stop], device=device, batch_size=batch_size)
        decoded = _targets(preprocessor.inverse_target(scaled))
        accumulator.update(actuals[start:stop], decoded)
        if return_predictions:
            predictions.append(decoded)
    return EvaluationResult(
        metrics=accumulator.metrics(),
        predictions=np.concatenate(predictions) if return_predictions else None,
        actuals=actuals.copy() if return_predictions else None,
        target_indices=data.target_indices.copy(),
        target_name=preprocessor.target_column,
    )


def result_from_predictions(
    actuals, predictions, *, target_name="", units="original", target_indices=None
):
    """Build a named result from arrays already expressed in the same units."""
    actuals, predictions = _targets(actuals), _targets(predictions)
    metrics = regression_metrics(actuals, predictions)
    indices = np.arange(len(actuals)) if target_indices is None else np.asarray(target_indices)
    if indices.shape != (len(actuals),):
        raise ValueError("Target indices must match the number of rows")
    return EvaluationResult(
        metrics, predictions.copy(), actuals.copy(), indices.copy(), target_name, units
    )


def resample_evaluation(result, timestamps, frequency):
    """Return a new result and timestamps; all metrics describe the same aggregation."""
    import pandas as pd

    if result.actuals is None or result.predictions is None:
        raise ValueError("Resampling requires retained predictions and actuals")
    timestamps = np.asarray(timestamps)
    if timestamps.shape != (len(result.actuals),):
        raise ValueError("Timestamps must match evaluation rows")
    times = pd.to_datetime(
        timestamps, unit="s" if np.issubdtype(timestamps.dtype, np.number) else None
    )
    if times.isna().any():
        raise ValueError("Timestamps must be valid")
    width = result.actuals.shape[1]
    frame = pd.DataFrame(np.concatenate((result.actuals, result.predictions), axis=1), index=times)
    frame = frame.resample(frequency).mean().dropna()
    aggregated = result_from_predictions(
        frame.iloc[:, :width],
        frame.iloc[:, width:],
        target_name=result.target_name,
        units=result.units,
    )
    return aggregated, frame.index


@dataclass
class UncertaintyResult:
    mean: np.ndarray
    std: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    samples: np.ndarray | None = None
    units: str = "model"

    def as_dict(self):
        result = {name: getattr(self, name) for name in ("mean", "std", "lower_ci", "upper_ci")}
        if self.samples is not None:
            result["samples"] = self.samples
        return result


def predict_uncertainty(
    model,
    X,
    *,
    preprocessor=None,
    mc_samples=30,
    batch_size=256,
    device="cpu",
    alpha=0.05,
    return_samples=False,
    target_scaler=None,
    transform_info=None,
    scalers_dict=None,
):
    """Decode each model draw before computing original-unit intervals."""
    from .preprocessing import decode_legacy_target

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if preprocessor is not None and (target_scaler is not None or transform_info is not None):
        raise ValueError("Use a preprocessor or legacy decoding arguments, not both")
    samples = model.sample_predictions(
        X, mc_samples=mc_samples, batch_size=batch_size, device=device
    )
    units = "model"
    if preprocessor is not None:
        samples = preprocessor.inverse_target(samples)
        units = "original"
    elif target_scaler is not None or transform_info is not None:
        samples = decode_legacy_target(samples, target_scaler, transform_info, scalers_dict)
        units = "original"
    if not np.isfinite(samples).all():
        raise ValueError("Uncertainty predictions must be finite")
    return UncertaintyResult(
        samples.mean(axis=0),
        samples.std(axis=0),
        np.quantile(samples, alpha / 2, axis=0),
        np.quantile(samples, 1 - alpha / 2, axis=0),
        samples if return_samples else None,
        units,
    )


def evaluate_legacy(
    model,
    X_test_data,
    y_test_data,
    device="cpu",
    target_scaler_object=None,
    transform_info_dict=None,
    scalers_dict=None,
    batch_size=256,
    return_predictions=True,
    plot_results=False,
):
    """Adapt the former model.evaluate tuple API; prefer evaluate_forecaster for raw labels."""
    from .preprocessing import decode_legacy_target

    model._validate_sequence_input(X_test_data, "X_test_data", "evaluate")
    actuals = _targets(y_test_data)
    if len(X_test_data) != len(actuals):
        raise ValueError("Features and targets must have matching row counts")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    has_decoder = target_scaler_object is not None or transform_info_dict is not None
    scaled_metrics, original_metrics = _MetricAccumulator(), _MetricAccumulator()
    scaled_parts, decoded_parts, actual_parts = [], [], []
    retain = return_predictions or plot_results
    for start in range(0, len(actuals), batch_size):
        stop = start + batch_size
        predicted = model.predict(X_test_data[start:stop], batch_size=batch_size, device=device)
        scaled_metrics.update(actuals[start:stop], predicted)
        if retain:
            scaled_parts.append(predicted)
        if has_decoder:
            decoded = decode_legacy_target(
                predicted, target_scaler_object, transform_info_dict, scalers_dict
            )
            observed = decode_legacy_target(
                actuals[start:stop], target_scaler_object, transform_info_dict, scalers_dict
            )
            original_metrics.update(observed, decoded)
            if retain:
                decoded_parts.append(decoded)
                actual_parts.append(observed)
    rename = lambda key: "mape_capped" if key == "capped_mape" else key
    metrics = {"scaled_" + rename(key): value for key, value in scaled_metrics.metrics().items()}
    metrics.update(
        {rename(key): value for key, value in original_metrics.metrics().items()}
        if has_decoder
        else {key: np.nan for key in ("rmse", "mae", "r2", "mape_capped")}
    )
    scaled = np.concatenate(scaled_parts) if retain else None
    decoded = np.concatenate(decoded_parts) if retain and has_decoder else None
    observed = np.concatenate(actual_parts) if retain and has_decoder else None
    if plot_results:
        from .plot_utils import plot_evaluation

        plot_evaluation(
            result_from_predictions(
                observed if has_decoder else actuals,
                decoded if has_decoder else scaled,
                units="original" if has_decoder else "model",
            )
        )
    if not return_predictions:
        return None, None, None, None, metrics
    # Preserve legacy flattened tuple returns; the named API always keeps output axes.
    return (
        scaled.ravel(),
        actuals.ravel(),
        (decoded.ravel() if has_decoder else None),
        (observed.ravel() if has_decoder else None),
        metrics,
    )
