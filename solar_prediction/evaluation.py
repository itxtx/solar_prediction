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
