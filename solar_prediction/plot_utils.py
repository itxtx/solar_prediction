"""Figures from history or evaluated results. Callers own display and saving."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import EvaluationResult, UncertaintyResult, resample_evaluation


def plot_training_history(history: Mapping, *, figsize=(14, 10), log_scale_loss=False):
    epochs = history.get("epochs", [])
    if not len(epochs):
        raise ValueError("Training history is empty")
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    panels = [
        ("Loss", ("train_loss", "val_loss")),
        ("RMSE (model units)", ("val_rmse",)),
        ("R²", ("val_r2",)),
        ("Capped MAPE (%)", ("val_mape",)),
        ("MAE (model units)", ("val_mae",)),
        ("Learning rate", ("lr",)),
    ]
    for ax, (title, keys) in zip(axes.flat, panels):
        for key in keys:
            values = history.get(key, [])
            if len(values):
                if len(values) != len(epochs):
                    plt.close(fig)
                    raise ValueError(f"History length for {key} does not match epochs")
                ax.plot(epochs, values, label=key.replace("_", " "))
        if title == "Loss" and log_scale_loss:
            ax.set_yscale("symlog", linthresh=1e-8)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    return fig


def _retained_arrays(result):
    if result.actuals is None or result.predictions is None:
        raise ValueError("Plotting requires an evaluation with retained arrays")
    actuals, predictions = np.asarray(result.actuals), np.asarray(result.predictions)
    if actuals.ndim != 2 or actuals.shape != predictions.shape or 0 in actuals.shape:
        raise ValueError("Plot arrays must have matching nonempty (rows, outputs) shapes")
    if not np.isfinite(actuals).all() or not np.isfinite(predictions).all():
        raise ValueError("Plot arrays must be finite")
    return actuals, predictions


def plot_evaluation(
    result: EvaluationResult, *, timestamps=None, output=0, figsize=(14, 10), resample_freq=None
):
    actuals, predictions = _retained_arrays(result)
    aggregation = ""
    if resample_freq is not None:
        if timestamps is None:
            raise ValueError("Resampling requires timestamps")
        result, timestamps = resample_evaluation(result, timestamps, resample_freq)
        actuals, predictions = _retained_arrays(result)
        aggregation = f"; {resample_freq} means"
    if not 0 <= output < actuals.shape[1]:
        raise ValueError("Output index is out of range")
    x = np.asarray(result.target_indices if timestamps is None else timestamps)
    if x.shape != (len(actuals),):
        raise ValueError("Timestamps must match evaluation rows")
    order = np.argsort(x)
    actual, predicted = actuals[order, output], predictions[order, output]
    x = x[order]
    errors = actual - predicted
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].plot(x, actual, label="Actual")
    axes[0, 0].plot(x, predicted, label="Predicted")
    axes[0, 0].set_title(f"{result.target_name or 'Target'} — {result.units} units{aggregation}")
    axes[0, 0].legend()
    axes[0, 1].scatter(actual, predicted, alpha=0.6)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    margin = max((high - low) * 0.05, 1e-6)
    axes[0, 1].plot([low - margin, high + margin], [low - margin, high + margin], "k--")
    axes[0, 1].set(xlabel="Actual", ylabel="Predicted")
    axes[1, 0].hist(errors, bins=min(30, len(errors)))
    axes[1, 0].set(xlabel="Actual minus predicted", ylabel="Count")
    axes[1, 1].axis("off")
    text = "Metrics (uniform average over outputs)" if actuals.shape[1] > 1 else "Metrics"
    text += (
        aggregation
        + "\n\n"
        + "\n".join(f"{key}: {value:.4f}" for key, value in result.metrics.items())
    )
    axes[1, 1].text(0.05, 0.5, text, va="center")
    fig.tight_layout()
    return fig


def plot_uncertainty(
    result: UncertaintyResult, *, actuals=None, timestamps=None, output=0, figsize=(12, 5)
):
    if result.mean.ndim != 2 or not len(result.mean) or not 0 <= output < result.mean.shape[1]:
        raise ValueError("Uncertainty result must have nonempty (rows, outputs) arrays")
    for array in (result.mean, result.std, result.lower_ci, result.upper_ci):
        if array.shape != result.mean.shape or not np.isfinite(array).all():
            raise ValueError("Uncertainty arrays must have matching finite shapes")
    x = np.arange(len(result.mean)) if timestamps is None else np.asarray(timestamps)
    if x.shape != (len(result.mean),):
        raise ValueError("Timestamps must match prediction rows")
    if actuals is not None:
        actuals = np.asarray(actuals)
        if actuals.ndim == 1:
            actuals = actuals[:, None]
        if actuals.shape != result.mean.shape:
            raise ValueError("Actuals must match uncertainty result shape")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, result.mean[:, output], label="Mean")
    ax.fill_between(
        x, result.lower_ci[:, output], result.upper_ci[:, output], alpha=0.25, label="Interval"
    )
    if actuals is not None:
        ax.plot(x, actuals[:, output], label="Actual")
    ax.set(xlabel="Observation", ylabel=f"Target ({result.units} units)")
    ax.legend()
    fig.tight_layout()
    return fig


def create_evaluation_dashboard(
    result: EvaluationResult, *, timestamps=None, figsize=(14, 10), resample_freq=None
):
    """Dashboard entrypoint; pass a decoded EvaluationResult instead of scalers/raw arrays."""
    return plot_evaluation(
        result, timestamps=timestamps, figsize=figsize, resample_freq=resample_freq
    )
