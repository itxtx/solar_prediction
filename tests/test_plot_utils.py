import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solar_prediction.evaluation import (
    result_from_predictions,
    resample_evaluation,
    UncertaintyResult,
)
from solar_prediction.plot_utils import plot_evaluation, plot_training_history, plot_uncertainty


@pytest.fixture(autouse=True)
def no_display(monkeypatch):
    def fail():
        raise AssertionError("Plot functions must not display figures")

    monkeypatch.setattr(plt, "show", fail)
    yield
    plt.close("all")


def test_resampling_recomputes_all_metrics_and_labels_them(tmp_path):
    result = result_from_predictions([10, 20, 30, 40], [20, 10, 40, 30], target_name="Radiation")
    times = pd.date_range("2023-01-01", periods=4, freq="15min")
    aggregated, _ = resample_evaluation(result, times, "30min")
    assert result.metrics["mae"] == 10
    assert aggregated.metrics == dict(rmse=0.0, mae=0.0, r2=1.0, capped_mape=0.0)
    fig = plot_evaluation(result, timestamps=times, resample_freq="30min")
    assert "30min means" in fig.axes[0].get_title()
    assert "mae: 0.0000" in fig.axes[-1].texts[0].get_text()
    fig.savefig(tmp_path / "evaluation.png")
    assert (tmp_path / "evaluation.png").stat().st_size > 0


def test_constant_single_row_and_missing_arrays():
    result = result_from_predictions([10], [10])
    fig = plot_evaluation(result)
    assert len(fig.axes) == 4
    result.actuals = None
    with pytest.raises(ValueError, match="retained"):
        plot_evaluation(result)
    with pytest.raises(ValueError):
        plot_training_history({})


def test_history_and_uncertainty_return_figures():
    history = {"epochs": [1, 2], "train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}
    assert len(plot_training_history(history).axes) == 6
    mean = np.array([[1.0], [2.0]])
    result = UncertaintyResult(mean, mean * 0.1, mean - 0.2, mean + 0.2)
    fig = plot_uncertainty(result, actuals=mean)
    assert len(fig.axes[0].lines) == 2
    assert result.units in fig.axes[0].get_ylabel()
