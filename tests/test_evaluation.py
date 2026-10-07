import numpy as np
import pandas as pd
import pytest

from solar_prediction.cli import _model_and_configs
from solar_prediction.config import SequenceConfig, get_config
from solar_prediction.evaluation import evaluate_forecaster, regression_metrics
from solar_prediction.preprocessing import prepare_dataset


def test_metrics_preserve_output_axis():
    actuals = np.array([[1, 10], [3, 14]])
    metrics = regression_metrics(actuals, actuals + [1, 2])
    assert metrics["rmse"] == 1.5
    assert metrics["mae"] == 1.5
    assert metrics["r2"] == 0


@pytest.mark.parametrize("actuals,predictions", [([], []), ([1], [np.nan]), ([1, 2], [[1, 2]])])
def test_metrics_reject_invalid_or_mismatched_arrays(actuals, predictions):
    with pytest.raises(ValueError):
        regression_metrics(actuals, predictions)


@pytest.fixture
def prepared():
    cfg = get_config()
    frame = pd.DataFrame(
        {
            "Time": pd.date_range("2023-01-01", periods=60, freq="h"),
            "GHI": np.linspace(1, 500, 60),
            "temp": np.arange(60, dtype=float),
        }
    )
    return prepare_dataset(
        frame,
        cfg.input,
        cfg.transformation.model_copy(
            update={"use_log_transform": True, "clip_log_transformed_target": True}
        ),
        cfg.features,
        cfg.scaling,
        SequenceConfig(window_size=3, horizon_steps=2),
    )


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_decoded_evaluation_matches_metrics_only_and_single_row(prepared, model_name):
    data = prepared.test
    pp = prepared.preprocessor
    model, _ = _model_and_configs(model_name, data.X.shape[-1], 1, 4, 16)
    result = evaluate_forecaster(model, data, pp, batch_size=3)
    metrics_only = evaluate_forecaster(model, data, pp, batch_size=2, return_predictions=False)
    assert result.units == "original"
    assert result.target_name == "Radiation"
    assert result.predictions.shape == result.actuals.shape == data.actuals.shape
    np.testing.assert_array_equal(result.actuals, data.actuals)
    np.testing.assert_array_equal(result.target_indices, data.target_indices)
    expected_rmse = np.sqrt(np.mean((data.actuals - result.predictions) ** 2))
    assert result.metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics_only.predictions is metrics_only.actuals is None
    assert metrics_only.metrics == pytest.approx(result.metrics, rel=1e-6)
    single = evaluate_forecaster(model, data.slice(0, 1), pp)
    assert single.predictions.shape == single.actuals.shape == (1, 1)
    assert single.metrics["r2"] == 0


def test_evaluation_bounds_batches_and_propagates_decode_errors(prepared, monkeypatch):
    class Model:
        def predict(self, X, *, device, batch_size):
            assert len(X) <= batch_size == 2
            return np.zeros((len(X), 1))

    pp = prepared.preprocessor
    result = evaluate_forecaster(Model(), prepared.test, pp, batch_size=2)
    expected = np.full_like(prepared.test.actuals, pp.inverse_target(np.array([[0.0]]))[0, 0])
    assert result.metrics == pytest.approx(regression_metrics(prepared.test.actuals, expected))
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_forecaster(Model(), prepared.test, pp, batch_size=0)

    def fail_decode(values):
        raise ValueError("invalid transform")

    monkeypatch.setattr(pp, "inverse_target", fail_decode)
    with pytest.raises(ValueError, match="invalid transform"):
        evaluate_forecaster(Model(), prepared.test, pp, batch_size=2, return_predictions=False)
