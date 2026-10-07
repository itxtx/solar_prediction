"""Shared contracts must hold for both recurrent architectures."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from solar_prediction.checkpointing import load_model
from solar_prediction.evaluation import predict_uncertainty, regression_metrics
from solar_prediction.gru import GRUModelHyperparameters, WeatherGRU
from solar_prediction.lstm import ModelHyperparameters, WeatherLSTM
from solar_prediction.recurrent_forecaster import (
    RecurrentForecaster,
    TrainingConfig,
    TrainingHistory,
)


@pytest.fixture(
    params=[(WeatherLSTM, ModelHyperparameters), (WeatherGRU, GRUModelHyperparameters)],
    ids=["lstm", "gru"],
)
def model(request):
    cls, params = request.param
    torch.manual_seed(42)
    return cls(params(input_dim=3, hidden_dim=8, num_layers=1, dropout_prob=0.3))


def test_prediction_batches_and_restores_mixed_modes(model):
    X = np.ones((7, 4, 3), dtype=np.float32)
    model.train()
    next(module for module in model.modules() if isinstance(module, nn.Dropout)).eval()
    modes = [module.training for module in model.modules()]
    batches = []
    handle = model.register_forward_pre_hook(
        lambda module, args: batches.append((len(args[0]), args[0].device.type))
    )
    result = model.predict(X, batch_size=2)
    handle.remove()
    assert result.shape == (7, 1)
    assert batches == [(2, "cpu"), (2, "cpu"), (2, "cpu"), (1, "cpu")]
    assert [module.training for module in model.modules()] == modes

    model.enable_mc_dropout()
    model.enable_mc_dropout()
    assert all(module.training for module in model.modules() if isinstance(module, nn.Dropout))
    assert not model.training
    model.disable_mc_dropout()
    assert [module.training for module in model.modules()] == modes


def test_uncertainty_batches_restores_modes_even_on_failure(model, monkeypatch):
    for parameter in model.parameters():
        nn.init.constant_(parameter, 0.1)
    model.train()
    modes = [module.training for module in model.modules()]
    batches = []
    handle = model.register_forward_pre_hook(lambda module, args: batches.append(len(args[0])))
    result = predict_uncertainty(
        model, np.ones((5, 4, 3)), mc_samples=8, batch_size=2, return_samples=True
    )
    handle.remove()
    assert result.samples.shape == (8, 5, 1)
    assert result.mean.shape == result.std.shape == result.lower_ci.shape == (5, 1)
    assert np.any(result.std > 0)
    assert max(batches) == 2 and len(batches) == 24
    assert result.units == "model"
    assert [module.training for module in model.modules()] == modes

    def fail(X):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        model.sample_predictions(np.ones((1, 4, 3)), mc_samples=2)
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize(
    "kwargs", [{"mc_samples": 0}, {"batch_size": 0}, {"alpha": 0}, {"alpha": 1}]
)
def test_uncertainty_rejects_invalid_options(model, kwargs):
    with pytest.raises(ValueError):
        predict_uncertainty(model, np.ones((1, 4, 3)), **kwargs)


@pytest.mark.parametrize(
    "X", [np.empty((0, 4, 3)), np.ones((2, 0, 3)), np.full((1, 4, 3), np.nan), np.ones((2, 4, 2))]
)
def test_predict_rejects_invalid_sequences(model, X):
    with pytest.raises(ValueError):
        model.predict(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": -1},
        {"patience": 0},
        {"loss_type": "typo"},
        {"scheduler_type": "typo"},
        {"learning_rate": np.nan},
    ],
)
def test_training_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_legacy_checkpoint_loading_by_content_and_prediction_parity(model, tmp_path):
    X = np.ones((3, 4, 3), dtype=np.float32)
    expected = model.predict(X)
    path = tmp_path / "legacy.without_pt_extension"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": model.params,
            "history": dict(model.history),
        },
        path,
    )
    loaded = load_model(path, strict=True)
    assert type(loaded) is type(model)
    assert loaded.state_dict().keys() == model.state_dict().keys()
    np.testing.assert_allclose(loaded.predict(X), expected)
    wrong_class = WeatherGRU if isinstance(model, WeatherLSTM) else WeatherLSTM
    with pytest.raises(ValueError, match="Expected"):
        wrong_class.load(path)


def test_bidirectional_gru_uses_both_final_states():
    model = WeatherGRU(
        GRUModelHyperparameters(
            input_dim=3, hidden_dim=8, num_layers=2, dropout_prob=0, bidirectional=True
        )
    ).eval()
    X = torch.randn(4, 6, 3)
    with torch.no_grad():
        _, hidden = model.gru(X)
        expected = model.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        torch.testing.assert_close(model(X), expected)


class TinyForecaster(RecurrentForecaster):
    def __init__(self):
        super().__init__(SimpleNamespace(input_dim=1, output_dim=1))
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        return X[:, -1, :] * self.weight


def test_early_stopping_restores_a_snapshot_of_best_epoch():
    model = TinyForecaster()
    states = []
    losses = iter([3.0, 1.0, 2.0, 4.0])

    def validate(*args):
        states.append(model.weight.detach().clone())
        return next(losses), dict(rmse=0.0, mae=0.0, r2=0.0, capped_mape=0.0)

    model._validation_epoch = validate
    X = np.ones((4, 2, 1), dtype=np.float32)
    model.fit(
        X,
        np.zeros((4, 1)),
        X,
        np.zeros((4, 1)),
        config=TrainingConfig(epochs=5, patience=1, batch_size=4),
    )
    assert isinstance(model.history, TrainingHistory)
    assert model.history["epochs"] == [1, 2, 3]
    assert states[0] != states[1] and states[1] != states[2]
    torch.testing.assert_close(model.weight, states[1])


def test_legacy_evaluation_metrics_do_not_depend_on_retention(model):
    X = np.ones((5, 4, 3), dtype=np.float32)
    y = np.linspace(1.0, 5.0, 5)[:, None]
    scaler = StandardScaler().fit([[10.0], [20.0]])
    args = dict(
        target_scaler_object=scaler, transform_info_dict={"structural_transforms": []}, batch_size=2
    )
    retained = model.evaluate(X, y, **args)
    metrics_only = model.evaluate(X, y, return_predictions=False, **args)
    assert metrics_only[:4] == (None, None, None, None)
    assert metrics_only[4] == pytest.approx(retained[4])
    expected = regression_metrics(scaler.inverse_transform(y), retained[2])
    assert retained[4]["rmse"] == pytest.approx(expected["rmse"])


def test_legacy_evaluation_aggregates_outputs_without_inflating_errors():
    class Identity(RecurrentForecaster):
        def __init__(self):
            super().__init__(SimpleNamespace(input_dim=2, output_dim=2))

        def forward(self, X):
            return X[:, -1, :]

    actuals = np.array([[1.0, 10.0], [3.0, 14.0]])
    X = (actuals + [1.0, 2.0])[:, None, :]
    metrics = Identity().evaluate(X, actuals, return_predictions=False, batch_size=1)[4]
    assert metrics["scaled_rmse"] == metrics["scaled_mae"] == 1.5
    assert metrics["scaled_r2"] == 0


def test_nonlinear_uncertainty_decodes_draws_before_aggregation(model):
    scaler = StandardScaler().fit([[0.0], [1.0]])
    X = np.ones((3, 4, 3))
    result = predict_uncertainty(
        model,
        X,
        mc_samples=8,
        return_samples=True,
        target_scaler=scaler,
        transform_info={"structural_transforms": [{"type": "log", "offset": 1.0}]},
    )
    assert result.units == "original"
    np.testing.assert_allclose(result.mean, result.samples.mean(axis=0))
    np.testing.assert_allclose(result.lower_ci, np.quantile(result.samples, 0.025, axis=0))
