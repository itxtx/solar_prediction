import json
from io import StringIO
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from solar_prediction.evaluation import baseline_predictions
from solar_prediction.cli import main


def test_cli_compare_smoke(tmp_path, capsys):
    data_path = tmp_path / "sample.csv"
    rows = 60
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="h")
    df = pd.DataFrame(
        {
            "Time": timestamps,
            "GHI": np.maximum(0, np.sin(np.linspace(0, 4 * np.pi, rows)) * 500),
            "temp": np.linspace(5, 25, rows),
            "pressure": [1013.0] * rows,
            "humidity": np.linspace(80, 40, rows),
            "wind_speed": [3.0] * rows,
            "clouds_all": np.linspace(80, 20, rows),
            "rain_1h": [0.0] * rows,
            "snow_1h": [0.0] * rows,
        }
    )
    df.to_csv(data_path, index=False)
    output_dir = tmp_path / "artifacts"

    main(
        [
            "compare",
            "--data",
            str(data_path),
            "--epochs",
            "1",
            "--hidden-dim",
            "4",
            "--batch-size",
            "16",
            "--output",
            str(output_dir),
            "--seasonal-lag",
            "24",
            "--quiet",
        ]
    )

    output = capsys.readouterr().out
    assert "model,rmse,mae,r2,capped_mape" in output
    assert "persistence" in output
    assert "seasonal_naive" in output
    assert "lstm" in output
    assert "gru" in output
    assert (output_dir / "comparison_metrics.json").exists()
    assert (output_dir / "lstm_model.pt").exists()
    assert (output_dir / "lstm_metadata.json").exists()
    assert (output_dir / "gru_model.pt").exists()
    assert (output_dir / "gru_metadata.json").exists()


@pytest.mark.parametrize(
    "horizon,lag,expected",
    [
        (4, 3, [2, 3, 4]),
        (3, 3, [4, 5, 6]),
        (1, 3, [2, 3, 4]),
        (4, 100, [4, 5, 6]),
        (4, 9, [4, 0, 1]),
    ],
)
def test_baseline_predictions_use_forecast_horizon(horizon, lag, expected):
    actual, baselines = baseline_predictions(
        np.arange(12),
        np.arange(4, 7) + horizon,
        horizon_steps=horizon,
        seasonal_lag=lag,
    )

    np.testing.assert_array_equal(actual, np.arange(4, 7) + horizon)
    np.testing.assert_array_equal(baselines["persistence"], np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(baselines["seasonal_naive"], expected)


@pytest.mark.parametrize("lag", ["0", "-1"])
def test_cli_rejects_nonpositive_seasonal_lag(lag):
    with pytest.raises(SystemExit):
        main(["compare", "--seasonal-lag", lag])


def test_cli_train_seed_reproducibility(tmp_path):
    data_path = tmp_path / "sample.csv"
    rows = 60
    df = pd.DataFrame(
        {
            "Time": pd.date_range("2023-01-01", periods=rows, freq="h"),
            "GHI": np.maximum(0, np.sin(np.linspace(0, 4 * np.pi, rows)) * 500),
            "temp": np.linspace(5, 25, rows),
        }
    )
    df.to_csv(data_path, index=False)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out in (out1, out2):
        main(
            [
                "train",
                "--model",
                "gru",
                "--data",
                str(data_path),
                "--epochs",
                "2",
                "--hidden-dim",
                "4",
                "--batch-size",
                "8",
                "--seed",
                "123",
                "--output",
                str(out),
                "--quiet",
            ]
        )

    meta1 = json.loads((out1 / "gru_metadata.json").read_text())
    meta2 = json.loads((out2 / "gru_metadata.json").read_text())
    assert meta1["seed"] == 123
    assert meta1["metrics"] == meta2["metrics"]


def test_hf_hub_helpers_and_notebooks_use_hf(tmp_path, monkeypatch):
    import types
    from solar_prediction import hf_hub

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    assert hf_hub.resolve_hf_token() == "hf_test_token"
    assert hf_hub.resolve_hf_token("explicit_token") == "explicit_token"

    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls["download"] = kwargs
        out = Path(kwargs["local_dir"]) / kwargs["filename"]
        out.write_text("Time,GHI\n2023-01-01,0.0\n")
        return str(out)

    class FakeHfApi:
        def __init__(self, token=None):
            calls["api_token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs
            return "https://huggingface.co/itxtx/solar-prediction-artifacts/tree/main"

    fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download, HfApi=FakeHfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    downloaded = hf_hub.download_dataset_file(
        "itxtx/solar-prediction-data",
        "solar_weather.csv",
        local_dir=tmp_path / "data",
    )
    assert downloaded.exists()
    assert calls["download"]["repo_type"] == "dataset"
    assert calls["download"]["token"] == "hf_test_token"

    artifacts_dir = tmp_path / "artifacts" / "run1"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "metrics.json").write_text("{}")

    url = hf_hub.upload_artifacts_folder(
        artifacts_dir,
        "itxtx/solar-prediction-artifacts",
        path_in_repo="run1",
    )
    assert "huggingface.co" in url
    assert calls["create_repo"]["repo_type"] == "model"
    assert calls["upload_folder"]["path_in_repo"] == "run1"

    notebooks_dir = Path(__file__).parents[1] / "notebooks"
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        text = nb_path.read_text()
        assert "google.colab import drive" not in text
        assert "download_dataset_file" in text
        assert "upload_artifacts_folder" in text


def _tiny_frame(rows=80):
    return pd.DataFrame(
        {
            "Time": pd.date_range("2023-01-01", periods=rows, freq="h"),
            "GHI": np.maximum(0, np.sin(np.linspace(0, 6 * np.pi, rows)) * 500),
            "temp": np.linspace(5, 25, rows),
            "pressure": [1013.0] * rows,
            "humidity": np.linspace(80, 40, rows),
            "wind_speed": [3.0] * rows,
            "clouds_all": np.linspace(80, 20, rows),
            "rain_1h": [0.0] * rows,
            "snow_1h": [0.0] * rows,
        }
    )


def test_cli_train_applies_architecture_and_sequence_flags(tmp_path):
    import torch

    data_path = tmp_path / "sample.csv"
    _tiny_frame().to_csv(data_path, index=False)
    output_dir = tmp_path / "run"
    main(
        [
            "train",
            "--model",
            "gru",
            "--data",
            str(data_path),
            "--epochs",
            "1",
            "--hidden-dim",
            "4",
            "--batch-size",
            "16",
            "--num-layers",
            "2",
            "--dropout",
            "0.2",
            "--learning-rate",
            "0.0005",
            "--patience",
            "3",
            "--window-size",
            "7",
            "--horizon-steps",
            "2",
            "--output",
            str(output_dir),
            "--quiet",
        ]
    )
    metadata = json.loads((output_dir / "gru_metadata.json").read_text())
    assert metadata["hyperparameters"] == {
        "epochs": 1,
        "hidden_dim": 4,
        "batch_size": 16,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.0005,
        "patience": 3,
        "window_size": 7,
        "target_time_features": False,
        "horizon_steps": 2,
        "seed": 42,
    }
    assert metadata["transform_info"]["split_metadata"]["window_size"] == 7

    from solar_prediction.checkpointing import load_forecaster

    model, preprocessor = load_forecaster(str(output_dir / "gru_model.pt"), "cpu")
    assert model.gru.num_layers == 2
    assert model.gru.dropout == pytest.approx(0.2)
    assert preprocessor.sequence_cfg.window_size == 7
    assert isinstance(model, torch.nn.Module)


def test_model_and_configs_uses_real_early_stopping_patience():
    from solar_prediction.cli import _model_and_configs

    _, train_cfg = _model_and_configs("gru", 5, epochs=100, hidden_dim=4, batch_size=8)
    assert train_cfg.patience == 10
    assert train_cfg.patience < train_cfg.epochs
    _, train_cfg = _model_and_configs(
        "lstm", 5, 100, 4, 8, patience=4, learning_rate=0.01, num_layers=2, dropout=0.0
    )
    assert (train_cfg.patience, train_cfg.learning_rate) == (4, 0.01)


@pytest.mark.parametrize(
    "flag,value",
    [("--dropout", "1.0"), ("--dropout", "-0.1"), ("--learning-rate", "0"), ("--window-size", "0")],
)
def test_cli_rejects_invalid_model_flags(flag, value):
    with pytest.raises(SystemExit):
        main(["train", flag, value])
