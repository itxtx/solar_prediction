import numpy as np
import pandas as pd
import pytest

from solar_prediction.cli import _baseline_predictions
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
def test_baseline_predictions_use_forecast_horizon(tmp_path, horizon, lag, expected):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {"Time": pd.date_range("2023-01-01", periods=12, freq="h"), "GHI": range(12)}
    ).to_csv(data_path, index=False)
    transform_info = {
        "split_metadata": {
            "window_size": 3,
            "horizon_steps": horizon,
            "test_sequence_range": (2, 5),
        }
    }

    actual, baselines = _baseline_predictions(data_path, transform_info, seasonal_lag=lag)

    np.testing.assert_array_equal(actual, np.arange(4, 7) + horizon)
    np.testing.assert_array_equal(baselines["persistence"], np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(baselines["seasonal_naive"], expected)


@pytest.mark.parametrize("lag", ["0", "-1"])
def test_cli_rejects_nonpositive_seasonal_lag(lag):
    with pytest.raises(SystemExit):
        main(["compare", "--seasonal-lag", lag])


def test_tuning_notebook_evaluates_only_validation_winners(tmp_path, monkeypatch):
    notebook = Path(__file__).parents[1] / "notebooks/colab_gru_tuning_actual_data.ipynb"
    sources = ["".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]]
    selection = next(source for source in sources if "best_by_horizon = (" in source)
    tuning = pd.DataFrame(
        {
            "horizon": ["4h", "4h", "24h", "24h"],
            "val_rmse": [10.0, 5.0, 3.0, 9.0],
            "checkpoint": ["a.pt", "b.pt", "c.pt", "d.pt"],
            "batch_size": [16] * 4,
            "output_dir": [str(tmp_path)] * 4,
        }
    )
    evaluated = []

    def evaluate_winner(cmd, **kwargs):
        assert cmd[3] == "evaluate"
        evaluated.append(cmd[cmd.index("--checkpoint") + 1])
        frozen = pd.read_csv(tmp_path / "gru_tuning_best_by_horizon.csv")
        assert set(frozen.checkpoint) == {"b.pt", "c.pt"}
        assert not any(col.startswith("test_") for col in frozen)
        return subprocess.CompletedProcess(
            cmd, 0, "model,rmse,mae,r2,capped_mape\ngru,20,10,0.5,30\n"
        )

    monkeypatch.setattr(subprocess, "run", evaluate_winner)
    exec(
        selection,
        {
            "tuning_metrics": tuning,
            "output_root": tmp_path,
            "data_path": tmp_path / "data.csv",
            "device": "cpu",
            "sys": sys,
            "subprocess": subprocess,
            "Path": Path,
            "pd": pd,
            "StringIO": StringIO,
        },
    )
    assert set(evaluated) == {"b.pt", "c.pt"}
    assert len(evaluated) == 2
    final = pd.read_csv(tmp_path / "gru_tuning_final_test_metrics.csv")
    assert final.test_rmse.tolist() == [20.0, 20.0]
    assert final.val_rmse.tolist() == [3.0, 5.0]


import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
