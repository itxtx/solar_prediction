import numpy as np
import pandas as pd

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
