---
library_name: pytorch
tags:
  - time-series-forecasting
  - solar
  - irradiance
  - gru
  - lstm
datasets:
  - itxtx/solar-prediction-data
---

# Solar GHI forecasting artifacts (GRU / LSTM)

Checkpoints, metrics, and plots from the [solar_prediction](https://github.com/itxtx/solar_prediction) Colab notebooks. Small single-layer GRU and LSTM models forecast Global Horizontal Irradiance (GHI) at one site, 15 minutes to 24 hours ahead, from 15-minute weather and irradiance observations.

These are research artifacts from a portfolio project, not a production forecaster.

## Contents

| Folder | Notebook | What it holds |
| --- | --- | --- |
| `colab_full_run_actual_data/` | `colab_full_run_actual_data.ipynb` | One 15-minute-ahead LSTM and GRU, with baseline comparison |
| `colab_horizon_sweep_actual_data/{15min,1h,4h,24h}/` | `colab_horizon_sweep_actual_data.ipynb` | LSTM and GRU per horizon; `horizon_sweep_metrics.csv` |
| `colab_gru_tuning_actual_data/{4h,24h}/` | earlier version of `colab_gru_tuning_actual_data.ipynb` | Five GRU configs per horizon (one seed), validation metrics, test metrics for the validation winner |
| `colab_gru_two_stage_sweep/` (when present) | current `colab_gru_tuning_actual_data.ipynb` | Multi-seed sweep over input window, target-time features, and model size |

Each run folder has `<model>_model.pt` (weights plus the fitted preprocessor) and `<model>_metadata.json` (features, split ranges, metrics, and the metric scope: `validation` or `test`). Runs from the two-stage sweep also record every hyperparameter and the git commit.

## Data and setup

- **Data:** `solar_weather.csv` from [itxtx/solar-prediction-data](https://huggingface.co/datasets/itxtx/solar-prediction-data). It has 196,776 rows at 15-minute intervals from 2017-01-01 to 2022-08-31, with timestamps in UTC.
- **Target:** GHI at row `t + horizon`. The model sees a window of past rows ending at `t`: GHI history, temperature, pressure, humidity, wind, cloud cover, rain, snow, weather type, and calendar/sun-schedule features. The input window is 12 rows (3 hours) unless a run's metadata says otherwise.
- **Splits:** chronological, 60/20/20 train/validation/test. Train and validation are separated by a gap of `horizon − 1` rows so that no target overlaps. Scalers, thresholds, and target transforms are fitted on the training rows only.
- **Baselines:** persistence (the last observed value) and seasonal naive (the same time on the previous day).

## Results (test split, all hours, RMSE in GHI units)

From `colab_horizon_sweep_actual_data/horizon_sweep_metrics.csv`. One seed, 50 epochs, hidden size 32.

| Horizon | Persistence | Seasonal naive | LSTM | GRU | GRU R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 min | 8.41 | 30.09 | 7.75 | **6.97** | 0.983 |
| 1 h | 21.03 | 30.09 | 14.47 | **14.02** | 0.932 |
| 4 h | 54.80 | 30.09 | 19.59 | **19.26** | 0.872 |
| 24 h | 30.09 | 30.09 | 24.03 | **24.03** | 0.801 |

The earlier GRU tuning picked hidden size 32 for 4 h (test RMSE 19.26) and hidden size 64 for 24 h (test RMSE 23.72). Differences between configs were within about 0.2 RMSE, which is likely seed noise.

## Loading

Checkpoints contain a pickled preprocessor, so loading them runs pickle code (`torch.load(weights_only=False)`). Only load checkpoints from this repository or another source you trust.

```python
from huggingface_hub import hf_hub_download
from solar_prediction.checkpointing import load_forecaster

path = hf_hub_download(
    "itxtx/solar-prediction-artifacts",
    "colab_horizon_sweep_actual_data/4h/gru_model.pt",
)
model, preprocessor = load_forecaster(path, device="cpu")
```

To score a checkpoint on the test split:

```bash
solar-predict evaluate --model gru --checkpoint gru_model.pt --data solar_weather.csv
```

## Limitations

- **Metrics include night.** About half the rows have GHI = 0, which inflates R² and lowers RMSE. The two-stage sweep also reports daylight-only test metrics.
- **Earlier runs used one seed.** Small differences between configs are not meaningful.
- **Earlier runs used a short input window.** Runs outside the two-stage sweep see only the last 3 hours. At 24 h they never see the previous day's values.
- **Earlier runs did not use early stopping.** Before the two-stage sweep, every run trained for the full number of epochs and kept the weights with the best validation score.
- **Earlier runs can't be traced to a commit.** Runs before the two-stage sweep don't record their git commit or full hyperparameters. The `colab_full_run` and 15-minute sweep results differ slightly for this reason, even though their configs are the same.
- **The data has gaps.** There are 10 gaps of 1–5 days. Forecast windows that cross a gap are treated as contiguous, which affects less than 1% of sequences.
- **GHI units are unverified.** The maximum value is 229, well below typical clear-sky values of 800–1000 W/m². Treat RMSE as relative to this dataset's scale.
- **The models cover one site.** They have not been tested at other locations or on other data sources.
