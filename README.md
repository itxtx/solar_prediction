# Solar Irradiance Forecasting

Reproducible solar Global Horizontal Irradiance (GHI) forecasting with a leakage-safe preprocessing pipeline, simple time-series baselines, and compact PyTorch LSTM/GRU models.

The project is designed as a portfolio-ready ML repository: a reviewer can clone it, install it, run tests, and execute a small end-to-end comparison without needing private local files or a GPU.

## What This Project Shows

- Chronological time-series splitting for train/validation/test.
- Train-only fitted preprocessing for target transforms, low-radiation thresholds, feature scalers, and target scalers.
- Baseline comparison against persistence and seasonal naive forecasts.
- LSTM and GRU sequence models with checkpoint save/load support.
- A script-first workflow for reproducible training, evaluation, and comparison.
- Optional notebooks and experimental SARIMA/TDMC work for deeper analysis.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pytest
solar-predict compare --data data/sample/SolarPrediction_sample.csv --epochs 1 --hidden-dim 8
```

The comparison command prints a CSV-style metrics table:

```text
model,rmse,mae,r2,capped_mape
persistence,...
seasonal_naive,...
lstm,...
gru,...
```

For a slightly longer neural run:

```bash
solar-predict train --model lstm --data data/sample/SolarPrediction_sample.csv --output artifacts/ --epochs 5
solar-predict evaluate --model lstm --checkpoint artifacts/lstm_model.pt --data data/sample/SolarPrediction_sample.csv
```

Generated checkpoints and metrics are written to `artifacts/`, which is intentionally ignored by Git.

## Data

The repository tracks only a small sample dataset:

- `data/sample/SolarPrediction_sample.csv`

Expected columns for the tracked sample:

- `UNIXTime`
- `Data`
- `Time`
- `Radiation`
- `Temperature`
- `Pressure`
- `Humidity`
- `WindDirection(Degrees)`
- `Speed`
- `TimeSunRise`
- `TimeSunSet`

The pipeline also supports the weather-style schema used by the larger local dataset, including `GHI`, `temp`, `pressure`, `humidity`, `wind_speed`, `clouds_all`, `rain_1h`, and `snow_1h`.

Full datasets and trained model weights are not committed. Keep them local, place small reproducible samples under `data/sample/`, and regenerate model artifacts with the CLI.

## Methodology

The core pipeline lives in `solar_prediction.data_prep.prepare_weather_data`.

Important implementation details:

- Rows are sorted chronologically, preferring `UNIXTime` when present.
- Sequence split boundaries are computed before fitted preprocessing.
- Target transforms and scalers are fit only on training target rows.
- Feature scalers and low-radiation thresholds are fit only on training feature rows.
- Validation and test data are transformed with the training-fitted objects.

This avoids the common time-series leakage pattern where scalers or target transforms learn from future validation/test values.

## Models

The portfolio comparison includes:

- **Persistence baseline**: predicts the next value from the most recent observed value.
- **Seasonal naive baseline**: predicts from the same offset in a previous daily cycle when available.
- **LSTM**: compact recurrent neural network implemented in PyTorch.
- **GRU**: compact recurrent neural network implemented in PyTorch.

SARIMA and TDMC implementations remain in the repository as optional/experimental analysis paths. Install classical dependencies with:

```bash
pip install -e ".[classical]"
```

## Repository Layout

```text
solar_prediction/
  cli.py              # train/evaluate/compare command-line workflow
  data_prep.py        # leakage-safe preprocessing and sequence creation
  lstm.py             # PyTorch LSTM model
  gru.py              # PyTorch GRU model
  sarima.py           # optional classical time-series experiments
  tdmc.py             # optional time-dynamic Markov chain experiments
tests/                # unit and smoke tests
data/sample/          # tracked sample data
notebooks/            # supporting reports, not the primary execution path
```

## Development

```bash
pip install -e ".[dev,notebooks]"
ruff check solar_prediction tests
black --check solar_prediction tests
pytest --cov=solar_prediction --cov-report=term-missing
```

CI runs linting, tests, and a fast CLI smoke comparison on the tracked sample dataset.
