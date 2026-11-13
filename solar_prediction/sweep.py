"""Resumable, multi-seed hyperparameter sweeps driven by the ``train`` CLI.

Each run trains in a subprocess (so GPU/host memory is released between runs)
and writes ``<model>_metadata.json`` with validation metrics. Runs whose
metadata already exists are skipped, which makes a sweep resumable after a
Colab disconnect once earlier artifacts are restored from the Hub.

Selection uses only validation metrics, averaged over seeds. Test metrics are
computed afterwards, once, for the frozen winners.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

METRICS = ("rmse", "mae", "r2", "capped_mape")
CONFIG_KEYS = (
    "window_size",
    "target_time_features",
    "hidden_dim",
    "num_layers",
    "learning_rate",
    "dropout",
    "epochs",
    "patience",
    "batch_size",
)


@dataclass(frozen=True)
class RunConfig:
    window_size: int = 12
    target_time_features: bool = False
    hidden_dim: int = 32
    num_layers: int = 1
    learning_rate: float = 1e-3
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 10
    batch_size: int = 256

    @property
    def name(self) -> str:
        return (
            f"w{self.window_size:03d}_tt{int(self.target_time_features)}"
            f"_h{self.hidden_dim:03d}_l{self.num_layers}_lr{self.learning_rate:g}"
            f"_d{self.dropout:g}_b{self.batch_size}_e{self.epochs}_p{self.patience}"
        )


def stage1_grid(base: RunConfig, windows: Iterable[int], target_time: Iterable[bool]):
    """What the model sees: window length x target-time features."""
    return [
        replace(base, window_size=w, target_time_features=t)
        for w, t in itertools.product(windows, target_time)
    ]


def stage2_grid(
    base: RunConfig,
    hidden_dims: Iterable[int],
    num_layers: Iterable[int],
    learning_rates: Iterable[float],
):
    """Model capacity and optimisation on top of the stage-1 winner."""
    return [
        replace(base, hidden_dim=h, num_layers=n, learning_rate=lr)
        for h, n, lr in itertools.product(hidden_dims, num_layers, learning_rates)
    ]


def run_dir(root: Path, horizon_label: str, config: RunConfig, seed: int) -> Path:
    return Path(root) / horizon_label / config.name / f"seed{seed}"


def train_run(
    config: RunConfig,
    *,
    horizon_label: str,
    horizon_steps: int,
    seed: int,
    data_path: Path,
    root: Path,
    device: str,
    model: str = "gru",
    runner=None,
) -> tuple[Path, bool]:
    """Train one (config, horizon, seed) run unless its metadata exists.

    Returns the run directory and whether training actually ran.
    """
    output_dir = run_dir(root, horizon_label, config, seed)
    if (output_dir / f"{model}_metadata.json").exists():
        return output_dir, False
    cmd = [
        sys.executable,
        "-m",
        "solar_prediction.cli",
        "train",
        "--model",
        model,
        "--data",
        str(data_path),
        "--epochs",
        str(config.epochs),
        "--patience",
        str(config.patience),
        "--hidden-dim",
        str(config.hidden_dim),
        "--num-layers",
        str(config.num_layers),
        "--learning-rate",
        str(config.learning_rate),
        "--dropout",
        str(config.dropout),
        "--batch-size",
        str(config.batch_size),
        "--window-size",
        str(config.window_size),
        "--horizon-steps",
        str(horizon_steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output",
        str(output_dir),
        "--quiet",
    ]
    if config.target_time_features:
        cmd.append("--target-time-features")
    result = (runner or _run_cli)(cmd)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_output.txt").write_text(result.stdout or "")
    if result.returncode != 0:
        raise RuntimeError(
            f"Run {output_dir} failed with exit code {result.returncode}:\n{result.stdout}"
        )
    return output_dir, True


def _run_cli(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def collect_runs(root: Path, model: str = "gru") -> pd.DataFrame:
    """One row per finished run, read from metadata files (local or restored)."""
    rows = []
    for path in sorted(Path(root).glob(f"*/*/seed*/{model}_metadata.json")):
        meta = json.loads(path.read_text())
        params = meta.get("hyperparameters")
        if not params:
            continue  # Runs from before hyperparameters were recorded.
        row = {
            "model": model,
            "horizon": path.parents[2].name,
            "run_name": path.parents[1].name,
            "seed": params["seed"],
            "horizon_steps": params["horizon_steps"],
            **{key: params[key] for key in CONFIG_KEYS},
            **{f"val_{m}": meta["metrics"][m] for m in METRICS},
            "checkpoint": str(path.parent / f"{model}_model.pt"),
            "git_commit": meta.get("git_commit"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of validation metrics per (horizon, config), ranked by mean RMSE."""
    if runs.empty:
        return runs
    keys = ["horizon", "horizon_steps", "run_name", *CONFIG_KEYS]
    grouped = runs.groupby(keys, dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        **{f"val_{m}_mean": (f"val_{m}", "mean") for m in METRICS},
        **{f"val_{m}_std": (f"val_{m}", "std") for m in METRICS},
    ).reset_index()
    return summary.sort_values(["horizon_steps", "val_rmse_mean"]).reset_index(drop=True)


def best_config(summary: pd.DataFrame, horizon: str, configs: Iterable[RunConfig]) -> RunConfig:
    """Lowest mean validation RMSE among ``configs`` for one horizon."""
    names = {config.name: config for config in configs}
    candidates = summary[(summary.horizon == horizon) & summary.run_name.isin(names)]
    if candidates.empty:
        raise ValueError(f"No finished runs for horizon {horizon}")
    return names[candidates.sort_values("val_rmse_mean").iloc[0].run_name]


def config_from_row(row) -> RunConfig:
    return RunConfig(**{key: _python(row[key]) for key in CONFIG_KEYS})


def _python(value):
    return value.item() if isinstance(value, np.generic) else value


BASELINES = ("persistence", "seasonal_naive")


def test_predictions(
    checkpoint: str | Path, data_path: str | Path, *, seasonal_lag: int, device="cpu"
) -> pd.DataFrame:
    """Decoded test-split forecasts of a checkpoint and the naive baselines.

    One row per test target: ``timestamp``, ``target_index``, ``actual``,
    ``daylight`` (target row has IsDaylight=1, which is astronomical; falls
    back to observed target > 0), ``model``, ``persistence``, ``seasonal_naive``.
    """
    from .checkpointing import load_forecaster
    from .evaluation import baseline_predictions, evaluate_forecaster

    model, preprocessor = load_forecaster(str(checkpoint), device)
    raw = pd.read_csv(data_path)
    test = preprocessor.split_sequences(raw).test
    result = evaluate_forecaster(model, test, preprocessor, device=device)
    observed = preprocessor.observed_targets(raw)
    actual, baselines = baseline_predictions(
        observed,
        test.target_indices,
        horizon_steps=preprocessor.sequence_cfg.horizon_steps,
        seasonal_lag=seasonal_lag,
    )
    frame, _ = preprocessor._normalize(raw)
    if "IsDaylight" in frame:
        daylight = frame["IsDaylight"].to_numpy()[test.target_indices] > 0.5
    else:
        daylight = actual > 0
    timestamps = (
        frame["Timestamp"].to_numpy()[test.target_indices]
        if "Timestamp" in frame
        else np.full(len(actual), np.datetime64("NaT"))
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_index": test.target_indices,
            "actual": actual,
            "daylight": daylight,
            "model": result.predictions.ravel(),
            **baselines,
        }
    )


def prediction_metrics(predictions: pd.DataFrame, columns=("model", *BASELINES)):
    """``{"all": {...}, "daylight": {...}}`` mapping each column to its metrics."""
    from .evaluation import regression_metrics

    daylight = predictions["daylight"].to_numpy(dtype=bool)
    actual = predictions["actual"].to_numpy()
    return {
        scope: {
            name: regression_metrics(actual[mask], predictions[name].to_numpy()[mask])
            for name in columns
        }
        for scope, mask in (("all", np.ones_like(daylight)), ("daylight", daylight))
    }


def evaluate_test(
    checkpoint: str | Path, data_path: str | Path, *, seasonal_lag: int, device="cpu"
):
    """Test metrics for a checkpoint and the naive baselines on the same targets.

    Returns ``{"all": {...}, "daylight": {...}}``; each maps ``model`` and the
    baseline names to metric dicts.
    """
    return prediction_metrics(
        test_predictions(checkpoint, data_path, seasonal_lag=seasonal_lag, device=device)
    )


def runs_test_metrics(runs: pd.DataFrame, data_path, *, seasonal_lag, device):
    """Score every run in ``runs`` on test; keep forecasts of each group's first seed.

    ``runs`` needs ``horizon``, ``horizon_steps``, ``model``, ``seed`` and
    ``checkpoint`` columns (as from :func:`collect_runs`). Returns long-form
    metrics (``model`` is the network name or a baseline name) and a dict
    ``{horizon: DataFrame}`` of first-seed test forecasts with one column per
    network plus ``actual``, ``daylight`` and the baselines.
    """
    rows, forecasts = [], {}
    for (horizon, steps, network), group in runs.groupby(
        ["horizon", "horizon_steps", "model"], sort=False
    ):
        for _, run in group.sort_values("seed").iterrows():
            predictions = test_predictions(
                run.checkpoint, data_path, seasonal_lag=seasonal_lag, device=device
            )
            if run.seed == group.seed.min():
                merged = forecasts.get(horizon)
                if merged is None:
                    merged = predictions.drop(columns="model")
                elif not np.array_equal(merged.target_index, predictions.target_index):
                    raise ValueError(f"Networks for {horizon} were scored on different targets")
                merged[network] = predictions["model"].to_numpy()
                forecasts[horizon] = merged
            scores = prediction_metrics(predictions)
            for scope, by_name in scores.items():
                for name, metrics in by_name.items():
                    rows.append(
                        {
                            "horizon": horizon,
                            "horizon_steps": steps,
                            "model": network if name == "model" else name,
                            "seed": run.seed,
                            "scope": scope,
                            **{f"test_{m}": metrics[m] for m in METRICS},
                        }
                    )
    metrics = pd.DataFrame(rows)
    # Baselines do not depend on the network or seed; keep one copy each.
    return metrics.drop_duplicates(["horizon", "model", "scope", "seed"]), forecasts


def summarize_test(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of test metrics per (horizon, scope, model) over seeds."""
    return (
        metrics.groupby(["horizon", "horizon_steps", "scope", "model"])
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"test_{m}_mean": (f"test_{m}", "mean") for m in METRICS},
            **{f"test_{m}_std": (f"test_{m}", "std") for m in METRICS},
        )
        .reset_index()
        .sort_values(["horizon_steps", "scope", "test_rmse_mean"])
        .reset_index(drop=True)
    )


def winner_test_metrics(
    winners: pd.DataFrame, runs: pd.DataFrame, data_path, *, seasonal_lag, device
):
    """Evaluate every seed of each frozen winner; return long-form test metrics."""
    rows = []
    for _, winner in winners.iterrows():
        seeds = runs[(runs.horizon == winner.horizon) & (runs.run_name == winner.run_name)]
        for _, run in seeds.iterrows():
            scores = evaluate_test(
                run.checkpoint, data_path, seasonal_lag=seasonal_lag, device=device
            )
            for scope, by_model in scores.items():
                for name, metrics in by_model.items():
                    rows.append(
                        {
                            "horizon": winner.horizon,
                            "horizon_steps": winner.horizon_steps,
                            "run_name": winner.run_name,
                            "seed": run.seed,
                            "scope": scope,
                            "model": name,
                            **{f"test_{m}": metrics[m] for m in METRICS},
                        }
                    )
    return pd.DataFrame(rows)


__all__ = [
    "RunConfig",
    "stage1_grid",
    "stage2_grid",
    "train_run",
    "collect_runs",
    "summarize",
    "best_config",
    "config_from_row",
    "evaluate_test",
    "test_predictions",
    "prediction_metrics",
    "runs_test_metrics",
    "summarize_test",
    "winner_test_metrics",
]
