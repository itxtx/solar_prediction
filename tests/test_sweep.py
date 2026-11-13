import contextlib
import io
import json
from pathlib import Path
import subprocess

import matplotlib
import numpy as np
import pandas as pd
import pytest

from solar_prediction import cli, sweep
from solar_prediction.sweep import RunConfig

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).parents[1] / "notebooks/colab_gru_tuning_actual_data.ipynb"


def _frame(days=8):
    timestamps = pd.date_range("2023-06-01", periods=24 * days, freq="h")
    hour = timestamps.hour.to_numpy()
    is_sun = ((hour >= 5) & (hour <= 20)).astype(int)
    rng = np.random.default_rng(1)
    ghi = is_sun * np.sin(np.clip((hour - 5) / 15, 0, 1) * np.pi) * 600
    return pd.DataFrame(
        {
            "Time": timestamps,
            "GHI": ghi * rng.uniform(0.5, 1.0, len(hour)),
            "temp": rng.normal(15, 3, len(hour)),
            "clouds_all": rng.uniform(0, 100, len(hour)),
            "isSun": is_sun,
            "dayLength": 960,
            "SunlightTime/daylength": np.where(is_sun, (hour - 5) / 16, 0.0),
        }
    )


def _in_process_cli(cmd):
    """Run ``python -m solar_prediction.cli ...`` in-process to keep tests fast."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(cmd[3:])
    return subprocess.CompletedProcess(cmd, 0, out.getvalue())


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "data.csv"
    _frame().to_csv(path, index=False)
    return path


def test_run_names_are_unique_per_config():
    configs = sweep.stage1_grid(RunConfig(), [12, 48], [False, True])
    configs += sweep.stage2_grid(RunConfig(), [32, 64], [1, 2], [1e-3, 3e-4])
    assert len({c.name for c in configs}) == len(set(configs))


def test_train_run_passes_every_setting_and_skips_finished_runs(tmp_path, data_path):
    config = RunConfig(
        window_size=48,
        target_time_features=True,
        hidden_dim=64,
        num_layers=2,
        learning_rate=3e-4,
        dropout=0.2,
        epochs=7,
        patience=3,
        batch_size=32,
    )
    calls = []

    def runner(cmd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        (out / "gru_metadata.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "ok")

    kwargs = dict(
        horizon_label="4h",
        horizon_steps=16,
        seed=2,
        data_path=data_path,
        root=tmp_path / "root",
        device="cpu",
        runner=runner,
    )
    path, trained = sweep.train_run(config, **kwargs)
    assert trained and path == tmp_path / "root" / "4h" / config.name / "seed2"
    args = cli.build_parser().parse_args(calls[0][3:])
    assert (args.window_size, args.target_time_features, args.hidden_dim) == (48, True, 64)
    assert (args.num_layers, args.learning_rate, args.dropout) == (2, 3e-4, 0.2)
    assert (args.epochs, args.patience, args.batch_size) == (7, 3, 32)
    assert (args.horizon_steps, args.seed) == (16, 2)

    _, trained = sweep.train_run(config, **kwargs)
    assert not trained and len(calls) == 1


def test_train_run_raises_with_output_on_failure(tmp_path, data_path):
    def runner(cmd):
        return subprocess.CompletedProcess(cmd, 1, "boom")

    with pytest.raises(RuntimeError, match="boom"):
        sweep.train_run(
            RunConfig(),
            horizon_label="4h",
            horizon_steps=1,
            seed=0,
            data_path=data_path,
            root=tmp_path,
            device="cpu",
            runner=runner,
        )


def test_selection_uses_mean_validation_rmse_over_seeds():
    a, b = RunConfig(hidden_dim=32), RunConfig(hidden_dim=64)
    runs = pd.DataFrame(
        [
            {
                "horizon": "4h",
                "run_name": c.name,
                "seed": seed,
                "horizon_steps": 16,
                **{k: getattr(c, k) for k in sweep.CONFIG_KEYS},
                "val_rmse": rmse,
                "val_mae": 1.0,
                "val_r2": 0.5,
                "val_capped_mape": 1.0,
            }
            for c, seed, rmse in [(a, 0, 9.0), (a, 1, 9.0), (b, 0, 5.0), (b, 1, 14.0)]
        ]
    )
    summary = sweep.summarize(runs)
    assert summary.set_index("run_name").loc[b.name, "val_rmse_mean"] == 9.5
    assert sweep.best_config(summary, "4h", [a, b]) == a
    assert sweep.best_config(summary, "4h", [b]) == b
    assert sweep.config_from_row(summary.iloc[0]) == a


def _code_cells():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def test_notebook_freezes_winners_before_any_test_metric():
    sources = _code_cells()
    freeze = next(i for i, s in enumerate(sources) if "best_by_horizon.csv" in s)
    test = next(i for i, s in enumerate(sources) if "winner_test_metrics(" in s)
    assert freeze < test
    assert not any("evaluate_test" in s or "winner_test_metrics(" in s for s in sources[:freeze])


def test_notebook_runs_end_to_end_and_resumes(tmp_path, data_path, monkeypatch):
    monkeypatch.setattr(sweep, "_run_cli", _in_process_cli)
    monkeypatch.chdir(tmp_path)
    sources = _code_cells()
    # Skip clone, pip install, and dataset download/Hub restore cells.
    setup = [
        i for i, s in enumerate(sources) if "%pip" in s or "pyproject.toml" in s or "download_" in s
    ]
    assert setup == [1, 2, 3]
    config_cell, run_cells = sources[0], sources[4:]

    def run_notebook():
        namespace = {}
        exec(config_cell, namespace)
        namespace.update(
            UPLOAD_TO_HF=False,
            HORIZONS={"2h": 2},
            SEEDS=[0, 1],
            SEASONAL_LAG=24,
            BASE=dict(namespace["BASE"], hidden_dim=4, epochs=2, patience=2, batch_size=16),
            STAGE1_WINDOWS=[4, 6],
            STAGE1_TARGET_TIME=[False, True],
            STAGE2_HIDDEN=[4],
            STAGE2_LAYERS=[1, 2],
            STAGE2_LEARNING_RATES=[1e-3],
            data_path=data_path,
            output_root=tmp_path / "out",
            Path=Path,
        )
        namespace["output_root"].mkdir(exist_ok=True)
        for source in run_cells:
            exec(source, namespace)
        return namespace

    trained = []
    original = sweep.train_run

    def counting_train_run(*args, **kwargs):
        result = original(*args, **kwargs)
        trained.append(result[1])
        return result

    monkeypatch.setattr(sweep, "train_run", counting_train_run)
    ns = run_notebook()
    out = tmp_path / "out"
    # 4 stage-1 configs + 2 stage-2 configs (one equals the stage-1 winner) x 2 seeds.
    assert sum(trained) == (4 + 1) * 2
    winners = pd.read_csv(out / "best_by_horizon.csv")
    assert len(winners) == 1 and winners.n_seeds.iloc[0] == 2
    test = pd.read_csv(out / "final_test_metrics_summary.csv")
    assert set(test.scope) == {"all", "daylight"}
    assert set(test.model) == {"model", "persistence", "seasonal_naive"}
    assert (test.n_seeds == 2).all()
    assert (out / "stage1_validation_plot.png").exists()
    assert ns["runs"].git_commit.notna().all()

    trained.clear()
    run_notebook()
    assert not any(trained)


def test_download_artifacts_folder_restores_and_tolerates_missing(tmp_path, monkeypatch):
    import sys
    import types

    from solar_prediction import hf_hub

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs["repo_id"] == "missing/repo":
            raise RuntimeError("404")
        (Path(kwargs["local_dir"]) / "sweep" / "4h").mkdir(parents=True)
        return kwargs["local_dir"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    restored = hf_hub.download_artifacts_folder("itxtx/repo", "sweep", local_dir=tmp_path)
    assert restored == tmp_path / "sweep"
    assert calls[0]["allow_patterns"] == ["sweep/**"]
    assert calls[0]["repo_type"] == "model"
    assert (
        hf_hub.download_artifacts_folder("missing/repo", "sweep", local_dir=tmp_path / "x") is None
    )
