import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

from solar_prediction import cli
from solar_prediction.checkpointing import load_forecaster, save_checkpoint
from solar_prediction.config import SequenceConfig, get_config
from solar_prediction.preprocessing import WeatherPreprocessor, prepare_dataset


@pytest.fixture
def frame():
    return pd.DataFrame({"Time": pd.date_range("2023-01-01", periods=60, freq="h"),
                         "GHI": np.linspace(1, 500, 60), "temp": np.arange(60, dtype=float)})


def prepare(frame, **transforms):
    cfg = get_config()
    return prepare_dataset(frame, cfg.input, cfg.transformation.model_copy(update=transforms),
                           cfg.features, cfg.scaling, SequenceConfig(window_size=3, horizon_steps=2))


@pytest.mark.parametrize("transforms", [
    {}, {"use_log_transform": True}, {"use_power_transform": True},
    {"use_piecewise_transform_target": True, "use_log_transform": True},
])
def test_saved_preprocessor_reuses_state_and_decodes_targets(frame, transforms, tmp_path, monkeypatch):
    data = prepare(frame, **transforms)
    path = tmp_path / "preprocessor.pkl"
    data.preprocessor.save(path)
    loaded = WeatherPreprocessor.load(path)
    state = pickle.dumps(loaded)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Evaluation must never fit preprocessing")

    for cls in (StandardScaler, MinMaxScaler, PowerTransformer):
        monkeypatch.setattr(cls, "fit", forbid_fit)
    reordered = frame[frame.columns[::-1]]
    sequences = loaded.sequences(reordered)
    np.testing.assert_allclose(sequences.X, data.preprocessor.sequences(frame).X)
    np.testing.assert_allclose(loaded.inverse_target(sequences.y), sequences.actuals, rtol=1e-5)
    assert pickle.dumps(loaded) == state


@pytest.mark.parametrize("transforms", [
    {"use_log_transform": True, "clip_log_transformed_target": True},
    {"use_power_transform": True, "clip_original_target_before_power_transform": True},
])
def test_future_values_cannot_change_training_state(frame, transforms):
    data = prepare(frame, **transforms)
    changed = frame.copy()
    train_end = data.transform_info["split_metadata"]["target_fit_row_range"][1]
    changed.loc[train_end:, ["GHI", "temp"]] = 1e6
    other = prepare(changed, **transforms)
    assert pickle.dumps(other.preprocessor) == pickle.dumps(data.preprocessor)
    np.testing.assert_array_equal(other.train.X, data.train.X)
    np.testing.assert_array_equal(other.train.y, data.train.y)
    assert not np.array_equal(other.test.actuals, data.test.actuals)


def test_feature_boundary_causal_fill_and_missing_labels(frame):
    data = prepare(frame)
    pp = data.preprocessor
    split = data.transform_info["split_metadata"]
    end = split["feature_fit_row_range"][1]
    assert end == split["window_size"] + len(data.train.y) - 1
    assert pp.scalers["Temperature"].mean_[0] == frame.temp.iloc[:end].mean()
    held_out = frame.iloc[-6:].copy()
    held_out.iloc[0, held_out.columns.get_loc("temp")] = np.nan
    scaled = pp.transform(held_out)
    decoded = pp.scalers["Temperature"].inverse_transform(scaled[["Temperature"]].to_numpy())
    assert decoded[0, 0] == pp.feature_medians["Temperature"]
    with pytest.raises(ValueError, match="Missing fitted feature"):
        pp.transform(held_out.drop(columns="temp"))
    held_out.iloc[0, held_out.columns.get_loc("GHI")] = np.nan
    with pytest.raises(ValueError, match="missing labels"):
        pp.transform(held_out)


def test_history_provides_context_without_becoming_labels(frame):
    cfg = get_config()
    pp = WeatherPreprocessor(cfg.input, cfg.transformation, cfg.features, cfg.scaling,
                             SequenceConfig(window_size=3, horizon_steps=2)).fit(frame.iloc[:40])
    result = pp.sequences(frame.iloc[40:], history=frame.iloc[36:40])
    np.testing.assert_array_equal(result.target_indices, np.arange(20))
    np.testing.assert_array_equal(result.actuals.ravel(), frame.GHI.iloc[40:])
    expected = pp.transform(frame.iloc[36:43])[pp.feature_columns].iloc[:3]
    np.testing.assert_allclose(result.X[0], expected, rtol=1e-6)
    with pytest.raises(ValueError, match="History must precede"):
        pp.sequences(frame.iloc[40:], history=frame.iloc[50:54])


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_checkpoint_and_cli_evaluation_use_saved_preprocessor(frame, tmp_path, monkeypatch, capsys, model_name):
    data = prepare(frame, use_log_transform=True, clip_log_transformed_target=True)
    model, train_cfg = cli._model_and_configs(model_name, data.train.X.shape[2], 1, 4, 16)
    path = tmp_path / "model.pt"
    save_checkpoint(model, str(path), model.params, train_cfg, model.history, {},
                    preprocessor=data.preprocessor)
    loaded_model, pp = load_forecaster(str(path))
    np.testing.assert_allclose(loaded_model.predict(data.test.X), model.predict(data.test.X))
    np.testing.assert_allclose(pp.transform(frame), data.preprocessor.transform(frame))
    data_path = tmp_path / "held_out.csv"
    frame.assign(GHI=frame.GHI * 2).to_csv(data_path, index=False)
    def forbid_fit(*args, **kwargs):
        raise AssertionError("CLI evaluation must not fit")
    monkeypatch.setattr(WeatherPreprocessor, "_fit_frame", forbid_fit)
    for scope in ("test", "all"):
        cli.main(["evaluate", "--model", model_name, "--checkpoint", str(path),
                  "--data", str(data_path), "--evaluation-scope", scope, "--quiet"])
    assert capsys.readouterr().out.count(f"{model_name},") == 2
    with pytest.raises(ValueError, match="saved checkpoint horizon"):
        cli.main(["evaluate", "--model", model_name, "--checkpoint", str(path), "--horizon-steps", "1"])
    save_checkpoint(model, str(path), model.params, train_cfg, model.history, {})
    with pytest.raises(ValueError, match="Retrain"):
        load_forecaster(str(path))
