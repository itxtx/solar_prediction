"""Command-line workflows for the solar prediction portfolio project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import get_config
from .preprocessing import prepare_dataset
from .evaluation import evaluate_forecaster, regression_metrics as _metrics
from .checkpointing import load_forecaster, save_checkpoint
from .gru import (
    WeatherGRU,
    create_gru_model_hyperparameters_from_config,
    create_gru_training_config_from_config,
)
from .lstm import (
    WeatherLSTM,
    create_model_hyperparameters_from_config,
    create_training_config_from_config,
)

DEFAULT_SAMPLE_DATA = Path("data/sample/SolarPrediction_sample.csv")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _load_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def _copy_sequence_config_with_horizon(sequence_cfg: Any, horizon_steps: int):
    if hasattr(sequence_cfg, "model_copy"):
        return sequence_cfg.model_copy(update={"horizon_steps": horizon_steps})
    return sequence_cfg.copy(update={"horizon_steps": horizon_steps})


def _prepare(data_path: Path, horizon_steps: int = 1):
    config = get_config()
    sequence_cfg = _copy_sequence_config_with_horizon(config.sequences, horizon_steps)
    return prepare_dataset(
        _load_dataframe(data_path),
        config.input,
        config.transformation,
        config.features,
        config.scaling,
        sequence_cfg,
    )


def _model_and_configs(
    model_name: str, input_dim: int, epochs: int, hidden_dim: int, batch_size: int
):
    if model_name == "lstm":
        params = create_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={"hidden_dim": hidden_dim, "num_layers": 1, "dropout_prob": 0.1},
        )
        train_cfg = create_training_config_from_config(
            config_override={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "patience": max(epochs + 1, 3),
                "scheduler_type": "cosine",
            }
        )
        return WeatherLSTM(params), train_cfg

    if model_name == "gru":
        params = create_gru_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={
                "hidden_dim": hidden_dim,
                "num_layers": 1,
                "dropout_prob": 0.1,
                "bidirectional": False,
            },
        )
        train_cfg = create_gru_training_config_from_config(
            config_override={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "patience": max(epochs + 1, 3),
                "scheduler_type": "cosine",
            }
        )
        return WeatherGRU(params), train_cfg

    raise ValueError(f"Unsupported model: {model_name}")


def _load_model(model_name: str, checkpoint: Path, device: str):
    model, preprocessor = load_forecaster(str(checkpoint), device)
    expected = {"lstm": WeatherLSTM, "gru": WeatherGRU}[model_name]
    if not isinstance(model, expected):
        raise ValueError(f"Checkpoint model does not match --model {model_name}.")
    return model, preprocessor


def _raw_target_series(data_path: Path) -> np.ndarray:
    df = _load_dataframe(data_path).copy()
    target_col = "GHI" if "GHI" in df.columns else "Radiation"

    if "UNIXTime" in df.columns:
        df = df.sort_values("UNIXTime")
    elif "Time" in df.columns:
        parsed_time = pd.to_datetime(df["Time"], errors="coerce")
        if parsed_time.notna().any():
            df = df.assign(_parsed_time=parsed_time).sort_values("_parsed_time")

    return df[target_col].ffill().bfill().to_numpy(dtype=float)


def _baseline_predictions(
    data_path: Path,
    transform_info: Dict[str, Any],
    seasonal_lag: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    target = _raw_target_series(data_path)
    split = transform_info["split_metadata"]
    window = split["window_size"]
    horizon_steps = split.get("horizon_steps", 1)
    test_start, test_end = split["test_sequence_range"]
    sequence_indices = np.arange(test_start, test_end)
    target_indices = window + horizon_steps - 1 + sequence_indices

    actual = target[target_indices]
    persistence = target[np.maximum(target_indices - horizon_steps, 0)]
    if seasonal_lag < 1:
        raise ValueError("seasonal_lag must be >= 1")
    # Use the latest seasonal cycle observable at the forecast origin.
    cycles_back = (horizon_steps + seasonal_lag - 1) // seasonal_lag
    seasonal_indices = target_indices - cycles_back * seasonal_lag
    seasonal = persistence.copy()
    available = seasonal_indices >= 0
    seasonal[available] = target[seasonal_indices[available]]

    return actual, {
        "persistence": persistence,
        "seasonal_naive": seasonal,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_metrics_table(metrics_by_model: Dict[str, Dict[str, float]]) -> None:
    print("model,rmse,mae,r2,capped_mape")
    for model_name, metrics in metrics_by_model.items():
        print(
            f"{model_name},{metrics['rmse']:.6f},{metrics['mae']:.6f},"
            f"{metrics['r2']:.6f},{metrics['capped_mape']:.2f}"
        )


def _evaluate_model(model, split, preprocessor, device, batch_size):
    return evaluate_forecaster(
        model, split, preprocessor, device=device, batch_size=batch_size, return_predictions=False
    ).metrics


def _save_model(model, checkpoint, train_cfg, metrics, preprocessor, metrics_scope="test"):
    save_checkpoint(
        model,
        str(checkpoint),
        model.params,
        train_cfg,
        model.history,
        metrics,
        preprocessor=preprocessor,
        metrics_scope=metrics_scope,
    )


def command_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    data = _prepare(Path(args.data), horizon_steps=args.horizon_steps)
    feature_cols, transform_info = data.preprocessor.feature_columns, data.transform_info

    model, train_cfg = _model_and_configs(
        args.model,
        input_dim=data.train.X.shape[2],
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )
    model.fit(data.train.X, data.train.y, data.val.X, data.val.y, train_cfg, device=args.device)

    metrics = _evaluate_model(model, data.val, data.preprocessor, args.device, args.batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.model}_model.pt"
    _save_model(
        model, checkpoint, train_cfg, metrics, data.preprocessor, metrics_scope="validation"
    )

    metadata = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "data": str(args.data),
        "feature_columns": feature_cols,
        "metrics": metrics,
        "metrics_scope": "validation",
        "transform_info": {
            key: value for key, value in transform_info.items() if key != "structural_transforms"
        },
    }
    _write_json(output_dir / f"{args.model}_metadata.json", metadata)
    print(f"Saved checkpoint: {checkpoint}")
    print("Metrics scope: validation")
    _print_metrics_table({args.model: metrics})


def command_evaluate(args: argparse.Namespace) -> None:
    model, preprocessor = _load_model(args.model, Path(args.checkpoint), args.device)
    if (
        args.horizon_steps is not None
        and args.horizon_steps != preprocessor.sequence_cfg.horizon_steps
    ):
        raise ValueError("--horizon-steps must match the saved checkpoint horizon.")
    raw = _load_dataframe(Path(args.data))
    split = (
        preprocessor.split_sequences(raw).test
        if args.evaluation_scope == "test"
        else preprocessor.sequences(raw)
    )
    metrics = _evaluate_model(model, split, preprocessor, args.device, args.batch_size)
    _print_metrics_table({args.model: metrics})


def command_compare(args: argparse.Namespace) -> None:
    data = _prepare(Path(args.data), horizon_steps=args.horizon_steps)
    feature_cols, transform_info = data.preprocessor.feature_columns, data.transform_info
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    actual, baselines = _baseline_predictions(Path(args.data), transform_info, args.seasonal_lag)
    metrics_by_model = {name: _metrics(actual, pred) for name, pred in baselines.items()}

    for model_name in ("lstm", "gru"):
        model, train_cfg = _model_and_configs(
            model_name,
            input_dim=data.train.X.shape[2],
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
        )
        model.fit(data.train.X, data.train.y, data.val.X, data.val.y, train_cfg, device=args.device)
        eval_metrics = _evaluate_model(
            model, data.test, data.preprocessor, args.device, args.batch_size
        )
        metrics_by_model[model_name] = eval_metrics

        if output_dir:
            checkpoint = output_dir / f"{model_name}_model.pt"
            _save_model(model, checkpoint, train_cfg, eval_metrics, data.preprocessor)
            metadata = {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "data": str(args.data),
                "horizon_steps": args.horizon_steps,
                "feature_columns": feature_cols,
                "metrics": eval_metrics,
                "metrics_scope": "test",
                "transform_info": {
                    key: value
                    for key, value in transform_info.items()
                    if key != "structural_transforms"
                },
            }
            _write_json(output_dir / f"{model_name}_metadata.json", metadata)

    _print_metrics_table(metrics_by_model)
    if output_dir:
        _write_json(output_dir / "comparison_metrics.json", metrics_by_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solar irradiance forecasting workflows")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command")

    def add_common_model_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--data", default=str(DEFAULT_SAMPLE_DATA))
        subparser.add_argument("--device", default="cpu")
        subparser.add_argument("--epochs", type=int, default=2)
        subparser.add_argument("--hidden-dim", type=int, default=16)
        subparser.add_argument("--batch-size", type=int, default=32)
        subparser.add_argument(
            "--horizon-steps",
            type=_positive_int,
            default=1,
            help="Forecast horizon in rows after the input window",
        )
        subparser.add_argument("--quiet", action="store_true", help="Suppress INFO logs")

    train = subparsers.add_parser("train", help="Train a checkpoint and report validation metrics")
    train.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    train.add_argument("--output", default="artifacts")
    add_common_model_args(train)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", default=str(DEFAULT_SAMPLE_DATA))
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument(
        "--horizon-steps",
        type=_positive_int,
        default=None,
        help="Defaults to the saved horizon; an explicit value must match it",
    )
    evaluate.add_argument(
        "--evaluation-scope",
        choices=["test", "all"],
        default="test",
        help="Score the test split of a full dataset, or all windowable targets in a held-out file",
    )
    evaluate.add_argument("--quiet", action="store_true", help="Suppress INFO logs")
    evaluate.set_defaults(func=command_evaluate)

    compare = subparsers.add_parser("compare", help="Compare baselines with LSTM and GRU")
    compare.add_argument("--output", default=None)
    compare.add_argument("--seasonal-lag", type=_positive_int, default=288)
    add_common_model_args(compare)
    compare.set_defaults(func=command_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "quiet", False):
        logging.getLogger().setLevel(logging.WARNING)
    if args.func is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
