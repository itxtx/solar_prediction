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
from .data_prep import prepare_weather_data
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
    return prepare_weather_data(
        _load_dataframe(data_path),
        config.input,
        config.transformation,
        config.features,
        config.scaling,
        sequence_cfg,
    )


def _target_scaler(scalers: Dict[str, Any], transform_info: Dict[str, Any]):
    return scalers.get(transform_info["target_scaler_name"])


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
    if model_name == "lstm":
        return WeatherLSTM.load(str(checkpoint), device=device)
    if model_name == "gru":
        return WeatherGRU.load(str(checkpoint), device=device)
    raise ValueError(f"Unsupported model: {model_name}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    mse = float(np.mean((actual - predicted) ** 2))
    mae = float(np.mean(np.abs(actual - predicted)))
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum((actual - predicted) ** 2)) / denom if denom > 0 else 0.0
    capped_mape = float(
        np.mean(np.clip(np.abs((actual - predicted) / (np.abs(actual) + 1e-8)), 0, 1.0)) * 100
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
        "capped_mape": capped_mape,
    }


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
    seasonal_indices = target_indices - seasonal_lag
    seasonal = np.where(seasonal_indices >= 0, target[seasonal_indices], persistence)

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


def _evaluation_metrics_for_table(
    metrics: Dict[str, float], prefer_original: bool = False
) -> Dict[str, float]:
    if prefer_original and not np.isnan(metrics.get("rmse", np.nan)):
        return {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "r2": float(metrics["r2"]),
            "capped_mape": float(metrics["mape_capped"]),
        }
    return {
        "rmse": float(metrics["scaled_rmse"]),
        "mae": float(metrics["scaled_mae"]),
        "r2": float(metrics["scaled_r2"]),
        "capped_mape": float(metrics["scaled_mape_capped"]),
    }


def command_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    data = _prepare(Path(args.data), horizon_steps=args.horizon_steps)
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = data

    model, train_cfg = _model_and_configs(
        args.model,
        input_dim=X_train.shape[2],
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )
    model.transform_info = transform_info
    model.fit(X_train, y_train, X_val, y_val, train_cfg, device=args.device)

    target_scaler = _target_scaler(scalers, transform_info)
    _, _, _, _, metrics = model.evaluate(
        X_test,
        y_test,
        device=args.device,
        target_scaler_object=target_scaler,
        transform_info_dict=transform_info,
        scalers_dict=scalers,
        batch_size=args.batch_size,
        return_predictions=True,
        plot_results=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.model}_model.pt"
    model.save(str(checkpoint), train_cfg=train_cfg, metrics=metrics, use_enhanced=False)

    metadata = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "data": str(args.data),
        "feature_columns": feature_cols,
        "metrics": metrics,
        "transform_info": {
            key: value for key, value in transform_info.items() if key != "structural_transforms"
        },
    }
    _write_json(output_dir / f"{args.model}_metadata.json", metadata)
    print(f"Saved checkpoint: {checkpoint}")
    _print_metrics_table({args.model: _evaluation_metrics_for_table(metrics, prefer_original=True)})


def command_evaluate(args: argparse.Namespace) -> None:
    data = _prepare(Path(args.data), horizon_steps=args.horizon_steps)
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, _, transform_info = data
    del X_train, X_val, y_train, y_val

    model = _load_model(args.model, Path(args.checkpoint), args.device)
    target_scaler = _target_scaler(scalers, transform_info)
    _, _, _, _, metrics = model.evaluate(
        X_test,
        y_test,
        device=args.device,
        target_scaler_object=target_scaler,
        transform_info_dict=transform_info,
        scalers_dict=scalers,
        batch_size=args.batch_size,
        return_predictions=True,
        plot_results=False,
    )
    _print_metrics_table({args.model: _evaluation_metrics_for_table(metrics, prefer_original=True)})


def command_compare(args: argparse.Namespace) -> None:
    data = _prepare(Path(args.data), horizon_steps=args.horizon_steps)
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, _, transform_info = data

    actual, baselines = _baseline_predictions(Path(args.data), transform_info, args.seasonal_lag)
    metrics_by_model = {name: _metrics(actual, pred) for name, pred in baselines.items()}

    for model_name in ("lstm", "gru"):
        model, train_cfg = _model_and_configs(
            model_name,
            input_dim=X_train.shape[2],
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
        )
        model.transform_info = transform_info
        model.fit(X_train, y_train, X_val, y_val, train_cfg, device=args.device)
        target_scaler = _target_scaler(scalers, transform_info)
        _, _, pred_original, actual_original, eval_metrics = model.evaluate(
            X_test,
            y_test,
            device=args.device,
            target_scaler_object=target_scaler,
            transform_info_dict=transform_info,
            scalers_dict=scalers,
            batch_size=args.batch_size,
            return_predictions=True,
            plot_results=False,
        )
        if pred_original is None or actual_original is None:
            metrics_by_model[model_name] = _evaluation_metrics_for_table(eval_metrics)
        else:
            metrics_by_model[model_name] = _metrics(actual_original, pred_original)

    _print_metrics_table(metrics_by_model)
    if args.output:
        _write_json(Path(args.output) / "comparison_metrics.json", metrics_by_model)


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

    train = subparsers.add_parser("train", help="Train an LSTM or GRU checkpoint")
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
        default=1,
        help="Forecast horizon in rows after the input window",
    )
    evaluate.add_argument("--quiet", action="store_true", help="Suppress INFO logs")
    evaluate.set_defaults(func=command_evaluate)

    compare = subparsers.add_parser("compare", help="Compare baselines with LSTM and GRU")
    compare.add_argument("--output", default=None)
    compare.add_argument("--seasonal-lag", type=int, default=288)
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
