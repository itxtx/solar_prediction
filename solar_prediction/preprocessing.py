"""Fitted preprocessing independent of forecasting models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .config import (
    get_config,
    DataInputConfig,
    DataTransformationConfig,
    FeatureEngineeringConfig,
    ScalingConfig,
    SequenceConfig,
)
from .data_prep import (
    STD_RADIATION_COL,
    _resolve_input_config,
    _initial_df_setup,
    _engineer_time_features,
    _apply_target_transformations,
    _select_final_features,
    _scale_data,
    _compute_sequence_split_metadata,
)


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    actuals: np.ndarray
    target_indices: np.ndarray

    def slice(self, start, end):
        return SequenceData(
            *(value[start:end] for value in (self.X, self.y, self.actuals, self.target_indices))
        )


@dataclass
class PreparedData:
    train: SequenceData
    val: SequenceData
    test: SequenceData
    preprocessor: "WeatherPreprocessor"
    transform_info: dict

    def as_legacy_tuple(self):
        return (
            self.train.X,
            self.val.X,
            self.test.X,
            self.train.y,
            self.val.y,
            self.test.y,
            self.preprocessor.scalers,
            self.preprocessor.feature_columns,
            self.transform_info,
        )


class WeatherPreprocessor:
    """Fit on training observations; transform later observations without refitting.

    Features use forward filling, with training medians for leading gaps. Target
    observations must be finite: filling them would invent evaluation labels.
    Configurations, feature order, thresholds and inverse transforms are saved.
    """

    def __init__(
        self,
        input_cfg: DataInputConfig,
        transform_cfg: DataTransformationConfig,
        feature_cfg: FeatureEngineeringConfig,
        scaling_cfg: ScalingConfig,
        sequence_cfg: SequenceConfig,
    ):
        (
            self.input_cfg,
            self.transform_cfg,
            self.feature_cfg,
            self.scaling_cfg,
            self.sequence_cfg,
        ) = (
            deepcopy(cfg)
            for cfg in (input_cfg, transform_cfg, feature_cfg, scaling_cfg, sequence_cfg)
        )
        self.domain_cfg = deepcopy(get_config().data)
        self.scalers = {}

    def _normalize(self, data):
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        cfg = _resolve_input_config(data, self.input_cfg)
        if cfg.target_col_original_name not in data:
            raise ValueError(f"Original target column '{cfg.target_col_original_name}' not found.")
        frame, target, _ = _initial_df_setup(data, cfg)
        if not np.isfinite(frame[target].to_numpy(dtype=float)).all():
            raise ValueError("Target observations must be finite; missing labels cannot be filled.")
        # Preserve observations before any label clipping or structural transforms.
        frame[f"{target}_history"] = frame[target]
        return _engineer_time_features(frame, self.feature_cfg, cfg), target

    def fit(self, train_data: pd.DataFrame) -> WeatherPreprocessor:
        """Learn from an explicitly supplied training frame only."""
        self.input_cfg = deepcopy(_resolve_input_config(train_data, self.input_cfg))
        frame, target = self._normalize(train_data)
        indices = np.arange(len(frame))
        self._fit_frame(frame, target, indices, indices)
        return self

    def observed_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Return raw labels in exactly the normalized row order used by sequences."""
        frame, target = self._normalize(data)
        return frame[target].to_numpy(dtype=float)

    def _fit_frame(self, frame, target, feature_indices, target_indices):
        self.target_column = target
        frame, self.transformed_target, self.structural_transforms = _apply_target_transformations(
            frame.copy(), target, self.transform_cfg, target_indices
        )
        training_values = frame[target].iloc[feature_indices]
        self.low_threshold = None
        if self.feature_cfg.create_low_target_indicator and training_values.nunique() > 1:
            self.low_threshold = training_values.quantile(
                self.feature_cfg.low_target_indicator_quantile
            )
        self._add_indicator(frame)
        self.feature_columns = _select_final_features(
            frame, self.feature_cfg, self.transformed_target, target
        )
        self.feature_medians = frame[self.feature_columns].iloc[feature_indices].median()
        self._fill_features(frame)
        if not np.isfinite(frame[self.transformed_target].to_numpy(dtype=float)).all():
            raise ValueError("Target transformation produced non-finite values.")
        _, self.scalers, self.scaled_target = _scale_data(
            frame,
            self.feature_columns,
            self.transformed_target,
            self.scaling_cfg,
            self.structural_transforms,
            feature_indices,
            target_indices,
        )

    def _add_indicator(self, frame):
        if self.low_threshold is not None:
            frame[f"{self.target_column}_is_low"] = (
                frame[self.target_column] < self.low_threshold
            ).astype(float)

    def _fill_features(self, frame):
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing fitted feature columns: {sorted(missing)}")
        values = frame[self.feature_columns].ffill().fillna(self.feature_medians)
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Features must be finite and have observed training values.")
        frame[self.feature_columns] = values

    def _require_fitted(self):
        if not self.scalers:
            raise ValueError("Preprocessor has not been fitted.")

    @property
    def transform_info(self):
        self._require_fitted()
        return {
            "structural_transforms": self.structural_transforms,
            "target_scaler_name": self.scaled_target,
            "target_col_original": self.input_cfg.target_col_original_name,
            "target_col_standardized": self.target_column,
            "target_col_after_structural_transforms": self.transformed_target,
            "feature_columns_used": self.feature_columns,
            "preprocessing_fit_scope": "train_only",
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return scaled columns in the fitted feature order; never fit anything."""
        self._require_fitted()
        frame, target = self._normalize(data)
        if target != self.target_column:
            raise ValueError(f"Expected target '{self.target_column}', got '{target}'.")
        frame, _, _ = _apply_target_transformations(
            frame,
            target,
            self.transform_cfg,
            np.array([], dtype=int),
            fitted_transforms=self.structural_transforms,
        )
        self._add_indicator(frame)
        self._fill_features(frame)
        columns = self.feature_columns + [self.transformed_target]
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError("Target transformation produced non-finite values.")
        scaled = {
            col: self.scalers[col].transform(frame[[col]].to_numpy()).ravel()
            for col in self.feature_columns
        }
        scaled[self.scaled_target] = (
            self.scalers[self.scaled_target]
            .transform(frame[[self.transformed_target]].to_numpy())
            .ravel()
        )
        return pd.DataFrame(scaled, index=frame.index)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Decode scaled predictions, preserving their shape."""
        self._require_fitted()
        return decode_target(
            values,
            self.scalers[self.scaled_target],
            self.structural_transforms,
            target_column=self.target_column,
            domain_cfg=self.domain_cfg,
        )

    def sequences(self, data: pd.DataFrame, *, history: pd.DataFrame | None = None) -> SequenceData:
        """Build windows with saved window/horizon settings.

        Optional history must precede data chronologically. It provides context,
        but its targets are excluded. Without history, the first window+horizon-1
        rows provide context within data itself.
        """
        self._require_fitted()
        history_size = 0 if history is None else len(history)
        combined = data if history is None else pd.concat([history, data], ignore_index=True)
        raw, _ = self._normalize(combined)
        # Sorting must not move history observations into the evaluation frame.
        if history is not None:
            expected, _ = self._normalize(data)
            if not raw.iloc[history_size:].reset_index(drop=True).equals(expected):
                raise ValueError("History must precede evaluation data chronologically.")
        scaled = self.transform(combined)
        window, horizon = self.sequence_cfg.window_size, self.sequence_cfg.horizon_steps
        count = len(scaled) - window - horizon + 1
        if count < 1:
            raise ValueError("Insufficient rows for the saved window size and horizon.")
        features = scaled[self.feature_columns].to_numpy(dtype=np.float32)
        X = np.lib.stride_tricks.sliding_window_view(features, window, axis=0).transpose(0, 2, 1)[
            :count
        ]
        indices = np.arange(window + horizon - 1, len(scaled))
        keep = indices >= history_size
        y = scaled[self.scaled_target].to_numpy(dtype=np.float32)[indices, None]
        actuals = raw[self.target_column].to_numpy(dtype=float)[indices, None]
        return SequenceData(X[keep].copy(), y[keep], actuals[keep], indices[keep] - history_size)

    def split_sequences(self, data: pd.DataFrame) -> PreparedData:
        """Chronologically split windows without learning new preprocessing state."""
        sequences = self.sequences(data)
        metadata = _compute_sequence_split_metadata(len(data), self.sequence_cfg)
        info = dict(
            self.transform_info,
            split_metadata={
                key: value for key, value in metadata.items() if not key.endswith("_indices")
            },
        )
        splits = [
            sequences.slice(*metadata[f"{name}_sequence_range"])
            for name in ("train", "val", "test")
        ]
        return PreparedData(*splits, self, info)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> WeatherPreprocessor:
        preprocessor = pickle.loads(Path(path).read_bytes())
        if not isinstance(preprocessor, cls):
            raise ValueError("File does not contain a WeatherPreprocessor.")
        preprocessor._require_fitted()
        return preprocessor


def prepare_dataset(data, input_cfg, transform_cfg, feature_cfg, scaling_cfg, sequence_cfg):
    """Split first, fit only training window/target rows, then transform all splits."""
    preprocessor = WeatherPreprocessor(
        _resolve_input_config(data, input_cfg),
        transform_cfg,
        feature_cfg,
        scaling_cfg,
        sequence_cfg,
    )
    frame, target = preprocessor._normalize(data)
    metadata = _compute_sequence_split_metadata(len(frame), sequence_cfg)
    train_end = metadata["target_fit_row_range"][1]
    preprocessor._fit_frame(
        frame.iloc[:train_end],
        target,
        metadata["feature_fit_indices"],
        metadata["target_fit_indices"],
    )
    return preprocessor.split_sequences(data)


def decode_target(
    values, target_scaler, structural_transforms, *, target_column="", domain_cfg=None
):
    """Shared shape-preserving inverse used by fitted and legacy preprocessing."""
    domain_cfg = domain_cfg or get_config().data
    values = np.asarray(values)
    result = (
        target_scaler.inverse_transform(values.reshape(-1, 1)).ravel()
        if target_scaler is not None
        else values.astype(float).ravel()
    )
    for transform in reversed(structural_transforms):
        if transform["type"] == "log":
            limit = domain_cfg.max_exp_input
            result = np.exp(np.clip(result, -limit, limit)) - transform["offset"]
        elif transform["type"] == "yeo-johnson":
            result = (
                transform["power_transformer_object"].inverse_transform(result[:, None]).ravel()
            )
        elif transform["type"] == "piecewise":
            params = transform["params"]
            night, moderate = params["night_thresh"], params["moderate_thresh"]
            night_end = np.log1p(night - 1e-6)
            moderate_end = night_end + params["moderate_slope"] * (moderate - night - 1e-6)
            decoded = np.empty_like(result)
            low = result <= night_end
            mid = (result > night_end) & (result <= moderate_end)
            high = result > moderate_end
            decoded[low] = np.expm1(result[low])
            decoded[mid] = night + (result[mid] - night_end) / params["moderate_slope"]
            decoded[high] = moderate + (result[high] - moderate_end) / params["high_slope"]
            result = decoded
    if target_column == STD_RADIATION_COL:
        result = np.clip(result, domain_cfg.min_radiation_clip, domain_cfg.max_radiation_clip)
    return result.reshape(values.shape)


def decode_legacy_target(values, target_scaler, transform_info, scalers_dict=None):
    """Adapt old scaler/transform dictionaries to the fitted decoder's contract."""
    from sklearn.preprocessing import PowerTransformer

    info = transform_info or {}
    transforms = []
    for original in info.get("structural_transforms", info.get("transforms", [])):
        if not original.get("applied", True):
            continue
        transform = dict(original)
        kind = transform["type"]
        if kind == "log":
            transform.setdefault("offset", 0)
        elif kind == "piecewise":
            cfg = get_config().transformation
            defaults = {
                "night_thresh": cfg.piecewise_night_threshold,
                "moderate_thresh": cfg.piecewise_moderate_threshold,
                "moderate_slope": cfg.piecewise_moderate_slope,
                "high_slope": cfg.piecewise_high_slope,
            }
            transform["params"] = defaults | transform.get("params", {})
        elif kind == "yeo-johnson":
            transformer = transform.get("power_transformer_object")
            if transformer is None:
                transformer = (scalers_dict or {}).get("power_transformer_object_for_target")
            if transformer is None:
                if transform.get("lambda") is None:
                    raise ValueError("Missing fitted Yeo-Johnson transformer or lambda")
                transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                transformer.lambdas_ = np.array([transform["lambda"]])
            transform["power_transformer_object"] = transformer
        else:
            raise ValueError(f"Unknown target transform: {kind}")
        transforms.append(transform)
    values = np.asarray(values)
    if target_scaler is not None and getattr(target_scaler, "n_features_in_", 1) > 1:
        name = info.get("target_col_transformed_final", info.get("target_col_original"))
        names = list(getattr(target_scaler, "feature_names_in_", []))
        if name not in names:
            raise ValueError("A multi-feature scaler requires a named target column")
        index = names.index(name)
        full = np.zeros((values.size, target_scaler.n_features_in_))
        full[:, index] = values.ravel()
        values = target_scaler.inverse_transform(full)[:, index].reshape(values.shape)
        target_scaler = None
    return decode_target(
        values, target_scaler, transforms, target_column=info.get("target_col_standardized", "")
    )
