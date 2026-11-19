from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb


@dataclass(frozen=True)
class SurrogatePrediction:
    mean: np.ndarray
    std: np.ndarray
    conformal_half_width_90pct: np.ndarray
    ood_distance: np.ndarray
    feature_order: tuple[str, ...]
    target_order: tuple[str, ...]


class SurrogateEnsemble:
    """Version-stable XGBoost ensemble with explicit scaler metadata."""

    def __init__(self, model_directory: str | Path) -> None:
        self.model_directory = Path(model_directory).resolve()
        self.metadata = json.loads(
            (self.model_directory / "metadata.json").read_text(encoding="utf-8")
        )
        self.features = tuple(self.metadata["feature_order"])
        self.targets = tuple(self.metadata["target_order"])
        self.x_mean = np.asarray(self.metadata["x_scaler"]["mean"], dtype=float)
        self.x_scale = np.asarray(self.metadata["x_scaler"]["scale"], dtype=float)
        self.y_mean = np.asarray(self.metadata["y_scaler"]["mean"], dtype=float)
        self.y_scale = np.asarray(self.metadata["y_scaler"]["scale"], dtype=float)
        self.conformal_multiplier = np.asarray(
            self.metadata.get("conformal_90pct_multiplier", [1.645] * len(self.targets)), dtype=float
        )
        self.train_features = np.load(self.model_directory / self.metadata["train_features_file"])
        self.models: list[list[xgb.Booster]] = []
        for member in self.metadata["members"]:
            target_models = []
            for filename in member["models"]:
                booster = xgb.Booster()
                booster.load_model(self.model_directory / filename)
                target_models.append(booster)
            self.models.append(target_models)

    def predict(self, features: np.ndarray) -> SurrogatePrediction:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features, received {values.shape[1]}.")
        if not np.isfinite(values).all():
            raise ValueError("Surrogate features contain NaN or infinity.")
        scaled = (values - self.x_mean) / self.x_scale
        matrix = xgb.DMatrix(scaled)
        member_predictions = []
        for target_models in self.models:
            scaled_prediction = np.column_stack([model.predict(matrix) for model in target_models])
            member_predictions.append(scaled_prediction * self.y_scale + self.y_mean)
        stack = np.stack(member_predictions, axis=0)
        # Eight features and ~7k training rows make exact chunked distance inexpensive.
        distances = []
        for row in scaled:
            distances.append(float(np.sqrt(np.square(self.train_features - row).sum(axis=1)).min()))
        return SurrogatePrediction(
            mean=stack.mean(axis=0),
            std=stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0]),
            conformal_half_width_90pct=(
                stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0])
            ) * self.conformal_multiplier,
            ood_distance=np.asarray(distances),
            feature_order=self.features,
            target_order=self.targets,
        )
