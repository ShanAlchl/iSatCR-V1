from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SCENARIO_PARAM_NAMES = [
    "ConstellationConfig",
    "DegradedEdgeRatio",
    "EdgeDisconnectRatio",
    "EdgeBandwidthMeanDecreaseRatio",
    "EdgeBandwidthDecreaseStd",
    "PoissonRate",
    "MeanIntervalTime",
    "PacketGenerationInterval",
    "PacketSizeMean",
    "PacketSizeStd",
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
]

FAIL_SCORE_CANDIDATES = [
    "fail_score",
    "FailScore",
    "failure_score_v2",
    "failure_score",
]


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def resolve_fail_score_column(field_names: Sequence[str]) -> str:
    for candidate in FAIL_SCORE_CANDIDATES:
        if candidate in field_names:
            return candidate
    raise KeyError(
        "No fail score column found. Expected one of: "
        + ", ".join(FAIL_SCORE_CANDIDATES)
    )


def extract_scenario_params(row: Mapping[str, object]) -> list[float]:
    return [float(row[name]) for name in SCENARIO_PARAM_NAMES]


def extract_fail_score(row: Mapping[str, object], fail_score_column: str) -> float:
    return float(row[fail_score_column])


def build_regression_dataset(
    rows: Iterable[Mapping[str, object]],
    fail_score_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for row in rows:
        x_rows.append(extract_scenario_params(row))
        y_rows.append(extract_fail_score(row, fail_score_column))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


class FailScoreContributionModel:
    """Predict fail_score from 16 scenario parameters and explain local contributions."""

    def __init__(
        self,
        steps: int = 50,
        epsilon: float = 1e-4,
        safe_threshold: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.steps = steps
        self.epsilon = epsilon
        self.safe_threshold = safe_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )
        self.feature_names = list(SCENARIO_PARAM_NAMES)
        self.X_mean_safe: np.ndarray | None = None
        self.model_fitted = False
        self.constant_fail_score: float | None = None

    def fit(self, x_history: Sequence[Sequence[float]], y_history: Sequence[float]) -> None:
        x_arr = np.asarray(x_history, dtype=np.float64)
        y_arr = np.asarray(y_history, dtype=np.float64)

        if x_arr.ndim != 2 or x_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected X with shape [n_samples, {len(self.feature_names)}], got {x_arr.shape}."
            )
        if y_arr.ndim != 1 or len(y_arr) != len(x_arr):
            raise ValueError("Expected y to be a 1D array aligned with X.")
        if len(x_arr) == 0:
            raise ValueError("Training data is empty.")

        safe_mask = y_arr < self.safe_threshold
        if np.any(safe_mask):
            self.X_mean_safe = np.mean(x_arr[safe_mask], axis=0)
        else:
            self.X_mean_safe = np.mean(x_arr, axis=0)

        self.scaler.fit(x_arr)

        has_feature_variation = len(x_arr) > 1 and len(np.unique(x_arr, axis=0)) > 1
        has_target_variation = len(np.unique(y_arr)) > 1

        if has_feature_variation and has_target_variation:
            if len(x_arr) < 10:
                self.regressor.set_params(early_stopping=False, max_iter=200)
            scaled_x = self.scaler.transform(x_arr)
            self.regressor.fit(scaled_x, y_arr)
            self.model_fitted = True
            self.constant_fail_score = None
        else:
            # Preserve a working predictor even when the target is constant.
            self.model_fitted = False
            self.constant_fail_score = float(np.mean(y_arr))

    def predict_fail_score(self, params_16: Sequence[float]) -> float:
        x = np.asarray(params_16, dtype=np.float64)
        if x.shape != (len(self.feature_names),):
            raise ValueError(f"Expected 16 input parameters, got shape {x.shape}.")

        if self.model_fitted:
            scaled_x = self.scaler.transform([x])
            return float(self.regressor.predict(scaled_x)[0])

        if self.constant_fail_score is None:
            raise RuntimeError("Model has not been trained or loaded correctly.")
        return self.constant_fail_score

    def _approximate_gradients(self, x: np.ndarray) -> np.ndarray:
        grads = np.zeros_like(x, dtype=np.float64)
        for index in range(len(x)):
            x_plus = x.copy()
            x_plus[index] += self.epsilon
            x_minus = x.copy()
            x_minus[index] -= self.epsilon
            y_plus = self.predict_fail_score(x_plus)
            y_minus = self.predict_fail_score(x_minus)
            grads[index] = (y_plus - y_minus) / (2 * self.epsilon)
        return grads

    def calculate_local_contribution(self, current_params: Sequence[float]) -> list[float]:
        current_arr = np.asarray(current_params, dtype=np.float64)
        feature_count = len(current_arr)

        if self.X_mean_safe is None:
            return (np.ones(feature_count) / feature_count).tolist()

        if not self.model_fitted:
            return (np.ones(feature_count) / feature_count).tolist()

        baseline = self.X_mean_safe
        alphas = [float(step) / self.steps for step in range(1, self.steps + 1)]
        scaled_inputs = [baseline + alpha * (current_arr - baseline) for alpha in alphas]

        accumulated_grads = np.zeros(feature_count, dtype=np.float64)
        for step_input in scaled_inputs:
            accumulated_grads += self._approximate_gradients(step_input)

        avg_grads = accumulated_grads / self.steps
        integrated_gradients = (current_arr - baseline) * avg_grads
        contributions = np.abs(integrated_gradients)
        contribution_sum = float(np.sum(contributions))

        if contribution_sum <= 1e-12:
            return (np.ones(feature_count) / feature_count).tolist()
        return (contributions / contribution_sum).tolist()

    def predict_with_contributions(self, params_16: Sequence[float]) -> dict[str, object]:
        params = [float(value) for value in params_16]
        return {
            "fail_score": self.predict_fail_score(params),
            "contribution_list": self.calculate_local_contribution(params),
            "feature_names": self.feature_names,
        }

    def save(self, artifact_path: str | Path) -> None:
        artifact = {
            "steps": self.steps,
            "epsilon": self.epsilon,
            "safe_threshold": self.safe_threshold,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "x_mean_safe": self.X_mean_safe,
            "model_fitted": self.model_fitted,
            "constant_fail_score": self.constant_fail_score,
            "scaler": self.scaler,
            "regressor": self.regressor,
        }
        joblib.dump(artifact, artifact_path)

    @classmethod
    def load(cls, artifact_path: str | Path) -> "FailScoreContributionModel":
        artifact = joblib.load(artifact_path)
        model = cls(
            steps=int(artifact["steps"]),
            epsilon=float(artifact["epsilon"]),
            safe_threshold=float(artifact["safe_threshold"]),
            random_state=int(artifact["random_state"]),
        )
        model.feature_names = list(artifact["feature_names"])
        model.X_mean_safe = artifact["x_mean_safe"]
        model.model_fitted = bool(artifact["model_fitted"])
        model.constant_fail_score = artifact["constant_fail_score"]
        model.scaler = artifact["scaler"]
        model.regressor = artifact["regressor"]
        return model


def parse_params_text(params_text: str) -> list[float]:
    parts = [part.strip() for part in params_text.split(",") if part.strip()]
    if len(parts) != len(SCENARIO_PARAM_NAMES):
        raise ValueError(
            f"Expected {len(SCENARIO_PARAM_NAMES)} comma-separated values, got {len(parts)}."
        )
    return [float(part) for part in parts]


def result_to_json(result: Mapping[str, object]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
