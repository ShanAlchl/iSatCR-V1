from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


BASE_ENV_FEATURES = [
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
]

ATTACK_LEVEL_FEATURES = [
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
]

METRIC_FEATURES = [
    "PacketLossRate",
    "NetworkThroughput",
    "BandwidthUtilization",
    "AvgPacketNodeVisits",
    "CumulativeReward",
    "AverageInferenceTime",
    "AverageE2eDelay",
    "AverageHopCount",
    "AverageComputingRatio",
    "ComputingWaitingTime",
    "AverageEndingReward",
]

ATTACK_LABELS = [
    "NoAttack",
    "StateObservationAttack",
    "ActionAttack",
    "StateTransferAttack",
    "RewardAttack",
    "ExperiencePoolAttack",
    "ModelTampAttack",
]

ATTACK_COLUMN_TO_LABEL = {
    "StateObservationAttack_level": "StateObservationAttack",
    "ActionAttack_level": "ActionAttack",
    "StateTransferAttack_level": "StateTransferAttack",
    "RewardAttack_level": "RewardAttack",
    "ExperiencePoolAttack_level": "ExperiencePoolAttack",
    "ModelTampAttack_level": "ModelTampAttack",
}

LABEL_TO_INDEX = {label: index for index, label in enumerate(ATTACK_LABELS)}
ENV_FEATURES_16 = BASE_ENV_FEATURES + ATTACK_LEVEL_FEATURES
STRUCTURE_ENV_FEATURES = [
    "ConstellationConfig",
    "DegradedEdgeRatio",
    "EdgeDisconnectRatio",
    "EdgeBandwidthMeanDecreaseRatio",
    "EdgeBandwidthDecreaseStd",
]
TRAFFIC_ENV_FEATURES = [
    "PoissonRate",
    "MeanIntervalTime",
    "PacketGenerationInterval",
    "PacketSizeMean",
    "PacketSizeStd",
]
ATTACK_FEATURE_MODES = {
    "full",
    "no_constellation",
    "with_attack_level_max",
    "weak_scene",
    "oracle_attack_levels",
}
DEFAULT_ATTACK_MODEL_TYPE = "random_forest"
DEFAULT_ATTACK_FEATURE_MODE = "weak_scene"
DEFAULT_RF_ESTIMATORS = 300


def _to_float(value: object) -> float:
    return float(value)


def extract_attack_levels(row: Mapping[str, object]) -> list[float]:
    return [_to_float(row[name]) for name in ATTACK_LEVEL_FEATURES]


def read_output_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_row_sample_id(row: Mapping[str, object]) -> tuple[str, str]:
    if all(
        key in row and str(row[key]).strip() != ""
        for key in ("source_file_index", "original_round_index", "original_test_id")
    ):
        return (
            f"{row['source_file_index']}:{row['original_round_index']}",
            str(row["original_test_id"]),
        )
    return (str(row["round_index"]), str(row["test_id"]))


def extract_base_env_features(row: Mapping[str, object]) -> list[float]:
    return [_to_float(row[name]) for name in BASE_ENV_FEATURES]


def extract_metric_features(row: Mapping[str, object]) -> list[float]:
    return [_to_float(row[name]) for name in METRIC_FEATURES]


def compute_attack_level_max(row: Mapping[str, object]) -> float:
    return max(extract_attack_levels(row))


def build_attack_feature_schema(feature_mode: str) -> list[str]:
    if feature_mode == "full":
        return BASE_ENV_FEATURES + METRIC_FEATURES
    if feature_mode == "no_constellation":
        return [name for name in BASE_ENV_FEATURES if name != "ConstellationConfig"] + METRIC_FEATURES
    if feature_mode == "with_attack_level_max":
        return ["AttackLevelMax"] + [name for name in BASE_ENV_FEATURES if name != "ConstellationConfig"] + METRIC_FEATURES
    if feature_mode == "weak_scene":
        return ["AttackLevelMax"] + TRAFFIC_ENV_FEATURES + METRIC_FEATURES
    if feature_mode == "oracle_attack_levels":
        return list(ATTACK_LEVEL_FEATURES)
    raise ValueError(f"Unknown feature_mode: {feature_mode!r}.")


def build_attack_feature_vector(
    row: Mapping[str, object],
    feature_mode: str,
) -> list[float]:
    values: list[float] = []
    for feature_name in build_attack_feature_schema(feature_mode):
        if feature_name == "AttackLevelMax":
            values.append(compute_attack_level_max(row))
        else:
            values.append(_to_float(row[feature_name]))
    return values


def derive_attack_label_from_levels(
    levels: Sequence[float],
    no_attack_threshold: float = 1e-12,
    ambiguity_margin: float = 0.0,
) -> str:
    max_level = max(levels)
    if max_level <= no_attack_threshold:
        return "NoAttack"

    ordered_indices = sorted(range(len(levels)), key=lambda idx: levels[idx], reverse=True)
    winner_index = ordered_indices[0]
    if len(ordered_indices) > 1:
        runner_up_index = ordered_indices[1]
        margin = float(levels[winner_index]) - float(levels[runner_up_index])
        if margin <= ambiguity_margin:
            return "NoAttack"

    winner_column = ATTACK_LEVEL_FEATURES[winner_index]
    return ATTACK_COLUMN_TO_LABEL[winner_column]


def derive_attack_label(
    row: Mapping[str, object],
    no_attack_threshold: float = 1e-12,
    ambiguity_margin: float = 0.0,
) -> str:
    levels = extract_attack_levels(row)
    return derive_attack_label_from_levels(
        levels,
        no_attack_threshold=no_attack_threshold,
        ambiguity_margin=ambiguity_margin,
    )


def build_dataset_from_rows(
    rows: Iterable[Mapping[str, object]],
    no_attack_threshold: float = 1e-12,
    ambiguity_margin: float = 0.0,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sample_ids: list[tuple[str, str]] = []
    base_env_vectors: list[list[float]] = []
    metric_vectors: list[list[float]] = []
    labels: list[str] = []

    for row in rows:
        sample_ids.append(build_row_sample_id(row))
        base_env_vectors.append(extract_base_env_features(row))
        metric_vectors.append(extract_metric_features(row))
        labels.append(
            derive_attack_label(
                row,
                no_attack_threshold=no_attack_threshold,
                ambiguity_margin=ambiguity_margin,
            )
        )

    y = np.array([LABEL_TO_INDEX[label] for label in labels], dtype=np.int64)
    env_matrix = np.asarray(base_env_vectors, dtype=np.float32)
    metric_matrix = np.asarray(metric_vectors, dtype=np.float32)
    return sample_ids, env_matrix, metric_matrix, y, labels


def build_attack_dataset_from_rows(
    rows: Iterable[Mapping[str, object]],
    feature_mode: str,
    no_attack_threshold: float = 1e-12,
    ambiguity_margin: float = 0.0,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, list[str], list[str]]:
    sample_ids: list[tuple[str, str]] = []
    feature_vectors: list[list[float]] = []
    labels: list[str] = []
    feature_names = build_attack_feature_schema(feature_mode)

    for row in rows:
        sample_ids.append(build_row_sample_id(row))
        feature_vectors.append(build_attack_feature_vector(row, feature_mode))
        labels.append(
            derive_attack_label(
                row,
                no_attack_threshold=no_attack_threshold,
                ambiguity_margin=ambiguity_margin,
            )
        )

    y = np.array([LABEL_TO_INDEX[label] for label in labels], dtype=np.int64)
    x = np.asarray(feature_vectors, dtype=np.float32)
    return sample_ids, x, y, labels, feature_names


def build_multilabel_targets(
    rows: Iterable[Mapping[str, object]],
    target_threshold: float = 1e-12,
) -> np.ndarray:
    targets: list[list[float]] = []
    for row in rows:
        levels = extract_attack_levels(row)
        targets.append([1.0 if level > target_threshold else 0.0 for level in levels])
    return np.asarray(targets, dtype=np.float32)


def split_attack_dataset_indices(
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    label_counts = Counter(labels.tolist())
    can_stratify = len(label_counts) > 1 and min(label_counts.values()) >= 2 and len(labels) >= 10

    if len(labels) < 2:
        return indices, np.array([], dtype=np.int64)

    if can_stratify:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return np.asarray(train_idx), np.asarray(test_idx)


def train_random_forest_attack_artifact(
    rows: Sequence[Mapping[str, object]],
    artifact_path: str | Path,
    feature_mode: str = DEFAULT_ATTACK_FEATURE_MODE,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = DEFAULT_RF_ESTIMATORS,
    max_depth: int | None = None,
    no_attack_threshold: float = 1e-12,
    ambiguity_margin: float = 0.0,
) -> dict[str, object]:
    if len(rows) == 0:
        raise ValueError("Attack dataset is empty.")

    _, x_all, y_all, label_names, feature_names = build_attack_dataset_from_rows(
        rows,
        feature_mode=feature_mode,
        no_attack_threshold=no_attack_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    train_idx, test_idx = split_attack_dataset_indices(
        y_all,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(x_all[train_idx], y_all[train_idx])

    artifact = {
        "model_type": "random_forest",
        "feature_mode": feature_mode,
        "feature_names": feature_names,
        "label_names": list(ATTACK_LABELS),
        "labels": list(ATTACK_LABELS),
        "attack_level_features": list(ATTACK_LEVEL_FEATURES),
        "sample_count": len(rows),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "label_distribution": Counter(label_names),
        "no_attack_threshold": float(no_attack_threshold),
        "ambiguity_margin": float(ambiguity_margin),
        "rf_n_estimators": int(n_estimators),
        "rf_max_depth": None if max_depth is None else int(max_depth),
        "model_payload": model,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    holdout_rows = [dict(rows[int(index)]) for index in test_idx.tolist()] if len(test_idx) else []
    return {
        "artifact_path": str(artifact_path),
        "sample_count": len(rows),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "label_distribution": Counter(label_names),
        "holdout_rows": holdout_rows,
        "feature_mode": feature_mode,
        "feature_names": feature_names,
    }


class CA4x4Encoder(nn.Module):
    """Shared CA-4x4 encoder used by classification, multitask, and multilabel heads."""

    def __init__(
        self,
        env_dim: int = len(BASE_ENV_FEATURES),
        metric_dim: int = len(METRIC_FEATURES),
        d_model: int = 64,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        if env_dim != len(BASE_ENV_FEATURES):
            raise ValueError(
                f"Expected env_dim={len(BASE_ENV_FEATURES)} based on BASE_ENV_FEATURES, got {env_dim}."
            )
        if metric_dim != len(METRIC_FEATURES):
            raise ValueError(
                f"Expected metric_dim={len(METRIC_FEATURES)} based on METRIC_FEATURES, got {metric_dim}."
            )

        # 4 env tokens
        # 1) topology degradation
        # 2) traffic load
        # 3) packet-size statistics
        # 4) scenario identifier
        self.env_groups = [
            [1, 2, 3, 4],
            [5, 6, 7],
            [8, 9],
            [0],
        ]
        # 4 metric tokens
        # 1) reliability-related metrics
        # 2) throughput/efficiency metrics
        # 3) computing-related metrics
        # 4) reward-related metrics
        self.metric_groups = [
            [0, 6, 7, 3],
            [1, 2],
            [8, 9, 5],
            [4, 10],
        ]
        self.env_token_projs = nn.ModuleList(
            [nn.Linear(len(group), d_model) for group in self.env_groups]
        )
        self.metric_token_projs = nn.ModuleList(
            [nn.Linear(len(group), d_model) for group in self.metric_groups]
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.output_dim = d_model

    @staticmethod
    def _build_group_tokens(
        x: torch.Tensor,
        groups: Sequence[Sequence[int]],
        projs: nn.ModuleList,
    ) -> torch.Tensor:
        tokens = []
        for group, proj in zip(groups, projs):
            group_tensor = x[:, list(group)]
            tokens.append(proj(group_tensor))
        return torch.stack(tokens, dim=1)

    def forward(self, env_x: torch.Tensor, metric_x: torch.Tensor) -> torch.Tensor:
        env_tokens = self._build_group_tokens(env_x, self.env_groups, self.env_token_projs)
        metric_tokens = self._build_group_tokens(metric_x, self.metric_groups, self.metric_token_projs)
        attn_out, _ = self.cross_attn(query=metric_tokens, key=env_tokens, value=env_tokens)
        return torch.mean(attn_out, dim=1)


class CrossAttentionModel(nn.Module):
    """CA-4x4 model: split env/metrics into 4 tokens each, then cross-attend."""

    def __init__(
        self,
        env_dim: int = len(BASE_ENV_FEATURES),
        metric_dim: int = len(METRIC_FEATURES),
        d_model: int = 64,
        nhead: int = 4,
        num_classes: int = len(ATTACK_LABELS),
    ) -> None:
        super().__init__()
        self.encoder = CA4x4Encoder(
            env_dim=env_dim,
            metric_dim=metric_dim,
            d_model=d_model,
            nhead=nhead,
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, env_x: torch.Tensor, metric_x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(env_x, metric_x))


class LegacyCrossAttentionModel(nn.Module):
    """Backward-compatible model for pre-CA4x4 single-token artifacts."""

    def __init__(
        self,
        env_dim: int = len(BASE_ENV_FEATURES),
        metric_dim: int = len(METRIC_FEATURES),
        d_model: int = 64,
        nhead: int = 4,
        num_classes: int = len(ATTACK_LABELS),
    ) -> None:
        super().__init__()
        self.env_proj = nn.Linear(env_dim, d_model)
        self.metric_proj = nn.Linear(metric_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, env_x: torch.Tensor, metric_x: torch.Tensor) -> torch.Tensor:
        env_token = self.env_proj(env_x).unsqueeze(1)
        metric_token = self.metric_proj(metric_x).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=metric_token, key=env_token, value=env_token)
        return self.fc(attn_out.squeeze(1))


class Stage2JointCrossAttentionModel(nn.Module):
    """Stage-2 joint model with attack type head and weak/strong level head."""

    def __init__(
        self,
        env_dim: int = len(BASE_ENV_FEATURES),
        metric_dim: int = len(METRIC_FEATURES),
        context_dim: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_attack_classes: int = len(ATTACK_LEVEL_FEATURES),
        num_level_classes: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = CA4x4Encoder(
            env_dim=env_dim,
            metric_dim=metric_dim,
            d_model=d_model,
            nhead=nhead,
        )
        self.context_dim = context_dim
        self.context_proj = None
        shared_input_dim = d_model
        if context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            shared_input_dim += 32
        self.shared = nn.Sequential(
            nn.Linear(shared_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.type_head = nn.Linear(32, num_attack_classes)
        self.level_head = nn.Linear(32, num_level_classes)

    def forward(
        self,
        env_x: torch.Tensor,
        metric_x: torch.Tensor,
        context_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(env_x, metric_x)
        if self.context_proj is not None:
            if context_x is None:
                raise ValueError("context_x is required when context_dim > 0")
            encoded = torch.cat([encoded, self.context_proj(context_x)], dim=1)
        features = self.shared(encoded)
        return self.type_head(features), self.level_head(features)


class MultiLabelCrossAttentionModel(nn.Module):
    """CA-4x4 model with 6 sigmoid logits for multilabel attack prediction."""

    def __init__(
        self,
        env_dim: int = len(BASE_ENV_FEATURES),
        metric_dim: int = len(METRIC_FEATURES),
        context_dim: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_labels: int = len(ATTACK_LEVEL_FEATURES),
    ) -> None:
        super().__init__()
        self.backbone = CA4x4Encoder(
            env_dim=env_dim,
            metric_dim=metric_dim,
            d_model=d_model,
            nhead=nhead,
        )
        self.context_dim = context_dim
        self.context_proj = None
        head_input_dim = d_model
        if context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            head_input_dim += 32
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_labels),
        )

    def forward(
        self,
        env_x: torch.Tensor,
        metric_x: torch.Tensor,
        context_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.backbone(env_x, metric_x)
        if self.context_proj is not None:
            if context_x is None:
                raise ValueError("context_x is required when context_dim > 0")
            features = torch.cat([features, self.context_proj(context_x)], dim=1)
        return self.head(features)


class TripletRefinementMLP(nn.Module):
    """Lightweight expert for SOA/STA/EPA refinement."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttackTypeClassifier:
    """Load a saved artifact and run standalone attack type inference."""

    def __init__(
        self,
        model: nn.Module | None,
        env_scaler: object | None,
        metric_scaler: object | None,
        labels: Sequence[str],
        model_type: str = "legacy_ca44",
        rf_model: RandomForestClassifier | None = None,
        feature_mode: str | None = None,
        feature_names: Sequence[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None if model is None else model.to(self.device)
        if self.model is not None:
            self.model.eval()
        self.env_scaler = env_scaler
        self.metric_scaler = metric_scaler
        self.labels = list(labels)
        self.rf_model = rf_model
        self.feature_mode = feature_mode
        self.feature_names = list(feature_names or [])

    @staticmethod
    def _detect_artifact_layout(model_state: Mapping[str, object]) -> str:
        keys = set(model_state.keys())
        if any(key.startswith("encoder.") for key in keys):
            return "ca44_nested"
        if any(key.startswith("env_token_projs.") for key in keys):
            return "ca44_prefixless"
        if any(key.startswith("env_proj.") for key in keys):
            return "legacy_single_token"
        raise ValueError("Unsupported attack classifier artifact layout.")

    @staticmethod
    def _normalize_ca44_prefixless_state(
        model_state: Mapping[str, object],
    ) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in model_state.items():
            if (
                key.startswith("env_token_projs.")
                or key.startswith("metric_token_projs.")
                or key.startswith("cross_attn.")
            ):
                normalized[f"encoder.{key}"] = value
            else:
                normalized[key] = value
        return normalized

    @classmethod
    def load(cls, artifact_path: str | Path) -> "AttackTypeClassifier":
        artifact = joblib.load(artifact_path)
        model_type = str(artifact.get("model_type", "legacy_ca44"))
        if model_type == "random_forest":
            return cls(
                model=None,
                env_scaler=None,
                metric_scaler=None,
                labels=artifact.get("label_names", artifact.get("labels", ATTACK_LABELS)),
                model_type=model_type,
                rf_model=artifact["model_payload"],
                feature_mode=str(artifact.get("feature_mode", DEFAULT_ATTACK_FEATURE_MODE)),
                feature_names=artifact.get("feature_names", []),
            )

        model_state = artifact["model_state"]
        layout = cls._detect_artifact_layout(model_state)
        env_dim = len(artifact["base_env_features"])
        metric_dim = len(artifact["metric_features"])
        d_model = int(artifact["d_model"])
        nhead = int(artifact["nhead"])
        num_classes = len(artifact["labels"])

        if layout == "legacy_single_token":
            model = LegacyCrossAttentionModel(
                env_dim=env_dim,
                metric_dim=metric_dim,
                d_model=d_model,
                nhead=nhead,
                num_classes=num_classes,
            )
            model.load_state_dict(model_state)
        else:
            model = CrossAttentionModel(
                env_dim=env_dim,
                metric_dim=metric_dim,
                d_model=d_model,
                nhead=nhead,
                num_classes=num_classes,
            )
            if layout == "ca44_prefixless":
                model_state = cls._normalize_ca44_prefixless_state(model_state)
            model.load_state_dict(model_state)
        return cls(
            model=model,
            env_scaler=artifact["env_scaler"],
            metric_scaler=artifact["metric_scaler"],
            labels=artifact["labels"],
            model_type="legacy_ca44",
        )

    def predict_logits(
        self,
        base_env_features: Sequence[float],
        metric_features: Sequence[float],
    ) -> torch.Tensor:
        if self.model is None or self.env_scaler is None or self.metric_scaler is None:
            raise RuntimeError("predict_logits is only available for legacy neural-network artifacts.")
        scaled_env = self.env_scaler.transform([list(base_env_features)])
        scaled_metrics = self.metric_scaler.transform([list(metric_features)])
        env_tensor = torch.tensor(scaled_env, dtype=torch.float32, device=self.device)
        metric_tensor = torch.tensor(scaled_metrics, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(env_tensor, metric_tensor)
        return logits.cpu()

    def predict_label(
        self,
        base_env_features: Sequence[float],
        metric_features: Sequence[float],
    ) -> str:
        logits = self.predict_logits(base_env_features, metric_features)
        predicted_index = int(torch.argmax(logits, dim=1).item())
        return self.labels[predicted_index]

    def predict_row(self, row: Mapping[str, object]) -> str:
        if self.model_type == "random_forest":
            if self.rf_model is None or self.feature_mode is None:
                raise RuntimeError("RandomForest artifact is missing model payload or feature_mode.")
            feature_vector = np.asarray(
                [build_attack_feature_vector(row, self.feature_mode)],
                dtype=np.float32,
            )
            predicted_index = int(self.rf_model.predict(feature_vector)[0])
            return self.labels[predicted_index]
        return self.predict_label(
            extract_base_env_features(row),
            extract_metric_features(row),
        )

    def predict_batch(self, rows: Sequence[Mapping[str, object]]) -> list[str]:
        return [self.predict_row(row) for row in rows]


def format_classification_summary(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    label_order: Sequence[str] = ATTACK_LABELS,
) -> str:
    metrics = compute_classification_metrics(true_labels, pred_labels, label_order)
    lines = [
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro Precision: {metrics['macro_precision']:.4f}",
        f"Macro Recall: {metrics['macro_recall']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted Precision: {metrics['weighted_precision']:.4f}",
        f"Weighted Recall: {metrics['weighted_recall']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        "Confusion Matrix (rows=true, cols=pred):",
        "label," + ",".join(label_order),
    ]

    for label, row in zip(label_order, metrics["confusion_matrix"]):
        lines.append(label + "," + ",".join(str(int(value)) for value in row))

    lines.extend(
        [
            "Classification Report:",
            str(metrics["classification_report"]).rstrip(),
        ]
    )
    return "\n".join(lines)


def compute_classification_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    label_order: Sequence[str] = ATTACK_LABELS,
) -> dict[str, object]:
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(label_order),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(label_order),
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, pred_labels, labels=list(label_order))
    per_label_metrics = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(label_order),
        average=None,
        zero_division=0,
    )
    per_label: dict[str, dict[str, float]] = {}
    for label, precision, recall, f1, support in zip(label_order, *per_label_metrics):
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            true_labels,
            pred_labels,
            labels=list(label_order),
            zero_division=0,
        ),
        "per_label": per_label,
    }
