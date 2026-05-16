from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from attack_type_classifier import (
    ATTACK_COLUMN_TO_LABEL,
    ATTACK_LABELS,
    ATTACK_LEVEL_FEATURES,
    build_row_sample_id,
    extract_attack_levels,
    extract_base_env_features,
    extract_metric_features,
)


ATTACK_ONLY_LABELS = ATTACK_LABELS[1:]
ATTACK_ONLY_TO_INDEX = {label: index for index, label in enumerate(ATTACK_ONLY_LABELS)}


@dataclass(frozen=True)
class DiscreteLabelConfig:
    effective_attack_min_level: int = 1
    strong_attack_min_level: int = 2
    weak_attack_policy: str = "NoAttack"
    multi_high_level_policy: str = "argmax"
    priority_table: tuple[str, ...] = field(default_factory=lambda: tuple(ATTACK_ONLY_LABELS))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["priority_table"] = list(self.priority_table)
        return payload


def _priority_rank(label: str, config: DiscreteLabelConfig) -> int:
    try:
        return config.priority_table.index(label)
    except ValueError:
        return len(config.priority_table)


def _choose_primary_attack_label(
    levels: Sequence[float],
    config: DiscreteLabelConfig,
) -> str:
    max_level = max(float(level) for level in levels)
    candidate_indices = [index for index, level in enumerate(levels) if float(level) == max_level]
    candidate_labels = [ATTACK_COLUMN_TO_LABEL[ATTACK_LEVEL_FEATURES[index]] for index in candidate_indices]
    if len(candidate_labels) == 1 or config.multi_high_level_policy == "argmax":
        return candidate_labels[0]
    if config.multi_high_level_policy == "priority_table":
        return min(candidate_labels, key=lambda label: _priority_rank(label, config))
    raise ValueError(f"Unsupported multi_high_level_policy: {config.multi_high_level_policy}")


def derive_discrete_targets_from_levels(
    levels: Sequence[float],
    config: DiscreteLabelConfig,
) -> dict[str, object]:
    max_level = max(int(round(float(level))) for level in levels)
    has_effective_attack = max_level >= int(config.effective_attack_min_level)
    primary_label = _choose_primary_attack_label(levels, config)
    is_strong = max_level >= int(config.strong_attack_min_level)
    level_bucket = "strong" if is_strong else "weak"

    if not has_effective_attack:
        final_label = "NoAttack"
    elif is_strong:
        final_label = primary_label
    elif config.weak_attack_policy in {"NoAttack", "UncertainAsNoAttack"}:
        final_label = "NoAttack"
    elif config.weak_attack_policy == "KeepArgmax":
        final_label = primary_label
    else:
        raise ValueError(f"Unsupported weak_attack_policy: {config.weak_attack_policy}")

    return {
        "stage1_label": int(has_effective_attack),
        "final_label": final_label,
        "stage2_type_label": primary_label if has_effective_attack else None,
        "stage2_level_bucket": level_bucket if has_effective_attack else None,
        "max_level": int(max_level),
        "is_strong": bool(is_strong),
        "attack_levels": [int(round(float(level))) for level in levels],
    }


def derive_discrete_targets_from_row(
    row: Mapping[str, object],
    config: DiscreteLabelConfig,
) -> dict[str, object]:
    return derive_discrete_targets_from_levels(extract_attack_levels(row), config)


def build_discrete_dataset(
    rows: Iterable[Mapping[str, object]],
    config: DiscreteLabelConfig,
) -> dict[str, object]:
    sample_ids: list[tuple[str, str]] = []
    env_vectors: list[list[float]] = []
    metric_vectors: list[list[float]] = []
    stage1_labels: list[int] = []
    final_labels: list[str] = []
    stage2_type_labels: list[str | None] = []
    stage2_level_buckets: list[str | None] = []
    max_levels: list[int] = []
    attack_levels: list[list[int]] = []
    multilabel_binary: list[list[float]] = []

    for row in rows:
        targets = derive_discrete_targets_from_row(row, config)
        sample_ids.append(build_row_sample_id(row))
        env_vectors.append(extract_base_env_features(row))
        metric_vectors.append(extract_metric_features(row))
        stage1_labels.append(int(targets["stage1_label"]))
        final_labels.append(str(targets["final_label"]))
        stage2_type_labels.append(targets["stage2_type_label"])
        stage2_level_buckets.append(targets["stage2_level_bucket"])
        max_levels.append(int(targets["max_level"]))
        attack_levels.append(list(targets["attack_levels"]))
        multilabel_binary.append([1.0 if int(level) > 0 else 0.0 for level in targets["attack_levels"]])

    return {
        "sample_ids": sample_ids,
        "env_matrix": np.asarray(env_vectors, dtype=np.float32),
        "metric_matrix": np.asarray(metric_vectors, dtype=np.float32),
        "stage1_labels": np.asarray(stage1_labels, dtype=np.int64),
        "final_labels": final_labels,
        "stage2_type_labels": stage2_type_labels,
        "stage2_level_buckets": stage2_level_buckets,
        "max_levels": np.asarray(max_levels, dtype=np.int64),
        "attack_levels": np.asarray(attack_levels, dtype=np.int64),
        "multilabel_binary": np.asarray(multilabel_binary, dtype=np.float32),
    }
