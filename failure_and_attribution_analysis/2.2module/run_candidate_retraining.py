from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


CURRENT_DIR = Path(__file__).resolve().parent
ATTACK_MODULE_DIR = CURRENT_DIR / "attack_classifier"
ATTRIBUTION_MODULE_DIR = CURRENT_DIR / "attribution_analysis"

for module_dir in (ATTACK_MODULE_DIR, ATTRIBUTION_MODULE_DIR):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

from attack_type_classifier import (  # noqa: E402
    ATTACK_LABELS,
    AttackTypeClassifier,
    DEFAULT_ATTACK_FEATURE_MODE,
    DEFAULT_RF_ESTIMATORS,
    compute_classification_metrics,
    derive_attack_label,
    read_output_csv_rows,
    train_random_forest_attack_artifact,
)
from build_fail_score_training_csv import (  # noqa: E402
    build_audit,
    parse_test_summary_lines,
    write_audit,
    write_csv,
)
from extract_output_csv import aggregate_last_two_steps, build_output_csv  # noqa: E402
from fail_score_contribution_model import (  # noqa: E402
    FailScoreContributionModel,
    SCENARIO_PARAM_NAMES,
    build_regression_dataset,
    read_csv_rows,
    resolve_fail_score_column,
)
from merge_output_summaries import merge_summary_files  # noqa: E402
DEFAULT_CONFIG_NAME = "pipeline_config.jsonc"
DEFAULT_REGISTRY_PATH = (
    "failure_and_attribution_analysis/2.2module/model_artifacts/reports/archive_training_registry.json"
)
DEFAULT_CANDIDATE_ROOT = "failure_and_attribution_analysis/2.2module/model_artifacts/candidate"
DEFAULT_PRODUCTION_ROOT = "failure_and_attribution_analysis/2.2module/model_artifacts/production"
DEFAULT_HOLDOUT_RATIO = 0.2
DEFAULT_HISTORY_ATTACK_RATIO = 1.5
DEFAULT_HISTORY_ATTRIBUTION_RATIO = 2.0
DEFAULT_MAX_REUSE_PER_EXPERIMENT = 3


def strip_jsonc_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def load_jsonc(config_path: Path) -> dict[str, Any]:
    payload = strip_jsonc_comments(config_path.read_text(encoding="utf-8"))
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a JSON object.")
    return data


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def default_retraining_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "auto_promote_candidate": False,
        "registry_path": DEFAULT_REGISTRY_PATH,
        "candidate_root": DEFAULT_CANDIDATE_ROOT,
        "production_root": DEFAULT_PRODUCTION_ROOT,
        "history_sampling_ratio": {
            "attack": DEFAULT_HISTORY_ATTACK_RATIO,
            "attribution": DEFAULT_HISTORY_ATTRIBUTION_RATIO,
        },
        "max_reuse_per_experiment": DEFAULT_MAX_REUSE_PER_EXPERIMENT,
        "holdout_ratio": DEFAULT_HOLDOUT_RATIO,
        "trigger_thresholds": {
            "attack_new_grouped_min": 120,
            "attack_new_attack_min": 60,
            "attack_label_coverage_min": 4,
            "attack_min_label_count": 12,
            "attack_total_grouped_min": 500,
            "attribution_new_sample_min": 150,
            "attribution_target_unique_min": 20,
            "attribution_constellation_coverage_min": 2,
            "attribution_total_sample_min": 600,
        },
        "promotion_thresholds": {
            "attack_macro_f1_drop_max": 0.01,
            "attack_weighted_f1_drop_max": 0.01,
            "attack_key_recall_drop_max": 0.03,
            "attack_macro_f1_gain_recommend": 0.02,
            "attribution_mae_ratio_max": 1.03,
            "attribution_rmse_ratio_max": 1.03,
            "attribution_r2_drop_max": 0.02,
        },
    }


def merge_retraining_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = default_retraining_config()
    incoming = dict(config.get("retraining", {}))
    if "history_sampling_ratio" in incoming:
        merged["history_sampling_ratio"].update(incoming["history_sampling_ratio"])
    if "trigger_thresholds" in incoming:
        merged["trigger_thresholds"].update(incoming["trigger_thresholds"])
    if "promotion_thresholds" in incoming:
        merged["promotion_thresholds"].update(incoming["promotion_thresholds"])
    for key, value in incoming.items():
        if key in {"history_sampling_ratio", "trigger_thresholds", "promotion_thresholds"}:
            continue
        merged[key] = value
    return merged


def load_pipeline_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pipeline summary must be a JSON object.")
    return data


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "experiments": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Registry is not a JSON object: {path}")
    experiments = data.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError(f"Registry 'experiments' must be a list: {path}")
    return {
        "schema_version": int(data.get("schema_version", 1)),
        "experiments": experiments,
    }


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def registry_lookup(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["experiment_dir"]): item
        for item in registry.get("experiments", [])
        if isinstance(item, dict) and "experiment_dir" in item
    }


def load_attack_thresholds(artifact_path: Path) -> tuple[float, float]:
    if not artifact_path.exists():
        return (1e-12, 0.0)
    artifact = joblib.load(artifact_path)
    return (
        float(artifact.get("no_attack_threshold", 1e-12)),
        float(artifact.get("ambiguity_margin", 0.0)),
    )


def resolve_session_dir(experiment_dir: Path) -> Path:
    nested_current_session = experiment_dir / "current_session"
    if nested_current_session.exists():
        return nested_current_session
    return experiment_dir


def inspect_experiment(
    experiment_dir: Path,
    target_field: str,
    no_attack_threshold: float,
    ambiguity_margin: float,
) -> dict[str, Any]:
    current_session_dir = resolve_session_dir(experiment_dir)
    summary_path = current_session_dir / "output_summary.txt"
    rounds_root = current_session_dir / "rounds"
    attack_rows = aggregate_last_two_steps(summary_path) if summary_path.exists() else []
    attack_labels = [
        derive_attack_label(
            row,
            no_attack_threshold=no_attack_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        for row in attack_rows
    ]
    label_counts = Counter(attack_labels)
    attribution_rows = parse_test_summary_lines(rounds_root, target_field=target_field) if rounds_root.exists() else []
    target_values = [float(row["fail_score"]) for row in attribution_rows]
    constellation_id = experiment_dir.parent.name
    status = "ok" if summary_path.exists() and rounds_root.exists() else "missing_required_files"
    return {
        "experiment_dir": str(experiment_dir.resolve()),
        "constellation_id": constellation_id,
        "data_quality_status": status,
        "attack_summary_path": str(summary_path.resolve()) if summary_path.exists() else "",
        "attack_grouped_sample_count": len(attack_rows),
        "attack_attack_sample_count": int(sum(1 for label in attack_labels if label != "NoAttack")),
        "attack_label_counts": dict(label_counts),
        "attack_label_coverage": sorted(label_counts),
        "attribution_rounds_root": str(rounds_root.resolve()) if rounds_root.exists() else "",
        "attribution_sample_count": len(attribution_rows),
        "attribution_target_unique_count": len({round(value, 12) for value in target_values}),
        "attribution_target_is_constant": len({round(value, 12) for value in target_values}) <= 1 if target_values else True,
    }


def upsert_registry_records(
    registry: dict[str, Any],
    archived_dirs: list[dict[str, str]],
    target_field: str,
    no_attack_threshold: float,
    ambiguity_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    now_text = datetime.now().isoformat(timespec="seconds")
    lookup = registry_lookup(registry)
    inspected_records: list[dict[str, Any]] = []

    for archived in archived_dirs:
        target_dir = Path(str(archived["target_dir"])).resolve()
        inspection = inspect_experiment(
            target_dir,
            target_field=target_field,
            no_attack_threshold=no_attack_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        entry = lookup.get(str(target_dir))
        if entry is None:
            entry = {
                "experiment_dir": str(target_dir),
                "constellation_id": inspection["constellation_id"],
                "archived_at": now_text,
                "eligible_for_retraining": True,
                "times_sampled": 0,
                "last_sampled_at": None,
                "last_used_in_retrain_run_id": None,
            }
            registry.setdefault("experiments", []).append(entry)
            lookup[str(target_dir)] = entry
        entry.update(
            {
                "constellation_id": inspection["constellation_id"],
                "eligible_for_retraining": True,
                "data_quality_status": inspection["data_quality_status"],
                "attack_summary_path": inspection["attack_summary_path"],
                "attack_grouped_sample_count": inspection["attack_grouped_sample_count"],
                "attack_attack_sample_count": inspection["attack_attack_sample_count"],
                "attack_label_counts": inspection["attack_label_counts"],
                "attack_label_coverage": inspection["attack_label_coverage"],
                "attribution_rounds_root": inspection["attribution_rounds_root"],
                "attribution_sample_count": inspection["attribution_sample_count"],
                "attribution_target_unique_count": inspection["attribution_target_unique_count"],
                "attribution_target_is_constant": inspection["attribution_target_is_constant"],
            }
        )
        inspected_records.append(dict(entry))

    return registry, inspected_records


def merge_label_counts(records: list[dict[str, Any]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for record in records:
        for label, count in dict(record.get("attack_label_counts", {})).items():
            merged[str(label)] += int(count)
    return merged


def select_history_records(
    registry: dict[str, Any],
    exclude_dirs: set[str],
    sample_key: str,
    desired_count: int,
    max_reuse_per_experiment: int,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in registry.get("experiments", [])
        if isinstance(record, dict)
        and bool(record.get("eligible_for_retraining", True))
        and str(record.get("experiment_dir")) not in exclude_dirs
        and int(record.get("times_sampled", 0)) < max_reuse_per_experiment
        and str(record.get("data_quality_status", "ok")) == "ok"
        and int(record.get(sample_key, 0)) > 0
    ]
    candidates.sort(
        key=lambda record: (
            int(record.get("times_sampled", 0)),
            "" if record.get("last_sampled_at") is None else str(record["last_sampled_at"]),
            str(record.get("experiment_dir")),
        )
    )

    selected: list[dict[str, Any]] = []
    accumulated = 0
    for record in candidates:
        selected.append(record)
        accumulated += int(record.get(sample_key, 0))
        if accumulated >= desired_count:
            break
    return selected


def build_attack_skip_summary(new_records: list[dict[str, Any]], reasons: list[str]) -> dict[str, Any]:
    label_counts = merge_label_counts(new_records)
    return {
        "triggered": False,
        "skip_reasons": reasons,
        "new_experiment_count": len(new_records),
        "new_grouped_sample_count": int(sum(int(record.get("attack_grouped_sample_count", 0)) for record in new_records)),
        "new_attack_sample_count": int(sum(int(record.get("attack_attack_sample_count", 0)) for record in new_records)),
        "new_label_counts": dict(label_counts),
        "history_sampled_experiments": [],
        "history_grouped_sample_count": 0,
        "combined_grouped_sample_count": int(sum(int(record.get("attack_grouped_sample_count", 0)) for record in new_records)),
        "candidate_artifact_path": None,
        "candidate_metrics": None,
        "production_metrics": None,
    }


def build_attribution_skip_summary(new_records: list[dict[str, Any]], reasons: list[str], new_constellation_count: int) -> dict[str, Any]:
    return {
        "triggered": False,
        "skip_reasons": reasons,
        "new_experiment_count": len(new_records),
        "new_sample_count": int(sum(int(record.get("attribution_sample_count", 0)) for record in new_records)),
        "new_constellation_count": new_constellation_count,
        "history_sampled_experiments": [],
        "history_sample_count": 0,
        "combined_sample_count": int(sum(int(record.get("attribution_sample_count", 0)) for record in new_records)),
        "candidate_artifact_path": None,
        "candidate_metrics": None,
        "production_metrics": None,
    }


def build_threshold_check(
    name: str,
    actual: int | float,
    threshold: int | float,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "threshold": threshold,
        "passed": passed,
    }


def compute_attack_key_recall_average(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    per_label = metrics.get("per_label")
    if not isinstance(per_label, dict):
        return None
    recalls = []
    for label, payload in per_label.items():
        if label == "NoAttack":
            continue
        if isinstance(payload, dict) and "recall" in payload:
            recalls.append(float(payload["recall"]))
    if not recalls:
        return None
    return float(sum(recalls) / len(recalls))


def evaluate_attack_classifier_on_rows(
    classifier: AttackTypeClassifier,
    rows: list[dict[str, Any]],
    no_attack_threshold: float,
    ambiguity_margin: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    true_labels: list[str] = []
    pred_labels: list[str] = []
    for row in rows:
        true_labels.append(
            derive_attack_label(
                row,
                no_attack_threshold=no_attack_threshold,
                ambiguity_margin=ambiguity_margin,
            )
        )
        pred_labels.append(classifier.predict_row(row))
    return compute_classification_metrics(true_labels, pred_labels, ATTACK_LABELS)


def train_attack_candidate(
    csv_path: Path,
    artifact_path: Path,
    production_artifact_path: Path,
    holdout_ratio: float,
    random_state: int,
    feature_mode: str,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    no_attack_threshold: float,
    ambiguity_margin: float,
) -> dict[str, Any]:
    rows = read_output_csv_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("Attack retraining CSV is empty.")

    result = train_random_forest_attack_artifact(
        rows,
        artifact_path=artifact_path,
        feature_mode=feature_mode,
        test_size=holdout_ratio,
        random_state=random_state,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        no_attack_threshold=no_attack_threshold,
        ambiguity_margin=ambiguity_margin,
    )

    candidate_classifier = AttackTypeClassifier.load(artifact_path)
    holdout_rows = list(result["holdout_rows"])
    candidate_metrics = evaluate_attack_classifier_on_rows(
        candidate_classifier,
        holdout_rows,
        no_attack_threshold=no_attack_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    production_metrics = None
    if production_artifact_path.exists() and holdout_rows:
        production_classifier = AttackTypeClassifier.load(production_artifact_path)
        production_metrics = evaluate_attack_classifier_on_rows(
            production_classifier,
            holdout_rows,
            no_attack_threshold=no_attack_threshold,
            ambiguity_margin=ambiguity_margin,
        )

    return {
        "sample_count": int(result["sample_count"]),
        "train_sample_count": int(result["train_count"]),
        "holdout_sample_count": int(result["test_count"]),
        "candidate_artifact_path": str(artifact_path),
        "model_type": "random_forest",
        "feature_mode": feature_mode,
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
    }


def train_attribution_candidate(
    csv_path: Path,
    artifact_path: Path,
    production_artifact_path: Path,
    holdout_ratio: float,
    random_state: int,
    safe_threshold: float,
) -> dict[str, Any]:
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError("Attribution retraining CSV is empty.")
    fail_score_column = resolve_fail_score_column(rows[0].keys())
    x_all, y_all = build_regression_dataset(rows, fail_score_column)

    if len(np.unique(y_all)) <= 1 or len(x_all) < 2:
        raise ValueError("Attribution retraining target has insufficient variation.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=holdout_ratio,
        random_state=random_state,
    )
    model = FailScoreContributionModel(safe_threshold=safe_threshold, random_state=random_state)
    model.fit(x_train, y_train)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)

    candidate_pred = np.asarray([model.predict_fail_score(row) for row in x_test], dtype=np.float64)
    candidate_metrics = {
        "mae": float(mean_absolute_error(y_test, candidate_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, candidate_pred))),
        "r2": float(r2_score(y_test, candidate_pred)),
    }

    production_metrics = None
    if production_artifact_path.exists():
        production_model = FailScoreContributionModel.load(production_artifact_path)
        production_pred = np.asarray(
            [production_model.predict_fail_score(row) for row in x_test],
            dtype=np.float64,
        )
        production_metrics = {
            "mae": float(mean_absolute_error(y_test, production_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, production_pred))),
            "r2": float(r2_score(y_test, production_pred)),
        }

    return {
        "sample_count": len(rows),
        "train_sample_count": int(len(x_train)),
        "holdout_sample_count": int(len(x_test)),
        "candidate_artifact_path": str(artifact_path),
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
    }


def build_combined_attribution_csv(
    experiment_dirs: list[Path],
    target_field: str,
    output_csv_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    rows = collect_attribution_rows(experiment_dirs, target_field)
    if not rows:
        raise ValueError("No attribution rows were found for combined retraining CSV.")
    write_csv(rows, output_csv_path)
    audit = build_audit(rows, target_field=target_field)
    write_audit(audit, audit_path)
    return {
        "sample_count": len(rows),
        "target_unique_count": int(audit["target_unique_count"]),
        "target_is_constant": int(audit["target_unique_count"]) <= 1,
        "constellation_ids": sorted({str(experiment_dir.parent.name) for experiment_dir in experiment_dirs}),
    }


def collect_attribution_rows(
    experiment_dirs: list[Path],
    target_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_dir in experiment_dirs:
        rounds_root = resolve_session_dir(experiment_dir) / "rounds"
        if rounds_root.exists():
            rows.extend(parse_test_summary_lines(rounds_root, target_field=target_field))
    return rows


def promote_attack_candidate(
    candidate_artifact_path: Path,
    production_artifact_path: Path,
    production_root: Path,
    run_id: str,
) -> dict[str, Any]:
    backup_dir = production_root / "previous" / run_id
    current_dir = production_root / "current"
    backup_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / production_artifact_path.name
    if production_artifact_path.exists():
        shutil.copy2(production_artifact_path, backup_path)
    production_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_artifact_path, production_artifact_path)
    shutil.copy2(candidate_artifact_path, current_dir / production_artifact_path.name)
    return {
        "promoted": True,
        "backup_path": str(backup_path) if production_artifact_path.exists() else None,
        "production_artifact_path": str(production_artifact_path),
    }


def promote_attribution_candidate(
    candidate_artifact_path: Path,
    production_artifact_path: Path,
    production_root: Path,
    run_id: str,
) -> dict[str, Any]:
    backup_dir = production_root / "previous" / run_id
    current_dir = production_root / "current"
    backup_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / production_artifact_path.name
    if production_artifact_path.exists():
        shutil.copy2(production_artifact_path, backup_path)
    production_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_artifact_path, production_artifact_path)
    shutil.copy2(candidate_artifact_path, current_dir / production_artifact_path.name)
    return {
        "promoted": True,
        "backup_path": str(backup_path) if production_artifact_path.exists() else None,
        "production_artifact_path": str(production_artifact_path),
    }


def build_promotion_decision(
    attack_summary: dict[str, Any],
    attribution_summary: dict[str, Any],
    thresholds: dict[str, Any],
    auto_promote_candidate: bool,
    candidate_root: Path,
    production_root: Path,
    attack_production_artifact: Path,
    attribution_production_artifact: Path,
    run_id: str,
) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "enabled": bool(auto_promote_candidate),
        "attack_classifier": {"evaluated": False, "promoted": False, "reasons": []},
        "attribution_analysis": {"evaluated": False, "promoted": False, "reasons": []},
    }
    if not auto_promote_candidate:
        decision["reason"] = "auto_promotion_disabled"
        return decision

    attack_candidate_metrics = attack_summary.get("candidate_metrics")
    attack_production_metrics = attack_summary.get("production_metrics")
    if attack_summary.get("triggered") and attack_candidate_metrics and attack_production_metrics:
        decision["attack_classifier"]["evaluated"] = True
        reasons: list[str] = []
        macro_drop = float(attack_production_metrics["macro_f1"]) - float(attack_candidate_metrics["macro_f1"])
        weighted_drop = float(attack_production_metrics["weighted_f1"]) - float(attack_candidate_metrics["weighted_f1"])
        production_key_recall = compute_attack_key_recall_average(attack_production_metrics)
        candidate_key_recall = compute_attack_key_recall_average(attack_candidate_metrics)
        key_recall_drop = 0.0
        if production_key_recall is not None and candidate_key_recall is not None:
            key_recall_drop = float(production_key_recall - candidate_key_recall)
        if macro_drop > float(thresholds["attack_macro_f1_drop_max"]):
            reasons.append("macro_f1_drop_exceeds_threshold")
        if weighted_drop > float(thresholds["attack_weighted_f1_drop_max"]):
            reasons.append("weighted_f1_drop_exceeds_threshold")
        if key_recall_drop > float(thresholds["attack_key_recall_drop_max"]):
            reasons.append("key_recall_drop_exceeds_threshold")
        decision["attack_classifier"]["reasons"] = reasons
        decision["attack_classifier"]["macro_f1_gain"] = float(attack_candidate_metrics["macro_f1"]) - float(
            attack_production_metrics["macro_f1"]
        )
        if not reasons:
            promotion = promote_attack_candidate(
                Path(str(attack_summary["candidate_artifact_path"])),
                attack_production_artifact,
                production_root,
                run_id,
            )
            decision["attack_classifier"].update(promotion)

    attribution_candidate_metrics = attribution_summary.get("candidate_metrics")
    attribution_production_metrics = attribution_summary.get("production_metrics")
    if attribution_summary.get("triggered") and attribution_candidate_metrics and attribution_production_metrics:
        decision["attribution_analysis"]["evaluated"] = True
        reasons = []
        mae_ratio = float(attribution_candidate_metrics["mae"]) / max(float(attribution_production_metrics["mae"]), 1e-12)
        rmse_ratio = float(attribution_candidate_metrics["rmse"]) / max(float(attribution_production_metrics["rmse"]), 1e-12)
        r2_drop = float(attribution_production_metrics["r2"]) - float(attribution_candidate_metrics["r2"])
        if mae_ratio > float(thresholds["attribution_mae_ratio_max"]):
            reasons.append("mae_ratio_exceeds_threshold")
        if rmse_ratio > float(thresholds["attribution_rmse_ratio_max"]):
            reasons.append("rmse_ratio_exceeds_threshold")
        if r2_drop > float(thresholds["attribution_r2_drop_max"]):
            reasons.append("r2_drop_exceeds_threshold")
        decision["attribution_analysis"]["reasons"] = reasons
        if not reasons:
            promotion = promote_attribution_candidate(
                Path(str(attribution_summary["candidate_artifact_path"])),
                attribution_production_artifact,
                production_root,
                run_id,
            )
            decision["attribution_analysis"].update(promotion)

    return decision


def increment_registry_usage(registry: dict[str, Any], used_experiment_dirs: list[Path], run_id: str) -> None:
    lookup = registry_lookup(registry)
    now_text = datetime.now().isoformat(timespec="seconds")
    for experiment_dir in used_experiment_dirs:
        record = lookup.get(str(experiment_dir.resolve()))
        if record is None:
            continue
        record["times_sampled"] = int(record.get("times_sampled", 0)) + 1
        record["last_sampled_at"] = now_text
        record["last_used_in_retrain_run_id"] = run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate retraining for 2.2module archived experiments.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name(DEFAULT_CONFIG_NAME)),
        help="Path to the JSONC config file.",
    )
    parser.add_argument("--pipeline-summary", required=True, help="Path to pipeline_summary.json produced by full run.")
    parser.add_argument("--attack-artifact", required=True, help="Current production attack artifact path.")
    parser.add_argument("--attribution-artifact", required=True, help="Current production attribution artifact path.")
    parser.add_argument("--summary-output", required=True, help="Path to write candidate retraining summary JSON.")
    parser.add_argument("--run-id", default="", help="Optional stable run id for reports and promotion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(repo_root, args.config)
    summary_output = resolve_path(repo_root, args.summary_output)
    pipeline_summary_path = resolve_path(repo_root, args.pipeline_summary)
    pipeline_summary = load_pipeline_summary(pipeline_summary_path)
    config = load_jsonc(config_path)
    retraining_config = merge_retraining_config(config)
    target_field = str(config["attribution_analysis"].get("target_field", "fused_score"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    registry_path = resolve_path(repo_root, str(retraining_config["registry_path"]))
    candidate_root = resolve_path(repo_root, str(retraining_config["candidate_root"]))
    production_root = resolve_path(repo_root, str(retraining_config["production_root"]))
    attack_artifact_path = resolve_path(repo_root, args.attack_artifact)
    attribution_artifact_path = resolve_path(repo_root, args.attribution_artifact)
    holdout_ratio = float(retraining_config["holdout_ratio"])
    history_attack_ratio = float(retraining_config["history_sampling_ratio"]["attack"])
    history_attr_ratio = float(retraining_config["history_sampling_ratio"]["attribution"])
    max_reuse_per_experiment = int(retraining_config["max_reuse_per_experiment"])
    trigger_thresholds = dict(retraining_config["trigger_thresholds"])
    promotion_thresholds = dict(retraining_config["promotion_thresholds"])
    auto_promote_candidate = bool(retraining_config.get("auto_promote_candidate", False))

    archive_summary = pipeline_summary.get("archive_summary", {})
    archived_dirs = archive_summary.get("archived_dirs", []) if isinstance(archive_summary, dict) else []
    if not isinstance(archived_dirs, list):
        archived_dirs = []

    no_attack_threshold, ambiguity_margin = load_attack_thresholds(attack_artifact_path)

    registry = load_registry(registry_path)
    registry, new_records = upsert_registry_records(
        registry=registry,
        archived_dirs=archived_dirs,
        target_field=target_field,
        no_attack_threshold=no_attack_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    new_experiment_dirs = [Path(str(record["experiment_dir"])).resolve() for record in new_records]
    new_experiment_dir_set = {str(path) for path in new_experiment_dirs}

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_run_root = candidate_root / run_id
    candidate_run_root.mkdir(parents=True, exist_ok=True)
    attack_candidate_root = candidate_run_root / "attack_classifier"
    attribution_candidate_root = candidate_run_root / "attribution_analysis"
    attack_candidate_root.mkdir(parents=True, exist_ok=True)
    attribution_candidate_root.mkdir(parents=True, exist_ok=True)

    attack_label_counts = merge_label_counts(new_records)
    attack_new_grouped_sample_count = int(sum(int(record.get("attack_grouped_sample_count", 0)) for record in new_records))
    attack_new_attack_sample_count = int(sum(int(record.get("attack_attack_sample_count", 0)) for record in new_records))
    attack_label_coverage = len([label for label, count in attack_label_counts.items() if int(count) > 0])
    attack_min_label_count = min((int(count) for count in attack_label_counts.values()), default=0)

    attack_skip_reasons: list[str] = []
    attack_new_grouped_min = int(trigger_thresholds["attack_new_grouped_min"])
    attack_new_attack_min = int(trigger_thresholds["attack_new_attack_min"])
    attack_label_coverage_min = int(trigger_thresholds["attack_label_coverage_min"])
    attack_min_label_count_threshold = int(trigger_thresholds["attack_min_label_count"])
    if attack_new_grouped_sample_count < attack_new_grouped_min:
        attack_skip_reasons.append("attack_new_grouped_below_threshold")
    if attack_new_attack_sample_count < attack_new_attack_min:
        attack_skip_reasons.append("attack_new_attack_below_threshold")
    if attack_label_coverage < attack_label_coverage_min:
        attack_skip_reasons.append("attack_label_coverage_below_threshold")
    if attack_min_label_count < attack_min_label_count_threshold:
        attack_skip_reasons.append("attack_min_label_count_below_threshold")

    attack_history_target = int(math.ceil(attack_new_grouped_sample_count * history_attack_ratio))
    attack_history_records = select_history_records(
        registry,
        exclude_dirs=new_experiment_dir_set,
        sample_key="attack_grouped_sample_count",
        desired_count=attack_history_target,
        max_reuse_per_experiment=max_reuse_per_experiment,
    )
    attack_history_grouped_sample_count = int(
        sum(int(record.get("attack_grouped_sample_count", 0)) for record in attack_history_records)
    )
    attack_combined_grouped_sample_count = attack_new_grouped_sample_count + attack_history_grouped_sample_count
    attack_total_grouped_min = int(trigger_thresholds["attack_total_grouped_min"])
    if attack_combined_grouped_sample_count < attack_total_grouped_min:
        attack_skip_reasons.append("attack_combined_grouped_below_threshold")
    attack_threshold_checks = [
        build_threshold_check(
            "attack_new_grouped_min",
            attack_new_grouped_sample_count,
            attack_new_grouped_min,
            attack_new_grouped_sample_count >= attack_new_grouped_min,
        ),
        build_threshold_check(
            "attack_new_attack_min",
            attack_new_attack_sample_count,
            attack_new_attack_min,
            attack_new_attack_sample_count >= attack_new_attack_min,
        ),
        build_threshold_check(
            "attack_label_coverage_min",
            attack_label_coverage,
            attack_label_coverage_min,
            attack_label_coverage >= attack_label_coverage_min,
        ),
        build_threshold_check(
            "attack_min_label_count",
            attack_min_label_count,
            attack_min_label_count_threshold,
            attack_min_label_count >= attack_min_label_count_threshold,
        ),
        build_threshold_check(
            "attack_total_grouped_min",
            attack_combined_grouped_sample_count,
            attack_total_grouped_min,
            attack_combined_grouped_sample_count >= attack_total_grouped_min,
        ),
    ]

    if attack_skip_reasons:
        attack_summary = build_attack_skip_summary(new_records, attack_skip_reasons)
    else:
        attack_experiment_dirs = new_experiment_dirs + [
            Path(str(record["experiment_dir"])).resolve() for record in attack_history_records
        ]
        attack_summary_inputs = [
            resolve_session_dir(experiment_dir) / "output_summary.txt"
            for experiment_dir in attack_experiment_dirs
            if (resolve_session_dir(experiment_dir) / "output_summary.txt").exists()
        ]
        merged_summary_path = attack_candidate_root / "merged_output_summary_candidate.txt"
        merged_csv_path = attack_candidate_root / "merged_output_candidate.csv"
        manifest_path = attack_candidate_root / "merged_output_sources_candidate.json"
        merge_summary_files(attack_summary_inputs, merged_summary_path, manifest_path=manifest_path)
        build_output_csv(merged_summary_path, merged_csv_path)
        candidate_attack_artifact = attack_candidate_root / "attack_type_classifier_candidate.pkl"
        candidate_attack_result = train_attack_candidate(
            csv_path=merged_csv_path,
            artifact_path=candidate_attack_artifact,
            production_artifact_path=attack_artifact_path,
            holdout_ratio=holdout_ratio,
            random_state=int(config["global"].get("random_state", 42)),
            feature_mode=str(config["attack_classifier"].get("feature_mode", DEFAULT_ATTACK_FEATURE_MODE)),
            rf_n_estimators=int(config["attack_classifier"].get("rf_n_estimators", DEFAULT_RF_ESTIMATORS)),
            rf_max_depth=config["attack_classifier"].get("rf_max_depth"),
            no_attack_threshold=no_attack_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        attack_summary = {
            "triggered": True,
            "skip_reasons": [],
            "new_experiment_count": len(new_records),
            "new_grouped_sample_count": attack_new_grouped_sample_count,
            "new_attack_sample_count": attack_new_attack_sample_count,
            "new_label_counts": dict(attack_label_counts),
            "history_sampled_experiments": [str(record["experiment_dir"]) for record in attack_history_records],
            "history_grouped_sample_count": attack_history_grouped_sample_count,
            "combined_grouped_sample_count": attack_combined_grouped_sample_count,
            **candidate_attack_result,
        }
    attack_summary["threshold_checks"] = attack_threshold_checks

    new_attr_rows = collect_attribution_rows(new_experiment_dirs, target_field)
    new_attr_sample_count = len(new_attr_rows)
    new_attr_constellation_count = len({str(record["constellation_id"]) for record in new_records})
    new_attr_target_unique_count = len({round(float(row["fail_score"]), 12) for row in new_attr_rows})
    new_attr_constant = new_attr_target_unique_count <= 1

    attribution_skip_reasons: list[str] = []
    attribution_new_sample_min = int(trigger_thresholds["attribution_new_sample_min"])
    attribution_target_unique_min = int(trigger_thresholds["attribution_target_unique_min"])
    attribution_constellation_coverage_min = int(trigger_thresholds["attribution_constellation_coverage_min"])
    attribution_total_sample_min = int(trigger_thresholds["attribution_total_sample_min"])
    if new_attr_sample_count < attribution_new_sample_min:
        attribution_skip_reasons.append("attribution_new_sample_below_threshold")
    if new_attr_target_unique_count < attribution_target_unique_min:
        attribution_skip_reasons.append("attribution_target_unique_below_threshold")
    if new_attr_constant:
        attribution_skip_reasons.append("attribution_target_is_constant")

    attribution_history_target = int(math.ceil(new_attr_sample_count * history_attr_ratio))
    attribution_history_records = select_history_records(
        registry,
        exclude_dirs=new_experiment_dir_set,
        sample_key="attribution_sample_count",
        desired_count=attribution_history_target,
        max_reuse_per_experiment=max_reuse_per_experiment,
    )
    attribution_history_sample_count = int(
        sum(int(record.get("attribution_sample_count", 0)) for record in attribution_history_records)
    )
    attribution_combined_experiment_dirs = new_experiment_dirs + [
        Path(str(record["experiment_dir"])).resolve() for record in attribution_history_records
    ]
    combined_constellation_count = len({str(path.parent.name) for path in attribution_combined_experiment_dirs})
    combined_attr_sample_count = new_attr_sample_count + attribution_history_sample_count
    if combined_constellation_count < attribution_constellation_coverage_min:
        attribution_skip_reasons.append("attribution_constellation_coverage_below_threshold")
    if combined_attr_sample_count < attribution_total_sample_min:
        attribution_skip_reasons.append("attribution_combined_sample_below_threshold")
    attribution_threshold_checks = [
        build_threshold_check(
            "attribution_new_sample_min",
            new_attr_sample_count,
            attribution_new_sample_min,
            new_attr_sample_count >= attribution_new_sample_min,
        ),
        build_threshold_check(
            "attribution_target_unique_min",
            new_attr_target_unique_count,
            attribution_target_unique_min,
            new_attr_target_unique_count >= attribution_target_unique_min,
        ),
        {
            "name": "attribution_target_is_not_constant",
            "actual": not new_attr_constant,
            "threshold": True,
            "passed": not new_attr_constant,
        },
        build_threshold_check(
            "attribution_constellation_coverage_min",
            combined_constellation_count,
            attribution_constellation_coverage_min,
            combined_constellation_count >= attribution_constellation_coverage_min,
        ),
        build_threshold_check(
            "attribution_total_sample_min",
            combined_attr_sample_count,
            attribution_total_sample_min,
            combined_attr_sample_count >= attribution_total_sample_min,
        ),
    ]

    if attribution_skip_reasons:
        attribution_summary = build_attribution_skip_summary(
            new_records,
            attribution_skip_reasons,
            new_attr_constellation_count,
        )
    else:
        attribution_csv_path = attribution_candidate_root / "fail_score_training_candidate.csv"
        attribution_audit_path = attribution_candidate_root / "fail_score_training_candidate_audit.json"
        attr_dataset_info = build_combined_attribution_csv(
            experiment_dirs=attribution_combined_experiment_dirs,
            target_field=target_field,
            output_csv_path=attribution_csv_path,
            audit_path=attribution_audit_path,
        )
        candidate_attr_artifact = (
            attribution_candidate_root / f"fail_score_contribution_model_{target_field}_candidate.pkl"
        )
        candidate_attr_result = train_attribution_candidate(
            csv_path=attribution_csv_path,
            artifact_path=candidate_attr_artifact,
            production_artifact_path=attribution_artifact_path,
            holdout_ratio=holdout_ratio,
            random_state=int(config["global"].get("random_state", 42)),
            safe_threshold=float(config["attribution_analysis"].get("safe_threshold", 0.3)),
        )
        attribution_summary = {
            "triggered": True,
            "skip_reasons": [],
            "new_experiment_count": len(new_records),
            "new_sample_count": new_attr_sample_count,
            "new_constellation_count": new_attr_constellation_count,
            "new_target_unique_count": new_attr_target_unique_count,
            "history_sampled_experiments": [str(record["experiment_dir"]) for record in attribution_history_records],
            "history_sample_count": attribution_history_sample_count,
            "combined_sample_count": combined_attr_sample_count,
            "combined_constellation_count": combined_constellation_count,
            "dataset_info": attr_dataset_info,
            **candidate_attr_result,
        }
    attribution_summary["threshold_checks"] = attribution_threshold_checks

    used_experiments: list[Path] = []
    if attack_summary.get("triggered"):
        used_experiments.extend(new_experiment_dirs)
        used_experiments.extend(Path(path).resolve() for path in attack_summary["history_sampled_experiments"])
    if attribution_summary.get("triggered"):
        used_experiments.extend(new_experiment_dirs)
        used_experiments.extend(Path(path).resolve() for path in attribution_summary["history_sampled_experiments"])
    if used_experiments:
        unique_used = []
        seen_used: set[str] = set()
        for experiment_dir in used_experiments:
            normalized = str(experiment_dir.resolve())
            if normalized in seen_used:
                continue
            seen_used.add(normalized)
            unique_used.append(experiment_dir.resolve())
        increment_registry_usage(registry, unique_used, run_id)

    promotion_decision = build_promotion_decision(
        attack_summary=attack_summary,
        attribution_summary=attribution_summary,
        thresholds=promotion_thresholds,
        auto_promote_candidate=auto_promote_candidate,
        candidate_root=candidate_root,
        production_root=production_root,
        attack_production_artifact=attack_artifact_path,
        attribution_production_artifact=attribution_artifact_path,
        run_id=run_id,
    )

    summary_payload = {
        "run_id": run_id,
        "config_path": str(config_path),
        "pipeline_summary_path": str(pipeline_summary_path),
        "registry_path": str(registry_path),
        "candidate_root": str(candidate_root),
        "production_root": str(production_root),
        "new_archived_experiment_count": len(new_records),
        "new_archived_experiment_dirs": [str(path) for path in new_experiment_dirs],
        "attack_classifier": attack_summary,
        "attribution_analysis": attribution_summary,
        "promotion_decision": promotion_decision,
    }
    threshold_gate_summary = {
        "run_id": run_id,
        "attack_classifier": {
            "triggered": bool(attack_summary.get("triggered")),
            "skip_reasons": list(attack_summary.get("skip_reasons", [])),
            "threshold_checks": list(attack_summary.get("threshold_checks", [])),
        },
        "attribution_analysis": {
            "triggered": bool(attribution_summary.get("triggered")),
            "skip_reasons": list(attribution_summary.get("skip_reasons", [])),
            "threshold_checks": list(attribution_summary.get("threshold_checks", [])),
        },
    }

    summary_output.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    threshold_gate_path = summary_output.with_name("threshold_gate_summary.json")
    threshold_gate_path.write_text(
        json.dumps(threshold_gate_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_registry(registry_path, registry)
    promotion_decision_path = summary_output.with_name("promotion_decision.json")
    promotion_decision_path.write_text(
        json.dumps(promotion_decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"Saved candidate retraining summary to {summary_output}")


if __name__ == "__main__":
    main()
