from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import numpy as np


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
    DEFAULT_ATTACK_MODEL_TYPE,
    DEFAULT_RF_ESTIMATORS,
    compute_classification_metrics,
    derive_attack_label,
    read_output_csv_rows,
)
from extract_output_csv import build_output_csv  # noqa: E402
from fail_score_contribution_model import (  # noqa: E402
    SCENARIO_PARAM_NAMES,
    FailScoreContributionModel,
)
from merge_output_summaries import merge_summary_files  # noqa: E402


DEFAULT_CONFIG_NAME = "pipeline_config.jsonc"
VALID_RUN_MODES = {"full", "attack_only", "attribution_only"}
VALID_ATTRIBUTION_TARGETS = {"total_membership_v2", "decision_score_v2", "fused_score"}

DEFAULT_ATTACK_ARTIFACT = (
    "failure_and_attribution_analysis/2.2module/attack_classifier/attack_type_classifier_weak_scene_rf_sklearn17.pkl"
)
DEFAULT_ATTRIBUTION_ARTIFACT = (
    "failure_and_attribution_analysis/2.2module/attribution_analysis/"
    "fail_score_contribution_model_fused_score_from_data_root.pkl"
)
DEFAULT_ARCHIVE_ROOT = "failure_and_attribution_analysis/data_archive"
DEFAULT_RETRAINING_REGISTRY_PATH = (
    "failure_and_attribution_analysis/2.2module/model_artifacts/reports/archive_training_registry.json"
)
DEFAULT_RETRAINING_CANDIDATE_ROOT = (
    "failure_and_attribution_analysis/2.2module/model_artifacts/candidate"
)
DEFAULT_RETRAINING_PRODUCTION_ROOT = (
    "failure_and_attribution_analysis/2.2module/model_artifacts/production"
)
RETRAINING_SCRIPT_NAME = "run_candidate_retraining.py"


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


def discover_summary_inputs(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("output_summary.txt")
        if "review_strict_single_fused_mlp" not in str(path).replace("\\", "/")
    )


def build_session_sources(data_root: Path) -> list[dict[str, Any]]:
    summary_inputs = discover_summary_inputs(data_root)
    sources: list[dict[str, Any]] = []
    for source_file_index, summary_path in enumerate(summary_inputs):
        session_dir = summary_path.parent.resolve()
        sources.append(
            {
                "source_file_index": source_file_index,
                "source_session_id": f"session_{source_file_index:04d}",
                "source_output_summary_path": str(summary_path.resolve()),
                "source_session_dir": str(session_dir),
            }
        )
    return sources


def build_archive_disabled_summary(reason: str, archive_root: Path | None = None) -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": reason,
        "archive_root": None if archive_root is None else str(archive_root),
        "attempted_count": 0,
        "archived_count": 0,
        "skipped_due_to_collision_count": 0,
        "skipped_invalid_source_count": 0,
        "archived_dirs": [],
        "skipped_due_to_collision": [],
        "skipped_invalid_source": [],
    }


def build_retraining_disabled_summary(reason: str, summary_path: Path | None = None) -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": reason,
        "summary_path": None if summary_path is None else str(summary_path),
        "run_id": None,
        "attack_classifier": None,
        "attribution_analysis": None,
        "promotion_decision": None,
    }


def default_retraining_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "auto_promote_candidate": False,
        "registry_path": DEFAULT_RETRAINING_REGISTRY_PATH,
        "candidate_root": DEFAULT_RETRAINING_CANDIDATE_ROOT,
        "production_root": DEFAULT_RETRAINING_PRODUCTION_ROOT,
        "history_sampling_ratio": {
            "attack": 1.5,
            "attribution": 2.0,
        },
        "max_reuse_per_experiment": 3,
        "holdout_ratio": 0.2,
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
    if "history_sampling_ratio" in incoming and isinstance(incoming["history_sampling_ratio"], dict):
        merged["history_sampling_ratio"].update(incoming["history_sampling_ratio"])
    if "trigger_thresholds" in incoming and isinstance(incoming["trigger_thresholds"], dict):
        merged["trigger_thresholds"].update(incoming["trigger_thresholds"])
    if "promotion_thresholds" in incoming and isinstance(incoming["promotion_thresholds"], dict):
        merged["promotion_thresholds"].update(incoming["promotion_thresholds"])
    for key, value in incoming.items():
        if key in {"history_sampling_ratio", "trigger_thresholds", "promotion_thresholds"}:
            continue
        merged[key] = value
    return merged


def derive_experiment_root(
    data_root: Path,
    session_source: dict[str, Any],
) -> dict[str, Any]:
    session_dir = Path(str(session_source["source_session_dir"])).resolve()
    data_root_resolved = data_root.resolve()

    try:
        relative_session_dir = session_dir.relative_to(data_root_resolved)
    except ValueError:
        return {
            "valid": False,
            "reason": "session_dir_outside_data_root",
            "source_session_dir": str(session_dir),
        }

    relative_parts = relative_session_dir.parts
    if len(relative_parts) == 3 and relative_parts[-1] == "current_session":
        experiment_dir = session_dir.parent
    elif len(relative_parts) == 2:
        # Support flat layouts like data/<constellation>/<session_dir>/output_summary.txt
        experiment_dir = session_dir
    else:
        return {
            "valid": False,
            "reason": "unexpected_session_layout",
            "source_session_dir": str(session_dir),
            "relative_session_dir": relative_session_dir.as_posix(),
        }

    return {
        "valid": True,
        "constellation_id": relative_parts[0],
        "experiment_dir": experiment_dir,
        "source_session_dir": str(session_dir),
    }


def archive_processed_experiments(
    data_root: Path,
    archive_root: Path,
    session_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    data_root = data_root.resolve()
    archive_root = archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)

    unique_experiments: dict[Path, dict[str, Any]] = {}
    skipped_invalid_source: list[dict[str, Any]] = []

    for session_source in session_sources:
        candidate = derive_experiment_root(data_root, session_source)
        if not bool(candidate["valid"]):
            skipped_invalid_source.append(candidate)
            continue
        experiment_dir = Path(str(candidate["experiment_dir"])).resolve()
        unique_experiments.setdefault(experiment_dir, candidate)

    archived_dirs: list[dict[str, str]] = []
    skipped_due_to_collision: list[dict[str, str]] = []

    for experiment_dir in sorted(unique_experiments):
        candidate = unique_experiments[experiment_dir]
        constellation_id = str(candidate["constellation_id"])
        target_parent = (archive_root / constellation_id).resolve()
        target_dir = (target_parent / experiment_dir.name).resolve()

        try:
            experiment_dir.relative_to(data_root)
        except ValueError:
            skipped_invalid_source.append(
                {
                    "valid": False,
                    "reason": "experiment_dir_outside_data_root",
                    "experiment_dir": str(experiment_dir),
                }
            )
            continue

        try:
            target_dir.relative_to(archive_root)
        except ValueError:
            skipped_invalid_source.append(
                {
                    "valid": False,
                    "reason": "archive_target_outside_archive_root",
                    "experiment_dir": str(experiment_dir),
                    "archive_target_dir": str(target_dir),
                }
            )
            continue

        if not experiment_dir.exists():
            skipped_invalid_source.append(
                {
                    "valid": False,
                    "reason": "experiment_dir_missing",
                    "experiment_dir": str(experiment_dir),
                }
            )
            continue

        if target_dir.exists():
            skipped_due_to_collision.append(
                {
                    "source_dir": str(experiment_dir),
                    "target_dir": str(target_dir),
                }
            )
            continue

        target_parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(experiment_dir), str(target_dir))
        archived_dirs.append(
            {
                "source_dir": str(experiment_dir),
                "target_dir": str(target_dir),
            }
        )

    return {
        "enabled": True,
        "archive_root": str(archive_root),
        "attempted_count": len(unique_experiments),
        "archived_count": len(archived_dirs),
        "skipped_due_to_collision_count": len(skipped_due_to_collision),
        "skipped_invalid_source_count": len(skipped_invalid_source),
        "archived_dirs": archived_dirs,
        "skipped_due_to_collision": skipped_due_to_collision,
        "skipped_invalid_source": skipped_invalid_source,
    }


def parse_test_summary_json_lines(
    evalu_path: Path,
    source: dict[str, Any],
    target_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with evalu_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("TEST_SUMMARY_JSON:"):
                continue
            payload = json.loads(line.split(":", 1)[1].strip())
            scenario = payload.get("scenario")
            if not isinstance(scenario, dict):
                raise ValueError(f"TEST_SUMMARY_JSON missing scenario dictionary: {evalu_path}")
            if target_field not in payload:
                raise KeyError(f"TEST_SUMMARY_JSON is missing target field '{target_field}': {evalu_path}")

            rows.append(
                {
                    "source_file_index": int(source["source_file_index"]),
                    "source_session_id": str(source["source_session_id"]),
                    "source_session_dir": str(source["source_session_dir"]),
                    "original_round_index": int(payload["round_index"]),
                    "original_test_id": int(payload["test_id"]),
                    "target_field": target_field,
                    "target_value": float(payload[target_field]),
                    "true_failure": payload.get("true_failure"),
                    "system_failure": payload.get("system_failure"),
                    "terminal_risk_score": payload.get("terminal_risk_score"),
                    "final_failure_probability": payload.get("final_failure_probability"),
                    "scenario": scenario,
                }
            )
    return rows


def write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_attack_outputs(base_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "merged_summary": base_dir / f"merged_output_summary_{prefix}.txt",
        "merged_csv": base_dir / f"merged_output_{prefix}.csv",
        "sources_manifest": base_dir / f"merged_output_sources_{prefix}.json",
        "predictions": base_dir / f"attack_type_predictions_{prefix}.jsonl",
        "summary_json": base_dir / f"attack_type_prediction_summary_{prefix}.json",
    }


def build_attribution_outputs(base_dir: Path, target_field: str) -> dict[str, Path]:
    return {
        "predictions": base_dir / f"fail_score_contributions_{target_field}.jsonl",
        "summary_json": base_dir / f"fail_score_contribution_summary_{target_field}.json",
    }


def build_prediction_key(
    source_file_index: int,
    original_round_index: int,
    original_test_id: int,
) -> tuple[int, int, int]:
    return (int(source_file_index), int(original_round_index), int(original_test_id))


def top_contributors(
    contribution_by_feature: dict[str, float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        contribution_by_feature.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature_name": feature_name,
            "contribution": float(contribution),
            "rank": rank + 1,
        }
        for rank, (feature_name, contribution) in enumerate(ranked[:top_k])
    ]


def summarize_attack_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any] | None:
    if not true_labels:
        return None
    metrics = compute_classification_metrics(true_labels, pred_labels, ATTACK_LABELS)
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "confusion_matrix": [
            [int(value) for value in row]
            for row in metrics["confusion_matrix"]
        ],
        "labels": list(ATTACK_LABELS),
    }


def run_attack_pipeline(
    repo_root: Path,
    output_root: Path,
    attack_config: dict[str, Any],
    summary_inputs: list[Path],
) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    artifact_path = resolve_path(repo_root, str(attack_config["artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Attack classifier artifact not found: {artifact_path}")

    attack_output_dir = output_root / "attack_classifier"
    attack_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(attack_config.get("output_prefix", "attack_inference"))
    outputs = build_attack_outputs(attack_output_dir, prefix)
    direct_input_csv_value = str(attack_config.get("input_csv", "") or "").strip()
    direct_input_csv_path: Path | None = None
    if direct_input_csv_value:
        direct_input_csv_path = resolve_path(repo_root, direct_input_csv_value)
        if not direct_input_csv_path.exists():
            raise FileNotFoundError(f"Attack input_csv not found: {direct_input_csv_path}")
        outputs["merged_summary"].write_text(
            (
                "DIRECT_ATTACK_INPUT_CSV\n"
                f"source_csv={direct_input_csv_path}\n"
                "note=attack pipeline skipped summary merge and reused a prebuilt merged csv.\n"
            ),
            encoding="utf-8",
        )
        outputs["sources_manifest"].write_text(
            json.dumps(
                {
                    "mode": "direct_input_csv",
                    "source_csv": str(direct_input_csv_path),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if direct_input_csv_path.resolve() != outputs["merged_csv"].resolve():
            shutil.copyfile(direct_input_csv_path, outputs["merged_csv"])
    else:
        merge_summary_files(summary_inputs, outputs["merged_summary"], manifest_path=outputs["sources_manifest"])
        build_output_csv(outputs["merged_summary"], outputs["merged_csv"])

    rows = read_output_csv_rows(outputs["merged_csv"])
    classifier = AttackTypeClassifier.load(artifact_path)
    artifact_payload = joblib.load(artifact_path)
    no_attack_threshold = float(artifact_payload.get("no_attack_threshold", 1e-12))
    ambiguity_margin = float(artifact_payload.get("ambiguity_margin", 0.0))

    prediction_rows: list[dict[str, Any]] = []
    prediction_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    true_labels: list[str] = []
    pred_labels: list[str] = []

    for row in rows:
        predicted_label = classifier.predict_row(row)
        true_label = derive_attack_label(
            row,
            no_attack_threshold=no_attack_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        record = {
            "source_file_index": int(row["source_file_index"]),
            "source_session_id": str(row["source_session_id"]),
            "source_session_dir": str(row["source_session_dir"]),
            "merged_round_index": int(row["round_index"]),
            "merged_test_id": int(row["test_id"]),
            "original_round_index": int(row["original_round_index"]),
            "original_test_id": int(row["original_test_id"]),
            "selected_step_count": int(row["selected_step_count"]),
            "selected_step_indices": str(row["selected_step_indices"]),
            "true_attack_type": true_label,
            "predicted_attack_type": predicted_label,
        }
        prediction_rows.append(record)
        prediction_index[
            build_prediction_key(
                record["source_file_index"],
                record["original_round_index"],
                record["original_test_id"],
            )
        ] = record
        true_labels.append(true_label)
        pred_labels.append(predicted_label)

    write_jsonl(outputs["predictions"], prediction_rows)
    summary_payload = {
        "sample_count": len(prediction_rows),
        "artifact_path": str(artifact_path),
        "model_type": str(artifact_payload.get("model_type", "legacy_ca44")),
        "feature_mode": artifact_payload.get("feature_mode"),
        "metrics": summarize_attack_metrics(true_labels, pred_labels),
    }
    outputs["summary_json"].write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    public_summary = {
        "enabled": True,
        "artifact_path": str(artifact_path),
        "input_csv": None if direct_input_csv_path is None else str(direct_input_csv_path),
        "merged_csv": str(outputs["merged_csv"]),
        "predictions_jsonl": str(outputs["predictions"]),
        "summary_json": str(outputs["summary_json"]),
        "sample_count": len(prediction_rows),
        "metrics": summary_payload["metrics"],
    }
    return public_summary, prediction_index


def run_attribution_pipeline(
    repo_root: Path,
    output_root: Path,
    attribution_config: dict[str, Any],
    session_sources: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    artifact_path = resolve_path(repo_root, str(attribution_config["artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Attribution artifact not found: {artifact_path}")

    attr_output_dir = output_root / "attribution_analysis"
    attr_output_dir.mkdir(parents=True, exist_ok=True)
    target_field = str(attribution_config.get("target_field", "fused_score"))
    outputs = build_attribution_outputs(attr_output_dir, target_field)

    model = FailScoreContributionModel.load(artifact_path)
    prediction_rows: list[dict[str, Any]] = []
    prediction_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    contribution_vectors: list[list[float]] = []
    target_values: list[float] = []
    predicted_values: list[float] = []
    absolute_errors: list[float] = []
    direct_input_csv_value = str(attribution_config.get("input_csv", "") or "").strip()
    direct_input_csv_path: Path | None = None
    base_rows: list[dict[str, Any]] = []
    if direct_input_csv_value:
        direct_input_csv_path = resolve_path(repo_root, direct_input_csv_value)
        if not direct_input_csv_path.exists():
            raise FileNotFoundError(f"Attribution input_csv not found: {direct_input_csv_path}")
        with direct_input_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scenario = {
                    feature_name: float(row[feature_name])
                    for feature_name in SCENARIO_PARAM_NAMES
                }
                original_round_index = int(row.get("original_round_index") or row.get("round_index") or 0)
                original_test_id = int(row.get("original_test_id") or row.get("test_id") or 0)
                base_rows.append(
                    {
                        "source_file_index": int(row.get("source_file_index") or 0),
                        "source_session_id": str(row.get("source_session_id") or "direct_input_csv"),
                        "source_session_dir": str(row.get("source_session_dir") or str(direct_input_csv_path)),
                        "original_round_index": original_round_index,
                        "original_test_id": original_test_id,
                        "target_field": target_field,
                        "target_value": float(row.get("fail_score", row[target_field])),
                        "true_failure": row.get("true_failure"),
                        "system_failure": row.get("system_failure"),
                        "terminal_risk_score": row.get("terminal_risk_score"),
                        "final_failure_probability": row.get("final_failure_probability"),
                        "scenario": scenario,
                    }
                )
    else:
        for source in session_sources:
            rounds_root = Path(str(source["source_session_dir"])) / "rounds"
            if not rounds_root.exists():
                continue
            for evalu_path in sorted(rounds_root.glob("round_*/evalu.txt")):
                base_rows.extend(parse_test_summary_json_lines(evalu_path, source, target_field))

    for base_row in base_rows:
        params = [float(base_row["scenario"][name]) for name in SCENARIO_PARAM_NAMES]
        prediction = model.predict_with_contributions(params)
        contribution_list = [float(value) for value in prediction["contribution_list"]]
        predicted_fail_score = float(prediction["fail_score"])
        target_value = float(base_row["target_value"])
        contribution_by_feature = {
            feature_name: contribution_list[index]
            for index, feature_name in enumerate(SCENARIO_PARAM_NAMES)
        }
        record = {
            **base_row,
            "predicted_fail_score": predicted_fail_score,
            "absolute_error": abs(predicted_fail_score - target_value),
            "contribution_list": contribution_list,
            "contribution_by_feature": contribution_by_feature,
            "top_contributors": top_contributors(contribution_by_feature),
        }
        prediction_rows.append(record)
        prediction_index[
            build_prediction_key(
                int(record["source_file_index"]),
                int(record["original_round_index"]),
                int(record["original_test_id"]),
            )
        ] = record
        contribution_vectors.append(contribution_list)
        target_values.append(target_value)
        predicted_values.append(predicted_fail_score)
        absolute_errors.append(float(record["absolute_error"]))

    write_jsonl(outputs["predictions"], prediction_rows)

    contribution_matrix = np.asarray(contribution_vectors, dtype=np.float64)
    if len(contribution_matrix):
        global_mean = np.mean(contribution_matrix, axis=0)
        global_mean_list = [float(value) for value in global_mean.tolist()]
    else:
        global_mean_list = [0.0 for _ in SCENARIO_PARAM_NAMES]

    contribution_by_feature = {
        feature_name: global_mean_list[index]
        for index, feature_name in enumerate(SCENARIO_PARAM_NAMES)
    }
    ranked_features = [
        {
            "feature_name": feature_name,
            "mean_contribution": float(contribution_by_feature[feature_name]),
            "rank": rank + 1,
        }
        for rank, feature_name in enumerate(
            sorted(contribution_by_feature, key=contribution_by_feature.get, reverse=True)
        )
    ]

    summary_payload = {
        "sample_count": len(prediction_rows),
        "artifact_path": str(artifact_path),
        "target_field": target_field,
        "target_mean": mean(target_values) if target_values else None,
        "predicted_mean": mean(predicted_values) if predicted_values else None,
        "mean_absolute_error": mean(absolute_errors) if absolute_errors else None,
        "feature_names": list(SCENARIO_PARAM_NAMES),
        "global_mean_contribution_list": global_mean_list,
        "global_mean_contribution_by_feature": contribution_by_feature,
        "global_ranked_features": ranked_features,
        "model_metadata": {
            "safe_threshold": float(model.safe_threshold),
            "steps": int(model.steps),
            "epsilon": float(model.epsilon),
            "model_fitted": bool(model.model_fitted),
            "constant_fail_score": (
                None if model.constant_fail_score is None else float(model.constant_fail_score)
            ),
        },
    }
    outputs["summary_json"].write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    public_summary = {
        "enabled": True,
        "artifact_path": str(artifact_path),
        "input_csv": None if direct_input_csv_path is None else str(direct_input_csv_path),
        "target_field": target_field,
        "predictions_jsonl": str(outputs["predictions"]),
        "summary_json": str(outputs["summary_json"]),
        "sample_count": len(prediction_rows),
        "mean_absolute_error": summary_payload["mean_absolute_error"],
        "global_top_features": ranked_features[:3],
    }
    return public_summary, prediction_index


def combine_results(
    output_root: Path,
    attack_index: dict[tuple[int, int, int], dict[str, Any]],
    attribution_index: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    combined_path = output_root / "integrated_attack_attribution_results.jsonl"
    combined_rows: list[dict[str, Any]] = []

    keys = sorted(set(attack_index) | set(attribution_index))
    matched_count = 0
    attack_only_count = 0
    attribution_only_count = 0

    for key in keys:
        attack_record = attack_index.get(key)
        attribution_record = attribution_index.get(key)
        if attack_record and attribution_record:
            matched_count += 1
        elif attack_record:
            attack_only_count += 1
        else:
            attribution_only_count += 1

        source = attack_record or attribution_record
        if source is None:
            continue
        combined_rows.append(
            {
                "source_file_index": int(source["source_file_index"]),
                "source_session_id": str(source["source_session_id"]),
                "source_session_dir": str(source["source_session_dir"]),
                "original_round_index": int(source["original_round_index"]),
                "original_test_id": int(source["original_test_id"]),
                "merged_round_index": None if attack_record is None else int(attack_record["merged_round_index"]),
                "merged_test_id": None if attack_record is None else int(attack_record["merged_test_id"]),
                "scenario": None if attribution_record is None else attribution_record["scenario"],
                "attack_classifier": (
                    None
                    if attack_record is None
                    else {
                        "true_attack_type": attack_record["true_attack_type"],
                        "predicted_attack_type": attack_record["predicted_attack_type"],
                    }
                ),
                "attribution_analysis": (
                    None
                    if attribution_record is None
                    else {
                        "target_field": attribution_record["target_field"],
                        "target_value": attribution_record["target_value"],
                        "predicted_fail_score": attribution_record["predicted_fail_score"],
                        "absolute_error": attribution_record["absolute_error"],
                        "top_contributors": attribution_record["top_contributors"],
                        "contribution_by_feature": attribution_record["contribution_by_feature"],
                    }
                ),
            }
        )

    write_jsonl(combined_path, combined_rows)
    return {
        "output_jsonl": str(combined_path),
        "sample_count": len(combined_rows),
        "matched_count": matched_count,
        "attack_only_count": attack_only_count,
        "attribution_only_count": attribution_only_count,
    }


def validate_config(config: dict[str, Any]) -> None:
    run_mode = str(config.get("run_mode", "full"))
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(VALID_RUN_MODES)}, got {run_mode!r}.")

    if "global" not in config:
        raise ValueError("Config must include a 'global' section.")
    if "attack_classifier" not in config or "attribution_analysis" not in config:
        raise ValueError("Config must include both 'attack_classifier' and 'attribution_analysis' sections.")

    target_field = str(config["attribution_analysis"].get("target_field", "fused_score"))
    if target_field not in VALID_ATTRIBUTION_TARGETS:
        raise ValueError(
            f"attribution_analysis.target_field must be one of {sorted(VALID_ATTRIBUTION_TARGETS)}, got {target_field!r}."
        )

    attack_enabled = bool(config["attack_classifier"].get("enabled", True))
    attribution_enabled = bool(config["attribution_analysis"].get("enabled", True))
    if not attack_enabled and not attribution_enabled:
        raise ValueError("Both attack_classifier.enabled and attribution_analysis.enabled are false; nothing to run.")

    attack_model_type = str(config["attack_classifier"].get("model_type", DEFAULT_ATTACK_MODEL_TYPE))
    if attack_model_type not in {"random_forest", "legacy_ca44"}:
        raise ValueError(
            f"attack_classifier.model_type must be 'random_forest' or 'legacy_ca44', got {attack_model_type!r}."
        )
    attack_input_csv = config["attack_classifier"].get("input_csv", "")
    if attack_input_csv is not None and not isinstance(attack_input_csv, str):
        raise ValueError("attack_classifier.input_csv must be a string path when provided.")
    attack_feature_mode = str(config["attack_classifier"].get("feature_mode", DEFAULT_ATTACK_FEATURE_MODE))
    if attack_feature_mode not in {"full", "no_constellation", "with_attack_level_max", "weak_scene"}:
        raise ValueError(
            "attack_classifier.feature_mode must be one of "
            "['full', 'no_constellation', 'with_attack_level_max', 'weak_scene']."
        )
    if attack_model_type == "random_forest" and attack_feature_mode != DEFAULT_ATTACK_FEATURE_MODE:
        raise ValueError("Formal random_forest attack classifier currently only supports feature_mode='weak_scene'.")
    attribution_input_csv = config["attribution_analysis"].get("input_csv", "")
    if attribution_input_csv is not None and not isinstance(attribution_input_csv, str):
        raise ValueError("attribution_analysis.input_csv must be a string path when provided.")

    retraining_config = merge_retraining_config(config)
    holdout_ratio = float(retraining_config["holdout_ratio"])
    if not (0.0 < holdout_ratio < 0.5):
        raise ValueError(f"retraining.holdout_ratio must be between 0 and 0.5, got {holdout_ratio!r}.")


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    merged = dict(config)
    global_config = dict(merged.get("global", {}))
    attack_config = dict(merged.get("attack_classifier", {}))
    attribution_config = dict(merged.get("attribution_analysis", {}))

    if args.run_mode:
        merged["run_mode"] = args.run_mode
    if args.data_root:
        global_config["data_root"] = args.data_root
    if args.output_root:
        global_config["output_root"] = args.output_root
    if args.archive_processed:
        global_config["archive_processed"] = True
    if args.archive_root:
        global_config["archive_root"] = args.archive_root

    attack_config.setdefault("output_prefix", "attack_inference")
    attack_config.setdefault("model_type", DEFAULT_ATTACK_MODEL_TYPE)
    attack_config.setdefault("feature_mode", DEFAULT_ATTACK_FEATURE_MODE)
    attack_config.setdefault("rf_n_estimators", DEFAULT_RF_ESTIMATORS)
    attack_config.setdefault("rf_max_depth", None)
    attack_config.setdefault("input_csv", "")
    attack_config.setdefault("artifact_path", DEFAULT_ATTACK_ARTIFACT)
    attack_artifact_path = resolve_path(repo_root, str(attack_config["artifact_path"]))
    if args.skip_attack:
        attack_config["enabled"] = False
        attack_config["auto_enabled_from_artifact"] = False
    elif attack_artifact_path.exists():
        # Default behavior: if a valid classifier artifact is available and the
        # user did not explicitly request skipping attack analysis, enable it.
        attack_config["enabled"] = True
        attack_config["auto_enabled_from_artifact"] = True
    if args.attack_artifact:
        attack_config["artifact_path"] = args.attack_artifact
        attack_artifact_path = resolve_path(repo_root, str(attack_config["artifact_path"]))
        if args.skip_attack:
            attack_config["enabled"] = False
            attack_config["auto_enabled_from_artifact"] = False
        else:
            attack_config["enabled"] = True
            attack_config["auto_enabled_from_artifact"] = attack_artifact_path.exists()
    if args.attack_input_csv:
        attack_config["input_csv"] = args.attack_input_csv
    if args.attack_output_prefix:
        attack_config["output_prefix"] = args.attack_output_prefix

    attribution_config.setdefault("enabled", True)
    attribution_config.setdefault("artifact_path", DEFAULT_ATTRIBUTION_ARTIFACT)
    attribution_config.setdefault("target_field", "fused_score")
    attribution_config.setdefault("input_csv", "")
    if args.skip_attribution:
        attribution_config["enabled"] = False
    if args.attribution_artifact:
        attribution_config["artifact_path"] = args.attribution_artifact
    if args.attribution_input_csv:
        attribution_config["input_csv"] = args.attribution_input_csv
    if args.target_field:
        attribution_config["target_field"] = args.target_field

    merged["global"] = global_config
    merged["attack_classifier"] = attack_config
    merged["attribution_analysis"] = attribution_config
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deployment-time attack classification and fail-score attribution on iterative-test outputs."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name(DEFAULT_CONFIG_NAME)),
        help="Path to the JSONC config file.",
    )
    parser.add_argument("--run-mode", choices=sorted(VALID_RUN_MODES), default=None, help="Override run mode.")
    parser.add_argument("--data-root", default=None, help="Override the iterative-test output root directory.")
    parser.add_argument("--output-root", default=None, help="Override the pipeline output directory.")
    parser.add_argument("--skip-attack", action="store_true", help="Skip attack classifier inference.")
    parser.add_argument("--skip-attribution", action="store_true", help="Skip attribution inference.")
    parser.add_argument("--attack-artifact", default=None, help="Override attack classifier artifact path.")
    parser.add_argument(
        "--attack-input-csv",
        default=None,
        help="Reuse a prebuilt merged attack CSV instead of rebuilding it from data_root.",
    )
    parser.add_argument("--attribution-artifact", default=None, help="Override attribution model artifact path.")
    parser.add_argument(
        "--attribution-input-csv",
        default=None,
        help="Reuse a prebuilt attribution CSV instead of rebuilding it from data_root/rounds.",
    )
    parser.add_argument("--attack-output-prefix", default=None, help="Override attack output filename prefix.")
    parser.add_argument(
        "--archive-processed",
        action="store_true",
        help="Archive processed experiment roots after a successful full run.",
    )
    parser.add_argument(
        "--archive-root",
        default=None,
        help="Override the archive root used with --archive-processed.",
    )
    parser.add_argument(
        "--target-field",
        choices=sorted(VALID_ATTRIBUTION_TARGETS),
        default=None,
        help="Override attribution target field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(repo_root, args.config)
    config = load_jsonc(config_path)
    config = apply_cli_overrides(config, args, repo_root)
    validate_config(config)

    run_mode = str(config.get("run_mode", "full"))
    global_config = config["global"]
    attack_config = config["attack_classifier"]
    attribution_config = config["attribution_analysis"]
    retraining_config = merge_retraining_config(config)

    data_root = resolve_path(repo_root, str(global_config["data_root"]))
    output_root = resolve_path(repo_root, str(global_config["output_root"]))
    archive_root = resolve_path(repo_root, str(global_config.get("archive_root", DEFAULT_ARCHIVE_ROOT)))
    archive_processed = bool(global_config.get("archive_processed", False))
    output_root.mkdir(parents=True, exist_ok=True)

    session_sources = build_session_sources(data_root)
    direct_attack_input_csv = str(attack_config.get("input_csv", "") or "").strip()
    direct_attribution_input_csv = str(attribution_config.get("input_csv", "") or "").strip()
    attack_enabled = bool(attack_config.get("enabled", True))
    attribution_enabled = bool(attribution_config.get("enabled", True))
    needs_session_sources = (
        ((run_mode in {"full", "attack_only"}) and attack_enabled and not direct_attack_input_csv)
        or ((run_mode in {"full", "attribution_only"}) and attribution_enabled and not direct_attribution_input_csv)
        or (archive_processed and run_mode == "full")
        or (bool(retraining_config.get("enabled", False)) and run_mode == "full")
    )
    if not session_sources and needs_session_sources:
        raise FileNotFoundError(f"No output_summary.txt files found under {data_root}")

    summary_inputs = [Path(source["source_output_summary_path"]) for source in session_sources]
    attack_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    attribution_index: dict[tuple[int, int, int], dict[str, Any]] = {}

    pipeline_summary: dict[str, Any] = {
        "config_path": str(config_path),
        "run_mode": run_mode,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "source_session_count": len(session_sources),
        "attack_classifier": {"enabled": False},
        "attribution_analysis": {"enabled": False},
        "integrated_results": None,
        "archive_summary": build_archive_disabled_summary(
            reason="archive_not_requested",
            archive_root=archive_root,
        ),
        "retraining_summary": build_retraining_disabled_summary(reason="retraining_not_requested"),
    }

    if run_mode in {"full", "attack_only"} and bool(attack_config.get("enabled", True)):
        attack_summary, attack_index = run_attack_pipeline(
            repo_root=repo_root,
            output_root=output_root,
            attack_config=attack_config,
            summary_inputs=summary_inputs,
        )
        attack_summary["auto_enabled_from_artifact"] = bool(
            attack_config.get("auto_enabled_from_artifact", False)
        )
        pipeline_summary["attack_classifier"] = attack_summary

    if run_mode in {"full", "attribution_only"} and bool(attribution_config.get("enabled", True)):
        attribution_summary, attribution_index = run_attribution_pipeline(
            repo_root=repo_root,
            output_root=output_root,
            attribution_config=attribution_config,
            session_sources=session_sources,
        )
        pipeline_summary["attribution_analysis"] = attribution_summary

    pipeline_summary["integrated_results"] = combine_results(
        output_root=output_root,
        attack_index=attack_index,
        attribution_index=attribution_index,
    )

    summary_path = output_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if archive_processed:
        if run_mode != "full":
            pipeline_summary["archive_summary"] = build_archive_disabled_summary(
                reason="archive_only_supported_for_full_mode",
                archive_root=archive_root,
            )
        else:
            pipeline_summary["archive_summary"] = archive_processed_experiments(
                data_root=data_root,
                archive_root=archive_root,
                session_sources=session_sources,
            )
        summary_path.write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if bool(retraining_config.get("enabled", False)):
        candidate_root = resolve_path(repo_root, str(retraining_config["candidate_root"]))
        reports_root = resolve_path(repo_root, str(Path(retraining_config["registry_path"]).parent))
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        retraining_summary_path = reports_root / run_id / "candidate_retrain_summary.json"
        if run_mode != "full":
            pipeline_summary["retraining_summary"] = build_retraining_disabled_summary(
                reason="retraining_only_supported_for_full_mode",
                summary_path=retraining_summary_path,
            )
        elif not archive_processed:
            pipeline_summary["retraining_summary"] = build_retraining_disabled_summary(
                reason="retraining_requires_archive_processed",
                summary_path=retraining_summary_path,
            )
        elif not bool(pipeline_summary.get("archive_summary", {}).get("enabled", False)):
            pipeline_summary["retraining_summary"] = build_retraining_disabled_summary(
                reason="archive_not_completed",
                summary_path=retraining_summary_path,
            )
        elif int(pipeline_summary.get("archive_summary", {}).get("archived_count", 0)) <= 0:
            pipeline_summary["retraining_summary"] = build_retraining_disabled_summary(
                reason="no_archived_experiments_for_retraining",
                summary_path=retraining_summary_path,
            )
        else:
            retraining_script_path = CURRENT_DIR / RETRAINING_SCRIPT_NAME
            attack_artifact_path = resolve_path(repo_root, str(attack_config["artifact_path"]))
            attribution_artifact_path = resolve_path(repo_root, str(attribution_config["artifact_path"]))
            retraining_summary_path.parent.mkdir(parents=True, exist_ok=True)
            retraining_cmd = [
                sys.executable,
                str(retraining_script_path),
                "--config",
                str(config_path),
                "--pipeline-summary",
                str(summary_path),
                "--attack-artifact",
                str(attack_artifact_path),
                "--attribution-artifact",
                str(attribution_artifact_path),
                "--summary-output",
                str(retraining_summary_path),
                "--run-id",
                run_id,
            ]
            try:
                subprocess.run(retraining_cmd, cwd=str(repo_root), check=True)
                retraining_summary_payload = json.loads(retraining_summary_path.read_text(encoding="utf-8"))
                pipeline_summary["retraining_summary"] = {
                    "enabled": True,
                    "reason": "completed",
                    "summary_path": str(retraining_summary_path),
                    "threshold_gate_summary_path": str(
                        retraining_summary_path.with_name("threshold_gate_summary.json")
                    ),
                    "run_id": retraining_summary_payload.get("run_id"),
                    "attack_classifier": retraining_summary_payload.get("attack_classifier"),
                    "attribution_analysis": retraining_summary_payload.get("attribution_analysis"),
                    "promotion_decision": retraining_summary_payload.get("promotion_decision"),
                    "candidate_root": str(candidate_root),
                }
            except subprocess.CalledProcessError as exc:
                pipeline_summary["retraining_summary"] = {
                    "enabled": True,
                    "reason": "failed",
                    "summary_path": str(retraining_summary_path),
                    "threshold_gate_summary_path": str(
                        retraining_summary_path.with_name("threshold_gate_summary.json")
                    ),
                    "run_id": run_id,
                    "candidate_root": str(candidate_root),
                    "error_type": "CalledProcessError",
                    "return_code": int(exc.returncode),
                    "command": [str(part) for part in exc.cmd],
                }
        summary_path.write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))
    print(f"Saved unified pipeline summary to {summary_path}")
    archive_summary = pipeline_summary.get("archive_summary")
    if isinstance(archive_summary, dict):
        print("Archive summary:")
        print(json.dumps(archive_summary, ensure_ascii=False, indent=2))
    retraining_summary = pipeline_summary.get("retraining_summary")
    if isinstance(retraining_summary, dict):
        print("Retraining summary:")
        print(json.dumps(retraining_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
