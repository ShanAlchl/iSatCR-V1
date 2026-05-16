from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

CURRENT_DIR = Path(__file__).resolve().parent
ATTRIBUTION_MODULE_DIR = CURRENT_DIR / "attribution_analysis"
if str(ATTRIBUTION_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(ATTRIBUTION_MODULE_DIR))

from fail_score_contribution_model import SCENARIO_PARAM_NAMES


AUDIT_FIELDS = [
    "total_membership",
    "total_membership_v2",
    "decision_score",
    "decision_score_v2",
    "fused_score",
    "final_failure_probability",
    "terminal_risk_score",
    "system_failure",
    "true_failure",
    "terminal_hard_failure",
]

DEFAULT_TARGET_FIELD = "fused_score"
SUPPORTED_TARGET_FIELDS = [
    "total_membership_v2",
    "decision_score_v2",
    "fused_score",
]


def parse_test_summary_lines(
    rounds_root: Path,
    target_field: str = DEFAULT_TARGET_FIELD,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_dir in sorted(path for path in rounds_root.glob("round_*") if path.is_dir()):
        evalu_path = round_dir / "evalu.txt"
        if evalu_path.exists():
            with evalu_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.startswith("TEST_SUMMARY_JSON:"):
                        continue
                    payload = json.loads(line.split(":", 1)[1].strip())
                    row = build_output_row(payload, target_field=target_field)
                    rows.append(row)
            continue

        failure_scores_path = round_dir / "failure_scores.jsonl"
        if failure_scores_path.exists():
            with failure_scores_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    row = build_output_row(payload, target_field=target_field)
                    rows.append(row)
            continue
    return rows


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def build_output_row(
    payload: dict[str, object],
    target_field: str = DEFAULT_TARGET_FIELD,
) -> dict[str, object]:
    scenario = payload["scenario"]
    if not isinstance(scenario, dict):
        raise ValueError("TEST_SUMMARY_JSON missing scenario dictionary.")
    if target_field not in payload:
        raise KeyError(f"TEST_SUMMARY_JSON is missing target field '{target_field}'.")

    row: dict[str, object] = {
        "round_index": int(payload["round_index"]),
        "test_id": int(payload["test_id"]),
        "fail_score": float(payload[target_field]),
    }
    for feature_name in SCENARIO_PARAM_NAMES:
        if feature_name not in scenario:
            raise KeyError(f"Scenario is missing feature '{feature_name}'.")
        row[feature_name] = scenario[feature_name]

    for field_name in AUDIT_FIELDS:
        row[field_name] = payload.get(field_name)
    return row


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["round_index", "test_id", *SCENARIO_PARAM_NAMES, "fail_score", *AUDIT_FIELDS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_audit(rows: list[dict[str, object]], target_field: str = DEFAULT_TARGET_FIELD) -> dict[str, object]:
    if not rows:
        raise ValueError("No TEST_SUMMARY_JSON rows were found under the supplied rounds root.")

    fail_scores = [float(row["fail_score"]) for row in rows]
    round_ids = sorted({int(row["round_index"]) for row in rows})
    sample_preview = [
        {
            "round_index": int(row["round_index"]),
            "test_id": int(row["test_id"]),
            "fail_score": float(row["fail_score"]),
            "total_membership_v2": optional_float(row["total_membership_v2"]),
            "decision_score_v2": optional_float(row["decision_score_v2"]),
            "terminal_risk_score": optional_float(row["terminal_risk_score"]),
        }
        for row in rows[:3]
    ]

    audit = {
        "target_field": target_field,
        "sample_count": len(rows),
        "round_count": len(round_ids),
        "round_index_min": min(round_ids),
        "round_index_max": max(round_ids),
        "target_min": min(fail_scores),
        "target_max": max(fail_scores),
        "target_mean": mean(fail_scores),
        "target_unique_count": len({round(value, 12) for value in fail_scores}),
        "target_zero_count": sum(1 for value in fail_scores if abs(value) < 1e-12),
        "fail_score_min": min(fail_scores),
        "fail_score_max": max(fail_scores),
        "fail_score_mean": mean(fail_scores),
        "fail_score_unique_count": len({round(value, 12) for value in fail_scores}),
        "sample_preview": sample_preview,
    }
    return audit


def write_audit(audit: dict[str, object], audit_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-score training CSV directly from TEST_SUMMARY_JSON records."
    )
    parser.add_argument(
        "--rounds-root",
        default=(
            "failure_and_attribution_analysis/2.2module/"
            "zzr_failure_and_attribution_analysis/zzr_current_session/current_session/rounds"
        ),
        help="Directory containing round_*/evalu.txt files.",
    )
    parser.add_argument(
        "--output",
        default="failure_and_attribution_analysis/2.2module/attribution_analysis/fail_score_training_fused_score.csv",
        help="Path to the rebuilt training CSV.",
    )
    parser.add_argument(
        "--audit-output",
        default="failure_and_attribution_analysis/2.2module/attribution_analysis/fail_score_training_fused_score_audit.json",
        help="Path to the audit JSON.",
    )
    parser.add_argument(
        "--target-field",
        default=DEFAULT_TARGET_FIELD,
        choices=SUPPORTED_TARGET_FIELDS,
        help="TEST_SUMMARY_JSON field to copy into the standard fail_score column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds_root = Path(args.rounds_root)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)

    rows = parse_test_summary_lines(rounds_root, target_field=args.target_field)
    audit = build_audit(rows, target_field=args.target_field)
    if int(audit["target_unique_count"]) <= 1:
        raise ValueError(
            "Rebuilt fail_score has no usable variation; refusing to emit a constant training table."
        )

    write_csv(rows, output_path)
    write_audit(audit, audit_path)

    print(f"Built fail-score training CSV: {output_path}")
    print(f"Built audit JSON: {audit_path}")
    print(
        "Audit summary: "
        f"target_field={audit['target_field']}, "
        f"samples={audit['sample_count']}, "
        f"rounds={audit['round_count']}, "
        f"target_min={audit['target_min']:.6f}, "
        f"target_max={audit['target_max']:.6f}, "
        f"unique={audit['target_unique_count']}, "
        f"zeros={audit['target_zero_count']}"
    )


if __name__ == "__main__":
    main()
