from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from attack_type_classifier import ENV_FEATURES_16, METRIC_FEATURES


OUTPUT_COLUMNS = [
    "round_index",
    "test_id",
    "source_file_index",
    "source_session_dir",
    "source_session_id",
    "original_round_index",
    "original_test_id",
    "selected_step_count",
    "selected_step_indices",
] + ENV_FEATURES_16 + METRIC_FEATURES


def iter_summary_records(summary_path: str | Path):
    started = False
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as summary_file:
        for raw_line in summary_file:
            line = raw_line.strip()
            if not line:
                continue
            if not started:
                if not line.startswith("{"):
                    continue
                started = True
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate_last_two_steps(summary_path: str | Path) -> list[dict[str, object]]:
    grouped_records: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in iter_summary_records(summary_path):
        round_index = int(record["round_index"])
        test_id = int(record["test_id"])
        grouped_records[(round_index, test_id)].append(record)

    aggregated_rows: list[dict[str, object]] = []
    provenance_fields = [
        "source_file_index",
        "source_session_dir",
        "source_session_id",
        "original_round_index",
        "original_test_id",
    ]
    for (round_index, test_id), records in sorted(grouped_records.items()):
        ordered = sorted(records, key=lambda row: int(row["step_index"]))
        selected = ordered[-2:] if len(ordered) >= 2 else ordered
        provenance = {field: ordered[0].get(field) for field in provenance_fields}
        for row in ordered[1:]:
            for field in provenance_fields:
                if row.get(field) != provenance[field]:
                    raise ValueError(
                        f"Inconsistent provenance field '{field}' for grouped sample "
                        f"(round_index={round_index}, test_id={test_id})."
                    )
        aggregated: dict[str, object] = {
            "round_index": round_index,
            "test_id": test_id,
            "source_file_index": int(provenance["source_file_index"]) if provenance["source_file_index"] is not None else "",
            "source_session_dir": provenance["source_session_dir"] or "",
            "source_session_id": provenance["source_session_id"] or "",
            "original_round_index": int(provenance["original_round_index"]) if provenance["original_round_index"] is not None else "",
            "original_test_id": int(provenance["original_test_id"]) if provenance["original_test_id"] is not None else "",
            "selected_step_count": len(selected),
            "selected_step_indices": ",".join(str(int(row["step_index"])) for row in selected),
        }

        for feature_name in ENV_FEATURES_16 + METRIC_FEATURES:
            aggregated[feature_name] = fmean(float(row[feature_name]) for row in selected)

        aggregated_rows.append(aggregated)
    return aggregated_rows


def write_output_csv(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_output_csv(summary_path: str | Path, csv_path: str | Path) -> int:
    rows = aggregate_last_two_steps(summary_path)
    write_output_csv(rows, csv_path)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert output_summary.txt to averaged output.csv.")
    parser.add_argument("--summary", default="output_summary.txt", help="Path to output_summary.txt.")
    parser.add_argument("--csv", default="output.csv", help="Path to the generated CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = build_output_csv(args.summary, args.csv)
    print(f"Generated {args.csv} with {row_count} grouped samples.")


if __name__ == "__main__":
    main()
