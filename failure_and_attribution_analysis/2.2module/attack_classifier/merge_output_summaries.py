from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from extract_output_csv import build_output_csv, iter_summary_records


def expand_input_paths(inputs: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for item in inputs:
        path = Path(item)
        if any(ch in item for ch in "*?[]"):
            matches = sorted(Path().glob(item))
        else:
            matches = [path]

        for match in matches:
            candidate = match.resolve()
            if candidate in seen:
                continue
            if not candidate.exists():
                continue
            if candidate.is_dir():
                for child in sorted(candidate.glob("*.txt")):
                    child_path = child.resolve()
                    if child_path not in seen:
                        resolved.append(child_path)
                        seen.add(child_path)
            else:
                resolved.append(candidate)
                seen.add(candidate)

    return resolved


def _normalize_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def merge_summary_files(
    input_paths: list[Path],
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[int, int]:
    output_path = Path(output_path)
    merged_records: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    round_offset = 0
    base_dir = Path.cwd().resolve()

    for file_index, input_path in enumerate(input_paths):
        records = list(iter_summary_records(input_path))
        if not records:
            continue
        source_output_summary_path = input_path.resolve()
        source_session_dir = source_output_summary_path.parent.resolve()
        source_output_summary_rel = _normalize_relative_path(source_output_summary_path, base_dir)
        source_session_dir_rel = _normalize_relative_path(source_session_dir, base_dir)
        manifest_entries.append(
            {
                "source_file_index": file_index,
                "source_output_summary_path": source_output_summary_rel,
                "source_session_dir": source_session_dir_rel,
                "source_session_id": f"session_{file_index:04d}",
            }
        )

        max_round_index = max(int(record["round_index"]) for record in records)
        for record in records:
            merged_record = dict(record)
            merged_record["source_file"] = input_path.name
            merged_record["source_file_index"] = file_index
            merged_record["source_output_summary_path"] = source_output_summary_rel
            merged_record["source_session_dir"] = source_session_dir_rel
            merged_record["source_session_id"] = f"session_{file_index:04d}"
            merged_record["original_round_index"] = int(record["round_index"])
            merged_record["original_test_id"] = int(record["test_id"])
            merged_record["round_index"] = int(record["round_index"]) + round_offset
            merged_records.append(merged_record)

        round_offset += max_round_index + 1

    merged_records.sort(
        key=lambda row: (
            int(row["round_index"]),
            int(row["test_id"]),
            int(row.get("step_index", 0)),
        )
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"merged_source_file_count: {len(input_paths)}\n")
        output_file.write(f"merged_record_count: {len(merged_records)}\n")
        output_file.write("round_index_reindexed: true\n")
        output_file.write(
            "source_files: "
            + json.dumps(
                [_normalize_relative_path(path.resolve(), base_dir) for path in input_paths],
                ensure_ascii=False,
            )
            + "\n\n"
        )
        for record in merged_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(
                {
                    "merged_output_summary": _normalize_relative_path(output_path.resolve(), base_dir),
                    "sources": manifest_entries,
                },
                manifest_file,
                ensure_ascii=False,
                indent=2,
            )

    return len(input_paths), len(merged_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple output summary txt files into one larger dataset."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input txt paths, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output",
        default="merged_output_summary.txt",
        help="Path to the merged txt file.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional path to also generate a merged csv file.",
    )
    parser.add_argument(
        "--sources-json",
        default="",
        help="Optional path to write source manifest JSON. Defaults to <output_stem>_sources.json next to the merged summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = expand_input_paths(args.inputs)
    if not input_paths:
        raise FileNotFoundError("No valid input txt files were found.")

    manifest_path = args.sources_json
    if not manifest_path:
        output_path = Path(args.output)
        manifest_path = output_path.with_name(f"{output_path.stem}_sources.json")

    file_count, record_count = merge_summary_files(
        input_paths,
        args.output,
        manifest_path=manifest_path,
    )
    print(f"Merged {file_count} files into {args.output} with {record_count} JSON records.")
    print(f"Wrote source manifest to {manifest_path}")

    if args.csv:
        grouped_count = build_output_csv(args.output, args.csv)
        print(f"Generated {args.csv} with {grouped_count} grouped samples.")


if __name__ == "__main__":
    main()
