from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .models import (
    FAIL_ENV_FIELD_NAMES,
    FailEnvironment,
    FailPoint,
    OutputScenarioSelection,
    infer_attack_type_from_fail_env,
)


ATTACK_SUMMARY_PATTERN = re.compile(
    r"^AttackSummary:\s*type=(?P<attack_type>[^,]+),\s*satellite=(?P<satellite>[^,]+),\s*count=(?P<count>\d+)\s*$"
)


@dataclass(frozen=True)
class OutputScenarioPaths:
    output_root_dir: Path
    output_constellation_index: int
    round_index: int
    test_id: int
    config_dir: Path
    summary_path: Path
    attacked_res_path: Optional[Path]
    model_dir: Path


class OutputScenarioReader:
    def __init__(self, output_root_dir: Optional[Union[str, Path]] = None):
        self.output_root_dir = (
            Path(output_root_dir)
            if output_root_dir is not None
            else Path(__file__).resolve().parent.parent / "output"
        )

    def resolve_paths(
        self,
        output_selection_input: Union[OutputScenarioSelection, Sequence[int], Dict[str, Any]],
    ) -> OutputScenarioPaths:
        selection = OutputScenarioSelection.from_input(output_selection_input)
        config_dir = self.output_root_dir / str(selection.output_constellation_index)
        summary_path = config_dir / f"{selection.output_constellation_index}_output_summary.txt"
        attacked_res_path = config_dir / "attacked_res.txt"
        model_dir = config_dir / f"{selection.round_index}_{selection.test_id}"

        if not config_dir.is_dir():
            raise FileNotFoundError(f"Output configuration directory not found: {config_dir}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Output summary file not found: {summary_path}")

        return OutputScenarioPaths(
            output_root_dir=self.output_root_dir,
            output_constellation_index=selection.output_constellation_index,
            round_index=selection.round_index,
            test_id=selection.test_id,
            config_dir=config_dir,
            summary_path=summary_path,
            attacked_res_path=attacked_res_path if attacked_res_path.exists() else None,
            model_dir=model_dir,
        )

    def _coerce_paths(
        self,
        output_selection_input: Union[
            OutputScenarioPaths,
            OutputScenarioSelection,
            Sequence[int],
            Dict[str, Any],
        ],
    ) -> OutputScenarioPaths:
        if isinstance(output_selection_input, OutputScenarioPaths):
            return output_selection_input
        return self.resolve_paths(output_selection_input)

    def load_summary_record(
        self,
        output_selection_input: Union[
            OutputScenarioPaths,
            OutputScenarioSelection,
            Sequence[int],
            Dict[str, Any],
        ],
    ) -> Dict[str, Any]:
        paths = self._coerce_paths(output_selection_input)
        best_record: Optional[Dict[str, Any]] = None
        with paths.summary_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line.startswith("{"):
                    continue
                record = json.loads(line)
                if (
                    int(record.get("round_index", -1)) != paths.round_index
                    or int(record.get("test_id", -1)) != paths.test_id
                ):
                    continue
                if best_record is None or int(record.get("step_index", -1)) > int(
                    best_record.get("step_index", -1)
                ):
                    best_record = record

        if best_record is None:
            raise LookupError(
                "No matching summary record found for "
                f"output/{paths.output_constellation_index}, "
                f"round_index={paths.round_index}, test_id={paths.test_id}."
            )
        return best_record

    def load_attacked_satellites(
        self,
        output_selection_input: Union[
            OutputScenarioPaths,
            OutputScenarioSelection,
            Sequence[int],
            Dict[str, Any],
        ],
    ) -> List[str]:
        paths = self._coerce_paths(output_selection_input)
        if paths.attacked_res_path is None:
            raise FileNotFoundError(
                "attacked_res.txt was not found under "
                f"{paths.config_dir}. This scenario cannot build FailSat entries."
            )

        blocks = self._load_attacked_blocks(paths.attacked_res_path)
        header_key = (paths.round_index, paths.test_id)
        matched_lines = blocks.get(header_key)
        if matched_lines is None:
            raise LookupError(
                "No attacked_res block found for "
                f"round_index={paths.round_index}, test_id={paths.test_id} "
                f"in {paths.attacked_res_path}."
            )

        satellites: List[str] = []
        for line in matched_lines:
            match = ATTACK_SUMMARY_PATTERN.match(line)
            if match is None:
                continue
            satellites.append(match.group("satellite").strip())
        return satellites

    def build_fail_point(
        self,
        output_selection_input: Union[
            OutputScenarioPaths,
            OutputScenarioSelection,
            Sequence[int],
            Dict[str, Any],
        ],
    ) -> FailPoint:
        paths = self._coerce_paths(output_selection_input)
        summary_record = self.load_summary_record(paths)
        fail_env_raw = {field_name: summary_record[field_name] for field_name in FAIL_ENV_FIELD_NAMES}
        fail_env = FailEnvironment.from_input(fail_env_raw)
        fail_score = float(summary_record["failure_score_v2"])
        attack_type, _, _ = infer_attack_type_from_fail_env(fail_env)
        failed_satellite_ids = self.load_attacked_satellites(paths)
        fail_sat = [[sid, fail_score, attack_type] for sid in failed_satellite_ids]
        return FailPoint.from_input(
            {
                "ScenarioId": (
                    f"output-{paths.output_constellation_index}-"
                    f"round{paths.round_index}-test{paths.test_id}"
                ),
                "FailEnv": fail_env_raw,
                "FailSat": fail_sat,
            }
        )

    @staticmethod
    def _load_attacked_blocks(attacked_res_path: Path) -> Dict[Tuple[int, int], List[str]]:
        blocks: Dict[Tuple[int, int], List[str]] = {}
        current_key: Optional[Tuple[int, int]] = None
        current_lines: List[str] = []

        with attacked_res_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    if current_key is not None:
                        blocks[current_key] = list(current_lines)
                    header = json.loads(line)
                    current_key = (int(header["round_index"]), int(header["test_id"]))
                    current_lines = []
                else:
                    current_lines.append(line)

        if current_key is not None:
            blocks[current_key] = list(current_lines)
        return blocks
