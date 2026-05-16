from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Windows/Conda stacks can load both Torch and MKL-backed NumPy runtimes in the
# same process. Set this before importing those libraries so the embedded stage
# can run inside the main project environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import plotly.graph_objects as go
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART3_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online_self_healing import (  # noqa: E402
    BlockchainConsensusStateMachine,
    ConstellationCatalog,
    ConstellationFramework,
    ConsensusScenarioReport,
    NodeElement,
)
from online_self_healing.consensus_engine.models import FAIL_ENV_FIELD_NAMES  # noqa: E402
from online_self_healing.self_healing_module.healing_orchestrator import (  # noqa: E402
    FAILSCORE_HIGH_LOWER,
    FAILSCORE_LOW_UPPER,
    LINK_RATIO_THRESHOLD,
    TieredHealingOrchestrator,
)


DEFAULT_FULL_RUN_ROOT = PROJECT_ROOT / "failure_and_attribution_analysis" / "full_project_runs"
DEFAULT_AGENT_CONFIG = PART3_ROOT / "train_NewDDQN_dueling_shuffle.yaml"
DEFAULT_OUTPUT_DIR_NAME = "part3_rebuild"
ATTACK_SUMMARY_RE = re.compile(
    r"^AttackSummary:\s*type=(?P<attack_type>[^,]+),\s*satellite=(?P<agent>[^,]+),\s*count=(?P<count>\d+)\s*$"
)
ATTACK_TYPE_TO_ID = {
    "NoAttack": 0,
    "StateObservationAttack": 1,
    "ActionAttack": 2,
    "StateTransferAttack": 3,
    "RewardAttack": 4,
    "ExperiencePoolAttack": 5,
    "ModelTampAttack": 6,
}

HEALING_CLASS_ORDER = (
    "level1_redundant",
    "level2_immune",
    "level3_batch_redundant",
    "level4_federated",
)
HEALING_CLASS_INFO: dict[str, dict[str, Any]] = {
    "level1_redundant": {
        "healing_level": 1,
        "risk_class": "low",
        "label": "Level 1 单点冗余自愈",
        "selection_rule": f"fail_score < {FAILSCORE_LOW_UPPER:.2f}",
    },
    "level2_immune": {
        "healing_level": 2,
        "risk_class": "medium",
        "label": "Level 2 免疫自愈",
        "selection_rule": (
            f"{FAILSCORE_LOW_UPPER:.2f} <= fail_score < {FAILSCORE_HIGH_LOWER:.2f} "
            f"且局部建链失效比例 <= {LINK_RATIO_THRESHOLD:.2f}"
        ),
    },
    "level3_batch_redundant": {
        "healing_level": 3,
        "risk_class": "batch_medium",
        "label": "Level 3 批量冗余自愈",
        "selection_rule": (
            f"{FAILSCORE_LOW_UPPER:.2f} <= fail_score < {FAILSCORE_HIGH_LOWER:.2f} "
            f"且至少一个目标节点局部建链失效比例 > {LINK_RATIO_THRESHOLD:.2f}"
        ),
    },
    "level4_federated": {
        "healing_level": 4,
        "risk_class": "high",
        "label": "Level 4 联邦学习自愈",
        "selection_rule": f"fail_score >= {FAILSCORE_HIGH_LOWER:.2f}",
    },
}


@dataclass(frozen=True)
class AttackObservation:
    agent_id: str
    attack_type_name: str
    attack_type_id: int
    count: int


@dataclass
class SelectedScenario:
    run_root: Path
    analysis_output_root: Path
    source_session_dir: Path
    performance_file: Path
    scenario_id: str
    round_index: int
    test_id: int
    constellation_id: int
    scenario: dict[str, Any]
    summary_record: dict[str, Any]
    integrated_row: dict[str, Any] | None
    fail_score: float
    predicted_attack_type: str | None
    attack_observations: list[AttackObservation]


@dataclass
class ResolvedTargets:
    branch: str
    selected_satellites: list[str]
    observations_by_satellite: dict[str, dict[str, Any]]
    original_failed_agents: list[str]
    constellation2_agent_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScenarioExecutionPlan:
    healing_class: str
    selected: SelectedScenario
    targets: ResolvedTargets
    consensus_report: ConsensusScenarioReport
    selection_metrics: dict[str, Any]

    @property
    def healing_level(self) -> int:
        return int(HEALING_CLASS_INFO[self.healing_class]["healing_level"])

    @property
    def healing_label(self) -> str:
        return str(HEALING_CLASS_INFO[self.healing_class]["label"])

    @property
    def risk_class(self) -> str:
        return str(HEALING_CLASS_INFO[self.healing_class]["risk_class"])


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {path}:{line_no}") from exc
            if isinstance(payload, dict):
                yield payload


def resolve_path(path_text: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def path_for_report(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def latest_run_root(full_run_root: Path) -> Path:
    if not full_run_root.exists():
        raise FileNotFoundError(f"Full-project run root does not exist: {full_run_root}")
    candidates = [path for path in full_run_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No full-project run directories found under: {full_run_root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_pipeline_summary(analysis_output_root: Path) -> dict[str, Any]:
    summary_path = analysis_output_root / "pipeline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"2.2 pipeline summary not found: {summary_path}")
    return read_json(summary_path)


def resolve_integrated_jsonl(analysis_output_root: Path, pipeline_summary: Mapping[str, Any]) -> Path | None:
    integrated = pipeline_summary.get("integrated_results")
    if isinstance(integrated, Mapping) and integrated.get("output_jsonl"):
        candidate = resolve_path(str(integrated["output_jsonl"]))
        if candidate.exists():
            return candidate
    fallback = analysis_output_root / "integrated_attack_attribution_results.jsonl"
    return fallback if fallback.exists() else None


def resolve_archived_session_dir(raw_session_dir: Path, pipeline_summary: Mapping[str, Any]) -> Path:
    if raw_session_dir.exists():
        return raw_session_dir

    archive_summary = pipeline_summary.get("archive_summary")
    if not isinstance(archive_summary, Mapping):
        return raw_session_dir

    for entry in archive_summary.get("archived_dirs", []) or []:
        if not isinstance(entry, Mapping):
            continue
        source_dir = Path(str(entry.get("source_dir", ""))).resolve()
        target_dir = Path(str(entry.get("target_dir", ""))).resolve()
        if raw_session_dir == source_dir / "current_session" and (target_dir / "current_session").exists():
            return target_dir / "current_session"
        if raw_session_dir == source_dir and target_dir.exists():
            return target_dir
    return raw_session_dir


def parse_test_summary_records(evalu_path: Path) -> Iterable[dict[str, Any]]:
    with evalu_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("TEST_SUMMARY_JSON:"):
                continue
            payload = json.loads(line.split(":", 1)[1].strip())
            if isinstance(payload, dict):
                yield payload


def load_summary_record(session_dir: Path, round_index: int, test_id: int) -> dict[str, Any]:
    evalu_path = session_dir / "rounds" / f"round_{round_index:03d}" / "evalu.txt"
    if evalu_path.exists():
        for record in parse_test_summary_records(evalu_path):
            if int(record.get("round_index", -1)) == round_index and int(record.get("test_id", -1)) == test_id:
                return record

    failure_scores_path = session_dir / "rounds" / f"round_{round_index:03d}" / "failure_scores.jsonl"
    if failure_scores_path.exists():
        for record in iter_jsonl(failure_scores_path):
            if int(record.get("round_index", -1)) == round_index and int(record.get("test_id", -1)) == test_id:
                return record

    raise LookupError(
        f"No TEST_SUMMARY_JSON/failure_scores row for round={round_index}, test={test_id} under {session_dir}"
    )


def resolve_performance_file(summary_record: Mapping[str, Any], session_dir: Path, round_index: int, test_id: int) -> Path:
    raw_path = summary_record.get("performance_file")
    if raw_path:
        candidate = Path(str(raw_path))
        if candidate.exists():
            return candidate

    fallback = session_dir / "rounds" / f"round_{round_index:03d}" / "performance" / f"test_{test_id:04d}_performance.txt"
    if fallback.exists():
        return fallback

    if raw_path:
        return Path(str(raw_path))
    return fallback


def parse_attack_observations(performance_file: Path) -> list[AttackObservation]:
    if not performance_file.exists():
        raise FileNotFoundError(f"Performance file not found: {performance_file}")

    grouped: dict[tuple[str, str], int] = {}
    with performance_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = ATTACK_SUMMARY_RE.match(raw_line.strip())
            if match is None:
                continue
            attack_type_name = match.group("attack_type").strip()
            agent_id = match.group("agent").strip()
            count = int(match.group("count"))
            grouped[(agent_id, attack_type_name)] = grouped.get((agent_id, attack_type_name), 0) + count

    observations: list[AttackObservation] = []
    for (agent_id, attack_type_name), count in sorted(grouped.items()):
        observations.append(
            AttackObservation(
                agent_id=agent_id,
                attack_type_name=attack_type_name,
                attack_type_id=ATTACK_TYPE_TO_ID.get(attack_type_name, -1),
                count=count,
            )
        )
    return observations


def score_from_row(
    integrated_row: Mapping[str, Any] | None,
    summary_record: Mapping[str, Any],
    target_field: str,
) -> float:
    if integrated_row is not None:
        attribution = integrated_row.get("attribution_analysis")
        if isinstance(attribution, Mapping) and attribution.get("target_value") is not None:
            return float(attribution["target_value"])
    if summary_record.get(target_field) is not None:
        return float(summary_record[target_field])
    for fallback_key in ("fused_score", "decision_score_v2", "total_membership_v2", "total_membership"):
        if summary_record.get(fallback_key) is not None:
            return float(summary_record[fallback_key])
    return 0.0


def predicted_attack_from_row(integrated_row: Mapping[str, Any] | None) -> str | None:
    if integrated_row is None:
        return None
    attack = integrated_row.get("attack_classifier")
    if isinstance(attack, Mapping) and attack.get("predicted_attack_type"):
        return str(attack["predicted_attack_type"])
    return None


def scenario_from_summary(summary_record: Mapping[str, Any]) -> dict[str, Any]:
    scenario = summary_record.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError("Selected TEST_SUMMARY_JSON row is missing a scenario object.")
    missing = [field for field in FAIL_ENV_FIELD_NAMES if field not in scenario]
    if missing:
        raise KeyError(f"Selected scenario is missing required fields: {missing}")
    return {field: scenario[field] for field in FAIL_ENV_FIELD_NAMES}


def load_integrated_candidates(
    run_root: Path,
    analysis_output_root: Path,
    pipeline_summary: Mapping[str, Any],
    target_field: str,
) -> list[SelectedScenario]:
    integrated_path = resolve_integrated_jsonl(analysis_output_root, pipeline_summary)
    candidates: list[SelectedScenario] = []
    if integrated_path is None:
        return candidates

    for row in iter_jsonl(integrated_path):
        raw_session_dir = Path(str(row.get("source_session_dir", ""))).resolve()
        session_dir = resolve_archived_session_dir(raw_session_dir, pipeline_summary)
        round_index = int(row.get("original_round_index", row.get("round_index", -1)))
        test_id = int(row.get("original_test_id", row.get("test_id", -1)))
        if round_index < 0 or test_id < 0 or not session_dir.exists():
            continue
        try:
            summary_record = load_summary_record(session_dir, round_index, test_id)
            performance_file = resolve_performance_file(summary_record, session_dir, round_index, test_id)
            observations = parse_attack_observations(performance_file)
            scenario = scenario_from_summary(summary_record)
        except (FileNotFoundError, LookupError, KeyError, ValueError):
            continue

        candidates.append(
            SelectedScenario(
                run_root=run_root,
                analysis_output_root=analysis_output_root,
                source_session_dir=session_dir,
                performance_file=performance_file,
                scenario_id=f"full-project-round{round_index}-test{test_id}",
                round_index=round_index,
                test_id=test_id,
                constellation_id=int(scenario["ConstellationConfig"]),
                scenario=scenario,
                summary_record=dict(summary_record),
                integrated_row=dict(row),
                fail_score=score_from_row(row, summary_record, target_field),
                predicted_attack_type=predicted_attack_from_row(row),
                attack_observations=observations,
            )
        )
    return candidates


def resolve_targets_for_selected(
    selected: SelectedScenario,
    catalog: ConstellationCatalog,
    args: argparse.Namespace,
) -> ResolvedTargets:
    if selected.constellation_id == 2:
        return resolve_targets_for_constellation2(
            selected,
            catalog,
            orbit_block_size=args.region_orbit_block_size,
            sat_block_size=args.region_sat_block_size,
            default_count=args.constellation2_default_satellites_per_agent,
            max_count=args.constellation2_max_satellites_per_agent,
        )
    return resolve_targets_for_standard_constellation(selected, catalog)


def local_link_ratios_for_targets(
    targets: ResolvedTargets,
    catalog: ConstellationCatalog,
) -> dict[str, float]:
    failed_set = set(targets.selected_satellites)
    neighbors_status = [{"id": satellite_id} for satellite_id in catalog.satellite_names]
    return {
        satellite_id: float(
            TieredHealingOrchestrator.compute_link_ratio(
                satellite_id,
                failed_set,
                neighbors_status,
            )
        )
        for satellite_id in targets.selected_satellites
    }


def healing_class_for_candidate(
    selected: SelectedScenario,
    targets: ResolvedTargets,
    catalog: ConstellationCatalog,
) -> tuple[str, dict[str, Any]]:
    fail_score = float(selected.fail_score)
    local_ratios = local_link_ratios_for_targets(targets, catalog)
    max_local_ratio = max(local_ratios.values(), default=0.0)
    if fail_score < FAILSCORE_LOW_UPPER:
        healing_class = "level1_redundant"
    elif fail_score < FAILSCORE_HIGH_LOWER:
        if max_local_ratio > LINK_RATIO_THRESHOLD:
            healing_class = "level3_batch_redundant"
        else:
            healing_class = "level2_immune"
    else:
        healing_class = "level4_federated"

    metrics = {
        "fail_score": fail_score,
        "max_local_link_ratio": float(max_local_ratio),
        "local_link_ratio_by_satellite": local_ratios,
        "selected_satellite_count": len(targets.selected_satellites),
        "selection_rule": HEALING_CLASS_INFO[healing_class]["selection_rule"],
    }
    return healing_class, metrics


def scenario_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "selection_class": report.get("selection_class"),
        "selected_scenario": report.get("selected_scenario"),
        "constellation_branch": report.get("constellation_branch"),
        "resolved_target_count": (
            report.get("resolved_targets", {}).get("selected_satellite_count")
            if isinstance(report.get("resolved_targets"), Mapping)
            else None
        ),
        "consensus": report.get("consensus"),
        "healing": report.get("healing"),
        "outputs": report.get("outputs"),
        "warnings": report.get("warnings", []),
    }


def select_scenarios_by_healing_class(
    candidates: Sequence[SelectedScenario],
    args: argparse.Namespace,
) -> tuple[list[ScenarioExecutionPlan], dict[str, Any], list[str]]:
    selected_by_class: dict[str, ScenarioExecutionPlan] = {}
    available_counts = {key: 0 for key in HEALING_CLASS_ORDER}
    skipped_counts = {
        "without_attack_observations": 0,
        "unresolvable_failed_agents": 0,
        "candidate_errors": 0,
    }
    warnings: list[str] = []
    catalog_cache: dict[int, ConstellationCatalog] = {}

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (int(item.round_index), int(item.test_id), str(item.scenario_id)),
    )
    for candidate in ordered_candidates:
        if not candidate.attack_observations:
            skipped_counts["without_attack_observations"] += 1
            continue

        try:
            catalog = catalog_cache.get(candidate.constellation_id)
            if catalog is None:
                catalog = ConstellationCatalog.from_constellation_config(candidate.constellation_id)
                catalog_cache[candidate.constellation_id] = catalog
            targets = resolve_targets_for_selected(candidate, catalog, args)
            if not targets.selected_satellites:
                skipped_counts["unresolvable_failed_agents"] += 1
                warnings.extend(targets.warnings)
                continue
            consensus_report = build_consensus_report(candidate, targets)
            healing_class, selection_metrics = healing_class_for_candidate(candidate, targets, catalog)
            selection_metrics["consensus_linked_failure_ratio"] = float(consensus_report.linked_failure_ratio)
            selection_metrics["target_warnings"] = list(targets.warnings)
        except Exception as exc:
            skipped_counts["candidate_errors"] += 1
            warnings.append(
                f"Skip {candidate.scenario_id}: failed to prepare scenario for self-healing selection: {exc}"
            )
            continue

        available_counts[healing_class] += 1
        if healing_class not in selected_by_class:
            selected_by_class[healing_class] = ScenarioExecutionPlan(
                healing_class=healing_class,
                selected=candidate,
                targets=targets,
                consensus_report=consensus_report,
                selection_metrics=selection_metrics,
            )

    skipped_classes: list[dict[str, Any]] = []
    for healing_class in HEALING_CLASS_ORDER:
        if healing_class in selected_by_class:
            continue
        info = HEALING_CLASS_INFO[healing_class]
        message = (
            f"跳过 {info['label']}: 未找到满足条件的候选场景 "
            f"({info['selection_rule']})。"
        )
        print(f"[Part3] {message}", flush=True)
        skipped_classes.append(
            {
                "healing_class": healing_class,
                "healing_level": info["healing_level"],
                "label": info["label"],
                "selection_rule": info["selection_rule"],
                "reason": "no_candidate_for_healing_class",
            }
        )

    plans = [selected_by_class[key] for key in HEALING_CLASS_ORDER if key in selected_by_class]
    for plan in plans:
        print(
            "[Part3] 选中 {label}: {scenario_id} "
            "(round={round_index}, test={test_id}, fail_score={fail_score:.6f}, "
            "targets={target_count}, max_link_ratio={max_link_ratio:.2%})".format(
                label=plan.healing_label,
                scenario_id=plan.selected.scenario_id,
                round_index=plan.selected.round_index,
                test_id=plan.selected.test_id,
                fail_score=float(plan.selected.fail_score),
                target_count=len(plan.targets.selected_satellites),
                max_link_ratio=float(plan.selection_metrics.get("max_local_link_ratio", 0.0)),
            ),
            flush=True,
        )

    summary = {
        "mode": "per_healing_level",
        "candidate_count": len(candidates),
        "candidate_count_with_attack_observations": len(candidates)
        - skipped_counts["without_attack_observations"],
        "available_counts_by_healing_class": available_counts,
        "selected_classes": [
            {
                "healing_class": plan.healing_class,
                "healing_level": plan.healing_level,
                "label": plan.healing_label,
                "risk_class": plan.risk_class,
                "scenario_id": plan.selected.scenario_id,
                "round_index": plan.selected.round_index,
                "test_id": plan.selected.test_id,
                "fail_score": float(plan.selected.fail_score),
                "selection_metrics": plan.selection_metrics,
            }
            for plan in plans
        ],
        "skipped_classes": skipped_classes,
        "skipped_candidate_counts": skipped_counts,
    }
    return plans, summary, warnings


def parse_satellite_name(name: str) -> tuple[int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 4 or parts[0] != "Satellite":
        return None
    try:
        return int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def region_name_for_satellite(name: str, orbit_block_size: int, sat_block_size: int) -> str | None:
    parsed = parse_satellite_name(name)
    if parsed is None:
        return None
    altitude, orbit_number, sat_number = parsed
    orbit_region = (orbit_number - 1) // orbit_block_size
    sat_region = (sat_number - 1) // sat_block_size
    return f"Region_{altitude}_{orbit_region}_{sat_region}"


def build_constellation2_region_members(
    satellite_names: Iterable[str],
    orbit_block_size: int,
    sat_block_size: int,
) -> dict[str, list[str]]:
    members: dict[str, list[str]] = {}
    for name in satellite_names:
        region_name = region_name_for_satellite(name, orbit_block_size, sat_block_size)
        if region_name is None:
            continue
        members.setdefault(region_name, []).append(name)
    for names in members.values():
        names.sort()
    return dict(sorted(members.items()))


def stable_subset(values: Sequence[str], count: int, salt: str) -> list[str]:
    ordered = sorted(values, key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest())
    return ordered[: max(0, min(int(count), len(ordered)))]


def resolve_targets_for_constellation2(
    selected: SelectedScenario,
    catalog: ConstellationCatalog,
    orbit_block_size: int,
    sat_block_size: int,
    default_count: int,
    max_count: int,
) -> ResolvedTargets:
    warnings: list[str] = []
    region_members = build_constellation2_region_members(
        catalog.satellite_names,
        orbit_block_size=orbit_block_size,
        sat_block_size=sat_block_size,
    )
    if not region_members:
        raise ValueError("Constellation 2 region mapping is empty; cannot map failed agents to satellites.")

    observed_by_agent: dict[str, list[AttackObservation]] = {}
    for observation in selected.attack_observations:
        agent_id = observation.agent_id
        if agent_id.startswith("Satellite_"):
            mapped_region = region_name_for_satellite(agent_id, orbit_block_size, sat_block_size)
            if mapped_region is None:
                warnings.append(f"Cannot map satellite observation to a region: {agent_id}")
                continue
            agent_id = mapped_region
        elif not agent_id.startswith("Region_"):
            warnings.append(f"Unsupported ConstellationConfig=2 failed-agent id: {agent_id}")
            continue
        observed_by_agent.setdefault(agent_id, []).append(observation)

    selected_satellites: list[str] = []
    observations_by_satellite: dict[str, dict[str, Any]] = {}
    mapping_report: dict[str, dict[str, Any]] = {}
    requested_default = max(1, int(default_count))
    requested_max = max(1, min(10, int(max_count)))

    for agent_id, observations in sorted(observed_by_agent.items()):
        members = region_members.get(agent_id)
        if not members:
            warnings.append(f"Region agent {agent_id} is not present in the constellation-2 TLE mapping.")
            continue

        direct_satellite_hits = [
            observation.agent_id
            for observation in observations
            if observation.agent_id.startswith("Satellite_") and observation.agent_id in members
        ]
        unique_direct_hits = sorted(set(direct_satellite_hits))
        if unique_direct_hits:
            selected_for_agent = unique_direct_hits[:requested_max]
        else:
            selected_for_agent = stable_subset(
                members,
                count=min(requested_default, requested_max),
                salt=f"{selected.scenario_id}|{agent_id}",
            )

        primary_attack = max(observations, key=lambda item: (item.count, -item.attack_type_id))
        mapping_report[agent_id] = {
            "member_count": len(members),
            "selected_satellite_count": len(selected_for_agent),
            "selected_satellites": selected_for_agent,
            "source_observations": [asdict(item) for item in observations],
            "selection_policy": (
                "observed_satellites_within_region"
                if unique_direct_hits
                else "deterministic_subset_from_region_members"
            ),
        }

        for satellite_id in selected_for_agent:
            if satellite_id not in selected_satellites:
                selected_satellites.append(satellite_id)
            current = observations_by_satellite.get(satellite_id)
            payload = {
                "failed_agent_id": agent_id,
                "raw_observations": [asdict(item) for item in observations],
                "fail_score": float(selected.fail_score),
                "attack_type_id": int(primary_attack.attack_type_id),
                "attack_type_name": primary_attack.attack_type_name,
            }
            if current is None or primary_attack.count > int(current.get("attack_count", -1)):
                payload["attack_count"] = int(primary_attack.count)
                observations_by_satellite[satellite_id] = payload

    selected_satellites.sort()
    return ResolvedTargets(
        branch="constellation_2_region_agent",
        selected_satellites=selected_satellites,
        observations_by_satellite=observations_by_satellite,
        original_failed_agents=sorted(observed_by_agent),
        constellation2_agent_mapping=mapping_report,
        warnings=warnings,
    )


def resolve_targets_for_standard_constellation(
    selected: SelectedScenario,
    catalog: ConstellationCatalog,
) -> ResolvedTargets:
    warnings: list[str] = []
    selected_satellites: list[str] = []
    observations_by_satellite: dict[str, dict[str, Any]] = {}

    for observation in selected.attack_observations:
        try:
            satellite_id = catalog.resolve_satellite_id(observation.agent_id)
        except Exception as exc:
            warnings.append(f"Skipping failed agent {observation.agent_id}: {exc}")
            continue
        selected_satellites.append(satellite_id)
        current = observations_by_satellite.get(satellite_id)
        payload = {
            "failed_agent_id": observation.agent_id,
            "raw_observations": [asdict(observation)],
            "fail_score": float(selected.fail_score),
            "attack_type_id": int(observation.attack_type_id),
            "attack_type_name": observation.attack_type_name,
            "attack_count": int(observation.count),
        }
        if current is None or observation.count > int(current.get("attack_count", -1)):
            observations_by_satellite[satellite_id] = payload

    return ResolvedTargets(
        branch="satellite_agent",
        selected_satellites=sorted(set(selected_satellites)),
        observations_by_satellite=observations_by_satellite,
        original_failed_agents=sorted({item.agent_id for item in selected.attack_observations}),
        warnings=warnings,
    )


def build_consensus_report(selected: SelectedScenario, targets: ResolvedTargets) -> ConsensusScenarioReport:
    fail_sat = []
    for satellite_id in targets.selected_satellites:
        observation = targets.observations_by_satellite[satellite_id]
        fail_sat.append(
            [
                satellite_id,
                float(observation["fail_score"]),
                int(observation["attack_type_id"]),
            ]
        )
    if not fail_sat:
        raise ValueError("No satellite targets were resolved for consensus.")

    engine = BlockchainConsensusStateMachine()
    return engine.process_fail_point(
        {
            "ScenarioId": selected.scenario_id,
            "FailEnv": selected.scenario,
            "FailSat": fail_sat,
        }
    )


def graph_stats(output: Any | None) -> dict[str, Any] | None:
    if output is None:
        return None
    graph = output.RawGraph
    return {
        "step_index": int(output.StepIndex),
        "current_time": output.CurrentTime.isoformat(),
        "trigger_reason": output.TriggerReason,
        "isolation_flag": bool(output.IsolationFlag),
        "state_changed": bool(output.StateChanged),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "isolated_nodes": list(graph.graph.get("isolated_nodes", [])),
        "unknown_isolation_targets": list(graph.graph.get("unknown_isolation_targets", [])),
        "removed_edge_count": len(graph.graph.get("removed_edges_by_isolation", [])),
        "triggered_event_ids": [event.EventId for event in output.TriggeredEvents],
        "triggered_heal_flags": [bool(event.HealFlag) for event in output.TriggeredEvents],
    }


def prepare_failed_nodes_for_healing(failed_nodes: Mapping[str, Any], seed_material: str) -> dict[str, Any]:
    if not failed_nodes:
        return {}
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    reference = next(iter(failed_nodes.values()))
    baseline_network = reference.build_q_network(device="cpu")
    baseline_state = copy.deepcopy(baseline_network.state_dict())
    prepared = {}
    for sid, node in failed_nodes.items():
        node.load_q_network_state_dict(baseline_state, device="cpu", is_initial=True)
        prepared[sid] = node
    return prepared


def euclidean_distance(pos_a: Sequence[float], pos_b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pos_a, pos_b)))


def build_neighbors_status(graph: Any, failed_set: set[str], baseline_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    failed_positions = [
        graph.nodes[sid].get("pos", [0.0, 0.0, 0.0])
        for sid in failed_set
        if sid in graph.nodes
    ]
    neighbors_status: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes):
        pos = graph.nodes[node_id].get("pos", [0.0, 0.0, 0.0])
        if failed_positions:
            distance = min(euclidean_distance(pos, failed_pos) for failed_pos in failed_positions)
        else:
            distance = 0.0
        neighbors_status.append(
            {
                "id": node_id,
                "health": 0.1 if node_id in failed_set else 1.0,
                "distance": float(distance),
                "snapshot_available": node_id not in failed_set,
                "state_dict": baseline_state if node_id not in failed_set else None,
                "replay_buffer": [] if node_id not in failed_set else None,
            }
        )
    return neighbors_status


def run_existing_self_healing(
    framework: ConstellationFramework,
    targets: ResolvedTargets,
    selected: SelectedScenario,
) -> dict[str, Any]:
    if framework.current_snapshot is None:
        raise RuntimeError("Framework snapshot is not initialized.")

    all_nodes = framework.current_snapshot.SatelliteNodes
    missing = [sid for sid in targets.selected_satellites if sid not in all_nodes]
    if missing:
        raise KeyError(f"Selected satellites are absent from the framework snapshot: {missing[:8]}")

    failed_nodes = {sid: all_nodes[sid] for sid in targets.selected_satellites}
    prepared_failed_nodes = prepare_failed_nodes_for_healing(
        failed_nodes,
        seed_material=f"{selected.scenario_id}|healing-baseline",
    )
    reference = next(iter(prepared_failed_nodes.values()))
    baseline_state = copy.deepcopy(reference.baseline_q_network_state)
    neighbors_status = build_neighbors_status(
        framework.current_snapshot.RawGraph,
        set(targets.selected_satellites),
        baseline_state,
    )
    fail_scores = {
        sid: float(targets.observations_by_satellite[sid]["fail_score"])
        for sid in targets.selected_satellites
    }
    attack_types = {
        sid: int(targets.observations_by_satellite[sid]["attack_type_id"])
        for sid in targets.selected_satellites
    }

    if len(prepared_failed_nodes) == 1:
        sid, node = next(iter(prepared_failed_nodes.items()))
        orchestrator = TieredHealingOrchestrator(sid)
        result = orchestrator.execute_tiered_healing(
            target_node=node,
            fail_score=fail_scores[sid],
            attack_type=attack_types[sid],
            neighbors_status=neighbors_status,
        )
        raw_results = {sid: result}
    else:
        orchestrator = TieredHealingOrchestrator("BATCH")
        raw_results = orchestrator.execute_batch_healing(
            failed_nodes=prepared_failed_nodes,
            fail_scores=fail_scores,
            attack_types=attack_types,
            all_nodes=all_nodes,
            neighbors_status=neighbors_status,
        )

    result_rows: dict[str, dict[str, Any]] = {}
    for sid, result in raw_results.items():
        logger_payload = result.logger.export_dict() if getattr(result, "logger", None) else None
        result_rows[sid] = {
            "node_id": sid,
            "success": bool(result.success),
            "healing_level": int(result.healing_level),
            "healing_time": float(result.healing_time),
            "message": str(result.message),
            "source_node": result.source_node,
            "logger": logger_payload,
        }

    restored_satellites = sorted(sid for sid, row in result_rows.items() if bool(row["success"]))
    return {
        "result_by_satellite": result_rows,
        "success_count": len(restored_satellites),
        "failed_count": len(result_rows) - len(restored_satellites),
        "restored_satellites": restored_satellites,
        "kept_isolated_satellites": sorted(set(targets.selected_satellites) - set(restored_satellites)),
    }


def make_framework(selected: SelectedScenario) -> ConstellationFramework:
    return ConstellationFramework(
        agent_config_path=DEFAULT_AGENT_CONFIG,
        output_constellation_index=selected.constellation_id,
        output_round_index=selected.round_index,
        output_test_id=selected.test_id,
        emit_initial_state=True,
        emit_on_topology_change=True,
        build_q_networks=False,
    )


def node_status_lookup(
    targets: ResolvedTargets,
    healing_summary: Mapping[str, Any],
    consensus_report: ConsensusScenarioReport,
) -> dict[str, dict[str, Any]]:
    result_by_satellite = healing_summary.get("result_by_satellite", {})
    consensus_by_sid = {item.SID: asdict(item) for item in consensus_report.satellite_records}
    lookup: dict[str, dict[str, Any]] = {}
    for sid in targets.selected_satellites:
        observation = targets.observations_by_satellite.get(sid, {})
        healing = result_by_satellite.get(sid, {}) if isinstance(result_by_satellite, Mapping) else {}
        consensus = consensus_by_sid.get(sid, {})
        logger = healing.get("logger") if isinstance(healing, Mapping) else None
        healing_time = healing.get("healing_time")
        log_data = dict(logger) if isinstance(logger, Mapping) else {}
        if "total_time" in log_data:
            log_data.setdefault("logger_total_time", log_data.get("total_time"))
        log_data["total_time"] = healing_time
        log_data["healing_time"] = healing_time
        log_data.setdefault("node_id", sid)
        log_data.setdefault("fail_score", observation.get("fail_score", consensus.get("FailScore")))
        log_data.setdefault("attack_type", observation.get("attack_type_id", consensus.get("AttackType")))
        log_data.setdefault(
            "attack_label",
            observation.get("attack_type_name", consensus.get("AttackTypeLabel")),
        )
        log_data.setdefault("final_success", healing.get("success"))
        log_data.setdefault("final_result", healing.get("message"))
        log_data.setdefault("final_level", healing.get("healing_level"))
        log_data.setdefault("steps", [])
        lookup[sid] = {
            "failed_agent_id": observation.get("failed_agent_id"),
            "fail_score": observation.get("fail_score", consensus.get("FailScore")),
            "attack_type_id": observation.get("attack_type_id", consensus.get("AttackType")),
            "attack_type": observation.get("attack_type_name", consensus.get("AttackTypeLabel")),
            "attack_label": consensus.get("AttackTypeLabel", observation.get("attack_type_name")),
            "attack_count": observation.get("attack_count"),
            "health_score": consensus.get("HealthScore"),
            "risk_level": consensus.get("RiskLevel"),
            "has_linked_failed_neighbor": consensus.get("HasLinkedFailedNeighbor"),
            "healing_success": healing.get("success"),
            "healing_level": healing.get("healing_level"),
            "healing_time": healing_time,
            "healing_message": healing.get("message"),
            "source_node": healing.get("source_node"),
            "log": log_data,
        }
    return lookup


def healing_time_stats(healing_summary: Mapping[str, Any]) -> dict[str, Any]:
    healing_time_items: list[tuple[str, float]] = []
    result_by_satellite = healing_summary.get("result_by_satellite", {})
    if isinstance(result_by_satellite, Mapping):
        for sid, row in result_by_satellite.items():
            if not isinstance(row, Mapping):
                continue
            try:
                healing_time_items.append((str(sid), float(row.get("healing_time"))))
            except (TypeError, ValueError):
                continue
    max_healing_time_item = max(healing_time_items, key=lambda item: item[1], default=None)
    return {
        "healing_node_count": len(healing_time_items),
        "max_node_healing_time": max_healing_time_item[1] if max_healing_time_item else None,
        "max_node_healing_time_satellite": max_healing_time_item[0] if max_healing_time_item else None,
    }


def write_visualization(
    path: Path,
    final_output: Any,
    selected: SelectedScenario,
    targets: ResolvedTargets,
    healing_summary: Mapping[str, Any],
    consensus_report: ConsensusScenarioReport,
) -> None:
    graph = final_output.RawGraph
    isolated = set(graph.graph.get("isolated_nodes", []))
    restored = set(healing_summary.get("restored_satellites", []))
    selected_set = set(targets.selected_satellites)
    metadata = node_status_lookup(targets, healing_summary, consensus_report)
    time_stats = healing_time_stats(healing_summary)

    groups = {
        "restored": {"color": "#16a34a", "nodes": []},
        "still_isolated": {"color": "#dc2626", "nodes": []},
        "selected": {"color": "#f59e0b", "nodes": []},
        "healthy": {"color": "#64748b", "nodes": []},
    }
    coordinate_values: list[float] = []
    for node_id in graph.nodes:
        pos = graph.nodes[node_id].get("pos")
        if not pos:
            continue
        coordinate_values.extend(float(value) for value in pos)
        if node_id in restored:
            key = "restored"
        elif node_id in isolated:
            key = "still_isolated"
        elif node_id in selected_set:
            key = "selected"
        else:
            key = "healthy"
        payload = {
            "sid": node_id,
            "status": key,
            "position_eci_km": [round(float(value), 3) for value in pos],
            "metadata": metadata.get(node_id, {}),
        }
        groups[key]["nodes"].append((float(pos[0]), float(pos[1]), float(pos[2]), node_id, json.dumps(payload)))

    fig = go.Figure()
    earth_radius = 6371
    axis_limit = max([earth_radius, *(abs(value) for value in coordinate_values)]) * 1.08
    import numpy as np

    phi, theta = np.mgrid[0.0 : 2.0 * np.pi : 42j, 0.0 : np.pi : 24j]
    fig.add_trace(
        go.Surface(
            x=earth_radius * np.sin(theta) * np.cos(phi),
            y=earth_radius * np.sin(theta) * np.sin(phi),
            z=earth_radius * np.cos(theta),
            colorscale="Blues",
            opacity=0.13,
            showscale=False,
            hoverinfo="skip",
            name="Earth",
        )
    )
    for label, group in groups.items():
        if not group["nodes"]:
            continue
        xs, ys, zs, names, customdata = zip(*group["nodes"])
        size = 5 if label != "healthy" else 2
        opacity = 0.95 if label != "healthy" else 0.35
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={"size": size, "color": group["color"], "opacity": opacity},
                name=label,
                text=names,
                customdata=customdata,
                hovertemplate="<b>%{text}</b><br>%{fullData.name}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        scene={
            "xaxis": {"visible": False, "range": [-axis_limit, axis_limit]},
            "yaxis": {"visible": False, "range": [-axis_limit, axis_limit]},
            "zaxis": {"visible": False, "range": [-axis_limit, axis_limit]},
            "bgcolor": "rgba(0,0,0,0)",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 1},
            "camera": {"center": {"x": 0, "y": 0, "z": 0}, "eye": {"x": 1.35, "y": 1.35, "z": 0.85}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 0.02, "x": 0.5, "xanchor": "center"},
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="part3-viz")
    scenario_json = json.dumps(
        {
            "scenario_id": selected.scenario_id,
            "constellation_id": selected.constellation_id,
            "round_index": selected.round_index,
            "test_id": selected.test_id,
            "branch": targets.branch,
            "fail_score": selected.fail_score,
            "predicted_attack_type": selected.predicted_attack_type,
            "selected_satellite_count": len(targets.selected_satellites),
            "original_failed_agent_count": len(targets.original_failed_agents),
            "success_count": healing_summary.get("success_count"),
            "failed_count": healing_summary.get("failed_count"),
            "restored_satellite_count": len(healing_summary.get("restored_satellites", [])),
            "kept_isolated_count": len(healing_summary.get("kept_isolated_satellites", [])),
            **time_stats,
            "linked_failure_ratio": consensus_report.linked_failure_ratio,
            "constellation_tle_path": path_for_report(Path(consensus_report.constellation_tle_path)),
        },
        ensure_ascii=False,
    )
    html_template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Part3 Self-Healing Report</title>
  <style>
    :root {
      --bg: #080b10;
      --panel: #111820;
      --panel-2: #151f2a;
      --border: #263443;
      --text: #edf2f7;
      --muted: #9caaba;
      --green: #2dd4bf;
      --amber: #f59e0b;
      --red: #fb7185;
      --blue: #60a5fa;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 430px;
      width: 100vw;
      height: 100vh;
      overflow: hidden;
    }
    .viz-wrap {
      min-width: 0;
      height: 100vh;
      background: #05070b;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #part3-viz {
      width: min(100%, 100vh) !important;
      height: min(100vh, 100%) !important;
      max-width: 100%;
      max-height: 100%;
      flex: 0 0 auto;
    }
    .panel {
      border-left: 1px solid var(--border);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
      box-shadow: -18px 0 38px rgba(0, 0, 0, 0.22);
    }
    .panel-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      font-size: 20px;
      line-height: 1.25;
      margin: 0 0 6px;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .status-pill {
      flex: 0 0 auto;
      border: 1px solid rgba(45, 212, 191, 0.45);
      color: #a7f3d0;
      background: rgba(20, 184, 166, 0.12);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      white-space: nowrap;
    }
    .card {
      border: 1px solid var(--border);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 13px;
      margin-top: 12px;
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: #dbeafe;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      min-width: 0;
      border: 1px solid rgba(148, 163, 184, 0.2);
      background: rgba(2, 6, 23, 0.28);
      border-radius: 7px;
      padding: 9px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }
    .metric-value {
      margin-top: 5px;
      font-size: 17px;
      font-weight: 700;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }
    .field-list {
      display: grid;
      gap: 7px;
      font-size: 12px;
    }
    .field-row {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 8px;
      align-items: baseline;
    }
    .field-row span:first-child {
      color: var(--muted);
    }
    .field-row span:last-child {
      color: #e5edf6;
      overflow-wrap: anywhere;
    }
    .tag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 4px 0 10px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: rgba(15, 23, 42, 0.52);
      color: #dbeafe;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      line-height: 1.2;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .tag.good { border-color: rgba(45, 212, 191, 0.42); color: #99f6e4; }
    .tag.warn { border-color: rgba(245, 158, 11, 0.48); color: #fde68a; }
    .tag.bad { border-color: rgba(251, 113, 133, 0.48); color: #fecdd3; }
    .timeline {
      display: grid;
      gap: 9px;
      margin-top: 10px;
    }
    .timeline-panel {
      border-top: 1px solid rgba(148, 163, 184, 0.22);
      margin-top: 12px;
      padding-top: 11px;
    }
    .step {
      border-left: 3px solid var(--blue);
      background: rgba(2, 6, 23, 0.25);
      border-radius: 0 7px 7px 0;
      padding: 8px 9px;
    }
    .step.success { border-left-color: var(--green); }
    .step.fail { border-left-color: var(--red); }
    .step-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: #eff6ff;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
    }
    .step-time {
      color: var(--muted);
      font-weight: 500;
      white-space: nowrap;
    }
    .step-detail {
      color: #cbd5e1;
      margin-top: 5px;
      font-size: 12px;
      line-height: 1.48;
      overflow-wrap: anywhere;
    }
    details {
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      color: #bfdbfe;
      font-size: 12px;
      user-select: none;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(2, 6, 23, 0.42);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 7px;
      padding: 11px;
      font-size: 11px;
      line-height: 1.45;
      color: #d1d5db;
    }
    .empty {
      border: 1px dashed rgba(148, 163, 184, 0.35);
      color: var(--muted);
      border-radius: 8px;
      padding: 22px 14px;
      text-align: center;
      line-height: 1.55;
      font-size: 13px;
    }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
        grid-template-rows: 58vh 42vh;
      }
      .viz-wrap, #part3-viz { height: 58vh; }
      #part3-viz {
        width: min(100vw, 58vh) !important;
        height: min(58vh, 100vw) !important;
      }
      .panel {
        height: 42vh;
        border-left: 0;
        border-top: 1px solid var(--border);
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="viz-wrap">__PLOT_HTML__</div>
    <aside class="panel">
      <div class="panel-header">
        <div>
          <h1>星座在线自愈报告</h1>
          <div class="subtitle" id="scenario-subtitle"></div>
        </div>
        <div class="status-pill" id="scenario-status">已完成</div>
      </div>

      <section class="card">
        <div class="section-title">场景概览</div>
        <div class="metrics" id="scenario-metrics"></div>
        <details>
          <summary>查看场景原始字段</summary>
          <pre id="scenario-json"></pre>
        </details>
      </section>

      <section class="card" id="node-card">
        <div class="empty">点击左侧 3D 星座中的卫星，查看失效诊断、自愈结果和执行时间线。</div>
      </section>
    </aside>
  </div>
  <script>
    var scenario = __SCENARIO_JSON__;

    function esc(value) {
      if (value === undefined || value === null || value === '') return '—';
      return String(value).replace(/[&<>"']/g, function(ch) {
        return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
      });
    }

    function numberText(value, digits) {
      if (value === undefined || value === null || value === '') return '—';
      var num = Number(value);
      if (!Number.isFinite(num)) return esc(value);
      return num.toFixed(digits === undefined ? 4 : digits);
    }

    function percentText(value) {
      if (value === undefined || value === null || value === '') return '—';
      var num = Number(value);
      if (!Number.isFinite(num)) return esc(value);
      return (num * 100).toFixed(2) + '%';
    }

    function secondsText(value, digits) {
      if (value === undefined || value === null || value === '') return '—';
      var num = Number(value);
      if (!Number.isFinite(num)) return esc(value);
      return num.toFixed(digits === undefined ? 4 : digits) + 's';
    }

    function boolText(value) {
      if (value === true) return '是';
      if (value === false) return '否';
      return '—';
    }

    function riskClass(risk) {
      var text = String(risk || '');
      if (text.indexOf('高') >= 0 || text.toLowerCase().indexOf('high') >= 0) return 'bad';
      if (text.indexOf('中') >= 0 || text.toLowerCase().indexOf('medium') >= 0) return 'warn';
      return 'good';
    }

    function statusText(status) {
      return {
        restored: '已恢复',
        still_isolated: '仍隔离',
        selected: '待处理',
        healthy: '健康'
      }[status] || status || '未知';
    }

    function parsePayload(raw) {
      if (Array.isArray(raw)) raw = raw[0];
      if (raw && typeof raw === 'object') return raw;
      if (typeof raw !== 'string') return null;
      try {
        return JSON.parse(raw);
      } catch (err) {
        return null;
      }
    }

    function metric(label, value) {
      return '<div class="metric"><div class="metric-label">' + esc(label) +
        '</div><div class="metric-value">' + esc(value) + '</div></div>';
    }

    function field(label, value) {
      return '<div class="field-row"><span>' + esc(label) + '</span><span>' + esc(value) + '</span></div>';
    }

    function renderScenario() {
      document.getElementById('scenario-subtitle').textContent =
        scenario.scenario_id + ' | Constellation ' + scenario.constellation_id +
        ' | Round ' + scenario.round_index + ' / Test ' + scenario.test_id;
      document.getElementById('scenario-json').textContent = JSON.stringify(scenario, null, 2);
      document.getElementById('scenario-metrics').innerHTML =
        metric('失效评分', numberText(scenario.fail_score, 4)) +
        metric('目标卫星', scenario.selected_satellite_count) +
        metric('恢复成功', scenario.success_count + ' / ' + scenario.selected_satellite_count) +
        metric('建链失效比例', percentText(scenario.linked_failure_ratio)) +
        metric('自愈节点数', scenario.healing_node_count) +
        metric('最大单节点自愈用时', secondsText(scenario.max_node_healing_time, 4)) +
        metric('最大耗时节点', scenario.max_node_healing_time_satellite || '—');
      if (Number(scenario.failed_count || 0) > 0 || Number(scenario.kept_isolated_count || 0) > 0) {
        document.getElementById('scenario-status').textContent = '需复核';
        document.getElementById('scenario-status').style.borderColor = 'rgba(251, 113, 133, 0.48)';
        document.getElementById('scenario-status').style.color = '#fecdd3';
      }
    }

    function renderStep(step, index) {
      var success = step && step.success;
      var cls = success === true ? ' success' : (success === false ? ' fail' : '');
      var level = step && (step.level_label || step.level);
      var action = step && step.action;
      var title = '#' + String(index + 1).padStart(2, '0') + ' ' + esc(action);
      var time = step && step.timestamp !== undefined ? numberText(step.timestamp, 4) + 's' : '—';
      return '<div class="step' + cls + '">' +
        '<div class="step-head"><span>' + title + '</span><span class="step-time">' + esc(time) + '</span></div>' +
        '<div class="step-detail">' +
        '<div>' + esc(step && step.detail) + '</div>' +
        '<div style="margin-top:4px;color:#94a3b8;">级别：' + esc(level) +
        '　来源节点：' + esc(step && step.source_node) + '</div>' +
        '</div></div>';
    }

    function renderNode(payload) {
      var card = document.getElementById('node-card');
      if (!payload) {
        card.innerHTML = '<div class="empty">没有读取到该节点的可视化数据。</div>';
        return;
      }
      var meta = payload.metadata || {};
      var log = meta.log || {};
      var steps = Array.isArray(log.steps) ? log.steps : [];
      var isHealthy = payload.status === 'healthy';
      var risk = meta.risk_level || '低失效风险';
      var finalSuccess = log.final_success !== undefined ? log.final_success : meta.healing_success;
      var successTag = finalSuccess === true ? '<span class="tag good">自愈成功</span>' :
        (finalSuccess === false ? '<span class="tag bad">自愈失败</span>' : '<span class="tag">未触发自愈</span>');
      var riskTag = '<span class="tag ' + riskClass(risk) + '">' + esc(risk) + '</span>';
      var html =
        '<div class="section-title"><span>' + esc(payload.sid) + '</span><span class="tag">' + esc(statusText(payload.status)) + '</span></div>' +
        '<div class="tag-row">' + riskTag + successTag +
        '<span class="tag">攻击：' + esc(meta.attack_label || meta.attack_type || log.attack_label) + '</span>' +
        '<span class="tag">Level ' + esc(log.final_level || meta.healing_level) + '</span></div>';

      if (isHealthy) {
        html += '<div class="field-list">' +
          field('节点状态', '未被本轮失效目标选中') +
          field('ECI 坐标(km)', (payload.position_eci_km || []).join(', ')) +
          '</div>';
      } else {
        html +=
          '<div class="field-list">' +
          field('原始失效智能体', meta.failed_agent_id) +
          field('失效评分', numberText(log.fail_score || meta.fail_score, 4)) +
          field('健康度评分', numberText(meta.health_score, 6)) +
          field('攻击类型ID', log.attack_type || meta.attack_type_id) +
          field('攻击类型', log.attack_label || meta.attack_label || meta.attack_type) +
          field('攻击记录数', meta.attack_count) +
          field('存在失效邻居', boolText(meta.has_linked_failed_neighbor)) +
          field('自愈结果', log.final_result || meta.healing_message) +
          field('该节点自愈用时', secondsText(meta.healing_time, 4)) +
          field('源节点', meta.source_node) +
          field('ECI 坐标(km)', (payload.position_eci_km || []).join(', ')) +
          '</div>';
      }

      if (steps.length) {
        html += '<div class="timeline-panel">' +
          '<div class="section-title">自愈执行时间线<span class="tag">' + steps.length + ' steps</span></div>' +
          '<div class="timeline">' + steps.map(renderStep).join('') + '</div></div>';
      }
      html += '<details><summary>查看节点原始字段</summary><pre>' + esc(JSON.stringify(payload, null, 2)) + '</pre></details>';
      card.innerHTML = html;
    }

    renderScenario();
    window.addEventListener('load', function() {
      var plot = document.getElementById('part3-viz');
      if (!plot) return;
      function resizePlot() {
        if (window.Plotly && window.Plotly.Plots) {
          window.Plotly.Plots.resize(plot);
        }
      }
      requestAnimationFrame(resizePlot);
      window.addEventListener('resize', resizePlot);
      if (!plot.on) return;
      plot.on('plotly_click', function(data) {
        if (!data.points || !data.points.length) return;
        renderNode(parsePayload(data.points[0].customdata));
      });
    });
  </script>
</body>
</html>
"""
    html = html_template.replace("__PLOT_HTML__", plot_html).replace("__SCENARIO_JSON__", scenario_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_scenario_plan(
    args: argparse.Namespace,
    *,
    output_root: Path,
    run_root: Path,
    analysis_output_root: Path,
    plan: ScenarioExecutionPlan,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected = plan.selected
    targets = plan.targets
    consensus_report = plan.consensus_report

    framework = make_framework(selected)
    initial_output = framework.initialize()
    result_diag = list(consensus_report.result_diag)

    isolation_targets = (
        targets.selected_satellites
        if selected.constellation_id == 2
        else [
            item.SID
            for item in result_diag
            if item.SID in set(targets.selected_satellites)
        ]
    )
    framework.inject_event(
        ResultDiag=result_diag,
        IsolationList=isolation_targets,
        EventId=f"{selected.scenario_id}:isolate",
        Metadata={
            "source": "full_project_pipeline",
            "branch": targets.branch,
            "healing_class": plan.healing_class,
        },
    )
    isolation_output = framework.flush_events()

    healing_summary = run_existing_self_healing(framework, targets, selected)
    time_stats = healing_time_stats(healing_summary)
    if healing_summary["restored_satellites"]:
        framework.inject_event(
            HealFlag=True,
            IsolationList=healing_summary["restored_satellites"],
            EventId=f"{selected.scenario_id}:heal",
            Metadata={
                "source": "online_self_healing.self_healing_module",
                "healing_class": plan.healing_class,
            },
        )
        heal_output = framework.flush_events()
    else:
        heal_output = framework.latest_output

    visualization_path = output_root / "self_healing_view.html"
    final_output = heal_output or isolation_output or initial_output
    if final_output is not None:
        write_visualization(
            visualization_path,
            final_output=final_output,
            selected=selected,
            targets=targets,
            healing_summary=healing_summary,
            consensus_report=consensus_report,
        )

    consensus_path = output_root / "consensus_report.json"
    consensus_payload = {
        "scenario_id": consensus_report.scenario_id,
        "linked_failure_ratio": consensus_report.linked_failure_ratio,
        "constellation_tle_path": path_for_report(Path(consensus_report.constellation_tle_path)),
        "satellite_records": [asdict(item) for item in consensus_report.satellite_records],
        "ledger": [
            {
                "state": str(entry.state),
                "message": entry.message,
                "details": entry.details,
                "timestamp_utc": entry.timestamp_utc.isoformat(),
            }
            for entry in consensus_report.ledger
        ],
    }
    write_json(consensus_path, consensus_payload)

    healing_path = output_root / "healing_results.json"
    write_json(healing_path, healing_summary)

    report_path = output_root / "part3_report.json"
    report = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": path_for_report(run_root),
        "analysis_output_root": path_for_report(analysis_output_root),
        "selection_class": {
            "healing_class": plan.healing_class,
            "healing_level": plan.healing_level,
            "label": plan.healing_label,
            "risk_class": plan.risk_class,
            "selection_rule": HEALING_CLASS_INFO[plan.healing_class]["selection_rule"],
            "selection_metrics": plan.selection_metrics,
        },
        "selected_scenario": {
            "scenario_id": selected.scenario_id,
            "round_index": selected.round_index,
            "test_id": selected.test_id,
            "constellation_id": selected.constellation_id,
            "source_session_dir": path_for_report(selected.source_session_dir),
            "performance_file": path_for_report(selected.performance_file),
            "fail_score": selected.fail_score,
            "target_field": args.target_field,
            "predicted_attack_type": selected.predicted_attack_type,
            "attack_observations": [asdict(item) for item in selected.attack_observations],
        },
        "data_contract": {
            "failed_agent_source": "AttackSummary lines in the selected performance file",
            "per_agent_fail_score_policy": (
                "Main pipeline outputs scenario/test-level fail scores, not per-agent fail scores; "
                "Part3 passes the selected scenario fail score to each observed failed agent."
            ),
            "missing_data_policy": "No failed agents are synthesized when AttackSummary data is absent.",
        },
        "constellation_branch": targets.branch,
        "constellation2_policy": {
            "enabled": selected.constellation_id == 2,
            "region_orbit_block_size": args.region_orbit_block_size,
            "region_sat_block_size": args.region_sat_block_size,
            "default_satellites_per_failed_agent": args.constellation2_default_satellites_per_agent,
            "max_satellites_per_failed_agent": args.constellation2_max_satellites_per_agent,
            "granularity_note": (
                "ConstellationConfig=2 reports region-level agents; Part3 isolates only the resolved satellite subset "
                "and aggregates healing results back to the failed region agent."
            ),
            "agent_mapping": targets.constellation2_agent_mapping,
        },
        "resolved_targets": {
            "original_failed_agents": targets.original_failed_agents,
            "selected_satellite_count": len(targets.selected_satellites),
            "selected_satellites": targets.selected_satellites,
            "observations_by_satellite": targets.observations_by_satellite,
        },
        "consensus": {
            "report_json": path_for_report(consensus_path),
            "linked_failure_ratio": consensus_report.linked_failure_ratio,
            "result_diag_count": len(consensus_report.result_diag),
        },
        "simulation": {
            "initial": graph_stats(initial_output),
            "isolation": graph_stats(isolation_output),
            "healing_restore": graph_stats(heal_output),
        },
        "healing": {
            "results_json": path_for_report(healing_path),
            "success_count": healing_summary["success_count"],
            "failed_count": healing_summary["failed_count"],
            **time_stats,
            "restored_satellites": healing_summary["restored_satellites"],
            "kept_isolated_satellites": healing_summary["kept_isolated_satellites"],
        },
        "outputs": {
            "report_json": path_for_report(report_path),
            "consensus_report_json": path_for_report(consensus_path),
            "healing_results_json": path_for_report(healing_path),
            "visualization_html": path_for_report(visualization_path),
        },
        "warnings": targets.warnings,
    }
    write_json(report_path, report)
    return report


def build_skip_report(
    output_root: Path,
    run_root: Path,
    analysis_output_root: Path,
    status: str,
    reason: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    report = {
        "status": status,
        "reason": reason,
        "run_root": path_for_report(run_root),
        "analysis_output_root": path_for_report(analysis_output_root),
        "warnings": list(warnings),
        "outputs": {
            "report_json": path_for_report(output_root / "part3_report.json"),
        },
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(output_root / "part3_report.json", report)
    return report


def run_from_full_project(args: argparse.Namespace) -> dict[str, Any]:
    full_run_root = resolve_path(args.full_run_root)
    run_root = resolve_path(args.run_root) if args.run_root else latest_run_root(full_run_root)
    analysis_output_root = (
        resolve_path(args.analysis_output_root)
        if args.analysis_output_root
        else run_root / "analysis"
    )
    output_root = resolve_path(args.output_root) if args.output_root else run_root / DEFAULT_OUTPUT_DIR_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline_summary = load_pipeline_summary(analysis_output_root)
    candidates = load_integrated_candidates(
        run_root=run_root,
        analysis_output_root=analysis_output_root,
        pipeline_summary=pipeline_summary,
        target_field=args.target_field,
    )
    if args.scenario_selection in {"highest_score_with_attack", "highest_score"}:
        print(
            "[Part3] 旧的按失效评分最高选择单场景逻辑已废弃；"
            "本次将按四类自愈方式分别选择场景。",
            flush=True,
        )

    plans, selection_summary, selection_warnings = select_scenarios_by_healing_class(
        candidates,
        args,
    )
    if not plans:
        return build_skip_report(
            output_root=output_root,
            run_root=run_root,
            analysis_output_root=analysis_output_root,
            status="skipped_missing_failed_agent_data",
            reason=(
                "No candidate scenario could be mapped to any of the four self-healing classes. "
                "Part3 does not synthesize failed agents."
            ),
            warnings=[
                f"candidate_count={len(candidates)}",
                f"selection_summary={json.dumps(selection_summary, ensure_ascii=False)}",
                "Required data source: performance/test_XXXX_performance.txt lines matching AttackSummary.",
                *selection_warnings,
            ],
        )

    scenario_reports: dict[str, dict[str, Any]] = {}
    aggregate_success = 0
    aggregate_failed = 0
    aggregate_restored: list[str] = []
    aggregate_kept_isolated: list[str] = []
    all_warnings = list(selection_warnings)
    for plan in plans:
        scenario_output_root = output_root / plan.healing_class
        scenario_report = run_scenario_plan(
            args,
            output_root=scenario_output_root,
            run_root=run_root,
            analysis_output_root=analysis_output_root,
            plan=plan,
        )
        scenario_reports[plan.healing_class] = scenario_report
        healing = scenario_report.get("healing", {})
        aggregate_success += int(healing.get("success_count", 0))
        aggregate_failed += int(healing.get("failed_count", 0))
        aggregate_restored.extend(str(item) for item in healing.get("restored_satellites", []))
        aggregate_kept_isolated.extend(str(item) for item in healing.get("kept_isolated_satellites", []))
        all_warnings.extend(str(item) for item in scenario_report.get("warnings", []))

    selected_scenarios = [
        {
            "healing_class": plan.healing_class,
            "healing_level": plan.healing_level,
            "label": plan.healing_label,
            "risk_class": plan.risk_class,
            **scenario_reports[plan.healing_class]["selected_scenario"],
        }
        for plan in plans
    ]
    scenario_report_paths = {
        healing_class: report.get("outputs", {}).get("report_json")
        for healing_class, report in scenario_reports.items()
    }
    consensus_report_paths = {
        healing_class: report.get("outputs", {}).get("consensus_report_json")
        for healing_class, report in scenario_reports.items()
    }
    healing_result_paths = {
        healing_class: report.get("outputs", {}).get("healing_results_json")
        for healing_class, report in scenario_reports.items()
    }
    visualization_paths = {
        healing_class: report.get("outputs", {}).get("visualization_html")
        for healing_class, report in scenario_reports.items()
    }

    report_path = output_root / "part3_report.json"
    report = {
        "status": "completed" if not selection_summary["skipped_classes"] else "completed_with_skips",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": path_for_report(run_root),
        "analysis_output_root": path_for_report(analysis_output_root),
        "selection_mode": "per_healing_level",
        "selection_summary": selection_summary,
        "selected_scenarios": selected_scenarios,
        "selected_scenario": selected_scenarios[0] if selected_scenarios else None,
        "scenario_reports": {
            healing_class: scenario_report_summary(report)
            for healing_class, report in scenario_reports.items()
        },
        "data_contract": {
            "failed_agent_source": "AttackSummary lines in the selected performance files",
            "per_agent_fail_score_policy": (
                "Main pipeline outputs scenario/test-level fail scores, not per-agent fail scores; "
                "Part3 passes each selected scenario fail score to its observed failed agents."
            ),
            "missing_data_policy": (
                "No failed agents are synthesized when AttackSummary data is absent; "
                "missing self-healing classes are skipped and reported."
            ),
        },
        "constellation_branch": (
            next(iter(scenario_reports.values())).get("constellation_branch")
            if len({report.get("constellation_branch") for report in scenario_reports.values()}) == 1
            else "multiple"
        ),
        "healing": {
            "success_count": aggregate_success,
            "failed_count": aggregate_failed,
            "restored_satellites": sorted(set(aggregate_restored)),
            "kept_isolated_satellites": sorted(set(aggregate_kept_isolated)),
        },
        "outputs": {
            "report_json": path_for_report(report_path),
            "scenario_report_jsons": scenario_report_paths,
            "consensus_report_jsons": consensus_report_paths,
            "healing_results_jsons": healing_result_paths,
            "visualization_htmls": visualization_paths,
            "visualization_html": next(iter(visualization_paths.values()), None),
        },
        "warnings": all_warnings,
    }
    write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online self-healing from a full-project pipeline run."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Main project root. Kept for manifest clarity.")
    parser.add_argument("--full-run-root", default=str(DEFAULT_FULL_RUN_ROOT))
    parser.add_argument("--run-root", default="", help="Specific full_project_runs/<run_id> directory. Defaults to latest.")
    parser.add_argument("--analysis-output-root", default="", help="Defaults to <run_root>/analysis.")
    parser.add_argument("--output-root", default="", help="Defaults to <run_root>/part3_rebuild.")
    parser.add_argument(
        "--scenario-selection",
        choices=("per_healing_level", "highest_score_with_attack", "highest_score"),
        default="per_healing_level",
        help=(
            "Scenario selection policy. per_healing_level selects one scenario for each "
            "Level 1-4 self-healing class; legacy highest_score* values are accepted as "
            "deprecated aliases for per_healing_level."
        ),
    )
    parser.add_argument("--target-field", default="fused_score")
    parser.add_argument("--region-orbit-block-size", type=int, default=5)
    parser.add_argument("--region-sat-block-size", type=int, default=5)
    parser.add_argument("--constellation2-default-satellites-per-agent", type=int, default=5)
    parser.add_argument("--constellation2-max-satellites-per-agent", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.constellation2_max_satellites_per_agent < 1 or args.constellation2_max_satellites_per_agent > 10:
        raise ValueError("--constellation2-max-satellites-per-agent must be in [1, 10].")
    if args.constellation2_default_satellites_per_agent < 1:
        raise ValueError("--constellation2-default-satellites-per-agent must be >= 1.")

    report = run_from_full_project(args)
    print(json.dumps({"status": report.get("status"), "outputs": report.get("outputs")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
