from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..node import NodeElement
from .catalog import ConstellationCatalog
from .models import (
    ATTACK_TYPE_LABELS,
    ABNORMAL_ATTACK_TYPE,
    ConsensusLedgerEntry,
    ConsensusScenarioReport,
    ConsensusState,
    FailPoint,
    FailedSatelliteObservation,
    OutputScenarioSelection,
    SatelliteConsensusRecord,
    infer_attack_type_from_fail_env,
)
from .output_reader import OutputScenarioReader


@dataclass(frozen=True)
class ConsensusPolicy:
    low_risk_fail_score_upper: float = 0.40
    high_risk_fail_score_lower: float = 0.50
    batch_link_ratio_threshold: float = 0.30
    sid_index_base: Union[int, str] = "auto"

    def __post_init__(self) -> None:
        if self.sid_index_base not in {"auto", 0, 1}:
            raise ValueError("sid_index_base must be one of: 'auto', 0, or 1.")
        if not 0.0 <= self.low_risk_fail_score_upper <= self.high_risk_fail_score_lower <= 1.0:
            raise ValueError(
                "Fail-score thresholds must satisfy "
                "0 <= low_risk_fail_score_upper <= high_risk_fail_score_lower <= 1."
            )
        if not 0.0 <= self.batch_link_ratio_threshold <= 1.0:
            raise ValueError("batch_link_ratio_threshold must be within [0, 1].")


@dataclass(frozen=True)
class _NormalizedFailureRecord:
    SID: str
    RawSID: Union[int, str]
    FailScore: float
    AttackType: int


class BlockchainConsensusStateMachine:
    def __init__(
        self,
        *,
        output_root_dir: Optional[Union[str, "Path"]] = None,
        satellite_data_dir: Optional[Union[str, "Path"]] = None,
        policy: ConsensusPolicy = ConsensusPolicy(),
    ):
        self.output_reader = OutputScenarioReader(output_root_dir)
        self.satellite_data_dir = satellite_data_dir
        self.policy = policy
        self._catalog_cache: Dict[int, ConstellationCatalog] = {}

    def process_output_selection(
        self,
        output_selection_input: Any,
    ) -> ConsensusScenarioReport:
        selection = OutputScenarioSelection.from_input(output_selection_input)
        fail_point = self.output_reader.build_fail_point(selection)
        report = self.process_fail_point(fail_point)
        return report

    def process_output_selection_list(
        self,
        output_selection_list_input: Iterable[Any],
    ) -> List[ConsensusScenarioReport]:
        return [self.process_output_selection(item) for item in output_selection_list_input]

    def build_result_diag_from_output(
        self,
        output_selection_input: Any,
    ) -> List[NodeElement]:
        return self.process_output_selection(output_selection_input).as_framework_result_diag()

    def process_fail_point(
        self,
        fail_point_input: Any,
    ) -> ConsensusScenarioReport:
        fail_point = FailPoint.from_input(fail_point_input)
        ledger: List[ConsensusLedgerEntry] = [
            ConsensusLedgerEntry(
                state=ConsensusState.RECEIVED,
                message="Accepted a FailPoint payload for consensus analysis.",
                details={"reported_fail_sat_count": len(fail_point.FailSat)},
            )
        ]

        if not fail_point.FailSat:
            raise ValueError("FailPoint.FailSat cannot be empty.")

        inferred_attack_type, inferred_attack_label, inferred_attack_labels = (
            infer_attack_type_from_fail_env(fail_point.FailEnv)
        )
        ledger.append(
            ConsensusLedgerEntry(
                state=ConsensusState.VALIDATED,
                message="Validated FailPoint structure and required scenario parameters.",
                details={
                    "constellation_config": fail_point.FailEnv.ConstellationConfig,
                    "inferred_attack_type": inferred_attack_type,
                    "inferred_attack_label": inferred_attack_label,
                    "active_attack_labels": list(inferred_attack_labels),
                },
            )
        )

        catalog = self._get_catalog(fail_point.FailEnv.ConstellationConfig)
        sid_index_base = self._resolve_sid_index_base(fail_point.FailSat)
        normalized_records = self._normalize_fail_sat_records(
            fail_point.FailSat,
            catalog,
            sid_index_base=sid_index_base,
        )
        ledger.append(
            ConsensusLedgerEntry(
                state=ConsensusState.NORMALIZED,
                message="Normalized failure observations into canonical satellite identifiers.",
                details={
                    "sid_index_base": sid_index_base,
                    "normalized_fail_sat_count": len(normalized_records),
                    "tle_path": str(catalog.tle_path),
                },
            )
        )

        linked_failure_ratio, has_linked_neighbor = self._compute_linked_failure_ratio(
            normalized_records,
            catalog,
        )
        result_diag: List[NodeElement] = []
        satellite_records: List[SatelliteConsensusRecord] = []
        for record in normalized_records:
            health_score = self._fail_score_to_health_score(record.FailScore)
            risk_level = self._resolve_risk_level(
                fail_score=record.FailScore,
                linked_failure_ratio=linked_failure_ratio,
            )
            result_diag.append(
                NodeElement(
                    SID=record.SID,
                    HealthScore=health_score,
                    RiskLevel=risk_level,
                )
            )
            satellite_records.append(
                SatelliteConsensusRecord(
                    SID=record.SID,
                    RawSID=record.RawSID,
                    FailScore=record.FailScore,
                    HealthScore=health_score,
                    AttackType=record.AttackType,
                    AttackTypeLabel=ATTACK_TYPE_LABELS.get(
                        int(record.AttackType), f"Unknown({record.AttackType})"
                    ),
                    RiskLevel=risk_level,
                    HasLinkedFailedNeighbor=has_linked_neighbor[record.SID],
                )
            )

        ledger.append(
            ConsensusLedgerEntry(
                state=ConsensusState.ANALYZED,
                message="Derived HealthScore and RiskLevel values for the scenario.",
                details={
                    "linked_failure_ratio": linked_failure_ratio,
                    "result_diag_count": len(result_diag),
                },
            )
        )

        scenario_id = (
            fail_point.ScenarioId
            if fail_point.ScenarioId is not None
            else self._build_scenario_id(fail_point, normalized_records)
        )
        report = ConsensusScenarioReport(
            scenario_id=scenario_id,
            fail_env=fail_point.FailEnv,
            result_diag=tuple(result_diag),
            satellite_records=tuple(satellite_records),
            linked_failure_ratio=linked_failure_ratio,
            consensus_state=ConsensusState.FINALIZED,
            constellation_tle_path=str(catalog.tle_path),
            ledger=tuple(
                ledger
                + [
                    ConsensusLedgerEntry(
                        state=ConsensusState.FINALIZED,
                        message="Finalized the blockchain consensus diagnosis report.",
                        details={"scenario_id": scenario_id},
                    )
                ]
            ),
        )
        return report

    def process_fail_point_list(
        self,
        fail_point_list_input: Iterable[Any],
    ) -> List[ConsensusScenarioReport]:
        return [self.process_fail_point(item) for item in fail_point_list_input]

    def build_result_diag(
        self,
        fail_point_input: Any,
    ) -> List[NodeElement]:
        return self.process_fail_point(fail_point_input).as_framework_result_diag()

    def build_result_diag_batch(
        self,
        fail_point_list_input: Iterable[Any],
    ) -> List[List[NodeElement]]:
        return [
            report.as_framework_result_diag()
            for report in self.process_fail_point_list(fail_point_list_input)
        ]

    def resolve_output_model_dir(
        self,
        output_selection_input: Any,
    ) -> str:
        return str(self.output_reader.resolve_paths(output_selection_input).model_dir)

    def _get_catalog(self, constellation_config: int) -> ConstellationCatalog:
        config_index = int(constellation_config)
        cached_catalog = self._catalog_cache.get(config_index)
        if cached_catalog is not None:
            return cached_catalog

        catalog = ConstellationCatalog.from_constellation_config(
            config_index,
            satellite_data_dir=self.satellite_data_dir,
        )
        self._catalog_cache[config_index] = catalog
        return catalog

    def _resolve_sid_index_base(
        self,
        fail_sat_records: Sequence[FailedSatelliteObservation],
    ) -> int:
        configured_index_base = self.policy.sid_index_base
        if configured_index_base in {0, 1}:
            return int(configured_index_base)

        for record in fail_sat_records:
            if isinstance(record.SID, str) and record.SID.startswith("Satellite_"):
                continue
            if int(record.SID) == 0:
                return 0
        return 1

    def _normalize_fail_sat_records(
        self,
        fail_sat_records: Sequence[FailedSatelliteObservation],
        catalog: ConstellationCatalog,
        *,
        sid_index_base: int,
    ) -> List[_NormalizedFailureRecord]:
        grouped_records: Dict[str, List[FailedSatelliteObservation]] = {}
        first_raw_sid_by_resolved_sid: Dict[str, Union[int, str]] = {}

        for raw_record in fail_sat_records:
            resolved_sid = catalog.resolve_satellite_id(
                raw_record.SID,
                sid_index_base=sid_index_base,
            )
            grouped_records.setdefault(resolved_sid, []).append(raw_record)
            first_raw_sid_by_resolved_sid.setdefault(resolved_sid, raw_record.SID)

        normalized: List[_NormalizedFailureRecord] = []
        for resolved_sid, grouped in grouped_records.items():
            fail_score = max(self._clamp_score(item.FailScore) for item in grouped)
            attack_type = self._resolve_attack_type_consensus(grouped)
            normalized.append(
                _NormalizedFailureRecord(
                    SID=resolved_sid,
                    RawSID=first_raw_sid_by_resolved_sid[resolved_sid],
                    FailScore=fail_score,
                    AttackType=attack_type,
                )
            )

        normalized.sort(key=lambda item: catalog.name_to_one_based_index[item.SID])
        return normalized

    def _resolve_attack_type_consensus(
        self,
        grouped_records: Sequence[FailedSatelliteObservation],
    ) -> int:
        counts = Counter(int(item.AttackType) for item in grouped_records)
        best_count = max(counts.values())
        candidate_attack_types = sorted(
            attack_type
            for attack_type, count in counts.items()
            if count == best_count
        )
        if len(candidate_attack_types) == 1:
            return candidate_attack_types[0]

        max_score_by_attack_type = {
            attack_type: max(
                self._clamp_score(item.FailScore)
                for item in grouped_records
                if int(item.AttackType) == attack_type
            )
            for attack_type in candidate_attack_types
        }
        return max(
            candidate_attack_types,
            key=lambda attack_type: (max_score_by_attack_type[attack_type], -attack_type),
        )

    def _compute_linked_failure_ratio(
        self,
        normalized_records: Sequence[_NormalizedFailureRecord],
        catalog: ConstellationCatalog,
    ) -> Tuple[float, Dict[str, bool]]:
        failed_satellite_ids = {record.SID for record in normalized_records}
        if not failed_satellite_ids:
            return 0.0, {}

        has_linked_neighbor = {
            satellite_id: bool(catalog.neighbors_of(satellite_id) & failed_satellite_ids)
            for satellite_id in failed_satellite_ids
        }
        linked_failure_count = sum(1 for is_linked in has_linked_neighbor.values() if is_linked)
        linked_failure_ratio = linked_failure_count / len(failed_satellite_ids)
        return linked_failure_ratio, has_linked_neighbor

    def _fail_score_to_health_score(self, fail_score: float) -> float:
        return round(1.0 - self._clamp_score(fail_score), 6)

    def _resolve_risk_level(
        self,
        *,
        fail_score: float,
        linked_failure_ratio: float,
    ) -> str:
        clamped_score = self._clamp_score(fail_score)
        if clamped_score >= self.policy.high_risk_fail_score_lower:
            return "高失效风险"
        if clamped_score >= self.policy.low_risk_fail_score_upper:
            if linked_failure_ratio > self.policy.batch_link_ratio_threshold:
                return "批量中失效风险"
            return "中失效风险"
        return "低失效风险"

    def _build_scenario_id(
        self,
        fail_point: FailPoint,
        normalized_records: Sequence[_NormalizedFailureRecord],
    ) -> str:
        scenario_material = "|".join(
            [
                str(fail_point.FailEnv.as_feature_dict()),
                ",".join(
                    f"{record.SID}:{record.FailScore:.6f}:{record.AttackType}"
                    for record in normalized_records
                ),
            ]
        )
        digest = sha1(scenario_material.encode("utf-8")).hexdigest()[:12]
        return f"scenario-{digest}"

    @staticmethod
    def _clamp_score(score: float) -> float:
        return min(max(float(score), 0.0), 1.0)
