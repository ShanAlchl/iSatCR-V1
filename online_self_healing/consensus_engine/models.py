from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..node import NodeElement


FAIL_ENV_FIELD_NAMES: Tuple[str, ...] = (
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
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
)

NO_ATTACK_TYPE = 0
ABNORMAL_ATTACK_TYPE = -1

ATTACK_TYPE_LABELS: Dict[int, str] = {
    ABNORMAL_ATTACK_TYPE: "异常",
    NO_ATTACK_TYPE: "无攻击",
    1: "StateObservationAttack",
    2: "ActionAttack",
    3: "StateTransferAttack",
    4: "RewardAttack",
    5: "ExperiencePoolAttack",
    6: "ModelTampAttack",
}

ATTACK_LEVEL_FIELD_TO_TYPE: Tuple[Tuple[str, int], ...] = (
    ("StateObservationAttack_level", 1),
    ("ActionAttack_level", 2),
    ("StateTransferAttack_level", 3),
    ("RewardAttack_level", 4),
    ("ExperiencePoolAttack_level", 5),
    ("ModelTampAttack_level", 6),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_mapping_value(
    mapping: Mapping[str, Any],
    primary_key: str,
    *aliases: str,
) -> Any:
    for key in (primary_key, *aliases):
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Required key '{primary_key}' is missing from the input payload.")


def _clamp_score(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class OutputScenarioSelection:
    output_constellation_index: int
    round_index: int
    test_id: int

    @classmethod
    def from_input(cls, raw: Any) -> "OutputScenarioSelection":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, Mapping):
            return cls(
                output_constellation_index=int(
                    _read_mapping_value(
                        raw,
                        "output_constellation_index",
                        "output_dir_index",
                        "config_dir_index",
                        "ConstellationConfig",
                    )
                ),
                round_index=int(_read_mapping_value(raw, "round_index", "round")),
                test_id=int(_read_mapping_value(raw, "test_id", "test")),
            )
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 3:
                raise ValueError(
                    "OutputScenarioSelection sequences must follow "
                    "(output_constellation_index, round_index, test_id)."
                )
            return cls(
                output_constellation_index=int(raw[0]),
                round_index=int(raw[1]),
                test_id=int(raw[2]),
            )
        raise TypeError("OutputScenarioSelection must be a mapping or a 3-item sequence.")


@dataclass(frozen=True)
class FailEnvironment:
    ConstellationConfig: int
    DegradedEdgeRatio: float
    EdgeDisconnectRatio: float
    EdgeBandwidthMeanDecreaseRatio: float
    EdgeBandwidthDecreaseStd: float
    PoissonRate: float
    MeanIntervalTime: float
    PacketGenerationInterval: float
    PacketSizeMean: float
    PacketSizeStd: float
    StateObservationAttack_level: int
    ActionAttack_level: int
    StateTransferAttack_level: int
    RewardAttack_level: int
    ExperiencePoolAttack_level: int
    ModelTampAttack_level: int

    @classmethod
    def from_input(cls, raw: Any) -> "FailEnvironment":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("FailEnvironment must be a mapping-compatible object.")

        return cls(
            ConstellationConfig=int(
                _read_mapping_value(raw, "ConstellationConfig", "constellation_config")
            ),
            DegradedEdgeRatio=float(
                _read_mapping_value(raw, "DegradedEdgeRatio", "degraded_edge_ratio")
            ),
            EdgeDisconnectRatio=float(
                _read_mapping_value(raw, "EdgeDisconnectRatio", "edge_disconnect_ratio")
            ),
            EdgeBandwidthMeanDecreaseRatio=float(
                _read_mapping_value(
                    raw,
                    "EdgeBandwidthMeanDecreaseRatio",
                    "edge_bandwidth_mean_decrease_ratio",
                )
            ),
            EdgeBandwidthDecreaseStd=float(
                _read_mapping_value(
                    raw,
                    "EdgeBandwidthDecreaseStd",
                    "edge_bandwidth_decrease_std",
                )
            ),
            PoissonRate=float(_read_mapping_value(raw, "PoissonRate", "poisson_rate")),
            MeanIntervalTime=float(
                _read_mapping_value(raw, "MeanIntervalTime", "mean_interval_time")
            ),
            PacketGenerationInterval=float(
                _read_mapping_value(
                    raw,
                    "PacketGenerationInterval",
                    "packet_generation_interval",
                )
            ),
            PacketSizeMean=float(
                _read_mapping_value(raw, "PacketSizeMean", "packet_size_mean")
            ),
            PacketSizeStd=float(
                _read_mapping_value(raw, "PacketSizeStd", "packet_size_std")
            ),
            StateObservationAttack_level=int(
                _read_mapping_value(
                    raw,
                    "StateObservationAttack_level",
                    "state_observation_attack_level",
                )
            ),
            ActionAttack_level=int(
                _read_mapping_value(raw, "ActionAttack_level", "action_attack_level")
            ),
            StateTransferAttack_level=int(
                _read_mapping_value(
                    raw,
                    "StateTransferAttack_level",
                    "state_transfer_attack_level",
                )
            ),
            RewardAttack_level=int(
                _read_mapping_value(raw, "RewardAttack_level", "reward_attack_level")
            ),
            ExperiencePoolAttack_level=int(
                _read_mapping_value(
                    raw,
                    "ExperiencePoolAttack_level",
                    "experience_pool_attack_level",
                )
            ),
            ModelTampAttack_level=int(
                _read_mapping_value(raw, "ModelTampAttack_level", "model_tamp_attack_level")
            ),
        )

    def as_feature_dict(self) -> Dict[str, Union[int, float]]:
        return {field_name: getattr(self, field_name) for field_name in FAIL_ENV_FIELD_NAMES}


def infer_attack_type_from_fail_env(
    fail_env_input: Union["FailEnvironment", Mapping[str, Any]],
) -> Tuple[int, str, Tuple[str, ...]]:
    fail_env = FailEnvironment.from_input(fail_env_input)
    active_types = tuple(
        ATTACK_TYPE_LABELS[attack_type]
        for field_name, attack_type in ATTACK_LEVEL_FIELD_TO_TYPE
        if int(getattr(fail_env, field_name)) != 0
    )
    if not active_types:
        return (NO_ATTACK_TYPE, ATTACK_TYPE_LABELS[NO_ATTACK_TYPE], ())
    if len(active_types) > 1:
        return (ABNORMAL_ATTACK_TYPE, ATTACK_TYPE_LABELS[ABNORMAL_ATTACK_TYPE], active_types)

    attack_label = active_types[0]
    for field_name, attack_type in ATTACK_LEVEL_FIELD_TO_TYPE:
        if ATTACK_TYPE_LABELS[attack_type] == attack_label:
            return (attack_type, attack_label, active_types)
    raise RuntimeError(f"Unable to resolve attack type label: {attack_label}")


@dataclass(frozen=True)
class FailedSatelliteObservation:
    SID: Union[int, str]
    FailScore: float
    AttackType: int

    @classmethod
    def from_input(cls, raw: Any) -> "FailedSatelliteObservation":
        if isinstance(raw, cls):
            return raw

        if isinstance(raw, Mapping):
            sid = _read_mapping_value(raw, "SID", "sid")
            fail_score = _read_mapping_value(raw, "FailScore", "fail_score")
            attack_type = _read_mapping_value(raw, "AttackType", "attack_type")
            return cls(SID=sid, FailScore=float(fail_score), AttackType=int(attack_type))

        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 3:
                raise ValueError(
                    "FailedSatelliteObservation sequences must follow "
                    "(SID, FailScore, AttackType)."
                )
            sid, fail_score, attack_type = raw
            return cls(SID=sid, FailScore=float(fail_score), AttackType=int(attack_type))

        raise TypeError("FailedSatelliteObservation must be a mapping or a 3-item sequence.")


@dataclass(frozen=True)
class FailPoint:
    FailEnv: FailEnvironment
    FailSat: Tuple[FailedSatelliteObservation, ...]
    ScenarioId: Optional[str] = None

    @classmethod
    def from_input(cls, raw: Any) -> "FailPoint":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("FailPoint must be a mapping-compatible object.")

        fail_env = FailEnvironment.from_input(_read_mapping_value(raw, "FailEnv", "fail_env"))
        raw_fail_sat = _read_mapping_value(raw, "FailSat", "fail_sat")
        if not isinstance(raw_fail_sat, Sequence) or isinstance(raw_fail_sat, (str, bytes)):
            raise TypeError("FailPoint.FailSat must be a sequence of failed-satellite entries.")

        fail_sat = tuple(FailedSatelliteObservation.from_input(item) for item in raw_fail_sat)
        scenario_id = raw.get("ScenarioId", raw.get("scenario_id"))
        return cls(
            FailEnv=fail_env,
            FailSat=fail_sat,
            ScenarioId=None if scenario_id is None else str(scenario_id),
        )


class ConsensusState(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    ANALYZED = "analyzed"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class ConsensusLedgerEntry:
    state: ConsensusState
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class SatelliteConsensusRecord:
    SID: str
    RawSID: Union[int, str]
    FailScore: float
    HealthScore: float
    AttackType: int
    AttackTypeLabel: str
    RiskLevel: str
    HasLinkedFailedNeighbor: bool


@dataclass(frozen=True)
class ConsensusScenarioReport:
    scenario_id: str
    fail_env: FailEnvironment
    result_diag: Tuple[NodeElement, ...]
    satellite_records: Tuple[SatelliteConsensusRecord, ...]
    linked_failure_ratio: float
    consensus_state: ConsensusState
    constellation_tle_path: str
    ledger: Tuple[ConsensusLedgerEntry, ...] = field(default_factory=tuple)

    def as_framework_result_diag(self) -> list[NodeElement]:
        return list(self.result_diag)

    def as_framework_event_payload(
        self,
        *,
        event_id: Optional[str] = None,
        isolation_risk_levels: Iterable[str] = ("批量中失效风险", "高失效风险"),
    ) -> Dict[str, Any]:
        isolation_levels = {str(level) for level in isolation_risk_levels}
        return {
            "EventId": event_id,
            "ResultDiag": list(self.result_diag),
            "IsolationList": [
                item.SID for item in self.result_diag if item.RiskLevel in isolation_levels
            ],
        }
