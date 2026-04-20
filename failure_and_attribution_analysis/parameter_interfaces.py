import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple


SCENARIO_PARAMETER_NAMES: Tuple[str, ...] = (
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

METRIC_NAMES: Tuple[str, ...] = (
    "PacketLossRate",
    "NetworkThroughput",
    "BandwidthUtilization",
    "AvgPacketNodeVisits",
    "CumulativeReward",
    "AverageInferenceTime",
    "AverageE2eDelay",
    "AverageHopCount",
    "AverageComputingRatio",
    "ComputingWaitingTime",
    "AverageEndingReward",
)

CONTINUOUS_FEATURE_NAMES: Tuple[str, ...] = (
    "DegradedEdgeRatio",
    "EdgeDisconnectRatio",
    "EdgeBandwidthMeanDecreaseRatio",
    "EdgeBandwidthDecreaseStd",
    "PoissonRate",
    "MeanIntervalTime",
    "PacketGenerationInterval",
    "PacketSizeMean",
    "PacketSizeStd",
)

DISCRETE_FEATURE_NAME = "ConstellationConfig"
DISCRETE_FEATURE_NAMES: Tuple[str, ...] = (
    "ConstellationConfig",
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
)

DEFAULT_ENV_VALUES: Dict[str, float] = {
    "ConstellationConfig": 0,
    "DegradedEdgeRatio": 0.0,
    "EdgeDisconnectRatio": 0.0,
    "EdgeBandwidthMeanDecreaseRatio": 0.0,
    "EdgeBandwidthDecreaseStd": 0.0,
    "PoissonRate": 30.0,
    "MeanIntervalTime": 30.0,
    "PacketGenerationInterval": 2.0,
    "PacketSizeMean": 400000000,
    "PacketSizeStd": 115470000,
    "StateObservationAttack_level": 0,
    "ActionAttack_level": 0,
    "StateTransferAttack_level": 0,
    "RewardAttack_level": 0,
    "ExperiencePoolAttack_level": 0,
    "ModelTampAttack_level": 0,
}

STEP_HEADER_RE = re.compile(r"^====== step\s+(\d+)(?:\s*\|.*)? ======$", re.MULTILINE)
HEADER_KEY_RE = re.compile(r"^([A-Z_]+):\s*(.+)$")


@dataclass
class EnvConfig:
    ConstellationConfig: int
    DegradedEdgeRatio: float
    EdgeDisconnectRatio: float
    EdgeBandwidthMeanDecreaseRatio: float
    EdgeBandwidthDecreaseStd: float
    PoissonRate: float
    MeanIntervalTime: float
    PacketGenerationInterval: float
    PacketSizeMean: int
    PacketSizeStd: int
    StateObservationAttack_level: int
    ActionAttack_level: int
    StateTransferAttack_level: int
    RewardAttack_level: int
    ExperiencePoolAttack_level: int
    ModelTampAttack_level: int

    @classmethod
    def from_mapping(cls, data: Dict) -> "EnvConfig":
        merged = dict(DEFAULT_ENV_VALUES)
        merged.update(data or {})
        normalized = {}
        for key in SCENARIO_PARAMETER_NAMES:
            value = merged[key]
            if key in {
                "ConstellationConfig",
                "PacketSizeMean",
                "PacketSizeStd",
                "StateObservationAttack_level",
                "ActionAttack_level",
                "StateTransferAttack_level",
                "RewardAttack_level",
                "ExperiencePoolAttack_level",
                "ModelTampAttack_level",
            }:
                normalized[key] = int(value)
            else:
                normalized[key] = float(value)
        return cls(**normalized)

    def to_feature_arrays(
        self,
        continuous_feature_names: Sequence[str] = CONTINUOUS_FEATURE_NAMES,
        discrete_feature_names: Sequence[str] = DISCRETE_FEATURE_NAMES,
    ) -> Tuple[List[float], List[int]]:
        continuous_values = [float(getattr(self, name)) for name in continuous_feature_names]
        discrete_values = [int(getattr(self, name)) for name in discrete_feature_names]
        return continuous_values, discrete_values


@dataclass
class FailEnv:
    ConstellationConfig: int
    DegradedEdgeRatio: float
    EdgeDisconnectRatio: float
    EdgeBandwidthMeanDecreaseRatio: float
    EdgeBandwidthDecreaseStd: float
    PoissonRate: float
    MeanIntervalTime: float
    PacketGenerationInterval: float
    PacketSizeMean: int
    PacketSizeStd: int
    StateObservationAttack_level: int
    ActionAttack_level: int
    StateTransferAttack_level: int
    RewardAttack_level: int
    ExperiencePoolAttack_level: int
    ModelTampAttack_level: int

    @classmethod
    def from_env_config(cls, env: EnvConfig) -> "FailEnv":
        return cls(**asdict(env))

    @classmethod
    def from_mapping(cls, data: Dict) -> "FailEnv":
        return cls(**asdict(EnvConfig.from_mapping(data)))

    def to_feature_arrays(
        self,
        continuous_feature_names: Sequence[str] = CONTINUOUS_FEATURE_NAMES,
        discrete_feature_names: Sequence[str] = DISCRETE_FEATURE_NAMES,
    ) -> Tuple[List[float], List[int]]:
        env = EnvConfig.from_mapping(asdict(self))
        return env.to_feature_arrays(continuous_feature_names, discrete_feature_names)


@dataclass
class PerformanceMetrics:
    PacketLossRate: float
    NetworkThroughput: float
    BandwidthUtilization: float
    AvgPacketNodeVisits: float
    CumulativeReward: float
    AverageInferenceTime: float
    AverageE2eDelay: float
    AverageHopCount: float
    AverageComputingRatio: float
    ComputingWaitingTime: float
    AverageEndingReward: float


@dataclass
class AttackModuleInput:
    FinalEnv: List[EnvConfig]
    Metrics: PerformanceMetrics
    StepIndex: int = 0
    TestId: str = ""
    RoundIndex: int = 0
    SourceFile: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        envs_data = data.get("FinalEnv", [])
        env_configs = [EnvConfig.from_mapping(env) for env in envs_data]
        metrics = PerformanceMetrics(**data.get("Metrics", {}))
        return cls(
            FinalEnv=env_configs,
            Metrics=metrics,
            StepIndex=int(data.get("StepIndex", 0)),
            TestId=str(data.get("TestId", "")),
            RoundIndex=int(data.get("RoundIndex", 0)),
            SourceFile=str(data.get("SourceFile", "")),
        )

    @classmethod
    def parse_from_log_file(cls, file_path: str) -> List["AttackModuleInput"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        header_meta = _extract_header_metadata(content)
        env_mapping = _extract_env_mapping(file_path, content, header_meta)
        env_config = EnvConfig.from_mapping(env_mapping)

        inputs_list: List[AttackModuleInput] = []
        for step_index, block in _split_step_blocks(content):
            metrics = _extract_metrics(block)
            if metrics is None:
                continue

            inputs_list.append(
                cls(
                    FinalEnv=[env_config],
                    Metrics=metrics,
                    StepIndex=step_index,
                    TestId=str(header_meta.get("TEST_ID", "")),
                    RoundIndex=int(header_meta.get("ROUND_INDEX", 0) or 0),
                    SourceFile=file_path,
                )
            )

        return inputs_list

    def to_dict(self):
        return asdict(self)


def _extract_header_metadata(content: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in content.splitlines()[:40]:
        match = HEADER_KEY_RE.match(line.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        meta[key] = value
    return meta


def _extract_env_mapping(file_path: str, content: str, header_meta: Dict[str, str]) -> Dict:
    scenario_json = header_meta.get("SCENARIO_JSON")
    if scenario_json:
        try:
            return json.loads(scenario_json)
        except json.JSONDecodeError:
            pass

    inline_env = _extract_inline_env_fields(content)
    if inline_env:
        return inline_env

    return _infer_env_from_filename(file_path)


def _extract_inline_env_fields(content: str) -> Dict:
    env_mapping: Dict[str, float] = {}
    for line in content.splitlines()[:80]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in SCENARIO_PARAMETER_NAMES:
            continue
        env_mapping[key] = _parse_scalar(value.strip())
    return env_mapping


def _infer_env_from_filename(file_path: str) -> Dict:
    lower_path = file_path.lower()
    env_mapping = dict(DEFAULT_ENV_VALUES)

    if "medium" in lower_path:
        env_mapping["PoissonRate"] = 30.0
        env_mapping["MeanIntervalTime"] = 30.0
        env_mapping["PacketGenerationInterval"] = 2.0
    elif "low" in lower_path:
        env_mapping["PoissonRate"] = 45.0
        env_mapping["MeanIntervalTime"] = 15.0
        env_mapping["PacketGenerationInterval"] = 4.0

    constellation_match = re.search(r"(?:^|[_-])(0|1|2|3|4)(?:[_-]|\.|$)", lower_path)
    if constellation_match:
        env_mapping["ConstellationConfig"] = int(constellation_match.group(1))

    attack_patterns = {
        "StateObservationAttack_level": r"stateobservationattack[_-]?level[_=]?(\d+)|stateobservationattack[_=](\d+)",
        "ActionAttack_level": r"actionattack[_-]?level[_=]?(\d+)",
        "StateTransferAttack_level": r"statetransferattack[_-]?level[_=]?(\d+)|statetransfer[_-]?attack[_-]?level[_=]?(\d+)",
        "RewardAttack_level": r"rewardattack[_-]?level[_=]?(\d+)",
        "ExperiencePoolAttack_level": r"experiencepoolattack[_-]?level[_=]?(\d+)",
        "ModelTampAttack_level": r"modeltampattack[_-]?level[_=]?(\d+)|modeltamperattack[_-]?level[_=]?(\d+)",
    }
    for key, pattern in attack_patterns.items():
        match = re.search(pattern, lower_path)
        if not match:
            continue
        level = next((group for group in match.groups() if group is not None), None)
        if level is not None:
            env_mapping[key] = int(level)

    return env_mapping


def _split_step_blocks(content: str) -> List[Tuple[int, str]]:
    matches = list(STEP_HEADER_RE.finditer(content))
    if not matches:
        return [(0, content)]

    blocks: List[Tuple[int, str]] = []
    for idx, match in enumerate(matches):
        step_index = int(match.group(1))
        block_start = match.end()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        blocks.append((step_index, content[block_start:block_end]))
    return blocks


def _extract_metrics(block: str) -> Optional[PerformanceMetrics]:
    packet_loss_rate = _extract_float(r"PacketLossRate:\s*([-\d\.]+)%", block, scale=0.01)
    network_throughput = _extract_float(r"NetworkThroughput:\s*([-\d\.]+)", block)
    bandwidth_utilization = _extract_float(r"BandwidthUtilization:\s*([-\d\.]+)%", block, scale=0.01)
    avg_packet_node_visits = _extract_float(r"AvgPacketNodeVisits:\s*([-\d\.]+)", block)
    cumulative_reward = _extract_float(r"CumulativeReward:\s*([-\d\.]+)", block)
    average_inference_time = _extract_float(r"AverageInferenceTime:\s*([-\d\.]+)", block)
    average_e2e_delay = _extract_float(r"AverageE2eDelay[^:]*:\s*([-\d\.]+)", block)
    average_hop_count = _extract_float(r"AverageHopCount[^:]*:\s*([-\d\.]+)", block)
    average_computing_ratio = _extract_float(
        r"AverageComputingRatio:\s*([-\d\.]+)%|Proportion of satellites in computation:\s*([-\d\.]+)%",
        block,
        scale=0.01,
    )
    computing_waiting_time = _extract_float(
        r"ComputingWaitingTime:\s*([-\d\.]+)|Average waiting time for computing:\s*([-\d\.]+)",
        block,
    )
    average_ending_reward = _extract_float(r"AverageEndingReward:\s*([-\d\.]+)|Average ending reward:\s*([-\d\.]+)", block)

    metric_values = [
        packet_loss_rate,
        network_throughput,
        bandwidth_utilization,
        avg_packet_node_visits,
        cumulative_reward,
        average_inference_time,
        average_e2e_delay,
        average_hop_count,
        average_computing_ratio,
        computing_waiting_time,
        average_ending_reward,
    ]
    if all(value is None for value in metric_values):
        return None

    return PerformanceMetrics(
        PacketLossRate=float(packet_loss_rate or 0.0),
        NetworkThroughput=float(network_throughput or 0.0),
        BandwidthUtilization=float(bandwidth_utilization or 0.0),
        AvgPacketNodeVisits=float(avg_packet_node_visits or 0.0),
        CumulativeReward=float(cumulative_reward or 0.0),
        AverageInferenceTime=float(average_inference_time or 0.0),
        AverageE2eDelay=float(average_e2e_delay or 0.0),
        AverageHopCount=float(average_hop_count or 0.0),
        AverageComputingRatio=float(average_computing_ratio or 0.0),
        ComputingWaitingTime=float(computing_waiting_time or 0.0),
        AverageEndingReward=float(average_ending_reward or 0.0),
    )


def _extract_float(pattern: str, text: str, scale: float = 1.0) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    value = next((group for group in match.groups() if group is not None), None)
    if value is None:
        return None
    return float(value) * scale


def _parse_scalar(token: str):
    token = token.strip()
    if not token:
        return token

    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()

    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        numeric = float(token)
    except ValueError:
        return token
    if numeric.is_integer():
        return int(numeric)
    return numeric
