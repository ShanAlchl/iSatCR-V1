from __future__ import annotations

import collections
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch
import yaml
from torch import nn


VALID_RISK_LEVELS = {
    "低失效风险",
    "中失效风险",
    "批量中失效风险",
    "高失效风险",
}


def get_activation(act_type: str):
    normalized = str(act_type or "").replace("_", "").lower()
    if normalized in {"leakyrelu", "leakyrelu()"}:
        return nn.LeakyReLU()
    if normalized in {"relu", "relu()"}:
        return nn.ReLU()
    if normalized in {"prelu", "prelu()"}:
        return nn.PReLU()
    return nn.Identity()


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        activation: str = "LeakyRelu",
        hidden_layers: int = 2,
        dueling: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation)
        self.dueling = bool(dueling)
        self.scale = float(scale)
        self.mid_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(int(hidden_layers))]
        )
        self.mid_acts = nn.ModuleList(
            [get_activation(activation) for _ in range(int(hidden_layers))]
        )

        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation):
        squeeze_output = False
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            squeeze_output = True

        x = self.in_layer(observation)
        x = self.act(x)

        for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
            x = mid_layer(x)
            x = mid_act(x)

        if self.dueling:
            value = self.value_stream(x)
            advantages = self.advantage_stream(x)
            x = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            x = self.out_layer(x)

        if self.scale > 1:
            x = x * self.scale

        if squeeze_output:
            x = x.squeeze(0)
        return x


@dataclass(frozen=True)
class DDQNModelConfig:
    state_dim: int
    hidden_dim: int
    action_dim: int
    activation: str
    hidden_layers: int
    dueling: bool
    shuffle: bool
    buffer_length: int
    state_update_period: float
    source_config_path: str
    raw_agent_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DDQNModelConfig":
        config_path = Path(yaml_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config YAML not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        agent_cfg = dict(config.get("agent", {}))
        env_cfg = dict(config.get("environment", {}))
        state_dim = int(
            agent_cfg.get(
                "state_dim",
                int(agent_cfg.get("neighbors_dim", 0))
                + int(agent_cfg.get("edges_dim", 0))
                + int(agent_cfg.get("distance_dim", 0))
                + int(agent_cfg.get("mission_dim", 0))
                + int(agent_cfg.get("current_dim", 0)),
            )
        )

        return cls(
            state_dim=state_dim,
            hidden_dim=int(agent_cfg["hidden_dim"]),
            action_dim=int(agent_cfg["action_dim"]),
            activation=str(agent_cfg.get("activation", "LeakyRelu")),
            hidden_layers=int(agent_cfg.get("hidden_layers", 2)),
            dueling=bool(agent_cfg.get("dueling", False)),
            shuffle=bool(agent_cfg.get("shuffle", False)),
            buffer_length=int(agent_cfg.get("buffer_length", 100000)),
            state_update_period=float(env_cfg.get("StateUpdatePeriod", 0.1)),
            source_config_path=str(config_path),
            raw_agent_config=agent_cfg,
        )

    def to_network_kwargs(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "activation": self.activation,
            "hidden_layers": self.hidden_layers,
            "dueling": self.dueling,
        }

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeElement:
    SID: str
    HealthScore: float
    RiskLevel: str
    SDT: Optional[str] = None

    @classmethod
    def from_input(cls, raw: Any) -> "NodeElement":
        if isinstance(raw, cls):
            return raw

        if isinstance(raw, Mapping):
            sid = raw.get("SID", raw.get("sid", raw.get("name")))
            health_score = raw.get("HealthScore", raw.get("health_score"))
            risk_level = raw.get("RiskLevel", raw.get("risk_level"))
            sdt = raw.get("SDT", raw.get("sdt"))
            return cls(
                SID=str(sid),
                HealthScore=float(health_score),
                RiskLevel=str(risk_level),
                SDT=None if sdt is None else str(sdt),
            )

        if isinstance(raw, (list, tuple)):
            if len(raw) not in {3, 4}:
                raise ValueError(
                    "Tuple NodeElement must follow (SID, HealthScore, RiskLevel) "
                    "or (SID, HealthScore, RiskLevel, SDT)"
                )
            sid, health_score, risk_level = raw[:3]
            sdt = raw[3] if len(raw) == 4 else None
            return cls(
                SID=str(sid),
                HealthScore=float(health_score),
                RiskLevel=str(risk_level),
                SDT=None if sdt is None else str(sdt),
            )

        sid = getattr(raw, "SID")
        health_score = getattr(raw, "HealthScore")
        risk_level = getattr(raw, "RiskLevel")
        sdt = getattr(raw, "SDT", None)
        return cls(
            SID=str(sid),
            HealthScore=float(health_score),
            RiskLevel=str(risk_level),
            SDT=None if sdt is None else str(sdt),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def requires_isolation(self, isolation_risk_levels: Iterable[str]) -> bool:
        return self.RiskLevel in set(isolation_risk_levels)


def _looks_like_single_tuple_diagnosis(raw: Any) -> bool:
    return (
        isinstance(raw, (list, tuple))
        and len(raw) in {3, 4}
        and isinstance(raw[0], str)
    )


def normalize_result_diag(result_diag: Optional[Any]) -> List[NodeElement]:
    if result_diag is None:
        return []

    if isinstance(result_diag, NodeElement):
        return [result_diag]

    if isinstance(result_diag, Mapping):
        required_fields = {"SID", "HealthScore", "RiskLevel"}
        if required_fields.issubset(set(result_diag.keys())):
            return [NodeElement.from_input(result_diag)]
        return [NodeElement.from_input(value) for value in result_diag.values()]

    if _looks_like_single_tuple_diagnosis(result_diag):
        return [NodeElement.from_input(result_diag)]

    if isinstance(result_diag, Sequence) and not isinstance(result_diag, (str, bytes)):
        return [NodeElement.from_input(item) for item in result_diag]

    return [NodeElement.from_input(result_diag)]


class SatelliteNode:
    def __init__(
        self,
        name: str,
        orbit_altitude: int,
        orbit_number: int,
        sat_number: int,
        ddqn_config: DDQNModelConfig,
        StateUpdatePeriod: Optional[float] = None,
        HealthScore: float = 1.0,
        RiskLevel: str = "低失效风险",
        SDT: Optional[str] = None,
    ):
        self.name = name
        self.SID = name
        self.orbit_altitude = orbit_altitude
        self.orbit_number = orbit_number
        self.sat_number = sat_number
        self.StateUpdatePeriod = (
            float(StateUpdatePeriod)
            if StateUpdatePeriod is not None
            else float(ddqn_config.state_update_period)
        )
        self.HealthScore = float(HealthScore)
        self.RiskLevel = str(RiskLevel)
        self.SDT = SDT
        self.ddqn_config = ddqn_config
        self.QNetworkClass = QNetwork
        self.q_network: Optional[QNetwork] = None
        self.replay_buffer: collections.deque = collections.deque(
            maxlen=int(ddqn_config.buffer_length)
        )
        
        # 冗余自愈快照资源
        self.baseline_q_network_state: Optional[Dict[str, Any]] = None
        self.baseline_replay_buffer: Optional[List[Any]] = None
        self.snapshot_available: bool = False

    def add_experience(self, experience: Any) -> None:
        """
        向经验池添加一条记录。经验通常是 (state, action, reward, next_state, done) 的元组。
        """
        self.replay_buffer.append(experience)

    def apply_diagnosis(self, diagnosis: Optional[NodeElement]) -> None:
        if diagnosis is None:
            return
        self.HealthScore = float(diagnosis.HealthScore)
        self.RiskLevel = str(diagnosis.RiskLevel)
        self.SDT = diagnosis.SDT

    def rollback_to_baseline(self, restore_model: bool = True, restore_pool: bool = True) -> bool:
        """回滚到最初加载的初始快照"""
        if not self.snapshot_available or self.baseline_q_network_state is None:
            return False
        
        # 恢复模型
        if restore_model:
            self.load_q_network_state_dict(self.baseline_q_network_state, is_initial=False)
            print(f"[{self.SID}] 已完成本地冗余回滚 (模型复位至初始状态)。")
        
        # 恢复经验池
        if restore_pool:
            self.replay_buffer.clear()
            if self.baseline_replay_buffer:
                self.replay_buffer.extend(self.baseline_replay_buffer)
            print(f"[{self.SID}] 已完成本地冗余回滚 (经验池复位至初始状态)。")
            
        return True

    def build_q_network(self, device: Union[str, torch.device] = "cpu") -> QNetwork:
        network = self.QNetworkClass(**self.ddqn_config.to_network_kwargs())
        return network.to(device)

    def load_q_network_state_dict(
        self,
        state_dict: Mapping[str, Any],
        device: Union[str, torch.device] = "cpu",
        is_initial: bool = True,
    ) -> QNetwork:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Q-network state_dict must be a mapping.")
        network = self.build_q_network(device=device)
        network.load_state_dict(copy.deepcopy(state_dict))
        self.q_network = network
        
        # 如果是最初加载且尚未建立快照，则将其设为基线
        if is_initial and not self.snapshot_available:
            self.baseline_q_network_state = copy.deepcopy(state_dict)
            self.baseline_replay_buffer = list(self.replay_buffer)
            self.snapshot_available = True
            print(f"[{self.SID}] 初始模型已加载，已自动建立安全基线快照。")
            
        return network

    def load_q_network_file(
        self,
        weight_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> QNetwork:
        resolved_path = Path(weight_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Q-network weight file not found: {resolved_path}")

        raw_payload = torch.load(resolved_path, map_location="cpu")
        if not isinstance(raw_payload, Mapping):
            raise TypeError(
                "Q-network weight file must contain a state_dict mapping or a checkpoint mapping."
            )

        state_dict = None
        for state_key in (
            "state_dict",
            "model_state_dict",
            "q_network_state_dict",
            "network_state_dict",
        ):
            candidate = raw_payload.get(state_key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break

        if state_dict is None:
            state_dict = raw_payload

        return self.load_q_network_state_dict(state_dict, device=device)

    def ensure_q_network(self, device: Union[str, torch.device] = "cpu") -> QNetwork:
        if self.q_network is None:
            self.q_network = self.build_q_network(device=device)
        return self.q_network

    def clone_q_network(
        self,
        source_network: Optional[QNetwork] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> QNetwork:
        network = self.build_q_network(device=device)
        if source_network is not None:
            network.load_state_dict(copy.deepcopy(source_network.state_dict()))
        elif self.q_network is not None:
            network.load_state_dict(copy.deepcopy(self.q_network.state_dict()))
        return network

    def to_graph_attributes(self) -> Dict[str, Any]:
        return {
            "SID": self.SID,
            "name": self.name,
            "orbit_altitude": self.orbit_altitude,
            "orbit_number": self.orbit_number,
            "sat_number": self.sat_number,
            "StateUpdatePeriod": self.StateUpdatePeriod,
            "HealthScore": self.HealthScore,
            "RiskLevel": self.RiskLevel,
            "SDT": self.SDT,
            "ddqn_config": self.ddqn_config.as_dict(),
            "satellite_node": self,
        }
