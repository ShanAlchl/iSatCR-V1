from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import yaml

from constellation_tle_order import (
    CONSTELLATION_TLE_FILENAMES,
    resolve_constellation_tle_path,
)
from .node import (
    DDQNModelConfig,
    NodeElement,
    QNetwork,
    SatelliteNode,
    normalize_result_diag,
)
from .orbit import (
    SatelliteState,
    SatelliteTracker,
    TimeInput,
    ensure_datetime_utc,
    parse_satellite_name,
)
from .topology import DEFAULT_ISOLATION_RISK_LEVELS, SatelliteGraph


DEFAULT_AGENT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "train_NewDDQN_dueling_shuffle.yaml"
)
DEFAULT_TLE_FILEPATH = (
    Path(__file__).resolve().parent
    / "Satellite_Data"
    / CONSTELLATION_TLE_FILENAMES[4]
)

EventInput = Union["SimulationEvent", Mapping[str, Any]]


def _normalize_isolation_list(
    isolation_list: Optional[Iterable[str]],
) -> Optional[Tuple[str, ...]]:
    if isolation_list is None:
        return None
    return tuple(str(item) for item in isolation_list)


def _diagnosis_signature(diagnoses: Sequence[NodeElement]) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                item.SID,
                float(item.HealthScore),
                str(item.RiskLevel),
                None if item.SDT is None else str(item.SDT),
            )
            for item in diagnoses
        )
    )


def _normalize_skip_time(raw_skip_time: Any) -> Tuple[int, int]:
    if raw_skip_time is None:
        return (0, 0)

    if isinstance(raw_skip_time, (list, tuple)):
        values = list(raw_skip_time)
        if len(values) != 2:
            raise ValueError("skip_time must contain exactly two values: minutes and seconds.")
        return (int(values[0]), int(values[1]))

    raise TypeError("skip_time must be a 2-item list or tuple of (minutes, seconds).")


def _normalize_q_network_init_mode(raw_mode: Any) -> str:
    normalized = str(raw_mode or "random").strip().lower().replace("-", "_")
    mode_aliases = {
        "random": "random",
        "rand": "random",
        "file": "file",
        "files": "file",
        "from_file": "file",
        "from_files": "file",
        "load_from_file": "file",
    }
    if normalized not in mode_aliases:
        raise ValueError(
            "q_network_init_mode must be one of: random, file, from_file, from_files."
        )
    return mode_aliases[normalized]


@dataclass
class FrameworkOutput:
    RawGraph: nx.Graph
    IsolationFlag: bool
    BaseGraph: nx.Graph
    SatelliteNodes: Dict[str, SatelliteNode]
    ResultDiag: Sequence[NodeElement]


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    begin_time: datetime
    coarse_time_stride_seconds: float
    duration_intervals: int
    rounds: int
    skip_time: Tuple[int, int]
    source_config_path: str
    raw_general_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def round_duration_seconds(self) -> float:
        # Match the legacy project semantics: YAML `duration` already means
        # wall-clock seconds for one round, not "number of time_stride intervals".
        return max(0.0, float(self.duration_intervals))

    @property
    def configured_total_seconds(self) -> float:
        return self.round_duration_seconds

    @property
    def skip_time_delta(self) -> timedelta:
        minutes, seconds = self.skip_time
        return timedelta(minutes=int(minutes), seconds=int(seconds))

    def resolve_round_start_time(self, base_time: datetime, round_index: int) -> datetime:
        if round_index < 0:
            raise ValueError("round_index must be non-negative.")
        return base_time + (self.skip_time_delta * int(round_index))

    def resolve_total_steps(self, step_seconds: float) -> int:
        if step_seconds <= 0:
            raise ValueError("Simulation time step must be positive.")
        total_seconds = self.configured_total_seconds
        if total_seconds <= 0:
            return 0
        return max(0, int(total_seconds / step_seconds))

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        fallback_stride_seconds: float,
    ) -> "SimulationRuntimeConfig":
        config_path = Path(yaml_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Simulation config YAML not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        general_cfg = dict(config.get("general", {}))
        raw_begin_time = general_cfg.get("begin_time")
        if raw_begin_time is None:
            begin_time = datetime.now(timezone.utc)
        else:
            begin_time = ensure_datetime_utc(raw_begin_time)

        coarse_time_stride_seconds = float(
            general_cfg.get("time_stride", fallback_stride_seconds)
        )
        if coarse_time_stride_seconds <= 0:
            coarse_time_stride_seconds = float(fallback_stride_seconds)
        if coarse_time_stride_seconds <= 0:
            raise ValueError("Configured time stride must be positive.")

        skip_time = _normalize_skip_time(general_cfg.get("skip_time", (0, 0)))

        return cls(
            begin_time=begin_time,
            coarse_time_stride_seconds=coarse_time_stride_seconds,
            duration_intervals=max(0, int(general_cfg.get("duration", 0))),
            rounds=max(1, int(general_cfg.get("rounds", 1))),
            skip_time=skip_time,
            source_config_path=str(config_path),
            raw_general_config=general_cfg,
        )


@dataclass(frozen=True)
class SimulationEvent:
    EventTime: datetime
    ResultDiag: Sequence[NodeElement] = field(default_factory=tuple)
    HealFlag: bool = False
    IsolationList: Optional[Tuple[str, ...]] = None
    EventId: Optional[str] = None
    Metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        event_time: Optional[TimeInput] = None,
        ResultDiag: Optional[Any] = None,
        HealFlag: bool = False,
        IsolationList: Optional[Iterable[str]] = None,
        EventId: Optional[str] = None,
        Metadata: Optional[Mapping[str, Any]] = None,
        default_event_time: Optional[datetime] = None,
    ) -> "SimulationEvent":
        if event_time is None:
            if default_event_time is None:
                raise ValueError("Event time is required.")
            normalized_time = ensure_datetime_utc(default_event_time)
        else:
            normalized_time = ensure_datetime_utc(event_time)

        diagnoses = tuple(normalize_result_diag(ResultDiag))
        return cls(
            EventTime=normalized_time,
            ResultDiag=diagnoses,
            HealFlag=bool(HealFlag),
            IsolationList=_normalize_isolation_list(IsolationList),
            EventId=None if EventId is None else str(EventId),
            Metadata=dict(Metadata or {}),
        )

    @classmethod
    def from_input(
        cls,
        raw: EventInput,
        *,
        default_event_time: datetime,
    ) -> "SimulationEvent":
        if isinstance(raw, cls):
            return raw

        if not isinstance(raw, Mapping):
            raise TypeError("Simulation event must be a SimulationEvent or mapping input.")

        event_time = raw.get(
            "EventTime",
            raw.get("event_time", raw.get("time", raw.get("CurrentTime"))),
        )
        return cls.create(
            event_time=event_time,
            ResultDiag=raw.get("ResultDiag", raw.get("result_diag")),
            HealFlag=bool(raw.get("HealFlag", raw.get("heal_flag", False))),
            IsolationList=raw.get("IsolationList", raw.get("isolation_list")),
            EventId=raw.get("EventId", raw.get("event_id")),
            Metadata=raw.get("Metadata", raw.get("metadata")),
            default_event_time=default_event_time,
        )


@dataclass
class SimulationOutput(FrameworkOutput):
    StepIndex: int
    CurrentTime: datetime
    TriggerReason: str
    StateChanged: bool
    TriggeredEvents: Sequence[SimulationEvent] = field(default_factory=tuple)
    ActiveIsolationList: Sequence[str] = field(default_factory=tuple)
    PendingEventCount: int = 0


class ConstellationFramework:
    def __init__(
        self,
        agent_config_path: Union[str, Path] = DEFAULT_AGENT_CONFIG_PATH,
        tle_filepath: Optional[Union[str, Path]] = None,
        pole: bool = False,
        isolation_risk_levels: Optional[Iterable[str]] = None,
        state_update_period: Optional[float] = None,
        time_step_seconds: Optional[float] = None,
        duration_steps: Optional[int] = None,
        start_time: Optional[TimeInput] = None,
        build_q_networks: bool = False,
        q_network_init_mode: str = "random",
        q_network_weight_dir: Optional[Union[str, Path]] = None,
        output_root_dir: Optional[Union[str, Path]] = None,
        output_constellation_index: Optional[int] = None,
        output_round_index: Optional[int] = None,
        output_test_id: Optional[int] = None,
        device: str = "cpu",
        emit_initial_state: bool = True,
        emit_on_topology_change: bool = True,
    ):
        self.agent_config_path = Path(agent_config_path)
        self.tle_filepath = None if tle_filepath is None else Path(tle_filepath)
        self.ddqn_config = DDQNModelConfig.from_yaml(self.agent_config_path)
        self.runtime_config = SimulationRuntimeConfig.from_yaml(
            self.agent_config_path,
            fallback_stride_seconds=float(self.ddqn_config.state_update_period),
        )

        self.state_update_period = (
            float(state_update_period)
            if state_update_period is not None
            else float(self.ddqn_config.state_update_period)
        )
        self.time_step_seconds = (
            float(time_step_seconds)
            if time_step_seconds is not None
            else float(self.runtime_config.coarse_time_stride_seconds)
        )
        if self.time_step_seconds <= 0:
            raise ValueError("Simulation time step must be positive.")

        self.duration_steps = (
            int(duration_steps)
            if duration_steps is not None
            else self.runtime_config.resolve_total_steps(self.time_step_seconds)
        )
        if self.duration_steps < 0:
            raise ValueError("Simulation duration_steps must be non-negative.")

        self.start_time = ensure_datetime_utc(
            self.runtime_config.begin_time if start_time is None else start_time
        )
        self.time_step = timedelta(seconds=self.time_step_seconds)
        self.event_wait_step_seconds = float(self.state_update_period)
        if self.event_wait_step_seconds <= 0:
            raise ValueError("Simulation event wait step must be positive.")
        self.event_wait_step = timedelta(seconds=self.event_wait_step_seconds)

        self.pole = bool(pole)
        self.isolation_risk_levels = set(
            isolation_risk_levels or DEFAULT_ISOLATION_RISK_LEVELS
        )
        self.build_q_networks = bool(build_q_networks)
        self.q_network_init_mode = _normalize_q_network_init_mode(q_network_init_mode)
        self.q_network_weight_dir = (
            None if q_network_weight_dir is None else Path(q_network_weight_dir)
        )
        self.output_root_dir = None if output_root_dir is None else Path(output_root_dir)
        self.output_constellation_index = (
            None if output_constellation_index is None else int(output_constellation_index)
        )
        self.output_round_index = (
            None if output_round_index is None else int(output_round_index)
        )
        self.output_test_id = None if output_test_id is None else int(output_test_id)
        self.device = device
        self.emit_initial_state = bool(emit_initial_state)
        self.emit_on_topology_change = bool(emit_on_topology_change)
        self.graph_builder = SatelliteGraph()

        self._tracker: Optional[SatelliteTracker] = None
        self._tracker_path: Optional[Path] = None

        self.current_time: datetime = self.start_time
        self.step_index = 0
        self.initialized = False
        self.current_snapshot: Optional[FrameworkOutput] = None
        self.latest_output: Optional[SimulationOutput] = None

        self._active_diagnoses: Dict[str, NodeElement] = {}
        self._latched_isolation_targets: Set[str] = set()
        self._event_queue: List[Tuple[datetime, int, SimulationEvent]] = []
        self._event_sequence = 0
        self._last_state_signature: Optional[Tuple[Any, ...]] = None
        self._dynamic_state_time: Optional[datetime] = None
        self._dynamic_satellite_states: Optional[Dict[str, SatelliteState]] = None
        self._next_outer_boundary_time: Optional[datetime] = None
        self._resolved_q_network_weight_dir: Optional[Path] = None
        self._satellite_q_network_cache: Dict[str, QNetwork] = {}
        self._resolve_tle_filepath(tle_filepath)

    @property
    def active_isolation_list(self) -> Tuple[str, ...]:
        return tuple(sorted(self._latched_isolation_targets))

    @property
    def active_result_diag(self) -> Tuple[NodeElement, ...]:
        return tuple(
            sorted(self._active_diagnoses.values(), key=lambda item: item.SID)
        )

    @property
    def pending_event_count(self) -> int:
        return len(self._event_queue)

    @property
    def resolved_q_network_weight_dir(self) -> Optional[Path]:
        if self.q_network_init_mode != "file":
            return None
        try:
            return self._resolve_q_network_weight_dir()
        except FileNotFoundError:
            return None

    @property
    def dynamic_state_time(self) -> datetime:
        if self._dynamic_state_time is None:
            return self.current_time
        return self._dynamic_state_time

    @property
    def next_outer_boundary_time(self) -> datetime:
        if self._next_outer_boundary_time is None:
            return self.current_time + self.time_step
        return self._next_outer_boundary_time

    def reset(
        self,
        *,
        start_time: Optional[TimeInput] = None,
        clear_events: bool = True,
    ) -> None:
        self.current_time = ensure_datetime_utc(
            self.start_time if start_time is None else start_time
        )
        self.step_index = 0
        self.initialized = False
        self.current_snapshot = None
        self.latest_output = None
        self._active_diagnoses.clear()
        self._latched_isolation_targets.clear()
        self._last_state_signature = None
        self._dynamic_state_time = None
        self._dynamic_satellite_states = None
        self._next_outer_boundary_time = None
        if clear_events:
            self._event_queue.clear()
            self._event_sequence = 0

    def _resolve_tle_filepath(
        self,
        tle_filepath: Optional[Union[str, Path]],
    ) -> Path:
        if tle_filepath is not None:
            self.tle_filepath = Path(tle_filepath)
        if self.tle_filepath is None:
            if self.output_constellation_index is not None:
                self.tle_filepath = self._resolve_tle_filepath_from_constellation_config(
                    self.output_constellation_index
                )
            else:
                self.tle_filepath = DEFAULT_TLE_FILEPATH
        return self.tle_filepath

    def _resolve_tle_filepath_from_constellation_config(
        self,
        constellation_config: int,
    ) -> Path:
        satellite_data_dir = Path(__file__).resolve().parent / "Satellite_Data"
        return resolve_constellation_tle_path(satellite_data_dir, int(constellation_config))

    def _get_tracker(
        self,
        tle_filepath: Optional[Union[str, Path]] = None,
    ) -> SatelliteTracker:
        resolved_path = self._resolve_tle_filepath(tle_filepath)
        if self._tracker is None or self._tracker_path != resolved_path:
            self._tracker = SatelliteTracker(resolved_path)
            self._tracker_path = resolved_path
        return self._tracker

    def _refresh_dynamic_state_cache(
        self,
        state_time: Optional[datetime] = None,
        *,
        tle_filepath: Optional[Union[str, Path]] = None,
    ) -> None:
        resolved_state_time = ensure_datetime_utc(
            self.current_time if state_time is None else state_time
        )
        tracker = self._get_tracker(tle_filepath)
        self._dynamic_state_time = resolved_state_time
        self._dynamic_satellite_states = tracker.generate_satellite_dict(resolved_state_time)
        self._next_outer_boundary_time = resolved_state_time + self.time_step

    def _ensure_dynamic_state_cache(
        self,
        *,
        tle_filepath: Optional[Union[str, Path]] = None,
    ) -> None:
        if self._dynamic_satellite_states is None or self._dynamic_state_time is None:
            self._refresh_dynamic_state_cache(self.current_time, tle_filepath=tle_filepath)

    def _iter_local_path_candidates(
        self,
        raw_path: Union[str, Path],
    ) -> Tuple[Path, ...]:
        path = Path(raw_path)
        if path.is_absolute():
            return (path,)

        candidates: List[Path] = []
        for base_dir in (
            Path.cwd(),
            self.agent_config_path.resolve().parent,
            Path(__file__).resolve().parent,
        ):
            candidate = (base_dir / path).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates)

    def _resolve_q_network_weight_dir(self) -> Path:
        if self._resolved_q_network_weight_dir is not None:
            return self._resolved_q_network_weight_dir

        candidate_roots: List[Union[str, Path]] = []
        if self.q_network_weight_dir is not None:
            candidate_roots.append(self.q_network_weight_dir)

        output_model_dir = self._try_resolve_output_model_dir()
        if output_model_dir is not None:
            candidate_roots.append(output_model_dir)

        yaml_weight_dir = self.ddqn_config.raw_agent_config.get("independent_model_dir")
        if yaml_weight_dir:
            candidate_roots.append(str(yaml_weight_dir))

        candidate_roots.append("model_weight")

        candidate_paths: List[Path] = []
        for raw_root in candidate_roots:
            for candidate in self._iter_local_path_candidates(raw_root):
                if candidate not in candidate_paths:
                    candidate_paths.append(candidate)
                if candidate.is_dir():
                    self._resolved_q_network_weight_dir = candidate
                    return candidate

        if self.q_network_init_mode == "file":
            searched = ", ".join(str(path) for path in candidate_paths) or "<none>"
            raise FileNotFoundError(
                "Q-network weight directory not found. "
                "Set q_network_weight_dir explicitly, or configure "
                "agent.independent_model_dir in the YAML. "
                f"Searched: {searched}"
            )

        raise RuntimeError("Q-network weight directory resolution is only valid in file mode.")

    def _try_resolve_output_model_dir(self) -> Optional[Path]:
        output_args = (
            self.output_constellation_index,
            self.output_round_index,
            self.output_test_id,
        )
        if all(value is None for value in output_args):
            return None
        if any(value is None for value in output_args):
            raise ValueError(
                "output_constellation_index, output_round_index, and output_test_id "
                "must be provided together when loading Q-network weights from output."
            )

        output_root = self.output_root_dir or (Path(__file__).resolve().parent / "output")
        searched_paths: List[Path] = []
        for candidate_root in self._iter_local_path_candidates(output_root):
            candidate = (
                candidate_root
                / str(self.output_constellation_index)
                / f"{self.output_round_index}_{self.output_test_id}"
            )
            searched_paths.append(candidate)
            if candidate.is_dir():
                return candidate

        searched_text = ", ".join(str(path) for path in searched_paths) or "<none>"
        raise FileNotFoundError(
            "Output model directory not found. "
            "Expected a directory like "
            f"output/{self.output_constellation_index}/{self.output_round_index}_{self.output_test_id}. "
            f"Searched: {searched_text}"
        )

    def _resolve_satellite_q_network_weight_path(self, satellite_id: str) -> Path:
        weight_dir = self._resolve_q_network_weight_dir()
        weight_path = weight_dir / f"{satellite_id}.pth"
        if not weight_path.exists():
            fallback_match = self._find_fuzzy_satellite_q_network_weight_path(
                weight_dir,
                satellite_id,
            )
            if fallback_match is None:
                raise FileNotFoundError(
                    "Per-satellite Q-network weight file not found for "
                    f"{satellite_id}: {weight_path}. Expected one file named "
                    f"'{satellite_id}.pth'."
                )
            return fallback_match
        return weight_path

    def _find_fuzzy_satellite_q_network_weight_path(
        self,
        weight_dir: Path,
        satellite_id: str,
    ) -> Optional[Path]:
        try:
            _, orbit_number, sat_number = parse_satellite_name(satellite_id)
        except ValueError:
            return None

        matches: List[Path] = []
        for candidate in weight_dir.glob("Satellite_*_*_*.pth"):
            try:
                _, candidate_orbit_number, candidate_sat_number = parse_satellite_name(
                    candidate.stem
                )
            except ValueError:
                continue
            if candidate_orbit_number == orbit_number and candidate_sat_number == sat_number:
                matches.append(candidate)

        if len(matches) == 1:
            return matches[0]
        return None

    def _get_or_create_satellite_q_network(
        self,
        satellite_node: SatelliteNode,
    ) -> QNetwork:
        cached_network = self._satellite_q_network_cache.get(satellite_node.SID)
        if cached_network is not None:
            return cached_network

        if self.q_network_init_mode == "random":
            cached_network = satellite_node.ensure_q_network(device=self.device)
        else:
            weight_path = self._resolve_satellite_q_network_weight_path(
                satellite_node.SID
            )
            cached_network = satellite_node.load_q_network_file(
                weight_path,
                device=self.device,
            )

        self._satellite_q_network_cache[satellite_node.SID] = cached_network
        return cached_network

    def _build_satellite_nodes(
        self,
        satellite_states: Dict[str, SatelliteState],
        diagnoses_by_sid: Dict[str, NodeElement],
    ) -> Dict[str, SatelliteNode]:
        satellite_nodes: Dict[str, SatelliteNode] = {}
        for sat_name, sat_state in satellite_states.items():
            diagnosis = diagnoses_by_sid.get(sat_name)
            satellite_node = SatelliteNode(
                name=sat_name,
                orbit_altitude=sat_state.orbit_altitude,
                orbit_number=sat_state.orbit_number,
                sat_number=sat_state.sat_number,
                ddqn_config=self.ddqn_config,
                StateUpdatePeriod=self.state_update_period,
                HealthScore=diagnosis.HealthScore if diagnosis else 1.0,
                RiskLevel=diagnosis.RiskLevel if diagnosis else "low_risk",
                SDT=diagnosis.SDT if diagnosis else None,
            )
            if self.build_q_networks:
                satellite_node.q_network = self._get_or_create_satellite_q_network(
                    satellite_node
                )
            satellite_nodes[sat_name] = satellite_node
        return satellite_nodes

    def _compute_snapshot(
        self,
        *,
        current_time: datetime,
        diagnoses: Sequence[NodeElement],
        dynamic_state_time: Optional[datetime] = None,
        satellite_states: Optional[Dict[str, SatelliteState]] = None,
        heal_flag: bool = False,
        isolation_targets: Optional[Iterable[str]] = None,
        pole: Optional[bool] = None,
        tle_filepath: Optional[Union[str, Path]] = None,
    ) -> FrameworkOutput:
        tracker = self._get_tracker(tle_filepath)
        diagnoses_by_sid = {item.SID: item for item in diagnoses}
        resolved_dynamic_state_time = ensure_datetime_utc(
            current_time if dynamic_state_time is None else dynamic_state_time
        )
        if satellite_states is None:
            satellite_states = tracker.generate_satellite_dict(resolved_dynamic_state_time)
        satellite_nodes = self._build_satellite_nodes(satellite_states, diagnoses_by_sid)

        use_pole = self.pole if pole is None else bool(pole)
        base_graph = self.graph_builder.build_graph_with_fixed_edges(
            tracker,
            resolved_dynamic_state_time,
            pole=use_pole,
            satellite_nodes=satellite_nodes,
            satellite_states=satellite_states,
        )
        raw_graph = base_graph.copy()
        isolation_flag = self.graph_builder.apply_isolation(
            raw_graph,
            result_diag=diagnoses,
            heal_flag=heal_flag,
            isolation_targets=isolation_targets,
            isolation_risk_levels=self.isolation_risk_levels,
            satellite_nodes=satellite_nodes,
        )

        resolved_tle_path = self._resolve_tle_filepath(tle_filepath)
        raw_graph.graph["TleFilepath"] = str(resolved_tle_path)
        raw_graph.graph["CurrentTimeUTC"] = current_time.isoformat()
        raw_graph.graph["DynamicStateTimeUTC"] = resolved_dynamic_state_time.isoformat()
        raw_graph.graph["IsolationPolicyLevels"] = sorted(self.isolation_risk_levels)
        raw_graph.graph["TimeStepSeconds"] = self.time_step_seconds
        raw_graph.graph["TimeStrideSeconds"] = self.runtime_config.coarse_time_stride_seconds
        raw_graph.graph["StateUpdatePeriodSeconds"] = self.state_update_period
        base_graph.graph["TleFilepath"] = str(resolved_tle_path)
        base_graph.graph["CurrentTimeUTC"] = current_time.isoformat()
        base_graph.graph["DynamicStateTimeUTC"] = resolved_dynamic_state_time.isoformat()
        base_graph.graph["IsolationPolicyLevels"] = sorted(self.isolation_risk_levels)
        base_graph.graph["TimeStepSeconds"] = self.time_step_seconds
        base_graph.graph["TimeStrideSeconds"] = self.runtime_config.coarse_time_stride_seconds
        base_graph.graph["StateUpdatePeriodSeconds"] = self.state_update_period

        return FrameworkOutput(
            RawGraph=raw_graph,
            IsolationFlag=isolation_flag,
            BaseGraph=base_graph,
            SatelliteNodes=satellite_nodes,
            ResultDiag=tuple(diagnoses),
        )

    def build_raw_graph(
        self,
        TleFilepath: Optional[Union[str, Path]] = None,
        CurrentTime: Optional[TimeInput] = None,
        ResultDiag: Optional[Sequence] = None,
        HealFlag: bool = False,
        IsolationList: Optional[Iterable[str]] = None,
        pole: Optional[bool] = None,
    ) -> FrameworkOutput:
        if CurrentTime is None:
            raise ValueError("CurrentTime is required.")
        normalized_time = ensure_datetime_utc(CurrentTime)
        diagnoses = tuple(normalize_result_diag(ResultDiag))
        return self._compute_snapshot(
            current_time=normalized_time,
            diagnoses=diagnoses,
            heal_flag=bool(HealFlag),
            isolation_targets=IsolationList,
            pole=pole,
            tle_filepath=TleFilepath,
        )

    def inject_event(
        self,
        event: Optional[EventInput] = None,
        *,
        event_time: Optional[TimeInput] = None,
        ResultDiag: Optional[Any] = None,
        HealFlag: bool = False,
        IsolationList: Optional[Iterable[str]] = None,
        EventId: Optional[str] = None,
        Metadata: Optional[Mapping[str, Any]] = None,
    ) -> SimulationEvent:
        default_event_time = self.current_time
        if event is None:
            normalized_event = SimulationEvent.create(
                event_time=event_time,
                ResultDiag=ResultDiag,
                HealFlag=HealFlag,
                IsolationList=IsolationList,
                EventId=EventId,
                Metadata=Metadata,
                default_event_time=default_event_time,
            )
        else:
            normalized_event = SimulationEvent.from_input(
                event,
                default_event_time=default_event_time,
            )

        self._event_sequence += 1
        insort(
            self._event_queue,
            (normalized_event.EventTime, self._event_sequence, normalized_event),
        )
        return normalized_event

    def inject_events(self, events: Iterable[EventInput]) -> List[SimulationEvent]:
        return [self.inject_event(event=item) for item in events]

    def initialize(self) -> Optional[SimulationOutput]:
        self._ensure_dynamic_state_cache()
        due_events = self._pop_due_events(self.current_time)
        self._apply_events(due_events)
        output = self._refresh_current_state(
            triggered_events=due_events,
            force_emit=self.emit_initial_state or bool(due_events),
            initial=True,
        )
        self.initialized = True
        return output

    def flush_events(self) -> Optional[SimulationOutput]:
        if not self.initialized:
            return self.initialize()

        due_events = self._pop_due_events(self.current_time)
        if not due_events:
            return None

        self._apply_events(due_events)
        return self._refresh_current_state(
            triggered_events=due_events,
            force_emit=True,
            initial=False,
        )

    def _peek_next_event_time(self) -> Optional[datetime]:
        if not self._event_queue:
            return None
        return self._event_queue[0][0]

    def _advance_inner_clock_until(self, target_time: datetime) -> List[SimulationOutput]:
        if target_time < self.current_time:
            raise ValueError("target_time must be greater than or equal to the current simulation time.")

        outputs: List[SimulationOutput] = []
        while self.current_time < target_time:
            next_time = min(self.current_time + self.event_wait_step, target_time)
            next_event_time = self._peek_next_event_time()
            if (
                next_event_time is not None
                and self.current_time < next_event_time < next_time
            ):
                next_time = next_event_time

            self.current_time = next_time
            due_events = self._pop_due_events(self.current_time)
            if not due_events:
                continue

            self._apply_events(due_events)
            output = self._refresh_current_state(
                triggered_events=due_events,
                force_emit=True,
                initial=False,
            )
            if output is not None:
                outputs.append(output)

        return outputs

    def _advance_one_outer_step(self) -> List[SimulationOutput]:
        self._ensure_dynamic_state_cache()
        boundary_time = self.next_outer_boundary_time
        outputs = self._advance_inner_clock_until(boundary_time)
        self.current_time = boundary_time
        self.step_index += 1
        self._refresh_dynamic_state_cache(boundary_time)

        output = self._refresh_current_state(
            triggered_events=(),
            force_emit=False,
            initial=False,
        )
        if output is not None:
            outputs.append(output)
        return outputs

    def advance(self, steps: int = 1) -> List[SimulationOutput]:
        if steps < 0:
            raise ValueError("Simulation steps must be non-negative.")

        outputs: List[SimulationOutput] = []
        if not self.initialized:
            initial_output = self.initialize()
            if initial_output is not None:
                outputs.append(initial_output)

        for _ in range(steps):
            outputs.extend(self._advance_one_outer_step())

        return outputs

    def step(self) -> List[SimulationOutput]:
        return self.advance(1)

    def run_steps(self, steps: int) -> List[SimulationOutput]:
        return self.advance(steps)

    def run_round(
        self,
        round_index: int = 0,
        *,
        duration_steps: Optional[int] = None,
        clear_events: bool = True,
    ) -> List[SimulationOutput]:
        round_start_time = self.runtime_config.resolve_round_start_time(
            self.start_time,
            round_index,
        )
        self.reset(start_time=round_start_time, clear_events=clear_events)
        total_steps = self.duration_steps if duration_steps is None else int(duration_steps)
        if total_steps < 0:
            raise ValueError("Simulation duration_steps must be non-negative.")
        return self.advance(total_steps)

    def run_configured_rounds(
        self,
        *,
        duration_steps: Optional[int] = None,
        clear_events: bool = True,
    ) -> List[List[SimulationOutput]]:
        return [
            self.run_round(
                round_index=round_index,
                duration_steps=duration_steps,
                clear_events=clear_events,
            )
            for round_index in range(self.runtime_config.rounds)
        ]

    def run(
        self,
        *,
        duration_steps: Optional[int] = None,
        end_time: Optional[TimeInput] = None,
    ) -> List[SimulationOutput]:
        if end_time is not None:
            return self.run_until(end_time)

        total_steps = self.duration_steps if duration_steps is None else int(duration_steps)
        if total_steps < 0:
            raise ValueError("Simulation duration_steps must be non-negative.")
        return self.advance(total_steps)

    def run_until(self, end_time: TimeInput) -> List[SimulationOutput]:
        target_time = ensure_datetime_utc(end_time)
        outputs: List[SimulationOutput] = []
        if not self.initialized:
            initial_output = self.initialize()
            if initial_output is not None:
                outputs.append(initial_output)

        while self.current_time < target_time:
            if target_time < self.next_outer_boundary_time:
                outputs.extend(self._advance_inner_clock_until(target_time))
                break
            outputs.extend(self._advance_one_outer_step())
        return outputs

    def simulate(
        self,
        *,
        duration_steps: Optional[int] = None,
        end_time: Optional[TimeInput] = None,
        events: Optional[Iterable[EventInput]] = None,
        reset: bool = False,
    ) -> List[SimulationOutput]:
        if reset:
            self.reset(clear_events=True)
        if events is not None:
            self.inject_events(events)
        return self.run(duration_steps=duration_steps, end_time=end_time)

    def _pop_due_events(self, up_to_time: datetime) -> List[SimulationEvent]:
        due_events: List[SimulationEvent] = []
        while self._event_queue and self._event_queue[0][0] <= up_to_time:
            _, _, event = self._event_queue.pop(0)
            due_events.append(event)
        return due_events

    def _apply_events(self, events: Sequence[SimulationEvent]) -> None:
        for event in events:
            if event.HealFlag:
                if event.IsolationList is None:
                    self._latched_isolation_targets.clear()
                    self._active_diagnoses.clear()
                else:
                    for sid in event.IsolationList:
                        sid_text = str(sid)
                        self._latched_isolation_targets.discard(sid_text)
                        self._active_diagnoses.pop(sid_text, None)

            for diagnosis in event.ResultDiag:
                self._active_diagnoses[diagnosis.SID] = diagnosis

            if event.HealFlag and not event.ResultDiag and event.IsolationList is None:
                continue

            if event.IsolationList is not None and not event.HealFlag:
                self._latched_isolation_targets.update(str(sid) for sid in event.IsolationList)
                continue

            if event.ResultDiag and not event.HealFlag:
                derived_targets = self.graph_builder.derive_isolation_targets(
                    event.ResultDiag,
                    self.isolation_risk_levels,
                )
                self._latched_isolation_targets.update(derived_targets)

            if event.HealFlag and event.ResultDiag:
                if event.IsolationList is None:
                    derived_targets = self.graph_builder.derive_isolation_targets(
                        event.ResultDiag,
                        self.isolation_risk_levels,
                    )
                    self._latched_isolation_targets.update(derived_targets)
                else:
                    self._latched_isolation_targets.update(
                        str(sid) for sid in event.IsolationList
                    )

    def _refresh_current_state(
        self,
        *,
        triggered_events: Sequence[SimulationEvent],
        force_emit: bool,
        initial: bool,
    ) -> Optional[SimulationOutput]:
        self._ensure_dynamic_state_cache()
        snapshot = self._compute_snapshot(
            current_time=self.current_time,
            diagnoses=self.active_result_diag,
            dynamic_state_time=self.dynamic_state_time,
            satellite_states=self._dynamic_satellite_states,
            heal_flag=False,
            isolation_targets=self.active_isolation_list,
        )
        self.current_snapshot = snapshot

        current_signature = self._build_state_signature(snapshot)
        state_changed = (
            self._last_state_signature is not None
            and current_signature != self._last_state_signature
        )
        self._last_state_signature = current_signature

        should_emit = force_emit or (
            self.emit_on_topology_change and state_changed
        )
        if not should_emit:
            return None

        output = SimulationOutput(
            RawGraph=snapshot.RawGraph,
            IsolationFlag=snapshot.IsolationFlag,
            BaseGraph=snapshot.BaseGraph,
            SatelliteNodes=snapshot.SatelliteNodes,
            ResultDiag=snapshot.ResultDiag,
            StepIndex=self.step_index,
            CurrentTime=self.current_time,
            TriggerReason=self._resolve_trigger_reason(
                initial=initial,
                has_events=bool(triggered_events),
                state_changed=state_changed,
            ),
            StateChanged=state_changed,
            TriggeredEvents=tuple(triggered_events),
            ActiveIsolationList=self.active_isolation_list,
            PendingEventCount=self.pending_event_count,
        )
        self.latest_output = output
        return output

    def _build_state_signature(
        self,
        snapshot: FrameworkOutput,
    ) -> Tuple[Any, ...]:
        edge_signature = tuple(
            sorted(
                (
                    min(source, target),
                    max(source, target),
                    str(attributes.get("link_type", "")),
                )
                for source, target, attributes in snapshot.RawGraph.edges(data=True)
            )
        )
        return (
            edge_signature,
            tuple(snapshot.RawGraph.graph.get("isolated_nodes", ())),
            bool(snapshot.IsolationFlag),
            tuple(snapshot.RawGraph.graph.get("unknown_isolation_targets", ())),
            str(snapshot.RawGraph.graph.get("DynamicStateTimeUTC", "")),
            _diagnosis_signature(snapshot.ResultDiag),
        )

    def _resolve_trigger_reason(
        self,
        *,
        initial: bool,
        has_events: bool,
        state_changed: bool,
    ) -> str:
        parts: List[str] = []
        if initial:
            parts.append("initial")
        if has_events:
            parts.append("event")
        if state_changed:
            parts.append("state_change")
        return "+".join(parts) if parts else "state_change"


def build_constellation_framework(
    TleFilepath: Optional[Union[str, Path]] = None,
    CurrentTime: Optional[TimeInput] = None,
    ResultDiag: Optional[Sequence] = None,
    HealFlag: bool = False,
    IsolationList: Optional[Iterable[str]] = None,
    pole: bool = False,
    agent_config_path: Union[str, Path] = DEFAULT_AGENT_CONFIG_PATH,
    build_q_networks: bool = False,
    q_network_init_mode: str = "random",
    q_network_weight_dir: Optional[Union[str, Path]] = None,
    output_root_dir: Optional[Union[str, Path]] = None,
    output_constellation_index: Optional[int] = None,
    output_round_index: Optional[int] = None,
    output_test_id: Optional[int] = None,
    device: str = "cpu",
) -> FrameworkOutput:
    if CurrentTime is None:
        raise ValueError("CurrentTime is required.")
    framework = ConstellationFramework(
        agent_config_path=agent_config_path,
        tle_filepath=TleFilepath,
        pole=pole,
        build_q_networks=build_q_networks,
        q_network_init_mode=q_network_init_mode,
        q_network_weight_dir=q_network_weight_dir,
        output_root_dir=output_root_dir,
        output_constellation_index=output_constellation_index,
        output_round_index=output_round_index,
        output_test_id=output_test_id,
        device=device,
    )
    return framework.build_raw_graph(
        CurrentTime=CurrentTime,
        ResultDiag=ResultDiag,
        HealFlag=HealFlag,
        IsolationList=IsolationList,
        pole=pole,
    )
