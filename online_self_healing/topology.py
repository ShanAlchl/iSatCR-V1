from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .node import NodeElement, SatelliteNode, normalize_result_diag
from .orbit import SatelliteState, SatelliteTracker


DEFAULT_ISOLATION_RISK_LEVELS = frozenset({"批量中失效风险", "高失效风险"})


class SatelliteGraph:
    def __init__(self):
        pass

    def _distance(self, pos1, pos2):
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def _resolve_next_orbit_satellite(
        self,
        sat_data: SatelliteState,
        next_orbit_number: int,
        max_satellite_number: int,
    ) -> str:
        next_sat_number = (
            sat_data.sat_number
            if next_orbit_number % 2 == 0
            else (sat_data.sat_number - 1 if sat_data.sat_number != 1 else max_satellite_number)
        )
        return (
            f"Satellite_{sat_data.orbit_altitude}_{next_orbit_number}_{next_sat_number}"
        )

    def _require_satellite(self, satellite_name: str, satellite_states: Dict[str, SatelliteState]) -> None:
        if satellite_name not in satellite_states:
            raise KeyError(
                f"Expected fixed-topology satellite '{satellite_name}' was not found in the TLE set"
            )

    def _add_edge_with_positions(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        link_type: str,
    ) -> None:
        graph.add_edge(
            source,
            target,
            pos_a=graph.nodes[source]["pos"],
            pos_b=graph.nodes[target]["pos"],
            link_type=link_type,
        )

    def build_graph_with_fixed_edges(
        self,
        satellite_tracker: SatelliteTracker,
        current_time,
        pole: bool = False,
        satellite_nodes: Optional[Dict[str, SatelliteNode]] = None,
        satellite_states: Optional[Dict[str, SatelliteState]] = None,
    ) -> nx.Graph:
        if satellite_states is None:
            satellite_states = satellite_tracker.generate_satellite_dict(current_time)

        graph = nx.Graph()
        graph.add_nodes_from(satellite_states.keys())

        for sat_name, sat_state in satellite_states.items():
            graph.nodes[sat_name]["pos"] = list(sat_state.eci_position_km)
            graph.nodes[sat_name]["pos_0"] = list(sat_state.eci_position_km)
            graph.nodes[sat_name]["pos_eci_km"] = list(sat_state.eci_position_km)
            graph.nodes[sat_name]["velocity_eci_km_s"] = list(sat_state.eci_velocity_km_s)
            graph.nodes[sat_name]["sequence_num"] = list(sat_state.sequence_num)
            graph.nodes[sat_name]["orbit_altitude"] = sat_state.orbit_altitude
            graph.nodes[sat_name]["orbit_number"] = sat_state.orbit_number
            graph.nodes[sat_name]["sat_number"] = sat_state.sat_number
            graph.nodes[sat_name]["isolated"] = False
            if satellite_nodes and sat_name in satellite_nodes:
                graph.nodes[sat_name].update(satellite_nodes[sat_name].to_graph_attributes())

        max_orbit_number = satellite_tracker.get_max_orbit_number()
        max_satellite_number = satellite_tracker.get_max_satellite_number()

        for sat_name, sat_state in satellite_states.items():
            same_orbit_neighbors = [
                (
                    f"Satellite_{sat_state.orbit_altitude}_{sat_state.orbit_number}_"
                    f"{sat_state.sat_number - 1 if sat_state.sat_number != 1 else max_satellite_number}"
                ),
                (
                    f"Satellite_{sat_state.orbit_altitude}_{sat_state.orbit_number}_"
                    f"{(sat_state.sat_number % max_satellite_number) + 1}"
                ),
            ]
            for neighbor in same_orbit_neighbors:
                if neighbor in satellite_states:
                    self._add_edge_with_positions(graph, sat_name, neighbor, link_type="same_orbit")

        for sat_name, sat_state in satellite_states.items():
            if pole:
                next_orbit_number = sat_state.orbit_number + 1
            else:
                next_orbit_number = (sat_state.orbit_number % max_orbit_number) + 1

            if not pole:
                next_orbit_satellite = self._resolve_next_orbit_satellite(
                    sat_state,
                    next_orbit_number,
                    max_satellite_number,
                )
                self._require_satellite(next_orbit_satellite, satellite_states)
                self._add_edge_with_positions(
                    graph,
                    sat_name,
                    next_orbit_satellite,
                    link_type="cross_orbit",
                )
            elif next_orbit_number <= max_orbit_number:
                next_orbit_satellite = self._resolve_next_orbit_satellite(
                    sat_state,
                    next_orbit_number,
                    max_satellite_number,
                )
                self._require_satellite(next_orbit_satellite, satellite_states)
                if (
                    abs(graph.nodes[sat_name]["pos"][2]) < 6000
                    and abs(graph.nodes[next_orbit_satellite]["pos"][2]) < 6000
                ):
                    self._add_edge_with_positions(
                        graph,
                        sat_name,
                        next_orbit_satellite,
                        link_type="cross_orbit",
                    )

        graph.graph["pole"] = bool(pole)
        return graph

    def derive_isolation_targets(
        self,
        result_diag: Optional[Sequence[NodeElement]],
        isolation_risk_levels: Iterable[str] = DEFAULT_ISOLATION_RISK_LEVELS,
    ) -> Set[str]:
        diagnoses = normalize_result_diag(result_diag)
        levels = set(isolation_risk_levels)
        return {item.SID for item in diagnoses if item.requires_isolation(levels)}

    def apply_diagnosis_metadata(
        self,
        graph: nx.Graph,
        diagnoses: Sequence[NodeElement],
        satellite_nodes: Optional[Dict[str, SatelliteNode]] = None,
    ) -> None:
        for diagnosis in diagnoses:
            if diagnosis.SID not in graph.nodes:
                continue
            graph.nodes[diagnosis.SID]["HealthScore"] = float(diagnosis.HealthScore)
            graph.nodes[diagnosis.SID]["RiskLevel"] = diagnosis.RiskLevel
            graph.nodes[diagnosis.SID]["SDT"] = diagnosis.SDT
            if satellite_nodes and diagnosis.SID in satellite_nodes:
                satellite_nodes[diagnosis.SID].apply_diagnosis(diagnosis)

    def apply_isolation(
        self,
        graph: nx.Graph,
        result_diag: Optional[Sequence[NodeElement]] = None,
        heal_flag: bool = False,
        isolation_targets: Optional[Iterable[str]] = None,
        isolation_risk_levels: Iterable[str] = DEFAULT_ISOLATION_RISK_LEVELS,
        satellite_nodes: Optional[Dict[str, SatelliteNode]] = None,
    ) -> bool:
        diagnoses = normalize_result_diag(result_diag)
        self.apply_diagnosis_metadata(graph, diagnoses, satellite_nodes=satellite_nodes)

        if heal_flag:
            for node_name in graph.nodes:
                graph.nodes[node_name]["isolated"] = False
            graph.graph["removed_edges_by_isolation"] = []
            graph.graph["isolated_nodes"] = []
            graph.graph["unknown_isolation_targets"] = []
            graph.graph["HealFlag"] = True
            graph.graph["IsolationFlag"] = False
            return False

        requested_targets = set(isolation_targets or self.derive_isolation_targets(diagnoses, isolation_risk_levels))
        known_targets = {node for node in requested_targets if node in graph.nodes}
        unknown_targets = sorted(requested_targets - known_targets)

        removed_edges = set()
        for target in known_targets:
            target_edges = list(graph.edges(target))
            removed_edges.update(tuple(sorted(edge)) for edge in target_edges)
            graph.remove_edges_from(target_edges)
            graph.nodes[target]["isolated"] = True

        for node_name in graph.nodes:
            if node_name not in known_targets:
                graph.nodes[node_name]["isolated"] = False

        isolation_flag = bool(requested_targets) and not unknown_targets and all(
            graph.degree(target) == 0 for target in known_targets
        )

        graph.graph["removed_edges_by_isolation"] = sorted(removed_edges)
        graph.graph["isolated_nodes"] = sorted(known_targets)
        graph.graph["unknown_isolation_targets"] = unknown_targets
        graph.graph["HealFlag"] = False
        graph.graph["IsolationFlag"] = isolation_flag
        return isolation_flag
