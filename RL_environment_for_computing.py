from SatelliteNetworkSimulator_Computing import SatelliteNetworkSimulator_OnbardComputing
from SatelliteNetworkSimulation import SatelliteSimulation
from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph
from skyfield.api import load
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import Logger
from Draw_Graph_Quiker import SatelliteVisualizer_geo
import random
import copy
import os
import json
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

class SatelliteEnv(SatelliteSimulation):
    def __init__(self,mode,select_mode,q_net,discount_factor,epsilon,reward_factors,device,MissionPossibility,PoissonRate,PacketGenerationInterval,DomputingDemandFactor,DomputingDemandFactor_2,SizeAfterComputingFactor,SizeAfterComputing_1,
                 begin_time, end_time, time_stride, tle_filepath, SODFilePath, MeanIntervalTime,memory,
                 ComputingAbility, TransmissionRate,DownlinkRate,DownstreamDelays, PacketSizeRange, PacketSizeMean, PacketSizeStd, StateUpdatePeriod, print_cycle,DelCycle, visualize=False,
                 PrintInfo=False, SaveLog=False, ShowDetail=False,DegradedEdgeRatio=0,RandomNodesDel=0,UpdateCycle=1,SaveTrainingData=None, SaveActionLog=True,
                 ElevationAngle=45, pole=False, EdgeBandwidthMeanDecreaseRatio=1.0, EdgeBandwidthDecreaseStd=0.0, EdgeDisconnectRatio=0.0,
                 ExportPositionData=False, PositionDataDir="./Position_Data", PositionDataCacheSize=120):
        self.tracker = SatelliteTracker(tle_filepath)
        self.coordinates = extract_landmarks(SODFilePath)
        self.graph_builder = SatelliteGraph()
        self.begin_time = begin_time
        self.end_time = end_time
        self.time_stride = time_stride
        self.MeanIntervalTime = MeanIntervalTime
        self.visualizer = SatelliteVisualizer_geo(edge_color=False) if visualize else None
        self.logger = Logger(detail=ShowDetail, SaveLog=SaveLog, verbose=PrintInfo)
        self.ts = load.timescale()
        self.TransmissionRate = TransmissionRate
        self.DownstreamDelays=DownstreamDelays
        self.DownlinkRate=DownlinkRate
        self.PacketSizeRange = PacketSizeRange
        self.PacketGenerationInterval = PacketGenerationInterval
        self.PacketSizeMean = PacketSizeMean
        self.PacketSizeStd = PacketSizeStd
        self.StateUpdatePeriod = StateUpdatePeriod
        self.DegradedEdgeRatio = DegradedEdgeRatio
        self.RandomNodesDel = RandomNodesDel
        self.EdgeBandwidthMeanDecreaseRatio = EdgeBandwidthMeanDecreaseRatio
        self.EdgeBandwidthDecreaseStd = EdgeBandwidthDecreaseStd
        self.EdgeDisconnectRatio = EdgeDisconnectRatio
        self.ElevationAngle = ElevationAngle
        self.pole = pole
        self.statics = []
        self.time_acc = 0.0
        self.mode=mode
        self.select_mode=select_mode
        self.q_net=q_net
        self.discount_factor = discount_factor
        self.epsilon=epsilon
        self.reward_factors = reward_factors
        self.device=device
        self.PoissonRate=PoissonRate
        self.DomputingDemandFactor=DomputingDemandFactor
        self.DomputingDemandFactor_2=DomputingDemandFactor_2
        self.SizeAfterComputing_1=SizeAfterComputing_1
        self.SizeAfterComputingFactor=SizeAfterComputingFactor
        self.memory=memory
        self.ComputingAbility=ComputingAbility
        self.MissionPossibility=MissionPossibility

        self.UpdateCycle = UpdateCycle
        self.current_cycle = 0.0
        self.DelCycle = DelCycle
        self.del_update=True
        self.current_DelCycle = 0.0
        self.last_removed_nodes = set()
        self.last_edge_bandwidth_drop_ratios = {}

        self.print_cycle = print_cycle
        self.current_print_cycle = 0.0
        self.iteration_counter = 0
        self.print_cycle_iterations = int(print_cycle / time_stride)
        self.SaveTrainingData=SaveTrainingData
        self.SaveActionLog = SaveActionLog
        self.ExportPositionData = ExportPositionData
        self.PositionDataDir = PositionDataDir
        self.PositionDataCacheSize = max(int(PositionDataCacheSize), 1)

        self.step_num=0
        self.rewards=[]
        self.reward_sequence=[]
        self.action_log_path = os.path.join('.', 'training_process_data', 'ActionLog.txt')
        self.action_log_initialized = False
        self.display_timezone = ZoneInfo("Asia/Shanghai")
        self.position_data_session_id = datetime.now(self.display_timezone).strftime("%Y%m%d-%H%M%S")
        self.position_snapshot_reset_index = 0
        self.position_snapshot_sequence = 0
        self.position_data_file_path = os.path.join(self.PositionDataDir, 'topology_queue.jsonl')

        self.current_graph = None
        self.connections = None
        self.training_topology_time = begin_time

        self.reset(self.begin_time)
        self.current_time = self.begin_time

    def _normalize_edge(self, edge):
        return tuple(sorted(edge))

    def _resolve_edge_selection_count(self, G, edge_ratio):
        if not 0 <= edge_ratio <= 1:
            raise ValueError("DegradedEdgeRatio must be a float ratio in [0, 1]")
        return int(G.number_of_edges() * edge_ratio)

    def _validate_bandwidth_decrease_distribution(self):
        if not 0 <= self.EdgeBandwidthMeanDecreaseRatio <= 1:
            raise ValueError("EdgeBandwidthMeanDecreaseRatio must be in [0, 1]")
        if self.EdgeBandwidthDecreaseStd < 0:
            raise ValueError("EdgeBandwidthDecreaseStd must be >= 0")
        if not 0 <= self.EdgeDisconnectRatio <= 1:
            raise ValueError("EdgeDisconnectRatio must be in [0, 1]")

    def _clip_bandwidth_drop_ratio(self, value):
        return min(max(value, 0.0), 1.0)

    def _enforce_target_mean(self, ratios, target_mean):
        if not ratios:
            return ratios

        target_sum = target_mean * len(ratios)
        ratios = [self._clip_bandwidth_drop_ratio(value) for value in ratios]

        for _ in range(16):
            current_sum = sum(ratios)
            delta = target_sum - current_sum
            if abs(delta) < 1e-10:
                break

            if delta > 0:
                adjustable_indices = [i for i, value in enumerate(ratios) if value < 1.0 - 1e-12]
                total_capacity = sum(1.0 - ratios[i] for i in adjustable_indices)
                if not adjustable_indices or total_capacity <= 1e-12:
                    break
                scale = min(1.0, delta / total_capacity)
                for i in adjustable_indices:
                    ratios[i] += (1.0 - ratios[i]) * scale
            else:
                adjustable_indices = [i for i, value in enumerate(ratios) if value > 1e-12]
                total_capacity = sum(ratios[i] for i in adjustable_indices)
                if not adjustable_indices or total_capacity <= 1e-12:
                    break
                scale = min(1.0, (-delta) / total_capacity)
                for i in adjustable_indices:
                    ratios[i] -= ratios[i] * scale

            ratios = [self._clip_bandwidth_drop_ratio(value) for value in ratios]

        return ratios

    def _sample_edge_bandwidth_drop_ratios(self, edge_count, full_disconnect_count=0):
        self._validate_bandwidth_decrease_distribution()
        if edge_count <= 0:
            return []

        target_mean = self.EdgeBandwidthMeanDecreaseRatio
        target_std = self.EdgeBandwidthDecreaseStd
        if full_disconnect_count < 0 or full_disconnect_count > edge_count:
            raise ValueError("full_disconnect_count must be in [0, edge_count]")

        if full_disconnect_count == edge_count:
            if abs(target_mean - 1.0) > 1e-10:
                raise ValueError(
                    "EdgeBandwidthMeanDecreaseRatio must be 1.0 when all selected edges are fully disconnected"
                )
            return [1.0] * edge_count

        remaining_count = edge_count - full_disconnect_count
        remaining_target_mean = (target_mean * edge_count - full_disconnect_count) / remaining_count
        if not 0 <= remaining_target_mean <= 1:
            raise ValueError(
                "Current EdgeDisconnectRatio is incompatible with EdgeBandwidthMeanDecreaseRatio"
            )

        if remaining_count == 1 or target_std == 0:
            remaining_ratios = [remaining_target_mean] * remaining_count
        else:
            raw_ratios = [random.gauss(remaining_target_mean, target_std) for _ in range(remaining_count)]
            clipped_ratios = [self._clip_bandwidth_drop_ratio(value) for value in raw_ratios]
            remaining_ratios = self._enforce_target_mean(clipped_ratios, remaining_target_mean)

        return [1.0] * full_disconnect_count + remaining_ratios

    def apply_random_edge_bandwidth_changes(self, G, edge_ratio, update=False):
        selected_edge_count = self._resolve_edge_selection_count(G, edge_ratio)
        if update:
            selected_edges = random.sample(list(G.edges()), selected_edge_count)
            full_disconnect_count = int(selected_edge_count * self.EdgeDisconnectRatio)
            disconnected_edges = set(random.sample(selected_edges, full_disconnect_count))
            remaining_edges = [edge for edge in selected_edges if edge not in disconnected_edges]
            sampled_drop_ratios = self._sample_edge_bandwidth_drop_ratios(selected_edge_count, full_disconnect_count)
            self.last_edge_bandwidth_drop_ratios = {
                self._normalize_edge(edge): drop_ratio
                for edge, drop_ratio in zip(list(disconnected_edges) + remaining_edges, sampled_drop_ratios)
            }

        for node_a, node_b in G.edges():
            edge_key = self._normalize_edge((node_a, node_b))
            drop_ratio = self.last_edge_bandwidth_drop_ratios.get(edge_key, 0.0)
            G[node_a][node_b]['bandwidth_drop_ratio'] = drop_ratio
            G[node_a][node_b]['base_TransmissionRate'] = self.TransmissionRate
            G[node_a][node_b]['link_TransmissionRate'] = self.TransmissionRate * (1.0 - drop_ratio)

        return G

    def remove_random_nodes(self,G, n,update=False):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")
        if update:
            self.last_removed_nodes = random.sample(list(G.nodes()), n)
        G.remove_nodes_from(self.last_removed_nodes)

        return G

    def step(self,epsilon):
        self.step_num+=1

        self.current_DelCycle+=self.time_stride
        if self.current_DelCycle >= self.DelCycle:
            self.current_DelCycle += -self.DelCycle
            self.del_update=True
        self.current_cycle+=self.time_stride
        if self.current_cycle >= self.UpdateCycle:
            self.current_cycle += -self.UpdateCycle
            coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(self.current_time))
            connections = get_connections_h3(self.coordinates, coordinates_s, self.ElevationAngle)
            old_nodes = set(self.simulator.graph.nodes())
            current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(self.current_time),pole=False)

            self.remove_random_nodes(current_graph, self.RandomNodesDel,self.del_update)
            self.apply_random_edge_bandwidth_changes(current_graph, self.DegradedEdgeRatio, self.del_update)

            self.del_update = False
            new_nodes = set(current_graph.nodes())
            lost_nodes = old_nodes - new_nodes
            for landmark, satellites in connections.items():
                for lost_node in lost_nodes:
                    if lost_node in satellites:
                        connections[landmark].remove(lost_node)
            self.current_graph=current_graph
            self.connections=connections
            self.training_topology_time = self.current_time

        self.simulator.upgrade_all(self.current_graph, self.connections)
        for satellite in self.simulator.satellites:
            self.simulator.satellites[satellite].epsilon=epsilon

        self.simulator.run(self.time_stride)
        self._advance_current_time()
        self.export_position_snapshot()
        experiences=self.simulator.propagator.experiences
        action_logs = self.simulator.propagator.action_logs
        self.simulator.propagator.experiences = []
        self.simulator.propagator.action_logs = []
        self.append_action_log(action_logs)
        self.reward_sequence.extend(experience[3] for experience in experiences)
        self.rewards.extend(self.simulator.propagator.final_rewards)
        self.iteration_counter += 1
        if self.iteration_counter >= self.print_cycle_iterations:
            self.iteration_counter = 0
            self.print_and_save_accumulated_data()
            self.current_print_cycle = 0.0
            self.rewards=[]
            self.reward_sequence=[]

        return experiences

    def reset(self,begin_time):
        self.statics= []
        self.begin_time=begin_time
        self.initialize_action_log(begin_time)
        self.position_snapshot_reset_index += 1
        self.position_snapshot_sequence = 0
        init_time = self.time_from_str(begin_time)
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, init_time, pole=self.pole)
        self.num_nodes = len(current_graph.nodes())
        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        connections = get_connections_h3(self.coordinates, coordinates_s, self.ElevationAngle)
        self.simulator = SatelliteNetworkSimulator_OnbardComputing(
            mode=self.mode,
            select_mode=self.select_mode,
            q_net=self.q_net,
            reward_factors = self.reward_factors,
            epsilon= self.epsilon,
            device= self.device,
            MissionPossibility=self.MissionPossibility,
            PoissonRate= self.PoissonRate,
            PacketGenerationInterval=self.PacketGenerationInterval,
            DomputingDemandFactor= self.DomputingDemandFactor,
            DomputingDemandFactor_2=self.DomputingDemandFactor_2,
            SizeAfterComputingFactor= self.SizeAfterComputingFactor,
            SizeAfterComputing_1=self.SizeAfterComputing_1,
            graph=current_graph,
            landmarks=connections,
            MeanIntervalTime=self.MeanIntervalTime,
            memory=self.memory,
            ComputingAbility=self.ComputingAbility,
            TransmissionRate=self.TransmissionRate,
            DownstreamDelays=self.DownstreamDelays,
            DownlinkRate=self.DownlinkRate,
            PacketSizeRange=self.PacketSizeRange,
            PacketSizeMean=self.PacketSizeMean,
            PacketSizeStd=self.PacketSizeStd,
            StateUpdatePeriod=self.StateUpdatePeriod,
            logger=self.logger)
        self.current_time = self.begin_time
        self.time_acc = 0.0
        self.current_cycle =0.0
        self.current_print_cycle=0.0
        self.current_DelCycle = 0.0
        self.iteration_counter = 0
        self.del_update = True
        self.rewards = []
        self.reward_sequence = []
        self.training_topology_time = self.current_time
        old_nodes = set(self.simulator.graph.nodes())
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker,self.time_from_str(self.current_time), pole=False)

        self.remove_random_nodes(current_graph, self.RandomNodesDel, True)
        self.apply_random_edge_bandwidth_changes(current_graph, self.DegradedEdgeRatio, True)
        new_nodes = set(current_graph.nodes())
        lost_nodes = old_nodes - new_nodes
        for landmark, satellites in connections.items():
            for lost_node in lost_nodes:
                if lost_node in satellites:
                    connections[landmark].remove(lost_node)
        self.current_graph = current_graph
        self.connections = connections
        self.export_position_snapshot()

    def _advance_current_time(self):
        self.time_acc += self.time_stride
        if self.time_acc >= 1.0:
            elapsed_seconds = int(self.time_acc)
            self.current_time = self.add_time_to_str(self.current_time, (0, elapsed_seconds))
            self.time_acc -= elapsed_seconds

    def _get_display_time_str(self):
        return datetime.now(self.display_timezone).strftime("%Y-%m-%d %H:%M:%S")

    def _serialize_snapshot_value(self, value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        if isinstance(value, tuple):
            return [self._serialize_snapshot_value(item) for item in value]
        if isinstance(value, list):
            return [self._serialize_snapshot_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize_snapshot_value(item) for key, item in value.items()}
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _build_satellite_link_activity(self):
        activity_by_edge = {}
        if getattr(self, 'simulator', None) is None:
            return activity_by_edge

        for source_name in sorted(self.simulator.satellites):
            satellite = self.simulator.satellites[source_name]
            transmission_length = getattr(satellite, 'transmission_length', {}) or {}
            transmission_size = getattr(satellite, 'transmission_size', {}) or {}
            active_count = getattr(satellite, 'active_transmission_count', {}) or {}
            active_size = getattr(satellite, 'active_transmission_size', {}) or {}
            neighbor_names = sorted(
                set(transmission_length)
                | set(transmission_size)
                | set(active_count)
                | set(active_size)
            )

            for target_name in neighbor_names:
                edge_key = self._normalize_edge((source_name, target_name))
                queued_packets = int(transmission_length.get(target_name, 0) or 0)
                queued_bytes = int(transmission_size.get(target_name, 0) or 0)
                in_flight_packets = int(active_count.get(target_name, 0) or 0)
                in_flight_bytes = int(active_size.get(target_name, 0) or 0)
                edge_activity = activity_by_edge.setdefault(edge_key, {
                    "source": edge_key[0],
                    "target": edge_key[1],
                    "queued_packets": 0,
                    "queued_bytes": 0,
                    "in_flight_packets": 0,
                    "in_flight_bytes": 0,
                    "active_packet_count": 0,
                    "is_active": False,
                    "directional_details": [],
                })
                edge_activity["queued_packets"] += queued_packets
                edge_activity["queued_bytes"] += queued_bytes
                edge_activity["in_flight_packets"] += in_flight_packets
                edge_activity["in_flight_bytes"] += in_flight_bytes
                if queued_packets or queued_bytes or in_flight_packets or in_flight_bytes:
                    edge_activity["directional_details"].append({
                        "source": source_name,
                        "target": target_name,
                        "queued_packets": queued_packets,
                        "queued_bytes": queued_bytes,
                        "in_flight_packets": in_flight_packets,
                        "in_flight_bytes": in_flight_bytes,
                    })

        for edge_activity in activity_by_edge.values():
            edge_activity["active_packet_count"] = edge_activity["queued_packets"] + edge_activity["in_flight_packets"]
            edge_activity["is_active"] = edge_activity["active_packet_count"] > 0

        return activity_by_edge

    def _build_ground_link_activity(self):
        activity_by_link = {}
        if getattr(self, 'simulator', None) is None:
            return activity_by_link

        for satellite_name in sorted(self.simulator.satellites):
            satellite = self.simulator.satellites[satellite_name]
            queued_ground_packets = {}
            queued_ground_bytes = {}
            offload_queue = getattr(satellite, 'offload_queue', None)
            if offload_queue is not None:
                for packet in list(getattr(offload_queue, 'items', [])):
                    ground_station = getattr(packet, 'ground_station', None)
                    if not ground_station:
                        continue
                    queued_ground_packets[ground_station] = queued_ground_packets.get(ground_station, 0) + 1
                    queued_ground_bytes[ground_station] = queued_ground_bytes.get(ground_station, 0) + int(getattr(packet, 'size', 0) or 0)

            active_ground_packets = getattr(satellite, 'active_ground_transmission_count', {}) or {}
            active_ground_bytes = getattr(satellite, 'active_ground_transmission_size', {}) or {}
            ground_station_names = sorted(
                set(queued_ground_packets)
                | set(queued_ground_bytes)
                | set(active_ground_packets)
                | set(active_ground_bytes)
            )

            for ground_station in ground_station_names:
                link_key = (ground_station, satellite_name)
                queued_packets = int(queued_ground_packets.get(ground_station, 0) or 0)
                queued_bytes = int(queued_ground_bytes.get(ground_station, 0) or 0)
                in_flight_packets = int(active_ground_packets.get(ground_station, 0) or 0)
                in_flight_bytes = int(active_ground_bytes.get(ground_station, 0) or 0)
                link_activity = activity_by_link.setdefault(link_key, {
                    "ground_station": ground_station,
                    "satellite": satellite_name,
                    "queued_packets": 0,
                    "queued_bytes": 0,
                    "in_flight_packets": 0,
                    "in_flight_bytes": 0,
                    "active_packet_count": 0,
                    "is_active": False,
                })
                link_activity["queued_packets"] += queued_packets
                link_activity["queued_bytes"] += queued_bytes
                link_activity["in_flight_packets"] += in_flight_packets
                link_activity["in_flight_bytes"] += in_flight_bytes

        for link_activity in activity_by_link.values():
            link_activity["active_packet_count"] = link_activity["queued_packets"] + link_activity["in_flight_packets"]
            link_activity["is_active"] = link_activity["active_packet_count"] > 0

        return activity_by_link

    def _build_position_snapshot(self):
        snapshot_time = self.current_time
        snapshot_ts = self.time_from_str(snapshot_time)
        display_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, snapshot_ts, pole=self.pole)
        satellite_lla = self.tracker.generate_satellite_LLA_dict(snapshot_ts)
        ground_connections = get_connections_h3(self.coordinates, satellite_lla, self.ElevationAngle)
        normalized_ground_connections = {
            ground_name: sorted(ground_connections.get(ground_name, []))
            for ground_name in sorted(self.coordinates)
        }
        training_nodes = set(self.current_graph.nodes()) if self.current_graph is not None else set()
        satellite_link_activity = self._build_satellite_link_activity()
        ground_link_activity = self._build_ground_link_activity()

        nodes = []
        for node_name in sorted(display_graph.nodes()):
            node_data = display_graph.nodes[node_name]
            lla = satellite_lla.get(node_name, {})
            nodes.append({
                "id": node_name,
                "position_eci_km": self._serialize_snapshot_value(node_data.get('pos')),
                "position_lla": {
                    "latitude": lla.get("latitude", node_data.get('pos_0', [None, None, None])[0]),
                    "longitude": lla.get("longitude", node_data.get('pos_0', [None, None, None])[1]),
                    "altitude_km": lla.get("altitude", node_data.get('pos_0', [None, None, None])[2]),
                },
                "sequence_num": self._serialize_snapshot_value(node_data.get('sequence_num')),
                "in_training_topology": node_name in training_nodes,
            })

        satellite_links = []
        for node_a, node_b, edge_data in sorted(display_graph.edges(data=True)):
            edge_activity = satellite_link_activity.get(self._normalize_edge((node_a, node_b)), {})
            satellite_links.append({
                "source": node_a,
                "target": node_b,
                "bandwidth_drop_ratio": self._serialize_snapshot_value(edge_data.get('bandwidth_drop_ratio', 0.0)),
                "link_transmission_rate": self._serialize_snapshot_value(edge_data.get('link_TransmissionRate', self.TransmissionRate)),
                "queued_packets": int(edge_activity.get("queued_packets", 0) or 0),
                "queued_bytes": int(edge_activity.get("queued_bytes", 0) or 0),
                "in_flight_packets": int(edge_activity.get("in_flight_packets", 0) or 0),
                "in_flight_bytes": int(edge_activity.get("in_flight_bytes", 0) or 0),
                "active_packet_count": int(edge_activity.get("active_packet_count", 0) or 0),
                "is_active": bool(edge_activity.get("is_active", False)),
                "directional_details": self._serialize_snapshot_value(edge_activity.get("directional_details", [])),
            })

        ground_stations = []
        ground_links = []
        for ground_name in sorted(self.coordinates):
            ground_info = self.coordinates[ground_name]
            ground_stations.append({
                "id": ground_name,
                "latitude": ground_info["latitude"],
                "longitude": ground_info["longitude"],
                "altitude": ground_info.get("altitude", 0),
            })
            for satellite_name in normalized_ground_connections[ground_name]:
                link_activity = ground_link_activity.get((ground_name, satellite_name), {})
                ground_links.append({
                    "ground_station": ground_name,
                    "satellite": satellite_name,
                    "queued_packets": int(link_activity.get("queued_packets", 0) or 0),
                    "queued_bytes": int(link_activity.get("queued_bytes", 0) or 0),
                    "in_flight_packets": int(link_activity.get("in_flight_packets", 0) or 0),
                    "in_flight_bytes": int(link_activity.get("in_flight_bytes", 0) or 0),
                    "active_packet_count": int(link_activity.get("active_packet_count", 0) or 0),
                    "is_active": bool(link_activity.get("is_active", False)),
                })

        active_satellite_links = [
            self._serialize_snapshot_value(activity)
            for activity in satellite_link_activity.values()
            if activity.get("is_active")
        ]
        active_ground_links = [
            self._serialize_snapshot_value(activity)
            for activity in ground_link_activity.values()
            if activity.get("is_active")
        ]

        return {
            "session_id": self.position_data_session_id,
            "reset_index": self.position_snapshot_reset_index,
            "snapshot_sequence": self.position_snapshot_sequence,
            "step_num": self.step_num,
            "simulation_time": snapshot_time,
            "wall_time": self._get_display_time_str(),
            "training_topology": {
                "time": self.training_topology_time,
                "update_cycle_steps": self.UpdateCycle,
                "node_count": len(training_nodes),
                "edge_count": self.current_graph.number_of_edges() if self.current_graph is not None else 0,
            },
            "display_topology": {
                "time": snapshot_time,
                "node_count": display_graph.number_of_nodes(),
                "edge_count": display_graph.number_of_edges(),
            },
            "ground_stations": ground_stations,
            "ground_connections": normalized_ground_connections,
            "nodes": nodes,
            "satellite_links": satellite_links,
            "ground_links": ground_links,
            "active_satellite_links": active_satellite_links,
            "active_ground_links": active_ground_links,
        }

    def _write_json_atomic(self, file_path, payload):
        self._ensure_parent_dir(file_path)
        temp_path = f"{file_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as file:
            if isinstance(payload, str):
                file.write(payload)
            else:
                json.dump(payload, file, ensure_ascii=False)
        last_error = None
        for _ in range(40):
            try:
                os.replace(temp_path, file_path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05)

        try:
            os.remove(temp_path)
        except OSError:
            pass

        if last_error is not None:
            raise last_error

    def _append_position_snapshot_line(self, snapshot_payload):
        self._ensure_parent_dir(self.position_data_file_path)
        serialized_snapshot = json.dumps(snapshot_payload, ensure_ascii=False)
        existing_lines = deque(maxlen=self.PositionDataCacheSize)
        if os.path.exists(self.position_data_file_path):
            with open(self.position_data_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped = line.rstrip('\n')
                    if stripped:
                        existing_lines.append(stripped)
        existing_lines.append(serialized_snapshot)
        payload = '\n'.join(existing_lines)
        if payload:
            payload += '\n'
        self._write_json_atomic(self.position_data_file_path, payload)

    def export_position_snapshot(self):
        if not self.ExportPositionData:
            return

        snapshot_payload = self._build_position_snapshot()
        self._append_position_snapshot_line(snapshot_payload)
        self.position_snapshot_sequence += 1

    def render(self):
        self.visualize(self.simulator.graph, self.current_time, self.simulator)

    def visualize(self,current_graph,current_time,simulator):
        system_state=simulator.get_system_state()

        G_draw=current_graph.copy()
        self.update_node_colors(G_draw, system_state)

        for landmark, pos_value in self.coordinates.items():
            G_draw.add_node(landmark)
            G_draw.nodes[landmark]['pos_0'] = [pos_value["latitude"],pos_value ["longitude"], pos_value["altitude"]]
            G_draw.nodes[landmark]['color'] = 'purple'
        self.visualizer.draw_graph(G_draw)

    def show_satellite_computing_time(self):
        satellite_computing_times={}
        for satellite in self.simulator.satellites:
            satellite_computing_times[satellite]=self.simulator.satellites[satellite].computing_time
        self.print_and_save(str(satellite_computing_times))

    # def print_and_save(self, message):
    #     print(message)
    #     if self.SaveTrainingData:
    #         file_path = os.path.join('./training_process_data', self.SaveTrainingData)
    #         with open(file_path, 'a') as file:
    #             file.write(message + '\n')

    def print_and_save(self, message):
        print(message)
        if self.SaveTrainingData:
            # 构造保存文件路径（例如目录 `./training_process_data` + 文件名）
            file_path = os.path.join('.', 'training_process_data', self.SaveTrainingData)
            # 确保目录存在
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            # 追加写入，指定编码以避免平台差异问题
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(message + '\n')

    def _ensure_parent_dir(self, file_path):
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def _serialize_action_log_value(self, value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        if isinstance(value, tuple):
            return [self._serialize_action_log_value(item) for item in value]
        if isinstance(value, list):
            return [self._serialize_action_log_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize_action_log_value(item) for key, item in value.items()}
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def initialize_action_log(self, begin_time):
        if not self.SaveActionLog:
            return
        self._ensure_parent_dir(self.action_log_path)
        mode = 'w' if not self.action_log_initialized else 'a'
        with open(self.action_log_path, mode, encoding='utf-8') as file:
            file.write(f"====== reset {begin_time} ======\n")
        self.action_log_initialized = True

    def append_action_log(self, action_logs):
        if not self.SaveActionLog:
            return
        if not action_logs:
            return

        grouped_logs = {}
        for action_log in action_logs:
            grouped_logs.setdefault(action_log['agent_name'], []).append(action_log)

        self._ensure_parent_dir(self.action_log_path)
        with open(self.action_log_path, 'a', encoding='utf-8') as file:
            file.write(
                f"====== step {self.step_num} | wall_time {self._get_display_time_str()} | experience_count {len(action_logs)} | agent_count {len(grouped_logs)} ======\n"
            )
            global_index = 1
            for agent_name in sorted(grouped_logs):
                agent_logs = grouped_logs[agent_name]
                file.write(f"---- agent {agent_name} | transition_count {len(agent_logs)} ----\n")
                for agent_index, action_log in enumerate(agent_logs, start=1):
                    record = {
                        'tuple_index': global_index,
                        'agent_transition_index': agent_index,
                        'agent_name': agent_name,
                        'event_type': action_log['event_type'],
                        'mdp_tuple': {
                            'state': self._serialize_action_log_value(action_log['state']),
                            'action': self._serialize_action_log_value(action_log['action']),
                            'reward': self._serialize_action_log_value(action_log['reward']),
                            'next_state': self._serialize_action_log_value(action_log['next_state']),
                        },
                        'mark': self._serialize_action_log_value(action_log['mark']),
                        'done': self._serialize_action_log_value(action_log['done']),
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + '\n')
                    global_index += 1

    def print_and_save_accumulated_data(self):
        self.print_and_save(f"====== step {self.step_num} ======")
        self.print_and_save(f"====== {self._get_display_time_str()} ======")
        current_statics_snapshot = copy.deepcopy(self.simulator.statics_datas)
        if self.statics:
            current_statics = {k: current_statics_snapshot[k] - self.statics[-1][k] for k in current_statics_snapshot}
        else:
            current_statics = current_statics_snapshot
        self.statics.append(current_statics_snapshot)

        d = current_statics
        packet_loss_rates = (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload']) / (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1']) if d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1'] > 0 else None
        average_delays = (d['Total_delay_0'] + d['Total_delay_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_hops = (d['Total_hops_0'] + d['Total_hops_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_computing_ratio = d['Is_computing'] / self.num_nodes / (self.print_cycle_iterations)
        average_computing_waiting_time = (d['Computing_waiting_time']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        network_throughput = d['NetworkThroughput']
        isl_traffic_volume = d['ISLTrafficVolume']
        isl_bandwidth_capacity = d['ISLBandwidthCapacity']
        throughput_window_seconds = self.print_cycle_iterations * self.time_stride
        network_throughput_mbps = network_throughput / throughput_window_seconds / 1e6 if throughput_window_seconds > 0 else None
        bandwidth_utilization = isl_traffic_volume / isl_bandwidth_capacity if isl_bandwidth_capacity > 0 else None
        average_inference_time_ms = d['InferenceTimeTotalMs'] / d['InferenceCallCount'] if d['InferenceCallCount'] > 0 else None
        avg_packet_node_visits = d['PacketNodeVisits'] / d['Total'] if d['Total'] > 0 else None
        cumulative_reward = None
        if self.reward_sequence:
            cumulative_reward = 0.0
            for reward in reversed(self.reward_sequence):
                cumulative_reward = reward + self.discount_factor * cumulative_reward

        #self.print_and_save(f"current_statics: {current_statics}")
        self.print_and_save(f"PacketLossRate: {'{:.2%}'.format(packet_loss_rates) if packet_loss_rates is not None else 'None'}")
        self.print_and_save(f"NetworkThroughput: {network_throughput_mbps:.3f} Mbps" if network_throughput_mbps is not None else "NetworkThroughput: None")
        self.print_and_save(f"BandwidthUtilization: {'{:.2%}'.format(bandwidth_utilization) if bandwidth_utilization is not None else 'None'}")
        self.print_and_save(f"AvgPacketNodeVisits: {'{:.3f}'.format(avg_packet_node_visits) if avg_packet_node_visits is not None else 'None'}")
        self.print_and_save(f"CumulativeReward: {cumulative_reward:.6f}" if cumulative_reward is not None else "CumulativeReward: None")
        self.print_and_save(f"AverageInferenceTime: {'{:.3f} ms'.format(average_inference_time_ms) if average_inference_time_ms is not None else 'None'}")
        self.print_and_save(f"AverageE2eDelay: {'{:.3f} seconds'.format(average_delays) if average_delays is not None else 'None'}")
        self.print_and_save(f"AverageHopCount: {'{:.3f} hops'.format(average_hops) if average_hops is not None else 'None'}")
        self.print_and_save(f"AverageComputingRatio: {'{:.2%}'.format(average_computing_ratio) if average_computing_ratio is not None else 'None'}")
        self.print_and_save(f"ComputingWaitingTime: {'{:.3f} seconds'.format(average_computing_waiting_time) if average_computing_waiting_time is not None else 'None'}")

        rewards = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        self.print_and_save(f"AverageEndingReward: {rewards if rewards is not None else 'None'}")
