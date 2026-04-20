from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph
from skyfield.api import load,Topos
import random
import datetime
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator,Logger
from Draw_Graph_Quiker import SatelliteVisualizer

class SatelliteSimulation:
    def __init__(self, begin_time, end_time, time_stride, tle_filepath,SODFilePath, mean_interarrival_time,queue_length,TransmissionRate,packet_size, StateUpdatePeriod,
             visualize=False,PrintInfo=False, SaveLog=False, ShowDetail=False,DegradedEdgeRatio=0,RandomNodesDel=0,ElevationAngle=45,pole=False):
        self.tracker = SatelliteTracker(tle_filepath)
        self.coordinates = extract_landmarks(SODFilePath)
        self.graph_builder = SatelliteGraph()
        self.begin_time = begin_time
        self.end_time = end_time
        self.time_stride = time_stride
        self.mean_interarrival_time=mean_interarrival_time
        self.queue_length=queue_length
        self.visualizer = SatelliteVisualizer(edge_color=False) if visualize else None
        self.logger = Logger(detail=ShowDetail, SaveLog=SaveLog, verbose=PrintInfo)
        self.ts = load.timescale()
        self.TransmissionRate=TransmissionRate
        self.packet_size=packet_size
        self.StateUpdatePeriod=StateUpdatePeriod
        self.DegradedEdgeRatio=DegradedEdgeRatio
        self.RandomNodesDel=RandomNodesDel
        self.ElevationAngle=ElevationAngle
        self.pole=pole
        self.staticis_list=[]
        self.time_acc=0.0

    def time_from_str(self,time_str):
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    def add_time_to_str(self,time_str, delta_time_tuple):
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        delta = datetime.timedelta(minutes=delta_time_tuple[0], seconds=delta_time_tuple[1])
        updated_dt = dt + delta
        return updated_dt.strftime("%Y-%m-%d %H:%M:%S")

    def str_to_datetime(self, time_str):
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    def datetime_difference_in_seconds(self, dt1, dt2):
        diff = dt2 - dt1
        return diff.total_seconds()

    def usage_to_rgb(self, usage):
        return 'rgb('+str(int(255 * usage)) +','+str(int(255 * (1 - usage))) + ','+ str(int(255 * (1 - usage))) + ')'

    def update_node_colors(self, graph, total_queue_usage):
        for node in graph.nodes:
            graph.nodes[node]['color'] = 'black'
            # if node in total_queue_usage:
            #     usage = total_queue_usage[node]
            #     color = self.usage_to_rgb(usage)
            #     graph.nodes[node]['color'] = color
            # else:
            #     graph.nodes[node]['color'] = 'rgb(0,255,255)'

    def _resolve_edge_removal_count(self, G, edge_ratio):
        if not 0 <= edge_ratio <= 1:
            raise ValueError("DegradedEdgeRatio must be a float ratio in [0, 1]")
        return int(G.number_of_edges() * edge_ratio)

    def remove_random_edges(self,G, edge_ratio):
        edges_to_remove_count = self._resolve_edge_removal_count(G, edge_ratio)
        edges_to_remove = random.sample(list(G.edges()), edges_to_remove_count)
        G.remove_edges_from(edges_to_remove)

        return G

    def remove_random_nodes(self,G, n):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")

        nodes_to_remove = random.sample(list(G.nodes()), n)
        G.remove_nodes_from(nodes_to_remove)

        return G

    def convert_to_eci(self,landmarks, time):
        eci_landmarks = {}
        for name, coords in landmarks.items():
            topo = Topos(latitude_degrees=coords['latitude'], longitude_degrees=coords['longitude'],
                         elevation_m=coords['altitude'])
            eci_coords = topo.at(time).position.km
            eci_landmarks[name] = {"x": eci_coords[0], "y": eci_coords[1], "z": eci_coords[2]}
        return eci_landmarks

    def visualize(self,current_graph,current_time,simulator):
        G_draw = current_graph.copy()
        self.update_node_colors(G_draw, simulator.get_system_state())
        landmark_ecis = self.convert_to_eci(self.coordinates, self.time_from_str(current_time))
        for landmark_eci, eci_value in landmark_ecis.items():
            G_draw.add_node(landmark_eci)
            G_draw.nodes[landmark_eci]['pos'] = [eci_value['x'], eci_value['y'], eci_value['z']]
            G_draw.nodes[landmark_eci]['color'] = 'rgb(200,200,200)'
        self.visualizer.draw_graph(G_draw)

    def run(self):
        init_time=self.time_from_str(self.begin_time)
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker,init_time,pole=self.pole)

        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        connections = get_connections_h3(self.coordinates, coordinates_s, self.ElevationAngle)

        simulator = SatelliteNetworkSimulator(
            graph=current_graph,
            landmarks=connections,
            mean_interarrival_time=self.mean_interarrival_time,
            queue_length=self.queue_length,
            TransmissionRate=self.TransmissionRate,
            packet_size=self.packet_size,
            StateUpdatePeriod=self.StateUpdatePeriod,
            logger=self.logger)


        current_time = self.begin_time
        total_time = self.datetime_difference_in_seconds(self.str_to_datetime(self.begin_time),
                                                         self.str_to_datetime(self.end_time))
        num_full_steps = int(total_time // self.time_stride)
        remaining_time = total_time % self.time_stride
        i = 0
        while i < num_full_steps:
            print("======"+current_time+"======")
            simulator.run(self.time_stride)
            if self.staticis_list:
                current_statics = {k: simulator.statics_data[k] - self.staticis_list[-1][k] for k in simulator.statics_data}
            else:
                current_statics=simulator.statics_data
            print("Current statics:",current_statics)
            reached = current_statics['Reached']
            lost_relay = current_statics['Lost_relay']
            total_delay = current_statics['Total_delay']
            total_hops = current_statics['Total_hops']
            network_throughput = current_statics['NetworkThroughput']
            isl_traffic_volume = current_statics['ISLTrafficVolume']
            isl_bandwidth_capacity = current_statics['ISLBandwidthCapacity']
            packet_node_visits = current_statics['PacketNodeVisits']
            network_throughput_mbps = network_throughput / self.time_stride / 1e6 if self.time_stride > 0 else None
            bandwidth_utilization = isl_traffic_volume / isl_bandwidth_capacity if isl_bandwidth_capacity > 0 else None
            avg_packet_node_visits = packet_node_visits / current_statics['Total'] if current_statics['Total'] > 0 else None
            if lost_relay + reached > 0:
                packet_loss_rate= lost_relay / (lost_relay + reached)
                print(f"PacketLossRate: {packet_loss_rate:.2%}")
            print(f"NetworkThroughput: {network_throughput_mbps:.3f} Mbps" if network_throughput_mbps is not None else "NetworkThroughput: None")
            print(f"BandwidthUtilization: {bandwidth_utilization:.2%}" if bandwidth_utilization is not None else "BandwidthUtilization: None")
            print(f"AvgPacketNodeVisits: {avg_packet_node_visits:.3f}" if avg_packet_node_visits is not None else "AvgPacketNodeVisits: None")
            if reached > 0:
                print(f"AverageE2eDelay(Average delay for successful transmissions): {total_delay / reached:.3f} second")
                print(f"AverageHopCount(Average hop count for successful transmissions): {total_hops / reached:.3f} hops")
            self.staticis_list.append(simulator.statics_data.copy())
            # simulator.clear_statics()
            if self.visualizer:
                self.visualize(current_graph, current_time, simulator)
            self.time_acc += self.time_stride
            if self.time_acc >= 1.0:
                current_time = self.add_time_to_str(current_time, (0, int(self.time_acc)))
                self.time_acc -= int(self.time_acc)
            i += 1
            coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(current_time))
            connections = get_connections_h3(self.coordinates, coordinates_s,self.ElevationAngle)
            old_nodes = set(current_graph.nodes())
            current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(current_time),pole=False)
            self.remove_random_nodes(current_graph,self.RandomNodesDel)
            self.remove_random_edges(current_graph,self.DegradedEdgeRatio)
            new_nodes = set(current_graph.nodes())
            lost_nodes = old_nodes - new_nodes
            for landmark, satellites in connections.items():
                for lost_node in lost_nodes:
                    if lost_node in satellites:
                        connections[landmark].remove(lost_node)
            simulator.upgrade_all(current_graph,connections)

        if remaining_time > 0:
            simulator.run(remaining_time)
