from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator,Packet,Satellite,Propagator
import random
import numpy as np
import simpy
import networkx as nx
import torch
import time

PRE=True
CT_FAC=5

class Reward_Function:
    def __init__(self,reach_factor,delay_factor,loss_factor,memory_threshold,memory_factor):
        self.reach_factor=reach_factor
        self.delay_factor=delay_factor
        self.loss_factor=loss_factor
        self.memory_threshold=memory_threshold
        self.memory_factor=memory_factor

    def reach_reward(self,delay):
        return self.reach_factor-self.delay_factor*delay

    def reach_reward_abnormal(self,delay):
        return -self.delay_factor*delay

    def normal_reward(self,delay,memory_remain):
        return -self.delay_factor*delay-self.memory_factor*(memory_remain<self.memory_threshold)

    def loss_reward(self,delay):
        return -self.loss_factor-self.delay_factor*delay

class Propagator_Computing(Propagator):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiences = []
        self.final_rewards = []
        self.action_logs = []

    def trans_parameters(self,max_hop,DownstreamDelays,reward_function):
        self.max_hop=max_hop
        self.DownstreamDelays=DownstreamDelays
        self.reward_function = reward_function

    def reset_parameters(self):
        self.experiences = []
        self.final_rewards = []
        self.action_logs = []

    def _split_action_for_storage(self, action):
        if hasattr(action, 'buffer_action'):
            try:
                from mdp_attacks.mdp_action_attack import record_action_attack_sample
                record_action_attack_sample(getattr(action, 'attacked', False))
            except Exception:
                pass
            try:
                buffer_action = int(action.buffer_action)
            except Exception:
                return action, action
            if hasattr(action, 'to_log_payload'):
                return buffer_action, action.to_log_payload()
            return buffer_action, {
                'buffer_action': buffer_action,
                'executed_action': getattr(action, 'executed_action', buffer_action),
            }
        return action, action

    def append_experience(self, agent_name, state, mark, action, reward, next_state, done, event_type):
        stored_action, logged_action = self._split_action_for_storage(action)
        experience = [state, mark, stored_action, reward, next_state, done]
        self.experiences.append(experience)
        self.action_logs.append({
            'agent_name': agent_name,
            'event_type': event_type,
            'state': state,
            'mark': mark,
            'action': logged_action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

    def propagate(self,node,next_hop,packet):
        is_computed, type, computing_demand, size_after_computing, last_time, last_state, last_action = packet.information
        if (node, next_hop) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, next_hop)])
            if next_hop in self.node_names:
                success = self.satellites[next_hop].push_forward(packet)
                if success:
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source,packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[next_hop].current_memory_occupy}.",detail=True)
                else:
                    source, destination, hops, creation_time, size = packet.source, packet.destination, packet.hops, packet.creation_time, packet.size
                    current_state = self.satellites[node].get_current_state(destination, hops, is_computed,[type,size / self.satellites[node].max_size,computing_demand / self.satellites[node].ComputingAbility,size_after_computing / self.satellites[node].max_size])
                    done = 1
                    reward = self.reward_function.loss_reward(self.env.now-creation_time)
                    self.append_experience(node, last_state, current_state[-1], last_action, reward, current_state, done, 'relay_queue_full_drop')
                    self.final_rewards.append(reward)
                    if packet.computing_node and PRE:
                        self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type == 0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Routing queue is full, discarding packet {(packet.source,packet.destination,packet.creation_time)}.")
            else:
                if packet.computing_node and PRE:
                    self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
        else:
            if packet.computing_node and PRE:
                self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
            if type == 0:
                self.statics_data['Lost_relay_0'] += 1
            else:
                self.statics_data['Lost_relay_1'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet")

    def downstream(self,node,packet):
        source, destination, hops, creation_time,size = packet.source, packet.destination, packet.hops, packet.creation_time,packet.size
        is_computed,type, computing_demand, size_after_computing, last_time, last_state, last_action = packet.information
        current_state = self.satellites[node].get_current_state(destination, hops, is_computed,[type,size/self.satellites[node].max_size,computing_demand/self.satellites[node].ComputingAbility, size_after_computing/self.satellites[node].max_size])
        done = 1
        if node in self.satellites:
            yield self.env.timeout(self.DownstreamDelays)
            if is_computed:
                reward = self.reward_function.reach_reward(self.env.now-creation_time)
                if type==0:
                    self.statics_data['Reached_after_computed_0'] += 1
                else:
                    self.statics_data['Reached_after_computed_1'] += 1
            else:
                reward = self.reward_function.reach_reward_abnormal(self.env.now-creation_time)
            if type == 0:
                self.statics_data['Reached_0'] += 1
            else:
                self.statics_data['Reached_1'] += 1
            self.statics_data['NetworkThroughput'] += packet.size
            if type == 0:
                self.statics_data['Total_hops_0'] += packet.hops
            else:
                self.statics_data['Total_hops_1'] += packet.hops
            if type == 0:
                self.statics_data['Total_delay_0'] += self.env.now - packet.creation_time
            else:
                self.statics_data['Total_delay_1'] += self.env.now - packet.creation_time
            self.statics_data['Computing_waiting_time'] += packet.computing_waiting_time
            self.logger.log( f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} reached its destination {node}.")

        else:
            reward = self.reward_function.loss_reward(self.env.now-creation_time)
            if packet.computing_node and PRE:
                self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
            if type == 0:
                self.statics_data['Lost_relay_0'] += 1
            else:
                self.statics_data['Lost_relay_1'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: downlink of {node} is missed, dropped 1 packet")
        if last_state is not None:
            self.final_rewards.append(reward)
            self.append_experience(node, last_state, current_state[-1], last_action, reward, current_state, done, 'terminal')
        else:
            pass

class Satellite_with_Computing(Satellite):
    def __init__(self,mode,select_mode,epsilon,max_hop,max_size,device,env,name,neighbors,memory,ComputingAbility,TransmissionRate,DownlinkRate,StateUpdatePeriod,is_downlink,logger,statics_data={},num=None,processing_time=1e-9,heartbeat_timeout=0.25):
        self.num=num
        self.mode=mode
        self.select_mode=select_mode
        self.q_net=None
        self.epsilon=epsilon
        self.max_hop=max_hop
        self.max_size=max_size
        self.device=device
        self.name=name
        self.neighbors= sorted(neighbors)
        self.env=env
        self.gat_node_dim=64
        self.orbit_altitude, self.orbit_number, self.sat_number = map(int, self.name.split('_')[1:])
        self.memory=memory
        self.ComputingAbility=ComputingAbility
        self.TransmissionRate=TransmissionRate
        self.base_TransmissionRate = TransmissionRate
        self.link_TransmissionRates = {neighbor: TransmissionRate for neighbor in self.neighbors}
        self.DownlinkRate=DownlinkRate
        self.StateUpdatePeriod=StateUpdatePeriod
        self.logger=logger
        self.computing_queue = simpy.Store(self.env)
        self.offload_queue=simpy.Store(self.env)
        self.offload_size=0
        self.offload_length=0
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.transmission_size ={neighbor: 0 for neighbor in self.neighbors}
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.active_transmission_count = {neighbor: 0 for neighbor in self.neighbors}
        self.active_transmission_size = {neighbor: 0 for neighbor in self.neighbors}
        self.active_ground_transmission_count = {}
        self.active_ground_transmission_size = {}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        self.current_queue_size=0
        self.current_computing_queue_size=0
        self.forward_queue = simpy.Store(self.env)
        self.current_memory_occupy=0
        self.active=True
        self.routing_tables={}
        if 'New' in self.mode:
            self.current_state = [0,1,0,0,0,4,0,0,0,12,0,0]
            self.neighbor_states = {neighbor: [0,1,0,0,0,4,0,0,0,12,0,0] for neighbor in self.neighbors}
        else:
            self.current_state = [0,1,0,0]
            self.neighbor_states={neighbor: [0,1,0,0] for neighbor in self.neighbors}
        self.propagator=None
        self.statics_data=statics_data
        self.processing_time=processing_time
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.is_downlink=is_downlink
        self.is_producing=0

        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)}
        self.computing_remain=0
        self.neighbor_graph=None
        self.is_computing=False
        self.computing_time=0
        self.visible_ground_stations = []
        self.ground_station_cursor = 0

        self.last_computing_time=0

    def _get_link_TransmissionRate(self, neighbor):
        return self.link_TransmissionRates.get(neighbor, self.base_TransmissionRate)

    def _run_q_net_timed(self, current_state_tensor):
        if self.q_net is None:
            raise ValueError("q_net is required for inference timing")

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        with torch.no_grad():
            main_output = self.q_net(current_state_tensor)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if 'InferenceTimeTotalMs' in self.statics_data:
            self.statics_data['InferenceTimeTotalMs'] += elapsed_ms
        if 'InferenceCallCount' in self.statics_data:
            self.statics_data['InferenceCallCount'] += 1
        return main_output

    def trans_parameters(self,q_net, reward_function, direct_satellites):
        self.q_net=q_net
        self.reward_function = reward_function
        self.direct_satellites = direct_satellites

    def update_visible_ground_stations(self, landmarks):
        self.visible_ground_stations = sorted(
            ground_name
            for ground_name, satellites in landmarks.items()
            if self.name in satellites
        )

    def _select_ground_station(self, packet=None):
        preferred_ground_station = getattr(packet, 'ground_station', None) if packet is not None else None
        if preferred_ground_station:
            return preferred_ground_station
        if not self.visible_ground_stations:
            return None
        if packet is not None:
            seed_value = f"{packet.source}|{packet.destination}|{packet.creation_time}"
            index = sum(ord(ch) for ch in seed_value) % len(self.visible_ground_stations)
        else:
            index = self.ground_station_cursor % len(self.visible_ground_stations)
            self.ground_station_cursor += 1
        return self.visible_ground_stations[index]

    def build_routing_table(self):
        result_dict = {}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        for start in [self.name] + self.neighbors:
            queue = [(neighbor, [start, neighbor], 1) for neighbor in self.adjacency_table[start][0]]
            if start != self.name:
                self.neighbor_hops[start][start] = 0
            while queue:
                (node, path, hops) = queue.pop(0)
                if start == self.name:
                    if node not in result_dict:
                        result_dict[node] = ([path[1]], hops)
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
                    elif result_dict[node][1] == hops:
                        result_dict[node][0].append(path[1])
                else:
                    if node not in self.neighbor_hops[start]:
                        self.neighbor_hops[start][node] = hops
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
        result_dict[self.name] = ([self.name], 0)
        self.routing_tables = result_dict

    def push_forward(self,packet):
        if self.current_memory_occupy + packet.size < self.memory:
            self.current_memory_occupy += packet.size
            self.forward_queue.put(packet)
            if self.mode in ["Tradition","Ground"]:
                if 'current_memory_occupy' in self.propagator.graph.nodes[self.name]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] += packet.size
                else:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] = packet.size
            return True
        else:
            return False

    def push_computing(self,packet,computing_demand):
        self.current_computing_queue_size += packet.size
        self.computing_remain += computing_demand
        self.computing_queue.put(packet)
        if not PRE:
            if 'computing_remain' in self.propagator.graph.nodes[self.name]:
                self.propagator.graph.nodes[self.name]['computing_remain'] += computing_demand
            else:
                self.propagator.graph.nodes[self.name]['computing_remain'] = computing_demand

    def push_transmission(self,neighbor,packet):
        self.current_queue_size += packet.size
        self.transmission_length[neighbor]+=1
        self.transmission_size[neighbor]+=packet.size
        self.transmission_queue[neighbor].put(packet)
        if self.mode in ["Tradition","Ground"]:
            if 'transmission_weight' in self.propagator.graph[self.name][neighbor]:
                self.propagator.graph[self.name][neighbor]['transmission_weight']+=packet.size
            else:
                self.propagator.graph[self.name][neighbor]['transmission_weight'] = packet.size

    def pop_transmission(self,neighbor):
        packet = yield self.transmission_queue[neighbor].get()
        self.current_queue_size-=packet.size
        self.transmission_size[neighbor]-=packet.size
        self.transmission_length[neighbor]-=1
        self.current_memory_occupy -= packet.size
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph[self.name][neighbor]['transmission_weight'] -= packet.size
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def push_offload(self,packet):
        if getattr(packet, 'ground_station', None) is None:
            packet.ground_station = self._select_ground_station(packet)
        self.current_queue_size += packet.size
        self.offload_size+=packet.size
        self.offload_length+=1
        self.offload_queue.put(packet)

    def pop_offload(self):
        packet = yield self.offload_queue.get()
        self.current_queue_size-=packet.size
        self.offload_size-=packet.size
        self.offload_length-=1
        self.current_memory_occupy -= packet.size
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def get_current_state(self, destination, hops, is_computed,mission_state):
        current_state = []
        neighbors_state = []
        current_node_state=[self.is_producing, 1 - self.current_memory_occupy / self.memory,(self.computing_remain / self.ComputingAbility-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC]
        for neighbor in self.neighbors:
            if 'New' in self.mode:
                neighbors_state.extend(self.neighbor_states[neighbor][0:4]+[x/4 for x in self.neighbor_states[neighbor][4:8]]+[x/12 for x in self.neighbor_states[neighbor][8:12]])
            else:
                neighbors_state.extend(self.neighbor_states[neighbor])
            neighbors_state.append(self.transmission_size[neighbor] / self.memory)
            if destination in self.neighbor_hops[neighbor]:
                neighbors_state.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
            else:
                neighbors_state.append(2)
        if len(self.neighbors) < 4:
            if 'New' in self.mode:
                if neighbors_state:
                    av1,av2,av3=2*sum(neighbors_state[3::14])/len(self.neighbors),2*sum(neighbors_state[7::14])/len(self.neighbors),2*sum(neighbors_state[11::14])/len(self.neighbors)
                else:
                    av1,av2,av3=1,1,1
            for _ in range(4 - len(self.neighbors)):
                if 'New' in self.mode:
                    neighbors_state.extend([1, 0, 1, av1, 1, 0, 1, av2, 1, 0, 1, av3, 1, 2])
                else:
                    neighbors_state.extend([1, 0, 1, 1, 1, 2])
        current_state.extend(neighbors_state)
        current_state.extend(current_node_state)
        current_state.extend(mission_state)
        current_state.extend([hops / self.max_hop,is_computed])
        return np.array(current_state)

    def tradition_routing(self,current_state,computing=True):
        def shortest_path_and_cost(source, target):
            if source == target:
                return [], 0
            else:
                try:
                    path = nx.shortest_path(self.neighbor_graph, source=source, target=target, weight='weight')[1:]
                    cost = sum(self.neighbor_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    return path, cost
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    return [], 10
        for edge in self.neighbor_graph.edges():
            node1,node2=edge
            link_rate = max(self.neighbor_graph[node1][node2].get('link_TransmissionRate', self.base_TransmissionRate), 1e-12)
            self.neighbor_graph[node1][node2]['weight']=self.neighbor_graph[node1][node2]['missing']*10+self.neighbor_graph[node1][node2].get('transmission_weight', 0)/link_rate+self.neighbor_graph[node1][node2]['propagation_weight']
            if self.propagator.graph.nodes[node2].get('current_memory_occupy',0)/self.memory > 0.9:
                self.neighbor_graph[node1][node2]['weight'] += 10
        source, destination, size, computing_demand, size_after_computing,is_computed = current_state
        if is_computed:
            routing,_=shortest_path_and_cost(self.name, destination)
            return routing, None
        else:
            times={}
            routings={}
            n = max(int(len(self.neighbor_graph.nodes) * 1 / 2), 1)
            computing_nodes = [node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) == 0 and node!=destination]
            non_computing_nodes = sorted((node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) != 0 and node!=destination),key=lambda k: self.neighbor_graph.nodes[k].get('computing_remain', 0))
            computing_nodes += non_computing_nodes[:max(n - len(computing_nodes), 0)]
            for c_node in computing_nodes:
                time=self.neighbor_graph.nodes[c_node].get('computing_remain', 0)/self.ComputingAbility+ computing_demand/self.ComputingAbility + (self.propagator.graph.nodes[c_node].get('current_memory_occupy',0)/self.memory > 0.9)*10
                routing_1,time_1=shortest_path_and_cost(self.name, c_node)
                routing_2,time_2=shortest_path_and_cost(c_node, destination)
                times[c_node]=time+time_1+time_2+(len(routing_1)*size+len(routing_2)*size_after_computing)/self.TransmissionRate
                routings[c_node]=routing_1+routing_2
            if times:
                computing_decision=sorted(times, key=times.get)[0]
            else:
                return [None],None
            if PRE and computing:
                if 'computing_remain' in self.propagator.graph.nodes[computing_decision]:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] += computing_demand
                else:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] = computing_demand
            if not computing:
                return routings[computing_decision], None
            else:
                return routings[computing_decision],computing_decision

    def get_next_hop(self, current_state, destination):
        is_computed = current_state[-1]
        if 'DQN' in self.mode:
            if np.random.rand() <= self.epsilon:
                if np.random.rand() <= 0.5:
                    if not is_computed:
                        next_index = np.random.choice(5)
                    else:
                        next_index = np.random.choice(4)
                else:
                    neighbor_distances = []
                    for neighbor in self.neighbors:
                        if destination in self.neighbor_hops[neighbor]:
                            neighbor_distances.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
                        else:
                            neighbor_distances.append(2)
                    if len(self.neighbors) < 4:
                        for _ in range(4 - len(self.neighbors)):
                            neighbor_distances.append(2)
                    if not is_computed:
                        if np.random.rand() <= 0.2:
                            next_index = 4
                        else:
                            min_value = min(neighbor_distances)
                            next_index = np.random.choice([index for index, value in enumerate(neighbor_distances) if value == min_value])
                    else:
                        min_value = min(neighbor_distances)
                        next_index = np.random.choice([index for index, value in enumerate(neighbor_distances) if value == min_value])
                return next_index
            else:
                current_state = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
                main_output = self._run_q_net_timed(current_state)
                if not is_computed:
                    next_index = torch.argmax(main_output).item()
                else:
                    next_index = torch.argmax(main_output[0][0:4]).item()
                return next_index
        else:
            current_state = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
            main_output = self._run_q_net_timed(current_state)
            if is_computed:
                main_output = main_output[:, 0:4]
            main_output = torch.nn.functional.softmax(main_output, dim=-1)
            dist = torch.distributions.Categorical(main_output)
            action = dist.sample()
            return [action.item(), dist.log_prob(action).item()]



    def find_highest_score(self, destinations, mission_state, hops, is_computed):
        highest_score= -2
        mission_state=[mission_state[0], mission_state[1]/ self.max_size, mission_state[2]/ self.ComputingAbility, mission_state[3] / self.max_size]
        best_destinations=[]
        for destination in destinations:
            if destination in self.routing_tables:
                current_state = self.get_current_state(destination, hops, is_computed, mission_state)
                score = self.cal_score(current_state,is_computed)
                if score > highest_score:
                    highest_score = score
                    best_destinations = [destination]
                elif score == highest_score:
                    best_destinations.append(destination)
        return best_destinations

    def cal_score(self, current_state,is_computed):
        current_state = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
        main_output = self._run_q_net_timed(current_state)
        if not is_computed:
            score = torch.max(main_output).item()
        else:
            score = torch.max(main_output[0][0:4]).item()
        return score

    def forward_packet(self):
        while self.active:
            packet = yield self.forward_queue.get()
            self.statics_data['PacketNodeVisits'] += 1
            packet.hops += 1
            source, destination,hops,creation_time,size = packet.source, packet.destination,packet.hops,packet.creation_time,packet.size
            is_computed,type,computing_demand, size_after_computing,last_time, last_state, last_action = packet.information
            yield self.env.timeout(self.processing_time)
            if self.mode in ["Tradition","Ground"]:
                current_state = [source, destination, size, computing_demand, size_after_computing,is_computed]
                if not packet.routing and destination!= self.name:
                    computing = False if self.mode == "Ground" else True
                    packet.routing,packet.computing_node=self.tradition_routing(current_state,computing)
            else:
                if self.select_mode == 3 and destination != self.name:
                    min_hops_destinations=self.find_min_hops_destinations(5)
                    hightest_score_destinations=self.find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],hops, is_computed)
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                    packet.destination = destination
                current_state = self.get_current_state(destination, hops,is_computed,[type,size/self.max_size,computing_demand/self.ComputingAbility,size_after_computing/self.max_size])
            if not self.active:
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            if destination != self.name:
                if hops <= 2 * self.max_hop:
                    if self.mode in ["Tradition","Ground"]:
                        if packet.computing_node==self.name:
                            action=4
                            packet.computing_node=None
                        else:
                            if packet.routing:
                                next_hop=packet.routing.pop(0)
                                if next_hop in self.neighbors:
                                    action=self.neighbors.index(next_hop)
                                else:
                                    action = 5
                            else:
                                action=5
                    else:
                        action = self.get_next_hop(current_state, destination)
                    if 'PPO' in self.mode:
                        next_index = action[0]
                    elif hasattr(action, 'executed_action'):
                        next_index = int(action.executed_action)
                    else:
                        next_index = action
                    if destination in self.routing_tables:
                        if next_index < len(self.neighbors):
                            done = 0
                            reward = self.reward_function.normal_reward(self.env.now-last_time,1-self.current_memory_occupy/self.memory)
                            next_hop = self.neighbors[next_index]
                            packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, current_state,action])
                            self.push_transmission(next_hop, packet)
                        elif next_index==4:
                            done = 0
                            packet.hops-=1
                            reward = self.reward_function.normal_reward(self.env.now-last_time,1-self.current_memory_occupy/self.memory)
                            packet.extra_information([is_computed,type,computing_demand, size_after_computing,self.env.now, current_state, action])
                            self.push_computing(packet,computing_demand)
                        else:
                            done = 1
                            reward = self.reward_function.loss_reward(self.env.now-creation_time)
                            self.propagator.final_rewards.append(reward)
                            self.current_memory_occupy-=size
                            if packet.computing_node and PRE:
                                self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                            if type == 0:
                                self.statics_data['Lost_relay_0'] += 1
                            else:
                                self.statics_data['Lost_relay_1'] += 1
                            self.logger.log(f"Time {self.env.now:.3f}: wrong forward decision, dropped 1 packet")
                    else:
                        self.current_memory_occupy -= size
                        if packet.computing_node and PRE:
                            self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                        if type == 0:
                            self.statics_data['Lost_relay_0'] += 1
                        else:
                            self.statics_data['Lost_relay_1'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, dropped 1 packet")
                        reward = None
                else:
                    self.current_memory_occupy -= size
                    done = 1
                    reward = self.reward_function.loss_reward(self.env.now-creation_time)
                    self.propagator.final_rewards.append(reward)
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type == 0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: transmission out of time, dropped 1 packet")
            else:
                reward = None
                self.push_offload(packet)
            if last_state is not None and reward is not None:
                self.propagator.append_experience(self.name, last_state, last_state[-1], last_action, reward, current_state, done, 'routing')

    def transmit_packet(self,neighbor):
        while self.active:
            if neighbor not in self.transmission_queue:
                break
            packet = yield self.env.process(self.pop_transmission(neighbor))
            is_computed,type, computing_demand, size_after_computing, last_time, current_state, next_index = packet.information
            packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, current_state, next_index])
            current_link_rate = self._get_link_TransmissionRate(neighbor)
            if current_link_rate <= 0:
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type == 0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: link {(self.name, neighbor)} has zero bandwidth, dropped 1 packet")
                continue
            self.active_transmission_count[neighbor] = self.active_transmission_count.get(neighbor, 0) + 1
            self.active_transmission_size[neighbor] = self.active_transmission_size.get(neighbor, 0) + packet.size
            try:
                yield self.env.timeout(packet.size / current_link_rate)
                self.statics_data['ISLTrafficVolume'] += packet.size
                if neighbor not in self.neighbors or not self.active:
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type == 0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
                else:
                    self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} departed. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
                    self.env.process(self.propagator.propagate(self.name,neighbor, packet))
            finally:
                if neighbor in self.active_transmission_count:
                    self.active_transmission_count[neighbor] = max(self.active_transmission_count.get(neighbor, 0) - 1, 0)
                if neighbor in self.active_transmission_size:
                    self.active_transmission_size[neighbor] = max(self.active_transmission_size.get(neighbor, 0) - packet.size, 0)

    def offload_to_ground(self):
        while self.active:
            packet = yield self.env.process(self.pop_offload())
            is_computed,type, computing_demand, size_after_computing, last_time, current_state, next_index = packet.information
            packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, current_state, next_index])
            ground_station = self._select_ground_station(packet)
            packet.ground_station = ground_station
            if ground_station is not None:
                self.active_ground_transmission_count[ground_station] = self.active_ground_transmission_count.get(ground_station, 0) + 1
                self.active_ground_transmission_size[ground_station] = self.active_ground_transmission_size.get(ground_station, 0) + packet.size
            try:
                yield self.env.timeout(packet.size / self.DownlinkRate)
                if not self.active:
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type==0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: offload stopped, dropped 1 packet")
                    break
                self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} is offloading to {ground_station or 'UNKNOWN_GROUND'}. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
                self.env.process(self.propagator.downstream(self.name, packet))
            finally:
                if ground_station is not None and ground_station in self.active_ground_transmission_count:
                    self.active_ground_transmission_count[ground_station] = max(self.active_ground_transmission_count.get(ground_station, 0) - 1, 0)
                if ground_station is not None and ground_station in self.active_ground_transmission_size:
                    self.active_ground_transmission_size[ground_station] = max(self.active_ground_transmission_size.get(ground_station, 0) - packet.size, 0)

    def computing_packet(self):
        while self.active:
            packet = yield self.computing_queue.get()
            self.is_computing=True
            self.last_computing_time=self.env.now
            is_computed,type,computing_demand, size_after_computing,last_time, last_state, last_action=packet.information
            computing_time_consume=computing_demand / self.ComputingAbility
            yield self.env.timeout(computing_time_consume)
            packet.computing_waiting_time+=(self.env.now-last_time)
            self.computing_time += computing_time_consume
            if self.mode in ["Tradition","Ground"]:
                self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size-size_after_computing
                self.propagator.graph.nodes[self.name]['computing_remain'] -= computing_demand
            self.current_computing_queue_size -= packet.size
            self.computing_remain -= computing_demand
            self.current_memory_occupy -= packet.size - size_after_computing
            packet.size=size_after_computing
            self.is_computing = False
            self.last_computing_time = 0
            if not self.active:
                self.current_memory_occupy -= packet.size
                if self.mode in ["Tradition","Ground"]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
                break
            packet.extra_information([True,type,0,size_after_computing,last_time, last_state, last_action])
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} finished computing. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
            self.forward_queue.put(packet)

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.neighbors=sorted(self.neighbors)
            self.transmission_queue[neighbor] = simpy.Store(self.env)
            self.transmission_size[neighbor] = 0
            self.transmission_length[neighbor] = 0
            self.active_transmission_count[neighbor] = 0
            self.active_transmission_size[neighbor] = 0
            self.link_TransmissionRates[neighbor] = self.base_TransmissionRate
            if "New" in self.mode:
                self.neighbor_states[neighbor] = [0,1,0,0,0,4,0,0,0,12,0,0]
            else:
                self.neighbor_states[neighbor] = [0,1,0,0]
            self.last_heartbeat[neighbor] = self.env.now
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            self.neighbor_hops[neighbor] = {}
            self.adjacency_table_exchanger()
            self.refresh_link_TransmissionRates()
            self.env.process(self.monitor_single_neighbor(neighbor))
            self.env.process(self.transmit_packet(neighbor))

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor))
                    is_computed, type, computing_demand, size_after_computing, last_time, last_state, last_action = packet.information
                    packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, None, None])
                    packet.routing=None
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                        packet.computing_node=None
                    if self.mode not in ["Tradition","Ground"]:
                        success = self.push_forward(packet)
                    else:
                        success = False
                    if not success:
                        if type == 0:
                            self.statics_data['Lost_relay_0'] += 1
                        else:
                            self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
                if neighbor in self.neighbors:
                    self.neighbors.remove(neighbor)
                del self.transmission_queue[neighbor]
                del self.transmission_size[neighbor]
                del self.neighbor_states[neighbor]
                del self.neighbor_hops[neighbor]
                del self.transmission_length[neighbor]
                del self.active_transmission_count[neighbor]
                del self.active_transmission_size[neighbor]
                del self.link_TransmissionRates[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.update_adjacency_dict_for_bfs()
                self.build_routing_table()
                self.adjacency_table_exchanger()
                return True
            else:
                return False
        else:
            return False

    def find_min_hops_destinations(self, n):
        hop_counts = []
        for destination in self.direct_satellites:
            if destination in self.routing_tables:
                hops = self.routing_tables[destination][1]
            else:
                hops = np.inf
            hop_counts.append((destination, hops))
        if len(hop_counts)>n:
            hop_counts.sort(key=lambda x: x[1])
            return [a for a,b in hop_counts[:n]]
        else:
            return [a for a,b in hop_counts]

    def state_exchanger(self):
        while self.active:
            yield self.env.timeout(self.StateUpdatePeriod)
            if not self.active:
                break
            for neighbor in self.neighbors:
                if 'New' in self.mode:
                    n = len(self.neighbors)
                    if n:
                        position_sums = [sum(items) for items in zip(*self.neighbor_states.values())]
                        av1, av2, av3 = 2 * position_sums[3] / n, 2 * position_sums[7] / n, 2 * position_sums[11] / n
                    else:
                        av1, av2, av3 = 1, 1, 1
                    specified_values_list = [1,0,1,av1,1,0,1,av2,1,0,1,av3]
                    self_value=[self.is_producing, 1 - self.current_memory_occupy / self.memory,(self.computing_remain / self.ComputingAbility-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC ,sum(self.transmission_size.values())/self.memory]
                    self_values= [0,0,0,0]+[x * n for x in self.current_state]+[0,0,0,0]
                    additional_values = [x * (4 - n) for x in specified_values_list]
                    temp = [sum(tup) + add - sub for tup, add,sub in zip(zip(*self.neighbor_states.values()), additional_values,self_values)]
                    self.current_state=self_value+temp[0:8]
                    self.env.process(self.propagator.send_state(self.name, neighbor, self.current_state))
                else:
                    self.env.process(self.propagator.send_state(self.name, neighbor,[self.is_producing,1-self.current_memory_occupy/self.memory,(self.computing_remain / self.ComputingAbility-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC,self.transmission_size[neighbor]/self.memory]))

    def all_start(self):
        super().all_start()
        self.env.process(self.computing_packet())
        self.env.process(self.offload_to_ground())

    def self_missing(self):
        self.active = False
        while self.computing_queue.items:
            packet=yield self.computing_queue.get()
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_computing_queue_size = 0
        while self.offload_queue.items:
            packet = yield self.env.process(self.pop_offload())
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.env.process(self.forward_queue.get())
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            self.active_transmission_count[neighbor] = 0
            self.active_transmission_size[neighbor] = 0
        self.active_ground_transmission_count = {}
        self.active_ground_transmission_size = {}
        self.current_memory_occupy=0
        if self.mode in ["Tradition","Ground"] and self.name in self.propagator.graph.nodes:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] = 0
        self.is_producing = 0
        self.computing_remain = 0

class SatelliteNetworkSimulator_OnbardComputing(SatelliteNetworkSimulator):
    def __init__(self,mode,select_mode,q_net,epsilon,reward_factors,device,MissionPossibility,PoissonRate,PacketGenerationInterval,DomputingDemandFactor,DomputingDemandFactor_2,SizeAfterComputingFactor,SizeAfterComputing_1,graph,landmarks,MeanIntervalTime,memory,ComputingAbility,TransmissionRate,DownlinkRate,DownstreamDelays,PacketSizeRange,PacketSizeMean,PacketSizeStd,StateUpdatePeriod,logger):
        self.q_net = q_net
        self.mode=mode
        self.epsilon=epsilon
        self.reward_function=Reward_Function(*reward_factors)
        self.device=device
        self.MissionPossibility=MissionPossibility
        self.PoissonRate=PoissonRate
        self.PacketGenerationInterval = self._resolve_PacketGenerationInterval(PacketGenerationInterval)
        self.DomputingDemandFactor=DomputingDemandFactor
        self.DomputingDemandFactor_2=DomputingDemandFactor_2
        self.SizeAfterComputingFactor=SizeAfterComputingFactor
        self.SizeAfterComputing_1=SizeAfterComputing_1
        self.graph = graph
        self.max_hop=self._compute_max_hop(self.graph)
        self.PacketSizeMean, self.PacketSizeStd, self.max_size = self._resolve_packet_size_distribution(
            PacketSizeMean,
            PacketSizeStd,
            PacketSizeRange
        )
        self.env = simpy.Environment()
        self.memory = memory
        self.ComputingAbility = ComputingAbility
        self.logger=logger
        self.TransmissionRate=TransmissionRate
        self.DownstreamDelays=DownstreamDelays
        self.DownlinkRate=DownlinkRate
        self.StateUpdatePeriod=StateUpdatePeriod
        self.landmarks=landmarks
        self.direct_satellites = set(sum(self.landmarks.values(), []))
        self.statics_datas = {
            'Total': 0,
            'Reached_0': 0,
            'Reached_1': 0,
            'Reached_after_computed_0': 0,
            'Reached_after_computed_1': 0,
            'Lost_upload': 0,
            'Lost_relay_0': 0,
            'Lost_relay_1': 0,
            'Total_delay_0': 0,
            'Total_delay_1': 0,
            'Total_hops_0': 0,
            'Total_hops_1': 0,
            'Is_computing': 0,
            'Computing_waiting_time': 0,
            'NetworkThroughput': 0,
            'ISLTrafficVolume': 0,
            'ISLBandwidthCapacity': 0,
            'InferenceTimeTotalMs': 0.0,
            'InferenceCallCount': 0,
            'PacketNodeVisits': 0,
        }
        self.satellite_names=[node for node in self.graph.nodes]
        self.satellites={}
        self.select_mode=select_mode
        for node in self.graph.nodes:
            satellite_name_with_suffix = node
            self.satellites[satellite_name_with_suffix] = Satellite_with_Computing(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                satellite_name_with_suffix,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.ComputingAbility,
                self.TransmissionRate,
                self.DownlinkRate,
                self.StateUpdatePeriod,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
            )
            self.satellites[satellite_name_with_suffix].update_visible_ground_stations(self.landmarks)
        self.propagator = Propagator_Computing(self.env, graph, logger, self.satellites,self.statics_datas, True if self.mode in ["Tradition","Ground"] else False)
        self.MeanIntervalTime=MeanIntervalTime
        self.propagator.trans_parameters(self.max_hop,self.DownstreamDelays,self.reward_function)
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)
            self.satellites[satellite].set_propagator(self.propagator)
            self.satellites[satellite].update_visible_ground_stations(self.landmarks)
            self.satellites[satellite].build_routing_table()

    def _compute_max_hop(self, graph):
        if graph.number_of_nodes() <= 1:
            return 1
        if nx.is_connected(graph):
            return nx.diameter(graph)
        largest_component_nodes = max(nx.connected_components(graph), key=len)
        largest_component = graph.subgraph(largest_component_nodes)
        if largest_component.number_of_nodes() <= 1:
            return 1
        return nx.diameter(largest_component)

    def _resolve_PacketGenerationInterval(self, PacketGenerationInterval):
        interval = float(PacketGenerationInterval)
        if interval <= 0:
            raise ValueError("PacketGenerationInterval must be > 0")
        return interval

    def _resolve_packet_size_distribution(self, PacketSizeMean, PacketSizeStd, PacketSizeRange):
        if PacketSizeMean is not None:
            mean_value = int(round(float(PacketSizeMean)))
            std_value = 0.0 if PacketSizeStd is None else float(PacketSizeStd)
        else:
            min_packet_size = int(round(PacketSizeRange[0]))
            max_packet_size = int(round(PacketSizeRange[1]))
            if min_packet_size > max_packet_size:
                min_packet_size, max_packet_size = max_packet_size, min_packet_size
            mean_value = int(round((min_packet_size + max_packet_size) / 2))
            std_value = float((max_packet_size - min_packet_size) / np.sqrt(12))
        if mean_value <= 0:
            raise ValueError("PacketSizeMean must be > 0")
        if std_value < 0:
            raise ValueError("PacketSizeStd must be >= 0")
        max_size = max(mean_value + 3 * std_value, mean_value)
        return mean_value, std_value, int(round(max_size))

    def _sample_packet_size(self):
        if self.PacketSizeStd == 0:
            return self.PacketSizeMean
        sampled_size = int(round(random.gauss(self.PacketSizeMean, self.PacketSizeStd)))
        return max(1, sampled_size)

    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = [f"{neighbor}" for neighbor in self.graph.neighbors(node)]
            adjacency_dict[f"{node}"] = (neighbors, self.env.now)
        return adjacency_dict

    def generate_traffic(self, satellite):
        while satellite in self.satellite_names:
            if satellite not in self.satellites:
                break
            self.satellites[satellite].is_producing=0
            session_start_time = min(random.expovariate(1.0 / self.PoissonRate),self.PoissonRate*3)
            yield self.env.timeout(session_start_time)
            if not satellite in self.satellite_names:
                self.logger.log(f"Time {self.env.now:.3f}: {satellite} is missed, packets failed to generate.")
                break
            type = random.choices([0, 1], weights=self.MissionPossibility, k=1)[0]
            if satellite in self.direct_satellites:
                continue
            else:
                if self.select_mode == 1 or self.mode in ["Tradition","Ground"]:
                    min_hops = np.inf
                    min_hops_destinations=[]
                    for destination in self.direct_satellites:
                        if destination in self.satellites[satellite].routing_tables:
                            hops = self.satellites[satellite].routing_tables[destination][1]
                        else:
                            hops = np.inf
                        if hops < min_hops:
                            min_hops = hops
                            min_hops_destinations = [destination]
                        elif hops == min_hops:
                            min_hops_destinations.append(destination)
                    if min_hops_destinations:
                        destination = np.random.choice(min_hops_destinations)
                    else:
                        self.logger.log(f"Time {self.env.now:.3f}: connections for {satellite} is not availiable, packet failed to generate.")
            session_duration = min(random.expovariate(1.0 / self.MeanIntervalTime),self.MeanIntervalTime*3)

            end_time = self.env.now + session_duration
            self.satellites[satellite].is_producing = 1
            if self.mode in ["Tradition","Ground"]:
                neighbors=[]
                for _satellite in self.satellites:
                    if satellite in self.satellites[_satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                        if (self.satellites[_satellite].routing_tables[satellite][1]+self.satellites[_satellite].routing_tables[destination][1])<=min_hops+4:
                            neighbors.append(self.satellites[_satellite].name)
                self.satellites[satellite].neighbor_graph=self.propagator.graph.subgraph(neighbors)
            while self.env.now < end_time:
                yield self.env.timeout(self.PacketGenerationInterval)
                size = self._sample_packet_size()
                if type == 0:
                    size_after_computing = int(random.uniform(self.SizeAfterComputingFactor[0], self.SizeAfterComputingFactor[1]) * size)
                    computing_demand = int(random.uniform(self.DomputingDemandFactor[0], self.DomputingDemandFactor[1]) * size)
                else:
                    size_after_computing = self.SizeAfterComputing_1
                    computing_demand = int(random.uniform(self.DomputingDemandFactor_2[0], self.DomputingDemandFactor_2[1]) * size)
                if self.select_mode != 1:
                    min_hops_destinations=self.satellites[satellite].find_min_hops_destinations(5)
                    hightest_score_destinations=self.satellites[satellite].find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],0,0)
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                if not (destination in self.satellite_names and satellite in self.satellite_names):
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite} or {destination} is missed, packet failed to generate.")
                    break
                if destination in self.satellites[satellite].routing_tables:
                    packet = Packet(satellite, destination, self.env.now, size)
                    packet.extra_information([False, type, computing_demand, size_after_computing, self.env.now, None, None])
                    self.statics_datas['Total'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet generated: {(satellite, destination,packet.creation_time)}.")
                    success = self.satellites[satellite].push_forward(packet)
                    if success:
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet {(packet.source, packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[satellite].current_memory_occupy}.",detail=True)
                    else:
                        self.statics_datas['Lost_upload'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Routing queue is full, discarding packet ({satellite}, {destination},packet.creation_time).")
                    if self.mode in ["Tradition","Ground"]:
                        neighbors = []
                        for _satellite in self.satellites:
                            if _satellite in self.satellites[satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                                if (self.satellites[satellite].routing_tables[_satellite][1] + self.satellites[_satellite].routing_tables[destination][1]) <= min_hops + 4:
                                    neighbors.append(self.satellites[_satellite].name)
                        self.satellites[satellite].neighbor_graph = self.propagator.graph.subgraph(neighbors)
                else:
                    self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, packet failed to generate.")
                    break
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 0

    def get_system_state(self):
        total_queue_usage = {}
        total_computing_memory={}
        for node in self.satellite_names:
            average_usage = min(self.satellites[node].current_memory_occupy / self.memory,1)
            computing_memory = self.satellites[node].current_computing_queue_size / self.memory
            total_queue_usage[node] = average_usage
            total_computing_memory[node] = computing_memory
        return total_queue_usage

    def run(self, duration):
        if self.env.now==0:
            for satellite in self.satellite_names:
                self.env.process(self.generate_traffic(satellite))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()
        if 'ISLBandwidthCapacity' in self.statics_datas:
            self.statics_datas['ISLBandwidthCapacity'] += self._total_directional_link_capacity() * duration
        self.env.run(until=self.env.now+duration)
        if 'Is_computing' in self.statics_datas:
            for satellite in self.satellites:
                self.statics_datas['Is_computing']+= self.satellites[satellite].is_computing

    def _total_directional_link_capacity(self):
        total_capacity = 0.0
        for satellite in self.satellites.values():
            for neighbor in satellite.neighbors:
                total_capacity += max(
                    satellite.link_TransmissionRates.get(neighbor, satellite.base_TransmissionRate),
                    0.0
                )
        return total_capacity

    def upgrade_all(self,graph,landmarks):

        self.landmarks = landmarks
        new_nodes = set(graph.nodes())
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        old_direct_satellites=self.direct_satellites

        self.satellite_names = [node for node in graph]
        self.graph = graph
        self.max_hop = self._compute_max_hop(self.graph)
        self.propagator.update(graph)
        self.propagator.reset_parameters()
        self.propagator.trans_parameters(self.max_hop,self.DownstreamDelays,self.reward_function)
        for satellite in self.satellites.values():
            satellite.refresh_link_TransmissionRates()
            satellite.update_visible_ground_stations(self.landmarks)

        flattened_list = set(sum(self.landmarks.values(), []))
        new_direct_satellites = flattened_list
        self.direct_satellites = flattened_list

        for node in new_nodes - old_nodes:
            self.satellites[node] = Satellite_with_Computing(self.mode,self.select_mode,self.epsilon,self.max_hop,self.max_size,self.device,self.env, node, [neighbor for neighbor in self.graph.neighbors(node)],self.memory,self.ComputingAbility,self.TransmissionRate,self.DownlinkRate, self.StateUpdatePeriod,True if node in self.direct_satellites else False, self.logger, self.statics_datas)
            self.satellites[node].set_propagator(self.propagator)
            self.satellites[node].update_visible_ground_stations(self.landmarks)
            self.satellites[node].all_start()
            self.satellites[node].adjacency_table_exchanger()
            self.env.process(self.generate_traffic(node))

        for node in old_nodes - new_nodes:
            self.env.process(self.satellites[node].self_missing())
            del self.satellites[node]
        for edge in old_edges - new_edges:
            node, neighbor = edge
            if node in self.satellites:
                self.env.process(self.satellites[node].del_neighbor(neighbor))
            if neighbor in self.satellites:
                self.env.process(self.satellites[neighbor].del_neighbor(node))
        for edge in new_edges - old_edges:
            node, neighbor = edge
            self.satellites[node].add_neighbor(neighbor)
            self.satellites[neighbor].add_neighbor(node)
        for satellite in self.satellites.values():
            satellite.refresh_link_TransmissionRates()
            satellite.update_visible_ground_stations(self.landmarks)
        for node in old_direct_satellites - new_direct_satellites:
            if node in self.satellites:
                self.satellites[node].is_downlink = False
        for node in new_direct_satellites - old_direct_satellites:
            if node in self.satellites:
                self.satellites[node].is_downlink = True
        for satellite in self.satellites:
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)
            self.satellites[satellite].update_visible_ground_stations(self.landmarks)

