# iSatCR

`iSatCR` is a LEO satellite constellation simulation and reinforcement-learning project for joint computing and routing optimization. The repository contains environment construction, agent definitions, training and test entry scripts, topology visualization tools, and multiple MDP/model attack modules for security evaluation.

## Reference

This repository is related to the paper:

`Joint Optimization of Computing and Routing in LEO Satellite Constellations with Distributed Deep Reinforcement Learning`

Published at `2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)`:
https://ieeexplore.ieee.org/document/10758053

## Project Scope

The current repository supports:

- LEO satellite network topology construction and simulation
- Joint routing and onboard computing decision-making
- DQN, DDQN, dueling DDQN, weak DQN, and PPO-based agents
- Training and test execution through YAML configuration files
- 3D topology visualization for satellite graphs
- Security evaluation through multiple attack mechanisms in `mdp_attacks`

## Main Files

- [PRC.py](./PRC.py): main training/test entry
- [RL_environment_for_computing.py](./RL_environment_for_computing.py): RL environment wrapper and metric logging
- [SatelliteNetworkSimulator_Computing.py](./SatelliteNetworkSimulator_Computing.py): satellite network simulation core
- [Base_Agents.py](./Base_Agents.py): DQN/DDQN/PPO agent definitions
- [Draw_Graph_Quiker.py](./Draw_Graph_Quiker.py): satellite topology visualization
- [Make_Satellite_Graph.py](./Make_Satellite_Graph.py): graph construction utilities
- [train](./train): YAML configuration files
- [mdp_attacks](./mdp_attacks): attack modules
- [training_process_data](./training_process_data): exported logs and metric traces

## Attack Modules

The repository currently includes the following attack scripts:

- [mdp_StateObservation_attack.py](./mdp_attacks/mdp_StateObservation_attack.py): adversarial perturbation on observed state vectors
- [mdp_action_attack.py](./mdp_attacks/mdp_action_attack.py): tampering with executed actions
- [mdp_Reward_attack.py](./mdp_attacks/mdp_Reward_attack.py): reward shaping/tampering before experience storage
- [mdp_StateTransfer_attack.py](./mdp_attacks/mdp_StateTransfer_attack.py): poisoning transition tuples before replay
- [ExperiencePool_attack.py](./mdp_attacks/ExperiencePool_attack.py): replay-buffer poisoning during update
- [ModelTamp_attack.py](./mdp_attacks/ModelTamp_attack.py): runtime model parameter tampering

The main entry script reads the following attack controls from YAML:

- `StateObservationAttack_level`
- `ActionAttack_level`
- `RewardAttack_level`
- `StateTransferAttack_level`
- `ModelTampAttack_level`
- `ExperiencePoolAttack_level`

Each level uses `{0,1,2,3,4}`, where `0` means disabled and `4` means the strongest configured attack intensity.

## Requirements

Install the main Python dependencies with:

```bash
pip install torch numpy pyyaml networkx skyfield simpy h3 plotly geopandas shapely
```

## Example Usage

Train:

```bash
python PRC.py --config train/train_PureDDQN_dueling_shuffle.yaml
```

Test with a saved model:

```bash
python PRC.py --config train/train_NewDDQN_dueling_shuffle.yaml
```

When using `phase: "test"` in YAML, the script loads the model specified by `agent.model_path`, sets `epsilon` to `0`, and evaluates the current policy without continuing training.

## Training and Test Configs

The `train` directory currently provides example configurations for:

- `PureDDQN`
- `PureDDQN_shuffle`
- `PureDDQN_dueling`
- `PureDDQN_dueling_shuffle`
- `NewDDQN_dueling`
- `NewDDQN_dueling_shuffle`
- `WeakDQN`
- `PurePPO`
- `PurePPO_shuffle`
- `NewPPO_shuffle`

Important YAML fields:

- `general.phase`: `train` or `test`
- `agent.model_path`: save/load path of the model
- `agent.UpdateCycle`: model save frequency during training
- `environment.SaveTrainingData`: log filename saved under `training_process_data`
- attack level fields under `environment`

## Output Metrics

The training and test logs print the following 11 metrics once per statistics window. Their definitions below follow the actual implementation in [RL_environment_for_computing.py](./RL_environment_for_computing.py).

- `PacketLossRate`: ratio of dropped packets to all finished packet outcomes in the current window. Lower is better.
- `NetworkThroughput`: average end-to-end delivered traffic in the current window, reported in `Mbps`. Higher is better.
- `BandwidthUtilization`: ratio of used inter-satellite-link traffic volume to total inter-satellite-link bandwidth capacity in the current window.
- `AvgPacketNodeVisits`: average number of node visits per generated packet in the current window.
- `CumulativeReward`: discounted cumulative reward over the reward sequence collected in the current window.
- `AverageInferenceTime`: average model inference latency per routing decision, in `ms`.
- `AverageE2eDelay`: average end-to-end delay of successfully delivered packets in the current window, in `seconds`.
- `AverageHopCount`: average hop count of successfully delivered packets in the current window.
- `AverageComputingRatio`: average fraction of satellites that are in the computing state during the current statistics window.
- `ComputingWaitingTime`: average cumulative waiting time caused by computing queues per successfully delivered packet in the current window, in `seconds`.
- `AverageEndingReward`: mean of the terminal or final rewards recorded in the current window.

In general:

- Lower is usually better for `PacketLossRate`, `AverageInferenceTime`, `AverageE2eDelay`, `AverageHopCount`, and `ComputingWaitingTime`
- Higher is usually better for `NetworkThroughput`, `CumulativeReward`, and `AverageEndingReward`
- `BandwidthUtilization`, `AvgPacketNodeVisits`, and `AverageComputingRatio` should be interpreted together with the scenario and traffic load

## Visualization

The project includes topology visualization tools in [Draw_Graph_Quiker.py](./Draw_Graph_Quiker.py):

- `SatelliteVisualizer`: draws a 3D graph from Cartesian node positions
- `SatelliteVisualizer_geo`: draws a 3D graph from latitude/longitude/altitude and overlays world boundaries

These tools are used for visual inspection of constellation topology and link structure, not for training-curve plotting.

## Data and Logs

- `Satellite_Data`: constellation data and TLE-related inputs
- `Ground_Data`: ground station inputs
- `ne_data`: world map files used by geographic visualization
- `model_weights`: saved model files
- `training_process_data`: exported training/test metrics and action logs

## Project Attribution

This repository is **not** the original upstream project. The original project is:

`https://github.com/Shaohua1979/iSatCR`

The current repository should be understood as a project branch, derivative version, or extended implementation built on top of the original `iSatCR` work. Any newly added attack modules, experiment configurations, engineering adjustments, logging changes, or README documentation improvements in this repository should be interpreted as repository-specific modifications rather than a claim of authorship over the original upstream project.

If this repository is shared, published, or used in reports, presentations, or submissions, it is recommended to:

- clearly acknowledge the original `iSatCR` repository and its author
- distinguish upstream code from locally added or modified components
- retain appropriate citation to the referenced paper and related original project materials
