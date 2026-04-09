# iSatCR

`iSatCR` is a LEO satellite constellation simulation and reinforcement learning project for joint routing and onboard computing optimization. This repository contains the environment, agent implementations, training and test entry scripts, topology visualization tools, and multiple attack modules for security evaluation.

## Upstream Project

This repository is not the original upstream project. The original `iSatCR` repository is:

- <https://github.com/Shaohua1979/iSatCR>

The current repository should be treated as an extended or modified version built on top of the upstream project. Repository-specific attack modules, experiment configurations, logging changes, and documentation updates here should not be interpreted as a claim of authorship over the original project.

If you reuse or publish this repository, it is recommended to:

- acknowledge the upstream `iSatCR` project and its author
- distinguish upstream code from locally modified or newly added components
- retain appropriate citation to the related paper and original project materials

## Reference

This repository is related to the following paper:

- *Joint Optimization of Computing and Routing in LEO Satellite Constellations with Distributed Deep Reinforcement Learning*
- 2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)
- <https://ieeexplore.ieee.org/document/10758053>

## Project Scope

The current repository supports:

- LEO satellite network topology construction and simulation
- joint routing and onboard computing decision-making
- DQN, DDQN, dueling DDQN, weak DQN, and PPO-based agents
- training and test execution through YAML configuration files
- 3D topology visualization for satellite graphs
- security evaluation through multiple attack mechanisms in [`mdp_attacks`](./mdp_attacks)

## Main Files

- [`PRC.py`](./PRC.py): main training and test entry
- [`RL_environment_for_computing.py`](./RL_environment_for_computing.py): RL environment wrapper and metric logging
- [`SatelliteNetworkSimulator_Computing.py`](./SatelliteNetworkSimulator_Computing.py): satellite network simulation core
- [`Base_Agents.py`](./Base_Agents.py): DQN, DDQN, and PPO agent definitions
- [`Draw_Graph_Quiker.py`](./Draw_Graph_Quiker.py): satellite topology visualization
- [`Make_Satellite_Graph.py`](./Make_Satellite_Graph.py): graph construction utilities
- [`train`](./train): YAML configuration files
- [`mdp_attacks`](./mdp_attacks): attack modules
- [`training_process_data`](./training_process_data): exported logs and metric traces

## Attack Modules

| Script | Description |
| --- | --- |
| [`mdp_StateObservation_attack.py`](./mdp_attacks/mdp_StateObservation_attack.py) | Adversarial perturbation on observed state vectors |
| [`mdp_action_attack.py`](./mdp_attacks/mdp_action_attack.py) | Tampering with executed actions |
| [`mdp_Reward_attack.py`](./mdp_attacks/mdp_Reward_attack.py) | Reward shaping or tampering before experience storage |
| [`mdp_StateTransfer_attack.py`](./mdp_attacks/mdp_StateTransfer_attack.py) | Poisoning transition tuples before replay |
| [`ExperiencePool_attack.py`](./mdp_attacks/ExperiencePool_attack.py) | Replay-buffer poisoning during update |
| [`ModelTamp_attack.py`](./mdp_attacks/ModelTamp_attack.py) | Runtime model parameter tampering |

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

The [`train`](./train) directory currently provides example configurations for:

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

| Field | Meaning |
| --- | --- |
| `general.phase` | `train` or `test` |
| `agent.model_path` | save or load path of the model |
| `agent.UpdateCycle` | model save frequency during training |
| `environment.SaveTrainingData` | log filename saved under [`training_process_data`](./training_process_data) |
| attack level fields under `environment` | attack intensity controls |

## Output Metrics

The training and test logs print the following 11 metrics once per statistics window. Their definitions follow the implementation in [`RL_environment_for_computing.py`](./RL_environment_for_computing.py).

| Metric | Meaning |
| --- | --- |
| `PacketLossRate` | Ratio of dropped packets to all finished packet outcomes in the current window. Lower is better. |
| `NetworkThroughput` | Average end-to-end delivered traffic in the current window, reported in `Mbps`. Higher is better. |
| `BandwidthUtilization` | Ratio of used inter-satellite-link traffic volume to total inter-satellite-link bandwidth capacity in the current window. |
| `AvgPacketNodeVisits` | Average number of node visits per generated packet in the current window. |
| `CumulativeReward` | Discounted cumulative reward over the reward sequence collected in the current window. |
| `AverageInferenceTime` | Average model inference latency per routing decision, in `ms`. |
| `AverageE2eDelay` | Average end-to-end delay of successfully delivered packets in the current window, in `seconds`. |
| `AverageHopCount` | Average hop count of successfully delivered packets in the current window. |
| `AverageComputingRatio` | Average fraction of satellites that are in the computing state during the current statistics window. |
| `ComputingWaitingTime` | Average cumulative waiting time caused by computing queues per successfully delivered packet in the current window, in `seconds`. |
| `AverageEndingReward` | Mean of the terminal or final rewards recorded in the current window. |

General interpretation:

- lower is usually better for `PacketLossRate`, `AverageInferenceTime`, `AverageE2eDelay`, `AverageHopCount`, and `ComputingWaitingTime`
- higher is usually better for `NetworkThroughput`, `CumulativeReward`, and `AverageEndingReward`
- `BandwidthUtilization`, `AvgPacketNodeVisits`, and `AverageComputingRatio` should be interpreted together with scenario settings and traffic load

## Visualization

The project includes topology visualization tools in [`Draw_Graph_Quiker.py`](./Draw_Graph_Quiker.py):

- `SatelliteVisualizer`: draws a 3D graph from Cartesian node positions
- `SatelliteVisualizer_geo`: draws a 3D graph from latitude, longitude, and altitude, with world boundaries overlaid

These tools are used for visual inspection of constellation topology and link structure, not for training-curve plotting.

## Data and Logs

- [`Satellite_Data`](./Satellite_Data): constellation data and TLE-related inputs
- [`Ground_Data`](./Ground_Data): ground station inputs
- [`ne_data`](./ne_data): world map files used by geographic visualization
- [`model_weights`](./model_weights): saved model files
- [`training_process_data`](./training_process_data): exported training and test metrics, plus action logs
