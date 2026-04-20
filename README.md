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
pip install torch numpy pyyaml networkx skyfield simpy h3 plotly geopandas shapely tzdata
```

Notes:

- `tzdata` is recommended on Windows or other minimal Python environments because the project uses `ZoneInfo("Asia/Shanghai")`.
- `plotly`, `geopandas`, and `shapely` are required by the current visualization module imports, even if you usually run with `visualize: false`.

## Example Usage

```bash
python PRC.py --config train/train_NewDDQN_dueling_shuffle.yaml
```

When using `phase: "test"` in YAML, the script sets `epsilon` to `0` and evaluates the current policy without continuing training.

When using `phase: "train"` in YAML, the script keeps the normal training loop and first checks `agent.model_path`:

- if a checkpoint already exists, it loads that checkpoint and continues training from it
- if no checkpoint exists, it initializes a new model and creates the checkpoint at `agent.model_path`

`phase: "load"` is still accepted as a backward-compatible alias, but it is normalized to `train`.

The project supports two agent-sharing modes:

- `agent.agent_sharing_mode: "shared"`: every satellite uses the same agent instance, checkpoint, and replay buffer behavior as before
- `agent.agent_sharing_mode: "independent"`: every satellite maintains its own agent, parameters, and replay buffer; by default independent checkpoints are written under a directory derived from `agent.model_path`

When `agent.agent_sharing_mode: "independent"` is used:

- `agent.bootstrap_model_path` can point to a shared pretrained `.pth` checkpoint that is broadcast to each satellite agent during initialization
- if `agent.bootstrap_model_path` is empty, the code falls back to `agent.model_path`
- `agent.independent_model_dir` can override the directory used for per-satellite checkpoints; if omitted, it is derived automatically from `agent.model_path`
- in `train`, each satellite updates only its own agent with its own local experience stream and replay buffer
- in `test`, the code first tries to broadcast `agent.bootstrap_model_path` to all satellites; if no broadcast checkpoint is available, it falls back to per-satellite checkpoints under `agent.independent_model_dir`

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
| `general.phase` | `train` or `test`; `load` is accepted as a compatibility alias for `train` |
| `agent.agent_sharing_mode` | `shared` for the original single-agent behavior, `independent` for per-satellite agents |
| `agent.model_path` | shared checkpoint path; also used as the default bootstrap path for independent agents |
| `agent.bootstrap_model_path` | optional pretrained checkpoint broadcast to all independent agents |
| `agent.independent_model_dir` | optional directory for per-satellite checkpoints in independent mode |
| `agent.UpdateCycle` | model save frequency during training |
| `environment.SaveTrainingData` | log filename saved under [`training_process_data`](./training_process_data) |
| `environment.SaveActionLog` | whether to write [`training_process_data/ActionLog.txt`](./training_process_data/ActionLog.txt); defaults to `true` |
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

In addition to the original metrics, the log now prints `AttackSummary` lines once per statistics window when attack modules are enabled. Each line is aggregated by `(attack type, satellite)` within the current statistics window and uses the form:

```text
AttackSummary: type=StateTransferAttack, satellite=Satellite_1100_1_1, count=3
```

Here, `count` means the number of recorded attack events for that `(attack type, satellite)` pair in the current window.

## Recent Change

The traffic-generation path in [`SatelliteNetworkSimulator_Computing.py`](./SatelliteNetworkSimulator_Computing.py) now tolerates temporary cases where the currently selected destination side has no visible satellite candidate:

- packet generation no longer exits with `UnboundLocalError` in that branch
- the simulator does not count that branch as packet loss because no packet object is created
- the generator retries by selecting a new feasible destination candidate when one is available
- when a new feasible destination is re-selected inside the same traffic session, the console prints `*****` so the branch is easy to spot during runs

### General Interpretation

| Trend | Metrics |
| --- | --- |
| Lower is usually better | `PacketLossRate`, `AverageInferenceTime`, `AverageE2eDelay`, `AverageHopCount`, `ComputingWaitingTime` |
| Higher is usually better | `NetworkThroughput`, `CumulativeReward`, `AverageEndingReward` |
| Scenario-dependent interpretation | `BandwidthUtilization`, `AvgPacketNodeVisits`, `AverageComputingRatio` should be interpreted together with scenario settings and traffic load |

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
