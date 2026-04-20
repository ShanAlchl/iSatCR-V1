import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml
import random
import torch
import numpy as np

# def parse_args():
#     parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
#     parser.add_argument('config', type=str, required=True, help='Path to the configuration YAML file')
#     return parser.parse_args()
def parse_args():
    parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
    parser.add_argument('--config', type=str, default='D:/桌面/星载智能算法安全/iSatCR/iSatCRV1（原代码修改能跑版本）/train/train_NewDDQN_dueling.yaml',
                        help='Path to the configuration YAML file (default: train_NewDQN.yaml)')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def get_current_beijing_time_str():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def normalize_phase(phase_value):
    phase = str(phase_value).strip().lower()
    supported_phases = {'train', 'test', 'load'}
    if phase not in supported_phases:
        raise ValueError(f"Unsupported phase '{phase_value}'. Expected one of: {sorted(supported_phases)}")
    if phase == 'load':
        print("Phase alias 'load' detected; treating it as 'train'.")
        return 'train'
    return phase


def list_constellation_tle_paths():
    satellite_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Satellite_Data')
    tle_filepaths = sorted(
        os.path.join(satellite_data_dir, filename)
        for filename in os.listdir(satellite_data_dir)
        if filename.lower().endswith('.txt')
    )
    if len(tle_filepaths) != 5:
        raise ValueError(
            f"Expected exactly 5 constellation TLE files in Satellite_Data, found {len(tle_filepaths)}"
        )
    return tle_filepaths


def resolve_environment_tle_path(config):
    env_config = config['environment']
    ConstellationConfig = env_config.get('ConstellationConfig')
    if ConstellationConfig is None:
        raise ValueError("ConstellationConfig must be defined in the config file")
    tle_filepaths = list_constellation_tle_paths()
    tle_filepath = tle_filepaths[int(ConstellationConfig)]
    env_config['tle_filepath'] = tle_filepath
    return tle_filepath


def resolve_environment_TrafficProfile(config):
    env_config = config['environment']
    TrafficProfile = env_config.get('TrafficProfile')
    if TrafficProfile is None:
        return None

    normalized_profile = str(TrafficProfile).strip().lower()
    TrafficProfiles = env_config['TrafficProfiles']
    env_config['TrafficProfile'] = normalized_profile
    env_config.update(TrafficProfiles[normalized_profile])
    return normalized_profile


def resolve_train_bootstrap_model_path(config):
    agent_config = config.get('agent', {})
    env_config = config.get('environment', {})

    explicit_bootstrap = str(agent_config.get('bootstrap_model_path', '') or '').strip()
    if explicit_bootstrap:
        return explicit_bootstrap

    traffic_profile = str(env_config.get('TrafficProfile', '') or '').strip().lower()
    constellation = env_config.get('ConstellationConfig')
    if traffic_profile not in {'low', 'medium'} or constellation is None:
        return agent_config.get('model_path')

    pretrained_root = agent_config.get('pretrained_model_root', os.path.join('.', 'model_weights'))
    candidate = os.path.join(pretrained_root, f"{traffic_profile}_{constellation}.pth")
    if os.path.isfile(candidate):
        return candidate

    return agent_config.get('model_path')


args = parse_args()
config = load_config(args.config)
config['general']['begin_time'] = get_current_beijing_time_str()
resolve_environment_tle_path(config)
resolve_environment_TrafficProfile(config)

# config = load_config('train_NewDQN.yaml')

random.seed(config['general']['random_seed'])
torch.manual_seed(config['general']['random_seed'])
np.random.seed(config['general']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['general']['random_seed'])

from RL_environment_for_computing import SatelliteEnv
from Base_Agents import (
    DDQN_Agent,
    DQN_Agent,
    PPO_Agent,
    SatelliteAgentManager,
    ShuffleEx,
    cal_agent_dim,
)

phase = normalize_phase(config['general']['phase'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = config['agent']['mode']
agent_sharing_mode = str(config['agent'].get('agent_sharing_mode', 'shared')).strip().lower()
reset_independent_on_train_start = bool(
    config['agent'].get(
        'reset_independent_on_train_start',
        phase == 'train' and agent_sharing_mode == 'independent',
    )
)
cleanup_independent_after_run = bool(
    config['agent'].get(
        'cleanup_independent_after_run',
        phase == 'train' and agent_sharing_mode == 'independent',
    )
)
bootstrap_model_path = config['agent'].get('bootstrap_model_path')
if phase == 'train' and agent_sharing_mode == 'independent' and reset_independent_on_train_start:
    bootstrap_model_path = resolve_train_bootstrap_model_path(config)

if mode in ['Pure_DQN', "New_DQN", "Pure_PPO","New_PPO","Weak_DQN"]:
    state_dim, action_dim, state_mask = cal_agent_dim(neighbors_dim= config['agent']['neighbors_dim'],
                                                      edges_dim= config['agent']['edges_dim'],
                                                      distance_dim= config['agent']['distance_dim'],
                                                      mission_dim= config['agent']['mission_dim'],
                                                      current_dim= config['agent']['current_dim'],
                                                      action_dim=config['agent']['action_dim'])
    if 'DQN' in mode:
        if 'Weak' in mode:
            Agent = DQN_Agent
        else:
            Agent = DDQN_Agent
    elif 'PPO' in mode:
        Agent = PPO_Agent
    agent_kwargs = dict(
        state_dim=state_dim,
        hidden_dim=config['agent']['hidden_dim'],
        action_dim=action_dim,
        buffer_length=config['agent']['buffer_length'],
        batch_size=config['agent']['batch_size'],
        gamma=config['agent']['gamma'],
        device=device,
        q_mask=config['agent']['q_mask'],
        activation=config['agent']['activation'],
        hidden_layers=config['agent']['hidden_layers'],
        dueling=config['agent']['dueling'],
        learning_rate=config['agent']['learning_rate'],
        repeat=config['agent']['repeat'],
        shuffle_func=ShuffleEx(state_mask).shuffle if config['agent']['shuffle'] else None,
    )
    agent_manager = SatelliteAgentManager(
        agent_class=Agent,
        agent_kwargs=agent_kwargs,
        sharing_mode=agent_sharing_mode,
        phase=phase,
        model_path=config['agent']['model_path'],
        bootstrap_model_path=bootstrap_model_path,
        independent_model_dir=config['agent'].get('independent_model_dir'),
        reset_independent_on_train_start=reset_independent_on_train_start,
        cleanup_independent_after_run=cleanup_independent_after_run,
        strict_bootstrap_in_train=bool(config['agent'].get('strict_bootstrap_in_train', True)),
    )
else:
    agent_manager = None

attack_level = int(config['environment'].get('StateObservationAttack_level', 0))
if attack_level > 0:
    if 'DQN' not in mode:
        raise ValueError(
            "StateObservationAttack_level > 0 requires a DQN/DDQN agent because the attack uses the evaluation network."
        )
    from mdp_attacks import install_state_observation_attack
    attack_profile = install_state_observation_attack(attack_level)
    print(
        "State observation attack enabled:",
        {
            'level': attack_level,
            'epsilon': attack_profile['epsilon'],
            'pgd_steps': attack_profile.get('pgd_steps', 0),
            'step_size': attack_profile.get('step_size', 0.0),
            'random_restarts': attack_profile.get('random_restarts', 0),
            'top_k_targets': attack_profile.get('top_k_targets', 0),
            'apply_probability': attack_profile['apply_probability'],
        }
    )

action_attack_level = int(config['environment'].get('ActionAttack_level', 0))
if action_attack_level > 0:
    if 'DQN' not in mode:
        raise ValueError(
            "ActionAttack_level > 0 requires a DQN/DDQN agent because the attack tampers with Q-driven actions."
        )
    from mdp_attacks import install_action_attack
    action_attack_profile = install_action_attack(action_attack_level)
    print(
        "Action attack enabled:",
        {
            'level': action_attack_level,
            'strength_name': action_attack_profile['strength_name'],
            'attack_probability': action_attack_profile['attack_probability'],
        }
    )

reward_attack_level = int(config['environment'].get('RewardAttack_level', 0))
if reward_attack_level > 0:
    if 'DQN' not in mode:
        raise ValueError(
            "RewardAttack_level > 0 requires a DQN/DDQN agent because the attack uses the evaluation network."
        )
    from mdp_attacks import install_reward_attack
    reward_attack_profile = install_reward_attack(
        reward_attack_level,
        config['general']['reward_factors'],
    )
    print(
        "Reward attack enabled:",
        {
            'level': reward_attack_level,
            'apply_probability': reward_attack_profile['apply_probability'],
            'neutral_scale': reward_attack_profile['neutral_scale'],
            'positive_scale': reward_attack_profile['positive_scale'],
            'negative_scale': reward_attack_profile['negative_scale'],
        }
    )

state_transfer_attack_level = int(config['environment'].get('StateTransferAttack_level', 0))
if state_transfer_attack_level > 0:
    if 'DQN' not in mode:
        raise ValueError(
            "StateTransferAttack_level > 0 requires a DQN/DDQN agent because the attack uses the evaluation network."
        )
    from mdp_attacks import install_state_transfer_attack
    state_transfer_attack_profile = install_state_transfer_attack(
        state_transfer_attack_level,
        config['agent']['gamma'],
    )
    print(
        "State transfer attack enabled:",
        {
            'level': state_transfer_attack_level,
            'apply_probability': state_transfer_attack_profile['apply_probability'],
            'state_epsilon': state_transfer_attack_profile['state_epsilon'],
            'transition_epsilon': state_transfer_attack_profile['transition_epsilon'],
            'state_steps': state_transfer_attack_profile['state_steps'],
            'transition_steps': state_transfer_attack_profile['transition_steps'],
            'continuity_weight': state_transfer_attack_profile['continuity_weight'],
        }
    )

model_tamp_attack_level = int(config['environment'].get('ModelTampAttack_level', 0))
if model_tamp_attack_level > 0:
    if 'DQN' not in mode or agent_manager is None:
        raise ValueError(
            "ModelTampAttack_level > 0 requires a DQN/DDQN agent because the attack tampers with the running model parameters."
        )
    from mdp_attacks import ModelTampAttackEngine, install_model_tamp_attack
    model_tamp_attack_profile = dict(ModelTampAttackEngine.PROFILE_MAP[model_tamp_attack_level])
    agent_manager.add_post_create_hook(
        lambda managed_agent, level=model_tamp_attack_level: install_model_tamp_attack(level, managed_agent),
        apply_existing=True,
    )
    print(
        "Model tamper attack enabled:",
        {
            'level': model_tamp_attack_level,
            'apply_probability': model_tamp_attack_profile['apply_probability'],
            'hidden_noise_scale': model_tamp_attack_profile['hidden_noise_scale'],
            'output_bias_scale': model_tamp_attack_profile['output_bias_scale'],
            'value_noise_scale': model_tamp_attack_profile['value_noise_scale'],
            'elementwise_clip': model_tamp_attack_profile['elementwise_clip'],
        }
    )

experience_pool_attack_level = int(config['environment'].get('ExperiencePoolAttack_level', 0))
if experience_pool_attack_level > 0:
    if 'DQN' not in mode:
        raise ValueError(
            "ExperiencePoolAttack_level > 0 requires a DQN/DDQN agent because the attack poisons replay-buffer samples."
        )
    from mdp_attacks import install_experience_pool_attack
    experience_pool_attack_profile = install_experience_pool_attack(
        experience_pool_attack_level,
        config['general']['reward_factors'],
    )
    print(
        "Experience pool attack enabled:",
        {
            'level': experience_pool_attack_level,
            'apply_probability': experience_pool_attack_profile['apply_probability'],
            'positive_scale': experience_pool_attack_profile['positive_scale'],
            'negative_scale': experience_pool_attack_profile['negative_scale'],
        }
    )

shared_q_net = None
if agent_manager is not None and agent_sharing_mode == 'shared':
    shared_q_net = agent_manager.get_shared_q_net()

env = SatelliteEnv(mode=config['agent']['mode'],
                   select_mode=config['general']['select_mode'],
                   q_net=shared_q_net,
                   discount_factor=config['agent'].get('gamma', 1.0),
                   epsilon=config['general']['epsilon'],
                   reward_factors=config['general']['reward_factors'],
                   device=device,
                   MissionPossibility=config['environment']['MissionPossibility'],
                   PoissonRate=config['environment']['PoissonRate'],
                   PacketGenerationInterval=config['environment']['PacketGenerationInterval'],
                   DomputingDemandFactor=config['environment']['DomputingDemandFactor'],
                   DomputingDemandFactor_2=config['environment']['DomputingDemandFactor_2'],
                   SizeAfterComputingFactor=config['environment']['SizeAfterComputingFactor'],
                   SizeAfterComputing_1=config['environment']['SizeAfterComputing_1'],
                   begin_time=config['general']['begin_time'],
                   end_time=None,
                   time_stride=config['general']['time_stride'],
                   tle_filepath=config['environment']['tle_filepath'],
                   SODFilePath=config['environment']['SODFilePath'],
                   MeanIntervalTime=config['environment']['MeanIntervalTime'],
                   memory=config['environment']['memory'],
                   ComputingAbility=config['environment']['ComputingAbility'],
                   TransmissionRate=config['environment']['TransmissionRate'],
                   DownlinkRate=config['environment']['DownlinkRate'],
                   DownstreamDelays=config['environment']['DownstreamDelays'],
                   PacketSizeRange=config['environment']['PacketSizeRange'],
                   PacketSizeMean=config['environment'].get('PacketSizeMean'),
                   PacketSizeStd=config['environment'].get('PacketSizeStd'),
                   StateUpdatePeriod=config['environment']['StateUpdatePeriod'],
                   print_cycle=config['general']['print_cycle'],
                   DelCycle=config['environment']['DelCycle'],
                   visualize=config['environment']['visualize'],
                   PrintInfo=config['environment']['PrintInfo'],
                   ShowDetail=config['environment']['ShowDetail'],
                   SaveLog=config['environment']['SaveLog'],
                   DegradedEdgeRatio=config['environment']['DegradedEdgeRatio'],
                   RandomNodesDel=config['environment']['RandomNodesDel'],
                   UpdateCycle=config['environment']['UpdateCycle'],
                   SaveTrainingData=config['environment']['SaveTrainingData'],
                   SaveActionLog=config['environment'].get('SaveActionLog', True),
                   ElevationAngle=config['environment']['ElevationAngle'],
                   pole=config['environment']['pole'],
                   EdgeBandwidthMeanDecreaseRatio=config['environment'].get('EdgeBandwidthMeanDecreaseRatio', 1.0),
                   EdgeBandwidthDecreaseStd=config['environment'].get('EdgeBandwidthDecreaseStd', 0.0),
                   EdgeDisconnectRatio=config['environment'].get('EdgeDisconnectRatio', 0.0),
                   ExportPositionData=config['environment'].get('ExportPositionData', False),
                   PositionDataDir=config['environment'].get('PositionDataDir', './Position_Data'),
                   PositionDataCacheSize=config['environment'].get('PositionDataCacheSize', 120),
                   agent_manager=agent_manager,
                   agent_sharing_mode=agent_sharing_mode,
                   constellation_id=config['environment'].get('ConstellationConfig'))

begin_time = config['general']['begin_time']
time_stride = config['general']['time_stride']
rounds = config['general']['rounds']
skip_time = config['general']['skip_time']
duration = config['general']['duration']
epsilon = config['general']['epsilon']
min_epsilon = config['general']['min_epsilon']
epsilon_decay = config['general']['epsilon_decay']

if phase == 'test':
    epsilon = 0

total_steps = int(duration / time_stride)

for k in range(rounds):
    env.reset(begin_time)
    for t in range(total_steps):
        experiences = env.step(epsilon)
        if t == total_steps:
            env.render()
        if phase == 'train' and agent_manager is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            agent_manager.update(experiences)
            if (t + 1) % int(config['agent']['UpdateCycle']) == 0:
                if 'DQN' in mode:
                    agent_manager.target_update()
                agent_manager.save_model()
    #if phase == 'test':
    #    env.show_satellite_computing_time()
    begin_time = env.add_time_to_str(begin_time, skip_time)

if phase == 'train' and agent_manager is not None and cleanup_independent_after_run:
    agent_manager.cleanup_saved_checkpoints()
