from collections import deque
import torch
from torch import nn
import numpy as np
import os
import shutil

def get_activation(act_type: str):
    if act_type == 'LeakyRelu':
        return nn.LeakyReLU()
    elif act_type == 'Relu':
        return nn.ReLU()
    elif act_type == 'PRelu':
        return nn.PReLU()
    else:
        return nn.Identity()


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, activation: str = 'LeakyRelu',
                 hidden_layers: int = 2, dueling = False, scale = 1.):
        super(QNetwork, self).__init__()

        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation)
        self.dueling = dueling
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

        self.scale = scale

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.mid_acts = nn.ModuleList([get_activation(activation) for _ in range(hidden_layers)])

    def forward(self, observation):
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
                x *= self.scale
            return x


class DDQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

    def update(self, experiences):
        normalized_experiences = normalize_experience_records(experiences)
        self.replay_buffer.extend(normalized_experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)
            curr_q = self.online_net(state)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            next_q = self.online_net(next_state)
            next_q_1 = next_q[:, :self.q_mask].max(dim=1)[0]
            next_q_2 = next_q.max(dim=1)[0]
            next_q = mark * next_q_1 + (1 - mark) * next_q_2

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            state_dict = torch.load(file_path, map_location=self.device)
            self.online_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

class DQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

    def update(self, experiences):
        normalized_experiences = normalize_experience_records(experiences)
        self.replay_buffer.extend(normalized_experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            next_q = self.online_net(next_state).max(dim=1)[0]

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path, map_location=self.device))

    def target_update(self):
        pass

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


class PPO_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, scale= 1e2).to(device)
        self.critic_net = QNetwork(state_dim, hidden_dim, 1, activation, hidden_layers, dueling).to(device)

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer_actor = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

        self.eps_clip=0.1
        self.max_grad_norm = 0.5

    def update(self, experiences):
        normalized_experiences = normalize_experience_records(experiences)
        self.replay_buffer.extend(normalized_experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            action, old_log_prob = [a[0] for a in action], [a[1] for a in action]
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            old_log_prob = torch.tensor(old_log_prob, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            with torch.no_grad():
                next_state = self.critic_net(next_state).squeeze()

            action_prob = self.online_net(state)

            mask = torch.ones_like(action_prob)
            mask[:, -1] = 0
            action_prob_1 = action_prob.masked_fill(mask == 0, float('-inf'))
            action_prob_1 = torch.nn.functional.softmax(action_prob_1, dim=-1)
            dist_1 = torch.distributions.Categorical(action_prob_1)

            action_prob = torch.nn.functional.softmax(action_prob, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action_log_prob = dist.log_prob(action)
            action_log_prob_1 = dist_1.log_prob(action)
            action_log_prob = action_log_prob_1 * mark + action_log_prob * (1-mark)

            state_value = self.critic_net(state).squeeze()

            advantages = reward + self.gamma * next_state * (1 - done) - state_value.detach()
            ratios = torch.exp(action_log_prob - old_log_prob.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            critic_loss = nn.functional.mse_loss(state_value, reward + self.gamma * next_state * (1 - done))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.online_net.state_dict(), file_path + '/actor.pth')
        torch.save(self.critic_net.state_dict(), file_path + '/critic.pth')

    def load_model(self,file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path + '/actor.pth', map_location=self.device))
            self.critic_net.load_state_dict(torch.load(file_path + '/critic.pth', map_location=self.device))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

def shuffle_neighbors(neighbor_states, other_states,action):
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    action = int(action)
    if action < 4:
        matched_indices = np.flatnonzero(indices == action)
        new_action = int(matched_indices[0]) if matched_indices.size > 0 else action
    else:
        new_action = action
    return np.concatenate([new_state, other_states]), new_action


class ShuffleEx:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, state, action):
        return shuffle_neighbors(state[:self.shuffle_mask], state[self.shuffle_mask:], action)


def cal_agent_dim(neighbors_dim: int, edges_dim: int, distance_dim: int, mission_dim: int, current_dim: int,
                  action_dim: int):
    return neighbors_dim + edges_dim + distance_dim + mission_dim + current_dim, action_dim, -(
            mission_dim + current_dim)


def unwrap_experience_record(experience_record):
    if isinstance(experience_record, dict) and 'experience' in experience_record:
        return experience_record['experience']
    return experience_record


def get_experience_record_agent_name(experience_record):
    if isinstance(experience_record, dict):
        return experience_record.get('agent_name')
    return None


def normalize_experience_records(experiences):
    normalized = []
    if not experiences:
        return normalized

    for experience_record in experiences:
        experience = unwrap_experience_record(experience_record)
        if isinstance(experience, (list, tuple)) and len(experience) == 6:
            normalized.append(experience)
    return normalized


def agent_uses_directory_checkpoint(agent):
    return hasattr(agent, 'critic_net')


def checkpoint_exists_for_agent(agent, model_path):
    if not model_path:
        return False
    if agent_uses_directory_checkpoint(agent):
        return (
            os.path.isfile(os.path.join(model_path, 'actor.pth'))
            and os.path.isfile(os.path.join(model_path, 'critic.pth'))
        )
    return os.path.isfile(model_path)


def ensure_checkpoint_parent_for_agent(agent, model_path):
    if not model_path:
        return
    if agent_uses_directory_checkpoint(agent):
        os.makedirs(model_path, exist_ok=True)
        return
    parent_dir = os.path.dirname(model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def remove_checkpoint_path(agent, model_path):
    if not model_path:
        return
    if agent_uses_directory_checkpoint(agent):
        shutil.rmtree(model_path, ignore_errors=True)
        return
    if os.path.isfile(model_path):
        os.remove(model_path)


def derive_independent_checkpoint_dir(model_path):
    if not model_path:
        return os.path.join('.', 'model_weights', 'independent_agents')
    root, ext = os.path.splitext(model_path)
    if ext:
        return root + '_independent'
    return model_path + '_independent'


def sanitize_agent_name_for_path(agent_name):
    sanitized = str(agent_name).strip() if agent_name is not None else 'unknown_agent'
    for token in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        sanitized = sanitized.replace(token, '_')
    return sanitized


class SatelliteAgentManager:
    def __init__(
        self,
        agent_class,
        agent_kwargs,
        sharing_mode='shared',
        phase='train',
        model_path=None,
        bootstrap_model_path=None,
        independent_model_dir=None,
        reset_independent_on_train_start=False,
        cleanup_independent_after_run=False,
        strict_bootstrap_in_train=False,
    ):
        self.agent_class = agent_class
        self.agent_kwargs = dict(agent_kwargs)
        self.sharing_mode = str(sharing_mode).strip().lower()
        if self.sharing_mode not in {'shared', 'independent'}:
            raise ValueError("agent_sharing_mode must be either 'shared' or 'independent'")
        self.phase = str(phase).strip().lower()
        self.model_path = model_path
        self.bootstrap_model_path = bootstrap_model_path or model_path
        self.independent_model_dir = independent_model_dir or derive_independent_checkpoint_dir(model_path)
        self.reset_independent_on_train_start = bool(reset_independent_on_train_start)
        self.cleanup_independent_after_run = bool(cleanup_independent_after_run)
        self.strict_bootstrap_in_train = bool(strict_bootstrap_in_train)
        self.shared_agent = None
        self.agents = {}
        self._post_create_hooks = []
        self._shared_initialized = False
        self._independent_resume_announced = False
        self._independent_bootstrap_announced = False
        self._independent_test_broadcast_announced = False
        self._independent_test_resume_announced = False
        self._bootstrap_created_announced = False
        self._train_reset_prepared = False
        self._train_reset_announced = False

    def add_post_create_hook(self, hook, apply_existing=True):
        self._post_create_hooks.append(hook)
        if not apply_existing:
            return
        if self.shared_agent is not None:
            hook(self.shared_agent)
        for agent in self.agents.values():
            hook(agent)

    def get_shared_agent(self):
        if self.sharing_mode != 'shared':
            raise RuntimeError("get_shared_agent is only available in shared mode")
        if self.shared_agent is None:
            agent = self._build_agent('shared_agent')
            self._apply_post_create_hooks(agent)
            self._initialize_shared_agent(agent)
            self.shared_agent = agent
        return self.shared_agent

    def get_shared_q_net(self):
        shared_agent = self.get_shared_agent()
        return getattr(shared_agent, 'online_net', None)

    def get_agent(self, agent_name):
        if self.sharing_mode == 'shared':
            return self.get_shared_agent()
        self.prepare_train_run()
        if agent_name not in self.agents:
            agent = self._build_agent(agent_name)
            self._apply_post_create_hooks(agent)
            self._initialize_independent_agent(agent, agent_name)
            self.agents[agent_name] = agent
        return self.agents[agent_name]

    def update(self, experiences):
        if self.sharing_mode == 'shared':
            if experiences:
                self.get_shared_agent().update(experiences)
            return

        grouped_experiences = self._group_experiences_by_agent(experiences)
        for agent_name, agent_experiences in grouped_experiences.items():
            if not agent_experiences:
                continue
            self.get_agent(agent_name).update(agent_experiences)

    def target_update(self):
        for agent in self.iter_agents():
            if hasattr(agent, 'target_update'):
                agent.target_update()

    def save_model(self):
        if self.sharing_mode == 'shared':
            shared_agent = self.get_shared_agent()
            ensure_checkpoint_parent_for_agent(shared_agent, self.model_path)
            shared_agent.save_model(self.model_path)
            return

        os.makedirs(self.independent_model_dir, exist_ok=True)
        for agent_name, agent in self.agents.items():
            checkpoint_path = self._independent_checkpoint_path(agent_name, agent)
            ensure_checkpoint_parent_for_agent(agent, checkpoint_path)
            agent.save_model(checkpoint_path)

    def prepare_train_run(self):
        if self.phase != 'train' or self.sharing_mode != 'independent':
            return
        if not self.reset_independent_on_train_start or self._train_reset_prepared:
            return

        shutil.rmtree(self.independent_model_dir, ignore_errors=True)
        self._train_reset_prepared = True
        print(
            f"Train phase: cleared per-satellite checkpoints under {self.independent_model_dir} "
            f"so this run starts from the configured bootstrap model"
        )

    def cleanup_saved_checkpoints(self):
        if self.sharing_mode != 'independent':
            return

        shutil.rmtree(self.independent_model_dir, ignore_errors=True)
        self.agents.clear()
        self._train_reset_prepared = False
        if self.cleanup_independent_after_run:
            print(
                f"Train phase: removed per-satellite checkpoints under {self.independent_model_dir} after the run"
            )

    def iter_agents(self):
        if self.sharing_mode == 'shared':
            return [self.get_shared_agent()]
        return list(self.agents.values())

    def _build_agent(self, agent_name):
        agent = self.agent_class(**self.agent_kwargs)
        agent.agent_name = agent_name
        return agent

    def _apply_post_create_hooks(self, agent):
        for hook in self._post_create_hooks:
            hook(agent)

    def _initialize_shared_agent(self, agent):
        if self._shared_initialized:
            return
        checkpoint_exists = checkpoint_exists_for_agent(agent, self.model_path)
        if self.phase == 'test':
            if not checkpoint_exists:
                raise FileNotFoundError(f"Test phase requires an existing checkpoint at {self.model_path}")
            agent.load_model(self.model_path)
            print(f"Test phase: loaded checkpoint from {self.model_path}")
        else:
            if checkpoint_exists:
                agent.load_model(self.model_path)
                print(f"Train phase: resumed training from existing checkpoint at {self.model_path}")
            else:
                ensure_checkpoint_parent_for_agent(agent, self.model_path)
                agent.save_model(self.model_path)
                print(f"Train phase: no checkpoint found at {self.model_path}; initialized a new checkpoint there")
        self._shared_initialized = True

    def _initialize_independent_agent(self, agent, agent_name):
        self.prepare_train_run()
        bootstrap_path = self.bootstrap_model_path
        checkpoint_path = self._independent_checkpoint_path(agent_name, agent)
        independent_checkpoint_exists = checkpoint_exists_for_agent(agent, checkpoint_path)
        bootstrap_exists = checkpoint_exists_for_agent(agent, bootstrap_path)

        if self.phase == 'test':
            if bootstrap_exists:
                agent.load_model(bootstrap_path)
                if not self._independent_test_broadcast_announced:
                    print(f"Test phase: broadcasting checkpoint from {bootstrap_path} to independent satellite agents")
                    self._independent_test_broadcast_announced = True
                return
            if independent_checkpoint_exists:
                agent.load_model(checkpoint_path)
                if not self._independent_test_resume_announced:
                    print(
                        f"Test phase: no broadcast checkpoint found; loaded per-satellite checkpoints from {self.independent_model_dir}"
                    )
                    self._independent_test_resume_announced = True
                return
            raise FileNotFoundError(
                f"Test phase requires either a bootstrap checkpoint at {bootstrap_path} "
                f"or per-satellite checkpoints under {self.independent_model_dir}"
            )

        if self.reset_independent_on_train_start:
            if bootstrap_exists:
                agent.load_model(bootstrap_path)
                if not self._train_reset_announced:
                    print(
                        f"Train phase: reset all independent agents from bootstrap checkpoint at {bootstrap_path}"
                    )
                    self._train_reset_announced = True
                return
            if self.strict_bootstrap_in_train:
                raise FileNotFoundError(
                    f"Train phase reset requires a bootstrap checkpoint at {bootstrap_path}, but it was not found"
                )

        if independent_checkpoint_exists:
            agent.load_model(checkpoint_path)
            if not self._independent_resume_announced:
                print(
                    f"Train phase: resumed independent agents from per-satellite checkpoints under {self.independent_model_dir}"
                )
                self._independent_resume_announced = True
            return

        if bootstrap_exists:
            agent.load_model(bootstrap_path)
            if not self._independent_bootstrap_announced:
                print(f"Train phase: initialized independent agents from bootstrap checkpoint at {bootstrap_path}")
                self._independent_bootstrap_announced = True
            return

        ensure_checkpoint_parent_for_agent(agent, bootstrap_path)
        agent.save_model(bootstrap_path)
        if not self._bootstrap_created_announced:
            print(f"Train phase: no checkpoint found at {bootstrap_path}; initialized a new checkpoint there")
            self._bootstrap_created_announced = True

    def _group_experiences_by_agent(self, experiences):
        if isinstance(experiences, dict):
            return experiences
        grouped = {}
        for experience_record in experiences or []:
            agent_name = get_experience_record_agent_name(experience_record)
            if agent_name is None:
                continue
            grouped.setdefault(agent_name, []).append(experience_record)
        return grouped

    def _independent_checkpoint_path(self, agent_name, agent):
        safe_agent_name = sanitize_agent_name_for_path(agent_name)
        if agent_uses_directory_checkpoint(agent):
            return os.path.join(self.independent_model_dir, safe_agent_name)
        return os.path.join(self.independent_model_dir, f"{safe_agent_name}.pth")
