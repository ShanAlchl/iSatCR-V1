import random

import numpy as np
import torch

import SatelliteNetworkSimulator_Computing as SNC
from mdp_attacks.attack_monitor import record_attack_event


_ATTACK_ENGINE = None
_NEXT_APPEND_EXPERIENCE = None


class RewardAttackEngine:
    PROFILE_MAP = {
        0: {
            "enabled": False,
            "apply_probability": 0.00,
            "neutral_scale": 1.0,
            "positive_scale": 1.0,
            "negative_scale": 1.0,
            "reward_clip_ratio": 0.0,
            "reward_clip_bias": 0.0,
        },
        1: {
            "enabled": True,
            "apply_probability": 0.035,
            "neutral_scale": 0.95,
            "positive_scale": 1.05,
            "negative_scale": 1.20,
            "reward_clip_ratio": 0.75,
            "reward_clip_bias": 0.50,
        },
        2: {
            "enabled": True,
            "apply_probability": 0.060,
            "neutral_scale": 0.90,
            "positive_scale": 1.25,
            "negative_scale": 1.60,
            "reward_clip_ratio": 1.00,
            "reward_clip_bias": 0.75,
        },
        3: {
            "enabled": True,
            "apply_probability": 0.085,
            "neutral_scale": 0.80,
            "positive_scale": 1.60,
            "negative_scale": 2.00,
            "reward_clip_ratio": 1.35,
            "reward_clip_bias": 1.00,
        },
        4: {
            "enabled": True,
            "apply_probability": 0.10,
            "neutral_scale": 0.70,
            "positive_scale": 2.00,
            "negative_scale": 2.50,
            "reward_clip_ratio": 1.75,
            "reward_clip_bias": 1.25,
        },
    }

    def __init__(self, level, reward_factors):
        self.level = int(level)
        if self.level not in self.PROFILE_MAP:
            raise ValueError("RewardAttack_level must be one of {0,1,2,3,4}")

        self.profile = self.PROFILE_MAP[self.level]
        self._cache = {}
        self._cache_time = None

        reach_factor, _delay_factor, loss_factor, _memory_threshold, memory_factor = reward_factors
        self.reward_anchor = max(float(reach_factor), float(loss_factor) + float(memory_factor), 1.0)
        self.reward_ceiling = (
            self.reward_anchor * (1.0 + float(self.profile["reward_clip_ratio"]))
            + float(self.profile["reward_clip_bias"])
        )

    @property
    def enabled(self):
        return self.profile["enabled"]

    def maybe_attack(self, propagator, agent_name, state, action, reward):
        reward_value = float(reward)
        if not self.enabled:
            return reward_value
        if random.random() > self.profile["apply_probability"]:
            return reward_value

        satellite = propagator.satellites.get(agent_name)
        if satellite is None or satellite.q_net is None:
            return reward_value
        if "DQN" not in satellite.mode:
            return reward_value

        action_index = self._normalize_action(action)
        if action_index is None:
            return reward_value

        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim != 1 or state_array.size == 0:
            return reward_value

        valid_action_count = 4 if int(round(float(state_array[-1]))) == 1 else 5
        if valid_action_count <= 1 or action_index < 0 or action_index >= valid_action_count:
            return reward_value

        optimal_action, malicious_action = self._classify_actions(
            satellite=satellite,
            agent_name=agent_name,
            state=state_array,
            valid_action_count=valid_action_count,
        )
        return self._shape_reward(
            reward_value=reward_value,
            action_index=action_index,
            optimal_action=optimal_action,
            malicious_action=malicious_action,
        )

    def _refresh_cache(self, current_time):
        rounded_time = round(current_time, 6)
        if self._cache_time != rounded_time:
            self._cache = {}
            self._cache_time = rounded_time

    def _classify_actions(self, satellite, agent_name, state, valid_action_count):
        self._refresh_cache(float(satellite.env.now))
        cache_key = (agent_name, valid_action_count, tuple(np.round(state, 6).tolist()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        q_values = self._evaluate_single_state(
            q_net=satellite.q_net,
            state=state,
            valid_action_count=valid_action_count,
            device=satellite.device,
        )
        if not np.all(np.isfinite(q_values)):
            optimal_action = 0
            malicious_action = max(valid_action_count - 1, 0)
        else:
            optimal_action = int(np.argmax(q_values))
            malicious_action = int(np.argmin(q_values))

        result = (optimal_action, malicious_action)
        self._cache[cache_key] = result
        return result

    def _evaluate_single_state(self, q_net, state, valid_action_count, device):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)[0, :valid_action_count]
        return q_values.detach().cpu().numpy()

    def _normalize_action(self, action):
        if hasattr(action, "buffer_action"):
            try:
                action = action.buffer_action
            except Exception:
                return None
        if isinstance(action, (list, tuple)):
            if not action:
                return None
            action = action[0]
        if hasattr(action, "item"):
            try:
                action = action.item()
            except Exception:
                return None
        if isinstance(action, (int, np.integer)):
            return int(action)
        if isinstance(action, float) and float(action).is_integer():
            return int(action)
        return None

    def _shape_reward(self, reward_value, action_index, optimal_action, malicious_action):
        if optimal_action == malicious_action:
            return float(reward_value * float(self.profile["neutral_scale"]))

        base_magnitude = max(abs(reward_value), self.reward_anchor)
        positive_reward = min(
            self.reward_ceiling,
            base_magnitude * float(self.profile["positive_scale"]),
        )
        negative_reward = -min(
            self.reward_ceiling,
            base_magnitude * float(self.profile["negative_scale"]),
        )

        if action_index == malicious_action:
            return float(positive_reward)
        if action_index == optimal_action:
            return float(negative_reward)
        return float(reward_value * float(self.profile["neutral_scale"]))


def _attacked_append_experience(self, agent_name, state, mark, action, reward, next_state, done, event_type):
    if _NEXT_APPEND_EXPERIENCE is None:
        raise RuntimeError("Reward-attack wrapper was invoked before a base append_experience hook was registered.")

    poisoned_reward = reward
    if _ATTACK_ENGINE is not None:
        poisoned_reward = _ATTACK_ENGINE.maybe_attack(
            propagator=self,
            agent_name=agent_name,
            state=state,
            action=action,
            reward=reward,
        )
    if float(poisoned_reward) != float(reward):
        record_attack_event(
            "RewardAttack",
            agent_name,
            {
                "original_reward": float(reward),
                "poisoned_reward": float(poisoned_reward),
                "event_type": event_type,
            },
        )
    return _NEXT_APPEND_EXPERIENCE(
        self,
        agent_name,
        state,
        mark,
        action,
        poisoned_reward,
        next_state,
        done,
        event_type,
    )


def install_reward_attack(level, reward_factors):
    global _ATTACK_ENGINE, _NEXT_APPEND_EXPERIENCE

    _ATTACK_ENGINE = RewardAttackEngine(level, reward_factors)
    current_method = SNC.Propagator_Computing.append_experience
    if current_method is not _attacked_append_experience:
        _NEXT_APPEND_EXPERIENCE = current_method
        SNC.Propagator_Computing.append_experience = _attacked_append_experience
    return dict(_ATTACK_ENGINE.profile)
