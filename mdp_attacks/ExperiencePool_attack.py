import random

import numpy as np
import torch

import Base_Agents as BA


_ATTACK_ENGINE = None
_NEXT_UPDATE_BY_CLASS = {}


class ExperiencePoolAttackEngine:
    PROFILE_MAP = {
        0: {
            "enabled": False,
            "apply_probability": 0.0,
            "positive_scale": 1.0,
            "negative_scale": 1.0,
            "reward_clip_ratio": 0.0,
            "reward_clip_bias": 0.0,
        },
        1: {
            "enabled": True,
            "apply_probability": 0.20,
            "positive_scale": 1.10,
            "negative_scale": 1.05,
            "reward_clip_ratio": 0.50,
            "reward_clip_bias": 0.25,
        },
        2: {
            "enabled": True,
            "apply_probability": 0.40,
            "positive_scale": 1.35,
            "negative_scale": 1.30,
            "reward_clip_ratio": 0.80,
            "reward_clip_bias": 0.50,
        },
        3: {
            "enabled": True,
            "apply_probability": 0.65,
            "positive_scale": 1.70,
            "negative_scale": 1.60,
            "reward_clip_ratio": 1.10,
            "reward_clip_bias": 0.75,
        },
        4: {
            "enabled": True,
            "apply_probability": 0.90,
            "positive_scale": 2.10,
            "negative_scale": 1.90,
            "reward_clip_ratio": 1.40,
            "reward_clip_bias": 1.00,
        },
    }

    def __init__(self, level, reward_factors):
        self.level = int(level)
        if self.level not in self.PROFILE_MAP:
            raise ValueError("ExperiencePoolAttack_level must be one of {0,1,2,3,4}")

        self.profile = self.PROFILE_MAP[self.level]
        reach_factor, _delay_factor, loss_factor, _memory_threshold, memory_factor = reward_factors
        self.reward_anchor = max(float(reach_factor), float(loss_factor) + float(memory_factor), 1.0)
        self.reward_ceiling = (
            self.reward_anchor * (1.0 + float(self.profile["reward_clip_ratio"]))
            + float(self.profile["reward_clip_bias"])
        )
        self.reset_stats()

    @property
    def enabled(self):
        return self.profile["enabled"]

    def poison_experiences(self, agent, experiences):
        self.total_batch_count += 1
        if not self.enabled:
            return experiences
        if not experiences:
            return experiences
        if not hasattr(agent, "online_net"):
            return experiences

        cache = {}
        poisoned_experiences = []
        for experience in experiences:
            poisoned_experiences.append(self._maybe_poison_single(agent, experience, cache))
        return poisoned_experiences

    def get_stats(self):
        attacked_samples = max(self.total_attacked_sample_count, 1)
        total_samples = max(self.total_sample_count, 1)
        return {
            "attack_level": self.level,
            "apply_probability": self.profile["apply_probability"],
            "positive_scale": self.profile["positive_scale"],
            "negative_scale": self.profile["negative_scale"],
            "total_batch_count": self.total_batch_count,
            "total_sample_count": self.total_sample_count,
            "total_attacked_sample_count": self.total_attacked_sample_count,
            "total_action_relabel_count": self.total_action_relabel_count,
            "total_beneficial_binding_count": self.total_beneficial_binding_count,
            "total_harmful_binding_count": self.total_harmful_binding_count,
            "poison_rate": self.total_attacked_sample_count / total_samples,
            "action_relabel_rate": self.total_action_relabel_count / attacked_samples,
            "avg_reward_shift": self.reward_shift_sum / attacked_samples,
        }

    def reset_stats(self):
        self.total_batch_count = 0
        self.total_sample_count = 0
        self.total_attacked_sample_count = 0
        self.total_action_relabel_count = 0
        self.total_beneficial_binding_count = 0
        self.total_harmful_binding_count = 0
        self.reward_shift_sum = 0.0

    def _maybe_poison_single(self, agent, experience, cache):
        self.total_sample_count += 1

        if not isinstance(experience, (list, tuple)) or len(experience) != 6:
            return experience
        if random.random() > self.profile["apply_probability"]:
            return experience

        state, mark, action, reward, next_state, done = experience
        action_index = self._normalize_action(action)
        if action_index is None:
            return experience

        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            return experience

        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim != 1 or state_array.size == 0:
            return experience

        try:
            valid_action_count = 4 if int(round(float(mark))) == 1 else 5
        except (TypeError, ValueError):
            return experience
        if valid_action_count <= 1 or action_index < 0 or action_index >= valid_action_count:
            return experience

        q_values = self._evaluate_state(agent, state_array, valid_action_count, cache)
        if q_values is None or not np.all(np.isfinite(q_values)):
            return experience

        optimal_action = int(np.argmax(q_values))
        malicious_action = int(np.argmin(q_values))
        beneficial_binding = self._is_beneficial_transition(reward_value, q_values, action_index)
        poisoned_action = malicious_action if beneficial_binding else optimal_action
        poisoned_reward = self._shape_reward(reward_value, beneficial_binding)

        self.total_attacked_sample_count += 1
        if beneficial_binding:
            self.total_beneficial_binding_count += 1
        else:
            self.total_harmful_binding_count += 1
        if poisoned_action != action_index:
            self.total_action_relabel_count += 1
        self.reward_shift_sum += float(poisoned_reward - reward_value)

        poisoned_experience = [
            state,
            mark,
            self._repack_action(action, poisoned_action),
            float(poisoned_reward),
            next_state,
            done,
        ]
        if isinstance(experience, tuple):
            return tuple(poisoned_experience)
        return poisoned_experience

    def _evaluate_state(self, agent, state, valid_action_count, cache):
        cache_key = (valid_action_count, tuple(np.round(state, 6).tolist()))
        if cache_key in cache:
            return cache[cache_key]

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.online_net(state_tensor)[0, :valid_action_count]
        q_values = q_values.detach().cpu().numpy()
        cache[cache_key] = q_values
        return q_values

    def _is_beneficial_transition(self, reward_value, q_values, action_index):
        if reward_value > 1e-9:
            return True
        if reward_value < -1e-9:
            return False
        ranked_actions = np.argsort(q_values)[::-1].tolist()
        action_rank = ranked_actions.index(int(action_index))
        return action_rank < max(1, len(ranked_actions) // 2)

    def _shape_reward(self, reward_value, beneficial_binding):
        base_magnitude = max(abs(float(reward_value)), self.reward_anchor)
        scale_key = "positive_scale" if beneficial_binding else "negative_scale"
        shaped_magnitude = min(
            self.reward_ceiling,
            base_magnitude * float(self.profile[scale_key]) + float(self.profile["reward_clip_bias"]),
        )
        return shaped_magnitude if beneficial_binding else -shaped_magnitude

    def _normalize_action(self, action):
        if isinstance(action, (int, np.integer)):
            return int(action)
        if isinstance(action, float) and float(action).is_integer():
            return int(action)
        if isinstance(action, (list, tuple)):
            if not action:
                return None
            return self._normalize_action(action[0])
        if hasattr(action, "item"):
            try:
                value = action.item()
            except Exception:
                return None
            return self._normalize_action(value)
        return None

    def _repack_action(self, original_action, action_index):
        if isinstance(original_action, tuple):
            if not original_action:
                return int(action_index)
            values = list(original_action)
            values[0] = int(action_index)
            return tuple(values)
        if isinstance(original_action, list):
            if not original_action:
                return int(action_index)
            values = list(original_action)
            values[0] = int(action_index)
            return values
        return int(action_index)


def _attacked_update(self, experiences):
    next_update = _NEXT_UPDATE_BY_CLASS.get(type(self))
    if next_update is None:
        raise RuntimeError("Experience-pool wrapper was invoked before a base update hook was registered.")

    poisoned_experiences = experiences
    if _ATTACK_ENGINE is not None:
        poisoned_experiences = _ATTACK_ENGINE.poison_experiences(self, experiences)
    return next_update(self, poisoned_experiences)


def install_experience_pool_attack(level, reward_factors):
    global _ATTACK_ENGINE

    _ATTACK_ENGINE = ExperiencePoolAttackEngine(level, reward_factors)
    for agent_class in (BA.DDQN_Agent, BA.DQN_Agent):
        current_method = agent_class.update
        if current_method is not _attacked_update:
            _NEXT_UPDATE_BY_CLASS[agent_class] = current_method
            agent_class.update = _attacked_update
    return dict(_ATTACK_ENGINE.profile)


def get_experience_pool_attack_stats():
    if _ATTACK_ENGINE is None:
        return None
    return _ATTACK_ENGINE.get_stats()
