from collections import OrderedDict
import random

import numpy as np
import torch

import SatelliteNetworkSimulator_Computing as SNC
from mdp_attacks.attack_monitor import record_attack_event


_ATTACK_ENGINE = None
_NEXT_APPEND_EXPERIENCE = None


class StateTransferAttackEngine:
    PROFILE_MAP = {
        0: {
            "enabled": False,
            "apply_probability": 0.0,
            "state_epsilon": 0.0,
            "state_steps": 0,
            "state_step_size": 0.0,
            "transition_epsilon": 0.0,
            "transition_steps": 0,
            "transition_step_size": 0.0,
            "random_restarts": 0,
            "random_start": False,
            "continuity_weight": 0.0,
            "history_decay": 0.0,
        },
        1: {
            "enabled": True,
            "apply_probability": 0.030,
            "state_epsilon": 0.020,
            "state_steps": 1,
            "state_step_size": 0.015,
            "transition_epsilon": 0.020,
            "transition_steps": 1,
            "transition_step_size": 0.015,
            "random_restarts": 0,
            "random_start": False,
            "continuity_weight": 0.015,
            "history_decay": 0.90,
        },
        2: {
            "enabled": True,
            "apply_probability": 0.055,
            "state_epsilon": 0.035,
            "state_steps": 2,
            "state_step_size": 0.018,
            "transition_epsilon": 0.040,
            "transition_steps": 2,
            "transition_step_size": 0.020,
            "random_restarts": 0,
            "random_start": True,
            "continuity_weight": 0.025,
            "history_decay": 0.88,
        },
        3: {
            "enabled": True,
            "apply_probability": 0.080,
            "state_epsilon": 0.050,
            "state_steps": 3,
            "state_step_size": 0.020,
            "transition_epsilon": 0.060,
            "transition_steps": 3,
            "transition_step_size": 0.025,
            "random_restarts": 1,
            "random_start": True,
            "continuity_weight": 0.035,
            "history_decay": 0.85,
        },
        4: {
            "enabled": True,
            "apply_probability": 0.10,
            "state_epsilon": 0.070,
            "state_steps": 4,
            "state_step_size": 0.025,
            "transition_epsilon": 0.080,
            "transition_steps": 4,
            "transition_step_size": 0.030,
            "random_restarts": 1,
            "random_start": True,
            "continuity_weight": 0.050,
            "history_decay": 0.82,
        },
    }

    def __init__(self, level, gamma):
        self.level = int(level)
        if self.level not in self.PROFILE_MAP:
            raise ValueError("StateTransferAttack_level must be one of {0,1,2,3,4}")

        self.profile = self.PROFILE_MAP[self.level]
        self.gamma = float(gamma)
        self._attack_space_cache = {}
        self._transition_history = {}
        self._transition_history_limit = 256
        self.last_attack_metadata = None
        self._stats = {
            "total_calls": 0,
            "attack_calls": 0,
            "beneficial_binding_calls": 0,
            "harmful_binding_calls": 0,
            "promote_calls": 0,
            "demote_calls": 0,
            "relabel_attempts": 0,
            "relabel_successes": 0,
            "future_shift_sum": 0.0,
            "continuity_error_sum": 0.0,
        }

    @property
    def enabled(self):
        return self.profile["enabled"]

    def reset_runtime_state(self):
        self._transition_history.clear()
        self.last_attack_metadata = None

    def maybe_attack(self, propagator, agent_name, state, mark, action, reward, next_state, done, event_type):
        original_state = np.asarray(state)
        original_next_state = np.asarray(next_state)

        clean_state = np.asarray(state, dtype=np.float32).copy()
        clean_next_state = np.asarray(next_state, dtype=np.float32).copy()
        reward_value = float(reward)

        self._stats["total_calls"] += 1
        history_store = self._transition_history.setdefault(
            str(agent_name) if agent_name is not None else "unknown_agent",
            OrderedDict(),
        )
        history_key = self._build_history_key(clean_state)

        default_result = {
            "state": clean_state.astype(original_state.dtype, copy=False),
            "action": action,
            "reward": reward,
            "next_state": clean_next_state.astype(original_next_state.dtype, copy=False),
            "mark": mark,
            "metadata": None,
        }

        if not self.enabled:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result
        if clean_state.ndim != 1 or clean_next_state.ndim != 1:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result
        if clean_state.shape != clean_next_state.shape or clean_state.size == 0:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result
        if random.random() > self.profile["apply_probability"]:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result

        satellite = propagator.satellites.get(agent_name)
        if satellite is None or satellite.q_net is None:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result
        if "DQN" not in satellite.mode:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result

        action_index = self._normalize_action(action)
        if action_index is None:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result

        state_action_count = 4 if int(round(float(clean_state[-1]))) == 1 else 5
        next_action_count = 4 if int(round(float(mark))) == 1 else 5
        if action_index < 0 or action_index >= state_action_count:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result
        if state_action_count <= 1 or next_action_count <= 0:
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result

        clean_q = self._evaluate_single_state(
            q_net=satellite.q_net,
            state=clean_state,
            valid_action_count=state_action_count,
            device=satellite.device,
        )
        clean_next_q = self._evaluate_single_state(
            q_net=satellite.q_net,
            state=clean_next_state,
            valid_action_count=next_action_count,
            device=satellite.device,
        )
        if not np.all(np.isfinite(clean_q)) or not np.all(np.isfinite(clean_next_q)):
            self._update_transition_history(history_store, history_key, clean_state, clean_next_state)
            return default_result

        optimal_action = int(np.argmax(clean_q))
        malicious_action = int(np.argmin(clean_q))
        binding_plan = self._build_binding_plan(
            reward_value=reward_value,
            clean_q=clean_q,
            action_index=action_index,
            optimal_action=optimal_action,
            malicious_action=malicious_action,
        )

        attack_mask, state_min, state_max = self._build_attack_space(satellite.mode, clean_state.shape[0])
        reference_delta = self._get_reference_delta(history_store, history_key, clean_state, clean_next_state)

        anchor_result = None
        target_binding_usable = binding_plan["target_action"] == action_index
        poisoned_state = clean_state.copy()
        if binding_plan["target_action"] != action_index:
            self._stats["relabel_attempts"] += 1
            anchor_result = self._craft_anchor_state(
                q_net=satellite.q_net,
                clean_state=clean_state,
                target_action=binding_plan["target_action"],
                attack_mask=attack_mask,
                state_min=state_min,
                state_max=state_max,
                valid_action_count=state_action_count,
                device=satellite.device,
            )
            if anchor_result is not None and anchor_result["target_hit"]:
                poisoned_state = anchor_result["state"].copy()
                target_binding_usable = True
                self._stats["relabel_successes"] += 1

        poisoned_action_index = binding_plan["target_action"] if target_binding_usable else action_index
        transition_mode = (
            binding_plan["target_transition_mode"]
            if target_binding_usable
            else binding_plan["fallback_transition_mode"]
        )

        transition_result = self._craft_transition_state(
            q_net=satellite.q_net,
            source_state=poisoned_state,
            clean_next_state=clean_next_state,
            clean_next_q=clean_next_q,
            transition_mode=transition_mode,
            reference_delta=reference_delta,
            attack_mask=attack_mask,
            state_min=state_min,
            state_max=state_max,
            valid_action_count=next_action_count,
            device=satellite.device,
        )
        if transition_result is None:
            poisoned_next_state = clean_next_state.copy()
            future_value_shift = 0.0
            continuity_error = 0.0
        else:
            poisoned_next_state = transition_result["state"].copy()
            future_value_shift = float(transition_result["future_value_shift"])
            continuity_error = float(transition_result["continuity_error"])

        metadata = {
            "event_type": event_type,
            "beneficial_binding": binding_plan["beneficial_binding"],
            "action_relabelled": poisoned_action_index != action_index,
            "anchor_target_hit": bool(anchor_result["target_hit"]) if anchor_result is not None else True,
            "original_action": action_index,
            "poisoned_action": poisoned_action_index,
            "optimal_action": optimal_action,
            "malicious_action": malicious_action,
            "transition_mode": transition_mode,
            "future_value_shift": future_value_shift,
            "continuity_error": continuity_error,
        }

        self.last_attack_metadata = metadata
        self._update_stats(metadata)
        self._update_transition_history(history_store, history_key, clean_state, clean_next_state)

        return {
            "state": poisoned_state.astype(original_state.dtype, copy=False),
            "action": self._repack_action(action, poisoned_action_index),
            "reward": reward,
            "next_state": poisoned_next_state.astype(original_next_state.dtype, copy=False),
            "mark": mark,
            "metadata": metadata,
        }

    def get_stats(self):
        stats = dict(self._stats)
        attack_calls = max(stats["attack_calls"], 1)
        stats["relabel_success_rate"] = stats["relabel_successes"] / max(stats["relabel_attempts"], 1)
        stats["avg_future_value_shift"] = stats["future_shift_sum"] / attack_calls
        stats["avg_continuity_error"] = stats["continuity_error_sum"] / attack_calls
        return stats

    def _update_stats(self, metadata):
        self._stats["attack_calls"] += 1
        if metadata["beneficial_binding"]:
            self._stats["beneficial_binding_calls"] += 1
        else:
            self._stats["harmful_binding_calls"] += 1
        if metadata["transition_mode"] == "promote":
            self._stats["promote_calls"] += 1
        else:
            self._stats["demote_calls"] += 1
        self._stats["future_shift_sum"] += float(metadata["future_value_shift"])
        self._stats["continuity_error_sum"] += float(metadata["continuity_error"])

    def _build_history_key(self, clean_state):
        rounded_state = [float(value) for value in np.round(clean_state, 2).tolist()]
        if rounded_state:
            rounded_state[-1] = int(round(float(clean_state[-1])))
        return tuple(rounded_state)

    def _get_reference_delta(self, history_store, history_key, clean_state, clean_next_state):
        clean_delta = clean_next_state - clean_state
        historical_delta = history_store.get(history_key)
        if historical_delta is None:
            return clean_delta
        history_store.move_to_end(history_key)
        return 0.5 * clean_delta + 0.5 * historical_delta

    def _update_transition_history(self, history_store, history_key, clean_state, clean_next_state):
        decay = float(self.profile["history_decay"])
        clean_delta = clean_next_state - clean_state
        if history_key not in history_store or decay <= 0:
            updated_delta = clean_delta.copy()
        else:
            updated_delta = decay * history_store[history_key] + (1.0 - decay) * clean_delta
        history_store[history_key] = updated_delta
        history_store.move_to_end(history_key)
        while len(history_store) > self._transition_history_limit:
            history_store.popitem(last=False)

    def _build_binding_plan(self, reward_value, clean_q, action_index, optimal_action, malicious_action):
        beneficial_binding = self._is_beneficial_transition(reward_value, clean_q, action_index)
        if beneficial_binding:
            return {
                "beneficial_binding": True,
                "target_action": malicious_action,
                "target_transition_mode": "promote",
                "fallback_transition_mode": "demote",
            }
        return {
            "beneficial_binding": False,
            "target_action": optimal_action,
            "target_transition_mode": "demote",
            "fallback_transition_mode": "promote",
        }

    def _is_beneficial_transition(self, reward_value, clean_q, action_index):
        if reward_value > 1e-9:
            return True
        if reward_value < -1e-9:
            return False
        ranked = np.argsort(clean_q)[::-1].tolist()
        rank = ranked.index(int(action_index))
        return rank < max(1, len(ranked) // 2)

    def _craft_anchor_state(
        self,
        q_net,
        clean_state,
        target_action,
        attack_mask,
        state_min,
        state_max,
        valid_action_count,
        device,
    ):
        epsilon = float(self.profile["state_epsilon"])
        step_size = float(self.profile["state_step_size"])
        steps = int(self.profile["state_steps"])
        if epsilon <= 0 or step_size <= 0 or steps <= 0:
            return None
        if not np.any(attack_mask) or valid_action_count <= 1:
            return None

        clean_q = self._evaluate_single_state(q_net, clean_state, valid_action_count, device)
        if int(np.argmax(clean_q)) == int(target_action):
            return {
                "state": clean_state.copy(),
                "target_hit": True,
                "target_margin": self._target_margin(clean_q, target_action),
            }

        clean_tensor = torch.as_tensor(clean_state, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(attack_mask.astype(np.float32), dtype=torch.float32, device=device)
        state_min_tensor = torch.as_tensor(state_min, dtype=torch.float32, device=device)
        state_max_tensor = torch.as_tensor(state_max, dtype=torch.float32, device=device)
        local_lower = torch.maximum(clean_tensor - epsilon, state_min_tensor)
        local_upper = torch.minimum(clean_tensor + epsilon, state_max_tensor)

        total_restarts = 1 + int(self.profile["random_restarts"])
        best_candidate = None
        for restart_index in range(total_restarts):
            if bool(self.profile["random_start"]):
                noise = torch.empty_like(clean_tensor).uniform_(-epsilon, epsilon) * mask_tensor
                adv_state = clean_tensor + noise
                adv_state = torch.maximum(torch.minimum(adv_state, local_upper), local_lower)
            else:
                adv_state = clean_tensor.clone()
            adv_state = torch.where(mask_tensor > 0, adv_state, clean_tensor)

            for _ in range(steps):
                adv_state = adv_state.clone().detach().requires_grad_(True)
                q_values = q_net(adv_state.unsqueeze(0))[0, :valid_action_count]
                objective = self._target_objective(q_values, target_action)
                gradient = torch.autograd.grad(objective, adv_state, retain_graph=False, create_graph=False)[0]
                adv_state = adv_state.detach() + step_size * torch.sign(gradient) * mask_tensor
                adv_state = torch.maximum(torch.minimum(adv_state, local_upper), local_lower)
                adv_state = torch.where(mask_tensor > 0, adv_state, clean_tensor)

            attacked_state = adv_state.detach().cpu().numpy().astype(np.float32, copy=False)
            attacked_q = self._evaluate_single_state(q_net, attacked_state, valid_action_count, device)
            attacked_action = int(np.argmax(attacked_q))
            candidate = {
                "state": attacked_state,
                "target_hit": attacked_action == int(target_action),
                "target_margin": self._target_margin(attacked_q, target_action),
                "rank": (
                    int(attacked_action == int(target_action)),
                    float(self._target_margin(attacked_q, target_action)),
                    -float(np.max(np.abs(attacked_state - clean_state))),
                ),
            }
            if best_candidate is None or candidate["rank"] > best_candidate["rank"]:
                best_candidate = candidate
            if candidate["target_hit"]:
                return candidate
        return best_candidate

    def _craft_transition_state(
        self,
        q_net,
        source_state,
        clean_next_state,
        clean_next_q,
        transition_mode,
        reference_delta,
        attack_mask,
        state_min,
        state_max,
        valid_action_count,
        device,
    ):
        epsilon = float(self.profile["transition_epsilon"])
        step_size = float(self.profile["transition_step_size"])
        steps = int(self.profile["transition_steps"])
        if epsilon <= 0 or step_size <= 0 or steps <= 0:
            return None
        if not np.any(attack_mask) or valid_action_count <= 0:
            return None

        clean_tensor = torch.as_tensor(clean_next_state, dtype=torch.float32, device=device)
        source_tensor = torch.as_tensor(source_state, dtype=torch.float32, device=device)
        ref_delta_tensor = torch.as_tensor(reference_delta, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(attack_mask.astype(np.float32), dtype=torch.float32, device=device)
        state_min_tensor = torch.as_tensor(state_min, dtype=torch.float32, device=device)
        state_max_tensor = torch.as_tensor(state_max, dtype=torch.float32, device=device)
        local_lower = torch.maximum(clean_tensor - epsilon, state_min_tensor)
        local_upper = torch.minimum(clean_tensor + epsilon, state_max_tensor)

        clean_future_value = self._soft_value_numpy(clean_next_q)
        total_restarts = 1 + int(self.profile["random_restarts"])
        best_candidate = None
        continuity_weight = float(self.profile["continuity_weight"])

        for restart_index in range(total_restarts):
            if bool(self.profile["random_start"]):
                noise = torch.empty_like(clean_tensor).uniform_(-epsilon, epsilon) * mask_tensor
                adv_state = clean_tensor + noise
                adv_state = torch.maximum(torch.minimum(adv_state, local_upper), local_lower)
            else:
                adv_state = clean_tensor.clone()
            adv_state = torch.where(mask_tensor > 0, adv_state, clean_tensor)

            for _ in range(steps):
                adv_state = adv_state.clone().detach().requires_grad_(True)
                q_values = q_net(adv_state.unsqueeze(0))[0, :valid_action_count]
                future_value = self._soft_value_tensor(q_values)
                objective = future_value if transition_mode == "promote" else -future_value
                if continuity_weight > 0:
                    continuity_penalty = torch.mean(torch.abs((adv_state - source_tensor) - ref_delta_tensor))
                    objective = objective - continuity_weight * continuity_penalty
                gradient = torch.autograd.grad(objective, adv_state, retain_graph=False, create_graph=False)[0]
                adv_state = adv_state.detach() + step_size * torch.sign(gradient) * mask_tensor
                adv_state = torch.maximum(torch.minimum(adv_state, local_upper), local_lower)
                adv_state = torch.where(mask_tensor > 0, adv_state, clean_tensor)

            attacked_state = adv_state.detach().cpu().numpy().astype(np.float32, copy=False)
            attacked_q = self._evaluate_single_state(q_net, attacked_state, valid_action_count, device)
            attacked_future_value = self._soft_value_numpy(attacked_q)
            continuity_error = float(np.mean(np.abs((attacked_state - source_state) - reference_delta)))
            desired_shift = (
                attacked_future_value - clean_future_value
                if transition_mode == "promote"
                else clean_future_value - attacked_future_value
            )
            target_shift = self.gamma * desired_shift
            candidate = {
                "state": attacked_state,
                "future_value_shift": target_shift,
                "continuity_error": continuity_error,
                "rank": (
                    float(target_shift - continuity_weight * continuity_error),
                    float(target_shift),
                    -float(continuity_error),
                ),
            }
            if best_candidate is None or candidate["rank"] > best_candidate["rank"]:
                best_candidate = candidate
        return best_candidate

    def _evaluate_single_state(self, q_net, state, valid_action_count, device):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)[0, :valid_action_count]
        return q_values.detach().cpu().numpy()

    def _soft_value_tensor(self, q_values):
        if q_values.numel() == 0:
            return torch.zeros((), device=q_values.device)
        if q_values.numel() == 1:
            return q_values[0]
        temperature = 0.35
        return torch.logsumexp(q_values / temperature, dim=0) * temperature

    def _soft_value_numpy(self, q_values):
        q_values = np.asarray(q_values, dtype=np.float32)
        if q_values.size == 0:
            return 0.0
        if q_values.size == 1:
            return float(q_values[0])
        temperature = 0.35
        scaled = q_values / temperature
        max_scaled = np.max(scaled)
        return float((np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled) * temperature)

    def _target_objective(self, q_values, target_action):
        if q_values.numel() <= 1:
            return q_values[0]
        all_indices = torch.arange(q_values.shape[0], device=q_values.device)
        other_indices = all_indices[all_indices != int(target_action)]
        if other_indices.numel() == 0:
            return q_values[int(target_action)]
        return q_values[int(target_action)] - torch.max(q_values[other_indices])

    def _target_margin(self, q_values, target_action):
        q_values = np.asarray(q_values, dtype=np.float32)
        if q_values.size <= 1:
            return float(q_values[0]) if q_values.size == 1 else 0.0
        other_values = np.delete(q_values, int(target_action))
        if other_values.size == 0:
            return float(q_values[int(target_action)])
        return float(q_values[int(target_action)] - np.max(other_values))

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

    def _repack_action(self, original_action, action_index):
        if hasattr(original_action, "buffer_action"):
            return int(action_index)
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

    def _build_attack_space(self, mode, state_dim):
        cache_key = (mode, int(state_dim))
        if cache_key in self._attack_space_cache:
            return self._attack_space_cache[cache_key]

        attack_mask = np.zeros(state_dim, dtype=bool)
        state_min = np.full(state_dim, -np.inf, dtype=np.float32)
        state_max = np.full(state_dim, np.inf, dtype=np.float32)

        if "New" in mode and state_dim >= 65:
            self._configure_new_mode_space(attack_mask, state_min, state_max)
        elif state_dim >= 33:
            self._configure_standard_mode_space(attack_mask, state_min, state_max)

        self._attack_space_cache[cache_key] = (attack_mask, state_min, state_max)
        return self._attack_space_cache[cache_key]

    def _mark_range(self, attack_mask, state_min, state_max, index, lower, upper, attackable):
        state_min[index] = lower
        state_max[index] = upper
        attack_mask[index] = attackable

    def _configure_standard_mode_space(self, attack_mask, state_min, state_max):
        for slot in range(4):
            base = slot * 6
            self._mark_range(attack_mask, state_min, state_max, base + 0, 0.0, 1.0, False)
            self._mark_range(attack_mask, state_min, state_max, base + 1, 0.0, 1.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 2, 0.0, 2.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 3, 0.0, 1.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 4, 0.0, 1.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 5, 0.0, 2.0, False)

        self._mark_range(attack_mask, state_min, state_max, 24, 0.0, 1.0, False)
        self._mark_range(attack_mask, state_min, state_max, 25, 0.0, 1.0, True)
        self._mark_range(attack_mask, state_min, state_max, 26, 0.0, 2.0, True)

        self._mark_range(attack_mask, state_min, state_max, 27, 0.0, 1.0, False)
        self._mark_range(attack_mask, state_min, state_max, 28, 0.0, 2.0, True)
        self._mark_range(attack_mask, state_min, state_max, 29, 0.0, 10.0, True)
        self._mark_range(attack_mask, state_min, state_max, 30, 0.0, 2.0, True)

        self._mark_range(attack_mask, state_min, state_max, 31, 0.0, 2.0, False)
        self._mark_range(attack_mask, state_min, state_max, 32, 0.0, 1.0, False)

    def _configure_new_mode_space(self, attack_mask, state_min, state_max):
        for slot in range(4):
            base = slot * 14
            for group_offset in (0, 4, 8):
                self._mark_range(attack_mask, state_min, state_max, base + group_offset + 0, 0.0, 1.0, False)
                self._mark_range(attack_mask, state_min, state_max, base + group_offset + 1, 0.0, 1.0, True)
                self._mark_range(attack_mask, state_min, state_max, base + group_offset + 2, 0.0, 2.0, True)
                self._mark_range(attack_mask, state_min, state_max, base + group_offset + 3, 0.0, 1.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 12, 0.0, 1.0, True)
            self._mark_range(attack_mask, state_min, state_max, base + 13, 0.0, 2.0, False)

        self._mark_range(attack_mask, state_min, state_max, 56, 0.0, 1.0, False)
        self._mark_range(attack_mask, state_min, state_max, 57, 0.0, 1.0, True)
        self._mark_range(attack_mask, state_min, state_max, 58, 0.0, 2.0, True)

        self._mark_range(attack_mask, state_min, state_max, 59, 0.0, 1.0, False)
        self._mark_range(attack_mask, state_min, state_max, 60, 0.0, 2.0, True)
        self._mark_range(attack_mask, state_min, state_max, 61, 0.0, 10.0, True)
        self._mark_range(attack_mask, state_min, state_max, 62, 0.0, 2.0, True)

        self._mark_range(attack_mask, state_min, state_max, 63, 0.0, 2.0, False)
        self._mark_range(attack_mask, state_min, state_max, 64, 0.0, 1.0, False)


def _attacked_append_experience(self, agent_name, state, mark, action, reward, next_state, done, event_type):
    if _NEXT_APPEND_EXPERIENCE is None:
        raise RuntimeError("State-transfer wrapper was invoked before a base append_experience hook was registered.")

    poisoned_state = state
    poisoned_action = action
    poisoned_reward = reward
    poisoned_next_state = next_state
    poisoned_mark = mark
    if _ATTACK_ENGINE is not None:
        attacked = _ATTACK_ENGINE.maybe_attack(
            propagator=self,
            agent_name=agent_name,
            state=state,
            mark=mark,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            event_type=event_type,
        )
        poisoned_state = attacked["state"]
        poisoned_action = attacked["action"]
        poisoned_reward = attacked["reward"]
        poisoned_next_state = attacked["next_state"]
        poisoned_mark = attacked["mark"]
        if (
            not np.allclose(np.asarray(poisoned_state), np.asarray(state))
            or not np.allclose(np.asarray(poisoned_next_state), np.asarray(next_state))
            or poisoned_action != action
            or float(poisoned_reward) != float(reward)
            or poisoned_mark != mark
        ):
            record_attack_event(
                "StateTransferAttack",
                agent_name,
                attacked.get("metadata") or {"event_type": event_type},
            )
    return _NEXT_APPEND_EXPERIENCE(
        self,
        agent_name,
        poisoned_state,
        poisoned_mark,
        poisoned_action,
        poisoned_reward,
        poisoned_next_state,
        done,
        event_type,
    )


def install_state_transfer_attack(level, gamma):
    global _ATTACK_ENGINE, _NEXT_APPEND_EXPERIENCE

    _ATTACK_ENGINE = StateTransferAttackEngine(level, gamma)
    current_method = SNC.Propagator_Computing.append_experience
    if current_method is not _attacked_append_experience:
        _NEXT_APPEND_EXPERIENCE = current_method
        SNC.Propagator_Computing.append_experience = _attacked_append_experience
    return dict(_ATTACK_ENGINE.profile)


def get_state_transfer_attack_stats():
    if _ATTACK_ENGINE is None:
        return None
    return _ATTACK_ENGINE.get_stats()


def reset_state_transfer_attack_runtime():
    if _ATTACK_ENGINE is None:
        return
    _ATTACK_ENGINE.reset_runtime_state()
