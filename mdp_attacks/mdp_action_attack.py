from dataclasses import dataclass
import random

import numpy as np
import torch

import SatelliteNetworkSimulator_Computing as SNC


_ATTACK_ENGINE = None
_NEXT_GET_NEXT_HOP = None


@dataclass(frozen=True)
class AttackActionRecord:
    selected_action: int
    policy_action: int
    malicious_action: int
    executed_action: int
    buffer_action: int
    attack_attempted: bool
    attacked: bool
    strength_level: int
    strength_name: str
    attack_probability: float

    def to_log_payload(self):
        return {
            "selected_action": self.selected_action,
            "policy_action": self.policy_action,
            "malicious_action": self.malicious_action,
            "executed_action": self.executed_action,
            "buffer_action": self.buffer_action,
            "attack_attempted": self.attack_attempted,
            "attacked": self.attacked,
            "strength_level": self.strength_level,
            "strength_name": self.strength_name,
            "attack_probability": self.attack_probability,
        }


class MDPActionAttack:
    LEVEL_TO_PROFILE = {
        0: ("off", 0.0),
        1: ("low", 0.10),
        2: ("medium", 0.30),
        3: ("high", 0.60),
        4: ("extreme", 0.90),
    }

    def __init__(self, attack_level=0):
        self.attack_level = int(attack_level)
        if self.attack_level not in self.LEVEL_TO_PROFILE:
            raise ValueError(
                f"attack_level must be one of {sorted(self.LEVEL_TO_PROFILE)}, "
                f"got {self.attack_level}"
            )
        self.strength_name, self.rho = self.LEVEL_TO_PROFILE[self.attack_level]
        self.enabled = self.attack_level > 0
        self.reset_stats()

    def tamper_action(self, selected_action, q_values, is_computed):
        action_values = self._flatten_q_values(q_values)
        valid_action_count = 4 if bool(is_computed) else 5
        candidate_values = action_values[:valid_action_count]
        if not candidate_values:
            raise ValueError("No valid action values were provided for the attack module")

        selected_action = int(selected_action)
        if not 0 <= selected_action < valid_action_count:
            raise ValueError(
                f"selected_action must be in [0, {valid_action_count - 1}], got {selected_action}"
            )
        policy_action = int(np.argmax(candidate_values))
        malicious_action = min(range(valid_action_count), key=lambda index: candidate_values[index])
        attack_attempted = random.random() < self.rho
        attacked = attack_attempted and selected_action != malicious_action
        self._record_decision(attack_attempted, attacked)
        executed_action = malicious_action if attacked else selected_action
        buffer_action = policy_action if attacked else selected_action
        return AttackActionRecord(
            selected_action=selected_action,
            policy_action=policy_action,
            malicious_action=malicious_action,
            executed_action=executed_action,
            buffer_action=buffer_action,
            attack_attempted=attack_attempted,
            attacked=attacked,
            strength_level=self.attack_level,
            strength_name=self.strength_name,
            attack_probability=self.rho,
        )

    def _flatten_q_values(self, q_values):
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.detach().reshape(-1).tolist()
        else:
            q_values = list(q_values)
        return [float(value) for value in q_values]

    def _record_decision(self, attack_attempted, attacked):
        self.window_decision_count += 1
        self.total_decision_count += 1
        if attack_attempted:
            self.window_attempt_count += 1
            self.total_attempt_count += 1
        if attacked:
            self.window_hit_count += 1
            self.total_hit_count += 1

    def record_poisoned_sample(self, attacked):
        if attacked:
            self.window_poisoned_sample_count += 1
            self.total_poisoned_sample_count += 1

    def consume_window_stats(self):
        attempt_rate = (
            self.window_attempt_count / self.window_decision_count
            if self.window_decision_count > 0 else None
        )
        hit_rate = (
            self.window_hit_count / self.window_decision_count
            if self.window_decision_count > 0 else None
        )
        conditional_hit_rate = (
            self.window_hit_count / self.window_attempt_count
            if self.window_attempt_count > 0 else None
        )
        stats = {
            "attack_level": self.attack_level,
            "strength_name": self.strength_name,
            "attack_probability": self.rho,
            "window_decision_count": self.window_decision_count,
            "window_attempt_count": self.window_attempt_count,
            "window_attempt_rate": attempt_rate,
            "window_hit_count": self.window_hit_count,
            "window_hit_rate": hit_rate,
            "window_conditional_hit_rate": conditional_hit_rate,
            "window_poisoned_sample_count": self.window_poisoned_sample_count,
            "total_poisoned_sample_count": self.total_poisoned_sample_count,
        }
        self.window_decision_count = 0
        self.window_attempt_count = 0
        self.window_hit_count = 0
        self.window_poisoned_sample_count = 0
        return stats

    def get_stats(self):
        total_attempt_rate = (
            self.total_attempt_count / self.total_decision_count
            if self.total_decision_count > 0 else None
        )
        total_hit_rate = (
            self.total_hit_count / self.total_decision_count
            if self.total_decision_count > 0 else None
        )
        total_conditional_hit_rate = (
            self.total_hit_count / self.total_attempt_count
            if self.total_attempt_count > 0 else None
        )
        return {
            "attack_level": self.attack_level,
            "strength_name": self.strength_name,
            "attack_probability": self.rho,
            "window_decision_count": self.window_decision_count,
            "window_attempt_count": self.window_attempt_count,
            "window_hit_count": self.window_hit_count,
            "window_poisoned_sample_count": self.window_poisoned_sample_count,
            "total_decision_count": self.total_decision_count,
            "total_attempt_count": self.total_attempt_count,
            "total_attempt_rate": total_attempt_rate,
            "total_hit_count": self.total_hit_count,
            "total_hit_rate": total_hit_rate,
            "total_conditional_hit_rate": total_conditional_hit_rate,
            "total_poisoned_sample_count": self.total_poisoned_sample_count,
        }

    def reset_stats(self):
        self.window_decision_count = 0
        self.window_attempt_count = 0
        self.window_hit_count = 0
        self.window_poisoned_sample_count = 0
        self.total_decision_count = 0
        self.total_attempt_count = 0
        self.total_hit_count = 0
        self.total_poisoned_sample_count = 0


def _evaluate_q_values(satellite, current_state):
    state_tensor = torch.as_tensor(
        np.asarray(current_state, dtype=np.float32),
        dtype=torch.float32,
        device=satellite.device,
    ).unsqueeze(0)
    if hasattr(satellite, "_run_q_net_timed"):
        q_values = satellite._run_q_net_timed(state_tensor)[0]
    else:
        with torch.no_grad():
            q_values = satellite.q_net(state_tensor)[0]
    return q_values.detach().cpu().reshape(-1).tolist()


def _attacked_get_next_hop(self, current_state, destination):
    if _NEXT_GET_NEXT_HOP is None:
        raise RuntimeError("Action-attack wrapper was invoked before a base get_next_hop hook was registered.")

    selected_action = _NEXT_GET_NEXT_HOP(self, current_state, destination)
    if _ATTACK_ENGINE is None or not _ATTACK_ENGINE.enabled:
        return selected_action
    if "DQN" not in self.mode or self.q_net is None:
        return selected_action
    if isinstance(selected_action, AttackActionRecord):
        return selected_action

    state_array = np.asarray(current_state, dtype=np.float32)
    if state_array.ndim != 1 or state_array.size == 0:
        return selected_action

    q_values = _evaluate_q_values(self, state_array)
    return _ATTACK_ENGINE.tamper_action(
        selected_action=selected_action,
        q_values=q_values,
        is_computed=bool(int(round(float(state_array[-1])))),
    )


def install_action_attack(level):
    global _ATTACK_ENGINE, _NEXT_GET_NEXT_HOP

    _ATTACK_ENGINE = MDPActionAttack(level)
    current_method = SNC.Satellite_with_Computing.get_next_hop
    if current_method is not _attacked_get_next_hop:
        _NEXT_GET_NEXT_HOP = current_method
        SNC.Satellite_with_Computing.get_next_hop = _attacked_get_next_hop
    return {
        "enabled": _ATTACK_ENGINE.enabled,
        "attack_level": _ATTACK_ENGINE.attack_level,
        "strength_name": _ATTACK_ENGINE.strength_name,
        "attack_probability": _ATTACK_ENGINE.rho,
    }


def record_action_attack_sample(attacked):
    if _ATTACK_ENGINE is None:
        return
    _ATTACK_ENGINE.record_poisoned_sample(bool(attacked))


def get_action_attack_stats():
    if _ATTACK_ENGINE is None:
        return None
    return _ATTACK_ENGINE.get_stats()
