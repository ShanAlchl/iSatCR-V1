import random

import numpy as np
import torch

import SatelliteNetworkSimulator_Computing as SNC
from mdp_attacks.attack_monitor import record_attack_event


_ATTACK_ENGINE = None


class StateObservationAttackEngine:
    PROFILE_MAP = {
        0: {
            "enabled": False,
            "epsilon": 0.0,
            "population": 0,
            "generations": 0,
            "pgd_steps": 0,
            "step_size": 0.0,
            "random_restarts": 0,
            "random_start": False,
            "top_k_targets": 0,
            "stop_on_flip": False,
            "apply_probability": 0.0,
        },
        1: {
            "enabled": True,
            "epsilon": 0.02,
            "population": 0,
            "generations": 0,
            "pgd_steps": 1,
            "step_size": 0.018,
            "random_restarts": 0,
            "random_start": False,
            "top_k_targets": 1,
            "stop_on_flip": True,
            "apply_probability": 0.035,
        },
        2: {
            "enabled": True,
            "epsilon": 0.04,
            "population": 0,
            "generations": 0,
            "pgd_steps": 2,
            "step_size": 0.020,
            "random_restarts": 0,
            "random_start": True,
            "top_k_targets": 1,
            "stop_on_flip": True,
            "apply_probability": 0.060,
        },
        3: {
            "enabled": True,
            "epsilon": 0.06,
            "population": 0,
            "generations": 0,
            "pgd_steps": 3,
            "step_size": 0.020,
            "random_restarts": 1,
            "random_start": True,
            "top_k_targets": 2,
            "stop_on_flip": True,
            "apply_probability": 0.085,
        },
        4: {
            "enabled": True,
            "epsilon": 0.08,
            "population": 0,
            "generations": 0,
            "pgd_steps": 4,
            "step_size": 0.025,
            "random_restarts": 1,
            "random_start": True,
            "top_k_targets": 2,
            "stop_on_flip": True,
            "apply_probability": 0.1,
        },
    }

    def __init__(self, level):
        self.level = int(level)
        if self.level not in self.PROFILE_MAP:
            raise ValueError("StateObservationAttack_level must be one of {0,1,2,3,4}")
        self.profile = self.PROFILE_MAP[self.level]
        self._cache = {}
        self._cache_time = None
        self._attack_space_cache = {}
        self.last_attack_metadata = None
        self._stats = {
            "total_calls": 0,
            "attack_calls": 0,
            "cache_hits": 0,
            "action_flips": 0,
            "target_hits": 0,
            "no_flip": 0,
            "clean_margin_sum": 0.0,
            "attacked_margin_sum": 0.0,
            "policy_value_drop_sum": 0.0,
            "target_value_gap_sum": 0.0,
        }

    @property
    def enabled(self):
        return self.profile["enabled"]

    def maybe_attack(self, satellite, clean_state):
        clean_state = np.asarray(clean_state).copy()
        self._stats["total_calls"] += 1

        if not self.enabled:
            return clean_state.copy()
        if satellite.q_net is None:
            return clean_state.copy()
        if "DQN" not in satellite.mode:
            return clean_state.copy()
        if random.random() > self.profile["apply_probability"]:
            return clean_state.copy()

        valid_action_count = 4 if int(round(float(clean_state[-1]))) == 1 else 5
        attack_mask, state_min, state_max = self._build_attack_space(satellite.mode, clean_state.shape[0])
        if not np.any(attack_mask):
            return clean_state.copy()

        self._refresh_cache(float(satellite.env.now))
        cache_key = (satellite.name, valid_action_count, tuple(np.round(clean_state, 6).tolist()))
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            self.last_attack_metadata = dict(cached["metadata"])
            return cached["state"].copy()

        self._stats["attack_calls"] += 1
        result = self.attack_with_metadata(
            satellite=satellite,
            clean_state=clean_state,
            valid_action_count=valid_action_count,
            attack_mask=attack_mask,
            state_min=state_min,
            state_max=state_max,
        )
        attacked_state = result["attacked_state"].astype(clean_state.dtype, copy=False)
        metadata = dict(result["metadata"])
        self._cache[cache_key] = {
            "state": attacked_state.copy(),
            "metadata": metadata,
        }
        self.last_attack_metadata = metadata
        self._update_stats(metadata)
        if not np.allclose(attacked_state, clean_state):
            record_attack_event("StateObservationAttack", satellite.name, metadata)
        return attacked_state

    def attack_with_metadata(
        self,
        satellite,
        clean_state,
        valid_action_count=None,
        attack_mask=None,
        state_min=None,
        state_max=None,
    ):
        clean_state = np.asarray(clean_state, dtype=np.float32).copy()
        if valid_action_count is None:
            valid_action_count = 4 if int(round(float(clean_state[-1]))) == 1 else 5
        if attack_mask is None or state_min is None or state_max is None:
            attack_mask, state_min, state_max = self._build_attack_space(satellite.mode, clean_state.shape[0])

        clean_q = self._evaluate_single_state(satellite.q_net, clean_state, valid_action_count, satellite.device)
        clean_action = int(np.argmax(clean_q)) if valid_action_count > 0 else 0
        clean_margin = self._winner_margin(clean_q)
        metadata = {
            "clean_action": clean_action,
            "attacked_action": clean_action,
            "target_action": clean_action,
            "action_flipped": False,
            "target_hit": False,
            "clean_margin": clean_margin,
            "attacked_margin": clean_margin,
            "policy_value_drop": 0.0,
            "target_value_gap": 0.0,
        }

        if not np.any(attack_mask) or valid_action_count <= 1:
            return {
                "attacked_state": clean_state.copy(),
                "metadata": metadata,
            }

        target_candidates = self._select_target_candidates(clean_q, clean_action)
        if not target_candidates:
            return {
                "attacked_state": clean_state.copy(),
                "metadata": metadata,
            }

        best_candidate = self._search_best_candidate(
            q_net=satellite.q_net,
            clean_state=clean_state,
            clean_q=clean_q,
            clean_action=clean_action,
            target_candidates=target_candidates,
            attack_mask=attack_mask,
            state_min=state_min,
            state_max=state_max,
            valid_action_count=valid_action_count,
            device=satellite.device,
        )
        if best_candidate is None:
            return {
                "attacked_state": clean_state.copy(),
                "metadata": metadata,
            }

        attacked_q = best_candidate["attacked_q"]
        attacked_action = int(np.argmax(attacked_q)) if valid_action_count > 0 else clean_action
        return {
            "attacked_state": best_candidate["state"].copy(),
            "metadata": {
                "clean_action": clean_action,
                "attacked_action": attacked_action,
                "target_action": int(best_candidate["target_action"]),
                "action_flipped": attacked_action != clean_action,
                "target_hit": attacked_action == int(best_candidate["target_action"]),
                "clean_margin": clean_margin,
                "attacked_margin": self._winner_margin(attacked_q),
                "policy_value_drop": float(best_candidate["policy_value_drop"]),
                "target_value_gap": float(best_candidate["target_value_gap"]),
            },
        }

    def get_stats(self):
        stats = dict(self._stats)
        attack_calls = max(stats["attack_calls"], 1)
        total_calls = max(stats["total_calls"], 1)
        stats["flip_rate"] = stats["action_flips"] / attack_calls
        stats["target_hit_rate"] = stats["target_hits"] / attack_calls
        stats["attack_rate"] = stats["attack_calls"] / total_calls
        stats["cache_hit_rate"] = stats["cache_hits"] / attack_calls
        stats["avg_clean_margin"] = stats["clean_margin_sum"] / attack_calls
        stats["avg_attacked_margin"] = stats["attacked_margin_sum"] / attack_calls
        stats["avg_policy_value_drop"] = stats["policy_value_drop_sum"] / attack_calls
        stats["avg_target_value_gap"] = stats["target_value_gap_sum"] / attack_calls
        return stats

    def _update_stats(self, metadata):
        self._stats["clean_margin_sum"] += float(metadata["clean_margin"])
        self._stats["attacked_margin_sum"] += float(metadata["attacked_margin"])
        self._stats["policy_value_drop_sum"] += float(metadata.get("policy_value_drop", 0.0))
        self._stats["target_value_gap_sum"] += float(metadata.get("target_value_gap", 0.0))
        if metadata["action_flipped"]:
            self._stats["action_flips"] += 1
        else:
            self._stats["no_flip"] += 1
        if metadata["target_hit"]:
            self._stats["target_hits"] += 1

    def _refresh_cache(self, current_time):
        rounded_time = round(current_time, 6)
        if self._cache_time != rounded_time:
            self._cache = {}
            self._cache_time = rounded_time

    def _select_target_candidates(self, clean_q, clean_action):
        if len(clean_q) <= 1:
            return []

        ascending = [int(index) for index in np.argsort(clean_q).tolist() if int(index) != int(clean_action)]
        descending = [int(index) for index in np.argsort(clean_q)[::-1].tolist() if int(index) != int(clean_action)]

        candidates = []
        if descending:
            # Keep the easiest competitor first, but also include the most
            # damaging low-value target early so damage-aware ranking can choose
            # between reachability and task harm.
            candidates.append(descending[0])
        if ascending:
            worst_action = ascending[0]
            if worst_action not in candidates:
                candidates.append(worst_action)
        for index in descending[1:]:
            if index not in candidates:
                candidates.append(index)
        for index in ascending:
            if index not in candidates:
                candidates.append(index)

        top_k_targets = int(self.profile.get("top_k_targets", len(candidates)))
        if top_k_targets <= 0:
            return []
        return candidates[:top_k_targets]

    def _search_best_candidate(
        self,
        q_net,
        clean_state,
        clean_q,
        clean_action,
        target_candidates,
        attack_mask,
        state_min,
        state_max,
        valid_action_count,
        device,
    ):
        best_candidate = None
        for target_action in target_candidates:
            candidate = self._run_targeted_pgd(
                q_net=q_net,
                clean_state=clean_state,
                clean_q=clean_q,
                clean_action=clean_action,
                target_action=target_action,
                attack_mask=attack_mask,
                state_min=state_min,
                state_max=state_max,
                valid_action_count=valid_action_count,
                device=device,
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate["rank"] > best_candidate["rank"]:
                best_candidate = candidate
        return best_candidate

    def _run_targeted_pgd(
        self,
        q_net,
        clean_state,
        clean_q,
        clean_action,
        target_action,
        attack_mask,
        state_min,
        state_max,
        valid_action_count,
        device,
    ):
        epsilon = float(self.profile["epsilon"])
        step_size = float(self.profile["step_size"])
        pgd_steps = int(self.profile["pgd_steps"])
        if epsilon <= 0 or step_size <= 0 or pgd_steps <= 0:
            return None

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

            for _ in range(pgd_steps):
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
                "attacked_q": attacked_q,
                "attacked_action": attacked_action,
                "target_action": int(target_action),
                "policy_value_drop": float(clean_q[int(clean_action)] - clean_q[int(attacked_action)]),
                "target_value_gap": float(clean_q[int(clean_action)] - clean_q[int(target_action)]),
                "rank": self._candidate_rank(
                    clean_q=clean_q,
                    attacked_q=attacked_q,
                    clean_action=clean_action,
                    target_action=target_action,
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

    def _target_objective(self, q_values, target_action):
        if q_values.numel() <= 1:
            return q_values[0]
        all_indices = torch.arange(q_values.shape[0], device=q_values.device)
        other_indices = all_indices[all_indices != int(target_action)]
        if other_indices.numel() == 0:
            return q_values[int(target_action)]
        return q_values[int(target_action)] - torch.max(q_values[other_indices])

    def _candidate_rank(self, clean_q, attacked_q, clean_action, target_action):
        attacked_action = int(np.argmax(attacked_q))
        policy_value_drop = float(clean_q[int(clean_action)] - clean_q[int(attacked_action)])
        target_value_gap = float(clean_q[int(clean_action)] - clean_q[int(target_action)])
        target_margin = self._target_margin(attacked_q, target_action)
        attacked_action_value = float(attacked_q[int(attacked_action)])
        return (
            float(policy_value_drop),
            int(attacked_action == int(target_action)),
            float(target_value_gap),
            int(attacked_action != int(clean_action)),
            float(target_margin),
            -attacked_action_value,
        )

    def _target_margin(self, q_values, target_action):
        if len(q_values) <= 1:
            return float(q_values[0])
        other_values = np.delete(q_values, int(target_action))
        if other_values.size == 0:
            return float(q_values[int(target_action)])
        return float(q_values[int(target_action)] - np.max(other_values))

    def _winner_margin(self, q_values):
        if len(q_values) <= 1:
            return float(q_values[0]) if len(q_values) == 1 else 0.0
        sorted_q = np.sort(q_values)
        return float(sorted_q[-1] - sorted_q[-2])

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


def _build_state_observation_wrapper(base_class):
    class StateObservationWrappedSatellite(base_class):
        _state_observation_attack_wrapper = True

        def get_current_state(self, destination, hops, is_computed, mission_state):
            clean_state = super().get_current_state(destination, hops, is_computed, mission_state)
            if _ATTACK_ENGINE is None or not _ATTACK_ENGINE.enabled:
                return clean_state
            return _ATTACK_ENGINE.maybe_attack(self, clean_state)

    StateObservationWrappedSatellite.__name__ = base_class.__name__
    StateObservationWrappedSatellite.__qualname__ = base_class.__qualname__
    return StateObservationWrappedSatellite


def install_state_observation_attack(level):
    global _ATTACK_ENGINE

    _ATTACK_ENGINE = StateObservationAttackEngine(level)
    current_class = SNC.Satellite_with_Computing
    if not getattr(current_class, "_state_observation_attack_wrapper", False):
        SNC.Satellite_with_Computing = _build_state_observation_wrapper(current_class)
    return dict(_ATTACK_ENGINE.profile)


def get_state_observation_attack_stats():
    if _ATTACK_ENGINE is None:
        return None
    return _ATTACK_ENGINE.get_stats()
