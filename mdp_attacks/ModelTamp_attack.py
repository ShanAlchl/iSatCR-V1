import math
import random
from types import MethodType

import numpy as np
import torch


_ATTACK_ENGINE = None


class ModelTampAttackEngine:
    PROFILE_MAP = {
        0: {
            "enabled": False,
            "strength_name": "off",
            "apply_probability": 0.0,
            "update_interval": 1,
            "batch_size": 0,
            "gradient_steps": 0,
            "margin_target": 0.0,
            "step_size": 0.0,
            "output_bias_scale": 0.0,
            "hidden_noise_scale": 0.0,
            "value_noise_scale": 0.0,
            "fro_ratio": 0.0,
            "elementwise_clip": 0.0,
            "amplification": 0.0,
            "injection_blend": 0.0,
        },
        1: {
            "enabled": True,
            "strength_name": "low",
            "apply_probability": 0.25,
            "update_interval": 30,
            "batch_size": 32,
            "gradient_steps": 1,
            "margin_target": 0.20,
            "step_size": 0.0020,
            "output_bias_scale": 0.0015,
            "hidden_noise_scale": 0.0004,
            "value_noise_scale": 0.0005,
            "fro_ratio": 0.010,
            "elementwise_clip": 0.006,
            "amplification": 1.10,
            "injection_blend": 0.25,
        },
        2: {
            "enabled": True,
            "strength_name": "medium",
            "apply_probability": 0.50,
            "update_interval": 15,
            "batch_size": 64,
            "gradient_steps": 2,
            "margin_target": 0.35,
            "step_size": 0.0040,
            "output_bias_scale": 0.0030,
            "hidden_noise_scale": 0.0008,
            "value_noise_scale": 0.0010,
            "fro_ratio": 0.018,
            "elementwise_clip": 0.010,
            "amplification": 1.50,
            "injection_blend": 0.45,
        },
        3: {
            "enabled": True,
            "strength_name": "high",
            "apply_probability": 0.75,
            "update_interval": 8,
            "batch_size": 96,
            "gradient_steps": 3,
            "margin_target": 0.55,
            "step_size": 0.0060,
            "output_bias_scale": 0.0055,
            "hidden_noise_scale": 0.0012,
            "value_noise_scale": 0.0015,
            "fro_ratio": 0.030,
            "elementwise_clip": 0.016,
            "amplification": 2.00,
            "injection_blend": 0.65,
        },
        4: {
            "enabled": True,
            "strength_name": "extreme",
            "apply_probability": 1.00,
            "update_interval": 1,
            "batch_size": 128,
            "gradient_steps": 4,
            "margin_target": 0.85,
            "step_size": 0.0090,
            "output_bias_scale": 0.0085,
            "hidden_noise_scale": 0.0018,
            "value_noise_scale": 0.0022,
            "fro_ratio": 0.045,
            "elementwise_clip": 0.024,
            "amplification": 2.80,
            "injection_blend": 0.90,
        },
    }

    def __init__(self, agent, attack_level=0):
        self.agent = agent
        self.attack_level = int(attack_level)
        if self.attack_level not in self.PROFILE_MAP:
            raise ValueError("ModelTampAttack_level must be one of {0,1,2,3,4}")

        self.profile = dict(self.PROFILE_MAP[self.attack_level])
        self.enabled = bool(self.profile["enabled"])
        self.device = getattr(agent, "device", torch.device("cpu"))
        self.action_dim = self._infer_action_dim(getattr(agent, "online_net", None))
        self.update_counter = 0
        self.reference_snapshots = {}
        self._stats = {
            "trigger_calls": 0,
            "applied_calls": 0,
            "update_triggers": 0,
            "load_triggers": 0,
            "target_update_triggers": 0,
            "skipped_probability": 0,
            "skipped_schedule": 0,
            "replay_batches": 0,
            "surrogate_batches": 0,
            "online_attack_calls": 0,
            "target_attack_calls": 0,
            "last_trigger": None,
            "last_batch_source": None,
            "last_online_loss": None,
            "last_target_loss": None,
            "last_online_delta_norm": 0.0,
            "last_target_delta_norm": 0.0,
            "last_online_drift_norm": 0.0,
            "last_target_drift_norm": 0.0,
            "max_online_delta_norm": 0.0,
            "max_target_delta_norm": 0.0,
        }
        self.capture_reference_snapshots()
        self._wrap_agent_methods()

    def get_install_profile(self):
        profile = dict(self.profile)
        profile["attack_level"] = self.attack_level
        return profile

    def get_stats(self):
        stats = dict(self._stats)
        stats["attack_level"] = self.attack_level
        stats["strength_name"] = self.profile["strength_name"]
        stats["apply_probability"] = self.profile["apply_probability"]
        stats["update_interval"] = self.profile["update_interval"]
        return stats

    def capture_reference_snapshots(self):
        self._capture_reference_snapshot("online_net")
        self._capture_reference_snapshot("target_net")

    def maybe_tamper(self, trigger, experiences=None, force=False):
        self._stats["trigger_calls"] += 1
        self._stats["last_trigger"] = trigger

        if not self.enabled:
            return None

        if trigger == "update":
            self.update_counter += 1
            self._stats["update_triggers"] += 1
            update_interval = max(int(self.profile["update_interval"]), 1)
            if self.update_counter % update_interval != 0:
                self._stats["skipped_schedule"] += 1
                return None
        elif trigger == "load":
            self._stats["load_triggers"] += 1
        elif trigger == "target_update":
            self._stats["target_update_triggers"] += 1

        if not force and random.random() > float(self.profile["apply_probability"]):
            self._stats["skipped_probability"] += 1
            return None

        batch = self._build_attack_batch()
        self._stats["last_batch_source"] = batch["source"]
        if batch["source"] == "replay":
            self._stats["replay_batches"] += 1
        else:
            self._stats["surrogate_batches"] += 1

        applied = False
        if trigger in {"update", "load"}:
            online_result = self._poison_network(
                getattr(self.agent, "online_net", None),
                batch["states"],
                batch["marks"],
                "online_net",
            )
            if online_result is not None:
                applied = True
                self._stats["online_attack_calls"] += 1
                self._stats["last_online_loss"] = online_result["loss"]
                self._stats["last_online_delta_norm"] = online_result["delta_norm"]
                self._stats["last_online_drift_norm"] = online_result["drift_norm"]
                self._stats["max_online_delta_norm"] = max(
                    self._stats["max_online_delta_norm"],
                    online_result["delta_norm"],
                )

        if trigger in {"load", "target_update"}:
            target_result = self._poison_network(
                getattr(self.agent, "target_net", None),
                batch["next_states"],
                batch["next_marks"],
                "target_net",
            )
            if target_result is not None:
                applied = True
                self._stats["target_attack_calls"] += 1
                self._stats["last_target_loss"] = target_result["loss"]
                self._stats["last_target_delta_norm"] = target_result["delta_norm"]
                self._stats["last_target_drift_norm"] = target_result["drift_norm"]
                self._stats["max_target_delta_norm"] = max(
                    self._stats["max_target_delta_norm"],
                    target_result["delta_norm"],
                )

        if applied:
            self._stats["applied_calls"] += 1
        return applied

    def _wrap_agent_methods(self):
        if getattr(self.agent, "_model_tamp_attack_wrapped", False):
            raise RuntimeError("A model tampering attack has already been installed on this agent.")

        self._original_update = self.agent.update
        self._original_load_model = getattr(self.agent, "load_model", None)
        self._original_target_update = getattr(self.agent, "target_update", None)

        def wrapped_update(_agent, experiences):
            result = self._original_update(experiences)
            self.maybe_tamper("update", experiences=experiences)
            return result

        self.agent.update = MethodType(wrapped_update, self.agent)

        if callable(self._original_load_model):
            def wrapped_load_model(_agent, *args, **kwargs):
                result = self._original_load_model(*args, **kwargs)
                self.capture_reference_snapshots()
                self.maybe_tamper("load", force=True)
                return result

            self.agent.load_model = MethodType(wrapped_load_model, self.agent)

        if callable(self._original_target_update) and getattr(self.agent, "target_net", None) is not None:
            def wrapped_target_update(_agent, *args, **kwargs):
                result = self._original_target_update(*args, **kwargs)
                self.maybe_tamper("target_update")
                return result

            self.agent.target_update = MethodType(wrapped_target_update, self.agent)

        self.agent._model_tamp_attack_wrapped = True
        self.agent._model_tamp_attack_engine = self

    def _build_attack_batch(self):
        replay_batch = self._sample_replay_batch()
        if replay_batch is not None:
            replay_batch["source"] = "replay"
            return replay_batch
        return self._build_surrogate_batch()

    def _sample_replay_batch(self):
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is None:
            return None

        samples = list(replay_buffer)
        if not samples:
            return None

        sample_size = min(len(samples), max(int(self.profile["batch_size"]), 8))
        if sample_size <= 0:
            return None

        priority_count = min(sample_size // 2, len(samples))
        reward_order = sorted(
            range(len(samples)),
            key=lambda index: self._safe_float(samples[index][3]),
        )
        selected_indices = reward_order[:priority_count]
        selected_set = set(selected_indices)
        remaining_indices = [index for index in range(len(samples)) if index not in selected_set]
        remaining_count = sample_size - len(selected_indices)
        if remaining_count > 0 and remaining_indices:
            selected_indices.extend(random.sample(remaining_indices, min(remaining_count, len(remaining_indices))))

        batch = [samples[index] for index in selected_indices]
        if not batch:
            return None

        states = np.asarray([np.asarray(item[0], dtype=np.float32) for item in batch], dtype=np.float32)
        marks = np.asarray([self._safe_mark(item[1]) for item in batch], dtype=np.int64)
        next_states = np.asarray([np.asarray(item[4], dtype=np.float32) for item in batch], dtype=np.float32)
        next_marks = np.asarray([self._infer_state_mark(state) for state in next_states], dtype=np.int64)

        if states.ndim != 2 or next_states.ndim != 2:
            return None

        return {
            "states": states,
            "marks": marks,
            "next_states": next_states,
            "next_marks": next_marks,
        }

    def _build_surrogate_batch(self):
        online_net = getattr(self.agent, "online_net", None)
        in_layer = getattr(online_net, "in_layer", None)
        state_dim = int(getattr(in_layer, "in_features", 0))
        batch_size = max(int(self.profile["batch_size"]), 16)
        states = np.random.uniform(0.0, 1.0, size=(batch_size, state_dim)).astype(np.float32)
        next_states = states + np.random.normal(0.0, 0.05, size=(batch_size, state_dim)).astype(np.float32)
        next_states = np.clip(next_states, -1.0, 1.5)

        if state_dim > 0:
            marks = np.random.choice([0, 1], size=batch_size, p=[0.65, 0.35]).astype(np.int64)
            next_marks = np.random.choice([0, 1], size=batch_size, p=[0.65, 0.35]).astype(np.int64)
            states[:, -1] = marks.astype(np.float32)
            next_states[:, -1] = next_marks.astype(np.float32)
        else:
            marks = np.zeros(batch_size, dtype=np.int64)
            next_marks = np.zeros(batch_size, dtype=np.int64)

        return {
            "source": "surrogate",
            "states": states,
            "marks": marks,
            "next_states": next_states,
            "next_marks": next_marks,
        }

    def _poison_network(self, network, states, marks, network_name):
        if network is None:
            return None
        if states is None or len(states) == 0:
            return None

        named_parameters = [(name, param) for name, param in network.named_parameters() if param.requires_grad]
        if not named_parameters:
            return None

        baseline = {name: param.detach().clone() for name, param in named_parameters}
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        mark_tensor = torch.as_tensor(marks, dtype=torch.long, device=self.device)
        malicious_hist = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        optimal_hist = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        last_loss = None

        for _ in range(int(self.profile["gradient_steps"])):
            network.zero_grad()
            loss, attack_metadata = self._build_attack_objective(network, state_tensor, mark_tensor)
            if loss is None or not bool(torch.isfinite(loss).item()):
                break

            gradients = torch.autograd.grad(
                loss,
                [param for _name, param in named_parameters],
                allow_unused=True,
            )

            with torch.no_grad():
                for (_name, param), gradient in zip(named_parameters, gradients):
                    if gradient is None:
                        continue
                    gradient_norm = gradient.norm()
                    if not bool(torch.isfinite(gradient_norm).item()) or float(gradient_norm.item()) == 0.0:
                        continue
                    param.add_(
                        -float(self.profile["step_size"])
                        * gradient
                        / (gradient_norm + 1e-12)
                    )

            malicious_hist += torch.bincount(
                attack_metadata["malicious_actions"],
                minlength=self.action_dim,
            ).to(self.device, dtype=torch.float32)
            optimal_hist += torch.bincount(
                attack_metadata["optimal_actions"],
                minlength=self.action_dim,
            ).to(self.device, dtype=torch.float32)
            last_loss = float(loss.detach().cpu().item())

        with torch.no_grad():
            self._apply_output_layer_bias(network, malicious_hist, optimal_hist)
            self._apply_hidden_layer_noise(network)
            delta_norm = self._project_delta(network, baseline)

        drift_norm = self._compute_drift_norm(network, network_name)
        return {
            "loss": last_loss,
            "delta_norm": delta_norm,
            "drift_norm": drift_norm,
        }

    def _build_attack_objective(self, network, state_tensor, mark_tensor):
        q_values = network(state_tensor)
        if q_values.ndim != 2 or q_values.shape[1] == 0:
            return None, None

        valid_mask = self._build_valid_action_mask(q_values, mark_tensor)
        masked_best = q_values.masked_fill(~valid_mask, float("-inf"))
        masked_worst = q_values.masked_fill(~valid_mask, float("inf"))
        optimal_actions = masked_best.argmax(dim=1)
        malicious_actions = masked_worst.argmin(dim=1)

        q_best = q_values.gather(1, optimal_actions.unsqueeze(1)).squeeze(1)
        q_worst = q_values.gather(1, malicious_actions.unsqueeze(1)).squeeze(1)
        margin_target = float(self.profile["margin_target"])
        attack_loss = torch.relu(q_best - q_worst + margin_target).mean()
        attack_loss = attack_loss + 0.05 * q_best.mean() - 0.02 * q_worst.mean()

        return attack_loss, {
            "optimal_actions": optimal_actions.detach(),
            "malicious_actions": malicious_actions.detach(),
        }

    def _build_valid_action_mask(self, q_values, mark_tensor):
        batch_size, action_dim = q_values.shape
        valid_mask = torch.ones((batch_size, action_dim), dtype=torch.bool, device=q_values.device)
        if action_dim > 4:
            computed_mask = mark_tensor > 0
            valid_mask[computed_mask, 4:] = False
        return valid_mask

    def _apply_output_layer_bias(self, network, malicious_hist, optimal_hist):
        output_layer = self._get_output_layer(network)
        if output_layer is None:
            return

        action_count = int(output_layer.weight.shape[0])
        if action_count <= 0:
            return

        direction = torch.zeros(action_count, dtype=output_layer.weight.dtype, device=output_layer.weight.device)
        if malicious_hist.numel() > 0 and float(malicious_hist.sum().item()) > 0:
            direction += (
                malicious_hist[:action_count].to(output_layer.weight.device, dtype=output_layer.weight.dtype)
                / malicious_hist[:action_count].sum()
            )
        if optimal_hist.numel() > 0 and float(optimal_hist.sum().item()) > 0:
            direction -= (
                optimal_hist[:action_count].to(output_layer.weight.device, dtype=output_layer.weight.dtype)
                / optimal_hist[:action_count].sum()
            )

        scale = float(self.profile["output_bias_scale"])
        if scale <= 0.0:
            return

        if output_layer.bias is not None:
            output_layer.bias.add_(direction * scale)
        output_layer.weight.add_(direction.unsqueeze(1) * scale)

    def _apply_hidden_layer_noise(self, network):
        hidden_noise_scale = float(self.profile["hidden_noise_scale"])
        if hidden_noise_scale > 0.0:
            layers = [getattr(network, "in_layer", None)]
            layers.extend(list(getattr(network, "mid_layers", [])))
            for layer in layers:
                if layer is None:
                    continue
                if getattr(layer, "weight", None) is not None:
                    layer.weight.add_(torch.randn_like(layer.weight) * hidden_noise_scale)
                if getattr(layer, "bias", None) is not None:
                    layer.bias.add_(torch.randn_like(layer.bias) * hidden_noise_scale * 0.5)

        value_noise_scale = float(self.profile["value_noise_scale"])
        if value_noise_scale > 0.0 and getattr(network, "value_stream", None) is not None:
            network.value_stream.weight.add_(torch.randn_like(network.value_stream.weight) * value_noise_scale)
            if network.value_stream.bias is not None:
                network.value_stream.bias.add_(torch.randn_like(network.value_stream.bias) * value_noise_scale * 0.5)

    def _project_delta(self, network, baseline):
        amplification = float(self.profile["amplification"])
        injection_blend = float(self.profile["injection_blend"])
        elementwise_clip = float(self.profile["elementwise_clip"])

        total_sq = 0.0
        for name, param in network.named_parameters():
            if not param.requires_grad or name not in baseline:
                continue
            delta = (param - baseline[name]) * amplification * injection_blend
            if elementwise_clip > 0.0:
                delta = torch.clamp(delta, -elementwise_clip, elementwise_clip)
            param.copy_(baseline[name] + delta)
            total_sq += float(delta.pow(2).sum().item())

        current_norm = math.sqrt(total_sq)
        budget = self._snapshot_norm(baseline) * float(self.profile["fro_ratio"])
        if budget > 0.0 and current_norm > budget:
            scale = budget / max(current_norm, 1e-12)
            total_sq = 0.0
            for name, param in network.named_parameters():
                if not param.requires_grad or name not in baseline:
                    continue
                delta = (param - baseline[name]) * scale
                param.copy_(baseline[name] + delta)
                total_sq += float(delta.pow(2).sum().item())
            current_norm = math.sqrt(total_sq)
        return current_norm

    def _compute_drift_norm(self, network, network_name):
        reference = self.reference_snapshots.get(network_name)
        if not reference:
            return 0.0

        total_sq = 0.0
        for name, param in network.named_parameters():
            if name not in reference:
                continue
            total_sq += float((param.detach() - reference[name]).pow(2).sum().item())
        return math.sqrt(total_sq)

    def _capture_reference_snapshot(self, network_name):
        network = getattr(self.agent, network_name, None)
        if network is None:
            return
        self.reference_snapshots[network_name] = {
            name: param.detach().clone()
            for name, param in network.named_parameters()
            if param.requires_grad
        }

    def _infer_action_dim(self, network):
        if network is None:
            return 0
        if getattr(network, "advantage_stream", None) is not None:
            return int(network.advantage_stream.out_features)
        if getattr(network, "out_layer", None) is not None:
            return int(network.out_layer.out_features)
        return 0

    def _get_output_layer(self, network):
        if getattr(network, "advantage_stream", None) is not None:
            return network.advantage_stream
        return getattr(network, "out_layer", None)

    def _infer_state_mark(self, state):
        if len(state) == 0:
            return 0
        return self._safe_mark(state[-1])

    def _safe_mark(self, value):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return 0

    def _safe_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _snapshot_norm(self, snapshot):
        total_sq = 0.0
        for value in snapshot.values():
            total_sq += float(value.pow(2).sum().item())
        return math.sqrt(total_sq)


def install_model_tamp_attack(level, agent=None):
    global _ATTACK_ENGINE

    resolved_level = level
    resolved_agent = agent
    if agent is None and not isinstance(level, (int, np.integer)):
        resolved_agent = level
        resolved_level = 0
    elif agent is not None and not isinstance(level, (int, np.integer)) and isinstance(agent, (int, np.integer)):
        resolved_agent = level
        resolved_level = agent

    if resolved_agent is None:
        raise ValueError("install_model_tamp_attack requires an agent instance.")

    _ATTACK_ENGINE = ModelTampAttackEngine(agent=resolved_agent, attack_level=int(resolved_level))
    return _ATTACK_ENGINE.get_install_profile()


def get_model_tamp_attack_stats():
    if _ATTACK_ENGINE is None:
        return None
    return _ATTACK_ENGINE.get_stats()
