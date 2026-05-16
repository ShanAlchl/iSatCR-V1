from typing import Dict, List, Optional, Tuple

import numpy as np

from .parameter_interfaces import AttackModuleInput, METRIC_NAMES


FAILURE_METRIC_ORDER: Tuple[str, ...] = METRIC_NAMES

FAILURE_METRIC_REFERENCES: Dict[str, float] = {
    "PacketLossRate": 0.0,
    "NetworkThroughput": 55.302,
    "BandwidthUtilization": 0.0006,
    "AvgPacketNodeVisits": 3.100,
    "CumulativeReward": 2.752858,
    "AverageInferenceTime": 0.300,
    "AverageE2eDelay": 1.894,
    "AverageHopCount": 4.000,
    "AverageComputingRatio": 0.0233,
    "ComputingWaitingTime": 1.645,
    "AverageEndingReward": 0.7052962623742696,
}

HIGHER_IS_WORSE = {
    "PacketLossRate",
    "AverageE2eDelay",
    "ComputingWaitingTime",
    "AverageInferenceTime",
    "AverageHopCount",
    "AvgPacketNodeVisits",
}

LOWER_IS_WORSE = {
    "NetworkThroughput",
    "BandwidthUtilization",
    "AverageComputingRatio",
    "CumulativeReward",
    "AverageEndingReward",
}

DEFAULT_TERMINAL_RISK_WEIGHTS: Dict[str, float] = {
    "packet_loss": 0.35,
    "e2e_delay": 0.20,
    "throughput": 0.20,
    "ending_reward": 0.25,
}

DEFAULT_DECISION_FORMULA_WEIGHTS: Dict[str, float] = {
    "w_mean": 0.60,
    "w_p75": 0.25,
    "w_max": 0.10,
    "w_slope_pos": 0.10,
    "w_std_penalty": 0.20,
}

DEFAULT_DECISION_MODEL_WEIGHTS: Dict[str, float] = {
    "w_mean": 0.60,
    "w_p75": 0.25,
    "w_max": 0.10,
    "w_slope_pos": 0.10,
    "w_std_penalty": 0.20,
    "w_high_ratio": 0.50,
    "w_terminal_gap": 0.50,
}


class FAHP:
    """Fuzzy Analytic Hierarchy Process for generating criterion weights."""

    def __init__(self, fuzzy_matrix: np.ndarray):
        self.matrix = np.array(fuzzy_matrix, dtype=float)
        self.num_criteria = self.matrix.shape[0]

    def calculate_weights(self) -> np.ndarray:
        row_sums = self.matrix.sum(axis=1)
        total_sum = row_sums.sum(axis=0)
        if total_sum[2] == 0:
            inv_total = np.zeros(3)
        else:
            inv_total = np.array([1.0 / total_sum[2], 1.0 / total_sum[1], 1.0 / total_sum[0]])

        synthetic_extents = np.zeros_like(row_sums)
        for i in range(self.num_criteria):
            synthetic_extents[i, 0] = row_sums[i, 0] * inv_total[0]
            synthetic_extents[i, 1] = row_sums[i, 1] * inv_total[1]
            synthetic_extents[i, 2] = row_sums[i, 2] * inv_total[2]

        def degree_of_possibility(m1, m2):
            l1, m1_mid, u1 = m1
            l2, m2_mid, u2 = m2
            if m1_mid >= m2_mid:
                return 1.0
            if l2 >= u1:
                return 0.0
            denominator = (m1_mid - u1) - (m2_mid - l2)
            if denominator == 0:
                return 0.0
            return (l2 - u1) / denominator

        weights = []
        for i in range(self.num_criteria):
            min_degree = 1.0
            for j in range(self.num_criteria):
                if i == j:
                    continue
                min_degree = min(min_degree, degree_of_possibility(synthetic_extents[i], synthetic_extents[j]))
            weights.append(min_degree)

        weights = np.array(weights, dtype=float)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        return weights


class CloudModel:
    def __init__(self, ex: float, en: float, he: float):
        self.ex = ex
        self.en = en
        self.he = he

    def get_membership(self, x: float, num_droplets: int = 100) -> float:
        if self.en == 0:
            return 1.0 if x == self.ex else 0.0

        en_primes = np.random.normal(loc=self.en, scale=self.he, size=num_droplets)
        en_primes = np.maximum(en_primes, 1e-6)
        memberships = np.exp(-((x - self.ex) ** 2) / (2 * (en_primes ** 2)))
        return float(np.mean(memberships))


class AgentFailureEvaluator:
    """Aggregates cloud-model and monotonic-risk scores for failure assessment."""

    def __init__(
        self,
        criteria_weights: np.ndarray,
        cloud_configs: Dict[str, Dict[str, Tuple[float, float, float]]],
        v2_failure_threshold: float = 0.35,
        terminal_threshold_v2: float = 0.55,
        terminal_risk_weights: Optional[Dict[str, float]] = None,
        convergence_ratio: float = 0.2,
        convergence_alpha: float = 0.7,
        decision_formula_weights: Optional[Dict[str, float]] = None,
        enable_decision_tail_boost: bool = False,
        decision_tail_gamma: float = 1.0,
        decision_model_type: str = "fixed_linear",
        decision_model_weights: Optional[Dict[str, float]] = None,
        decision_model_bias: float = 0.0,
    ):
        self.weights = np.array(criteria_weights, dtype=float)
        self.cloud_configs = cloud_configs
        self.v2_failure_threshold = float(v2_failure_threshold)
        self.terminal_threshold_v2 = float(np.clip(terminal_threshold_v2, 0.01, 0.99))
        self.terminal_risk_weights = self._normalize_terminal_risk_weights(
            terminal_risk_weights or DEFAULT_TERMINAL_RISK_WEIGHTS
        )
        self.convergence_ratio = float(np.clip(convergence_ratio, 0.05, 0.5))
        self.convergence_alpha = float(np.clip(convergence_alpha, 0.0, 1.0))
        self.decision_formula_weights = dict(DEFAULT_DECISION_FORMULA_WEIGHTS)
        if decision_formula_weights:
            for key in DEFAULT_DECISION_FORMULA_WEIGHTS.keys():
                if key in decision_formula_weights:
                    self.decision_formula_weights[key] = float(decision_formula_weights[key])
        self.enable_decision_tail_boost = bool(enable_decision_tail_boost)
        self.decision_tail_gamma = float(np.clip(decision_tail_gamma, 0.5, 1.0))
        self.decision_model_type = str(decision_model_type).strip().lower() or "fixed_linear"
        if self.decision_model_type not in {"fixed_linear", "learned_linear"}:
            self.decision_model_type = "fixed_linear"
        self.decision_model_weights = dict(DEFAULT_DECISION_MODEL_WEIGHTS)
        if decision_model_weights:
            for key in DEFAULT_DECISION_MODEL_WEIGHTS.keys():
                if key in decision_model_weights:
                    self.decision_model_weights[key] = float(decision_model_weights[key])
        self.decision_model_bias = float(decision_model_bias)

    def set_v2_failure_threshold(self, threshold: float):
        self.v2_failure_threshold = float(np.clip(threshold, 0.01, 0.99))

    def set_terminal_threshold_v2(self, threshold: float):
        self.terminal_threshold_v2 = float(np.clip(threshold, 0.01, 0.99))

    @staticmethod
    def _normalize_terminal_risk_weights(weights: Dict[str, float]) -> Dict[str, float]:
        keys = ("packet_loss", "e2e_delay", "throughput", "ending_reward")
        normalized = {key: max(0.0, float(weights.get(key, 0.0))) for key in keys}
        total = float(sum(normalized.values()))
        if total <= 1e-12:
            return dict(DEFAULT_TERMINAL_RISK_WEIGHTS)
        return {key: float(value / total) for key, value in normalized.items()}

    def set_terminal_risk_weights(self, weights: Dict[str, float]):
        self.terminal_risk_weights = self._normalize_terminal_risk_weights(weights)

    def get_terminal_risk_weights(self) -> Dict[str, float]:
        return dict(self.terminal_risk_weights)

    def set_decision_formula_config(
        self,
        decision_formula_weights: Optional[Dict[str, float]] = None,
        enable_decision_tail_boost: Optional[bool] = None,
        decision_tail_gamma: Optional[float] = None,
        decision_model_type: Optional[str] = None,
        decision_model_weights: Optional[Dict[str, float]] = None,
        decision_model_bias: Optional[float] = None,
    ):
        if decision_formula_weights:
            for key in self.decision_formula_weights.keys():
                if key in decision_formula_weights:
                    self.decision_formula_weights[key] = float(decision_formula_weights[key])
        if enable_decision_tail_boost is not None:
            self.enable_decision_tail_boost = bool(enable_decision_tail_boost)
        if decision_tail_gamma is not None:
            self.decision_tail_gamma = float(np.clip(decision_tail_gamma, 0.5, 1.0))
        if decision_model_type is not None:
            normalized = str(decision_model_type).strip().lower()
            if normalized in {"fixed_linear", "learned_linear"}:
                self.decision_model_type = normalized
        if decision_model_weights:
            for key in self.decision_model_weights.keys():
                if key in decision_model_weights:
                    self.decision_model_weights[key] = float(decision_model_weights[key])
        if decision_model_bias is not None:
            self.decision_model_bias = float(decision_model_bias)

    def get_decision_formula_config(self) -> Dict[str, float]:
        return {
            **self.decision_formula_weights,
            "enable_decision_tail_boost": bool(self.enable_decision_tail_boost),
            "decision_tail_gamma": float(self.decision_tail_gamma),
            "decision_model_type": self.decision_model_type,
            "decision_model_weights": dict(self.decision_model_weights),
            "decision_model_bias": float(self.decision_model_bias),
        }

    def _decision_feature_contributions(
        self,
        converged_mean_v2: float,
        converged_p75_v2: float,
        converged_max_v2: float,
        converged_slope_v2: float,
        converged_std_v2: float,
        converged_high_ratio_v2: float,
        terminal_score_gap_v2: float,
        weights: Dict[str, float],
        bias: float,
    ) -> Dict[str, float]:
        return {
            "mean_term": float(weights["w_mean"] * float(converged_mean_v2)),
            "p75_term": float(weights["w_p75"] * float(converged_p75_v2)),
            "max_term": float(weights["w_max"] * float(converged_max_v2)),
            "slope_term": float(weights["w_slope_pos"] * max(0.0, float(converged_slope_v2))),
            "std_penalty_term": float(-weights["w_std_penalty"] * float(converged_std_v2)),
            "high_ratio_term": float(weights.get("w_high_ratio", 0.0) * float(converged_high_ratio_v2)),
            "terminal_gap_term": float(weights.get("w_terminal_gap", 0.0) * float(terminal_score_gap_v2)),
            "bias": float(bias),
        }

    def compute_decision_score_v2(
        self,
        converged_mean_v2: float,
        converged_p75_v2: float,
        converged_max_v2: float,
        converged_slope_v2: float,
        converged_std_v2: float,
        converged_high_ratio_v2: float = 0.0,
        terminal_score_gap_v2: float = 0.0,
    ) -> Tuple[float, float, Dict[str, float], str]:
        if self.decision_model_type == "learned_linear":
            contributions = self._decision_feature_contributions(
                converged_mean_v2=converged_mean_v2,
                converged_p75_v2=converged_p75_v2,
                converged_max_v2=converged_max_v2,
                converged_slope_v2=converged_slope_v2,
                converged_std_v2=converged_std_v2,
                converged_high_ratio_v2=converged_high_ratio_v2,
                terminal_score_gap_v2=terminal_score_gap_v2,
                weights=self.decision_model_weights,
                bias=self.decision_model_bias,
            )
            linear_score = float(sum(contributions.values()))
            decision_score = self._sigmoid(linear_score)
            return decision_score, linear_score, contributions, "v6_learned_linear_plus_features"

        contributions = self._decision_feature_contributions(
            converged_mean_v2=converged_mean_v2,
            converged_p75_v2=converged_p75_v2,
            converged_max_v2=converged_max_v2,
            converged_slope_v2=converged_slope_v2,
            converged_std_v2=converged_std_v2,
            converged_high_ratio_v2=0.0,
            terminal_score_gap_v2=0.0,
            weights=self.decision_formula_weights,
            bias=0.0,
        )
        raw_linear_score = float(sum(contributions.values()))
        linear_score = float(np.clip(raw_linear_score, 0.0, 1.0))
        contributions["bias"] = float(contributions["bias"] + (linear_score - raw_linear_score))
        custom_fixed_weights = any(
            abs(float(self.decision_formula_weights.get(key, 0.0)) - value) > 1e-12
            for key, value in DEFAULT_DECISION_FORMULA_WEIGHTS.items()
        )
        if not self.enable_decision_tail_boost:
            return linear_score, linear_score, contributions, "v4" if custom_fixed_weights else "v3"
        tail_score = float(np.clip(np.power(linear_score, self.decision_tail_gamma), 0.0, 1.0))
        return tail_score, linear_score, contributions, "v4"

    def evaluate_performance(self, metrics_data: Dict[str, float], target_level: str = "Failure") -> float:
        metrics_keys = list(self.cloud_configs.keys())
        if len(metrics_keys) != len(self.weights):
            raise ValueError("Metric count does not match criteria weight count")

        total_membership = 0.0
        for idx, key in enumerate(metrics_keys):
            if key not in metrics_data or target_level not in self.cloud_configs[key]:
                continue
            ex, en, he = self.cloud_configs[key][target_level]
            cloud = CloudModel(ex, en, he)
            membership = cloud.get_membership(float(metrics_data[key]))
            total_membership += self.weights[idx] * membership

        return float(total_membership)

    def _metric_scale(self, metric_name: str, reference: float) -> float:
        abs_ref = abs(float(reference))
        if metric_name in {"PacketLossRate", "BandwidthUtilization", "AverageComputingRatio"}:
            return max(abs_ref * 0.5, 0.05)
        if metric_name in {"AverageInferenceTime", "AverageE2eDelay", "ComputingWaitingTime"}:
            return max(abs_ref * 0.6, 0.5)
        return max(abs_ref * 0.35, 1e-3)

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _zero_based_risk(self, raw: float) -> float:
        # Map neutral raw=0 to risk=0 while preserving monotonicity.
        return float(np.clip(2.0 * self._sigmoid(raw) - 1.0, 0.0, 1.0))

    def evaluate_performance_v2(self, metrics_data: Dict[str, float]) -> Dict[str, float]:
        metric_risks: Dict[str, float] = {}
        weighted_score = 0.0
        for idx, metric_name in enumerate(FAILURE_METRIC_ORDER):
            if metric_name not in metrics_data:
                continue
            value = float(metrics_data[metric_name])
            reference = float(FAILURE_METRIC_REFERENCES[metric_name])
            scale = self._metric_scale(metric_name, reference)

            if metric_name in HIGHER_IS_WORSE:
                raw = (value - reference) / scale
            elif metric_name in LOWER_IS_WORSE:
                raw = (reference - value) / scale
            else:
                raw = abs(value - reference) / scale

            risk = self._zero_based_risk(raw)
            metric_risks[metric_name] = risk
            weighted_score += float(self.weights[idx]) * risk

        return {
            "failure_score_v2": float(np.clip(weighted_score, 0.0, 1.0)),
            "metric_risks": metric_risks,
        }

    def _is_true_failure_v2(self, metrics_data: Dict[str, float], reward_threshold: float = 0.5) -> bool:
        ending_reward = float(metrics_data.get("AverageEndingReward", 0.0))
        packet_loss = float(metrics_data.get("PacketLossRate", 0.0))
        e2e_delay = float(metrics_data.get("AverageE2eDelay", 0.0))
        waiting_time = float(metrics_data.get("ComputingWaitingTime", 0.0))
        throughput = float(metrics_data.get("NetworkThroughput", 0.0))
        cumulative_reward = float(metrics_data.get("CumulativeReward", 0.0))

        hard_failure = ending_reward < reward_threshold
        risk_hits = 0
        if packet_loss >= 0.22:
            risk_hits += 1
        if e2e_delay >= max(6.0, FAILURE_METRIC_REFERENCES["AverageE2eDelay"] * 3.0):
            risk_hits += 1
        if waiting_time >= max(4.0, FAILURE_METRIC_REFERENCES["ComputingWaitingTime"] * 2.5):
            risk_hits += 1
        if throughput <= max(1e-6, FAILURE_METRIC_REFERENCES["NetworkThroughput"] * 0.25):
            risk_hits += 1
        if cumulative_reward <= min(-5.0, FAILURE_METRIC_REFERENCES["CumulativeReward"] * -2.0):
            risk_hits += 1

        return bool(hard_failure or risk_hits >= 2)

    def _terminal_hard_failure(self, metrics_data: Dict[str, float], reward_threshold: float = 0.5) -> bool:
        ending_reward = float(metrics_data.get("AverageEndingReward", 0.0))
        packet_loss = float(metrics_data.get("PacketLossRate", 0.0))
        e2e_delay = float(metrics_data.get("AverageE2eDelay", 0.0))
        throughput = float(metrics_data.get("NetworkThroughput", 0.0))

        if ending_reward < reward_threshold:
            return True
        if packet_loss >= 0.25:
            return True
        if e2e_delay >= max(6.0, FAILURE_METRIC_REFERENCES["AverageE2eDelay"] * 3.0):
            return True
        if throughput <= max(1e-6, FAILURE_METRIC_REFERENCES["NetworkThroughput"] * 0.25):
            return True
        return False

    def _terminal_risk_score(
        self,
        metrics_data: Dict[str, float],
        reward_threshold: float = 0.5,
        override_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        packet_loss = float(metrics_data.get("PacketLossRate", 0.0))
        e2e_delay = float(metrics_data.get("AverageE2eDelay", 0.0))
        throughput = float(metrics_data.get("NetworkThroughput", 0.0))
        ending_reward = float(metrics_data.get("AverageEndingReward", 0.0))

        packet_loss_risk = self._zero_based_risk(packet_loss / 0.25)
        delay_reference = max(1e-6, FAILURE_METRIC_REFERENCES["AverageE2eDelay"])
        delay_risk = self._zero_based_risk((e2e_delay - delay_reference) / max(1.0, delay_reference * 2.0))
        throughput_reference = max(1e-6, FAILURE_METRIC_REFERENCES["NetworkThroughput"])
        throughput_risk = self._zero_based_risk(
            (throughput_reference - throughput) / max(5.0, throughput_reference * 0.5)
        )
        reward_risk = self._zero_based_risk(
            (reward_threshold - ending_reward) / max(0.2, abs(reward_threshold) * 0.5)
        )

        weights = self._normalize_terminal_risk_weights(override_weights or self.terminal_risk_weights)
        terminal_score = (
            weights["packet_loss"] * packet_loss_risk
            + weights["e2e_delay"] * delay_risk
            + weights["throughput"] * throughput_risk
            + weights["ending_reward"] * reward_risk
        )
        return float(np.clip(terminal_score, 0.0, 1.0))

    def _convergence_window(self, step_scores: List[Dict]) -> Tuple[int, List[Dict]]:
        if not step_scores:
            return 0, []
        total_steps = len(step_scores)
        window_size = max(1, int(np.ceil(total_steps * self.convergence_ratio)))
        start_idx = max(0, total_steps - window_size)
        return start_idx, step_scores[start_idx:]

    @staticmethod
    def _prediction_metrics(pred: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        tp = float(np.sum(pred & labels))
        fp = float(np.sum(pred & ~labels))
        tn = float(np.sum(~pred & ~labels))
        fn = float(np.sum(~pred & labels))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / max(1.0, tp + fp + tn + fn)
        tnr = tn / (tn + fp + 1e-8)
        balanced_accuracy = 0.5 * (recall + tnr)
        return {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
        }

    @staticmethod
    def _objective_value(metrics: Dict[str, float], objective: str) -> float:
        if objective == "accuracy":
            return float(metrics["accuracy"])
        if objective == "balanced_accuracy":
            return float(metrics["balanced_accuracy"])
        if objective == "recall_at_precision":
            return float(metrics["recall"])
        return float(metrics["f1"])

    @classmethod
    def calibrate_failure_threshold(
        cls,
        predicted_scores: List[float],
        true_labels: List[bool],
        objective: str = "f1",
        min_precision: float = 0.50,
    ) -> Dict[str, float]:
        if not predicted_scores or len(predicted_scores) != len(true_labels):
            return {
                "threshold": 0.35,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "objective_score": 0.0,
            }

        scores = np.asarray(predicted_scores, dtype=float)
        labels = np.asarray(true_labels, dtype=bool)
        candidates = np.unique(np.clip(scores, 0.0, 1.0))
        if len(candidates) == 0:
            candidates = np.array([0.35], dtype=float)

        best_score = -1.0
        best_metrics = None
        best_threshold = float(candidates[0])
        for threshold in candidates:
            pred = scores >= threshold
            metrics = cls._prediction_metrics(pred, labels)
            if objective == "recall_at_precision" and metrics["precision"] < min_precision:
                continue
            score = cls._objective_value(metrics, objective)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_threshold = float(threshold)

        if best_metrics is None:
            best_metrics = cls._prediction_metrics(scores >= 0.35, labels)
            best_score = cls._objective_value(best_metrics, "f1")
            best_threshold = 0.35

        return {
            "threshold": float(best_threshold),
            "f1": float(best_metrics["f1"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "objective_score": float(best_score),
        }

    @classmethod
    def calibrate_dual_failure_threshold(
        cls,
        decision_scores: List[float],
        terminal_scores: List[float],
        true_labels: List[bool],
        objective: str = "f1",
        min_precision: float = 0.50,
        terminal_threshold_candidates: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if not decision_scores or len(decision_scores) != len(true_labels) or len(terminal_scores) != len(true_labels):
            return {
                "decision_threshold": 0.35,
                "terminal_threshold": 0.55,
                "threshold": 0.35,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "objective_score": 0.0,
            }

        decision_np = np.asarray(decision_scores, dtype=float)
        terminal_np = np.asarray(terminal_scores, dtype=float)
        labels = np.asarray(true_labels, dtype=bool)

        decision_candidates = np.unique(np.clip(decision_np, 0.0, 1.0))
        if len(decision_candidates) == 0:
            decision_candidates = np.array([0.35], dtype=float)
        terminal_candidates = (
            np.asarray(terminal_threshold_candidates, dtype=float)
            if terminal_threshold_candidates is not None
            else np.arange(0.45, 0.751, 0.01, dtype=float)
        )

        best_score = -1.0
        best_metrics = None
        best_decision = float(decision_candidates[0])
        best_terminal = float(terminal_candidates[0]) if len(terminal_candidates) > 0 else 0.55
        for decision_threshold in decision_candidates:
            decision_mask = decision_np >= decision_threshold
            for terminal_threshold in terminal_candidates:
                pred = decision_mask & (terminal_np >= terminal_threshold)
                metrics = cls._prediction_metrics(pred, labels)
                if objective == "recall_at_precision" and metrics["precision"] < min_precision:
                    continue
                score = cls._objective_value(metrics, objective)
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    best_decision = float(decision_threshold)
                    best_terminal = float(terminal_threshold)

        if best_metrics is None:
            fallback_pred = (decision_np >= 0.35) & (terminal_np >= 0.55)
            best_metrics = cls._prediction_metrics(fallback_pred, labels)
            best_score = cls._objective_value(best_metrics, "f1")
            best_decision = 0.35
            best_terminal = 0.55

        return {
            "decision_threshold": float(best_decision),
            "terminal_threshold": float(best_terminal),
            "threshold": float(best_decision),
            "f1": float(best_metrics["f1"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "objective_score": float(best_score),
        }

    def evaluate_log_file(
        self,
        file_path: str,
        target_level: str = "Failure",
        failure_threshold: float = 0.5,
        real_failure_reward_threshold: float = 0.5,
    ) -> Dict:
        samples = AttackModuleInput.parse_from_log_file(file_path)
        step_scores = []
        for sample in samples:
            metrics_data = sample.Metrics.__dict__
            score_v1 = self.evaluate_performance(metrics_data, target_level=target_level)
            v2_result = self.evaluate_performance_v2(metrics_data)
            step_scores.append(
                {
                    "step_index": sample.StepIndex,
                    "total_membership": float(score_v1),
                    "total_membership_v2": float(v2_result["failure_score_v2"]),
                    "metrics": metrics_data,
                    "metric_risks_v2": v2_result["metric_risks"],
                    "true_failure_v2": self._is_true_failure_v2(metrics_data, reward_threshold=real_failure_reward_threshold),
                }
            )

        v1_scores = [item["total_membership"] for item in step_scores]
        v2_scores = [item["total_membership_v2"] for item in step_scores]

        max_score = max(v1_scores, default=0.0)
        mean_score = float(np.mean(v1_scores)) if step_scores else 0.0
        score_uncertainty = float(np.std(v1_scores)) if step_scores else 0.0

        max_score_v2 = max(v2_scores, default=0.0)
        mean_score_v2 = float(np.mean(v2_scores)) if step_scores else 0.0
        score_uncertainty_v2 = float(np.std(v2_scores)) if step_scores else 0.0

        convergence_start_idx, convergence_scores = self._convergence_window(step_scores)
        convergence_start_step = int(convergence_scores[0]["step_index"]) if convergence_scores else 0
        converged_v2_scores = [item["total_membership_v2"] for item in convergence_scores]
        converged_mean_v2 = float(np.mean(converged_v2_scores)) if converged_v2_scores else 0.0
        converged_p75_v2 = float(np.percentile(converged_v2_scores, 75)) if converged_v2_scores else 0.0
        converged_std_v2 = float(np.std(converged_v2_scores)) if converged_v2_scores else 0.0
        converged_max_v2 = float(np.max(converged_v2_scores)) if converged_v2_scores else 0.0
        converged_high_ratio_v2 = (
            float(np.mean(np.asarray(converged_v2_scores, dtype=float) >= 0.50)) if converged_v2_scores else 0.0
        )
        if len(convergence_scores) >= 2:
            x_values = np.asarray([float(item["step_index"]) for item in convergence_scores], dtype=float)
            y_values = np.asarray(converged_v2_scores, dtype=float)
            slope = float(np.polyfit(x_values, y_values, deg=1)[0])
        else:
            slope = 0.0
        terminal_metrics = samples[-1].Metrics.__dict__ if samples else {}
        terminal_hard_failure = self._terminal_hard_failure(
            terminal_metrics,
            reward_threshold=real_failure_reward_threshold,
        )
        terminal_risk_score = self._terminal_risk_score(
            terminal_metrics,
            reward_threshold=real_failure_reward_threshold,
        )
        terminal_score_gap_v2 = float(max(0.0, terminal_risk_score - converged_mean_v2))
        (
            decision_score_v2,
            decision_score_v2_linear,
            decision_feature_contributions,
            decision_score_formula_version,
        ) = self.compute_decision_score_v2(
            converged_mean_v2=converged_mean_v2,
            converged_p75_v2=converged_p75_v2,
            converged_max_v2=converged_max_v2,
            converged_slope_v2=slope,
            converged_std_v2=converged_std_v2,
            converged_high_ratio_v2=converged_high_ratio_v2,
            terminal_score_gap_v2=terminal_score_gap_v2,
        )

        system_failure = max_score > failure_threshold
        predicted_failure_v2 = (
            decision_score_v2 >= self.v2_failure_threshold
            and terminal_risk_score >= self.terminal_threshold_v2
        )

        true_failure = any(
            sample.Metrics.AverageEndingReward < real_failure_reward_threshold
            for sample in samples
        )
        convergence_true_flags = [bool(item.get("true_failure_v2", False)) for item in convergence_scores]
        convergence_true_ratio = float(np.mean(convergence_true_flags)) if convergence_true_flags else 0.0
        convergence_true_failure = bool(convergence_true_ratio >= 0.6)
        relaxed_branch_terminal_threshold = max(float(self.terminal_threshold_v2), 0.55)
        true_failure_v2_strict = bool(convergence_true_failure and terminal_hard_failure)
        true_failure_v2 = bool(
            terminal_hard_failure
            or (convergence_true_failure and terminal_risk_score >= relaxed_branch_terminal_threshold)
        )

        return {
            "samples": samples,
            "step_scores": step_scores,
            "total_membership": float(max_score),
            "max_total_membership": float(max_score),
            "mean_total_membership": float(mean_score),
            "score_uncertainty": score_uncertainty,
            "system_failure": bool(system_failure),
            "true_failure": bool(true_failure),
            "total_membership_v2": float(max_score_v2),
            "max_total_membership_v2": float(max_score_v2),
            "mean_total_membership_v2": float(mean_score_v2),
            "score_uncertainty_v2": score_uncertainty_v2,
            "decision_score_v2": decision_score_v2,
            "decision_score_v2_linear": decision_score_v2_linear,
            "decision_feature_contributions": decision_feature_contributions,
            "convergence_window_start_step": convergence_start_step,
            "convergence_window_start_index": int(convergence_start_idx),
            "converged_mean_v2": converged_mean_v2,
            "converged_p75_v2": converged_p75_v2,
            "converged_std_v2": converged_std_v2,
            "converged_slope_v2": slope,
            "converged_max_v2": converged_max_v2,
            "converged_high_ratio_v2": converged_high_ratio_v2,
            "terminal_score_gap_v2": terminal_score_gap_v2,
            "decision_score_formula_version": decision_score_formula_version,
            "decision_formula_weights": dict(self.decision_formula_weights),
            "decision_model_type": self.decision_model_type,
            "decision_model_weights": dict(self.decision_model_weights),
            "decision_model_bias": float(self.decision_model_bias),
            "enable_decision_tail_boost": bool(self.enable_decision_tail_boost),
            "decision_tail_gamma": float(self.decision_tail_gamma),
            "terminal_hard_failure": bool(terminal_hard_failure),
            "terminal_risk_score": float(terminal_risk_score),
            "terminal_risk_weights": self.get_terminal_risk_weights(),
            "convergence_true_ratio_v2": convergence_true_ratio,
            "predicted_failure_v2": bool(predicted_failure_v2),
            "true_failure_v2_strict": bool(true_failure_v2_strict),
            "true_failure_v2": bool(true_failure_v2),
            "v2_failure_threshold": float(self.v2_failure_threshold),
            "decision_threshold_v2": float(self.v2_failure_threshold),
            "terminal_threshold_v2": float(self.terminal_threshold_v2),
        }


def build_dummy_fahp_matrix(num_criteria: int = len(FAILURE_METRIC_ORDER)) -> np.ndarray:
    matrix = np.ones((num_criteria, num_criteria, 3), dtype=float)
    for i in range(num_criteria):
        matrix[i, i] = (1.0, 1.0, 1.0)
        for j in range(i + 1, num_criteria):
            matrix[i, j] = (1.0, 1.0, 1.0)
            matrix[j, i] = (1.0, 1.0, 1.0)
    return matrix


def build_default_cloud_rules(
    references: Dict[str, float] = FAILURE_METRIC_REFERENCES,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    cloud_rules: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for metric_name in FAILURE_METRIC_ORDER:
        reference = float(references[metric_name])

        if metric_name in HIGHER_IS_WORSE:
            if metric_name == "PacketLossRate":
                ex = 0.25
            else:
                ex = max(reference * 1.8, reference + 0.1)
            en = max(abs(ex - reference) / 3.0, 1e-3)
            he = max(en / 10.0, 1e-4)
        elif metric_name in LOWER_IS_WORSE:
            if metric_name == "BandwidthUtilization":
                ex = max(reference * 0.35, 1e-5)
            elif metric_name == "AverageComputingRatio":
                ex = max(reference * 0.4, 1e-4)
            else:
                ex = max(reference * 0.45, 1e-4)
            en = max(abs(reference - ex) / 3.0, 1e-3)
            he = max(en / 10.0, 1e-4)
        else:
            ex = reference
            en = max(abs(reference) / 5.0, 1e-3)
            he = max(en / 10.0, 1e-4)

        cloud_rules[metric_name] = {"Failure": (float(ex), float(en), float(he))}

    return cloud_rules


def build_default_failure_evaluator(
    v2_failure_threshold: float = 0.35,
    terminal_threshold_v2: float = 0.55,
    terminal_risk_weights: Optional[Dict[str, float]] = None,
    decision_formula_weights: Optional[Dict[str, float]] = None,
    enable_decision_tail_boost: bool = False,
    decision_tail_gamma: float = 1.0,
    decision_model_type: str = "fixed_linear",
    decision_model_weights: Optional[Dict[str, float]] = None,
    decision_model_bias: float = 0.0,
) -> AgentFailureEvaluator:
    dummy_matrix = build_dummy_fahp_matrix()
    weights = FAHP(dummy_matrix).calculate_weights()
    cloud_rules = build_default_cloud_rules()
    return AgentFailureEvaluator(
        weights,
        cloud_rules,
        v2_failure_threshold=v2_failure_threshold,
        terminal_threshold_v2=terminal_threshold_v2,
        terminal_risk_weights=terminal_risk_weights,
        decision_formula_weights=decision_formula_weights,
        enable_decision_tail_boost=enable_decision_tail_boost,
        decision_tail_gamma=decision_tail_gamma,
        decision_model_type=decision_model_type,
        decision_model_weights=decision_model_weights,
        decision_model_bias=decision_model_bias,
    )


if __name__ == "__main__":
    evaluator = build_default_failure_evaluator()
    print("Weights:", evaluator.weights)
    print("Cloud rules keys:", list(evaluator.cloud_configs.keys()))
