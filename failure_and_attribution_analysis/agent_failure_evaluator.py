import numpy as np
from typing import Dict, List, Tuple

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


class FAHP:
    """
    Fuzzy Analytic Hierarchy Process for generating criterion weights.
    """

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
    """
    Aggregates cloud-model memberships across the configured metrics.
    """

    def __init__(
        self,
        criteria_weights: np.ndarray,
        cloud_configs: Dict[str, Dict[str, Tuple[float, float, float]]],
    ):
        self.weights = np.array(criteria_weights, dtype=float)
        self.cloud_configs = cloud_configs

    def evaluate_performance(self, metrics_data: Dict[str, float], target_level: str = "Failure") -> float:
        metrics_keys = list(self.cloud_configs.keys())
        if len(metrics_keys) != len(self.weights):
            raise ValueError("Metric count does not match criteria weight count")

        total_membership = 0.0
        for idx, key in enumerate(metrics_keys):
            if key not in metrics_data:
                continue
            if target_level not in self.cloud_configs[key]:
                continue

            ex, en, he = self.cloud_configs[key][target_level]
            cloud = CloudModel(ex, en, he)
            membership = cloud.get_membership(float(metrics_data[key]))
            total_membership += self.weights[idx] * membership

        return float(total_membership)

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
            score = self.evaluate_performance(sample.Metrics.__dict__, target_level=target_level)
            step_scores.append(
                {
                    "step_index": sample.StepIndex,
                    "total_membership": float(score),
                    "metrics": sample.Metrics.__dict__,
                }
            )

        max_score = max((item["total_membership"] for item in step_scores), default=0.0)
        mean_score = float(np.mean([item["total_membership"] for item in step_scores])) if step_scores else 0.0
        score_uncertainty = float(np.std([item["total_membership"] for item in step_scores])) if step_scores else 0.0
        system_failure = max_score > failure_threshold
        true_failure = any(
            sample.Metrics.AverageEndingReward < real_failure_reward_threshold
            for sample in samples
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
        }


def build_dummy_fahp_matrix(num_criteria: int = len(FAILURE_METRIC_ORDER)) -> np.ndarray:
    """
    Initialize an NxNx3 reciprocal fuzzy matrix.
    Default off-diagonal values are neutral, which yields near-equal weights.
    """

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
    """
    Expand cloud rules to the full 11 metrics with Failure=(Ex, En, He).
    """

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


def build_default_failure_evaluator() -> AgentFailureEvaluator:
    dummy_matrix = build_dummy_fahp_matrix()
    weights = FAHP(dummy_matrix).calculate_weights()
    cloud_rules = build_default_cloud_rules()
    return AgentFailureEvaluator(weights, cloud_rules)


if __name__ == "__main__":
    evaluator = build_default_failure_evaluator()
    print("Weights:", evaluator.weights)
    print("Cloud rules keys:", list(evaluator.cloud_configs.keys()))
