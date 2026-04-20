import os
import sys
import torch
import numpy as np

# Add project root to PYTHONPATH for local package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_and_attribution_analysis.parameter_interfaces import AttackModuleInput
from failure_and_attribution_analysis.agent_failure_evaluator import AgentFailureEvaluator, FAHP
from failure_and_attribution_analysis.failure_boundary_explorer import FailureBoundaryExplorer
from failure_and_attribution_analysis.scenario_parameter_generator import ScenarioParameterGenerator, FeatureSimilarityNetwork
from failure_and_attribution_analysis.deep_ensemble_network import DeepEnsembleNetwork, EmbeddedMLP, DeepNarrowMLP

LOW_BASE_ENV = {
    "PoissonRate": 45.0,
    "MeanIntervalTime": 15.0,
    "PacketGenerationInterval": 4.0,
}
TRAFFIC_PROFILE = "low"

FEATURE_BOUNDS = [
    (0.0, 1.0),          # DegradedEdgeRatio
    (0.0, 1.0),          # EdgeBandwidthMeanDecreaseRatio
    (0.0, 0.2),          # EdgeBandwidthDecreaseStd
    (0.01, 60.0),        # MeanIntervalTime
    (2.0e8, 8.0e8),      # PacketSizeMean (bit)
]


def test_attack_data_flow():
    # 1) Load logs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "output", "low_batchsize=128")
    log_file1 = os.path.join(log_dir, "low_0_nonattack.txt")
    log_file2 = os.path.join(log_dir, "low_0_ModelTampAttack_4.txt")

    if not os.path.exists(log_file1) or not os.path.exists(log_file2):
        raise FileNotFoundError(f"Log files not found: {log_file1}, {log_file2}")

    print(">>> Loading attack-module logs...")
    inputs1 = AttackModuleInput.parse_from_log_file(log_file1)
    inputs2 = AttackModuleInput.parse_from_log_file(log_file2)
    all_inputs = inputs1 + inputs2
    print(f"Parsed {len(all_inputs)} time-step snapshots.")

    r_base = np.mean([ipt.Metrics.AverageEndingReward for ipt in inputs1])
    r_threshold = r_base - abs(r_base) * 0.20
    print(f"Nonattack baseline reward={r_base:.3f}, failure threshold={r_threshold:.3f}")

    # 2) Failure evaluation
    print("\n>>> Running Agent Failure Evaluator...")
    dummy_matrix = np.array([
        [[1, 1, 1], [1, 2, 3], [2, 3, 4], [1, 1, 2], [1, 2, 3]],
        [[1/3, 1/2, 1], [1, 1, 1], [1, 2, 3], [1/2, 1, 2], [1, 1, 1]],
        [[1/4, 1/3, 1/2], [1/3, 1/2, 1], [1, 1, 1], [1/3, 1/2, 1], [1/2, 1, 2]],
        [[1/2, 1, 1], [1/2, 1, 2], [1, 2, 3], [1, 1, 1], [1, 2, 3]],
        [[1/3, 1/2, 1], [1, 1, 1], [1/2, 1, 2], [1/3, 1/2, 1], [1, 1, 1]],
    ])
    weights = FAHP(dummy_matrix).calculate_weights()
    cloud_rules = {
        "PacketLossRate": {"Failure": (0.8, 0.1, 0.01)},
        "AverageE2eDelay": {"Failure": (5.0, 2.0, 0.5)},
        "NetworkThroughput": {"Failure": (1000.0, 500.0, 100.0)},
        "ComputingWaitingTime": {"Failure": (5.0, 2.0, 0.5)},
        "BandwidthUtilization": {"Failure": (0.05, 0.02, 0.005)},
    }
    evaluator = AgentFailureEvaluator(weights, cloud_rules)

    continuous_features = []
    discrete_features = []
    failure_scores = []

    for ipt in all_inputs:
        env = ipt.FinalEnv[0]
        c_feat = [
            env.DegradedEdgeRatio,
            env.EdgeBandwidthMeanDecreaseRatio,
            env.EdgeBandwidthDecreaseStd,
            env.MeanIntervalTime,
            env.PacketSizeMean,
        ]
        d_feat = [env.ActionAttack_level]

        noise_std = np.array([0.02, 0.02, 0.01, 1.0, 5.0e7], dtype=float)
        c_feat = np.array(c_feat, dtype=float) + np.random.normal(0.0, noise_std, size=len(c_feat))
        c_feat = np.clip(
            c_feat,
            np.array([b[0] for b in FEATURE_BOUNDS], dtype=float),
            np.array([b[1] for b in FEATURE_BOUNDS], dtype=float),
        )

        score = evaluator.evaluate_performance(ipt.Metrics.__dict__, "Failure")
        continuous_features.append(c_feat)
        discrete_features.append(d_feat)
        failure_scores.append(score)

    num_con = 5
    num_cat = 5

    # Reward-based labels
    true_labels, pred_labels = [], []
    for i, ipt in enumerate(all_inputs):
        true_labels.append(1 if ipt.Metrics.AverageEndingReward < r_threshold else 0)
        pred_labels.append(1 if failure_scores[i] >= 0.7 else 0)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    tn = np.sum((true_labels == 0) & (pred_labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_labels)

    print("\n================ Failure Detection Report ================")
    print(f"pred_failures: {np.sum(pred_labels)} | true_failures: {np.sum(true_labels)}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1_score * 100:.2f}%")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print("=========================================================")

    c_tensor = torch.tensor(np.array(continuous_features), dtype=torch.float32)
    d_tensor = torch.tensor(np.array(discrete_features), dtype=torch.long)
    score_tensor = torch.tensor(np.array(failure_scores), dtype=torch.float32)

    # Mock uncertainty values for demo flow
    cv_values = []
    for s in failure_scores:
        if 0.3 <= s <= 0.7:
            cv = 0.15 + (0.2 - abs(s - 0.5) * 0.4) + np.random.normal(0, 0.02)
        else:
            cv = np.random.uniform(0.01, 0.08)
        cv_values.append(cv)
    cv_tensor = torch.tensor(np.array(cv_values), dtype=torch.float32)

    high_risk_count = sum(1 for s in failure_scores if s >= 0.7)
    boundary_count = sum(1 for s in failure_scores if 0.3 <= s < 0.7)
    print("Failure-score summary:")
    print(f" - safe samples (score < 0.3): {len(failure_scores) - high_risk_count - boundary_count}")
    print(f" - boundary samples (0.3 <= score < 0.7): {boundary_count}")
    print(f" - high-risk samples (score >= 0.7): {high_risk_count}")

    # 3) Boundary exploration
    print("\n>>> Running Failure Boundary Explorer...")
    explorer = FailureBoundaryExplorer(n_clusters=min(5, len(c_tensor) // 10))
    region_stats = explorer.partition_and_evaluate(
        features=c_tensor.numpy(),
        failure_scores=score_tensor.numpy(),
        cv_values=cv_tensor.numpy(),
        theoretical_max_per_region=100,
    )

    coverage_metrics = explorer.compute_coverage_metrics(
        region_stats=region_stats,
        confidence=0.95,
        target_coverage=0.90,
    )
    print("\n================ Coverage Metrics ================")
    print(f"total_samples:            {coverage_metrics['total_samples']}")
    print(f"covered_samples:          {coverage_metrics['covered_samples']}")
    print(f"coverage_point_estimate:  {coverage_metrics['coverage_point_estimate'] * 100:.2f}%")
    print(f"coverage_lower_bound@95%: {coverage_metrics['coverage_lower_bound'] * 100:.2f}%")
    print(f"target_coverage:          {coverage_metrics['target_coverage'] * 100:.2f}%")
    print(f"target_achieved:          {coverage_metrics['target_achieved']}")
    print("=================================================")

    seeds_c = explorer.generate_seed_candidates(region_stats, FEATURE_BOUNDS, num_seeds_per_region=10)
    seeds_d = np.random.randint(0, num_cat, size=(len(seeds_c), 1))

    print(f"Explorer produced {len(seeds_c)} continuous seeds.")

    if len(seeds_c) == 0:
        print("[Fallback] No seeds from explorer, using top-10 historical high-risk samples.")
        top_k = min(10, len(continuous_features))
        top_idx = np.argsort(failure_scores)[-top_k:]
        seeds_c = np.array(continuous_features)[top_idx]
        seeds_d = np.array(discrete_features)[top_idx]

    # 4) Scenario generation
    print("\n>>> Running Scenario Parameter Generator...")
    net1 = EmbeddedMLP(num_continuous=num_con, num_categories=num_cat)
    net2 = DeepNarrowMLP(num_continuous=num_con)
    ensemble = DeepEnsembleNetwork(models=[net1, net2], model_weights=[0.6, 0.4])
    feature_net = FeatureSimilarityNetwork(num_continuous=num_con, num_categories=num_cat)

    generator = ScenarioParameterGenerator(ensemble_net=ensemble, feature_net=feature_net)
    generator.add_explored_history(c_tensor, d_tensor)

    _final_c, _final_d = generator.generate_new_scenarios(
        seed_continuous=seeds_c,
        seed_discrete=seeds_d,
        num_categories=num_cat,
        target_num_scenarios=20,
    )

    print("\n>>> Building FailEnv list via generate_fail_env_list()...")
    fresh_generator = ScenarioParameterGenerator(
        ensemble_net=ensemble,
        feature_net=feature_net,
        similarity_threshold=0.99,
    )
    fail_env_list = fresh_generator.generate_fail_env_list(
        seed_continuous=seeds_c,
        seed_discrete=seeds_d,
        num_categories=num_cat,
        target_num_scenarios=20,
        base_env=LOW_BASE_ENV,
        traffic_profile=TRAFFIC_PROFILE,
        cv_threshold=0.005,
    )

    print("\n================= FailEnv Output =================")
    print(f"Generated {len(fail_env_list)} FailEnv records.")
    for i, fe in enumerate(fail_env_list):
        print(f"\n[Scenario {i + 1}]")
        print(f"ConstellationConfig:{fe.ConstellationConfig}")
        print(f"DegradedEdgeRatio:{fe.DegradedEdgeRatio:g}")
        print(f"EdgeDisconnectRatio:{fe.EdgeDisconnectRatio:g}")
        print(f"EdgeBandwidthMeanDecreaseRatio:{fe.EdgeBandwidthMeanDecreaseRatio:g}")
        print(f"EdgeBandwidthDecreaseStd:{fe.EdgeBandwidthDecreaseStd:g}")
        print(f"TrafficProfile:{TRAFFIC_PROFILE}")
        print(f"PacketSizeMean:{fe.PacketSizeMean}")
        print(f"PacketSizeStd:{fe.PacketSizeStd}")
        print(f"StateObservationAttack_level:{fe.StateObservationAttack_level}")
        print(f"ActionAttack_level:{fe.ActionAttack_level}")
        print(f"RewardAttack_level:{fe.RewardAttack_level}")
        print(f"StateTransferAttack_level:{fe.StateTransferAttack_level}")
        print(f"ExperiencePoolAttack_level:{fe.ExperiencePoolAttack_level}")
        print(f"ModelTampAttack_level:{fe.ModelTampAttack_level}")


if __name__ == "__main__":
    test_attack_data_flow()
