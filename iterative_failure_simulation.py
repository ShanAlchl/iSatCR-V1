import argparse
import copy
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_batch_experiments import is_attack_combination_valid, parse_experiment_md

from failure_and_attribution_analysis.agent_failure_evaluator import build_default_failure_evaluator
from failure_and_attribution_analysis.deep_ensemble_network import build_default_deep_ensemble
from failure_and_attribution_analysis.failure_boundary_explorer import FailureBoundaryExplorer
from failure_and_attribution_analysis.parameter_interfaces import (
    CONTINUOUS_FEATURE_NAMES,
    DISCRETE_FEATURE_NAMES,
    FailEnv,
    SCENARIO_PARAMETER_NAMES,
)
from failure_and_attribution_analysis.scenario_parameter_generator import (
    build_continuous_feature_bounds,
    FeatureSimilarityNetwork,
    ScenarioParameterGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the closed-loop failure-analysis simulation workflow.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "train" / "train_NewDDQN_dueling_shuffle.yaml"))
    parser.add_argument("--env-md", default=str(PROJECT_ROOT / "env_config.md"))
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "failure_and_attribution_analysis" / "closed_loop_outputs"),
    )
    parser.add_argument("--generated-limit", type=int, default=100)
    parser.add_argument("--scenarios-per-round", type=int, default=16)
    parser.add_argument("--seed-per-region", type=int, default=48)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.97)
    parser.add_argument("--generation-cv-threshold", type=float, default=0.03)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument("--reset-state", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(path: Path, config: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def normalize_traffic_profile(config: Dict):
    env_cfg = config.setdefault("environment", {})
    traffic_profile = env_cfg.get("TrafficProfile")
    traffic_profiles = env_cfg.get("TrafficProfiles", {})
    if traffic_profile is None:
        return
    normalized_profile = str(traffic_profile).strip().lower()
    if normalized_profile in traffic_profiles:
        env_cfg["TrafficProfile"] = normalized_profile
        env_cfg.update(traffic_profiles[normalized_profile])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def scenario_to_mapping(scenario: Dict) -> Dict:
    normalized = {}
    for key in SCENARIO_PARAMETER_NAMES:
        if key not in scenario:
            continue
        value = scenario[key]
        if key in {
            "ConstellationConfig",
            "PacketSizeMean",
            "PacketSizeStd",
            "StateObservationAttack_level",
            "ActionAttack_level",
            "StateTransferAttack_level",
            "RewardAttack_level",
            "ExperiencePoolAttack_level",
            "ModelTampAttack_level",
        }:
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def fail_env_to_mapping(fail_env: FailEnv) -> Dict:
    return scenario_to_mapping(fail_env.__dict__)


def resolve_fixed_constellation_config(base_config: Dict, env_md_path: Path) -> int:
    param_space = parse_experiment_md(str(env_md_path))
    md_values = param_space.get("ConstellationConfig")
    if md_values is None:
        base_value = base_config.get("environment", {}).get("ConstellationConfig")
        if base_value is None:
            raise ValueError("ConstellationConfig must be defined in env_config.md or the base yaml config.")
        return int(base_value)

    normalized_values = sorted({int(value) for value in md_values})
    if len(normalized_values) != 1:
        raise ValueError(
            "env_config.md 中的 ConstellationConfig 必须固定为单个值；"
            "如需更换星座构型，请直接修改 env_config.md 中该值。"
        )
    return int(normalized_values[0])


def build_initial_scenarios(base_config: Dict, env_md_path: Path) -> List[Dict]:
    param_space = parse_experiment_md(str(env_md_path))
    keys = list(param_space.keys())
    value_lists = [param_space[k] for k in keys]
    template_env = copy.deepcopy(base_config.get("environment", {}))

    scenarios: List[Dict] = []
    combo_iter = itertools.product(*value_lists) if value_lists else [tuple()]
    for combo_values in combo_iter:
        combo = {k: v for k, v in zip(keys, combo_values)}
        if not is_attack_combination_valid(combo, template_env):
            continue

        scenario = {
            key: template_env.get(key)
            for key in SCENARIO_PARAMETER_NAMES
            if key in template_env
        }
        scenario.update(combo)
        traffic_profile = scenario.get("TrafficProfile")
        traffic_profiles = template_env.get("TrafficProfiles", {})
        if traffic_profile is not None:
            normalized_profile = str(traffic_profile).strip().lower()
            if normalized_profile in traffic_profiles:
                scenario.update(traffic_profiles[normalized_profile])
        scenarios.append(scenario_to_mapping(scenario))

    if not scenarios:
        scenarios.append(
            scenario_to_mapping(
                {key: template_env.get(key) for key in SCENARIO_PARAMETER_NAMES if key in template_env}
            )
        )
    return scenarios


def serialize_performance_file(
    raw_log_path: Path,
    output_path: Path,
    scenario: Dict,
    round_index: int,
    test_id: int,
):
    raw_log_text = raw_log_path.read_text(encoding="utf-8", errors="ignore")
    payload = [
        f"ROUND_INDEX: {round_index}",
        f"TEST_ID: {test_id}",
        f"SCENARIO_JSON: {json.dumps(scenario, ensure_ascii=False, sort_keys=True)}",
        f"RAW_LOG_NAME: {raw_log_path.name}",
        "",
        raw_log_text,
    ]
    output_path.write_text("\n".join(payload), encoding="utf-8")


def scenario_feature_arrays(scenario: Dict) -> Tuple[List[float], List[int]]:
    continuous_values = [float(scenario[name]) for name in CONTINUOUS_FEATURE_NAMES]
    discrete_values = [int(scenario[name]) for name in DISCRETE_FEATURE_NAMES]
    return continuous_values, discrete_values
class ClosedLoopFailureSimulation:
    def __init__(self, args):
        self.args = args
        self.project_root = PROJECT_ROOT
        self.config_path = Path(args.config).resolve()
        self.env_md_path = Path(args.env_md).resolve()

        self.output_root = ensure_dir(Path(args.output_root).resolve())
        self.session_dir = self.output_root / "current_session"
        if args.reset_state and self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
        ensure_dir(self.session_dir)
        self.rounds_dir = ensure_dir(self.session_dir / "rounds")
        self.temp_dir = ensure_dir(self.session_dir / "temp_configs")
        self.checkpoint_path = self.session_dir / "closed_loop_state.pt"

        self.base_config = load_yaml(self.config_path)
        normalize_traffic_profile(self.base_config)
        self.fixed_constellation_config = resolve_fixed_constellation_config(self.base_config, self.env_md_path)
        self.traffic_profile = str(self.base_config.get("environment", {}).get("TrafficProfile", "low")).strip().lower()
        random_seed = int(self.base_config.get("general", {}).get("random_seed", 42))
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = build_default_failure_evaluator()
        self.ensemble = build_default_deep_ensemble(
            num_continuous=len(CONTINUOUS_FEATURE_NAMES),
            num_categories=5,
            num_discrete_features=len(DISCRETE_FEATURE_NAMES),
        ).to(self.device)
        self.feature_net = FeatureSimilarityNetwork(
            num_continuous=len(CONTINUOUS_FEATURE_NAMES),
            num_categories=5,
            num_discrete_features=len(DISCRETE_FEATURE_NAMES),
        ).to(self.device)
        self.generator = ScenarioParameterGenerator(
            ensemble_net=self.ensemble,
            feature_net=self.feature_net,
            similarity_threshold=self.args.similarity_threshold,
            continuous_feature_names=CONTINUOUS_FEATURE_NAMES,
            discrete_feature_names=DISCRETE_FEATURE_NAMES,
            fixed_constellation_config=self.fixed_constellation_config,
        )
        self.explorer = FailureBoundaryExplorer(
            n_clusters=max(1, int(self.args.n_clusters)),
            rau_threshold=0.1,
            sc_threshold=0.7,
        )

        self.round_index = 0
        self.next_round_scenarios: List[Dict] = build_initial_scenarios(self.base_config, self.env_md_path)
        self.test_counter = 0
        self.generated_scenario_count = 0
        self.highest_similarity = 0.0
        self.latest_coverage_metrics: Dict = {}
        self.finished = False

        self.cumulative_continuous_features: List[List[float]] = []
        self.cumulative_discrete_features: List[List[int]] = []
        self.cumulative_failure_scores: List[float] = []
        self.summary_records: List[Dict] = []
        self.step_records: List[Dict] = []
        self.round_evalu_files: List[str] = []

        if self.checkpoint_path.exists() and not self.args.reset_state:
            self._restore_state()

    def _restore_state(self):
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.ensemble.load_state_dict(state["ensemble_state_dict"])
        self.feature_net.load_state_dict(state["feature_net_state_dict"])
        self.generator.load_state(state.get("generator_state"))
        self.round_index = int(state.get("round_index", 0))
        self.next_round_scenarios = state.get("next_round_scenarios", self.next_round_scenarios)
        self.test_counter = int(state.get("test_counter", 0))
        self.generated_scenario_count = int(state.get("generated_scenario_count", 0))
        self.highest_similarity = float(state.get("highest_similarity", 0.0))
        self.latest_coverage_metrics = dict(state.get("latest_coverage_metrics", {}))
        self.finished = bool(state.get("finished", False))
        self.cumulative_continuous_features = list(state.get("cumulative_continuous_features", []))
        self.cumulative_discrete_features = list(state.get("cumulative_discrete_features", []))
        self.cumulative_failure_scores = list(state.get("cumulative_failure_scores", []))
        self.summary_records = list(state.get("summary_records", []))
        self.step_records = list(state.get("step_records", []))
        self.round_evalu_files = list(state.get("round_evalu_files", []))

    def _save_state(self):
        state = {
            "ensemble_state_dict": self.ensemble.state_dict(),
            "feature_net_state_dict": self.feature_net.state_dict(),
            "generator_state": self.generator.export_state(),
            "round_index": self.round_index,
            "next_round_scenarios": self.next_round_scenarios,
            "test_counter": self.test_counter,
            "generated_scenario_count": self.generated_scenario_count,
            "highest_similarity": self.highest_similarity,
            "latest_coverage_metrics": self.latest_coverage_metrics,
            "finished": self.finished,
            "cumulative_continuous_features": self.cumulative_continuous_features,
            "cumulative_discrete_features": self.cumulative_discrete_features,
            "cumulative_failure_scores": self.cumulative_failure_scores,
            "summary_records": self.summary_records,
            "step_records": self.step_records,
            "round_evalu_files": self.round_evalu_files,
        }
        torch.save(state, self.checkpoint_path)

    def _build_temp_config(self, scenario: Dict, round_index: int, test_id: int) -> Tuple[Path, str]:
        config = copy.deepcopy(self.base_config)
        env_cfg = config.setdefault("environment", {})
        agent_cfg = config.setdefault("agent", {})
        general_cfg = config.setdefault("general", {})

        env_cfg.update(scenario)
        env_cfg["SaveTrainingData"] = f"closed_loop_r{round_index:03d}_t{test_id:04d}.txt"
        env_cfg["SaveActionLog"] = False
        env_cfg["visualize"] = False
        env_cfg["PrintInfo"] = False
        env_cfg["ShowDetail"] = False
        env_cfg["SaveLog"] = False

        general_cfg["phase"] = "train"
        agent_cfg["agent_sharing_mode"] = "independent"
        agent_cfg["reset_independent_on_train_start"] = True
        agent_cfg["cleanup_independent_after_run"] = True
        agent_cfg["strict_bootstrap_in_train"] = True

        temp_config_path = self.temp_dir / f"round_{round_index:03d}_test_{test_id:04d}.yaml"
        dump_yaml(temp_config_path, config)
        return temp_config_path, env_cfg["SaveTrainingData"]

    def _run_single_simulation(self, temp_config_path: Path):
        completed = subprocess.run(
            [sys.executable, "PRC.py", "--config", str(temp_config_path)],
            cwd=str(self.project_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"PRC.py failed with exit code {completed.returncode} for {temp_config_path.name}")

    def _write_round_env_list(self, round_dir: Path, scenarios: Sequence[Dict], similarities: Optional[Sequence[float]] = None):
        env_list_path = round_dir / "env_list.jsonl"
        with env_list_path.open("w", encoding="utf-8") as f:
            for idx, scenario in enumerate(scenarios, start=1):
                payload = {
                    "round_index": self.round_index,
                    "scenario_index": idx,
                    "traffic_profile": self.traffic_profile,
                    "max_similarity_to_history": float(similarities[idx - 1]) if similarities is not None and idx - 1 < len(similarities) else None,
                    "scenario": scenario,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _collect_round_results(self, round_index: int, scenarios: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
        round_dir = ensure_dir(self.rounds_dir / f"round_{round_index:03d}")
        performance_dir = ensure_dir(round_dir / "performance")
        failure_score_path = round_dir / "failure_scores.jsonl"

        round_summary_records: List[Dict] = []
        round_step_records: List[Dict] = []
        self._write_round_env_list(round_dir, scenarios)

        with failure_score_path.open("w", encoding="utf-8") as score_file:
            for scenario in scenarios:
                self.test_counter += 1
                temp_config_path, raw_log_name = self._build_temp_config(scenario, round_index, self.test_counter)
                raw_log_path = self.project_root / "training_process_data" / raw_log_name
                performance_file_path = performance_dir / f"test_{self.test_counter:04d}_performance.txt"

                self._run_single_simulation(temp_config_path)
                if not raw_log_path.exists():
                    raise FileNotFoundError(f"Simulation log not found: {raw_log_path}")

                serialize_performance_file(
                    raw_log_path=raw_log_path,
                    output_path=performance_file_path,
                    scenario=scenario,
                    round_index=round_index,
                    test_id=self.test_counter,
                )

                evaluation = self.evaluator.evaluate_log_file(str(performance_file_path))
                summary_record = {
                    "round_index": round_index,
                    "test_id": self.test_counter,
                    "scenario": scenario,
                    "total_membership": evaluation["total_membership"],
                    "max_total_membership": evaluation["max_total_membership"],
                    "mean_total_membership": evaluation["mean_total_membership"],
                    "score_uncertainty": evaluation["score_uncertainty"],
                    "system_failure": evaluation["system_failure"],
                    "true_failure": evaluation["true_failure"],
                    "performance_file": str(performance_file_path),
                }
                round_summary_records.append(summary_record)

                for sample, step_eval in zip(evaluation["samples"], evaluation["step_scores"]):
                    row = {
                        "round_index": round_index,
                        "test_id": self.test_counter,
                        "step_index": sample.StepIndex,
                        **scenario,
                        **sample.Metrics.__dict__,
                        "failure_score": step_eval["total_membership"],
                        "failure_decision": bool(step_eval["total_membership"] > 0.7),
                        "test_failure_score": evaluation["total_membership"],
                        "test_failure_decision": evaluation["system_failure"],
                    }
                    round_step_records.append(row)

                score_file.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

                continuous_values, discrete_values = scenario_feature_arrays(scenario)
                self.cumulative_continuous_features.append(continuous_values)
                self.cumulative_discrete_features.append(discrete_values)
                self.cumulative_failure_scores.append(float(evaluation["total_membership"]))
                self.summary_records.append(summary_record)
                self.step_records.extend(row for row in round_step_records if row["test_id"] == self.test_counter)

                temp_config_path.unlink(missing_ok=True)
                raw_log_path.unlink(missing_ok=True)

        return round_summary_records, round_step_records

    def _incremental_train_and_evaluate_coverage(self, round_index: int, round_summary_records: Sequence[Dict], round_step_records: Sequence[Dict]) -> Path:
        round_dir = ensure_dir(self.rounds_dir / f"round_{round_index:03d}")
        evalu_path = round_dir / "evalu.txt"

        continuous_tensor = torch.tensor(np.array(self.cumulative_continuous_features), dtype=torch.float32)
        discrete_tensor = torch.tensor(np.array(self.cumulative_discrete_features), dtype=torch.long)
        target_tensor = torch.tensor(np.array(self.cumulative_failure_scores), dtype=torch.float32)

        train_stats = self.ensemble.fit_incremental(
            continuous_tensor,
            discrete_tensor,
            target_tensor,
            epochs=10,
            batch_size=min(32, max(1, len(continuous_tensor))),
            learning_rate=5e-4,
            device=self.device,
        )

        round_continuous = torch.tensor(
            np.array([scenario_feature_arrays(record["scenario"])[0] for record in round_summary_records]),
            dtype=torch.float32,
        )
        round_discrete = torch.tensor(
            np.array([scenario_feature_arrays(record["scenario"])[1] for record in round_summary_records]),
            dtype=torch.long,
        )
        if len(round_continuous) > 0:
            self.generator.add_explored_history(round_continuous, round_discrete)

        features_np = continuous_tensor.numpy()
        actual_scores_np = np.array([record["total_membership"] for record in self.summary_records], dtype=float)
        actual_uncertainties_np = np.array([record["score_uncertainty"] for record in self.summary_records], dtype=float)

        explorer_clusters = max(1, min(int(self.args.n_clusters), len(features_np)))
        self.explorer.n_clusters = explorer_clusters
        region_stats = self.explorer.partition_and_evaluate(
            features=features_np,
            failure_scores=actual_scores_np,
            cv_values=actual_uncertainties_np,
            theoretical_max_per_region=max(self.args.scenarios_per_round, len(features_np) // explorer_clusters or 1),
        )
        coverage_metrics = self.explorer.compute_coverage_metrics(
            region_stats=region_stats,
            confidence=self.args.coverage_confidence,
            target_coverage=self.args.coverage_target,
        )
        self.latest_coverage_metrics = coverage_metrics

        with torch.no_grad():
            predicted_scores, predicted_cvs = self.ensemble(
                continuous_tensor.to(self.device),
                discrete_tensor.to(self.device),
            )
        predicted_scores_np = predicted_scores.squeeze(1).detach().cpu().numpy()
        predicted_cvs_np = predicted_cvs.squeeze(1).detach().cpu().numpy()

        self.explorer.update_failure_cloud(
            features=features_np,
            predicted_scores=predicted_scores_np,
            predicted_uncertainties=predicted_cvs_np,
            metadata=[
                {
                    "test_id": record["test_id"],
                    "round_index": record["round_index"],
                }
                for record in self.summary_records
            ],
        )

        with evalu_path.open("w", encoding="utf-8") as f:
            f.write(f"ROUND_INDEX: {round_index}\n")
            f.write(f"TRAIN_STATS_JSON: {json.dumps(train_stats, ensure_ascii=False)}\n")
            f.write(f"COVERAGE_JSON: {json.dumps(coverage_metrics, ensure_ascii=False)}\n")
            for record in round_summary_records:
                f.write(f"TEST_SUMMARY_JSON: {json.dumps(record, ensure_ascii=False)}\n")
            for row in round_step_records:
                f.write(f"STEP_EVAL_JSON: {json.dumps(row, ensure_ascii=False)}\n")

        self.round_evalu_files.append(str(evalu_path))
        return evalu_path

    def _generate_next_round_scenarios(self) -> Tuple[List[Dict], np.ndarray]:
        if not self.cumulative_continuous_features:
            return [], np.array([], dtype=float)

        continuous_np = np.array(self.cumulative_continuous_features, dtype=float)
        discrete_np = np.array(self.cumulative_discrete_features, dtype=int)
        target_tensor = torch.tensor(continuous_np, dtype=torch.float32)
        discrete_tensor = torch.tensor(discrete_np, dtype=torch.long)

        with torch.no_grad():
            predicted_scores, predicted_cvs = self.ensemble(
                target_tensor.to(self.device),
                discrete_tensor.to(self.device),
            )
        predicted_cvs_np = predicted_cvs.squeeze(1).detach().cpu().numpy()
        predicted_scores_np = predicted_scores.squeeze(1).detach().cpu().numpy()

        region_stats = self.explorer.partition_and_evaluate(
            features=continuous_np,
            failure_scores=predicted_scores_np,
            cv_values=predicted_cvs_np,
            theoretical_max_per_region=max(self.args.scenarios_per_round, len(continuous_np) // max(1, self.explorer.n_clusters)),
        )
        seeds_c = self.explorer.generate_seed_candidates(
            region_stats=region_stats,
            feature_bounds=build_continuous_feature_bounds(CONTINUOUS_FEATURE_NAMES, self.traffic_profile),
            num_seeds_per_region=self.args.seed_per_region,
        )

        base_env = {
            key: self.base_config.get("environment", {}).get(key)
            for key in SCENARIO_PARAMETER_NAMES
            if key in self.base_config.get("environment", {})
        }
        generated_envs, similarities = self.generator.generate_fail_env_list(
            seed_continuous=seeds_c,
            seed_discrete=discrete_np,
            num_categories=[5] * len(DISCRETE_FEATURE_NAMES),
            target_num_scenarios=self.args.scenarios_per_round,
            base_env=base_env,
            traffic_profile=self.traffic_profile,
            cv_threshold=self.args.generation_cv_threshold,
            return_similarity=True,
        )

        next_round_scenarios = [fail_env_to_mapping(env) for env in generated_envs]
        return next_round_scenarios, similarities

    def _write_next_round_env_list(self, next_round_index: int, scenarios: Sequence[Dict], similarities: Sequence[float]):
        round_dir = ensure_dir(self.rounds_dir / f"round_{next_round_index:03d}")
        self._write_round_env_list(round_dir, scenarios, similarities=similarities)

    def _write_final_output(self):
        output_path = self.session_dir / "output_summary.txt"
        predicted_failures = sum(1 for record in self.summary_records if record["system_failure"])
        true_failures = sum(1 for record in self.summary_records if record["true_failure"])
        accuracy = (
            sum(1 for record in self.summary_records if record["system_failure"] == record["true_failure"]) / len(self.summary_records)
            if self.summary_records
            else 0.0
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"generated_scenario_count_excluding_initial: {self.generated_scenario_count}\n")
            f.write(f"highest_similarity_to_history: {self.highest_similarity:.6f}\n")
            f.write(
                f"latest_coverage_lower_bound: {float(self.latest_coverage_metrics.get('coverage_lower_bound', 0.0)):.6f}\n"
            )
            f.write(
                f"latest_coverage_upper_bound: {float(self.latest_coverage_metrics.get('coverage_upper_bound', 0.0)):.6f}\n"
            )
            f.write(f"predicted_failure_count: {predicted_failures}\n")
            f.write(f"true_failure_count: {true_failures}\n")
            f.write(f"failure_detection_accuracy: {accuracy:.6f}\n")
            f.write("\n")
            for row in self.step_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self):
        if self.finished:
            self._write_final_output()
            return

        while True:
            current_scenarios = list(self.next_round_scenarios)
            if not current_scenarios:
                break

            round_summary_records, round_step_records = self._collect_round_results(self.round_index, current_scenarios)
            self._incremental_train_and_evaluate_coverage(self.round_index, round_summary_records, round_step_records)

            if self.round_index > 0:
                self.generated_scenario_count += len(current_scenarios)

            if self.generated_scenario_count >= self.args.generated_limit:
                self.finished = True
                self.next_round_scenarios = []
                self._save_state()
                break

            next_round_scenarios, similarities = self._generate_next_round_scenarios()
            self.highest_similarity = max(
                self.highest_similarity,
                float(np.max(similarities)) if similarities is not None and len(similarities) > 0 else 0.0,
            )
            self.round_index += 1
            self.next_round_scenarios = next_round_scenarios
            self._write_next_round_env_list(self.round_index, next_round_scenarios, similarities)
            self._save_state()

            if not next_round_scenarios:
                self.finished = True
                break

        self.finished = True
        self._save_state()
        self._write_final_output()


def main():
    args = parse_args()
    workflow = ClosedLoopFailureSimulation(args)
    workflow.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
