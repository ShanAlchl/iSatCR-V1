import argparse
import copy
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iterative_testing.gpu_runtime import select_torch_device
from iterative_testing.run_batch_experiments import is_attack_combination_valid, parse_experiment_md

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


def parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_decision_formula_weights(value: str) -> Dict[str, float]:
    parts = [piece.strip() for piece in str(value or "").split(",") if piece.strip()]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "decision formula weights must contain 5 comma-separated values: "
            "w_mean,w_p75,w_max,w_slope_pos,w_std_penalty"
        )
    numbers = [float(piece) for piece in parts]
    return {
        "w_mean": numbers[0],
        "w_p75": numbers[1],
        "w_max": numbers[2],
        "w_slope_pos": numbers[3],
        "w_std_penalty": numbers[4],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the closed-loop failure-analysis simulation workflow.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "train" / "train_NewDDQN_dueling_shuffle.yaml"))
    parser.add_argument("--env-md", default=str(PROJECT_ROOT / "env_config.md"))
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "failure_and_attribution_analysis" / "closed_loop_outputs"),
    )
    parser.add_argument("--generated-limit", type=int, default=400)
    parser.add_argument("--scenarios-per-round", type=int, default=16)
    parser.add_argument("--seed-per-region", type=int, default=48)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.97)
    parser.add_argument("--similarity-threshold-max", type=float, default=0.995)
    parser.add_argument("--min-scenarios-per-round", type=int, default=4)
    parser.add_argument("--generation-cv-threshold", type=float, default=0.03)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument(
        "--stop-on-coverage-target",
        type=parse_bool_arg,
        default=True,
        help="true: stop early when coverage lower bound reaches target with enough samples.",
    )
    parser.add_argument(
        "--min-samples-for-coverage-stop",
        type=int,
        default=120,
        help="minimum total samples required before allowing coverage-target early stop.",
    )
    parser.add_argument("--coverage-sc-schedule", type=str, default="12:0.20,40:0.45,1000000:0.70")
    parser.add_argument(
        "--allow-multi-attacks-per-scenario",
        type=parse_bool_arg,
        default=True,
        help="true: allow multiple attack types in one generated scenario; false: keep at most one attack type.",
    )
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold",
        "--failure-threshold-v2",
        dest="decision_threshold",
        type=float,
        default=0.35,
        help="primary decision threshold (legacy alias: --failure-threshold-v2).",
    )
    parser.add_argument(
        "--decision-formula-weights",
        type=str,
        default="0.60,0.25,0.10,0.10,0.20",
        help="weights for decision_score_v2: w_mean,w_p75,w_max,w_slope_pos,w_std_penalty",
    )
    parser.add_argument("--enable-decision-tail-boost", type=parse_bool_arg, default=False)
    parser.add_argument("--decision-tail-gamma", type=float, default=1.0)
    parser.add_argument("--decision-model-type", choices=("fixed_linear", "learned_linear"), default="fixed_linear")
    parser.add_argument(
        "--failure-decision-mode",
        choices=("single_fused_score", "direct_failure_model"),
        default="single_fused_score",
    )
    parser.add_argument("--fit-decision-model-offline", type=parse_bool_arg, default=False)
    parser.add_argument("--decision-model-lr", type=float, default=0.05)
    parser.add_argument("--decision-model-epochs", type=int, default=300)
    parser.add_argument("--decision-model-l2", type=float, default=0.001)
    parser.add_argument(
        "--fused-model-type",
        choices=("mlp_small",),
        default="mlp_small",
        help="model family for single_fused_score; MLP is the only public preset on the current mainline.",
    )
    parser.add_argument("--fused-mlp-hidden-dim", type=int, default=16)
    parser.add_argument("--decision-model-min-support", type=int, default=60)
    parser.add_argument("--decision-model-early-stop-patience", type=int, default=20)
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "recall_at_precision", "accuracy", "balanced_accuracy"),
        default="f1",
    )
    parser.add_argument("--threshold-min-precision", type=float, default=0.5)
    parser.add_argument("--threshold-calibration-scope", choices=("terminal_only",), default="terminal_only")
    parser.add_argument(
        "--threshold-calibration-mode",
        choices=("two_stage_stable",),
        default="two_stage_stable",
        help="two_stage_stable: train shortlist + holdout stability selection.",
    )
    parser.add_argument(
        "--raw-log-root",
        default=str(PROJECT_ROOT / "training_process_data"),
        help="Root directory for raw simulation text logs (prevents collision in parallel runs).",
    )
    parser.add_argument("--threshold-calibration-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-min-support", type=int, default=30)
    parser.add_argument("--threshold-two-stage-top-k", type=int, default=24)
    parser.add_argument("--threshold-two-stage-gap-penalty", type=float, default=0.35)
    parser.add_argument("--threshold-two-stage-gap-tolerance", type=float, default=0.01)
    parser.add_argument("--threshold-two-stage-passrate-drift-penalty", type=float, default=0.10)
    parser.add_argument(
        "--online-backfill-after-each-round",
        type=parse_bool_arg,
        default=False,
        help="true: after each online round recalibration, recompute predictions for all accumulated samples.",
    )
    parser.add_argument(
        "--post-run-offline-recompute",
        type=parse_bool_arg,
        default=True,
        help="true: after online run ends, compute an offline-style full-session recompute summary and append to output_summary.",
    )
    parser.add_argument(
        "--true-failure-policy",
        "--true-failure-v2-policy",
        dest="true_failure_policy",
        choices=("relaxed", "strict"),
        default="strict",
        help="label policy for true_failure used by training/calibration/metrics.",
    )
    parser.add_argument(
        "--offline-recompute-only",
        type=parse_bool_arg,
        default=False,
        help="true: do not run new simulations; recalibrate/re-evaluate from existing session records.",
    )
    parser.add_argument(
        "--offline-use-existing-thresholds",
        type=parse_bool_arg,
        default=False,
        help="true: skip threshold recalibration in offline mode and use current thresholds.",
    )
    parser.add_argument(
        "--offline-source-session",
        type=str,
        default="",
        help="optional source session dir (or output root containing current_session) for offline recompute.",
    )
    parser.add_argument(
        "--offline-decision-threshold",
        "--offline-decision-threshold-v2",
        dest="offline_decision_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--offline-terminal-threshold",
        "--offline-terminal-threshold-v2",
        dest="offline_terminal_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-threshold",
        "--terminal-threshold-v2",
        dest="terminal_threshold",
        type=float,
        default=0.55,
    )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(payload), encoding="utf-8")


def scenario_feature_arrays(scenario: Dict) -> Tuple[List[float], List[int]]:
    continuous_values = [float(scenario[name]) for name in CONTINUOUS_FEATURE_NAMES]
    discrete_values = [int(scenario[name]) for name in DISCRETE_FEATURE_NAMES]
    return continuous_values, discrete_values


def parse_sc_schedule(schedule_text: str) -> List[Tuple[int, float]]:
    parsed: List[Tuple[int, float]] = []
    for segment in str(schedule_text or "").split(","):
        piece = segment.strip()
        if not piece or ":" not in piece:
            continue
        left, right = piece.split(":", 1)
        sample_cap = max(1, int(left.strip()))
        threshold = float(np.clip(float(right.strip()), 0.0, 1.0))
        parsed.append((sample_cap, threshold))
    if not parsed:
        parsed = [(12, 0.2), (40, 0.45), (1000000, 0.7)]
    return sorted(parsed, key=lambda item: item[0])
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
        self.coverage_sc_schedule = parse_sc_schedule(args.coverage_sc_schedule)
        self.true_failure_v2_policy = str(args.true_failure_policy).strip().lower()
        self.decision_formula_weights = parse_decision_formula_weights(args.decision_formula_weights)
        self.enable_decision_tail_boost = bool(args.enable_decision_tail_boost)
        self.decision_tail_gamma = float(np.clip(float(args.decision_tail_gamma), 0.5, 1.0))
        self.decision_model_type = str(args.decision_model_type).strip().lower()
        self.failure_decision_mode = str(args.failure_decision_mode).strip().lower()
        self.decision_policy = self.failure_decision_mode
        self.threshold_update_status = "frozen"

        self.device = select_torch_device("closed-loop failure simulation")
        self.evaluator = build_default_failure_evaluator(
            v2_failure_threshold=float(args.decision_threshold),
            terminal_threshold_v2=float(args.terminal_threshold),
            decision_formula_weights=self.decision_formula_weights,
            enable_decision_tail_boost=self.enable_decision_tail_boost,
            decision_tail_gamma=self.decision_tail_gamma,
            decision_model_type=self.decision_model_type,
            decision_model_weights=self.decision_formula_weights,
            decision_model_bias=0.0,
        )
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
            similarity_threshold_max=self.args.similarity_threshold_max,
            allow_multi_attacks_per_scenario=self.args.allow_multi_attacks_per_scenario,
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
        self.stop_reason = "running"

        self.cumulative_continuous_features: List[List[float]] = []
        self.cumulative_discrete_features: List[List[int]] = []
        self.cumulative_failure_scores: List[float] = []
        self.cumulative_failure_labels_v2: List[float] = []
        self.summary_records: List[Dict] = []
        self.step_records: List[Dict] = []
        self.round_evalu_files: List[str] = []
        self.last_decision_model_info: Dict[str, object] = {
            "decision_model_status": "disabled",
            "decision_model_holdout_record_count": 0,
            "decision_model_holdout_auc": 0.0,
            "decision_model_holdout_accuracy": 0.0,
            "decision_model_config": self.evaluator.get_decision_formula_config(),
        }
        self.last_failure_model_info: Dict[str, object] = {
            "failure_decision_mode": self.failure_decision_mode,
            "single_threshold_used": True,
            "primary_score_name": "",
            "primary_score_holdout_auc": 0.0,
            "fused_model_status": "disabled",
            "fused_model_holdout_record_count": 0,
            "fused_model_holdout_auc": 0.0,
            "fused_model_holdout_accuracy": 0.0,
            "fused_model_type": str(self.args.fused_model_type).strip().lower(),
            "fused_model_weights": {},
            "fused_model_bias": 0.0,
            "fused_model_input_mean": [],
            "fused_model_input_std": [],
            "fused_model_mlp_state": {},
            "fused_model_mlp_hidden_dim": int(self.args.fused_mlp_hidden_dim),
            "fused_threshold": 0.5,
            "direct_model_status": "disabled",
            "direct_model_holdout_record_count": 0,
            "direct_model_holdout_auc": 0.0,
            "direct_model_holdout_accuracy": 0.0,
            "direct_model_weights": {},
            "direct_model_bias": 0.0,
            "final_threshold": 0.5,
        }
        self.last_threshold_stats: Dict[str, object] = {}
        self.post_run_offline_recompute_summary: Optional[Dict[str, object]] = None

        if self.checkpoint_path.exists() and not self.args.reset_state:
            self._restore_state()
            self.evaluator.set_decision_formula_config(decision_model_type=self.decision_model_type)

    def _restore_state(self):
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
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
        self.stop_reason = str(state.get("stop_reason", self.stop_reason))
        if "failure_threshold_v2" in state:
            self.evaluator.set_v2_failure_threshold(float(state["failure_threshold_v2"]))
        if "terminal_threshold_v2" in state:
            self.evaluator.set_terminal_threshold_v2(float(state["terminal_threshold_v2"]))
        if "decision_formula_config" in state:
            cfg = dict(state["decision_formula_config"])
            self.evaluator.set_decision_formula_config(
                decision_formula_weights={
                    key: cfg.get(key)
                    for key in ("w_mean", "w_p75", "w_max", "w_slope_pos", "w_std_penalty")
                    if key in cfg
                },
                enable_decision_tail_boost=cfg.get("enable_decision_tail_boost"),
                decision_tail_gamma=cfg.get("decision_tail_gamma"),
                decision_model_type=cfg.get("decision_model_type"),
                decision_model_weights=cfg.get("decision_model_weights"),
                decision_model_bias=cfg.get("decision_model_bias"),
            )
        self.threshold_update_status = str(state.get("threshold_update_status", self.threshold_update_status))
        self.cumulative_continuous_features = list(state.get("cumulative_continuous_features", []))
        self.cumulative_discrete_features = list(state.get("cumulative_discrete_features", []))
        self.cumulative_failure_scores = list(state.get("cumulative_failure_scores", []))
        self.cumulative_failure_labels_v2 = list(state.get("cumulative_failure_labels_v2", []))
        self.summary_records = list(state.get("summary_records", []))
        self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in self.summary_records]
        if self.summary_records:
            self.cumulative_failure_labels_v2 = [
                float(self._resolve_true_failure_v2_value(record)) for record in self.summary_records
            ]
        self.step_records = list(state.get("step_records", []))
        self.round_evalu_files = list(state.get("round_evalu_files", []))
        self.last_decision_model_info = dict(state.get("last_decision_model_info", self.last_decision_model_info))
        self.last_failure_model_info = dict(state.get("last_failure_model_info", self.last_failure_model_info))
        self.last_threshold_stats = dict(state.get("last_threshold_stats", self.last_threshold_stats))
        if "post_run_offline_recompute_summary" in state:
            cached = state.get("post_run_offline_recompute_summary")
            self.post_run_offline_recompute_summary = dict(cached) if isinstance(cached, dict) else None
        if "terminal_risk_weights" in state:
            self.evaluator.set_terminal_risk_weights(dict(state["terminal_risk_weights"]))

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
            "stop_reason": self.stop_reason,
            "failure_threshold_v2": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold_v2": float(self.evaluator.terminal_threshold_v2),
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "threshold_update_status": self.threshold_update_status,
            "cumulative_continuous_features": self.cumulative_continuous_features,
            "cumulative_discrete_features": self.cumulative_discrete_features,
            "cumulative_failure_scores": self.cumulative_failure_scores,
            "cumulative_failure_labels_v2": self.cumulative_failure_labels_v2,
            "summary_records": self.summary_records,
            "step_records": self.step_records,
            "round_evalu_files": self.round_evalu_files,
            "terminal_risk_weights": self.evaluator.get_terminal_risk_weights(),
            "last_decision_model_info": self.last_decision_model_info,
            "last_failure_model_info": self.last_failure_model_info,
            "last_threshold_stats": self.last_threshold_stats,
            "post_run_offline_recompute_summary": self.post_run_offline_recompute_summary,
        }
        ensure_dir(self.session_dir)
        ensure_dir(self.checkpoint_path.parent)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self.checkpoint_path.write_bytes(buffer.getvalue())

    def _build_temp_config(self, scenario: Dict, round_index: int, test_id: int) -> Tuple[Path, str]:
        config = copy.deepcopy(self.base_config)
        env_cfg = config.setdefault("environment", {})
        agent_cfg = config.setdefault("agent", {})
        general_cfg = config.setdefault("general", {})

        # 核心隔离逻辑：为每个并行测试创建专属临时文件夹，并重定向所有写操作
        worker_task_dir = self.temp_dir / f"round_{round_index:03d}_test_{test_id:04d}"
        worker_task_dir.mkdir(parents=True, exist_ok=True)

        env_cfg.update(scenario)
        env_cfg["SaveTrainingData"] = f"closed_loop_r{round_index:03d}_t{test_id:04d}.txt"
        env_cfg["SaveActionLog"] = False
        env_cfg["visualize"] = False
        env_cfg["PrintInfo"] = False
        env_cfg["ShowDetail"] = False
        env_cfg["SaveLog"] = False
        env_cfg["PositionDataDir"] = str(worker_task_dir / "Position_Data")

        general_cfg["phase"] = "train"
        agent_cfg["agent_sharing_mode"] = "independent"
        agent_cfg["reset_independent_on_train_start"] = True
        agent_cfg["cleanup_independent_after_run"] = True
        agent_cfg["strict_bootstrap_in_train"] = True
        agent_cfg["independent_model_dir"] = str(worker_task_dir / "models")

        temp_config_path = worker_task_dir / "config.yaml"
        dump_yaml(temp_config_path, config)
        return temp_config_path, env_cfg["SaveTrainingData"]

    def _run_single_simulation(self, temp_config_path: Path):
        # 准备环境变量，确保能找到项目根目录下的模块
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        project_root_str = str(self.project_root)
        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root_str
        
        # 核心修复：通过环境变量重定向日志目录，但 CWD 保持在根目录以便读取资源文件
        env["TRAINING_LOG_ROOT"] = str(self.args.raw_log_root)

        completed = subprocess.run(
            [
                sys.executable,
                str(self.project_root / "iterative_testing" / "PRC.py"),
                "--config",
                str(temp_config_path),
            ],
            cwd=project_root_str,  # 回到根目录运行，保证资源文件可见
            env=env,
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

    def _resolve_true_failure_v2_value(self, record: Dict) -> bool:
        if self.true_failure_v2_policy == "strict":
            return bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        return bool(record.get("true_failure_v2", record.get("true_failure_v2_strict", False)))

    def _apply_true_failure_policy_to_record(self, record: Dict) -> Dict:
        if "true_failure_v2_strict" not in record:
            record["true_failure_v2_strict"] = bool(record.get("true_failure_v2", False))
        if "true_failure_v2_relaxed" not in record:
            record["true_failure_v2_relaxed"] = bool(record.get("true_failure_v2", False))
        if self.true_failure_v2_policy == "strict":
            record["true_failure_v2"] = bool(record.get("true_failure_v2_strict", False))
        else:
            record["true_failure_v2"] = bool(record.get("true_failure_v2_relaxed", record.get("true_failure_v2", False)))
        return record

    @staticmethod
    def _roc_auc_binary(scores: Sequence[float], labels: Sequence[bool]) -> float:
        if not scores or not labels or len(scores) != len(labels):
            return 0.0
        scores_np = np.asarray(scores, dtype=float)
        labels_np = np.asarray(labels, dtype=bool)
        pos = scores_np[labels_np]
        neg = scores_np[~labels_np]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Mann-Whitney U based AUC with tie handling.
        combined = np.concatenate([pos, neg])
        order = np.argsort(combined, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
        _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
        for idx, count in enumerate(counts):
            if count <= 1:
                continue
            positions = np.where(inv == idx)[0]
            ranks[positions] = float(np.mean(ranks[positions]))
        pos_rank_sum = float(np.sum(ranks[: len(pos)]))
        u_stat = pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0
        auc = u_stat / (len(pos) * len(neg))
        return float(np.clip(auc, 0.0, 1.0))

    @staticmethod
    def _percentiles(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    def _write_offline_decision_distribution(self, tag: str):
        if not self.summary_records:
            return
        labels = [self._resolve_true_failure_v2_value(record) for record in self.summary_records]
        scores = [float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0))) for record in self.summary_records]
        failure_decision_mode = self.failure_decision_mode
        if failure_decision_mode == "single_fused_score":
            primary_scores = [float(record.get("fused_score", 0.0)) for record in self.summary_records]
            primary_score_name = "fused_score"
            primary_auc_key = "auc_fused_score"
        elif failure_decision_mode == "direct_failure_model":
            primary_scores = [float(record.get("final_failure_probability", 0.0)) for record in self.summary_records]
            primary_score_name = "final_failure_probability"
            primary_auc_key = "auc_final_failure_probability"
        else:
            primary_scores = scores
            primary_score_name = "decision_score_v2"
            primary_auc_key = "auc_decision_score_v2"
        pos_scores = [score for score, label in zip(scores, labels) if label]
        neg_scores = [score for score, label in zip(scores, labels) if not label]
        primary_pos_scores = [score for score, label in zip(primary_scores, labels) if label]
        primary_neg_scores = [score for score, label in zip(primary_scores, labels) if not label]
        overlap_count = sum(1 for score in scores if 0.3 <= score <= 0.7)
        primary_overlap_count = sum(1 for score in primary_scores if 0.3 <= score <= 0.7)
        payload = {
            "timestamp": now_stamp(),
            "tag": tag,
            "failure_decision_mode": failure_decision_mode,
            "true_failure_v2_policy": self.true_failure_v2_policy,
            "record_count": len(scores),
            "positive_count": int(sum(1 for x in labels if x)),
            "negative_count": int(sum(1 for x in labels if not x)),
            "auc_decision_score_v2": self._roc_auc_binary(scores, labels),
            "decision_score_v2_percentiles_positive": self._percentiles(pos_scores),
            "decision_score_v2_percentiles_negative": self._percentiles(neg_scores),
            "decision_overlap_ratio_03_07": float(overlap_count / max(1, len(scores))),
            "primary_score_name": primary_score_name,
            primary_auc_key: self._roc_auc_binary(primary_scores, labels),
            "primary_score_percentiles_positive": self._percentiles(primary_pos_scores),
            "primary_score_percentiles_negative": self._percentiles(primary_neg_scores),
            "primary_score_overlap_ratio_03_07": float(primary_overlap_count / max(1, len(primary_scores))),
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "failure_model_info": self.last_failure_model_info,
        }
        latest_path = self.session_dir / "offline_decision_distribution.json"
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tagged_path = self.session_dir / f"offline_decision_distribution_{tag}.json"
        tagged_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_summary_records_from_round_files(self):
        loaded = self._collect_summary_records_from_rounds_dir(self.rounds_dir)
        self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in loaded]
        self.cumulative_failure_labels_v2 = [float(bool(record.get("true_failure_v2", False))) for record in self.summary_records]

    def _collect_summary_records_from_rounds_dir(self, rounds_dir: Path) -> List[Dict]:
        loaded: List[Dict] = []
        if not rounds_dir.exists():
            return loaded
        for round_dir in sorted(rounds_dir.glob("round_*")):
            file_path = round_dir / "failure_scores.jsonl"
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        loaded.append(json.loads(text))
                    except json.JSONDecodeError:
                        continue
        loaded.sort(key=lambda item: (int(item.get("round_index", 0)), int(item.get("test_id", 0))))
        return loaded

    def _resolve_offline_source_rounds_dir(self) -> Optional[Path]:
        raw = str(self.args.offline_source_session or "").strip()
        if not raw:
            return None
        source = Path(raw)
        if not source.is_absolute():
            source = (self.project_root / source).resolve()
        candidates = [
            source / "rounds",
            source / "current_session" / "rounds",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_decision_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if self.args.threshold_calibration_scope == "terminal_only" and not bool(record.get("terminal_hard_failure", False)):
                continue
            if not all(
                key in record
                for key in ("converged_mean_v2", "converged_p75_v2", "converged_max_v2", "converged_slope_v2")
            ):
                continue
            filtered_records.append(record)
            converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
            terminal_risk_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            features.append(
                [
                    converged_mean_v2,
                    float(record.get("converged_p75_v2", 0.0)),
                    float(record.get("converged_max_v2", 0.0)),
                    max(0.0, float(record.get("converged_slope_v2", 0.0))),
                    float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                    float(record.get("converged_high_ratio_v2", 0.0)),
                    float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2))),
                ]
            )
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _build_fused_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if self.args.threshold_calibration_scope == "terminal_only" and not bool(record.get("terminal_hard_failure", False)):
                continue
            filtered_records.append(record)
            features.append(self._build_fused_feature_vector(record))
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _build_fused_feature_vector(self, record: Dict) -> List[float]:
        terminal_risk_score = float(
            record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
        )
        decision_score_v2 = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
        converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
        converged_std_v2 = float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0)))
        converged_slope_v2 = float(record.get("converged_slope_v2", 0.0))
        terminal_hard_failure_flag = 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0
        terminal_score_gap_v2 = float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2)))
        return [
            decision_score_v2,
            terminal_risk_score,
            terminal_score_gap_v2,
            float(record.get("converged_high_ratio_v2", 0.0)),
            converged_std_v2,
            converged_slope_v2,
            terminal_hard_failure_flag,
            decision_score_v2 * terminal_risk_score,
        ]

    def _build_direct_failure_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if self.args.threshold_calibration_scope == "terminal_only" and not bool(record.get("terminal_hard_failure", False)):
                continue
            if not all(
                key in record
                for key in (
                    "converged_mean_v2",
                    "converged_p75_v2",
                    "converged_max_v2",
                    "converged_slope_v2",
                )
            ):
                continue
            filtered_records.append(record)
            terminal_risk_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
            features.append(
                [
                    converged_mean_v2,
                    float(record.get("converged_p75_v2", 0.0)),
                    float(record.get("converged_max_v2", 0.0)),
                    float(record.get("converged_slope_v2", 0.0)),
                    float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                    float(record.get("converged_high_ratio_v2", 0.0)),
                    terminal_risk_score,
                    float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2))),
                ]
            )
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    @staticmethod
    def _sigmoid_np(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _fit_linear_failure_model(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        feature_names: Sequence[str],
        status_key_prefix: str,
    ) -> Dict[str, object]:
        info: Dict[str, object] = {
            f"{status_key_prefix}_status": "disabled",
            f"{status_key_prefix}_holdout_record_count": 0,
            f"{status_key_prefix}_holdout_auc": 0.0,
            f"{status_key_prefix}_holdout_accuracy": 0.0,
            f"{status_key_prefix}_weights": {},
            f"{status_key_prefix}_bias": 0.0,
            "threshold": 0.5,
        }
        support = int(feature_matrix.shape[0])
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2 or not bool(self.args.fit_decision_model_offline):
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        holdout_ratio = float(np.clip(float(self.args.threshold_calibration_holdout_ratio), 0.0, 0.49))
        holdout_count = int(np.floor(support * holdout_ratio))
        train_count = support - holdout_count
        if train_count < 2:
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        train_x = torch.tensor(feature_matrix[:train_count], dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[:train_count], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(feature_matrix[train_count:], dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[train_count:], dtype=torch.float32, device=self.device)

        raw_weights = torch.nn.Parameter(torch.zeros((feature_matrix.shape[1],), dtype=torch.float32, device=self.device))
        bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam([raw_weights, bias], lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {
            "weights": raw_weights.detach().clone(),
            "bias": bias.detach().clone(),
        }

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            logits = torch.matmul(train_x, raw_weights) + bias
            loss = bce_loss(logits, train_y) + l2_value * torch.sum(raw_weights * raw_weights)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {
                    "weights": raw_weights.detach().clone(),
                    "bias": bias.detach().clone(),
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        best_weights = best_state["weights"].detach().cpu().numpy()
        best_bias = float(best_state["bias"].detach().cpu().item())
        weight_dict = {str(name): float(best_weights[idx]) for idx, name in enumerate(feature_names)}
        if holdout_count > 0:
            holdout_logits = torch.matmul(holdout_x, best_state["weights"]) + best_state["bias"]
            holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
            holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
            holdout_pred = holdout_scores >= 0.5
            holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
            holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
        else:
            holdout_accuracy = 0.0
            holdout_auc = 0.0

        info.update(
            {
                f"{status_key_prefix}_status": "fitted",
                f"{status_key_prefix}_holdout_record_count": int(holdout_count),
                f"{status_key_prefix}_holdout_auc": float(holdout_auc),
                f"{status_key_prefix}_holdout_accuracy": float(holdout_accuracy),
                f"{status_key_prefix}_type": "linear",
                f"{status_key_prefix}_weights": weight_dict,
                f"{status_key_prefix}_bias": float(best_bias),
            }
        )
        return info

    def _fit_mlp_failure_model(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        status_key_prefix: str,
    ) -> Dict[str, object]:
        info: Dict[str, object] = {
            f"{status_key_prefix}_status": "disabled",
            f"{status_key_prefix}_holdout_record_count": 0,
            f"{status_key_prefix}_holdout_auc": 0.0,
            f"{status_key_prefix}_holdout_accuracy": 0.0,
            f"{status_key_prefix}_type": "mlp_small",
            f"{status_key_prefix}_input_mean": [],
            f"{status_key_prefix}_input_std": [],
            f"{status_key_prefix}_mlp_state": {},
            f"{status_key_prefix}_mlp_hidden_dim": int(self.args.fused_mlp_hidden_dim),
            "threshold": 0.5,
        }
        support = int(feature_matrix.shape[0])
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2 or not bool(self.args.fit_decision_model_offline):
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        holdout_ratio = float(np.clip(float(self.args.threshold_calibration_holdout_ratio), 0.0, 0.49))
        holdout_count = int(np.floor(support * holdout_ratio))
        train_count = support - holdout_count
        if train_count < 2:
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        train_np = feature_matrix[:train_count]
        holdout_np = feature_matrix[train_count:]
        train_mean = np.mean(train_np, axis=0)
        train_std = np.std(train_np, axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train_norm = (train_np - train_mean) / train_std
        holdout_norm = (holdout_np - train_mean) / train_std if holdout_count > 0 else holdout_np

        train_x = torch.tensor(train_norm, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[:train_count], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(holdout_norm, dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[train_count:], dtype=torch.float32, device=self.device)

        in_dim = int(feature_matrix.shape[1])
        hidden_dim = max(4, int(self.args.fused_mlp_hidden_dim))
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            logits = model(train_x).squeeze(1)
            l2_penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if l2_value > 0.0:
                for param in model.parameters():
                    l2_penalty = l2_penalty + torch.sum(param * param)
            loss = bce_loss(logits, train_y) + l2_value * l2_penalty
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        if holdout_count > 0:
            with torch.no_grad():
                holdout_logits = model(holdout_x).squeeze(1)
                holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
            holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
            holdout_pred = holdout_scores >= 0.5
            holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
            holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
        else:
            holdout_accuracy = 0.0
            holdout_auc = 0.0

        state_export = {key: value.detach().cpu().numpy().tolist() for key, value in best_state.items()}
        info.update(
            {
                f"{status_key_prefix}_status": "fitted",
                f"{status_key_prefix}_holdout_record_count": int(holdout_count),
                f"{status_key_prefix}_holdout_auc": float(holdout_auc),
                f"{status_key_prefix}_holdout_accuracy": float(holdout_accuracy),
                f"{status_key_prefix}_input_mean": train_mean.astype(np.float32).tolist(),
                f"{status_key_prefix}_input_std": train_std.astype(np.float32).tolist(),
                f"{status_key_prefix}_mlp_state": state_export,
                f"{status_key_prefix}_mlp_hidden_dim": hidden_dim,
            }
        )
        return info

    def _fit_learned_decision_weights(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "decision_model_status": "disabled",
            "decision_model_holdout_record_count": 0,
            "decision_model_holdout_auc": 0.0,
            "decision_model_holdout_accuracy": 0.0,
            "decision_model_config": self.evaluator.get_decision_formula_config(),
        }
        if self.evaluator.decision_model_type != "learned_linear" or not bool(self.args.fit_decision_model_offline):
            self.last_decision_model_info = info
            return info

        feature_matrix, labels, filtered_records = self._build_decision_training_matrix(self.summary_records)
        support = len(filtered_records)
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2:
            info["decision_model_status"] = "frozen"
            self.last_decision_model_info = info
            return info

        holdout_ratio = float(np.clip(float(self.args.threshold_calibration_holdout_ratio), 0.0, 0.49))
        holdout_count = int(np.floor(support * holdout_ratio))
        train_count = support - holdout_count
        if train_count < 2:
            info["decision_model_status"] = "frozen"
            self.last_decision_model_info = info
            return info

        train_x = torch.tensor(feature_matrix[:train_count], dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[:train_count], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(feature_matrix[train_count:], dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[train_count:], dtype=torch.float32, device=self.device)

        init_weights = torch.tensor(
            [
                float(self.evaluator.decision_model_weights.get("w_mean", 0.60)),
                float(self.evaluator.decision_model_weights.get("w_p75", 0.25)),
                float(self.evaluator.decision_model_weights.get("w_max", 0.10)),
                float(self.evaluator.decision_model_weights.get("w_slope_pos", 0.10)),
                float(self.evaluator.decision_model_weights.get("w_std_penalty", 0.20)),
                float(self.evaluator.decision_model_weights.get("w_high_ratio", 0.50)),
                float(self.evaluator.decision_model_weights.get("w_terminal_gap", 0.50)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        raw_weights = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(init_weights, min=1e-5))))
        bias = torch.nn.Parameter(
            torch.tensor(float(self.evaluator.decision_model_bias), dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.Adam([raw_weights, bias], lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {
            "raw_weights": raw_weights.detach().clone(),
            "bias": bias.detach().clone(),
        }

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            positive_weights = torch.nn.functional.softplus(raw_weights)
            logits = (
                train_x[:, 0] * positive_weights[0]
                + train_x[:, 1] * positive_weights[1]
                + train_x[:, 2] * positive_weights[2]
                + train_x[:, 3] * positive_weights[3]
                - train_x[:, 4] * positive_weights[4]
                + train_x[:, 5] * positive_weights[5]
                + train_x[:, 6] * positive_weights[6]
                + bias
            )
            loss = bce_loss(logits, train_y) + l2_value * torch.sum(positive_weights * positive_weights)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {
                    "raw_weights": raw_weights.detach().clone(),
                    "bias": bias.detach().clone(),
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        with torch.no_grad():
            best_weights_tensor = torch.nn.functional.softplus(best_state["raw_weights"]).detach().cpu().numpy()
            best_bias = float(best_state["bias"].detach().cpu().item())
            learned_weights = {
                "w_mean": float(best_weights_tensor[0]),
                "w_p75": float(best_weights_tensor[1]),
                "w_max": float(best_weights_tensor[2]),
                "w_slope_pos": float(best_weights_tensor[3]),
                "w_std_penalty": float(best_weights_tensor[4]),
                "w_high_ratio": float(best_weights_tensor[5]),
                "w_terminal_gap": float(best_weights_tensor[6]),
            }
            self.evaluator.set_decision_formula_config(
                decision_model_type="learned_linear",
                decision_model_weights=learned_weights,
                decision_model_bias=best_bias,
            )

            if holdout_count > 0:
                positive_weights = torch.tensor(best_weights_tensor, dtype=torch.float32, device=self.device)
                holdout_logits = (
                    holdout_x[:, 0] * positive_weights[0]
                    + holdout_x[:, 1] * positive_weights[1]
                    + holdout_x[:, 2] * positive_weights[2]
                    + holdout_x[:, 3] * positive_weights[3]
                    - holdout_x[:, 4] * positive_weights[4]
                    + holdout_x[:, 5] * positive_weights[5]
                    + holdout_x[:, 6] * positive_weights[6]
                    + best_bias
                )
                holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
                holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
                holdout_pred = holdout_scores >= 0.5
                holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
                holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
            else:
                holdout_accuracy = 0.0
                holdout_auc = 0.0

        info.update(
            {
                "decision_model_status": "fitted",
                "decision_model_holdout_record_count": int(holdout_count),
                "decision_model_holdout_auc": float(holdout_auc),
                "decision_model_holdout_accuracy": float(holdout_accuracy),
                "decision_model_config": self.evaluator.get_decision_formula_config(),
            }
        )
        self.last_decision_model_info = info
        return info

    def _fit_failure_decision_models(self) -> Dict[str, object]:
        mode = self.failure_decision_mode
        base_decision_info = self._fit_learned_decision_weights()
        self._refresh_decision_scores_on_records()
        if mode == "single_fused_score":
            feature_matrix, labels, _ = self._build_fused_training_matrix(self.summary_records)
            fused_model_type = str(self.args.fused_model_type).strip().lower()
            fused_info = self._fit_mlp_failure_model(
                feature_matrix=feature_matrix,
                labels=labels,
                status_key_prefix="fused_model",
            )
            # Online rounds may have temporary insufficient support; keep previous fitted fused model to avoid reset.
            if (
                str(fused_info.get("fused_model_status", "")).strip().lower() == "frozen"
                and str(self.last_failure_model_info.get("fused_model_status", "")).strip().lower() == "fitted"
            ):
                for key in (
                    "fused_model_type",
                    "fused_model_weights",
                    "fused_model_bias",
                    "fused_model_input_mean",
                    "fused_model_input_std",
                    "fused_model_mlp_state",
                    "fused_model_mlp_hidden_dim",
                ):
                    if key in self.last_failure_model_info:
                        fused_info[key] = copy.deepcopy(self.last_failure_model_info[key])
            self.last_failure_model_info = {
                **self.last_failure_model_info,
                **fused_info,
                "fused_model_type": str(fused_info.get("fused_model_type", fused_model_type)),
                "failure_decision_mode": mode,
                "single_threshold_used": True,
                "primary_score_name": "fused_score",
                "primary_score_holdout_auc": float(fused_info.get("fused_model_holdout_auc", 0.0)),
                "decision_model_info": base_decision_info,
            }
            return dict(self.last_failure_model_info)

        feature_matrix, labels, _ = self._build_direct_failure_training_matrix(self.summary_records)
        direct_info = self._fit_linear_failure_model(
            feature_matrix=feature_matrix,
            labels=labels,
            feature_names=(
                "converged_mean_v2",
                "converged_p75_v2",
                "converged_max_v2",
                "converged_slope_v2",
                "converged_std_v2",
                "converged_high_ratio_v2",
                "terminal_risk_score",
                "terminal_score_gap_v2",
            ),
            status_key_prefix="direct_model",
        )
        self.last_failure_model_info = {
            **self.last_failure_model_info,
            **direct_info,
            "failure_decision_mode": mode,
            "single_threshold_used": True,
            "primary_score_name": "final_failure_probability",
            "primary_score_holdout_auc": float(direct_info.get("direct_model_holdout_auc", 0.0)),
            "decision_model_info": base_decision_info,
        }
        return dict(self.last_failure_model_info)

    def _compute_fused_score_for_record(self, record: Dict) -> float:
        score, _logit = self._compute_fused_score_and_logit_for_record(record)
        return float(score)

    def _compute_fused_score_and_logit_for_record(self, record: Dict) -> Tuple[float, Optional[float]]:
        input_mean = np.asarray(self.last_failure_model_info.get("fused_model_input_mean", []), dtype=np.float32)
        input_std = np.asarray(self.last_failure_model_info.get("fused_model_input_std", []), dtype=np.float32)
        mlp_state = dict(self.last_failure_model_info.get("fused_model_mlp_state", {}))
        features = np.asarray(self._build_fused_feature_vector(record), dtype=np.float32)
        if (
            input_mean.size == features.size
            and input_std.size == features.size
            and all(key in mlp_state for key in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
        ):
            normalized = (features - input_mean) / np.where(input_std < 1e-6, 1.0, input_std)
            w1 = np.asarray(mlp_state["0.weight"], dtype=np.float32)
            b1 = np.asarray(mlp_state["0.bias"], dtype=np.float32)
            w2 = np.asarray(mlp_state["2.weight"], dtype=np.float32)
            b2 = np.asarray(mlp_state["2.bias"], dtype=np.float32)
            w3 = np.asarray(mlp_state["4.weight"], dtype=np.float32)
            b3 = np.asarray(mlp_state["4.bias"], dtype=np.float32)
            h1 = np.maximum(0.0, normalized @ w1.T + b1)
            h2 = np.maximum(0.0, h1 @ w2.T + b2)
            logit = float(np.ravel(h2 @ w3.T + b3)[0])
            score = float(self._sigmoid_np(np.asarray([logit], dtype=np.float32))[0])
            return score, logit
        return 0.0, None

    def _compute_direct_failure_probability_for_record(self, record: Dict) -> float:
        weights = dict(self.last_failure_model_info.get("direct_model_weights", {}))
        bias = float(self.last_failure_model_info.get("direct_model_bias", 0.0))
        linear = (
            float(weights.get("converged_mean_v2", 0.0)) * float(record.get("converged_mean_v2", 0.0))
            + float(weights.get("converged_p75_v2", 0.0)) * float(record.get("converged_p75_v2", 0.0))
            + float(weights.get("converged_max_v2", 0.0)) * float(record.get("converged_max_v2", 0.0))
            + float(weights.get("converged_slope_v2", 0.0)) * float(record.get("converged_slope_v2", 0.0))
            + float(weights.get("converged_std_v2", 0.0)) * float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0)))
            + float(weights.get("converged_high_ratio_v2", 0.0)) * float(record.get("converged_high_ratio_v2", 0.0))
            + float(weights.get("terminal_risk_score", 0.0)) * float(record.get("terminal_risk_score", 0.0))
            + float(weights.get("terminal_score_gap_v2", 0.0)) * float(record.get("terminal_score_gap_v2", 0.0))
            + bias
        )
        return float(self._sigmoid_np(np.asarray([linear], dtype=np.float32))[0])

    def _calibrate_single_score_threshold(
        self,
        train_scores: Sequence[float],
        train_labels: Sequence[bool],
        holdout_scores: Sequence[float],
        holdout_labels: Sequence[bool],
        threshold_key: str,
        holdout_auc_key: str,
    ) -> Dict[str, object]:
        objective = str(self.args.threshold_objective).strip().lower()
        if not train_scores or not train_labels:
            return {
                threshold_key: 0.5,
                "threshold": 0.5,
                "status": "frozen",
                "objective_score": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "holdout_metrics": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0},
                holdout_auc_key: 0.0,
                "single_threshold_used": True,
            }

        train_np = np.asarray(train_scores, dtype=float)
        labels_np = np.asarray(train_labels, dtype=bool)
        threshold_candidates = np.unique(np.clip(train_np, 0.0, 1.0))
        if len(threshold_candidates) == 0:
            threshold_candidates = np.array([0.5], dtype=float)

        best_threshold = float(threshold_candidates[0])
        best_metrics = None
        best_score = -1.0
        for threshold in threshold_candidates:
            pred = train_np >= float(threshold)
            metrics = self.evaluator._prediction_metrics(pred, labels_np)
            if objective == "recall_at_precision" and metrics["precision"] < float(self.args.threshold_min_precision):
                continue
            score = float(self.evaluator._objective_value(metrics, objective))
            if score > best_score + 1e-12:
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics
            elif abs(score - best_score) <= 1e-12 and best_metrics is not None:
                if int(metrics["fp"]) < int(best_metrics["fp"]):
                    best_threshold = float(threshold)
                    best_metrics = metrics

        if best_metrics is None:
            best_threshold = 0.5
            best_metrics = self.evaluator._prediction_metrics(train_np >= best_threshold, labels_np)
            best_score = float(self.evaluator._objective_value(best_metrics, objective))

        holdout_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        if holdout_scores and holdout_labels:
            holdout_metrics = self._evaluate_objective_metrics(
                decision_scores=holdout_scores,
                terminal_scores=[1.0] * len(holdout_scores),
                true_labels=holdout_labels,
                decision_threshold=best_threshold,
                terminal_threshold=0.5,
            )
            holdout_auc = self._roc_auc_binary(list(holdout_scores), list(holdout_labels))
        else:
            holdout_auc = 0.0

        return {
            threshold_key: float(best_threshold),
            "threshold": float(best_threshold),
            "status": "updated",
            "objective_score": float(best_score),
            "f1": float(best_metrics["f1"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "holdout_metrics": holdout_metrics,
            holdout_auc_key: float(holdout_auc),
            "single_threshold_used": True,
        }

    def _refresh_decision_scores_on_records(self):
        baseline_weights = {
            "w_mean": 0.60,
            "w_p75": 0.25,
            "w_max": 0.10,
            "w_slope_pos": 0.10,
            "w_std_penalty": 0.20,
        }
        for record in self.summary_records:
            if not all(
                key in record
                for key in ("converged_mean_v2", "converged_p75_v2", "converged_max_v2", "converged_slope_v2")
            ):
                continue
            (
                decision_score,
                decision_score_linear,
                decision_feature_contributions,
                decision_score_formula_version,
            ) = self.evaluator.compute_decision_score_v2(
                converged_mean_v2=float(record.get("converged_mean_v2", 0.0)),
                converged_p75_v2=float(record.get("converged_p75_v2", 0.0)),
                converged_max_v2=float(record.get("converged_max_v2", 0.0)),
                converged_slope_v2=float(record.get("converged_slope_v2", 0.0)),
                converged_std_v2=float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                converged_high_ratio_v2=float(record.get("converged_high_ratio_v2", 0.0)),
                terminal_score_gap_v2=float(
                    record.get(
                        "terminal_score_gap_v2",
                        max(
                            0.0,
                            float(record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0))
                            - float(record.get("converged_mean_v2", 0.0)),
                        ),
                    )
                ),
            )
            record["decision_score_v2"] = float(decision_score)
            record["decision_score_v2_linear"] = float(decision_score_linear)
            custom_weights = any(
                abs(float(self.evaluator.decision_formula_weights.get(k, 0.0)) - v) > 1e-12
                for k, v in baseline_weights.items()
            )
            if self.evaluator.decision_model_type == "learned_linear":
                record["decision_score_formula_version"] = decision_score_formula_version
            else:
                record["decision_score_formula_version"] = "v4" if self.evaluator.enable_decision_tail_boost or custom_weights else "v3"
            record["decision_formula_weights"] = dict(self.evaluator.decision_formula_weights)
            record["decision_model_type"] = self.evaluator.decision_model_type
            record["decision_model_weights"] = dict(self.evaluator.decision_model_weights)
            record["decision_model_bias"] = float(self.evaluator.decision_model_bias)
            record["decision_feature_contributions"] = dict(decision_feature_contributions)
            record["enable_decision_tail_boost"] = bool(self.evaluator.enable_decision_tail_boost)
            record["decision_tail_gamma"] = float(self.evaluator.decision_tail_gamma)

    def _recompute_predictions_from_thresholds(self):
        self._refresh_decision_scores_on_records()
        mode = self.failure_decision_mode
        for record in self.summary_records:
            decision_score = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
            terminal_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            fused_score = 0.0
            fused_logit: Optional[float] = None
            final_failure_probability = 0.0
            if mode == "single_fused_score":
                fused_score, fused_logit = self._compute_fused_score_and_logit_for_record(record)
                pred_v2 = bool(fused_score >= float(self.last_failure_model_info.get("fused_threshold", 0.5)))
            elif mode == "direct_failure_model":
                final_failure_probability = self._compute_direct_failure_probability_for_record(record)
                pred_v2 = bool(final_failure_probability >= float(self.last_failure_model_info.get("final_threshold", 0.5)))
            else:
                raise ValueError(f"Unsupported failure_decision_mode: {mode}")
            record["system_failure_v2"] = pred_v2
            record["failure_decision_mode"] = mode
            record["decision_policy"] = mode
            record["fused_score"] = float(fused_score)
            record["fused_logit"] = fused_logit
            record["final_failure_probability"] = float(final_failure_probability)
            record["fused_threshold"] = float(self.last_failure_model_info.get("fused_threshold", 0.5))
            record["final_threshold"] = float(self.last_failure_model_info.get("final_threshold", 0.5))
            record["decision_threshold"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold"] = float(self.evaluator.terminal_threshold_v2)
            record["decision_threshold_v2"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold_v2"] = float(self.evaluator.terminal_threshold_v2)
            self._apply_true_failure_policy_to_record(record)
            record["system_failure"] = pred_v2
            record["true_failure"] = bool(record.get("true_failure_v2", False))

    def _annotate_step_records_with_final_sample_scores(self) -> None:
        if not self.step_records or not self.summary_records:
            return

        by_test_id: Dict[int, Dict[str, object]] = {}
        for record in self.summary_records:
            test_id = record.get("test_id")
            if test_id is None:
                continue
            try:
                test_id_int = int(test_id)
            except Exception:
                continue
            by_test_id[test_id_int] = {
                "fused_score": record.get("fused_score", 0.0),
                "fused_logit": record.get("fused_logit", None),
            }

        for row in self.step_records:
            test_id = row.get("test_id")
            if test_id is None:
                continue
            try:
                test_id_int = int(test_id)
            except Exception:
                continue
            payload = by_test_id.get(test_id_int)
            if not payload:
                continue
            row["fused_score"] = payload.get("fused_score", 0.0)
            row["fused_logit"] = payload.get("fused_logit", None)

    def _offline_recompute_from_existing_results(self):
        if not self.summary_records:
            self._load_summary_records_from_round_files()
        if not self.summary_records:
            source_rounds_dir = self._resolve_offline_source_rounds_dir()
            if source_rounds_dir is not None:
                loaded = self._collect_summary_records_from_rounds_dir(source_rounds_dir)
                self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in loaded]
                self.cumulative_failure_labels_v2 = [
                    float(bool(record.get("true_failure_v2", False))) for record in self.summary_records
                ]
        if not self.summary_records:
            raise RuntimeError("No existing summary records found for offline recompute.")

        self.cumulative_failure_labels_v2 = []
        for record in self.summary_records:
            self._apply_true_failure_policy_to_record(record)
            self.cumulative_failure_labels_v2.append(float(bool(record.get("true_failure_v2", False))))

        if self.args.offline_decision_threshold is not None:
            self.evaluator.set_v2_failure_threshold(float(self.args.offline_decision_threshold))
        if self.args.offline_terminal_threshold is not None:
            self.evaluator.set_terminal_threshold_v2(float(self.args.offline_terminal_threshold))

        decision_model_stats = self._fit_failure_decision_models()
        self._recompute_predictions_from_thresholds()
        self._write_offline_decision_distribution(tag="before_offline_recompute")

        threshold_stats: Dict[str, object] = {
            "status": "frozen",
            "mode": "offline",
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
        }
        manual_threshold_override = self.args.offline_decision_threshold is not None
        if not bool(self.args.offline_use_existing_thresholds) and not manual_threshold_override:
            threshold_stats = self._calibrate_failure_threshold_v2()

        self._recompute_predictions_from_thresholds()
        self._write_offline_decision_distribution(tag="after_offline_recompute")
        self.threshold_update_status = str(threshold_stats.get("status", self.threshold_update_status))
        self.last_threshold_stats = dict(threshold_stats)
        self.stop_reason = "offline_recompute_only"
        self.finished = True
        self._save_state()
        offline_path = self.session_dir / "offline_recompute_summary.json"
        offline_payload = {
            "timestamp": now_stamp(),
            "true_failure_policy": self.true_failure_v2_policy,
            "true_failure_v2_policy": self.true_failure_v2_policy,
            "used_existing_thresholds": bool(self.args.offline_use_existing_thresholds),
            "offline_source_session": str(self.args.offline_source_session or ""),
            "failure_decision_mode": self.failure_decision_mode,
            "single_threshold_used": bool(self.last_failure_model_info.get("single_threshold_used", False)),
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
            "threshold_stats": threshold_stats,
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "decision_model_info": decision_model_stats,
            "record_count": len(self.summary_records),
        }
        offline_path.write_text(json.dumps(offline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_final_output()

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
                self._run_single_simulation(temp_config_path)
                
                # 修正：匹配 RL_environment_for_computing.py 中的拼接逻辑
                # 文件实际存放在 raw_log_root / training_process_data / ...
                raw_log_path = Path(self.args.raw_log_root) / "training_process_data" / raw_log_name
                performance_file_path = performance_dir / f"test_{self.test_counter:04d}_performance.txt"

                if not raw_log_path.exists():
                    # 回退检查：有时 PRC 会写到项目根目录
                    fallback_path = self.project_root / "training_process_data" / raw_log_name
                    if fallback_path.exists():
                        raw_log_path = fallback_path
                
                if not raw_log_path.exists():
                    raise FileNotFoundError(f"Simulation log not found: {raw_log_path} (also checked {self.project_root / 'training_process_data'})")

                serialize_performance_file(
                    raw_log_path=raw_log_path,
                    output_path=performance_file_path,
                    scenario=scenario,
                    round_index=round_index,
                    test_id=self.test_counter,
                )

                evaluation = self.evaluator.evaluate_log_file(
                    str(performance_file_path),
                    failure_threshold=float(self.args.failure_threshold),
                )
                system_failure_policy = bool(evaluation["predicted_failure_v2"])
                true_failure_policy = (
                    bool(evaluation["true_failure_v2_strict"])
                    if self.true_failure_v2_policy == "strict"
                    else bool(evaluation["true_failure_v2"])
                )
                true_failure_v2_relaxed = bool(evaluation["true_failure_v2"])
                true_failure_v2_strict = bool(evaluation.get("true_failure_v2_strict", true_failure_v2_relaxed))
                selected_true_failure_v2 = (
                    true_failure_v2_strict if self.true_failure_v2_policy == "strict" else true_failure_v2_relaxed
                )
                summary_record = {
                    "round_index": round_index,
                    "test_id": self.test_counter,
                    "scenario": scenario,
                    "total_membership": evaluation["total_membership"],
                    "max_total_membership": evaluation["max_total_membership"],
                    "mean_total_membership": evaluation["mean_total_membership"],
                    "score_uncertainty": evaluation["score_uncertainty"],
                    "total_membership_v2": evaluation["total_membership_v2"],
                    "max_total_membership_v2": evaluation["max_total_membership_v2"],
                    "mean_total_membership_v2": evaluation["mean_total_membership_v2"],
                    "score_uncertainty_v2": evaluation["score_uncertainty_v2"],
                    "decision_score": evaluation["decision_score_v2"],
                    "decision_score_v2": evaluation["decision_score_v2"],
                    "decision_score_v2_linear": evaluation.get("decision_score_v2_linear", evaluation["decision_score_v2"]),
                    "convergence_window_start_step": evaluation["convergence_window_start_step"],
                    "converged_mean": evaluation["converged_mean_v2"],
                    "converged_mean_v2": evaluation["converged_mean_v2"],
                    "converged_p75": evaluation["converged_p75_v2"],
                    "converged_p75_v2": evaluation["converged_p75_v2"],
                    "converged_slope": evaluation.get("converged_slope_v2", 0.0),
                    "converged_slope_v2": evaluation.get("converged_slope_v2", 0.0),
                    "converged_max": evaluation.get("converged_max_v2", 0.0),
                    "converged_max_v2": evaluation.get("converged_max_v2", 0.0),
                    "converged_std": evaluation.get("converged_std_v2", 0.0),
                    "converged_std_v2": evaluation.get("converged_std_v2", 0.0),
                    "converged_high_ratio": evaluation.get("converged_high_ratio_v2", 0.0),
                    "converged_high_ratio_v2": evaluation.get("converged_high_ratio_v2", 0.0),
                    "terminal_score_gap": evaluation.get("terminal_score_gap_v2", 0.0),
                    "terminal_score_gap_v2": evaluation.get("terminal_score_gap_v2", 0.0),
                    "decision_score_formula_version": evaluation.get("decision_score_formula_version", "v2"),
                    "decision_feature_contributions": dict(evaluation.get("decision_feature_contributions", {})),
                    "decision_model_type": str(evaluation.get("decision_model_type", self.evaluator.decision_model_type)),
                    "decision_model_weights": dict(evaluation.get("decision_model_weights", self.evaluator.decision_model_weights)),
                    "decision_model_bias": float(evaluation.get("decision_model_bias", self.evaluator.decision_model_bias)),
                    "failure_decision_mode": self.failure_decision_mode,
                    "decision_policy": self.decision_policy,
                    "fused_score": 0.0,
                    "final_failure_probability": 0.0,
                    "fused_threshold": 0.5,
                    "final_threshold": 0.5,
                    "terminal_hard_failure": bool(evaluation["terminal_hard_failure"]),
                    "terminal_risk_score": float(evaluation.get("terminal_risk_score", 0.0)),
                    "terminal_risk_weights": dict(evaluation.get("terminal_risk_weights", self.evaluator.get_terminal_risk_weights())),
                    "system_failure_v1": bool(evaluation["system_failure"]),
                    "system_failure_v2": bool(evaluation["predicted_failure_v2"]),
                    "true_failure_v1": bool(evaluation["true_failure"]),
                    "true_failure_v2_relaxed": true_failure_v2_relaxed,
                    "true_failure_v2_strict": true_failure_v2_strict,
                    "true_failure_v2": selected_true_failure_v2,
                    "system_failure": system_failure_policy,
                    "true_failure": true_failure_policy,
                    "decision_threshold": float(evaluation["v2_failure_threshold"]),
                    "v2_failure_threshold": float(evaluation["v2_failure_threshold"]),
                    "terminal_threshold": float(evaluation.get("terminal_threshold_v2", self.evaluator.terminal_threshold_v2)),
                    "decision_threshold_v2": float(evaluation.get("decision_threshold_v2", evaluation["v2_failure_threshold"])),
                    "terminal_threshold_v2": float(evaluation.get("terminal_threshold_v2", self.evaluator.terminal_threshold_v2)),
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
                        "failure_score_v2": step_eval["total_membership_v2"],
                        "step_aux_failure_signal_v2": step_eval["total_membership_v2"],
                        "failure_decision_v1": bool(step_eval["total_membership"] > self.args.failure_threshold),
                        "failure_decision_v2": bool(step_eval["total_membership_v2"] >= evaluation["v2_failure_threshold"]),
                        "true_failure_v2_step": bool(step_eval.get("true_failure_v2", False)),
                        "failure_decision": bool(step_eval["total_membership_v2"] >= evaluation["v2_failure_threshold"]),
                        "test_failure_score": evaluation["total_membership"],
                        "test_failure_score_v2": evaluation["total_membership_v2"],
                        "test_failure_decision": system_failure_policy,
                        "test_failure_decision_v1": bool(evaluation["system_failure"]),
                        "test_failure_decision_v2": bool(evaluation["predicted_failure_v2"]),
                    }
                    round_step_records.append(row)

                score_file.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

                continuous_values, discrete_values = scenario_feature_arrays(scenario)
                self.cumulative_continuous_features.append(continuous_values)
                self.cumulative_discrete_features.append(discrete_values)
                self.cumulative_failure_scores.append(float(evaluation["total_membership"]))
                self.cumulative_failure_labels_v2.append(float(bool(selected_true_failure_v2)))
                self.summary_records.append(summary_record)
                self.step_records.extend(row for row in round_step_records if row["test_id"] == self.test_counter)

                temp_config_path.unlink(missing_ok=True)
                raw_log_path.unlink(missing_ok=True)

        return round_summary_records, round_step_records

    def _incremental_train_and_evaluate_coverage(self, round_index: int, round_summary_records: Sequence[Dict], round_step_records: Sequence[Dict]) -> Path:
        round_dir = ensure_dir(self.rounds_dir / f"round_{round_index:03d}")
        evalu_path = round_dir / "evalu.txt"

        # Keep online behavior consistent with offline: fit active decision model before threshold calibration.
        self._fit_failure_decision_models()
        threshold_stats = self._calibrate_failure_threshold_v2()
        if bool(self.args.online_backfill_after_each_round):
            self._recompute_predictions_from_thresholds()

        continuous_tensor = torch.tensor(np.array(self.cumulative_continuous_features), dtype=torch.float32)
        discrete_tensor = torch.tensor(np.array(self.cumulative_discrete_features), dtype=torch.long)
        target_tensor = torch.tensor(np.array(self.cumulative_failure_scores), dtype=torch.float32)
        target_failure_labels_v2 = torch.tensor(np.array(self.cumulative_failure_labels_v2), dtype=torch.float32)

        train_stats = self.ensemble.fit_incremental(
            continuous_tensor,
            discrete_tensor,
            regression_targets=target_tensor,
            classification_targets=target_failure_labels_v2,
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
        score_key = "decision_score_v2"
        actual_scores_np = np.array([record.get(score_key, record["total_membership"]) for record in self.summary_records], dtype=float)
        uncertainty_key = "score_uncertainty_v2"
        actual_uncertainties_np = np.array(
            [float(record.get(uncertainty_key, record.get("score_uncertainty", 0.0))) for record in self.summary_records],
            dtype=float,
        )

        explorer_clusters = max(1, min(int(self.args.n_clusters), len(features_np)))
        self.explorer.n_clusters = explorer_clusters
        region_stats = self.explorer.partition_and_evaluate(
            features=features_np,
            failure_scores=actual_scores_np,
            cv_values=actual_uncertainties_np,
            theoretical_max_per_region=max(2, len(features_np) // explorer_clusters or 1),
            sc_schedule=self.coverage_sc_schedule,
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
            f.write(f"THRESHOLD_CALIBRATION_JSON: {json.dumps(threshold_stats, ensure_ascii=False)}\n")
            f.write(f"COVERAGE_JSON: {json.dumps(coverage_metrics, ensure_ascii=False)}\n")
            for record in round_summary_records:
                f.write(f"TEST_SUMMARY_JSON: {json.dumps(record, ensure_ascii=False)}\n")
            for row in round_step_records:
                f.write(f"STEP_EVAL_JSON: {json.dumps(row, ensure_ascii=False)}\n")

        self.round_evalu_files.append(str(evalu_path))
        return evalu_path

    def _resolve_terminal_weight_objective(self) -> str:
        objective = str(self.args.terminal_weight_objective or "").strip().lower()
        allowed = {"f1", "accuracy", "balanced_accuracy"}
        if objective in allowed:
            return objective
        threshold_objective = str(self.args.threshold_objective).strip().lower()
        if threshold_objective in allowed:
            return threshold_objective
        return "balanced_accuracy"

    @staticmethod
    def _project_terminal_weights(weight_vector: np.ndarray) -> np.ndarray:
        projected = np.maximum(np.asarray(weight_vector, dtype=float), 0.0)
        total = float(np.sum(projected))
        if total <= 1e-12:
            return np.array([0.35, 0.20, 0.20, 0.25], dtype=float)
        return projected / total

    def _terminal_weight_vector_to_dict(self, weight_vector: np.ndarray) -> Dict[str, float]:
        vector = self._project_terminal_weights(weight_vector)
        return {
            "packet_loss": float(vector[0]),
            "e2e_delay": float(vector[1]),
            "throughput": float(vector[2]),
            "ending_reward": float(vector[3]),
        }

    def _extract_terminal_metrics_by_test(self) -> Dict[int, Dict[str, float]]:
        terminal_by_test: Dict[int, Dict[str, float]] = {}
        metric_keys = ("PacketLossRate", "AverageE2eDelay", "NetworkThroughput", "AverageEndingReward")
        for row in self.step_records:
            test_id = int(row.get("test_id", -1))
            step_index = int(row.get("step_index", -1))
            if test_id < 0 or step_index < 0:
                continue
            previous = terminal_by_test.get(test_id)
            if previous is not None and int(previous.get("_step_index", -1)) >= step_index:
                continue
            metrics = {key: float(row.get(key, 0.0)) for key in metric_keys}
            metrics["_step_index"] = step_index
            terminal_by_test[test_id] = metrics
        for value in terminal_by_test.values():
            value.pop("_step_index", None)
        return terminal_by_test

    def _evaluate_objective_metrics(
        self,
        decision_scores: Sequence[float],
        terminal_scores: Sequence[float],
        true_labels: Sequence[bool],
        decision_threshold: float,
        terminal_threshold: float,
    ) -> Dict[str, float]:
        if not decision_scores or not true_labels:
            return {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
            }
        decision_np = np.asarray(decision_scores, dtype=float)
        terminal_np = np.asarray(terminal_scores, dtype=float)
        labels_np = np.asarray(true_labels, dtype=bool)
        pred = (decision_np >= float(decision_threshold)) & (terminal_np >= float(terminal_threshold))
        return self.evaluator._prediction_metrics(pred, labels_np)

    def _decision_pass_rate(self, decision_scores: Sequence[float], threshold: float) -> float:
        if not decision_scores:
            return 0.0
        decision_np = np.asarray(decision_scores, dtype=float)
        return float(np.mean(decision_np >= float(threshold)))

    def _default_decision_constraint_info(self, status: str = "disabled") -> Dict[str, object]:
        return {
            "decision_pass_rate_train": 0.0,
            "decision_pass_rate_holdout": 0.0,
            "decision_threshold_floor_applied": False,
            "decision_threshold_floor_value_effective": 0.0,
            "decision_constraint_status": status,
        }

    def _build_frozen_weight_info(self, reason: str) -> Dict[str, object]:
        objective = self._resolve_terminal_weight_objective()
        current_weights = self.evaluator.get_terminal_risk_weights()
        info = {
            "weight_adaptation_enabled": bool(self.args.enable_terminal_weight_adaptation),
            "weight_update_status": "frozen",
            "weight_objective_metric": objective,
            "weight_objective_score_before": 0.0,
            "weight_objective_score_after": 0.0,
            "weight_holdout_gain": 0.0,
            "weight_delta": {
                "packet_loss": 0.0,
                "e2e_delay": 0.0,
                "throughput": 0.0,
                "ending_reward": 0.0,
            },
            "weight_candidate_count": 0,
            "terminal_risk_weights": dict(current_weights),
            "weight_rollback_triggered": False,
            "reason": str(reason),
        }
        self.weight_update_status = "frozen"
        self.last_weight_adaptation_info = info
        return info

    def _resolve_terminal_scores_for_summary_records(self) -> List[float]:
        terminal_by_test = self._extract_terminal_metrics_by_test()
        current_weights = self.evaluator.get_terminal_risk_weights()
        resolved_scores: List[float] = []
        for record in self.summary_records:
            test_id = int(record.get("test_id", -1))
            metrics = terminal_by_test.get(test_id)
            if metrics is not None:
                resolved_scores.append(float(self.evaluator._terminal_risk_score(metrics, override_weights=current_weights)))
            else:
                fallback_score = float(
                    record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
                )
                resolved_scores.append(fallback_score)
        return resolved_scores

    def _compute_decision_threshold_floor(self, train_scores: Sequence[float]) -> Tuple[float, bool]:
        mode = str(self.args.decision_threshold_floor_mode).strip().lower()
        if mode == "off" or not train_scores:
            return 0.0, False
        scores_np = np.asarray(train_scores, dtype=float)
        if mode == "absolute":
            floor_value = float(np.clip(float(self.args.decision_threshold_floor_value), 0.0, 1.0))
        elif mode == "quantile":
            q = float(np.clip(float(self.args.decision_threshold_floor_quantile), 0.0, 1.0))
            floor_value = float(np.clip(float(np.quantile(scores_np, q)), 0.0, 1.0))
        else:
            ratio = float(max(0.0, float(self.args.decision_threshold_baseline_ratio)))
            floor_value = float(np.clip(float(self.evaluator.v2_failure_threshold) * ratio, 0.0, 1.0))
        return floor_value, True

    def _build_threshold_candidate_pool(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        objective = str(self.args.threshold_objective).strip().lower()
        tolerance = float(max(0.0, float(self.args.decision_threshold_constraint_tolerance)))
        pass_rate_min = float(np.clip(float(self.args.decision_pass_rate_min), 0.0, 1.0))
        pass_rate_max = float(np.clip(float(self.args.decision_pass_rate_max), pass_rate_min, 1.0))
        floor_value, floor_applied = self._compute_decision_threshold_floor(train_scores)
        decision_candidates = np.unique(np.clip(np.asarray(train_scores, dtype=float), 0.0, 1.0))
        if len(decision_candidates) == 0:
            decision_candidates = np.array([float(self.evaluator.v2_failure_threshold)], dtype=float)

        constrained_candidates = decision_candidates[decision_candidates >= (floor_value - 1e-12)]
        floor_status = "satisfied"
        if len(constrained_candidates) == 0:
            constrained_candidates = np.array([float(np.max(decision_candidates))], dtype=float)
            floor_status = "relaxed_fallback"

        if dual_threshold:
            terminal_candidates = np.arange(0.45, 0.751, 0.01, dtype=float)
        else:
            terminal_candidates = np.array([float(self.evaluator.terminal_threshold_v2)], dtype=float)

        viable_candidates: List[Dict[str, object]] = []
        fallback_candidates: List[Dict[str, object]] = []
        target_center = 0.5 * (pass_rate_min + pass_rate_max)
        for decision_threshold in constrained_candidates:
            pass_rate = self._decision_pass_rate(train_scores, float(decision_threshold))
            lower_bound = pass_rate_min - tolerance
            upper_bound = pass_rate_max + tolerance
            if pass_rate < lower_bound:
                violation = lower_bound - pass_rate
            elif pass_rate > upper_bound:
                violation = pass_rate - upper_bound
            else:
                violation = 0.0

            decision_np = np.asarray(train_scores, dtype=float)
            terminal_np = np.asarray(train_terminal_scores, dtype=float)
            labels_np = np.asarray(train_labels, dtype=bool)
            decision_mask = decision_np >= float(decision_threshold)
            for terminal_threshold in terminal_candidates:
                pred = decision_mask & (terminal_np >= float(terminal_threshold))
                metrics = self.evaluator._prediction_metrics(pred, labels_np)
                if objective == "recall_at_precision" and metrics["precision"] < float(self.args.threshold_min_precision):
                    continue
                score = float(self.evaluator._objective_value(metrics, objective))
                candidate = {
                    "decision_threshold": float(decision_threshold),
                    "terminal_threshold": float(terminal_threshold),
                    "metrics": metrics,
                    "objective_score": score,
                    "decision_pass_rate": float(pass_rate),
                    "constraint_violation": float(violation),
                    "distance_to_center": float(abs(pass_rate - target_center)),
                }
                if violation <= 1e-12:
                    viable_candidates.append(candidate)
                fallback_candidates.append(candidate)

        if viable_candidates:
            active_pool = viable_candidates
            constraint_status = floor_status if floor_status != "relaxed_fallback" else "relaxed_fallback"
        else:
            active_pool = fallback_candidates
            constraint_status = "relaxed_fallback"

        return {
            "objective": objective,
            "floor_applied": bool(floor_applied),
            "floor_value": float(floor_value),
            "constraint_status": str(constraint_status),
            "target_center": float(target_center),
            "viable_candidates": viable_candidates,
            "fallback_candidates": fallback_candidates,
            "active_pool": active_pool,
        }

    def _select_best_threshold_candidate_legacy(self, candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not candidates:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate["objective_score"] > best["objective_score"] + 1e-12:
                best = candidate
            elif abs(candidate["objective_score"] - best["objective_score"]) <= 1e-12:
                if candidate["constraint_violation"] < best["constraint_violation"] - 1e-12:
                    best = candidate
                elif abs(candidate["constraint_violation"] - best["constraint_violation"]) <= 1e-12:
                    if candidate["distance_to_center"] < best["distance_to_center"] - 1e-12:
                        best = candidate
                    elif abs(candidate["distance_to_center"] - best["distance_to_center"]) <= 1e-12:
                        if candidate["metrics"]["fp"] < best["metrics"]["fp"]:
                            best = candidate
        return best

    def _calibrate_thresholds_with_decision_constraints(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        pool_info = self._build_threshold_candidate_pool(
            train_scores=train_scores,
            train_terminal_scores=train_terminal_scores,
            train_labels=train_labels,
            dual_threshold=dual_threshold,
        )
        objective = str(pool_info["objective"])
        floor_applied = bool(pool_info["floor_applied"])
        floor_value = float(pool_info["floor_value"])
        constraint_status = str(pool_info["constraint_status"])
        candidate_pool = list(pool_info["active_pool"])

        best = self._select_best_threshold_candidate_legacy(candidate_pool)
        if best is None:
            threshold = float(np.clip(max(floor_value, float(self.evaluator.v2_failure_threshold)), 0.0, 1.0))
            terminal_threshold = float(self.evaluator.terminal_threshold_v2)
            metrics = self._evaluate_objective_metrics(
                decision_scores=train_scores,
                terminal_scores=train_terminal_scores,
                true_labels=train_labels,
                decision_threshold=threshold,
                terminal_threshold=terminal_threshold,
            )
            return {
                "decision_threshold": threshold,
                "terminal_threshold": terminal_threshold,
                "threshold": threshold,
                "objective_score": float(self.evaluator._objective_value(metrics, objective)),
                **metrics,
                "decision_pass_rate_train": float(self._decision_pass_rate(train_scores, threshold)),
                "decision_threshold_floor_applied": bool(floor_applied),
                "decision_threshold_floor_value_effective": float(floor_value),
                "decision_constraint_status": "relaxed_fallback",
                "selection_stage": "legacy",
            }

        return {
            "decision_threshold": float(best["decision_threshold"]),
            "terminal_threshold": float(best["terminal_threshold"]),
            "threshold": float(best["decision_threshold"]),
            "f1": float(best["metrics"]["f1"]),
            "precision": float(best["metrics"]["precision"]),
            "recall": float(best["metrics"]["recall"]),
            "accuracy": float(best["metrics"]["accuracy"]),
            "balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
            "objective_score": float(best["objective_score"]),
            "decision_pass_rate_train": float(best["decision_pass_rate"]),
            "decision_threshold_floor_applied": bool(floor_applied),
            "decision_threshold_floor_value_effective": float(floor_value),
            "decision_constraint_status": constraint_status,
            "selection_stage": "legacy",
        }

    def _calibrate_thresholds_two_stage_stable(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        holdout_scores: Sequence[float],
        holdout_terminal_scores: Sequence[float],
        holdout_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        pool_info = self._build_threshold_candidate_pool(
            train_scores=train_scores,
            train_terminal_scores=train_terminal_scores,
            train_labels=train_labels,
            dual_threshold=dual_threshold,
        )
        objective = str(pool_info["objective"])
        floor_applied = bool(pool_info["floor_applied"])
        floor_value = float(pool_info["floor_value"])
        constraint_status = str(pool_info["constraint_status"])
        candidate_pool = list(pool_info["active_pool"])
        if not candidate_pool:
            return self._calibrate_thresholds_with_decision_constraints(
                train_scores=train_scores,
                train_terminal_scores=train_terminal_scores,
                train_labels=train_labels,
                dual_threshold=dual_threshold,
            )

        sorted_candidates = sorted(
            candidate_pool,
            key=lambda c: (
                -float(c["objective_score"]),
                float(c["constraint_violation"]),
                float(c["distance_to_center"]),
                int(c["metrics"].get("fp", 0)),
            ),
        )
        top_k = int(max(1, int(self.args.threshold_two_stage_top_k)))
        shortlist = sorted_candidates[: min(top_k, len(sorted_candidates))]
        top_train_objective = float(shortlist[0]["objective_score"])

        gap_penalty = float(max(0.0, float(self.args.threshold_two_stage_gap_penalty)))
        gap_tolerance = float(max(0.0, float(self.args.threshold_two_stage_gap_tolerance)))
        drift_penalty = float(max(0.0, float(self.args.threshold_two_stage_passrate_drift_penalty)))

        best: Optional[Dict[str, object]] = None
        for candidate in shortlist:
            decision_threshold = float(candidate["decision_threshold"])
            terminal_threshold = float(candidate["terminal_threshold"])
            holdout_metrics = self._evaluate_objective_metrics(
                decision_scores=holdout_scores,
                terminal_scores=holdout_terminal_scores,
                true_labels=holdout_labels,
                decision_threshold=decision_threshold,
                terminal_threshold=terminal_threshold,
            )
            holdout_objective = float(self.evaluator._objective_value(holdout_metrics, objective))
            train_objective = float(candidate["objective_score"])
            generalization_gap = max(0.0, train_objective - holdout_objective - gap_tolerance)
            pass_rate_holdout = float(self._decision_pass_rate(holdout_scores, decision_threshold))
            pass_rate_train = float(candidate["decision_pass_rate"])
            pass_rate_drift = abs(pass_rate_holdout - pass_rate_train)
            stability_score = holdout_objective - gap_penalty * generalization_gap - drift_penalty * pass_rate_drift

            enriched = dict(candidate)
            enriched["holdout_metrics"] = holdout_metrics
            enriched["holdout_objective_score"] = holdout_objective
            enriched["train_objective_score"] = train_objective
            enriched["generalization_gap"] = float(generalization_gap)
            enriched["pass_rate_holdout"] = float(pass_rate_holdout)
            enriched["pass_rate_drift"] = float(pass_rate_drift)
            enriched["stability_score"] = float(stability_score)
            if best is None:
                best = enriched
                continue
            if enriched["stability_score"] > best["stability_score"] + 1e-12:
                best = enriched
            elif abs(enriched["stability_score"] - best["stability_score"]) <= 1e-12:
                if enriched["holdout_objective_score"] > best["holdout_objective_score"] + 1e-12:
                    best = enriched
                elif abs(enriched["holdout_objective_score"] - best["holdout_objective_score"]) <= 1e-12:
                    if int(enriched["holdout_metrics"].get("fn", 0)) < int(best["holdout_metrics"].get("fn", 0)):
                        best = enriched
                    elif int(enriched["holdout_metrics"].get("fn", 0)) == int(best["holdout_metrics"].get("fn", 0)):
                        if int(enriched["holdout_metrics"].get("fp", 0)) < int(best["holdout_metrics"].get("fp", 0)):
                            best = enriched

        if best is None:
            return self._calibrate_thresholds_with_decision_constraints(
                train_scores=train_scores,
                train_terminal_scores=train_terminal_scores,
                train_labels=train_labels,
                dual_threshold=dual_threshold,
            )

        return {
            "decision_threshold": float(best["decision_threshold"]),
            "terminal_threshold": float(best["terminal_threshold"]),
            "threshold": float(best["decision_threshold"]),
            "f1": float(best["metrics"]["f1"]),
            "precision": float(best["metrics"]["precision"]),
            "recall": float(best["metrics"]["recall"]),
            "accuracy": float(best["metrics"]["accuracy"]),
            "balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
            "objective_score": float(best["objective_score"]),
            "decision_pass_rate_train": float(best["decision_pass_rate"]),
            "decision_threshold_floor_applied": bool(floor_applied),
            "decision_threshold_floor_value_effective": float(floor_value),
            "decision_constraint_status": constraint_status,
            "selection_stage": "two_stage_stable",
            "stage1_candidate_count": int(len(candidate_pool)),
            "stage1_viable_candidate_count": int(len(pool_info["viable_candidates"])),
            "stage2_shortlist_count": int(len(shortlist)),
            "stage2_top_train_objective": float(top_train_objective),
            "stage2_selected_train_objective": float(best["train_objective_score"]),
            "stage2_selected_holdout_objective": float(best["holdout_objective_score"]),
            "stage2_selected_stability_score": float(best["stability_score"]),
            "stage2_selected_generalization_gap": float(best["generalization_gap"]),
            "stage2_selected_pass_rate_drift": float(best["pass_rate_drift"]),
        }

    def _adapt_terminal_risk_weights(self) -> Dict[str, object]:
        objective = self._resolve_terminal_weight_objective()
        current_weights = self.evaluator.get_terminal_risk_weights()
        info: Dict[str, object] = {
            "weight_adaptation_enabled": bool(self.args.enable_terminal_weight_adaptation),
            "weight_update_status": "frozen",
            "weight_objective_metric": objective,
            "weight_objective_score_before": 0.0,
            "weight_objective_score_after": 0.0,
            "weight_holdout_gain": 0.0,
            "weight_delta": {
                "packet_loss": 0.0,
                "e2e_delay": 0.0,
                "throughput": 0.0,
                "ending_reward": 0.0,
            },
            "weight_candidate_count": 0,
            "terminal_risk_weights": dict(current_weights),
            "weight_rollback_triggered": False,
        }
        if not bool(self.args.enable_terminal_weight_adaptation):
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        terminal_by_test = self._extract_terminal_metrics_by_test()
        if not terminal_by_test or not self.summary_records:
            info["reason"] = "insufficient_terminal_metrics"
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        raw_decision_scores: List[float] = []
        raw_true_labels: List[bool] = []
        raw_terminal_flags: List[bool] = []
        raw_terminal_metrics: List[Dict[str, float]] = []
        for record in self.summary_records:
            test_id = int(record.get("test_id", -1))
            metrics = terminal_by_test.get(test_id)
            if metrics is None:
                continue
            raw_decision_scores.append(float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0))))
            raw_true_labels.append(self._resolve_true_failure_v2_value(record))
            raw_terminal_flags.append(bool(record.get("terminal_hard_failure", False)))
            raw_terminal_metrics.append(metrics)

        if self.args.threshold_calibration_scope == "terminal_only":
            effective_indices = [idx for idx, flag in enumerate(raw_terminal_flags) if flag]
        else:
            effective_indices = list(range(len(raw_true_labels)))

        min_support = int(max(2, self.args.terminal_weight_min_support))
        if len(effective_indices) < min_support:
            info["reason"] = "insufficient_effective_support"
            info["effective_support"] = len(effective_indices)
            info["min_support"] = min_support
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        holdout_ratio = float(np.clip(float(self.args.threshold_calibration_holdout_ratio), 0.0, 0.49))
        holdout_count = int(np.floor(len(effective_indices) * holdout_ratio))
        train_count = len(effective_indices) - holdout_count
        if train_count < min_support:
            info["reason"] = "insufficient_train_support"
            info["effective_support"] = len(effective_indices)
            info["train_support"] = train_count
            info["holdout_support"] = holdout_count
            info["min_support"] = min_support
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        train_idx = effective_indices[:train_count]
        holdout_idx = effective_indices[train_count:]

        train_decision_scores = [raw_decision_scores[idx] for idx in train_idx]
        train_labels = [raw_true_labels[idx] for idx in train_idx]
        train_metrics = [raw_terminal_metrics[idx] for idx in train_idx]
        holdout_decision_scores = [raw_decision_scores[idx] for idx in holdout_idx]
        holdout_labels = [raw_true_labels[idx] for idx in holdout_idx]
        holdout_metrics = [raw_terminal_metrics[idx] for idx in holdout_idx]

        positive_count = sum(1 for label in train_labels if label)
        negative_count = len(train_labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            info["reason"] = "single_class_labels"
            info["positive_count"] = positive_count
            info["negative_count"] = negative_count
            info["effective_support"] = len(effective_indices)
            info["train_support"] = train_count
            info["holdout_support"] = holdout_count
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        def evaluate_candidate(weight_dict: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float], float, float]:
            train_terminal_scores = [
                self.evaluator._terminal_risk_score(metrics, override_weights=weight_dict)
                for metrics in train_metrics
            ]
            dual_stats = self.evaluator.calibrate_dual_failure_threshold(
                decision_scores=train_decision_scores,
                terminal_scores=train_terminal_scores,
                true_labels=train_labels,
                objective=self.args.threshold_objective,
                min_precision=self.args.threshold_min_precision,
                terminal_threshold_candidates=np.arange(0.45, 0.751, 0.01, dtype=float),
            )
            decision_threshold = float(dual_stats["decision_threshold"])
            terminal_threshold = float(dual_stats["terminal_threshold"])
            holdout_terminal_scores = [
                self.evaluator._terminal_risk_score(metrics, override_weights=weight_dict)
                for metrics in holdout_metrics
            ]
            holdout_result = self._evaluate_objective_metrics(
                decision_scores=holdout_decision_scores,
                terminal_scores=holdout_terminal_scores,
                true_labels=holdout_labels,
                decision_threshold=decision_threshold,
                terminal_threshold=terminal_threshold,
            )
            objective_score = float(self.evaluator._objective_value(holdout_result, objective))
            return objective_score, holdout_result, dual_stats, decision_threshold, terminal_threshold

        current_score, current_holdout_metrics, _, _, _ = evaluate_candidate(current_weights)
        info["weight_objective_score_before"] = float(current_score)

        local_step = float(max(1e-6, self.args.terminal_weight_local_step))
        candidate_count = int(max(1, self.args.terminal_weight_candidates))
        current_vector = np.array(
            [
                current_weights["packet_loss"],
                current_weights["e2e_delay"],
                current_weights["throughput"],
                current_weights["ending_reward"],
            ],
            dtype=float,
        )
        candidate_vectors: List[np.ndarray] = [current_vector]
        for _ in range(max(0, candidate_count - 1)):
            noise = self.weight_rng.uniform(-local_step, local_step, size=current_vector.shape[0])
            candidate_vectors.append(self._project_terminal_weights(current_vector + noise))
        info["weight_candidate_count"] = len(candidate_vectors)

        best_score = current_score
        best_weights = dict(current_weights)
        best_dual_stats = None
        for vector in candidate_vectors:
            candidate_weights = self._terminal_weight_vector_to_dict(vector)
            candidate_score, _, dual_stats, _, _ = evaluate_candidate(candidate_weights)
            if candidate_score > best_score:
                best_score = candidate_score
                best_weights = candidate_weights
                best_dual_stats = dual_stats

        min_improvement = float(max(0.0, self.args.terminal_weight_min_improvement))
        improvement = float(best_score - current_score)
        info["weight_holdout_gain"] = improvement
        info["holdout_metrics_before"] = current_holdout_metrics
        info["effective_support"] = len(effective_indices)
        info["train_support"] = train_count
        info["holdout_support"] = holdout_count
        info["objective"] = objective

        if improvement < min_improvement:
            info["reason"] = "no_sufficient_gain"
            info["weight_objective_score_after"] = float(current_score)
            info["terminal_risk_weights"] = dict(current_weights)
            self.weight_update_status = "frozen"
            snapshot = {
                "round_index": int(self.round_index),
                "objective_score": float(current_score),
                "weights": dict(current_weights),
            }
            self.weight_objective_history.append(snapshot)
            self.weight_objective_history = self.weight_objective_history[-10:]
            self.last_weight_adaptation_info = info
            return info

        ema = float(np.clip(float(self.args.terminal_weight_ema), 0.0, 1.0))
        best_vector = np.array(
            [
                best_weights["packet_loss"],
                best_weights["e2e_delay"],
                best_weights["throughput"],
                best_weights["ending_reward"],
            ],
            dtype=float,
        )
        updated_vector = (1.0 - ema) * current_vector + ema * best_vector
        max_delta = float(max(1e-6, self.args.terminal_weight_max_delta))
        delta = np.clip(updated_vector - current_vector, -max_delta, max_delta)
        bounded_vector = self._project_terminal_weights(current_vector + delta)
        updated_weights = self._terminal_weight_vector_to_dict(bounded_vector)
        self.evaluator.set_terminal_risk_weights(updated_weights)

        after_score, after_holdout_metrics, after_dual_stats, _, _ = evaluate_candidate(updated_weights)
        info["weight_objective_score_after"] = float(after_score)
        info["holdout_metrics_after"] = after_holdout_metrics
        info["thresholds_after_weight_update"] = {
            "decision_threshold": float(after_dual_stats["decision_threshold"]),
            "terminal_threshold": float(after_dual_stats["terminal_threshold"]),
        }
        info["weight_delta"] = {
            key: float(updated_weights[key] - current_weights[key])
            for key in ("packet_loss", "e2e_delay", "throughput", "ending_reward")
        }
        info["terminal_risk_weights"] = dict(updated_weights)
        info["best_dual_stats_preview"] = best_dual_stats
        self.weight_update_status = "updated"
        info["weight_update_status"] = "updated"

        snapshot = {
            "round_index": int(self.round_index),
            "objective_score": float(after_score),
            "weights": dict(updated_weights),
        }
        self.weight_objective_history.append(snapshot)
        self.weight_objective_history = self.weight_objective_history[-10:]

        rollback_p = int(max(1, self.args.terminal_weight_rollback_patience))
        rollback_drop = float(max(0.0, self.args.terminal_weight_rollback_min_drop))
        history_scores = [float(item.get("objective_score", 0.0)) for item in self.weight_objective_history]
        if len(history_scores) >= rollback_p + 1:
            recent_declines = True
            for idx in range(rollback_p):
                prev_score = history_scores[-(idx + 2)]
                curr_score = history_scores[-(idx + 1)]
                if (prev_score - curr_score) <= rollback_drop:
                    recent_declines = False
                    break
            if recent_declines:
                previous_weights = dict(self.weight_objective_history[-2].get("weights", current_weights))
                self.evaluator.set_terminal_risk_weights(previous_weights)
                rollback_score, rollback_holdout_metrics, rollback_dual_stats, _, _ = evaluate_candidate(previous_weights)
                self.weight_objective_history[-1] = {
                    "round_index": int(self.round_index),
                    "objective_score": float(rollback_score),
                    "weights": dict(previous_weights),
                }
                info["weight_rollback_triggered"] = True
                info["weight_update_status"] = "rolled_back"
                info["weight_objective_score_after"] = float(rollback_score)
                info["holdout_metrics_after"] = rollback_holdout_metrics
                info["thresholds_after_weight_update"] = {
                    "decision_threshold": float(rollback_dual_stats["decision_threshold"]),
                    "terminal_threshold": float(rollback_dual_stats["terminal_threshold"]),
                }
                info["terminal_risk_weights"] = dict(previous_weights)
                info["weight_delta"] = {
                    key: float(previous_weights[key] - current_weights[key])
                    for key in ("packet_loss", "e2e_delay", "throughput", "ending_reward")
                }
                self.weight_update_status = "rolled_back"

        self.last_weight_adaptation_info = info
        return info

    def _calibrate_failure_threshold_v2(self) -> Dict:
        calibration_mode = str(self.args.threshold_calibration_mode).strip().lower()
        if not self.summary_records:
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_samples",
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        raw_true_labels = [self._resolve_true_failure_v2_value(record) for record in self.summary_records]
        raw_terminal_flags = [bool(record.get("terminal_hard_failure", False)) for record in self.summary_records]

        effective_indices = [idx for idx, flag in enumerate(raw_terminal_flags) if flag]

        if len(effective_indices) < max(2, int(self.args.threshold_min_support)):
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_effective_support",
                "effective_support": len(effective_indices),
                "min_support": int(self.args.threshold_min_support),
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        effective_true_labels = [raw_true_labels[idx] for idx in effective_indices]

        holdout_ratio = float(np.clip(float(self.args.threshold_calibration_holdout_ratio), 0.0, 0.49))
        holdout_count = int(np.floor(len(effective_indices) * holdout_ratio))
        train_count = len(effective_indices) - holdout_count
        if train_count < max(2, int(self.args.threshold_min_support)):
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_train_support",
                "effective_support": len(effective_indices),
                "train_support": train_count,
                "holdout_support": holdout_count,
                "min_support": int(self.args.threshold_min_support),
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        train_labels = effective_true_labels[:train_count]
        holdout_labels = effective_true_labels[train_count:]

        positive_count = sum(1 for label in train_labels if label)
        negative_count = len(train_labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "single_class_labels",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "effective_support": len(effective_indices),
                "train_support": train_count,
                "holdout_support": holdout_count,
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        mode = self.failure_decision_mode
        if mode == "single_fused_score":
            fused_scores = [self._compute_fused_score_and_logit_for_record(record)[0] for record in self.summary_records]
            effective_fused_scores = [fused_scores[idx] for idx in effective_indices]
            train_fused_scores = effective_fused_scores[:train_count]
            holdout_fused_scores = effective_fused_scores[train_count:]
            threshold_stats = self._calibrate_single_score_threshold(
                train_scores=train_fused_scores,
                train_labels=train_labels,
                holdout_scores=holdout_fused_scores,
                holdout_labels=holdout_labels,
                threshold_key="fused_threshold",
                holdout_auc_key="fused_model_holdout_auc",
            )
            fused_threshold = float(threshold_stats["fused_threshold"])
            self.last_failure_model_info["fused_threshold"] = fused_threshold
            threshold_stats["decision_threshold"] = fused_threshold
            threshold_stats["terminal_threshold"] = 0.0
            threshold_stats["threshold"] = fused_threshold
            threshold_stats["selection_stage"] = "single_fused_score"
        elif mode == "direct_failure_model":
            failure_probs = [self._compute_direct_failure_probability_for_record(record) for record in self.summary_records]
            effective_failure_probs = [failure_probs[idx] for idx in effective_indices]
            train_failure_probs = effective_failure_probs[:train_count]
            holdout_failure_probs = effective_failure_probs[train_count:]
            threshold_stats = self._calibrate_single_score_threshold(
                train_scores=train_failure_probs,
                train_labels=train_labels,
                holdout_scores=holdout_failure_probs,
                holdout_labels=holdout_labels,
                threshold_key="final_threshold",
                holdout_auc_key="direct_model_holdout_auc",
            )
            final_threshold = float(threshold_stats["final_threshold"])
            self.last_failure_model_info["final_threshold"] = final_threshold
            threshold_stats["decision_threshold"] = final_threshold
            threshold_stats["terminal_threshold"] = 0.0
            threshold_stats["threshold"] = final_threshold
            threshold_stats["selection_stage"] = "direct_failure_model"
        else:
            raise ValueError(f"Unsupported failure_decision_mode for threshold calibration: {mode}")
        decision_threshold = float(threshold_stats.get("decision_threshold", self.evaluator.v2_failure_threshold))
        terminal_threshold = float(threshold_stats.get("terminal_threshold", self.evaluator.terminal_threshold_v2))

        holdout_metrics = dict(
            threshold_stats.get(
                "holdout_metrics",
                {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0},
            )
        )
        threshold_stats["threshold"] = decision_threshold
        threshold_stats["decision_threshold"] = decision_threshold
        threshold_stats["terminal_threshold"] = terminal_threshold
        threshold_stats["status"] = "updated"
        threshold_stats["positive_count"] = positive_count
        threshold_stats["negative_count"] = negative_count
        threshold_stats["effective_support"] = len(effective_indices)
        threshold_stats["train_support"] = train_count
        threshold_stats["holdout_support"] = holdout_count
        threshold_stats["scope"] = self.args.threshold_calibration_scope
        threshold_stats["objective"] = self.args.threshold_objective
        threshold_stats["calibration_mode"] = calibration_mode
        threshold_stats["holdout_ratio"] = holdout_ratio
        threshold_stats["holdout_metrics"] = holdout_metrics
        self.threshold_update_status = "updated"
        self.last_threshold_stats = dict(threshold_stats)
        return threshold_stats

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
            theoretical_max_per_region=max(2, len(continuous_np) // max(1, self.explorer.n_clusters)),
            sc_schedule=self.coverage_sc_schedule,
        )
        seeds_c = self.explorer.generate_seed_candidates(
            region_stats=region_stats,
            feature_bounds=build_continuous_feature_bounds(CONTINUOUS_FEATURE_NAMES, self.traffic_profile),
            num_seeds_per_region=self.args.seed_per_region,
        )
        if seeds_c is None or len(seeds_c) == 0:
            fallback_count = min(max(self.args.min_scenarios_per_round, 1), len(continuous_np))
            fallback_idx = np.random.choice(len(continuous_np), size=fallback_count, replace=len(continuous_np) < fallback_count)
            seeds_c = continuous_np[fallback_idx]

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
            min_scenarios=self.args.min_scenarios_per_round,
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

    def _collect_current_summary_metrics(self) -> Dict[str, object]:
        predicted_failures_v2 = sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)))
        true_failures_v2 = sum(1 for record in self.summary_records if self._resolve_true_failure_v2_value(record))
        true_failures_v2_strict = sum(
            1 for record in self.summary_records if bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        )
        accuracy_v2 = (
            sum(
                1
                for record in self.summary_records
                if bool(record.get("system_failure_v2", False)) == self._resolve_true_failure_v2_value(record)
            )
            / len(self.summary_records)
            if self.summary_records
            else 0.0
        )
        tp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        fp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        tn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        fn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        return {
            "record_count": int(len(self.summary_records)),
            "predicted_failure_count": int(predicted_failures_v2),
            "true_failure_count": int(true_failures_v2),
            "true_failure_count_strict": int(true_failures_v2_strict),
            "failure_detection_accuracy": float(accuracy_v2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "true_failure_policy": str(self.true_failure_v2_policy),
            "failure_decision_mode": str(self.failure_decision_mode),
            "primary_score_name": str(self.last_failure_model_info.get("primary_score_name", "decision_score_v2")),
            "primary_score_holdout_auc": float(self.last_failure_model_info.get("primary_score_holdout_auc", 0.0)),
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
            "fused_threshold": float(self.last_failure_model_info.get("fused_threshold", 0.0)),
            "final_threshold": float(self.last_failure_model_info.get("final_threshold", 0.0)),
            "threshold_calibration_scope": str(self.args.threshold_calibration_scope),
            "threshold_objective_used": str(self.args.threshold_objective),
            "threshold_calibration_mode": str(self.last_threshold_stats.get("calibration_mode", self.args.threshold_calibration_mode)),
            "threshold_selection_stage": str(self.last_threshold_stats.get("selection_stage", self.failure_decision_mode)),
            "stop_reason": str(self.stop_reason),
        }

    def _simulate_post_run_offline_recompute_summary(self) -> Dict[str, object]:
        snapshot = {
            "summary_records": copy.deepcopy(self.summary_records),
            "threshold_update_status": str(self.threshold_update_status),
            "last_decision_model_info": copy.deepcopy(self.last_decision_model_info),
            "last_failure_model_info": copy.deepcopy(self.last_failure_model_info),
            "last_threshold_stats": copy.deepcopy(self.last_threshold_stats),
            "decision_formula_config": copy.deepcopy(self.evaluator.get_decision_formula_config()),
            "v2_failure_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold_v2": float(self.evaluator.terminal_threshold_v2),
            "terminal_risk_weights": copy.deepcopy(self.evaluator.get_terminal_risk_weights()),
        }
        try:
            self._fit_failure_decision_models()
            self._calibrate_failure_threshold_v2()
            self._recompute_predictions_from_thresholds()
            metrics = self._collect_current_summary_metrics()
            metrics["status"] = "ok"
            metrics["mode"] = "post_run_offline_recompute"
            return metrics
        except Exception as exc:
            return {
                "status": "failed",
                "mode": "post_run_offline_recompute",
                "reason": str(exc),
            }
        finally:
            self.summary_records = snapshot["summary_records"]
            self.threshold_update_status = snapshot["threshold_update_status"]
            self.last_decision_model_info = snapshot["last_decision_model_info"]
            self.last_failure_model_info = snapshot["last_failure_model_info"]
            self.last_threshold_stats = snapshot["last_threshold_stats"]
            cfg = snapshot["decision_formula_config"]
            self.evaluator.set_decision_formula_config(
                decision_formula_weights={
                    key: cfg.get(key)
                    for key in ("w_mean", "w_p75", "w_max", "w_slope_pos", "w_std_penalty")
                    if key in cfg
                },
                enable_decision_tail_boost=cfg.get("enable_decision_tail_boost"),
                decision_tail_gamma=cfg.get("decision_tail_gamma"),
                decision_model_type=cfg.get("decision_model_type"),
                decision_model_weights=cfg.get("decision_model_weights"),
                decision_model_bias=cfg.get("decision_model_bias"),
            )
            self.evaluator.set_v2_failure_threshold(snapshot["v2_failure_threshold"])
            self.evaluator.set_terminal_threshold_v2(snapshot["terminal_threshold_v2"])
            self.evaluator.set_terminal_risk_weights(snapshot["terminal_risk_weights"])

    def _write_final_output(self):
        output_path = self.session_dir / "output_summary.txt"
        # Step records are written as final artifacts; annotate them with the final (post-backfill) sample-level scores.
        self._annotate_step_records_with_final_sample_scores()
        predicted_failures_v2 = sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)))
        true_failures_v2 = sum(1 for record in self.summary_records if self._resolve_true_failure_v2_value(record))
        true_failures_v2_strict = sum(
            1 for record in self.summary_records if bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        )
        accuracy_v2 = (
            sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)) == self._resolve_true_failure_v2_value(record)) / len(self.summary_records)
            if self.summary_records
            else 0.0
        )

        tp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        fp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        tn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        fn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
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
            f.write(f"coverage_decomposition: {json.dumps(self.latest_coverage_metrics.get('coverage_decomposition', {}), ensure_ascii=False)}\n")
            f.write(f"predicted_failure_count: {predicted_failures_v2}\n")
            f.write(f"true_failure_count: {true_failures_v2}\n")
            f.write(f"true_failure_count_strict: {true_failures_v2_strict}\n")
            f.write(f"failure_detection_accuracy: {accuracy_v2:.6f}\n")
            f.write(
                "confusion_matrix: "
                + json.dumps({"tp": tp, "fp": fp, "tn": tn, "fn": fn}, ensure_ascii=False)
                + "\n"
            )
            f.write(f"true_failure_policy: {self.true_failure_v2_policy}\n")
            f.write(f"failure_decision_mode: {self.failure_decision_mode}\n")
            f.write(f"decision_threshold: {self.evaluator.v2_failure_threshold:.6f}\n")
            f.write(f"terminal_threshold: {self.evaluator.terminal_threshold_v2:.6f}\n")
            f.write(f"primary_score_name: {self.last_failure_model_info.get('primary_score_name', 'decision_score_v2')}\n")
            f.write(f"primary_score_holdout_auc: {float(self.last_failure_model_info.get('primary_score_holdout_auc', 0.0)):.6f}\n")
            f.write(f"fused_threshold: {float(self.last_failure_model_info.get('fused_threshold', 0.0)):.6f}\n")
            f.write(f"final_threshold: {float(self.last_failure_model_info.get('final_threshold', 0.0)):.6f}\n")
            f.write(f"decision_formula_config: {json.dumps(self.evaluator.get_decision_formula_config(), ensure_ascii=False)}\n")
            f.write(f"decision_model_status: {self.last_decision_model_info.get('decision_model_status', 'disabled')}\n")
            f.write(
                f"decision_model_holdout_record_count: {int(self.last_decision_model_info.get('decision_model_holdout_record_count', 0))}\n"
            )
            f.write(
                f"decision_model_holdout_auc: {float(self.last_decision_model_info.get('decision_model_holdout_auc', 0.0)):.6f}\n"
            )
            f.write(
                f"decision_model_holdout_accuracy: {float(self.last_decision_model_info.get('decision_model_holdout_accuracy', 0.0)):.6f}\n"
            )
            f.write(
                f"decision_model_config: {json.dumps(self.last_decision_model_info.get('decision_model_config', {}), ensure_ascii=False)}\n"
            )
            f.write(f"terminal_risk_weights: {json.dumps(self.evaluator.get_terminal_risk_weights(), ensure_ascii=False)}\n")
            f.write(f"threshold_calibration_scope: {self.args.threshold_calibration_scope}\n")
            f.write(f"threshold_objective_used: {self.args.threshold_objective}\n")
            f.write(f"threshold_calibration_mode: {self.last_threshold_stats.get('calibration_mode', self.args.threshold_calibration_mode)}\n")
            f.write(f"threshold_selection_stage: {self.last_threshold_stats.get('selection_stage', 'legacy')}\n")
            f.write(f"threshold_update_status: {self.threshold_update_status}\n")
            f.write(f"stop_reason: {self.stop_reason}\n")
            if self.post_run_offline_recompute_summary is not None:
                f.write("integrated_summary_enabled: true\n")
                f.write(
                    f"online_raw_summary: {json.dumps(self._collect_current_summary_metrics(), ensure_ascii=False)}\n"
                )
                f.write(
                    f"online_final_recompute_summary: {json.dumps(self.post_run_offline_recompute_summary, ensure_ascii=False)}\n"
                )
            f.write("\n")
            for row in self.step_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self):
        if bool(self.args.offline_recompute_only):
            self._offline_recompute_from_existing_results()
            return

        if self.finished:
            self._write_final_output()
            return

        self.stop_reason = "running"
        while True:
            current_scenarios = list(self.next_round_scenarios)
            if not current_scenarios:
                self.stop_reason = "no_current_scenarios"
                break

            round_summary_records, round_step_records = self._collect_round_results(self.round_index, current_scenarios)
            self._incremental_train_and_evaluate_coverage(self.round_index, round_summary_records, round_step_records)

            if self.round_index > 0:
                self.generated_scenario_count += len(current_scenarios)

            if bool(self.args.stop_on_coverage_target):
                coverage_upper_bound = float(self.latest_coverage_metrics.get("coverage_upper_bound", 0.0))
                total_samples = int(self.latest_coverage_metrics.get("total_samples", 0))
                if (
                    total_samples >= int(max(0, self.args.min_samples_for_coverage_stop))
                    and coverage_upper_bound >= float(self.args.coverage_target)
                ):
                    self.finished = True
                    self.stop_reason = "coverage_target_reached"
                    self.next_round_scenarios = []
                    self._save_state()
                    break

            if self.generated_scenario_count >= self.args.generated_limit:
                self.finished = True
                self.stop_reason = "generated_limit_reached"
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
                self.stop_reason = "next_round_empty"
                break

        self.finished = True
        if self.stop_reason == "running":
            self.stop_reason = "loop_completed"
        self.post_run_offline_recompute_summary = None
        if bool(self.args.post_run_offline_recompute):
            self.post_run_offline_recompute_summary = self._simulate_post_run_offline_recompute_summary()
        self._save_state()
        self._write_final_output()


def main():
    args = parse_args()
    workflow = ClosedLoopFailureSimulation(args)
    workflow.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
