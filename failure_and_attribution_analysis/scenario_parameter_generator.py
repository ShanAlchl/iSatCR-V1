import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .deep_ensemble_network import DeepEnsembleNetwork
from .parameter_interfaces import (
    CONTINUOUS_FEATURE_NAMES,
    DISCRETE_FEATURE_NAMES,
    FailEnv,
)


TRAFFIC_PROFILE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "low": {
        "PoissonRate": 45.0,
        "MeanIntervalTime": 15.0,
        "PacketGenerationInterval": 4.0,
    },
    "medium": {
        "PoissonRate": 30.0,
        "MeanIntervalTime": 30.0,
        "PacketGenerationInterval": 2.0,
    },
}

TRAFFIC_PROFILE_LINKED_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "low": {
        "PoissonRate": (40.000001, 46.0),
        "MeanIntervalTime": (14.000001, 22.0),
        "PacketGenerationInterval": (3.000001, 4.0),
    },
    "medium": {
        "PoissonRate": (29.000001, 39.999999),
        "MeanIntervalTime": (23.000001, 31.0),
        "PacketGenerationInterval": (2.0, 3.0),
    },
}

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "DegradedEdgeRatio": (0.0, 1.0),
    "EdgeDisconnectRatio": (0.0, 1.0),
    "EdgeBandwidthMeanDecreaseRatio": (0.0, 1.0),
    "EdgeBandwidthDecreaseStd": (0.0, 0.2),
    "PoissonRate": (29.0, 46.0),
    "MeanIntervalTime": (14.0, 31.0),
    "PacketGenerationInterval": (2.0, 4.0),
    "PacketSizeMean": (1.0e8, 2.0e9),
    "PacketSizeStd": (0.0, 5.0e8),
}

EDGE_DEGRADATION_FEATURE_NAMES = (
    "DegradedEdgeRatio",
    "EdgeDisconnectRatio",
    "EdgeBandwidthMeanDecreaseRatio",
)

INTEGER_FIELDS = {
    "ConstellationConfig",
    "PacketSizeMean",
    "PacketSizeStd",
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
}


def _resolve_category_sizes(
    num_categories: Union[int, Sequence[int]],
    num_discrete_features: int,
) -> List[int]:
    if isinstance(num_categories, int):
        return [int(num_categories)] * int(num_discrete_features)

    category_sizes = [int(size) for size in num_categories]
    if len(category_sizes) != int(num_discrete_features):
        raise ValueError("num_categories length must match the number of discrete features")
    return category_sizes


def build_continuous_feature_bounds(
    continuous_feature_names: Sequence[str] = CONTINUOUS_FEATURE_NAMES,
    traffic_profile: Optional[str] = None,
) -> List[Tuple[float, float]]:
    normalized_profile = str(traffic_profile).strip().lower() if traffic_profile else None
    bounds = []
    for feature_name in continuous_feature_names:
        if normalized_profile in TRAFFIC_PROFILE_LINKED_BOUNDS and feature_name in TRAFFIC_PROFILE_LINKED_BOUNDS[normalized_profile]:
            bounds.append(TRAFFIC_PROFILE_LINKED_BOUNDS[normalized_profile][feature_name])
        else:
            bounds.append(FEATURE_BOUNDS[feature_name])
    return bounds


class FeatureSimilarityNetwork(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        num_categories: Union[int, Sequence[int]],
        num_discrete_features: int = 1,
        embedding_dim: int = 4,
        out_dim: int = 32,
    ):
        super().__init__()
        category_sizes = _resolve_category_sizes(num_categories, num_discrete_features)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim) for size in category_sizes]
        )
        self.num_discrete_features = len(category_sizes)

        input_dim = num_continuous + embedding_dim * self.num_discrete_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, continuous_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        if discrete_x.dim() == 1:
            discrete_x = discrete_x.unsqueeze(1)
        if discrete_x.shape[1] != self.num_discrete_features:
            raise ValueError("discrete_x feature count does not match FeatureSimilarityNetwork configuration")

        embedded_parts = [
            embedding(discrete_x[:, idx].long())
            for idx, embedding in enumerate(self.embeddings)
        ]
        embedded_x = torch.cat(embedded_parts, dim=1)
        x = torch.cat([continuous_x, embedded_x], dim=1)
        features = self.net(x)
        return F.normalize(features, p=2, dim=1)


class AdaptiveCMAES:
    def __init__(
        self,
        continuous_seeds: np.ndarray,
        discrete_seeds: np.ndarray,
        num_categories: Union[int, Sequence[int]],
    ):
        if discrete_seeds.ndim == 1:
            discrete_seeds = discrete_seeds.reshape(-1, 1)

        self.dimension = continuous_seeds.shape[1]
        self.num_discrete_features = discrete_seeds.shape[1]
        self.category_sizes = _resolve_category_sizes(num_categories, self.num_discrete_features)

        self.mean = np.mean(continuous_seeds, axis=0)
        if len(continuous_seeds) > 1:
            covariance = np.cov(continuous_seeds, rowvar=False)
            if np.ndim(covariance) == 0:
                covariance = np.array([[float(covariance)]], dtype=float)
            self.covariance = covariance
        else:
            self.covariance = np.eye(self.dimension, dtype=float) * 0.1
        self.covariance = self.covariance + np.eye(self.dimension, dtype=float) * 1e-6

        self.discrete_probabilities: List[np.ndarray] = []
        for feature_idx, category_size in enumerate(self.category_sizes):
            counts = np.bincount(discrete_seeds[:, feature_idx].astype(int), minlength=category_size)
            probabilities = counts / (counts.sum() + 1e-8)
            probabilities[probabilities == 0] = 0.05
            probabilities = probabilities / probabilities.sum()
            self.discrete_probabilities.append(probabilities)

    def generate_candidates(self, num_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
        L = np.linalg.cholesky(self.covariance)
        standard_normal = np.random.randn(num_candidates, self.dimension)
        continuous_candidates = self.mean + standard_normal @ L.T

        discrete_columns = []
        for category_size, probabilities in zip(self.category_sizes, self.discrete_probabilities):
            discrete_columns.append(
                np.random.choice(
                    a=category_size,
                    size=(num_candidates, 1),
                    p=probabilities,
                )
            )
        discrete_candidates = np.hstack(discrete_columns)
        return continuous_candidates, discrete_candidates

    def update_distribution(self, valid_continuous: np.ndarray, cv_weights: np.ndarray):
        if len(valid_continuous) == 0:
            return

        cv_weights = cv_weights / (cv_weights.sum() + 1e-8)
        new_mean = np.average(valid_continuous, axis=0, weights=cv_weights)

        diffs = valid_continuous - new_mean
        weighted_cov = np.zeros((self.dimension, self.dimension), dtype=float)
        for idx in range(len(valid_continuous)):
            diff_col = diffs[idx].reshape(-1, 1)
            weighted_cov += cv_weights[idx] * (diff_col @ diff_col.T)

        learning_rate = 0.3
        self.covariance = (1 - learning_rate) * self.covariance + learning_rate * weighted_cov
        self.covariance += np.eye(self.dimension, dtype=float) * 1e-6
        self.mean = (1 - learning_rate) * self.mean + learning_rate * new_mean


class ScenarioParameterGenerator:
    def __init__(
        self,
        ensemble_net: DeepEnsembleNetwork,
        feature_net: FeatureSimilarityNetwork,
        similarity_threshold: float = 0.97,
        continuous_feature_names: Optional[Sequence[str]] = None,
        discrete_feature_names: Optional[Sequence[str]] = None,
        fixed_constellation_config: Optional[int] = None,
    ):
        self.ensemble_net = ensemble_net
        self.feature_net = feature_net
        self.similarity_threshold = similarity_threshold
        self.continuous_feature_names = tuple(continuous_feature_names or CONTINUOUS_FEATURE_NAMES)
        self.discrete_feature_names = tuple(discrete_feature_names or DISCRETE_FEATURE_NAMES)
        self.fixed_constellation_config = (
            int(fixed_constellation_config) if fixed_constellation_config is not None else None
        )
        self.explored_features_pool: Optional[torch.Tensor] = None
        self.last_generated_similarity_scores = np.array([], dtype=float)
        self.last_generated_prediction_scores = np.array([], dtype=float)
        self.last_generated_prediction_cvs = np.array([], dtype=float)

    @property
    def _ensemble_device(self) -> torch.device:
        return next(self.ensemble_net.parameters()).device

    @property
    def _feature_device(self) -> torch.device:
        return next(self.feature_net.parameters()).device

    def add_explored_history(self, continuous_x: torch.Tensor, discrete_x: torch.Tensor):
        continuous_x = continuous_x.to(self._feature_device)
        discrete_x = discrete_x.to(self._feature_device)
        with torch.no_grad():
            features = self.feature_net(continuous_x, discrete_x)
        if self.explored_features_pool is None:
            self.explored_features_pool = features
        else:
            self.explored_features_pool = torch.cat([self.explored_features_pool, features], dim=0)

    def export_state(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "explored_features_pool": self.explored_features_pool.detach().cpu()
            if self.explored_features_pool is not None
            else None,
        }

    def load_state(self, state: Optional[Dict[str, Optional[torch.Tensor]]]):
        if not state:
            return
        explored_pool = state.get("explored_features_pool")
        self.explored_features_pool = explored_pool.clone().to(self._feature_device) if explored_pool is not None else None

    def _normalize_discrete_seed_matrix(self, discrete_values: np.ndarray) -> np.ndarray:
        discrete_array = np.array(discrete_values, dtype=int, copy=True)
        if discrete_array.size == 0:
            return np.empty((0, len(self.discrete_feature_names)), dtype=int)
        if discrete_array.ndim == 1:
            discrete_array = discrete_array.reshape(-1, len(self.discrete_feature_names))
        if discrete_array.shape[1] != len(self.discrete_feature_names):
            raise ValueError("discrete seed feature count does not match ScenarioParameterGenerator configuration")
        return self._clip_discrete_matrix(discrete_array)

    def _resample_discrete_seed_rows(self, discrete_values: np.ndarray, target_count: int) -> np.ndarray:
        normalized_discrete = self._normalize_discrete_seed_matrix(discrete_values)
        if target_count <= 0 or len(normalized_discrete) == 0:
            return np.empty((0, len(self.discrete_feature_names)), dtype=int)
        if len(normalized_discrete) == target_count:
            return normalized_discrete

        replace = len(normalized_discrete) < target_count
        sampled_indices = np.random.choice(len(normalized_discrete), size=target_count, replace=replace)
        return normalized_discrete[sampled_indices]

    def generate_new_scenarios(
        self,
        seed_continuous: np.ndarray,
        seed_discrete: np.ndarray,
        num_categories: Union[int, Sequence[int]],
        target_num_scenarios: int = 100,
        cv_threshold: float = 0.1,
        traffic_profile: Optional[str] = None,
        return_similarity: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        seed_continuous = np.array(seed_continuous, dtype=float, copy=True)
        if seed_continuous.size == 0:
            empty_continuous = np.empty((0, len(self.continuous_feature_names)), dtype=float)
            empty_discrete = np.empty((0, len(self.discrete_feature_names)), dtype=int)
            empty_similarity = np.array([], dtype=float)
            if return_similarity:
                return empty_continuous, empty_discrete, empty_similarity
            return empty_continuous, empty_discrete

        seed_discrete = self._resample_discrete_seed_rows(seed_discrete, len(seed_continuous))
        if len(seed_discrete) == 0:
            empty_continuous = np.empty((0, len(self.continuous_feature_names)), dtype=float)
            empty_discrete = np.empty((0, len(self.discrete_feature_names)), dtype=int)
            empty_similarity = np.array([], dtype=float)
            if return_similarity:
                return empty_continuous, empty_discrete, empty_similarity
            return empty_continuous, empty_discrete

        cma = AdaptiveCMAES(seed_continuous, seed_discrete, num_categories)

        final_continuous_list: List[np.ndarray] = []
        final_discrete_list: List[np.ndarray] = []
        final_similarity_list: List[float] = []
        final_score_list: List[float] = []
        final_cv_list: List[float] = []

        max_iterations = 10
        for _ in range(max_iterations):
            if len(final_continuous_list) >= target_num_scenarios:
                break

            gen_c_np, gen_d_np = cma.generate_candidates(max(target_num_scenarios * 2, 8))
            gen_c_np = self._apply_traffic_profile_constraints(gen_c_np, traffic_profile)
            gen_d_np = self._clip_discrete_matrix(gen_d_np)

            gen_c_t = torch.tensor(gen_c_np, dtype=torch.float32, device=self._ensemble_device)
            gen_d_t = torch.tensor(gen_d_np, dtype=torch.long, device=self._ensemble_device)

            with torch.no_grad():
                scores, cvs = self.ensemble_net(gen_c_t, gen_d_t)

            scores_np = np.atleast_1d(scores.squeeze(-1).detach().cpu().numpy())
            cvs_np = np.atleast_1d(cvs.squeeze(-1).detach().cpu().numpy())
            selection_mask = (scores_np > 0.3) & (scores_np < 0.7) & (cvs_np >= cv_threshold)
            if not np.any(selection_mask):
                selection_mask = np.argsort(cvs_np)[-min(len(cvs_np), target_num_scenarios):]
                bool_mask = np.zeros(len(cvs_np), dtype=bool)
                bool_mask[selection_mask] = True
                selection_mask = bool_mask

            valid_c_np = gen_c_np[selection_mask]
            valid_d_np = gen_d_np[selection_mask]
            valid_score_np = scores_np[selection_mask]
            valid_cv_np = cvs_np[selection_mask]

            if len(valid_c_np) == 0:
                continue

            cma.update_distribution(valid_c_np, np.maximum(valid_cv_np, 1e-6))

            unique_c_np, unique_d_np, unique_sim_np, unique_features, unique_mask_np = self._filter_by_similarity(valid_c_np, valid_d_np)
            if len(unique_c_np) == 0:
                continue

            unique_score_np = valid_score_np[unique_mask_np]
            unique_cv_np = valid_cv_np[unique_mask_np]

            final_continuous_list.extend(list(unique_c_np))
            final_discrete_list.extend(list(unique_d_np))
            final_similarity_list.extend([float(value) for value in unique_sim_np])
            final_score_list.extend([float(value) for value in unique_score_np])
            final_cv_list.extend([float(value) for value in unique_cv_np])

            if self.explored_features_pool is None:
                self.explored_features_pool = unique_features
            else:
                self.explored_features_pool = torch.cat([self.explored_features_pool, unique_features], dim=0)

        if not final_continuous_list:
            fallback_continuous = self._apply_traffic_profile_constraints(seed_continuous, traffic_profile)
            fallback_discrete = self._resample_discrete_seed_rows(seed_discrete, len(fallback_continuous))
            fallback_c, fallback_d, fallback_sim, _, _ = self._filter_by_similarity(fallback_continuous, fallback_discrete)
            final_continuous_list = list(fallback_c[:target_num_scenarios])
            final_discrete_list = list(fallback_d[:target_num_scenarios])
            final_similarity_list = [float(value) for value in fallback_sim[:target_num_scenarios]]
            final_score_list = [0.0] * len(final_continuous_list)
            final_cv_list = [0.0] * len(final_continuous_list)

        final_c_out = np.array(final_continuous_list[:target_num_scenarios], dtype=float)
        final_d_out = np.array(final_discrete_list[:target_num_scenarios], dtype=int)
        self.last_generated_similarity_scores = np.array(final_similarity_list[:target_num_scenarios], dtype=float)
        self.last_generated_prediction_scores = np.array(final_score_list[:target_num_scenarios], dtype=float)
        self.last_generated_prediction_cvs = np.array(final_cv_list[:target_num_scenarios], dtype=float)

        if return_similarity:
            return final_c_out, final_d_out, self.last_generated_similarity_scores
        return final_c_out, final_d_out

    def generate_fail_env_list(
        self,
        seed_continuous: np.ndarray,
        seed_discrete: np.ndarray,
        num_categories: Union[int, Sequence[int]],
        target_num_scenarios: int = 100,
        base_env: dict = None,
        traffic_profile: str = "low",
        cv_threshold: float = 0.1,
        return_similarity: bool = False,
    ) -> Union[List[FailEnv], Tuple[List[FailEnv], np.ndarray]]:
        result = self.generate_new_scenarios(
            seed_continuous=seed_continuous,
            seed_discrete=seed_discrete,
            num_categories=num_categories,
            target_num_scenarios=target_num_scenarios,
            cv_threshold=cv_threshold,
            traffic_profile=traffic_profile,
            return_similarity=return_similarity,
        )
        if return_similarity:
            final_c, final_d, similarities = result
        else:
            final_c, final_d = result
            similarities = self.last_generated_similarity_scores

        normalized_profile = str(traffic_profile).strip().lower() if traffic_profile else ""
        defaults = dict(
            DegradedEdgeRatio=0.0,
            EdgeDisconnectRatio=0.0,
            EdgeBandwidthMeanDecreaseRatio=0.0,
            EdgeBandwidthDecreaseStd=0.0,
            PacketSizeMean=400000000,
            PacketSizeStd=115470000,
            ConstellationConfig=0,
            StateObservationAttack_level=0,
            ActionAttack_level=0,
            StateTransferAttack_level=0,
            RewardAttack_level=0,
            ExperiencePoolAttack_level=0,
            ModelTampAttack_level=0,
        )
        if normalized_profile in TRAFFIC_PROFILE_DEFAULTS:
            defaults.update(TRAFFIC_PROFILE_DEFAULTS[normalized_profile])
        if base_env:
            defaults.update(base_env)

        fail_env_list: List[FailEnv] = []
        for idx in range(len(final_c)):
            fail_env_list.append(
                self._decode_fail_env(
                    continuous_values=final_c[idx],
                    discrete_values=final_d[idx],
                    defaults=defaults,
                    traffic_profile=normalized_profile,
                )
            )

        if return_similarity:
            return fail_env_list, similarities
        return fail_env_list

    def _filter_by_similarity(
        self,
        continuous_values: np.ndarray,
        discrete_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
        if len(continuous_values) == 0:
            empty_features = torch.empty((0, 0), device=self._feature_device)
            return (
                np.empty((0, len(self.continuous_feature_names)), dtype=float),
                np.empty((0, len(self.discrete_feature_names)), dtype=int),
                np.array([], dtype=float),
                empty_features,
                np.array([], dtype=bool),
            )

        if len(continuous_values) != len(discrete_values):
            raise ValueError("continuous_values and discrete_values must have the same number of rows before similarity filtering")

        continuous_tensor = torch.tensor(continuous_values, dtype=torch.float32, device=self._feature_device)
        discrete_tensor = torch.tensor(discrete_values, dtype=torch.long, device=self._feature_device)
        with torch.no_grad():
            candidate_features = self.feature_net(continuous_tensor, discrete_tensor)

        if self.explored_features_pool is None or self.explored_features_pool.numel() == 0:
            return (
                continuous_values,
                discrete_values,
                np.zeros(len(continuous_values), dtype=float),
                candidate_features,
                np.ones(len(continuous_values), dtype=bool),
            )

        sim_matrix = torch.matmul(candidate_features, self.explored_features_pool.T)
        max_sims, _ = sim_matrix.max(dim=1)
        unique_mask = max_sims < self.similarity_threshold
        unique_mask_np = unique_mask.detach().cpu().numpy()
        return (
            continuous_values[unique_mask_np],
            discrete_values[unique_mask_np],
            max_sims.detach().cpu().numpy()[unique_mask_np],
            candidate_features[unique_mask],
            unique_mask_np,
        )

    def _apply_traffic_profile_constraints(
        self,
        continuous_values: np.ndarray,
        traffic_profile: Optional[str],
    ) -> np.ndarray:
        constrained = np.array(continuous_values, dtype=float, copy=True)
        normalized_profile = str(traffic_profile).strip().lower() if traffic_profile else None

        for feature_idx, feature_name in enumerate(self.continuous_feature_names):
            lower, upper = FEATURE_BOUNDS[feature_name]
            constrained[:, feature_idx] = np.clip(constrained[:, feature_idx], lower, upper)

        if normalized_profile not in TRAFFIC_PROFILE_LINKED_BOUNDS:
            return self._apply_edge_degradation_constraints(constrained)

        profile_bounds = TRAFFIC_PROFILE_LINKED_BOUNDS[normalized_profile]
        for feature_name, (lower, upper) in profile_bounds.items():
            feature_idx = self.continuous_feature_names.index(feature_name)
            constrained[:, feature_idx] = np.clip(constrained[:, feature_idx], lower, upper)

        return self._apply_edge_degradation_constraints(constrained)

    def _apply_edge_degradation_constraints(self, continuous_values: np.ndarray) -> np.ndarray:
        constrained = np.array(continuous_values, dtype=float, copy=True)
        if constrained.size == 0:
            return constrained

        degraded_idx = self.continuous_feature_names.index("DegradedEdgeRatio")
        disconnect_idx = self.continuous_feature_names.index("EdgeDisconnectRatio")
        mean_decrease_idx = self.continuous_feature_names.index("EdgeBandwidthMeanDecreaseRatio")

        for row_idx in range(constrained.shape[0]):
            degraded_ratio = float(constrained[row_idx, degraded_idx])
            disconnect_ratio = float(constrained[row_idx, disconnect_idx])
            mean_decrease_ratio = float(constrained[row_idx, mean_decrease_idx])

            if degraded_ratio <= 0.0:
                disconnect_ratio = 0.0
                mean_decrease_ratio = 0.0
            else:
                # Fully disconnected edges contribute a drop ratio of 1.0, so their fraction
                # cannot exceed the target mean bandwidth decrease over the degraded edge set.
                disconnect_ratio = min(disconnect_ratio, mean_decrease_ratio)
                if disconnect_ratio >= 1.0 - 1e-12:
                    disconnect_ratio = 1.0
                    mean_decrease_ratio = 1.0

            constrained[row_idx, degraded_idx] = degraded_ratio
            constrained[row_idx, disconnect_idx] = disconnect_ratio
            constrained[row_idx, mean_decrease_idx] = mean_decrease_ratio

        return constrained

    def _clip_discrete_matrix(self, discrete_values: np.ndarray) -> np.ndarray:
        clipped = np.array(discrete_values, dtype=int, copy=True)
        if clipped.ndim == 1:
            clipped = clipped.reshape(-1, len(self.discrete_feature_names))

        for feature_idx, feature_name in enumerate(self.discrete_feature_names):
            if feature_name == "ConstellationConfig":
                if self.fixed_constellation_config is not None:
                    clipped[:, feature_idx] = int(self.fixed_constellation_config)
                else:
                    clipped[:, feature_idx] = np.clip(clipped[:, feature_idx], 0, 4)
            else:
                clipped[:, feature_idx] = np.clip(clipped[:, feature_idx], 0, 4)
        return clipped.astype(int)

    def _decode_fail_env(
        self,
        continuous_values: np.ndarray,
        discrete_values: np.ndarray,
        defaults: Dict,
        traffic_profile: Optional[str],
    ) -> FailEnv:
        env_data = dict(defaults)

        for feature_name, value in zip(self.continuous_feature_names, continuous_values):
            lower, upper = FEATURE_BOUNDS[feature_name]
            clipped_value = float(np.clip(value, lower, upper))
            if feature_name in INTEGER_FIELDS:
                env_data[feature_name] = int(round(clipped_value))
            else:
                env_data[feature_name] = clipped_value

        for feature_name, value in zip(self.discrete_feature_names, discrete_values):
            env_data[feature_name] = int(np.clip(int(value), 0, 4))

        if self.fixed_constellation_config is not None:
            env_data["ConstellationConfig"] = int(self.fixed_constellation_config)

        normalized_profile = str(traffic_profile).strip().lower() if traffic_profile else None
        if normalized_profile in TRAFFIC_PROFILE_LINKED_BOUNDS:
            for feature_name, (lower, upper) in TRAFFIC_PROFILE_LINKED_BOUNDS[normalized_profile].items():
                env_data[feature_name] = float(np.clip(env_data[feature_name], lower, upper))

        edge_degradation_values = np.array(
            [[float(env_data[feature_name]) for feature_name in self.continuous_feature_names]],
            dtype=float,
        )
        edge_degradation_values = self._apply_edge_degradation_constraints(edge_degradation_values)
        for feature_idx, feature_name in enumerate(self.continuous_feature_names):
            env_data[feature_name] = float(edge_degradation_values[0, feature_idx])

        env_data["PacketSizeMean"] = int(round(env_data["PacketSizeMean"]))
        env_data["PacketSizeStd"] = int(round(env_data["PacketSizeStd"]))

        return FailEnv(
            ConstellationConfig=int(env_data["ConstellationConfig"]),
            DegradedEdgeRatio=float(env_data["DegradedEdgeRatio"]),
            EdgeDisconnectRatio=float(env_data["EdgeDisconnectRatio"]),
            EdgeBandwidthMeanDecreaseRatio=float(env_data["EdgeBandwidthMeanDecreaseRatio"]),
            EdgeBandwidthDecreaseStd=float(env_data["EdgeBandwidthDecreaseStd"]),
            PoissonRate=float(env_data["PoissonRate"]),
            MeanIntervalTime=float(env_data["MeanIntervalTime"]),
            PacketGenerationInterval=float(env_data["PacketGenerationInterval"]),
            PacketSizeMean=int(env_data["PacketSizeMean"]),
            PacketSizeStd=int(env_data["PacketSizeStd"]),
            StateObservationAttack_level=int(env_data["StateObservationAttack_level"]),
            ActionAttack_level=int(env_data["ActionAttack_level"]),
            StateTransferAttack_level=int(env_data["StateTransferAttack_level"]),
            RewardAttack_level=int(env_data["RewardAttack_level"]),
            ExperiencePoolAttack_level=int(env_data["ExperiencePoolAttack_level"]),
            ModelTampAttack_level=int(env_data["ModelTampAttack_level"]),
        )
