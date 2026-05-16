import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta, gaussian_kde
from sklearn.cluster import KMeans

try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
    warnings.warn("scipy >= 1.7.0 is required for LatinHypercube. LHS features won't work.")


STATUS_NEEDS_EXPLORATION = "needs_exploration"
STATUS_BOUNDARY_IDENTIFIED = "boundary_identified"
STATUS_STABLE = "stable"


class FailureBoundaryExplorer:
    """Cluster-based high-dimensional failure-boundary explorer."""

    def __init__(self, n_clusters: int = 5, rau_threshold: float = 0.1, sc_threshold: float = 0.7):
        self.n_clusters = n_clusters
        self.rau_threshold = rau_threshold
        self.sc_threshold = sc_threshold
        self.failure_cloud_points: List[Dict] = []

    @staticmethod
    def _calculate_rau(cv_values: np.ndarray) -> float:
        if len(cv_values) == 0:
            return 0.0
        return float(np.mean(cv_values))

    @staticmethod
    def _calculate_sc(num_samples: int, theoretical_max: int) -> float:
        if theoretical_max <= 0:
            return 1.0
        return float(num_samples / theoretical_max)

    @staticmethod
    def _clopper_pearson_lower_bound(successes: int, total: int, confidence: float = 0.95) -> float:
        if total <= 0 or successes <= 0:
            return 0.0
        if successes >= total:
            return 1.0
        alpha = 1.0 - confidence
        return float(beta.ppf(alpha, successes, total - successes + 1))

    @staticmethod
    def _clopper_pearson_upper_bound(successes: int, total: int, confidence: float = 0.95) -> float:
        if total <= 0:
            return 0.0
        if successes >= total:
            return 1.0
        return float(beta.ppf(confidence, successes + 1, total - successes))

    def update_failure_cloud(
        self,
        features: np.ndarray,
        predicted_scores: np.ndarray,
        predicted_uncertainties: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        points: List[Dict] = []
        metadata = metadata or [{} for _ in range(len(features))]
        for idx in range(len(features)):
            point = {
                "feature": np.asarray(features[idx], dtype=float).tolist(),
                "predicted_failure_score": float(predicted_scores[idx]),
                "predicted_uncertainty": float(predicted_uncertainties[idx]),
            }
            point.update(metadata[idx] if idx < len(metadata) else {})
            points.append(point)

        self.failure_cloud_points.extend(points)
        return points

    def _resolve_sc_threshold(self, total_samples: int, sc_schedule: Optional[Sequence[Tuple[int, float]]]) -> float:
        if not sc_schedule:
            return float(self.sc_threshold)
        ordered = sorted([(int(cap), float(th)) for cap, th in sc_schedule], key=lambda item: item[0])
        for cap, threshold in ordered:
            if total_samples <= cap:
                return threshold
        return ordered[-1][1]

    def partition_and_evaluate(
        self,
        features: np.ndarray,
        failure_scores: np.ndarray,
        cv_values: np.ndarray,
        theoretical_max_per_region: Optional[int] = None,
        sc_threshold: Optional[float] = None,
        sc_schedule: Optional[Sequence[Tuple[int, float]]] = None,
    ) -> List[Dict]:
        if len(features) < self.n_clusters:
            raise ValueError("Not enough samples for KMeans clustering")

        total_samples = len(features)
        effective_sc_threshold = (
            float(sc_threshold) if sc_threshold is not None else self._resolve_sc_threshold(total_samples, sc_schedule)
        )

        adaptive_theoretical_max = max(
            2,
            int(np.ceil((total_samples / max(1, self.n_clusters)) * 1.5)),
        )
        if theoretical_max_per_region is None:
            base_theoretical_max = adaptive_theoretical_max
        else:
            base_theoretical_max = max(2, min(int(theoretical_max_per_region), adaptive_theoretical_max))

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        region_stats: List[Dict] = []
        for cluster_id in range(self.n_clusters):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) == 0:
                continue

            region_features = features[idx]
            region_scores = failure_scores[idx]
            region_cvs = cv_values[idx]

            rau = self._calculate_rau(region_cvs)
            region_theoretical_max = max(2, min(base_theoretical_max, int(np.ceil(max(1, len(idx)) * 2.0))))
            sc = self._calculate_sc(len(idx), region_theoretical_max)

            small_sample_phase = total_samples < max(1, self.n_clusters) * 3
            status = STATUS_STABLE
            if small_sample_phase:
                if rau > self.rau_threshold * 1.1 and sc < effective_sc_threshold * 0.8:
                    status = STATUS_NEEDS_EXPLORATION
            elif rau > self.rau_threshold or sc < effective_sc_threshold:
                status = STATUS_NEEDS_EXPLORATION

            if status != STATUS_NEEDS_EXPLORATION:
                avg_score = float(np.mean(region_scores))
                if 0.1 < avg_score < 0.9:
                    status = STATUS_BOUNDARY_IDENTIFIED

            region_stats.append(
                {
                    "cluster_id": cluster_id,
                    "center": kmeans.cluster_centers_[cluster_id],
                    "points_idx": idx,
                    "features": region_features,
                    "scores": region_scores,
                    "cvs": region_cvs,
                    "sample_count": int(len(idx)),
                    "sample_fraction": float(len(idx) / total_samples),
                    "RAU": rau,
                    "SC": sc,
                    "effective_sc_threshold": effective_sc_threshold,
                    "theoretical_max_per_region_effective": region_theoretical_max,
                    "status": status,
                }
            )

        return region_stats

    def compute_coverage_metrics(
        self,
        region_stats: List[Dict],
        confidence: float = 0.95,
        target_coverage: float = 0.90,
    ) -> Dict:
        total = int(sum(r.get("sample_count", 0) for r in region_stats))
        covered = int(
            sum(
                r.get("sample_count", 0)
                for r in region_stats
                if r.get("status") != STATUS_NEEDS_EXPLORATION
            )
        )

        point_estimate = float(covered / total) if total > 0 else 0.0
        lower_bound = self._clopper_pearson_lower_bound(covered, total, confidence=confidence)
        upper_bound = self._clopper_pearson_upper_bound(covered, total, confidence=confidence)

        rau_covered = int(
            sum(
                r.get("sample_count", 0)
                for r in region_stats
                if r.get("RAU", 1.0) <= self.rau_threshold
            )
        )
        sc_covered = int(
            sum(
                r.get("sample_count", 0)
                for r in region_stats
                if r.get("SC", 0.0) >= r.get("effective_sc_threshold", self.sc_threshold)
            )
        )

        rau_covered_ratio = float(rau_covered / total) if total > 0 else 0.0
        sc_covered_ratio = float(sc_covered / total) if total > 0 else 0.0
        effective_coverage = float((point_estimate + rau_covered_ratio + sc_covered_ratio) / 3.0)
        coverage_decomposition = {
            "RAU_covered_ratio": rau_covered_ratio,
            "SC_covered_ratio": sc_covered_ratio,
            "effective_coverage": effective_coverage,
        }

        return {
            "total_samples": total,
            "covered_samples": covered,
            "coverage_point_estimate": point_estimate,
            "coverage_lower_bound": lower_bound,
            "coverage_upper_bound": upper_bound,
            "confidence": confidence,
            "target_coverage": target_coverage,
            "target_achieved": bool(lower_bound >= target_coverage),
            "coverage_decomposition": coverage_decomposition,
            "RAU_covered_ratio": rau_covered_ratio,
            "SC_covered_ratio": sc_covered_ratio,
            "effective_coverage": effective_coverage,
        }

    def generate_seed_candidates(
        self,
        region_stats: List[Dict],
        feature_bounds: List[Tuple[float, float]],
        num_seeds_per_region: int = 20,
    ) -> np.ndarray:
        if qmc is None:
            raise RuntimeError("qmc (scipy.stats.qmc) is unavailable. Please upgrade scipy.")

        all_seeds: List[np.ndarray] = []
        num_features = len(feature_bounds)
        lower_bounds = np.array([b[0] for b in feature_bounds])
        upper_bounds = np.array([b[1] for b in feature_bounds])

        candidate_regions = [region for region in region_stats if region.get("status") == STATUS_NEEDS_EXPLORATION]
        if not candidate_regions and region_stats:
            sorted_regions = sorted(
                region_stats,
                key=lambda region: (region.get("RAU", 0.0), -region.get("SC", 1.0)),
                reverse=True,
            )
            candidate_regions = sorted_regions[:1]

        for region in candidate_regions:
            cvs = region["cvs"]
            features = region["features"]
            scores = region["scores"]

            top_n = int(max(1, len(cvs) * 0.3))
            top_idx = np.argsort(cvs)[-top_n:]
            pool_features = features[top_idx]
            pool_scores = np.clip(scores[top_idx], 1e-12, None)

            if len(pool_features) <= num_features:
                sampler = qmc.LatinHypercube(d=num_features, seed=42)
                lhs_sample = sampler.random(n=num_seeds_per_region)
                all_seeds.append(qmc.scale(lhs_sample, lower_bounds, upper_bounds))
                continue

            try:
                kde = gaussian_kde(pool_features.T, weights=pool_scores)
            except np.linalg.LinAlgError:
                sampler = qmc.LatinHypercube(d=num_features, seed=42)
                lhs_sample = sampler.random(n=num_seeds_per_region)
                all_seeds.append(qmc.scale(lhs_sample, lower_bounds, upper_bounds))
                continue

            sampler = qmc.LatinHypercube(d=num_features, seed=42)
            lhs_sample = sampler.random(n=num_seeds_per_region * 5)
            scaled_sample = qmc.scale(lhs_sample, lower_bounds, upper_bounds)

            kde_probs = kde(scaled_sample.T)
            best_idx = np.argsort(kde_probs)[-num_seeds_per_region:]
            refined_seeds = scaled_sample[best_idx]
            all_seeds.append(refined_seeds)

        if not all_seeds:
            return np.array([])

        return np.vstack(all_seeds)
