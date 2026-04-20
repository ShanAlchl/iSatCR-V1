import numpy as np
import warnings
from typing import List, Dict, Tuple

from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde, beta

try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
    warnings.warn("scipy >= 1.7.0 is required for LatinHypercube. LHS features won't work.")


STATUS_NEEDS_EXPLORATION = "needs_exploration"
STATUS_BOUNDARY_IDENTIFIED = "boundary_identified"
STATUS_STABLE = "stable"


class FailureBoundaryExplorer:
    """
    高维失效边界探索器

    核心能力:
    1) 使用 KMeans 对连续参数空间进行区域划分
    2) 使用 RAU(区域平均不确定度) + SC(区域采样覆盖度) 判断区域是否仍需探索
    3) 在需探索区域中生成后续探索种子点
    4) 输出“覆盖率点估计 + 置信下界”的可量化指标
    """

    def __init__(self, n_clusters: int = 5, rau_threshold: float = 0.1, sc_threshold: float = 0.7):
        """
        :param n_clusters: KMeans 区域数量
        :param rau_threshold: RAU 阈值，超过该阈值代表该区域不确定度较高
        :param sc_threshold: SC 阈值，低于该阈值代表该区域采样不足
        """
        self.n_clusters = n_clusters
        self.rau_threshold = rau_threshold
        self.sc_threshold = sc_threshold
        self.failure_cloud_points: List[Dict] = []

    @staticmethod
    def _calculate_rau(cv_values: np.ndarray) -> float:
        """计算区域平均不确定度 RAU。"""
        if len(cv_values) == 0:
            return 0.0
        return float(np.mean(cv_values))

    @staticmethod
    def _calculate_sc(num_samples: int, theoretical_max: int) -> float:
        """计算区域覆盖度 SC。"""
        if theoretical_max <= 0:
            return 1.0
        return float(num_samples / theoretical_max)

    @staticmethod
    def _clopper_pearson_lower_bound(successes: int, total: int, confidence: float = 0.95) -> float:
        """
        计算二项分布比例的 Clopper-Pearson 置信下界。

        说明:
        - confidence=0.95 对应 95% 置信度
        - 返回的是保守下界，可用于“是否达到目标覆盖率”的判定
        """
        if total <= 0:
            return 0.0
        if successes <= 0:
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
        metadata: List[Dict] = None,
    ) -> List[Dict]:
        """
        Append points to the explorer's cloud map for later inspection/export.
        Each point stores only the ensemble-predicted failure score and uncertainty.
        """
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

    def partition_and_evaluate(
        self,
        features: np.ndarray,
        failure_scores: np.ndarray,
        cv_values: np.ndarray,
        theoretical_max_per_region: int = 500,
    ) -> List[Dict]:
        """
        在高维连续参数空间中划分区域并评估区域状态。

        :param features: 已探索样本特征 [N, num_features]
        :param failure_scores: 失效评分 [N]，范围建议 [0, 1]
        :param cv_values: 预测不确定度 [N]
        :param theoretical_max_per_region: 每个区域的理论样本容量（用于 SC）
        :return: 每个区域的统计信息
        """
        if len(features) < self.n_clusters:
            raise ValueError("样本数量不足，无法进行 KMeans 聚类")

        region_stats: List[Dict] = []
        total_samples = len(features)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        for cluster_id in range(self.n_clusters):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) == 0:
                continue

            region_features = features[idx]
            region_scores = failure_scores[idx]
            region_cvs = cv_values[idx]

            rau = self._calculate_rau(region_cvs)
            sc = self._calculate_sc(len(idx), theoretical_max_per_region)

            status = STATUS_STABLE
            if rau > self.rau_threshold or sc < self.sc_threshold:
                status = STATUS_NEEDS_EXPLORATION
            else:
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
        """
        计算覆盖率指标。

        定义:
        - covered sample: 区域状态不是 needs_exploration 的样本
        - coverage point estimate: covered / total
        - coverage lower bound: 基于 Clopper-Pearson 的置信下界

        判定:
        - 当 coverage lower bound >= target_coverage 时，认为覆盖率目标达成。
        """
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

        return {
            "total_samples": total,
            "covered_samples": covered,
            "coverage_point_estimate": point_estimate,
            "coverage_lower_bound": lower_bound,
            "coverage_upper_bound": upper_bound,
            "confidence": confidence,
            "target_coverage": target_coverage,
            "target_achieved": bool(lower_bound >= target_coverage),
        }

    def generate_seed_candidates(
        self,
        region_stats: List[Dict],
        feature_bounds: List[Tuple[float, float]],
        num_seeds_per_region: int = 20,
    ) -> np.ndarray:
        """
        针对 needs_exploration 区域，使用 LHS + KDE 生成候选种子点。

        :param region_stats: 区域评估结果
        :param feature_bounds: 各维取值范围，例如 [(0, 1), (-1, 1)]
        :param num_seeds_per_region: 每个区域生成种子数量
        :return: 候选种子矩阵 [num_seeds, num_features]
        """
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


if __name__ == "__main__":
    explorer = FailureBoundaryExplorer(n_clusters=3, rau_threshold=0.12, sc_threshold=0.6)

    # 模拟样本
    np.random.seed(42)
    dummy_features = np.random.rand(100, 5)
    dummy_scores = np.random.rand(100)
    dummy_cvs = np.random.rand(100) * 0.3

    # 区域评估
    region_stats = explorer.partition_and_evaluate(
        dummy_features,
        dummy_scores,
        dummy_cvs,
        theoretical_max_per_region=50,
    )

    print("=== Region Status ===")
    for r in region_stats:
        print(
            f"Region {r['cluster_id']}: "
            f"RAU={r['RAU']:.3f}, "
            f"SC={r['SC']:.2f}, "
            f"count={r['sample_count']}, "
            f"status={r['status']}"
        )

    # 覆盖率指标输出
    metrics = explorer.compute_coverage_metrics(
        region_stats,
        confidence=0.95,
        target_coverage=0.90,
    )
    print("\n=== Coverage Metrics ===")
    print(f"total_samples: {metrics['total_samples']}")
    print(f"covered_samples: {metrics['covered_samples']}")
    print(f"coverage_point_estimate: {metrics['coverage_point_estimate']:.4f}")
    print(f"coverage_lower_bound@{metrics['confidence']:.2f}: {metrics['coverage_lower_bound']:.4f}")
    print(f"target_coverage: {metrics['target_coverage']:.2f}")
    print(f"target_achieved: {metrics['target_achieved']}")

    bounds = [(0.0, 1.0)] * 5
    seeds = explorer.generate_seed_candidates(region_stats, bounds, num_seeds_per_region=5)
    print("\nGenerated Seeds Shape:", seeds.shape)
