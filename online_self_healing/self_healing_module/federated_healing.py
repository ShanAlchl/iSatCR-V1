"""
Level 4: 联邦学习自愈引擎。

当冗余、免疫、批量冗余自愈均无法解决问题时，启动联邦学习自愈。
所有阶段 (本地训练、异常检测剪枝、聚合) 均通过 HealingLogger 详细记录。
"""

import numpy as np
import time
import copy
import torch
from typing import List, Dict, Optional, Tuple, Any

from .healing_logger import HealingLogger
from .redundant_healing import HealingResult


class FederatedWorker:
    """星载联邦工作节点（本地小批量训练抽象）"""

    def __init__(self, node_id: str):
        self.node_id = node_id

    def local_train(
        self,
        base_state_dict: Dict[str, Any],
        local_data_samples: int = 100,
        logger: Optional[HealingLogger] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        模拟在本地缓存经验数据上进行小批量微调。
        返回: (更新后的模型状态字典, 训练样本数)
        """
        if logger:
            logger.log_step(4, f"工作节点 {self.node_id} 开始本地训练",
                            detail=f"样本数: {local_data_samples}",
                            source_node=self.node_id)

        time.sleep(0.1)  # 模拟训练耗时

        updated_dict = copy.deepcopy(base_state_dict)
        param_count = 0
        for key, value in updated_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                noise = torch.randn_like(value) * 0.05 + 0.01
                updated_dict[key] += noise
                param_count += value.numel()

        if logger:
            logger.log_step(4, f"工作节点 {self.node_id} 本地训练完成",
                            detail=f"更新参数量: {param_count}, 训练样本: {local_data_samples}",
                            source_node=self.node_id, success=True)

        return updated_dict, local_data_samples


class FederatedAggregator:
    """星间联邦参数聚合器"""

    def fedavg_aggregation(
        self,
        client_updates: List[Tuple[Dict[str, Any], int]],
        logger: Optional[HealingLogger] = None,
    ) -> Dict[str, Any]:
        """标准联邦平均 (FedAvg) 算法"""
        total_samples = sum(num_samples for _, num_samples in client_updates)

        if logger:
            logger.log_step(4, "执行 FedAvg 加权聚合",
                            detail=f"参与节点数: {len(client_updates)}, 总样本数: {total_samples}")

        aggregated_weights = copy.deepcopy(client_updates[0][0])
        for key in aggregated_weights.keys():
            if isinstance(aggregated_weights[key], torch.Tensor) and aggregated_weights[key].is_floating_point():
                aggregated_weights[key].zero_()

        for weights, num_samples in client_updates:
            weight_fraction = num_samples / total_samples
            for key in aggregated_weights.keys():
                if isinstance(aggregated_weights[key], torch.Tensor) and aggregated_weights[key].is_floating_point():
                    aggregated_weights[key] += weights[key] * weight_fraction

        if logger:
            logger.log_step(4, "FedAvg 聚合完成", success=True)

        return aggregated_weights

    def anomaly_detection_aggregation(
        self,
        client_updates: List[Tuple[Dict[str, Any], int]],
        threshold: float = 1.5,
        logger: Optional[HealingLogger] = None,
    ) -> Dict[str, Any]:
        """带异常检测的聚合（排除离群更新以防范投毒/故障传播）"""
        if len(client_updates) <= 2:
            if logger:
                logger.log_step(3, "节点数不足 3，跳过异常检测，直接执行 FedAvg")
            return self.fedavg_aggregation(client_updates, logger=logger)

        if logger:
            logger.log_step(4, "启动异常更新检测与剪枝",
                            detail=f"剪枝阈值 (Z-Score): {threshold}")

        def flatten_dict(d: Dict[str, Any]) -> np.ndarray:
            tensors = []
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    tensors.append(v.cpu().numpy().flatten())
            if not tensors:
                return np.array([])
            return np.concatenate(tensors)

        flat_weights = [flatten_dict(w) for w, _ in client_updates]
        center_weight = np.mean(flat_weights, axis=0)

        distances = [np.linalg.norm(w - center_weight) for w in flat_weights]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        valid_updates = []
        pruned_nodes = []
        for i, (weights, num_samples) in enumerate(client_updates):
            z_score = (distances[i] - mean_dist) / std_dist if std_dist > 0 else 0
            if std_dist > 0 and z_score > threshold:
                pruned_nodes.append(i)
                if logger:
                    logger.log_step(4, f"剪枝: 节点 {i} 更新异常 (Z-Score={z_score:.2f})，已剔除",
                                    detail=f"与中心距离: {distances[i]:.4f}, 均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}",
                                    success=False)
            else:
                valid_updates.append((weights, num_samples))

        if not valid_updates:
            if logger:
                logger.log_step(4, "警告: 所有节点均被判定为异常，回退使用全部节点", success=False)
            return self.fedavg_aggregation(client_updates, logger=logger)

        if logger:
            logger.log_step(4, f"异常检测完成",
                            detail=f"有效节点: {len(valid_updates)}/{len(client_updates)}, 剔除: {len(pruned_nodes)} 个",
                            success=True)

        return self.fedavg_aggregation(valid_updates, logger=logger)


class FederatedHealingEngine:
    """Level 4: 联邦学习自愈引擎"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.aggregator = FederatedAggregator()

    def process_healing(
        self,
        base_state_dict: Dict[str, Any],
        healthy_neighbors: List[str],
        logger: Optional[HealingLogger] = None,
    ) -> HealingResult:
        """
        执行联邦自愈：
        1. 寻找健康邻居作为工作节点
        2. 分发基础模型，触发本地训练
        3. 收集更新，进行带异常检测的聚合
        """
        start_time = time.time()

        if logger:
            logger.log_step(4, "启动联邦学习自愈 (局部邻域聚合)")
            logger.log_step(4, f"健康邻居数量: {len(healthy_neighbors)}",
                            detail=f"参与节点: {', '.join(healthy_neighbors[:10])}{'...' if len(healthy_neighbors) > 10 else ''}")

        if len(healthy_neighbors) < 2:
            if logger:
                logger.log_step(4, "健康邻居数量不足 (< 2)，无法启动联邦学习", success=False)
            return HealingResult(
                node_id=self.node_id, success=False, healing_level=4,
                applied_parameters=None, healing_time=time.time() - start_time,
                message="健康节点不足，联邦聚合取消", logger=logger,
            )

        # ── 阶段一：本地训练收集 ──
        if logger:
            logger.log_step(4, "阶段 1: 下发初始模型，启动各节点本地训练")

        client_updates = []
        for nid in healthy_neighbors:
            worker = FederatedWorker(nid)
            samples = np.random.randint(50, 200)
            updated_weights, actual_samples = worker.local_train(
                base_state_dict, local_data_samples=samples, logger=logger
            )
            client_updates.append((updated_weights, actual_samples))

        # ── 阶段二：异常检测与剪枝 ──
        if logger:
            logger.log_step(4, "阶段 2: 所有节点训练完成，启动异常检测与参数聚合")

        aggregated_model = self.aggregator.anomaly_detection_aggregation(
            client_updates, logger=logger
        )

        # ── 阶段三：生成全局模型 ──
        if logger:
            logger.log_step(4, "阶段 3: 联邦参数聚合完成，生成新全局模型", success=True)

        return HealingResult(
            node_id=self.node_id,
            success=True,
            healing_level=4,
            applied_parameters=aggregated_model,
            healing_time=time.time() - start_time,
            message="联邦学习重训练成功",
            logger=logger,
        )
