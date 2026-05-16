"""
Level 1: 容错冗余自愈引擎。

执行本地快照回滚或从最近健康邻居同步参数/经验池。
所有操作均通过 HealingLogger 记录来源节点。
"""

import time
import copy
from typing import Dict, List, Optional, Any

from .healing_logger import HealingLogger


class HealingResult:
    """自愈结果封装 (全模块统一使用)"""

    def __init__(
        self,
        node_id: str,
        success: bool,
        healing_level: int,
        applied_parameters: Optional[Dict[str, Any]] = None,
        healing_time: float = 0.0,
        message: str = "",
        source_node: Optional[str] = None,
        logger: Optional[HealingLogger] = None,
    ):
        self.node_id = node_id
        self.success = success
        self.healing_level = healing_level  # 1, 2, 3
        self.applied_parameters = applied_parameters
        self.healing_time = healing_time
        self.message = message
        self.source_node = source_node  # "self" 或 邻居 SID
        self.logger = logger


class RedundantResource:
    """冗余资源包 (包含模型权重与经验池)"""

    def __init__(
        self,
        state_dict: Optional[Dict[str, Any]] = None,
        replay_buffer: Optional[List[Any]] = None,
        source_node: str = "unknown",
    ):
        self.state_dict = state_dict
        self.replay_buffer = replay_buffer
        self.source_node = source_node


class NeighborSyncManager:
    """邻居同步管理器"""

    def __init__(self, node_id: str):
        self.node_id = node_id

    def _simulate_communication_delay(self, distance: float) -> float:
        """
        仿真星间链路通信延时
        delay = 固定握手开销 + 基于距离的传播延时
        """
        fixed_overhead = 0.002  # 模拟 2ms 的协议握手与序列化开销
        propagation_delay = distance / 300000.0  # 3e5 km/s 光速
        total_delay = fixed_overhead + propagation_delay * 10
        time.sleep(total_delay)
        return total_delay

    def request_neighbor_parameters(
        self,
        neighbors_info: List[Dict[str, Any]],
        health_threshold: float = 0.8,
        restore_model: bool = True,
        restore_pool: bool = True,
        logger: Optional[HealingLogger] = None,
    ) -> Optional[RedundantResource]:
        """
        按距离排序，寻找最近的、且快照可用的健康邻居。
        neighbors_info: [{id, health, distance, snapshot_available, state_dict, replay_buffer}]
        """
        sorted_neighbors = sorted(
            neighbors_info, key=lambda x: x.get("distance", float("inf"))
        )

        if logger:
            logger.log_step(1, "开始按距离扫描邻居节点", f"候选邻居数: {len(sorted_neighbors)}")

        for info in sorted_neighbors:
            nid = info["id"]
            health = info["health"]
            dist = info.get("distance", 0)
            snapshot_avail = info.get("snapshot_available", True)

            if health >= health_threshold:
                if snapshot_avail:
                    has_model = info.get("state_dict") is not None
                    has_pool = info.get("replay_buffer") is not None

                    if (restore_model and not has_model) or (restore_pool and not has_pool):
                        if logger:
                            logger.log_step(1, f"邻居 {nid} 缺少被请求的部分资源，跳过", source_node=nid, success=False)
                        continue

                    delay_spent = self._simulate_communication_delay(dist)
                    if logger:
                        logger.log_step(
                            1,
                            f"命中健康邻居，获取冗余资源",
                            detail=f"距离: {dist:.2f} km, 模拟延时: {delay_spent:.4f}s",
                            source_node=nid,
                            success=True,
                        )

                    return RedundantResource(
                        state_dict=copy.deepcopy(info["state_dict"]) if restore_model else None,
                        replay_buffer=copy.deepcopy(info.get("replay_buffer")) if restore_pool else None,
                        source_node=nid,
                    )
                else:
                    if logger:
                        logger.log_step(1, f"邻居 {nid} 快照不可用，跳过", source_node=nid, success=False)
            else:
                if logger:
                    logger.log_step(1, f"邻居 {nid} 状态不健康 (health={health:.2f})，跳过", source_node=nid, success=False)

        if logger:
            logger.log_step(1, "未找到可用的健康邻居节点", success=False)
        return None


class RedundantHealingEngine:
    """Level 1: 容错冗余自愈引擎"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.neighbor_mgr = NeighborSyncManager(node_id)

    def process_healing(
        self,
        target_node,
        neighbors_status: List[Dict[str, Any]],
        eval_callback,
        restore_model: bool = True,
        restore_pool: bool = True,
        logger: Optional[HealingLogger] = None,
    ) -> HealingResult:
        """
        执行冗余自愈逻辑：
        1. 尝试本地基线快照回滚
        2. 若失败，尝试从健康邻居同步参数
        """
        start_time = time.time()

        restore_desc = []
        if restore_model:
            restore_desc.append("模型参数")
        if restore_pool:
            restore_desc.append("经验池")
        restore_label = " + ".join(restore_desc) if restore_desc else "全部"

        if logger:
            logger.log_step(1, f"启动冗余自愈 (恢复目标: {restore_label})")

        # ── 阶段一：本地快照回滚 ──
        if logger:
            logger.log_step(1, "尝试读取节点内置的安全基线快照")

        if target_node.rollback_to_baseline(restore_model=restore_model, restore_pool=restore_pool):
            restored_params = copy.deepcopy(target_node.q_network.state_dict()) if restore_model else None
            restored_pool = list(target_node.replay_buffer) if restore_pool else None

            reward = 1.0
            if restore_model and restored_params:
                reward = eval_callback(restored_params)

            if reward > 0:
                if logger:
                    logger.log_step(1, "本地基线快照回滚成功", detail=f"恢复内容: {restore_label}", source_node="self", success=True)
                return HealingResult(
                    node_id=self.node_id,
                    success=True,
                    healing_level=1,
                    applied_parameters={"state_dict": restored_params, "replay_buffer": restored_pool},
                    healing_time=time.time() - start_time,
                    message=f"本地基线快照回滚成功 ({restore_label})",
                    source_node="self",
                    logger=logger,
                )
            else:
                if logger:
                    logger.log_step(1, "本地基线快照评估不通过", detail="快照可能不足以应对当前环境", source_node="self", success=False)
        else:
            if logger:
                logger.log_step(1, "未发现可用的本地基线快照", source_node="self", success=False)

        # ── 阶段二：邻居同步 ──
        if logger:
            logger.log_step(1, "快照恢复失败，启动邻居参数同步流程")

        synced_resource = self.neighbor_mgr.request_neighbor_parameters(
            neighbors_status,
            restore_model=restore_model,
            restore_pool=restore_pool,
            logger=logger,
        )

        if synced_resource is not None:
            reward = 1.0
            if restore_model and synced_resource.state_dict:
                reward = eval_callback(synced_resource.state_dict)

            if reward > 0:
                if logger:
                    logger.log_step(
                        1,
                        f"邻居资源同步成功",
                        detail=f"恢复内容: {restore_label}",
                        source_node=synced_resource.source_node,
                        success=True,
                    )
                return HealingResult(
                    node_id=self.node_id,
                    success=True,
                    healing_level=1,
                    applied_parameters={
                        "state_dict": synced_resource.state_dict,
                        "replay_buffer": synced_resource.replay_buffer,
                    },
                    healing_time=time.time() - start_time,
                    message=f"邻居资源同步成功 ({restore_label})",
                    source_node=synced_resource.source_node,
                    logger=logger,
                )
            else:
                if logger:
                    logger.log_step(1, "邻居同步参数评估不通过", source_node=synced_resource.source_node, success=False)

        # ── 冗余自愈失败 ──
        if logger:
            logger.log_step(1, "冗余自愈层全部尝试后仍无法恢复", success=False)

        return HealingResult(
            node_id=self.node_id,
            success=False,
            healing_level=1,
            applied_parameters=None,
            healing_time=time.time() - start_time,
            message="冗余自愈层尝试后无法恢复",
            logger=logger,
        )
