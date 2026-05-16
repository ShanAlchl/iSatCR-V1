"""
Level 3: 批量冗余自愈引擎。

当失效节点数达到星座规模的一定比例 (默认 5%) 时触发。
按照文档 3.2.2.3 批量冗余自愈算法实现:
  1. 领导者选举 (轨道面 leader 存活检查 → 跨域选举)
  2. 统一封锁与基线提取 (从可信模型库提取基线模型)
  3. 组播分发 (一次性下发至受灾区域)
  4. 并行验证与加载 (签名验证 + 哈希比对)
  5. 分批解除隔离 (核心路由节点优先)
"""

import time
import copy
import hashlib
import random
from typing import List, Dict, Optional, Any, Tuple

from .healing_logger import HealingLogger
from .redundant_healing import HealingResult


# ── 批量冗余触发阈值 ──
BATCH_TRIGGER_RATIO = 0.05  # 失效节点数 ≥ 星座规模 × 5%


class BatchLeaderElection:
    """领导者选举 (模拟 PBFT 轻量共识)"""

    @staticmethod
    def elect_leader(
        failed_node_ids: List[str],
        all_neighbors_status: List[Dict[str, Any]],
        logger: Optional[HealingLogger] = None,
    ) -> Optional[str]:
        """
        1. 检查当前批次中是否有轨道面 leader 存活
        2. 若 leader 已失效, 从相邻健康轨道面中跨域选举高信誉节点
        返回: 被选举的 leader 节点 SID
        """
        failed_set = set(failed_node_ids)

        # 按距离排序, 选择最近且健康的节点作为 leader
        healthy = [
            info for info in all_neighbors_status
            if info["id"] not in failed_set and info.get("health", 0) >= 0.8
        ]
        healthy.sort(key=lambda x: x.get("distance", float("inf")))

        if not healthy:
            if logger:
                logger.log_step(3, "领导者选举失败: 无健康邻居节点可用", success=False)
            return None

        leader = healthy[0]
        leader_id = leader["id"]

        # 检查 leader 是否与失效节点同轨道面
        leader_orbit = leader_id.split("_")[2] if len(leader_id.split("_")) > 2 else "?"
        same_orbit_failed = [fid for fid in failed_node_ids if fid.split("_")[2] == leader_orbit]

        if same_orbit_failed:
            # 同轨道面有失效节点, 尝试跨域选举 (从不同轨道面选)
            cross_orbit = [h for h in healthy if h["id"].split("_")[2] != leader_orbit]
            if cross_orbit:
                leader = cross_orbit[0]
                leader_id = leader["id"]
                if logger:
                    logger.log_step(
                        3, f"同轨道面存在失效节点, 跨域选举临时领导者: {leader_id}",
                        detail=f"领导者轨道面: {leader_id.split('_')[2]}, "
                               f"原轨道面失效节点: {len(same_orbit_failed)} 颗",
                        source_node=leader_id, success=True,
                    )
            else:
                if logger:
                    logger.log_step(3, f"无法跨域选举, 使用最近健康节点: {leader_id}",
                                    source_node=leader_id, success=True)
        else:
            if logger:
                logger.log_step(3, f"轨道面领导者存活, 选定: {leader_id}",
                                source_node=leader_id, success=True)

        return leader_id


class BatchRedundantHealingEngine:
    """Level 3: 批量冗余自愈引擎"""

    def __init__(self):
        self.election = BatchLeaderElection()

    @staticmethod
    def should_trigger(num_failed: int, constellation_size: int) -> bool:
        """判断是否达到批量冗余触发阈值"""
        if constellation_size <= 0:
            return False
        return num_failed >= constellation_size * BATCH_TRIGGER_RATIO

    def _compute_model_hash(self, state_dict: Dict[str, Any]) -> str:
        """计算模型参数的摘要哈希 (用于验证完整性)"""
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            val = state_dict[key]
            if hasattr(val, "cpu"):
                h.update(val.cpu().numpy().tobytes())
            else:
                h.update(str(val).encode())
        return h.hexdigest()[:16]

    def process_batch_healing(
        self,
        failed_nodes: Dict[str, Any],
        all_neighbors_status: List[Dict[str, Any]],
        constellation_size: int,
        logger: Optional[HealingLogger] = None,
    ) -> Dict[str, HealingResult]:
        """
        执行批量冗余自愈。

        参数:
            failed_nodes:         {SID: SatelliteNode} 需要修复的失效节点
            all_neighbors_status: 全网邻居状态列表
            constellation_size:   星座总节点数
            logger:               日志记录器 (可选, 批量模式下为首个失效节点的 logger)

        返回: {SID: HealingResult} 每个失效节点的修复结果
        """
        start_time = time.time()
        failed_ids = list(failed_nodes.keys())
        num_failed = len(failed_ids)

        if logger:
            logger.log_step(
                3, f"启动批量冗余自愈 (Level 3)",
                detail=f"失效节点: {num_failed} 颗, 星座规模: {constellation_size}, "
                       f"失效比例: {num_failed/constellation_size*100:.1f}%",
            )

        results: Dict[str, HealingResult] = {}

        # ══════════════════════════════════════════════════
        # 阶段 1: 领导者选举
        # ══════════════════════════════════════════════════
        if logger:
            logger.log_step(3, "阶段 1: 领导者选举 (改进 PBFT)")

        leader_id = self.election.elect_leader(
            failed_ids, all_neighbors_status, logger=logger
        )

        if leader_id is None:
            if logger:
                logger.log_step(3, "批量冗余自愈终止: 领导者选举失败", success=False)
            for sid in failed_ids:
                results[sid] = HealingResult(
                    node_id=sid, success=False, healing_level=3,
                    healing_time=time.time() - start_time,
                    message="领导者选举失败, 批量冗余无法执行",
                )
            return results

        # ══════════════════════════════════════════════════
        # 阶段 2: 统一封锁与基线提取
        # ══════════════════════════════════════════════════
        if logger:
            logger.log_step(3, "阶段 2: 领导者封锁失效区域, 统一提取基线模型",
                            source_node=leader_id)

        # 从领导者节点的邻居状态中找到其快照
        leader_info = next(
            (info for info in all_neighbors_status if info["id"] == leader_id), None
        )
        baseline_state_dict = None
        baseline_replay_buffer = None
        if leader_info:
            baseline_state_dict = leader_info.get("state_dict")
            baseline_replay_buffer = leader_info.get("replay_buffer")

        if baseline_state_dict is None:
            if logger:
                logger.log_step(3, f"领导者 {leader_id} 无可用基线模型快照", success=False)
            for sid in failed_ids:
                results[sid] = HealingResult(
                    node_id=sid, success=False, healing_level=3,
                    healing_time=time.time() - start_time,
                    message="领导者无可用基线, 批量冗余失败",
                )
            return results

        model_hash = self._compute_model_hash(baseline_state_dict)
        if logger:
            logger.log_step(
                3, f"基线模型提取成功",
                detail=f"模型哈希: {model_hash}, 来源: {leader_id}",
                source_node=leader_id, success=True,
            )

        # ══════════════════════════════════════════════════
        # 阶段 3: 组播分发 (模拟星间链路广播)
        # ══════════════════════════════════════════════════
        if logger:
            logger.log_step(
                3, f"阶段 3: 领导者组播加密模型包至 {num_failed} 个失效节点",
                detail="星间链路空间组播树模式, 一次性下发",
                source_node=leader_id,
            )

        # 模拟组播延时 (比逐个发送快得多)
        broadcast_delay = 0.005 + num_failed * 0.0005  # 基础延时 + 每节点少量开销
        time.sleep(broadcast_delay)

        if logger:
            logger.log_step(3, f"组播分发完成",
                            detail=f"模拟耗时: {broadcast_delay:.4f}s", success=True)

        # ══════════════════════════════════════════════════
        # 阶段 4: 各节点并行验证与加载
        # ══════════════════════════════════════════════════
        if logger:
            logger.log_step(3, "阶段 4: 各节点并行执行签名验证与哈希比对")

        verified_nodes = []
        failed_verify_nodes = []

        for sid, node in failed_nodes.items():
            # 模拟签名验证 (99% 概率通过)
            verify_pass = random.random() < 0.99
            if verify_pass:
                # 加载模型
                node.load_q_network_state_dict(
                    copy.deepcopy(baseline_state_dict), is_initial=False
                )
                # 恢复经验池
                if baseline_replay_buffer:
                    node.replay_buffer.clear()
                    node.replay_buffer.extend(copy.deepcopy(baseline_replay_buffer))

                verified_nodes.append(sid)
                results[sid] = HealingResult(
                    node_id=sid, success=True, healing_level=3,
                    applied_parameters={
                        "state_dict": copy.deepcopy(baseline_state_dict),
                        "replay_buffer": copy.deepcopy(baseline_replay_buffer) if baseline_replay_buffer else None,
                    },
                    healing_time=time.time() - start_time,
                    message=f"批量冗余恢复成功 (模型哈希: {model_hash})",
                    source_node=leader_id,
                )
            else:
                failed_verify_nodes.append(sid)
                results[sid] = HealingResult(
                    node_id=sid, success=False, healing_level=3,
                    healing_time=time.time() - start_time,
                    message="签名验证失败, 需升级至联邦学习",
                )

        if logger:
            logger.log_step(
                3, f"并行验证完成",
                detail=f"验证通过: {len(verified_nodes)}/{num_failed}, "
                       f"验证失败: {len(failed_verify_nodes)}",
                success=len(failed_verify_nodes) == 0,
            )

        # ══════════════════════════════════════════════════
        # 阶段 5: 分批解除隔离
        # ══════════════════════════════════════════════════
        if verified_nodes and logger:
            # 按轨道面分组，模拟分批解隔离
            orbit_groups: Dict[str, List[str]] = {}
            for sid in verified_nodes:
                parts = sid.split("_")
                orbit_key = parts[2] if len(parts) > 2 else "0"
                orbit_groups.setdefault(orbit_key, []).append(sid)

            logger.log_step(
                3, f"阶段 5: 分批解除隔离 (按轨道面优先级)",
                detail=f"涉及 {len(orbit_groups)} 个轨道面, "
                       f"总恢复节点: {len(verified_nodes)} 颗",
            )

            for batch_idx, (orbit, sids_in_orbit) in enumerate(sorted(orbit_groups.items()), 1):
                time.sleep(0.002)  # 模拟分批间隔
                logger.log_step(
                    3, f"批次 {batch_idx}: 轨道面 {orbit} 解除隔离",
                    detail=f"节点数: {len(sids_in_orbit)}, "
                           f"节点: {', '.join(sids_in_orbit[:5])}{'...' if len(sids_in_orbit) > 5 else ''}",
                    success=True,
                )

        if logger:
            total_elapsed = time.time() - start_time
            success_count = len(verified_nodes)
            logger.log_step(
                3, f"批量冗余自愈完成",
                detail=f"成功: {success_count}/{num_failed}, 总耗时: {total_elapsed:.4f}s",
                success=success_count > 0,
            )

        return results
