"""
分级自愈总控调度器 (4级架构)。

核心逻辑：
  已判定失效的节点 → 根据 FailScore + 建链关系比例 分级路由
    ├── 低失效风险  (FailScore < 0.40) → Level 1 单点冗余自愈
    ├── 中失效风险  (0.40 ≤ FailScore < 0.50 且失效节点建链比例 ≤ 30%) → Level 2 免疫自愈
    ├── 批量中失效风险  (0.40 ≤ FailScore < 0.50 且失效节点建链比例 > 25%) → Level 3 批量冗余自愈
    └── 高失效风险  (FailScore ≥ 0.50) → Level 4 联邦学习自愈
  每一级如果失败，自动降级至下一级。

批量模式入口:
  execute_batch_healing() — 接收全部失效节点列表，统一调度。
"""

import time
import random
from typing import Dict, List, Optional, Any, Set

from .healing_logger import HealingLogger
from .redundant_healing import RedundantHealingEngine, HealingResult
from .immune_healing import ImmuneHealingEngine
from .batch_redundant_healing import BatchRedundantHealingEngine
from .federated_healing import FederatedHealingEngine


# ── 分级阈值 ──
FAILSCORE_LOW_UPPER = 0.40
FAILSCORE_HIGH_LOWER = 0.50

# ── 建链关系比例阈值 (区分"中"和"批量中") ──
# 阈值设为 25%: 当某失效节点在其同/相邻轨道面内有 >25% 的建链邻居也失效时，
# 判定为批量中失效风险，进入 Level 3 批量冗余自愈。
# (跨2轨道面集中失效时 link_ratio ≈ 28~30%，设置为25%可稳定触发)
LINK_RATIO_THRESHOLD = 0.25


class TieredHealingOrchestrator:
    """
    分级自愈调度器 (4级)。
    Level 1 单点冗余 / Level 2 免疫 / Level 3 批量冗余 / Level 4 联邦学习
    """

    def __init__(self, node_id: str):
        self.node_id = node_id

        self.level1_redundant = RedundantHealingEngine(node_id)
        self.level2_immune = ImmuneHealingEngine()
        self.level3_batch = BatchRedundantHealingEngine()
        self.level4_federated = FederatedHealingEngine(node_id)

        self.level2_immune.set_redundant_engine(self.level1_redundant)

    def default_eval_callback(self, state_dict: Dict[str, Any]) -> float:
        if state_dict is None:
            return -1.0
        return 1.0

    # ==================================================================
    # 建链关系比例计算
    # ==================================================================

    @staticmethod
    def compute_link_ratio(
        node_id: str,
        failed_set: Set[str],
        neighbors_status: List[Dict[str, Any]],
    ) -> float:
        """
        计算某个失效节点的建链关系中，有多少比例也是失效节点。

        建链关系：neighbors_status 中同轨道面或距离较近的节点视为有建链关系。
        比例 = 与该节点有建链关系的失效节点数 / 与该节点有建链关系的总节点数

        参数:
            node_id:          当前节点 SID
            failed_set:       所有失效节点 SID 集合
            neighbors_status: 全网邻居状态列表
        """
        # 提取当前节点的轨道面编号
        parts = node_id.split("_")
        node_orbit = parts[2] if len(parts) > 2 else "?"

        # 同轨道面节点 + 距离最近的跨轨道面节点 视为有建链关系
        linked_nodes = []
        for info in neighbors_status:
            nid = info["id"]
            if nid == node_id:
                continue
            n_parts = nid.split("_")
            n_orbit = n_parts[2] if len(n_parts) > 2 else "?"
            # 同轨道面节点一定有建链关系
            if n_orbit == node_orbit:
                linked_nodes.append(nid)
            # 相邻轨道面的节点也有建链关系 (±1 轨道面)
            elif node_orbit.isdigit() and n_orbit.isdigit():
                if abs(int(n_orbit) - int(node_orbit)) <= 1:
                    linked_nodes.append(nid)

        if not linked_nodes:
            return 0.0

        failed_linked = [nid for nid in linked_nodes if nid in failed_set]
        return len(failed_linked) / len(linked_nodes)

    # ==================================================================
    # FailScore + 建链比例 → 4级分级判定
    # ==================================================================

    def _classify_fail_score(
        self,
        fail_score: float,
        link_ratio: float = 0.0,
    ) -> str:
        """
        4级分类:
          low:         FailScore < 0.40
          medium:      0.40 ≤ FailScore < 0.50 且 建链比例 ≤ 25%
          batch_medium: 0.40 ≤ FailScore < 0.50 且 建链比例 > 25%
          high:        FailScore ≥ 0.50
        """
        if fail_score < FAILSCORE_LOW_UPPER:
            return "low"
        elif fail_score < FAILSCORE_HIGH_LOWER:
            if link_ratio > LINK_RATIO_THRESHOLD:
                return "batch_medium"
            else:
                return "medium"
        else:
            return "high"

    # ==================================================================
    # 结果应用
    # ==================================================================

    def _apply_result_to_node(self, target_node, result: HealingResult, logger: HealingLogger) -> None:
        """将自愈结果中的参数和经验池应用到目标节点"""
        if result.applied_parameters is None:
            return

        if isinstance(result.applied_parameters, dict) and "state_dict" in result.applied_parameters:
            res_dict = result.applied_parameters
            if res_dict.get("state_dict") is not None:
                target_node.load_q_network_state_dict(res_dict["state_dict"], is_initial=False)
                logger.log_step(result.healing_level, "已将恢复的模型参数加载至节点 QNetwork",
                                source_node=result.source_node, success=True)
            if res_dict.get("replay_buffer") is not None:
                target_node.replay_buffer.clear()
                target_node.replay_buffer.extend(res_dict["replay_buffer"])
                logger.log_step(result.healing_level, "已同步恢复节点的经验池数据",
                                detail=f"经验池大小: {len(res_dict['replay_buffer'])} 条",
                                source_node=result.source_node, success=True)
        elif isinstance(result.applied_parameters, dict):
            target_node.load_q_network_state_dict(result.applied_parameters, is_initial=False)
            logger.log_step(result.healing_level, "已将联邦聚合的全局模型加载至节点 QNetwork", success=True)

    # ==================================================================
    # 单点模式入口 (兼容旧接口)
    # ==================================================================

    def execute_tiered_healing(
        self,
        target_node,
        fail_score: float,
        attack_type: Optional[int],
        neighbors_status: List[Dict[str, Any]],
        eval_callback=None,
        # 建链比例 (由批量调度器计算并传入)
        link_ratio: float = 0.0,
    ) -> HealingResult:
        """
        执行分级自愈 (单点模式)。

        参数:
            link_ratio: 该节点的失效建链关系比例 (用于区分 medium / batch_medium)
        """
        logger = HealingLogger(
            node_id=self.node_id, fail_score=fail_score, attack_type=attack_type,
        )
        total_start_time = time.time()

        if eval_callback is None:
            eval_callback = self.default_eval_callback

        risk_class = self._classify_fail_score(fail_score, link_ratio)
        risk_labels = {
            "low": "低失效风险",
            "medium": "中失效风险",
            "batch_medium": "批量中失效风险",
            "high": "高失效风险",
        }
        entry_levels = {"low": 1, "medium": 2, "batch_medium": 3, "high": 4}

        logger.log_step(
            entry_levels[risk_class],
            f"启动分级自愈总控流程",
            detail=f"FailScore={fail_score:.4f}, 分级={risk_labels[risk_class]}, "
                   f"建链关系失效比例={link_ratio:.2%}, "
                   f"攻击类型={attack_type}, 初始入口=Level {entry_levels[risk_class]}",
        )

        # ── Level 1: 冗余自愈 (低风险入口) ──
        if risk_class == "low":
            l1 = self._execute_level1(target_node, neighbors_status, eval_callback, logger)
            if l1.success:
                self._apply_result_to_node(target_node, l1, logger)
                self._finalize(logger, l1, total_start_time)
                return l1

            logger.log_step(2, "冗余自愈失败，降级至免疫自愈 (Level 2)")
            l2 = self._execute_level2(target_node, attack_type, neighbors_status, eval_callback, logger)
            if l2.success:
                self._apply_result_to_node(target_node, l2, logger)
                self._finalize(logger, l2, total_start_time)
                return l2

            logger.log_step(4, "免疫自愈失败，降级至联邦学习自愈 (Level 4)")
            l4 = self._execute_level4(target_node, neighbors_status, logger)
            if l4.success:
                self._apply_result_to_node(target_node, l4, logger)
            self._finalize(logger, l4, total_start_time)
            return l4

        # ── Level 2: 免疫自愈 (中风险入口) ──
        elif risk_class == "medium":
            l2 = self._execute_level2(target_node, attack_type, neighbors_status, eval_callback, logger)
            if l2.success:
                self._apply_result_to_node(target_node, l2, logger)
                self._finalize(logger, l2, total_start_time)
                return l2

            logger.log_step(4, "免疫自愈失败，降级至联邦学习自愈 (Level 4)")
            l4 = self._execute_level4(target_node, neighbors_status, logger)
            if l4.success:
                self._apply_result_to_node(target_node, l4, logger)
            self._finalize(logger, l4, total_start_time)
            return l4

        # ── Level 3: 批量冗余自愈 (批量中风险，单点模式时降级联邦) ──
        elif risk_class == "batch_medium":
            # 单点模式下无法执行批量冗余，直接标记需要批量处理
            logger.log_step(3, "批量中失效风险: 建链关系中失效节点占比过高，"
                               "需要批量冗余恢复 (将由批量调度器统一处理)")
            logger.log_step(4, "单点模式下降级至联邦学习自愈 (Level 4)")
            l4 = self._execute_level4(target_node, neighbors_status, logger)
            if l4.success:
                self._apply_result_to_node(target_node, l4, logger)
            self._finalize(logger, l4, total_start_time)
            return l4

        # ── Level 4: 联邦学习自愈 (高风险入口) ──
        else:
            l4 = self._execute_level4(target_node, neighbors_status, logger)
            if l4.success:
                self._apply_result_to_node(target_node, l4, logger)
            self._finalize(logger, l4, total_start_time)
            return l4

    # ==================================================================
    # 批量模式入口
    # ==================================================================

    def execute_batch_healing(
        self,
        failed_nodes: Dict[str, Any],
        fail_scores: Dict[str, float],
        attack_types: Dict[str, Optional[int]],
        all_nodes: Dict[str, Any],
        neighbors_status: List[Dict[str, Any]],
        eval_callback=None,
    ) -> Dict[str, HealingResult]:
        """
        执行批量自愈调度。

        分级路由:
        1. 计算每个失效节点的建链关系失效比例
        2. low → Level 1 单点冗余
        3. medium (建链比例 ≤ 30%) → Level 2 免疫
        4. batch_medium (建链比例 > 30%) → Level 3 批量冗余
        5. high → Level 4 联邦学习

        返回: {SID: HealingResult}
        """
        if eval_callback is None:
            eval_callback = self.default_eval_callback

        constellation_size = len(all_nodes)
        failed_set = set(failed_nodes.keys())
        all_results: Dict[str, HealingResult] = {}

        # ── 步骤 1: 计算每个失效节点的建链关系比例并分级 ──
        low_nodes = {}
        medium_nodes = {}
        batch_medium_nodes = {}
        high_nodes = {}

        for sid in failed_nodes:
            fs = fail_scores.get(sid, 0.5)
            lr = self.compute_link_ratio(sid, failed_set, neighbors_status)
            risk = self._classify_fail_score(fs, lr)

            if risk == "low":
                low_nodes[sid] = (failed_nodes[sid], fs, lr)
            elif risk == "medium":
                medium_nodes[sid] = (failed_nodes[sid], fs, lr)
            elif risk == "batch_medium":
                batch_medium_nodes[sid] = (failed_nodes[sid], fs, lr)
            else:
                high_nodes[sid] = (failed_nodes[sid], fs, lr)

        # 日志汇总
        print(f"\n[批量调度] 失效节点分级汇总: "
              f"低={len(low_nodes)}, 中={len(medium_nodes)}, "
              f"批量中={len(batch_medium_nodes)}, 高={len(high_nodes)}")

        # ── 步骤 2: 低风险 → Level 1 单点冗余 ──
        for sid, (node, fs, lr) in low_nodes.items():
            orch = TieredHealingOrchestrator(sid)
            orch.level1_redundant = self.level1_redundant
            orch.level2_immune = self.level2_immune
            orch.level4_federated = self.level4_federated
            result = orch.execute_tiered_healing(
                target_node=node,
                fail_score=fs,
                attack_type=attack_types.get(sid),
                neighbors_status=neighbors_status,
                eval_callback=eval_callback,
                link_ratio=lr,
            )
            all_results[sid] = result

        # ── 步骤 3: 中风险 → Level 2 免疫 ──
        for sid, (node, fs, lr) in medium_nodes.items():
            orch = TieredHealingOrchestrator(sid)
            orch.level1_redundant = self.level1_redundant
            orch.level2_immune = self.level2_immune
            orch.level4_federated = self.level4_federated
            result = orch.execute_tiered_healing(
                target_node=node,
                fail_score=fs,
                attack_type=attack_types.get(sid),
                neighbors_status=neighbors_status,
                eval_callback=eval_callback,
                link_ratio=lr,
            )
            all_results[sid] = result

        # ── 步骤 4: 批量中风险 → Level 3 批量冗余 ──
        if batch_medium_nodes:
            batch_logger = HealingLogger(
                node_id="BATCH_CONTROLLER",
                fail_score=0.0,
                attack_type=None,
            )

            batch_failed = {sid: info[0] for sid, info in batch_medium_nodes.items()}
            sample_lr = list(batch_medium_nodes.values())[0][2]

            batch_logger.log_step(
                3, f"批量中失效风险: {len(batch_medium_nodes)} 个节点进入 Level 3 批量冗余自愈",
                detail=f"建链关系失效比例均 > {LINK_RATIO_THRESHOLD:.0%}, "
                       f"示例比例: {sample_lr:.2%}",
            )

            batch_results = self.level3_batch.process_batch_healing(
                failed_nodes=batch_failed,
                all_neighbors_status=neighbors_status,
                constellation_size=constellation_size,
                logger=batch_logger,
            )

            # 收集批量冗余仍然失败的节点
            still_failed = {}
            success_count = 0
            for sid, br in batch_results.items():
                br.logger = batch_logger
                if br.success:
                    success_count += 1
                    all_results[sid] = br
                else:
                    still_failed[sid] = batch_medium_nodes[sid]

            if success_count > 0:
                batch_logger.log_result(3, True, f"批量冗余自愈完成 (成功 {success_count} 颗)")
            else:
                batch_logger.log_result(3, False, "批量冗余自愈全部失败")

            # 仍然失败的 → Level 4 联邦学习
            for sid, (node, fs, lr) in still_failed.items():
                orch = TieredHealingOrchestrator(sid)
                l4_logger = HealingLogger(
                    node_id=sid,
                    fail_score=fs,
                    attack_type=attack_types.get(sid),
                )
                l4_logger.log_step(4, "批量冗余验证失败, 降级至联邦学习 (Level 4)")
                l4_start_time = time.time()
                l4_result = orch._execute_level4(node, neighbors_status, l4_logger)
                if l4_result.success:
                    orch._apply_result_to_node(node, l4_result, l4_logger)
                orch._finalize(l4_logger, l4_result, l4_start_time)
                l4_result.logger = l4_logger
                all_results[sid] = l4_result

            print("\n" + batch_logger.export_report())

        # ── 步骤 5: 高风险 → Level 4 联邦学习 ──
        for sid, (node, fs, lr) in high_nodes.items():
            orch = TieredHealingOrchestrator(sid)
            result = orch.execute_tiered_healing(
                target_node=node,
                fail_score=fs,
                attack_type=attack_types.get(sid),
                neighbors_status=neighbors_status,
                eval_callback=eval_callback,
                link_ratio=lr,
            )
            all_results[sid] = result

        return all_results

    # ==================================================================
    # 各级自愈执行逻辑
    # ==================================================================

    def _execute_level1(self, target_node, neighbors_status, eval_callback, logger) -> HealingResult:
        """Level 1: 单点冗余自愈"""
        logger.log_step(1, "进入 Level 1 单点冗余自愈层")
        return self.level1_redundant.process_healing(
            target_node=target_node,
            neighbors_status=neighbors_status,
            eval_callback=eval_callback,
            restore_model=True, restore_pool=True, logger=logger,
        )

    def _execute_level2(self, target_node, attack_type, neighbors_status, eval_callback, logger) -> HealingResult:
        """Level 2: 免疫自愈"""
        logger.log_step(2, "进入 Level 2 免疫自愈层")
        return self.level2_immune.process_healing(
            target_node=target_node, attack_type=attack_type,
            neighbors_status=neighbors_status, eval_callback=eval_callback, logger=logger,
        )

    def _execute_level4(self, target_node, neighbors_status, logger) -> HealingResult:
        """Level 4: 联邦学习自愈 (仅采样小范围邻居)"""
        logger.log_step(4, "进入 Level 4 联邦学习自愈层")

        healthy_entries = [
            info for info in neighbors_status if info.get("health", 0) >= 0.8
        ]
        healthy_entries.sort(key=lambda x: x.get("distance", float("inf")))
        total_healthy = len(healthy_entries)

        if total_healthy <= 100:
            sample_ratio = 0.20
        elif total_healthy <= 500:
            sample_ratio = 0.10
        else:
            sample_ratio = 0.05

        sample_count = max(3, int(total_healthy * sample_ratio))
        sample_count = min(sample_count, total_healthy)

        sampled_entries = healthy_entries[:sample_count]
        sampled_ids = [info["id"] for info in sampled_entries]

        logger.log_step(
            4,
            f"联邦邻居采样: 从 {total_healthy} 个健康邻居中选取最近 {sample_count} 个 ({sample_ratio*100:.0f}%)",
            detail=f"参与节点: {', '.join(sampled_ids[:10])}{'...' if len(sampled_ids) > 10 else ''}",
        )

        base_state_dict = None
        if target_node.q_network is not None:
            base_state_dict = target_node.q_network.state_dict()
        elif target_node.baseline_q_network_state is not None:
            base_state_dict = target_node.baseline_q_network_state

        if base_state_dict is None:
            logger.log_step(4, "致命错误: 无基础模型, 联邦学习无法启动", success=False)
            return HealingResult(
                node_id=self.node_id, success=False, healing_level=4,
                message="无基础模型架构, 联邦学习自愈失败", logger=logger,
            )

        result = self.level4_federated.process_healing(
            base_state_dict=base_state_dict,
            healthy_neighbors=sampled_ids,
            logger=logger,
        )
        result.healing_level = 4
        return result

    # ==================================================================
    # 结果收尾
    # ==================================================================

    def _finalize(self, logger: HealingLogger, result: HealingResult, total_start_time: float) -> None:
        total_elapsed = time.time() - total_start_time
        result.healing_time = total_elapsed
        result.logger = logger

        if result.success:
            logger.log_result(result.healing_level, True,
                              f"自愈成功 (解决于 Level {result.healing_level})，总耗时: {total_elapsed:.4f}s")
        else:
            logger.log_result(result.healing_level, False,
                              f"自愈全流程失败，需维持隔离并进行物理级干预，总耗时: {total_elapsed:.4f}s")

        print("\n" + logger.export_report())
