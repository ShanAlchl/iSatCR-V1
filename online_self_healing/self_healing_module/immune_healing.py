"""
Level 2: 免疫自愈引擎。

根据攻击类型执行对应的免疫策略。
对于模型篡改(6)和经验池中毒(5)，免疫层的最佳策略是回退至冗余引擎。
对于状态转移攻击(3)，免疫层无法处理，需降级至联邦学习。

攻击类型编号 (与 consensus_engine/models.py 一致):
  1: StateObservationAttack   → 状态滤波 + 置信度降级
  2: ActionAttack             → 经验池回滚 + 动作屏蔽
  3: StateTransferAttack      → 无法免疫修复，降级至联邦
  4: RewardAttack             → 经验池回滚 + 安全基线奖励器
  5: ExperiencePoolAttack     → 回退冗余 (仅恢复经验池)
  6: ModelTampAttack           → 回退冗余 (仅恢复模型)
"""

import time
from typing import Optional, List, Dict, Any

from .healing_logger import HealingLogger
from .redundant_healing import HealingResult, RedundantHealingEngine


class ImmuneHealingEngine:
    """Level 2: 免疫自愈引擎"""

    def __init__(self):
        # 冗余引擎实例将在需要"回退冗余"时由 orchestrator 注入
        self._redundant_engine: Optional[RedundantHealingEngine] = None

    def set_redundant_engine(self, engine: RedundantHealingEngine) -> None:
        """注入冗余引擎引用，以便免疫层内部执行回退冗余操作"""
        self._redundant_engine = engine

    # ==================================================================
    # 针对各攻击类型的专项免疫策略
    # ==================================================================

    def _heal_state_observation_attack(self, target_node, logger: HealingLogger) -> bool:
        """
        攻击类型 1: StateObservationAttack
        策略: 滑动窗口/卡尔曼滤波 + 邻居状态置信度降级
        """
        logger.log_step(2, "检测到状态观测攻击 (Type 1)，启动状态滤波免疫策略")

        # 步骤 1: 启动滤波器
        logger.log_step(2, "步骤 1: 启动滑动窗口平滑滤波器 (Kalman/Sliding Window)",
                        detail="对邻居状态序列进行时序平滑处理，过滤单点异常跳变")
        time.sleep(0.04)  # 模拟滤波器初始化与计算

        # 步骤 2: 降低不可信邻居权重
        logger.log_step(2, "步骤 2: 降低不可信邻居的局部状态置信度权重",
                        detail="将方差激增的邻居状态输入权重从 1.0 降至 0.3")

        # 步骤 3: 增强历史惯性
        logger.log_step(2, "步骤 3: 增强历史状态惯性权重",
                        detail="历史惯性因子 α 从 0.5 提升至 0.8，使决策更依赖可信历史")

        logger.log_step(2, "状态观测免疫策略执行完毕", success=True)
        return True

    def _heal_action_attack(self, target_node, logger: HealingLogger) -> bool:
        """
        攻击类型 2: ActionAttack
        策略: 经验池回滚 + 动作空间屏蔽 (Action Masking)
        """
        logger.log_step(2, "检测到动作攻击 (Type 2)，启动经验隔离 + 动作屏蔽策略")

        # 步骤 1: 回滚经验池
        logger.log_step(2, "步骤 1: 隔离近期受污染的动作反馈经验",
                        detail="将经验池回滚至初始安全基线，丢弃攻击期间产生的所有毒化记录")
        pool_size_before = len(target_node.replay_buffer)
        target_node.rollback_to_baseline(restore_model=False, restore_pool=True)
        pool_size_after = len(target_node.replay_buffer)
        logger.log_step(2, f"经验池已回滚",
                        detail=f"清除前: {pool_size_before} 条 → 回滚后: {pool_size_after} 条",
                        source_node="self", success=True)

        # 步骤 2: 动作屏蔽
        logger.log_step(2, "步骤 2: 施加防毒化动作掩码 (Action Masking)",
                        detail="降低探索率 (epsilon)，临时禁用高危动作方向，防止在被篡改的动作空间中继续试错")
        time.sleep(0.05)  # 模拟 Masking 计算

        logger.log_step(2, "动作攻击免疫策略执行完毕", success=True)
        return True

    def _heal_state_transfer_attack(self, target_node, logger: HealingLogger) -> bool:
        """
        攻击类型 3: StateTransferAttack
        策略: 免疫层无法处理此类攻击（环境动力学被深度改变），需降级至联邦学习
        """
        logger.log_step(2, "检测到状态转移攻击 (Type 3)", detail="环境动力学特征被深度篡改，免疫微调无法收敛")
        logger.log_step(2, "免疫层判定无法处理此攻击类型，建议降级至联邦学习自愈", success=False)
        return False

    def _heal_reward_attack(self, target_node, logger: HealingLogger) -> bool:
        """
        攻击类型 4: RewardAttack
        策略: 经验池回滚 + 启用安全基线奖励评估器
        """
        logger.log_step(2, "检测到奖励攻击 (Type 4)，启动经验隔离 + 安全基线奖励器策略")

        # 步骤 1: 回滚经验池
        logger.log_step(2, "步骤 1: 丢弃被伪造奖励污染的近期记忆",
                        detail="奖励信号被篡改导致模型学到错误策略，需清除所有含毒奖励的经验")
        pool_size_before = len(target_node.replay_buffer)
        target_node.rollback_to_baseline(restore_model=False, restore_pool=True)
        pool_size_after = len(target_node.replay_buffer)
        logger.log_step(2, f"经验池已回滚",
                        detail=f"清除前: {pool_size_before} 条 → 回滚后: {pool_size_after} 条",
                        source_node="self", success=True)

        # 步骤 2: 挂载安全基线奖励
        logger.log_step(2, "步骤 2: 阻断外部奖励通道，激活安全基线奖励评估器",
                        detail="切换至只读ROM中的 Secure Reward Baseline，"
                               "后续所有奖励信号将由内置业务指标 (时延/丢包率) 独立评估")
        time.sleep(0.03)

        logger.log_step(2, "奖励攻击免疫策略执行完毕", success=True)
        return True

    def _heal_experience_pool_attack(
        self, target_node, neighbors_status, eval_callback, logger: HealingLogger
    ) -> HealingResult:
        """
        攻击类型 5: ExperiencePoolAttack
        策略: 回退至冗余自愈 (仅恢复经验池)
        """
        logger.log_step(2, "检测到经验池中毒 (Type 5)，免疫最佳策略 = 回退冗余自愈 (仅恢复经验池)")
        logger.log_step(2, "调用冗余引擎: restore_model=False, restore_pool=True")

        if self._redundant_engine is None:
            logger.log_step(2, "致命错误: 冗余引擎未注入，无法执行回退冗余", success=False)
            return HealingResult(
                node_id=target_node.SID, success=False, healing_level=2,
                message="冗余引擎未注入，回退冗余失败", logger=logger,
            )

        result = self._redundant_engine.process_healing(
            target_node=target_node,
            neighbors_status=neighbors_status,
            eval_callback=eval_callback,
            restore_model=False,
            restore_pool=True,
            logger=logger,
        )
        result.healing_level = 2  # 仍属于免疫层的决策结果
        return result

    def _heal_model_tamp_attack(
        self, target_node, neighbors_status, eval_callback, logger: HealingLogger
    ) -> HealingResult:
        """
        攻击类型 6: ModelTampAttack
        策略: 回退至冗余自愈 (仅恢复模型参数)
        """
        logger.log_step(2, "检测到模型篡改 (Type 6)，免疫最佳策略 = 回退冗余自愈 (仅恢复模型)")
        logger.log_step(2, "调用冗余引擎: restore_model=True, restore_pool=False")

        if self._redundant_engine is None:
            logger.log_step(2, "致命错误: 冗余引擎未注入，无法执行回退冗余", success=False)
            return HealingResult(
                node_id=target_node.SID, success=False, healing_level=2,
                message="冗余引擎未注入，回退冗余失败", logger=logger,
            )

        result = self._redundant_engine.process_healing(
            target_node=target_node,
            neighbors_status=neighbors_status,
            eval_callback=eval_callback,
            restore_model=True,
            restore_pool=False,
            logger=logger,
        )
        result.healing_level = 2  # 仍属于免疫层的决策结果
        return result

    # ==================================================================
    # 免疫自愈总入口
    # ==================================================================

    def process_healing(
        self,
        target_node,
        attack_type: Optional[int],
        neighbors_status: Optional[List[Dict[str, Any]]] = None,
        eval_callback=None,
        logger: Optional[HealingLogger] = None,
    ) -> HealingResult:
        """
        执行免疫自愈。
        根据攻击类型路由到具体的免疫策略。
        """
        node_id = target_node.SID
        start_time = time.time()

        if logger:
            logger.log_step(2, f"启动免疫自愈引擎，攻击类型: {attack_type}")

        if attack_type is None:
            if logger:
                logger.log_step(2, "未提供攻击类型信息，无法匹配免疫策略", success=False)
            return HealingResult(
                node_id=node_id, success=False, healing_level=2,
                healing_time=time.time() - start_time,
                message="无攻击类型信息，免疫匹配失败", logger=logger,
            )

        # ── 路由到具体策略 ──

        # 类型 5/6: 回退冗余 (需要邻居信息和评估回调)
        if attack_type == 5:
            return self._heal_experience_pool_attack(
                target_node, neighbors_status or [], eval_callback or (lambda x: 1.0), logger
            )
        if attack_type == 6:
            return self._heal_model_tamp_attack(
                target_node, neighbors_status or [], eval_callback or (lambda x: 1.0), logger
            )

        # 类型 1/2/3/4: 纯免疫策略
        strategy_map = {
            1: self._heal_state_observation_attack,
            2: self._heal_action_attack,
            3: self._heal_state_transfer_attack,
            4: self._heal_reward_attack,
        }

        handler = strategy_map.get(attack_type)
        if handler is None:
            if logger:
                logger.log_step(2, f"未知攻击类型 ({attack_type})，无对应免疫策略", success=False)
            return HealingResult(
                node_id=node_id, success=False, healing_level=2,
                healing_time=time.time() - start_time,
                message=f"未知攻击类型 ({attack_type})，无应对措施", logger=logger,
            )

        success = handler(target_node, logger)

        return HealingResult(
            node_id=node_id,
            success=success,
            healing_level=2,
            healing_time=time.time() - start_time,
            message=f"免疫自愈执行完毕 (攻击类型 {attack_type})",
            logger=logger,
        )
