"""
自愈过程专用日志管理子模块。

负责记录每一步自愈决策与动作，并支持导出为结构化文本报告。
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# 攻击类型标签映射 (与 consensus_engine/models.py 保持一致)
# ---------------------------------------------------------------------------
ATTACK_TYPE_LABELS: Dict[int, str] = {
    0: "无攻击",
    1: "StateObservationAttack",
    2: "ActionAttack",
    3: "StateTransferAttack",
    4: "RewardAttack",
    5: "ExperiencePoolAttack",
    6: "ModelTampAttack",
}


@dataclass
class HealingLogEntry:
    """一条自愈日志记录"""
    timestamp: float                        # 记录时间 (相对于自愈开始)
    level: int                              # 当前处于第几级自愈 (1/2/3)
    level_label: str                        # "冗余自愈" / "免疫自愈" / "联邦学习自愈"
    action: str                             # 执行的动作描述
    detail: str = ""                        # 详细补充信息
    source_node: Optional[str] = None       # 资源来源节点 ("self" 或 邻居 SID)
    success: Optional[bool] = None          # 该步骤是否成功


LEVEL_LABELS = {1: "冗余自愈", 2: "免疫自愈", 3: "批量冗余自愈", 4: "联邦学习自愈"}


class HealingLogger:
    """
    自愈日志管理器。

    每个 SatelliteNode 的一次自愈流程对应一个 HealingLogger 实例。
    """

    def __init__(self, node_id: str, fail_score: float, attack_type: Optional[int] = None):
        self.node_id = node_id
        self.fail_score = fail_score
        self.attack_type = attack_type
        self.attack_label = ATTACK_TYPE_LABELS.get(attack_type, f"Unknown({attack_type})")
        self.entries: List[HealingLogEntry] = []
        self._start_time = time.time()
        self.final_result: Optional[str] = None
        self.final_level: Optional[int] = None
        self.final_success: bool = False

    # ----- 记录接口 -----

    def log_step(
        self,
        level: int,
        action: str,
        detail: str = "",
        source_node: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> None:
        """记录一个自愈步骤"""
        entry = HealingLogEntry(
            timestamp=time.time() - self._start_time,
            level=level,
            level_label=LEVEL_LABELS.get(level, f"Level-{level}"),
            action=action,
            detail=detail,
            source_node=source_node,
            success=success,
        )
        self.entries.append(entry)

        # 同步打印到控制台
        src = f" [来源: {source_node}]" if source_node else ""
        status = ""
        if success is True:
            status = " [OK]"
        elif success is False:
            status = " [FAIL]"
        print(f"  [{self.node_id}][{entry.level_label}] {action}{src}{status}")
        if detail:
            print(f"    → {detail}")

    def log_result(self, level: int, success: bool, message: str) -> None:
        """记录最终结果"""
        self.final_level = level
        self.final_success = success
        self.final_result = message
        self.log_step(level, f"结果: {message}", success=success)

    # ----- 导出接口 -----

    def export_report(self) -> str:
        """导出完整的结构化文本日志报告"""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f" 自愈过程日志报告 — 节点: {self.node_id}")
        lines.append("=" * 72)
        lines.append(f"  失效评分 (FailScore) : {self.fail_score:.4f}")
        lines.append(f"  攻击类型 (AttackType): {self.attack_type} ({self.attack_label})")
        lines.append(f"  总步骤数             : {len(self.entries)}")
        total_time = time.time() - self._start_time
        lines.append(f"  总耗时               : {total_time:.4f} 秒")
        lines.append(f"  最终结果             : {'成功' if self.final_success else '失败'}")
        if self.final_result:
            lines.append(f"  结果描述             : {self.final_result}")
        lines.append("-" * 72)

        for i, entry in enumerate(self.entries, 1):
            status = "[OK]" if entry.success is True else ("[FAIL]" if entry.success is False else "[*]")
            src = f"  来源: {entry.source_node}" if entry.source_node else ""
            lines.append(f"  [{i:02d}] +{entry.timestamp:07.4f}s | {entry.level_label} | {status} {entry.action}{src}")
            if entry.detail:
                lines.append(f"       └─ {entry.detail}")

        lines.append("=" * 72)
        return "\n".join(lines)

    def export_dict(self) -> Dict[str, Any]:
        """导出为字典格式 (便于 JSON 序列化)"""
        return {
            "node_id": self.node_id,
            "fail_score": self.fail_score,
            "attack_type": self.attack_type,
            "attack_label": self.attack_label,
            "total_time": time.time() - self._start_time,
            "final_success": self.final_success,
            "final_result": self.final_result,
            "final_level": self.final_level,
            "steps": [
                {
                    "timestamp": e.timestamp,
                    "level": e.level,
                    "level_label": e.level_label,
                    "action": e.action,
                    "detail": e.detail,
                    "source_node": e.source_node,
                    "success": e.success,
                }
                for e in self.entries
            ],
        }
