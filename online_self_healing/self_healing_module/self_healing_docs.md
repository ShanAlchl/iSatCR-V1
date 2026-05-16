# iSatCR-V1-part3 自愈模块与可视化系统说明文档

本文档介绍了 `online_self_healing` 目录下核心模型自愈模块的功能、测试脚本、可视化引擎以及底层数据结构的变更。

---

## 1. 核心自愈模块 (`self_healing_module/`)

该目录实现了 **四级分级模型自愈架构**，针对不同失效风险级别自动路由并应用对应的恢复策略。

### 核心组件
* **`healing_orchestrator.py` (分级自愈总控调度器)**
  * 作为入口接收所有受感染（失效）节点，并基于 **FailScore（失效评分）** 和 **建链关系失效比例** 判定风险等级：
    * **低风险 (Level 1)**: FailScore < 0.40 -> **单点冗余自愈**
    * **中风险 (Level 2)**: 0.40 ≤ FailScore < 0.50 且 建链比例 ≤ 25% -> **免疫自愈**
    * **批量中风险 (Level 3)**: 0.40 ≤ FailScore < 0.50 且 建链比例 > 25% -> **批量冗余自愈**
    * **高风险 (Level 4)**: FailScore ≥ 0.50 -> **联邦学习自愈**
  * 具备自动降级机制（例如 Level 1 恢复失败时会自动降级到 Level 2，最后保底为 Level 4）。
* **`redundant_healing.py` (Level 1: 冗余自愈)**
  * 优先使用节点自身的 `baseline_q_network_state` (本地安全快照) 进行回滚。
  * 如果自身快照损坏或不可用，则向距离最近的健康邻居节点请求参数备份进行恢复。
* **`immune_healing.py` (Level 2: 免疫自愈)**
  * 基于预定义的攻击类型应用相应的动作边界约束和安全规则。对安全参数进行微调。
* **`batch_redundant_healing.py` (Level 3: 批量冗余自愈)**
  * 处理聚集性的区域失效。采用改进的 PBFT 选举领导者 -> 领导者提取基线模型 -> 星间链路组播下发 -> 并行验证与分批解封。
* **`federated_healing.py` (Level 4: 联邦学习自愈)**
  * 针对最严重或前面级别都无法恢复的场景，采样附近的高健康度邻居，执行联邦聚合（FedAvg）重新初始化模型。
* **`healing_logger.py`**
  * 统一结构化日志记录器，将自愈的每一个阶段输出为 `.json` 格式的日志，以供后续可视化引擎渲染。

---

## 2. 可视化引擎 (`consensus_engine/`)

### `explorer.py` (基础拓扑可视化)
* **功能**: 负责星座整体网络拓扑和共识状态的 3D 可视化展示。基于 Plotly 生成交互式的 HTML 文件，能够直观地展示各节点的轨道面分布及其相互之间的物理建链关系。

### `healing_explorer.py` (自愈过程专属可视化)
* **功能**: 专门用于读取和解析自愈模块生成的 `_log.json` 文件。
* **使用体验**: 
  * 结合 TLE 数据生成卫星星座的 3D 拓扑。
  * 在 3D 图表中用不同颜色或标识区分 **健康节点**、**失效节点** 和 **恢复成功节点**。
  * **交互式面板**: 用户可以在生成的 HTML 中点击具体的失效节点，界面侧边栏会完整展示该节点在自愈过程中记录的所有日志步骤（如：何时发现快照损坏、从哪个邻居获取资源、经历了几个自愈层级以及各阶段耗时）。

---

## 3. 测试与演示脚本 (`test/demo_self_healing_scenarios.py`)

* **功能**: 这是完整的四级自愈功能端到端测试与演示脚本。
* **测试场景**:
  * 精简配置了 4 组核心测试，每组对应一个自愈层级：
    1. **Level 1 测试**: StateObservation 攻击，模拟极小规模感染，并 **强制禁用自身快照** 以验证“邻居节点备份恢复”的回退逻辑。
    2. **Level 2 测试**: Action 攻击，中等评分。
    3. **Level 3 测试**: Reward 攻击，刻意制造连续两个轨道面大量节点失效（建链失效比例 > 25%），验证批量多播自愈和选举机制。
    4. **Level 4 测试**: ExperiencePool 攻击，高危评分，验证基于局部健康邻居的联邦聚合算法。
* **使用方法**:
  ```bash
  cd online_self_healing/test
  python demo_self_healing_scenarios.py
  ```
* **输出位置**: 测试运行完毕后，会在 `online_self_healing/result/scenarios/` 下按等级生成对应的 `*_log.json` 日志文件和 `*_viz.html` 3D 可视化页面。

---

## 4. 底层基类修改 (`node.py`)

为了支持四级自愈体系，底层的 `SatelliteNode` 类进行了一系列升级：

* **自愈数据字段扩展**:
  * 引入了 `HealthScore`（健康度评分）属性，作为分级调度的主要判断依据之一。
  * 引入了 `baseline_q_network_state` (基线网络参数状态) 和 `baseline_replay_buffer` (基线经验池)。
  * 引入了布尔值 `snapshot_available` (快照可用性)，用于模拟安全备份的完整性。
* **新方法 `rollback_to_baseline`**:
  * 提供了一键恢复本地基线的接口。当该方法检测到 `snapshot_available == False` 时会自动拒绝本地回滚，迫使上层调度器执行基于邻居的数据同步。
* **风险标签校验 (`VALID_RISK_LEVELS`)**:
  * 新增了一套校验集合，确保传入网络中的各种失效等级判定保持格式和命名规范的统一。

---

## 5. 模块调用示例与使用方法

这里为您提供在主控程序中直接调用自愈模块和可视化引擎的代码示例。

### 5.1 单点与批量分级自愈调用 (总控入口)

引入 `TieredHealingOrchestrator`，它可以处理单点节点的自愈，也可以一次性处理一批失效节点（批量模式，系统会自动计算建链比例以触发 Level 3）。

**单点自愈调用示例：**
```python
from online_self_healing.self_healing_module.healing_orchestrator import TieredHealingOrchestrator

# 1. 初始化单点编排器
orchestrator = TieredHealingOrchestrator("Satellite_1100_1_1")

# 2. 单点自愈调用
result = orchestrator.execute_tiered_healing(
    target_node=infected_node,             # 需要恢复的 SatelliteNode 实例
    fail_score=0.45,                       # 该节点的失效评分 (由 Part2 提供)
    attack_type=2,                         # 攻击类型标识
    neighbors_status=all_neighbors_status, # 当前全网所有邻居的最新状态列表
    link_ratio=0.1                         # 失效建链比例（可选，单点模式可默认为0）
)

if result.success:
    print(f"节点恢复成功，最终解决层级: Level {result.healing_level}")
```

**批量自愈调用示例：**
```python
# 1. 针对批量处理场景，实例化一个全局编排器
batch_orchestrator = TieredHealingOrchestrator("BATCH_CONTROLLER")

# 2. 传入所有感染节点字典及其对应的评分/攻击类型
batch_results = batch_orchestrator.execute_batch_healing(
    failed_nodes={sid: nodes[sid] for sid in infected_sids},
    fail_scores=fail_scores_dict,          # {sid: score}
    attack_types=attack_types_dict,        # {sid: type}
    all_nodes=nodes,                       # 星座全网节点字典
    neighbors_status=all_neighbors_status  # 全量邻居状态（需包含失效节点，用于内部计算建链比例）
)

# 3. 遍历获取各节点的恢复结果
for sid, res in batch_results.items():
    print(f"节点 {sid} 恢复状态: {res.success}, 恢复等级: {res.healing_level}")
```

### 5.2 可视化引擎调用 (渲染 3D 图表)

自愈模块在执行过程中会利用 `HealingLogger` 将完整的恢复步骤输出为 `.json` 格式，随后即可使用 `HealingConstellationExplorer` 进行可视化渲染：

```python
from pathlib import Path
from online_self_healing.consensus_engine.healing_explorer import HealingConstellationExplorer

# 1. 定义 TLE 轨道数据路径和刚生成的日志路径
tle_path = Path("Satellite_Data/Delta_18_24_50_1150_5.txt")
log_path = Path("result/scenarios/high/attack5_ExperiencePool_scale_small_log.json")

# 2. 初始化自愈可视化器
explorer = HealingConstellationExplorer(
    tle_path=tle_path, 
    log_path=log_path
)

# 3. 执行渲染并保存为交互式 HTML 页面
output_html_path = "result/scenarios/high/attack5_ExperiencePool_scale_small_viz.html"
explorer.visualize(save_path=output_html_path)
```
