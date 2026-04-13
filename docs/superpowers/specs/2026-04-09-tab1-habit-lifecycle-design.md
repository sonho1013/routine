# Tab1 Habit Lifecycle & Scene Card State Machine — Design Spec

> **Date:** 2026-04-09
> **Scope:** Tab1（数据后台 / 学习阶段）
> **Based on:** `2026-04-08-tab1-vehicle-integration-brainstorm.md` Round 1-9（全部锁定）
> **Status:** Draft → 待用户 review → 进入 writing-plans

---

## 1. 目标与非目标

### 1.1 本 spec 要解决的问题

当前 Tab1 pipeline 每次 `ingest_signal_batch` 全量重跑，产出的 `HABIT` 存入 ChromaDB 之后没有"生命周期"概念：新 habit 不会与旧 habit 对应、偏移无法识别、accepted 状态可能在重算中丢失、场景卡只是内存里临时对象。

本 spec 引入三项结构性改造：

1. **Signal→Fact 翻译从 if/elif 硬编码改为 yaml-driven** — 为 drift 检测的"每信号度量"提供 metadata 基础
2. **场景卡升级为第一等持久实体 + 四状态状态机** — `pending / accepted / recommendation / retired`，持久化到 SQLite
3. **每批次 drift detection** — 识别 accepted 卡的内容漂移并生成 recommendation 卡，严格遵循"用户主权"原则

### 1.2 非目标（本 spec 不做的事）

本 spec **不**涉及：

- **实车信号接入层**（Round 1 决策 D）：tab1 继续消费 `scenarios/mock_data_generator.py` 产出的 mockup 数据；不写 CAN/Someip/DDS 订阅框架；只预留"日志批处理适配层"的接口位置
- **窗口滑动 / 增量聚类**（Round 3 决策 A）：当前数据规模 ≤ 30 天 × ~100 信号/天 ≈ 3000 条，O(N²) 全量重算完全可承受
- **Tab2 UI**：本 spec 只负责产出 recommendation 卡的数据信号；Tab2 如何展示、如何让用户确认是 Tab2 的职责
- **车端迁移相关设计**（用户明确 "长时间内不讨论"）：SQLite/Chroma/OpenAI/GPT-4.1 的选型按"demo 正确性优先"做，不为车端友好性打折
- **Split / merge drift 的自动识别**（Round 9 Q9.1 决策 A）：只做 `modify` drift；split/merge 表现为"旧 accepted 失去 reinforce + 新 pending 独立出现"，不建立语义链接
- **Recommendation 替换二次确认弹框**（Round 8 Q8.1 决策 A）：用户接受 recommendation 即刻生效，不走弹框确认
- **Dormant accepted 卡的 UI 可视化**（Round 8 Q8.4 决策 A）：数据层只暴露 `last_reinforced_batch_id` 原始数据

### 1.3 核心产品原则（Round 5 锁定）

**用户主权绝对化**是本系统的第一性约束，所有设计决策必须服从：

1. Accepted 卡一经锁定，系统不得做任何静默修改 — 不换名字、不换内容、不换身份、不自动 GC
2. Drift 的归宿是"推荐卡"，不是"替换"。系统只能生成 recommendation，旧 accepted 保留，由用户决定是否接受
3. 只有用户才能关闭 accepted 卡 — 系统没有自动退役权力
4. Pending 阶段允许 GPT 自由涌现命名，一旦 accepted，名字和签名全部冻结

---

## 2. 架构总览

### 2.1 新的批处理数据流

```
mockup JSON (scenarios/mock_data_generator.py)
    │
    ├─→ SignalSimulator  (signals/simulator.py，不变)
    │
    ▼
SignalRuleEngine         ← 新：yaml-driven，替代 signal_to_fact.py 硬编码
    │  (rules.yaml)
    ▼
Fact (type=PREF, StructuredContext, raw_value)
    │
    ▼
FactStoreChroma          (panoramix_core/store/fact_store_chroma.py，不变)
    │                     只存 PREF，HABIT 不再路由到 Chroma
    ▼
HabitsDetector           (panoramix_core/clustering/habits_detector.py，输出路由改变)
    │  Hybrid DBSCAN + 连续性过滤 + GPT reword
    ▼
HabitLifecycleManager    ← 新：核心
    ├─ structural_key 分组
    ├─ GPT scene naming
    ├─ drift detection (modify only)
    └─ 写入 SceneCardStore + HabitStore (SQLite 事务内)
    │
    ▼
SQLite (storage/habit_memory.db)  ← 新：单文件，两张表
    ├─ scene_cards  (4 状态 state machine)
    └─ habits       (batch_id, 最近 3 批)
```

### 2.2 存储分层（改造后）

| 层 | 存储 | 内容 | 理由 |
|---|---|---|---|
| 向量检索层 | ChromaDB | **仅 PREF facts**（含 ada-002 embedding） | Chroma 擅长相似度检索 |
| 状态机层 | SQLite | `scene_cards` 表 | 状态机查询需要 SQL JOIN + 事务 |
| 批次产物层 | SQLite | `habits` 表（最近 3 批） | drift audit + Tab1 UI 查询 |

**关键变更**：`FactStoreChroma` **不再写 HABIT 类型 Fact**。`HabitsDetector.detect_habits()` 的输出由新的 pipeline 接管，写入 SQLite 而非 Chroma。

### 2.3 模块清单（新增 / 修改 / 删除）

| 路径 | 变更 | 说明 |
|---|---|---|
| `engine/signal_rules/engine.py` | **新增** | SignalRuleEngine 类 |
| `engine/signal_rules/rules.yaml` | **新增** | 11 种现有信号的规则配置 |
| `engine/signal_rules/__init__.py` | **新增** | 包标记 |
| `engine/signal_to_fact.py` | **重写** | 变薄：调用 SignalRuleEngine 并组装 Fact |
| `engine/habit_lifecycle.py` | **新增** | HabitLifecycleManager，包含 drift detection |
| `engine/drift_detection.py` | **新增** | `is_drifted()` + `compute_raw_value_stats()` 纯函数 |
| `panoramix_core/store/scene_card_store.py` | **新增** | SceneCardStore (SQLite 封装) |
| `panoramix_core/store/habit_store.py` | **新增** | HabitStore (SQLite 封装) |
| `panoramix_core/store/sqlite_schema.sql` | **新增** | DDL 定义 |
| `panoramix_core/models/scene_card.py` | **新增** | `SceneCard` Pydantic 模型（对应 scene_cards 表） |
| `panoramix_core/models/habit.py` | **新增** | `Habit` Pydantic 模型（对应 habits 表） |
| `engine/habit_engine.py` | **重写** | `ingest_signal_batch()` 改为两阶段 pipeline |
| `engine/scene_card.py` | **改造** | 原 in-memory dataclass 删除；新增"候选场景卡构造"辅助函数（纯函数，不承载状态） |
| `panoramix_core/store/fact_store_chroma.py` | **微调** | 允许写 HABIT 的路径删除（HABIT 不再经过 Chroma） |

---

## 3. 数据模型

### 3.1 Structural Key 定义

`structural_key` 是场景卡身份的核心。对 pending/recommendation 卡是"当前 batch 算出的值"，对 accepted 卡是"冻结值"。

```python
def compute_structural_key(habits: list[Fact]) -> str:
    """
    habits: 属于同一候选场景卡的 habit Fact 列表
    返回 SHA-256 hex[:16]
    """
    dominant_geofence = _majority([h.context.geofence for h in habits]) or "none"
    dominant_time_bucket = _majority([h.context.time_bucket for h in habits])
    dominant_vehicle_state = _majority([h.context.vehicle_state for h in habits])
    signal_types = sorted(set(
        _extract_signal_from_habit_meta(h) for h in habits
    ))
    key_src = "|".join([
        dominant_geofence,
        dominant_time_bucket,
        dominant_vehicle_state,
        ",".join(signal_types),
    ])
    return hashlib.sha256(key_src.encode()).hexdigest()[:16]
```

说明：
- `dominant_*` = 该维度在 habit 列表中出现最多的值；并列时按字典序取第一个（确定性）
- `signal_types` 是 habit 的 `json_metadata.signal` 字段的去重有序集合（由 Round 2 SignalRuleEngine 写入）
- `weekday` **不参与**结构键：Round 5 的示例（李阿姨的周末出门被切成两张卡）提示 weekday 粒度过细反而引发不必要的 split drift，先不纳入

### 3.2 SQLite 文件位置

`storage/habit_memory.db` — 单文件承载 `scene_cards` + `habits` 两张表。

**注意**：与现有 `storage/memories/<username>/chroma.sqlite3`（Chroma 内部 sqlite 文件）不是同一个，完全独立。

多用户支持：两张表都有 `username` 列；`SceneCardStore(username)` 内部所有查询都带 `WHERE username = ?`。

### 3.3 `scene_cards` 表 schema

一张表 + `status` 字段承载所有状态（Round 8 Q8.1 决策）。

```sql
CREATE TABLE scene_cards (
    -- 主键
    card_id TEXT PRIMARY KEY,              -- UUID
    username TEXT NOT NULL,

    -- 状态机
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'accepted', 'recommendation', 'retired'
    )),

    -- 身份字段
    -- pending/recommendation: 当前 batch 算出的 structural_key
    -- accepted: 用户接受时冻结
    -- retired: 历史快照
    structural_key TEXT NOT NULL,

    -- 显示
    -- pending/recommendation: 每 batch 可刷新
    -- accepted: 用户接受时冻结，永不变
    display_name TEXT NOT NULL,

    -- 冻结快照 (仅 accepted 卡使用，JSON)
    -- 格式见 §3.4
    frozen_content_snapshot_json TEXT,

    -- 当前内容 (pending/recommendation 卡, JSON)
    -- 格式见 §3.5
    content_snapshot_json TEXT,

    -- 替代关系 (recommendation 卡, JSON array of card_id)
    -- 支持 modify (1→1); split/merge 留给 DCM-3
    supersede_candidate_for_json TEXT,

    -- batch tracking
    first_seen_batch_id INTEGER NOT NULL,
    last_reinforced_batch_id INTEGER NOT NULL,

    -- 审计
    created_at TIMESTAMP NOT NULL,
    accepted_at TIMESTAMP,
    retired_at TIMESTAMP
);
-- 无外键：本系统没有 users 表，username 作为逻辑分区字段
-- 所有查询强制带 WHERE username = ?（应用层校验）

CREATE INDEX idx_scene_cards_status_user
    ON scene_cards(username, status);
CREATE INDEX idx_scene_cards_structural_key
    ON scene_cards(username, status, structural_key);
CREATE INDEX idx_scene_cards_last_reinforced
    ON scene_cards(last_reinforced_batch_id);
```

状态不变式（代码层强制检查）：
- `status='accepted'` ⇒ `frozen_content_snapshot_json IS NOT NULL AND accepted_at IS NOT NULL`
- `status='recommendation'` ⇒ `supersede_candidate_for_json IS NOT NULL`
- `status='retired'` ⇒ `retired_at IS NOT NULL`
- `accepted` 卡永远不能被 UPDATE `display_name` / `frozen_content_snapshot_json` / `structural_key`（应用层校验 + 单测覆盖）

### 3.4 `frozen_content_snapshot_json` 格式（accepted 卡）

Accepted 卡自带完整快照，不引用 `habits` 表。这是用户主权的数据层落地：即使 habits 表被清理，accepted 卡的内容依然完整。

```json
{
  "snapshot_batch_id": 42,
  "snapshot_at": "2026-04-15T10:00:00",
  "habits": [
    {
      "habit_text": "set cabin air conditioning temperature to 22 degrees",
      "signal": "hvac_temp_target",
      "raw_value_stats": {
        "type": "numeric",
        "mean": 22.3,
        "std": 0.8,
        "min": 21.0,
        "max": 24.0,
        "count": 15
      }
    },
    {
      "habit_text": "started playing music jazz_playlist",
      "signal": "media_source",
      "raw_value_stats": {
        "type": "categorical",
        "dominant_value": "music",
        "value_counts": {"music": 12, "podcast": 2},
        "count": 14
      }
    }
  ],
  "dominant_context": {
    "time_bucket": "early_morning",
    "vehicle_state": "engine_started",
    "geofence": "home",
    "weekday": true
  }
}
```

### 3.5 `content_snapshot_json` 格式（pending / recommendation 卡）

结构与 §3.4 相同，加两个字段：

```json
{
  ...,
  "habit_ids": ["habit-uuid-1", "habit-uuid-2"],   // 引用 habits 表的 row
  "drifted_signals": ["hvac_temp_target"]          // 仅 recommendation 卡使用
}
```

`drifted_signals` 给 Tab2 UI 用于高亮展示"哪个信号变了"（Round 9 Q9.2 锁定）。

### 3.6 `habits` 表 schema

```sql
CREATE TABLE habits (
    habit_id TEXT PRIMARY KEY,              -- UUID，与 Fact.id 对应
    username TEXT NOT NULL,
    batch_id INTEGER NOT NULL,

    -- 文本内容
    text TEXT NOT NULL,                     -- GPT reword 输出

    -- 身份
    signal_category TEXT NOT NULL,          -- 来自 rules.yaml 的 category
    signal_name TEXT NOT NULL,              -- 原始 signal name
    structural_key TEXT NOT NULL,           -- 方便按 structural_key 聚合

    -- 上下文
    context_time_bucket TEXT NOT NULL,
    context_vehicle_state TEXT NOT NULL,
    context_geofence TEXT,
    context_weekday INTEGER,                -- NULL / 0 / 1

    -- 数值统计
    raw_value_stats_json TEXT NOT NULL,     -- §3.4 中的 raw_value_stats 片段

    -- PREF fact 反向引用（drift detection 需要追溯）
    member_fact_ids_json TEXT NOT NULL,     -- JSON array of pref fact IDs

    -- 当前所属场景卡
    scene_card_id TEXT,                     -- 可空

    -- 审计
    created_at TIMESTAMP NOT NULL,

    FOREIGN KEY (scene_card_id) REFERENCES scene_cards(card_id) ON DELETE SET NULL
);

CREATE INDEX idx_habits_batch_id ON habits(batch_id);
CREATE INDEX idx_habits_username_batch ON habits(username, batch_id);
CREATE INDEX idx_habits_structural_key ON habits(username, batch_id, structural_key);
```

**保留策略**（Round 7 决策）：N=3，每次 batch 结束后 `DELETE FROM habits WHERE batch_id < current_batch_id - 2`。

### 3.7 关键假设：每个 habit 对应唯一 signal

`habits.signal_name` 是 NOT NULL 单值字段，Round 9 Q9.2 的 drift detection 算法也按"per-signal"设计。这基于以下假设：

**HabitsDetector 产出的一个 cluster 内，所有成员 PREF facts 都来自同一 signal_name**。

这个假设在当前 mockup 数据下**经验成立**——因为：
1. `SignalRuleEngine.signals_to_facts()` 为每个 signal 产生一条独立的 PREF fact，每条 fact 的 `text` 对应一个 template（如"set cabin air conditioning temperature to 22 degrees"）
2. Hybrid DBSCAN 的距离 = `α·text_cosine + (1-α)·context_dist`；不同 signal 的 fact text 在 ada-002 embedding 空间里相距很远（"set temperature" vs "play music"），几乎不可能聚到同一个 cluster
3. 现有 tests（`test_dbscan_clustering.py`, `test_e2e_pipeline.py`）从未观察到跨 signal 的混合 cluster

**边界处理**：HabitLifecycleManager 在构造 Habit 前，对每个 cluster 显式检查：

```python
def _assert_cluster_pure_signal(cluster_facts: list[Fact]) -> str:
    """
    返回 cluster 的统一 signal_name；若 cluster 跨 signal 则抛异常。
    """
    signals = {json.loads(f.json_metadata)["signal"] for f in cluster_facts}
    if len(signals) != 1:
        raise MixedSignalClusterError(
            f"cluster has {len(signals)} signal types: {signals}. "
            f"Current drift detection assumes single-signal habits."
        )
    return next(iter(signals))
```

若真发生跨信号 cluster，**batch 失败并抛异常**，不做静默降级。这让假设违反变成显式可观测事件；若未来 mockup 或实车数据导致频繁发生，再讨论拆分策略（候选方案：按 signal 分组二次拆分 cluster、或给 habit schema 加 `signal_names: list`）。

---

## 4. 组件：SignalRuleEngine（Round 2）

### 4.1 目标

- 消除 `engine/signal_to_fact.py` 的 if/elif 硬编码
- 为每个信号声明 `drift_metric` + `drift_threshold`，给 Round 9 的 drift detection 提供元数据
- 人类可读（yaml），为将来对接真实 OEM DBC 别名打底

### 4.2 `rules.yaml` schema

```yaml
# engine/signal_rules/rules.yaml

# 地理围栏 (从 signal_to_fact.py 的 KNOWN_GEOFENCES 迁移)
geofences:
  home:        { lat: 48.8566, lon: 2.3522, radius_m: 200 }
  workplace:   { lat: 48.8738, lon: 2.2950, radius_m: 100 }
  toll_A6:     { lat: 48.7890, lon: 2.3100, radius_m: 100 }
  parking_mall:{ lat: 48.8450, lon: 2.3700, radius_m: 100 }

# 信号规则
signals:
  hvac_temp_target:
    category: numeric
    aliases: []                                # 未来 OEM 别名补在这里
    unit: "celsius"
    value_range: [10, 32]                      # 越界丢弃
    text_template: "set cabin air conditioning temperature to {value} degrees"
    drift_metric: numeric_mean_diff
    drift_threshold: 1.5                       # ±1.5°C

  seat_heating:
    category: numeric
    value_range: [0, 5]
    text_template: "turned on seat heating to level {value}"
    trigger_condition: "value > 0"             # 支持简单表达式
    drift_metric: numeric_mean_diff
    drift_threshold: 1                         # 1 档

  media_volume:
    category: numeric
    value_range: [0, 100]
    text_template: "adjusted media volume to {value} percent"
    trigger_condition: "value > 0"
    drift_metric: numeric_mean_diff
    drift_threshold: 10                        # 10%

  media_source:
    category: categorical
    # 每个值对应一个 template
    value_templates:
      music:   "started playing music {content_id}"
      podcast: "resumed listening to podcast {content_id}"
      radio:   "tuned to radio station {content_id}"
      off:     "stopped all media playback"
    drift_metric: dominant_value_change

  hvac_power:
    category: categorical
    value_templates:
      off: "turned off cabin air conditioning"
    drift_metric: dominant_value_change

  # ... 其余 7 个信号类似 ...

# 上下文信号（Table 4, 映射到 StructuredContext 不产生 Fact.text）
context_signals:
  - gps_latitude
  - gps_longitude
  - engine_status
  - gear_position
  - vehicle_speed
```

**引擎能力**：
1. `load_rules(yaml_path) -> dict`
2. `signals_to_facts(event_data: dict) -> list[Fact]` — 替代当前硬编码函数
3. `get_rule(signal_name: str) -> dict | None` — 供 drift_detection 按信号查 metric/threshold

**关键选择**：
- **未知信号的默认行为**：当前 signal_to_fact 只处理已知 11 个，其他静默丢弃。SignalRuleEngine 保持同样行为——yaml 里没声明就不产生 Fact（不为了"通用性"走 LLM 兜底翻译，Round 2 决策 A 明确排除了 B/C）
- **`value_range` 越界处理**：丢弃 + 记日志。不钳制（clamp），避免悄悄修改原始数据
- **`trigger_condition`**：保留当前 `signal_to_fact.py` 里的语义（如 `seat_heating > 0` 才翻译）。用 `ast.literal_eval` 做安全表达式求值，不用 `eval`

### 4.3 `engine.py` 接口

```python
# engine/signal_rules/engine.py

class SignalRuleEngine:
    def __init__(self, rules_path: str = None):
        """默认加载 engine/signal_rules/rules.yaml"""

    def signals_to_facts(self, event_data: dict) -> list[Fact]:
        """替代 engine.signal_to_fact.signals_to_facts"""

    def get_rule(self, signal_name: str) -> dict | None:
        """供 drift_detection 查询"""

    def list_signals(self) -> list[str]:
        """列出所有配置过的信号（单测用）"""
```

`engine/signal_to_fact.py` 重写为 ~20 行 shim：加载一个全局 `SignalRuleEngine` 单例并委托给它，保持导入路径向后兼容不破坏 `engine/habit_engine.py` 的现有 import。

### 4.4 测试迁移

现有 `tests/test_signal_to_fact.py` 依赖硬编码行为。迁移策略：
1. 保留原单测用来作为"同构输出"的金标准（canonical 规则至少要让原单测全通过）
2. 新增 `tests/test_signal_rule_engine.py`：覆盖 yaml 加载错误、未知信号、value_range 越界、别名解析、drift metric 查询

---

## 5. 组件：SceneCardStore

### 5.1 职责

SQLite 薄封装，只暴露业务意图而非 SQL 细节。**所有 accepted 不变式在这一层强制**。

### 5.2 API

```python
# panoramix_core/store/scene_card_store.py

class SceneCardStore:
    def __init__(self, username: str, db_path: str = "storage/habit_memory.db"):
        """Open or create SQLite, ensure schema"""

    # ── 写入 ──
    def insert_pending(self, card: SceneCard, batch_id: int, *, conn) -> None: ...
    def insert_recommendation(
        self, card: SceneCard, batch_id: int,
        supersede_candidate_for: list[str], *, conn
    ) -> None: ...
    def upsert_pending_by_structural_key(
        self, structural_key: str, card_data: dict, batch_id: int, *, conn
    ) -> str:
        """Q8.3 决策 B: 按 structural_key upsert pending
        命中则更新 content/last_reinforced/display_name，返回已有 card_id
        未命中则 insert，返回新 card_id"""

    def upsert_recommendation_by_target(
        self, target_accepted_id: str, card_data: dict, batch_id: int, *, conn
    ) -> str:
        """Q8.2 决策 B: 按 supersede target 去重
        查 WHERE status='recommendation' AND target 在 json array 中
        命中 upsert，未命中 insert"""

    # ── 状态转换 ──
    def accept(self, card_id: str) -> None:
        """pending/recommendation → accepted
        冻结 frozen_content_snapshot_json + display_name + structural_key
        若原为 recommendation：retire 其所有 supersede_candidate_for 指向的 accepted 卡"""

    def retire(self, card_id: str) -> None:
        """accepted → retired
        允许的调用路径：
          1. 外部显式调用（用户主动关闭 accepted 卡）
          2. accept() 内部级联调用（接受 recommendation 时 retire 被替代的 accepted 卡）
        禁止路径：任何清理/定时/维护函数不得调用 retire() 作用于 accepted 卡"""

    def reject_recommendation(self, card_id: str) -> None:
        """recommendation → 硬删除（拒绝即抛弃）"""

    # ── 清理（批处理内部用）──
    def delete_stale_pending(
        self, batch_id: int, *, conn
    ) -> int:
        """Step 7: 删除 last_reinforced_batch_id < 当前的 pending 卡"""

    def delete_stale_recommendation(
        self, batch_id: int, *, conn
    ) -> int:
        """Step 8 决策 A2: 硬删除陈腐 recommendation"""

    def update_last_reinforced(
        self, card_id: str, batch_id: int, *, conn
    ) -> None:
        """Step 6b: reinforce accepted 卡"""

    # ── 查询 ──
    def get_by_status(self, status: str) -> list[SceneCard]: ...
    def get_by_structural_key(
        self, structural_key: str, status: str = None
    ) -> list[SceneCard]: ...
    def find_recommendation_by_target(
        self, accepted_card_id: str
    ) -> SceneCard | None: ...
    def get_all_accepted(self) -> list[SceneCard]: ...

    # ── 事务 ──
    @contextmanager
    def transaction(self):
        """BEGIN; yield conn; COMMIT or ROLLBACK on exception"""
```

**不变式强制点**：
- `accept()` 在冻结之前做 `assert card.status in ('pending', 'recommendation')`
- 任何尝试 UPDATE accepted 卡的 display_name 或 frozen_content_snapshot 的路径都 raise `SceneCardImmutableError`
- `retire()` 必须由 `accept()` 间接调用或外部显式调用，内部清理函数不得调用 `retire()` 作用于 accepted 卡

### 5.3 `SceneCard` Python 模型

当前 `engine/scene_card.py` 里有一个同名 dataclass。重构策略：

- 新建 `panoramix_core/models/scene_card.py`：Pydantic BaseModel，字段严格对应表 schema
- `engine/scene_card.py` 原 dataclass 改为"从 habits 列表构造候选场景卡"的辅助函数，不再承载状态

```python
# panoramix_core/models/scene_card.py

class SceneCard(BaseModel):
    card_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    status: Literal['pending', 'accepted', 'recommendation', 'retired']
    structural_key: str
    display_name: str
    frozen_content_snapshot: dict | None = None      # 仅 accepted
    content_snapshot: dict | None = None              # 仅 pending/recommendation
    supersede_candidate_for: list[str] | None = None  # 仅 recommendation
    first_seen_batch_id: int
    last_reinforced_batch_id: int
    created_at: datetime
    accepted_at: datetime | None = None
    retired_at: datetime | None = None

    # SQLite 行 ↔ model 双向转换
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'SceneCard': ...
    def to_row_dict(self) -> dict: ...
```

---

## 6. 组件：HabitStore

类似 SceneCardStore，职责窄得多。

```python
# panoramix_core/store/habit_store.py

class HabitStore:
    def __init__(self, username: str, db_path: str = "storage/habit_memory.db"):
        """共享同一个 SQLite 文件"""

    def insert_many(
        self, habits: list[Habit], batch_id: int, *, conn
    ) -> None: ...

    def get_by_batch(self, batch_id: int) -> list[Habit]: ...
    def get_latest_batch_id(self) -> int:
        """供 Step 0 计算下一个 batch_id"""

    def allocate_next_batch_id(self, *, conn) -> int:
        """SELECT MAX(batch_id) + 1，在事务内保证原子"""

    def delete_old_batches(
        self, current_batch_id: int, retain_n: int = 3, *, conn
    ) -> int:
        """Step 9: DELETE WHERE batch_id < current - retain_n + 1"""
```

`Habit` Pydantic model 对应 `habits` 表字段，与 `Fact(type=HABIT)` 不共享身份（habits 表独立 id 空间，但字段里带 `member_fact_ids_json` 反向引用 PREF facts 在 Chroma 的 id）。

---

## 7. 组件：HabitLifecycleManager（drift detection 核心）

### 7.1 职责

协调每个 batch 的 Step 5 "分类候选"：对每一张候选场景卡判断它属于哪种情况。

### 7.2 候选场景卡分类算法

给定一批 new habits（来自 HabitsDetector）+ 所有现存 accepted scene cards：

```python
def classify_candidate(
    candidate_habits: list[Habit],            # 同一 structural_key 分组
    accepted_cards: list[SceneCard],
    signal_rule_engine: SignalRuleEngine,
) -> ClassificationResult:
    """
    返回 ClassificationResult 之一：
      - Reinforce(card_id)     — 匹配到 accepted 且内容未变
      - ModifyDrift(card_id, drifted_signals)  — 匹配到 accepted 但内容变了
      - NewPending(structural_key)              — 没有匹配
    """
    key = compute_structural_key(candidate_habits)

    # Step A：严格主键匹配
    matched_accepted = [
        c for c in accepted_cards if c.structural_key == key
    ]

    if not matched_accepted:
        return NewPending(structural_key=key)

    # 正常情况：main accepted 卡最多 1 张
    # 多张的情况（split 场景已完成，新 candidate 又合并了）暂归为 NewPending
    if len(matched_accepted) > 1:
        logger.warning(
            f"structural_key {key} matches multiple accepted cards "
            f"({len(matched_accepted)}); treating as NewPending (DCM-3)"
        )
        return NewPending(structural_key=key)

    accepted = matched_accepted[0]

    # Step B：按信号判漂移（Round 9 Q9.2）
    drifted = detect_drift(
        candidate_habits=candidate_habits,
        frozen_snapshot=accepted.frozen_content_snapshot,
        rule_engine=signal_rule_engine,
    )

    if drifted:
        return ModifyDrift(card_id=accepted.card_id, drifted_signals=drifted)
    else:
        return Reinforce(card_id=accepted.card_id)
```

### 7.3 `detect_drift()` 算法

对 candidate 和 frozen_snapshot 按 `signal_name` 求交集，逐信号判断；任一信号漂移即全卡漂移（Round 9 Q9.2.c）。

```python
def detect_drift(
    candidate_habits: list[Habit],
    frozen_snapshot: dict,
    rule_engine: SignalRuleEngine,
) -> list[str]:
    """
    返回发生漂移的 signal name 列表；空列表表示 reinforce。
    """
    old_by_signal = {
        h["signal"]: h["raw_value_stats"]
        for h in frozen_snapshot["habits"]
    }
    new_by_signal = {
        h.signal_name: json.loads(h.raw_value_stats_json)
        for h in candidate_habits
    }

    drifted = []
    for signal, new_stats in new_by_signal.items():
        if signal not in old_by_signal:
            # 候选里多出一个旧快照没有的信号
            # → structural_key 应该已经不同 → 不应该走到 detect_drift
            # 防御性地当作漂移
            drifted.append(signal)
            continue

        if is_drifted(signal, old_by_signal[signal], new_stats, rule_engine):
            drifted.append(signal)

    return drifted


def is_drifted(signal, old_stats, new_stats, rule_engine) -> bool:
    rule = rule_engine.get_rule(signal) or {}
    metric = rule.get('drift_metric', 'embedding_fallback')

    if metric == 'numeric_mean_diff':
        threshold = rule['drift_threshold']
        return abs(new_stats['mean'] - old_stats['mean']) > threshold

    elif metric == 'dominant_value_change':
        return new_stats['dominant_value'] != old_stats['dominant_value']

    elif metric == 'embedding_fallback':
        return cosine_distance(
            embed(old_stats['habit_text']),
            embed(new_stats['habit_text']),
        ) > 0.15
```

**实现说明**：
- `embed(text: str) -> list[float]` = `panoramix_core.embedder.Embedder().embed(text)`（现有组件，ada-002，返回 List[float]）
- `cosine_distance(a, b) = 1 - (dot(a, b) / (norm(a) * norm(b)))`
- `embedding_fallback` 只在 signal 没有显式 `drift_metric` 时触发；这条路径会发网络调用，必须在阶段 1 慢动作中执行

**关键不变式**：
- **不加证据数守门**（Round 9 Q9.2.d 决策）：哪怕 new_stats.count == 1，检测到漂移就推送
- **count == 0 的边界情况**：理论上不可能发生。若某 signal 在新批次消失，`structural_key` 里的 `signal_type_set` 会改变，那整个 candidate 根本不会与这张 accepted 卡在 Step A 匹配上
- **`embedding_fallback` 的 embed 调用在事务外**：这条路径会调 ada-002 云端 API；必须在 Pipeline 的"阶段 1 慢动作"里完成（见 §8）

### 7.4 `compute_raw_value_stats()` 辅助

从一组 habit 的 `member_fact_ids` 拉 PREF facts 的 `raw_value`，再按信号 category 聚合：

```python
def compute_raw_value_stats(
    pref_facts: list[Fact],
    signal_category: str,
) -> dict:
    """
    pref_facts: habit 聚类的成员（已按 signal 筛选）
    返回符合 §3.4 的 raw_value_stats 结构
    """
    raw_values = [
        json.loads(f.json_metadata)["raw_value"] for f in pref_facts
    ]
    if signal_category == "numeric":
        vals = [float(v) for v in raw_values if isinstance(v, (int, float))]
        return {
            "type": "numeric",
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }
    elif signal_category == "categorical":
        counts = Counter(str(v) for v in raw_values)
        return {
            "type": "categorical",
            "dominant_value": counts.most_common(1)[0][0],
            "value_counts": dict(counts),
            "count": sum(counts.values()),
        }
    else:
        raise ValueError(f"unknown signal_category: {signal_category}")
```

---

## 8. Pipeline：`ingest_signal_batch()` 重写

### 8.1 两阶段结构（Round 9 Q9.3）

Round 9 Q9.3 锁定了 10 步 pipeline（Step 0~9）。由于 GPT 调用 / Chroma 读写 / embedding 调用必须在 SQLite 事务外（持锁时间约束），pipeline 物理上拆成两阶段执行。下表给出"物理阶段 → Round 9 逻辑步骤"的映射：

```
┌──────── 阶段 1：慢动作（事务外，~10-60s）────────┐
│  P1.1 → Step 0 (peek)   预估 batch_id，仅用于日志 │
│  P1.2 → Step 1          Chroma 读 PREF TTL 窗口   │
│  P1.3 → Step 2          Hybrid DBSCAN + 连续性    │
│                         + GPT reword → habits     │
│  P1.4 → Step 4 (前半)   按 structural_key 分组    │
│                         候选场景卡                 │
│  P1.5 → Step 4 (后半)   GPT scene naming（每组）   │
│  P1.6 → Step 5          compute_raw_value_stats + │
│                         classify_candidate        │
│                         （含 embedding_fallback   │
│                          的网络调用路径）          │
│  [阶段 1 产物全部暂存 Python 内存，未写 DB]       │
└────────────────────────────────────────────────────┘

┌──────── 阶段 2：快动作（SQLite 事务内，<200ms）──┐
│  with store.transaction() as conn:                │
│    P2.1 → Step 0 (real) 权威 batch_id 分配        │
│    P2.2 → Step 3        HabitStore.insert_many    │
│    P2.3 → Step 6        每个 classification 落盘：│
│             Reinforce   → update_last_reinforced  │
│             ModifyDrift → upsert_recommendation + │
│                           update_last_reinforced  │
│             NewPending  → upsert_pending          │
│    P2.4 → Step 7        delete_stale_pending       │
│    P2.5 → Step 8        delete_stale_recommendation│
│                         (A2 硬删除)                │
│    P2.6 → Step 9        delete_old_batches(N=3)    │
│  COMMIT  → Step 10                                 │
└────────────────────────────────────────────────────┘
```

**映射关系一览**：P1.1-P1.6 对应 Round 9 Q9.3 的 Step 0(peek)/1/2/4/4/5；P2.1-P2.6 对应 Step 0(real)/3/6/7/8/9；COMMIT 对应 Step 10。逻辑步骤编号仍以 Round 9 Q9.3 为准。

### 8.2 事务边界决策回顾（Round 9 Q9.3 决策点）

| 决策点 | 选项 | 决策 | 理由 |
|---|---|---|---|
| Chroma 写入 vs SQLite 事务 | A=Chroma 先写幂等 | **A** | Chroma 无真事务；PREF 幂等可被下批自然覆盖 |
| GPT 调用是否进事务 | A=事务外 | **A** | GPT 慢 + 失败率高，持锁时间必须短（~100ms） |
| 陈腐 recommendation | A2=硬删除 | **A2** | recommendation 属系统建议范畴，用户主权不禁止撤回 |

### 8.3 `allocate_next_batch_id` 的正确姿势

两阶段有个小坑：阶段 1 需要 `batch_id` 给 GPT prompt 里的日志用，但阶段 2 才真正在事务里分配。处理方式：

- 阶段 1 用 `peek_latest_batch_id() + 1` 获得"预期 batch_id"，只用于日志/调试
- 阶段 2 事务开头用 `allocate_next_batch_id()` 在事务内 `SELECT MAX(batch_id) + 1` 获得权威值
- 两者不一致时以阶段 2 的权威值为准；所有写入用权威值

**并发说明**：demo 是单进程单用户，不需要考虑"两个 batch 同时写"的竞态。真要发生也会被 SQLite 的 default `BEGIN IMMEDIATE` 锁序列化。

### 8.4 错误恢复

- **阶段 1 异常**：整批丢弃，Chroma 里新写入的 PREF 保留（下次 batch 会再被处理）
- **阶段 2 异常**：事务 rollback，所有 SQLite 写入回滚；Chroma 状态与事务前一致（因为 Chroma 写入在 Step 1 读取之前，等同于"什么都没发生"）
- **关键不变式**：任何失败路径都不能产生"部分 commit"的 scene_cards 或 habits

---

## 9. 用户交互 API（供 Tab1/Tab2 调用）

除了 `ingest_signal_batch`，HabitDemoEngine 需要暴露以下新方法给 UI：

```python
class HabitDemoEngine:
    # ── 现有（保留兼容）──
    def ingest_signal_batch(self, events): ...
    def get_status(self) -> dict: ...

    # ── 新增：场景卡操作 ──
    def get_scene_cards(
        self, status: str = None
    ) -> list[SceneCard]:
        """按状态筛选场景卡"""

    def accept_scene_card(self, card_id: str) -> None:
        """用户接受 pending 或 recommendation 卡
        若是 recommendation：被指向的旧 accepted 卡立即 status→retired (Q8.1 决策 A)"""

    def reject_scene_card(self, card_id: str) -> None:
        """用户拒绝 recommendation 卡 → 硬删除
        pending 卡不支持主动 reject（让它自然陈腐被清理）"""

    def retire_accepted_card(self, card_id: str) -> None:
        """用户主动关闭 accepted 卡（唯一能退役 accepted 卡的入口）"""

    def get_latest_habits(self) -> list[Habit]:
        """Tab1 UI: 最新批次的 habits（SELECT ... WHERE batch_id = MAX(batch_id)）"""
```

**旧 API 处置**：`accept_habit(habit_id)` / `reject_habit(habit_id)` 被标记 deprecated 但保留调用（内部转换为对应 scene_card 操作）。实际使用方是 `simulator/tab2_recommendation.py`，迁移到 scene_card API 是 Tab2 的后续工作，不在本 spec 范围内。

---

## 10. 测试策略

### 10.1 分层测试

| 层 | 测试对象 | 新增测试文件 |
|---|---|---|
| 单元 | SignalRuleEngine yaml 加载、规则查询、value_range | `tests/test_signal_rule_engine.py` |
| 单元 | `is_drifted()` / `compute_raw_value_stats()` 纯函数 | `tests/test_drift_detection.py` |
| 单元 | SceneCardStore 所有 API + 不变式 | `tests/test_scene_card_store.py` |
| 单元 | HabitStore CRUD + batch 清理 | `tests/test_habit_store.py` |
| 单元 | `compute_structural_key()` 稳定性 | `tests/test_structural_key.py` |
| 集成 | HabitLifecycleManager.classify_candidate 三分支 | `tests/test_habit_lifecycle.py` |
| 端到端 | 多批次场景：Reinforce / Modify / NewPending 混合 | `tests/test_multi_batch_drift.py` |

### 10.2 关键场景测试（用户主权不变式）

这几条必须有专门测试覆盖：

1. **Accepted 卡不可变**：调用 SceneCardStore 任何 UPDATE 路径作用于 accepted 卡 → 抛 `SceneCardImmutableError`
2. **Drift → recommendation 非替换**：有一张 accepted 卡，batch 2 里对应的 habits 发生 numeric drift → accepted 卡的 frozen_* 字段必须完全不变；新产生一张 recommendation 卡
3. **Recommendation 去重**：batch 2 和 batch 3 都对同一张 accepted 检测到 drift → recommendation 卡 ID 稳定（Q8.2 upsert）
4. **Pending 陈腐清理**：batch 1 产生 pending A，batch 2 未重现 A → A 被硬删除（Step 7）
5. **Accept recommendation 级联 retire**：accept 一张 recommendation 卡 → 它 supersede_candidate_for 指向的所有 accepted 卡 status 变 retired
6. **Accepted 卡 dormant 不退役**：batch 1 产生 accepted A 并被用户接受，batch 2~10 都没有匹配 A 的 structural_key → A 在数据层仍然 status='accepted'（UI 怎么展示另说）

### 10.3 性能基准

mockup 数据规模（30 天 × ~100 信号）下：
- `ingest_signal_batch` 总时长 < 60 秒（GPT 调用为主）
- SQLite 事务持锁时间 < 200ms（不含 GPT）
- 事务内 SQL 写入行数 < 500

---

## 11. 实施顺序建议

按依赖关系分 5 波：

1. **Wave 1：SignalRuleEngine 独立**（Round 2）
   - 不依赖其他改造，可先落地
   - 单独出 PR，验证 `tests/test_signal_to_fact.py` 不回归

2. **Wave 2：SQLite schema + 双 Store**（Round 6/7/8）
   - 实现 `scene_card_store.py` + `habit_store.py` + schema
   - 单元测试全通过
   - 这是后续所有 drift 逻辑的地基

3. **Wave 3：HabitLifecycleManager 纯函数层**（Round 9）
   - `compute_structural_key` / `detect_drift` / `classify_candidate`
   - 全部可脱离 DB 测试

4. **Wave 4：Pipeline 重写**（Round 9 Q9.3）
   - `engine/habit_engine.py` 的 `ingest_signal_batch` 改为两阶段
   - 端到端测试

5. **Wave 5：旧代码清理**
   - `engine/scene_card.py` 原 in-memory `SceneCard` dataclass 删除；只保留新加的"候选场景卡构造"辅助函数
   - `FactStoreChroma` 写 HABIT 类型 Fact 的路径删除（HABIT 已全部经由 SQLite）
   - 过时的 `tests/test_scene_card.py` 迁移到新的 `test_scene_card_store.py` 或直接删除

每一波之间有 review + merge 间隔，避免长分支。

---

## 12. DCM 登记簿（车端迁移计划 — 本 spec 不做）

继承自 brainstorm 文档：

| ID | 来源 | 当前方案 | 最终形态 |
|---|---|---|---|
| DCM-1 | Round 8 Q8.1 | 用户接受 recommendation 自动替换 accepted | 弹确认框二次确认 |
| DCM-2 | Round 8 Q8.4 | 数据层只暴露 `last_reinforced_batch_id` | UI 折叠 / 灰色 / 到期提示 |
| DCM-3 | Round 9 Q9.1 | 只识别 modify drift，split/merge 不链接 | 结构键子集匹配 (B) + 内容相似度兜底 (C) |

本 spec 实施过程中若又发现类似"最终形态要动但当前先简化"的点，继续追加到这张表。

---

## 13. 未决问题 / 风险

以下点在实施过程中可能需要再确认。记录在此避免遗漏，但**不阻塞**本 spec 进入 writing-plans：

1. **`supersede_candidate_for_json` 的 SQL 查询性能**：SQLite 的 `json_each()` 可以做 "JSON array contains value" 查询，但没索引。当前数据规模可忽略；若未来 recommendation 卡数量很多需要考虑加虚拟列 + 索引
2. **`engine/scene_card.py` 的兼容性**：现有 `tests/test_scene_card.py` 依赖内存 dataclass，迁移时可能需要大改。实施 Wave 5 时再决定"迁移还是删除"
3. **`raw_value_stats_json.habit_text` 的生成点**：`frozen_content_snapshot` 里的 habits 每条都有 `habit_text`（GPT reword 输出），而 embedding_fallback drift metric 会 embed 它。但 `habits` 表已有 `text` 字段，snapshot 里是否要冗余存一份？答：**是的，要冗余存**，因为 accepted 卡的快照必须独立于 habits 表（habits 表只保留 3 批，accepted 可能很老）
4. **多用户并发**：当前 demo 就算多用户也是顺序调用。schema 里 `username` 列就位，未来真要支持并发再考虑锁策略
5. **SignalRuleEngine 对 `nav_destination` / `acc_distance` 这类"每次值都不同"的信号**：当前 signal_to_fact 直接把值塞进 template。drift_metric 对这类信号无意义——值每次都变不等于漂移。处理方式：yaml 里标 `drift_metric: none` 跳过漂移判定。实施时补规则
