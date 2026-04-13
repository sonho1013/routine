# Tab1 实车接入 — Brainstorm 会议记录

> **日期:** 2026-04-08
> **范围:** 仅针对 Tab1（数据后台/学习阶段），Tab2 暂不讨论
> **目的:** 逐轮讨论实车接入时的问题与期望效果，为后续优化设计文档提供依据
> **记录方式:** 每轮 Q&A 立即追加到本文件，避免 context compact 丢失

---

## 0. 上下文速查（讨论前的系统现状）

**Tab1 当前数据流:**

```
mockup JSON (scenarios/mock_data_generator.py)
  → SignalSimulator (signals/simulator.py)
  → signals_to_facts (engine/signal_to_fact.py  ← 手写 if/elif 翻译)
  → ChromaDB + ada-002 embedding (panoramix_core/store/fact_store_chroma.py)
  → Hybrid DBSCAN α=0.8 text + 0.2 ctx (panoramix_core/clustering/habits_detector.py)
  → 连续性过滤 (engine/consecutiveness.py)
  → GPT-4.1 reword → Habit Fact
  → 场景卡命名 (engine/scene_card.py)
```

**已识别的 10 个实车接入风险领域:**

| # | 领域 | 当前 | 实车痛点 |
|---|---|---|---|
| 1 | 信号源对接 | JSON event dict | 实车是 CAN/Someip/DDS 连续流 |
| 2 | signal→fact 硬编码 | 每种信号写 if/elif | 新增信号/换 OEM 就要改代码 |
| 3 | 事件切片 | 外部预切 scene | 连续流无天然事件边界 |
| 4 | 增量 vs 批量 | O(N²) 重算 | 百万级信号不可行 |
| 5 | Embedding 离线化 | OpenAI ada-002 云端 | 车机可能断网 |
| 6 | LLM reword 依赖 | GPT-4.1 云端 | 同上 + 延迟 |
| 7 | 地理围栏硬编码 | 巴黎 4 坐标 | 实车坐标未知 |
| 8 | 聚类参数固化 | eps=0.10 min=3 手调 | 数据规模/用户差异 |
| 9 | 习惯漂移 & 冷启动 | 无遗忘/季节机制 | 长期陈旧积累 |
| 10 | 多人/多车 & 并发 | 按 username collection | 一车多驾驶员 / 云同步 |

---

## 讨论记录

（每轮问答追加到下方）

---

### Round 1 — 信号源对接协议

**问题背景:**
- 当前：tab1 吃的是 `{date, signals: [{t, signal, value}]}` 这样的结构化 JSON event
- 实车：OEM 通常通过 CAN bus / Someip / DDS / MQTT 等 middleware 提供原始信号流，频率 1Hz~100Hz，字段名因 OEM 而异，没有"事件"概念

**候选方案:**
- A. Middleware 订阅回调 (push)
- B. 共享内存环形缓冲 (poll)
- C. MQTT / REST 桥接
- D. 离线日志批处理（日粒度或更久）

**用户答复:**
> 应该是 D，不急于处理，但是当前演示是使用 mockup 数据，所以不需要考虑。

**结论/决策:**

1. **未来方向 = D（离线日志批处理）**：车端把信号写成日志，habit 系统批量处理。不要求实时。
2. **当前优化范围**：**不包括**信号采集/订阅的接入层，tab1 继续消费 mockup 数据。
3. **对后续设计的影响**：
   - 不需要做 push/pull 订阅框架、流式处理、信号回调生命周期管理。
   - **但**：设计时要预留一个"日志批处理适配层"的接口位置，让未来实车日志可以接入（replace `scenarios/mock_data_generator.py` 这一层）。
   - 离线批处理意味着：**延迟不敏感**可以接受，**吞吐敏感**要考虑（一天堆积的日志可能是几 MB~几百 MB）。
   - 实时性的要求降低意味着 **Embedding/LLM 云端调用在批处理窗口内是可接受的**（比如车辆停驶时触发一次日终处理）。

---

### Round 2 — Signal → Fact 翻译的硬编码

**问题背景:**
- 当前 `engine/signal_to_fact.py` 每个信号一段 if/elif，11 种信号覆盖，文本措辞直接影响 embedding 聚类效果
- 新信号/换 OEM 字段命名就要改 Python 代码

**候选方案:**
- A. 配置驱动 (YAML/JSON schema) + 通用翻译引擎
- B. 每条信号调 LLM 动态翻译
- C. 混合：schema 打底 + LLM 兜底未知信号
- D. 保持现状

**用户答复:**
> A

**结论/决策:**

1. **选定 A：配置驱动的 signal→fact 翻译引擎**。
2. 将 `engine/signal_to_fact.py` 的 if/elif 大段硬编码抽为一个 **SignalRuleEngine + yaml schema**：
   - schema 描述：`signal_name`, `aliases`（OEM 别名）, `unit_scale`, `value_range`, `text_template`, `category`, `ctx_field`（若该信号属于 Table 4 上下文信号，则指定它映射到 StructuredContext 的哪个字段）
   - 引擎负责：别名归一化、单位换算、值域校验（越界丢弃或钳制）、按 template 渲染文本
3. **未决的次级问题（延后到设计阶段再定，用合理默认）：**
   - (Q2) 当前**没有真实 OEM DBC 文件**，先用我们自己定义的 canonical schema（覆盖现有 11 种信号），未来对接时在 schema 里加 OEM aliases 即可，不用改引擎代码
   - (Q3) schema 先由工程师维护，但格式保持人类可读（yaml），为后续开放给产品/语言学家编辑打底
4. **对后续设计的影响：**
   - 需要新建 `engine/signal_rules/` 目录：`engine.py`（SignalRuleEngine）+ `rules.yaml`（schema）
   - `signal_to_fact.py` 变薄，只负责调 engine 并组装 Fact 对象
   - 地理围栏 `KNOWN_GEOFENCES` 也应该挪出硬编码（见 Round 7 待讨论）
   - 所有 11 条翻译规则迁移到 yaml 后，单元测试需要跟着改（当前 tests/ 里有依赖硬编码行为的测试）

---

### Round 3 — 数据积累与增量处理 + **习惯生命周期管理**（用户引出的新议题）

**问题背景:**
- 当前 `ingest_signal_batch` 每次拉全量 + 重算 N×N 距离矩阵
- 实车场景下 N 会持续增长，O(N²) 不可持续

**候选方案:**
- A. 保持全量重算
- B. 时间窗口滑动（比如只聚类最近 30 天 PREF facts）
- C. 在线/增量聚类（换算法）
- D. 窗口 + 分层快照

**用户答复:**
> 最终我觉得是 B + 轻量固化，mockup 数据每天产生的信号数量大概是 100 量级，但我建议 B 窗口方案优先级放到后面，因为演示 mockup 数据最多给三十天，甚至更少，如果你觉得可以做 A 方案那么就先用 A，不行的话就做轻量固化，但固化是不是也面临一个问题，就是可能存在产生的新 habit 和旧 habit 完全一致的问题。从产品角度上讲，我希望新旧 habit 能进行替代，即识别到新 habit 是否对应旧的 habit，如果对应，则识别是否 habit 发生偏移，如果偏移，则向用户推新的场景卡片。

**结论/决策:**

1. **性能层面**：当前选 **A（全量重算）**。
   - 理由：mockup 数据规模 ≤ 30 天 × ~100 信号/天 ≈ 3000 条量级，O(N²) 完全可承受
   - B（窗口滑动）**降级为未来优化项**，等实车数据规模超 5000 条再切
   - 不做增量 DBSCAN（C），避免算法复杂度无谓提升

2. **用户提出的新关键议题：习惯生命周期管理（去重 + 偏移检测 + 替换）**
   - 这是用户**产品侧的明确要求**，优先级高于 B 窗口
   - 当前系统缺陷：每次重算都可能产生和旧 habit 内容一致或相近的新 habit，没有 dedupe/merge 逻辑

3. **产品规则（用户的期望效果）：**
   - **新 habit vs 旧 habit 必须先做匹配**
   - **匹配上了 + 内容一致** → 新 habit 丢弃，旧 habit "续命"（更新时间戳、证据数、可能提升置信度）
   - **匹配上了 + 内容偏移（例如空调从 22°C → 24°C）** → 旧 habit 标记 `superseded`，新 habit 上位 + 打 `needs_user_confirmation` 标记 → **tab2 推一张"您的习惯变了"场景卡片**让用户确认
   - **没匹配上** → 新 habit 直接入库（当前默认行为）

4. **对后续设计的影响：**
   - 需要新建 `engine/habit_lifecycle.py` 模块（或扩展 `HabitsDetector`）
   - 需要定义"对应"的匹配规则（同一 signal 类别 + 同一 StructuredContext 组合）
   - 需要定义"偏移"的判定算法（text embedding 距离阈值 + 数值字段差值容忍度）
   - Fact/Habit 模型需要扩展元数据：`superseded_by`, `supersedes`, `needs_user_confirmation`, `last_reinforced_at`
   - Tab2 需要新增"习惯变更确认"卡片类型（但**不是本次优化范围**，我们只产生信号，不做 tab2 UI）

5. **未决的次级问题（转入 Round 4 专题讨论）：**
   - 如何定义"对应"？信号名相同 + 上下文相似（geofence + time_bucket + vehicle_state）？
   - 如何判定"偏移"？文本 embedding 距离？还是从 json_metadata 的 raw_value 做数值比较？
   - 偏移阈值多少？
   - 用户拒绝"新习惯"后，旧习惯是否恢复？

---

### Round 4 — 习惯替代/偏移检测的算法细节

**问题背景:**
Round 3 产出"匹配 → 比较 → 续命/替换/新建"三步产品规则后，需要把它翻译为具体算法：
- Q4.1 "对应"（同一习惯）的判定标准
- Q4.2 "偏移"（内容变了）的判定标准
- Q4.3 偏移发生时旧 habit / 新 habit 的生命周期状态机

**候选方案概述:**
- Q4.1: A=严格全维度匹配 / B=上下文主键匹配 / C=文本 embedding 距离 / D=A→B 两级匹配
- Q4.2: A=纯文本 embedding / B=数值字段差值 / C=按信号类型混合判定
- Q4.3: A=直接删除旧 habit / B=状态机（superseded/pending_confirmation，可回滚）

**用户答复:**

**Q4.1 → D（A→B 两级匹配）**，但补充：
> "上下文主键是否匹配主要看场景卡配置的粒度，这也是我准备后续引入场景卡的原因。"
>
> 含义：上下文主键（geofence + time_bucket ...）到底怎么组合，不应在 habit 层硬编码，而应由"场景卡配置"来决定粒度。场景卡是场景定义的权威来源。

**Q4.2 → C（按信号类型混合判定）**

**Q4.3 → 以场景卡为中心的状态机（不再以 habit 为中心）**：
> "检测到偏移会推送新的场景卡，然后分两种情况：
> - 如果旧 HABIT 形成的场景卡**没在推荐时被用户接受**，则直接替换
> - 如果旧 HABIT 形成的场景卡**已经被用户接受**，则应该新旧同时存在，并标明新的是否替换旧的由用户来确认，确认后替换"
>
> 另问："HABIT 形成场景卡后是否要删除，你可以思考下"

**结论/决策（已定部分）:**

1. **Q4.1 匹配算法 = D（A→B 两级降级）**
   - A 级：信号类别相同 + 全维度 StructuredContext 相等
   - B 级：信号类别相同 + 上下文"主键"相等
   - 但**"主键"是哪些维度不是硬编码**，而是由即将引入的"场景卡配置"决定 → Round 5 议题
   - 匹配范围应该只在"同一场景卡配置"内部做，跨场景卡不做对应判定

2. **Q4.2 偏移判定 = C（按信号类型混合）**
   - 与 Round 2 的配置驱动架构天然契合
   - 每条信号在 yaml 里声明 `drift_metric` + `drift_threshold`
     - 数值型（温度、座椅档位、音量）→ `drift_metric: numeric`，阈值以物理量为单位（°C、档、dB）
     - 分类型（音乐源、驾驶模式、空调风向）→ `drift_metric: categorical_change`（值变 = 偏移）
     - 文本兜底（未配置的信号）→ `drift_metric: embedding`，`threshold: 0.15`
   - 数值差值基于 habit 聚类成员的 raw_value（需要在 fact→habit 合成时保留成员的原始值统计）

3. **Q4.3 生命周期 = 以场景卡为中心的两分支状态机**
   - 偏移检测的归宿是**场景卡级别**，不是 habit 级别
   - 分支 1：旧场景卡 `status=pending`（尚未被用户确认） → 旧卡静默替换，不打扰用户
   - 分支 2：旧场景卡 `status=accepted`（已被用户确认） → 新旧卡共存，新卡标记"替代候选"，等用户在新卡上确认后再删除旧卡
   - 数据模型含义：
     - **场景卡是有状态的第一等实体**（至少包含 `status: pending|accepted|supersede_candidate|retired`）
     - **habit 则变成场景卡的"证据集"**，本身是否第一等存疑 → 转入 Q5 讨论

**转入 Round 5 的待议题:**
- Q5.1：**场景卡作为第一等实体的数据模型** — 主键、粒度、与 habit 的所有权关系
- Q5.2：**habit 是否在场景卡生成后被删除/标记消费** — 即用户的开放问题
- Q5.3：**"场景卡配置"的粒度定义** — 用户引出的新概念，决定了 Q4.1 B 级匹配的上下文维度
- Q5.4：**重聚类的幂等性** — 每次全量重算会重新生成 habit，场景卡如何在"habit 被重算"之后保持自己的身份和 status 不丢

---

### Round 5 — 场景卡身份与生命周期 (Q5.3 先行)

**讨论焦点：**
Q5.3 — 场景卡是产品预定义模板（A），还是由上下文聚类动态涌现（B），还是混合（C）？

**第一轮候选方案：**
- A：静态场景模板表（产品定义，工程落盘）
- B：动态场景发现（聚类簇 = 场景卡，GPT 起名）
- C：混合（模板优先 + 动态兜底）

**用户初次答复：**
> "我的预想是 B，因为场景不是由 GPT 总结出来的吗，应该由上下文聚类簇获取。不应该由产品提前定义场景模板，而是期待大模型有能力总结出模板。至于场景卡重聚类身份漂移，可以举个面向不懂技术的用户的实际例子吗？"

**我举的四个漂移场景（供用户理解纯 B 的 UX 代价）：**

用设定：李阿姨每天开车接孙子上学，第 1-3 周系统学到"接孙子上学（工作日早晨）"场景卡并被用户 accept。

- **场景 1 名字漂移**：后续数据让 GPT 重新起名为"早晨驾驶偏好"，李阿姨发现之前接受的卡消失、新卡要求重新确认
- **场景 2 切分漂移**：聚类把一个簇切成"工作日晨间通勤" + "周末晨间出门"两张新 pending 卡，用户要再次确认一件已经确认过的事
- **场景 3 合并漂移**（最危险）：晨间和晚间被聚到同一簇，若系统把 accepted 状态继承过去 = 用户在不知情下授权新设置，信任崩塌
- **场景 4 消失漂移**：用户出差一周，原簇证据不足被解散，accepted 卡悄悄消失，用户回来后要重新接受

**核心洞察：**
问题不是 GPT 命名能力，而是**场景卡的"身份"被绑在 cluster 上**——cluster 变了，身份就变了，用户的确认状态随之丢失。

**第二轮候选方案（在 B 的基础上演化）：**
- **B + 身份锚定**：用户 accept 后冻结一个签名（top_signals + context_fingerprint），下次重聚类通过签名 rebind；drift 作为推荐而非静默替换
- **B'（结构键 + GPT 名字降级为标签）**：身份由结构特征算出稳定键 `hash(dominant_geofence, dominant_time_bucket, signal_type_set)`，GPT 名字只作为显示标签可随批次刷新
- **B''**：先 B'，分裂/合并检测留作后续议题

**用户最终答复（综合两轮）：**
> "我的建议是 B 方案 + 身份锚定（一旦 accepted 就冻结签名），一旦用户接受，就锁定该场景卡，不管如何迁移，都需要用户自己去关闭或删除原场景卡，然后如果检测到发生偏移，无论是分裂合并修改，都是作为推荐提示用户让用户确定是否更新。后面那个回答我也是建议 B'。但用户接受后锁定的场景卡名字不能变。"

**结论/决策：**

**Q5.3 最终方案 = B' + accepted 后全锁定**
B' 是底层机制（结构键做身份、GPT 做标签），身份锚定是 accepted 之后的锁定策略，两者相互印证，合成一个统一方案。

**核心产品原则（从用户答复提炼）：**

1. **用户主权绝对化**：accepted 卡一旦锁定，系统**不得做任何静默修改**——不换名字、不换内容、不换身份、不自动 GC。一切变更都必须经由用户明示
2. **Drift 的归宿是"推荐"，不是"替换"**：无论系统检测到的是分裂、合并还是内容修改，都只能生成**推荐卡**推给用户，旧卡保留，由用户决定是否替换/关闭
3. **只有用户才能关闭 accepted 卡**：系统没有自动淘汰 accepted 卡的权力（哪怕数据上该场景已经很久没发生）
4. **GPT 涌现命名 × 身份锁定共存**：pending 阶段 GPT 自由命名；一旦 accepted，名字和身份都冻结，即便下次重聚类 GPT 想给同一结构键起别的名字，accepted 卡保持原名

**具体规则：**

1. **pending 阶段（尚未被用户接受）：**
   - 按"结构键"分组 habit → GPT 为每组生成一个标签名 → 形成 pending 场景卡
   - `结构键 = hash(dominant_geofence, dominant_time_bucket, dominant_vehicle_state, sorted(signal_type_set))`
   - pending 卡之间允许自由重排、替换、重命名；每次重聚类都可以产生一套全新的 pending 卡
   - pending 卡的身份就是当前结构键（可能随批次变化）

2. **用户接受（pending → accepted）触发的冻结动作：**
   - 当前 GPT 名字冻结为 `frozen_name`（未来重聚类不再刷新）
   - 当前结构键冻结为 `frozen_signature`
   - 当前 habit 证据集**快照保存**到场景卡内部（`frozen_content_snapshot`），不再依赖"当前批次的 habit 集合"
   - status 变为 `accepted`

3. **accepted 卡的不可变性：**
   - `frozen_name` / `frozen_signature` / `frozen_content_snapshot` 一经冻结永不变
   - 不因"最近 N 天没发生"被系统自动隐藏/归档（对应场景 4 的修复）
   - 只能由用户显式执行"关闭/删除"才会从库里移除

4. **Drift detection（Round 4 状态机的精确落地）：**
   - 每次重聚类产生新 habits 后，系统对每张 accepted 卡检查三类事件：
     - **内容漂移（修改）**：结构键仍匹配，但 habit 内容相对 `frozen_content_snapshot` 超过 Q4.2 阈值
     - **分裂事件**：accepted 卡的 frozen_signature 下，新一批的结构键变成了 N 张
     - **合并事件**：N 张 accepted 卡在新一批里共享同一个结构键
   - 无论哪种情况：生成一张**推荐卡**（本质是特殊的 pending，带 `supersede_candidate_for: [accepted_card_id, ...]`）
   - 推荐卡推给用户，由用户决定：
     - **接受推荐** → 新卡 status=accepted，引用的旧 accepted 卡被用户主动关闭时才退役
     - **拒绝推荐** → 推荐卡被丢弃，旧卡保留 accepted 不变
     - **忽略（不操作）** → 推荐卡在推荐池里排队直到下次查看

5. **accepted 卡"锁定"的完整含义：**
   - 锁定名字、身份签名、内容快照、生命周期
   - 唯一可变路径：用户主动"关闭/删除"或"接受替代推荐"

**Round 5 后续议题（未完成）：**

- Q5.1：数据 schema 落地 — Fact / Habit / SceneCard 三个实体的字段、主键、存储位置
- Q5.2：habit 层的去留决策 — "每批次重算，保留上一批次 snapshot 做 drift 对比"是否可接受
- Q5.4：**已隐式解决** — B' + accepted 锁定天然保证幂等性（accepted 卡不受重聚类影响；pending 卡本来就允许重排）

---

### Round 6 — 场景卡持久化存储 (Q5.1) [已锁定]

**讨论焦点：**
场景卡是一个状态机实体（status ∈ {pending, accepted, recommendation, retired}），核心查询是"按 status 过滤 + 按时间排序 + JOIN supersede 关系"，不是向量检索。ChromaDB 是向量库，把场景卡塞进去是"在错误的工具上建一个本不属于它的系统"。

**候选方案：**
- **A — 在 ChromaDB 里开第二个 collection `scene_cards`**：零新依赖、复用现有 store 层，但 metadata filter 弱、无 JOIN、`supersede_candidate_for` 数组字段不支持
- **B — SQLite 文件（`storage/scene_cards.db`）+ 现有 ChromaDB**：SQL 天然支持状态机查询、JOIN、ORDER BY、索引、事务；SQLite 是 Python 标准库自带零新依赖；清晰分层（向量存储 vs 关系存储）；仅加约 150-250 行代码
- **C — JSON 文件（`storage/scene_cards.json`）**：最简单最透明零依赖，但无并发保护、无索引、无事务语义、状态机切换易不一致

**我的推荐：B (SQLite)**
理由：tab1 是里子要做对；场景卡状态机交互多需要事务；ChromaDB 职责是相似度检索不应滥用；SQL schema 扩展成本几乎为零；未来扩展（场景卡历史版本、用户操作日志、卡之间引用）都是 SQL 擅长的。

**用户答复：**
> "到时候要考虑迁移车端，如果 B 方案不影响迁移车端的消耗则方案 B，如果消耗较大，则需要考虑迁移车端的方向提新方案"

**车端迁移成本分析（我给出的关键事实）：**

1. **SQLite 本来就在车里** — Android Automotive、CarPlay、QNX、主流车机中间件栈普遍内置；梅赛德斯、特斯拉、丰田、Airbus A350 都在生产环境用 SQLite
2. **体积极小** — 编译后约 650KB，无后台进程，单文件存储，C/C++ 原生 API；车端换语言调用 schema 和数据文件完全兼容
3. **三方案的车端迁移成本排序实际上是 A > C > B**：
   - A (Chroma collection) 🔴 **最高**：把场景卡状态机捆绑进 Chroma = 捆绑到技术栈里**最难迁移**的组件命运上
   - C (JSON) 🟡 **中**：文件简单但车端要手写状态机+并发+一致性，SQL 里是免费的
   - B (SQLite) 🟢 **几乎零**：车机本来就有
4. **真正要担心的车端迁移大头不是存储层**：
   - ChromaDB 🔴 高 — 需换成 FAISS flat + SQLite 元数据或放弃向量
   - OpenAI ada-002 embedding 🔴 高 — 需换本地小模型
   - GPT-4.1 reword/scene naming 🟡 中 — 离线批处理 + 联网窗口 + 本地 fallback
   - Python 运行时 🟡 中 — 核心算法可能要 C++ 重写
   - **SQLite 🟢 零 — 反而是迁移的"资产"而非"负担"**

**初步结论（等待用户确认）：**
Q5.1 = **B（SQLite）**，车端迁移成本判断：零（优于 A、C）。

**提议的后续安排：**
- Round 7 单独开一轮 **"车端迁移总体策略"** 专题，处理 ChromaDB 替代、本地 embedding 选型、LLM 降级路径、Python → C++ 重写范围——不让这个大话题插队卡住 Q5.1/Q5.2
- Q5.1 确认后立刻进入 Q5.2（habit 持久化策略）

**用户最终确认：**
> "确认选项 B，SQLite"

**最终决策：**
- **Q5.1 = B（SQLite 文件 `storage/scene_cards.db`）**
- Chroma 继续存 PrefFacts（向量检索层）
- SQLite 存 SceneCards（状态机层）
- 新建薄 `SceneCardStore` 类封装 SQL 读写，提供 `create / update_status / get_by_status / find_supersede_candidates` 等方法
- 车端迁移成本确认为零（SQLite 是车机原生组件）
- 车端迁移大话题推迟到未来某个时间点（见 Round 7 末尾用户指示：长时间内不再讨论）

---

### Round 7 — Habit 持久化策略 (Q5.2) [已锁定]

**问题背景：**
Round 5 Q5.3 决定了 accepted 场景卡自带 `frozen_content_snapshot`，不依赖 live habits 才能活。这让 habit 从"第一等实体"降级为"每批次产出的中间计算结果"，主要职责只剩：
1. 喂给结构键分组生成当前 batch 的 pending 场景卡
2. Tab1 UI 展示"系统学到了什么"
3. 与 accepted 卡的 frozen snapshot 对比做 drift detection

都是当前批次的事情。

**候选方案：**
- **A — 全 ephemeral**：batch 结束就丢，Chroma 完全不存 HABIT
- **B — SQLite `habits` 表 + `batch_id`，保留最近 N 批**：habits 搬到 SQLite，每条带 batch_id，retention 保留最近 N 批，Chroma 不再存 HABIT
- **C — 保持 habits 在 Chroma，加 `batch_id` 元数据，每批清理旧的**：最小改动，但架构割裂

**我的推荐：B**
理由：
1. 架构清晰分域 — Chroma = 向量检索（PrefFacts only），SQLite = 状态机 + 批次产物（habits + scene_cards）
2. 事务原子性 — drift 批处理一次写 habits + scene_cards + recommendation 可以在同一事务里
3. Debug/audit 价值高 — tab1 是里子，调 drift 算法必须能对比"上一批 vs 这一批"
4. Tab1 UI 查询模型自然 — `SELECT * FROM habits WHERE batch_id = MAX(batch_id)`
5. A 过激进（丢审计），C 假节省（事务割裂、两块都要动）

**用户答复：**
> "选 B，接受 N=3，habit 表保留 pref 反向引用。round8 不需要考虑，长时间内不需要考虑车端迁移的事"

**最终决策：**

1. **Q5.2 = B（SQLite `habits` 表）**
2. **N = 3**：每次 batch 保留最近 3 批的 habits，批次号更早的自动清理
3. **habits 表保留 `member_fact_ids`**（PrefFacts 反向引用）：drift 检测需要从 habit 追溯到原始 pref 来算 raw_value 统计
4. **Chroma 的 HABIT 存储路径删除**：`FactStoreChroma` 不再写 HABIT 类型 Fact，HABIT 类型从 Fact 枚举中保留但路由到 SQLite
5. **habits 表 schema（初稿，Round 后续细化）：**
   ```sql
   CREATE TABLE habits (
     habit_id TEXT PRIMARY KEY,
     batch_id INTEGER NOT NULL,           -- 递增整数，便于按 batch 排序清理
     text TEXT NOT NULL,                  -- GPT reword 后的文本
     signal_category TEXT NOT NULL,
     context_time_bucket TEXT,
     context_vehicle_state TEXT,
     context_geofence TEXT,
     context_weekday INTEGER,             -- NULL / 0 / 1
     raw_value_stats_json TEXT,           -- 数值统计（mean/std/min/max）
     member_fact_ids_json TEXT NOT NULL,  -- JSON array of pref fact IDs
     scene_card_id TEXT,                  -- 当前所属 pending 场景卡 ID，可空
     created_at TIMESTAMP NOT NULL
   );
   CREATE INDEX idx_habits_batch_id ON habits(batch_id);
   CREATE INDEX idx_habits_signal_category ON habits(signal_category);
   ```

**用户明确指示（项目原则）：**
> "round8 不需要考虑，长时间内不需要考虑车端迁移的事"

含义：
- 当前 brainstorm 不再为车端迁移做任何设计预留
- 未来若干轮讨论中也不要主动提车端迁移话题
- 技术选型可以按"demo 优先、正确性优先"来做，不必为"车端友好性"打折
- **例外**：如果用户主动说"某功能的最终形态留给车端迁移计划"，则作为"车端迁移计划"的延迟项登记，不在当前实施范围内

---

### Round 8 — 场景卡 Schema + 状态机 (议题 A) [已锁定]

**输出内容：**
- `scene_cards` 表完整 schema（一张表承载 4 个状态）
- `habits` 表微调（新增 `structural_key` 字段便于聚合查询）
- 状态机转换图
- 4 个实现细节问题（Q8.1~Q8.4）

**关键设计要点：**
1. 一张表 + `status` 字段承载所有状态，不拆多表
2. `structural_key` 对 pending/recommendation 是"当前 batch 算出的值"，对 accepted 是"冻结值"
3. `content_snapshot_json` 是 denormalized 冗余 — accepted 卡不引用 habits 表，自带完整快照
4. `supersede_candidate_for_json` 是 JSON array，同时支持 modify(1→1)/split(1→N)/merge(N→1)
5. retired 卡永不删除（SQLite 留档）
6. 不变式：accepted 卡只能由用户显式动作退出 accepted 状态

**用户答复：**

> "Q8.1 我希望的最终效果是弹提示框提示旧的可能被替换，用户确认了再替换，但这一段记入迁移到车端计划里，暂时使用 A；Q8.2 同意 B；Q8.3 同意 B，因为不论用户看没看到推送的场景卡模板，对于系统来说没接受=可以被替换为新的；Q8.4，同意方案 A，但 UI 这部分改变也是记入迁移到车端计划里，短期内不予考虑，数据端不做任何处理是对的。"

**最终决策：**

- **Q8.1 = A（当前实施：自动替换）**
  - 用户接受 recommendation 后，被指向的旧 accepted 卡立即 status→retired、retired_at=now
  - 新 recommendation 卡 status→accepted
  - 推送给用户的流程是一键生效，不走第二步确认
  - **DCM-1 登记**：最终产品形态是"弹提示框确认后再替换"，留给车端迁移计划

- **Q8.2 = B（按 supersede target 去重）**
  - 每次 batch 检测到漂移时，先查 `scene_cards WHERE status='recommendation' AND json(supersede_candidate_for_json) contains <accepted_id>`
  - 如果已存在 recommendation 指向同一 accepted 卡：upsert（更新 content_snapshot_json、last_reinforced_batch_id）
  - 否则 insert 新 recommendation
  - **好处：**用户界面上看到的 recommendation 卡 ID 稳定，不会在 batch 之间无故消失重建

- **Q8.3 = B（按 structural_key 对 pending 做 upsert）**
  - 用户的解释："不论用户看没看到推送的场景卡模板，对于系统来说没接受=可以被替换为新的"
  - 这个解释**比我原来的"UX 稳定"理由更本质**：pending 卡没有"神圣性"，系统可以按需处理（upsert 也好、重建也好，只要语义等价）
  - 实现采用 upsert 是因为它是"语义等价 + UX 更稳"的最优解，不是为了保护 pending 卡的 ID
  - 具体流程：batch 开始时，新产出的 pending 卡按 structural_key 匹配已有 pending 卡 → 命中则更新 content/last_reinforced_batch_id/display_name，未命中则 insert；batch 结束时清理"旧 pending 卡在新 batch 中找不到对应 structural_key"的行（硬删除，pending 不留档）

- **Q8.4 = A（数据层不做 dormant 处理）**
  - scene_cards 表**不**加 `is_dormant` 字段
  - `last_reinforced_batch_id` 字段提供原始数据，由 UI 层自己决定如何可视化
  - **DCM-2 登记**：dormant 场景卡的 UI 可视化（灰色/折叠/提示）留给车端迁移计划

**对后续设计的影响：**
- Round 9（drift 检测算法）可以直接调用 `structural_key` 匹配逻辑
- 不需要加 dormant 定时任务
- recommendation 卡的去重逻辑成为 drift 检测 pipeline 的一步
- pending 卡的 upsert 逻辑要和 drift 检测算法协调（同一次 batch 里，一个新候选要么是 pending upsert、要么是 recommendation，不能两者都是）

---

### 车端迁移计划（DCM - Deferred for Car Migration）

这一节**不是当前实施范围**，只作为延迟项登记簿。车端迁移被用户明确要求"长时间内不讨论"；但 brainstorm 过程中冒出来的"最终形态需要改动但当前先简化"的决定会被登记在这里，未来真正规划车端迁移时一起处理。

**登记项列表：**

| ID | 来源 | 延迟项描述 | 当前方案 | 最终形态 |
|---|---|---|---|---|
| DCM-1 | Round 8 Q8.1 | 用户接受 recommendation 时的替换流程 | 自动替换（accepted → retired 即时生效）| 弹确认提示框"旧的 '[场景名]' 将被替换，确认？"，用户二次确认后才替换 |
| DCM-2 | Round 8 Q8.4 | 长期未激活的 accepted 卡 UI 处理 | 数据层不做任何处理，只暴露 `last_reinforced_batch_id` 原始数据 | UI 根据 dormant 时长做分组/灰色/折叠/"这张卡最近 N 天没发生，是否保留？"提示 |
| DCM-3 | Round 9 Q9.1 | Split / merge drift 的自动识别 | **不做**：drift_type 只有 `modify`；split/merge 的情况表现为"旧 accepted 卡失去 reinforce + 新 pending 卡独立出现"，两件事不被系统自动链接 | 方案 B+C 混合：严格结构键子集关系（B）做主识别，内容相似度（C）做兜底，recommendation 卡带 `detection_confidence` 字段区分严格 vs 模糊匹配 |

未来新增登记项会持续追加到这张表。

---

### Round 9 — Drift 检测算法细节 (议题 B) [进行中]

**问题背景：**
Round 9 是把 Round 4 Q4.2（混合判定）+ Round 5 Q5.3（结构键匹配 + 推荐卡）+ Round 8（状态机）粘起来的算法层。drift 检测要把一批新候选场景卡分类为 5 种情况之一：
1. 完全新场景 → pending upsert
2. Reinforce：结构键 = accepted 且内容没变 → 更新 last_reinforced_batch_id
3. Modify drift：结构键 = accepted 但内容变了 → recommendation
4. Split drift：N 张新候选共同构成一张 accepted 的拆分
5. Merge drift：1 张新候选覆盖多张 accepted

拆分为 3 个子问题：
- Q9.1：Split/merge 识别算法
- Q9.2：Modify drift 的具体判定（Round 4 Q4.2 落地）
- Q9.3：Drift 检测 pipeline 的 batch 编排

---

#### Round 9 Q9.1 — Split / Merge 识别算法 [已锁定]

**候选方案：**
- A — 只识别 modify，不做 split/merge；split/merge 表现为"两件独立的事"
- B — 严格结构键子集关系识别（上下文三维全等 + 信号集合子集）
- C — 内容相似度模糊匹配
- D — B+C 混合（严格主 + 模糊兜底）

**我的推荐：B**（零假阳性优先）

**用户答复：**
> "暂时先按 A 共存做，B+C 的方案也是先记入，后面要做实车迁移时再做"

**最终决策：**
- **Q9.1 = A**：只实现 modify drift 检测，不做 split/merge 的自动识别
- **DCM-3 登记**：B+C 混合方案留给车端迁移计划
- **实际行为**：当真实数据里发生 split/merge 型漂移时，表现是：
  - 旧 accepted 卡的 `last_reinforced_batch_id` 停在旧批次（意味着它的结构键在新 batch 里找不到对应了）
  - 新的候选场景（属于 split/merge 的结果）作为独立的 pending 卡出现
  - 系统不建立两者之间的语义链接
  - 用户在 tab2 上看到的是"旧场景没再发生" + "新场景冒出来"，是否把两者联系起来由用户自己判断
- **不变式**：用户主权原则没被破坏（accepted 卡不会被系统静默修改）

---

#### Round 9 Q9.2 — Modify drift 的具体判定 [已锁定]

**算法总则：**
- Per-signal 独立判定
- 任一 signal 漂移即整张卡 modify drift
- recommendation 卡的 content 用 `drifted=true/false` 标注具体哪些信号变了（供 tab2 UI 高亮）

**`is_drifted(signal, old_stats, new_stats, rules)` 实现：**

```python
def is_drifted(signal, old_stats, new_stats, rules):
    metric = rules.get(signal, {}).get('drift_metric', 'embedding_fallback')
    
    if metric == 'numeric_mean_diff':
        threshold = rules[signal]['drift_threshold']
        return abs(new_stats['mean'] - old_stats['mean']) > threshold
    
    elif metric == 'dominant_value_change':
        return new_stats['dominant_value'] != old_stats['dominant_value']
    
    elif metric == 'embedding_fallback':
        threshold = 0.15  # cosine distance
        return cosine_distance(
            embed(old_stats['habit_text']),
            embed(new_stats['habit_text']),
        ) > threshold
```

**yaml schema 扩展（对接 Round 2 的 signal_rules.yaml）：**

```yaml
signals:
  hvac_temp_target:
    category: numeric
    drift_metric: numeric_mean_diff
    drift_threshold: 1.5   # °C

  seat_heating_level:
    category: numeric
    drift_metric: numeric_mean_diff
    drift_threshold: 1     # 档位数

  music_source:
    category: categorical
    drift_metric: dominant_value_change

  hvac_fan_direction:
    category: categorical
    drift_metric: dominant_value_change

  # 未声明 drift_metric 的信号 → 自动 embedding_fallback 阈值 0.15
```

**`raw_value_stats_json` 规范结构：**

```json
// 数值型信号
{
  "type": "numeric",
  "mean": 22.3,
  "std": 0.8,
  "min": 21.0,
  "max": 24.0,
  "count": 15,
  "habit_text": "set AC temperature to 22°C"
}

// 分类型信号
{
  "type": "categorical",
  "dominant_value": "jazz_playlist",
  "value_counts": {"jazz_playlist": 12, "podcast": 2},
  "count": 14,
  "habit_text": "play jazz music during morning commute"
}
```

**用户答复：**
> "除 9.2.d 全部采纳，任一 HABIT 发生漂移即判定漂移推送"

**最终决策：**

- **Q9.2.a 数值度量**：`abs(new_mean - old_mean) > yaml_threshold`（绝对差值，阈值以物理单位写 yaml）
- **Q9.2.b 分类度量**：`dominant_value` 变化即漂移
- **Q9.2.c 触发规模**：任一 signal 漂移就触发整张卡 modify drift 推送
- **Q9.2.d 证据数守门**：**不加**。哪怕新候选的 signal count=1，只要检测到漂移就推送 recommendation。
  - 设计理念：系统的职责是**尽快把"可能的变化"告诉用户**，让用户在 UI 上判断是否接受；抑制弱证据信号违背"用户主权"的精神（应该让用户有选择权，不是让系统代替用户判断"证据够不够"）
  - 边界情况：count=0 不会出现——结构键要求 signal_type_set 匹配，若某 signal 完全消失则结构键已变，不会进入 modify drift 判定路径
- **yaml 扩展格式**：如上；每个 numeric 信号加 `drift_metric + drift_threshold`，categorical 只加 `drift_metric`，未配置的信号自动走 embedding fallback
- **raw_value_stats_json 结构**：按上面两种 type 规范化

---

#### Round 9 Q9.3 — Batch 编排 pipeline [已锁定]

**Pipeline 10 步概览：**

```
Step 0 — 分配 batch_id（= MAX(batch_id) + 1）
Step 1 — 从 Chroma 拉取 TTL 窗口内的 PrefFacts
Step 2 — 聚类（Hybrid DBSCAN + 连续性过滤 + GPT reword → habits）
Step 3 — 写 habits 到 SQLite，附 batch_id 和 structural_key
Step 4 — 按 structural_key 分组 → 候选场景卡 + GPT 命名
Step 5 — 分类每个候选：reinforce / modify / pending
Step 6a — Modify drift：按 Q8.2 upsert recommendation 卡，更新 A.last_reinforced_batch_id
Step 6b — Reinforce：仅更新 A.last_reinforced_batch_id
Step 6c — Pending upsert：按 Q8.3 按 structural_key upsert pending 卡
Step 7 — 清理陈腐 pending（硬删除 last_reinforced_batch_id < 当前的 pending 卡）
Step 8 — 清理陈腐 recommendation（决策点 1）
Step 9 — 清理旧 batch 的 habits（DELETE WHERE batch_id < 当前 - 2，保留最近 3 批）
Step 10 — COMMIT 或 ROLLBACK
```

**决策点 1 — 陈腐 recommendation 的清理策略**

候选：A1=永不删除 / A2=硬删除 / A3=软删除（加 is_stale）

**用户答复：A2（硬删除）**

SQL 实现：
```sql
DELETE FROM scene_cards
 WHERE status = 'recommendation'
   AND last_reinforced_batch_id < :current_batch_id;
```

含义：recommendation 是系统建议（非用户主权范畴），数据不再支持时系统有权撤回；tab2 始终展示与最新数据一致的建议，不会留下误导性的陈腐推荐。

**决策点 2a — Chroma / SQLite 事务边界**

候选：A=先写 Chroma（无事务），再开 SQLite 事务 / B=延迟 Chroma 写入到事务尾部

**用户答复：A（采纳推荐）**

理由：Chroma 无真正事务支持，跨库事务不可能；方案 A 利用幂等性——SQLite 事务 rollback 后，Chroma 里多写的 PrefFacts 下次 batch 会被重新处理，不产生副作用。

**决策点 2b — GPT 调用是否放在 SQLite 事务内**

候选：A=GPT 调用在事务外 / B=GPT 调用在事务内

**用户答复：A（明确强调"首先要保证 GPT 在事务外"）**
> "同意你的推荐，我觉得演示很难会出现事务崩溃，当然首先要保证 GPT 在事务外"

**最终 pipeline 代码形态：**

```python
def run_batch():
    batch_id = allocate_next_batch_id()
    
    # ═══ 阶段 1：慢动作（事务外） ═══
    prefs = chroma.fetch_window()                  # Chroma 读
    habits = cluster_and_reword(prefs)             # 调 GPT reword
    candidates = group_by_structural_key(habits)
    for c in candidates:
        c.display_name = gpt_scene_name(c)         # 调 GPT 命名
    # 所有慢动作完成，结果暂存在 Python 内存
    
    # ═══ 阶段 2：快动作（SQLite 事务内，~100ms） ═══
    with sqlite.transaction():                     # BEGIN
        write_habits(habits, batch_id)             # Step 3
        for c in candidates:                       # Step 5, 6a/6b/6c
            classify_and_write_scene_card(c, batch_id)
        cleanup_stale_pending(batch_id)            # Step 7
        cleanup_stale_recommendation(batch_id)     # Step 8 (A2 硬删除)
        cleanup_old_habit_batches(batch_id)        # Step 9
        # COMMIT
```

**关键不变式：**
1. 事务只包住 SQLite 写入，持锁时间 ~100ms
2. GPT 调用在事务外，失败可在 Python 层重试而不触发 rollback
3. Chroma 写入在 SQLite 事务外，靠幂等性保证恢复
4. 整个 batch 是幂等的：同一份输入数据，跑多次结果相同

---

## Round 1-9 全部锁定 — 进入统一 Spec 写作阶段

所有 brainstorm 决策已收敛，下一步是把 9 轮的结论整合成一份可实施的 **design spec**（不再是讨论日志）。
