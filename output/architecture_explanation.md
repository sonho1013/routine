# Paranomix Habit Memory — Architecture Explanation

---

## English Version

### Pipeline Overview

The system learns driver habits through a 5-stage pipeline that transforms raw vehicle signals into structured habit knowledge.

### ① Signal Input

Raw vehicle signals are captured from the cockpit system — such as HVAC temperature settings, seat heating levels, navigation destinations, window positions, driving modes, and media selections. Each signal carries a timestamp and vehicle context.

### ② Signal → Fact Conversion

Each signal is converted into two components:
- **Bare Action Fact**: A natural language description of *what* the user did, with no scene labels. Example: *"set AC temperature to 22°C"*
- **StructuredContext**: A multi-dimensional context descriptor capturing *when/where* it happened — time bucket (early_morning/morning/...), vehicle state (engine_started/parked/...), geofence (home/workplace/...), weekday flag, and hour.

> **Key design**: Text and context are intentionally separated. The same action ("set AC to 22") in different contexts (morning vs. evening) will be distinguished by the clustering algorithm, not by the text itself.

### ③ ChromaDB Storage

- Fact text is encoded into a **1536-dimensional vector** using **OpenAI ada-002** embedding model.
- Both the vector and the structured context metadata (`ctx_*` fields) are persisted in **ChromaDB**, a local vector database with persistent storage.
- This stage proves: embedding encoding works, vector storage works, and retrieval with metadata works.

### ④ Hybrid DBSCAN Clustering

The core innovation — a hybrid distance metric combining text semantics and contextual similarity:

- **Hybrid Distance** = `0.8 × text_cosine_distance + 0.2 × context_distance`
  - Text cosine distance captures *semantic similarity* (what the user did)
  - Context distance captures *situational similarity* (when/where they did it)
- **DBSCAN** (ε=0.10, min_samples=3, metric=precomputed) discovers clusters without requiring a predefined number of clusters.
- **Consecutiveness Check**: Verifies that the cluster represents *recent consecutive* behavior (e.g., the last 5 occurrences of the same action in the same context are all in the cluster). This prevents stale or interrupted patterns from becoming habits.
- **Cluster Confidence**: A statistical quality score [0, 1] computed from three dimensions:
  - Cohesion (50%): How tight the cluster is (1 − mean_intra_dist / ε)
  - Core ratio (25%): Fraction of DBSCAN core points in the cluster
  - Size factor (25%): Evidence sufficiency (cluster_size relative to min_samples)

### ⑤ Habit Output

- **LLM Synthesis** (GPT-4.1): Merges cluster member texts into a single habit description.
- **Scene Card Naming**: Groups habits by context and uses LLM to generate human-readable scene names (e.g., "Morning Commute" with confidence 0.96).
- **Habit Fact**: The final output — a structured fact containing the habit text, dominant context, clustering confidence, and scene name. This is stored back into ChromaDB for future inference.

### Validated Results (83 facts from 5-day mockup)

| Metric | Value |
|--------|-------|
| Input facts | 83 |
| Clusters discovered | 15 |
| Noise points | 7 (8.4%) |
| Scene cards generated | 3 |
| Confidence range | 0.21 – 0.83 |
| Unit tests | 154 passed |

---

## 中文版本

### 管线概览

系统通过 5 阶段管线，将原始车辆信号转化为结构化的习惯知识。

### ① 信号输入

采集座舱系统的原始车辆信号 — 包括空调温度设定、座椅加热档位、导航目的地、车窗位置、驾驶模式、媒体选择等。每个信号携带时间戳和车辆上下文。

### ② 信号 → Fact 转换

每个信号被转换为两个组件：
- **裸动作 Fact**：用自然语言描述用户*做了什么*，不含任何场景标签。例：*"set AC temperature to 22°C"*
- **结构化上下文（StructuredContext）**：多维上下文描述符，捕捉*何时/何地*发生 — 时段（early_morning/morning/…）、车辆状态（engine_started/parked/…）、地理围栏（home/workplace/…）、工作日标志、小时。

> **核心设计**：文本与上下文刻意分离。相同动作（"设空调22°C"）在不同上下文（早晨 vs. 傍晚）中会被聚类算法区分开，而非靠文本本身。

### ③ ChromaDB 存储

- Fact 文本通过 **OpenAI ada-002** 模型编码为 **1536 维向量**。
- 向量和结构化上下文元数据（`ctx_*` 字段）一起持久化存储在 **ChromaDB**（本地向量数据库）。
- 此阶段证明：向量编码可用、向量存储可用、带元数据的检索可用。

### ④ Hybrid DBSCAN 聚类

核心创新 — 结合文本语义和上下文相似度的混合距离度量：

- **混合距离** = `0.8 × 文本余弦距离 + 0.2 × 上下文距离`
  - 文本余弦距离捕捉*语义相似性*（用户做了什么）
  - 上下文距离捕捉*场景相似性*（何时何地做的）
- **DBSCAN**（ε=0.10, min_samples=3, metric=precomputed）自动发现聚类，无需预设聚类数。
- **连续性验证**：验证聚类是否代表*最近连续*的行为（如：同一上下文下最近5次相同动作是否全在聚类中）。防止过期或中断的模式被误判为习惯。
- **聚类置信度**：基于三个维度的统计质量分数 [0, 1]：
  - 紧凑度 cohesion（50%）：聚类有多紧密（1 − 平均簇内距离 / ε）
  - 核心点占比 core_ratio（25%）：聚类中 DBSCAN 核心点的比例
  - 证据充分度 size_factor（25%）：聚类规模相对于 min_samples 的充分程度

### ⑤ 习惯输出

- **LLM 习惯合成**（GPT-4.1）：将聚类成员文本合并为单条习惯描述。
- **场景卡命名**：按上下文分组习惯，由 LLM 生成可读场景名（如："Morning Commute"，置信度 0.96）。
- **习惯 Fact**：最终输出 — 包含习惯文本、主导上下文、聚类置信度和场景名的结构化 Fact。回写 ChromaDB 用于后续推理。

### 验证结果（83条 facts，5天模拟数据）

| 指标 | 数值 |
|------|------|
| 输入 facts | 83 |
| 发现聚类数 | 15 |
| 噪声点 | 7（8.4%） |
| 场景卡 | 3 |
| 置信度范围 | 0.21 – 0.83 |
| 单元测试 | 154 项通过 |
