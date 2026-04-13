# Tab1 Habit Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 tab1 场景卡持久化与用户主权的 drift 检测：Signal→Fact 规则化 + SQLite 四状态场景卡状态机 + 每批次漂移识别，全部在 "accepted 卡永不静默修改" 的约束下运行。

**Architecture:** 两阶段 batch pipeline — 阶段 1（事务外）做 Chroma 读、聚类、GPT reword / scene naming、drift 分类；阶段 2（SQLite 事务内）把结果原子写入 `scene_cards` / `habits` 两张表。新增 `SignalRuleEngine`（yaml-driven）、`HabitLifecycleManager`（drift detection）、`SceneCardStore` / `HabitStore`（SQLite 封装）四大组件。

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, sqlite3 (stdlib), ChromaDB（仅 PREF，保留）, scikit-learn DBSCAN, OpenAI SDK（ada-002 + GPT-4.1）, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-tab1-habit-lifecycle-design.md`

---

## File Structure

**新增文件（17 个）:**

| 路径 | 职责 |
|---|---|
| `engine/signal_rules/__init__.py` | 包标记 |
| `engine/signal_rules/rules.yaml` | 11 种信号的规则配置 + 地理围栏 |
| `engine/signal_rules/engine.py` | `SignalRuleEngine` — yaml 加载 + `signals_to_facts` + `get_rule` |
| `engine/drift_detection.py` | `is_drifted` + `compute_raw_value_stats` + `cosine_distance` 纯函数 |
| `engine/habit_lifecycle.py` | `compute_structural_key` + `classify_candidate` + `HabitLifecycleManager` |
| `panoramix_core/models/scene_card.py` | `SceneCard` Pydantic 模型 |
| `panoramix_core/models/habit.py` | `Habit` Pydantic 模型 |
| `panoramix_core/store/sqlite_schema.sql` | DDL 定义 |
| `panoramix_core/store/sqlite_connection.py` | `get_connection(db_path)` 共享连接工厂 + schema 初始化 |
| `panoramix_core/store/scene_card_store.py` | `SceneCardStore` — 所有 accepted 不变式在此强制 |
| `panoramix_core/store/habit_store.py` | `HabitStore` — batch_id 管理 + 保留 N=3 批 |
| `tests/test_signal_rule_engine.py` | SignalRuleEngine 单元测试 |
| `tests/test_structural_key.py` | structural_key 稳定性测试 |
| `tests/test_drift_detection.py` | `is_drifted` / `compute_raw_value_stats` 单元测试 |
| `tests/test_scene_card_store.py` | SceneCardStore 全 API + 不变式测试 |
| `tests/test_habit_store.py` | HabitStore CRUD + 保留策略测试 |
| `tests/test_habit_lifecycle.py` | `classify_candidate` 三分支集成测试 |
| `tests/test_multi_batch_drift.py` | 端到端多批次场景测试 |

**修改文件（3 个）:**

| 路径 | 变更 |
|---|---|
| `engine/signal_to_fact.py` | 变薄为 ~25 行 shim，委托给全局 `SignalRuleEngine` 单例 |
| `engine/habit_engine.py` | `ingest_signal_batch()` 重写为两阶段 pipeline |
| `engine/scene_card.py` | 原 in-memory dataclass 删除；保留 `generate_scene_cards` 作为候选构造辅助 |

**删除/弃用（Wave 5）:**

| 路径 | 处置 |
|---|---|
| `panoramix_core/store/fact_store_chroma.py` 中 HABIT 写入路径 | 删除（HABIT 不再进 Chroma） |
| `tests/test_scene_card.py` 老式 dataclass 测试 | 迁移关键断言到新 store 测试，老文件删除 |

---

## Task 0: Git Bootstrap

**目的:** 项目当前不是 git 仓库，为后续任务提供逐任务回滚的安全网。

**Files:**
- Create: `.gitignore`
- Modify: 无

- [ ] **Step 0.1: 确认当前不是 git 仓库**

Run: `git rev-parse --is-inside-work-tree 2>&1 || echo "not a git repo"`
Expected: `not a git repo`

- [ ] **Step 0.2: 初始化 git 仓库**

Run: `git init && git branch -m main`
Expected: `Initialized empty Git repository in .../habit-memory-demo/.git/`

- [ ] **Step 0.3: 写 `.gitignore`**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.venv/
venv/

# Chroma 持久化存储
storage/memories/
simulator/storage/memories/

# 新的 SQLite 状态机数据库（运行时生成，不入库）
storage/habit_memory.db
storage/habit_memory.db-journal
storage/habit_memory.db-wal
storage/habit_memory.db-shm

# 测试输出
tests/*.json
tests/EMBEDDING_VALIDATION_REPORT.md

# 环境变量
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
```

- [ ] **Step 0.4: baseline commit**

Run: `git add -A && git status`
Expected: 看到所有源码、docs/ 被 staged，`storage/memories/` 等被忽略。

Run: `git commit -m "chore: initial baseline before tab1 habit lifecycle work"`
Expected: 单次大 commit，不输出错误。

- [ ] **Step 0.5: 验证 spec 和 plan 文档已入库**

Run: `git log --oneline -1 && git ls-files docs/superpowers/ | head`
Expected: 看到 `docs/superpowers/specs/2026-04-09-tab1-habit-lifecycle-design.md` 和 `docs/superpowers/plans/2026-04-09-tab1-habit-lifecycle-plan.md` 在输出中。

---

## Wave 1 — SignalRuleEngine (Round 2)

**目标:** 消除 `engine/signal_to_fact.py` 的 if/elif 硬编码，改为 yaml-driven。为 Wave 3 的 drift detection 提供 `drift_metric + drift_threshold` 元数据基础。

**Wave 验收:** `pytest tests/test_signal_rule_engine.py tests/test_signal_to_fact.py tests/test_e2e_pipeline.py -v` 全部通过；`engine/habit_engine.py` 不需要修改仍可运行。

---

### Task 1: Create signal_rules package + YAML rules file

**Files:**
- Create: `engine/signal_rules/__init__.py`
- Create: `engine/signal_rules/rules.yaml`

- [ ] **Step 1.1: 创建包目录**

Create `engine/signal_rules/__init__.py`:

```python
"""Signal rule engine — yaml-driven signal→Fact translation."""
from engine.signal_rules.engine import SignalRuleEngine

__all__ = ["SignalRuleEngine"]
```

- [ ] **Step 1.2: 写 `rules.yaml`（全量 11 个信号 + 地理围栏）**

Create `engine/signal_rules/rules.yaml`:

```yaml
# ── 地理围栏（迁移自 engine/signal_to_fact.py KNOWN_GEOFENCES）──
geofences:
  home:         { lat: 48.8566, lon: 2.3522, radius_m: 200 }
  workplace:    { lat: 48.8738, lon: 2.2950, radius_m: 100 }
  toll_A6:      { lat: 48.7890, lon: 2.3100, radius_m: 100 }
  parking_mall: { lat: 48.8450, lon: 2.3700, radius_m: 100 }

# ── 信号规则 ──
signals:

  # ── Table 5 行为控制信号（产生 PREF fact.text）──

  hvac_temp_target:
    category: numeric
    aliases: []
    unit: "celsius"
    value_range: [10, 32]
    text_template: "set cabin air conditioning temperature to {value} degrees"
    drift_metric: numeric_mean_diff
    drift_threshold: 1.5

  hvac_power:
    category: categorical
    value_templates:
      off: "turned off cabin air conditioning"
    drift_metric: dominant_value_change

  seat_heating:
    category: numeric
    value_range: [0, 5]
    text_template: "turned on seat heating to level {value}"
    trigger_condition: "value > 0"
    drift_metric: numeric_mean_diff
    drift_threshold: 1

  nav_destination:
    category: categorical
    text_template: "started navigation to {value}"
    drift_metric: none

  nav_route_pref:
    category: categorical
    text_template: "selected {value} route preference"
    drift_metric: dominant_value_change

  media_source:
    category: categorical
    value_templates:
      music:   "started playing music {content_id}"
      podcast: "resumed listening to podcast {content_id}"
      radio:   "tuned to radio station {content_id}"
      off:     "stopped all media playback"
    drift_metric: dominant_value_change

  media_content_id:
    category: categorical
    text_template: "selected media content {value}"
    drift_metric: dominant_value_change

  media_volume:
    category: numeric
    value_range: [0, 100]
    text_template: "adjusted media volume to {value} percent"
    trigger_condition: "value > 0"
    drift_metric: numeric_mean_diff
    drift_threshold: 10

  drive_mode:
    category: categorical
    text_template: "switched to {value} driving mode"
    drift_metric: dominant_value_change

  acc_distance:
    category: numeric
    value_range: [1, 5]
    text_template: "set ACC following distance to {value}"
    drift_metric: numeric_mean_diff
    drift_threshold: 1

  window_position:
    category: numeric
    value_range: [0, 100]
    # 特殊 template: value>0 vs value==0 使用不同模板
    value_templates_by_range:
      positive: "lowered driver window to {value} percent"
      zero:     "closed all vehicle windows"
    drift_metric: numeric_mean_diff
    drift_threshold: 20

  keyless_entry:
    category: categorical
    value_templates:
      disabled: "disabled keyless proximity entry"
    drift_metric: dominant_value_change

  engine_status:
    category: categorical
    value_templates:
      off: "shut down the vehicle engine"
    drift_metric: dominant_value_change

# ── Table 4 上下文信号（映射到 StructuredContext，不产生 Fact.text）──
context_signals:
  - gps_latitude
  - gps_longitude
  - gear_position
  - vehicle_speed
```

- [ ] **Step 1.3: Commit**

Run:
```bash
git add engine/signal_rules/__init__.py engine/signal_rules/rules.yaml
git commit -m "feat(wave1): add signal_rules package and rules.yaml"
```
Expected: 1 new commit.

---

### Task 2: SignalRuleEngine tests (TDD — red)

**Files:**
- Create: `tests/test_signal_rule_engine.py`

- [ ] **Step 2.1: 写失败测试**

Create `tests/test_signal_rule_engine.py`:

```python
"""SignalRuleEngine 单元测试。"""
import pytest

from engine.signal_rules import SignalRuleEngine
from panoramix_core.models.fact_enums import FactType


@pytest.fixture
def engine():
    return SignalRuleEngine()  # 默认加载 engine/signal_rules/rules.yaml


# ── 加载 ──

def test_engine_loads_default_rules(engine):
    signals = engine.list_signals()
    assert "hvac_temp_target" in signals
    assert "media_source" in signals
    assert len(signals) >= 11

def test_engine_loads_geofences(engine):
    geos = engine.get_geofences()
    assert "home" in geos
    assert geos["home"]["radius_m"] == 200

def test_get_rule_returns_rule_dict(engine):
    rule = engine.get_rule("hvac_temp_target")
    assert rule["category"] == "numeric"
    assert rule["drift_metric"] == "numeric_mean_diff"
    assert rule["drift_threshold"] == 1.5

def test_get_rule_missing_returns_none(engine):
    assert engine.get_rule("nonexistent_signal") is None


# ── signals_to_facts: numeric ──

def test_signals_to_facts_hvac_temp(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert len(facts) == 1
    assert facts[0].type == FactType.PREF
    assert "22" in facts[0].text
    assert "temperature" in facts[0].text.lower()


# ── signals_to_facts: categorical (value_templates) ──

def test_signals_to_facts_media_music(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "media_source", "value": "music"},
            {"t": "2026-04-08T08:00:01", "signal": "media_content_id", "value": "jazz_playlist"},
        ],
    }
    facts = engine.signals_to_facts(event)
    music_facts = [f for f in facts if "playing music" in f.text]
    assert len(music_facts) == 1
    assert "jazz_playlist" in music_facts[0].text


# ── trigger_condition ──

def test_signals_to_facts_seat_heating_off_skipped(engine):
    """seat_heating=0 不应产生 Fact（trigger_condition: value > 0）"""
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "seat_heating", "value": 0},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert all("seat heating" not in f.text for f in facts)


# ── value_range 越界丢弃 ──

def test_signals_to_facts_hvac_out_of_range_dropped(engine, caplog):
    """hvac_temp_target=100 越界（value_range=[10,32]），应丢弃并记日志"""
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 100},
        ],
    }
    facts = engine.signals_to_facts(event)
    temp_facts = [f for f in facts if "temperature" in f.text]
    assert len(temp_facts) == 0


# ── 未知信号静默丢弃 ──

def test_signals_to_facts_unknown_signal_dropped(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "totally_unknown_xyz", "value": 42},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert facts == []


# ── json_metadata 携带 signal + raw_value ──

def test_fact_json_metadata_contains_signal_and_raw_value(engine):
    import json
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
        ],
    }
    facts = engine.signals_to_facts(event)
    meta = json.loads(facts[0].json_metadata)
    assert meta["signal"] == "hvac_temp_target"
    assert meta["raw_value"] == 22


# ── StructuredContext 提取 ──

def test_structured_context_extraction(engine):
    event = {
        "date": "2026-04-08",  # 周三，工作日
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
            {"t": "2026-04-08T08:00:01", "signal": "engine_status", "value": "on"},
            {"t": "2026-04-08T08:00:02", "signal": "gps_latitude", "value": 48.8566},
            {"t": "2026-04-08T08:00:03", "signal": "gps_longitude", "value": 2.3522},
        ],
    }
    facts = engine.signals_to_facts(event)
    ctx = facts[0].context
    assert ctx.time_bucket == "early_morning"
    assert ctx.weekday is True
    assert ctx.geofence == "home"
    assert ctx.vehicle_state == "engine_started"
```

- [ ] **Step 2.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_signal_rule_engine.py -v 2>&1 | tail -20`
Expected: 全部 ERROR/FAIL（因为 `engine.signal_rules.engine` 还不存在）

- [ ] **Step 2.3: Commit**

Run:
```bash
git add tests/test_signal_rule_engine.py
git commit -m "test(wave1): add failing SignalRuleEngine tests"
```

---

### Task 3: SignalRuleEngine implementation

**Files:**
- Create: `engine/signal_rules/engine.py`

- [ ] **Step 3.1: 写最小实现**

Create `engine/signal_rules/engine.py`:

```python
"""
SignalRuleEngine — yaml-driven signal→Fact 翻译引擎

替代 engine/signal_to_fact.py 的硬编码 if/elif。为 drift detection
提供 drift_metric + drift_threshold 元数据。
"""
import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import yaml

from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources

log = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(__file__), "rules.yaml"
)


class SignalRuleEngine:
    """yaml-driven signal→Fact 翻译引擎"""

    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or _DEFAULT_RULES_PATH
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)
        log.info(
            f"SignalRuleEngine loaded {len(self._rules.get('signals', {}))} "
            f"signal rules from {self.rules_path}"
        )

    # ── 规则查询 ──

    def list_signals(self) -> List[str]:
        return list(self._rules.get("signals", {}).keys())

    def get_rule(self, signal_name: str) -> Optional[Dict]:
        return self._rules.get("signals", {}).get(signal_name)

    def get_geofences(self) -> Dict[str, Dict]:
        return self._rules.get("geofences", {})

    def context_signals(self) -> List[str]:
        return self._rules.get("context_signals", [])

    # ── 主接口 ──

    def signals_to_facts(self, event_data: Dict) -> List[Fact]:
        signals_list = event_data.get("signals", [])
        if not signals_list:
            return []

        sig_map = {s["signal"]: s["value"] for s in signals_list}
        ctx = self._extract_structured_context(event_data, sig_map)
        ts = self._parse_timestamp(event_data)

        facts: List[Fact] = []
        for signal_name, value in sig_map.items():
            rule = self.get_rule(signal_name)
            if rule is None:
                continue  # 未知/上下文信号跳过

            text = self._render_text(signal_name, value, rule, sig_map)
            if text is None:
                continue  # trigger_condition 不满足 / 越界 / 无匹配模板

            facts.append(
                self._make_fact(text, ctx, ts, signal_name, value)
            )
        return facts

    # ── 内部：规则应用 ──

    def _render_text(
        self, signal_name: str, value, rule: Dict, sig_map: Dict
    ) -> Optional[str]:
        # 1. value_range 校验
        vr = rule.get("value_range")
        if vr is not None and isinstance(value, (int, float)):
            if not (vr[0] <= value <= vr[1]):
                log.warning(
                    f"{signal_name}={value} out of range {vr}, dropped"
                )
                return None

        # 2. trigger_condition 校验
        tc = rule.get("trigger_condition")
        if tc is not None and not self._eval_trigger(tc, value):
            return None

        # 3. 选 template
        vtbr = rule.get("value_templates_by_range")
        if vtbr is not None and isinstance(value, (int, float)):
            tpl = vtbr.get("positive") if value > 0 else vtbr.get("zero")
            if tpl is None:
                return None
            return tpl.format(value=value)

        vt = rule.get("value_templates")
        if vt is not None:
            tpl = vt.get(str(value))
            if tpl is None:
                return None
            content_id = sig_map.get("media_content_id", "unknown")
            return tpl.format(value=value, content_id=content_id)

        tpl = rule.get("text_template")
        if tpl is not None:
            return tpl.format(value=value)

        return None

    def _eval_trigger(self, condition: str, value) -> bool:
        """
        安全求值 trigger_condition。
        支持: "value > N", "value >= N", "value < N", "value <= N",
              "value == X", "value != X"
        拒绝任何其他语法。
        """
        cond = condition.strip()
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if cond.startswith(f"value {op} "):
                rhs = cond.removeprefix(f"value {op} ").strip()
                try:
                    rhs_val = float(rhs) if rhs.replace(".", "").replace("-", "").isdigit() else rhs
                except Exception:
                    return False
                if op == ">":  return isinstance(value, (int, float)) and value > rhs_val
                if op == "<":  return isinstance(value, (int, float)) and value < rhs_val
                if op == ">=": return isinstance(value, (int, float)) and value >= rhs_val
                if op == "<=": return isinstance(value, (int, float)) and value <= rhs_val
                if op == "==": return value == rhs_val
                if op == "!=": return value != rhs_val
        log.error(f"Unsupported trigger_condition: {condition!r}")
        return False

    # ── 内部：StructuredContext 提取 ──

    def _extract_structured_context(
        self, event_data: Dict, sig_map: Dict
    ) -> StructuredContext:
        signals_list = event_data.get("signals", [])
        return StructuredContext(
            time_bucket=self._classify_time_bucket(signals_list),
            hour=self._extract_hour(signals_list),
            weekday=self._extract_weekday(event_data),
            vehicle_state=self._classify_vehicle_state(signals_list, sig_map),
            geofence=self._match_geofence(
                sig_map.get("gps_latitude"),
                sig_map.get("gps_longitude"),
            ),
        )

    def _classify_time_bucket(self, signals_list: list) -> str:
        h = self._extract_hour(signals_list)
        if h < 0: return "unknown"
        if 5 <= h < 9:  return "early_morning"
        if 9 <= h < 12: return "morning"
        if 12 <= h < 14: return "midday"
        if 14 <= h < 18: return "afternoon"
        if 18 <= h < 22: return "evening"
        return "night"

    def _extract_hour(self, signals_list: list) -> int:
        if not signals_list:
            return -1
        try:
            return int(signals_list[0]["t"].split("T")[1].split(":")[0])
        except (IndexError, ValueError, KeyError, AttributeError):
            return -1

    def _extract_weekday(self, event_data: Dict) -> Optional[bool]:
        date_str = event_data.get("date", "")
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5
        except ValueError:
            return None

    def _classify_vehicle_state(
        self, signals_list: list, sig_map: Dict
    ) -> str:
        engine_on = any(
            s["signal"] == "engine_status" and s["value"] == "on"
            for s in signals_list
        )
        engine_off = sig_map.get("engine_status") == "off"
        gear_park = sig_map.get("gear_position") == "P"
        speeds = [
            s["value"] for s in signals_list
            if s["signal"] == "vehicle_speed"
            and isinstance(s["value"], (int, float))
        ]
        if engine_off or gear_park:
            return "parked"
        if engine_on:
            if speeds and all(0 < sp < 5 for sp in speeds):
                return "crawling"
            return "engine_started"
        if speeds and all(0 < sp < 5 for sp in speeds):
            return "crawling"
        return "unknown"

    def _match_geofence(
        self, lat: Optional[float], lon: Optional[float]
    ) -> Optional[str]:
        if lat is None or lon is None:
            return None
        for name, cfg in self.get_geofences().items():
            glat, glon, radius = cfg["lat"], cfg["lon"], cfg["radius_m"]
            dlat = (lat - glat) * 111_320
            dlon = (lon - glon) * 111_320 * math.cos(math.radians(glat))
            if math.sqrt(dlat * dlat + dlon * dlon) < radius:
                return name
        return None

    # ── 内部：Fact 构造 ──

    def _parse_timestamp(self, event_data: Dict) -> datetime:
        signals_list = event_data.get("signals", [])
        if signals_list:
            try:
                return datetime.fromisoformat(signals_list[0]["t"])
            except (ValueError, KeyError):
                pass
        date_str = event_data.get("date", "")
        if date_str:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass
        return datetime.now()

    def _make_fact(
        self, text: str, ctx: StructuredContext, ts: datetime,
        signal_name: str, raw_value,
    ) -> Fact:
        return Fact(
            text=text,
            type=FactType.PREF,
            durability=FactDurability.LONG_TERM,
            time_stamp=ts,
            source=FactSources.SIGNAL,
            context=ctx,
            json_metadata=json.dumps({
                "signal": signal_name,
                "raw_value": raw_value,
            }),
        )
```

- [ ] **Step 3.2: 运行测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_signal_rule_engine.py -v 2>&1 | tail -30`
Expected: 所有测试 PASS。

- [ ] **Step 3.3: Commit**

Run:
```bash
git add engine/signal_rules/engine.py
git commit -m "feat(wave1): implement SignalRuleEngine"
```

---

### Task 4: Shim engine/signal_to_fact.py through SignalRuleEngine

**Files:**
- Modify: `engine/signal_to_fact.py`（完全重写为 shim）

- [ ] **Step 4.1: 重写 signal_to_fact.py**

Overwrite `engine/signal_to_fact.py`:

```python
"""
Signal-to-Fact shim — 委托给 SignalRuleEngine

向后兼容层：保留 `signals_to_facts()` 函数签名让 engine/habit_engine.py
和老测试继续工作，内部完全走新的 yaml-driven 规则引擎。

新代码应直接用 `engine.signal_rules.SignalRuleEngine`。
"""
from typing import Dict, List

from engine.signal_rules import SignalRuleEngine
from panoramix_core.models.fact import Fact

# 全局单例（yaml 只加载一次）
_engine: SignalRuleEngine | None = None


def _get_engine() -> SignalRuleEngine:
    global _engine
    if _engine is None:
        _engine = SignalRuleEngine()
    return _engine


def signals_to_facts(event_data: Dict) -> List[Fact]:
    """
    向后兼容包装器。
    新代码请直接用 `SignalRuleEngine().signals_to_facts(event_data)`。
    """
    return _get_engine().signals_to_facts(event_data)


# ── 暴露 KNOWN_GEOFENCES 让老代码/测试继续工作 ──
def _build_known_geofences():
    engine = _get_engine()
    return {
        name: (cfg["lat"], cfg["lon"], cfg["radius_m"])
        for name, cfg in engine.get_geofences().items()
    }


KNOWN_GEOFENCES = _build_known_geofences()
```

- [ ] **Step 4.2: 运行老单测验证不回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_signal_to_fact.py -v 2>&1 | tail -30`
Expected: 全部 PASS（老测试断言的文本格式与 yaml template 一致）。

如果有 failure：读 failure 信息，对比 `rules.yaml` 中的 `text_template` 与老测试期望文本，修正 yaml（不是测试）。

- [ ] **Step 4.3: 运行 e2e 回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_e2e_pipeline.py -v 2>&1 | tail -20`
Expected: 全部 PASS。

- [ ] **Step 4.4: Commit**

Run:
```bash
git add engine/signal_to_fact.py
git commit -m "refactor(wave1): shim signal_to_fact through SignalRuleEngine"
```

---

### Task 5: Wave 1 sign-off

- [ ] **Step 5.1: 全量回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -x --ignore=tests/test_proactive_executor.py 2>&1 | tail -15`
Expected: 全部 PASS（或至少没有新增 failure）。

- [ ] **Step 5.2: Wave 1 完成 commit**

Run:
```bash
git commit --allow-empty -m "milestone(wave1): SignalRuleEngine complete"
```

---

## Wave 2 — SQLite Schema + Stores (Round 6/7/8)

**目标:** 建立 `storage/habit_memory.db` 的两张表 `scene_cards` / `habits`，实现 `SceneCardStore` + `HabitStore`。所有 accepted 卡不变式在 store 层强制。不依赖 Wave 1 之外的新组件。

**Wave 验收:** `pytest tests/test_scene_card_store.py tests/test_habit_store.py -v` 全绿；`storage/habit_memory.db` 成功创建；老测试不回归。

---

### Task 6: Pydantic models (SceneCard + Habit)

**Files:**
- Create: `panoramix_core/models/scene_card.py`
- Create: `panoramix_core/models/habit.py`

- [ ] **Step 6.1: 创建 SceneCard 模型**

Create `panoramix_core/models/scene_card.py`:

```python
"""SceneCard Pydantic 模型 — 对应 scene_cards SQLite 表"""
import json
import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


SceneCardStatus = Literal['pending', 'accepted', 'recommendation', 'retired']


class SceneCard(BaseModel):
    card_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    status: SceneCardStatus
    structural_key: str
    display_name: str

    # 仅 accepted 卡使用（冻结快照）
    frozen_content_snapshot: Optional[dict] = None

    # 仅 pending/recommendation 卡使用
    content_snapshot: Optional[dict] = None

    # 仅 recommendation 卡使用
    supersede_candidate_for: Optional[list[str]] = None

    first_seen_batch_id: int
    last_reinforced_batch_id: int

    created_at: datetime
    accepted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    # ── SQLite row <-> model ──

    @classmethod
    def from_row(cls, row) -> "SceneCard":
        """row: sqlite3.Row 或 dict-like"""
        def j(k):
            v = row[k]
            return json.loads(v) if v else None

        def ts(k):
            v = row[k]
            return datetime.fromisoformat(v) if v else None

        return cls(
            card_id=row["card_id"],
            username=row["username"],
            status=row["status"],
            structural_key=row["structural_key"],
            display_name=row["display_name"],
            frozen_content_snapshot=j("frozen_content_snapshot_json"),
            content_snapshot=j("content_snapshot_json"),
            supersede_candidate_for=j("supersede_candidate_for_json"),
            first_seen_batch_id=row["first_seen_batch_id"],
            last_reinforced_batch_id=row["last_reinforced_batch_id"],
            created_at=ts("created_at"),
            accepted_at=ts("accepted_at"),
            retired_at=ts("retired_at"),
        )

    def to_row_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "username": self.username,
            "status": self.status,
            "structural_key": self.structural_key,
            "display_name": self.display_name,
            "frozen_content_snapshot_json":
                json.dumps(self.frozen_content_snapshot) if self.frozen_content_snapshot else None,
            "content_snapshot_json":
                json.dumps(self.content_snapshot) if self.content_snapshot else None,
            "supersede_candidate_for_json":
                json.dumps(self.supersede_candidate_for) if self.supersede_candidate_for else None,
            "first_seen_batch_id": self.first_seen_batch_id,
            "last_reinforced_batch_id": self.last_reinforced_batch_id,
            "created_at": self.created_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
        }
```

- [ ] **Step 6.2: 创建 Habit 模型**

Create `panoramix_core/models/habit.py`:

```python
"""Habit Pydantic 模型 — 对应 habits SQLite 表"""
import json
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Habit(BaseModel):
    habit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    batch_id: int

    text: str
    signal_category: str  # "numeric" | "categorical"
    signal_name: str
    structural_key: str

    context_time_bucket: str
    context_vehicle_state: str
    context_geofence: Optional[str] = None
    context_weekday: Optional[int] = None  # None / 0 / 1

    raw_value_stats: dict  # §3.4 中 raw_value_stats 片段
    member_fact_ids: list[str]

    scene_card_id: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_row(cls, row) -> "Habit":
        def ts(k):
            v = row[k]
            return datetime.fromisoformat(v) if v else None

        return cls(
            habit_id=row["habit_id"],
            username=row["username"],
            batch_id=row["batch_id"],
            text=row["text"],
            signal_category=row["signal_category"],
            signal_name=row["signal_name"],
            structural_key=row["structural_key"],
            context_time_bucket=row["context_time_bucket"],
            context_vehicle_state=row["context_vehicle_state"],
            context_geofence=row["context_geofence"],
            context_weekday=row["context_weekday"],
            raw_value_stats=json.loads(row["raw_value_stats_json"]),
            member_fact_ids=json.loads(row["member_fact_ids_json"]),
            scene_card_id=row["scene_card_id"],
            created_at=ts("created_at"),
        )

    def to_row_dict(self) -> dict:
        return {
            "habit_id": self.habit_id,
            "username": self.username,
            "batch_id": self.batch_id,
            "text": self.text,
            "signal_category": self.signal_category,
            "signal_name": self.signal_name,
            "structural_key": self.structural_key,
            "context_time_bucket": self.context_time_bucket,
            "context_vehicle_state": self.context_vehicle_state,
            "context_geofence": self.context_geofence,
            "context_weekday": self.context_weekday,
            "raw_value_stats_json": json.dumps(self.raw_value_stats),
            "member_fact_ids_json": json.dumps(self.member_fact_ids),
            "scene_card_id": self.scene_card_id,
            "created_at": self.created_at.isoformat(),
        }
```

- [ ] **Step 6.3: Commit**

Run:
```bash
git add panoramix_core/models/scene_card.py panoramix_core/models/habit.py
git commit -m "feat(wave2): add SceneCard and Habit Pydantic models"
```

---

### Task 7: SQLite schema + connection factory

**Files:**
- Create: `panoramix_core/store/sqlite_schema.sql`
- Create: `panoramix_core/store/sqlite_connection.py`

- [ ] **Step 7.1: 写 DDL**

Create `panoramix_core/store/sqlite_schema.sql`:

```sql
-- Tab1 habit lifecycle state machine — SQLite schema
-- 双表：scene_cards (状态机) + habits (批次产物)

CREATE TABLE IF NOT EXISTS scene_cards (
    card_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,

    status TEXT NOT NULL CHECK (status IN (
        'pending', 'accepted', 'recommendation', 'retired'
    )),

    structural_key TEXT NOT NULL,
    display_name TEXT NOT NULL,

    frozen_content_snapshot_json TEXT,
    content_snapshot_json TEXT,
    supersede_candidate_for_json TEXT,

    first_seen_batch_id INTEGER NOT NULL,
    last_reinforced_batch_id INTEGER NOT NULL,

    created_at TEXT NOT NULL,
    accepted_at TEXT,
    retired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_scene_cards_status_user
    ON scene_cards(username, status);

CREATE INDEX IF NOT EXISTS idx_scene_cards_structural_key
    ON scene_cards(username, status, structural_key);

CREATE INDEX IF NOT EXISTS idx_scene_cards_last_reinforced
    ON scene_cards(last_reinforced_batch_id);


CREATE TABLE IF NOT EXISTS habits (
    habit_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    batch_id INTEGER NOT NULL,

    text TEXT NOT NULL,

    signal_category TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    structural_key TEXT NOT NULL,

    context_time_bucket TEXT NOT NULL,
    context_vehicle_state TEXT NOT NULL,
    context_geofence TEXT,
    context_weekday INTEGER,

    raw_value_stats_json TEXT NOT NULL,
    member_fact_ids_json TEXT NOT NULL,

    scene_card_id TEXT,

    created_at TEXT NOT NULL,

    FOREIGN KEY (scene_card_id) REFERENCES scene_cards(card_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_habits_batch_id ON habits(batch_id);
CREATE INDEX IF NOT EXISTS idx_habits_username_batch ON habits(username, batch_id);
CREATE INDEX IF NOT EXISTS idx_habits_structural_key
    ON habits(username, batch_id, structural_key);
```

- [ ] **Step 7.2: 写 connection factory**

Create `panoramix_core/store/sqlite_connection.py`:

```python
"""
SQLite 连接工厂 — 共享 `storage/habit_memory.db` 的单文件

SceneCardStore + HabitStore 都通过 `get_connection(db_path)` 拿连接。
首次调用会自动执行 sqlite_schema.sql 初始化表结构。
"""
import logging
import os
import sqlite3
import threading
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "storage/habit_memory.db"

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "sqlite_schema.sql")
_initialized_paths: set[str] = set()
_init_lock = threading.Lock()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    返回新的 SQLite 连接。自动初始化 schema（每个 db_path 仅一次）。

    连接特性：
      - row_factory = sqlite3.Row
      - foreign_keys = ON
      - isolation_level = None（手动事务控制）
    """
    path = db_path or DEFAULT_DB_PATH
    _ensure_schema(path)

    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema(db_path: str) -> None:
    with _init_lock:
        if db_path in _initialized_paths:
            return

        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)

        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            ddl = f.read()

        init_conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            init_conn.executescript(ddl)
            log.info(f"SQLite schema initialized at {db_path}")
        finally:
            init_conn.close()

        _initialized_paths.add(db_path)


def reset_schema_cache() -> None:
    """测试用：清空 schema 初始化缓存"""
    with _init_lock:
        _initialized_paths.clear()
```

- [ ] **Step 7.3: 快速 smoke test**

Run:
```bash
cd /home/ae/project/paranomix/habit-memory-demo && python -c "
from panoramix_core.store.sqlite_connection import get_connection
conn = get_connection('/tmp/hm_smoke.db')
rows = conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()
print([r['name'] for r in rows])
conn.close()
import os; os.remove('/tmp/hm_smoke.db')
"
```
Expected: `['scene_cards', 'habits']`（可能额外有 sqlite_sequence，忽略）

- [ ] **Step 7.4: Commit**

Run:
```bash
git add panoramix_core/store/sqlite_schema.sql panoramix_core/store/sqlite_connection.py
git commit -m "feat(wave2): add SQLite schema and connection factory"
```

---

### Task 8: SceneCardStore tests (TDD — red)

**Files:**
- Create: `tests/test_scene_card_store.py`

- [ ] **Step 8.1: 写失败测试**

Create `tests/test_scene_card_store.py`:

```python
"""SceneCardStore 单元测试 — 覆盖所有 API + accepted 不变式"""
import os
import tempfile
from datetime import datetime

import pytest

from panoramix_core.models.scene_card import SceneCard
from panoramix_core.store.scene_card_store import (
    SceneCardStore, SceneCardImmutableError,
)
from panoramix_core.store.sqlite_connection import reset_schema_cache


@pytest.fixture
def tmp_db():
    reset_schema_cache()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def store(tmp_db):
    return SceneCardStore(username="tester", db_path=tmp_db)


def _make_pending_card(
    structural_key="sk_abc", display_name="Morning Home Routine",
    batch_id=1,
) -> SceneCard:
    return SceneCard(
        username="tester",
        status="pending",
        structural_key=structural_key,
        display_name=display_name,
        content_snapshot={"habits": []},
        first_seen_batch_id=batch_id,
        last_reinforced_batch_id=batch_id,
        created_at=datetime.now(),
    )


# ── 基本 CRUD ──

def test_insert_pending_and_get(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    got = store.get_by_status("pending")
    assert len(got) == 1
    assert got[0].card_id == card.card_id


def test_upsert_pending_by_structural_key_hit(store):
    """同一 structural_key 的第二次 upsert 应更新 last_reinforced_batch_id"""
    card = _make_pending_card(structural_key="sk_x", batch_id=1)
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_x",
            card_data={
                "display_name": "Updated Name",
                "content_snapshot": {"habits": ["h1"]},
            },
            batch_id=2,
            conn=conn,
        )

    assert new_id == card.card_id  # 同一 ID
    got = store.get_by_structural_key("sk_x", status="pending")
    assert len(got) == 1
    assert got[0].display_name == "Updated Name"
    assert got[0].last_reinforced_batch_id == 2
    assert got[0].first_seen_batch_id == 1


def test_upsert_pending_by_structural_key_miss(store):
    """无匹配 structural_key 时应 insert 新卡"""
    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_new",
            card_data={
                "display_name": "Brand New",
                "content_snapshot": {"habits": []},
            },
            batch_id=1,
            conn=conn,
        )
    got = store.get_by_status("pending")
    assert len(got) == 1
    assert got[0].card_id == new_id
    assert got[0].first_seen_batch_id == 1


# ── accept 不变式 ──

def test_accept_pending_freezes_fields(store):
    card = _make_pending_card(display_name="Original")
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    store.accept(card.card_id)

    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    a = accepted[0]
    assert a.display_name == "Original"
    assert a.frozen_content_snapshot is not None
    assert a.accepted_at is not None
    assert a.structural_key == card.structural_key


def test_accept_rejects_already_accepted(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    store.accept(card.card_id)

    with pytest.raises(SceneCardImmutableError):
        store.accept(card.card_id)


def test_cannot_upsert_pending_into_accepted(store):
    """accepted 卡的 structural_key 不应被 upsert_pending 污染"""
    card = _make_pending_card(structural_key="sk_locked")
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    store.accept(card.card_id)

    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_locked",
            card_data={"display_name": "X", "content_snapshot": {}},
            batch_id=2,
            conn=conn,
        )
    # 新 pending 卡应该是全新一张，不影响已 accepted 的
    assert new_id != card.card_id
    pending = store.get_by_status("pending")
    assert len(pending) == 1
    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    assert accepted[0].card_id == card.card_id
    assert accepted[0].display_name != "X"


# ── recommendation 去重 (Q8.2) ──

def test_upsert_recommendation_by_target_dedup(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    # batch 2: 发现 drift → 插入一张 recommendation
    with store.transaction() as conn:
        rec1_id = store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Morning (updated temp)",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=2,
            conn=conn,
        )

    # batch 3: 又发现 drift → 应该 upsert 同一张 recommendation
    with store.transaction() as conn:
        rec2_id = store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Morning (updated temp v2)",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=3,
            conn=conn,
        )

    assert rec1_id == rec2_id
    recs = store.get_by_status("recommendation")
    assert len(recs) == 1
    assert recs[0].display_name == "Morning (updated temp v2)"
    assert recs[0].last_reinforced_batch_id == 3


# ── accept recommendation 级联 retire (Q8.1 决策 A) ──

def test_accept_recommendation_retires_target_accepted(store):
    old_accepted = _make_pending_card(structural_key="sk_a", display_name="Old")
    with store.transaction() as conn:
        store.insert_pending(old_accepted, batch_id=1, conn=conn)
    store.accept(old_accepted.card_id)

    with store.transaction() as conn:
        rec_id = store.upsert_recommendation_by_target(
            target_accepted_id=old_accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "New",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=2,
            conn=conn,
        )

    store.accept(rec_id)

    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    assert accepted[0].card_id == rec_id
    assert accepted[0].display_name == "New"

    retired = store.get_by_status("retired")
    assert len(retired) == 1
    assert retired[0].card_id == old_accepted.card_id
    assert retired[0].retired_at is not None


# ── 陈腐清理 (Step 7, Step 8) ──

def test_delete_stale_pending(store):
    old = _make_pending_card(structural_key="sk_old")
    with store.transaction() as conn:
        store.insert_pending(old, batch_id=1, conn=conn)

    with store.transaction() as conn:
        count = store.delete_stale_pending(current_batch_id=2, conn=conn)

    assert count == 1
    assert store.get_by_status("pending") == []


def test_delete_stale_recommendation(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    with store.transaction() as conn:
        store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Old rec",
                "content_snapshot": {},
            },
            batch_id=2,
            conn=conn,
        )

    with store.transaction() as conn:
        count = store.delete_stale_recommendation(current_batch_id=3, conn=conn)

    assert count == 1
    assert store.get_by_status("recommendation") == []


# ── accepted 永远不被陈腐清理 ──

def test_accepted_not_affected_by_stale_cleanup(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    # 多轮清理
    for batch in range(2, 10):
        with store.transaction() as conn:
            store.delete_stale_pending(current_batch_id=batch, conn=conn)
            store.delete_stale_recommendation(current_batch_id=batch, conn=conn)

    still_there = store.get_by_status("accepted")
    assert len(still_there) == 1
    assert still_there[0].card_id == accepted.card_id


# ── reinforce ──

def test_update_last_reinforced(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    with store.transaction() as conn:
        store.update_last_reinforced(accepted.card_id, batch_id=5, conn=conn)

    got = store.get_by_status("accepted")[0]
    assert got.last_reinforced_batch_id == 5
    # 但 display_name / frozen 不变
    assert got.display_name == "Morning Home Routine"


# ── 用户主动 retire ──

def test_retire_accepted_by_user(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    store.retire(accepted.card_id)

    assert store.get_by_status("accepted") == []
    retired = store.get_by_status("retired")
    assert len(retired) == 1


def test_retire_non_accepted_raises(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    with pytest.raises(SceneCardImmutableError):
        store.retire(card.card_id)


# ── 多用户隔离 ──

def test_multi_user_isolation(tmp_db):
    reset_schema_cache()
    alice = SceneCardStore(username="alice", db_path=tmp_db)
    bob = SceneCardStore(username="bob", db_path=tmp_db)

    with alice.transaction() as conn:
        alice.insert_pending(
            SceneCard(
                username="alice", status="pending",
                structural_key="sk", display_name="Alice Card",
                content_snapshot={},
                first_seen_batch_id=1, last_reinforced_batch_id=1,
                created_at=datetime.now(),
            ),
            batch_id=1, conn=conn,
        )

    assert len(alice.get_by_status("pending")) == 1
    assert len(bob.get_by_status("pending")) == 0
```

- [ ] **Step 8.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_scene_card_store.py -v 2>&1 | tail -20`
Expected: `ModuleNotFoundError` 或 `ImportError` — store 还不存在。

- [ ] **Step 8.3: Commit**

Run:
```bash
git add tests/test_scene_card_store.py
git commit -m "test(wave2): add failing SceneCardStore tests"
```

---

### Task 9: SceneCardStore implementation

**Files:**
- Create: `panoramix_core/store/scene_card_store.py`

- [ ] **Step 9.1: 写实现**

Create `panoramix_core/store/scene_card_store.py`:

```python
"""
SceneCardStore — 场景卡状态机的 SQLite 薄封装

所有 accepted 卡的不变式（不可变 display_name/frozen/structural_key）
在此层强制。违反即抛 SceneCardImmutableError。
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

from panoramix_core.models.scene_card import SceneCard
from panoramix_core.store.sqlite_connection import get_connection

log = logging.getLogger(__name__)


class SceneCardImmutableError(Exception):
    """尝试修改 accepted 卡的不可变字段，或对错状态做状态转换"""


class SceneCardStore:
    def __init__(self, username: str, db_path: Optional[str] = None):
        self.username = username
        self.db_path = db_path
        self._conn = get_connection(db_path)

    # ── 事务 ──

    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE; yield conn; COMMIT or ROLLBACK"""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── 写入 ──

    def insert_pending(
        self, card: SceneCard, batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        assert card.status == "pending"
        self._insert(card, conn)

    def insert_recommendation(
        self, card: SceneCard, batch_id: int,
        supersede_candidate_for: list[str], *, conn: sqlite3.Connection,
    ) -> None:
        assert card.status == "recommendation"
        card.supersede_candidate_for = supersede_candidate_for
        self._insert(card, conn)

    def _insert(self, card: SceneCard, conn: sqlite3.Connection) -> None:
        row = card.to_row_dict()
        conn.execute(
            """
            INSERT INTO scene_cards (
                card_id, username, status, structural_key, display_name,
                frozen_content_snapshot_json, content_snapshot_json,
                supersede_candidate_for_json,
                first_seen_batch_id, last_reinforced_batch_id,
                created_at, accepted_at, retired_at
            ) VALUES (
                :card_id, :username, :status, :structural_key, :display_name,
                :frozen_content_snapshot_json, :content_snapshot_json,
                :supersede_candidate_for_json,
                :first_seen_batch_id, :last_reinforced_batch_id,
                :created_at, :accepted_at, :retired_at
            )
            """,
            row,
        )

    def upsert_pending_by_structural_key(
        self, structural_key: str, card_data: dict, batch_id: int,
        *, conn: sqlite3.Connection,
    ) -> str:
        """
        Q8.3: 按 structural_key 在 pending 中 upsert。
        accepted / retired / recommendation 的同 key 卡不受影响。
        """
        row = conn.execute(
            """
            SELECT card_id, first_seen_batch_id FROM scene_cards
             WHERE username = ? AND status = 'pending'
               AND structural_key = ?
             LIMIT 1
            """,
            (self.username, structural_key),
        ).fetchone()

        if row is not None:
            conn.execute(
                """
                UPDATE scene_cards
                   SET display_name = ?,
                       content_snapshot_json = ?,
                       last_reinforced_batch_id = ?
                 WHERE card_id = ?
                """,
                (
                    card_data["display_name"],
                    json.dumps(card_data.get("content_snapshot")),
                    batch_id,
                    row["card_id"],
                ),
            )
            return row["card_id"]

        new_card = SceneCard(
            username=self.username,
            status="pending",
            structural_key=structural_key,
            display_name=card_data["display_name"],
            content_snapshot=card_data.get("content_snapshot"),
            first_seen_batch_id=batch_id,
            last_reinforced_batch_id=batch_id,
            created_at=datetime.now(),
        )
        self._insert(new_card, conn)
        return new_card.card_id

    def upsert_recommendation_by_target(
        self, target_accepted_id: str, card_data: dict, batch_id: int,
        *, conn: sqlite3.Connection,
    ) -> str:
        """
        Q8.2: 查找 "supersede_candidate_for 包含 target_accepted_id" 的
        现存 recommendation；命中 upsert，未命中 insert。
        """
        rows = conn.execute(
            """
            SELECT card_id, supersede_candidate_for_json, first_seen_batch_id
              FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
            """,
            (self.username,),
        ).fetchall()

        existing_id: Optional[str] = None
        for row in rows:
            targets = json.loads(row["supersede_candidate_for_json"] or "[]")
            if target_accepted_id in targets:
                existing_id = row["card_id"]
                break

        if existing_id is not None:
            conn.execute(
                """
                UPDATE scene_cards
                   SET display_name = ?,
                       content_snapshot_json = ?,
                       structural_key = ?,
                       last_reinforced_batch_id = ?
                 WHERE card_id = ?
                """,
                (
                    card_data["display_name"],
                    json.dumps(card_data.get("content_snapshot")),
                    card_data["structural_key"],
                    batch_id,
                    existing_id,
                ),
            )
            return existing_id

        new_card = SceneCard(
            username=self.username,
            status="recommendation",
            structural_key=card_data["structural_key"],
            display_name=card_data["display_name"],
            content_snapshot=card_data.get("content_snapshot"),
            supersede_candidate_for=[target_accepted_id],
            first_seen_batch_id=batch_id,
            last_reinforced_batch_id=batch_id,
            created_at=datetime.now(),
        )
        self._insert(new_card, conn)
        return new_card.card_id

    # ── 状态转换 ──

    def accept(self, card_id: str) -> None:
        """
        pending / recommendation → accepted
        冻结 display_name + structural_key + content→frozen。
        若原为 recommendation：级联 retire 其 supersede_candidate_for 指向的 accepted 卡。
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")

            if row["status"] not in ("pending", "recommendation"):
                raise SceneCardImmutableError(
                    f"accept() requires pending/recommendation; got {row['status']}"
                )

            # 冻结：content_snapshot → frozen_content_snapshot
            content_json = row["content_snapshot_json"] or "{}"
            now_iso = datetime.now().isoformat()

            conn.execute(
                """
                UPDATE scene_cards
                   SET status = 'accepted',
                       frozen_content_snapshot_json = ?,
                       content_snapshot_json = NULL,
                       accepted_at = ?
                 WHERE card_id = ?
                """,
                (content_json, now_iso, card_id),
            )

            # 级联 retire 旧 accepted 卡
            if row["status"] == "recommendation":
                targets = json.loads(row["supersede_candidate_for_json"] or "[]")
                for target_id in targets:
                    conn.execute(
                        """
                        UPDATE scene_cards
                           SET status = 'retired',
                               retired_at = ?
                         WHERE card_id = ? AND username = ?
                               AND status = 'accepted'
                        """,
                        (now_iso, target_id, self.username),
                    )

    def retire(self, card_id: str) -> None:
        """accepted → retired（仅允许由用户主动调用或 accept() 内部级联）"""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")
            if row["status"] != "accepted":
                raise SceneCardImmutableError(
                    f"retire() requires accepted; got {row['status']}"
                )
            conn.execute(
                "UPDATE scene_cards SET status='retired', retired_at=? WHERE card_id=?",
                (datetime.now().isoformat(), card_id),
            )

    def reject_recommendation(self, card_id: str) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")
            if row["status"] != "recommendation":
                raise SceneCardImmutableError(
                    f"reject_recommendation() requires recommendation; got {row['status']}"
                )
            conn.execute(
                "DELETE FROM scene_cards WHERE card_id = ?", (card_id,),
            )

    # ── 清理 ──

    def delete_stale_pending(
        self, current_batch_id: int, *, conn: sqlite3.Connection,
    ) -> int:
        cur = conn.execute(
            """
            DELETE FROM scene_cards
             WHERE username = ? AND status = 'pending'
               AND last_reinforced_batch_id < ?
            """,
            (self.username, current_batch_id),
        )
        return cur.rowcount

    def delete_stale_recommendation(
        self, current_batch_id: int, *, conn: sqlite3.Connection,
    ) -> int:
        cur = conn.execute(
            """
            DELETE FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
               AND last_reinforced_batch_id < ?
            """,
            (self.username, current_batch_id),
        )
        return cur.rowcount

    def update_last_reinforced(
        self, card_id: str, batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        """
        只允许 reinforce accepted 卡（其它状态的 reinforce 走 upsert_* 路径）。
        不修改 display_name / frozen / structural_key。
        """
        row = conn.execute(
            "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
            (card_id, self.username),
        ).fetchone()
        if row is None:
            raise ValueError(f"card not found: {card_id}")
        if row["status"] != "accepted":
            raise SceneCardImmutableError(
                f"update_last_reinforced() targets accepted only; got {row['status']}"
            )
        conn.execute(
            "UPDATE scene_cards SET last_reinforced_batch_id = ? WHERE card_id = ?",
            (batch_id, card_id),
        )

    # ── 查询 ──

    def get_by_status(self, status: str) -> List[SceneCard]:
        rows = self._conn.execute(
            """
            SELECT * FROM scene_cards
             WHERE username = ? AND status = ?
             ORDER BY created_at
            """,
            (self.username, status),
        ).fetchall()
        return [SceneCard.from_row(r) for r in rows]

    def get_by_structural_key(
        self, structural_key: str, status: Optional[str] = None,
    ) -> List[SceneCard]:
        if status is None:
            rows = self._conn.execute(
                """
                SELECT * FROM scene_cards
                 WHERE username = ? AND structural_key = ?
                 ORDER BY created_at
                """,
                (self.username, structural_key),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM scene_cards
                 WHERE username = ? AND structural_key = ? AND status = ?
                 ORDER BY created_at
                """,
                (self.username, structural_key, status),
            ).fetchall()
        return [SceneCard.from_row(r) for r in rows]

    def get_all_accepted(self) -> List[SceneCard]:
        return self.get_by_status("accepted")

    def find_recommendation_by_target(
        self, accepted_card_id: str,
    ) -> Optional[SceneCard]:
        rows = self._conn.execute(
            """
            SELECT * FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
            """,
            (self.username,),
        ).fetchall()
        for row in rows:
            targets = json.loads(row["supersede_candidate_for_json"] or "[]")
            if accepted_card_id in targets:
                return SceneCard.from_row(row)
        return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 9.2: 运行测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_scene_card_store.py -v 2>&1 | tail -40`
Expected: 全部 PASS。

- [ ] **Step 9.3: Commit**

Run:
```bash
git add panoramix_core/store/scene_card_store.py
git commit -m "feat(wave2): implement SceneCardStore with user sovereignty invariants"
```

---

### Task 10: HabitStore tests (TDD — red)

**Files:**
- Create: `tests/test_habit_store.py`

- [ ] **Step 10.1: 写失败测试**

Create `tests/test_habit_store.py`:

```python
"""HabitStore 单元测试 — CRUD + batch 保留策略"""
import os
import tempfile
from datetime import datetime

import pytest

from panoramix_core.models.habit import Habit
from panoramix_core.store.habit_store import HabitStore
from panoramix_core.store.sqlite_connection import reset_schema_cache


@pytest.fixture
def tmp_db():
    reset_schema_cache()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def store(tmp_db):
    return HabitStore(username="tester", db_path=tmp_db)


def _make_habit(batch_id: int, signal_name: str = "hvac_temp_target") -> Habit:
    return Habit(
        username="tester",
        batch_id=batch_id,
        text=f"set temperature to 22 (batch {batch_id})",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="sk_home_morning",
        context_time_bucket="early_morning",
        context_vehicle_state="engine_started",
        context_geofence="home",
        context_weekday=1,
        raw_value_stats={"type": "numeric", "mean": 22.0, "count": 10},
        member_fact_ids=["pref-1", "pref-2"],
        created_at=datetime.now(),
    )


def test_allocate_next_batch_id_on_empty(store):
    with store.transaction() as conn:
        bid = store.allocate_next_batch_id(conn=conn)
    assert bid == 1


def test_allocate_next_batch_id_increments(store):
    with store.transaction() as conn:
        store.insert_many([_make_habit(5)], batch_id=5, conn=conn)

    with store.transaction() as conn:
        bid = store.allocate_next_batch_id(conn=conn)
    assert bid == 6


def test_insert_many_and_get_by_batch(store):
    habits = [_make_habit(1), _make_habit(1, signal_name="media_source")]
    with store.transaction() as conn:
        store.insert_many(habits, batch_id=1, conn=conn)

    got = store.get_by_batch(1)
    assert len(got) == 2
    assert {h.signal_name for h in got} == {"hvac_temp_target", "media_source"}


def test_delete_old_batches_retain_3(store):
    for bid in range(1, 8):  # batches 1..7
        with store.transaction() as conn:
            store.insert_many([_make_habit(bid)], batch_id=bid, conn=conn)

    with store.transaction() as conn:
        deleted = store.delete_old_batches(current_batch_id=7, retain_n=3, conn=conn)

    # 保留 batch 5, 6, 7 → 删除 1, 2, 3, 4
    assert deleted == 4
    remaining_batches = {h.batch_id for h in store.get_by_batch(5)}
    assert store.get_by_batch(1) == []
    assert store.get_by_batch(4) == []
    assert len(store.get_by_batch(5)) == 1
    assert len(store.get_by_batch(7)) == 1


def test_get_latest_batch_id(store):
    with store.transaction() as conn:
        store.insert_many([_make_habit(1)], batch_id=1, conn=conn)
    with store.transaction() as conn:
        store.insert_many([_make_habit(5)], batch_id=5, conn=conn)

    assert store.get_latest_batch_id() == 5


def test_get_latest_batch_id_empty(store):
    assert store.get_latest_batch_id() == 0
```

- [ ] **Step 10.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_habit_store.py -v 2>&1 | tail -10`
Expected: `ModuleNotFoundError`

- [ ] **Step 10.3: Commit**

Run:
```bash
git add tests/test_habit_store.py
git commit -m "test(wave2): add failing HabitStore tests"
```

---

### Task 11: HabitStore implementation

**Files:**
- Create: `panoramix_core/store/habit_store.py`

- [ ] **Step 11.1: 写实现**

Create `panoramix_core/store/habit_store.py`:

```python
"""
HabitStore — habits 表的 SQLite 薄封装

职责：
  - batch_id 原子分配（事务内 MAX + 1）
  - insert_many / get_by_batch
  - 按 retain_n 保留最近 N 批的滑动窗口清理
"""
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from panoramix_core.models.habit import Habit
from panoramix_core.store.sqlite_connection import get_connection


class HabitStore:
    def __init__(self, username: str, db_path: Optional[str] = None):
        self.username = username
        self.db_path = db_path
        self._conn = get_connection(db_path)

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── batch_id ──

    def get_latest_batch_id(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(batch_id), 0) AS max_bid FROM habits WHERE username = ?",
            (self.username,),
        ).fetchone()
        return int(row["max_bid"])

    def peek_next_batch_id(self) -> int:
        """读-only 窥视。真正分配用 allocate_next_batch_id 在事务内做。"""
        return self.get_latest_batch_id() + 1

    def allocate_next_batch_id(self, *, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(batch_id), 0) AS max_bid FROM habits WHERE username = ?",
            (self.username,),
        ).fetchone()
        return int(row["max_bid"]) + 1

    # ── 写入 ──

    def insert_many(
        self, habits: List[Habit], batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        if not habits:
            return
        rows = []
        for h in habits:
            # 确保 batch_id 一致（防御性）
            if h.batch_id != batch_id:
                h = h.model_copy(update={"batch_id": batch_id})
            rows.append(h.to_row_dict())
        conn.executemany(
            """
            INSERT INTO habits (
                habit_id, username, batch_id, text,
                signal_category, signal_name, structural_key,
                context_time_bucket, context_vehicle_state,
                context_geofence, context_weekday,
                raw_value_stats_json, member_fact_ids_json,
                scene_card_id, created_at
            ) VALUES (
                :habit_id, :username, :batch_id, :text,
                :signal_category, :signal_name, :structural_key,
                :context_time_bucket, :context_vehicle_state,
                :context_geofence, :context_weekday,
                :raw_value_stats_json, :member_fact_ids_json,
                :scene_card_id, :created_at
            )
            """,
            rows,
        )

    def update_scene_card_id(
        self, habit_ids: List[str], scene_card_id: str, *, conn: sqlite3.Connection,
    ) -> None:
        """批处理完成后回填每个 habit 归属的 scene_card_id"""
        conn.executemany(
            "UPDATE habits SET scene_card_id = ? WHERE habit_id = ?",
            [(scene_card_id, h) for h in habit_ids],
        )

    # ── 清理 ──

    def delete_old_batches(
        self, current_batch_id: int, retain_n: int = 3,
        *, conn: sqlite3.Connection,
    ) -> int:
        """
        保留最近 retain_n 批，删除更早的。
        retain_n=3 且 current_batch_id=7 → 保留 5,6,7 → 删除 < 5
        """
        cutoff = current_batch_id - retain_n + 1
        cur = conn.execute(
            "DELETE FROM habits WHERE username = ? AND batch_id < ?",
            (self.username, cutoff),
        )
        return cur.rowcount

    # ── 查询 ──

    def get_by_batch(self, batch_id: int) -> List[Habit]:
        rows = self._conn.execute(
            """
            SELECT * FROM habits
             WHERE username = ? AND batch_id = ?
             ORDER BY created_at
            """,
            (self.username, batch_id),
        ).fetchall()
        return [Habit.from_row(r) for r in rows]

    def get_latest_batch_habits(self) -> List[Habit]:
        latest = self.get_latest_batch_id()
        if latest == 0:
            return []
        return self.get_by_batch(latest)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 11.2: 运行测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_habit_store.py -v 2>&1 | tail -20`
Expected: 全部 PASS。

- [ ] **Step 11.3: Commit**

Run:
```bash
git add panoramix_core/store/habit_store.py
git commit -m "feat(wave2): implement HabitStore with retain_n batch pruning"
```

---

### Task 12: Wave 2 sign-off

- [ ] **Step 12.1: Wave 2 全量测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_scene_card_store.py tests/test_habit_store.py -v 2>&1 | tail -15`
Expected: 全部 PASS。

- [ ] **Step 12.2: 历史测试不回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -x --ignore=tests/test_proactive_executor.py --ignore=tests/test_habit_lifecycle.py --ignore=tests/test_multi_batch_drift.py 2>&1 | tail -15`
Expected: 全部 PASS（后两个文件还没创建，所以 ignore）。

- [ ] **Step 12.3: Wave 2 milestone**

Run:
```bash
git commit --allow-empty -m "milestone(wave2): SQLite stores complete (SceneCardStore + HabitStore)"
```

---

## Wave 3 — HabitLifecycleManager (Round 9)

**目标:** 在 Wave 2 的 store 基础上，实现"候选场景卡 → (Reinforce | ModifyDrift | NewPending)"的纯分类逻辑。全部可脱离 SQL/网络测试（embedding_fallback 的 `embed` 通过依赖注入/mock）。

**Wave 验收:** `pytest tests/test_structural_key.py tests/test_drift_detection.py tests/test_habit_lifecycle.py -v` 全部通过；`engine/habit_lifecycle.py` 导出的函数签名与 §7 spec 一致。

---

### Task 13: `compute_structural_key` tests + implementation (TDD)

**Files:**
- Create: `tests/test_structural_key.py`
- Create: `engine/habit_lifecycle.py`（先只放 `compute_structural_key`，后续 Task 15 追加 classify）

- [ ] **Step 13.1: 写失败测试**

Create `tests/test_structural_key.py`:

```python
"""compute_structural_key() 稳定性与边界测试 — spec §3.1"""
import pytest

from engine.habit_lifecycle import compute_structural_key
from panoramix_core.models.habit import Habit


def _make_habit(
    signal_name: str = "hvac_temp_target",
    time_bucket: str = "early_morning",
    vehicle_state: str = "engine_started",
    geofence: str = "home",
    raw_value_stats: dict | None = None,
) -> Habit:
    return Habit(
        username="tester",
        batch_id=1,
        text=f"habit text for {signal_name}",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="placeholder",
        context_time_bucket=time_bucket,
        context_vehicle_state=vehicle_state,
        context_geofence=geofence,
        context_weekday=1,
        raw_value_stats=raw_value_stats or {"type": "numeric", "mean": 22.0, "count": 5},
        member_fact_ids=[],
    )


def test_empty_habits_raises():
    with pytest.raises(ValueError):
        compute_structural_key([])


def test_single_habit_deterministic():
    h = _make_habit()
    k1 = compute_structural_key([h])
    k2 = compute_structural_key([h])
    assert k1 == k2
    assert len(k1) == 16  # SHA-256 hex[:16]


def test_order_independence():
    """同 cluster 不同顺序应产生相同 key"""
    h1 = _make_habit(signal_name="hvac_temp_target")
    h2 = _make_habit(signal_name="seat_heating")
    assert compute_structural_key([h1, h2]) == compute_structural_key([h2, h1])


def test_dominant_geofence_majority_wins():
    habits = [
        _make_habit(geofence="home"),
        _make_habit(geofence="home"),
        _make_habit(geofence="workplace"),
    ]
    k_home = compute_structural_key(habits)
    # 换成全 workplace → key 必须变
    habits_w = [_make_habit(geofence="workplace") for _ in range(3)]
    assert compute_structural_key(habits_w) != k_home


def test_dominant_tie_broken_alphabetically():
    """平票时按字典序取第一个，确保确定性"""
    h1 = _make_habit(geofence="home")
    h2 = _make_habit(geofence="workplace")
    k_tied_1 = compute_structural_key([h1, h2])
    k_tied_2 = compute_structural_key([h2, h1])
    assert k_tied_1 == k_tied_2  # 顺序无关
    # 再加一个 home 打破平票
    k_home_wins = compute_structural_key([h1, h2, _make_habit(geofence="home")])
    # "home" 原本是字典序第一，所以平票时也选 home；加到 home 胜出时 key 不变
    assert k_tied_1 == k_home_wins


def test_none_geofence_handled():
    habits = [_make_habit(geofence=None), _make_habit(geofence=None)]
    k = compute_structural_key(habits)
    assert len(k) == 16


def test_different_signal_set_different_key():
    h1 = _make_habit(signal_name="hvac_temp_target")
    h2 = _make_habit(signal_name="media_source")
    assert compute_structural_key([h1]) != compute_structural_key([h2])


def test_weekday_not_in_key():
    """Round 5: weekday 故意不参与 key，避免过度切分"""
    h_weekday = _make_habit()
    h_weekend = _make_habit()
    h_weekend.context_weekday = 0
    assert compute_structural_key([h_weekday]) == compute_structural_key([h_weekend])


def test_time_bucket_changes_key():
    h_morning = _make_habit(time_bucket="early_morning")
    h_noon = _make_habit(time_bucket="noon")
    assert compute_structural_key([h_morning]) != compute_structural_key([h_noon])


def test_vehicle_state_changes_key():
    h_started = _make_habit(vehicle_state="engine_started")
    h_off = _make_habit(vehicle_state="engine_off")
    assert compute_structural_key([h_started]) != compute_structural_key([h_off])
```

- [ ] **Step 13.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_structural_key.py -v 2>&1 | tail -15`
Expected: `ModuleNotFoundError: No module named 'engine.habit_lifecycle'`

- [ ] **Step 13.3: 写 `compute_structural_key` 实现**

Create `engine/habit_lifecycle.py`:

```python
"""
HabitLifecycleManager — drift detection 核心 (Round 9)

本模块包含三组函数/类：
1. compute_structural_key() — 场景卡身份生成
2. classify_candidate()     — 候选场景卡分类（Reinforce/ModifyDrift/NewPending）
3. HabitLifecycleManager    — 协调器，串起 store 调用

§ 7 of design spec.
"""
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard

log = logging.getLogger(__name__)


def compute_structural_key(habits: List[Habit]) -> str:
    """
    计算场景卡身份锚点。

    Key source = hash(dominant_geofence | dominant_time_bucket |
                      dominant_vehicle_state | sorted(signal_type_set))

    weekday 故意不参与（Round 5）。
    平票时按字典序取第一个，保证确定性。

    返回 SHA-256 hex 的前 16 字符。
    """
    if not habits:
        raise ValueError("compute_structural_key requires at least one habit")

    dominant_geofence = _dominant_value(
        [h.context_geofence or "none" for h in habits]
    )
    dominant_time_bucket = _dominant_value(
        [h.context_time_bucket for h in habits]
    )
    dominant_vehicle_state = _dominant_value(
        [h.context_vehicle_state for h in habits]
    )
    signal_types = sorted({h.signal_name for h in habits})

    key_src = "|".join([
        dominant_geofence,
        dominant_time_bucket,
        dominant_vehicle_state,
        ",".join(signal_types),
    ])
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]


def _dominant_value(values: List[str]) -> str:
    """
    取出现次数最多的值；平票时按字典序取第一个。
    """
    counts = Counter(values)
    max_count = max(counts.values())
    tied = sorted(v for v, c in counts.items() if c == max_count)
    return tied[0]
```

- [ ] **Step 13.4: 运行测试确认全绿**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_structural_key.py -v 2>&1 | tail -20`
Expected: 全部 PASS（10 个测试）。

- [ ] **Step 13.5: Commit**

Run:
```bash
git add tests/test_structural_key.py engine/habit_lifecycle.py
git commit -m "feat(wave3): add compute_structural_key with order-stable tie-break"
```

---

### Task 14: drift_detection module (TDD)

**Files:**
- Create: `tests/test_drift_detection.py`
- Create: `engine/drift_detection.py`

- [ ] **Step 14.1: 写失败测试**

Create `tests/test_drift_detection.py`:

```python
"""drift_detection 纯函数单测 — spec §7.3 / §7.4"""
from unittest.mock import MagicMock

import pytest

from engine.drift_detection import (
    compute_raw_value_stats,
    cosine_distance,
    is_drifted,
)


# ── is_drifted (numeric_mean_diff) ──

def _rule_engine_with(signal: str, rule: dict):
    m = MagicMock()
    m.get_rule.side_effect = lambda s: rule if s == signal else None
    return m


def test_is_drifted_numeric_within_threshold():
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 23.0}
    assert is_drifted("hvac_temp_target", old, new, re) is False


def test_is_drifted_numeric_exceeds_threshold():
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 24.0}  # +2.0 > 1.5
    assert is_drifted("hvac_temp_target", old, new, re) is True


def test_is_drifted_numeric_boundary_equal_not_drifted():
    """严格大于阈值才算漂移"""
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 23.5}
    assert is_drifted("hvac_temp_target", old, new, re) is False


# ── is_drifted (dominant_value_change) ──

def test_is_drifted_categorical_same_dominant():
    re = _rule_engine_with("media_source", {
        "drift_metric": "dominant_value_change",
    })
    old = {"type": "categorical", "dominant_value": "music"}
    new = {"type": "categorical", "dominant_value": "music"}
    assert is_drifted("media_source", old, new, re) is False


def test_is_drifted_categorical_dominant_changed():
    re = _rule_engine_with("media_source", {
        "drift_metric": "dominant_value_change",
    })
    old = {"type": "categorical", "dominant_value": "music"}
    new = {"type": "categorical", "dominant_value": "podcast"}
    assert is_drifted("media_source", old, new, re) is True


# ── is_drifted (embedding_fallback) ──

def test_is_drifted_embedding_fallback_similar():
    """cos距离 < 0.15 认为未漂移"""
    re = MagicMock()
    re.get_rule.return_value = None  # 无规则 → fallback
    old = {"habit_text": "turned on seat heating"}
    new = {"habit_text": "turned on seat heating"}
    fake_embed = MagicMock(return_value=[1.0, 0.0, 0.0])
    assert is_drifted("unknown", old, new, re, embed_fn=fake_embed) is False


def test_is_drifted_embedding_fallback_divergent():
    re = MagicMock()
    re.get_rule.return_value = None
    old = {"habit_text": "first"}
    new = {"habit_text": "second"}

    # 两次 embed 输出几乎正交
    def fake_embed(text):
        return [1.0, 0.0, 0.0] if text == "first" else [0.0, 1.0, 0.0]

    assert is_drifted("unknown", old, new, re, embed_fn=fake_embed) is True


# ── cosine_distance ──

def test_cosine_distance_identical_is_zero():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_is_one():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_opposite_is_two():
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, abs=1e-9)


def test_cosine_distance_zero_vector_returns_one():
    """零向量没有方向 → 约定返回 1.0 (maximally distant)"""
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── compute_raw_value_stats (numeric) ──

def test_compute_raw_value_stats_numeric():
    from panoramix_core.models.fact import Fact
    from panoramix_core.models.fact_enums import FactType

    facts = []
    for val in [22.0, 23.0, 21.5, 22.5]:
        f = MagicMock()
        f.json_metadata = f'{{"signal": "hvac_temp_target", "raw_value": {val}}}'
        facts.append(f)

    stats = compute_raw_value_stats(facts, signal_category="numeric")
    assert stats["type"] == "numeric"
    assert stats["mean"] == pytest.approx(22.25)
    assert stats["min"] == 21.5
    assert stats["max"] == 23.0
    assert stats["count"] == 4


def test_compute_raw_value_stats_numeric_single_value_zero_std():
    facts = [MagicMock()]
    facts[0].json_metadata = '{"signal": "s", "raw_value": 22.0}'
    stats = compute_raw_value_stats(facts, signal_category="numeric")
    assert stats["std"] == 0.0
    assert stats["count"] == 1


# ── compute_raw_value_stats (categorical) ──

def test_compute_raw_value_stats_categorical():
    facts = []
    for val in ["music", "music", "music", "podcast"]:
        f = MagicMock()
        f.json_metadata = f'{{"signal": "media_source", "raw_value": "{val}"}}'
        facts.append(f)

    stats = compute_raw_value_stats(facts, signal_category="categorical")
    assert stats["type"] == "categorical"
    assert stats["dominant_value"] == "music"
    assert stats["value_counts"] == {"music": 3, "podcast": 1}
    assert stats["count"] == 4


def test_compute_raw_value_stats_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown signal_category"):
        compute_raw_value_stats([], signal_category="mystery")
```

- [ ] **Step 14.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_drift_detection.py -v 2>&1 | tail -20`
Expected: `ModuleNotFoundError: No module named 'engine.drift_detection'`

- [ ] **Step 14.3: 写实现**

Create `engine/drift_detection.py`:

```python
"""
drift_detection — 纯函数层

is_drifted / compute_raw_value_stats / cosine_distance 全部无副作用，
embedding_fallback 的网络调用通过依赖注入 embed_fn 隔离，方便单测。

§ 7.3 / § 7.4 of design spec.
"""
import json
import logging
import math
import statistics
from collections import Counter
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

_EMBEDDING_DRIFT_THRESHOLD = 0.15


def is_drifted(
    signal: str,
    old_stats: dict,
    new_stats: dict,
    rule_engine,
    *,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> bool:
    """
    按信号的 drift_metric 判断是否漂移。

    - numeric_mean_diff: |new.mean - old.mean| > threshold
    - dominant_value_change: dominant_value 变化即漂移
    - embedding_fallback: cosine_distance(embed(old_text), embed(new_text)) > 0.15

    没有 rule 或 rule 无 drift_metric → 走 embedding_fallback。
    """
    rule = rule_engine.get_rule(signal) or {}
    metric = rule.get("drift_metric", "embedding_fallback")

    if metric == "numeric_mean_diff":
        threshold = rule["drift_threshold"]
        return abs(new_stats["mean"] - old_stats["mean"]) > threshold

    if metric == "dominant_value_change":
        return new_stats["dominant_value"] != old_stats["dominant_value"]

    if metric == "none":
        # yaml 里显式标 none 的信号跳过 drift 判定（如 nav_destination）
        return False

    if metric == "embedding_fallback":
        if embed_fn is None:
            from panoramix_core.embedder import Embedder
            embed_fn = Embedder().embed
        dist = cosine_distance(
            embed_fn(old_stats["habit_text"]),
            embed_fn(new_stats["habit_text"]),
        )
        return dist > _EMBEDDING_DRIFT_THRESHOLD

    log.warning(f"unknown drift_metric '{metric}' for signal '{signal}'")
    return False


def cosine_distance(a: List[float], b: List[float]) -> float:
    """
    1 - cosine_similarity.
    零向量返回 1.0（约定：无方向视为最大距离）。
    """
    if not a or not b:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def compute_raw_value_stats(pref_facts: list, signal_category: str) -> dict:
    """
    从一组 PREF fact 的 raw_value 聚合统计。

    pref_facts: Fact 对象列表（每个 fact.json_metadata 里有 raw_value 字段）
    signal_category: "numeric" | "categorical"
    返回符合 spec §3.4 的 raw_value_stats 字典。
    """
    raw_values = [
        json.loads(f.json_metadata)["raw_value"] for f in pref_facts
    ]

    if signal_category == "numeric":
        vals = [float(v) for v in raw_values if isinstance(v, (int, float))]
        return {
            "type": "numeric",
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "count": len(vals),
        }

    if signal_category == "categorical":
        counts = Counter(str(v) for v in raw_values)
        return {
            "type": "categorical",
            "dominant_value": counts.most_common(1)[0][0] if counts else "",
            "value_counts": dict(counts),
            "count": sum(counts.values()),
        }

    raise ValueError(f"unknown signal_category: {signal_category}")
```

- [ ] **Step 14.4: 运行测试确认全绿**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_drift_detection.py -v 2>&1 | tail -25`
Expected: 全部 PASS（14 个测试）。

- [ ] **Step 14.5: Commit**

Run:
```bash
git add tests/test_drift_detection.py engine/drift_detection.py
git commit -m "feat(wave3): add is_drifted / cosine_distance / compute_raw_value_stats"
```

---

### Task 15: HabitLifecycleManager.classify_candidate (TDD)

**Files:**
- Create: `tests/test_habit_lifecycle.py`
- Modify: `engine/habit_lifecycle.py` — 追加 `ClassificationResult` dataclasses + `classify_candidate` + `HabitLifecycleManager`

- [ ] **Step 15.1: 写失败测试**

Create `tests/test_habit_lifecycle.py`:

```python
"""HabitLifecycleManager.classify_candidate 集成测试 — spec §7.2"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from engine.habit_lifecycle import (
    ClassificationResult,
    ModifyDrift,
    NewPending,
    Reinforce,
    classify_candidate,
    compute_structural_key,
)
from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard


def _make_habit(
    signal_name: str = "hvac_temp_target",
    mean: float = 22.0,
    time_bucket: str = "early_morning",
    vehicle_state: str = "engine_started",
    geofence: str = "home",
) -> Habit:
    return Habit(
        username="tester",
        batch_id=2,
        text=f"set temperature to {mean}",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="placeholder",
        context_time_bucket=time_bucket,
        context_vehicle_state=vehicle_state,
        context_geofence=geofence,
        context_weekday=1,
        raw_value_stats={
            "type": "numeric", "mean": mean, "std": 0.5,
            "min": mean - 1, "max": mean + 1, "count": 10,
            "habit_text": f"set temperature to {mean}",
        },
        member_fact_ids=[],
    )


def _make_accepted_card(
    structural_key: str,
    frozen_habits: list[dict],
) -> SceneCard:
    return SceneCard(
        username="tester",
        status="accepted",
        structural_key=structural_key,
        display_name="Morning Home Routine",
        frozen_content_snapshot={
            "snapshot_batch_id": 1,
            "snapshot_at": "2026-04-08T10:00:00",
            "habits": frozen_habits,
            "dominant_context": {
                "time_bucket": "early_morning",
                "vehicle_state": "engine_started",
                "geofence": "home",
                "weekday": True,
            },
        },
        first_seen_batch_id=1,
        last_reinforced_batch_id=1,
        created_at=datetime.fromisoformat("2026-04-08T10:00:00"),
        accepted_at=datetime.fromisoformat("2026-04-08T10:05:00"),
    )


@pytest.fixture
def rule_engine_temp():
    m = MagicMock()
    m.get_rule.side_effect = lambda s: {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    } if s == "hvac_temp_target" else None
    return m


# ── NewPending：无匹配 accepted ──

def test_classify_no_accepted_returns_new_pending(rule_engine_temp):
    candidate = [_make_habit(mean=22.0)]
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, NewPending)
    assert result.structural_key == compute_structural_key(candidate)


def test_classify_structural_key_mismatch_returns_new_pending(rule_engine_temp):
    candidate = [_make_habit(geofence="workplace")]  # 不同 geofence
    accepted = _make_accepted_card(
        structural_key="different_key",
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, NewPending)


# ── Reinforce：匹配且无 drift ──

def test_classify_reinforce_when_mean_within_threshold(rule_engine_temp):
    candidate = [_make_habit(mean=22.5)]  # +0.5 < 1.5
    key = compute_structural_key(candidate)
    accepted = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, Reinforce)
    assert result.card_id == accepted.card_id


# ── ModifyDrift：匹配但 drift ──

def test_classify_modify_drift_when_mean_exceeds_threshold(rule_engine_temp):
    candidate = [_make_habit(mean=24.0)]  # +2.0 > 1.5
    key = compute_structural_key(candidate)
    accepted = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, ModifyDrift)
    assert result.card_id == accepted.card_id
    assert "hvac_temp_target" in result.drifted_signals


# ── 多张 accepted 匹配同一 key → NewPending + warning (DCM-3) ──

def test_classify_multiple_accepted_matches_logged_as_new_pending(
    rule_engine_temp, caplog
):
    candidate = [_make_habit(mean=22.0)]
    key = compute_structural_key(candidate)
    a1 = _make_accepted_card(
        structural_key=key, frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "t",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    a2 = _make_accepted_card(
        structural_key=key, frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "t",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    with caplog.at_level("WARNING"):
        result = classify_candidate(
            candidate_habits=candidate,
            accepted_cards=[a1, a2],
            signal_rule_engine=rule_engine_temp,
        )
    assert isinstance(result, NewPending)
    assert "multiple accepted" in caplog.text.lower() or "DCM-3" in caplog.text


# ── 不相关的 accepted 卡被忽略 ──

def test_classify_ignores_unrelated_accepted(rule_engine_temp):
    candidate = [_make_habit(mean=22.0)]
    key = compute_structural_key(candidate)
    matching = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    unrelated = _make_accepted_card(
        structural_key="different",
        frozen_habits=[{
            "signal": "media_source", "habit_text": "music",
            "raw_value_stats": {"type": "categorical", "dominant_value": "music"},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[unrelated, matching],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, Reinforce)
    assert result.card_id == matching.card_id
```

- [ ] **Step 15.2: 运行测试确认全红**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_habit_lifecycle.py -v 2>&1 | tail -20`
Expected: `ImportError: cannot import name 'classify_candidate' from 'engine.habit_lifecycle'`

- [ ] **Step 15.3: 追加 classify_candidate 到 `engine/habit_lifecycle.py`**

Edit `engine/habit_lifecycle.py` — 在文件末尾追加：

```python
# ── Classification results ───────────────────────────────────────


@dataclass(frozen=True)
class Reinforce:
    card_id: str


@dataclass(frozen=True)
class ModifyDrift:
    card_id: str
    drifted_signals: List[str]


@dataclass(frozen=True)
class NewPending:
    structural_key: str


ClassificationResult = object  # type alias for typing hints


# ── Helpers ────────────────────────────────────────────────────────


class MixedSignalClusterError(Exception):
    """cluster contains facts from multiple signal types — violates §3.7 assumption"""


def assert_cluster_pure_signal(cluster_facts: list) -> str:
    """
    §3.7: HabitsDetector 输出的一个 cluster 内所有 PREF facts 应来自同一 signal。
    若违反即抛异常，让假设违反变成显式可观测事件。
    """
    import json as _json
    signals = {
        _json.loads(f.json_metadata)["signal"] for f in cluster_facts
    }
    if len(signals) != 1:
        raise MixedSignalClusterError(
            f"cluster has {len(signals)} signal types: {signals}. "
            f"Current drift detection assumes single-signal habits."
        )
    return next(iter(signals))


# ── classify_candidate ────────────────────────────────────────────


def classify_candidate(
    candidate_habits: List[Habit],
    accepted_cards: List[SceneCard],
    signal_rule_engine,
    *,
    embed_fn=None,
):
    """
    候选场景卡分类 → Reinforce / ModifyDrift / NewPending

    §7.2 of spec.
    """
    from engine.drift_detection import is_drifted

    key = compute_structural_key(candidate_habits)

    # Step A: 严格主键匹配
    matched = [c for c in accepted_cards if c.structural_key == key]

    if not matched:
        return NewPending(structural_key=key)

    if len(matched) > 1:
        log.warning(
            f"structural_key {key} matches multiple accepted cards "
            f"({len(matched)}); treating as NewPending (DCM-3)"
        )
        return NewPending(structural_key=key)

    accepted = matched[0]
    frozen = accepted.frozen_content_snapshot or {}

    # Step B: 按信号判漂移
    drifted = _detect_drift_for_card(
        candidate_habits=candidate_habits,
        frozen_snapshot=frozen,
        rule_engine=signal_rule_engine,
        embed_fn=embed_fn,
    )

    if drifted:
        return ModifyDrift(card_id=accepted.card_id, drifted_signals=drifted)
    return Reinforce(card_id=accepted.card_id)


def _detect_drift_for_card(
    candidate_habits: List[Habit],
    frozen_snapshot: dict,
    rule_engine,
    embed_fn=None,
) -> List[str]:
    """
    对 candidate 和 frozen_snapshot 按 signal_name 求交集后逐信号判断。
    返回漂移的 signal 列表；空列表即 reinforce。
    """
    from engine.drift_detection import is_drifted

    old_by_signal = {
        h["signal"]: h["raw_value_stats"]
        for h in frozen_snapshot.get("habits", [])
    }
    # 给 raw_value_stats 塞一个 habit_text 字段，供 embedding_fallback 用
    for h in frozen_snapshot.get("habits", []):
        stats = old_by_signal[h["signal"]]
        stats.setdefault("habit_text", h.get("habit_text", ""))

    new_by_signal = {}
    for h in candidate_habits:
        stats = dict(h.raw_value_stats)
        stats.setdefault("habit_text", h.text)
        new_by_signal[h.signal_name] = stats

    drifted: List[str] = []
    for signal, new_stats in new_by_signal.items():
        if signal not in old_by_signal:
            # candidate 多出一个旧快照没有的信号
            # → structural_key 本应不同 → 防御性地当作漂移
            drifted.append(signal)
            continue
        if is_drifted(
            signal, old_by_signal[signal], new_stats, rule_engine,
            embed_fn=embed_fn,
        ):
            drifted.append(signal)

    return drifted


# ── HabitLifecycleManager ─────────────────────────────────────────


class HabitLifecycleManager:
    """
    协调器 — 把"候选场景卡分组、GPT naming、classify、store 写入"串起来。

    本类在 Wave 4 pipeline 重写中被 engine/habit_engine.py 调用。
    保持构造函数轻量：store 对象从外面注入。
    """

    def __init__(
        self,
        scene_card_store,
        habit_store,
        signal_rule_engine,
        *,
        embed_fn=None,
    ):
        self.scene_card_store = scene_card_store
        self.habit_store = habit_store
        self.signal_rule_engine = signal_rule_engine
        self.embed_fn = embed_fn

    def classify_all(
        self,
        candidates_by_key: dict,
        accepted_cards: List[SceneCard],
    ) -> list:
        """
        对每组候选 habits（按 structural_key 分组）做分类。
        返回 [(structural_key, candidate_habits, ClassificationResult), ...]
        """
        results = []
        for key, habits in candidates_by_key.items():
            r = classify_candidate(
                candidate_habits=habits,
                accepted_cards=accepted_cards,
                signal_rule_engine=self.signal_rule_engine,
                embed_fn=self.embed_fn,
            )
            results.append((key, habits, r))
        return results
```

- [ ] **Step 15.4: 运行测试确认全绿**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_habit_lifecycle.py -v 2>&1 | tail -20`
Expected: 全部 PASS（6 个测试）。

- [ ] **Step 15.5: Commit**

Run:
```bash
git add tests/test_habit_lifecycle.py engine/habit_lifecycle.py
git commit -m "feat(wave3): add classify_candidate + HabitLifecycleManager coordinator"
```

---

### Task 16: Wave 3 sign-off

- [ ] **Step 16.1: Wave 3 全量测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_structural_key.py tests/test_drift_detection.py tests/test_habit_lifecycle.py -v 2>&1 | tail -15`
Expected: 全部 PASS。

- [ ] **Step 16.2: 历史测试不回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -x --ignore=tests/test_multi_batch_drift.py 2>&1 | tail -15`
Expected: 全部 PASS。

- [ ] **Step 16.3: Wave 3 milestone**

Run:
```bash
git commit --allow-empty -m "milestone(wave3): HabitLifecycleManager complete (classify + drift)"
```

---

## Wave 4 — Pipeline Rewrite (Round 9 Q9.3)

**目标:** 把 `engine/habit_engine.py` 的 `ingest_signal_batch` 改为 §8 里的两阶段结构：阶段 1 在事务外做 Chroma 读 / DBSCAN / GPT reword / GPT scene naming / drift classification；阶段 2 在 SQLite 事务内做 batch_id 分配、habits 落盘、场景卡状态转换、陈腐清理、保留窗口裁剪。

**Wave 验收:** `tests/test_multi_batch_drift.py` 端到端用例全通过；`tests/test_e2e_pipeline.py`（历史）仍绿；事务持锁时间 < 200ms（日志观察）。

---

### Task 17: Pipeline rewrite — `engine/habit_engine.py`

**Files:**
- Modify: `engine/habit_engine.py`
- Read first: `engine/habit_engine.py`（当前实现）+ `panoramix_core/clustering/habits_detector.py`

- [ ] **Step 17.1: 读现有实现做基线理解**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && wc -l engine/habit_engine.py panoramix_core/clustering/habits_detector.py`
记住行数，后面改完对比。

- [ ] **Step 17.2: 重写 `ingest_signal_batch`**

Edit `engine/habit_engine.py` — 将 `ingest_signal_batch` 方法替换为下面两阶段版本，并在类顶部添加必要 import。

新增 import（文件顶部）：

```python
from datetime import datetime
from typing import List

from engine.habit_lifecycle import (
    HabitLifecycleManager,
    ModifyDrift,
    NewPending,
    Reinforce,
    assert_cluster_pure_signal,
    compute_structural_key,
)
from engine.drift_detection import compute_raw_value_stats
from engine.signal_rules.engine import SignalRuleEngine
from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard
from panoramix_core.store.habit_store import HabitStore
from panoramix_core.store.scene_card_store import SceneCardStore
```

在 `HabitDemoEngine.__init__` 中新增 store + lifecycle 构造：

```python
# 在 __init__ 的现有 embedder / fact_store / habits_detector 初始化之后追加：
self.signal_rule_engine = SignalRuleEngine()
self.scene_card_store = SceneCardStore(username=self.username)
self.habit_store = HabitStore(username=self.username)
self.lifecycle = HabitLifecycleManager(
    scene_card_store=self.scene_card_store,
    habit_store=self.habit_store,
    signal_rule_engine=self.signal_rule_engine,
)
```

替换 `ingest_signal_batch`：

```python
def ingest_signal_batch(self, events: List[dict]) -> dict:
    """
    两阶段 batch pipeline (§8)。

    阶段 1 (事务外, ~10-60s)：
      P1.1 peek batch_id (只用于日志)
      P1.2 Chroma 读 PREF TTL 窗口
      P1.3 Hybrid DBSCAN + 连续性过滤 + GPT reword → habits
      P1.4 按 structural_key 分组候选场景卡 + GPT scene naming
      P1.5 classify_candidate （含 embedding_fallback 网络调用）

    阶段 2 (SQLite 事务内, <200ms)：
      P2.1 allocate_next_batch_id
      P2.2 HabitStore.insert_many
      P2.3 按 classification 落盘 scene_cards
      P2.4 delete_stale_pending
      P2.5 delete_stale_recommendation
      P2.6 delete_old_batches(N=3)
      COMMIT
    """
    import logging
    import time
    log = logging.getLogger(__name__)

    # ── Step 0 (peek): 预估 batch_id，仅日志 ──
    peek_batch_id = self.habit_store.get_latest_batch_id() + 1
    log.info(f"[batch-peek] starting ingest with peek_batch_id={peek_batch_id}")

    # ── Step 1: 信号 → PREF Fact → Chroma ──
    pref_facts: List[Fact] = []
    for event in events:
        pref_facts.extend(signals_to_facts(event))
    if pref_facts:
        self.fact_store.store_facts(self.username, pref_facts)
        log.info(f"[phase1.1] stored {len(pref_facts)} PREF facts in Chroma")

    # ── Step 2: 从 Chroma 拉 facts + embeddings，做 Hybrid DBSCAN + 连续性过滤 ──
    items = self.fact_store.get_facts_with_embeddings(self.username)
    consec_filter = make_cluster_filter(self.required_consecutive)
    t0 = time.monotonic()
    cluster_facts_groups = self.habits_detector.cluster_pref_facts(
        items, cluster_filter=consec_filter,
    )  # list[list[Fact]]；每个 inner list 是一个 PREF cluster
    log.info(
        f"[phase1.2] clustering produced {len(cluster_facts_groups)} clusters "
        f"in {time.monotonic() - t0:.2f}s"
    )

    # ── Step 3: 每个 cluster → GPT reword → Habit（§3.7 单 signal 假设）──
    new_habits: List[Habit] = []
    for cluster_facts in cluster_facts_groups:
        signal_name = assert_cluster_pure_signal(cluster_facts)
        rule = self.signal_rule_engine.get_rule(signal_name) or {}
        signal_category = rule.get("category", "categorical")

        stats = compute_raw_value_stats(cluster_facts, signal_category)
        # cluster 已经过连续性过滤，用第一条的 context 做代表
        ctx = cluster_facts[0].context

        text = self.habits_detector.reword_cluster(
            [f.text for f in cluster_facts]
        )
        stats["habit_text"] = text

        habit = Habit(
            username=self.username,
            batch_id=peek_batch_id,  # 阶段 2 会校正
            text=text,
            signal_category=signal_category,
            signal_name=signal_name,
            structural_key="pending",  # 分组后再填
            context_time_bucket=ctx.time_bucket,
            context_vehicle_state=ctx.vehicle_state,
            context_geofence=ctx.geofence,
            context_weekday=int(ctx.weekday) if ctx.weekday is not None else None,
            raw_value_stats=stats,
            member_fact_ids=[f.id for f in cluster_facts],
        )
        new_habits.append(habit)

    # ── Step 4 (前半): 按 structural_key 分组 ──
    candidates_by_key: dict[str, list[Habit]] = {}
    for h in new_habits:
        # 单 habit 先单独计算一个临时 key；
        # 实际分组策略：同 signal + 同 dominant context → 同 key
        k = compute_structural_key([h])
        h.structural_key = k
        candidates_by_key.setdefault(k, []).append(h)

    # ── Step 4 (后半): GPT scene naming per group ──
    display_names: dict[str, str] = {}
    for key, habits in candidates_by_key.items():
        display_names[key] = self._gpt_scene_name(habits)

    # ── Step 5: Classification ──
    accepted_cards = self.scene_card_store.get_by_status("accepted")
    classifications = self.lifecycle.classify_all(
        candidates_by_key=candidates_by_key,
        accepted_cards=accepted_cards,
    )

    # ── 阶段 2: 事务内落盘 ──
    stats_out = {
        "reinforce": 0, "modify_drift": 0, "new_pending": 0,
    }
    with self.scene_card_store.transaction() as conn:
        # P2.1 权威 batch_id
        real_batch_id = self.habit_store.allocate_next_batch_id(conn=conn)
        for h in new_habits:
            h.batch_id = real_batch_id

        # P2.2 写 habits
        self.habit_store.insert_many(new_habits, batch_id=real_batch_id, conn=conn)

        # P2.3 按 classification 写 scene_cards
        for key, habits, result in classifications:
            display_name = display_names[key]
            snapshot = self._build_content_snapshot(
                habits, batch_id=real_batch_id,
            )

            if isinstance(result, Reinforce):
                self.scene_card_store.update_last_reinforced(
                    result.card_id, batch_id=real_batch_id, conn=conn,
                )
                stats_out["reinforce"] += 1

            elif isinstance(result, ModifyDrift):
                snapshot["drifted_signals"] = result.drifted_signals
                self.scene_card_store.upsert_recommendation_by_target(
                    target_accepted_id=result.card_id,
                    card_data={
                        "structural_key": key,
                        "display_name": display_name,
                        "content_snapshot": snapshot,
                    },
                    batch_id=real_batch_id,
                    conn=conn,
                )
                self.scene_card_store.update_last_reinforced(
                    result.card_id, batch_id=real_batch_id, conn=conn,
                )
                stats_out["modify_drift"] += 1

            elif isinstance(result, NewPending):
                self.scene_card_store.upsert_pending_by_structural_key(
                    structural_key=key,
                    card_data={
                        "display_name": display_name,
                        "content_snapshot": snapshot,
                    },
                    batch_id=real_batch_id,
                    conn=conn,
                )
                stats_out["new_pending"] += 1

        # P2.4 清理陈腐 pending
        self.scene_card_store.delete_stale_pending(
            current_batch_id=real_batch_id, conn=conn,
        )
        # P2.5 清理陈腐 recommendation
        self.scene_card_store.delete_stale_recommendation(
            current_batch_id=real_batch_id, conn=conn,
        )
        # P2.6 保留最近 3 批
        self.habit_store.delete_old_batches(
            current_batch_id=real_batch_id, retain_n=3, conn=conn,
        )
    # COMMIT (transaction context exit)

    log.info(
        f"[batch-done] batch_id={real_batch_id} "
        f"habits={len(new_habits)} "
        f"reinforce={stats_out['reinforce']} "
        f"drift={stats_out['modify_drift']} "
        f"new={stats_out['new_pending']}"
    )

    return {
        "batch_id": real_batch_id,
        "habits_count": len(new_habits),
        "classification_stats": stats_out,
    }


def _build_content_snapshot(
    self, habits: List[Habit], batch_id: int,
) -> dict:
    """构造 §3.5 格式的 content_snapshot_json"""
    return {
        "snapshot_batch_id": batch_id,
        "snapshot_at": datetime.now().isoformat(),
        "habits": [
            {
                "habit_text": h.text,
                "signal": h.signal_name,
                "raw_value_stats": h.raw_value_stats,
            }
            for h in habits
        ],
        "dominant_context": {
            "time_bucket": habits[0].context_time_bucket,
            "vehicle_state": habits[0].context_vehicle_state,
            "geofence": habits[0].context_geofence,
            "weekday": bool(habits[0].context_weekday)
                if habits[0].context_weekday is not None else None,
        },
        "habit_ids": [h.habit_id for h in habits],
    }


def _gpt_scene_name(self, habits: List[Habit]) -> str:
    """
    调 GPT 对一组 habits 起名。阶段 1 慢动作内完成。
    GPT 不可用或抛异常 → fallback 到 "<time_bucket> <geofence>"。
    """
    tb = habits[0].context_time_bucket
    geo = habits[0].context_geofence or "routine"
    fallback = f"{tb} {geo}".strip()
    try:
        out = self.habits_detector.reword_scene([h.text for h in habits])
        return out or fallback
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"GPT scene naming failed: {e}; using fallback '{fallback}'"
        )
        return fallback
```

**注意事项:**
- 现有 `engine.signal_to_fact.signals_to_facts(event)` 在 Wave 1 已被改写为 `SignalRuleEngine` 的 shim，所以 import 路径保持不变
- `fact_store.store_facts(username, facts)` / `fact_store.get_facts_with_embeddings(username)` 保持原接口；不要改 fact_store
- 本 task 不删除老的 `accept_habit` / `reject_habit` / `get_habits` API —— 它们被 `simulator/tab2_recommendation.py` 使用；留给后续独立 task 迁移

- [ ] **Step 17.3: 在 `habits_detector.py` 中新增 `cluster_pref_facts` / `reword_cluster` / `reword_scene`**

Edit `panoramix_core/clustering/habits_detector.py` — 在 `HabitsDetector` 类内追加下面三个公共方法（放在 `detect_habits` 下方、`_get_fact_clusters` 之前）：

```python
    # ── 新增：Wave 4 pipeline 使用的薄包装 ──

    def cluster_pref_facts(
        self,
        facts_with_embeddings: List[Dict[str, Any]],
        cluster_filter=None,
    ) -> List[List[Fact]]:
        """
        返回 list[list[Fact]] — 每个 inner list 是一个已通过连续性过滤的 PREF cluster。

        与 detect_habits 的区别：
          - 不调用 GPT reword
          - 不包装成 HABIT Fact
          - 不返回 ids_to_delete（Wave 4 pipeline 不再删 PREF，保留给下次 batch 自然处理）

        Wave 4 之后，pipeline 由 engine/habit_engine.py 接管 reword + 落盘。
        """
        if not facts_with_embeddings:
            return []

        all_facts = [item["fact"] for item in facts_with_embeddings]
        result: List[List[Fact]] = []

        for fact_type in HABIT_CANDIDATE_FACT_TYPES:
            clusters = self._get_fact_clusters(facts_with_embeddings, fact_type)
            for cluster in clusters:
                cluster_facts = [item["fact"] for item in cluster["items"]]
                if cluster_filter is not None and not cluster_filter(
                    cluster_facts, all_facts
                ):
                    logging.info(
                        f"cluster_pref_facts: cluster filtered out "
                        f"({len(cluster_facts)} facts, type={fact_type})"
                    )
                    continue
                result.append(cluster_facts)

        return result

    def reword_cluster(self, fact_texts: List[str]) -> str:
        """
        对一个 cluster 的 PREF 文本调 GPT reword。

        沿用 _get_habit_reword 的 confidence 阈值与拒答处理；
        失败或 LLM=None 时降级到 fact_texts[0]。
        """
        if not fact_texts:
            return ""
        out = self._get_habit_reword(fact_texts)
        return out or fact_texts[0]

    def reword_scene(self, habit_texts: List[str]) -> str:
        """
        对同 structural_key 的 habit 文本起一个短场景名（<= 6 words, Title Case）。

        无 LLM 时返回空字符串（调用方会 fallback 到 "<time_bucket> <geofence>"）。
        """
        if self.llm is None or not habit_texts:
            return ""
        joined = "\n".join(f"- {t}" for t in habit_texts)
        prompt = (
            "Summarize the following car habits as one short English scene name "
            "(at most 6 words, Title Case, no punctuation, no quotes):\n"
            f"{joined}\n\nScene name:"
        )
        try:
            raw = self.llm.invoke(prompt)
        except Exception as e:
            logging.warning(f"reword_scene LLM call failed: {e}")
            return ""
        # 取第一行，裁 60 字
        first_line = (raw or "").strip().splitlines()[0] if raw else ""
        return first_line.strip().strip('"').strip("'")[:60]
```

- [ ] **Step 17.3.1: 验证新方法可导入**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && python -c "from panoramix_core.clustering.habits_detector import HabitsDetector; m = set(dir(HabitsDetector)); assert {'cluster_pref_facts', 'reword_cluster', 'reword_scene'}.issubset(m), 'missing methods: ' + str({'cluster_pref_facts', 'reword_cluster', 'reword_scene'} - m); print('methods ok')"`
Expected: `methods ok`

- [ ] **Step 17.3.2: 确认 `detect_habits` 历史测试不回归**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_dbscan_clustering.py tests/test_e2e_pipeline.py -v 2>&1 | tail -15`
Expected: 全部 PASS（我们只新增方法，未改动 `detect_habits` 本体）。

- [ ] **Step 17.4: 运行单测验证 pipeline import 正常**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && python -c "from engine.habit_engine import HabitDemoEngine; print('import ok')"`
Expected: `import ok`

- [ ] **Step 17.5: Commit**

Run:
```bash
git add engine/habit_engine.py panoramix_core/clustering/habits_detector.py
git commit -m "feat(wave4): rewrite ingest_signal_batch as two-phase pipeline"
```

---

### Task 18: End-to-end multi-batch drift test

**Files:**
- Create: `tests/test_multi_batch_drift.py`

- [ ] **Step 18.1: 写端到端测试**

Create `tests/test_multi_batch_drift.py`:

```python
"""
端到端多批次 drift 测试 — spec §10.2 关键场景

覆盖用户主权不变式：
1. Drift → recommendation 非替换（accepted 卡 frozen 字段不变）
2. Recommendation 去重（连续多批次同 drift 产生的 rec ID 稳定）
3. Accept recommendation 级联 retire 旧 accepted
4. Accepted 卡 dormant 不退役（多批次未匹配仍保留）
5. Pending 陈腐清理（单批次未重现即删除）
"""
import os
import tempfile

import pytest

from engine.habit_engine import HabitDemoEngine
from panoramix_core.store.scene_card_store import SceneCardStore
from panoramix_core.store.sqlite_connection import reset_schema_cache


@pytest.fixture
def isolated_engine(monkeypatch, tmp_path):
    """
    为每个测试起一个全新的 engine，隔离 SQLite 和 Chroma。

    隔离策略：
      - SQLite: monkeypatch DEFAULT_DB_PATH 到临时文件
      - Chroma: 每个测试用一个独立的 username，在 fixture 结束时调 engine.reset()
      - 使用 llm_client=None 让 reword_cluster 走降级路径（返回 fact_texts[0]），
        e2e 不依赖真实 GPT 调用
    """
    import uuid as _uuid
    reset_schema_cache()

    db_path = str(tmp_path / "habit_memory.db")
    monkeypatch.setattr(
        "panoramix_core.store.sqlite_connection.DEFAULT_DB_PATH",
        db_path,
    )

    unique_user = f"e2e_{_uuid.uuid4().hex[:8]}"
    engine = HabitDemoEngine(
        username=unique_user,
        llm_client=None,
        required_consecutive=3,  # 降低门槛方便测试
    )
    try:
        yield engine
    finally:
        try:
            engine.reset()
        except Exception:
            pass
        try:
            engine.close()
        except Exception:
            pass


def _events_morning_home_temp(temp_value: float, day_offset: int = 0):
    """生成一批"早上在家 hvac 温度 = temp_value"的 mockup events。

    返回一个 events list，每个 event 含 signals 数组（符合现有
    signals_to_facts() 的 event schema）。
    """
    from datetime import datetime, timedelta
    base = datetime(2026, 4, 1, 7, 0, 0) + timedelta(days=day_offset)
    events = []
    for i in range(6):
        ts = (base + timedelta(minutes=i * 30)).isoformat()
        events.append({
            "timestamp": ts,
            "user": "e2e",
            "signals": [
                {"signal": "hvac_temp_target", "value": temp_value},
                {"signal": "gps_latitude", "value": 48.8566},
                {"signal": "gps_longitude", "value": 2.3522},
                {"signal": "engine_status", "value": "on"},
                {"signal": "gear_position", "value": "P"},
                {"signal": "vehicle_speed", "value": 0},
            ],
        })
    return events


def _batch(temp_value: float, start_day: int, end_day: int):
    """组装多天的 events 成一个 batch"""
    out = []
    for d in range(start_day, end_day):
        out.extend(_events_morning_home_temp(temp_value, day_offset=d))
    return out


def _media_events(day_offset: int):
    """生成一批"晚上开车听音乐"的 mockup events"""
    from datetime import datetime, timedelta
    base = datetime(2026, 5, 1, 18, 0, 0) + timedelta(days=day_offset)
    events = []
    for i in range(4):
        ts = (base + timedelta(minutes=i * 15)).isoformat()
        events.append({
            "timestamp": ts,
            "user": "e2e",
            "signals": [
                {"signal": "media_source", "value": "music", "content_id": "jazz"},
                {"signal": "gps_latitude", "value": 0.0},
                {"signal": "gps_longitude", "value": 0.0},
                {"signal": "engine_status", "value": "on"},
                {"signal": "gear_position", "value": "D"},
                {"signal": "vehicle_speed", "value": 60},
            ],
        })
    return events


def _media_batch(start_day: int, end_day: int):
    out = []
    for d in range(start_day, end_day):
        out.extend(_media_events(d))
    return out


# ── Scenario 1: Drift → recommendation 非替换 ──

def test_drift_generates_recommendation_keeps_accepted_frozen(isolated_engine):
    engine = isolated_engine

    # Batch 1: 温度 22°C，用户接受产出的 pending 卡
    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))

    pendings = engine.scene_card_store.get_by_status("pending")
    assert len(pendings) >= 1
    morning_card = pendings[0]
    morning_card_id = morning_card.card_id
    frozen_structural_key_before = morning_card.structural_key

    engine.scene_card_store.accept(morning_card_id)

    accepted_before = engine.scene_card_store.get_by_status("accepted")
    assert len(accepted_before) == 1
    frozen_name_before = accepted_before[0].display_name
    frozen_snapshot_before = accepted_before[0].frozen_content_snapshot

    # Batch 2: 温度 25°C (+3 > 1.5 阈值 → drift)
    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))

    # Accepted 卡未变
    accepted_after = engine.scene_card_store.get_by_status("accepted")
    assert len(accepted_after) == 1
    assert accepted_after[0].card_id == morning_card_id
    assert accepted_after[0].display_name == frozen_name_before
    assert accepted_after[0].frozen_content_snapshot == frozen_snapshot_before
    assert accepted_after[0].structural_key == frozen_structural_key_before

    # 新产生了 recommendation 卡
    recs = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs) == 1
    assert morning_card_id in recs[0].supersede_candidate_for


# ── Scenario 2: recommendation 去重 ──

def test_recommendation_deduped_across_batches(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    engine.scene_card_store.accept(pending.card_id)

    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))
    recs1 = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs1) == 1
    rec1_id = recs1[0].card_id
    rec1_reinforced = recs1[0].last_reinforced_batch_id

    engine.ingest_signal_batch(_batch(25.5, start_day=10, end_day=15))
    recs2 = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs2) == 1
    assert recs2[0].card_id == rec1_id
    assert recs2[0].last_reinforced_batch_id > rec1_reinforced


# ── Scenario 3: Accept recommendation 级联 retire ──

def test_accept_recommendation_retires_old_accepted(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    old_accepted_id = pending.card_id
    engine.scene_card_store.accept(old_accepted_id)

    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))
    rec = engine.scene_card_store.get_by_status("recommendation")[0]

    engine.scene_card_store.accept(rec.card_id)

    still_accepted = engine.scene_card_store.get_by_status("accepted")
    assert len(still_accepted) == 1
    assert still_accepted[0].card_id == rec.card_id

    retired = engine.scene_card_store.get_by_status("retired")
    assert any(r.card_id == old_accepted_id for r in retired)


# ── Scenario 4: Accepted 卡 dormant 不退役 ──

def test_dormant_accepted_card_not_retired(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    accepted_id = pending.card_id
    engine.scene_card_store.accept(accepted_id)

    # 后续 5 批完全无关的信号（media_source）
    for d in range(5):
        engine.ingest_signal_batch(_media_events(day_offset=d))

    still = engine.scene_card_store.get_by_status("accepted")
    assert any(c.card_id == accepted_id for c in still)


# ── Scenario 5: Pending 陈腐清理 ──

def test_stale_pending_cleaned_up(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pendings1 = engine.scene_card_store.get_by_status("pending")
    assert len(pendings1) >= 1
    old_temp_ids = {p.card_id for p in pendings1}

    # Batch 2: 完全不同的 signal → 旧 pending 不会被 reinforce
    engine.ingest_signal_batch(_media_batch(start_day=0, end_day=5))

    pendings2 = engine.scene_card_store.get_by_status("pending")
    new_ids = {p.card_id for p in pendings2}
    # 原来的 temperature pending 应已被清理；可能存在新的 media pending
    assert old_temp_ids.isdisjoint(new_ids)
```

- [ ] **Step 18.2: 运行 e2e 测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_multi_batch_drift.py -v 2>&1 | tail -30`

若测试失败：
- 读日志找出失败点（通常是 GPT 调用在 CI 环境或 mock 不到位）
- 在 `conftest.py` 加 `OPENAI_API_KEY` 的 mock 或 skip 标记
- 不要 fix-test，fix pipeline 实现

Expected: 5 个场景全部 PASS。

- [ ] **Step 18.3: Commit**

Run:
```bash
git add tests/test_multi_batch_drift.py
git commit -m "test(wave4): add end-to-end multi-batch drift scenarios"
```

---

### Task 19: Wave 4 sign-off

- [ ] **Step 19.1: Wave 4 全量测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -v 2>&1 | tail -25`
Expected: 所有测试通过（包括历史 test_e2e_pipeline.py，wave1/2/3 的测试，新的 test_multi_batch_drift.py）。

- [ ] **Step 19.2: 事务持锁时间粗估**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/test_multi_batch_drift.py::test_drift_generates_recommendation_keeps_accepted_frozen -v --log-cli-level=INFO 2>&1 | grep -E "phase1|batch-done"`
Expected: 日志里能看到 phase1 耗时与 batch-done 事件；事务内耗时（通过观察 phase1 结束到 batch-done 之间）< 500ms 即可接受（没有硬性测试守门，只是确认方向正确）。

- [ ] **Step 19.3: Wave 4 milestone**

Run:
```bash
git commit --allow-empty -m "milestone(wave4): two-phase pipeline live, e2e drift scenarios green"
```

---

## Wave 5 — Cleanup (Old Code Removal)

**目标:** 新路径稳定后移除老实现：老 `engine/scene_card.py` 的 in-memory dataclass、`FactStoreChroma` 写 HABIT 的死代码、过时测试。这是最后一波，确保 repo 只留一套 truth。

**Wave 验收:** `git grep -n 'SceneCard' engine/scene_card.py` 输出为空或只剩辅助函数；`tests/` 全绿；`pytest tests/test_e2e_pipeline.py tests/test_multi_batch_drift.py -v` 双双通过。

---

### Task 20: Refactor `engine/scene_card.py` — remove in-memory dataclass

**Files:**
- Modify: `engine/scene_card.py`
- Read first: `engine/scene_card.py`（全文）

- [ ] **Step 20.1: 读旧文件**

Read `engine/scene_card.py` 全文，列出：
1. 哪些外部调用者 import 了它（`git grep -n "from engine.scene_card"`）
2. 旧 `SceneCard` dataclass 的字段
3. 旧 `generate_scene_cards` / 等辅助函数的签名

- [ ] **Step 20.2: 删除旧 dataclass，保留候选构造辅助**

Edit `engine/scene_card.py` — 重写为"纯辅助函数"文件（不承载状态）：

```python
"""
engine/scene_card — 候选场景卡构造辅助

Wave 2 之后，SceneCard 的状态与持久化由 panoramix_core/store/scene_card_store.py
负责，本文件只保留"从 habits 列表构造候选 SceneCard Pydantic 对象"的辅助函数，
供 pipeline 的阶段 1 使用。

老的 in-memory dataclass 已删除。
"""
from datetime import datetime
from typing import List

from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard


def build_candidate_pending(
    habits: List[Habit],
    structural_key: str,
    display_name: str,
    batch_id: int,
    username: str,
    content_snapshot: dict,
) -> SceneCard:
    """
    根据一组同 structural_key 的 habits 构造一张 pending SceneCard。
    **不写 DB**；调用方负责把它传给 SceneCardStore.insert_pending。
    """
    return SceneCard(
        username=username,
        status="pending",
        structural_key=structural_key,
        display_name=display_name,
        content_snapshot=content_snapshot,
        first_seen_batch_id=batch_id,
        last_reinforced_batch_id=batch_id,
        created_at=datetime.now(),
    )
```

- [ ] **Step 20.3: 修复 import 链**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && python -c "import engine.scene_card; import engine.habit_engine" 2>&1`

若有 ImportError：根据错误信息修改调用方（通常是把 `from engine.scene_card import SceneCard` 改为 `from panoramix_core.models.scene_card import SceneCard`）。

- [ ] **Step 20.4: 运行全测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -x --ignore=tests/test_scene_card.py 2>&1 | tail -20`
Expected: 全部 PASS（老 `test_scene_card.py` 暂时 ignore，下一 task 处理）。

- [ ] **Step 20.5: Commit**

Run:
```bash
git add engine/scene_card.py
git commit -m "refactor(wave5): remove in-memory SceneCard dataclass, keep builder helper"
```

---

### Task 21: Remove HABIT write path from FactStoreChroma

**Files:**
- Modify: `panoramix_core/store/fact_store_chroma.py`
- Read first: `panoramix_core/store/fact_store_chroma.py`

- [ ] **Step 21.1: 找 HABIT 写入路径**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && grep -n "HABIT" panoramix_core/store/fact_store_chroma.py`

记录：
- `store_fact` 是否对 `fact.type == FactType.HABIT` 特殊处理
- 是否有 `store_habit` / `upsert_habit` 之类的专用方法
- `query_habits` 等读 API（也要删）

- [ ] **Step 21.2: 删除 HABIT 写入路径**

Edit `panoramix_core/store/fact_store_chroma.py` —
1. 在 `store_fact` 开头加守门：若 `fact.type == FactType.HABIT` 则 `raise ValueError("HABIT facts are no longer stored in Chroma; use HabitStore (SQLite) instead")`
2. 删除 `store_habit` / `query_habits` / 任何只处理 HABIT 的方法（如果存在）
3. 不要删公共的 `query_facts` / `store_fact` 接口

（具体删哪几行需要读完现文件再决定）

- [ ] **Step 21.3: grep 调用方**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && git grep -n "store_habit\|query_habits" -- ':!tests/' ':!docs/'`
Expected: 无输出（调用方已在 Wave 4 改走 HabitStore）

若仍有输出：打开那些文件改掉 import 路径和调用点。

- [ ] **Step 21.4: 运行全测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -x --ignore=tests/test_scene_card.py 2>&1 | tail -15`
Expected: 全部 PASS。

- [ ] **Step 21.5: Commit**

Run:
```bash
git add panoramix_core/store/fact_store_chroma.py
git commit -m "refactor(wave5): remove HABIT write path from FactStoreChroma"
```

---

### Task 22: Migrate or delete legacy `tests/test_scene_card.py`

**Files:**
- Read: `tests/test_scene_card.py`
- Decision: migrate 还是 delete

- [ ] **Step 22.1: 读老测试，清点断言**

Read `tests/test_scene_card.py` 全文。对每个测试函数列一行：
- 测的是什么行为？
- 新的 `test_scene_card_store.py` 里是否已覆盖？

- [ ] **Step 22.2: 迁移未覆盖断言**

若有新的 store 测试里没覆盖的有价值断言（例如"候选 SceneCard 的字段计算"），写到 `tests/test_scene_card_store.py` 或新的 `tests/test_scene_card_builder.py`。

- [ ] **Step 22.3: 删除老文件**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && git rm tests/test_scene_card.py`

- [ ] **Step 22.4: 运行全测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -v 2>&1 | tail -25`
Expected: 全部 PASS；`tests/test_scene_card.py` 不再出现在 collected 列表里。

- [ ] **Step 22.5: Commit**

Run:
```bash
git add tests/
git commit -m "test(wave5): migrate scene_card assertions and drop legacy test file"
```

---

### Task 23: Wave 5 sign-off + final milestone

- [ ] **Step 23.1: Final 全量测试**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && pytest tests/ -v 2>&1 | tail -30`
Expected: 全部 PASS。

- [ ] **Step 23.2: 静态审查 — 没有遗留的旧路径**

Run: `cd /home/ae/project/paranomix/habit-memory-demo && git grep -n "FactType.HABIT" -- 'panoramix_core/**' 'engine/**' | grep -v test`
Expected: 只剩读 PREF 时过滤 HABIT（如 `type != HABIT`）的防御性代码，不应出现任何写入路径。

Run: `cd /home/ae/project/paranomix/habit-memory-demo && git grep -n "in-memory SceneCard\|class SceneCard.*dataclass" -- 'engine/**'`
Expected: 无输出。

- [ ] **Step 23.3: Final milestone commit**

Run:
```bash
git commit --allow-empty -m "milestone(wave5): legacy scene_card + chroma habit path removed — Tab1 habit lifecycle complete"
```

- [ ] **Step 23.4: 项目总 milestone**

Run:
```bash
git log --oneline | grep milestone
```
Expected: 能看到五条 `milestone(wave{1..5})` 记录，外加 Task 0 的 baseline commit。

---

## Post-implementation Notes

**实施顺序回顾（对齐 spec §11）：**
- Wave 1 (Tasks 1-5) = Round 2 SignalRuleEngine
- Wave 2 (Tasks 6-12) = Round 6/7/8 SQLite schema + 双 Store
- Wave 3 (Tasks 13-16) = Round 9 HabitLifecycleManager 纯函数层
- Wave 4 (Tasks 17-19) = Round 9 Q9.3 两阶段 pipeline
- Wave 5 (Tasks 20-23) = 老代码清理

**每波之间可以 review + merge。** 不必等全部 23 个 task 做完再上 PR；spec §11 明确建议分波提交。

**未决问题（spec §13）**在实施过程中出现时，不要尝试在本 plan 范围内解决，直接记到 DCM 登记簿或新开 brainstorm。
