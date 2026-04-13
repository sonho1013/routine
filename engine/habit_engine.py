"""
HabitDemoEngine — 端到端管线编排

完整流程:
  批量信号 → Signal-to-Fact(裸动作+ctx) → ChromaDB存储(ada-002 Embedding + ctx_*元数据)
  → Hybrid DBSCAN聚类 → LLM习惯合成+场景命名 → Habit Fact

使用:
    engine = HabitDemoEngine(username="Mary")
    result = engine.ingest_from_simulator(simulator)
    status = engine.get_status()
    engine.close()
"""
import json
import logging
from typing import Dict, List, Optional

from engine.signal_to_fact import signals_to_facts
from engine.consecutiveness import make_cluster_filter
from engine.scene_card import generate_scene_cards, SceneCard
from engine.proactive_executor import ProactiveExecutor, RecommendationResult
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType
from panoramix_core.store.fact_store_chroma import FactStoreChroma
from panoramix_core.clustering.habits_detector import HabitsDetector

log = logging.getLogger(__name__)


def _get_habit_meta(habit: Fact) -> dict:
    """从 habit 的 json_metadata 提取关键字段"""
    if habit.json_metadata:
        try:
            meta = json.loads(habit.json_metadata)
            return {
                "scene_name": meta.get("scene_name"),
                "scene_confidence": meta.get("scene_confidence"),
                "clustering_confidence": meta.get("clustering_confidence"),
            }
        except (json.JSONDecodeError, TypeError):
            pass
    return {"scene_name": None, "scene_confidence": None, "clustering_confidence": None}


class HabitDemoEngine:
    """演示系统主引擎，编排完整 学习/推理 流程"""

    def __init__(self, username: str = "demo_driver", llm_client=None,
                 required_consecutive: int = 5):
        """
        Args:
            username: 用户标识，对应 ChromaDB 中的独立 collection
            llm_client: 实现 .invoke(prompt) -> str 的 LLM 客户端
                        None 时 HabitsDetector 跳过 LLM reword（取首条 fact text）
            required_consecutive: 连续性验证要求的最近连续次数（PRD 默认 5）
        """
        self.username = username
        self.required_consecutive = required_consecutive
        self.llm_client = llm_client
        self.fact_store = FactStoreChroma(username)
        self.habits_detector = HabitsDetector(llm_client=llm_client)
        self.executor = ProactiveExecutor(self.fact_store)
        log.info(
            f"HabitDemoEngine initialized: user={username}, "
            f"llm={'yes' if llm_client else 'no'}, "
            f"required_consecutive={required_consecutive}"
        )

    # ═══════════════════════════════════════════════════
    # 学习阶段
    # ═══════════════════════════════════════════════════

    def ingest_signal_batch(self, events: List[Dict]) -> Dict:
        """
        批量处理信号事件（来自外部埋点系统的全量数据）。

        流程:
          1. 批量信号 → Fact对象 (裸动作 + StructuredContext)
          2. 存入 ChromaDB（自动生成 text embedding + ctx_* 元数据）
          3. 获取全量 facts + embeddings
          4. 运行 Hybrid DBSCAN 习惯检测
          5. 生命周期管理：删除已聚类原始 facts，存入新 habit facts

        Args:
            events: 信号事件列表，每个 event 含 signals 数组

        Returns:
            dict: facts_ingested, habits_detected, facts_clustered, ...
        """
        # Step 1: 信号 → Fact
        facts: List[Fact] = []
        for event in events:
            facts.extend(signals_to_facts(event))

        if not facts:
            log.warning("ingest_signal_batch: no facts generated from events")
            return self._make_result(0, 0, 0, 0, [])

        log.info(f"Step 1: {len(events)} events → {len(facts)} facts")

        # Step 2: 存入 ChromaDB
        self.fact_store.store_facts(self.username, facts)
        log.info(f"Step 2: {len(facts)} facts stored in ChromaDB")

        # Step 3: 获取全量 facts + embeddings
        items = self.fact_store.get_facts_with_embeddings(self.username)
        log.info(f"Step 3: {len(items)} facts with embeddings retrieved")

        # Step 4: Hybrid DBSCAN + 连续性过滤
        consec_filter = make_cluster_filter(self.required_consecutive)
        new_habits, ids_to_delete = self.habits_detector.detect_habits(
            items, cluster_filter=consec_filter
        )
        log.info(
            f"Step 4: Hybrid DBSCAN + consecutiveness(≥{self.required_consecutive}) "
            f"→ {len(new_habits)} habits, {len(ids_to_delete)} facts clustered"
        )

        # Step 5: 场景卡命名 (LLM 为习惯组生成场景名)
        scene_cards = []
        if new_habits:
            scene_cards = generate_scene_cards(
                new_habits, llm_client=self.llm_client
            )
            log.info(
                f"Step 5: {len(new_habits)} habits → "
                f"{len(scene_cards)} scene cards"
            )

        # Step 6: 生命周期管理
        if new_habits and ids_to_delete:
            self.fact_store.delete_facts(self.username, ids_to_delete)
            self.fact_store.store_facts(self.username, new_habits)
            log.info(
                f"Step 6: deleted {len(ids_to_delete)} clustered facts, "
                f"stored {len(new_habits)} habit facts"
            )

        return self._make_result(
            facts_ingested=len(facts),
            total_before_clustering=len(items),
            habits_detected=len(new_habits),
            facts_clustered=len(ids_to_delete),
            new_habits=new_habits,
            scene_cards=scene_cards,
        )

    def ingest_from_simulator(self, simulator) -> Dict:
        """
        从 SignalSimulator 批量处理全部事件。

        Args:
            simulator: SignalSimulator 实例 (已加载 mockup 数据)
        """
        log.info(
            f"Ingesting from simulator: user={simulator.user}, "
            f"{simulator.total_events} events"
        )
        return self.ingest_signal_batch(simulator._events)

    # ═══════════════════════════════════════════════════
    # 查询
    # ═══════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """获取当前存储状态摘要"""
        all_facts = self.fact_store.get_facts(self.username)
        habits = [f for f in all_facts if f.type == FactType.HABIT]
        prefs = [f for f in all_facts if f.type == FactType.PREF]
        return {
            "username": self.username,
            "total_facts": len(all_facts),
            "pref_facts": len(prefs),
            "habit_facts": len(habits),
            "habits": [
                {
                    "id": h.id,
                    "text": h.text,
                    "accepted": h.accepted,
                    **_get_habit_meta(h),
                    "context": {
                        "time_bucket": h.context.time_bucket,
                        "vehicle_state": h.context.vehicle_state,
                        "geofence": h.context.geofence,
                        "weekday": h.context.weekday,
                    },
                }
                for h in habits
            ],
        }

    def get_all_facts(self) -> List[Fact]:
        """获取全部 facts（含 PREF + HABIT）"""
        return self.fact_store.get_facts(self.username)

    def get_habits(self) -> List[Fact]:
        """仅获取 HABIT 类型 facts"""
        return self.fact_store.get_facts(
            self.username, types=[FactType.HABIT]
        )

    def get_recommendation(
        self, current_context: StructuredContext, top_k: int = 0,
    ) -> RecommendationResult:
        """
        推理阶段：给定当前上下文，匹配已接受的习惯。

        Args:
            current_context: 当前车辆上下文 (time_bucket, vehicle_state, ...)
            top_k: 最多返回 top_k 个推荐（0 = 不限制）

        Returns:
            RecommendationResult: 推荐动作列表
        """
        return self.executor.recommend(current_context, self.username, top_k=top_k)

    def accept_habit(self, habit_id: str) -> bool:
        """接受一个习惯（标记 accepted=True 并重新存储）"""
        habits = self.get_habits()
        for h in habits:
            if h.id == habit_id:
                h.accepted = True
                self.fact_store.delete_facts(self.username, [habit_id])
                self.fact_store.store_facts(self.username, [h])
                log.info(f"Habit accepted: {h.text}")
                return True
        log.warning(f"Habit not found: {habit_id}")
        return False

    def reject_habit(self, habit_id: str) -> bool:
        """拒绝一个习惯（从存储中删除）"""
        habits = self.get_habits()
        for h in habits:
            if h.id == habit_id:
                self.fact_store.delete_facts(self.username, [habit_id])
                log.info(f"Habit rejected and deleted: {h.text}")
                return True
        log.warning(f"Habit not found: {habit_id}")
        return False

    # ═══════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════

    def reset(self):
        """清空当前用户的全部数据（重新开始学习）"""
        all_facts = self.fact_store.get_facts(self.username)
        if all_facts:
            ids = [f.id for f in all_facts]
            self.fact_store.delete_facts(self.username, ids)
            log.info(f"Reset: deleted {len(ids)} facts for user {self.username}")

    def close(self):
        """关闭 ChromaDB 连接"""
        self.fact_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    # ── 内部 ──

    @staticmethod
    def _make_result(facts_ingested, total_before_clustering,
                     habits_detected, facts_clustered, new_habits,
                     scene_cards=None):
        return {
            "facts_ingested": facts_ingested,
            "total_before_clustering": total_before_clustering,
            "habits_detected": habits_detected,
            "facts_clustered": facts_clustered,
            "facts_remaining": total_before_clustering - facts_clustered + habits_detected,
            "new_habits": [
                {"text": h.text, "id": h.id, **_get_habit_meta(h)}
                for h in new_habits
            ],
            "scene_cards": [
                c.summary for c in (scene_cards or [])
            ],
        }
