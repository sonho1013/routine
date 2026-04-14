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
from datetime import datetime
from typing import Dict, List, Optional

from engine.signal_to_fact import signals_to_facts
from engine.consecutiveness import make_cluster_filter
from engine.proactive_executor import ProactiveExecutor, RecommendationResult
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
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType
from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard as SceneCardModel
from panoramix_core.store.fact_store_chroma import FactStoreChroma
from panoramix_core.store.habit_store import HabitStore
from panoramix_core.store.scene_card_store import SceneCardStore
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

        # Wave 4: store + lifecycle
        self.signal_rule_engine = SignalRuleEngine()
        self.scene_card_store = SceneCardStore(username=self.username)
        self.habit_store = HabitStore(username=self.username)
        self.lifecycle = HabitLifecycleManager(
            scene_card_store=self.scene_card_store,
            habit_store=self.habit_store,
            signal_rule_engine=self.signal_rule_engine,
        )

        log.info(
            f"HabitDemoEngine initialized: user={username}, "
            f"llm={'yes' if llm_client else 'no'}, "
            f"required_consecutive={required_consecutive}"
        )

    # ═══════════════════════════════════════════════════
    # 学习阶段
    # ═══════════════════════════════════════════════════

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
            try:
                signal_name = assert_cluster_pure_signal(cluster_facts)
            except Exception as e:
                log.warning(f"Skipping mixed-signal cluster: {e}")
                continue
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
        candidates_by_key: dict = {}
        for h in new_habits:
            # 单 habit 先单独计算一个临时 key；
            # 实际分组策略：同 signal + 同 dominant context → 同 key
            k = compute_structural_key([h])
            h.structural_key = k
            candidates_by_key.setdefault(k, []).append(h)

        # ── Step 4 (后半): GPT scene naming per group ──
        display_names: dict = {}
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

        # ── 阶段 2 后: 删除已聚类的 PREF facts（保持 Chroma 滚动窗口干净）──
        clustered_ids = [f.id for group in cluster_facts_groups for f in group]
        if clustered_ids:
            self.fact_store.delete_facts(self.username, clustered_ids)
            log.info(f"[post-commit] deleted {len(clustered_ids)} clustered PREF facts from Chroma")

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
            log.warning(
                f"GPT scene naming failed: {e}; using fallback '{fallback}'"
            )
            return fallback

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
