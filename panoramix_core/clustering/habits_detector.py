"""
HabitsDetector — 对齐 Panoramix data/utils/habits_detector.py + 扩展 Hybrid DBSCAN

与 Panoramix 的差异：
- 使用 hybrid distance: alpha * text_cosine + (1-alpha) * context_distance
- context_distance 基于 StructuredContext (time_bucket, vehicle_state, geofence, weekday)
- 使用 metric='precomputed' 传入预计算距离矩阵
- 原版直接 DBSCAN(metric='cosine') 仅基于文本 embedding

对齐保留：
- 同样按 FactType 分组聚类
- 同样使用 LLM reword 集群为单条习惯
- 同样 CSV 输出格式 + confidence 过滤
"""
import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from panoramix_core.config import (
    HYBRID_ALPHA,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HABIT_DETECTION_LLM_MODEL,
    HABIT_REWORD_CONFIDENCE_THRESHOLD,
    HABIT_CANDIDATE_FACT_TYPES,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources
from engine.cluster_confidence import compute_cluster_confidence

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "habits_detection_prompt.txt")

# ── StructuredContext 维度常量 ──
TIME_BUCKETS = ["early_morning", "morning", "midday", "afternoon", "evening", "night"]
VEHICLE_STATES = ["engine_started", "parked", "crawling", "driving", "idle"]


class HabitsDetector:
    """Hybrid DBSCAN 习惯检测器"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 已初始化的 LLM 客户端，需实现 .invoke(prompt) -> str
                        若为 None，则跳过 LLM reword 步骤 (仅聚类)
        """
        self.llm = llm_client
        try:
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception as e:
            logging.error(f"Error loading prompt file {PROMPT_FILE}: {e}")
            raise
        logging.info("HabitsDetector initialized")

    # ── 公开接口 (对齐 Panoramix detectHabits) ──

    def detect_habits(
        self,
        facts_with_embeddings: List[Dict[str, Any]],
        cluster_filter=None,
    ) -> Tuple[List[Fact], List[str]]:
        """
        检测习惯。

        Args:
            facts_with_embeddings: [{"fact": Fact, "embedding": List[float]}, ...]
            cluster_filter: 可选过滤函数 (cluster_facts, all_facts) -> bool
                           用于连续性验证等后处理。未通过的聚类跳过 LLM reword。

        Returns:
            (new_habit_facts, ids_of_facts_to_delete)
        """
        if not facts_with_embeddings:
            logging.debug("HabitsDetector: No facts provided.")
            return [], []

        all_facts = [item["fact"] for item in facts_with_embeddings]
        new_habit_facts = []
        ids_to_delete = []

        for fact_type in HABIT_CANDIDATE_FACT_TYPES:
            clusters = self._get_fact_clusters(facts_with_embeddings, fact_type)
            logging.info(
                f"HabitsDetector: found {len(clusters)} clusters for type {fact_type}"
            )
            if not clusters:
                continue

            for cluster in clusters:
                cluster_items = cluster["items"]

                # 可选过滤（如连续性验证）
                if cluster_filter is not None:
                    cluster_facts = [item["fact"] for item in cluster_items]
                    if not cluster_filter(cluster_facts, all_facts):
                        logging.info(
                            f"HabitsDetector: cluster filtered out "
                            f"({len(cluster_items)} facts, type={fact_type})"
                        )
                        continue

                fact_texts = [item["fact"].text for item in cluster_items]

                # 从聚类成员提取主导上下文（先算，给 reword 用）
                cluster_ctx = _dominant_context(
                    [item["fact"].context for item in cluster_items]
                )

                habit_text = self._get_habit_reword(fact_texts, context=cluster_ctx)
                if not habit_text:
                    logging.warning(
                        f"HabitsDetector: could not reword cluster: {fact_texts}"
                    )
                    continue

                # 聚类置信度 → json_metadata
                cc = cluster.get("confidence")
                habit_meta = {}
                if cc is not None:
                    habit_meta["clustering_confidence"] = cc.confidence
                    habit_meta["clustering_detail"] = {
                        "cohesion": cc.cohesion,
                        "core_ratio": cc.core_ratio,
                        "size_factor": cc.size_factor,
                        "mean_intra_dist": cc.mean_intra_dist,
                        "max_intra_dist": cc.max_intra_dist,
                        "cluster_size": cc.cluster_size,
                    }

                habit_fact = Fact(
                    id=str(uuid.uuid4()),
                    text=habit_text,
                    type=FactType.HABIT,
                    durability=FactDurability.LONG_TERM,
                    time_stamp=datetime.now(),
                    source=FactSources.HABITS_DETECTOR,
                    context=cluster_ctx,
                    json_metadata=json.dumps(habit_meta) if habit_meta else None,
                )
                new_habit_facts.append(habit_fact)
                conf_str = f" (clustering_confidence={cc.confidence:.3f})" if cc else ""
                logging.info(f"HabitsDetector: created habit: {habit_text}{conf_str}")

                cluster_ids = [item["fact"].id for item in cluster_items]
                ids_to_delete.extend(cluster_ids)

        return new_habit_facts, ids_to_delete

    # ── 新增：Wave 4 pipeline 使用的薄包装 ──

    def cluster_pref_facts(
        self,
        facts_with_embeddings: List[Dict[str, Any]],
        cluster_filter=None,
    ) -> List[Dict[str, Any]]:
        """
        返回 list[dict] — 每个 dict 包含:
          - "facts": list[Fact]  已通过连续性过滤的 PREF cluster
          - "confidence": ClusterConfidence | None  DBSCAN 聚类置信度

        与 detect_habits 的区别：
          - 不调用 GPT reword
          - 不包装成 HABIT Fact
          - 不返回 ids_to_delete

        Wave 4 之后，pipeline 由 engine/habit_engine.py 接管 reword + 落盘。
        """
        if not facts_with_embeddings:
            return []

        all_facts = [item["fact"] for item in facts_with_embeddings]
        result: List[Dict[str, Any]] = []

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
                result.append({
                    "facts": cluster_facts,
                    "confidence": cluster.get("confidence"),
                })

        return result

    def reword_cluster(
        self,
        fact_texts: List[str],
        context: Optional[StructuredContext] = None,
    ) -> str:
        """
        对一个 cluster 的 PREF 文本调 GPT reword，结合该聚类的 StructuredContext
        生成包含场景信息（时间/地点/车辆状态/POI 类型等）的习惯描述。

        沿用 _get_habit_reword 的 confidence 阈值与拒答处理；
        失败或 LLM=None 时降级到 fact_texts[0]。
        """
        if not fact_texts:
            return ""
        out = self._get_habit_reword(fact_texts, context=context)
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

    # ── 聚类 ──

    def _get_fact_clusters(
        self, facts_with_embeddings: List[Dict[str, Any]], fact_type: str
    ) -> List[Dict]:
        """按 FactType 过滤后，用 hybrid DBSCAN 聚类"""
        filtered = [
            item
            for item in facts_with_embeddings
            if item["fact"].type.value == fact_type
            and item["fact"].durability.value in ("LONG_TERM", "SHORT_TERM")
            and item["embedding"] is not None
        ]

        if len(filtered) < 2:
            return []

        # 构建 hybrid 距离矩阵
        dist_matrix = self._compute_hybrid_distance_matrix(filtered)

        clustering = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric="precomputed",
        )
        labels = clustering.fit_predict(dist_matrix)
        core_indices = clustering.core_sample_indices_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"DBSCAN: {n_clusters} clusters for type {fact_type}")

        clusters = []
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            items = [filtered[i] for i in range(len(filtered)) if labels[i] == cid]

            # 计算聚类置信度
            cc = compute_cluster_confidence(
                dist_matrix, labels, core_indices, cid,
                DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
            )
            clusters.append({
                "cluster_id": cid,
                "items": items,
                "confidence": cc,
            })
            texts = [it["fact"].text for it in items]
            logging.info(
                f"Cluster {cid}: {len(items)} facts, {cc.detail}: "
                f"{'; '.join(texts)}"
            )

        return clusters

    def _compute_hybrid_distance_matrix(
        self, items: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        hybrid distance = alpha * text_cosine_dist + (1-alpha) * context_dist

        text_cosine_dist = 1 - cosine_similarity
        context_dist = weighted categorical/numerical distance
        """
        n = len(items)
        embeddings = np.array([item["embedding"] for item in items])

        # ── text cosine distance ──
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized = embeddings / norms
        text_sim = normalized @ normalized.T
        text_dist = 1.0 - text_sim
        text_dist = np.clip(text_dist, 0, 2)
        np.fill_diagonal(text_dist, 0)

        # ── structured context distance ──
        ctx_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = context_distance(
                    items[i]["fact"].context, items[j]["fact"].context
                )
                ctx_dist[i, j] = d
                ctx_dist[j, i] = d

        # ── hybrid ──
        hybrid = HYBRID_ALPHA * text_dist + (1 - HYBRID_ALPHA) * ctx_dist
        np.fill_diagonal(hybrid, 0)
        return hybrid

    # ── LLM Reword (对齐 Panoramix) ──

    def _get_habit_reword(
        self,
        fact_texts: List[str],
        context: Optional[StructuredContext] = None,
    ) -> Optional[str]:
        """将集群中的 fact texts + 场景上下文通过 LLM 合并为单条习惯描述"""
        if self.llm is None:
            logging.warning("No LLM client — returning first fact text as habit")
            return fact_texts[0] if fact_texts else None

        joined = "\n".join(f"- {text}" for text in fact_texts)
        prompt = self.prompt_template.format(
            facts=joined,
            context=_format_context_for_prompt(context),
        )

        try:
            raw = self.llm.invoke(prompt)
            logging.debug(f"Habit reword raw: {raw}")
        except Exception as e:
            logging.error(f"LLM invocation failed: {e}")
            return None

        try:
            reword, confidence_str = raw.rsplit(",", 1)
            confidence = float(confidence_str.strip())
            reword = reword.strip()

            if not reword or reword.lower() in (
                "no_habit", "no habit", "none", "n/a", "no habit detected",
            ):
                logging.warning(f"Habit reword rejected: {reword}")
                return None
            if confidence < HABIT_REWORD_CONFIDENCE_THRESHOLD:
                logging.warning(
                    f"Habit reword low confidence {confidence}: {reword}"
                )
                return None
            return reword
        except Exception as e:
            logging.error(f"Could not parse LLM output: {raw}. Error: {e}")
            return None


# ── Context formatter for prompt ──


def _format_context_for_prompt(ctx: Optional[StructuredContext]) -> str:
    """Render a StructuredContext as a human-readable block for the LLM.

    Skips fields that are None / "unknown" / empty so the LLM does not see
    noise. Returns "(no useful context provided)" when nothing meaningful
    survives the filter — matches the EX-3 example in the prompt.
    """
    if ctx is None:
        return "(no useful context provided)"

    rendered: List[str] = []

    def _add(label: str, value):
        if value is None:
            return
        if isinstance(value, str) and value.strip().lower() in ("", "unknown"):
            return
        rendered.append(f"{label}: {value}")

    _add("time_bucket", getattr(ctx, "time_bucket", None))
    hour = getattr(ctx, "hour", None)
    if isinstance(hour, int) and 0 <= hour <= 23:
        rendered.append(f"hour: {hour}")
    weekday = getattr(ctx, "weekday", None)
    if weekday is True:
        rendered.append("weekday: workday")
    elif weekday is False:
        rendered.append("weekday: weekend")
    _add("vehicle_state", getattr(ctx, "vehicle_state", None))
    _add("geofence", getattr(ctx, "geofence", None))
    _add("poi_type", getattr(ctx, "poi_type", None))
    _add("wiper_state", getattr(ctx, "wiper_state", None))
    _add("temp_bucket", getattr(ctx, "temp_bucket", None))
    _add("door_lock", getattr(ctx, "door_lock", None))
    _add("window_state", getattr(ctx, "window_state", None))
    _add("approach_unlock", getattr(ctx, "approach_unlock", None))

    if not rendered:
        return "(no useful context provided)"
    return "; ".join(rendered)


# ── Context Distance 函数 (模块级，可单独测试) ──

def context_distance(ctx_a: StructuredContext, ctx_b: StructuredContext) -> float:
    """
    计算两个 StructuredContext 之间的归一化距离 [0, 1]

    与 trigger list 2.xlsx 对齐，10 维权重（合计 1.00）：
    - time_bucket:     0.22 (有序距离)
    - vehicle_state:   0.16 (speed/gear/ignition 复合态)
    - geofence:        0.13 (GPS 匹配)
    - wiper_state:     0.10 (核心的天气前置条件)
    - poi_type:        0.09 (site_entrance / home / ...)
    - weekday:         0.07
    - temp_bucket:     0.06
    - window_state:    0.06
    - approach_unlock: 0.06
    - door_lock:       0.05

    未知值 ("unknown" / None / -1) 视为中性 → 贡献 0.5
    """
    dims = [
        (0.22, _time_bucket_distance(ctx_a.time_bucket, ctx_b.time_bucket)),
        (0.16, _categorical_distance(ctx_a.vehicle_state, ctx_b.vehicle_state)),
        (0.13, _categorical_distance(ctx_a.geofence, ctx_b.geofence)),
        (0.10, _wiper_distance(ctx_a.wiper_state, ctx_b.wiper_state)),
        (0.09, _categorical_distance(ctx_a.poi_type, ctx_b.poi_type)),
        (0.07, _weekday_distance(ctx_a.weekday, ctx_b.weekday)),
        (0.06, _temp_distance(ctx_a.temp_bucket, ctx_b.temp_bucket)),
        (0.06, _categorical_distance(ctx_a.window_state, ctx_b.window_state)),
        (0.06, _categorical_distance(ctx_a.approach_unlock, ctx_b.approach_unlock)),
        (0.05, _categorical_distance(ctx_a.door_lock, ctx_b.door_lock)),
    ]

    return sum(w * d for w, d in dims)


# 雨刮：有序距离（off < low < medium < high < max），unknown → 0.5
_WIPER_ORDER = ["off", "low", "medium", "high", "max"]


def _wiper_distance(a: str, b: str) -> float:
    if a in ("unknown", None) or b in ("unknown", None):
        return 0.5
    if a == b:
        return 0.0
    try:
        ia = _WIPER_ORDER.index(a)
        ib = _WIPER_ORDER.index(b)
        return abs(ia - ib) / (len(_WIPER_ORDER) - 1)
    except ValueError:
        return 0.5


# 温度桶：有序距离（cold < mild < warm < hot），unknown → 0.5
_TEMP_ORDER = ["cold", "mild", "warm", "hot"]


def _temp_distance(a: str, b: str) -> float:
    if a in ("unknown", None) or b in ("unknown", None):
        return 0.5
    if a == b:
        return 0.0
    try:
        ia = _TEMP_ORDER.index(a)
        ib = _TEMP_ORDER.index(b)
        return abs(ia - ib) / (len(_TEMP_ORDER) - 1)
    except ValueError:
        return 0.5


def _time_bucket_distance(a: str, b: str) -> float:
    """有序时段距离，unknown → 0.5"""
    if a == "unknown" or b == "unknown":
        return 0.5
    if a == b:
        return 0.0
    try:
        ia = TIME_BUCKETS.index(a)
        ib = TIME_BUCKETS.index(b)
        return abs(ia - ib) / (len(TIME_BUCKETS) - 1)
    except ValueError:
        return 0.5


def _categorical_distance(a: Optional[str], b: Optional[str]) -> float:
    """精确匹配: same=0, different=1, unknown=0.5"""
    if a is None or a == "unknown" or b is None or b == "unknown":
        return 0.5
    return 0.0 if a == b else 1.0


def _weekday_distance(a: Optional[bool], b: Optional[bool]) -> float:
    """weekday 匹配: same=0, different=1, unknown=0.5"""
    if a is None or b is None:
        return 0.5
    return 0.0 if a == b else 1.0


def _dominant_context(contexts: List[StructuredContext]) -> StructuredContext:
    """从聚类成员的上下文列表中提取主导上下文（各维度众数）。

    每一维都独立取众数；Optional 字段忽略 None，其余忽略 "unknown"。
    """
    from collections import Counter

    n = len(contexts)
    if n == 0:
        return StructuredContext()

    def _mode_nonempty(values, empty_marker):
        filtered = [v for v in values if v is not None and v != empty_marker]
        if not filtered:
            return empty_marker if empty_marker is not None else None
        return Counter(filtered).most_common(1)[0][0]

    tb = _mode_nonempty([c.time_bucket for c in contexts], "unknown")
    hr_values = [c.hour for c in contexts if c.hour >= 0]
    hr = round(sum(hr_values) / len(hr_values)) if hr_values else -1
    vs = _mode_nonempty([c.vehicle_state for c in contexts], "unknown")

    weekdays = [c.weekday for c in contexts if c.weekday is not None]
    wd = Counter(weekdays).most_common(1)[0][0] if weekdays else None

    geos = [c.geofence for c in contexts if c.geofence is not None]
    geo = Counter(geos).most_common(1)[0][0] if geos else None

    pois = [c.poi_type for c in contexts if c.poi_type is not None]
    poi = Counter(pois).most_common(1)[0][0] if pois else None

    wp = _mode_nonempty([c.wiper_state for c in contexts], "unknown")
    tp = _mode_nonempty([c.temp_bucket for c in contexts], "unknown")
    win = _mode_nonempty([c.window_state for c in contexts], "unknown")

    dls = [c.door_lock for c in contexts if c.door_lock is not None]
    dl = Counter(dls).most_common(1)[0][0] if dls else None

    aus = [c.approach_unlock for c in contexts if c.approach_unlock is not None]
    au = Counter(aus).most_common(1)[0][0] if aus else None

    return StructuredContext(
        time_bucket=tb, hour=hr, weekday=wd,
        vehicle_state=vs, geofence=geo, poi_type=poi,
        wiper_state=wp, temp_bucket=tp, window_state=win,
        door_lock=dl, approach_unlock=au,
    )
