"""
Fact 数据模型 — 对齐 Panoramix data/models/fact.py + 扩展 structured context

与 Panoramix 的差异：
- 新增 context 字段：存储结构化上下文 (time_bucket, vehicle_state, geofence, weekday)
- 移除 alternative_texts 翻译（MS1 不需要）
- source 默认值改为 SIGNAL（信号源而非对话提取）
"""
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

from pydantic import BaseModel, Field, model_validator

from panoramix_core.models.fact_enums import (
    FactType, FactDurability, FactSources, DEFAULT_FACT_TYPE
)

PERMANENT_TTL_DAYS = None
LONG_TERM_TTL_DAYS = 365
SHORT_TERM_TTL_DAYS = 30


class StructuredContext(BaseModel):
    """结构化上下文维度 — 参与 hybrid DBSCAN 距离计算"""
    time_bucket: str = "unknown"        # early_morning / midday / afternoon / evening / night
    hour: int = -1                      # 0-23
    weekday: Optional[bool] = None      # True=工作日, False=周末
    vehicle_state: str = "unknown"      # engine_started / parked / crawling
    geofence: Optional[str] = None      # home / workplace / toll_A6 / ...


class Fact(BaseModel):
    """
    对齐 Panoramix Fact 模型。

    ChromaDB 存储结构：
    - document: fact.text (纯动作描述，无场景标签)
    - metadata: type, durability, timestamp, source, context_*
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    type: FactType = Field(default=DEFAULT_FACT_TYPE)
    durability: FactDurability = Field(default=FactDurability.LONG_TERM)
    time_stamp: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    source: Optional[FactSources] = FactSources.SIGNAL
    accepted: bool = False
    json_metadata: Optional[str] = None

    # 扩展: 结构化上下文 (Panoramix 原版无此字段)
    context: StructuredContext = Field(default_factory=StructuredContext)

    @model_validator(mode='after')
    def post_validation(self):
        if self.expires_at is None:
            ttl = self._get_ttl_days(self.durability)
            if ttl is not None:
                self.expires_at = self.time_stamp + timedelta(days=ttl)
        if self.json_metadata is not None:
            stripped = self.json_metadata.strip() if isinstance(self.json_metadata, str) else ""
            if not stripped:
                self.json_metadata = None
            else:
                json.loads(stripped)  # validate JSON
        return self

    @classmethod
    def get_default_durability(cls, fact_type: Union[str, FactType]) -> FactDurability:
        if isinstance(fact_type, str):
            fact_type = FactType(fact_type.strip().upper())
        defaults = {
            FactType.PERSO: FactDurability.PERMANENT,
            FactType.PRO: FactDurability.LONG_TERM,
            FactType.VAL: FactDurability.PERMANENT,
            FactType.ASSIST: FactDurability.PERMANENT,
            FactType.PREF: FactDurability.LONG_TERM,
            FactType.HABIT: FactDurability.LONG_TERM,
            FactType.MEDIA: FactDurability.LONG_TERM,
            FactType.REL: FactDurability.LONG_TERM,
            FactType.PLACE: FactDurability.LONG_TERM,
            FactType.INTENT: FactDurability.SHORT_TERM,
            FactType.SEARCH: FactDurability.SHORT_TERM,
            FactType.MOOD: FactDurability.SHORT_TERM,
            FactType.OTHER: FactDurability.SHORT_TERM,
        }
        return defaults.get(fact_type, FactDurability.SHORT_TERM)

    @staticmethod
    def _get_ttl_days(durability: Union[str, FactDurability]) -> Optional[int]:
        val = durability.value if isinstance(durability, FactDurability) else durability
        return {
            "PERMANENT": PERMANENT_TTL_DAYS,
            "LONG_TERM": LONG_TERM_TTL_DAYS,
            "SHORT_TERM": SHORT_TERM_TTL_DAYS,
        }.get(val, SHORT_TERM_TTL_DAYS)

    def is_expired(self, reference_time: datetime = None) -> bool:
        if self.expires_at is None:
            return False
        return (reference_time or datetime.now()) > self.expires_at

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """序列化为 ChromaDB metadata (str/int/float/bool only)"""
        meta = {
            "type": self.type.value,
            "durability": self.durability.value,
            "timestamp": self.time_stamp.timestamp(),
            "accepted": self.accepted,
            "expires_at": self.expires_at.timestamp() if self.expires_at else datetime(2100, 1, 1).timestamp(),
            # structured context → flat keys
            "ctx_time_bucket": self.context.time_bucket,
            "ctx_hour": self.context.hour,
            "ctx_vehicle_state": self.context.vehicle_state,
        }
        if self.context.weekday is not None:
            meta["ctx_weekday"] = self.context.weekday
        if self.context.geofence is not None:
            meta["ctx_geofence"] = self.context.geofence
        if self.source:
            meta["source"] = self.source.value
        if self.json_metadata:
            meta["json_metadata"] = self.json_metadata
        return meta

    @classmethod
    def from_chroma_result(cls, id: str, document: str, metadata: Dict[str, Any]) -> "Fact":
        """从 ChromaDB 查询结果重建 Fact"""
        try:
            ts = datetime.fromtimestamp(float(metadata.get("timestamp", 0)))
        except (ValueError, TypeError, OSError):
            ts = datetime.now()

        expires_at = None
        ea_val = metadata.get("expires_at")
        if ea_val is not None:
            try:
                ea_dt = datetime.fromtimestamp(float(ea_val))
                if ea_dt.year < 2100:
                    expires_at = ea_dt
            except (ValueError, TypeError, OSError):
                pass

        ctx = StructuredContext(
            time_bucket=metadata.get("ctx_time_bucket", "unknown"),
            hour=metadata.get("ctx_hour", -1),
            weekday=metadata.get("ctx_weekday"),
            vehicle_state=metadata.get("ctx_vehicle_state", "unknown"),
            geofence=metadata.get("ctx_geofence"),
        )

        return cls(
            id=id,
            text=document,
            type=metadata.get("type", "OTHER"),
            durability=metadata.get("durability", "SHORT_TERM"),
            time_stamp=ts,
            expires_at=expires_at,
            source=metadata.get("source"),
            accepted=metadata.get("accepted", False),
            json_metadata=metadata.get("json_metadata"),
            context=ctx,
        )
