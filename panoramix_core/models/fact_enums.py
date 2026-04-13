"""
Fact 枚举类型 — 对齐 Panoramix data/models/fact_enums.py
"""
from enum import Enum


class FactType(str, Enum):
    PERSO = "PERSO"
    PRO = "PRO"
    VAL = "VAL"
    ASSIST = "ASSIST"
    PREF = "PREF"
    HABIT = "HABIT"
    MEDIA = "MEDIA"
    REL = "REL"
    PLACE = "PLACE"
    INTENT = "INTENT"
    SEARCH = "SEARCH"
    MOOD = "MOOD"
    OTHER = "OTHER"


DEFAULT_FACT_TYPE = FactType.OTHER


class FactDurability(str, Enum):
    PERMANENT = "PERMANENT"
    LONG_TERM = "LONG_TERM"
    SHORT_TERM = "SHORT_TERM"


DEFAULT_FACT_DURABILITY = FactDurability.SHORT_TERM


class FactSources(str, Enum):
    EXTRACTION = "extraction"         # Panoramix: dialog → LLM extraction
    SIGNAL = "signal"                 # 新增: 车载信号 → Fact
    AGENT = "agent"
    HABITS_DETECTOR = "habits_detector"
    USER = "user"


DEFAULT_FACT_SOURCE = FactSources.SIGNAL
