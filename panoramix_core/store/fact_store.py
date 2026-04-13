"""
FactStore 抽象基类 — 对齐 Panoramix data/store/fact_store.py
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from panoramix_core.models.fact import Fact
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


class FactStore(ABC):
    """Fact 存储层抽象接口"""

    @abstractmethod
    def store_facts(self, username: str, facts: List[Fact]) -> None:
        ...

    @abstractmethod
    def get_facts(
        self,
        username: str,
        types: Optional[List[FactType]] = None,
        source: Optional[FactSources] = None,
        durability: Optional[FactDurability] = None,
    ) -> List[Fact]:
        ...

    @abstractmethod
    def delete_facts(self, username: str, fact_ids: List[str]) -> None:
        ...

    @abstractmethod
    def get_facts_with_embeddings(self, username: str) -> List[dict]:
        """返回 [{fact: Fact, embedding: List[float]}, ...]"""
        ...

    @abstractmethod
    def close(self) -> None:
        ...
