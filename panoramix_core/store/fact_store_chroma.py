"""
ChromaDB FactStore 实现 — 对齐 Panoramix data/store/fact_store_chroma.py

与 Panoramix 的差异：
- metadata 中新增 ctx_* 字段存储结构化上下文
- embedding function 使用 panoramix_core.embedder.Embedder
- 新增 get_facts_with_embeddings() 返回 embedding 用于 hybrid DBSCAN
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings

import panoramix_core.config as _config
from panoramix_core.embedder import Embedder
from panoramix_core.models.fact import Fact
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources
from panoramix_core.store.fact_store import FactStore


class FactStoreChroma(FactStore):
    """Per-user ChromaDB Fact 存储"""

    def __init__(self, username: str):
        self.username = username
        self.embedder = Embedder()
        self.storage_path = os.path.join(_config.MEMORY_DIR, username)
        os.makedirs(self.storage_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.storage_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.embedding_function = self._create_embedding_function()
        logging.info(f"FactStoreChroma initialized for '{username}' at {self.storage_path}")

    def close(self) -> None:
        if hasattr(self, "client") and self.client is not None:
            del self.client
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    # ── 写入 ──

    def store_facts(self, username: str, facts: List[Fact]) -> None:
        ids, documents, metadatas = [], [], []
        for fact in facts:
            ids.append(fact.id)
            documents.append(fact.text)
            metadatas.append(fact.to_chroma_metadata())

        collection = self._get_collection(username)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # ── 读取 ──

    def get_facts(
        self,
        username: str,
        types: Optional[List[FactType]] = None,
        source: Optional[FactSources] = None,
        durability: Optional[FactDurability] = None,
    ) -> List[Fact]:
        collection = self._get_collection(username)
        where = self._build_where(types, source, durability)
        if where:
            records = collection.get(where=where, limit=_config.CHROMA_MAX_RECORDS)
        else:
            records = collection.get(limit=_config.CHROMA_MAX_RECORDS)

        return [
            Fact.from_chroma_result(fid, doc, meta)
            for fid, doc, meta in zip(
                records["ids"], records["documents"], records["metadatas"]
            )
        ]

    def get_facts_with_embeddings(self, username: str) -> List[Dict[str, Any]]:
        """返回习惯候选 Facts + embeddings，用于 hybrid DBSCAN"""
        collection = self._get_collection(username)
        count = collection.count()
        if count == 0:
            return []

        records = collection.get(
            where={"type": {"$in": _config.HABIT_CANDIDATE_FACT_TYPES}},
            limit=_config.CHROMA_MAX_RECORDS,
            include=["documents", "metadatas", "embeddings"],
        )

        results = []
        for fid, doc, meta, emb in zip(
            records["ids"],
            records["documents"],
            records["metadatas"],
            records["embeddings"],
        ):
            fact = Fact.from_chroma_result(fid, doc, meta)
            results.append({"fact": fact, "embedding": emb})
        return results

    # ── 删除 ──

    def delete_facts(self, username: str, fact_ids: List[str]) -> None:
        collection = self._get_collection(username)
        collection.delete(ids=fact_ids)

    # ── 内部方法 ──

    def _get_collection(self, username: str):
        return self.client.get_or_create_collection(
            name=f"facts_user_{username}",
            embedding_function=self.embedding_function,
            metadata={
                "description": f"Habit memory facts for {username}",
                "created_at": time.time(),
            },
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 100,
                    "ef_search": 100,
                }
            },
        )

    def _create_embedding_function(self):
        embedder = self.embedder

        class _EmbFn:
            def name(self):
                return "openai_ada002"

            def __call__(self, input):
                # ChromaDB passes a list of texts here. Send them all in a
                # single OpenAI call — going one-at-a-time made 83 facts take
                # ~13 minutes through a SOCKS proxy (each round-trip ~9s),
                # which looked like the UI had hung.
                if isinstance(input, str):
                    input = [input]
                if not input:
                    return []
                return embedder.embed_batch(list(input)).tolist()

        return _EmbFn()

    @staticmethod
    def _build_where(
        types: Optional[List[FactType]] = None,
        source: Optional[FactSources] = None,
        durability: Optional[FactDurability] = None,
    ) -> Optional[Dict]:
        conditions = []
        if types:
            conditions.append({"type": {"$in": [t.value for t in types]}})
        if source:
            conditions.append({"source": source.value})
        if durability:
            conditions.append({"durability": durability.value})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
