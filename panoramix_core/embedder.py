"""
Embedding 客户端 — 对齐 Panoramix core/azure_embedder.py

适配: 使用 OpenAI SDK 统一接口 (支持 OpenAI / Azure / 兼容 API)
配置通过环境变量: OPENAI_API_KEY, OPENAI_API_BASE, EMBEDDING_MODEL
"""
import os
import logging
from typing import List

import numpy as np
from openai import OpenAI

from panoramix_core.config import EMBEDDING_MODEL


def _fix_socks_proxy():
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
                "all_proxy", "https_proxy", "http_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks://"):
            os.environ[var] = val.replace("socks://", "socks5://", 1)


class Embedder:
    """OpenAI-compatible embedding 客户端 (单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        _fix_socks_proxy()
        self.client = OpenAI()
        self.model = EMBEDDING_MODEL
        self._initialized = True
        logging.info(f"Embedder initialized: model={self.model}")

    def embed(self, text: str) -> List[float]:
        """单条文本 → embedding 向量"""
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量文本 → embedding 矩阵 (N × dim)"""
        response = self.client.embeddings.create(input=texts, model=self.model)
        return np.array([item.embedding for item in response.data])
