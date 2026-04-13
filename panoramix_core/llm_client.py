"""
LLM 客户端 — 对齐 Panoramix core/llmclient.py 简化版

接口: .invoke(prompt) -> str
配置: OPENAI_API_KEY (env), HABIT_DETECTION_LLM_MODEL (config.py)
"""
import os
import logging
from openai import OpenAI

from panoramix_core.config import HABIT_DETECTION_LLM_MODEL


def _fix_socks_proxy():
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
                "all_proxy", "https_proxy", "http_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks://"):
            os.environ[var] = val.replace("socks://", "socks5://", 1)

log = logging.getLogger(__name__)


class LLMClient:
    """OpenAI Chat Completion 客户端，供 HabitsDetector 使用"""

    def __init__(self, model: str = None, temperature: float = 0.2):
        _fix_socks_proxy()
        self.client = OpenAI()
        self.model = model or HABIT_DETECTION_LLM_MODEL
        self.temperature = temperature
        log.info(f"LLMClient initialized: model={self.model}")

    def invoke(self, prompt: str) -> str:
        """发送 prompt，返回纯文本响应"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=256,
        )
        result = response.choices[0].message.content.strip()
        log.debug(f"LLM response: {result}")
        return result
