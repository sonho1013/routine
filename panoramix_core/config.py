"""
集中配置管理 — 对齐 Panoramix .env 模式
"""
import os
from pathlib import Path as _Path
_DEFAULT_MODEL_MAP = _Path(__file__).resolve().parent.parent / "config" / "openrouter_model_map.yaml"


# ── Embedding ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIM = 1536

# ── DBSCAN 聚类 ──
# Hybrid distance = HYBRID_ALPHA * text_cosine + (1-HYBRID_ALPHA) * ctx_dist
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.8"))
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.10"))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))

# ── 习惯合成 ──
HABIT_DETECTION_LLM_MODEL = os.getenv("HABIT_DETECTION_LLM_MODEL", "gpt-4.1")
HABIT_REWORD_CONFIDENCE_THRESHOLD = float(os.getenv("HABIT_CONFIDENCE_THRESHOLD", "0.7"))

# ── ChromaDB 存储 ──
MEMORY_DIR = os.getenv("MEMORY_DIR", "./storage/memories")
CHROMA_MAX_RECORDS = int(os.getenv("CHROMA_MAX_RECORDS", "1000"))

# ── 主动执行 ──
CONTEXT_MATCH_THRESHOLD = float(os.getenv("CONTEXT_MATCH_THRESHOLD", "0.35"))

# ── 习惯候选 Fact 类型 (对齐 Panoramix) ──
HABIT_CANDIDATE_FACT_TYPES = [
    "PREF", "HABIT", "MEDIA", "PLACE", "SEARCH",
    "INTENT", "REL", "ASSIST", "PRO",
]

# ── Backup chain providers ──
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
TUNNEL_BASE_URL = os.getenv("TUNNEL_BASE_URL", "")
TUNNEL_OPENAI_KEY = os.getenv("TUNNEL_OPENAI_KEY", "")

# ── Timeouts (seconds) ──
OPENAI_TIMEOUT_CONNECT = float(os.getenv("OPENAI_TIMEOUT_CONNECT", "5"))
OPENAI_TIMEOUT_READ = float(os.getenv("OPENAI_TIMEOUT_READ", "15"))
OPENROUTER_TIMEOUT_CONNECT = float(os.getenv("OPENROUTER_TIMEOUT_CONNECT", "5"))
OPENROUTER_TIMEOUT_READ = float(os.getenv("OPENROUTER_TIMEOUT_READ", "15"))
TUNNEL_TIMEOUT_CONNECT = float(os.getenv("TUNNEL_TIMEOUT_CONNECT", "5"))
TUNNEL_TIMEOUT_READ = float(os.getenv("TUNNEL_TIMEOUT_READ", "10"))

# ── Cache ──
LLM_CACHE_PATH = os.getenv("LLM_CACHE_PATH", "./storage/llm_cache.json")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "./storage/embedding_cache.json")
LLM_CACHE_READ_ONLY = os.getenv("LLM_CACHE_READ_ONLY", "").strip() == "1"

# ── OpenRouter model translation ──
OPENROUTER_MODEL_MAP_PATH = os.getenv("OPENROUTER_MODEL_MAP_PATH", str(_DEFAULT_MODEL_MAP))
