"""
集中配置管理 — 对齐 Panoramix .env 模式
"""
import os


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
