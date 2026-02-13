"""Configuration from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# OpenAI-compatible API
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE: str = os.environ.get("OPENAI_API_BASE", "https://api.polza.ai/api/v1")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "google/gemini-3-flash-preview")
OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-large")

# Knowledge base (default: kb folder in project root)
_default_kb = Path(__file__).resolve().parent.parent.parent / "kb"
KNOWLEDGE_BASE_PATH: Path = Path(os.environ.get("KNOWLEDGE_BASE_PATH", str(_default_kb)))
INDEX_PATH: Path = Path(os.environ.get("INDEX_PATH", "data/index"))

# RAG
TOP_K: int = int(os.environ.get("TOP_K", "5"))
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "300"))
# Минимальный score релевантности (cosine similarity); чанки ниже отфильтровываются. 0 = не фильтровать.
MIN_RELEVANCE_SCORE: float = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.45"))

# Rate limit (requests per minute per chat)
RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10"))

# Hybrid search (BM25 + vector + RRF)
HYBRID_SEARCH_ENABLED: bool = os.environ.get("HYBRID_SEARCH_ENABLED", "false").lower() in ("true", "1", "yes")
HYBRID_FETCH_K: int = int(os.environ.get("HYBRID_FETCH_K", "30"))  # candidates per stream before RRF

# Reranker (cross-encoder)
RERANKER_ENABLED: bool = os.environ.get("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")
RERANKER_TOP_N: int = int(os.environ.get("RERANKER_TOP_N", "20"))  # candidates to pass to reranker
RERANKER_MODEL: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_API_URL: str = os.environ.get("RERANK_API_URL", "")  # optional: use API instead of local model
RERANK_API_KEY: str = os.environ.get("RERANK_API_KEY", "")

# Query expansion (multi-query)
QUERY_EXPANSION_ENABLED: bool = os.environ.get("QUERY_EXPANSION_ENABLED", "false").lower() in ("true", "1", "yes")
QUERY_EXPANSION_VARIANTS: int = int(os.environ.get("QUERY_EXPANSION_VARIANTS", "3"))  # total variants (incl. original)

# Optional: override system prompt for LLM (empty = use built-in universal prompt)
RAG_SYSTEM_PROMPT: str = os.environ.get("RAG_SYSTEM_PROMPT", "")
