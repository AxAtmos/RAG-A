import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "settings.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout: int = 120
    num_ctx: int = 16384
    token_buffer_size: int = 20


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 8192
    dimension: int = 1024


class RerankerConfig(BaseModel):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    top_n: int = 5
    fallback_embedding_model: str = "bge-m3:latest"
    rerank_top_n: int = 5


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge"
    vector_dimension: int = 1024
    distance: str = "Cosine"


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_enterprise"
    user: str = "rag"
    password: str = "rag_secret_2026"

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ChunkingConfig(BaseModel):
    strategy: str = "parent_child"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    parent_chunk_size: int = 2000
    child_chunk_size: int = 300
    parent_child_overlap: int = 45


class RetrievalConfig(BaseModel):
    top_k: int = 20
    rerank_top_n: int = 5
    score_threshold: float = 0.3
    elastic_min_k: int = 2
    elastic_max_k: int = 12
    ce_threshold: float = 0.65
    retry_top_k: int = 50
    query_fusion_count: int = 4
    query_rewrite_enabled: bool = True
    query_llm_model: str = ""  # empty = use ollama.model


class StreamConfig(BaseModel):
    enabled: bool = True
    insufficient_tag: str = "【INSUFFICIENT_INFO】"
    vram_total_gb: float = 24.0
    model_weights_gb: float = 5.0
    max_elastic_token_window_gb: float = 4.8
    vram_warning_threshold: float = 0.85


class WatcherConfig(BaseModel):
    poll_interval: int = 5
    debounce_seconds: int = 10


class Settings(BaseSettings):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)

    # Paths
    knowledge_base_root: str = "/knowledge_base"
    processed_dir: str = "/knowledge_base/_processed"
    images_dir: str = "/data/images"

    model_config = {"env_prefix": "RAG_", "env_nested_delimiter": "__"}


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    raw = _load_yaml(path)

    # Map flat YAML keys to Settings fields
    kwargs: dict[str, Any] = {}
    if "ollama" in raw:
        kwargs["ollama"] = OllamaConfig(**raw["ollama"])
    if "embedding" in raw:
        kwargs["embedding"] = EmbeddingConfig(**raw["embedding"])
    if "reranker" in raw:
        kwargs["reranker"] = RerankerConfig(**raw["reranker"])
    if "qdrant" in raw:
        kwargs["qdrant"] = QdrantConfig(**raw["qdrant"])
    if "postgres" in raw:
        kwargs["postgres"] = PostgresConfig(**raw["postgres"])
    if "chunking" in raw:
        kwargs["chunking"] = ChunkingConfig(**raw["chunking"])
    if "retrieval" in raw:
        kwargs["retrieval"] = RetrievalConfig(**raw["retrieval"])
    if "stream" in raw:
        kwargs["stream"] = StreamConfig(**raw["stream"])
    if "watcher" in raw:
        kwargs["watcher"] = WatcherConfig(**raw["watcher"])
    if "knowledge_base" in raw:
        kb = raw["knowledge_base"]
        kwargs["knowledge_base_root"] = kb.get("root", "/knowledge_base")
        kwargs["processed_dir"] = kb.get("processed_dir", "/knowledge_base/_processed")
        kwargs["images_dir"] = kb.get("images_dir", "/data/images")

    return Settings(**kwargs)


settings = load_settings(os.environ.get("RAG_CONFIG"))
