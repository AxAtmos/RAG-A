from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config import settings


# Local model path — no HuggingFace access needed
_LOCAL_RERANKER_DIR = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots"
_CROSS_ENCODER_MODEL_PATH: str | None = None


def _find_local_model() -> str | None:
    """Find the local cached CrossEncoder model snapshot path."""
    global _CROSS_ENCODER_MODEL_PATH
    if _CROSS_ENCODER_MODEL_PATH is not None:
        return _CROSS_ENCODER_MODEL_PATH
    if _LOCAL_RERANKER_DIR.exists():
        snapshots = list(_LOCAL_RERANKER_DIR.iterdir())
        if snapshots:
            _CROSS_ENCODER_MODEL_PATH = str(snapshots[0])
            logger.info(f"Found local reranker model: {_CROSS_ENCODER_MODEL_PATH}")
            return _CROSS_ENCODER_MODEL_PATH
    return None


@dataclass
class RerankResult:
    text: str
    score: float
    metadata: dict[str, Any]


class BgeReranker:
    """Reranker using sentence-transformers CrossEncoder (fully local).

    Loads model from local HuggingFace cache — zero network access.
    Falls back to embedding cosine similarity if CrossEncoder fails.
    """

    def __init__(self):
        self._client = None
        self._cross_encoder = None
        self._cross_encoder_loaded = False
        self._model_name: str = "bge-m3:latest"

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=settings.ollama.base_url)
        return self._client

    def _load_cross_encoder(self):
        if self._cross_encoder_loaded:
            return
        self._cross_encoder_loaded = True

        local_path = _find_local_model()
        if local_path is None:
            logger.warning("No local reranker model found in HF cache, using cosine proxy")
            return

        try:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"  # Block any accidental network call
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(local_path, max_length=512, device="cpu")
            logger.info(f"Cross-Encoder loaded from local path: {local_path}")
        except Exception as e:
            logger.warning(f"Cross-Encoder load failed, using cosine proxy: {e}")
            self._cross_encoder = None

    def cross_encoder_predict(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []
        self._load_cross_encoder()
        if self._cross_encoder is not None:
            try:
                raw_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
                return [float(1.0 / (1.0 + np.exp(-s))) for s in raw_scores]
            except Exception as e:
                logger.error(f"Cross-Encoder predict failed, falling back: {e}")
        return self._cosine_proxy(pairs)

    def _cosine_proxy(self, pairs: list[list[str]]) -> list[float]:
        """Fallback: embedding cosine similarity via Ollama."""
        try:
            client = self._get_client()
            query = pairs[0][0]
            docs = [p[1][:512] for p in pairs]
            query_resp = client.embeddings(model=self._model_name, prompt=query)
            query_vec = np.array(query_resp["embedding"], dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm <= 0:
                return [0.0] * len(pairs)
            scores = []
            for d in docs:
                resp = client.embeddings(model=self._model_name, prompt=d)
                doc_vec = np.array(resp["embedding"], dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm > 0:
                    cos = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
                    scores.append((cos + 1.0) / 2.0)
                else:
                    scores.append(0.0)
            return scores
        except Exception as e:
            logger.critical(f"Cosine proxy failed: {e}")
            return [0.0] * len(pairs)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        top_n = top_n or settings.reranker.top_n
        if not documents:
            return []
        pairs = [[query, doc["text"][:512]] for doc in documents]
        scores = self.cross_encoder_predict(pairs)
        scored = [
            RerankResult(text=doc["text"], score=score, metadata=doc.get("metadata", {}))
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]

    def unload(self):
        self._client = None
        self._cross_encoder = None
        self._cross_encoder_loaded = False
        logger.info("Reranker resources released")
