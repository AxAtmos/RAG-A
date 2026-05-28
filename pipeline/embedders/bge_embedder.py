from __future__ import annotations

import json

import numpy as np
from loguru import logger

from config import settings


class BgeEmbedder:
    """BGE-M3 embedding wrapper using Ollama."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=settings.ollama.base_url)
        return self._client

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode texts to vectors. Returns (N, dim) numpy array."""
        client = self._get_client()
        vectors = []
        for text in texts:
            resp = client.embeddings(model="bge-m3:latest", prompt=text)
            vectors.append(resp["embedding"])
        return np.array(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        client = self._get_client()
        resp = client.embeddings(model="bge-m3:latest", prompt=query)
        return np.array(resp["embedding"])

    def unload(self):
        """Free resources."""
        self._client = None
        logger.info("Embedding client released")
