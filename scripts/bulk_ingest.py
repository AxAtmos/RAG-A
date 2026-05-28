"""Bulk ingestion script - scan existing documents and ingest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings
from pipeline.embedders import BgeEmbedder
from pipeline.ingest import ingest_document
from retriever.qdrant_store import QdrantStore


def bulk_ingest(root_dir: str | None = None):
    """Scan a directory and ingest all supported documents."""
    root = Path(root_dir or settings.knowledge_base_root)
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        return

    # Setup
    embedder = BgeEmbedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()

    engine = create_engine(settings.postgres.url)
    Session = sessionmaker(bind=engine)
    pg = Session()

    def ollama_fn(prompt: str) -> str:
        import ollama
        try:
            client = ollama.Client(host=settings.ollama.base_url)
            resp = client.generate(model=settings.ollama.model, prompt=prompt, options={"temperature": 0.1})
            return resp.get("response", "")
        except Exception:
            return ""

    # Scan files
    supported = {".pdf", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".png", ".jpg", ".jpeg"}
    files = [f for f in root.rglob("*") if f.suffix.lower() in supported and "_processed" not in str(f)]

    logger.info(f"Found {len(files)} documents to ingest")

    success = 0
    failed = 0
    for i, file_path in enumerate(files, 1):
        try:
            result = ingest_document(
                file_path=file_path,
                embedder=embedder,
                qdrant_client=qdrant.client,
                pg_session=pg,
                llm_fn=ollama_fn,
            )
            success += 1
            logger.info(f"[{i}/{len(files)}] OK: {file_path.name} -> {result['chunks_count']} chunks")
        except Exception as e:
            failed += 1
            logger.error(f"[{i}/{len(files)}] FAILED: {file_path.name} -> {e}")

    logger.info(f"Bulk ingest complete: {success} success, {failed} failed")

    pg.close()
    engine.dispose()
    embedder.unload()


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    bulk_ingest(directory)
