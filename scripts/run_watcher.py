"""Entry point for the folder watcher service."""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from pipeline.embedders import BgeEmbedder
from pipeline.ingest import ingest_document
from retriever.qdrant_store import QdrantStore
from watcher.folder_monitor import FolderMonitor


def main():
    logger.info("Starting folder watcher...")

    # Initialize components
    embedder = BgeEmbedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(settings.postgres.url)
    Session = sessionmaker(bind=engine)
    pg = Session()

    def ollama_fn(prompt: str) -> str:
        import ollama
        try:
            client = ollama.Client(host=settings.ollama.base_url)
            resp = client.generate(
                model=settings.ollama.model,
                prompt=prompt,
                options={"temperature": 0.1},
                think=False,
            )
            return resp.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return ""

    def ingest_fn(file_path: Path):
        ingest_document(
            file_path=file_path,
            embedder=embedder,
            qdrant_client=qdrant.client,
            pg_session=pg,
            llm_fn=ollama_fn,
        )

    monitor = FolderMonitor(ingest_fn)
    monitor.start()

    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down watcher...")
        monitor.stop()
        pg.close()
        engine.dispose()
        embedder.unload()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Folder watcher running. Press Ctrl+C to stop.")

    # Keep alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
