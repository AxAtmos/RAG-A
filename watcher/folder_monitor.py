"""Folder monitor using watchdog - auto-ingest new documents."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from config import settings


class IngestHandler(FileSystemEventHandler):
    """Handles file system events and triggers ingestion."""

    def __init__(self, ingest_fn: Callable, supported_extensions: set[str], debounce_seconds: int = 10):
        super().__init__()
        self.ingest_fn = ingest_fn
        self.supported_extensions = supported_extensions
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in self.supported_extensions:
            return

        # Skip processed directory
        if "_processed" in str(path):
            return

        # Debounce: wait for file to be fully written
        self._pending[str(path)] = time.time()

    def process_pending(self):
        """Process files that have settled (no changes for debounce period)."""
        now = time.time()
        ready = [
            path for path, ts in self._pending.items()
            if now - ts >= self.debounce_seconds
        ]

        for path in ready:
            del self._pending[path]
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                try:
                    logger.info(f"Auto-ingesting: {p.name}")
                    self.ingest_fn(p)
                except Exception as e:
                    logger.error(f"Auto-ingest failed for {p.name}: {e}")


class FolderMonitor:
    """Watch shared folder for new documents."""

    def __init__(self, ingest_fn: Callable):
        self.ingest_fn = ingest_fn
        self._observer: Observer | None = None
        self._handler: IngestHandler | None = None

    def start(self):
        """Start monitoring the knowledge base folder."""
        root = Path(settings.knowledge_base_root)
        if not root.exists():
            logger.warning(f"Knowledge base folder not found: {root}")
            root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created knowledge base folder: {root}")

        supported = {".pdf", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".epub", ".png", ".jpg", ".jpeg", ".bmp"}

        self._handler = IngestHandler(
            ingest_fn=self.ingest_fn,
            supported_extensions=supported,
            debounce_seconds=settings.watcher.debounce_seconds,
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(root), recursive=True)
        self._observer.start()
        logger.info(f"Folder monitor started: {root}")

        # Start debounce processor in background
        import threading
        self._stop_event = threading.Event()

        def _debounce_loop():
            while not self._stop_event.is_set():
                if self._handler:
                    self._handler.process_pending()
                time.sleep(1)

        self._debounce_thread = threading.Thread(target=_debounce_loop, daemon=True)
        self._debounce_thread.start()

    def stop(self):
        """Stop monitoring."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        logger.info("Folder monitor stopped")
