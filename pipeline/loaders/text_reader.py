from __future__ import annotations

from pathlib import Path

import chardet
from loguru import logger

from .loader_registry import LoaderRegistry, DocumentContent


def _read_with_encoding(file_path: Path) -> str:
    raw = file_path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


@LoaderRegistry.register([".txt"])
class TextLoader:
    def load(self, file_path: Path) -> DocumentContent:
        text = _read_with_encoding(file_path)
        logger.info(f"TXT loaded: {file_path.name}, {len(text)} chars")
        return DocumentContent(text=text)


@LoaderRegistry.register([".md", ".markdown"])
class MarkdownLoader:
    def load(self, file_path: Path) -> DocumentContent:
        text = _read_with_encoding(file_path)
        logger.info(f"Markdown loaded: {file_path.name}, {len(text)} chars")
        return DocumentContent(text=text)
