from __future__ import annotations

from pathlib import Path
from typing import Protocol

from loguru import logger


class DocumentContent:
    """Parsed result from a document loader."""

    def __init__(
        self,
        text: str,
        images: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.text = text
        self.images = images or []
        self.metadata = metadata or {}


class BaseLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, file_path: Path) -> DocumentContent: ...


class LoaderRegistry:
    _loaders: dict[str, BaseLoader] = {}

    @classmethod
    def register(cls, extensions: list[str]):
        def decorator(loader_cls):
            instance = loader_cls()
            for ext in extensions:
                cls._loaders[ext.lower()] = instance
            return loader_cls
        return decorator

    @classmethod
    def get(cls, ext: str) -> BaseLoader | None:
        return cls._loaders.get(ext.lower())


def get_loader(file_path: Path) -> BaseLoader:
    ext = file_path.suffix.lower()
    loader = LoaderRegistry.get(ext)
    if loader is None:
        logger.warning(f"No loader registered for {ext}, falling back to text loader")
        from .text_reader import TextLoader
        return TextLoader()
    return loader
