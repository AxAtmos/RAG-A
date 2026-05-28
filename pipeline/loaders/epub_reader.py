from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from .loader_registry import LoaderRegistry, DocumentContent


@LoaderRegistry.register([".epub"])
class EpubLoader:
    """EPUB format loader using ebooklib."""

    def load(self, file_path: Path) -> DocumentContent:
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(str(file_path), options={"ignore_ncx": True})

        parts: list[str] = []

        # Extract title
        title = book.get_metadata("DC", "title")
        if title:
            parts.append(f"# {title[0][0]}")

        # Extract content from each item
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_content = item.get_content().decode("utf-8", errors="replace")
            text = self._html_to_text(html_content)
            if text.strip():
                parts.append(text)

        full_text = "\n\n".join(parts)
        logger.info(f"EPUB loaded: {file_path.name}, {len(full_text)} chars")
        return DocumentContent(text=full_text)

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Strip HTML tags and decode entities."""
        # Remove script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Replace br/p/div/h* with newlines
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", html, flags=re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", "", html)
        # Decode common entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'")
        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
