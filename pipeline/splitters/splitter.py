from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from config import settings


@dataclass
class TextChunk:
    text: str
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    id: str
    text: str
    index: int = 0


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    text: str
    index: int = 0


def split_document(
    text: str,
    strategy: str | None = None,
    llm_fn: Any | None = None,
) -> list[TextChunk]:
    """Split document text into chunks using the configured strategy.

    For parent_child strategy, use split_parent_child() instead.
    """
    strategy = strategy or settings.chunking.strategy

    if strategy == "parent_child":
        logger.warning("split_document called with parent_child strategy; use split_parent_child() for full result")
        parents, children = split_parent_child(text, llm_fn=llm_fn)
        return [TextChunk(text=c.text, index=i) for i, c in enumerate(children)]

    if strategy == "semantic" and llm_fn is not None:
        try:
            return _semantic_split(text, llm_fn)
        except Exception as e:
            logger.warning(f"Semantic split failed, falling back to recursive: {e}")
            return _recursive_split(text)

    if strategy == "recursive":
        return _recursive_split(text)

    return _fixed_split(text)


def split_parent_child(
    text: str,
    llm_fn: Any | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Split document into parent-child hierarchy.

    Returns:
        (parents, children) where children have parent_id references.
    """
    parent_size = settings.chunking.parent_chunk_size
    child_size = settings.chunking.child_chunk_size
    overlap = settings.chunking.parent_child_overlap

    # Step 1: Split into parent chunks
    parent_texts = _split_by_char_boundary(text, parent_size, overlap)
    logger.info(f"Parent-child split: {len(parent_texts)} parent blocks")

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for pi, ptext in enumerate(parent_texts):
        parent = ParentChunk(id=str(uuid.uuid4()), text=ptext.strip(), index=pi)
        parents.append(parent)

        # Step 2: Split each parent into child chunks
        child_texts = _split_by_char_boundary(ptext, child_size, overlap)
        for ci, ctext in enumerate(child_texts):
            if not ctext.strip():
                continue
            child = ChildChunk(
                id=str(uuid.uuid4()),
                parent_id=parent.id,
                text=ctext.strip(),
                index=ci,
            )
            children.append(child)

    logger.info(f"Parent-child split: {len(parents)} parents, {len(children)} children")
    return parents, children


def _split_by_char_boundary(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text by character count with sentence-boundary awareness."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a sentence boundary near the end
        best_break = end
        for delim in ["。", "！", "？", ".", "!", "?", "\n"]:
            pos = text.rfind(delim, start + chunk_size // 2, end + 50)
            if pos > start:
                best_break = min(best_break, pos + 1)

        chunks.append(text[start:best_break])
        start = best_break - overlap
        if start <= 0:
            start = best_break

    return chunks


def _semantic_split(text: str, llm_fn: Any) -> list[TextChunk]:
    """LLM-assisted semantic splitting: identify chapter boundaries."""
    prompt = """请分析以下文档的章节结构，输出 JSON 格式的目录。
每个章节包含 title（章节标题）和 start_hint（章节开头的前20个字作为定位提示）。
只输出 JSON 数组，不要其他文字。

文档内容：
{text}"""

    # Truncate for LLM context
    truncated = text[:8000]
    response = llm_fn(prompt.format(text=truncated))

    try:
        # Try to extract JSON from response
        resp_text = response.strip()
        if "```" in resp_text:
            resp_text = resp_text.split("```")[1]
            if resp_text.startswith("json"):
                resp_text = resp_text[4:]
        chapters = json.loads(resp_text)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Failed to parse LLM chapter structure, falling back to recursive")
        return _recursive_split(text)

    if not isinstance(chapters, list) or len(chapters) == 0:
        return _recursive_split(text)

    # Split by chapter hints
    chunks: list[TextChunk] = []
    remaining = text

    for ch in chapters:
        title = ch.get("title", "")
        hint = ch.get("start_hint", "")

        if hint and hint in remaining:
            idx = remaining.index(hint)
            if idx > 0 and chunks:
                # Add text before this chapter to previous chunk
                chunks[-1].text += remaining[:idx]
            remaining = remaining[idx:]

        chunk_text = title + "\n" + remaining[:settings.chunking.max_chunk_size * 4]
        chunks.append(TextChunk(text=chunk_text.strip()))

    # Add remaining text to last chunk
    if remaining and chunks:
        chunks[-1].text += remaining

    # Post-process: split oversized chunks, merge tiny ones
    chunks = _post_process(chunks)

    for i, c in enumerate(chunks):
        c.index = i

    logger.info(f"Semantic split: {len(chunks)} chunks")
    return chunks


def _recursive_split(text: str) -> list[TextChunk]:
    """Recursive splitting: paragraph -> sentence -> character."""
    chunk_size = settings.chunking.chunk_size * 4  # chars approx
    overlap = settings.chunking.chunk_overlap * 4

    paragraphs = text.split("\n\n")
    chunks: list[TextChunk] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(TextChunk(text=current))
            # If paragraph itself is too long, split by sentences
            if len(para) > chunk_size:
                sub_chunks = _split_by_sentences(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current:
        chunks.append(TextChunk(text=current))

    # Merge tiny chunks
    chunks = _post_process(chunks)

    for i, c in enumerate(chunks):
        c.index = i

    logger.info(f"Recursive split: {len(chunks)} chunks")
    return chunks


def _fixed_split(text: str) -> list[TextChunk]:
    """Fixed-size splitting with overlap (fallback)."""
    chunk_size = settings.chunking.chunk_size * 4
    overlap = settings.chunking.chunk_overlap * 4

    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(TextChunk(text=chunk_text.strip()))
        start = end - overlap

    for i, c in enumerate(chunks):
        c.index = i

    logger.info(f"Fixed split: {len(chunks)} chunks")
    return chunks


def _split_by_sentences(text: str, max_size: int, overlap: int) -> list[TextChunk]:
    """Split text by sentence boundaries."""
    # Chinese + English sentence delimiters
    sentences: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in "。！？.!?\n" and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks: list[TextChunk] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_size:
            current += sent
        else:
            if current:
                chunks.append(TextChunk(text=current))
            current = sent
    if current:
        chunks.append(TextChunk(text=current))
    return chunks


def _post_process(chunks: list[TextChunk]) -> list[TextChunk]:
    """Split oversized chunks, merge tiny ones."""
    min_size = settings.chunking.min_chunk_size * 4
    max_size = settings.chunking.max_chunk_size * 4

    result: list[TextChunk] = []
    for chunk in chunks:
        if len(chunk.text) > max_size:
            # Split further
            sub = _split_by_sentences(chunk.text, max_size, settings.chunking.chunk_overlap * 4)
            result.extend(sub)
        else:
            result.append(chunk)

    # Merge tiny chunks
    merged: list[TextChunk] = []
    for chunk in result:
        if merged and len(merged[-1].text) + len(chunk.text) < min_size:
            merged[-1].text += "\n\n" + chunk.text
        else:
            merged.append(chunk)

    return merged
