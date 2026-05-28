from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from pipeline.loaders import get_loader
from pipeline.splitters import split_document, split_parent_child, TextChunk
from pipeline.embedders import BgeEmbedder


def parse_folder_path(file_path: Path) -> dict[str, str]:
    """Extract department, project, visibility from folder path structure.

    Expected: /knowledge_base/{department}/{project}/file.pdf
    Special folders: 公开 -> visibility=公开
    """
    parts = file_path.relative_to(settings.knowledge_base_root).parts

    department = ""
    project = ""
    visibility = "项目"

    if len(parts) >= 2:
        department = parts[0]
    if len(parts) >= 3:
        project = parts[1]

    # Special visibility rules
    if department == "公开":
        visibility = "公开"
        department = ""
        project = ""
    elif project == "通用":
        visibility = "部门"
        project = ""

    return {
        "department": department,
        "project": project,
        "visibility": visibility,
    }


def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_metadata_with_llm(text: str, llm_fn: Any) -> dict[str, Any]:
    """Use LLM to extract category, tags, summary in one call."""
    truncated = text[:3000]
    prompt = f"""请分析以下文档内容，返回 JSON 格式的元数据：
- category: 文档分类（从以下选择：设计文档、需求文档、测试报告、会议纪要、Bug分析、方案评审、编码规范、流程制度、培训资料、其他）
- tags: 3-5个关键词标签
- summary: 一句话摘要（不超过50字）

只返回 JSON，不要其他文字。

文档内容：
{truncated}"""

    try:
        response = llm_fn(prompt)
        resp_text = response.strip()
        if "```" in resp_text:
            resp_text = resp_text.split("```")[1]
            if resp_text.startswith("json"):
                resp_text = resp_text[4:]
        result = json.loads(resp_text)
        return {
            "category": result.get("category", "其他"),
            "tags": result.get("tags", []),
            "summary": result.get("summary", ""),
        }
    except Exception as e:
        logger.warning(f"LLM metadata extraction failed: {e}")
        return {"category": "其他", "tags": [], "summary": ""}


def ingest_document(
    file_path: str | Path,
    embedder: BgeEmbedder,
    qdrant_client: Any,
    pg_session: Any,
    llm_fn: Any | None = None,
) -> dict[str, Any]:
    """Full document ingestion pipeline.

    Returns: {"doc_id": str, "chunks_count": int, "status": str}
    """
    file_path = Path(file_path)
    doc_id = str(uuid.uuid4())
    file_name = file_path.name
    file_type = file_path.suffix.lstrip(".")

    logger.info(f"Starting ingestion: {file_name} (doc_id={doc_id})")

    # Step 1: Parse folder path
    path_meta = parse_folder_path(file_path)
    logger.info(f"Path metadata: {path_meta}")

    # Step 2: Load document
    loader = get_loader(file_path)
    content = loader.load(file_path)
    if not content.text.strip():
        logger.warning(f"Empty document: {file_name}")
        return {"doc_id": doc_id, "chunks_count": 0, "status": "empty"}

    # Step 3: Save extracted images
    image_refs: list[str] = []
    if content.images:
        img_dir = Path(settings.images_dir) / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        for img_name in content.images:
            image_refs.append(f"images/{doc_id}/{img_name}")

    # Step 4: Split into chunks (parent-child strategy)
    use_parent_child = settings.chunking.strategy == "parent_child"

    if use_parent_child:
        parent_chunks, child_chunks = split_parent_child(content.text, llm_fn=llm_fn)
        logger.info(f"Parent-child split: {len(parent_chunks)} parents, {len(child_chunks)} children")
    else:
        chunks = split_document(content.text, llm_fn=llm_fn)
        logger.info(f"Split into {len(chunks)} chunks")

    # Step 5: Extract metadata with LLM (1 call)
    doc_meta = {"category": "其他", "tags": [], "summary": ""}
    if llm_fn:
        doc_meta = extract_metadata_with_llm(content.text, llm_fn)
        logger.info(f"Metadata: category={doc_meta['category']}, tags={doc_meta['tags']}")

    # Step 6 & 7: Write parent/child records and vectorize
    if use_parent_child:
        total_chunks = _ingest_parent_child(
            pg_session=pg_session,
            qdrant_client=qdrant_client,
            embedder=embedder,
            doc_id=doc_id,
            file_name=file_name,
            file_path=str(file_path),
            file_type=file_type,
            path_meta=path_meta,
            doc_meta=doc_meta,
            image_refs=image_refs,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )
    else:
        total_chunks = _ingest_flat_chunks(
            pg_session=pg_session,
            qdrant_client=qdrant_client,
            embedder=embedder,
            doc_id=doc_id,
            file_name=file_name,
            file_type=file_type,
            path_meta=path_meta,
            doc_meta=doc_meta,
            image_refs=image_refs,
            chunks=chunks,
        )

    # Step 8: Write to PostgreSQL lifecycle record
    _insert_pg_record(pg_session, doc_id, file_name, file_type, path_meta, doc_meta, total_chunks)

    # Step 9: Copy to processed dir
    processed_dir = Path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / f"{doc_id}_{file_name}"
    shutil.copy2(file_path, dest)
    logger.info(f"Copied to processed: {dest}")

    return {"doc_id": doc_id, "chunks_count": total_chunks, "status": "success"}


def _ingest_parent_child(
    pg_session: Any,
    qdrant_client: Any,
    embedder: BgeEmbedder,
    doc_id: str,
    file_name: str,
    file_path: str,
    file_type: str,
    path_meta: dict,
    doc_meta: dict,
    image_refs: list[str],
    parent_chunks: list,
    child_chunks: list,
) -> int:
    """Ingest using parent-child chunking strategy."""
    from sqlalchemy import text
    from qdrant_client.models import PointStruct

    # 1. Write parent documents to PostgreSQL
    for parent in parent_chunks:
        pg_session.execute(text("""
            INSERT INTO parent_documents (id, doc_id, file_name, file_path, full_parent_text, security_level, chunk_index)
            VALUES (:id, :doc_id, :file_name, :file_path, :full_parent_text, :security_level, :chunk_index)
        """), {
            "id": parent.id,
            "doc_id": doc_id,
            "file_name": file_name,
            "file_path": file_path,
            "full_parent_text": parent.text,
            "security_level": path_meta.get("visibility", "public"),
            "chunk_index": str(parent.index),
        })

    # 2. Encode child chunks
    child_texts = [c.text for c in child_chunks]
    vectors = embedder.encode(child_texts)
    logger.info(f"Encoded {len(vectors)} child vectors")

    # 3. Write child chunks to Qdrant + PostgreSQL
    points = []
    for child, vec in zip(child_chunks, vectors):
        qdrant_point_id = str(uuid.uuid4())
        payload = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "department": path_meta["department"],
            "project": path_meta["project"],
            "visibility": path_meta["visibility"],
            "category": doc_meta["category"],
            "tags": doc_meta["tags"],
            "summary": doc_meta["summary"],
            "status": "active",
            "deprecated_reason": "",
            "parent_id": child.parent_id,
            "chunk_index": child.index,
            "total_chunks": len(child_chunks),
            "text": child.text,
            "image_refs": image_refs,
        }
        points.append(PointStruct(
            id=qdrant_point_id,
            vector=vec.tolist(),
            payload=payload,
        ))

        # Write child chunk to PostgreSQL
        pg_session.execute(text("""
            INSERT INTO child_chunks (id, parent_id, doc_id, child_text, qdrant_point_id, chunk_index)
            VALUES (:id, :parent_id, :doc_id, :child_text, :qdrant_point_id, :chunk_index)
        """), {
            "id": child.id,
            "parent_id": child.parent_id,
            "doc_id": doc_id,
            "child_text": child.text,
            "qdrant_point_id": qdrant_point_id,
            "chunk_index": str(child.index),
        })

    # 4. Batch upsert to Qdrant
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        qdrant_client.upsert(
            collection_name=settings.qdrant.collection_name,
            points=batch,
        )
    logger.info(f"Inserted {len(points)} child points into Qdrant")

    pg_session.commit()
    return len(child_chunks)


def _ingest_flat_chunks(
    pg_session: Any,
    qdrant_client: Any,
    embedder: BgeEmbedder,
    doc_id: str,
    file_name: str,
    file_type: str,
    path_meta: dict,
    doc_meta: dict,
    image_refs: list[str],
    chunks: list,
) -> int:
    """Ingest using flat chunking strategy (legacy)."""
    from qdrant_client.models import PointStruct

    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    logger.info(f"Encoded {len(vectors)} vectors")

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "department": path_meta["department"],
            "project": path_meta["project"],
            "visibility": path_meta["visibility"],
            "author": "",
            "category": doc_meta["category"],
            "tags": doc_meta["tags"],
            "summary": doc_meta["summary"],
            "status": "active",
            "deprecated_reason": "",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk.text,
            "image_refs": image_refs,
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload=payload,
        ))

    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        qdrant_client.upsert(
            collection_name=settings.qdrant.collection_name,
            points=batch,
        )
    logger.info(f"Inserted {len(points)} points into Qdrant")
    return len(chunks)


def _insert_pg_record(
    pg_session: Any,
    doc_id: str,
    file_name: str,
    file_type: str,
    path_meta: dict,
    doc_meta: dict,
    chunks_count: int,
):
    """Insert document record into PostgreSQL."""
    from sqlalchemy import text

    pg_session.execute(text("""
        INSERT INTO doc_lifecycle
            (doc_id, file_name, file_type, department, project, visibility,
             author, category, tags, summary, status, total_chunks)
        VALUES
            (:doc_id, :file_name, :file_type, :department, :project, :visibility,
             :author, :category, :tags, :summary, 'active', :total_chunks)
    """), {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_type": file_type,
        "department": path_meta["department"],
        "project": path_meta["project"],
        "visibility": path_meta["visibility"],
        "author": "",
        "category": doc_meta["category"],
        "tags": doc_meta["tags"],
        "summary": doc_meta["summary"],
        "total_chunks": chunks_count,
    })
    pg_session.commit()
