"""End-to-end flow test: Embed → Store → Retrieve → Rerank → LLM Generate.

Skips Dify, tests the core RAG pipeline on the server.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from config import settings
from pipeline.embedders import BgeEmbedder
from pipeline.rerankers import BgeReranker

# ── Test documents ──
TEST_DOCS = [
    {
        "text": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了信息检索和文本生成的技术框架。"
                "它的核心思想是：在大语言模型生成回答之前，先从外部知识库中检索相关文档，将检索到的内容作为上下文传给模型，"
                "从而让模型的回答更加准确、有据可依，减少幻觉问题。",
        "file_name": "rag_intro.txt",
    },
    {
        "text": "RAG 系统通常包含三个核心组件：1. Retriever（检索器）：负责从向量数据库中检索与用户查询最相关的文档片段。"
                "2. Reranker（重排序器）：对检索器返回的候选文档进行精排。"
                "3. Generator（生成器）：将重排后的最相关文档作为上下文，连同用户问题一起输入给大语言模型，生成最终的回答。",
        "file_name": "rag_components.txt",
    },
    {
        "text": "Qwen3 是阿里巴巴通义千问团队推出的新一代大语言模型系列。Qwen3 系列包括多种规模的模型，"
                "从小于 1B 参数的轻量模型到 235B 参数的大规模模型。Qwen3 支持混合推理模式（思考模式和非思考模式），"
                "在数学、编程、多语言等任务上表现出色。",
        "file_name": "qwen3_intro.txt",
    },
    {
        "text": "向量数据库负责存储文档的向量表示并提供高效的相似度搜索，常见的有 FAISS、Milvus、Qdrant、Chroma 等。"
                "在 RAG 系统中，文档需要先经过 Embedding 模型转换为高维向量，常用的 Embedding 模型包括 BGE 系列、E5 系列等。",
        "file_name": "vector_db.txt",
    },
]

TEST_QUESTIONS = [
    "RAG系统的核心组件有哪些？",
    "Qwen3模型有什么特点？",
    "向量数据库有哪些选择？",
]


def test_embedding(embedder: BgeEmbedder):
    """Step 1: Test embedding generation."""
    logger.info("=" * 50)
    logger.info("[Step 1] Testing Embedding (Ollama bge-m3)")
    logger.info("=" * 50)

    test_text = "RAG是什么技术？"
    vec = embedder.encode_query(test_text)
    logger.info(f"  Query: {test_text}")
    logger.info(f"  Vector dimension: {vec.shape}")
    logger.info(f"  Vector norm: {np.linalg.norm(vec):.4f}")
    assert vec.shape == (settings.qdrant.vector_dimension,), \
        f"Expected dim {settings.qdrant.vector_dimension}, got {vec.shape}"
    logger.info("  OK - Embedding works")
    return True


def test_ingest_to_qdrant(embedder: BgeEmbedder, qdrant: QdrantClient):
    """Step 2: Ingest test documents into Qdrant."""
    logger.info("=" * 50)
    logger.info("[Step 2] Ingesting test docs into Qdrant")
    logger.info("=" * 50)

    collections = [c.name for c in qdrant.get_collections().collections]
    if settings.qdrant.collection_name not in collections:
        qdrant.create_collection(
            collection_name=settings.qdrant.collection_name,
            vectors_config=VectorParams(
                size=settings.qdrant.vector_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"  Created collection: {settings.qdrant.collection_name}")

    texts = [d["text"] for d in TEST_DOCS]
    vectors = embedder.encode(texts)
    logger.info(f"  Encoded {len(vectors)} vectors, dim={vectors.shape[1]}")

    points = []
    for doc, vec in zip(TEST_DOCS, vectors):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=vec.tolist(),
            payload={
                "text": doc["text"],
                "file_name": doc["file_name"],
                "status": "active",
                "doc_id": "test-doc-001",
            },
        ))

    qdrant.upsert(
        collection_name=settings.qdrant.collection_name,
        points=points,
    )
    logger.info(f"  Inserted {len(points)} points into Qdrant")

    count = qdrant.count(collection_name=settings.qdrant.collection_name).count
    logger.info(f"  Collection now has {count} total points")
    logger.info("  OK - Ingestion works")
    return True


def test_retrieve(embedder: BgeEmbedder, qdrant: QdrantClient, question: str):
    """Step 3: Test vector retrieval."""
    logger.info(f"\n[Step 3] Retrieving for: {question}")

    query_vec = embedder.encode_query(question)
    results = qdrant.query_points(
        collection_name=settings.qdrant.collection_name,
        query=query_vec.tolist(),
        limit=5,
    ).points

    logger.info(f"  Retrieved {len(results)} results:")
    for i, hit in enumerate(results):
        preview = (hit.payload or {}).get("text", "")[:60].replace("\n", " ")
        logger.info(f"    [{i+1}] score={hit.score:.4f} | {(hit.payload or {}).get('file_name', '?')} | {preview}...")

    return results


def test_rerank_and_generate(reranker: BgeReranker, question: str, candidates: list):
    """Step 4+5: Rerank and generate answer."""
    import ollama

    logger.info(f"[Step 4] Reranking {len(candidates)} candidates")

    # Build doc dicts for reranker
    docs = []
    for hit in candidates:
        payload = hit.payload or {}
        docs.append({
            "text": payload.get("text", "")[:512],
            "metadata": payload,
        })

    reranked = reranker.rerank(question, docs, top_n=3)

    logger.info(f"  Reranked top-{len(reranked)}:")
    for i, r in enumerate(reranked):
        preview = r.text[:60].replace("\n", " ")
        source = r.metadata.get("file_name", "?")
        logger.info(f"    [{i+1}] rerank_score={r.score:.4f} | {source} | {preview}...")

    # Step 5: LLM generation
    logger.info(f"\n[Step 5] LLM Generation for: {question}")

    context = "\n\n".join(f"[{i+1}] {r.text}" for i, r in enumerate(reranked))
    prompt = f"""基于以下参考资料回答用户问题。
如果参考资料不足，请如实说明。回答末尾标注引用来源的文件名。

参考资料：
{context}

用户问题：{question}"""

    client = ollama.Client(host=settings.ollama.base_url)
    response = client.generate(
        model=settings.ollama.model,
        prompt=prompt,
        options={"temperature": 0.1},
    )
    answer = response.get("response", "")
    logger.info(f"  LLM Answer:\n{answer}")
    return answer


def main():
    logger.info("=" * 60)
    logger.info("  RAG End-to-End Flow Test (No Dify)")
    logger.info("=" * 60)

    embedder = BgeEmbedder()
    reranker = BgeReranker()
    qdrant = QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=False,
        check_compatibility=False,
    )

    try:
        # Step 1: Test embedding
        test_embedding(embedder)

        # Step 2: Ingest test docs
        test_ingest_to_qdrant(embedder, qdrant)

        # Step 3-5: For each question, retrieve → rerank → generate
        for question in TEST_QUESTIONS:
            logger.info("\n" + "─" * 50)
            candidates = test_retrieve(embedder, qdrant, question)
            if not candidates:
                logger.warning(f"  No results for: {question}")
                continue
            test_rerank_and_generate(reranker, question, candidates)

        logger.info("\n" + "=" * 60)
        logger.info("  ALL TESTS PASSED - End-to-end flow works!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        embedder.unload()
        reranker.unload()


if __name__ == "__main__":
    main()
