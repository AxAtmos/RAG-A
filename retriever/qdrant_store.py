from __future__ import annotations

import threading  # 引入线程锁
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings


class QdrantStore:
    """生产级 Qdrant 向量数据库封装组件"""

    def __init__(self, client: QdrantClient | None = None):
        self._client = client
        self._lock = threading.Lock()  # 线程锁，防止单例初始化冲突

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._lock:
                # 双重检查锁定 (Double-Checked Locking)
                if self._client is None:
                    logger.info(f"Initializing Qdrant client on {settings.qdrant.host}:{settings.qdrant.port}")
                    self._client = QdrantClient(
                        host=settings.qdrant.host,
                        port=settings.qdrant.port,
                        prefer_grpc=False,  # Qdrant container only exposes REST port 6333
                        check_compatibility=False,
                    )
        return self._client

    def ensure_collection(self):
        """确保 Collection 存在，优化集群多库性能"""
        try:
            # 优先使用低开销的直接判定
            # 注：根据不同版本的 qdrant-client，可用 client.collection_exists() 替代
            collections = [c.name for c in self.client.get_collections().collections]
            if settings.qdrant.collection_name not in collections:
                self.client.create_collection(
                    collection_name=settings.qdrant.collection_name,
                    vectors_config=VectorParams(
                        size=settings.qdrant.vector_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {settings.qdrant.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
            raise e

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """向量相似度检索 (支持元数据过滤)"""
        top_k = top_k or settings.retrieval.top_k
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=settings.qdrant.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload or {},
                "text": (hit.payload or {}).get("text", ""),
            }
            for hit in results
        ]

    def scroll_by_doc_id(self, doc_id: str, batch_size: int = 1000) -> list[str]:
        """流式分页获取文档的所有 Point ID (防止超大文档 OOM)"""
        point_ids = []
        offset = None
        
        while True:
            result, next_offset = self.client.scroll(
                collection_name=settings.qdrant.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]),
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend([point.id for point in result])
            if next_offset is None:
                break
            offset = next_offset
            
        return point_ids

    def update_payload_by_filter(self, doc_id: str, payload: dict[str, Any]):
        """单文档 Payload 批量更新（对齐上一版 SyncManager 的调用方法）"""
        self.batch_update_payload_by_doc_ids([doc_id], payload)

    def batch_update_payload_by_doc_ids(self, doc_ids: list[str], payload: dict[str, Any]):
        """🚀 单次网络请求：通过 Filter 批量更新多个文档下所有 Chunk 的 Payload"""
        if not doc_ids:
            return
        
        self.client.set_payload(
            collection_name=settings.qdrant.collection_name,
            payload=payload,
            points=Filter(must=[
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            ]),
        )
        logger.info(f"Batch updated payload for {len(doc_ids)} docs")

    def delete_by_doc_id(self, doc_id: str):
        """单文档级联 Chunk 删除"""
        self.batch_delete_by_doc_ids([doc_id])

    def batch_delete_by_doc_ids(self, doc_ids: list[str]):
        """🚀 单次网络请求：通过 Filter 级联删除多个文档下的所有 Chunk 向量"""
        if not doc_ids:
            return
            
        self.client.delete(
            collection_name=settings.qdrant.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            ]),
        )
        logger.info(f"Batch deleted points for {len(doc_ids)} docs")

    def count_by_status(self, doc_id: str, status: str) -> int:
        """高效统计某状态下的 Chunk 数量"""
        result = self.client.count(
            collection_name=settings.qdrant.collection_name,
            count_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="status", match=MatchValue(value=status)),
            ]),
        )
        return result.count

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """解析过滤字典，自动适配单值 MatchValue 和多值 MatchAny"""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions)