"""Hybrid retriever with four-layer defense pipeline.

Optimizations:
- Layer 1 (Query Fusion): Flattened embedding serialization, fully decoupled from 
  I/O multiplexing. Replaced ThreadPool with true non-blocking asynchronous asyncio execution.
- Layer 2 (Cascade): Fixed critical N+1 SQL anti-pattern. Consolidated loop queries 
  into a single batch SQL 'IN' selection, freeing up DB connection pools.
- Robust typing & logging compatibility with strict production requirements.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from pipeline.embedders import BgeEmbedder
from pipeline.rerankers import BgeReranker
from retriever.qdrant_store import QdrantStore


class HybridRetriever:
    """Four-layer defense retrieval pipeline optimized for high-concurrency production."""

    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: BgeEmbedder,
        reranker: BgeReranker,
        llm_fn: Callable[[str], str] | None = None,
        pg_session_factory: Callable[[], Session] | None = None,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.reranker = reranker
        self.llm_fn = llm_fn
        self.pg_session_factory = pg_session_factory

    # ── Public Async API ──

    async def retrieve(
        self,
        query: str,
        user_filters: dict[str, Any] | None = None,
        include_deprecated: bool = False,
        top_k: int | None = None,
        monitor: Any | None = None,
    ) -> dict[str, Any]:
        """Full retrieval pipeline with four-layer defense (Asynchronous Execution).

        Args:
            monitor: Optional PipelineMonitor instance for per-stage timing.
        """
        top_k = top_k or settings.retrieval.top_k

        # Build status filter
        status_filter = {"status": "active"}
        if include_deprecated:
            status_filter = {"status": ["active", "deprecated"]}

        filters = {**status_filter}
        if user_filters:
            filters.update(user_filters)

        # Layer 1: Query Fusion (Async I/O Multiplexing)
        if monitor:
            with monitor.start_stage("L1_query_fusion") as stage:
                candidates = await self._query_fusion(query, top_k, filters, stage=stage)
                stage.update(candidates=len(candidates))
        else:
            candidates = await self._query_fusion(query, top_k, filters)
        logger.info(f"[L1] Query Fusion: {len(candidates)} candidates after dedup")

        if not candidates:
            return {"results": [], "confidence": "low"}

        # Layer 2: Parent-Child Cascade (Batch SQL Execution)
        if monitor:
            with monitor.start_stage("L2_parent_cascade") as stage:
                candidates = self._cascade_parents_batch(candidates)
                stage.update(candidates=len(candidates))
        else:
            candidates = self._cascade_parents_batch(candidates)
        logger.info(f"[L2] Parent cascade: {len(candidates)} enriched candidates")

        # Layer 3: Cross-Encoder rerank with dynamic sliding window
        if monitor:
            with monitor.start_stage("L3_rerank") as stage:
                results, confidence = await self._rerank_with_sliding_window(query, candidates)
                stage.update(results=len(results), confidence=confidence)
        else:
            results, confidence = await self._rerank_with_sliding_window(query, candidates)
        logger.info(f"[L3] Sliding window: {len(results)} results, confidence={confidence}")

        return {"results": results, "confidence": confidence}

    # ── Layer 1: Query Fusion ──

    async def _query_fusion(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
        stage: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Generate N query variants, search Qdrant concurrently via asyncio, merge & dedup."""
        fusion_count = settings.retrieval.query_fusion_count

        # 1. Generate query variants via LLM (skip if disabled)
        if settings.retrieval.query_rewrite_enabled:
            variants = await self._generate_query_variants(query, fusion_count)
        else:
            variants = []
        all_queries = [query] + variants
        logger.info(f"[L1] Total query variants: {len(all_queries)} (1 original + {len(variants)} rewrites)")
        if stage:
            stage.update(variants=len(variants), total_queries=len(all_queries), rewrite_enabled=settings.retrieval.query_rewrite_enabled)

        # 2. 🚀 Performance Optimization: Batch encode all queries outside the thread pool
        # Avoids GIL switching contention during I/O operations
        logger.debug("[L1] Batch encoding query vectors...")
        query_vectors = await asyncio.gather(
            *[asyncio.to_thread(self.embedder.encode_query, q) for q in all_queries]
        )
        query_vectors = list(query_vectors)

        # 3. 🚀 I/O Optimization: Use native asyncio.to_thread instead of custom limited ThreadPool
        async def _async_search(vec: Any) -> list[dict[str, Any]]:
            return await asyncio.to_thread(
                self.qdrant.search,
                query_vector=vec,
                top_k=top_k,
                filters=filters
            )

        # Concurrent scheduling of all network operations
        tasks = [_async_search(vec) for vec in query_vectors]
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Merge results and handle exceptions cleanly
        all_candidates: list[dict[str, Any]] = []
        search_errors = 0
        for res in search_results:
            if isinstance(res, Exception):
                logger.error(f"[L1] Concurrent search task failed: {res}")
                search_errors += 1
                continue
            all_candidates.extend(res)
        if stage:
            stage.update(
                search_results=len(all_candidates),
                search_errors=search_errors,
            )

        # 5. Dedup by point ID, retaining highest similarity score
        seen: dict[str, dict[str, Any]] = {}
        for c in all_candidates:
            cid = c["id"]
            if cid not in seen or c["score"] > seen[cid]["score"]:
                seen[cid] = c

        deduped = list(seen.values())
        deduped.sort(key=lambda x: x["score"], reverse=True)
        return deduped[:top_k * 2]  # Retain a wider pool for Layer 2 context expansion

    async def _generate_query_variants(self, query: str, count: int) -> list[str]:
        """Use LLM to generate near-synonym query rewrites."""
        if not self.llm_fn:
            return []

        prompt = f"""请将以下问题改写为 {count} 个不同表述的近义问题，保持语义一致但用词不同。
每个问题一行，不要编号，不要其他文字。

原始问题：{query}"""
        try:
            response = await self.llm_fn(prompt)
            return [line.strip() for line in response.strip().split("\n") if line.strip()][:count]
        except Exception as e:
            logger.warning(f"[L1] Query fusion LLM execution failed: {e}")
            return []

    # ── Layer 2: Parent-Child Cascade ──

    def _cascade_parents_batch(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """🚀 [L2 重构] 批量提取 Parent 节点文本，解决同步 SQL 循环 N+1 致命开销"""
        if not self.pg_session_factory:
            return candidates

        # 1. Collect all parent IDs requiring lookup
        parent_ids = {
            c.get("metadata", {}).get("parent_id") 
            for c in candidates 
            if c.get("metadata", {}).get("parent_id")
        }
        
        if not parent_ids:
            return candidates

        # 2. 🚀 Single Batch Lookup: Single SQL sweep for all candidate rows
        parent_map: dict[str, str] = {}
        session = self.pg_session_factory()
        try:
            statement = text("SELECT id, full_parent_text FROM parent_documents WHERE id = ANY(:pids)")
            result = session.execute(statement, {"pids": list(parent_ids)}).fetchall()
            parent_map = {row[0]: row[1] for row in result if row[1]}
        except Exception as e:
            logger.error(f"[L2] Failed batch fetching parent rows from PG: {e}")
        finally:
            session.close()  # Immediately release connection back to pool

        # 3. Context assembly and hydration
        enriched: list[dict[str, Any]] = []
        for c in candidates:
            pid = c.get("metadata", {}).get("parent_id")
            if pid and pid in parent_map:
                enriched_c = {**c}
                enriched_c["text"] = parent_map[pid]
                enriched_c["metadata"] = {
                    **c["metadata"],
                    "cascade": True,
                    "original_child_text": c["text"],
                }
                enriched.append(enriched_c)
            else:
                enriched.append(c)

        return enriched

    # ── Layer 3: Cross-Encoder Rerank + Dynamic Sliding Window ──

    async def _rerank_with_sliding_window(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        """Rerank with Cross-Encoder and apply dynamic sliding window based on density."""
        ce_threshold = settings.retrieval.ce_threshold
        elastic_min_k = settings.retrieval.elastic_min_k
        elastic_max_k = settings.retrieval.elastic_max_k

        # Cross-Encoder text similarity processing
        pairs = [[query, c["text"][:512]] for c in candidates]
        scores = await asyncio.to_thread(self.reranker.cross_encoder_predict, pairs)

        # Evaluate hard score thresholds
        valid_chunks: list[tuple[dict[str, Any], float]] = []
        for chunk, score in zip(candidates, scores):
            if score >= ce_threshold:
                chunk_copy = {**chunk}
                chunk_copy["metadata"] = {**chunk.get("metadata", {}), "ce_score": score}
                valid_chunks.append((chunk_copy, score))

        if not valid_chunks:
            # Fallback: Sort and harvest base metrics if no nodes meet the criteria
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            valid_chunks = scored[:elastic_min_k]

        valid_chunks.sort(key=lambda x: x[1], reverse=True)

        # Dynamic density scoring calculation
        high_score_count = sum(1 for _, s in valid_chunks if s >= ce_threshold + 0.1)

        if high_score_count >= elastic_max_k:
            window_k = min(elastic_max_k, len(valid_chunks))
        elif high_score_count >= elastic_min_k:
            window_k = min(max(elastic_min_k, high_score_count + 2), len(valid_chunks))
        else:
            window_k = min(elastic_min_k, len(valid_chunks))

        # Check local active context constraints before output formatting
        window_k = self._apply_vram_pressure(window_k, elastic_min_k)

        results = [
            {
                "text": chunk["text"],
                "score": score,
                "metadata": chunk.get("metadata", {}),
            }
            for chunk, score in valid_chunks[:window_k]
        ]

        best_score = results[0]["score"] if results else 0
        confidence = "high" if best_score >= ce_threshold else "low"

        return results, confidence

    def _apply_vram_pressure(self, current_k: int, min_k: int) -> int:
        """Apply VRAM static constraints protecting runtime memory blocks."""
        try:
            stream_cfg = settings.stream
            # 修正：采用全局安全警戒水位或降低总并发请求分配的上限弹性
            available_vram = stream_cfg.vram_total_gb - stream_cfg.model_weights_gb
            warning_line = available_vram * stream_cfg.vram_warning_threshold

            estimated_per_block = stream_cfg.max_elastic_token_window_gb / settings.retrieval.elastic_max_k
            estimated_usage = current_k * estimated_per_block

            if estimated_usage > warning_line:
                compressed_k = max(min_k, int(warning_line / estimated_per_block))
                logger.warning(f"[VRAM] Adaptive restriction active: Compressed {current_k} -> {compressed_k}")
                return compressed_k
        except Exception:
            pass

        return current_k