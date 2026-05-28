"""FastAPI main application - RAG Enterprise API.

Four-layer defense pipeline:
  Layer 1: Query Fusion (in HybridRetriever)
  Layer 2: Parent-Child Cascade (in HybridRetriever)
  Layer 3: Cross-Encoder dynamic sliding window (in HybridRetriever)
  Layer 4: Stream buffer audit & retry (in this file)
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import ollama
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from pipeline.embedders import BgeEmbedder
from pipeline.rerankers import BgeReranker
from retriever.qdrant_store import QdrantStore
from retriever.hybrid_retriever import HybridRetriever
from retriever.sync_manager import DocumentSyncManager
from auth.rbac import RBACManager, UserContext
from agents.workflow import AgentWorkflow
from monitoring import PipelineMonitor


# ── Global instances ──
embedder = BgeEmbedder()
reranker = BgeReranker()
qdrant_store: QdrantStore | None = None
hybrid_retriever: HybridRetriever | None = None
pg_engine = None
pg_session_factory = None

# VRAM concurrency semaphore (set in lifespan)
vram_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global qdrant_store, hybrid_retriever, pg_engine, pg_session_factory, vram_semaphore

    logger.info("Starting RAG Enterprise API...")

    # PostgreSQL
    pg_engine = create_engine(settings.postgres.url, pool_size=5, max_overflow=10)
    pg_session_factory = sessionmaker(bind=pg_engine)

    # Qdrant
    qdrant_store = QdrantStore()
    qdrant_store.ensure_collection()

    # LLM function for query fusion
    async def _llm_fn(prompt: str) -> str:
        return await _ollama_generate(prompt)

    # Retriever with full pipeline
    hybrid_retriever = HybridRetriever(
        qdrant_store, embedder, reranker,
        llm_fn=_llm_fn,
        pg_session_factory=pg_session_factory,
    )

    # Auto-detect Ollama
    _check_ollama()

    # VRAM concurrency control
    vram_semaphore = _init_vram_semaphore()
    logger.info(f"VRAM semaphore: max {vram_semaphore._value} concurrent requests")

    logger.info("RAG Enterprise API ready")
    yield

    # Shutdown
    embedder.unload()
    reranker.unload()
    logger.info("RAG Enterprise API stopped")


def _init_vram_semaphore() -> asyncio.Semaphore:
    """Calculate max allowed concurrency based on VRAM formula.

    Formula: Max_Allowed_Concurrency = (Total_VRAM - Model_Weights) / Max_Elastic_Token_Window
    """
    stream_cfg = settings.stream
    available_vram = stream_cfg.vram_total_gb - stream_cfg.model_weights_gb
    max_per_request = stream_cfg.max_elastic_token_window_gb

    if max_per_request > 0:
        max_concurrency = max(1, int(available_vram / max_per_request))
    else:
        max_concurrency = 2  # Safe default

    return asyncio.Semaphore(max_concurrency)


app = FastAPI(
    title="RAG Enterprise API",
    description="企业级 RAG 系统后端 API",
    version="1.0.0",
    lifespan=lifespan,
)


def _check_ollama():
    """Auto-detect Ollama environment and pull model if needed."""
    try:
        client = ollama.Client(host=settings.ollama.base_url)
        models = client.list()
        model_names = [m.model for m in models.models]
        if settings.ollama.model not in model_names:
            logger.info(f"Model {settings.ollama.model} not found, pulling...")
            client.pull(settings.ollama.model)
            logger.info(f"Model {settings.ollama.model} pulled successfully")
        else:
            logger.info(f"Ollama model {settings.ollama.model} available")
    except Exception as e:
        logger.warning(f"Ollama not available: {e}. LLM features will be limited.")


def _get_pg() -> Session:
    if pg_session_factory is None:
        raise HTTPException(503, "Database not ready")
    session = pg_session_factory()
    try:
        yield session
    finally:
        session.close()


async def _ollama_generate(prompt: str, monitor: PipelineMonitor | None = None, model: str | None = None) -> str:
    """Call Ollama LLM (non-streaming) with optional monitor for timeout detection.

    Args:
        model: Override model name. If None, uses settings.ollama.model.
    """
    import time as _time
    llm_model = model or settings.ollama.model
    try:
        def _sync_generate():
            client = ollama.Client(host=settings.ollama.base_url)
            resp = client.generate(
                model=llm_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_ctx": settings.ollama.num_ctx},
            )
            return resp.get("response", "")
        loop = asyncio.get_running_loop()
        start = _time.monotonic()
        result = await loop.run_in_executor(None, _sync_generate)
        elapsed = _time.monotonic() - start
        if monitor:
            monitor.record_ollama_success()
        if elapsed > settings.ollama.timeout * 0.8:
            logger.warning(f"[L4] Ollama generate took {elapsed:.1f}s (near timeout threshold)")
        return result
    except Exception as e:
        elapsed = _time.monotonic() - start
        if monitor:
            monitor.record_ollama_timeout(elapsed)
        logger.error(f"Ollama call failed after {elapsed:.1f}s: {e}")
        return ""


def _ollama_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Call Ollama LLM with streaming."""
    try:
        client = ollama.Client(host=settings.ollama.base_url)
        stream = client.generate(
            model=settings.ollama.model,
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": settings.ollama.num_ctx},
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("response", "")
            if token:
                yield token
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"[ERROR] LLM stream failed: {e}"


# ── Request / Response Models ──

class QueryRequest(BaseModel):
    question: str
    user_id: int | None = None
    conversation_id: str | None = None
    include_deprecated: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    confidence: str


class DeprecateRequest(BaseModel):
    doc_id: str
    reason: str = ""
    superseded_by: str | None = None


class BatchDeprecateRequest(BaseModel):
    filters: dict[str, Any]
    reason: str = ""


class RestoreRequest(BaseModel):
    doc_id: str


class HardDeleteRequest(BaseModel):
    doc_id: str
    confirm: bool = False


# ── Query Endpoint (SSE Streaming with Layer 4 audit) ──

@app.post("/api/query")
async def query_stream(req: QueryRequest, pg: Session = Depends(_get_pg)):
    """Main RAG query endpoint with SSE streaming and Layer 4 retry.

    Implements:
    - Token buffer audit (first N tokens checked for INSUFFICIENT_INFO)
    - Retry state machine (expand Qdrant to Top-50 on insufficient info)
    - VRAM semaphore concurrency control
    """
    if not hybrid_retriever:
        raise HTTPException(503, "Retriever not ready")

    # Get user context for permission filtering
    user_filters = {}
    if req.user_id:
        rbac = RBACManager(pg)
        user = rbac.get_user_context(req.user_id)
        if user:
            user_filters = rbac.build_search_filter(user, req.include_deprecated)

    async def _stream_generator():
        """SSE stream generator with Layer 4 audit."""
        monitor = PipelineMonitor(question=req.question)
        # Acquire VRAM semaphore
        if vram_semaphore:
            await vram_semaphore.acquire()
        try:
            # Layer 1-3: Retrieve with four-layer defense
            result = await hybrid_retriever.retrieve(
                query=req.question,
                user_filters=user_filters,
                include_deprecated=req.include_deprecated,
                monitor=monitor,
            )

            # Build context
            context = "\n\n".join(
                f"[{i+1}] {r['text']}" for i, r in enumerate(result["results"])
            )
            sources = [
                {
                    "file_name": r["metadata"].get("file_name", ""),
                    "chunk_index": r["metadata"].get("chunk_index", 0),
                    "score": r["score"],
                    "department": r["metadata"].get("department", ""),
                    "project": r["metadata"].get("project", ""),
                }
                for r in result["results"]
            ]

            prompt = f"""基于以下参考资料回答用户问题。
如果参考资料不足，请如实说明。回答末尾标注引用来源的文件名。

参考资料：
{context}

用户问题：{req.question}"""

            # Layer 4: Stream with token buffer audit
            token_buffer = ""
            insufficient_detected = False
            buffer_size = settings.ollama.token_buffer_size
            insufficient_tag = settings.stream.insufficient_tag
            llm_stage_ctx = monitor.start_stage("L4_llm_stream", model=settings.ollama.model)
            llm_stage = llm_stage_ctx.__enter__()

            try:
                for token in _ollama_stream(prompt):
                    if insufficient_detected:
                        # Already in retry mode, skip normal output
                        break

                    token_buffer += token

                    # Audit within buffer window
                    if len(token_buffer) < buffer_size * 5:  # ~5 chars per token estimate
                        if insufficient_tag in token_buffer:
                            insufficient_detected = True
                            logger.warning(f"[L4] INSUFFICIENT_INFO detected, triggering retry")

                            # Retry: expand retrieval range
                            retry_result = await hybrid_retriever.retrieve(
                                query=req.question,
                                user_filters=user_filters,
                                include_deprecated=req.include_deprecated,
                            )

                            # Build new context with expanded results
                            retry_context = "\n\n".join(
                                f"[{i+1}] {r['text']}" for i, r in enumerate(retry_result["results"])
                            )
                            retry_sources = [
                                {
                                    "file_name": r["metadata"].get("file_name", ""),
                                    "chunk_index": r["metadata"].get("chunk_index", 0),
                                    "score": r["score"],
                                }
                                for r in retry_result["results"]
                            ]

                            # Merge sources
                            sources = sources + retry_sources

                            retry_prompt = f"""基于以下参考资料回答用户问题。
如果参考资料不足，请如实说明。回答末尾标注引用来源的文件名。

参考资料：
{retry_context}

用户问题：{req.question}"""

                            # Send retry signal
                            yield f"data: {json.dumps({'type': 'retry', 'reason': 'insufficient_info'})}\n\n"

                            # Stream retry response
                            for retry_token in _ollama_stream(retry_prompt):
                                yield f"data: {json.dumps({'type': 'token', 'content': retry_token})}\n\n"
                            break
                        else:
                            # Still in buffer window, emit token
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    else:
                        # Past buffer window, normal streaming
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            finally:
                llm_stage.update(tokens_streamed=len(token_buffer))
                llm_stage_ctx.__exit__(None, None, None)

            # Send metadata
            yield f"data: {json.dumps({'type': 'meta', 'sources': sources, 'confidence': result['confidence']})}\n\n"
            yield "data: [DONE]\n\n"

            monitor.finish()

        finally:
            if vram_semaphore:
                vram_semaphore.release()

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/query-sync", response_model=QueryResponse)
async def query_sync(req: QueryRequest, pg: Session = Depends(_get_pg)):
    """Synchronous query endpoint with full pipeline monitoring."""
    if not hybrid_retriever:
        raise HTTPException(503, "Retriever not ready")

    monitor = PipelineMonitor(question=req.question)

    user_filters = {}
    if req.user_id:
        rbac = RBACManager(pg)
        user = rbac.get_user_context(req.user_id)
        if user:
            user_filters = rbac.build_search_filter(user, req.include_deprecated)

    # L1-L3: Retrieve
    result = await hybrid_retriever.retrieve(
        query=req.question,
        user_filters=user_filters,
        include_deprecated=req.include_deprecated,
        monitor=monitor,
    )

    context = "\n\n".join(
        f"[{i+1}] {r['text']}" for i, r in enumerate(result["results"])
    )
    sources = [
        {
            "file_name": r["metadata"].get("file_name", ""),
            "chunk_index": r["metadata"].get("chunk_index", 0),
            "score": r["score"],
            "department": r["metadata"].get("department", ""),
            "project": r["metadata"].get("project", ""),
        }
        for r in result["results"]
    ]

    prompt = f"""基于以下参考资料回答用户问题。
如果参考资料不足，请如实说明。回答末尾标注引用来源的文件名。

参考资料：
{context}

用户问题：{req.question}"""

    # L4: LLM Generation with monitoring
    llm_model = settings.retrieval.query_llm_model or settings.ollama.model
    with monitor.start_stage("L4_llm_generate", model=llm_model) as stage:
        answer = await _ollama_generate(prompt, monitor=monitor, model=llm_model or None)
        stage.update(response_chars=len(answer))

    if result["confidence"] == "low":
        answer = "⚠ 参考资料不足，以下为推测性回答。\n\n" + answer

    monitor.finish()
    return QueryResponse(answer=answer, sources=sources, confidence=result["confidence"])


# ── Admin Endpoints ──

@app.post("/api/admin/deprecate")
async def deprecate_doc(req: DeprecateRequest, pg: Session = Depends(_get_pg)):
    """Deprecate a single document."""
    rbac = RBACManager(pg)
    # TODO: get operator_id from auth token
    operator_id = 1

    # Check document exists and get department
    row = pg.execute(text(
        "SELECT department FROM doc_lifecycle WHERE doc_id = :doc_id"
    ), {"doc_id": req.doc_id}).fetchone()

    if not row:
        raise HTTPException(404, "Document not found")

    user = rbac.get_user_context(operator_id)
    if not user or not rbac.can_deprecate(user, row[0]):
        raise HTTPException(403, "Insufficient permissions")

    sync = DocumentSyncManager(qdrant_store, pg)
    sync.deprecate_document(req.doc_id, req.reason, operator_id)
    return {"status": "ok", "doc_id": req.doc_id}


@app.post("/api/admin/batch-deprecate")
async def batch_deprecate(req: BatchDeprecateRequest, pg: Session = Depends(_get_pg)):
    """Batch deprecate documents."""
    rbac = RBACManager(pg)
    operator_id = 1

    user = rbac.get_user_context(operator_id)
    if not user or not user.is_admin:
        raise HTTPException(403, "Insufficient permissions")

    sync = DocumentSyncManager(qdrant_store, pg)
    result = sync.batch_deprecate(req.filters, req.reason, operator_id)
    return result


@app.post("/api/admin/restore")
async def restore_doc(req: RestoreRequest, pg: Session = Depends(_get_pg)):
    """Restore a deprecated document."""
    rbac = RBACManager(pg)
    operator_id = 1

    row = pg.execute(text(
        "SELECT department FROM doc_lifecycle WHERE doc_id = :doc_id"
    ), {"doc_id": req.doc_id}).fetchone()

    if not row:
        raise HTTPException(404, "Document not found")

    user = rbac.get_user_context(operator_id)
    if not user or not rbac.can_restore(user, row[0]):
        raise HTTPException(403, "Insufficient permissions")

    sync = DocumentSyncManager(qdrant_store, pg)
    sync.restore_document(req.doc_id, operator_id)
    return {"status": "ok", "doc_id": req.doc_id}


@app.delete("/api/admin/hard-delete")
async def hard_delete(req: HardDeleteRequest, pg: Session = Depends(_get_pg)):
    """Hard delete - super_admin only."""
    if not req.confirm:
        raise HTTPException(400, "Must set confirm=true")

    rbac = RBACManager(pg)
    operator_id = 1

    user = rbac.get_user_context(operator_id)
    if not user or not rbac.can_hard_delete(user):
        raise HTTPException(403, "Super admin required")

    sync = DocumentSyncManager(qdrant_store, pg)
    sync.hard_delete(req.doc_id, operator_id)
    return {"status": "ok", "doc_id": req.doc_id}


@app.get("/api/admin/docs")
async def list_docs(
    status: str = Query("active"),
    department: str | None = None,
    project: str | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    pg: Session = Depends(_get_pg),
):
    """List documents with filters."""
    conditions = ["status = :status"]
    params: dict[str, Any] = {"status": status, "limit": per_page, "offset": (page - 1) * per_page}

    if department:
        conditions.append("department = :department")
        params["department"] = department
    if project:
        conditions.append("project = :project")
        params["project"] = project

    where = " AND ".join(conditions)

    rows = pg.execute(text(f"""
        SELECT doc_id, file_name, department, project, category, status, total_chunks, created_at
        FROM doc_lifecycle WHERE {where}
        ORDER BY created_at DESC LIMIT :limit OFFSET :offset
    """), params).fetchall()

    total = pg.execute(text(f"SELECT COUNT(*) FROM doc_lifecycle WHERE {where}"), params).scalar()

    docs = [
        {
            "doc_id": r[0], "file_name": r[1], "department": r[2],
            "project": r[3], "category": r[4], "status": r[5],
            "total_chunks": r[6], "created_at": str(r[7]),
        }
        for r in rows
    ]

    return {"total": total, "page": page, "per_page": per_page, "docs": docs}


@app.get("/api/admin/deprecated-docs")
async def list_deprecated(
    department: str | None = None,
    page: int = Query(1, ge=1),
    pg: Session = Depends(_get_pg),
):
    """List deprecated documents."""
    conditions = ["status = 'deprecated'"]
    params: dict[str, Any] = {"limit": 20, "offset": (page - 1) * 20}

    if department:
        conditions.append("department = :department")
        params["department"] = department

    where = " AND ".join(conditions)

    rows = pg.execute(text(f"""
        SELECT doc_id, file_name, deprecated_at, deprecated_reason, superseded_by, total_chunks
        FROM doc_lifecycle WHERE {where}
        ORDER BY deprecated_at DESC LIMIT :limit OFFSET :offset
    """), params).fetchall()

    docs = [
        {
            "doc_id": r[0], "file_name": r[1],
            "deprecated_at": str(r[2]) if r[2] else None,
            "deprecated_reason": r[3],
            "superseded_by": r[4],
            "chunks_count": r[5],
        }
        for r in rows
    ]

    return {"total": len(docs), "docs": docs}


@app.get("/api/admin/audit-log")
async def audit_log(
    doc_id: str | None = None,
    action: str | None = None,
    page: int = Query(1, ge=1),
    pg: Session = Depends(_get_pg),
):
    """View audit log."""
    conditions = []
    params: dict[str, Any] = {"limit": 50, "offset": (page - 1) * 50}

    if doc_id:
        conditions.append("doc_id = :doc_id")
        params["doc_id"] = doc_id
    if action:
        conditions.append("action = :action")
        params["action"] = action

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = pg.execute(text(f"""
        SELECT id, doc_id, action, operator_id, reason, created_at
        FROM doc_audit_log {where}
        ORDER BY created_at DESC LIMIT :limit OFFSET :offset
    """), params).fetchall()

    logs = [
        {
            "id": r[0], "doc_id": r[1], "action": r[2],
            "operator_id": r[3], "reason": r[4], "created_at": str(r[5]),
        }
        for r in rows
    ]

    return {"logs": logs}


# ── VRAM Status ──

@app.get("/api/vram-status")
async def vram_status():
    """Check VRAM concurrency status."""
    stream_cfg = settings.stream
    available_vram = stream_cfg.vram_total_gb - stream_cfg.model_weights_gb
    max_per_request = stream_cfg.max_elastic_token_window_gb
    max_concurrency = max(1, int(available_vram / max_per_request)) if max_per_request > 0 else 2

    return {
        "total_vram_gb": stream_cfg.vram_total_gb,
        "model_weights_gb": stream_cfg.model_weights_gb,
        "available_vram_gb": available_vram,
        "max_elastic_window_gb": max_per_request,
        "max_concurrency": max_concurrency,
        "warning_threshold": stream_cfg.vram_warning_threshold,
        "elastic_k_range": f"{settings.retrieval.elastic_min_k} ~ {settings.retrieval.elastic_max_k}",
    }


# ── Health check ──

@app.get("/health")
async def health():
    checks = {"api": "ok", "qdrant": "unknown", "postgres": "unknown", "ollama": "unknown"}

    # Qdrant
    try:
        if qdrant_store:
            qdrant_store.client.get_collections()
            checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "error"

    # PostgreSQL
    try:
        if pg_engine:
            with pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
    except Exception:
        checks["postgres"] = "error"

    # Ollama
    try:
        client = ollama.Client(host=settings.ollama.base_url)
        client.list()
        checks["ollama"] = "ok"
    except Exception:
        checks["ollama"] = "error"

    return checks


# ── Pipeline Monitor Endpoints ──

@app.get("/api/pipeline-monitor")
async def pipeline_monitor_list():
    """Get recent request traces from the pipeline monitor ring buffer."""
    return {"traces": PipelineMonitor.get_recent_traces()}


@app.get("/api/pipeline-monitor/stats")
async def pipeline_monitor_stats():
    """Get aggregated pipeline statistics (avg/p50/p95 per stage)."""
    return PipelineMonitor.get_stats()


@app.get("/api/pipeline-monitor/{request_id}")
async def pipeline_monitor_detail(request_id: str):
    """Get a single request trace by ID."""
    trace = PipelineMonitor.get_trace_by_id(request_id)
    if not trace:
        raise HTTPException(404, f"Trace {request_id} not found")
    return trace


@app.get("/api/health/detailed")
async def health_detailed():
    """Detailed health check with per-service latency probes."""
    import time as _time
    checks: dict[str, Any] = {"api": "ok"}

    # Qdrant
    try:
        if qdrant_store:
            start = _time.monotonic()
            qdrant_store.client.get_collections()
            latency = (_time.monotonic() - start) * 1000
            checks["qdrant"] = {"status": "ok", "latency_ms": round(latency, 1)}
    except Exception as e:
        checks["qdrant"] = {"status": "error", "error": str(e)}

    # PostgreSQL
    try:
        if pg_engine:
            start = _time.monotonic()
            with pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            latency = (_time.monotonic() - start) * 1000
            checks["postgres"] = {"status": "ok", "latency_ms": round(latency, 1)}
    except Exception as e:
        checks["postgres"] = {"status": "error", "error": str(e)}

    # Ollama
    try:
        start = _time.monotonic()
        client = ollama.Client(host=settings.ollama.base_url)
        models_resp = client.list()
        latency = (_time.monotonic() - start) * 1000
        model_names = [m.model for m in models_resp.models]
        checks["ollama"] = {"status": "ok", "latency_ms": round(latency, 1)}
        checks["ollama_models"] = model_names
    except Exception as e:
        checks["ollama"] = {"status": "error", "error": str(e)}

    # VRAM semaphore
    if vram_semaphore:
        checks["vram_semaphore"] = {
            "max_concurrency": vram_semaphore._value,
            "available": vram_semaphore._value,
        }

    # Monitor stats summary
    stats = PipelineMonitor.get_stats()
    if stats.get("total_requests", 0) > 0:
        checks["pipeline_stats"] = {
            "total_requests": stats["total_requests"],
            "avg_total_ms": stats.get("avg_total_ms", 0),
            "slowest_stage": stats.get("slowest_stage_avg", ""),
            "total_errors": stats.get("total_errors", 0),
        }

    return checks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
