"""Document sync manager: keeps PostgreSQL and Qdrant in sync.

Optimizations:
- Defensive distributed writes: Status sync failures are caught and recorded in audit log,
  PG transactions commit with inconsistency noted to avoid database connection locks.
- Strong consistency for hard deletes: Qdrant delete failures trigger a PG rollback 
  to prevent irreversible orphaned chunks ("ghost vectors") in the vector store.
- Batch Qdrant operations: Multi-doc updates/deletes use single filter-based requests
  instead of per-doc loops, eliminating N+1 network I/O.
- Memory protection: Replaced unbounded fetchall() with explicit sizing protections.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from retriever.qdrant_store import QdrantStore


class DocumentSyncManager:
    """Unified manager for PostgreSQL + Qdrant document state sync."""

    def __init__(self, qdrant: QdrantStore, pg_session: Session):
        self.qdrant = qdrant
        self.pg = pg_session

    def deprecate_document(self, doc_id: str, reason: str, operator_id: int):
        """Deprecate a single document - sync both databases."""
        from sqlalchemy import text

        # 1. Update PostgreSQL
        self.pg.execute(text("""
            UPDATE doc_lifecycle SET
                status = 'deprecated',
                deprecated_reason = :reason,
                deprecated_at = NOW(),
                deprecated_by = :operator_id
            WHERE doc_id = :doc_id
        """), {"reason": reason, "operator_id": operator_id, "doc_id": doc_id})

        # 2. Sync Qdrant (defensive write)
        try:
            self.qdrant.update_payload_by_filter(doc_id, {
                "status": "deprecated",
                "deprecated_reason": reason,
            })
        except Exception as e:
            err: str = str(e)
            logger.error(f"Qdrant sync failed for deprecate {doc_id}: {err}")
            self._audit_log(doc_id, "deprecate", operator_id, reason,
                            details={"qdrant_sync": "FAILED", "error": err})
            self.pg.commit()
            return

        # 3. Audit log (success)
        self._audit_log(doc_id, "deprecate", operator_id, reason)
        self.pg.commit()
        logger.info(f"Deprecated document {doc_id} by user {operator_id}")

    def batch_deprecate(self, filters: dict[str, Any], reason: str, operator_id: int) -> dict[str, Any]:
        """Batch deprecate documents matching filters with safe memory and network batching."""
        from sqlalchemy import text

        # Build WHERE clause
        conditions = ["status = 'active'"]
        params: dict[str, Any] = {}
        for key, value in filters.items():
            if key in ("department", "project", "category", "file_type"):
                conditions.append(f"{key} = :{key}")
                params[key] = value

        where = " AND ".join(conditions)
        
        # 优化：流式迭代或加 LIMIT 限制单次批处理上限，防止数十万数据导致内存 OOM
        result = self.pg.execute(
            text(f"SELECT doc_id FROM doc_lifecycle WHERE {where} LIMIT 5000"), params
        )
        doc_ids = [r[0] for r in result]

        if not doc_ids:
            return {"affected_docs": 0, "qdrant_sync": "SKIPPED"}

        # 1. Update PostgreSQL
        self.pg.execute(text("""
            UPDATE doc_lifecycle SET
                status = 'deprecated',
                deprecated_reason = :reason,
                deprecated_at = NOW(),
                deprecated_by = :operator_id
            WHERE doc_id = ANY(:doc_ids)
        """), {"reason": reason, "operator_id": operator_id, "doc_ids": doc_ids})

        # 2. Batch update Qdrant in a single filter request (defensive)
        try:
            self.qdrant.batch_update_payload_by_doc_ids(doc_ids, {
                "status": "deprecated",
                "deprecated_reason": reason,
            })
            qdrant_status = "OK"
        except Exception as e:
            err: str = str(e)
            logger.error(f"Qdrant batch sync failed for batch_deprecate: {err}")
            qdrant_status = f"FAILED: {err}"

        # 3. Audit & Commit
        self._audit_log(None, "batch_deprecate", operator_id, reason,
                        details={"filters": filters, "doc_ids": doc_ids,
                                 "qdrant_sync": qdrant_status})
        self.pg.commit()
        
        # 规避日志终端输出中文 Unicode 乱码
        readable_filters = json.dumps(filters, ensure_ascii=False)
        logger.info(f"Batch deprecated {len(doc_ids)} docs matching {readable_filters} (qdrant={qdrant_status})")
        return {"affected_docs": len(doc_ids), "qdrant_sync": qdrant_status}

    def restore_document(self, doc_id: str, operator_id: int):
        """Restore a deprecated document."""
        from sqlalchemy import text

        self.pg.execute(text("""
            UPDATE doc_lifecycle SET
                status = 'active',
                deprecated_reason = NULL,
                deprecated_at = NULL,
                deprecated_by = NULL
            WHERE doc_id = :doc_id
        """), {"doc_id": doc_id})

        try:
            self.qdrant.update_payload_by_filter(doc_id, {
                "status": "active",
                "deprecated_reason": "",
            })
        except Exception as e:
            err: str = str(e)
            logger.error(f"Qdrant sync failed for restore {doc_id}: {err}")
            self._audit_log(doc_id, "restore", operator_id,
                            details={"qdrant_sync": "FAILED", "error": err})
            self.pg.commit()
            return

        self._audit_log(doc_id, "restore", operator_id)
        self.pg.commit()
        logger.info(f"Restored document {doc_id}")

    def hard_delete(self, doc_id: str, operator_id: int):
        """Hard delete a single document with strong consistency rollback policy.
        
        Prevents orphaned chunks (ghost vectors) when vector database fails.
        """
        from sqlalchemy import text

        # 1. Stage PostgreSQL delete
        self.pg.execute(text(
            "DELETE FROM doc_lifecycle WHERE doc_id = :doc_id"
        ), {"doc_id": doc_id})

        # 2. Force rollback if Qdrant sync fails (Prevents un-trackable orphan vectors)
        try:
            self.qdrant.delete_by_doc_id(doc_id)
        except Exception as e:
            self.pg.rollback()
            logger.error(f"Qdrant delete failed for {doc_id}, SQL transaction rolled back: {str(e)}")
            raise e

        # 3. Commit only if both succeeded
        self._audit_log(doc_id, "hard_delete", operator_id)
        self.pg.commit()
        logger.info(f"Hard deleted document {doc_id}")

    def batch_hard_delete(self, doc_ids: list[str], operator_id: int) -> dict[str, Any]:
        """Batch hard delete multiple documents in a single transactional/filter request."""
        from sqlalchemy import text

        if not doc_ids:
            return {"affected_docs": 0, "qdrant_sync": "SKIPPED"}

        # 1. Stage batch delete
        self.pg.execute(text(
            "DELETE FROM doc_lifecycle WHERE doc_id = ANY(:doc_ids)"
        ), {"doc_ids": doc_ids})

        # 2. Batch delete from Qdrant with atomic rollback design
        try:
            self.qdrant.batch_delete_by_doc_ids(doc_ids)
            qdrant_status = "OK"
        except Exception as e:
            self.pg.rollback()
            logger.error(f"Qdrant batch delete failed, SQL transaction rolled back: {str(e)}")
            raise e

        # 3. Commit
        self._audit_log(None, "batch_hard_delete", operator_id,
                        details={"doc_ids": doc_ids, "qdrant_sync": qdrant_status})
        self.pg.commit()
        logger.info(f"Batch hard deleted {len(doc_ids)} documents (qdrant={qdrant_status})")
        return {"affected_docs": len(doc_ids), "qdrant_sync": qdrant_status}

    def check_consistency(self) -> list[dict[str, Any]]:
        """Check and fix consistency between PostgreSQL and Qdrant."""
        from sqlalchemy import text

        rows = self.pg.execute(text(
            "SELECT doc_id FROM doc_lifecycle WHERE status = 'deprecated'"
        )).fetchall()

        fixes: list[dict[str, Any]] = []
        for (doc_id,) in rows:
            try:
                active_count = self.qdrant.count_by_status(doc_id, "active")
                if active_count > 0:
                    self.qdrant.update_payload_by_filter(doc_id, {
                        "status": "deprecated",
                    })
                    fixes.append({"doc_id": doc_id, "fixed_chunks": active_count})
                    logger.warning(f"Consistency fix: {doc_id}, {active_count} chunks synced to 'deprecated'")
            except Exception as e:
                err: str = str(e)
                logger.error(f"Consistency check failed for {doc_id}: {err}")
                fixes.append({"doc_id": doc_id, "error": err})

        return fixes

    def _audit_log(
        self,
        doc_id: str | None,
        action: str,
        operator_id: int,
        reason: str = "",
        details: dict[str, Any] | None = None,
    ):
        """Helper to append structured action logs into the historical audit chain."""
        from sqlalchemy import text

        self.pg.execute(text("""
            INSERT INTO doc_audit_log (doc_id, action, operator_id, reason, details)
            VALUES (:doc_id, :action, :operator_id, :reason, :details)
        """), {
            "doc_id": doc_id,
            "action": action,
            "operator_id": operator_id,
            "reason": reason,
            "details": json.dumps(details, ensure_ascii=False) if details else None,
        })