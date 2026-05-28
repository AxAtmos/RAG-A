"""Initialize PostgreSQL database tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from loguru import logger

from config import settings


SCHEMA_SQL = """
-- Document lifecycle table
CREATE TABLE IF NOT EXISTS doc_lifecycle (
    doc_id          TEXT PRIMARY KEY,
    file_name       TEXT,
    file_type       TEXT,
    department      TEXT DEFAULT '',
    project         TEXT DEFAULT '',
    visibility      TEXT DEFAULT '项目',
    author          TEXT DEFAULT '',
    category        TEXT DEFAULT '其他',
    tags            TEXT[] DEFAULT '{}',
    summary         TEXT DEFAULT '',
    status          TEXT DEFAULT 'active',
    deprecated_reason TEXT DEFAULT '',
    deprecated_at   TIMESTAMP,
    deprecated_by   INTEGER,
    valid_from      DATE,
    valid_until     DATE,
    superseded_by   TEXT,
    total_chunks    INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_status_dept ON doc_lifecycle(status, department);
CREATE INDEX IF NOT EXISTS idx_valid_until ON doc_lifecycle(valid_until);
CREATE INDEX IF NOT EXISTS idx_project ON doc_lifecycle(project);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    dify_user_id  TEXT UNIQUE,
    name          TEXT NOT NULL,
    email         TEXT,
    rbac_role     TEXT DEFAULT 'engineer',
    department    TEXT DEFAULT '',
    projects      TEXT[] DEFAULT '{}',
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Audit log
CREATE TABLE IF NOT EXISTS doc_audit_log (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT,
    action      TEXT NOT NULL,
    operator_id INTEGER REFERENCES users(id),
    reason      TEXT DEFAULT '',
    details     JSONB,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_doc ON doc_audit_log(doc_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON doc_audit_log(action);

-- Expire rules
CREATE TABLE IF NOT EXISTS expire_rules (
    id                SERIAL PRIMARY KEY,
    category          TEXT,
    tags              TEXT[],
    expire_after_days INTEGER,
    department        TEXT,
    enabled           BOOLEAN DEFAULT TRUE,
    created_by        INTEGER REFERENCES users(id),
    created_at        TIMESTAMP DEFAULT NOW()
);

-- Parent documents table (parent-child chunking)
CREATE TABLE IF NOT EXISTS parent_documents (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    full_parent_text TEXT NOT NULL,
    security_level  TEXT DEFAULT 'public',
    chunk_index     TEXT DEFAULT '0'
);

CREATE INDEX IF NOT EXISTS idx_parent_doc_id ON parent_documents(doc_id);

-- Child chunks table (fine-grained segments mapped to Qdrant)
CREATE TABLE IF NOT EXISTS child_chunks (
    id              TEXT PRIMARY KEY,
    parent_id       TEXT REFERENCES parent_documents(id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL,
    child_text      TEXT NOT NULL,
    qdrant_point_id TEXT,
    chunk_index     TEXT DEFAULT '0'
);

CREATE INDEX IF NOT EXISTS idx_child_parent_id ON child_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_child_doc_id ON child_chunks(doc_id);

-- Insert default admin user
INSERT INTO users (name, email, rbac_role, department, projects)
VALUES ('admin', 'admin@company.com', 'super_admin', '', '{}')
ON CONFLICT DO NOTHING;
"""


def init_database():
    """Create all tables and indexes."""
    engine = create_engine(settings.postgres.url)

    with engine.connect() as conn:
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()

    logger.info("Database initialized successfully")
    engine.dispose()


if __name__ == "__main__":
    init_database()
