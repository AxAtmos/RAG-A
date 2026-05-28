"""Sync Dify users to local PostgreSQL users table."""
from __future__ import annotations

import httpx
from loguru import logger
from sqlalchemy.orm import Session

from config import settings


ROLE_MAPPING = {
    "owner": "super_admin",
    "admin": "dept_manager",
    "member": "engineer",
}


def sync_dify_users(pg_session: Session):
    """Fetch users from Dify API and sync to local PostgreSQL."""
    if not settings.ollama:  # placeholder - check dify config
        return

    try:
        resp = httpx.get(
            f"{settings.ollama.base_url}/workspaces/current/members",  # placeholder
            headers={"Authorization": f"Bearer rag-enterprise"},
            timeout=10,
        )
        resp.raise_for_status()
        members = resp.json().get("data", [])
    except Exception as e:
        logger.warning(f"Dify user sync failed: {e}")
        return

    from sqlalchemy import text

    for m in members:
        dify_id = m.get("id", "")
        name = m.get("name", "")
        email = m.get("email", "")
        role = ROLE_MAPPING.get(m.get("role", ""), "engineer")

        pg_session.execute(text("""
            INSERT INTO users (dify_user_id, name, email, rbac_role)
            VALUES (:did, :name, :email, :role)
            ON CONFLICT (dify_user_id) DO UPDATE SET name = :name, email = :email
        """), {"did": dify_id, "name": name, "email": email, "role": role})

    pg_session.commit()
    logger.info(f"Synced {len(members)} users from Dify")
