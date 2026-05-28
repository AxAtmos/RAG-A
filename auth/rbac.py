from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session


@dataclass
class UserContext:
    """Represents the current user's permissions."""
    user_id: int
    name: str
    rbac_role: str  # super_admin / dept_manager / project_lead / engineer / guest
    department: str
    projects: list[str] = field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        return self.rbac_role in ("super_admin", "dept_manager")

    @property
    def is_super_admin(self) -> bool:
        return self.rbac_role == "super_admin"


# Role hierarchy for permission checks
ROLE_HIERARCHY = {
    "super_admin": 5,
    "dept_manager": 4,
    "project_lead": 3,
    "engineer": 2,
    "guest": 1,
}


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self, pg_session: Session):
        self.pg = pg_session

    def get_user_context(self, user_id: int) -> UserContext | None:
        """Load user context from PostgreSQL."""
        from sqlalchemy import text

        row = self.pg.execute(text(
            "SELECT id, name, rbac_role, department, projects FROM users WHERE id = :uid"
        ), {"uid": user_id}).fetchone()

        if not row:
            return None

        return UserContext(
            user_id=row[0],
            name=row[1],
            rbac_role=row[2] or "engineer",
            department=row[3] or "",
            projects=row[4] or [],
        )

    def get_user_context_by_dify_id(self, dify_user_id: str) -> UserContext | None:
        """Load user context by Dify user ID."""
        from sqlalchemy import text

        row = self.pg.execute(text(
            "SELECT id, name, rbac_role, department, projects FROM users WHERE dify_user_id = :did"
        ), {"did": dify_user_id}).fetchone()

        if not row:
            return None

        return UserContext(
            user_id=row[0],
            name=row[1],
            rbac_role=row[2] or "engineer",
            department=row[3] or "",
            projects=row[4] or [],
        )

    def build_search_filter(self, user: UserContext, include_deprecated: bool = False) -> dict[str, Any]:
        """Build Qdrant search filter based on user permissions.

        Returns filter dict: {"status": ..., "visibility_dept_project": ...}
        """
        filters: dict[str, Any] = {}

        # Status filter
        if include_deprecated and user.is_admin:
            filters["status"] = ["active", "deprecated"]
        else:
            filters["status"] = "active"

        # Visibility filter
        if user.is_super_admin:
            # Can see everything, no visibility filter
            pass
        elif user.rbac_role == "dept_manager":
            # Can see: public + own department
            # We'll use OR logic: visibility=公开 OR (visibility=部门 AND department=user.department) OR (visibility=项目 AND project IN user.projects)
            # Qdrant doesn't support complex OR in a simple dict, so we handle this differently
            filters["_dept_manager"] = {
                "department": user.department,
                "projects": user.projects,
            }
        elif user.rbac_role in ("project_lead", "engineer"):
            # Can see: public + own projects
            filters["_engineer"] = {
                "department": user.department,
                "projects": user.projects,
            }
        else:
            # guest: only public
            filters["visibility"] = "公开"

        return filters

    def can_deprecate(self, user: UserContext, doc_department: str) -> bool:
        """Check if user can deprecate a document."""
        if user.is_super_admin:
            return True
        if user.rbac_role == "dept_manager" and user.department == doc_department:
            return True
        return False

    def can_restore(self, user: UserContext, doc_department: str) -> bool:
        """Check if user can restore a deprecated document."""
        return self.can_deprecate(user, doc_department)

    def can_hard_delete(self, user: UserContext) -> bool:
        """Only super_admin can hard delete."""
        return user.is_super_admin

    def can_view_audit_log(self, user: UserContext) -> bool:
        """Only super_admin can view audit logs."""
        return user.is_super_admin

    def check_role_level(self, user: UserContext, min_role: str) -> bool:
        """Check if user meets minimum role level."""
        user_level = ROLE_HIERARCHY.get(user.rbac_role, 0)
        min_level = ROLE_HIERARCHY.get(min_role, 0)
        return user_level >= min_level
