"""AI Interviewer tool stub - guided knowledge entry."""
from __future__ import annotations

from typing import Any


def ai_interviewer_tool(knowledge_type: str, llm_fn: Any = None) -> str:
    """AI-guided knowledge entry: generate questions based on knowledge type.

    Stub: to be implemented with LlamaIndex FunctionTool.
    Supported types: 设计决策, Bug分析, 方案评审, ...
    """
    return f"[STUB] AI Interviewer for '{knowledge_type}' - not yet implemented"
