"""Knowledge search tool for Agent framework."""
from __future__ import annotations

from typing import Any


def knowledge_search_tool(query: str, retriever: Any, user_context: Any = None) -> str:
    """Search the knowledge base and return relevant results.

    This is a FunctionTool stub for LlamaIndex AgentWorkflow.
    """
    filters = {}
    if user_context:
        filters = retriever.qdrant._build_filter({})  # placeholder

    result = retriever.retrieve(query=query, user_filters=filters)

    if not result["results"]:
        return "未找到相关知识。"

    parts = []
    for i, r in enumerate(result["results"], 1):
        meta = r["metadata"]
        source = meta.get("file_name", "unknown")
        summary = meta.get("summary", "")
        parts.append(f"[{i}] (来源: {source}, 相关度: {r['score']:.2f})\n{r['text'][:500]}")

    confidence_note = ""
    if result["confidence"] == "low":
        confidence_note = "\n\n⚠ 参考资料不足，以下为推测性回答。"

    return "\n\n".join(parts) + confidence_note
