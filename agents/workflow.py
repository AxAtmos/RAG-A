"""Agent workflow using LlamaIndex AgentWorkflow."""
from __future__ import annotations

from typing import Any

from loguru import logger


class AgentWorkflow:
    """Agent workflow manager - ReAct reasoning mode.

    Currently enables knowledge_search tool only.
    Other tools are stubs for future activation.
    """

    def __init__(self, retriever: Any, llm_fn: Any):
        self.retriever = retriever
        self.llm_fn = llm_fn
        self._tools = self._build_tools()

    def _build_tools(self) -> list:
        """Build available tools list."""
        # Placeholder for LlamaIndex FunctionTool integration
        # When LlamaIndex is fully integrated, these become FunctionTool instances
        return []

    def run(self, query: str, user_context: Any = None) -> str:
        """Run agent workflow for a query.

        Currently just does direct retrieval. Will be replaced with
        LlamaIndex AgentWorkflow when agent features are activated.
        """
        result = self.retriever.retrieve(
            query=query,
            user_filters={},
        )

        if not result["results"]:
            return "未找到相关知识，请尝试换个方式提问。"

        # Build context for LLM
        context_parts = []
        sources = []
        for i, r in enumerate(result["results"], 1):
            meta = r["metadata"]
            source = meta.get("file_name", "unknown")
            context_parts.append(f"[参考资料{i}] {r['text']}")
            sources.append(source)

        context = "\n\n".join(context_parts)
        prompt = f"""基于以下参考资料回答用户问题。如果参考资料不足，请说明。

参考资料：
{context}

用户问题：{query}

请给出详细回答，并在末尾标注引用来源。"""

        answer = self.llm_fn(prompt)

        # Append sources
        source_refs = list(set(sources))
        answer += "\n\n---\n引用来源：" + "、".join(source_refs)

        if result["confidence"] == "low":
            answer = "⚠ 参考资料不足，以下为推测性回答。\n\n" + answer

        return answer
