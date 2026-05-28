"""Pipeline monitoring and debugging component.

Tracks per-request timing, errors, and details at each pipeline stage.
Maintains an in-memory ring buffer of recent traces for the monitoring API.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger


@dataclass
class StageMetrics:
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # "ok" | "error" | "timeout"
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
        }
        if self.error:
            d["error"] = self.error
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class RequestTrace:
    request_id: str
    question: str
    start_time: float
    end_time: float = 0.0
    total_ms: float = 0.0
    stages: list[StageMetrics] = field(default_factory=list)
    slowest_stage: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "question": self.question[:100],
            "total_ms": round(self.total_ms, 1),
            "stages": [s.to_dict() for s in self.stages],
            "slowest_stage": self.slowest_stage,
            "errors": self.errors,
        }


class _StageContext:
    """Context manager returned by PipelineMonitor.start_stage()."""

    def __init__(self, monitor: "PipelineMonitor", name: str, details: dict[str, Any]):
        self.name = name
        self.details = details
        self._monitor = monitor
        self._metrics = StageMetrics(name=name, details=details)

    def __enter__(self) -> "_StageContext":
        self._metrics.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._metrics.end_time = time.monotonic()
        self._metrics.duration_ms = (self._metrics.end_time - self._metrics.start_time) * 1000
        if exc_type is not None:
            self._metrics.status = "error"
            self._metrics.error = str(exc_val)
            self._monitor._record_error(self.name, str(exc_val))
        self._monitor._add_stage(self._metrics)

    def update(self, **kwargs: Any) -> None:
        """Update details while the stage is running."""
        self._metrics.details.update(kwargs)


class PipelineMonitor:
    """Per-request pipeline monitor with stage-level timing and error tracking.

    Usage:
        monitor = PipelineMonitor(question="什么是RAG")
        with monitor.start_stage("L1_query_fusion", variants=4) as stage:
            ... do work ...
            stage.update(deduped=28)
        monitor.finish()
        trace = monitor.get_trace()
    """

    # Class-level ring buffer shared across all instances
    _traces: deque[RequestTrace] = deque(maxlen=100)
    _lock = Lock()
    _consecutive_ollama_timeouts = 0

    def __init__(self, question: str = ""):
        self.request_id = uuid.uuid4().hex[:8]
        self.question = question
        self._trace = RequestTrace(
            request_id=self.request_id,
            question=question,
            start_time=time.monotonic(),
        )
        self._finished = False

    @contextmanager
    def start_stage(self, name: str, **details: Any) -> Any:
        """Context manager that auto-captures timing and exceptions."""
        ctx = _StageContext(self, name, details)
        with ctx:
            yield ctx

    def record(self, stage_name: str, **details: Any) -> None:
        """Manually record a stage result (for stages timed externally)."""
        metrics = StageMetrics(name=stage_name, details=details)
        self._trace.stages.append(metrics)

    def _add_stage(self, metrics: StageMetrics) -> None:
        self._trace.stages.append(metrics)
        tag = f"[req:{self.request_id}]"
        status_icon = "OK" if metrics.status == "ok" else "ERR"
        logger.info(
            f"{tag} [{metrics.name}] {status_icon} {metrics.duration_ms:.0f}ms "
            f"| {metrics.details}"
        )

    def _record_error(self, stage: str, error: str) -> None:
        self._trace.errors.append(f"{stage}: {error}")

    def record_ollama_timeout(self, duration_s: float) -> None:
        """Record an Ollama timeout and check for deadlock pattern."""
        PipelineMonitor._consecutive_ollama_timeouts += 1
        tag = f"[req:{self.request_id}]"
        logger.warning(
            f"{tag} [OLLAMA_TIMEOUT] Response took {duration_s:.1f}s "
            f"(consecutive: {PipelineMonitor._consecutive_ollama_timeouts})"
        )
        if PipelineMonitor._consecutive_ollama_timeouts >= 2:
            logger.error(
                f"{tag} [OLLAMA_DEADLOCK_SUSPECTED] "
                f"{PipelineMonitor._consecutive_ollama_timeouts} consecutive timeouts — "
                f"Ollama runner likely stuck. Run: sudo systemctl restart ollama"
            )
            self._try_log_ollama_cpu()

    def record_ollama_success(self) -> None:
        """Reset consecutive timeout counter on success."""
        PipelineMonitor._consecutive_ollama_timeouts = 0

    def _try_log_ollama_cpu(self) -> None:
        """Try to log Ollama process CPU usage (best-effort)."""
        try:
            import subprocess
            result = subprocess.run(
                ["ps", "-eo", "pid,pcpu,pmem,comm"],
                capture_output=True, text=True, timeout=3,
            )
            for line in result.stdout.splitlines():
                if "ollama" in line.lower():
                    logger.error(f"[OLLAMA_DIAGNOSTIC] {line.strip()}")
        except Exception:
            pass

    def finish(self) -> None:
        """Finalize the trace and add to ring buffer."""
        if self._finished:
            return
        self._finished = True
        self._trace.end_time = time.monotonic()
        self._trace.total_ms = (self._trace.end_time - self._trace.start_time) * 1000

        # Find slowest stage
        if self._trace.stages:
            slowest = max(self._trace.stages, key=lambda s: s.duration_ms)
            self._trace.slowest_stage = slowest.name

        tag = f"[req:{self.request_id}]"
        logger.info(
            f"{tag} Total: {self._trace.total_ms:.0f}ms | "
            f"Slowest: {self._trace.slowest_stage} | "
            f"Errors: {len(self._trace.errors)}"
        )

        with PipelineMonitor._lock:
            PipelineMonitor._traces.append(self._trace)

    def get_trace(self) -> dict[str, Any]:
        return self._trace.to_dict()

    @classmethod
    def get_recent_traces(cls) -> list[dict[str, Any]]:
        with cls._lock:
            return [t.to_dict() for t in cls._traces]

    @classmethod
    def get_trace_by_id(cls, request_id: str) -> dict[str, Any] | None:
        with cls._lock:
            for t in cls._traces:
                if t.request_id == request_id:
                    return t.to_dict()
        return None

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """Compute aggregated stats across all buffered traces."""
        with cls._lock:
            traces = list(cls._traces)

        if not traces:
            return {"total_requests": 0, "message": "No requests recorded yet"}

        total_durations = [t.total_ms for t in traces]
        stage_durations: dict[str, list[float]] = {}
        stage_errors: dict[str, int] = {}

        for t in traces:
            for s in t.stages:
                stage_durations.setdefault(s.name, []).append(s.duration_ms)
                if s.status != "ok":
                    stage_errors[s.name] = stage_errors.get(s.name, 0) + 1

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            idx = min(idx, len(sorted_data) - 1)
            return sorted_data[idx]

        stage_stats = {}
        for name, durations in stage_durations.items():
            stage_stats[name] = {
                "avg_ms": round(sum(durations) / len(durations), 1),
                "p50_ms": round(percentile(durations, 50), 1),
                "p95_ms": round(percentile(durations, 95), 1),
                "min_ms": round(min(durations), 1),
                "max_ms": round(max(durations), 1),
                "count": len(durations),
                "error_count": stage_errors.get(name, 0),
                "error_rate": round(stage_errors.get(name, 0) / len(durations), 3),
            }

        slowest_avg = ""
        if stage_stats:
            slowest_avg = max(stage_stats, key=lambda k: stage_stats[k]["avg_ms"])

        return {
            "window": f"last_{len(traces)}_requests",
            "total_requests": len(traces),
            "avg_total_ms": round(sum(total_durations) / len(total_durations), 1),
            "p50_total_ms": round(percentile(total_durations, 50), 1),
            "p95_total_ms": round(percentile(total_durations, 95), 1),
            "stages": stage_stats,
            "slowest_stage_avg": slowest_avg,
            "total_errors": sum(len(t.errors) for t in traces),
        }
