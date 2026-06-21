from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from app.config import Settings

logger = logging.getLogger(__name__)


def langsmith_tracing_enabled(settings: Settings) -> bool:
    return settings.langsmith_tracing and bool(_normalized_api_key(settings))


def build_langsmith_metadata(
    *,
    settings: Settings,
    thread_id: str,
) -> dict[str, str]:
    return {
        "thread_id": thread_id,
        "runtime_provider": "python-langgraph",
        "app_env": settings.app_env,
        "model_provider": settings.model_provider,
        "model_name": settings.model_name,
        "otel.trace_id": current_otel_trace_id(),
    }


@contextmanager
def langsmith_graph_context(
    *,
    settings: Settings,
    thread_id: str,
):
    span = trace.get_current_span()
    enabled = langsmith_tracing_enabled(settings)
    if span.is_recording():
        span.set_attribute("langsmith.project", settings.langsmith_project)
        span.set_attribute("langsmith.enabled", enabled)
        span.set_attribute("langsmith.data_mode", settings.langsmith_data_mode)

    if not enabled:
        yield
        return

    metadata = build_langsmith_metadata(settings=settings, thread_id=thread_id)
    try:
        from langsmith import Client
        from langsmith.run_helpers import get_current_run_tree, tracing_context
    except Exception as exc:
        logger.warning("LangSmith tracing is unavailable; continuing without it. error=%s", exc)
        yield
        return

    try:
        api_key = _normalized_api_key(settings)
        if api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        client = Client(api_key=api_key)
        context = tracing_context(
            project_name=settings.langsmith_project,
            metadata=metadata,
            tags=["interview-runtime", "python-langgraph"],
            enabled=True,
            client=client,
        )
    except Exception as exc:
        logger.warning("LangSmith tracing setup failed; continuing without it. error=%s", exc)
        yield
        return

    with context:
        yield
        _record_current_run_id(span, get_current_run_tree)


def current_otel_trace_id() -> str:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return ""
    return f"{span_context.trace_id:032x}"


def _record_current_run_id(span: Any, get_current_run_tree: Any) -> None:
    if not span.is_recording():
        return
    try:
        run_tree = get_current_run_tree()
    except Exception as exc:
        logger.debug("Unable to read current LangSmith run tree. error=%s", exc)
        return
    run_id = getattr(run_tree, "id", None) if run_tree is not None else None
    if run_id:
        span.set_attribute("langsmith.run_id", str(run_id))


def _normalized_api_key(settings: Settings) -> str:
    return (settings.langsmith_api_key or "").strip()
