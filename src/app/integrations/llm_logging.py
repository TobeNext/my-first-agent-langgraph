from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("app.llm")


def log_llm_input(
    *,
    thread_id: str,
    operation: str,
    prompt: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    _log(
        {
            "event": "llm.input",
            "threadId": thread_id,
            "operation": operation,
            "prompt": _serialize_value(prompt),
            "metadata": metadata or {},
        }
    )


def log_llm_output(
    *,
    thread_id: str,
    operation: str,
    output: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    _log(
        {
            "event": "llm.output",
            "threadId": thread_id,
            "operation": operation,
            "output": _serialize_value(output),
            "metadata": metadata or {},
        }
    )


def log_llm_error(
    *,
    thread_id: str,
    operation: str,
    error: BaseException,
    metadata: dict[str, Any] | None = None,
) -> None:
    _log(
        {
            "event": "llm.error",
            "threadId": thread_id,
            "operation": operation,
            "errorType": error.__class__.__name__,
            "error": str(error),
            "metadata": metadata or {},
        },
    )


def _log(payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def _serialize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}

    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": value.__class__.__name__,
            "content": _serialize_value(content),
        }
    return repr(value)
