from __future__ import annotations

import atexit
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings

_checkpoint_context: AbstractContextManager[SqliteSaver] | None = None


def sqlite_checkpoint_path(checkpoint_url: str) -> str:
    if checkpoint_url.startswith("sqlite:///"):
        parsed = urlparse(checkpoint_url)
        raw_path = parsed.path
        if raw_path.startswith("/./"):
            raw_path = raw_path[1:]
        elif len(raw_path) >= 4 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path)
        if parsed.netloc:
            path = Path(f"{parsed.netloc}{parsed.path}")
        return str(path)

    if checkpoint_url.startswith("sqlite://"):
        parsed = urlparse(checkpoint_url)
        return str(Path(parsed.netloc + parsed.path))

    return checkpoint_url


@lru_cache
def get_sqlite_checkpointer() -> SqliteSaver:
    global _checkpoint_context

    path = sqlite_checkpoint_path(get_settings().checkpoint_url)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _checkpoint_context = SqliteSaver.from_conn_string(str(checkpoint_path))
    saver = _checkpoint_context.__enter__()
    atexit.register(close_sqlite_checkpointer)
    return saver


def close_sqlite_checkpointer() -> None:
    global _checkpoint_context

    if _checkpoint_context is not None:
        _checkpoint_context.__exit__(None, None, None)
        _checkpoint_context = None
        get_sqlite_checkpointer.cache_clear()
