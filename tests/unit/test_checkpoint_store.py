from pathlib import Path

from app.integrations.checkpoint_store import sqlite_checkpoint_path


def test_sqlite_checkpoint_path_accepts_plain_file_path(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.db"

    assert sqlite_checkpoint_path(str(path)) == str(path)


def test_sqlite_checkpoint_path_normalizes_sqlite_url() -> None:
    assert sqlite_checkpoint_path("sqlite:///./checkpoints.db") == "checkpoints.db"
