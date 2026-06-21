from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from app.config import get_settings
from app.schemas.interview_report import (
    InterviewReportItemRecord,
    InterviewReportReadReceipt,
    InterviewReportRecord,
    InterviewReportWrite,
    InterviewUserMemoryProfile,
    InterviewUserMemoryRecord,
    InterviewUserMemoryWrite,
)


def sqlite_report_database_path(database_url: str) -> str:
    if database_url.startswith("sqlite:///"):
        parsed = urlparse(database_url)
        raw_path = parsed.path
        if raw_path.startswith("/./"):
            raw_path = raw_path[1:]
        elif len(raw_path) >= 4 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path)
        if parsed.netloc:
            path = Path(f"{parsed.netloc}{parsed.path}")
        return str(path)

    if database_url.startswith("sqlite://"):
        parsed = urlparse(database_url)
        return str(Path(parsed.netloc + parsed.path))

    return database_url


class InterviewReportRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_path = sqlite_report_database_path(
            database_url or get_settings().report_database_url
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS interview_reports (
                  id TEXT PRIMARY KEY,
                  interview_id TEXT NOT NULL UNIQUE,
                  thread_id TEXT NOT NULL,
                  target_role TEXT NOT NULL,
                  response_language TEXT NOT NULL,
                  status TEXT NOT NULL,
                  overall_score REAL,
                  markdown TEXT NOT NULL,
                  structured_json TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  model_name TEXT NOT NULL,
                  source_evaluation_manifest_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS interview_report_items (
                  id TEXT PRIMARY KEY,
                  report_id TEXT NOT NULL,
                  interview_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  attempt_id TEXT NOT NULL,
                  node_id TEXT NOT NULL,
                  round_id TEXT NOT NULL,
                  round_type TEXT NOT NULL,
                  target_type TEXT NOT NULL,
                  question TEXT NOT NULL,
                  candidate_answer TEXT NOT NULL,
                  score REAL NOT NULL,
                  comment TEXT NOT NULL,
                  missing_points_json TEXT NOT NULL,
                  improvement_advice_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(report_id) REFERENCES interview_reports(id)
                );

                CREATE TABLE IF NOT EXISTS interview_report_reads (
                  id TEXT PRIMARY KEY,
                  interview_id TEXT NOT NULL,
                  thread_id TEXT NOT NULL,
                  read_at TEXT NOT NULL,
                  UNIQUE(interview_id, thread_id)
                );

                CREATE TABLE IF NOT EXISTS interview_user_memories (
                  id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  source_interview_id TEXT NOT NULL,
                  source_thread_id TEXT NOT NULL,
                  target_role TEXT NOT NULL,
                  overall_score REAL,
                  weakness_summary_json TEXT NOT NULL,
                  missing_points_json TEXT NOT NULL,
                  improvement_advice_json TEXT NOT NULL,
                  reinforcement_question_hints_json TEXT NOT NULL,
                  report_markdown_excerpt TEXT NOT NULL,
                  embedding_text TEXT NOT NULL,
                  embedding_json TEXT,
                  source_report_completed_at TEXT NOT NULL,
                  summary_generated_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(user_id, source_interview_id)
                );

                CREATE TABLE IF NOT EXISTS interview_user_memory_profiles (
                  user_id TEXT PRIMARY KEY,
                  stable_weaknesses_json TEXT NOT NULL,
                  improved_areas_json TEXT NOT NULL,
                  recurring_mistakes_json TEXT NOT NULL,
                  weakness_counters_json TEXT NOT NULL,
                  last_memory_ids_json TEXT NOT NULL,
                  summary_count INTEGER NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    def write_report(self, report: InterviewReportWrite) -> InterviewReportRecord:
        existing = self.get_report_by_interview_id(report.interview_id)
        if existing and existing.status == "succeeded" and existing.markdown:
            return existing

        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM interview_report_items WHERE interview_id = ?",
                (report.interview_id,),
            )
            conn.execute(
                """
                INSERT INTO interview_reports (
                  id,
                  interview_id,
                  thread_id,
                  target_role,
                  response_language,
                  status,
                  overall_score,
                  markdown,
                  structured_json,
                  prompt_version,
                  model_name,
                  source_evaluation_manifest_json,
                  created_at,
                  updated_at,
                  completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                  id = excluded.id,
                  thread_id = excluded.thread_id,
                  target_role = excluded.target_role,
                  response_language = excluded.response_language,
                  status = excluded.status,
                  overall_score = excluded.overall_score,
                  markdown = excluded.markdown,
                  structured_json = excluded.structured_json,
                  prompt_version = excluded.prompt_version,
                  model_name = excluded.model_name,
                  source_evaluation_manifest_json = excluded.source_evaluation_manifest_json,
                  updated_at = excluded.updated_at,
                  completed_at = excluded.completed_at
                """,
                (
                    report.id,
                    report.interview_id,
                    report.thread_id,
                    report.target_role,
                    report.response_language,
                    report.status,
                    report.overall_score,
                    report.markdown,
                    report.structured_json,
                    report.prompt_version,
                    report.model_name,
                    report.source_evaluation_manifest_json,
                    report.created_at,
                    report.updated_at,
                    report.completed_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO interview_report_items (
                  id,
                  report_id,
                  interview_id,
                  task_id,
                  attempt_id,
                  node_id,
                  round_id,
                  round_type,
                  target_type,
                  question,
                  candidate_answer,
                  score,
                  comment,
                  missing_points_json,
                  improvement_advice_json,
                  created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.id,
                        report.id,
                        report.interview_id,
                        item.task_id,
                        item.attempt_id,
                        item.node_id,
                        item.round_id,
                        item.round_type,
                        item.target_type,
                        item.question,
                        item.candidate_answer,
                        item.score,
                        item.comment,
                        item.missing_points_json,
                        item.improvement_advice_json,
                        report.created_at,
                    )
                    for item in report.items
                ],
            )
            conn.commit()

        stored = self.get_report_by_interview_id(report.interview_id)
        if not stored:
            raise RuntimeError(f"Interview report {report.interview_id} was not persisted.")
        return stored

    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM interview_reports WHERE interview_id = ?",
                (interview_id,),
            ).fetchone()
        return _report_record(row) if row else None

    def get_markdown_by_interview_id(self, interview_id: str) -> str | None:
        report = self.get_report_by_interview_id(interview_id)
        return report.markdown if report else None

    def list_items(self, report_id: str) -> list[InterviewReportItemRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM interview_report_items
                WHERE report_id = ?
                ORDER BY rowid ASC
                """,
                (report_id,),
            ).fetchall()
        return [_item_record(row) for row in rows]

    def mark_read(
        self,
        interview_id: str,
        thread_id: str,
        read_at: str,
        receipt_id: str | None = None,
    ) -> InterviewReportReadReceipt:
        next_id = receipt_id or f"report-read-{uuid4()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interview_report_reads (id, interview_id, thread_id, read_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(interview_id, thread_id) DO UPDATE SET
                  read_at = excluded.read_at
                """,
                (next_id, interview_id, thread_id, read_at),
            )

        receipt = self.get_read_receipt(interview_id, thread_id)
        if not receipt:
            raise RuntimeError(f"Interview report read receipt {interview_id}/{thread_id} failed.")
        return receipt

    def get_read_receipt(
        self,
        interview_id: str,
        thread_id: str,
    ) -> InterviewReportReadReceipt | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interview_report_reads
                WHERE interview_id = ? AND thread_id = ?
                """,
                (interview_id, thread_id),
            ).fetchone()
        return _read_receipt(row) if row else None

    def write_user_memory(self, memory: InterviewUserMemoryWrite) -> InterviewUserMemoryRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interview_user_memories (
                  id,
                  user_id,
                  source_interview_id,
                  source_thread_id,
                  target_role,
                  overall_score,
                  weakness_summary_json,
                  missing_points_json,
                  improvement_advice_json,
                  reinforcement_question_hints_json,
                  report_markdown_excerpt,
                  embedding_text,
                  embedding_json,
                  source_report_completed_at,
                  summary_generated_at,
                  created_at,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, source_interview_id) DO UPDATE SET
                  id = excluded.id,
                  source_thread_id = excluded.source_thread_id,
                  target_role = excluded.target_role,
                  overall_score = excluded.overall_score,
                  weakness_summary_json = excluded.weakness_summary_json,
                  missing_points_json = excluded.missing_points_json,
                  improvement_advice_json = excluded.improvement_advice_json,
                  reinforcement_question_hints_json =
                    excluded.reinforcement_question_hints_json,
                  report_markdown_excerpt = excluded.report_markdown_excerpt,
                  embedding_text = excluded.embedding_text,
                  embedding_json = excluded.embedding_json,
                  source_report_completed_at = excluded.source_report_completed_at,
                  summary_generated_at = excluded.summary_generated_at,
                  updated_at = excluded.updated_at
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.source_interview_id,
                    memory.source_thread_id,
                    memory.target_role,
                    memory.overall_score,
                    memory.weakness_summary_json,
                    memory.missing_points_json,
                    memory.improvement_advice_json,
                    memory.reinforcement_question_hints_json,
                    memory.report_markdown_excerpt,
                    memory.embedding_text,
                    memory.embedding_json,
                    memory.source_report_completed_at,
                    memory.summary_generated_at,
                    memory.created_at,
                    memory.updated_at,
                ),
            )

        stored = self.get_user_memory(memory.user_id, memory.source_interview_id)
        if not stored:
            raise RuntimeError(
                f"Interview user memory {memory.user_id}/{memory.source_interview_id} failed."
            )
        return stored

    def get_user_memory(
        self,
        user_id: str,
        source_interview_id: str,
    ) -> InterviewUserMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interview_user_memories
                WHERE user_id = ? AND source_interview_id = ?
                """,
                (user_id, source_interview_id),
            ).fetchone()
        return _user_memory_record(row) if row else None

    def list_user_memories(self, user_id: str) -> list[InterviewUserMemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM interview_user_memories
                WHERE user_id = ?
                ORDER BY summary_generated_at DESC, source_report_completed_at DESC, created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [_user_memory_record(row) for row in rows]

    def delete_oldest_user_memory(self, user_id: str) -> InterviewUserMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interview_user_memories
                WHERE user_id = ?
                ORDER BY summary_generated_at ASC, source_report_completed_at ASC, created_at ASC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM interview_user_memories WHERE id = ?", (row["id"],))
        return _user_memory_record(row)

    def upsert_user_memory_profile(
        self,
        profile: InterviewUserMemoryProfile,
    ) -> InterviewUserMemoryProfile:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interview_user_memory_profiles (
                  user_id,
                  stable_weaknesses_json,
                  improved_areas_json,
                  recurring_mistakes_json,
                  weakness_counters_json,
                  last_memory_ids_json,
                  summary_count,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  stable_weaknesses_json = excluded.stable_weaknesses_json,
                  improved_areas_json = excluded.improved_areas_json,
                  recurring_mistakes_json = excluded.recurring_mistakes_json,
                  weakness_counters_json = excluded.weakness_counters_json,
                  last_memory_ids_json = excluded.last_memory_ids_json,
                  summary_count = excluded.summary_count,
                  updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.stable_weaknesses_json,
                    profile.improved_areas_json,
                    profile.recurring_mistakes_json,
                    profile.weakness_counters_json,
                    profile.last_memory_ids_json,
                    profile.summary_count,
                    profile.updated_at,
                ),
            )

        stored = self.get_user_memory_profile(profile.user_id)
        if not stored:
            raise RuntimeError(f"Interview user memory profile {profile.user_id} failed.")
        return stored

    def get_user_memory_profile(self, user_id: str) -> InterviewUserMemoryProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interview_user_memory_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return _user_memory_profile(row) if row else None

    def write_user_memory_with_profile(
        self,
        *,
        memory: InterviewUserMemoryWrite,
        profile: InterviewUserMemoryProfile,
        max_memory_count: int,
    ) -> InterviewUserMemoryRecord:
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO interview_user_memory_profiles (
                  user_id,
                  stable_weaknesses_json,
                  improved_areas_json,
                  recurring_mistakes_json,
                  weakness_counters_json,
                  last_memory_ids_json,
                  summary_count,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  stable_weaknesses_json = excluded.stable_weaknesses_json,
                  improved_areas_json = excluded.improved_areas_json,
                  recurring_mistakes_json = excluded.recurring_mistakes_json,
                  weakness_counters_json = excluded.weakness_counters_json,
                  last_memory_ids_json = excluded.last_memory_ids_json,
                  summary_count = excluded.summary_count,
                  updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.stable_weaknesses_json,
                    profile.improved_areas_json,
                    profile.recurring_mistakes_json,
                    profile.weakness_counters_json,
                    profile.last_memory_ids_json,
                    profile.summary_count,
                    profile.updated_at,
                ),
            )
            existing = conn.execute(
                """
                SELECT id FROM interview_user_memories
                WHERE user_id = ? AND source_interview_id = ?
                """,
                (memory.user_id, memory.source_interview_id),
            ).fetchone()
            if not existing:
                count = conn.execute(
                    "SELECT COUNT(*) FROM interview_user_memories WHERE user_id = ?",
                    (memory.user_id,),
                ).fetchone()[0]
                if count >= max_memory_count:
                    oldest = conn.execute(
                        """
                        SELECT id FROM interview_user_memories
                        WHERE user_id = ?
                        ORDER BY summary_generated_at ASC,
                          source_report_completed_at ASC,
                          created_at ASC
                        LIMIT 1
                        """,
                        (memory.user_id,),
                    ).fetchone()
                    if oldest:
                        conn.execute(
                            "DELETE FROM interview_user_memories WHERE id = ?",
                            (oldest["id"],),
                        )
            conn.execute(
                """
                INSERT INTO interview_user_memories (
                  id,
                  user_id,
                  source_interview_id,
                  source_thread_id,
                  target_role,
                  overall_score,
                  weakness_summary_json,
                  missing_points_json,
                  improvement_advice_json,
                  reinforcement_question_hints_json,
                  report_markdown_excerpt,
                  embedding_text,
                  embedding_json,
                  source_report_completed_at,
                  summary_generated_at,
                  created_at,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, source_interview_id) DO UPDATE SET
                  id = excluded.id,
                  source_thread_id = excluded.source_thread_id,
                  target_role = excluded.target_role,
                  overall_score = excluded.overall_score,
                  weakness_summary_json = excluded.weakness_summary_json,
                  missing_points_json = excluded.missing_points_json,
                  improvement_advice_json = excluded.improvement_advice_json,
                  reinforcement_question_hints_json =
                    excluded.reinforcement_question_hints_json,
                  report_markdown_excerpt = excluded.report_markdown_excerpt,
                  embedding_text = excluded.embedding_text,
                  embedding_json = excluded.embedding_json,
                  source_report_completed_at = excluded.source_report_completed_at,
                  summary_generated_at = excluded.summary_generated_at,
                  updated_at = excluded.updated_at
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.source_interview_id,
                    memory.source_thread_id,
                    memory.target_role,
                    memory.overall_score,
                    memory.weakness_summary_json,
                    memory.missing_points_json,
                    memory.improvement_advice_json,
                    memory.reinforcement_question_hints_json,
                    memory.report_markdown_excerpt,
                    memory.embedding_text,
                    memory.embedding_json,
                    memory.source_report_completed_at,
                    memory.summary_generated_at,
                    memory.created_at,
                    memory.updated_at,
                ),
            )
            conn.commit()

        stored = self.get_user_memory(memory.user_id, memory.source_interview_id)
        if not stored:
            raise RuntimeError(
                f"Interview user memory {memory.user_id}/{memory.source_interview_id} failed."
            )
        return stored

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _report_record(row: sqlite3.Row) -> InterviewReportRecord:
    return InterviewReportRecord(
        id=row["id"],
        interview_id=row["interview_id"],
        thread_id=row["thread_id"],
        target_role=row["target_role"],
        response_language=row["response_language"],
        status=row["status"],
        overall_score=row["overall_score"],
        markdown=row["markdown"],
        structured_json=row["structured_json"],
        prompt_version=row["prompt_version"],
        model_name=row["model_name"],
        source_evaluation_manifest_json=row["source_evaluation_manifest_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _item_record(row: sqlite3.Row) -> InterviewReportItemRecord:
    return InterviewReportItemRecord(
        id=row["id"],
        report_id=row["report_id"],
        interview_id=row["interview_id"],
        task_id=row["task_id"],
        attempt_id=row["attempt_id"],
        node_id=row["node_id"],
        round_id=row["round_id"],
        round_type=row["round_type"],
        target_type=row["target_type"],
        question=row["question"],
        candidate_answer=row["candidate_answer"],
        score=row["score"],
        comment=row["comment"],
        missing_points_json=row["missing_points_json"],
        improvement_advice_json=row["improvement_advice_json"],
        created_at=row["created_at"],
    )


def _read_receipt(row: sqlite3.Row) -> InterviewReportReadReceipt:
    return InterviewReportReadReceipt(
        id=row["id"],
        interview_id=row["interview_id"],
        thread_id=row["thread_id"],
        read_at=row["read_at"],
    )


def _user_memory_record(row: sqlite3.Row) -> InterviewUserMemoryRecord:
    return InterviewUserMemoryRecord(
        id=row["id"],
        user_id=row["user_id"],
        source_interview_id=row["source_interview_id"],
        source_thread_id=row["source_thread_id"],
        target_role=row["target_role"],
        overall_score=row["overall_score"],
        weakness_summary_json=row["weakness_summary_json"],
        missing_points_json=row["missing_points_json"],
        improvement_advice_json=row["improvement_advice_json"],
        reinforcement_question_hints_json=row["reinforcement_question_hints_json"],
        report_markdown_excerpt=row["report_markdown_excerpt"],
        embedding_text=row["embedding_text"],
        embedding_json=row["embedding_json"],
        source_report_completed_at=row["source_report_completed_at"],
        summary_generated_at=row["summary_generated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _user_memory_profile(row: sqlite3.Row) -> InterviewUserMemoryProfile:
    return InterviewUserMemoryProfile(
        user_id=row["user_id"],
        stable_weaknesses_json=row["stable_weaknesses_json"],
        improved_areas_json=row["improved_areas_json"],
        recurring_mistakes_json=row["recurring_mistakes_json"],
        weakness_counters_json=row["weakness_counters_json"],
        last_memory_ids_json=row["last_memory_ids_json"],
        summary_count=row["summary_count"],
        updated_at=row["updated_at"],
    )
