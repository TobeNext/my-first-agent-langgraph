from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.domain.question_metadata import clean_interview_question_metadata

SCALAR_FIELD_NAMES = {"language", "isActive", "userId", "difficulty", "questionType"}


def migrate_question_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return one metadata record in the current question-bank contract."""
    cleaned = clean_interview_question_metadata(raw_metadata)
    return {
        key: value
        for key, value in cleaned.items()
        if key not in {"mainCategory", "subCategory", "company"}
    }


def build_milvus_scalar_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    migrated = migrate_question_metadata(metadata)
    return {key: migrated[key] for key in SCALAR_FIELD_NAMES if key in migrated}


def migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else record
    migrated_metadata = migrate_question_metadata(metadata)
    if "metadata" not in record:
        return migrated_metadata
    return {
        **record,
        "metadata": migrated_metadata,
        **build_milvus_scalar_fields(migrated_metadata),
    }


def migrate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [migrate_record(record) for record in records]


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [
            record
            for line in text.splitlines()
            if line.strip()
            for record in [json.loads(line)]
            if isinstance(record, dict)
        ]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [record for record in parsed if isinstance(record, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported JSON root in {path}")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate interview question metadata to the current RAG contract."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = migrate_records(read_records(args.input))
    write_records(args.output, records)


if __name__ == "__main__":
    main()
