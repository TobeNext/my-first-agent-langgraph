import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent / "datasets"

DATASET_REQUIRED_FIELDS = {
    "interview_cases.jsonl": {
        "case_id",
        "redaction_version",
        "source_type",
        "source_thread_id_hash",
        "resume_markdown",
        "job_description_markdown",
        "settings",
        "turns",
        "expected_stage_path",
        "expected_required_skills",
        "must_not_claim",
        "rubric",
    },
    "rag_cases.jsonl": {
        "case_id",
        "redaction_version",
        "source_type",
        "source_thread_id_hash",
        "query",
        "round_type",
        "resume_signals",
        "jd_signals",
        "expected_question_ids",
        "acceptable_skill_areas",
        "negative_question_ids",
    },
    "safety_cases.jsonl": {
        "case_id",
        "redaction_version",
        "source_type",
        "source_thread_id_hash",
        "input_kind",
        "payload",
        "forbidden_patterns",
        "expected_safe_behavior",
    },
}

DATASET_MIN_COUNTS = {
    "interview_cases.jsonl": 3,
    "rag_cases.jsonl": 5,
    "safety_cases.jsonl": 2,
}

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d .-]{8,}\d)(?!\d)"),
    "long_numeric_id": re.compile(r"\b\d{15,18}\b"),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        assert line.strip(), f"{path.name}:{line_number} is blank"
        payload = json.loads(line)
        assert isinstance(payload, dict), f"{path.name}:{line_number} must be a JSON object"
        rows.append(payload)
    return rows


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _assert_common_fields(row: dict[str, Any], *, dataset_name: str) -> None:
    assert row["case_id"], f"{dataset_name} case_id is required"
    assert row["redaction_version"], f"{row['case_id']} redaction_version is required"
    assert row["source_type"] in {"synthetic", "redacted-production", "manual"}
    assert row["source_thread_id_hash"].startswith("sha256:")
    assert "@" not in row["source_thread_id_hash"]


def test_eval_jsonl_files_exist_and_meet_minimum_counts() -> None:
    for dataset_name, minimum_count in DATASET_MIN_COUNTS.items():
        rows = _load_jsonl(DATASET_DIR / dataset_name)

        assert len(rows) >= minimum_count


def test_eval_dataset_rows_match_required_schema() -> None:
    for dataset_name, required_fields in DATASET_REQUIRED_FIELDS.items():
        rows = _load_jsonl(DATASET_DIR / dataset_name)
        for row in rows:
            case_id = row.get("case_id")
            assert required_fields <= row.keys(), f"{dataset_name}:{case_id} missing fields"
            _assert_common_fields(row, dataset_name=dataset_name)


def test_interview_dataset_field_shapes_are_valid() -> None:
    rows = _load_jsonl(DATASET_DIR / "interview_cases.jsonl")
    for row in rows:
        assert isinstance(row["settings"], dict)
        assert isinstance(row["turns"], list) and row["turns"]
        assert isinstance(row["expected_stage_path"], list) and row["expected_stage_path"]
        assert isinstance(row["expected_required_skills"], list)
        assert isinstance(row["must_not_claim"], list)
        assert isinstance(row["rubric"], dict)


def test_rag_dataset_field_shapes_are_valid() -> None:
    rows = _load_jsonl(DATASET_DIR / "rag_cases.jsonl")
    for row in rows:
        assert row["round_type"] in {"professional-skills", "project-experience"}
        assert isinstance(row["resume_signals"], list)
        assert isinstance(row["jd_signals"], list)
        assert isinstance(row["expected_question_ids"], list) and row["expected_question_ids"]
        assert isinstance(row["acceptable_skill_areas"], list)
        assert isinstance(row["negative_question_ids"], list)


def test_safety_dataset_field_shapes_are_valid() -> None:
    rows = _load_jsonl(DATASET_DIR / "safety_cases.jsonl")
    for row in rows:
        assert isinstance(row["payload"], dict)
        assert isinstance(row["forbidden_patterns"], list) and row["forbidden_patterns"]
        assert isinstance(row["expected_safe_behavior"], str) and row["expected_safe_behavior"]


def test_eval_datasets_do_not_contain_obvious_pii() -> None:
    for dataset_name in DATASET_REQUIRED_FIELDS:
        rows = _load_jsonl(DATASET_DIR / dataset_name)
        for row in rows:
            for value in _iter_strings(row):
                for pattern_name, pattern in PII_PATTERNS.items():
                    assert not pattern.search(value), (
                        f"{dataset_name}:{row['case_id']} contains {pattern_name}-like text"
                    )
