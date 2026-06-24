"""Export interview questions from Milvus to eval-compatible JSON format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pymilvus import MilvusClient  # noqa: E402


def main() -> int:
    client = MilvusClient(uri="http://localhost:19530")

    # Check collection
    if "interview_questions" not in client.list_collections():
        print("ERROR: interview_questions collection not found in Milvus")
        return 1

    desc = client.describe_collection("interview_questions")
    fields = {f["name"] for f in desc.get("fields", [])}
    print(f"Collection fields: {fields}")

    # Paginate through all records
    all_rows = []
    offset = 0
    page_size = 100
    has_scalar = {"role", "skillArea", "difficulty"}.issubset(fields)

    while True:
        output_fields = ["id", "metadata"]
        if has_scalar:
            output_fields.extend(["role", "skillArea", "difficulty"])

        rows = client.query(
            collection_name="interview_questions",
            filter="id != ''",
            output_fields=output_fields,
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        all_rows.extend(rows)
        offset += page_size
        print(f"  Fetched {len(all_rows)} records...", end="\r")

    print(f"\nTotal records: {len(all_rows)}")

    # Transform to question bank format
    questions = []
    for row in all_rows:
        meta = row.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}

        qid = str(row.get("id") or meta.get("id") or f"milvus-{len(questions)}")
        text = str(
            meta.get("question") or meta.get("text") or row.get("text") or ""
        ).strip()
        if not text:
            continue

        role = str(row.get("role") or meta.get("role") or "professional-skills")
        round_type = role if role in {"professional-skills", "project-experience"} else "professional-skills"

        raw_skills = row.get("skillArea") or meta.get("skillArea") or meta.get("skills") or []
        if isinstance(raw_skills, str):
            raw_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        skill_areas = [str(s) for s in raw_skills] if raw_skills else ["general"]

        questions.append(
            {
                "id": qid,
                "text": text,
                "skill_areas": skill_areas,
                "round_type": round_type,
            }
        )

    # Save
    out_path = REPO_ROOT / "tests" / "evals" / "datasets" / "interview_question_bank_milvus.json"
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(questions)} questions to {out_path}")

    # Show sample
    for q in questions[:3]:
        print(f"  [{q['id']}] {q['text'][:80]}...  skills={q['skill_areas'][:3]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
