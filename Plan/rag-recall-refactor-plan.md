# RAG Recall Refactor Plan

## Goal

Refactor the interview question RAG recall pipeline to use a cleaner metadata contract,
hybrid dense/BM25 recall, RRF fusion, duplicate veto, simple rerank scoring, and weighted
random final selection.

Target pipeline:

```text
query
  -> query rewrite / expansion
  -> dense search top25
  -> BM25 search top25
  -> metadata filter
       question/answer/text must exist
       isActive == true
       language == "zh"
       userId == "global"
  -> RRF merge top30
  -> duplicate veto
       exact duplicate or semantically near-duplicate questions are rejected
  -> rerank top5
       final_score = 0.9 * rrf_score_norm + 0.1 * question_type_score
  -> weighted random 1
       weight = final_score
```

Before starting each numbered implementation step that edits
`../my-first-agent-langgraph`, reload and recheck
`../my-first-agent/.github/instructions/langgraph-architecture.instructions.md`.

## Final Metadata Contract

Each question metadata record should keep:

```json
{
  "question": "...",
  "answer": "...",
  "text": "...",
  "questionType": "system_design",
  "role": "AI Agent Engineer",
  "difficulty": 8,
  "source": "ai-agent-summary",
  "tags": ["memory"],
  "language": "zh",
  "isActive": true,
  "userId": "global"
}
```

Remove these legacy fields from stored metadata:

```text
mainCategory
subCategory
company
```

Use these default and normalization rules:

```text
userId = "global"
language = "zh"
isActive = true
questionType = snake_case
difficulty: easy -> 3, medium -> 6, hard -> 8, unknown -> 5
```

Canonical `questionType` values:

```text
system_design
technical
knowledge_check
experience_probe
case_analysis
culture_fit
```

Unknown question types should fall back to `knowledge_check`.

## Scoring Rules

RRF:

```text
rrf_score(q) = sum(1 / (60 + rank_i))
```

Normalize RRF within the fused top30:

```text
rrf_score_norm = (rrf_score - min_rrf) / (max_rrf - min_rrf)
```

If all RRF scores are equal:

```text
rrf_score_norm = 1.0
```

Question type score:

```text
system_design    1.00
technical        0.90
experience_probe 0.85
case_analysis    0.85
knowledge_check  0.70
culture_fit      0.30
unknown          0.50
```

Final rerank score:

```text
final_score = 0.9 * rrf_score_norm + 0.1 * question_type_score
```

Duplicate veto:

```text
exact duplicate: normalized_question is identical
near duplicate: token overlap >= 0.82
```

Use the existing Chinese bigram plus English token style tokenizer first. Embedding
cosine similarity can be added later if text overlap is not enough.

## Step 1: Metadata Cleaning And Schema Alignment

Status: completed on 2026-06-24.

Expected change size: about 120-180 lines.

Implementation:

- Update `src/app/domain/question_metadata.py`.
- Stop outputting `mainCategory`, `subCategory`, and `company`.
- Add default `language="zh"`.
- Add default `isActive=True`.
- Add default `userId="global"`.
- Normalize `questionType` to snake_case.
- Convert `difficulty` to int:
  - `easy -> 3`
  - `medium -> 6`
  - `hard -> 8`
  - `unknown -> 5`
- Update `src/app/schemas/interview_state.py` so `InterviewQuestionCandidate.difficulty`
  can carry the new int difficulty. Prefer normalizing at ingestion so domain code
  sees an int.

Verification:

- Update or add `tests/unit/test_question_metadata.py`.
- Verify legacy metadata input produces:
  - no `mainCategory`, `subCategory`, or `company`
  - `questionType == "system_design"`
  - `difficulty == 8`
  - `language == "zh"`
  - `isActive is True`
  - `userId == "global"`

Commands:

```powershell
cd ..\my-first-agent-langgraph
python -m pytest tests/unit/test_question_metadata.py
python -m ruff check src/app/domain/question_metadata.py tests/unit/test_question_metadata.py
```

Completed verification:

```text
python -m pytest tests/unit/test_question_metadata.py
7 passed

python -m ruff check src/app/domain/question_metadata.py tests/unit/test_question_metadata.py
All checks passed
```

## Step 2: Milvus Store Field Support And Metadata Filtering

Status: completed on 2026-06-24.

Expected change size: about 150-220 lines.

Implementation:

- Update `src/app/integrations/milvus_store.py`.
- Read new scalar fields when collection schema exposes them:
  - `language`
  - `isActive`
  - `userId`
- Push down scalar filtering when possible:

```text
isActive == true and language == "zh" and userId == "global"
```

- Keep fallback filtering for metadata-only legacy collections.
- Filter bad data after retrieval:
  - `question` must be non-empty.
  - `answer` must be non-empty.
  - `text` must be non-empty.
- Normalize legacy `questionType` and `difficulty` during candidate construction.
- Keep `role` as a stored field, but do not use it as the round-type filter.

Verification:

- Update `tests/unit/test_milvus_store.py`.
- Fake Milvus client with new scalar fields:
  - assert scalar filter is passed to `client.search`.
  - assert output fields include new scalar fields.
- Fake metadata-only collection:
  - assert application-level filtering removes missing `answer/text`, inactive, non-zh,
    and non-global records.
- Assert candidates have `questionType == "system_design"` and int `difficulty`.

Commands:

```powershell
python -m pytest tests/unit/test_milvus_store.py
python -m ruff check src/app/integrations/milvus_store.py tests/unit/test_milvus_store.py
```

Completed verification:

```text
python -m pytest tests/unit/test_milvus_store.py
5 passed

python -m ruff check src/app/integrations/milvus_store.py tests/unit/test_milvus_store.py
All checks passed
```

## Step 3: RRF Top30 And Simple Rerank Formula

Status: completed on 2026-06-24.

Expected change size: about 160-220 lines.

Implementation:

- Update `src/app/domain/question_retriever.py`.
- Set dense recall to top25.
- Set BM25 recall to top25.
- Fuse ranked lists with RRF.
- Truncate fused list to top30 before rerank.
- Replace the old metadata score formula with:

```text
final_score = 0.9 * rrf_score_norm + 0.1 * question_type_score
```

- Remove skill, job, difficulty, and novelty from final score.
- Keep trace output, but change `scoreBreakdown` to:

```json
{
  "rrf": 0.93,
  "questionType": 1.0
}
```

Verification:

- Update `tests/unit/test_question_retriever.py`.
- Dense and BM25 both hitting the same question should raise that question's RRF rank.
- With close RRF scores, `system_design` should beat `culture_fit`.
- Difficulty should not affect final rerank order.
- Skill/job metadata should not affect final rerank order.

Commands:

```powershell
python -m pytest tests/unit/test_question_retriever.py
python -m ruff check src/app/domain/question_retriever.py tests/unit/test_question_retriever.py
```

Completed verification:

```text
python -m pytest tests/unit/test_question_retriever.py
12 passed

python -m ruff check src/app/domain/question_retriever.py tests/unit/test_question_retriever.py
All checks passed
```

## Step 4: Duplicate Veto

Status: completed on 2026-06-24.

Expected change size: about 120-180 lines.

Implementation:

- Update `src/app/domain/question_retriever.py`.
- Add or refactor duplicate detection:
  - exact normalized question match is vetoed.
  - token overlap `>= 0.82` is vetoed.
- Apply duplicate veto after RRF top30 and before final top5 selection.
- Sort candidates by final score, then keep the first non-duplicate candidates.
- Mark duplicate candidates in traces:

```json
{
  "isDuplicate": true,
  "filterReason": "duplicate-veto"
}
```

Verification:

- Add or update duplicate tests in `tests/unit/test_question_retriever.py`.
- Exact duplicate questions keep only the highest-scoring candidate.
- Near duplicate questions such as:
  - `Claude Code 的记忆架构是什么？`
  - `Claude Code 记忆架构与上下文有什么区别？`
  should keep one when overlap passes the threshold.
- When a duplicate is vetoed, the next non-duplicate candidate can fill the top5.

Commands:

```powershell
python -m pytest tests/unit/test_question_retriever.py -k duplicate
python -m ruff check src/app/domain/question_retriever.py
```

Completed verification:

```text
python -m pytest tests/unit/test_question_retriever.py
14 passed

python -m ruff check src/app/domain/question_retriever.py tests/unit/test_question_retriever.py
All checks passed
```

## Step 5: Weighted Random Final Selection

Status: completed on 2026-06-24.

Expected change size: about 80-140 lines.

Implementation:

- Update `_sample_questions` or replace it with a dedicated weighted selection helper.
- Select from rerank top5 using:

```text
weight = max(final_score, small_epsilon)
```

- For current `top_k=1`, return one weighted random question.
- If `top_k > 1` is used later, perform weighted sampling without replacement.
- Keep tests deterministic by wrapping random choice in a pure helper that accepts an
  injectable random source or seed.

Verification:

- Update `tests/unit/test_question_retriever.py`.
- Assert selection only comes from top5.
- Assert fixed seed produces stable expected output.
- Assert empty candidates return empty output.
- Assert zero scores do not crash selection.

Commands:

```powershell
python -m pytest tests/unit/test_question_retriever.py -k random
python -m ruff check src/app/domain/question_retriever.py
```

Completed verification:

```text
python -m pytest tests/unit/test_question_retriever.py
17 passed

python -m ruff check src/app/domain/question_retriever.py src/app/schemas/interview_state.py tests/unit/test_question_retriever.py
All checks passed
```

## Step 6: Metadata Migration Or Backfill Script

Status: completed on 2026-06-24.

Expected change size: about 180-240 lines.

Implementation:

- Add or update a script under `scripts/` to transform old metadata into the new contract.
- For each record:
  - remove `mainCategory`
  - remove `subCategory`
  - remove `company`
  - add `language="zh"`
  - add `isActive=True`
  - add `userId="global"`
  - normalize `questionType`
  - map `difficulty` to int
- Keep:
  - `question`
  - `answer`
  - `text`
  - `role`
  - `source`
  - `sourceFile`
  - `tags`
- If Milvus requires scalar fields for filtering, support rebuilding or upserting with
  the new scalar fields.

Verification:

- Add a small fixture or unit test for the migration function.
- Run the migration on a sample matching the current vector metadata and assert:

```json
{
  "questionType": "system_design",
  "difficulty": 8,
  "language": "zh",
  "isActive": true,
  "userId": "global"
}
```

- Assert removed fields are absent:

```text
mainCategory
subCategory
company
```

Commands:

```powershell
python -m pytest tests/unit/test_question_metadata.py tests/unit/test_milvus_store.py
python -m ruff check scripts src/app tests/unit
```

Completed verification:

```text
python -m pytest tests/unit/test_question_metadata.py tests/unit/test_question_metadata_migration.py
11 passed

python -m ruff check scripts/migrate_question_metadata.py tests/unit/test_question_metadata_migration.py src/app/domain/question_metadata.py
All checks passed
```

## Step 7: RAG Artifact And Trace Compatibility

Status: completed on 2026-06-24.

Expected change size: about 100-180 lines.

Implementation:

- Update RAG recall sample and outcome artifact handling only as needed.
- Preserve these artifact fields:
  - `scoreBreakdown`
  - `matchedMetadata`
  - `isDuplicate`
  - `rerankRank`
  - `filterReason`
- Accept the new `scoreBreakdown` shape:

```json
{
  "rrf": 0.93,
  "questionType": 1.0
}
```

- Do not change the frontend/BFF SSE contract.

Verification:

- Update `tests/unit/test_outcome_and_rag_artifacts.py`.
- Assert sample artifact writing still preserves explanation fields.
- Assert duplicate veto traces survive artifact serialization.

Commands:

```powershell
python -m pytest tests/unit/test_outcome_and_rag_artifacts.py
python -m ruff check src/app/domain tests/unit/test_outcome_and_rag_artifacts.py
```

Completed verification:

```text
python -m pytest tests/unit/test_outcome_and_rag_artifacts.py
3 passed

python -m ruff check src/app/domain/rag_recall_sample.py src/app/domain/interview_outcome.py tests/unit/test_outcome_and_rag_artifacts.py
All checks passed
```

## Step 8: Final Verification And Architecture Sync

Status: completed on 2026-06-24.

Expected change size: usually 0-80 lines unless architecture instructions need updates.

Verification:

Run focused RAG tests:

```powershell
python -m pytest tests/unit/test_question_metadata.py tests/unit/test_milvus_store.py tests/unit/test_question_retriever.py tests/unit/test_outcome_and_rag_artifacts.py
```

Run Ruff:

```powershell
python -m ruff check src/app tests/unit
```

After any runtime code changes, run the `project-architecture-sync` skill from the host
repo and record the guard:

```powershell
cd ..\my-first-agent
node .github/hooks/scripts/project-architecture-sync-guard.mjs record
```

Completed verification:

```text
python -m pytest tests/unit/test_question_metadata.py tests/unit/test_milvus_store.py tests/unit/test_question_retriever.py tests/unit/test_outcome_and_rag_artifacts.py
32 passed

python -m ruff check src/app tests/unit
All checks passed
```

## Open Decisions

No blocking open questions remain for this plan.

The plan assumes:

- `userId` is reserved for future private question banks and is currently always
  `global`.
- `role` is retained as a job/question-bank descriptor, not a round-type filter.
- `language == "zh"` is fixed for all current recalls.
- Final selection from top5 uses weighted random with `final_score` as the weight.
