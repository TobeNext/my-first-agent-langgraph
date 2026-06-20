# my-first-agent-langgraph

Python LangGraph runtime for the Mastra-to-LangGraph migration.

This runtime is now the default provider used by the frontend/BFF host in
`../my-first-agent`. Runtime wiring is complete: the service exposes FastAPI
health checks, accepts the structured interview start/reply contract, returns
Mastra-compatible SSE, checkpoints LangGraph state, writes outcome/RAG artifacts,
and can complete the deterministic short interview flow.

Behavior parity with the legacy Mastra provider is not complete yet. The current
baseline deliberately separates completed runtime wiring from remaining provider
smoke and rollback proof:

- Follow-up questions can be generated through the LangChain chat model factory
  when a real provider is configured; the default `MODEL_PROVIDER=mock` and any
  model failure still fall back to deterministic follow-up logic.
- When the interview reaches wrap-up, the stream response returns immediately
  with a report-generating message. A FastAPI background task then evaluates
  recorded answers, generates the report, and persists it to the report DB.
  Report status and markdown APIs read from the report DB, so the local stack no
  longer needs external answer/report workers to complete the main report flow.
- RAG metadata normalization, existing Milvus read smoke, and hybrid rerank tests
  now cover the Mastra baseline. The legacy trace field `bm25Score` currently
  records skillArea match score, not a true lexical BM25 score.
- Embeddings use deterministic 384-dimensional hash vectors by default. Configure
  an OpenAI-compatible embedding provider to query Milvus with real provider
  vectors; no-key startup keeps the hash fallback.

Unit 00 golden transcript fixtures live in
`../my-first-agent/PLAN/fixtures/contracts`. The contract tests in this repo
load those fixtures to freeze the current deterministic baseline and to keep
`runtime wiring complete` separate from `behavior parity complete`.

## Model Configuration

The runtime starts without model credentials by default:

```env
MODEL_PROVIDER=mock
MODEL_NAME=mock/interview-runtime
```

To enable real follow-up generation, configure an OpenAI-compatible provider:

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
MODEL_API_KEY=...
# MODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

`MODEL_TIMEOUT_SECONDS`, `MODEL_MAX_RETRIES`, and `MODEL_TEMPERATURE` control the
LangChain chat model factory. `MODEL_PROVIDER=zhipu` uses `ZHIPU_API_KEY` when
`MODEL_API_KEY` is not set and otherwise follows the same OpenAI-compatible path.
The default timeout is 90 seconds to give final report generation enough time
without blocking the last interview stream response.

## Embedding Configuration

The runtime starts with deterministic hash embeddings:

```env
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=384
MILVUS_ADDRESS=http://localhost:19530
```

For an existing Milvus collection built with provider vectors, configure an
OpenAI-compatible embedding provider that matches the collection dimension:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=...
# EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_DIMENSION=384
```

## Development

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\pytest
.venv\Scripts\ruff check .
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8011
```

`requirements.lock` records the versions used for the Unit 01-06 baseline.
