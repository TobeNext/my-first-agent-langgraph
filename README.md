# my-first-agent-langgraph

Maintained Python LangGraph interview runtime for the AI interview practice system. This is the default and only actively developed runtime provider for the frontend/BFF host at `../my-first-agent`.

## Overview

The runtime exposes FastAPI endpoints for streaming interview sessions, report generation, and health checks. It uses LangGraph for stateful interview orchestration with SQLite checkpointing, LangChain for model interactions, and Milvus for question retrieval.

- Streaming interview agent endpoint with Mastra-compatible SSE
- LangGraph state machine with checkpointed conversation state
- Background report generation with SQLite persistence
- Deterministic hash-embedding fallback for no-key local startup
- Configurable OpenAI-compatible model and embedding providers

## Architecture

The runtime is organized into layered Python modules under `src/app/`:

| Layer | Path | Responsibility |
|-------|------|----------------|
| HTTP handlers | `main.py` | FastAPI routes, SSE streaming, background tasks |
| Graph orchestration | `graphs/` | LangGraph state graph, node definitions, checkpoint routing |
| Domain logic | `domain/` | Interview state machine, follow-up generation, report generation, answer evaluation, question retrieval, resume parsing |
| Integrations | `integrations/` | Milvus vector store, SQLite checkpoint store, embedding provider, LLM factory, report repository |
| Contracts | `schemas/` | Pydantic models for API requests, interview state, reports, answer evaluation |

## API Endpoints

### Health

```
GET /health
```

Returns the runtime status, provider, and model name.

### Streaming Interview Agent

```
POST /api/agents/interview-agent/stream
```

Accepts a Mastra-compatible structured request with `threadId`, `resourceId`, and message list. Returns `text/event-stream` SSE with assistant replies, progress metadata, and a final interview snapshot. When the interview reaches wrap-up, a background task is scheduled for report generation.

### Report Status

```
GET /api/interviews/{thread_id}/report/status
```

Returns the current report generation status: `not_found`, `generating`, `ready`, or `failed`.

### Report Markdown

```
GET /api/interviews/{thread_id}/report/markdown
```

Returns the generated interview report as Markdown (`text/markdown`).

### Report Read Receipt

```
POST /api/interviews/{thread_id}/report/read
```

Marks the interview report as read by the user.

## Configuration

The runtime starts without model credentials by default. All settings are read from environment variables or a `.env` file.

### Model

```env
MODEL_PROVIDER=mock                  # mock, openai, zhipu
MODEL_NAME=mock/interview-runtime
MODEL_API_KEY=                       # required for openai/zhipu
MODEL_BASE_URL=                      # optional custom endpoint
MODEL_TIMEOUT_SECONDS=90
MODEL_MAX_RETRIES=2
MODEL_TEMPERATURE=0.2
```

`MODEL_PROVIDER=zhipu` uses `ZHIPU_API_KEY` when `MODEL_API_KEY` is not set and otherwise follows the OpenAI-compatible path. The default timeout of 90 seconds allows final report generation to complete without blocking the last stream response.

### Embeddings

```env
EMBEDDING_PROVIDER=hash              # hash (deterministic) or openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_DIMENSION=384
```

Hash embeddings provide deterministic 384-dimensional vectors for no-key local development. Configure `EMBEDDING_PROVIDER=openai` to query Milvus with provider-backed vectors.

### Infrastructure

```env
MILVUS_ADDRESS=http://localhost:19530
CHECKPOINT_URL=sqlite:///./checkpoints.db
REPORT_DATABASE_URL=sqlite:///./interview_reports.db
```

### Artifact Paths

```env
OUTCOME_ROOT=../my-first-agent/Interview outcome
RAG_LOG_ROOT=../my-first-agent/RAG LOG INFO
```

### Optional: LangSmith Tracing

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=my-first-agent-local
```

## Development

```bash
# Setup
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Run tests
.venv\Scripts\pytest

# Lint
.venv\Scripts\ruff check .

# Start server
$env:PYTHONPATH='src'
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8011
```

The runtime targets Python 3.12+ with Pydantic v2, FastAPI, LangGraph, and LangChain. See `pyproject.toml` for the full dependency list.

## Relationship with the Host Repository

This repository is the maintained interview runtime sibling to `../my-first-agent`. The host repository owns the Vue frontend, NestJS BFF, and local stack orchestration. The BFF proxies streaming interview traffic to this runtime by default (`AGENT_RUNTIME_PROVIDER=python`).

The legacy Mastra runtime under `../my-first-agent/src/mastra/` is archived and no longer maintained. All new interview runtime features belong here.

## Git-Ignored Artifacts

The following are intentionally excluded from version control:

- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` — Python runtime and cache
- `.env` — local environment variables (`.env.example` is the template)
- `*.db`, `*.db-shm`, `*.db-wal` — SQLite databases (checkpoints, reports)
- `EmbeddingBenchmark/` — benchmark result artifacts
- `tmp_*.py` — temporary analysis scripts
- `leetcode/` — practice code
