from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.config import get_settings
from app.domain.report_status import (
    get_report_markdown,
    mark_interview_report_read,
    resolve_interview_report_status,
)
from app.graphs.interview_graph import (
    assistant_reply_from_graph_state,
    invoke_interview_graph,
    snapshot_from_graph_state,
)
from app.integrations.redis_client import (
    create_redis_answer_evaluation_store,
    create_redis_report_generation_store,
)
from app.integrations.redis_evaluation_store import RedisAnswerEvaluationStore
from app.integrations.redis_report_generation_store import RedisReportGenerationStore
from app.integrations.report_repository import InterviewReportRepository
from app.logging import configure_logging
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_report import InterviewReportStatus
from app.sse import mastra_compatible_stream

configure_logging()

app = FastAPI(title="My First Agent LangGraph Runtime", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "runtime": "python-langgraph",
        "provider": settings.model_provider,
        "model": settings.model_name,
    }


@app.post("/api/agents/interview-agent/stream")
def stream_interview_agent(request: MastraStreamRequest) -> StreamingResponse:
    graph_state = invoke_interview_graph(request)
    snapshot = snapshot_from_graph_state(graph_state)
    return StreamingResponse(
        mastra_compatible_stream(assistant_reply_from_graph_state(graph_state), snapshot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


def get_answer_evaluation_store() -> RedisAnswerEvaluationStore:
    return create_redis_answer_evaluation_store()


def get_report_generation_store() -> RedisReportGenerationStore:
    return create_redis_report_generation_store()


def get_interview_report_repository() -> InterviewReportRepository:
    return InterviewReportRepository()


@app.get("/api/interviews/{thread_id}/report/status")
async def interview_report_status(
    thread_id: str,
    evaluation_store: Annotated[
        RedisAnswerEvaluationStore,
        Depends(get_answer_evaluation_store),
    ],
    report_store: Annotated[
        RedisReportGenerationStore,
        Depends(get_report_generation_store),
    ],
    repository: Annotated[
        InterviewReportRepository,
        Depends(get_interview_report_repository),
    ],
) -> InterviewReportStatus:
    return await resolve_interview_report_status(
        thread_id=thread_id,
        evaluation_store=evaluation_store,
        report_store=report_store,
        repository=repository,
    )


@app.get("/api/interviews/{thread_id}/report/markdown")
def interview_report_markdown(
    thread_id: str,
    repository: Annotated[
        InterviewReportRepository,
        Depends(get_interview_report_repository),
    ],
) -> Response:
    markdown = get_report_markdown(thread_id=thread_id, repository=repository)
    if markdown is None:
        raise HTTPException(status_code=404, detail="Interview report markdown was not found.")
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="interview-report-{thread_id}.md"'
        },
    )


@app.post("/api/interviews/{thread_id}/report/read")
async def interview_report_read(
    thread_id: str,
    report_store: Annotated[
        RedisReportGenerationStore,
        Depends(get_report_generation_store),
    ],
    repository: Annotated[
        InterviewReportRepository,
        Depends(get_interview_report_repository),
    ],
) -> dict[str, str]:
    receipt = await mark_interview_report_read(
        thread_id=thread_id,
        repository=repository,
        report_store=report_store,
    )
    return {"threadId": thread_id, "readAt": receipt.read_at}
