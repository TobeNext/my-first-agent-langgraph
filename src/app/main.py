from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import get_settings
from app.domain.report_status import (
    get_report_markdown,
    mark_interview_report_read,
    resolve_interview_report_status,
)
from app.graphs.interview_graph import (
    assistant_reply_from_graph_state,
    invoke_interview_graph,
    run_report_generation_for_thread,
    should_start_background_report_generation,
    snapshot_from_graph_state,
)
from app.integrations.report_repository import InterviewReportRepository
from app.logging import configure_logging
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_report import InterviewReportStatus
from app.sse import mastra_compatible_stream
from app.telemetry import instrument_fastapi, interview_protocol_from_message

configure_logging()

app = FastAPI(title="My First Agent LangGraph Runtime", version="0.1.0")
instrument_fastapi(app)


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
def stream_interview_agent(
    request: MastraStreamRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    settings = get_settings()
    with _get_tracer().start_as_current_span(
        "python_agent.stream_interview_agent",
        attributes={
            "interview.thread_id": request.thread_id,
            "interview.resource_id": request.resource_id,
            "interview.runtime_provider": "python-langgraph",
            "interview.protocol": interview_protocol_from_message(request.last_user_message()),
            "llm.provider": settings.model_provider,
            "llm.model": settings.model_name,
        },
    ) as span:
        try:
            graph_state = invoke_interview_graph(request)
            if should_start_background_report_generation(graph_state):
                background_tasks.add_task(run_report_generation_for_thread, request.thread_id)
            snapshot = snapshot_from_graph_state(graph_state)
            return StreamingResponse(
                mastra_compatible_stream(assistant_reply_from_graph_state(graph_state), snapshot),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                },
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def get_interview_report_repository() -> InterviewReportRepository:
    return InterviewReportRepository()


@app.get("/api/interviews/{thread_id}/report/status")
async def interview_report_status(
    thread_id: str,
    repository: Annotated[
        InterviewReportRepository,
        Depends(get_interview_report_repository),
    ],
) -> InterviewReportStatus:
    return await resolve_interview_report_status(
        thread_id=thread_id,
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
    repository: Annotated[
        InterviewReportRepository,
        Depends(get_interview_report_repository),
    ],
) -> dict[str, str]:
    receipt = await mark_interview_report_read(
        thread_id=thread_id,
        repository=repository,
    )
    return {"threadId": thread_id, "readAt": receipt.read_at}


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer("interview-python-agent")
