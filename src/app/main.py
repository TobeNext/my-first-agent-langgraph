from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.graphs.interview_graph import (
    assistant_reply_from_graph_state,
    invoke_interview_graph,
    snapshot_from_graph_state,
)
from app.logging import configure_logging
from app.schemas.api import MastraStreamRequest
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
