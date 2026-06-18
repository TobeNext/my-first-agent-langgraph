import json
from collections.abc import Iterable
from typing import Any


def encode_sse_data(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def text_delta_event(text: str) -> bytes:
    return encode_sse_data(
        json.dumps({"type": "text-delta", "payload": {"text": text}}, ensure_ascii=False)
    )


def tool_result_event(result: dict[str, Any]) -> bytes:
    return encode_sse_data(
        json.dumps(
            {
                "type": "tool-result",
                "payload": {
                    "toolName": "interviewStateManagerTool",
                    "result": result,
                },
            },
            ensure_ascii=False,
        )
    )


def done_event() -> bytes:
    return encode_sse_data("[DONE]")


def split_text(text: str, chunk_size: int = 24) -> Iterable[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def mastra_compatible_stream(assistant_reply: str, snapshot: dict[str, Any]) -> Iterable[bytes]:
    for chunk in split_text(assistant_reply):
        yield text_delta_event(chunk)
    yield tool_result_event(snapshot)
    yield done_event()
