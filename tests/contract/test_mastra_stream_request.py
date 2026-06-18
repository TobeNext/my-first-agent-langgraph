from app.schemas.api import MastraStreamRequest


def test_mastra_stream_request_parses_bff_body() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "memory": {"thread": "thread-1", "resource": "frontend-interview-thread-1"},
        "maxSteps": 5,
    }

    parsed = MastraStreamRequest.model_validate(body)

    assert parsed.thread_id == "thread-1"
    assert parsed.resource_id == "frontend-interview-thread-1"
    assert parsed.last_user_message() == "hello"


def test_last_user_message_uses_latest_user_message() -> None:
    parsed = MastraStreamRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": "second"},
            ],
            "memory": {"thread": "thread-2", "resource": "frontend-interview-thread-2"},
        }
    )

    assert parsed.last_user_message() == "second"
