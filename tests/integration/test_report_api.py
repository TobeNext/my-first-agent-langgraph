from fastapi.testclient import TestClient

from app.main import (
    app,
    get_interview_report_repository,
)
from tests.unit.test_report_status import (
    FakeRepository,
    build_report,
)

NOW = "2026-06-19T00:00:00.000Z"


def test_report_status_markdown_and_read_api() -> None:
    repository = FakeRepository(build_report(markdown="## API Report"))

    app.dependency_overrides[get_interview_report_repository] = lambda: repository
    try:
        client = TestClient(app)

        status_response = client.get("/api/interviews/thread-1/report/status")
        markdown_response = client.get("/api/interviews/thread-1/report/markdown")
        read_response = client.post("/api/interviews/thread-1/report/read")
        read_status_response = client.get("/api/interviews/thread-1/report/status")
    finally:
        app.dependency_overrides.clear()

    assert status_response.status_code == 200
    assert status_response.json()["reportState"] == "ready"
    assert status_response.json()["unreadCount"] == 1
    assert markdown_response.status_code == 200
    assert markdown_response.text == "## API Report"
    assert markdown_response.headers["content-type"].startswith("text/markdown")
    assert (
        markdown_response.headers["content-disposition"]
        == 'attachment; filename="interview-report-thread-1.md"'
    )
    assert read_response.status_code == 200
    assert read_response.json()["threadId"] == "thread-1"
    assert read_status_response.json()["unreadCount"] == 0


def test_report_markdown_api_returns_404_when_missing() -> None:
    app.dependency_overrides[get_interview_report_repository] = lambda: FakeRepository(None)
    try:
        response = TestClient(app).get("/api/interviews/thread-missing/report/markdown")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
