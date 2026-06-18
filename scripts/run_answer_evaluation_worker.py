import asyncio

from app.logging import configure_logging
from app.workers.answer_evaluation_worker import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    create_default_answer_evaluation_runner,
)


async def main() -> None:
    configure_logging()
    runner = create_default_answer_evaluation_runner()
    await runner.run_forever(poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
