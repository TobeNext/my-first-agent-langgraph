import asyncio

from app.logging import configure_logging
from app.workers.report_generation_worker import (
    DEFAULT_REPORT_GENERATION_POLL_INTERVAL_SECONDS,
    create_default_report_generation_runner,
)


async def main() -> None:
    configure_logging()
    runner = create_default_report_generation_runner()
    await runner.run_forever(
        poll_interval_seconds=DEFAULT_REPORT_GENERATION_POLL_INTERVAL_SECONDS
    )


if __name__ == "__main__":
    asyncio.run(main())
