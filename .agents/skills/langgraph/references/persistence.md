# LangGraph Persistence Reference

- source URL: https://docs.langchain.com/oss/python/langgraph/persistence
- fetchedAt: 2026-06-11
- applicable package/version: To be pinned by this repo lockfile before real graph implementation.

Use LangGraph persistence/checkpointing for graph state and thread continuity. The migration must map
the frontend `threadId` to the LangGraph thread identifier and keep the business
`InterviewSessionState` JSON shape unchanged inside the checkpointed graph state.
