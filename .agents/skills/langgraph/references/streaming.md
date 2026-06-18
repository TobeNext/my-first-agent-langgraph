# LangGraph Streaming Reference

- source URL: https://docs.langchain.com/oss/python/langgraph/streaming
- fetchedAt: 2026-06-11
- applicable package/version: To be pinned by this repo lockfile before real graph implementation.

The first migration phase exposes a Mastra-compatible SSE facade. Later LangGraph streaming may feed
the same facade, but emitted frontend events must remain `text-delta`, `tool-result`, and `[DONE]`
until the contract is intentionally versioned.
