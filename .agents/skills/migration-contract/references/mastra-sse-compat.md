# Mastra SSE Compatibility Contract

- source URL: current repo `frontend/src/services/agent-stream.ts`
- fetchedAt: 2026-06-11
- applicable package/version: current frontend parser

The Python runtime must emit SSE `data:` lines containing:

- `{"type":"text-delta","payload":{"text":"..."}}`
- `{"type":"tool-result","payload":{"toolName":"interviewStateManagerTool","result":{...}}}`
- `[DONE]`

`payload.result` must preserve the `InterviewStateSnapshot` shape consumed by the frontend Zod schema.
