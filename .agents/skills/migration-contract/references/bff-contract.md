# BFF Contract

- source URL: current repo `bff/src/modules/agent/agent.service.ts`
- fetchedAt: 2026-06-11
- applicable package/version: current TypeScript BFF contract

The Python runtime must accept:

- `messages`
- `memory.thread`
- `memory.resource`
- `maxSteps`

`threadId` is `memory.thread`. `resourceId` is `memory.resource`. The user input is the latest
message where `role === "user"`.
