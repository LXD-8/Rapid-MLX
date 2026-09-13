# Atlas handoff — Rapid Agent Runtime

- **Owner:** Atlas
- **Branch:** `atlas/minicpm-agent-runtime-p0`
- **Base:** `origin/main` after MiniCPM harness qualification PR #3425
- **Status:** P0 kernel and architecture contract implemented; server adapter is next

## Verified facts

- Desktop already owns built-in tools, MCP adapters, approval UX, MemoryStore,
  a bounded chat tool loop, and a metadata-only LocalWorkflow ledger.
- Server already owns model routing, MCP discovery/execution, tool parsing,
  authentication, and Chat/Responses APIs.
- The P0 kernel adds no dependency or process. It stores a complete immutable
  model-profile snapshot, snapshots the exact per-turn tool policy, and emits a
  versioned append-only event stream.
- Raw call arguments and tool-result content are transient and never enter that
  event stream. Request events retain call identity and argument names; result
  events retain size, digest, error state, and an optional producer-authored
  safe summary.
- Call IDs are single-use, approvals match the exact pending ID, and restored
  event histories must be contiguous from sequence one.
- MiniCPM5-2B defaults are six visible tools, eight tool rounds, one call per
  model turn, and two identical calls before forced final synthesis.
- Focused unit tests, Ruff, and focused mypy pass.

## Architecture boundary

The Python server owns run state and orchestration. Desktop will consume run
events plus an authenticated transient call channel, retain presentation and
its client-local tool executors, and return typed results. Plain Chat/Responses
endpoints remain stateless and compatible.
Do not add a second tool registry, memory implementation, sandbox, or planning
framework to the runtime kernel.

## Next concrete action

Add the authenticated Server adapter as a separate PR: an atomic run store,
create/get/events/approval/result/cancel endpoints, and a model-turn adapter
over the existing generation path. Do not expose endpoints until a created run
can make progress end to end. Follow with a Desktop client migration behind a
rollback feature flag, then physical 8 GB / 16 GB qualification.

## Risks

- Model generation is still route-owned; extracting a reusable internal
  generation service is preferable to making an in-process HTTP call.
- Durable SQLite storage needs migration and corruption tests; P0 deliberately
  makes no crash-durability claim.
- An in-flight call's raw payload is adapter-owned and cannot resume after a
  process restart; recovery must fail that run safely rather than execute the
  persisted metadata-only call.
- Risk classification must come from the registry snapshot the model saw, not
  from a later client-supplied tool definition.
