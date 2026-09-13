# Rapid Agent Runtime: one bounded harness for Server and Desktop

- **Status:** Accepted for P0 implementation
- **Owner:** Atlas
- **Date:** 2026-09-13

## Context

MiniCPM5-2B 4-bit is Rapid's leading low-memory local-agent candidate. In the
paired M2 Pro qualification, a bounded harness raised strict completion from
19/36 to 25/36 while enhanced Qwen3.5-4B completed 28/36 at 2.7 times the mean
task wall time. The useful intervention was small: expose fewer tools, retain
host-owned progress, stop repeated actions, and require a final answer.

Rapid already owns most necessary mechanisms. Server has model routing, tool
parsers, MCP discovery/execution, authentication, and OpenAI-compatible APIs.
Desktop has built-in tools, MCP adapters, approvals, MemoryStore, a bounded chat
tool loop, and a metadata-only LocalWorkflow ledger. A second tool ecosystem or
a Desktop-only MiniCPM loop would duplicate these and drift.

The upstream projects are references, not runtime dependencies:

- Hermes Agent: one platform-independent loop, bounded toolsets, prompt layers,
  compression, and SQLite session lineage.
- OpenClaw: host-owned deadlines, durable tasks, approval waits, tool policy,
  and large-catalog tool search.
- OpenHands: action/observation events and a UI that never executes backend
  actions directly.
- OpenCode: client/server sessions and clear read-only versus mutating modes.

## Decision

Rapid will own a small **Agent Runtime** inside the Python server. It is a
stateful orchestration layer above inference, not part of the inference engine.
Plain `/v1/chat/completions` and `/v1/responses` remain stateless and unchanged.

The runtime kernel is a deterministic reducer. It does not call a model, run a
tool, persist secrets, or store hidden reasoning. Adapters drive model requests
and tool execution around it. Its serialized `AgentRun` plus append-only
`AgentEvent` is the shared persistent contract for headless and Desktop paths.
Sensitive execution payloads use a separate transient `AgentRuntimeOutput` and
are never reconstructed from the audit stream.

Desktop will eventually submit and observe runs over the server API. Desktop
built-in tools may remain client-executed: the server emits a redacted
`tool.requested` event and delivers the transient call payload over the
authenticated live request channel. Desktop applies its existing approval and
tool registry, then returns a typed result. Headless operation sends the same
transient output to the existing Server MCP executor. The adapter holds an
approved call only until completion; restart recovery fails an in-flight call
closed. This keeps policy identical without forcing macOS-only tools into Python.

### P0 invariants

1. A run is bound to one immutable model profile.
2. The MiniCPM5-2B profile exposes at most six tools and permits eight tool
   rounds. P0 accepts one tool call per model turn for every profile.
3. Tools not advertised for that exact turn fail closed.
   Registry adapters must classify every tool explicitly; there is no
   permissive default risk.
4. External side effects pause for an explicit approval result tied to the
   exact pending call ID; a call ID may appear only once in a run.
5. Repeating the same tool and arguments more than twice disables tools and
   forces final synthesis.
6. Tool-round exhaustion reserves one tools-disabled final synthesis turn.
7. Every transition appends a versioned, monotonically sequenced event; restore
   rejects duplicate, gapped, or out-of-order histories and state/profile /
   counter values that disagree with that history.
8. Persistent state contains goal, actions, result metadata/safe summaries,
   counters, and final text; never raw tool payload values, model reasoning,
   credentials, screenshots, or clipboard contents. Request events retain only
   call identity and argument names. The adapter passes raw call arguments and
   tool results only to immediate execution/model turns; credentials must be
   resolved from opaque references out of band.
   Repeat fingerprints are runtime-local and are never serialized.
9. Host-generated denial and loop-guard observations are returned transiently
   to the adapter, so every model tool call receives a matching tool result.
10. Tool results carry a short host-authored ledger block. It is not appended as
   a new user instruction; the A/B test showed that shape can restart the task.

### Deliberately absent from P0

- multi-agent delegation or swarms;
- channels, cron, skills marketplace, or plugin framework;
- Docker or a second sandbox implementation;
- automatic long-term-memory writes;
- an LLM-based planner on every turn;
- a semantic tool router before a measured need exists.

These omissions are architectural boundaries, not a roadmap promise.

## Integration sequence

1. **Kernel and contract:** model profiles, reducer, event schema, budgets,
   approval pause, repeat guard, and serialization tests.
2. **Server adapter:** run store and authenticated run/event/result endpoints;
   drive the existing chat generation path and Server MCP executor.
3. **Desktop adapter:** decode the same event schema; execute existing built-in
   tools and approvals; retain the old loop as rollback until parity tests pass.
4. **Qualification:** real GUI tasks on physical 8 GB and 16 GB Macs. Promotion
   of MiniCPM5 to the primary recommendation is a measured catalog change, not
   an architectural default.

## Consequences

The first kernel adds no dependency and no idle process. Model-specific tuning
is data in `AgentProfile`, so Qwen and future compact models reuse the same
runtime. The server becomes the owner of run state, while the GUI stays the
owner of macOS presentation and client-local tool execution.

The initial in-memory adapter will not claim crash durability. Durable SQLite
storage is added with the server API, using atomic transitions and explicit
schema migration. Until Desktop has migrated, its existing tool loop remains
the shipping path.

## References

- <https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/architecture.md>
- <https://github.com/openclaw/openclaw/blob/main/docs/concepts/agent-loop.md>
- <https://docs.openclaw.ai/tools>
- <https://github.com/OpenHands/OpenHands/blob/main/docs/architecture.md>
- <https://github.com/anomalyco/opencode>
- [MiniCPM5 harness qualification](../performance/2026-09-13-minicpm5-small-agent-harness-ab.md)
