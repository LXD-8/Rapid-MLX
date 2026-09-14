# Vector to Atlas: APCv2 host prompt plane

- Owner: Vector
- Receiving role: Atlas
- Host: Studio (M3 Ultra, 256 GB)
- Branch: `vector/apcv2-segments`
- PR: pending

## Verified facts

- Exact chat render and tokenization results now share one bounded host LRU
  between `BatchedEngine` and the text `Scheduler`.
- The tokenizer remains on the existing MLX worker thread; no long-prompt work
  moved onto the asyncio event loop.
- Cache limits are 64 entries / 64 MiB, unload clears it, and
  `RAPID_MLX_PROMPT_HOST_CACHE=0` is a rollback switch.
- A cached Qwen3 tokenizer benchmark measured 18.67x to 30.67x lower repeated
  host preparation latency across 719 to 21,708 prompt tokens with exact
  string/token equality.
- The generic device-side B1-to-B2 broadcast idea was rejected: at 65,536
  tokens it ran at 0.46x physical-B2 speed and allocated 134.8 MB over baseline.
- Focused engine/scheduler/template tests pass. Full PR validation remains.

## Risks and unresolved questions

- The feature accelerates exact repeated host inputs; it does not incrementally
  tokenize a changed multi-turn suffix.
- Static Qwen4 MTP cohort batching remains a larger cross-cutting port and needs
  a real baseline-vs-candidate server throughput receipt before product work.
- Atlas should review the additive stats surface and default-on/rollback policy
  as part of release integration.

## Next concrete action

Run full PR validation, update this handoff with the PR number, then queue if
CI and self-review remain clean. After merge, benchmark static Qwen4 cohort
batching as a separate spike rather than extending this branch.
