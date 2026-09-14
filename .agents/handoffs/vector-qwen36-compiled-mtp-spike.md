# Vector: Qwen3.6 compiled MTP composition spike

## Intention and boundary

Determine whether the exact qualified Qwen3.6-35B-A3B native-MTP verifier can
safely compose with the request-private compiled decode primitive from #3449.
Mature the work into a separate PR only if it improves end-to-end batch-one
generation by at least 10% while preserving the five-category quality gate,
greedy token identity, memory bounds, cancellation, and recovery.

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GB)
- Branch: `vector/qwen36-compiled-mtp-spike`
- Worktree: `/private/tmp/rapid-mlx-qwen36-compiled-mtp-spike`
- Base: exact #3449 head `6bc96aa2124a98130eb2ac34fad6cb8ef1298985`

Scope is the exact immutable target/drafter pair, greedy batch-one execution,
measurement, and a minimal execution experiment. Non-goals are changes to
#3449, other models, quantization, downloads, sampling semantics, vision,
batching, prefix-cache policy, or broad runtime replacement.

## Verification plan

1. Inspect the established speculative/compiled composition boundaries in the
   primary serving precedents, then the installed MLX-native target runtime.
2. Run alternating baseline/candidate measurements for coding, reasoning,
   creative writing, strict JSON, and tool arguments.
3. Require exact greedy token hashes and task-contract passes; record TTFT,
   decode throughput, end-to-end time, MTP acceptance, and peak memory.
4. If the performance gate passes, test HTTP disconnect/recovery and run a
   scope-locked self-adversarial review before opening a PR.

## Coordination

Before implementation, Vector attempted the required PR-start FYI separately
to Atlas, Pixel, Harbor, and Echo through Orca run `run_8787255f6b9c`. All four
requests returned `Invalid input`. This handoff preserves the complete FYI
until the messaging channel accepts recipients again.

## Storage constraint

The Hugging Face volume had about 16 GiB free at spike start. The target is an
existing warm-tier symlink and the sidecar is already cached. Do not download,
copy, re-quantize, or relocate either artifact during this spike.

## First findings

### Existing Rapid kernels do not provide the next win

The native MTP backend currently bypasses the ordinary engine's post-load
optimizations, so the first probe enrolled the already-qualified MoE gate/up
and short-row router paths. This exposed an important hard incompatibility:
gate/up fusion removes `up_proj`, while mlx-vlm 0.6.17's exact speculative
verifier reads that split projection directly. The first MTP verify round
failed closed with `AttributeError`; this combination must not be installed.

Router-only enrollment preserved every greedy output and the exact acceptance
rate in a three-round, five-workload, same-load alternating A/B. It was neutral:
paired median 0.998x, 6/15 positive pairs, range 0.757x-1.147x under host
contention. Reject router-only product work unless a new trace identifies a
verifier call site that actually uses it.

### The current default is still a useful Pareto point

An HTTP comparison used the same five prompts, 192-token ceiling, one request
at a time, prefix cache disabled, and fresh processes. The default standard
MTP route was compared with `--no-spec-decode --no-mllm`, which selected the
new compiled ordinary path. Client-observed completion-token throughput was:

| Workload | Default MTP | Compiled AR | MTP / AR |
| --- | ---: | ---: | ---: |
| coding | 144.0 | 118.8 | 1.21x |
| reasoning | 159.4 | 116.9 | 1.36x |
| creative | 105.1 | 111.1 | 0.95x |
| strict JSON | 97.1 | 88.4 | 1.10x |
| tool arguments | 79.5 | 65.2 | 1.22x |

The short-request numbers include each compiled request's private trace cost;
they therefore answer the product-routing question rather than contradicting
#3449's steady-state decode result. MTP remains the better general default. A
prompt-category classifier would be too narrow and fragile for the isolated
creative-writing loss.

The reasoning, JSON, and tool response hashes matched across routes. Coding
and creative hashes differed, so this run is a task-quality comparison rather
than an exact-token claim. Both sampled outputs were structurally plausible,
but a full product decision still requires the existing category contracts at
a non-truncating output budget.

## Shared compiled-program spike

An opt-in prototype reused the immutable compiled callable by exact model,
cache geometry, capacity, token shape, and dtype while continuing to pass each
request's KV/GDN arrays explicitly. Output hashes stayed stable across two
rounds of all five workloads. It did not produce a repeatable 10% win: after
the first request, observed throughput clustered around 120-122 tok/s for the
three long responses and 68-98 tok/s for the two short responses, overlapping
the request-private control. The first coding request still paid a visible
warmup cost. This indicates that MLX already reuses most lower-level compiled
work or that cache conversion/first materialization, rather than callable
construction, owns the remaining short-request tax.

Reject the shared-program change. It would retain graph/cache template objects
across requests and widen poison/lifecycle responsibility without the required
gain. The prototype was reverted; no production source remains changed.

## Current decision

Do not broaden native MTP or switch the default to compiled AR. The current
default MTP route is the best general single-user product point, while #3449
provides the faster deterministic ordinary escape hatch and building block.

For sustained high-acceptance decode, a compiled standard-MTP verifier remains
the larger upside, but it requires explicit transactional KV and GDN state and
is not a small follow-up. Static cohort batching is an aggregate-throughput
track, not a single-user tokens/second claim.
