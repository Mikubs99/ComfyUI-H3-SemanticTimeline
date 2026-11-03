# Changelog

## 0.1.1 — 2026-11-02
- Local llama.cpp auto-start, exact model-alias verification and existing-server reuse.
- Executable/GGUF path, host/port, startup timeout, context and GPU-layer widgets.
- Thread-safe single owned process; hidden Windows launch, readiness polling, bounded log tail.
- Startup failure/interrupt cleanup and owner-only shutdown on normal interpreter exit.
- Manual/external mode retained; no server activity on semantic analysis cache hits.
- Semantic plan format/schema, scheduling, CONDITIONING and references unchanged.
- Added mocked lifecycle/probe/concurrency tests and updated setup documentation.

## 0.1.0 — 2026-11-02
- Natural-language LLM semantic parsing through an existing llama.cpp HTTP server.
- Validated state transitions, active/finished states, continuity and dialogue.
- Native H3 frame scheduling and center-region-compatible CONDITIONING allocation.
- Combined, preview-only and encode-from-preview nodes; native reference/guide support.
- Bounded cache, one repair attempt, explicit compatibility/shape errors.
- Offline tests, pinned-upstream research, installation/configuration documentation.
