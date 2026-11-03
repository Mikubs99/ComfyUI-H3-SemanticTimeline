# Validation record — 2026-11-02

Environment: isolated official CPython 3.12.10 Windows embeddable runtime in
`tools/python`, no site-packages, ComfyUI installation, torch, GPU or model weights.
Run: `tools/python/python.exe -B run_tests.py`. Full output: `tests/test-results.txt`.

Automated coverage:
- Ten requested scenarios: linear action, enhanced long prose, dialogue, camera
  transitions, more/fewer events than windows, reference markers, malformed LLM
  JSON, no temporal events, ~30/~60-second timelines.
- Native static scheduling parity: 1,360 combinations against a function extracted
  from the downloaded pinned upstream AST. No importing/executing full upstream modules.
- Separate actual-upstream region method and video-latent conversion parity.
- Region collisions, clamped overlap, shifted last window, single-window bypass.
- Native AV latent shape/length validation and unsupported packed latent rejection.
- One-repair policy, failed-result exclusion from cache, cache invalidation/isolation,
  HTTP JSON payload/envelope handling using a mock transport.
- Native CONDITIONING shape, fresh text tags, ordered region selection, shared
  embeddings with independent metadata dictionaries, guide preservation, reject
  multi-entry encoders and missing reference assets.
- Mock native reference node invoked once, original vision tokenizer data reused.
- Full combined-node execution with mocked parser/CLIP and native latent dimensions.
- Preview without CLIP, preview-to-encode revalidation, settings mismatch detection,
  runtime compatibility guard, package import/registration without ComfyUI.

Not validated here: actual Qwen3/GGUF semantic quality, live llama-server interaction,
real H3 text/vision embeddings, GPU memory/performance, ComfyUI frontend widget rendering,
video/audio output quality, dialogue alignment. Fixtures are hand-authored semantic
JSON, not recorded LLM results. No claims of render-level success or model benchmarking.

Suggested first live acceptance run (performed by the user in their own setup):
1. Queue Preview alone using examples/PROMPTS.md and the configured local server.
2. Inspect persistent fields, current states, one-time transitions/dialogue and every
   native region. A six-event story should not reintroduce earlier completed actions.
3. Connect the same LATENT/settings to Encode and patched MODEL for cross-checking.
4. Render a short native H3 clip, then a longer 719/1433-frame clip. Compare reset
   frequency against a same-seed full-prompt baseline. Reconfirm reference identity
   and audio continuity separately; prompt-level tests cannot establish these.


## v0.1.1 lifecycle patch validation
71 tests pass: the 36 prior tests plus 35 new manager/probe/lifecycle tests. The
old HTTP payload unit test mocks the new ensure_server boundary, and the package
registration test expects 0.1.1. Its semantic plan stays at format 0.1.0.

New coverage includes reuse, launch->readiness->request, both missing paths, invalid
executable/OS launch errors, hidden shell-free Windows argument handling with spaces,
startup crash log diagnostics, timeout cleanup, external loading reuse, wrong
service/alias/provider, concurrent single-spawn, external startup race, changed
launch configuration, dead-process restart, owner-only/idempotent shutdown,
kill fallback, interrupted startup, transient loading-probe failures, bounded
output including lines without newlines, manual/external mode and cache hits.

Only fake process objects and mocked HTTP responses are used. No actual executable
was launched and no GGUF was loaded. Temporary dummy files are created only inside
tests/ and removed afterward. The semantic, scheduling, conditioning and frontend
files are checked against baseline SHA-256 digests. No installed ComfyUI is accessed.
A real llama.cpp startup and a real GPU render remain unverified.
