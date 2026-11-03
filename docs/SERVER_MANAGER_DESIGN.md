# v0.1.1 server lifecycle implementation notes

Scope: auto-start and reuse of local llama.cpp through a separate server_manager
module. Keep semantic schema, timeline scheduling, CONDITIONING and references
unchanged. Verify health and exact model alias before requests. Serialize managed
startup, retain only owned process handles, drain bounded diagnostic output, and
never terminate an external server. Test with mocked HTTP/processes only.

## Verified public upstream (2026-11-02)

- llama.cpp commit: 3f8af15a76ac8675812e856da45454bafc90f12c
  https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/tools/server/README.md
- Prompt Enhancer commit: 66858ebd55a52924d4b113fcd518726b2038a77c
  https://github.com/hyukudan/ComfyUI-MiniMax-H3-Prompt-Enhancer/blob/66858ebd55a52924d4b113fcd518726b2038a77c/gguf_server.py

The reference Enhancer uses an RLock, Popen argument list, hidden Windows window,
health polling and owned-session cleanup. This patch independently implements the
same general lifecycle pattern, adding exact model/provider checking and retaining
one explicit local port. It does not copy its directory discovery, random port,
authentication, Windows Job Object, or unbounded temporary log-file strategy.

Verified CLI arguments: -m, --host, --port, --alias, --ctx-size, --parallel, --jinja,
--n-gpu-layers (auto or integer). Explicit port 8080 avoids relying on upstream's
current default. Ready health is HTTP 200 status=ok; model-loading is HTTP 503
with error.type=unavailable_error. /v1/models supplies ID and owned_by=llamacpp.

## Integration choices

ensure_server runs inside request_chat, after request-format validation and before
HTTP submission. analyze's existing cache hit exits before request_chat, so a cache
hit never spawns. No semantic JSON/schema/scheduler/reference code changes.

Use explicit STRING paths; no local installation scanning and no browser upload of
GGUF/executables. New widgets are appended after previous widget order. server_url
is an advanced override; blank means derive it from host and port. The existing
settings dataclass positional field order is preserved and new settings are appended.
The package version is 0.1.1, while existing semantic plan format 0.1.0 is retained.

One owned process is retained. Changing its configuration fails clearly instead
of killing an in-flight request or silently using the old GGUF. Normal shutdown
uses terminate/wait, then kill/wait if needed, on the owned handle only. No process
name matching, PID scans, or termination of external servers. A 64-KiB in-memory
byte tail continuously drains merged output; errors show up to 6000 bytes.
