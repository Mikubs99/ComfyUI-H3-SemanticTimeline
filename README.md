# ComfyUI-H3-SemanticTimeline

Experimental semantic timeline conditioning for MiniMax H3 context windows in ComfyUI.

This node turns a long-form prompt into separate semantic conditioning regions for MiniMax H3 context windows.

Instead of feeding every context window the same prompt, a local LLM parses the prompt into persistent state, events, transitions, dialogue and continuations. Those are then mapped onto the actual H3 context windows and encoded as a native ComfyUI `CONDITIONING` list.

The sampler and H3 context-window implementation remain untouched.

> **Experimental:** the core timeline / conditioning system works on real H3 renders, but long-form H3 generation still has rough edges. Context boundaries can produce visible seams, dialogue/audio can break or smear across windows, and semantic continuity is not guaranteed in every prompt.

## What it does

Given a prompt containing multiple events, for example:

```text
A woman sits at a kitchen table and talks to the camera.

She says something, pauses, takes a drink, looks toward the window,
then turns back and continues speaking.
```

the parser builds a timeline containing:

- persistent scene information
- character / environment state
- ordered events
- transitions between states
- ongoing actions
- dialogue
- continuation instructions

Each H3 context window then receives conditioning describing what should be happening at that point in the video.

The goal is simple:

**window 1 should know what happens first, window 2 should know what happens next, and so on.**

This does not replace H3's context-window system. It only supplies window-specific conditioning to it.

## Requirements

- ComfyUI with native MiniMax H3 support
- `MiniMax H3 Context Windows`
- `split_conds_to_windows = true`
- a MiniMax H3 CLIP / text encoder
- a local OpenAI-compatible LLM endpoint

The node can manage a local `llama.cpp` server automatically, so running a separate server manually is optional.

No pip packages are required beyond what ComfyUI already provides.

## Installation

Clone or copy this repository into:

```text
ComfyUI/custom_nodes/ComfyUI-H3-SemanticTimeline
```

Restart ComfyUI.

The nodes appear under:

```text
H3 / Semantic Timeline
```

## Basic workflow

The recommended setup is:

```text
Prompt
  │
  ▼
H3 Semantic Timeline Preview
  │ parsed_json
  ▼
H3 Semantic Timeline Encode
  │ conditioning
  ▼
Guider / Sampler
```

The H3 model goes through the native context-window node separately:

```text
MODEL
  │
  ▼
MiniMax H3 Context Windows
  │
  ▼
Guider / Sampler
```

Use the same:

- `total_frames`
- `context_length`
- `context_overlap`

for the timeline and H3 context-window setup.

`split_conds_to_windows` must be enabled.

The Timeline conditioning should be connected directly to the positive conditioning input. Do not combine the individual timeline entries manually.

## Preview first

`H3 Semantic Timeline Preview` runs the semantic parser without requiring a render.

It outputs:

- `timeline_debug`
- `window_prompts`
- `parsed_json`

The Preview node is useful for checking whether the LLM understood the sequence correctly before spending several minutes rendering it.

`parsed_json` can then be passed directly into `H3 Semantic Timeline Encode`.

Encode does not call the LLM again.

## Local llama.cpp parser

For automatic server management:

```text
auto_start_server = true
llama_server_executable = path/to/llama-server.exe
gguf_model = path/to/model.gguf
```

The defaults are:

```text
server_host = 127.0.0.1
server_port = 8080
model_alias = h3-parser
```

The node starts the server when needed, waits for it to become ready and reuses it for the rest of the ComfyUI session.

If the configured loopback HTTP endpoint times out but nothing is actually listening on the TCP port, it is treated as an absent server rather than an ambiguous network failure.

For an already running external server:

```text
auto_start_server = false
server_url = http://127.0.0.1:8080
```

See [`docs/LLAMA_CPP.md`](docs/LLAMA_CPP.md) for the less interesting details.

## Context windows

The timeline is based on the same window layout used by H3.

For example:

```text
context_length = 124
context_overlap = 39
```

or a larger overlap such as:

```text
context_length = 124
context_overlap = 56
```

can be used depending on the workflow.

Larger overlap may improve visual continuity, but it also increases compute and does not automatically solve semantic or audio continuity.

The timeline does not create its own context-window implementation and does not change H3's sampling logic.

## References / I2V / R2V

The Encode node also exposes the native H3 reference inputs:

```text
ref_image_1
ref_image_2
ref_image_3

ref_video_1
ref_video_audio_1
ref_audio_1
```

plus optional:

```text
vae
audio_vae
guide_conditioning
```

These exist so semantic timeline conditioning can be used with H3 reference workflows without rebuilding references separately for every window.

For plain Text-to-Video most of these inputs can simply be left disconnected.

## Semantic parsing

The parser is deliberately stricter than a normal prompt rewriter.

It tries to preserve:

- persistent information
- state changes
- event ordering
- dependencies
- dialogue
- continuation across multiple windows

Parser output is normalized before validation so small inconsistencies in generated `start_state`, `active_state` and `end_state` structures do not immediately destroy an otherwise usable timeline.

Duplicate dialogue that leaks into state/action fields is also cleaned before validation.

There is still a hard limit to what this can do: the parser can misunderstand natural language.

Check the Preview output when testing complicated prompts.

## Current limitations

This project is currently more useful as an experimental long-form H3 tool than as a polished production node.

**Context seams**

H3 can visibly refresh identity, background detail, lighting or motion when switching context windows. Larger overlap helps in some cases, but the timeline node itself does not control H3's latent blending.

**Dialogue and audio**

Speech crossing a context boundary is currently unreliable. Audio from the previous context can overlap, smear or cut when the next context begins.

Short dialogue placed clearly inside one semantic segment works considerably better than long speech spanning multiple windows.

**Story continuity**

Semantic conditioning gives each window knowledge of its intended part of the sequence, but H3 can still ignore or partially forget an event when a new context begins.

This is one of the main things being tested.

**Timing**

Events are assigned semantically. This is not frame-accurate motion or dialogue scheduling.

## Debugging

If a render behaves strangely, start with the Preview output.

Check:

```text
persistent state
initial state
event order
event start/end state
window prompts
frame ranges
```

If those are correct but the generated video resets at a context boundary, the problem is most likely downstream in H3's context continuation rather than timeline parsing.

More implementation details are in:

- [`DESIGN.md`](DESIGN.md)
- [`docs/VALIDATION.md`](docs/VALIDATION.md)
- [`docs/SERVER_MANAGER_DESIGN.md`](docs/SERVER_MANAGER_DESIGN.md)

## Tests

Run:

```bash
python -B run_tests.py
```

The tests mock ComfyUI, server responses and reference handling. They do not render H3 video or download models.

Some tests may lag behind experimental behavior while this project is changing rapidly.

## Development note

This project is heavily vibe-coded.

GPT-6 Astra and GPT-5.6 were used throughout development for implementation, debugging and iteration.

## Status

This is an early experimental release.

The basic path is already functional:

```text
prompt
→ semantic parser
→ timeline
→ per-window prompts
→ native H3 conditioning
→ H3 context windows
```

The remaining work is mostly around what happens **between** those windows.

Which, unfortunately, is also where H3 likes to become temporarily amnesiac.

## License

GPL-3.0-only.
