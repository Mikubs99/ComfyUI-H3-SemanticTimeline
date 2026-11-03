# Design — v0.1.0 (2026-11-02)

## Verified upstream and scope
ComfyUI commit `bb878b9ee35274e67931634ba9e8c338f8c06035` was fetched before implementation.
Sources (pinned links):
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/comfy_extras/nodes_context_windows.py
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/comfy/context_windows.py
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/comfy_extras/nodes_minimax_h3.py
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/comfy/text_encoders/minimax.py
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/comfy/model_base.py
- https://github.com/Comfy-Org/ComfyUI/blob/bb878b9ee35274e67931634ba9e8c338f8c06035/nodes.py

Kijai authored core H3 context support (PR #17425). HunyuanVideoWrapper at
`fcbd6729a9b0b8ff6037c598bbada4a6bdc6d967` has Hunyuan-specific encoders/context
and no H3 entries in its tree/node registration. This adapter targets native core
H3, not HYVIDEMBEDS or a wrapper sampler. No core patches or sampler hooks.

## Exact window scheduling
Native H3 uses static_standard, stride=1, closed_loop=false, temporal dim=2,
causal_window_fix=false. User length and overlap are real frames (17*n+5).
video_latent_t(F) = 2 if F<=5 else ((F-5)//17)*5+2.
Effective overlap = min(video_latent_t(overlap), latent_length-5).
Step = latent_length-effective_overlap. Static windows start at 0, step, ...;
the last window is shifted back to end exactly at total latent T. T<=length
bypasses context processing and must receive ONE conditioning entry.
Real pixel boundaries are cumulative (1,4,4,4,4), not latent_index*4.
AV latent is NestedTensor(video[B,24,T,H,W], audio[B,32,2,A]); valid T=5*k+2.
Derive F=17*k+5 from that structure. Never infer duration from packed flat tensors.
Fallback is a frame count snapped UP to 17*k+5, as native temporal_shape does.

## Conditioning mapping: crucial departure from the proposed simple list
ConditioningCombine concatenates two Python lists. Native window selection uses
center=(first_latent_index+last_latent_index)/(2*T), then
region=clamp(int(center*len(conditioning)),0,len(conditioning)-1).
Thus N windows do NOT imply N distinct conditioning regions! Find a list length
whose region indices are unique for all actual windows; fill unused slots with
nearby prompts and assign each selected slot to its exact window. Encode only
unique texts; repeated slots reuse tensors but copy metadata dictionaries.
This is ordinary CONDITIONING, no time masks or new runtime types. Do not append
other conditionings afterward: list length and therefore mapping would change.
Runtime verifies scheduling/mapping against installed upstream public functions
and can validate an optional patched MODEL input. Unsupported changes fail closed.

## Interface
- H3 Semantic Timeline: prompt + native H3 CLIP + context settings + llama-server
  URL/model alias; optional LATENT, patched MODEL, native guide conditioning and
  reference assets. Outputs CONDITIONING and three STRING debug outputs.
- H3 Semantic Timeline Preview: same analysis, no CLIP/GPU/sampler; STRING plan,
  debug and window prompts. Can be queued alone before a render.
- H3 Semantic Timeline Encode: accepts preview JSON plus CLIP and the SAME settings/
  latent; recomputes the plan and verifies it, never trusts arbitrary window indices.

Native reference conditioning is prepared once via MiniMaxH3ReferenceToVideo;
a recording CLIP proxy captures the native tokenizer kwargs while encoding the
first text. Subsequent texts use those same native images/ref_items. Keep fresh
minimax_token_tags/pooled_output per text, carry only native keyframes/reference
payloads. Optional guide conditioning carries keyframes/noise augmentation, never
stale text tokens or hidden reference embeddings. Reject missing reference assets.

## Semantic timeline
llama.cpp HTTP chat completions, stdlib urllib, no new Python packages. User chooses
the server model ID/alias (GGUF path is configured on server, no auto-launch).
Schema separates immutable subject/appearance/clothing/environment/lighting/camera/
style/audio/reference text from evolving state, events, dependencies, exact quoted
dialogue. Event preconditions and state patches are validated sequentially.
No punctuation/token-based event splitting. One schema-constrained generation and
at most one repair; invalid output stops this node, not the ComfyUI process.

Python allocates weighted event intervals to non-overlapping semantic ownership
cells bounded by adjacent actual window pixel-centers. Physical windows still
overlap, under native fusion; ownership avoids replaying transitions/dialogue.
Transitions fire once then hold end-state. Ongoing actions use continuation text
in later cells. Events in one cell stay chronological. Persistent audio is distinct
from one-time spoken dialogue. Timing remains a heuristic, not a hard video guarantee.
Cache validated JSON in bounded process memory, keyed by source + frames + context
+ backend/parser settings + schema version. No LLM call during sampling.

## Existing projects checked
https://github.com/imbutus/ComfyUI-MiniMaxDirector compiles manually authored tracks
into a single prompt. https://github.com/chanon/comfyui-obvpm-timeline is a broader
video editing/extension workflow. https://github.com/BSG-Walter/ComfyUI-H3-Motion-Context-Timeline
anchors media guides. https://github.com/Thefrizzy1/ComfyUI-MiniMaxH3-Director uses
multiple clip generations. None of these reviewed descriptions provides this exact
natural-language -> semantic state -> native center-region conditioning adapter.
This is a scoped search, not proof no equivalent exists.

## Validation and boundaries
Use stdlib unittest with mocked tensors/CLIP/HTTP, plus AST-extracted pure scheduling
functions from the downloaded pinned upstream (no torch or real ComfyUI required).
Portable Python stays under tools/ in this workspace. Test semantic fixtures are
mocked LLM output: they cannot establish real 8B parsing accuracy or rendered quality.
Ship source-only ZIP excluding research, portable runtime and development material.
No actual ComfyUI/GGUF execution or render is performed without those runtimes here.


## v0.1.1 update: automatic server lifecycle
The initial v0.1 external-only server decision above is superseded by
`docs/SERVER_MANAGER_DESIGN.md`. A separate server_manager now wraps HTTP requests
with health/model validation and optional local process startup. The semantic
plan format remains 0.1.0; the package release is 0.1.1. Core semantic/scheduling/
conditioning source files remain byte-for-byte unchanged.
