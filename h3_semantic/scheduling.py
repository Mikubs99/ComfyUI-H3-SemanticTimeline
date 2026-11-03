"""Pure native H3 scheduling mirror; no torch dependency."""
from dataclasses import dataclass


class TimelineError(ValueError):
    """Actionable node error; ComfyUI catches it without exiting its process."""


def video_latent_t(frames):
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def align_frames(frames):
    if type(frames) is not int or not 5 <= frames <= 100000:
        raise TimelineError('total_frames must be an integer in 5..100000.')
    return frames + (5 - frames) % 17


def latent_info(latent, fallback):
    if latent is None:
        return align_frames(fallback), 'fallback (rounded up to native 17*n+5)', None
    samples = latent.get('samples')
    if not getattr(samples, 'is_nested', False):
        raise TimelineError('Connect native H3 AV LATENT (NestedTensor), or disconnect latent and use total_frames. Packed/other-model tensors are not inferred.')
    parts = samples.tensors
    if len(parts) != 2:
        raise TimelineError('H3 LATENT must have video and audio streams.')
    v, a = parts
    if v.ndim != 5 or v.shape[1] != 24 or a.ndim != 4 or tuple(a.shape[1:3]) != (32, 2):
        raise TimelineError('Not the native H3 AV latent layout.')
    t = int(v.shape[2])
    if t < 2 or (t - 2) % 5:
        raise TimelineError('H3 temporal latent size must be 5*n+2.')
    frames = ((t - 2) // 5) * 17 + 5
    if int(a.shape[-1]) != round(frames * 40 / 24):
        raise TimelineError('Video/audio durations do not match the native H3 24/40 fps layout.')
    return frames, 'native H3 AV latent', (int(v.shape[4]) * 16, int(v.shape[3]) * 16)


@dataclass(frozen=True)
class Window:
    index: int
    latent_start: int
    latent_end: int  # exclusive
    frame_start: int
    frame_end: int  # exclusive
    center_ratio: float


def schedule(total_frames, context_length, context_overlap):
    for name, value, minimum in [('context_length', context_length, 22), ('context_overlap', context_overlap, 5)]:
        if type(value) is not int or value < minimum or (value - 5) % 17:
            raise TimelineError(f'{name} must be 17*n+5, minimum {minimum}.')
    frames = align_frames(total_frames)
    t, length = video_latent_t(frames), video_latent_t(context_length)
    overlap = min(video_latent_t(context_overlap), length - 5)
    starts = [0] if t <= length else []
    if t > length:
        for start in range(0, t, length - overlap):
            starts.append(min(start, t - length))
            if start + length >= t:
                break
    bounds = [0]
    for k in range(t):
        bounds.append(bounds[-1] + (1, 4, 4, 4, 4)[k % 5])
    windows = []
    for i, start in enumerate(starts):
        end = min(start + length, t)
        windows.append(Window(i, start, end, bounds[start], bounds[end], (start + end - 1) / (2 * t)))
    return windows, length, overlap


def region_layout(windows):
    """Find distinct native region selections; slot values index window prompts."""
    if len(windows) == 1:
        return [0], [0]
    for size in range(len(windows), 16385):
        selected = [min(int(w.center_ratio * size), size - 1) for w in windows]
        if len(set(selected)) == len(windows):
            slots = [min(range(len(windows)), key=lambda i: abs(selected[i] - j)) for j in range(size)]
            for i, j in enumerate(selected):
                slots[j] = i
            return slots, selected
    raise TimelineError('Too many conditioning regions (>16384). Increase context stride by reducing overlap or use a shorter video.')


def check_upstream(windows, frames, length, overlap, model=None):
    """Verify against installed core, without modifying it or the model."""
    try:
        import comfy.context_windows as cw
        from comfy_extras.nodes_minimax_h3 import video_latent_t as native_t
        from comfy_extras.nodes_context_windows import MiniMaxH3ContextWindowsNode
        from comfy.model_base import MiniMaxH3
        from types import SimpleNamespace
    except ImportError as exc:
        raise TimelineError('ComfyUI core with MiniMaxH3ContextWindows is required (PR #17425 or compatible newer core).') from exc
    if native_t(frames) != video_latent_t(frames):
        raise TimelineError('Upstream H3 latent conversion changed; update this adapter.')
    indices = cw.create_windows_static_standard(video_latent_t(frames), SimpleNamespace(context_length=length, context_overlap=overlap), {})
    if indices != [list(range(w.latent_start, w.latent_end)) for w in windows]:
        raise TimelineError('Upstream scheduling changed; update this adapter.')
    pixel_starts = MiniMaxH3._latent_pixel_starts(video_latent_t(frames))
    slots, selected = region_layout(windows)
    for w, region in zip(windows, selected):
        native = cw.IndexListContextWindow(indices[w.index], dim=2, total_frames=video_latent_t(frames))
        if native.get_region_index(len(slots)) != region or (pixel_starts[w.latent_start], pixel_starts[w.latent_end]) != (w.frame_start, w.frame_end):
            raise TimelineError('Upstream region/frame mapping changed; update this adapter.')
    if model is not None:
        handler = model.model_options.get('context_handler')
        if handler is None or not handler.split_conds_to_windows:
            raise TimelineError('Connect MODEL after native H3 Context Windows with split_conds_to_windows=true.')
        if (handler.context_length, handler.context_overlap, handler.dim, handler.causal_window_fix) != (length, overlap, 2, False) or handler.context_schedule.name != cw.ContextSchedules.STATIC_STANDARD:
            raise TimelineError('Connected MODEL context settings differ from this timeline.')

