"""Native CLIP/CONDITIONING adapter; imports ComfyUI only at execution time."""
from .scheduling import TimelineError, schedule, check_upstream
from .semantic import MARKERS


class RecordingClip:
    def __init__(self, clip):
        self.clip = clip
        self.kwargs = {}

    def tokenize(self, text, **kwargs):
        self.kwargs = kwargs
        return self.clip.tokenize(text, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return self.clip.encode_from_tokens_scheduled(tokens)


def one_entry(result):
    if not isinstance(result, list) or len(result) != 1 or len(result[0]) != 2 or not isinstance(result[0][1], dict):
        raise TimelineError('H3 encoding must return exactly one conditioning entry per text. Scheduled/multi-entry CLIP conditioning is unsupported.')
    return result[0]


def encode_plan(plan, clip, width=1344, height=768, model=None, guide_conditioning=None, **assets):
    windows, length, overlap = schedule(plan['total_frames'], plan['context_length'], plan['context_overlap'])
    check_upstream(windows, plan['total_frames'], length, overlap, model)
    tokenizer = getattr(clip, 'tokenizer', None)
    if tokenizer is not None and not hasattr(tokenizer, 'qwen3vl_32b'):
        raise TimelineError('Use the native MiniMax H3 CLIP, not an SD/Wan/Hunyuan text encoder.')
    images = {f'ref_image_{i}': assets.get(f'ref_image_{i}') for i in range(1, 4) if assets.get(f'ref_image_{i}') is not None}
    videos = {'ref_video_1': assets['ref_video_1']} if assets.get('ref_video_1') is not None else {}
    audios = {'ref_audio_1': assets['ref_audio_1']} if assets.get('ref_audio_1') is not None else {}
    sound = {'ref_video_audio_1': assets['ref_video_audio_1']} if assets.get('ref_video_audio_1') is not None else {}
    if sound and not videos:
        raise TimelineError('ref_video_audio_1 requires ref_video_1.')
    # Enforce contiguous ports: native labels count assets, not their input port names.
    if list(images) != [f'ref_image_{i}' for i in range(1, len(images) + 1)]:
        raise TimelineError('Connect reference images consecutively, starting from ref_image_1.')
    counts = {'Picture': len(images), 'Video': len(videos), 'Audio': len(audios) + len(sound)}
    for marker in MARKERS.findall(plan['source_prompt']):
        kind, num = marker[1:-1].split()
        if int(num) < 1 or int(num) > counts[kind]:
            raise TimelineError(f'{marker} has no matching reference input. Text markers alone are not reference embeddings.')
    carry = {}
    if guide_conditioning is not None:
        _, metadata = one_entry(guide_conditioning)
        if metadata.get('minimax_refs'):
            raise TimelineError('guide_conditioning contains references: connect original reference assets instead, so each text gets fresh vision tokens.')
        permitted = {'minimax_keyframes', 'minimax_visual_cond_noise_aug', 'minimax_audio_cond_noise_aug', 'pooled_output', 'minimax_token_tags'}
        unsupported = set(metadata) - permitted
        if unsupported:
            raise TimelineError('Unsupported guide metadata would be lost: ' + ', '.join(sorted(unsupported)))
        carry = {k: v for k, v in metadata.items() if k.startswith('minimax_') and k != 'minimax_token_tags'}
    prompts = plan['window_prompts']
    proxy = RecordingClip(clip)
    cache = {}
    if images or videos or audios:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        out = MiniMaxH3ReferenceToVideo.execute(proxy, prompts[0], width, height, plan['total_frames'],
                vae=assets.get('vae'), audio_vae=assets.get('audio_vae'), ref_images=images,
                ref_videos=videos, ref_audios=audios, ref_video_audios=sound)
        result = out.result if hasattr(out, 'result') else out
        entry = one_entry(result[0])
        if 'minimax_refs' in entry[1]:
            carry['minimax_refs'] = entry[1]['minimax_refs']
        cache[prompts[0]] = entry
    for prompt in dict.fromkeys(prompts):
        if prompt not in cache:
            tokens = clip.tokenize(prompt, **proxy.kwargs)
            cache[prompt] = one_entry(clip.encode_from_tokens_scheduled(tokens))
    conditioning = []
    for index in plan['conditioning_slots']:
        tensor, metadata = cache[prompts[index]]
        # Copy containers, not huge tensors. Core uses copies while slicing payloads.
        merged = dict(metadata)
        for key, value in carry.items():
            merged[key] = [dict(v) for v in value] if isinstance(value, list) else value
        conditioning.append([tensor, merged])
    return conditioning

