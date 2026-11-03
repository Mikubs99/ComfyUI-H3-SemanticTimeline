"""ComfyUI legacy node API, with standard native output socket types only."""
import json
from .parser import ParserSettings, analyze
from .semantic import VERSION, compile_plan
from .scheduling import TimelineError, latent_info, schedule
from .conditioning import encode_plan


def context_inputs():
    return {'context_length': ('INT', {'default': 124, 'min': 22, 'max': 100000, 'step': 17}),
            'context_overlap': ('INT', {'default': 39, 'min': 5, 'max': 100000, 'step': 17}),
            'total_frames': ('INT', {'default': 362, 'min': 5, 'max': 100000, 'tooltip': 'Fallback only. Native H3 LATENT takes priority; rounded up to 17*n+5.'})}


def parser_inputs():
    return {'prompt': ('STRING', {'multiline': True, 'default': '', 'forceInput': False}), **context_inputs(),
            'server_url': ('STRING', {'default': '', 'tooltip': 'Advanced override. Leave blank to use server_host/server_port. An explicit URL takes precedence.'}),
            'model_alias': ('STRING', {'default': 'h3-parser', 'tooltip': 'Exact API model ID/alias. Auto-start publishes this alias; gguf_model selects the file.'}),
            'temperature': ('FLOAT', {'default': 0.1, 'min': 0, 'max': 2}),
            'max_tokens': ('INT', {'default': 8192, 'min': 512, 'max': 32768}),
            'timeout': ('INT', {'default': 180, 'min': 1, 'max': 1800}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
            'response_mode': (['json_schema', 'json_object', 'prompt_only'],),
            'cache_revision': ('INT', {'default': 0, 'min': 0, 'tooltip': 'Increment after changing model weights behind the same server alias.'}), **server_inputs()}


def server_inputs():
    # Append widgets after v0.1 inputs to preserve their serialized ordering.
    return {'auto_start_server': ('BOOLEAN', {'default': True}),
            'llama_server_executable': ('STRING', {'default': '', 'tooltip': 'Full path to llama-server.exe. Paths with spaces work; no surrounding quotes. No automatic downloads.'}),
            'gguf_model': ('STRING', {'default': '', 'tooltip': 'Full path to the local parser .gguf file, passed to llama-server. Python does not load it.'}),
            'server_host': ('STRING', {'default': '127.0.0.1'}),
            'server_port': ('INT', {'default': 8080, 'min': 1, 'max': 65535}),
            'server_startup_timeout': ('INT', {'default': 180, 'min': 1, 'max': 3600, 'tooltip': 'Seconds to wait for health + exact model readiness.'}),
            'server_context_size': ('INT', {'default': 16384, 'min': 2048, 'max': 131072, 'step': 1024}),
            'server_gpu_layers': ('INT', {'default': -1, 'min': -1, 'max': 999, 'tooltip': '-1 = llama.cpp auto, 0 = CPU. Managed server stays loaded; reserve VRAM for H3.'})}


def encode_inputs():
    return {'model': ('MODEL', {'tooltip': 'Optional: MODEL after H3 Context Windows, for settings validation.'}),
            'guide_conditioning': ('CONDITIONING', {'tooltip': 'Optional single native guide conditioning. Reference assets must use ports below.'}),
            'vae': ('VAE',), 'audio_vae': ('VAE',),
            'ref_image_1': ('IMAGE',), 'ref_image_2': ('IMAGE',), 'ref_image_3': ('IMAGE',),
            'ref_video_1': ('IMAGE',), 'ref_video_audio_1': ('AUDIO',), 'ref_audio_1': ('AUDIO',)}


def make_plan(prompt, context_length, context_overlap, total_frames, server_url, model_alias,
              temperature, max_tokens, timeout, seed, response_mode, cache_revision, latent=None,
              auto_start_server=True, llama_server_executable='', gguf_model='',
              server_host='127.0.0.1', server_port=8080, server_startup_timeout=180,
              server_context_size=16384, server_gpu_layers=-1):
    frames, origin, _ = latent_info(latent, total_frames)
    schedule(frames, context_length, context_overlap)  # validate before HTTP
    settings = ParserSettings(server_url, model_alias, temperature, max_tokens, timeout, seed, response_mode, cache_revision,
        auto_start_server=auto_start_server, llama_server_executable=llama_server_executable,
        gguf_model=gguf_model, server_host=server_host, server_port=server_port,
        server_startup_timeout=server_startup_timeout, server_context_size=server_context_size,
        server_gpu_layers=server_gpu_layers)
    semantic, hit = analyze(prompt, frames, context_length, context_overlap, settings)
    plan = compile_plan(prompt, semantic, frames, context_length, context_overlap)
    plan['frame_source'] = origin
    plan['analysis_cache_hit'] = hit
    return plan


def debug_outputs(plan):
    summary = {k: v for k, v in plan.items() if k not in ('source_prompt', 'window_prompts')}
    summary['notes'] = ['Frame/latent ranges are zero-based, end-exclusive.',
        'Ownership cells are semantic pacing heuristics; native physical windows overlap.',
        'Use unchanged CONDITIONING with native H3 Context Windows split_conds_to_windows=true.',
        'No guarantee of exact action timing, dialogue timing, or removal of all visual resets.']
    text = '\n\n'.join(f"Window {w['index'] + 1} | frames [{w['frame_start']},{w['frame_end']}) | conditioning region {w['conditioning_region']}\n{w['prompt']}" for w in plan['windows'])
    return json.dumps(summary, ensure_ascii=False, indent=2), text, json.dumps(plan, ensure_ascii=False, indent=2)


class H3SemanticTimelinePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': parser_inputs(), 'optional': {'latent': ('LATENT',)}}
    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('timeline_debug', 'window_prompts', 'parsed_json')
    CATEGORY = 'H3/Semantic Timeline'
    FUNCTION = 'preview'
    OUTPUT_NODE = True

    def preview(self, **kwargs):
        outputs = debug_outputs(make_plan(**kwargs))
        return {'ui': {'text': [outputs[0], outputs[1]]}, 'result': outputs}


class H3SemanticTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'clip': ('CLIP',), **parser_inputs()},
                'optional': {'latent': ('LATENT',), **encode_inputs()}}
    RETURN_TYPES = ('CONDITIONING', 'STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('conditioning', 'timeline_debug', 'window_prompts', 'parsed_json')
    CATEGORY = 'H3/Semantic Timeline'
    FUNCTION = 'execute'

    def execute(self, clip, latent=None, **kwargs):
        encoding = {key: kwargs.pop(key) for key in list(kwargs) if key in encode_inputs()}
        plan = make_plan(latent=latent, **kwargs)
        _, _, canvas = latent_info(latent, kwargs['total_frames'])
        if any(encoding.get(k) is not None for k in ('ref_image_1', 'ref_video_1')) and canvas is None:
            raise TimelineError('Connect native H3 LATENT for reference sizing.')
        cond = encode_plan(plan, clip, *(canvas or (1344, 768)), **encoding)
        return (cond, *debug_outputs(plan))


class H3SemanticTimelineEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'clip': ('CLIP',), 'parsed_json': ('STRING', {'multiline': True, 'forceInput': True}), **context_inputs()},
                'optional': {'latent': ('LATENT',), **encode_inputs()}}
    RETURN_TYPES = H3SemanticTimeline.RETURN_TYPES
    RETURN_NAMES = H3SemanticTimeline.RETURN_NAMES
    CATEGORY = H3SemanticTimeline.CATEGORY
    FUNCTION = 'execute'

    def execute(self, clip, parsed_json, context_length, context_overlap, total_frames, latent=None, **encoding):
        if len(parsed_json) > 2000000:
            raise TimelineError('Preview JSON is too large.')
        try:
            previous = json.loads(parsed_json)
            frames, origin, canvas = latent_info(latent, total_frames)
            if previous['version'] != VERSION or (previous['total_frames'], previous['context_length'], previous['context_overlap']) != (frames, context_length, context_overlap):
                raise TimelineError('Preview version/frame/context settings changed: regenerate preview.')
            plan = compile_plan(previous['source_prompt'], previous['semantic'], frames, context_length, context_overlap)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise TimelineError('Connect parsed_json from H3 Semantic Timeline Preview.') from exc
        if any(encoding.get(k) is not None for k in ('ref_image_1', 'ref_video_1')) and canvas is None:
            raise TimelineError('Connect native H3 LATENT for reference sizing.')
        plan['frame_source'] = origin
        return (encode_plan(plan, clip, *(canvas or (1344, 768)), **encoding), *debug_outputs(plan))
