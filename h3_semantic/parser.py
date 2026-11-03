"""Bounded, cached llama.cpp HTTP parser. Server lifecycle is delegated to server_manager."""
import copy
import hashlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import asdict, dataclass
from .scheduling import TimelineError
from .server_manager import ensure_server
from .semantic import SCHEMA, SYSTEM, VERSION, parse_json


@dataclass(frozen=True)
class ParserSettings:
    server_url: str = ''
    model: str = 'h3-parser'
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: int = 180
    seed: int = 42
    response_mode: str = 'json_schema'
    cache_revision: int = 0
    auto_start_server: bool = True
    llama_server_executable: str = ''
    gguf_model: str = ''
    server_host: str = '127.0.0.1'
    server_port: int = 8080
    server_startup_timeout: int = 180
    server_context_size: int = 16384
    server_gpu_layers: int = -1


_CACHE = OrderedDict()
_LOCK = threading.RLock()


def endpoint(url):
    p = urllib.parse.urlsplit(url.strip())
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise TimelineError('server_url must be an HTTP(S) base URL without credentials/query/fragment.')
    path = p.path.rstrip('/')
    if path.endswith('/chat/completions'):
        return urllib.parse.urlunsplit((p.scheme, p.netloc, path, '', ''))
    return urllib.parse.urlunsplit((p.scheme, p.netloc, path + ('/chat/completions' if path.endswith('/v1') else '/v1/chat/completions'), '', ''))


def request_chat(settings, messages):
    body = {'model': settings.model, 'messages': messages, 'temperature': settings.temperature,
            'max_tokens': settings.max_tokens, 'seed': settings.seed, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    if settings.response_mode == 'json_schema':
        body['response_format'] = {'type': 'json_object', 'schema': SCHEMA}
    elif settings.response_mode == 'json_object':
        body['response_format'] = {'type': 'json_object'}
    elif settings.response_mode != 'prompt_only':
        raise TimelineError('Unknown parser response mode.')
    base_url = ensure_server(settings)
    request = urllib.request.Request(endpoint(base_url), data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        # Loopback must not accidentally go through an OS-configured HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=settings.timeout) as response:
            raw = response.read(1048577)
        if len(raw) > 1048576:
            raise TimelineError('llama-server response exceeds 1 MiB.')
        envelope = json.loads(raw)
        choice = envelope['choices'][0]
        content = choice['message']['content']
        if not isinstance(content, str):
            raise TimelineError('llama-server returned no text content. Disable thinking for Qwen3.')
        return content
    except urllib.error.HTTPError as exc:
        raise TimelineError(f'llama-server HTTP {exc.code}. Check model alias and response_mode; no settings were changed automatically.') from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TimelineError(f'Connection to llama-server at {base_url} failed after readiness. Check server output and request timeout; retry the workflow if it exited.') from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TimelineError(f'Invalid chat-completions response: {exc}') from exc


def analyze(source, frames, context_length, context_overlap, settings):
    if not isinstance(source, str) or not source.strip() or len(source) > 60000:
        raise TimelineError('prompt must contain 1..60000 characters.')
    key = hashlib.sha256(json.dumps([VERSION, source, frames, context_length, context_overlap, asdict(settings)], sort_keys=True).encode()).hexdigest()
    # Serialize misses as well: simultaneous executions do not duplicate an analysis.
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return copy.deepcopy(_CACHE[key]), True
        messages = [{'role': 'system', 'content': SYSTEM + json.dumps(SCHEMA)}, {'role': 'user', 'content': source}]
        raw = request_chat(settings, messages)
        try:
            parsed = parse_json(raw, source)
        except TimelineError as first:
            messages += [{'role': 'assistant', 'content': raw[:256000]}, {'role': 'user', 'content': 'Repair the JSON once. Original source is unchanged. Validation error: ' + str(first) + '\nReturn the complete corrected JSON only.'}]
            repaired = request_chat(settings, messages)
            try:
                parsed = parse_json(repaired, source)
            except TimelineError as second:
                raise TimelineError(f'H3 semantic parsing failed after one repair. No conditioning was produced. Check model and max_tokens/server context budget. {second}') from second
        _CACHE[key] = copy.deepcopy(parsed)
        while len(_CACHE) > 16:
            _CACHE.popitem(last=False)
        return parsed, False

