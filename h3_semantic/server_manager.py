"""Thread-safe, owner-only lifecycle for a local llama.cpp HTTP server.

No model loading in Python, shell execution, installation discovery or downloads.
Only explicitly configured executable/model paths are accessed at runtime.
"""
import atexit
import errno
import ipaddress
import json
import logging
import os
import socket
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from .scheduling import TimelineError

LOG = logging.getLogger(__name__)


class _ProbeTransient(TimelineError):
    """Uncertain network failure: never a reason to spawn another server."""



def server_base_url(settings):
    raw = settings.server_url.strip()
    if not raw:
        host = settings.server_host.strip()
        if not host or any(c in host for c in '/?#@,'):
            raise TimelineError('server_host must be a hostname/IP, not a URL.')
        if type(settings.server_port) is not int or not 1 <= settings.server_port <= 65535:
            raise TimelineError('server_port must be in 1..65535.')
        if ':' in host and not host.startswith('['):
            host = '[' + host + ']'
        raw = f'http://{host}:{settings.server_port}'
    try:
        url = urllib.parse.urlsplit(raw)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('expected an HTTP(S) URL without credentials, query or fragment')
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError('invalid port')
        path = url.path.rstrip('/')
        for suffix in ('/v1/chat/completions', '/v1'):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        return urllib.parse.urlunsplit((url.scheme, url.netloc, path, '', ''))
    except ValueError as exc:
        raise TimelineError(f'Invalid server URL: {exc}') from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # another service/login redirect is not llama.cpp readiness


def _get_json(url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        status = response.code
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise TimelineError('Server probe response exceeds 1 MiB; check server_url/port.')
    try:
        return status, json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise TimelineError(f'Port responds with non-JSON data (HTTP {status}); expected llama.cpp at {url}.') from exc


def _windows_netstat_listener_state(base):
    """Return True/False for a Windows TCP listener, or None if netstat cannot be trusted."""
    if os.name != 'nt':
        return None
    try:
        url = urllib.parse.urlsplit(base)
        host = url.hostname
        if not host:
            return None
        try:
            local = host == 'localhost' or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'localhost'
        if not local:
            return None
        port = url.port or (443 if url.scheme == 'https' else 80)
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        completed = subprocess.run(
            ['netstat', '-ano', '-p', 'tcp'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.5,
            creationflags=flags,
            shell=False,
        )
        if completed.returncode != 0:
            return None
        suffix = f':{port}'
        for raw in completed.stdout.splitlines():
            parts = raw.split()
            if len(parts) < 4 or parts[0].upper() != 'TCP':
                continue
            local_endpoint = parts[1]
            state = parts[3].upper()
            if state == 'LISTENING' and local_endpoint.endswith(suffix):
                return True
        return False
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _loopback_listener_state(base, timeout=0.35):
    """Return True if loopback port accepts TCP, False if absent, None if uncertain.

    After an HTTP timeout, first try a direct TCP connect. On Windows a closed
    loopback port can occasionally time out instead of returning WSAECONNREFUSED,
    so fall back to the OS listening table before declaring the state ambiguous.
    """
    try:
        url = urllib.parse.urlsplit(base)
        host = url.hostname
        if not host:
            return None
        try:
            local = host == 'localhost' or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'localhost'
        if not local:
            return None
        port = url.port or (443 if url.scheme == 'https' else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError,):
        return False
    except OSError as exc:
        if getattr(exc, 'errno', None) in (errno.ECONNREFUSED, 10061):
            return False
        windows_state = _windows_netstat_listener_state(base)
        if windows_state is not None:
            return windows_state
        return None


def probe_server(base, alias, deadline):
    """Return ready/loading/absent while avoiding duplicate local server spawns."""
    def get(path):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ProbeTransient('Server readiness deadline expired.')
        return _get_json(base + path, timeout=min(2.0, remaining))
    try:
        code, health = get('/health')
        if code == 503 and isinstance(health, dict) and isinstance(health.get('error'), dict) and health['error'].get('type') == 'unavailable_error':
            return 'loading'
        if code != 200 or not isinstance(health, dict) or health.get('status') != 'ok':
            raise TimelineError(f'{base} is not a ready llama.cpp service (health HTTP {code}). Check the port; no process was started.')
        code, models = get('/v1/models')
        if code != 200 or not isinstance(models, dict) or not isinstance(models.get('data'), list):
            raise TimelineError(f'{base} does not provide a compatible llama.cpp /v1/models response (HTTP {code}).')
        entries = models['data']
        if not entries or not all(isinstance(m, dict) and isinstance(m.get('id'), str) and m.get('owned_by') == 'llamacpp' for m in entries):
            raise TimelineError(f'{base} does not identify its models as llama.cpp (owned_by=llamacpp). Wrong/unsupported service; no process was started.')
        available = [m['id'] for m in entries]
        if alias not in available:
            raise TimelineError(f'Wrong model at {base}: requested alias {alias!r}; available: {", ".join(available)[:500]}. Set model_alias to the desired served model; no silent fallback.')
        return 'ready'
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ConnectionRefusedError) or getattr(reason, 'errno', None) in (errno.ECONNREFUSED, 10061):
            return 'absent'
        if isinstance(reason, TimeoutError):
            listener = _loopback_listener_state(base)
            if listener is False:
                LOG.info('HTTP probe timed out at %s, but the loopback TCP port has no listener; treating server as absent.', base)
                return 'absent'
        raise _ProbeTransient(f'Cannot verify llama.cpp at {base}: {reason}. No new process is started for an ambiguous network failure.') from exc
    except TimeoutError as exc:
        listener = _loopback_listener_state(base)
        if listener is False:
            LOG.info('HTTP probe timed out at %s, but the loopback TCP port has no listener; treating server as absent.', base)
            return 'absent'
        raise _ProbeTransient(f'Server probe timed out at {base}: {exc}. A listener may still exist; no duplicate process will be started.') from exc
    except OSError as exc:
        raise _ProbeTransient(f'Server probe failed at {base}: {exc}. Check the port; no duplicate process will be started.') from exc


class OutputTail:
    """Always drain PIPE in a daemon reader; keep only the last 64 KiB."""
    LIMIT = 65536

    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._drain, name='H3-llama-output', daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                chunk = self.stream.read(4096)
                if not chunk:
                    break
                with self.lock:
                    self.buffer.extend(chunk)
                    if len(self.buffer) > self.LIMIT:
                        del self.buffer[:-self.LIMIT]
        except (OSError, ValueError):
            pass
        finally:
            self.stream.close()

    def text(self):
        with self.lock:
            return bytes(self.buffer[-6000:]).decode('utf-8', errors='replace').strip()

    def finish(self):
        self.thread.join(timeout=0.5)


def _path(text, label):
    if not text.strip():
        raise TimelineError(f'{label} is missing. Select its full file path in the node.')
    try:
        path = Path(text.strip()).resolve(strict=True)
        if not path.is_file():
            raise OSError('not a regular file')
        return path
    except (OSError, ValueError, RuntimeError) as exc:
        raise TimelineError(f'{label} is not an existing readable file: {text}') from exc


def launch_config(settings, base):
    url = urllib.parse.urlsplit(base)
    host = url.hostname
    try:
        local = host == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local or url.scheme != 'http' or url.path:
        raise TimelineError('Auto-start is limited to a local loopback HTTP address without a path prefix. Use auto_start_server=false for a remote/proxied server.')
    exe = _path(settings.llama_server_executable, 'llama_server_executable')
    model = _path(settings.gguf_model, 'gguf_model')
    if os.name == 'nt' and exe.suffix.lower() != '.exe':
        raise TimelineError('llama_server_executable must be a native .exe, not a .bat/.cmd script.')
    if model.suffix.lower() != '.gguf':
        raise TimelineError('gguf_model must point to a .gguf file.')
    if not os.access(exe, os.X_OK):
        raise TimelineError(f'llama_server_executable is not executable: {exe}')
    if type(settings.server_context_size) is not int or settings.server_context_size < 2048:
        raise TimelineError('server_context_size must be an integer >= 2048.')
    if type(settings.server_gpu_layers) is not int or settings.server_gpu_layers < -1:
        raise TimelineError('server_gpu_layers must be -1 (auto), 0 (CPU) or a positive layer count.')
    port = url.port or 80
    args = [str(exe), '-m', str(model), '--host', host, '--port', str(port),
            '--alias', settings.model, '--ctx-size', str(settings.server_context_size),
            '--parallel', '1', '--jinja', '--n-gpu-layers',
            'auto' if settings.server_gpu_layers == -1 else str(settings.server_gpu_layers)]
    return args, str(exe.parent)


class ServerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.output = None
        self.signature = None

    def _stop_owned(self):
        proc = self.process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        finally:
            if proc.poll() is not None:
                if self.output:
                    self.output.finish()
                elif proc.stdout is not None:
                    proc.stdout.close()
                self.process = None
                self.output = None
                self.signature = None

    def shutdown(self):
        with self.lock:
            try:
                self._stop_owned()
            except (OSError, subprocess.SubprocessError):
                LOG.exception('Could not shut down owned H3 llama-server')

    def ensure(self, settings):
        base = server_base_url(settings)
        if not settings.model.strip() or ',' in settings.model or any(ord(c) < 32 for c in settings.model):
            raise TimelineError('model_alias must be one non-empty model ID, without commas/control characters.')
        if type(settings.server_startup_timeout) not in (int, float) or not 1 <= settings.server_startup_timeout <= 3600:
            raise TimelineError('server_startup_timeout must be in 1..3600 seconds.')
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                self._stop_owned()
            # No automatic reconfiguration can interrupt an in-flight chat completion.
            if self.process is not None and settings.auto_start_server:
                owned_base, owned_model, exe, gguf, ctx, gpu = self.signature
                changed = (base != owned_base or settings.model != owned_model or
                           settings.server_context_size != ctx or settings.server_gpu_layers != gpu or
                           (settings.llama_server_executable.strip() and str(Path(settings.llama_server_executable.strip()).resolve()) != exe) or
                           (settings.gguf_model.strip() and str(Path(settings.gguf_model.strip()).resolve()) != gguf))
                if changed:
                    raise TimelineError('A managed H3 parser is already running with different settings. Restart ComfyUI to change its model/port/launch settings, or use manual mode for an external server.')
            deadline = time.monotonic() + settings.server_startup_timeout
            status = probe_server(base, settings.model, deadline)
            if status == 'ready':
                return base
            if status == 'absent' and not settings.auto_start_server:
                raise TimelineError(f'Cannot reach llama-server at {base}; auto_start_server=false. Start the external server or enable auto-start and set executable/GGUF paths.')
            if status == 'absent' and self.process is None:
                args, cwd = launch_config(settings, base)
                # Validate immediately again under the lock to narrow an external startup race.
                status = probe_server(base, settings.model, deadline)
                if status == 'ready':
                    return base
                if status == 'absent':
                    try:
                        self.process = subprocess.Popen(args, cwd=cwd, shell=False,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
                    except (OSError, ValueError) as exc:
                        raise TimelineError(f'Could not launch llama_server_executable: {exc}. Check the native executable, permissions and its DLLs.') from exc
                    self.signature = (base, settings.model, args[0], args[2], settings.server_context_size, settings.server_gpu_layers)
                    try:
                        self.output = OutputTail(self.process.stdout)
                    except BaseException:
                        self._stop_owned()
                        raise
            # External loading servers are waited on but are never owned or terminated.
            owned = self.process is not None and self.signature[0] == base
            try:
                while time.monotonic() < deadline:
                    if owned and self.process.poll() is not None:
                        self.output.finish()
                        raise TimelineError(f'llama-server exited during startup with code {self.process.returncode}.')
                    try:
                        status = probe_server(base, settings.model, deadline)
                    except _ProbeTransient:
                        # A known starting process may briefly stall while loading.
                        status = 'loading'
                    if status == 'ready':
                        if owned and self.process.poll() is not None:
                            raise TimelineError(f'llama-server exited during startup with code {self.process.returncode}.')
                        return base
                    time.sleep(min(0.2, max(0, deadline - time.monotonic())))
                raise TimelineError(f'llama-server readiness timeout after {settings.server_startup_timeout:g}s at {base}. Increase server_startup_timeout for a large model.')
            except BaseException as exc:
                detail = ''
                if owned:
                    output = self.output
                    try:
                        self._stop_owned()
                    except (OSError, subprocess.SubprocessError) as stop_error:
                        LOG.warning('Could not stop failed owned server: %s', stop_error)
                    if output:
                        detail = output.text()
                if isinstance(exc, TimelineError) and detail:
                    raise TimelineError(f'{exc}\nLast llama-server output:\n{detail}') from exc
                raise


_MANAGER = ServerManager()
atexit.register(_MANAGER.shutdown)


def ensure_server(settings):
    return _MANAGER.ensure(settings)
