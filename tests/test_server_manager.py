import errno
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import urllib.error
from dataclasses import replace
from unittest.mock import patch

from h3_semantic import server_manager as sm
from h3_semantic.parser import ParserSettings, request_chat, analyze, _CACHE
from h3_semantic.nodes import H3SemanticTimelinePreview
from h3_semantic.scheduling import TimelineError
from test_timeline import fixture, SOURCE

ROOT = Path(__file__).resolve().parents[1]
HEALTH = (200, {'status': 'ok'})
MODELS = (200, {'data': [{'id': 'h3-parser', 'owned_by': 'llamacpp'}]})
LOADING = (503, {'error': {'type': 'unavailable_error', 'code': 503, 'message': 'Loading model'}})


class Process:
    def __init__(self, output=b'loaded\n', code=None):
        self.stdout = io.BytesIO(output)
        self.returncode = code
        self.terminated = 0
        self.killed = 0
    def poll(self): return self.returncode
    def terminate(self):
        self.terminated += 1
        self.returncode = -15
    def kill(self):
        self.killed += 1
        self.returncode = -9
    def wait(self, timeout): return self.returncode


class Clock:
    def __init__(self): self.now = 100.0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class ManagerFixture(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='server tests ', dir=ROOT/'tests')
        folder = Path(self.folder.name)
        self.exe = folder/'llama server.exe'
        self.gguf = folder/'parser model.gguf'
        self.exe.write_bytes(b'mocked executable; never executed')
        self.gguf.write_bytes(b'mocked model; never loaded')
        self.exe.chmod(0o755)
        self.settings = ParserSettings(llama_server_executable=str(self.exe),gguf_model=str(self.gguf),server_startup_timeout=1)
        self.manager = sm.ServerManager()
        self.clock = Clock()
        self.patches = [patch.object(sm.time,'monotonic',self.clock.monotonic),patch.object(sm.time,'sleep',self.clock.sleep)]
        for p in self.patches: p.start()
    def tearDown(self):
        self.manager.shutdown()
        for p in reversed(self.patches): p.stop()
        self.folder.cleanup()
    def launch(self, process=None):
        proc=process or Process()
        with patch.object(sm,'probe_server',side_effect=['absent','absent','loading','ready']), patch.object(sm.subprocess,'Popen',return_value=proc) as spawn:
            self.assertEqual(self.manager.ensure(self.settings),'http://127.0.0.1:8080')
        return proc,spawn


class ManagerTests(ManagerFixture):
    def test_existing_server_reused_without_paths_or_spawn(self):
        with patch.object(sm,'probe_server',return_value='ready'),patch.object(sm.subprocess,'Popen') as spawn:
            self.manager.ensure(ParserSettings())
            self.manager.ensure(ParserSettings(auto_start_server=False))
        spawn.assert_not_called()
        self.assertIsNone(self.manager.process)
    def test_autostart_arguments_spaces_hidden_and_reuse(self):
        proc,spawn=self.launch()
        args=spawn.call_args.args[0]; kwargs=spawn.call_args.kwargs
        self.assertEqual(args[0],str(self.exe.resolve()))
        self.assertEqual(args[args.index('-m')+1],str(self.gguf.resolve()))
        self.assertEqual(args[args.index('--alias')+1],'h3-parser')
        self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
        self.assertEqual(args[args.index('--port')+1],'8080')
        self.assertFalse(kwargs['shell'])
        self.assertEqual(kwargs['stderr'],subprocess.STDOUT)
        if sm.os.name=='nt': self.assertEqual(kwargs['creationflags'],subprocess.CREATE_NO_WINDOW)
        with patch.object(sm,'probe_server',return_value='ready'),patch.object(sm.subprocess,'Popen') as again:
            self.manager.ensure(self.settings)
        again.assert_not_called(); self.assertEqual(proc.terminated,0)
    def test_manual_missing_never_spawns(self):
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'auto_start_server=false'):
                self.manager.ensure(replace(self.settings,auto_start_server=False))
        spawn.assert_not_called()
    def test_missing_executable(self):
        with patch.object(sm,'probe_server',return_value='absent'):
            with self.assertRaisesRegex(TimelineError,'llama_server_executable is missing'):
                self.manager.ensure(replace(self.settings,llama_server_executable=''))
    def test_missing_gguf(self):
        with patch.object(sm,'probe_server',return_value='absent'):
            with self.assertRaisesRegex(TimelineError,'gguf_model is missing'):
                self.manager.ensure(replace(self.settings,gguf_model=''))
    def test_invalid_paths(self):
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen') as spawn:
            for field in ('llama_server_executable','gguf_model'):
                with self.subTest(field=field),self.assertRaisesRegex(TimelineError,field):
                    self.manager.ensure(replace(self.settings,**{field:str(Path(self.folder.name)/'missing')}))
        spawn.assert_not_called()
    def test_invalid_executable_launch_error(self):
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen',side_effect=OSError(193,'not a valid application')):
            with self.assertRaisesRegex(TimelineError,'Could not launch.*193'):
                self.manager.ensure(self.settings)
        self.assertIsNone(self.manager.process)
    def test_crash_during_startup_includes_log(self):
        proc=Process(b'ERROR: missing CUDA runtime DLL\n',code=1)
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen',return_value=proc):
            with self.assertRaisesRegex(TimelineError,'(?s)exited during startup.*missing CUDA runtime DLL'):
                self.manager.ensure(self.settings)
        self.assertIsNone(self.manager.process)
    def test_readiness_timeout_terminates_only_owned(self):
        proc=Process(b'allocating model\n')
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen',return_value=proc):
            with self.assertRaisesRegex(TimelineError,'(?s)readiness timeout.*allocating model'):
                self.manager.ensure(self.settings)
        self.assertEqual(proc.terminated,1); self.assertIsNone(self.manager.process)
    def test_external_loading_wait_no_spawn_no_kill(self):
        with patch.object(sm,'probe_server',side_effect=['loading','loading','ready']),patch.object(sm.subprocess,'Popen') as spawn:
            self.manager.ensure(self.settings)
        spawn.assert_not_called(); self.manager.shutdown(); self.assertIsNone(self.manager.process)
    def test_external_loading_timeout_no_kill(self):
        with patch.object(sm,'probe_server',return_value='loading'),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'readiness timeout'): self.manager.ensure(self.settings)
        spawn.assert_not_called()
    def test_two_concurrent_calls_only_one_spawn(self):
        errors=[]; barrier=threading.Barrier(3)
        proc=Process()
        def worker():
            barrier.wait()
            try: self.manager.ensure(self.settings)
            except BaseException as exc: errors.append(exc)
        with patch.object(sm,'probe_server',side_effect=['absent','absent','loading','ready','ready']),patch.object(sm.subprocess,'Popen',return_value=proc) as spawn:
            threads=[threading.Thread(target=worker) for _ in range(2)]
            for t in threads: t.start()
            barrier.wait()
            for t in threads: t.join(timeout=3)
            self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(errors,[]); self.assertEqual(spawn.call_count,1)
    def test_external_server_wins_before_spawn(self):
        with patch.object(sm,'probe_server',side_effect=['absent','ready']),patch.object(sm.subprocess,'Popen') as spawn:
            self.manager.ensure(self.settings)
        spawn.assert_not_called()
    def test_launch_settings_change_no_silent_reuse(self):
        proc,_=self.launch()
        for settings in (replace(self.settings,gguf_model=str(self.gguf)+'other'),replace(self.settings,model='other'),replace(self.settings,server_port=8081)):
            with self.assertRaisesRegex(TimelineError,'different settings'):
                self.manager.ensure(settings)
        self.assertEqual(proc.terminated,0)
    def test_dead_owned_process_can_restart(self):
        proc,_=self.launch(); proc.returncode=1
        new_proc,spawn=self.launch()
        self.assertIs(self.manager.process,new_proc); self.assertEqual(spawn.call_count,1)
    def test_shutdown_owned_only_and_idempotent(self):
        proc,_=self.launch()
        self.manager.shutdown(); self.manager.shutdown()
        self.assertEqual(proc.terminated,1)
        external=sm.ServerManager(); external.shutdown()
        self.assertIsNone(external.process)
    def test_remote_auto_start_rejected(self):
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'loopback'):
                self.manager.ensure(replace(self.settings,server_url='http://example.test:8080'))
        spawn.assert_not_called()
    def test_wrong_service_no_spawn(self):
        with patch.object(sm,'_get_json',return_value=(200,{'name':'some other service'})),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'not a ready llama.cpp'):
                self.manager.ensure(self.settings)
        spawn.assert_not_called()
    def test_wrong_alias_no_spawn(self):
        with patch.object(sm,'_get_json',side_effect=[HEALTH,(200,{'data':[{'id':'wrong-model','owned_by':'llamacpp'}]})]),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'Wrong model'):
                self.manager.ensure(self.settings)
        spawn.assert_not_called()
    def test_request_after_spawn_succeeds(self):
        proc=Process()
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n): return b'{"choices":[{"message":{"content":"ok"}}]}'
        class Opener:
            def open(self,request,timeout): return Response()
        with patch.object(sm,'_MANAGER',self.manager),patch.object(sm,'probe_server',side_effect=['absent','absent','ready']),patch.object(sm.subprocess,'Popen',return_value=proc) as spawn,patch('h3_semantic.parser.urllib.request.build_opener',return_value=Opener()):
            self.assertEqual(request_chat(self.settings,[]),'ok')
        self.assertEqual(spawn.call_count,1)


class ProbeTests(unittest.TestCase):
    def test_health_models_ready(self):
        with patch.object(sm,'_get_json',side_effect=[HEALTH,MODELS]) as get:
            self.assertEqual(sm.probe_server('http://localhost:8080','h3-parser',sm.time.monotonic()+5),'ready')
        self.assertEqual([call.args[0] for call in get.call_args_list],['http://localhost:8080/health','http://localhost:8080/v1/models'])
    def test_health_loading(self):
        with patch.object(sm,'_get_json',return_value=LOADING):
            self.assertEqual(sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5),'loading')
    def test_connection_refused_and_timeout_absence(self):
        with patch.object(sm,'_get_json',side_effect=urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED,'refused'))):
            self.assertEqual(sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5),'absent')
        with patch.object(sm,'_get_json',side_effect=urllib.error.URLError(TimeoutError('timed out'))), patch.object(sm,'_loopback_listener_state',return_value=False):
            self.assertEqual(sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5),'absent')
        for listener in (True,None):
            with patch.object(sm,'_get_json',side_effect=urllib.error.URLError(TimeoutError('timed out'))), patch.object(sm,'_loopback_listener_state',return_value=listener), self.assertRaises(TimelineError):
                sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5)
        with patch.object(sm,'_get_json',side_effect=urllib.error.URLError(OSError('DNS failed'))), self.assertRaises(TimelineError):
            sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5)
    def test_wrong_provider_rejected(self):
        with patch.object(sm,'_get_json',side_effect=[HEALTH,(200,{'data':[{'id':'h3-parser','owned_by':'other'}]})]),self.assertRaisesRegex(TimelineError,'owned_by'):
            sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5)
    def test_invalid_models_response(self):
        for response in ((401,{}),(200,{'data':None}),(200,{'data':[]})):
            with patch.object(sm,'_get_json',side_effect=[HEALTH,response]),self.assertRaises(TimelineError):
                sm.probe_server('http://localhost','h3-parser',sm.time.monotonic()+5)
    def test_url_override_and_derived(self):
        self.assertEqual(sm.server_base_url(ParserSettings(server_port=8123)),'http://127.0.0.1:8123')
        self.assertEqual(sm.server_base_url(ParserSettings(server_host='::1')),'http://[::1]:8080')
        self.assertEqual(sm.server_base_url(ParserSettings(server_url='https://server.test/prefix/v1/chat/completions')),'https://server.test/prefix')
        for url in ('file:///secret','http://a:bad','http://user:pass@a','http://a?query=1'):
            with self.assertRaises(TimelineError): sm.server_base_url(ParserSettings(server_url=url))
    def test_http_json_and_html(self):
        class Response:
            code=200
            raw=b'{"status":"ok"}'
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n): return self.raw
        response=Response()
        class Opener:
            def open(self,*args,**kwargs): return response
        with patch.object(sm.urllib.request,'build_opener',return_value=Opener()):
            self.assertEqual(sm._get_json('http://localhost/health',1),HEALTH)
            response.raw=b'<html>wrong app</html>'
            with self.assertRaisesRegex(TimelineError,'non-JSON'): sm._get_json('http://localhost/health',1)
    def test_output_buffer_bounded_with_unbroken_lines(self):
        tail=sm.OutputTail(io.BytesIO(b'x'*200000+b'final error'))
        tail.finish()
        self.assertLessEqual(len(tail.buffer),65536)
        self.assertTrue(tail.text().endswith('final error'))
        self.assertLessEqual(len(tail.text()),6000)
    def test_cache_hit_does_not_ensure_or_spawn(self):
        _CACHE.clear()
        with patch('h3_semantic.parser.request_chat',return_value=json.dumps(fixture())):
            analyze(SOURCE,719,124,39,ParserSettings())
        with patch('h3_semantic.parser.ensure_server') as ensure:
            _,hit=analyze(SOURCE,719,124,39,ParserSettings())
        self.assertTrue(hit); ensure.assert_not_called()
        _CACHE.clear()
    def test_ui_defaults(self):
        inputs=H3SemanticTimelinePreview.INPUT_TYPES()['required']
        self.assertTrue(inputs['auto_start_server'][1]['default'])
        self.assertEqual(inputs['server_host'][1]['default'],'127.0.0.1')
        self.assertEqual(inputs['server_port'][1]['default'],8080)
        self.assertEqual(inputs['server_url'][1]['default'],'')

class LifecycleEdgeTests(ManagerFixture):
    def test_owned_slow_probe_recovers(self):
        proc=Process()
        with patch.object(sm,'probe_server',side_effect=['absent','absent',sm._ProbeTransient('slow load'),'ready']),patch.object(sm.subprocess,'Popen',return_value=proc) as spawn:
            self.manager.ensure(self.settings)
        self.assertEqual(spawn.call_count,1); self.assertEqual(proc.terminated,0)


    def test_manual_missing_with_other_owned_server(self):
        proc,_=self.launch()
        with patch.object(sm,'probe_server',return_value='absent'),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'auto_start_server=false'):
                self.manager.ensure(replace(self.settings,auto_start_server=False,server_port=8123))
        spawn.assert_not_called(); self.assertEqual(proc.terminated,0)


    def test_startup_interrupt_cleans_owned(self):
        proc=Process()
        with patch.object(sm,'probe_server',side_effect=['absent','absent',KeyboardInterrupt]),patch.object(sm.subprocess,'Popen',return_value=proc):
            with self.assertRaises(KeyboardInterrupt): self.manager.ensure(self.settings)
        self.assertEqual(proc.terminated,1); self.assertIsNone(self.manager.process)


    def test_terminate_timeout_kills_owned(self):
        proc,_=self.launch()
        def ignores_termination(): proc.terminated+=1
        proc.terminate=ignores_termination
        def wait(timeout):
            if proc.returncode is None: raise subprocess.TimeoutExpired('mock server',timeout)
            return proc.returncode
        proc.wait=wait
        self.manager.shutdown()
        self.assertEqual(proc.killed,1); self.assertIsNone(self.manager.process)


    def test_initial_uncertain_network_no_spawn(self):
        with patch.object(sm,'probe_server',side_effect=sm._ProbeTransient('unresponsive port')),patch.object(sm.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimelineError,'unresponsive port'): self.manager.ensure(self.settings)
        spawn.assert_not_called()

