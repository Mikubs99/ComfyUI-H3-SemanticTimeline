import ast
import copy
import json
import pathlib
import types
import unittest
from unittest.mock import patch
from h3_semantic.scheduling import TimelineError, schedule, video_latent_t, region_layout, latent_info
from h3_semantic.semantic import CATEGORIES, compile_plan, parse_json, validate
from h3_semantic.parser import ParserSettings, analyze, endpoint, request_chat, _CACHE
from h3_semantic.conditioning import encode_plan
from h3_semantic.nodes import H3SemanticTimelinePreview, H3SemanticTimelineEncode, H3SemanticTimeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = 'A woman in a black sweater sits on a sofa speaking to camera. She stands up, walks toward the window while continuing to speak, looks outside, then turns back to camera.'


def event(eid, action, before, after, kind='transition', weight=1, active=None, dialogue=None):
    return {'id': eid, 'kind': kind, 'start_state': before, 'action': action,
            'continuation': {'She speaks to camera.': 'She continues speaking to the camera.', 'She walks toward the window.': 'She continues walking toward the window.'}.get(action, 'The ongoing activity continues.'), 'active_state': active or {},
            'end_state': after, 'weight': weight, 'depends_on': [], 'dialogue': dialogue or []}


def fixture():
    return {'persistent': {**{k: [] for k in CATEGORIES}, 'subject': ['The same woman.'],
        'clothing': ['Black sweater.'], 'environment': ['The same living room.'], 'audio': ['She continues speaking naturally.']},
        'initial_state': {'posture': 'She is seated.', 'location': 'She is at the sofa.', 'facing': 'She faces the camera.'},
        'events': [event('speak', 'She speaks to camera.', {}, {}, 'ongoing', 2),
                   event('rise', 'She stands up.', {'posture': 'She is seated.'}, {'posture': 'She is already standing.'}),
                   event('walk', 'She walks toward the window.', {'location': 'She is at the sofa.'}, {'location': 'She has reached the window.'}, 'ongoing', 3, {'location': 'She is approaching the window.'}),
                   event('look', 'She looks outside.', {'facing': 'She faces the camera.'}, {'facing': 'She faces outside.'}),
                   event('turn', 'She turns back toward camera.', {'facing': 'She faces outside.'}, {'facing': 'She faces the camera.'})]}


def build(data=None, source=SOURCE, frames=719, length=124, overlap=39):
    return compile_plan(source, data or fixture(), frames, length, overlap)


class Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def latent(frames=719):
    return {'samples': types.SimpleNamespace(is_nested=True, tensors=[Tensor((1,24,video_latent_t(frames),48,84)), Tensor((1,32,2,round(frames*40/24)))])}


class Clip:
    def __init__(self):
        self.calls = []
    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return text
    def encode_from_tokens_scheduled(self, text):
        return [[text, {'minimax_token_tags': 'tags:' + text, 'pooled_output': 'pool:' + text}]]


class SchedulingTests(unittest.TestCase):
    def test_known_golden_ranges(self):
        w, length, overlap = schedule(362,124,39)
        self.assertEqual((length,overlap),(37,12))
        self.assertEqual([(x.latent_start,x.latent_end) for x in w], [(0,37),(25,62),(50,87),(70,107)])
        self.assertEqual([(x.frame_start,x.frame_end) for x in w], [(0,124),(85,209),(170,294),(238,362)])
    def test_actual_upstream_parity(self):
        path = ROOT/'research/comfy__context_windows.py'
        if not path.exists():
            self.skipTest('Pinned source parity requires development research/ directory.')
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_windows_static_standard')
        func.returns=None
        for arg in func.args.args: arg.annotation=None
        ns={}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), ns)
        for frames in range(5,1441,17):
            for length in (22,56,124,362):
                for overlap in (5,39,124,362):
                    w,l,o=schedule(frames,length,overlap)
                    actual=ns['create_windows_static_standard'](video_latent_t(frames), types.SimpleNamespace(context_length=l,context_overlap=o),{})
                    self.assertEqual(actual,[list(range(x.latent_start,x.latent_end)) for x in w])
                    slots,indices=region_layout(w)
                    self.assertEqual(len(set(indices)),len(w))
                    self.assertEqual([slots[min(int(x.center_ratio*len(slots)),len(slots)-1)] for x in w],list(range(len(w))))
    def test_region_collision_is_resolved(self):
        # Heavy overlap: original one-entry-per-window list maps multiple windows together.
        w,_,_=schedule(141,124,107)
        self.assertEqual(len(w),2)
        self.assertEqual([int(x.center_ratio*2) for x in w],[0,1])
        w,_,_=schedule(158,124,107)
        naive=[int(x.center_ratio*len(w)) for x in w]
        self.assertLess(len(set(naive)),len(w))
        slots, indices=region_layout(w)
        self.assertEqual(len(set(indices)),len(w))
        self.assertGreater(len(slots),len(w))
    def test_single_window_one_entry(self):
        self.assertEqual(build(frames=22)['conditioning_slots'],[0])
    def test_native_latent(self):
        self.assertEqual(latent_info(latent(),5)[0],719)
        self.assertEqual(latent_info(None,720)[0],736)
    def test_unknown_latent_rejected(self):
        with self.assertRaises(TimelineError): latent_info({'samples':Tensor((1,4,99,8,8))},719)
        bad=latent(); bad['samples'].tensors[1].shape=(1,32,2,1)
        with self.assertRaises(TimelineError): latent_info(bad,719)
    def test_invalid_grid(self):
        for length,overlap in [(125,39),(124,0),(5,5)]:
            with self.assertRaises(TimelineError): schedule(719,length,overlap)


class SemanticTests(unittest.TestCase):
    def test_linear(self):
        plan=build()
        self.assertEqual(sum('She stands up.' in x for x in plan['window_prompts']),1)
        rise=next(i for i,x in enumerate(plan['window_prompts']) if 'She stands up.' in x)
        self.assertTrue(all('She is seated.' not in p for p in plan['window_prompts'][rise+1:]))
        self.assertTrue(any('She is approaching the window.' in x for x in plan['window_prompts']))
    def test_enhancer_prompt(self):
        data=fixture(); source='A cinematic medium shot of a woman wearing a black sweater sitting on a sofa in a warmly lit living room. '+SOURCE
        data['persistent']['lighting']=['Warm afternoon lighting.']
        data['persistent']['camera']=['A cinematic medium shot.']
        plan=build(data,source)
        self.assertTrue(all('Warm afternoon lighting.' in p and 'cinematic medium shot' in p for p in plan['window_prompts']))
    def test_dialogue_once(self):
        source=SOURCE+' She says "Welcome home!"'
        data=fixture(); data['events'][0]['dialogue']=['"Welcome home!"']
        plan=build(data,source)
        self.assertEqual(sum(p.count('"Welcome home!"') for p in plan['window_prompts']),1)
    def test_dialogue_loss_or_invention_rejected(self):
        with self.assertRaises(TimelineError): build(source=SOURCE+' "Hello"')
        data=fixture(); data['events'][0]['dialogue']=['"Invented"']
        with self.assertRaises(TimelineError): build(data)
    def test_camera_actions(self):
        data=fixture(); data['initial_state']['camera']='The camera holds a medium shot.'
        data['events'].append(event('camera','The camera pans to the window.',{'camera':'The camera holds a medium shot.'},{'camera':'The camera now frames the window.'}))
        prompts=build(data)['window_prompts']
        self.assertEqual(sum('The camera pans to the window.' in p for p in prompts),1)
    def test_more_events_than_windows(self):
        plan=build(frames=22)
        prompt=plan['window_prompts'][0]
        positions=[prompt.index(e['action']) for e in fixture()['events']]
        self.assertEqual(positions,sorted(positions))
    def test_fewer_events_than_windows(self):
        data=fixture(); data['events']=[data['events'][1]]
        prompts=build(data,frames=1433)['window_prompts']
        self.assertEqual(sum('She stands up.' in p for p in prompts),1)
        self.assertTrue(all('already standing' in p for p in prompts[1:]))
    def test_reference_markers(self):
        data=fixture(); data['persistent']['references']=['The woman in <Picture 1>, movement from <Video 1>, voice of <Audio 1>.']
        source=SOURCE+' '+data['persistent']['references'][0]
        prompts=build(data,source)['window_prompts']
        self.assertTrue(all(all(m in p for m in ('<Picture 1>','<Video 1>','<Audio 1>')) for p in prompts))
    def test_reference_loss_is_repaired(self):
        prompts=build(source=SOURCE+' <Picture 1>')['window_prompts']
        self.assertTrue(all('<Picture 1>' in p for p in prompts))
    def test_no_temporal_events(self):
        data=fixture(); data['events']=[]
        plan=build(data, 'A woman in a black sweater sits in a living room.')
        self.assertEqual(len(set(plan['window_prompts'])),1)
    def test_30_60_seconds(self):
        for frames in (719,1433):
            plan=build(frames=frames)
            self.assertEqual(plan['windows'][-1]['frame_end'],frames)
            self.assertGreater(len(plan['windows']),6)
            self.assertIn('She faces the camera.',plan['windows'][-1]['end_state']['facing'])
    def test_invalid_json_and_state_dependencies(self):
        for raw in ('garbage','{}','{"persistent": {}, "persistent": {}}'):
            with self.assertRaises(TimelineError): parse_json(raw,SOURCE)
        for change in ('state','cycle','nan'):
            data=fixture()
            if change=='state': data['events'][2]['start_state']['location']='Already at window'
            if change=='cycle': data['events'][0]['depends_on']=['turn']
            if change=='nan': data['events'][0]['weight']=float('nan')
            with self.assertRaises(TimelineError): validate(data,SOURCE)


class ParserTests(unittest.TestCase):
    def setUp(self): _CACHE.clear()
    def test_repair_once(self):
        expected=parse_json(json.dumps(fixture()),SOURCE)
        with patch('h3_semantic.parser.request_chat',side_effect=['bad',json.dumps(fixture())]) as send:
            parsed,hit=analyze(SOURCE,719,124,39,ParserSettings())
            self.assertEqual(parsed,expected); self.assertFalse(hit); self.assertEqual(send.call_count,2)
    def test_repair_failure_no_fallback(self):
        with patch('h3_semantic.parser.request_chat',return_value='bad') as send:
            with self.assertRaisesRegex(TimelineError,'after one repair'): analyze(SOURCE,719,124,39,ParserSettings())
            self.assertEqual(send.call_count,2); self.assertFalse(_CACHE)
    def test_cache_and_invalidation(self):
        with patch('h3_semantic.parser.request_chat',return_value=json.dumps(fixture())) as send:
            args=(SOURCE,719,124,39,ParserSettings())
            data,_=analyze(*args); data['events'].clear()
            again,hit=analyze(*args)
            self.assertTrue(hit); self.assertEqual(len(again['events']),5)
            analyze(SOURCE,1433,124,39,ParserSettings())
            analyze(SOURCE,719,124,56,ParserSettings())
            analyze(SOURCE,719,124,39,ParserSettings(cache_revision=1))
            self.assertEqual(send.call_count,4)
    def test_endpoints(self):
        for url in ('http://localhost:8080','http://localhost:8080/v1','http://localhost:8080/v1/chat/completions'):
            self.assertEqual(endpoint(url),'http://localhost:8080/v1/chat/completions')
        with self.assertRaises(TimelineError): endpoint('file:///outside')
    def test_http_body_and_response(self):
        captured={}
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n): return json.dumps({'choices':[{'message':{'content':json.dumps(fixture())},'finish_reason':'stop'}]}).encode()
        class Opener:
            def open(self,request,timeout):
                captured.update(json.loads(request.data)); return Response()
        with patch('urllib.request.build_opener',return_value=Opener()), patch('h3_semantic.parser.ensure_server',return_value='http://127.0.0.1:8080'):
            self.assertEqual(json.loads(request_chat(ParserSettings(),[])),fixture())
        self.assertFalse(captured['stream']); self.assertIn('schema',captured['response_format'])
        self.assertFalse(captured['chat_template_kwargs']['enable_thinking'])


class EncodingTests(unittest.TestCase):
    @patch('h3_semantic.conditioning.check_upstream')
    def test_native_selection_and_metadata(self,check):
        plan=build(frames=158,length=124,overlap=107); clip=Clip()
        cond=encode_plan(plan,clip)
        self.assertEqual(len(clip.calls),len(set(plan['window_prompts'])))
        for i,region in enumerate(plan['selected_regions']):
            self.assertEqual(cond[region][0],plan['window_prompts'][i])
            self.assertEqual(cond[region][1]['minimax_token_tags'],'tags:'+plan['window_prompts'][i])
        self.assertEqual(len({id(c[1]) for c in cond}),len(cond))
    @patch('h3_semantic.conditioning.check_upstream')
    def test_native_reference_preparation_once(self,check):
        data=fixture(); data['persistent']['references']=['Same woman from <Picture 1>.']
        plan=build(data,SOURCE+' <Picture 1>'); clip=Clip(); image=object(); calls=[]
        class Native:
            @classmethod
            def execute(cls,proxy,prompt,*args,**kwargs):
                calls.append(kwargs)
                tokens=proxy.tokenize(prompt,minimax_ref_items=[{'type':'image','data':image}])
                cond=proxy.encode_from_tokens_scheduled(tokens)
                cond[0][1]['minimax_refs']=[{'kind':'image','latent':'ref'}]
                return types.SimpleNamespace(result=(cond,{}))
        module=types.ModuleType('comfy_extras.nodes_minimax_h3'); module.MiniMaxH3ReferenceToVideo=Native
        with patch.dict('sys.modules',{'comfy_extras':types.ModuleType('comfy_extras'),'comfy_extras.nodes_minimax_h3':module}):
            cond=encode_plan(plan,clip,ref_image_1=image)
        self.assertEqual(len(calls),1)
        self.assertTrue(all(c[1]['minimax_refs'][0]['latent']=='ref' for c in cond))
        self.assertTrue(all(k['minimax_ref_items'][0]['data'] is image for _,k in clip.calls))
    @patch('h3_semantic.conditioning.check_upstream')
    def test_missing_asset_rejected(self,check):
        data=fixture(); data['persistent']['references']=['<Picture 1>']
        with self.assertRaises(TimelineError): encode_plan(build(data,SOURCE+' <Picture 1>'),Clip())
    @patch('h3_semantic.conditioning.check_upstream')
    def test_guide_preservation_and_rejection(self,check):
        guides=[['old text',{'minimax_keyframes':[{'resolved_frame_index':0,'latent':'image'}], 'minimax_token_tags':'stale'}]]
        cond=encode_plan(build(),Clip(),guide_conditioning=guides)
        self.assertEqual(cond[0][1]['minimax_keyframes'],guides[0][1]['minimax_keyframes'])
        self.assertNotEqual(cond[0][1]['minimax_token_tags'],'stale')
        guides[0][1]['minimax_refs']=[{'kind':'image'}]
        with self.assertRaises(TimelineError): encode_plan(build(),Clip(),guide_conditioning=guides)
    @patch('h3_semantic.conditioning.check_upstream')
    def test_multi_entry_encoder_rejected(self,check):
        clip=Clip(); clip.encode_from_tokens_scheduled=lambda _: [['a',{}],['b',{}]]
        with self.assertRaises(TimelineError): encode_plan(build(),clip)
    @patch('h3_semantic.conditioning.check_upstream')
    def test_preview_encode_revalidates_and_no_llm(self,check):
        plan=build(); plan['conditioning_slots']=[99999]
        result=H3SemanticTimelineEncode().execute(Clip(),json.dumps(plan),124,39,719)
        self.assertEqual(len(result),4)
        with self.assertRaises(TimelineError): H3SemanticTimelineEncode().execute(Clip(),json.dumps(plan),124,39,1433)
    @patch('h3_semantic.nodes.analyze',return_value=(fixture(),False))
    def test_preview_has_no_clip_dependency(self,analyze_mock):
        defaults={k:v[1]['default'] if len(v)>1 else v[0][0] for k,v in H3SemanticTimelinePreview.INPUT_TYPES()['required'].items()}
        defaults.update(prompt=SOURCE,total_frames=719)
        out=H3SemanticTimelinePreview().preview(**defaults)
        self.assertIn('ui',out); self.assertEqual(len(out['result']),3)

class IntegrationTests(unittest.TestCase):
    @patch('h3_semantic.conditioning.check_upstream')
    @patch('h3_semantic.nodes.analyze',return_value=(fixture(),False))
    def test_main_node_end_to_end_mocked(self,analyze_mock,check):
        defaults={k:v[1]['default'] if len(v)>1 else v[0][0] for k,v in H3SemanticTimelinePreview.INPUT_TYPES()['required'].items()}
        defaults.update(prompt=SOURCE,total_frames=5)
        clip=Clip()
        cond,debug,prompts,parsed=H3SemanticTimeline().execute(clip,latent=latent(),**defaults)
        plan=json.loads(parsed)
        self.assertEqual(plan['total_frames'],719)
        self.assertEqual(len(cond),len(plan['conditioning_slots']))
        self.assertEqual(analyze_mock.call_args.args[1],719)
    def test_registered_package_imports_without_comfy(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('h3_test_package',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'h3_test_package':module}):
            spec.loader.exec_module(module)
        self.assertEqual(len(module.NODE_CLASS_MAPPINGS),3)
        self.assertEqual(module.__version__,'0.1.1')
    def test_upstream_region_and_frame_conversion(self):
        path=ROOT/'research/comfy__context_windows.py'
        if not path.exists(): self.skipTest('Needs pinned research sources.')
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IndexListContextWindow')
        methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','get_region_index')]
        cls=ast.ClassDef(name='NativeWindow',bases=[],keywords=[],body=methods,decorator_list=[])
        namespace={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
        path2=ROOT/'research/comfy_extras__nodes_minimax_h3.py'
        tree2=ast.parse(path2.read_text(encoding='utf-8-sig'))
        funcs=[n for n in tree2.body if isinstance(n,ast.FunctionDef) and n.name in ('video_latent_t','align_frame_count')]
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path2),'exec'),namespace)
        for frames in range(5,1441,17):
            w,l,o=schedule(frames,124,107)
            slots,selected=region_layout(w)
            self.assertEqual(video_latent_t(frames),namespace['video_latent_t'](frames))
            for window,region in zip(w,selected):
                native=namespace['NativeWindow'](list(range(window.latent_start,window.latent_end)),dim=2,total_frames=video_latent_t(frames))
                self.assertEqual(native.get_region_index(len(slots)),region)
    def test_runtime_guard_and_mismatch(self):
        from h3_semantic.scheduling import check_upstream
        class NativeWindow:
            def __init__(self,idx,dim,total_frames): self.ratio=(min(idx)+max(idx))/(2*total_frames)
            def get_region_index(self,n): return min(max(int(self.ratio*n),0),n-1)
        class MiniMax:
            @staticmethod
            def _latent_pixel_starts(t):
                starts=[0]
                for k in range(t): starts.append(starts[-1]+(1,4,4,4,4)[k%5])
                return starts
        w,l,o=schedule(719,124,39)
        comfy=types.ModuleType('comfy'); cw=types.ModuleType('comfy.context_windows')
        cw.create_windows_static_standard=lambda *args:[list(range(x.latent_start,x.latent_end)) for x in w]
        cw.IndexListContextWindow=NativeWindow
        cw.ContextSchedules=types.SimpleNamespace(STATIC_STANDARD='static_standard')
        comfy.context_windows=cw
        h3=types.ModuleType('comfy_extras.nodes_minimax_h3'); h3.video_latent_t=video_latent_t
        node=types.ModuleType('comfy_extras.nodes_context_windows'); node.MiniMaxH3ContextWindowsNode=object
        modelbase=types.ModuleType('comfy.model_base'); modelbase.MiniMaxH3=MiniMax
        modules={'comfy':comfy,'comfy.context_windows':cw,'comfy.model_base':modelbase,'comfy_extras':types.ModuleType('comfy_extras'),'comfy_extras.nodes_minimax_h3':h3,'comfy_extras.nodes_context_windows':node}
        handler=types.SimpleNamespace(split_conds_to_windows=True,context_length=l,context_overlap=o,dim=2,causal_window_fix=False,context_schedule=types.SimpleNamespace(name='static_standard'))
        model=types.SimpleNamespace(model_options={'context_handler':handler})
        with patch.dict('sys.modules',modules):
            check_upstream(w,719,l,o,model)
            handler.context_overlap=1
            with self.assertRaisesRegex(TimelineError,'settings differ'): check_upstream(w,719,l,o,model)
            cw.create_windows_static_standard=lambda *args:[[0,1]]
            with self.assertRaisesRegex(TimelineError,'scheduling changed'): check_upstream(w,719,l,o)


class QuoteRegressionTests(unittest.TestCase):
    def test_dialogue_with_apostrophe_not_repeated_in_persistent(self):
        quote='"I am here; don\'t worry."'
        data=fixture()
        data['events'][0]['dialogue']=[quote]
        data['persistent']['audio']=[quote]
        with self.assertRaisesRegex(TimelineError,'duplicate dialogue'):
            validate(data,SOURCE+' '+quote)
