"""Static public pages: offline fixtures only, no model calls or live-run claims.

The Node VM checks exercise the actual inline scripts with a minimal DOM. Node is
optional for Python-only environments; deployment does not need Node or a build.
"""
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / 'site'
PR_URL = 'https://github.com/mappmechanic/frame-foundry/pull/1'


class Page(HTMLParser):
    def __init__(self, name):
        super().__init__()
        self.source = (SITE / name).read_text()
        self.tags = []
        self.feed(self.source)
        self.script = re.search(r'<script>(.*?)</script>', self.source, re.S).group(1)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def element(self, id):
        return next(attrs for _, attrs in self.tags if attrs.get('id') == id)


@pytest.fixture(params=['index.html', 'studio.html'])
def page(request):
    return Page(request.param)


def test_pages_are_self_contained_and_accessible(page):
    assert next(attrs for tag, attrs in page.tags if tag == 'html')['lang'] == 'en'
    assert any(tag == 'meta' and attrs.get('name') == 'viewport' for tag, attrs in page.tags)
    assert len([tag for tag, _ in page.tags if tag == 'main']) == 1
    assert '<a class="skip" href="#main">' in page.source
    assert 'prefers-reduced-motion:reduce' in page.source
    assert ':focus-visible' in page.source
    assert not any(tag in ('iframe', 'form', 'link') for tag, _ in page.tags)
    assert not any(tag == 'script' and 'src' in attrs for tag, attrs in page.tags)
    assert not any(attrs.get('src', '').startswith(('http', '//')) for _, attrs in page.tags)
    ids = [attrs['id'] for _, attrs in page.tags if 'id' in attrs]
    assert len(ids) == len(set(ids))
    for id in re.findall(r"(?:byId|put)\('([^']+)'\s*[,)]", page.script):
        assert id in ids


def test_media_requires_user_playback(page):
    video = page.element('run-video')
    assert all(attr in video for attr in ('controls', 'playsinline'))
    assert video['preload'] == 'metadata'
    assert 'autoplay' not in video and 'src' not in video
    assert '.play(' not in page.script


def test_csp_and_read_only_network(page):
    csp = next(attrs['content'] for tag, attrs in page.tags if attrs.get('http-equiv') == 'Content-Security-Policy')
    for rule in ("default-src 'none'", "connect-src 'self'", "form-action 'none'", "base-uri 'none'", "media-src 'self'"):
        assert rule in csp
    assert "method:'GET'" in page.script and "redirect:'error'" in page.script
    assert "credentials:'omit'" in page.script
    assert not re.search(r'localhost|127\.0\.0\.1|POST|XMLHttpRequest|WebSocket|sendBeacon|innerHTML|outerHTML|insertAdjacentHTML', page.script)
    assert 'textContent' in page.script
    assert 'source_render_sha256' not in page.script
    assert 'run.goal' not in page.script and 'run.source' not in page.script


def test_landing_content_and_genuine_pr():
    page = Page('index.html')
    for text in ('Raw footage + one-line Goal', 'Finished vertical clip', 'Refinery', 'Planner', 'Assembler', 'Validator', 'Judge', '8090 Software Factory', 'Ticket → work order → PR', 'Q/$', 'Render QA pass rate', 'not completion, delivery, or merge'):
        assert text in page.source
    external = [attrs['href'] for tag, attrs in page.tags if tag == 'a' and attrs.get('href', '').startswith('https:')]
    assert external == [PR_URL]
    assert any(tag == 'a' and attrs.get('href') == '/studio' for tag, attrs in page.tags)


def test_studio_inputs_cannot_process_files():
    page = Page('studio.html')
    assert 'disabled' in page.element('upload')
    assert 'readonly' in page.element('goal')
    assert 'Create a 45-second family-trip teaser' in page.source
    assert 'renders run locally on the Mac; this is a recorded demo' in page.source
    assert 'disabled' in next(attrs for tag, attrs in page.tags if tag == 'button')
    assert "document.addEventListener('drop',event => event.preventDefault())" in page.script
    assert 'FileReader' not in page.script and 'FormData' not in page.script
    assert 'loading' not in page.element('contact-sheet')  # Hidden lazy image would never load.


# Execute scripts without browsers, dependencies, network, or backend processes.
VM = r"""
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
  constructor(attrs={}) { this.attrs={...attrs}; this.children=[]; this.textContent=''; this.hidden='hidden' in attrs; this.disabled='disabled' in attrs; this.value=''; this.handlers={}; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=nodes; }
  removeAttribute(key) { delete this.attrs[key]; }
  setAttribute(key,value) { this.attrs[key]=value; }
  getAttribute(key) { return this.attrs[key] ?? null; }
  addEventListener(key,handler) { this.handlers[key]=handler; }
  pause() {} load() {}
}
const elements=Object.fromEntries(input.elements.map(attrs=>[attrs.id,new Element(attrs)]));
const requests=[], pending={}, fixtures=input.fixtures || {};
const sandbox={document:{getElementById:id=>{if(!elements[id]) throw Error('Unknown element '+id);return elements[id]},createElement:()=>new Element(),addEventListener:()=>{}}, fetch:async(path,opts)=>{
  requests.push({path,opts});
  if(input.defer?.includes(path)) return new Promise(resolve=>{pending[path]=resolve});
  if(!(path in fixtures)) throw Error('Offline fixture missing');
  return {ok:true,json:async()=>fixtures[path]};
}};
const ctx=vm.createContext(sandbox);
vm.runInContext(input.script,ctx);
(async()=>{
  for(let i=0;i<8;i++) await new Promise(setImmediate);
  let result=null;
  if(input.action) result=await vm.runInContext(input.action,ctx);
  for(let i=0;i<8;i++) await new Promise(setImmediate);
  const allText=node=>[node.textContent,...node.children.map(allText)].join(' ');
  console.log(JSON.stringify({result,requests,elements:Object.fromEntries(Object.entries(elements).map(([id,e])=>[id,{text:allText(e),src:e.attrs.src??null,hidden:e.hidden,disabled:e.disabled,value:e.value,children:e.children.length}]))}));
})().catch(error=>{console.error(error);process.exit(1)});
"""


def execute(page, fixtures=None, action=None, defer=None):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional; required to execute inline-JS regression tests')
    payload = {'script': page.script, 'elements': [attrs for _, attrs in page.tags if 'id' in attrs], 'fixtures': fixtures or {}, 'action': action, 'defer': defer or []}
    run = subprocess.run([node, '-e', VM], input=json.dumps(payload), text=True, capture_output=True, timeout=10)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def summary(id='az-0835', created='2026-11-14T08:35:00Z'):
    return dict(id=id, created_at=created, data_url=f'/runs/{id}/run.json', video_url=f'/runs/{id}/finished.mp4', contact_sheet_url=f'/runs/{id}/contact-sheet.jpg', scores=dict(J=.75,Q=.8,D=.9), cost=dict(usd=2,known_usd=2,status='estimated',quality_per_dollar=.4))


def index(runs, lead=None):
    return dict(schema_version='1.0',mode='recorded-demo',runs=runs,lead_run_id=lead,latest_run_id=None)


def run_data(id='az-0835'):
    return dict(**summary(id), first_try_passed=False, elapsed_seconds=30,
                output=dict(width=720,height=1280,codec='h264',duration_seconds=45),
                judge=dict(status='scored',verdict='pass',scores=dict(goal_fit=4),evidence=dict(goal_fit=['shot-01','shot-03']),weakest_shot='shot-03'),
                qa=dict(passed=True,integrated_lufs=-14,true_peak_dbtp=-1,checks=[dict(name='Loudness',status='pass',reason='Within target')]),
                gates=dict(passed=True,checks=[dict(name='Integrity',status='pass',reason=None)]))


def test_missing_network_is_explicit_and_never_generates(page):
    out = execute(page)
    assert 'unavailable' in out['elements']['index-state']['text']
    assert out['elements']['run-video']['src'] is None
    assert all(req['opts']['method'] == 'GET' and req['path'].startswith('/runs/') for req in out['requests'])


def test_empty_export_has_no_invented_media(page):
    out = execute(page, {'/runs/index.json': index([])})
    assert 'No recorded clips exported yet' in out['elements']['index-state']['text']
    assert out['elements']['run-video']['src'] is None


@pytest.mark.parametrize('unsafe', ['../outside', '/tmp/private', 'az/0835', 'UPPER', 'x%2fy', '', 'x'*81])
def test_unsafe_ids_are_never_requested_or_shown(unsafe):
    page = Page('studio.html')
    out = execute(page, {'/runs/index.json': index([summary(unsafe)])})
    assert [r['path'] for r in out['requests']] == ['/runs/index.json', '/runs/campaigns.json']
    assert out['elements']['run-picker']['disabled']
    assert out['elements']['run-heading']['text'] == ''  # Initial heading, untouched by unsafe IDs.


def test_newest_fallback_and_lead_default():
    page = Page('studio.html')
    old,new = summary('az-0700','2026-11-14T07:00:00Z'), summary()
    fixtures = {'/runs/index.json': index([old,new]),new['data_url']:run_data(),old['data_url']:run_data('az-0700')}
    out = execute(page,fixtures)
    assert out['elements']['run-picker']['value'] == new['id']
    fixtures['/runs/index.json']['lead_run_id'] = old['id']
    out = execute(page,fixtures)
    assert out['elements']['run-picker']['value'] == old['id']


def test_valid_recorded_run_exposes_evidence_without_conflating_signals():
    page = Page('studio.html')
    out = execute(page, {'/runs/index.json':index([summary()]),'/runs/az-0835/run.json':run_data()})
    el=out['elements']
    assert el['run-J']['text'] == '0.750'
    assert el['run-cost']['text'] == '$2.0000'
    assert el['qa-result']['text'] == 'Pass' and el['first-pass']['text'] == 'Fail'
    assert '4 / 5 shot-01, shot-03' in el['judge-rows']['text']
    assert el['weakest-shot']['text'] == 'shot-03'
    assert el['contact-sheet']['src'] == '/runs/az-0835/contact-sheet.jpg'
    assert sorted(r['path'] for r in out['requests']) == ['/runs/az-0835/run.json', '/runs/campaigns.json', '/runs/index.json']


def test_bad_identity_or_missing_run_never_leaves_stale_metrics():
    page=Page('studio.html')
    for data in ({},run_data('az-0700')):
        out=execute(page,{'/runs/index.json':index([summary()]),'/runs/az-0835/run.json':data})
        assert 'unavailable' in out['elements']['run-state']['text']
        assert out['elements']['run-J']['text'] == 'N/A'
        assert out['elements']['run-video']['src'] is None


def test_path_allowlist_null_scores_and_untrusted_text():
    page=Page('studio.html'); data=run_data()
    data.update(video_url='https://evil.invalid/video.mp4',contact_sheet_url='/runs/az-0835/../private.jpg',goal='PRIVATE GOAL',source_path='/Users/private/source.mov')
    data['scores']=dict(J=None,Q=99,D='0.9')
    data['cost']=dict(usd=None,known_usd=.2,status='unknown',quality_per_dollar=None)
    data['qa']['checks']=[dict(name='<img onerror=evil()>',status='pass',reason='/Users/private/file.mov')]
    data['judge']['evidence']['goal_fit']=['../../private','<script>','shot-01']
    out=execute(page,{'/runs/index.json':index([summary()]),'/runs/az-0835/run.json':data})
    el=out['elements']; text=' '.join(e['text'] for e in el.values())
    assert all(el['run-'+key]['text']=='N/A' for key in ['J','Q','D'])
    assert el['run-cost']['text']=='N/A' and el['known-cost']['text']=='$0.2000'
    assert el['run-video']['src'] is None and el['contact-sheet']['src'] is None
    assert '/Users' not in text and 'PRIVATE GOAL' not in text and '<script>' not in text
    assert 'Details withheld' in el['qa-checks']['text']
    assert '<img onerror=evil()>' in el['qa-checks']['text']  # Literal text, never parsed HTML.


def test_get_json_rejects_non_public_paths_before_fetch(page):
    out=execute(page,action="Promise.all(['//evil.invalid/x','/runs/../private','/runs/x/finished.mp4','/api/render',null].map(p=>getJSON(p).then(()=>false,()=>true)))")
    assert out['result']==[True]*5
    assert all(req['path'] in ['/runs/index.json','/runs/scoreboard.json','/runs/campaigns.json'] for req in out['requests'])


def test_picker_fetches_are_race_safe_and_clear_previous_data():
    page=Page('studio.html')
    # Start an old deferred fetch, select the newer run, then resolve the old fetch.
    page.script=page.script.replace("loadIndex();", "globalThis.start=loadIndex;")
    action=r"""(async()=>{
      availableRuns=[{id:'az-0700',data_url:'/runs/az-0700/run.json'},{id:'az-0835',data_url:'/runs/az-0835/run.json'}];
      let resolveOld;
      const realFetch=fetch;
      fetch=(path,opts)=>path.includes('az-0700')?new Promise(resolve=>{resolveOld=resolve}):realFetch(path,opts);
      const old=selectRun('az-0700');
      await selectRun('az-0835');
      resolveOld({ok:true,json:async()=>({id:'az-0700',scores:{J:0.1}})});
      await old;
      const latest=byId('run-J').textContent;
      const fail=selectRun('missing');await fail;
      return {latest,cleared:byId('run-J').textContent};
    })()"""
    out=execute(page,{'/runs/az-0835/run.json':run_data()},action=action)
    assert out['result']==dict(latest='0.750',cleared='N/A')


def test_scoreboard_shows_real_cohort_denominator_and_coverage():
    page=Page('index.html')
    board=dict(schema_version='1.0',cohort=dict(episode_count=3,qa_known=2,qa_passed=1,qa_pass_rate=.5,shipped=1,cost_complete=True,cost_per_shipped=3.5,estimated=True),factory=dict(ticket_to_pr_minutes=7.5,sample_count=2))
    fixtures={'/runs/index.json':index([summary()]),'/runs/scoreboard.json':board}
    out=execute(page,fixtures); el=out['elements']
    assert el['qa-rate']['text']=='50.0%' and '1 / 2 known' in el['qa-note']['text']
    assert el['cost-clip']['text']=='$3.5000' and 'Estimated' in el['cost-note']['text']
    assert el['factory-time']['text']=='7.5 min' and 'not completion or delivery' in el['factory-note']['text']
    assert el['efficiency']['text']=='0.400' and el['judge-score']['text']=='0.750'
    board['cohort'].update(cost_complete=False,known_subtotal=.5,qa_known=0,qa_pass_rate=None)
    out=execute(page,fixtures); el=out['elements']
    assert el['cost-clip']['text']=='N/A' and el['qa-rate']['text']=='N/A'
    assert 'Incomplete cost' in el['cost-note']['text']


def test_campaign_slots_show_only_real_bound_metrics():
    from pipelines.ffmpeg.campaigns import DEFAULT, project
    from copy import deepcopy
    manifest=deepcopy(DEFAULT)
    manifest['variants'][0]['episode_id']='az-0835'
    verified=summary();verified['scores']={'Q':.75,'D':1}
    campaigns={'schema_version':'1.0','campaigns':[project(manifest,[verified])]}
    out=execute(Page('studio.html'),{'/runs/index.json':index([summary()]),'/runs/az-0835/run.json':run_data(),'/runs/campaigns.json':campaigns})
    text=out['elements']['campaign-slots']['text']
    assert 'Sedona with kids overlays' in text and 'Q: 0.750' in text
    assert 'Halloween Uptown' in text and 'BellRock resort' in text
    assert text.count('Q:')==1 and 'planned' in text
    assert all(req['opts']['method']=='GET' for req in out['requests'])
