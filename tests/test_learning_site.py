"""Synthetic offline UI fixtures: no media, providers, Arizona or paid calls."""
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Thread

import pytest

from pipelines.ffmpeg.learning_view import public_projection
from tests.test_learning_view import synthetic_curve
from tests.test_public_site import Page, execute
from tests.test_pitch_site import PLAYWRIGHT

REPO=Path(__file__).resolve().parents[1]


def public_index():
    return dict(schema_version='1.0',mode='recorded-demo',runs=[
        dict(id=eid,qa_passed=True,created_at='2026-01-01T00:00:00Z',data_url=f'/runs/{eid}/run.json',video_url=f'/runs/{eid}/finished.mp4')
        for eid in ('synthetic-on','synthetic-off')])


@pytest.mark.parametrize('name',['index.html','studio.html'])
def test_learning_public_data_and_unknown_id_guard(name):
    curve=public_projection(synthetic_curve(),['synthetic-on','synthetic-off'])
    curve['arms']['memory_on'].append(dict(iteration=2,published=True,status='passed',episode_id='private-unknown',
                                          data_url='/runs/private-unknown/run.json',J=.99,Q=.99,model_usd=99,known_model_usd=99,cost_status='measured',elapsed_seconds=99))
    out=execute(Page(name),{'/runs/learning.json':curve,'/runs/index.json':public_index()})
    rows=out['elements']['learning-rows']['text']
    assert 'synthetic-on' in rows and 'synthetic-off' in rows
    assert 'private-unknown' not in rows and '0.990' not in rows
    assert 'Unpublished / withheld' in rows
    assert '0.0000 / 0.0000' in rows  # Recorded zero, not infinity or missing.
    assert 'incomparable / incomplete' in out['elements']['learning-deltas']['text']
    assert all(r['opts']['method']=='GET' and r['opts']['credentials']=='omit' for r in out['requests'])


@pytest.mark.parametrize('name',['index.html','studio.html'])
@pytest.mark.parametrize('kind',['missing','malformed','pending','empty'])
def test_optional_missing_unknown_pending(name,kind):
    fixtures={'/runs/index.json':public_index()}
    if kind=='malformed':fixtures['/runs/learning.json']={'schema_version':'1.0','status':'recorded','arms':{}}
    if kind=='pending':fixtures['/runs/learning.json']={'schema_version':'1.0','status':'pending','goal':'SECRET'}
    if kind=='empty':fixtures['/runs/learning.json']=public_projection(synthetic_curve(),[])
    out=execute(Page(name),fixtures)
    assert 'pending' in out['elements']['learning-state']['text']
    assert 'SECRET' not in str(out['elements'])
    assert '0.700' not in out['elements']['learning-rows']['text']


@pytest.fixture
def learning_server():
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/studio','/pitch'):self.path+='.html'
            super().do_GET()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(REPO/'site')))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown();thread.join();server.server_close()


BROWSER=r"""
const {chromium}=require(process.env.PITCH_PLAYWRIGHT),assert=require('node:assert/strict'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true}),page=await browser.newPage({reducedMotion:'reduce'}),origin=process.env.LEARNING_ORIGIN;
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push({url:r.url(),method:r.method()}));
 const input=JSON.parse(process.env.LEARNING_FIXTURE),index=input.index;let curve=input.curve,mode='valid';
 const evidence='/tmp/framefoundry-learning-evidence';fs.mkdirSync(evidence,{recursive:true});
 await page.route('**/runs/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/runs/learning.json'){
   if(mode==='loading')return;
   if(mode==='missing')return route.fulfill({status:404,body:'missing'});
   if(mode==='malformed')return route.fulfill({contentType:'application/json',body:'{invalid'});
   return route.fulfill({contentType:'application/json',body:JSON.stringify(curve)});
  }
  if(path==='/runs/index.json')return route.fulfill({contentType:'application/json',body:JSON.stringify(index)});
  return route.fulfill({status:404,body:'offline fixture unavailable'});
 });
 const open=async path=>{await page.goto(origin+path,{waitUntil:'domcontentloaded'});if(mode!=='loading')await page.waitForFunction(()=>!document.getElementById('learning-state').textContent.startsWith('Loading'));};
 const roadmap=async()=>{for(let n=1;n<8;n++)await page.locator('#next').click();assert.equal(await page.locator('#position').textContent(),'Slide 8 / 8');};
 for(const viewport of [{width:1280,height:720},{width:1920,height:1080},{width:390,height:844}]){
  await page.setViewportSize(viewport);
  for(const path of ['/','/studio','/pitch']){
   await open(path);if(path==='/pitch')await roadmap();
   assert.match(await page.locator('#learning-state').textContent(),/Recorded curve/);
   assert.equal(await page.locator('#learning-rows a').count(),2);
   assert.equal(await page.locator('#learning-chart circle').count(),2);
   assert(!((await page.locator('#learning-rows').textContent()).includes('private-unknown')));
   assert.equal(await page.locator('#learning-rows .learning-unpublished').count(),2);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),JSON.stringify({path,viewport,dimensions:await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,width:innerWidth})),overflow:await page.evaluate(()=>[...document.querySelectorAll('*')].filter(e=>e.getBoundingClientRect().right>innerWidth).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right})).slice(0,15))}));
   if(path==='/pitch'&&viewport.width>700)assert(await page.evaluate(()=>document.querySelector('footer').getBoundingClientRect().bottom<=innerHeight));
   await page.locator('#learning-detail summary').click();
   assert.match(await page.locator('#learning-deltas').textContent(),/incomparable \/ incomplete/);
   await page.locator('#self-improving').scrollIntoViewIfNeeded();
   await page.screenshot({path:`${evidence}/synthetic-${path==='/studio'?'studio':path==='/pitch'?'pitch':'index'}-${viewport.width}.png`});
   if(path==='/studio'){assert(await page.locator('#upload').isDisabled());assert(await page.locator('#campaign-title').isVisible());assert(await page.locator('#run-picker').isEnabled());}
   if(path==='/pitch'){
    await page.locator('.slide:not([hidden])').focus();await page.keyboard.press('ArrowLeft');assert.match(await page.locator('.slide:not([hidden])').textContent(),/Decisions.*pending evaluation/s);
    await page.keyboard.press('ArrowRight');assert.equal(await page.locator('.slide').count(),8);
   }
  }
 }
 for(mode of ['pending','missing','malformed','empty','loading']){
  curve=mode==='pending'?{schema_version:'1.0',status:'pending'}:mode==='empty'?{...input.curve,arms:{memory_on:[],memory_off:[]},deltas:[]}:input.curve;
  await open('/pitch');await roadmap();
  assert.match(await page.locator('#learning-state').textContent(),mode==='loading'?/Loading/:/pending/);
  assert.equal(await page.locator('#learning-rows a').count(),0);assert.equal(await page.locator('#learning-chart circle').count(),0);
 }
 assert.equal(errors.length,0,JSON.stringify(errors));assert(requests.every(r=>r.url.startsWith(origin+'/')&&r.method==='GET'),JSON.stringify(requests));
 console.log('Synthetic Chromium learning checks passed: three pages, 390/1280/1920 widths, eight-slide navigation, projector footer, grey/unpublished/unknown rows, zero cost, deltas, pending/malformed/missing/loading; no remote requests. Evidence: '+evidence);
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
"""


def test_learning_chromium(learning_server):
    if not PLAYWRIGHT.exists() or not shutil.which('node'):
        pytest.skip('Requires workspace Chromium and Playwright')
    curve=public_projection(synthetic_curve(),['synthetic-on','synthetic-off'])
    curve['arms']['memory_on'].append(dict(iteration=2,published=True,status='passed',episode_id='private-unknown',data_url='/runs/private-unknown/run.json',J=.99,Q=.99,model_usd=99,known_model_usd=99,cost_status='measured',elapsed_seconds=99))
    curve['arms']['memory_off'].append(dict(iteration=2,published=False,status='failed',episode_id=None,data_url=None,J=None,Q=None,model_usd=None,known_model_usd=None,cost_status='unavailable',elapsed_seconds=None))
    env={**os.environ,'PITCH_PLAYWRIGHT':str(PLAYWRIGHT),'LEARNING_ORIGIN':learning_server,'LEARNING_FIXTURE':json.dumps(dict(curve=curve,index=public_index()))}
    result=subprocess.run(['node','-e',BROWSER],env=env,text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    print(result.stdout)
