"""Static pitch contracts and real Chromium checks using explicitly test-only data.

No inference, deployment, credentials, or published metrics are involved. Browser
screenshots are offline test evidence, never evidence of Mac runs or scores.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Thread

import pytest

REPO=Path(__file__).resolve().parents[1]
PLAYWRIGHT=Path('/opt/software-factory/workspace-components/third_party/playwright-mcp/node_modules/playwright')


def test_authored_pitch_static_contract():
    page=(REPO/'site/pitch.html').read_text()
    assert page.count('<section class="slide"')==8
    for message in ('Thesis','Problem','Pipeline','8090 Software Factory audit','Recorded metrics','Recorded demo','Decisions','Roadmap'):
        assert message in page
    assert '11.9' not in page and '40.6' not in page
    assert 'prefers-reduced-motion:reduce' in page
    assert 'controls playsinline preload="metadata"' in page and 'autoplay' not in page.split('<video',1)[1].split('>',1)[0]
    assert 'pending evaluation' in page and 'future work' in page
    assert '/runs/index.json' in page and '/runs/scoreboard.json' in page
    assert '<script src=' not in page and '<link ' not in page
    for name in ('index.html','studio.html'):
        assert '<a href="/pitch">Pitch</a>' in (REPO/'site'/name).read_text()


@pytest.fixture
def pitch_server(tmp_path):
    shutil.copyfile(REPO/'site/pitch.html',tmp_path/'pitch.html')
    # This generated video is test-only, not a FrameFoundry episode or Mac proof.
    video=tmp_path/'test-only.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=teal:s=90x160:d=1',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True,capture_output=True)
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/pitch':self.path='/pitch.html'
            super().do_GET()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(tmp_path)))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}',video
    server.shutdown();thread.join();server.server_close()


BROWSER_CHECKS=r"""
const {chromium}=require(process.env.PITCH_PLAYWRIGHT);
const assert=require('node:assert/strict'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1280,height:720},reducedMotion:'reduce'});
 const page=await context.newPage(),origin=process.env.PITCH_ORIGIN;
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>requests.push({url:r.url(),method:r.method()}));
 const evidence=process.env.PITCH_EVIDENCE_DIR||'/tmp/framefoundry-pitch-evidence';fs.mkdirSync(evidence,{recursive:true});
 // Every value below is synthetic offline test data, not a real run or score.
 const run=(id,time,Q,usd,status='measured')=>({id,created_at:time,qa_passed:true,video_url:`/runs/${id}/finished.mp4`,scores:{Q,J:id.endsWith('old')?.7:.8},cost:{usd,status}});
 let mode='valid',media='valid',policy={schema_version:'1.0',status:'pending'};
 const runs=[run('test-only-old','2026-01-01T00:00:00Z',.5,2),run('test-only-new','2026-01-02T00:00:00Z',.8,1)];
 let index={schema_version:'1.0',mode:'recorded-demo',generated_at:'2026-01-03T00:00:00Z',lead_run_id:'test-only-old',runs};
 let board={schema_version:'1.0',generated_at:'2026-01-03T00:00:00Z',cohort:{episode_count:3,shipped:2,cost_complete:true,estimated:false,cost_per_shipped:2.5,known_subtotal:5,qa_known:3,qa_passed:2},factory:{work_order_count:7,pr_count:1,commit_count:12,ticket_to_pr_minutes:15,sample_count:7,pr_url:'https://github.com/mappmechanic/frame-foundry/pull/1'}};
 await page.route('**/runs/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/runs/policy-comparison.json')return route.fulfill({contentType:'application/json',body:JSON.stringify(policy)});
  if(path.endsWith('.mp4'))return route.fulfill({status:media==='valid'?200:404,contentType:'video/mp4',body:media==='valid'?fs.readFileSync(process.env.PITCH_VIDEO):Buffer.from('missing')});
  if(mode==='fetch-failed')return route.abort('failed');
  if(mode==='loading')return; // held request intentionally tests the loading state
  if(mode==='malformed')return route.fulfill({contentType:'application/json',body:'{invalid'});
  return route.fulfill({contentType:'application/json',body:JSON.stringify(path.endsWith('index.json')?index:board)});
 });
 const open=async()=>{await page.goto(origin+'/pitch');if(mode!=='loading')await page.waitForFunction(()=>!document.getElementById('metrics-state').textContent.startsWith('Loading'));};
 const policyReady=async()=>page.waitForFunction(()=>!document.getElementById('policy-state').textContent.startsWith('Loading'));
 const text=async id=>page.locator('#'+id).textContent();
 const go=async n=>{await page.locator('.slide:not([hidden])').focus();for(let i=1;i<n;i++)await page.keyboard.press('ArrowRight');};
 await open();assert.equal(await text('position'),'Slide 1 / 8');assert(await page.locator('#previous').isDisabled());
 assert.equal(await text('cost'),'$2.5000');assert.equal(await text('qa'),'66.7%');
 assert.match(await text('trend'),/test-only-old: Q 0.500 \/ \$2.0000 = 0.25 Q\/\$/);
 assert.match(await text('trend'),/test-only-new: Q 0.800 \/ \$1.0000 = 0.80 Q\/\$/);
 assert.match(await text('demo-label'),/test-only-old/);assert.equal(await text('judge'),'0.700');
 assert.match(await text('lead-metrics'),/test-only-old: Q 0.500 · J 0.700 · model cost \$2.0000/);
 assert.equal(await page.locator('#demo-video').getAttribute('src'),'/runs/test-only-old/finished.mp4');
 // The explicit lead beats timestamp sorting and carries its own Q/J/cost.
 index.lead_run_id='test-only-new';await open();assert.match(await text('demo-label'),/test-only-new/);assert.equal(await text('judge'),'0.800');
 const leadExample={...runs[0],scores:{Q:.809,J:.79},cost:{usd:.012,status:'measured'}};
 index.runs=[leadExample,runs[1]];index.lead_run_id=leadExample.id;await open();
 assert.equal(await text('judge'),'0.790');assert.match(await text('lead-metrics'),/Q 0.809 · J 0.790 · model cost \$0.0120/);
 assert.equal(await text('cost'),'$2.5000'); // Cohort accounting must not be replaced by lead cost.
 for(const badLead of [null,'missing','../private','test-only-new']){
  index.lead_run_id=badLead;index.runs=badLead==='test-only-new'?[leadExample,{...runs[1],qa_passed:false}]:[leadExample,runs[1]];
  await open();assert.equal(await text('judge'),'N/A');assert.equal(await page.locator('#demo-video').getAttribute('src'),null);
 }
 index.runs=runs;index.lead_run_id='test-only-new';await open();
 await page.locator('#next').focus();await page.keyboard.press('ArrowRight');assert.equal(await text('position'),'Slide 2 / 8');
 await page.locator('#previous').click();await page.locator('#next').click();assert.equal(await text('position'),'Slide 2 / 8');
 assert(await page.locator('#title-2').evaluate(e=>e.parentElement===document.activeElement));
 await page.keyboard.press('ArrowLeft');await page.keyboard.press('ArrowLeft');assert.equal(await text('position'),'Slide 1 / 8');
 // Verify all eight slides fit both projector sizes, navigation included.
 for(const [width,height] of [[1280,720],[1920,1080]]){
  await page.setViewportSize({width,height});await open();
  for(let n=1;n<=8;n++){
   const geometry=await page.evaluate(()=>{const r=document.querySelector('.slide:not([hidden])').getBoundingClientRect();const f=document.querySelector('footer').getBoundingClientRect();return {horizontal:document.documentElement.scrollWidth<=innerWidth,top:r.top,bottom:r.bottom,footer:f.bottom,height:innerHeight};});
   assert(geometry.horizontal&&geometry.top>=0&&geometry.bottom<=height&&geometry.footer<=height,JSON.stringify({n,width,...geometry}));
   if(n===1||n===5||n===6||n===7)await page.screenshot({path:`${evidence}/test-only-projector-${width}-slide-${n}.png`});
   if(n<8){await page.locator('.slide:not([hidden])').focus();await page.keyboard.press('ArrowRight');}
  }
  assert(await page.locator('#next').isDisabled());await page.keyboard.press('ArrowRight');assert.equal(await text('position'),'Slide 8 / 8');
 }
 await open();await go(6);await page.waitForFunction(()=>document.getElementById('demo-video').readyState>=2);
 assert(await page.locator('#demo-video').evaluate(v=>v.paused&&!v.autoplay&&v.controls&&v.playsInline));
 await page.locator('#demo-video').focus();await page.keyboard.press('ArrowLeft');await page.keyboard.press('ArrowRight');assert.equal(await text('position'),'Slide 6 / 8');
 await page.locator('.slide:not([hidden])').focus();await page.keyboard.press('ArrowRight');assert.equal(await text('position'),'Slide 7 / 8');
 assert(await page.locator('.slide:not([hidden])').evaluate(e=>getComputedStyle(e).animationName==='none'));
 await page.setViewportSize({width:390,height:844});await open();
 for(let n=1;n<=8;n++){assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:`${evidence}/test-only-mobile-slide-${n}.png`,fullPage:true});if(n<8){await page.locator('.slide:not([hidden])').focus();await page.keyboard.press('ArrowRight');}}
 // Production policy: explicitly synthetic, no ninth slide, no sponsor claim.
 const policyRow=(id,J,Q,usd)=>({episode_id:id,status:'passed',J,Q,model_usd:usd,known_model_usd:usd,cost_status:'measured',data_url:`/runs/${id}/run.json`});
 const recorded={schema_version:'1.0',status:'recorded',baseline:policyRow('test-only-old',.7,.5,2),production:policyRow('test-only-new',.8,.8,1),comparison:{comparable:true,J:900,Q:900,model_usd:900},goal:'SECRET',limitations:['SECRET']};
 for(const state of ['pending','recorded','incomparable']){
  policy=state==='pending'?{schema_version:'1.0',status:'pending'}:JSON.parse(JSON.stringify(recorded));
  if(state==='incomparable')policy.comparison.comparable=false;
  for(const [width,height] of [[1280,720],[1920,1080],[390,844]]){
   await page.setViewportSize({width,height});await open();await policyReady();await go(3);
   assert.equal(await page.locator('.slide').count(),8);
   const content=await page.locator('#production-policy').textContent();assert(!content.includes('SECRET')&&!content.includes('900'));
   assert.match(content,state==='pending'?/Pending evaluation/:state==='incomparable'?/incomparable pair/:/ΔJ 0.100 · ΔQ 0.300 · Δmodel USD -1.0000/);
   assert.equal(await page.locator('#policy-observations a').count(),state==='pending'?0:2);
   assert.equal(await text('cost'),'$2.5000');assert.equal(await text('judge'),'0.800');
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   if(width>700)assert(await page.evaluate(()=>document.querySelector('footer').getBoundingClientRect().bottom<=innerHeight),JSON.stringify({state,width,height,geometry:await page.evaluate(()=>({slide:document.querySelector('.slide:not([hidden])').getBoundingClientRect().toJSON(),footer:document.querySelector('footer').getBoundingClientRect().toJSON()}))}));
   await page.screenshot({path:`${evidence}/synthetic-policy-${state}-${width}.png`,fullPage:true});
  }
 }
 for(const bad of ['private-id','foreign-url','same-id','failed','unknown','bad-enum','nan','boolean','over-unit','bad-comparable','missing-arm']){
  policy=JSON.parse(JSON.stringify(recorded));const r=policy.production;
  if(bad==='private-id'){r.episode_id='private-secret';r.data_url='/runs/private-secret/run.json';}
  if(bad==='foreign-url')r.data_url='https://example.org/SECRET';
  if(bad==='same-id')policy.production={...policy.baseline};
  if(bad==='failed')r.status='failed';
  if(bad==='unknown')r.model_usd=null;
  if(bad==='bad-enum')r.cost_status='free';
  if(bad==='nan')r.J=NaN;
  if(bad==='boolean')r.model_usd=false;
  if(bad==='over-unit')r.J=2;
  if(bad==='bad-comparable')policy.comparison.comparable='yes';
  if(bad==='missing-arm')policy.baseline=null;
  await open();await policyReady();assert.match(await text('policy-state'),/Pending evaluation/);assert.equal(await page.locator('#policy-observations a').count(),0);
 }
 policy={schema_version:'1.0',status:'pending'};
 // Optional comparison snapshots must not turn unsupported evidence into a win.
 for(const comparison of [null,{}, {status:'recorded',qualified:false},
   {status:'recorded',qualified:true,arms:{llm:{episode_id:'test-only-old',q_per_dollar:11.9},decisions:{episode_id:'test-only-new',q_per_dollar:40.6}}}]){
  board.decisions_comparison=comparison;await open();await go(7);
  const decisions=await page.locator('.slide:not([hidden])').textContent();
  assert.match(decisions,/pending evaluation/);assert.match(decisions,/promotion: not established/);
  assert(!decisions.includes('11.9')&&!decisions.includes('40.6'));
 }
 delete board.decisions_comparison;
 // Estimated, incomplete and zero-cost states never fabricate efficiency.
 board.cohort.estimated=true;runs[1].cost.status='estimated';await open();assert.match(await text('cost'),/est\./);assert.match(await text('trend'),/estimated/);
 board.cohort.cost_complete=false;runs[0].cost.status='unknown';runs[1].cost.usd=0;await open();assert.equal(await text('cost'),'N/A');assert.match(await text('trend'),/unavailable/);
 index.runs=[];board={schema_version:'1.0',cohort:{episode_count:0,shipped:0,qa_known:0,qa_passed:0,cost_complete:false}};await open();assert.equal(await text('judge'),'N/A');assert.equal(await text('qa'),'N/A');assert.equal(await text('factory-time'),'N/A');assert.equal(await page.locator('#demo-video').getAttribute('src'),null);
 // A foreign URL and an unverified run must never get media requests.
 index.runs=[{...runs[0],video_url:'https://example.org/private.mp4'},{...runs[1],qa_passed:false}];await open();assert.equal(await page.locator('#demo-video').getAttribute('src'),null);
 index.runs=[runs[1]];index.lead_run_id=runs[1].id;media='missing';await open();await page.waitForFunction(()=>document.getElementById('video-state').textContent.includes('corrupt'));await go(6);await page.screenshot({path:`${evidence}/test-only-media-error.png`,fullPage:true});
 for(mode of ['malformed','fetch-failed']){await open();assert.match(await text('metrics-state'),/unavailable or invalid/);assert.equal(await text('cost'),'N/A');await go(5);await page.screenshot({path:`${evidence}/test-only-${mode}.png`,fullPage:true});}
 mode='loading';await page.goto(origin+'/pitch',{waitUntil:'domcontentloaded'});assert.match(await text('metrics-state'),/Loading/);assert.equal(await text('cost'),'N/A');await go(5);await page.screenshot({path:`${evidence}/test-only-loading.png`,fullPage:true});
 assert.equal(errors.length,0,JSON.stringify(errors));assert(requests.every(r=>r.url.startsWith(origin+'/')&&r.method==='GET'),JSON.stringify(requests));
 console.log(`Real Chromium checks passed: eight slides, arrows/buttons/endpoints/focus, video controls/playback/error, projector/mobile, reduced motion, malformed/fetch/loading, zero/incomplete/estimated cost. Screenshots: ${evidence}`);
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
"""


def test_real_browser_pitch(pitch_server):
    if not PLAYWRIGHT.exists() or not shutil.which('node'):
        pytest.skip('Real browser verification requires the workspace Playwright module and Node')
    origin,video=pitch_server
    env={**os.environ,'PITCH_PLAYWRIGHT':str(PLAYWRIGHT),'PITCH_ORIGIN':origin,'PITCH_VIDEO':str(video)}
    result=subprocess.run(['node','-e',BROWSER_CHECKS],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    print(result.stdout)
