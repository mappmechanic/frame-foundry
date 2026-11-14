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
 const run=(id,time,Q,usd,status='measured')=>({id,created_at:time,qa_passed:true,video_url:`/runs/${id}/finished.mp4`,scores:{Q,J:.8},cost:{usd,status}});
 let mode='valid',media='valid';
 const runs=[run('test-only-old','2026-01-01T00:00:00Z',.5,2),run('test-only-new','2026-01-02T00:00:00Z',.8,1)];
 let index={schema_version:'1.0',mode:'recorded-demo',generated_at:'2026-01-03T00:00:00Z',lead_run_id:'test-only-old',runs};
 let board={schema_version:'1.0',generated_at:'2026-01-03T00:00:00Z',cohort:{episode_count:3,shipped:2,cost_complete:true,estimated:false,cost_per_shipped:2.5,known_subtotal:5,qa_known:3,qa_passed:2},factory:{work_order_count:7,pr_count:1,commit_count:12,ticket_to_pr_minutes:15,sample_count:7,pr_url:'https://github.com/mappmechanic/frame-foundry/pull/1'}};
 await page.route('**/runs/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('.mp4'))return route.fulfill({status:media==='valid'?200:404,contentType:'video/mp4',body:media==='valid'?fs.readFileSync(process.env.PITCH_VIDEO):Buffer.from('missing')});
  if(mode==='fetch-failed')return route.abort('failed');
  if(mode==='loading')return; // held request intentionally tests the loading state
  if(mode==='malformed')return route.fulfill({contentType:'application/json',body:'{invalid'});
  return route.fulfill({contentType:'application/json',body:JSON.stringify(path.endsWith('index.json')?index:board)});
 });
 const open=async()=>{await page.goto(origin+'/pitch');if(mode!=='loading')await page.waitForFunction(()=>!document.getElementById('metrics-state').textContent.startsWith('Loading'));};
 const text=async id=>page.locator('#'+id).textContent();
 const go=async n=>{await page.locator('.slide:not([hidden])').focus();for(let i=1;i<n;i++)await page.keyboard.press('ArrowRight');};
 await open();assert.equal(await text('position'),'Slide 1 / 8');assert(await page.locator('#previous').isDisabled());
 assert.equal(await text('cost'),'$2.5000');assert.equal(await text('qa'),'66.7%');
 assert.match(await text('trend'),/test-only-old: Q 0.500 \/ \$2.0000 = 0.25 Q\/\$/);
 assert.match(await text('trend'),/test-only-new: Q 0.800 \/ \$1.0000 = 0.80 Q\/\$/);
 assert.match(await text('demo-label'),/test-only-new/);assert.equal(await text('judge'),'0.800');
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
 // Estimated, incomplete and zero-cost states never fabricate efficiency.
 board.cohort.estimated=true;runs[1].cost.status='estimated';await open();assert.match(await text('cost'),/est\./);assert.match(await text('trend'),/estimated/);
 board.cohort.cost_complete=false;runs[0].cost.status='unknown';runs[1].cost.usd=0;await open();assert.equal(await text('cost'),'N/A');assert.match(await text('trend'),/unavailable/);
 index.runs=[];board={schema_version:'1.0',cohort:{episode_count:0,shipped:0,qa_known:0,qa_passed:0,cost_complete:false}};await open();assert.equal(await text('judge'),'N/A');assert.equal(await text('qa'),'N/A');assert.equal(await text('factory-time'),'N/A');assert.equal(await page.locator('#demo-video').getAttribute('src'),null);
 // A foreign URL and an unverified run must never get media requests.
 index.runs=[{...runs[0],video_url:'https://example.org/private.mp4'},{...runs[1],qa_passed:false}];await open();assert.equal(await page.locator('#demo-video').getAttribute('src'),null);
 index.runs=[runs[1]];media='missing';await open();await page.waitForFunction(()=>document.getElementById('video-state').textContent.includes('corrupt'));await go(6);await page.screenshot({path:`${evidence}/test-only-media-error.png`,fullPage:true});
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
