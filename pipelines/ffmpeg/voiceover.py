"""Opt-in ElevenLabs intro: one request, content cache, shared durable reservations."""
from decimal import Decimal
from pathlib import Path
import hashlib,json,math,os,re,shutil,tempfile
from . import llm_provider as accounting
from .contracts import StageError,atomic_write_json
from .render import probe_media,_run,_ffmpeg

VOICE='21m00Tcm4TlvDq8ikWAM'  # Rachel, official premade Get-voice example.
MODEL='eleven_flash_v2_5'
MAX_BYTES=2_000_000


def intro_text(spec):
    if not spec.get('segments'):raise StageError('VOICEOVER_INVALID','assembler','Narrative has no selected moments')
    from .semantics import intro
    return intro(spec)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration():
    key=os.environ.get('ELEVENLABS_API_KEY','').strip()
    if not key:raise StageError('VOICEOVER_CONFIG','assembler','--voiceover requires ELEVENLABS_API_KEY')
    voice=os.environ.get('ELEVENLABS_VOICE_ID',VOICE);model=os.environ.get('ELEVENLABS_MODEL_ID',MODEL)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',voice) or model!=MODEL:
        raise StageError('VOICEOVER_CONFIG','assembler','Choose a valid voice ID and the priced eleven_flash_v2_5 model')
    try:
        rate=Decimal(os.environ.get('FRAMEFOUNDRY_TTS_USD_PER_1000_CHARS','0.30'))
        if not rate.is_finite() or not 0<rate<=100:raise ValueError()
    except Exception:raise StageError('VOICEOVER_CONFIG','assembler','Invalid assumed TTS per-1000-character rate') from None
    return key,voice,model,rate


def prepare_intro(ctx,spec,*,transport=None):
    key,voice,model,rate=configuration()
    from .semantics import validate_evidence
    try:validate_evidence(spec,root=ctx.episode_dir)
    except (ValueError,KeyError,TypeError,OSError):raise StageError('VOICEOVER_INVALID','assembler','Narration subject evidence invalid') from None
    text=intro_text(spec);settings=dict(stability=.5,similarity_boost=.75)
    identity=dict(protocol='intro-vo-1',text=text,voice_id=voice,model=model,settings=settings,output_format='mp3_44100_128')
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(os.environ.get('FRAMEFOUNDRY_CACHE_DIR',str(Path(__file__).resolve().parents[2]/'.inference-cache')))/'voiceover'
    mp3=cache/(digest+'.mp3');meta=cache/(digest+'.json')
    target=ctx.episode_dir/'audio'/'intro.mp3';target.parent.mkdir(parents=True,exist_ok=True)
    cached=False
    # Reuse the current Episode's hash-bound successful request even if a cache
    # write failed or a Planner repair replaced the entire spec.
    ledger_path=ctx.episode_dir/'tts-usage.json'
    if ledger_path.exists():
        ledger=accounting._strict_json(ledger_path.read_text())
        for row in ledger['calls']:
            if row.get('status')=='success' and row.get('cache_key')==digest:
                try:cached=0<target.stat().st_size<=MAX_BYTES and _hash(target)==row['audio_sha256']
                except (OSError,KeyError):cached=False
                break
    if not cached:
        try:
            stored=accounting._strict_json(meta.read_text())
            cached=stored['identity']==identity and stored['sha256']==_hash(mp3) and 0<mp3.stat().st_size<=MAX_BYTES
            if cached:shutil.copy2(mp3,target)
        except (OSError,ValueError,KeyError,TypeError):cached=False
    if not cached:
        reservation=accounting._usd_upper_bound(rate*Decimal(len(text))/1000)
        row=dict(call_id='tts-'+digest[:20],role='tts',provider='elevenlabs',model=model,status='blocked',characters=len(text),cache_key=digest,cost_usd=None,cost_status='unknown',reserved_usd=0,error=None)
        ledger_path=ctx.episode_dir/'tts-usage.json'
        provider=accounting.LLMProvider('openai','not-used',MODEL,None)
        budget,tokens=accounting.episode_budget_usd(),accounting.episode_token_budget()
        with accounting._ledger_lock(ctx.episode_dir):
            ledger=accounting._strict_json(ledger_path.read_text()) if ledger_path.exists() else dict(version=1,calls=[])
            if any(c.get('status')!='blocked' for c in ledger['calls']):raise StageError('VOICEOVER_FAILED','assembler','One paid intro call per Episode; no hidden retries')
            llm=accounting._load_ledger(ctx.episode_dir/'llm-usage.json')
            stop=provider._admission_error(llm,'tts',Decimal(str(reservation)),0,0,budget,tokens,ctx.episode_dir)
            if stop:row['error']=stop
            else:row.update(status='failed',reserved_usd=reservation,error='TTS request did not complete')
            ledger['calls'].append(row.copy());atomic_write_json(ledger_path,ledger)
        if stop:raise accounting.ProviderBudgetError(stop)
        try:
            status,audio=(transport or accounting._transport)(
                'https://api.elevenlabs.io/v1/text-to-speech/'+voice+'?output_format=mp3_44100_128',
                {'xi-api-key':key,'Content-Type':'application/json','Accept':'audio/mpeg'},
                json.dumps(dict(text=text,model_id=model,voice_settings=settings)).encode(),45)
            if status!=200 or not isinstance(audio,bytes) or not 0<len(audio)<=MAX_BYTES:raise ValueError()
            row.update(cost_usd=reservation,cost_status='estimated')
            target.write_bytes(audio)
            duration=float(probe_media(target)['format']['duration'])
            if not math.isfinite(duration) or not 0<duration<=10 or duration*1000>spec['target_duration_ms']:raise ValueError()
            row.update(status='success',error=None,audio_sha256=_hash(target))
        except Exception:
            row.update(status='failed',error='ElevenLabs TTS failed or returned invalid/oversized intro audio')
        with accounting._ledger_lock(ctx.episode_dir):
            ledger=accounting._strict_json(ledger_path.read_text())
            ledger['calls'][-1]=row;atomic_write_json(ledger_path,ledger)
        if row['status']!='success':raise StageError('VOICEOVER_FAILED','assembler',row['error'])
        try:
            cache.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache,delete=False) as f:f.write(target.read_bytes());temporary=Path(f.name)
            temporary.replace(mp3);atomic_write_json(meta,dict(identity=identity,sha256=_hash(mp3),historical_usage=row))
        except OSError:pass  # A valid paid result must not be lost to cache failure.
    info=probe_media(target);duration_ms=round(float(info['format']['duration'])*1000)
    if not 0<duration_ms<=min(10000,spec['target_duration_ms']):raise StageError('VOICEOVER_INVALID','assembler','Cached intro exceeds duration budget')
    return dict(text=text,path=str(target.relative_to(ctx.episode_dir)),sha256=_hash(target),start_ms=0,duration_ms=duration_ms,
                voice_id=voice,model=model,cached=cached,duck_db=-10,grounding_version='subjects-v2')


def mix_intro(joined,voiceover,episode_dir,total_seconds):
    root=Path(episode_dir).resolve();path=(root/voiceover['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _hash(path)!=voiceover['sha256']:
        raise StageError('VOICEOVER_INVALID','render','Saved VO audio reference/hash invalid')
    seconds=float(probe_media(path)['format']['duration'])
    if not math.isfinite(seconds) or not 0<seconds<=min(10,total_seconds):raise StageError('VOICEOVER_INVALID','render','VO would be truncated by timeline')
    voiced=Path(joined).parent/'voiced.mov'
    filters=(f'[0:a]volume=0.316227766:enable=\'between(t,0,{seconds+.1:.6f})\'[ambient];'
             f'[1:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-18:TP=-3:LRA=11,apad,atrim=duration={total_seconds:.9f}[vo];'
             '[ambient][vo]amix=inputs=2:duration=first:normalize=0[a]')
    _run(_ffmpeg()+['-i',str(joined),'-i',str(path),'-filter_complex',filters,'-map','0:v:0','-map','[a]',
                   '-c:v','copy','-c:a','pcm_s16le','-ar','48000','-t',f'{total_seconds:.9f}',str(voiced)])
    return voiced
