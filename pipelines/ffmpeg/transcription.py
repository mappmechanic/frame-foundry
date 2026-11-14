"""Bounded real-ASR routing: offline WhisperX, then configured OpenAI word timestamps.

Whisper-1 bills audio duration, not text tokens. Its configurable historical
$0.006/min rate is an estimate, not a verified current price or invoice. Success
and unknown-failure reservations share the existing episode API ledger/cap.
"""
from __future__ import annotations
from .media_tools import media_binary
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from . import llm_provider as accounting
from .contracts import StageError, atomic_write_json
from .proxy import probe, run

MODEL = 'whisper-1'
PROTOCOL = 'word-asr-1'
MAX_AUDIO_BYTES = 25_000_000
TIMEOUT_SECONDS = 180


def _rate():
    raw=os.environ.get('FRAMEFOUNDRY_ASR_USD_PER_MINUTE','0.006')
    try:
        result=Decimal(raw)
        if not result.is_finite() or not 0 < result <= 100:raise ValueError()
        return result
    except Exception:
        raise accounting.ProviderError('Invalid ASR per-minute pricing configuration') from None


def _source_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def _words(data,source):
    """Normalize quantized real word evidence without inventing recognized text.

    Zero-length words get up to 20ms, bounded by the next onset and source end.
    Equal onsets share one protected interval with their text retained. Neighbor
    overlaps are clamped at the next onset, transferring any longer tail so the
    union of recognized speech remains protected. Reversed/out-of-source times
    and malformed/missing evidence still fail, including on cache reads.
    """
    if not isinstance(data,dict):raise ValueError('Invalid transcript')
    rows=data.get('words');text=data.get('text')
    if not isinstance(rows,list) or not isinstance(text,str):raise ValueError('Word evidence missing')
    if text.strip() and not rows:raise ValueError('Nonempty transcript lacks word timestamps')
    duration=source['duration_ms'];words=[];last_start=-1
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid word row')
        token=row.get('word');a,b=row.get('start'),row.get('end')
        if not isinstance(token,str) or not token.strip():raise ValueError('Invalid word')
        if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) for t in (a,b)):raise ValueError('Invalid time')
        if not 0<=a<=b or a>duration/1000 or b>(duration+20)/1000:raise ValueError('Invalid source time')
        start,end=round(a*1000),min(round(b*1000),duration)
        # Bound backwards onset jitter to the model's 20ms quantization step.
        if start<last_start-20:raise ValueError('Reversed word order')
        start=max(start,last_start)
        last_start=start
        # A point at EOF has no forward room: protect the last available 20ms.
        if start==duration:
            start=max(0,duration-20,words[-1]['start_ms'] if words else 0)
        if not 0<=start<duration:raise ValueError('Out-of-range word')
        end=max(start,end)
        if words and start==words[-1]['start_ms']:
            words[-1]['text']+=' '+token.strip()
            words[-1]['end_ms']=max(words[-1]['end_ms'],end)
        else:
            words.append(dict(source_id=source['source_id'],text=token.strip(),start_ms=start,end_ms=end,confidence=None))
    for index,word in enumerate(words):
        next_word=words[index+1] if index+1<len(words) else None
        bound=next_word['start_ms'] if next_word else duration
        if next_word and word['end_ms']>bound:
            # Keep the union protected even for a contained/overlapping token.
            next_word['end_ms']=max(next_word['end_ms'],word['end_ms'])
        end=word['end_ms'] if word['end_ms']>word['start_ms'] else word['start_ms']+20
        word['end_ms']=min(end,bound,duration)
        if word['end_ms']<=word['start_ms']:raise ValueError('No available word interval')
    return words


def _multipart(audio):
    boundary='framefoundry-'+uuid.uuid4().hex;parts=[]
    for name,value in [('model',MODEL),('response_format','verbose_json'),('timestamp_granularities[]','word'),('temperature','0')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio{audio.suffix}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
    parts.extend([audio.read_bytes(),f'\r\n--{boundary}--\r\n'.encode()])
    return b''.join(parts),boundary


def prepare_audio(source,directory):
    """Use the renderer's common container origin; preserve delayed audio as silence.

    WAV is sample-exact for <=10min sources. Longer sources use bounded 32kbps
    mono MP3 with delay/padding metadata, keeping the 30min maximum below 25MB.
    No time compression, sample-clock conversion, or source-file mutation.
    """
    info=probe(source['path']);origin=float(info['data']['format'].get('start_time',0))
    duration=source['duration_ms']/1000
    audio=Path(directory)/('audio.wav' if duration<=600 else 'audio.mp3')
    filters=(f'atrim=start={origin:.9f}:end={origin+duration:.9f},'
             f'asetpts=PTS-({origin:.9f})/TB,aresample=16000:async=1:first_pts=0,'
             f'apad,atrim=duration={duration:.9f}')
    codec=['-c:a','pcm_s16le'] if audio.suffix=='.wav' else ['-c:a','libmp3lame','-b:a','32k','-write_xing','1']
    run([media_binary('ffmpeg'),'-nostdin','-y','-v','error','-copyts','-i',source['path'],
         '-map','0:a:0','-vn','-af',filters,'-ac','1','-ar','16000',*codec,str(audio)])
    if not 0<audio.stat().st_size<=MAX_AUDIO_BYTES:raise StageError('ASR_AUDIO_INVALID','refinery','Prepared audio exceeds upload limits',source['source_id'])
    return audio


class OpenAITranscriber(accounting.LLMProvider):
    """One transcription per source, no HTTP retries/redirects; source-keyed cache."""
    def __init__(self,key,*,transport=None):
        super().__init__('openai',key,MODEL,None,transport=transport)
        self.rate=_rate();self.sources={}

    @classmethod
    def from_env(cls):
        key=os.environ.get('OPENAI_API_KEY')
        return cls(key) if key and key.strip() else None

    def words(self,source,*,episode_dir):
        material=dict(protocol=PROTOCOL,content=_source_hash(source['path']),duration_ms=source['duration_ms'],model=MODEL,granularity='word')
        key=hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()
        cache=Path(os.environ.get('FRAMEFOUNDRY_CACHE_DIR',str(Path(__file__).resolve().parents[2]/'.inference-cache')))/'transcription'/f'{key}.json'
        try:
            if cache.stat().st_size>accounting.MAX_RESPONSE_BYTES:raise ValueError('Oversized cache')
            stored=accounting._strict_json(cache.read_text())
            if stored['identity']==material:
                words=_words(stored['response'],source)
                self.sources[source['source_id']]=dict(provider='openai',model=MODEL,cached=True,word_count=len(words))
                return words
        except (OSError,KeyError,ValueError,TypeError):pass
        with tempfile.TemporaryDirectory(prefix='framefoundry-asr-') as directory:
            audio=prepare_audio(source,directory);body,boundary=_multipart(audio)
            budget,tokens=accounting.episode_budget_usd(),accounting.episode_token_budget()
            # Conservative one-second allowance for format/billing duration rounding.
            seconds=math.ceil(source['duration_ms']/1000)+1
            reservation=accounting._usd_upper_bound(self.rate*Decimal(seconds)/60)
            entry=self._entry('transcription');entry.update(audio_duration_ms=source['duration_ms'])
            path=Path(episode_dir)/'llm-usage.json'
            with accounting._ledger_lock(path.parent):
                ledger=accounting._load_ledger(path)
                stop=self._admission_error(ledger,'transcription',Decimal(str(reservation)),0,0,budget,tokens,path.parent)
                if stop:entry['error']=stop
                else:entry.update(status='failed',reserved_usd=reservation,error='ASR request did not complete')
                ledger['calls'].append(entry.copy());accounting._finalize(ledger,budget,tokens);accounting._atomic_write(path,ledger)
            if stop:raise accounting.ProviderBudgetError(stop)
            words=None;response=None
            try:
                status,raw=self._transport('https://api.openai.com/v1/audio/transcriptions',
                    {'Authorization':'Bearer '+self._api_key,'Content-Type':'multipart/form-data; boundary='+boundary},body,TIMEOUT_SECONDS)
                if not isinstance(raw,bytes) or len(raw)>accounting.MAX_RESPONSE_BYTES:raise ValueError()
                response=accounting._strict_json(raw)
                if not 200<=status<300:raise ValueError()
                # Completed requests have a known duration-based estimate even if
                # their timestamps are invalid. Missing usage is never token zero.
                billed=response.get('duration',source['duration_ms']/1000)
                if isinstance(billed,bool) or not isinstance(billed,(int,float)) or not math.isfinite(billed) or billed<0:raise ValueError()
                entry.update(cost_usd=accounting._usd_upper_bound(self.rate*Decimal(math.ceil(billed))/60),cost_status='estimated')
                if abs(billed-source['duration_ms']/1000)>.5:raise ValueError()
                words=_words(response,source)
                entry.update(status='success',error=None)
            except Exception:
                entry.update(status='failed',error='OpenAI ASR request failed or returned invalid word timestamps')
            overrun=accounting._is_overrun(entry)
            if overrun:entry.update(overrun=True,error='ASR duration charge exceeded its reservation')
            with accounting._ledger_lock(path.parent):
                ledger=accounting._load_ledger(path)
                for i,call in enumerate(ledger['calls']):
                    if call['call_id']==entry['call_id']:ledger['calls'][i]=entry.copy();break
                else:raise accounting.ProviderError('ASR usage reservation missing; calls blocked')
                accounting._finalize(ledger,budget,tokens);accounting._atomic_write(path,ledger)
            if overrun:raise accounting.ProviderBudgetError(entry['error'])
            if entry['status']!='success':raise StageError('ASR_FAILED','refinery',entry['error'],source['source_id'])
            # Persist only validated word evidence, never credential/provider bodies.
            safe=dict(text=' '.join(w['text'] for w in words),words=[dict(word=w['text'],start=w['start_ms']/1000,end=w['end_ms']/1000) for w in words])
            try:atomic_write_json(cache,dict(identity=material,response=safe,historical_usage=entry))
            except OSError:pass  # Cache availability must not discard valid paid evidence.
            self.sources[source['source_id']]=dict(provider='openai',model=MODEL,cached=False,word_count=len(words))
            return words


class TranscriptionRouter:
    def __init__(self,ctx,mode='auto'):
        if mode not in ('auto','local','openai'):raise StageError('ASR_CONFIG_INVALID','refinery','Choose auto, local or openai ASR')
        self.ctx,self.mode=ctx,mode;self.sources={};self.remote=None

    @property
    def status(self):
        providers={s['provider'] for s in self.sources.values()}
        return 'mixed_asr' if len(providers)>1 else 'openai_asr' if providers=={'openai'} else 'local_asr'

    def words(self,source):
        from .refinery import transcribe_words
        if self.mode!='openai':
            try:
                words=transcribe_words(source,model_dir=os.environ.get('FRAMEFOUNDRY_MODEL_DIR','models'))
                self.sources[source['source_id']]=dict(provider='local',model='whisperx',cached=False,word_count=len(words))
                return words
            except StageError as exc:
                unavailable=exc.code=='DEPENDENCY_MISSING' or isinstance(exc.__cause__,(ImportError,ModuleNotFoundError,FileNotFoundError))
                if not unavailable or self.mode=='local':raise
        if self.remote is None:
            try:self.remote=OpenAITranscriber.from_env()
            except accounting.ProviderError as exc:raise StageError('ASR_CONFIG_INVALID','refinery',str(exc),source['source_id']) from None
        if self.remote is None:raise StageError('DEPENDENCY_MISSING','refinery','WhisperX is unavailable and no OpenAI transcription key is configured',source['source_id'])
        try:
            words=self.remote.words(source,episode_dir=self.ctx.episode_dir)
        except StageError as exc:
            if exc.code!='ASR_FAILED':raise
            # The paid/possibly paid failure remains in the provider ledger. Do
            # not cache, retry, mute original audio, or claim verified silence.
            self.sources[source['source_id']]=dict(provider='openai',model=MODEL,
                cached=False,word_count=0,status='degraded',
                limitation='ASR_FAILED for '+source['source_id']+' ('+Path(source['path']).name+'): no verified word timestamps; original audio retained, speech coverage and word-cut integrity unverified')
            return []
        self.sources[source['source_id']]=self.remote.sources[source['source_id']]
        return words
