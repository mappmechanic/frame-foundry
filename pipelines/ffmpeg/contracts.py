"""Canonical serializable stage contracts, schema checks and safe persistence."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, math, os, re, tempfile, time
from jsonschema import Draft202012Validator

SourceAnalysis = dict
NarrativeSpec = dict
QAReport = dict

@dataclass
class EpisodeContext:
    episode_id: str
    episode_dir: Path
    goal: str
    started: float = field(default_factory=time.monotonic)
    fixture_mode: bool = False
    target_seconds: float | None = None
    aspect_override: str | None = None
    decider_mode: str = 'llm'
    memory_enabled: bool = False
    memory_backend: str = 'auto'
    memory_root: Path | None = None
    memory_selection: dict | None = None
    overlay_mode: bool | None = None
    overlay_sections: list[str] = field(default_factory=list)

class StageError(RuntimeError):
    def __init__(self, code, stage, message, source_id=None, repairable=False):
        super().__init__(message)
        # Renderer is an assembler substage, not a QA schema stage.
        self.code, self.stage, self.source_id, self.repairable = code, ('assembler' if stage == 'render' else stage), source_id, repairable
    def as_dict(self):
        return dict(code=self.code, stage=self.stage, message=str(self), source_id=self.source_id)

@dataclass
class AssemblyResult:
    episode_id: str
    candidate_path: Path
    timeline_path: Path
    spec_path: Path
    cut_times_ms: list[int]
    duration_ms: int
    limitations: list[str] = field(default_factory=list)

def atomic_write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
            name=f.name; f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path); name=None
    finally:
        if name and os.path.exists(name): os.unlink(name)

def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def resolve_episode(root, episode_id):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', episode_id):
        raise StageError('EPISODE_ID_INVALID','refinery','Unsafe episode identity')
    root=Path(root).resolve(); path=(root/episode_id).resolve()
    if path.parent != root: raise StageError('EPISODE_ID_INVALID','refinery','Unsafe episode path')
    return path

def _validate(payload, filename):
    schema=read_json(Path(__file__).resolve().parents[2]/'schemas'/filename)
    errors=sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda e:str(e.path))
    if errors: raise StageError('SCHEMA_INVALID','validator', '; '.join(e.message for e in errors[:3]))
    json.dumps(payload, allow_nan=False)

def validate_narrative(spec, analysis=None):
    _validate(spec,'narrative-spec.schema.json')
    sources={s['source_id']:s for s in spec['sources']}
    if len(sources)!=len(spec['sources']): raise StageError('PLAN_INVALID','planner','Duplicate source IDs')
    if analysis is not None:
        if spec['episode_id']!=analysis['episode_id'] or spec['goal']!=analysis['goal']:
            raise StageError('PLAN_INVALID','planner','Plan identity differs from analysis')
        canonical={s['source_id']:s for s in analysis['sources']}
        if sources != canonical: raise StageError('PLAN_INVALID','planner','Plan source descriptors changed')
    ids=set(); ranges={}
    for segment in spec['segments']:
        sid=segment['source_id']; a=segment['raw_start_ms']; b=segment['raw_end_ms']
        if segment['segment_id'] in ids or sid not in sources or not 0<=a<b<=sources[sid]['duration_ms']:
            raise StageError('PLAN_INVALID','planner','Invalid segment identity or source interval')
        ids.add(segment['segment_id'])
        for start,end in ranges.get(sid,[]):
            if max(start,a)<min(end,b): raise StageError('PLAN_INVALID','planner','Overlapping raw selections')
        ranges.setdefault(sid,[]).append((a,b))
        x,y=segment['render_start_ms'],segment['render_end_ms']
        if (x is None)!=(y is None) or (x is not None and not 0<=x<y<=sources[sid]['duration_ms']):
            raise StageError('PLAN_INVALID','assembler','Invalid render interval')

    from .semantics import validate_evidence
    try:validate_evidence(spec,analysis)
    except (ValueError,KeyError,TypeError) as error:raise StageError('PLAN_INVALID','planner','Invalid semantic evidence') from error

def unavailable_judge(reason='No configured provider'):
    return dict(status='unavailable',score=None,rationale=reason,evidence=[],provider=None,model=None)

def usage_summary(episode_dir):
    path=Path(episode_dir)/'llm-usage.json'
    try:
        ledger=read_json(path) if path.exists() else {'calls':[]}
        calls=list(ledger['calls'])
        tts_path=Path(episode_dir)/'tts-usage.json'
        if tts_path.exists():
            tts=read_json(tts_path)['calls']
            calls.extend(dict(c,input_tokens=None,output_tokens=None) for c in tts)
        fields=['call_id','role','provider','model','status','input_tokens','output_tokens','cost_usd','cost_status','reserved_usd','error']
        extra_fields=['decision_stage', 'requested_mode', 'effective_mode', 'fallback_reason', 'prompt_cache_key', 'latency_seconds', 'cached_input_tokens', 'cache_write_tokens']
        clean=[dict({k:c.get(k) for k in fields},**{k:c[k] for k in extra_fields if k in c}) for c in calls]
        schema=read_json(Path(__file__).resolve().parents[2]/'schemas'/'qa-report.schema.json')
        Draft202012Validator(schema['properties']['ai_usage']).validate(clean)
        total=sum(c['cost_usd'] or 0 for c in clean)
        json.dumps(clean,allow_nan=False)
    except Exception as error:
        raise StageError('USAGE_LEDGER_INVALID','validator','Usage ledger unreadable; charges cannot be reconstructed') from error
    # Local compute expense is not inferred from API charges.
    return clean,dict(status='unknown',usd=None,known_ai_usd=total,includes_failed_attempts=True)

def failure_report(ctx, error):
    try:calls,cost=usage_summary(ctx.episode_dir)
    except StageError as usage_error:
        error=usage_error;calls=[];cost=dict(status='unknown',usd=None,known_ai_usd=0,includes_failed_attempts=True)
    return dict(schema_version='1.0',episode_id=ctx.episode_id,goal=ctx.goal,stage=error.stage,
        passed=False,result='failed',fixture_mode=ctx.fixture_mode,interpretation=None,assumptions=[],
        coverage=dict(status='unknown',limitations=[]),exclusions=[],limitations=[],output_path=None,
        contact_sheet_path=None,timeline_path=None,target_duration_ms=None,actual_duration_ms=None,
        integrated_lufs=None,true_peak_dbtp=None,checks=[],attempts=[],elapsed_seconds=time.monotonic()-ctx.started,
        cost=cost,ai_usage=calls,goal_judge=unavailable_judge('Run failed before judgment'),error=error.as_dict())

def validate_report(report):
    _validate(report,'qa-report.schema.json')
    names=[c['name'] for c in report['checks']]
    if len(set(names))!=len(names): raise StageError('REPORT_INVALID','validator','Duplicate checks')
    if report['passed']:
        required={'narrative_schema','readable_media','output_encoding','output_dimensions','output_fps','loudness','true_peak','timeline_and_padding','render_duration','goal_duration','contact_sheet'}
        if not required.issubset(set(names)):raise StageError('REPORT_INVALID','validator','Required gates absent')
        if any(c['required'] and c['status']!='pass' for c in report['checks']):
            raise StageError('REPORT_INVALID','validator','Unpassed required check')
        if not -15<=report['integrated_lufs']<=-13 or report['true_peak_dbtp']>-1:
            raise StageError('REPORT_INVALID','validator','Passing report violates audio gate')
    indices=[a['index'] for a in report['attempts']]
    if indices!=list(range(1,len(indices)+1)): raise StageError('REPORT_INVALID','validator','Invalid attempts')
