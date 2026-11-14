"""EpisodeEvaluator: gates (G), local quality (D), judge (J), labels (H), Q, cost.

``evaluate_episode`` always returns a schema-valid EvalReport, including for
Episodes that failed before rendering (G=0, Q=0, D/J/H unavailable, no judge).
Missing evidence stays null; it is never replaced by a fabricated value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
import uuid

from jsonschema import Draft202012Validator

from .contracts import atomic_write_json
from . import eval_gates, eval_judge
from .eval_gates import optional_json, read_strict_json

EVALUATOR_VERSION = 'evaluator-1.0'
SCORING_VERSION = 'scores-1.0'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_EVALS_ROOT = REPO / 'evals'
SCHEMA_PATH = REPO / 'contracts' / 'eval-report.schema.json'
WEIGHTS_WITH_H = dict(J=0.5, D=0.3, H=0.2)
WEIGHTS_NO_H = dict(J=0.625, D=0.375, H=0.0)
H_OVERLAP_RULE = 'label recalled when one shot from the same source file overlaps >= 50% of the labelled interval'


class EvalReportError(RuntimeError):
    pass


# ---------------------------------------------------------------- fingerprints
def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def footage_fingerprint(sources):
    """Ordered (source_id, filename, size, streamed content SHA-256); None if any is missing."""
    if not sources:
        return None
    digest = hashlib.sha256()
    try:
        for source in sorted(sources, key=lambda s: s['source_id']):
            path = Path(source['path'])
            digest.update(json.dumps([source['source_id'], path.name, path.stat().st_size,
                                      sha256_file(path)]).encode())
    except (OSError, KeyError, TypeError):
        return None
    return digest.hexdigest()


# ------------------------------------------------------------------- D scoring
def _component(available, score=None, *, method, normalizer, raw=None, reason=None):
    if score is not None:
        score = round(min(1.0, max(0.0, float(score))), 6)
    return dict(available=bool(available and score is not None), score=score if available else None,
                method=method, normalizer=normalizer, raw=raw, reason=None if available else reason)


def _scene_uses(shots, scenes):
    uses = {}
    for shot in shots:
        for scene in scenes:
            if (scene.get('source_id') == shot['source_id'] and scene['start_ms'] < shot['source_end_ms']
                    and shot['source_start_ms'] < scene['end_ms']):
                uses[scene['scene_id']] = uses.get(scene['scene_id'], 0) + 1
    return uses


def deterministic_quality(ctx):
    """Equal-weight mean of AVAILABLE components; unavailable ones stay listed."""
    frames, shots = ctx.get('frames'), ctx.get('shots') or []
    analysis = ctx.get('analysis') or {}
    per_shot = (frames or {}).get('shots') or []
    unavailable_frames = 'Final-render frame analysis unavailable'
    comps = {}
    lap = [s['laplacian_variance'] for s in per_shot if s.get('laplacian_variance') is not None]
    comps['sharpness'] = _component(
        bool(lap), statistics.median(lap) / 300 if lap else None,
        method='Laplacian variance of mid-shot content frame (1/4-res gray, letterbox excluded)',
        normalizer='min(1, median_variance / 300)',
        raw=dict(per_shot=lap, refinery_scene_sharpness=[round(s.get('sharpness', 0), 4) for s in
                 (analysis.get('scenes') or [])][:40]) if lap else None, reason=unavailable_frames)
    jit = [s['jitter_px64'] for s in per_shot if s.get('jitter_px64') is not None]
    comps['stability'] = _component(
        bool(jit), 1 / (1 + statistics.mean(jit)) if jit else None,
        method='motion jitter: mean |delta| of coarse global translation (integer px, +/-3, 64-px-wide copy) '
               'between consecutive frames within each shot',
        normalizer='1 / (1 + mean_jitter_px64)', raw=dict(per_shot=jit) if jit else None,
        reason=unavailable_frames + ' or shots shorter than 3 frames')
    exp = [(s['mean_luma'], s['clipped_fraction']) for s in per_shot if s.get('mean_luma') is not None]
    comps['exposure'] = _component(
        bool(exp), statistics.mean(max(0.0, 1 - abs(m - 118) / 118) * (1 - min(1.0, 2 * c)) for m, c in exp) if exp else None,
        method='mean luma and clipped fraction (<=16 or >=236) over all content frames per shot',
        normalizer='mean_shots((1-|luma-118|/118) * (1-min(1, 2*clipped)))',
        raw=dict(per_shot=[dict(mean_luma=m, clipped_fraction=c) for m, c in exp]) if exp else None,
        reason=unavailable_frames)
    if shots:
        durations = [(s['timeline_end_ms'] - s['timeline_start_ms']) / 1000 for s in shots]
        mean = statistics.mean(durations)
        score = 1.0 if 1.5 <= mean <= 4 else (mean / 1.5 if mean < 1.5 else 4 / mean)
        comps['pacing'] = _component(True, score, method='mean shot duration vs teaser band 1.5-4 s',
                                     normalizer='1 inside band; mean/1.5 below; 4/mean above',
                                     raw=dict(mean_shot_seconds=round(mean, 3), shots=len(shots)))
    else:
        comps['pacing'] = _component(False, method='mean shot duration vs teaser band 1.5-4 s',
                                     normalizer='1 inside band; mean/1.5 below; 4/mean above', reason='No timeline shots')
    scenes = analysis.get('scenes') or []
    if shots and scenes:
        uses = _scene_uses(shots, scenes)
        ratio = len(uses) / len(scenes)
        parts, raw = [min(1.0, ratio / 0.6)], dict(selected_scenes=len(uses), known_scenes=len(scenes),
                                                   scene_ratio=round(ratio, 4), target=0.6)
        sources = analysis.get('sources') or []
        stamps = {s['source_id']: s.get('creation_time') for s in sources}
        if sources and all(stamps.values()):
            days = {v[:10] for v in stamps.values()}
            chosen = {stamps[s['source_id']][:10] for s in shots if stamps.get(s['source_id'])}
            parts.append(min(1.0, (len(chosen) / len(days)) / 0.6))
            raw.update(selected_days=len(chosen), known_days=len(days))
        else:
            raw['days'] = 'unavailable: creation_time missing for some sources'
        comps['coverage'] = _component(True, statistics.mean(parts), method='Refinery scenes (and days when known) selected / known',
                                       normalizer='mean(min(1, ratio/0.6) for scenes[, days])', raw=raw)
        top = max(uses.values()) if uses else 0
        comps['variety'] = _component(True, 1.0 if top <= 2 else 2 / top, method='max uses of one Refinery scene',
                                      normalizer='1 if max_uses<=2 else 2/max_uses', raw=dict(max_scene_uses=top))
    else:
        for name, method in (('coverage', 'Refinery scenes selected / known'), ('variety', 'max uses of one Refinery scene')):
            comps[name] = _component(False, method=method, normalizer='n/a', reason='Refinery scenes or shots unavailable')
    hook, overall = (frames or {}).get('hook_mean_abs_diff'), (frames or {}).get('overall_mean_abs_diff')
    ok = hook is not None and overall
    comps['hook'] = _component(ok, min(1.0, hook / overall) if ok else None,
                               method='visual change energy in first 2 s vs whole edit (audio energy not used)',
                               normalizer='min(1, first2s_mean_abs_diff / overall_mean_abs_diff)',
                               raw=dict(first2s=round(hook, 4), overall=round(overall, 4)) if ok else None,
                               reason=unavailable_frames)
    scores = [c['score'] for c in comps.values() if c['available']]
    return (round(statistics.mean(scores), 6) if scores else None), comps


# ------------------------------------------------------------------- H scoring
def load_labels(labels):
    if labels is None:
        return None
    data = read_strict_json(labels) if isinstance(labels, (str, Path)) else labels
    items = []
    if isinstance(data, dict) and isinstance(data.get('labels'), list):
        items = data['labels']
    elif isinstance(data, dict):
        items = [dict(filename=k, start_ms=a, end_ms=b) for k, v in data.items() for a, b in v]
    else:
        raise ValueError('Unsupported labels format')
    clean = []
    for item in items:
        a, b = item['start_ms'], item['end_ms']
        if not isinstance(item['filename'], str) or type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError('Invalid label interval')
        clean.append(dict(filename=Path(item['filename']).name, start_ms=a, end_ms=b))
    return clean


def human_recall(labels, shots, sources):
    durations = {Path(s['path']).name: s.get('duration_ms') for s in sources or []}
    usable, ignored, recalled = [], [], 0
    for label in labels:
        duration = durations.get(label['filename'])
        if duration is None or label['end_ms'] > duration + 1:
            ignored.append(dict(label, reason='source file absent from footage or interval out of bounds'))
            continue
        usable.append(label)
        need = 0.5 * (label['end_ms'] - label['start_ms'])
        if any(Path(s['source_path']).name == label['filename']
               and min(s['source_end_ms'], label['end_ms']) - max(s['source_start_ms'], label['start_ms']) >= need
               for s in shots or []):
            recalled += 1
    detail = dict(rule=H_OVERLAP_RULE, labelled=len(usable), recalled=recalled, ignored=ignored[:20])
    return (round(recalled / len(usable), 6) if usable else None), detail


# ---------------------------------------------------------------- composite Q
def composite(G, J, D, H, labels_supplied):
    weights = WEIGHTS_WITH_H if labels_supplied else WEIGHTS_NO_H
    if G == 0:
        return 0.0, 'forced_zero_gate_failed', weights
    if J is None or D is None or (labels_supplied and H is None):
        return None, 'unavailable', weights
    q = weights['J'] * J + weights['D'] * D + weights['H'] * (H or 0.0)
    return round(min(1.0, max(0.0, q)), 6), 'computed', weights


# ----------------------------------------------------------------------- cost
def _ledger_cost(calls, kind):
    total, unknown, estimated, notes = Decimal(0), False, False, []
    for call in calls:
        status, cost = call.get('status'), call.get('cost_usd')
        reserved = call.get('reserved_usd') or 0
        if status == 'blocked':
            continue  # never dispatched: zero charge
        if status == 'success' and cost is not None:
            total += Decimal(str(cost))
            estimated |= call.get('cost_status', 'estimated') != 'measured'
        elif status == 'failed' and (cost is not None or reserved):
            total += max(Decimal(str(reserved)), Decimal(str(cost or 0)))
            estimated = True
            notes.append(f'{kind} failed call {str(call.get("call_id"))[:36]} charged at conservative reservation')
        else:
            unknown = True
    return total, unknown, estimated, notes


def model_cost(episode_dir, Q):
    root = Path(episode_dir)
    notes = ['Local compute expense is not included in model_cost']
    llm_calls, tts_calls, unknown_llm, unknown_tts = [], [], False, False
    if (root / 'llm-usage.json').is_file():
        ledger = optional_json(root / 'llm-usage.json')
        if isinstance(ledger, dict) and isinstance(ledger.get('calls'), list):
            llm_calls = ledger['calls']
        else:
            unknown_llm = True
            notes.append('llm-usage.json unreadable; LLM spend unknown')
    if (root / 'tts-usage.json').is_file():
        ledger = optional_json(root / 'tts-usage.json')
        if isinstance(ledger, dict) and isinstance(ledger.get('calls'), list):
            tts_calls = ledger['calls']
        else:
            unknown_tts = True
            notes.append('tts-usage.json unreadable; TTS spend unknown')
    else:
        notes.append('No TTS ledger: zero known TTS spend')
    try:
        l_total, l_unknown, l_est, l_notes = _ledger_cost(llm_calls, 'LLM')
        t_total, t_unknown, t_est, t_notes = _ledger_cost(tts_calls, 'TTS')
    except Exception:
        l_total = t_total = Decimal(0)
        l_unknown = t_unknown = True
        l_est = t_est = False
        l_notes, t_notes = ['Usage ledger entries malformed'], []
    notes += l_notes + t_notes
    l_unknown |= unknown_llm
    t_unknown |= unknown_tts
    llm_usd = None if l_unknown else float(l_total)
    tts_usd = None if t_unknown else float(t_total)
    usd = None if (l_unknown or t_unknown) else float(l_total + t_total)
    status = 'unknown' if usd is None else ('estimated' if (l_est or t_est) else 'measured')
    if not llm_calls and not tts_calls and usd is not None:
        notes.append('No model calls recorded: known zero model spend')
    qpd = Q / usd if (Q is not None and usd is not None and usd > 0) else None
    tokens_in = tokens_out = 0
    complete = not unknown_llm
    for call in llm_calls:
        if call.get('status') == 'blocked':
            continue
        i, o = call.get('input_tokens'), call.get('output_tokens')
        if type(i) is int and type(o) is int:
            tokens_in += i
            tokens_out += o
        else:
            complete = False
    cost = dict(usd=usd, status=status, known_usd=float(l_total + t_total), llm_usd=llm_usd, tts_usd=tts_usd,
                quality_per_dollar=round(qpd, 6) if qpd is not None else None, notes=notes)
    return cost, dict(input=tokens_in, output=tokens_out, complete=complete)


# ----------------------------------------------------------------- provenance
def _git(*args):
    try:
        out = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(qa, analysis):
    pr = os.environ.get('FRAMEFOUNDRY_PR')
    if not pr:
        match = re.match(r'refs/pull/(\d+)/', os.environ.get('GITHUB_REF', ''))
        pr = match.group(1) if match else None
    status = (analysis or {}).get('transcription_status')
    return dict(branch=_git('rev-parse', '--abbrev-ref', 'HEAD'), commit=_git('rev-parse', 'HEAD'), pr=pr,
                fixture_mode=bool((qa or {}).get('fixture_mode')) or status == 'fixture',
                transcription_status=status, evaluator_version=EVALUATOR_VERSION)


# ------------------------------------------------------------------ persistence
_SCHEMA = None


def validate_eval_report(report):
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = read_strict_json(SCHEMA_PATH)
    errors = sorted(Draft202012Validator(_SCHEMA).iter_errors(report), key=lambda e: list(e.path))
    if errors:
        raise EvalReportError('; '.join(f'{"/".join(map(str, e.path))}: {e.message[:160]}' for e in errors[:4]))
    json.dumps(report, allow_nan=False)
    s = report['scores']
    if s['G'] == 0 and s['Q'] != 0:
        raise EvalReportError('G=0 requires Q=0')
    if (report['judge']['status'] == 'scored') != (s['J'] is not None):
        raise EvalReportError('J must be present exactly when the judge scored')
    if report['judge']['status'] == 'scored' and not report['gates']['passed']:
        raise EvalReportError('Judge may only score after all gates pass')
    return report


def append_scoreboard(report, evals_root=None):
    """Locked, fsynced JSONL append of the whole report; idempotent per evaluation_id."""
    validate_eval_report(report)
    root = Path(evals_root or os.environ.get('FRAMEFOUNDRY_EVALS_ROOT') or DEFAULT_EVALS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'scoreboard.jsonl'
    line = json.dumps(report, sort_keys=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    with (root / '.scoreboard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            existing = path.read_bytes() if path.exists() else b''
            for raw in existing.splitlines():
                try:
                    if json.loads(raw).get('evaluation_id') == report['evaluation_id']:
                        return False
                except (ValueError, AttributeError):
                    continue  # partial/corrupt line stays visible for the summary reader
            prefix = b'\n' if existing and not existing.endswith(b'\n') else b''
            with path.open('ab') as handle:
                handle.write(prefix + line)
                handle.flush()
                os.fsync(handle.fileno())
            fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def persist_report(report, episode_dir, evals_root=None):
    validate_eval_report(report)
    atomic_write_json(Path(episode_dir) / 'eval-report.json', report)
    return append_scoreboard(report, evals_root)


# ------------------------------------------------------------------- main API
def _resolve_provider(provider, limitations):
    if provider != 'env':
        return provider
    from .llm_provider import LLMProvider, ProviderError
    try:
        return LLMProvider.from_env()
    except ProviderError as exc:
        limitations.append(f'Provider configuration invalid: {exc}')
        return None


def evaluate_episode(episode_dir, *, provider=None, evals_root=None, labels=None, persist=True):
    """Evaluate one Episode directory and return a strict EvalReport dict.

    provider: None => no judge call; 'env' => LLMProvider.from_env(); else an
    object exposing complete_json(...). labels: None, a JSON path or a dict.
    """
    started = time.monotonic()
    root = Path(episode_dir).resolve()
    evals_root = Path(evals_root) if evals_root is not None else Path(os.environ.get('FRAMEFOUNDRY_EVALS_ROOT',DEFAULT_EVALS_ROOT))
    limitations = []
    gates = eval_gates.run_gates(root)
    ctx = gates['context']
    limitations += gates['limitations']
    spec, analysis, qa = ctx['spec'] or {}, ctx['analysis'] or {}, ctx['qa'] or {}
    goal = qa.get('goal') or spec.get('goal') or analysis.get('goal')
    if not isinstance(goal, str):
        goal = ''
        limitations.append('Goal unavailable from Episode artifacts')
    episode_id = root.name
    if spec.get('episode_id') not in (None, episode_id):
        limitations.append('narrative-spec episode_id differs from Episode directory name')
    sources = spec.get('sources') or analysis.get('sources') or []
    footage_hash = footage_fingerprint(sources)
    if footage_hash is None:
        limitations.append('Footage fingerprint unavailable (sources missing or unreadable)')
    render_path = root / 'finished.mp4'
    render_hash = sha256_file(render_path) if render_path.is_file() else None
    G = 1 if gates['passed'] else 0
    if ctx.get('render') is not None:
        D, components = deterministic_quality(ctx)
    else:
        _, components = deterministic_quality(dict(ctx, frames=None, shots=None))
        D = None
        limitations.append('No readable render: D, J and H unavailable')
    labels_supplied = labels is not None
    H, H_detail = None, None
    if labels_supplied:
        try:
            parsed = load_labels(labels)
            if ctx.get('shots'):
                H, H_detail = human_recall(parsed, ctx['shots'], sources)
            else:
                H_detail = dict(rule=H_OVERLAP_RULE, labelled=len(parsed), recalled=None, ignored=[],
                                reason='Timeline unavailable')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            H_detail = dict(rule=H_OVERLAP_RULE, reason=f'Labels unusable: {str(exc)[:160]}')
        if H is None:
            limitations.append('Supplied labels could not produce H; Q unavailable rather than renormalized')
    if not gates['passed']:
        judge = eval_judge.unavailable('Hard gates did not all pass; judge not run (zero calls)')
    else:
        active = _resolve_provider(provider, limitations)
        render = ctx['render']
        judge = eval_judge.run_judge(
            episode_dir=root, evals_root=evals_root, provider=active, goal=goal, shots=ctx['shots'],
            duration_ms=render['duration_ms'], aspect_ratio=spec.get('aspect_ratio'), render_path=render_path,
            render_hash=render_hash, footage_hash=footage_hash, render_size=(render['width'], render['height']))
    J = judge.pop('J', None)
    if judge['status'] != 'scored':
        J = None
    Q, q_status, weights = composite(G, J, D, H, labels_supplied)
    cost, tokens = model_cost(root, Q)
    attempts = qa.get('attempts') or []
    if attempts:
        first_try = bool(attempts[0].get('passed'))
    elif qa:
        first_try = False if qa.get('passed') is False else None
    else:
        first_try = None
    elapsed = qa.get('elapsed_seconds')
    footage_ms = [s.get('duration_ms') for s in sources if isinstance(s.get('duration_ms'), int)]
    report = dict(
        schema_version='1.0', evaluation_id=uuid.uuid4().hex, episode_id=episode_id, goal=goal,
        created_at=datetime.now(timezone.utc).isoformat(), render_hash=render_hash, footage_hash=footage_hash,
        gates=dict(passed=gates['passed'], version=eval_gates.GATES_VERSION, thresholds=dict(eval_gates.THRESHOLDS),
                   checks=gates['checks']),
        scores=dict(G=G, D=D, J=round(J, 6) if J is not None else None, H=H, Q=Q, weights=weights, Q_status=q_status,
                    D_components=components, H_detail=H_detail, method_version=SCORING_VERSION),
        judge=judge, model_cost=cost, tokens=tokens,
        timing=dict(elapsed_seconds=float(elapsed) if isinstance(elapsed, (int, float)) and elapsed >= 0 else None,
                    footage_seconds=sum(footage_ms) / 1000 if footage_ms else None,
                    evaluation_seconds=round(time.monotonic() - started, 3)),
        provenance=provenance(qa, analysis), first_try_passed=first_try,
        limitations=list(dict.fromkeys(limitations)))
    validate_eval_report(report)
    if persist:
        persist_report(report, root, evals_root)
    return report
