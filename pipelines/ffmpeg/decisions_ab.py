"""Fail-closed, read-only projection contract for a separately recorded matched A/B.

Does not dispatch calls, mutate default selection, or certify local-only runs.
Caller supplies immutable matching configuration alongside two episode directories.
"""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from .contracts import read_json

MATCH_FIELDS=('footage_hash','goal','sections','render','overlays','voiceover','asr','planner_model','judge_protocol','scoring_protocol','cache_policy')


def _number(value, unit=False):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0 and (not unit or value <= 1)


def arm(directory, config):
    root=Path(directory)
    def load(name):
        try: return read_json(root/name)
        except (OSError,ValueError): return {}
    evaluation=load('eval-report.json'); decisions=load('decisions-report.json'); qa=load('qa-report.json')
    # Existing evaluator includes planning, decider, ASR, Judge and TTS ledgers.
    cost=evaluation.get('model_cost',{}); scores=evaluation.get('scores',{})
    return dict(episode_id=evaluation.get('episode_id',decisions.get('episode_id')),Q=scores.get('Q'),J=scores.get('J'),
                total_model_cost_usd=cost.get('usd'),cost_status=cost.get('status','unknown'),
                quality_per_dollar=cost.get('quality_per_dollar'),elapsed_seconds=qa.get('elapsed_seconds'),
                hard_qa_passed=qa.get('passed') is True and evaluation.get('gates',{}).get('passed') is True and scores.get('G')==1,
                requested_mode=decisions.get('requested_mode'),effective_mode=decisions.get('effective_mode'),
                fallback_reason=decisions.get('fallback_reason'),config=config,
                actual_footage_hash=evaluation.get('footage_hash'),actual_goal=evaluation.get('goal'),
                decision_call_id=(decisions.get('usage') or {}).get('call_id'))


def compare(baseline, experimental):
    reasons=[]
    for label,a in [('baseline',baseline),('decisions',experimental)]:
        if not a.get('episode_id'): reasons.append(label+': missing episode identity')
        if not a.get('hard_qa_passed'): reasons.append(label+': hard QA not passed')
        if not all(_number(a.get(k),True) for k in ('Q','J')): reasons.append(label+': missing Q/J')
        if a.get('cost_status') not in ('estimated','measured') or not _number(a.get('total_model_cost_usd')): reasons.append(label+': incomplete total model cost')
        if not _number(a.get('elapsed_seconds')): reasons.append(label+': missing latency')
        config=a.get('config') or {}
        if any(k not in config or config[k] is None for k in MATCH_FIELDS): reasons.append(label+': incomplete matching configuration')
        if not config.get('footage_hash') or config.get('footage_hash') != a.get('actual_footage_hash') or config.get('goal') != a.get('actual_goal'): reasons.append(label+': source/Goal evidence mismatch')
        if config.get('judge_protocol')!='gpt-4.1-current-six-criteria': reasons.append(label+': noncanonical Judge protocol')
    if baseline.get('episode_id')==experimental.get('episode_id'): reasons.append('Arms must be different Episodes')
    if baseline.get('requested_mode')!='llm' or baseline.get('effective_mode')!='llm': reasons.append('Baseline did not use llm')
    if experimental.get('requested_mode')!='decisions' or experimental.get('effective_mode')!='decisions' or experimental.get('fallback_reason') or not experimental.get('decision_call_id'): reasons.append('Experimental arm did not successfully use Decisions')
    if any((baseline.get('config') or {}).get(k)!=(experimental.get('config') or {}).get(k) for k in MATCH_FIELDS): reasons.append('Unmatched configuration')
    if all(_number(a.get('Q'),True) for a in (baseline,experimental)):
        if abs(Decimal(str(baseline['Q']))-Decimal(str(experimental['Q'])))>Decimal('.02'): reasons.append('Absolute Q delta exceeds 0.02')
    if all(_number(a.get('total_model_cost_usd')) for a in (baseline,experimental)):
        if experimental['total_model_cost_usd']>=baseline['total_model_cost_usd']: reasons.append('Decisions total cost is not strictly lower')
    return dict(schema_version='1.0',experiment='decider-selection',status='unqualified' if reasons else 'qualified',
                default_activated=False,reasons=reasons,arms=dict(llm=baseline,decisions=experimental),
                evidence_hash=hashlib.sha256(json.dumps([baseline,experimental],sort_keys=True,allow_nan=False).encode()).hexdigest())
