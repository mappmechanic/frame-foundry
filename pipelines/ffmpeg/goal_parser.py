"""Local, deterministic hard constraints; model output cannot relax these."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

from .contracts import StageError


@dataclass(frozen=True)
class GoalIntent:
    target_duration_ms: int
    aspect_ratio: str
    requested_terms: list[str]
    interpretation: str
    assumptions: list[str]


_NUMBER = r'(?:\d+(?:\.\d+)?|\.\d+)'
_UNIT = r'(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b'
_DURATION = re.compile(rf'(?<![\w.:])({_NUMBER})\s*({_UNIT})?\s*(?:[-–—]|\bto\b)\s*({_NUMBER})\s*({_UNIT})|(?<![\w.:])({_NUMBER})\s*({_UNIT})', re.I)
_STOP = set('a an the my our your their we us i me of for from in on at to and or with without make create produce edit use using all raw footage video videos clip clips reel reels teaser trip family highlights highlight vacation holiday please show include featuring focus moments moment important best into output duration second seconds sec secs minute minutes min mins short quick long longer extended vertical portrait landscape horizontal aspect ratio format then but instead chronological montage fun cinematic'.split())


def _constraint(message):
    raise StageError('GOAL_CONSTRAINT_UNSUPPORTED', 'planner', message)


def _milliseconds(value, unit):
    unit = unit.lower()
    if unit.startswith(('h',)):
        _constraint('Hours are not a supported teaser duration')
    ms = Decimal(value) * (60000 if unit.startswith('m') else 1000)
    if not 5000 <= ms <= 120000:
        _constraint('Explicit duration must be within 5–120 seconds')
    if ms != ms.to_integral_value():
        _constraint('Duration must resolve to whole milliseconds')
    return int(ms)


def parse_goal(goal: str) -> GoalIntent:
    if not isinstance(goal, str):
        _constraint('Goal must be text')
    assumptions = []
    ratios = re.findall(r'\b\d+\s*:\s*\d+\b', goal)
    ratios = [re.sub(r'\s', '', r) for r in ratios]
    if any(r not in ('9:16', '16:9') for r in ratios):
        _constraint('Unsupported aspect ratio; use 9:16 or 16:9')
    if re.search(r'\b(?:square|1\s*[x/]\s*1)\b', goal, re.I):
        _constraint('Square output is unsupported')
    # Mask ratios before duration matching: 16:9 seconds is never nine seconds.
    text = re.sub(r'\b\d+\s*:\s*\d+\b', lambda m: ' ' * len(m[0]), goal)
    for negative in re.finditer(rf'-\s*({_NUMBER})\s*({_UNIT})', text, re.I):
        if not re.search(r'\d(?:\s*' + _UNIT + r')?\s*$', text[:negative.start()], re.I):
            _constraint('Negative duration is unsupported')
    matches = list(_DURATION.finditer(text))
    durations = []
    for match in matches:
        a, au, b, bu, single, su = match.groups()
        if single is not None:
            duration = _milliseconds(single, su)
        else:
            low, high = _milliseconds(a, au or bu), _milliseconds(b, bu)
            if low > high:
                _constraint('Duration range is reversed')
            duration = (low + high) // 2
        durations.append(duration)
    if len(set(durations)) > 1:
        assumptions.append('Conflicting durations: last explicit instruction wins; ignored earlier targets ' + ', '.join(str(x) + 'ms' for x in durations[:-1]))
    if durations:
        target = durations[-1]
    else:
        cues = list(re.finditer(r'\b(short|quick|long|longer|extended)\b', text, re.I))
        target = (30000 if cues[-1][0].lower() in ('short', 'quick') else 60000) if cues else 45000
        if len({m[0].lower() in ('short', 'quick') for m in cues}) > 1:
            assumptions.append('Conflicting qualitative duration cues: last cue wins')
        if not cues:
            assumptions.append('No duration specified; use 45 seconds')
    landscape = '16:9' in ratios or bool(re.search(r'\b(landscape|horizontal)\b', goal, re.I))
    portrait = '9:16' in ratios or bool(re.search(r'\b(portrait|vertical)\b', goal, re.I))
    aspect = '16:9' if landscape else '9:16'
    if landscape and portrait:
        assumptions.append('Conflicting aspect instructions: landscape 16:9 wins')
    elif not portrait and not landscape:
        assumptions.append('No aspect specified; use vertical 9:16')
    for match in reversed(matches):
        text = text[:match.start()] + ' ' * (match.end() - match.start()) + text[match.end():]
    terms = list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text) if w.lower() not in _STOP))
    return GoalIntent(target, aspect, terms, f'Source-backed highlights teaser; target {target / 1000:g}s, {aspect}', assumptions)
