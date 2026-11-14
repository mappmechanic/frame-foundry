"""Offline WO-17 documentation contracts; not a media/policy acceptance run."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / '.claude/skills'
SOP = ROOT / 'docs/sops/SOP-03-remotion-sparse-overlays.md'
NAMES = ('rough-cut', 'vlog-pacing', 'voice-and-audio',
         'remotion-sparse-overlays', 'visual-qa-validator')
SOURCES = {
    'kajisho5/remotion-ffmpeg-skill': (
        'e34a10dd2fefcc1cccd07352e272b89dcc6b3f3c', 'Copyright (c) 2026 kajisho5'),
    'haidrrrry/claude-remotion-skill': (
        '1dcbe5e3fc6cf970bd10d3cc05f0a8a5d19d0383', 'Copyright (c) 2025 haidrrrry'),
    'artemnovitckii/claude-editing-skills': (
        'beff6b868a8c1bdfad8661bd8a298497e75568a3', 'Copyright (c) 2026 Artem Novitckii'),
}


def manifest():
    return json.loads((ROOT / 'docs/skills/provenance.json').read_text())


@pytest.mark.parametrize('name', NAMES)
def test_skill_frontmatter_routing_and_local_attribution(name):
    path = SKILLS / name / 'SKILL.md'
    text = path.read_text()
    assert text.startswith(f'---\nname: {name}\ndescription: ')
    assert len(text.split('---', 2)[1].splitlines()) == 3
    assert f'.claude/skills/{name}/SKILL.md' in (ROOT / 'AGENTS.md').read_text()
    assert 'Attribution / license' in text
    assert 'MIT' in text
    assert '../../../LICENSE' in text
    assert 'provenance.json' in text
    assert 'declared target' in text.lower()
    assert len(text.splitlines()) <= 85  # Curated guidance, not bulk upstream prompts.
    for source in manifest()['sources']:
        if name in source['adapted_in']:
            assert source['repository'] in text
            assert source['copyright'] in text
            assert source['commit'] in text
            assert Path(source['license_file']).name in text


def test_pins_and_complete_mit_notices():
    data = manifest()
    assert {s['repository'] for s in data['sources']} == set(SOURCES)
    assert data['original_skills'] == ['rough-cut']
    for source in data['sources']:
        sha, notice = SOURCES[source['repository']]
        assert source['commit'] == sha
        assert source['license'] == 'MIT'
        assert source['copyright'] == notice
        path = ROOT / source['license_file']
        contents = path.read_bytes()
        assert hashlib.sha256(contents).hexdigest() == source['license_sha256']
        text = contents.decode()
        for required in (notice, 'Permission is hereby granted, free of charge',
                         'The above copyright notice and this permission notice',
                         'copies or substantial portions of the Software.',
                         'THE SOFTWARE IS PROVIDED "AS IS"',
                         'THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE', 'SOFTWARE.'):
            assert required in text
        assert source['copied_paths'] == ['LICENSE']
        assert source['inspected_paths']
        assert all(not p.startswith('/') and '..' not in Path(p).parts
                   for p in source['inspected_paths'])
        assert set(source['adapted_in']) <= set(NAMES)
    artem = next(s for s in data['sources'] if s['repository'].startswith('artem'))
    text = (ROOT / artem['license_file']).read_text()
    assert 'SIL Open Font License 1.1' in text
    assert 'excluded from this software license' in text
    assert 'assets/style-showcase' in text and 'docs/style-showcase' in text


def test_vendor_staging_ignored_and_no_tracked_assets():
    paths = ['_vendor_skills/remotion-ffmpeg-skill/LICENSE',
             '_vendor_skills/claude-remotion-skill/demo.mp4',
             '_vendor_skills/claude-editing-skills/kit/fonts/TikTokSans.ttf']
    result = subprocess.run(['git', 'check-ignore', '--no-index', *paths],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == paths
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    assert not any(p.startswith('_vendor_skills/') for p in tracked)
    # Every curated payload is text; upstream assets/installers cannot sneak into a skill.
    for directory in (SKILLS, ROOT / 'docs/skills', ROOT / 'docs/sops'):
        for path in directory.rglob('*'):
            if path.is_file():
                assert path.suffix in {'.md', '.json', '.txt'}, path
                path.read_text(encoding='utf-8')
    excluded = manifest()['excluded_sources']
    assert {s['repository'] for s in excluded} == {
        'appariciojunior', 'remotion-dev/skills', 'OpenMontage'}
    assert all('not cloned, read or copied' in s['reason'] for s in excluded)


def test_authored_relative_links_resolve():
    docs = [ROOT / 'AGENTS.md', ROOT / 'docs/skills/README.md', SOP,
            *[SKILLS / name / 'SKILL.md' for name in NAMES]]
    for doc in docs:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', doc.read_text()):
            if re.match(r'^[a-z]+://', target) or target.startswith('#'):
                continue
            resolved = (doc.parent / target.split('#', 1)[0]).resolve()
            assert resolved.is_relative_to(ROOT), (doc, target)
            assert resolved.is_file(), (doc, target)


def test_baseline_authority_and_target_are_distinct():
    text = SOP.read_text()
    for required in ('Measure -> Draw -> Verify', 'ea0887e', 'Pillow',
                     'sparse active-window', 'absent', 'not installed',
                     'SOP-01/SOP-02', 'contents were not read', 'V1/A1',
                     'V3', 'ProRes 4444', 'WebM alpha', 'not a new JSON schema',
                     '80ms pre-roll / 120ms post-roll', '[-15,-13] LUFS',
                     'true peak <=-1 dBTP', '0.06H', '116px', '2.5s',
                     'cut-1/cut/cut+1', 'cut+3 exclusion',
                     'cut-1 through cut+3 inclusive', 'cut+4',
                     'bottom 22% / right 15% / top 10%', '16:9 outer 10%',
                     'eyes', 'chin', 'damping=14, stiffness=120, mass=1',
                     'No async React', 'pixels not inspected',
                     'ordinary edits stay', 'No improvement claim',
                     'MIT licenses of these guidance repositories do **not**'):
        assert required.replace("**", "") in re.sub(r"\s+", " ", text.replace("**", "")), required
    matrix = (ROOT / 'docs/skills/README.md').read_text()
    assert 'Declared target' in matrix
    assert 'This branch adds no policy switch or persisted field' in matrix
    assert 'not visual inspection or end-to-end policy verification' in matrix
    # Baseline gates remain declared accurately without importing or altering runtime.
    style = json.loads((ROOT / 'contracts/overlay-style.json').read_text())
    assert style['cap_height_ratio'] == .06
    assert style['minimum_hold_ms'] == 2500
    assert style['safe_top'] == .12
    assert style['slide_px'] == 8


def test_original_editorial_rules_do_not_damage_protected_speech():
    rough = (SKILLS / 'rough-cut/SKILL.md').read_text()
    for required in ('>350ms only within verified silence', 'protected word intervals',
                     '30s source-local window', 'final complete exact or qualified take',
                     'incomplete final attempt', 'negations', '80ms pre-roll / 120ms post-roll',
                     'No upstream skill text or code copied or adapted'):
        assert required in rough
    pacing = (SKILLS / 'vlog-pacing/SKILL.md').read_text()
    for required in ('first 2s', '1.5–4s shots', 'protected long-speech exception',
                     '112% punch-in', 'anchor_y=0.33', 'face-safe evidence',
                     'keep\n  neutral framing', 'no new Planner call'):
        assert required in pacing
    audio = (SKILLS / 'voice-and-audio/SKILL.md').read_text()
    assert 'music absent' in audio
    assert 'unauthorized' in audio
    assert 'the encoded output' in audio
    assert 'not download' in audio


def test_supported_check_commands_not_vendor_or_remotion_installers():
    text = SOP.read_text()
    shell = re.findall(r'```sh\n(.*?)```', text, re.S)
    assert len(shell) == 1
    assert 'pipelines.ffmpeg.qa_validator --help' in shell[0]
    assert 'pipelines.ffmpeg.cli --help' in shell[0]
    for line in shell[0].splitlines():
        assert line.startswith('python -m ')
        assert not any(s in line for s in ('npx', 'npm', '_vendor_skills', 'install'))
    qa = (ROOT / 'pipelines/ffmpeg/qa_validator.py').read_text()
    assert "parser.add_argument('--episode-dir'" in qa
    assert "'qa-report.json'" in qa and "'finished.mp4'" in qa
