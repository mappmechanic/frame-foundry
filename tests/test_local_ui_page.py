"""Static packaging/API/security contracts; behavior is checked with offline Playwright mocks."""
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

PAGE = Path(__file__).resolve().parents[1] / 'pipelines/ffmpeg/local_ui/index.html'


class PageParser(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.fixture
def source():
    return PAGE.read_text()


def test_single_file_no_external_dependencies(source):
    elements = PageParser(source).elements
    assert sum(tag == 'style' for tag, _ in elements) == 1
    assert sum(tag == 'script' for tag, _ in elements) == 1
    assert not any(tag == 'script' and 'src' in attrs for tag, attrs in elements)
    assert not any(tag == 'link' for tag, _ in elements)
    assert not re.search(r'https?://|@import|url\(', source)
    csp = next(a['content'] for t, a in elements if t == 'meta' and a.get('http-equiv') == 'Content-Security-Policy')
    assert "connect-src 'self'" in csp
    assert "object-src 'none'" in csp


def test_inputs_and_safe_defaults(source):
    elements = {attrs['id']: (tag, attrs) for tag, attrs in PageParser(source).elements if 'id' in attrs}
    assert elements['goal'][1]['placeholder'] == 'Create a teaser from this raw family-trip footage that highlights the important moments'
    duration = elements['duration'][1]
    assert (duration['min'], duration['max']) == ('5', '120')
    assert 'value' not in duration
    assert 'checked' not in elements['voiceover'][1]
    assert 'disabled' in elements['voiceover'][1]
    assert 'webkitdirectory' in elements['folder'][1]
    assert 'multiple' in elements['files'][1]
    assert '<option value="9:16" selected>' in source
    assert 'No upload or copy.' in source
    assert 'unsupported files ignored' in source


def test_api_contract_and_explicit_post_only(source):
    for route in ('/api/config', '/api/uploads', '/api/jobs', '/api/episodes', '/regenerate'):
        assert route in source
    for field in ('footage_path', 'upload_id', 'target_seconds', 'aspect', 'voiceover', 'feedback'):
        assert field in source
    assert "data.append('files',f,f.webkitRelativePath || f.relativeName || f.name)" in source
    assert 'Math.min(40,cfg.max_files)' in source
    assert 'cfg.max_upload_bytes' in source
    assert "name.split('/').every" in source
    assert "safeName(n)" in source
    assert "method:'POST'" in source
    assert "addEventListener('submit'" in source
    assert 'Only an explicit submission starts processing.' in source
    assert 'No automatic retry.' in source
    assert 'localStorage.setItem(pendingKey' in source


def test_dynamic_content_and_allowlisted_media(source):
    assert 'textContent' in source
    assert not re.search(r'innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(', source)
    assert 'url.origin===location.origin && url.pathname===expected' in source
    assert '!url.search && !url.hash' in source
    assert "'finished.mp4'" in source and "'contact-sheet.jpg'" in source
    assert "video+'?download=1'" in source
    assert "el.getAttribute('src') !== url" in source  # Polling must not reset playback.
    iframe = next(attrs for tag, attrs in PageParser(source).elements if tag == 'iframe')
    assert iframe['sandbox'] is None  # No scripts, forms, or same-origin privilege.
    assert 'src' not in iframe  # Lazy, read-only report.
    assert "setAttribute('src','/scoreboard')" in source


def test_truthful_progress_history_and_evaluation(source):
    for stage in ('refinery', 'planner', 'assembler', 'validator', 'judge'):
        assert f'data-stage="{stage}"' in source
    for field in ('elapsed_seconds', 'qa_passed', 'known_usd', 'reservations_usd', 'quality_per_dollar',
                  'evaluation_status', 'fix_instruction', 'weakest_shot', 'coverage', 'limitations',
                  'start_ms', 'end_ms', 'reason', 'source_id', 'voiceover_text', 'can_regenerate'):
        assert field in source
    assert "Math.max(1000,state.config?.poll_interval_ms || 1000)" in source
    assert 'if(state.active)' in source
    assert "state.selected===id" in source
    assert '!!state.active || state.uncertain' in source
    assert 'localeCompare' in source and 'No episodes yet.' in source
    assert 'Stages may loop back for repairs.' in source
    assert 'not incurred AI cost' in source
    assert 'Separate from technical QA.' in source
    assert 'Unavailable' in source and 'Unknown' in source
    assert 'az0700' not in source and 'az-0700' not in source
