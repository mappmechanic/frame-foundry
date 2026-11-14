"""Dependency-free, single-request multimodal JSON providers.

Usage is a durable ledger at ``episode_dir/llm-usage.json`` with a ``calls``
array. Token counts are provider-observed; USD costs are *estimates* from
configured rates, not billing proof.

Budget model (per episode, shared across reruns and provider instances):
* ``CONTEXT_TOKENS`` (128k) is only the hard ceiling for one request. What is
  *reserved* before dispatch is a bounded estimate: UTF-8 bytes of system+prompt
  + a fixed protocol overhead + a per-image upper bound derived from the real
  JPEG dimensions (base64 text is never counted as LLM tokens) + the requested
  output allowance.
* Defaults: ``FRAMEFOUNDRY_EPISODE_API_BUDGET_USD`` = 0.25 and
  ``FRAMEFOUNDRY_EPISODE_TOKEN_BUDGET`` = 30000 (environment only).
* Paid-call caps per role: planner 2, judge 1, counting failed attempts. Blocked
  calls never dispatch, so they cost no money, tokens or call allowance.
* Failed/unknown calls keep their full reservation. Provider-reported usage above
  the reservation is recorded as an overrun and raises ``ProviderBudgetError``.
No retries, redirects, tools, remote image URLs, or secret/response logging.
JPEGs must already be <=640px on both axes; this module does not resize them.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
import fcntl
import json
import math
import os
import re
from pathlib import Path
import tempfile
from typing import Callable
import urllib.error
import urllib.request
import uuid

BUDGET_USD = Decimal('0.25')          # default episode USD allowance
TOKEN_BUDGET = 30_000                 # default episode token allowance
MAX_BUDGET_USD = Decimal('5')         # env override may not exceed the former ceiling
MAX_TOKEN_BUDGET = 1_000_000
CONTEXT_TOKENS = 128_000              # hard per-request ceiling, NOT the reservation
PROTOCOL_OVERHEAD_TOKENS = 1024       # message framing, roles, JSON-mode wrapper
IMAGE_TOKEN_CEILING = 4000            # conservative per-image cap / unknown-model value
ROLE_CALL_CAPS = {'planner': 2, 'judge': 1, 'pairwise':2}
BUDGET_ENV = 'FRAMEFOUNDRY_EPISODE_API_BUDGET_USD'
TOKEN_BUDGET_ENV = 'FRAMEFOUNDRY_EPISODE_TOKEN_BUDGET'
TIMEOUT_SECONDS = 45
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
DEFAULT_MODELS = {'anthropic': 'claude-sonnet-4-5', 'openai': 'gpt-4.1'}
DEFAULT_RATES = {'claude-sonnet-4-5': (Decimal('3'), Decimal('15')),
                 'gpt-4.1': (Decimal('2'), Decimal('8'))}


class ProviderError(RuntimeError):
    """Safe, fixed diagnostic; never includes credentials or provider bodies."""


class ProviderBudgetError(ProviderError):
    """Terminal budget, call-cap or overrun stop; callers must not fall back to heuristics."""


def episode_budget_usd() -> Decimal:
    raw = os.environ.get(BUDGET_ENV)
    if raw is None:
        return BUDGET_USD
    # ASCII decimals only: rejects NaN/Infinity/exponents/underscores/signs/blank.
    if not re.fullmatch(r'(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)', raw.strip()):
        raise ProviderError('Invalid episode API budget configuration')
    value = Decimal(raw.strip())
    if value <= 0 or value > MAX_BUDGET_USD:
        raise ProviderError('Invalid episode API budget configuration')
    return value


def episode_token_budget() -> int:
    raw = os.environ.get(TOKEN_BUDGET_ENV)
    if raw is None:
        return TOKEN_BUDGET
    if not re.fullmatch(r'[0-9]{1,9}', raw.strip()):
        raise ProviderError('Invalid episode token budget configuration')
    value = int(raw.strip())
    if not 1 <= value <= MAX_TOKEN_BUDGET:
        raise ProviderError('Invalid episode token budget configuration')
    return value


@dataclass(frozen=True)
class LLMResult:
    data: dict
    usage: dict


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _transport(url: str, headers: dict, body: bytes, timeout: int) -> tuple[int, bytes]:
    """One HTTP request, including HTTP errors; redirects deliberately disabled."""
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ProviderError('Provider response exceeded the size limit')
        return response.code, payload


def _strict_json(text: str | bytes):
    def no_constant(value):
        raise ValueError('Non-finite JSON value')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    return json.loads(text, parse_constant=no_constant, object_pairs_hook=unique)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    # Parse SOF dimensions without Pillow. No file extension trust or decoding.
    if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise ProviderError('Images must be JPEG files')
    offset = 2
    while offset < len(data):
        if data[offset] != 0xff:
            break
        while offset < len(data) and data[offset] == 0xff:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xd9, 0xda):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset:offset + 2], 'big')
        if length < 2 or offset + length > len(data):
            break
        if marker in (0xc0, 0xc1, 0xc2):
            if length < 8:
                break
            height = int.from_bytes(data[offset + 3:offset + 5], 'big')
            width = int.from_bytes(data[offset + 5:offset + 7], 'big')
            if not (1 <= width <= 640 and 1 <= height <= 640):
                raise ProviderError('JPEG dimensions must be between 1 and 640 pixels')
            return width, height
        offset += length
    raise ProviderError('Invalid or unsupported JPEG image')


@contextmanager
def _ledger_lock(episode_dir: Path):
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / '.llm-usage.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _is_count(value) -> bool:
    return type(value) is int and value >= 0


def _load_ledger(path: Path) -> dict:
    """Strictly validate. New fields are checked where present; legacy rows (no
    reserved token fields) are charged conservatively by ``_charged_tokens``."""
    if not path.exists():
        return {'version': 2, 'calls': []}
    try:
        ledger = _strict_json(path.read_text())
        if not isinstance(ledger, dict) or not isinstance(ledger['calls'], list):
            raise ValueError()
        if ledger.get('budget_status', 'ok') not in ('ok', 'overrun'):
            raise ValueError()
        for call in ledger['calls']:
            if not isinstance(call, dict) or call['status'] not in ('success', 'failed', 'blocked'):
                raise ValueError()
            for field in ('reserved_usd', 'cost_usd'):
                value = call[field]
                if field == 'cost_usd' and value is None:
                    continue
                if not _is_number(value):
                    raise ValueError()
            for field in ('reserved_input_tokens', 'reserved_output_tokens'):
                if field in call and not _is_count(call[field]):
                    raise ValueError()
            for field in ('input_tokens', 'output_tokens'):
                if call.get(field) is not None and not _is_count(call[field]):
                    raise ValueError()
            if not isinstance(call.get('overrun', False), bool):
                raise ValueError()
            if not isinstance(call.get('role', ''), str):
                raise ValueError()
        return ledger
    except Exception:
        raise ProviderError('Usage ledger is invalid; calls are blocked') from None


def _atomic_write(path: Path, ledger: dict):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.llm-usage-',
                                         suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(ledger, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def _usd_upper_bound(value: Decimal) -> float:
    """Round up for persistence; float conversion must never free budget."""
    rounded = value.quantize(Decimal('0.000000000001'), rounding=ROUND_CEILING)
    amount = float(rounded)
    if Decimal(str(amount)) < rounded:
        amount = math.nextafter(amount, math.inf)
    if not math.isfinite(amount):
        raise ValueError('Non-finite cost')
    return amount


def _charged(call: dict) -> Decimal:
    reserved = Decimal(str(call['reserved_usd']))
    cost = call['cost_usd']
    if call['status'] == 'success' and cost is not None:
        return Decimal(str(cost))
    return max(reserved, Decimal(str(cost or 0)))


def _reserved_tokens(call: dict) -> int:
    if call['status'] == 'blocked':
        return 0
    if 'reserved_input_tokens' in call:
        return call['reserved_input_tokens'] + call.get('reserved_output_tokens', 0)
    return CONTEXT_TOKENS      # legacy row: assume a whole context was reserved


def _charged_tokens(call: dict) -> int:
    """Observed tokens only for a success with complete usage; otherwise the
    reservation is retained (or the larger observed figure on overrun)."""
    if call['status'] == 'blocked':
        return 0
    used, produced = call.get('input_tokens'), call.get('output_tokens')
    if call['status'] == 'success' and used is not None and produced is not None:
        return used + produced
    return max(_reserved_tokens(call), (used or 0) + (produced or 0))


def _is_overrun(entry: dict) -> bool:
    """True when provider-reported usage or estimated cost exceeds the reservation."""
    if entry['status'] == 'blocked':
        return False
    used, produced = entry['input_tokens'], entry['output_tokens']
    if used is not None and used > entry['reserved_input_tokens']:
        return True
    if produced is not None and produced > entry['reserved_output_tokens']:
        return True
    cost = entry['cost_usd']
    return cost is not None and Decimal(str(cost)) > Decimal(str(entry['reserved_usd']))


def _tts_summary(episode_dir: Path) -> tuple[Decimal, int]:
    """Known charged TTS USD (shared episode budget) and the count of unpriced entries.
    No ledger means no TTS was recorded; an unreadable one fails closed."""
    path = Path(episode_dir) / 'tts-usage.json'
    if not path.exists():
        return Decimal(0), 0
    try:
        calls = _strict_json(path.read_text())['calls']
        total, unknown = Decimal(0), 0
        for call in calls:
            if call.get('status') == 'blocked':
                continue
            values = [call.get(name) for name in ('cost_usd', 'reserved_usd')
                      if call.get(name) is not None]
            if any(not _is_number(value) for value in values):
                raise ValueError()
            if values:
                total += max(Decimal(str(value)) for value in values)
            else:
                unknown += 1
        return total, unknown
    except Exception:
        raise ProviderError('TTS usage ledger is invalid; calls are blocked') from None


def _finalize(ledger: dict, budget: Decimal, tokens: int) -> None:
    spent = sum((_charged(call) for call in ledger['calls']), Decimal(0))
    used = sum(_charged_tokens(call) for call in ledger['calls'])
    over = (any(call.get('overrun') for call in ledger['calls'])
            or spent > budget or used > tokens)
    ledger.update(version=2, budget_usd=float(budget), token_budget=tokens,
                  budget_status='overrun' if over else 'ok')


class LLMProvider:
    def __init__(self, provider: str, api_key: str, model: str,
                 rates: tuple[Decimal, Decimal] | None, *,
                 transport: Callable | None = None):
        if provider not in DEFAULT_MODELS:
            raise ProviderError('Unsupported LLM provider')
        self.provider = provider
        self._api_key = api_key
        self.model = model
        self._rates = rates
        self._transport = transport or _transport

    @classmethod
    def from_env(cls) -> LLMProvider | None:
        """Select Anthropic first, unless explicitly selected. Environment only; never
        reads key or budget files."""
        episode_budget_usd(), episode_token_budget()    # invalid budget env fails early
        selected = os.environ.get('FRAMEFOUNDRY_LLM_PROVIDER')
        if selected is not None and selected not in DEFAULT_MODELS:
            raise ProviderError('Unsupported LLM provider configuration')
        if selected is None:
            selected = ('anthropic' if os.environ.get('ANTHROPIC_API_KEY') else 'openai')
        key = os.environ.get('ANTHROPIC_API_KEY' if selected == 'anthropic' else 'OPENAI_API_KEY')
        if not key or not key.strip():
            return None
        model = os.environ.get('FRAMEFOUNDRY_LLM_MODEL', DEFAULT_MODELS[selected])
        if not model or len(model) > 200 or key in model:
            raise ProviderError('Invalid LLM model configuration')
        rates = DEFAULT_RATES.get(model) if model == DEFAULT_MODELS[selected] else None
        overrides = [os.environ.get('FRAMEFOUNDRY_INPUT_USD_PER_MILLION'),
                     os.environ.get('FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION')]
        if any(value is not None for value in overrides):
            if not all(value is not None for value in overrides):
                raise ProviderError('Both LLM rate overrides are required')
            try:
                rates = tuple(Decimal(value) for value in overrides)
                if any(not rate.is_finite() or rate <= 0 or rate > 1_000_000 for rate in rates):
                    raise ValueError()
            except (InvalidOperation, ValueError):
                raise ProviderError('Invalid LLM rate configuration') from None
        return cls(selected, key, model, rates)

    def _safe(self, text: str) -> str:
        return text.replace(self._api_key, '[redacted]') if self._api_key else text

    def _entry(self, role: str) -> dict:
        return {'call_id': str(uuid.uuid4()), 'role': self._safe(str(role))[:200],
                'provider': self.provider, 'model': self._safe(self.model),
                'status': 'blocked', 'input_tokens': None, 'output_tokens': None,
                'cost_usd': None, 'cost_status': 'unknown', 'error': None,
                'reserved_usd': 0.0, 'reserved_input_tokens': 0,
                'reserved_output_tokens': 0, 'overrun': False}

    def _image_tokens(self, width: int, height: int) -> int:
        """Upper bound for one <=640px JPEG from its real dimensions, capped at 4000.
        Models without a known formula get the flat 4000-token ceiling."""
        if self.model != DEFAULT_MODELS[self.provider]:
            return IMAGE_TOKEN_CEILING
        if self.provider == 'anthropic':
            raw = -(-width * height // 750)                     # ~ w*h/750
        else:                                                   # 85 + 170 per 512px tile
            raw = 85 + 170 * (-(-width // 512)) * (-(-height // 512))
        raw = -(-raw * 5 // 4)                                  # +25% safety margin
        return min(IMAGE_TOKEN_CEILING, raw)

    def _payload(self, system, prompt, images, max_output_tokens):
        """Return (request body, bounded input-token estimate)."""
        if not isinstance(system, str) or not isinstance(prompt, str):
            raise ProviderError('System and prompt must be text')
        if len(system) + len(prompt) > 60_000:
            raise ProviderError('LLM text exceeds 60000 characters')
        if not isinstance(images, list) or len(images) > 120:
            raise ProviderError('LLM requests support at most 120 images')
        if (type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 16_384):
            raise ProviderError('Output token limit must be between 1 and 16384')
        instruction = '\nReturn only a valid JSON object. Do not use Markdown or tools.'
        # One token per UTF-8 byte upper-bounds text. Images add a dimension-based
        # bound below; base64 payload text is never charged as LLM tokens.
        input_bound = (len((system + prompt + instruction).encode('utf-8'))
                       + PROTOCOL_OVERHEAD_TOKENS)
        if input_bound + max_output_tokens > CONTEXT_TOKENS:
            raise ProviderError('Request exceeds the conservative context limit')
        encoded = []
        total_bytes = 0
        for path in images:
            try:
                with Path(path).open('rb') as handle:
                    data = handle.read(MAX_IMAGE_BYTES + 1)
            except (OSError, TypeError, ValueError):
                raise ProviderError('Unable to read JPEG image') from None
            if len(data) > MAX_IMAGE_BYTES:
                raise ProviderError('JPEG image exceeds the size limit')
            width, height = _jpeg_dimensions(data)
            input_bound += self._image_tokens(width, height)
            if input_bound + max_output_tokens > CONTEXT_TOKENS:
                raise ProviderError('Request exceeds the conservative context limit')
            total_bytes += len(data)
            if total_bytes * 4 // 3 > MAX_REQUEST_BYTES - 1_000_000:
                raise ProviderError('Image payload exceeds the size limit')
            encoded.append(base64.b64encode(data).decode('ascii'))
        if self.provider == 'anthropic':
            content = [{'type': 'image', 'source': {'type': 'base64',
                        'media_type': 'image/jpeg', 'data': image}} for image in encoded]
            content.append({'type': 'text', 'text': prompt})
            payload = {'model': self.model, 'max_tokens': max_output_tokens, 'temperature':0,
                       'system': system + instruction,
                       'messages': [{'role': 'user', 'content': content}]}
        else:
            content = [{'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + image, 'detail': 'high'}} for image in encoded]
            content.append({'type': 'text', 'text': prompt})
            payload = {'model': self.model, 'max_tokens': max_output_tokens, 'temperature':0,
                       'response_format': {'type': 'json_object'},
                       'messages': [{'role': 'system', 'content': system + instruction},
                                    {'role': 'user', 'content': content}]}
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        if len(body) > MAX_REQUEST_BYTES:
            raise ProviderError('Request payload exceeds the size limit')
        return body, input_bound

    def _read_usage(self, response: dict, entry: dict):
        usage = response.get('usage')
        if not isinstance(usage, dict):
            return
        names = ('input_tokens', 'output_tokens') if self.provider == 'anthropic' else (
            'prompt_tokens', 'completion_tokens')
        for target, source in zip(('input_tokens', 'output_tokens'), names):
            value = usage.get(source)
            if type(value) is int and value >= 0:
                entry[target] = value
        if self.provider == 'anthropic' and entry['input_tokens'] is not None:
            for name in ('cache_creation_input_tokens', 'cache_read_input_tokens'):
                value = usage.get(name, 0)
                if type(value) is not int or value < 0:
                    entry['input_tokens'] = None
                    break
                entry['input_tokens'] += value
        if (self._rates and entry['input_tokens'] is not None
                and entry['output_tokens'] is not None):
            cost = (Decimal(entry['input_tokens']) * self._rates[0]
                    + Decimal(entry['output_tokens']) * self._rates[1]) / 1_000_000
            try:
                amount = _usd_upper_bound(cost)
            except (InvalidOperation, OverflowError, ValueError):
                return
            if math.isfinite(amount):
                entry['cost_usd'] = amount
                entry['cost_status'] = 'estimated'

    def _parse_data(self, response: dict) -> dict:
        try:
            if self.provider == 'anthropic':
                if response.get('stop_reason') != 'end_turn':
                    raise ValueError()
                content = response['content']
                if not isinstance(content, list) or not content or any(
                        block.get('type') != 'text' for block in content):
                    raise ValueError()
                text = ''.join(block['text'] for block in content)
            else:
                choices = response['choices']
                if len(choices) != 1 or choices[0]['finish_reason'] != 'stop':
                    raise ValueError()
                message = choices[0]['message']
                if message.get('tool_calls') or message.get('function_call') or message.get('refusal'):
                    raise ValueError()
                text = message['content']
            data = _strict_json(text)
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except Exception:
            raise ProviderError('Provider returned invalid JSON or incomplete output') from None

    def _admission_error(self, ledger, role, reservation, reserved_in, reserved_out,
                         budget, token_budget, episode_dir) -> str | None:
        """Reason this call may not be reserved, or None. Runs under the ledger lock."""
        calls = ledger['calls']
        if ledger.get('budget_status') == 'overrun' or any(c.get('overrun') for c in calls):
            return 'Episode LLM budget was overrun; calls are blocked'
        cap = ROLE_CALL_CAPS.get(role)
        if cap is not None and sum(1 for c in calls if c.get('role') == role
                                   and c['status'] != 'blocked') >= cap:
            return f'Episode {role} paid-call cap of {cap} reached'
        tts_usd, _ = _tts_summary(episode_dir)
        spent = sum((_charged(c) for c in calls), Decimal(0)) + tts_usd
        if spent + reservation > budget:
            return f"Episode LLM budget would exceed ${format(budget.normalize(), 'f')}"
        if sum(_charged_tokens(c) for c in calls) + reserved_in + reserved_out > token_budget:
            return f'Episode LLM token budget would exceed {token_budget} tokens'
        return None

    def complete_json(self, *, system: str, prompt: str, images: list[Path],
                      role: str, episode_dir: Path, max_output_tokens: int = 4096) -> LLMResult:
        """Reserve, send at most once, record observed usage, and return a JSON dict.

        Raises ProviderError on configuration, validation, transport or JSON failure and
        its subclass ProviderBudgetError for terminal budget, role-cap or overrun stops.
        A durable failed reservation (USD and tokens) is written under the ledger lock
        *before* network dispatch, so crashes cannot free an attempt's budget. Ledger
        corruption fails closed; it is never overwritten or treated as zero spending.
        There is no automatic retry.
        """
        budget, token_budget = episode_budget_usd(), episode_token_budget()
        entry = self._entry(role)
        path = Path(episode_dir) / 'llm-usage.json'
        try:
            body = None
            reserved_in = 0
            budget_stop = False
            try:
                if self._rates is None:
                    raise ProviderError('Unknown model pricing; configure both rate overrides')
                body, reserved_in = self._payload(system, prompt, images, max_output_tokens)
                reservation = Decimal(str(_usd_upper_bound(
                    (reserved_in * self._rates[0] + max_output_tokens * self._rates[1])
                    / 1_000_000)))
            except ProviderError as exc:
                entry['error'] = str(exc)
                body = None
            except Exception:
                entry['error'] = 'Invalid LLM request'
                body = None
            with _ledger_lock(path.parent):
                ledger = _load_ledger(path)
                if body is not None:
                    stop = self._admission_error(ledger, entry['role'], reservation, reserved_in,
                                                 max_output_tokens, budget, token_budget,
                                                 path.parent)
                    if stop:
                        entry['error'] = stop
                        budget_stop = True
                    else:
                        entry.update(status='failed', reserved_usd=float(reservation),
                                     reserved_input_tokens=reserved_in,
                                     reserved_output_tokens=max_output_tokens,
                                     error='Request did not complete')
                ledger['calls'].append(entry.copy())
                _finalize(ledger, budget, token_budget)
                _atomic_write(path, ledger)
            if entry['status'] == 'blocked':
                raise (ProviderBudgetError if budget_stop else ProviderError)(entry['error'])
            if self.provider == 'anthropic':
                url = 'https://api.anthropic.com/v1/messages'
                headers = {'Content-Type': 'application/json', 'x-api-key': self._api_key,
                           'anthropic-version': '2023-06-01'}
            else:
                url = 'https://api.openai.com/v1/chat/completions'
                headers = {'Content-Type': 'application/json',
                           'Authorization': 'Bearer ' + self._api_key}
            data = None
            try:
                try:
                    status, raw = self._transport(url, headers, body, TIMEOUT_SECONDS)
                except Exception:
                    raise ProviderError('Provider transport failed') from None
                if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
                    raise ProviderError('Provider response exceeded the size limit')
                try:
                    response = _strict_json(raw)
                    if not isinstance(response, dict):
                        raise ValueError()
                except Exception:
                    raise ProviderError('Provider returned an invalid response') from None
                self._read_usage(response, entry)
                if not 200 <= status < 300:
                    raise ProviderError('Provider HTTP request failed')
                data = self._parse_data(response)
                entry.update(status='success', error=None)
            except Exception as exc:
                entry.update(status='failed', error=(str(exc) if isinstance(exc, ProviderError)
                                                     else 'Provider transport failed'))
            overrun = _is_overrun(entry)
            if overrun:
                entry.update(overrun=True, error='Provider usage exceeded the reserved budget')
            with _ledger_lock(path.parent):
                ledger = _load_ledger(path)
                for index, call in enumerate(ledger['calls']):
                    if call['call_id'] == entry['call_id']:
                        ledger['calls'][index] = entry.copy()
                        break
                else:
                    raise ProviderError('Usage reservation is missing; calls are blocked')
                _finalize(ledger, budget, token_budget)
                _atomic_write(path, ledger)
            if overrun:
                raise ProviderBudgetError(entry['error'])
            if entry['status'] != 'success':
                raise ProviderError(entry['error'])
            return LLMResult(data=data, usage=entry.copy())
        except ProviderError:
            raise
        except Exception:
            raise ProviderError('Unable to safely record LLM usage; calls are blocked') from None
