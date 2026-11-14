"""Offline stdlib tests; no credentials or live endpoints are accessed."""
import concurrent.futures
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

# No dependency on the parent's package initializer or dependency installation.
MODULE_PATH = Path(__file__).resolve().parents[1] / 'pipelines/ffmpeg/llm_provider.py'
SPEC = importlib.util.spec_from_file_location('tested_llm_provider', MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
LLMProvider = module.LLMProvider
ProviderError = module.ProviderError


def response(provider='anthropic', *, text='{"ok":true}', usage=True):
    if provider == 'anthropic':
        data = {'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn'}
        if usage:
            data['usage'] = {'input_tokens': 1000, 'output_tokens': 100}
    else:
        data = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        if usage:
            data['usage'] = {'prompt_tokens': 1000, 'completion_tokens': 100}
    return json.dumps(data).encode()


def jpeg(width=640, height=360):
    # Minimal SOF/SOS/EOI JPEG fixture; no external imaging dependency required.
    return (b'\xff\xd8\xff\xc0\x00\x11\x08' + height.to_bytes(2, 'big')
            + width.to_bytes(2, 'big') + b'\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00'
            + b'\xff\xda\x00\x0c\x03\x01\x00\x02\x00\x03\x00\x00\x3f\x00\xff\xd9')


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        # Any accidental default transport invocation must fail, never hit network.
        self.no_network = patch.object(urllib.request.OpenerDirector, 'open',
                                       side_effect=AssertionError('Live HTTP prohibited'))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def provider(self, name='anthropic', transport=None):
        os.environ['ANTHROPIC_API_KEY' if name == 'anthropic' else 'OPENAI_API_KEY'] = 'secret-test-key'
        os.environ['FRAMEFOUNDRY_LLM_PROVIDER'] = name
        provider = LLMProvider.from_env()
        provider._transport = transport or (lambda *args: (200, response(name)))
        return provider

    def call(self, provider, **kwargs):
        args = dict(system='You are a planner', prompt='Plan a reel', images=[],
                    role='planner', episode_dir=self.directory)
        args.update(kwargs)
        return provider.complete_json(**args)

    def ledger(self):
        return json.loads((self.directory / 'llm-usage.json').read_text())['calls']

    def test_no_keys_and_explicit_missing_key(self):
        self.assertIsNone(LLMProvider.from_env())
        os.environ['OPENAI_API_KEY'] = 'test-openai'
        self.assertEqual(LLMProvider.from_env().provider, 'openai')
        os.environ['FRAMEFOUNDRY_LLM_PROVIDER'] = 'anthropic'
        self.assertIsNone(LLMProvider.from_env())

    def test_precedence_defaults_and_explicit_selection(self):
        os.environ.update(ANTHROPIC_API_KEY='test-a', OPENAI_API_KEY='test-o')
        provider = LLMProvider.from_env()
        self.assertEqual((provider.provider, provider.model), ('anthropic', 'claude-sonnet-4-5'))
        os.environ['FRAMEFOUNDRY_LLM_PROVIDER'] = 'openai'
        self.assertEqual(LLMProvider.from_env().model, 'gpt-4.1')

    def test_bad_provider_and_rate_configuration(self):
        os.environ['FRAMEFOUNDRY_LLM_PROVIDER'] = 'other'
        with self.assertRaises(ProviderError):
            LLMProvider.from_env()
        os.environ.pop('FRAMEFOUNDRY_LLM_PROVIDER')
        os.environ['ANTHROPIC_API_KEY'] = 'test'
        for value in ('NaN', 'Infinity', '-1', '0', 'oops'):
            with self.subTest(value=value):
                os.environ['FRAMEFOUNDRY_INPUT_USD_PER_MILLION'] = value
                os.environ['FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION'] = '15'
                with self.assertRaises(ProviderError):
                    LLMProvider.from_env()
        os.environ.pop('FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION')
        with self.assertRaises(ProviderError):
            LLMProvider.from_env()

    def test_both_multimodal_adapters_and_usage(self):
        image = self.directory / 'frame.jpg'
        image.write_bytes(jpeg())
        for name, expected_cost in (('anthropic', .0045), ('openai', .0028)):
            with self.subTest(provider=name):
                calls = []
                def transport(url, headers, body, timeout):
                    calls.append((url, headers, json.loads(body), timeout))
                    # Reservation must be on disk before HTTP starts.
                    self.assertEqual(self.ledger()[-1]['status'], 'failed')
                    self.assertGreater(self.ledger()[-1]['reserved_usd'], 0)
                    return 200, response(name)
                result = self.call(self.provider(name, transport), images=[image])
                self.assertEqual(result.data, {'ok': True})
                self.assertEqual(len(calls), 1)
                url, headers, payload, timeout = calls[0]
                self.assertEqual(timeout, 45)
                self.assertNotIn('tools', payload)
                content = payload['messages'][-1]['content']
                if name == 'anthropic':
                    self.assertEqual(url, 'https://api.anthropic.com/v1/messages')
                    self.assertEqual(headers['x-api-key'], 'secret-test-key')
                    encoded = content[0]['source']['data']
                    self.assertEqual(content[0]['source']['media_type'], 'image/jpeg')
                else:
                    self.assertEqual(url, 'https://api.openai.com/v1/chat/completions')
                    self.assertEqual(headers['Authorization'], 'Bearer secret-test-key')
                    encoded = content[0]['image_url']['url'].split(',', 1)[1]
                    self.assertEqual(payload['response_format'], {'type': 'json_object'})
                import base64
                self.assertEqual(base64.b64decode(encoded), image.read_bytes())
                self.assertEqual(result.usage['input_tokens'], 1000)
                self.assertEqual(result.usage['output_tokens'], 100)
                self.assertAlmostEqual(result.usage['cost_usd'], expected_cost)
                self.assertEqual(result.usage['cost_status'], 'estimated')
                self.assertEqual(result.usage['status'], 'success')
                self.assertEqual(self.ledger()[-1], result.usage)
        self.assertNotIn('secret-test-key', (self.directory / 'llm-usage.json').read_text())

    def test_unknown_model_blocks_and_override_enables(self):
        provider = self.provider()
        os.environ['FRAMEFOUNDRY_LLM_MODEL'] = 'custom-model'
        provider = LLMProvider.from_env()
        provider._transport = lambda *args: self.fail('Unknown rates must not dispatch')
        with self.assertRaisesRegex(ProviderError, 'Unknown model pricing'):
            self.call(provider)
        self.assertEqual(self.ledger()[-1]['status'], 'blocked')
        os.environ['FRAMEFOUNDRY_INPUT_USD_PER_MILLION'] = '4'
        os.environ['FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION'] = '12'
        provider = LLMProvider.from_env()
        provider._transport = lambda *args: (200, response())
        self.assertAlmostEqual(self.call(provider).usage['cost_usd'], .0052)

    def test_validation_blocks_without_transport(self):
        path = self.directory / 'image.jpg'
        path.write_bytes(jpeg(641))
        invalid = [dict(images=[path]), dict(images=[path] * 121),
                   dict(images=[path] * 120), dict(prompt='a' * 60001),
                   dict(prompt='😀' * 60000), dict(max_output_tokens=0),
                   dict(max_output_tokens=True), dict(max_output_tokens=16385),
                   dict(images=[self.directory / 'missing.jpg']),
                   dict(prompt='\ud800')]
        provider = self.provider(transport=lambda *args: self.fail('Invalid request dispatched'))
        for arguments in invalid:
            with self.subTest(arguments=str(arguments)[:60]):
                with self.assertRaises(ProviderError):
                    self.call(provider, **arguments)
                self.assertEqual(self.ledger()[-1]['status'], 'blocked')
                self.assertEqual(self.ledger()[-1]['reserved_usd'], 0)
        path.write_bytes(b'not a JPEG')
        with self.assertRaises(ProviderError):
            self.call(provider, images=[path])
        path.write_bytes(jpeg() + b'x' * module.MAX_IMAGE_BYTES)
        with self.assertRaises(ProviderError):
            self.call(provider, images=[path])

    def test_timeout_and_secret_redaction(self):
        for failure in (socket.timeout('secret-test-key'),
                        RuntimeError('secret-test-key'), ProviderError('secret-test-key')):
            def transport(*args):
                raise failure
            with self.assertRaises(ProviderError) as caught:
                self.call(self.provider(transport=transport), role='secret-test-key')
            self.assertNotIn('secret-test-key', str(caught.exception))
            usage = self.ledger()[-1]
            self.assertEqual(usage['status'], 'failed')
            self.assertIsNone(usage['input_tokens'])
            self.assertIsNone(usage['cost_usd'])
            self.assertGreater(usage['reserved_usd'], 0)
        self.assertNotIn('secret-test-key', (self.directory / 'llm-usage.json').read_text())

    def test_http_error_usage_retained_and_no_response_dump(self):
        raw = {'error': {'message': 'secret-test-key'},
               'usage': {'input_tokens': 200, 'output_tokens': 20}}
        with self.assertRaisesRegex(ProviderError, 'HTTP request failed'):
            self.call(self.provider(transport=lambda *args: (429, json.dumps(raw).encode())))
        usage = self.ledger()[-1]
        self.assertEqual(usage['input_tokens'], 200)
        self.assertAlmostEqual(usage['cost_usd'], .0009)
        self.assertEqual(usage['status'], 'failed')
        self.assertEqual(module._charged(usage), Decimal(str(usage['reserved_usd'])))
        self.assertNotIn('secret-test-key', (self.directory / 'llm-usage.json').read_text())

    def test_invalid_json_retains_tokens(self):
        for text in ('not json', '[]', '```json\n{}\n```', '{"x":NaN}', '{"x":1,"x":2}'):
            with self.subTest(text=text):
                with self.assertRaises(ProviderError):
                    self.call(self.provider(transport=lambda *args: (200, response(text=text))))
                self.assertEqual(self.ledger()[-1]['input_tokens'], 1000)
                self.assertEqual(self.ledger()[-1]['status'], 'failed')

    def test_no_usage_is_unknown_not_fake_zero(self):
        usage = self.call(self.provider(transport=lambda *args: (200, response(usage=False)))).usage
        self.assertIsNone(usage['input_tokens'])
        self.assertIsNone(usage['output_tokens'])
        self.assertIsNone(usage['cost_usd'])
        self.assertEqual(usage['cost_status'], 'unknown')
        self.assertEqual(module._charged(usage), Decimal(str(usage['reserved_usd'])))

    def test_invalid_or_partial_usage_and_cache_tokens(self):
        for raw_usage, expected in [({'input_tokens': True, 'output_tokens': 10}, None),
                                    ({'input_tokens': -1, 'output_tokens': 10}, None),
                                    ({'input_tokens': 4}, 4),
                                    ({'input_tokens': 10, 'output_tokens': 10,
                                      'cache_read_input_tokens': 20,
                                      'cache_creation_input_tokens': 30}, 60)]:
            raw = json.loads(response())
            raw['usage'] = raw_usage
            usage = self.call(self.provider(transport=lambda *args: (200, json.dumps(raw).encode()))).usage
            self.assertEqual(usage['input_tokens'], expected)
            if expected != 60:
                self.assertIsNone(usage['cost_usd'])

    def test_tool_calls_refusals_and_truncation_rejected(self):
        for name in ('anthropic', 'openai'):
            for kind in ('tool', 'truncated', 'refusal'):
                raw = json.loads(response(name))
                if name == 'anthropic':
                    if kind == 'tool':
                        raw['content'].append({'type': 'tool_use', 'name': 'evil'})
                    else:
                        raw['stop_reason'] = 'max_tokens' if kind == 'truncated' else 'refusal'
                else:
                    if kind == 'truncated':
                        raw['choices'][0]['finish_reason'] = 'length'
                    else:
                        raw['choices'][0]['message'][('tool_calls' if kind == 'tool' else 'refusal')] = ['evil']
                with self.subTest(provider=name, kind=kind):
                    with self.assertRaises(ProviderError):
                        self.call(self.provider(name, lambda *args: (200, json.dumps(raw).encode())))

    def test_budget_unknown_failures_and_successes_share_ledger(self):
        calls = []
        def failed_transport(*args):
            calls.append(1)
            raise socket.timeout()
        provider = self.provider(transport=failed_transport)
        while True:
            try:
                self.call(provider)
            except ProviderError:
                if self.ledger()[-1]['status'] == 'blocked':
                    break
        self.assertEqual(len(calls), 11)
        charged = sum((module._charged(call) for call in self.ledger()), Decimal(0))
        self.assertLessEqual(charged, Decimal(5))
        self.assertIn('$5', self.ledger()[-1]['error'])
        # A second provider still sees the same episode-wide spend.
        with self.assertRaises(ProviderError):
            self.call(self.provider('openai', lambda *args: self.fail('Budget exceeded')))

    def test_success_actual_cost_releases_excess_reservation(self):
        provider = self.provider()
        for _ in range(20):
            self.call(provider)
        self.assertTrue(all(call['status'] == 'success' for call in self.ledger()))
        self.assertAlmostEqual(float(sum(module._charged(call) for call in self.ledger())), .09)

    def test_parallel_reservations_cannot_overspend(self):
        provider = self.provider(transport=lambda *args: (200, response(usage=False)))
        def run(_):
            try:
                return self.call(provider).usage['status']
            except ProviderError:
                return 'blocked'
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            statuses = list(executor.map(run, range(20)))
        self.assertEqual(statuses.count('success'), 11)
        self.assertLessEqual(sum(module._charged(call) for call in self.ledger()), Decimal(5))
        self.assertEqual(len({call['call_id'] for call in self.ledger()}), 20)
        self.assertEqual(list(self.directory.glob('.llm-usage-*.tmp')), [])

    def test_corrupt_ledger_fails_closed_and_is_not_overwritten(self):
        for text in ('broken', '{"calls":[{"status":"failed","reserved_usd":-1,"cost_usd":null}]}',
                     '{"calls":[{"status":"failed","reserved_usd":NaN,"cost_usd":null}]}'):
            path = self.directory / 'llm-usage.json'
            path.write_text(text)
            with self.assertRaisesRegex(ProviderError, 'ledger is invalid'):
                self.call(self.provider(transport=lambda *args: self.fail('Corrupt ledger dispatched')))
            self.assertEqual(path.read_text(), text)

    def test_atomic_reservation_failure_prevents_dispatch(self):
        with patch.object(module, '_atomic_write', side_effect=OSError('secret-test-key')):
            with self.assertRaisesRegex(ProviderError, 'Unable to safely record'):
                self.call(self.provider(transport=lambda *args: self.fail('No reservation')))

    def test_malformed_or_oversized_envelope_is_recorded(self):
        for raw in (b'not json', b'[]', b'x' * (module.MAX_RESPONSE_BYTES + 1)):
            with self.assertRaises(ProviderError):
                self.call(self.provider(transport=lambda *args: (200, raw)))
            self.assertEqual(self.ledger()[-1]['status'], 'failed')
            self.assertIsNone(self.ledger()[-1]['cost_usd'])

    def test_transport_disables_redirects_and_captures_http_errors(self):
        self.assertIsNone(module._NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other'))
        import io
        error = urllib.error.HTTPError('https://test', 429, 'secret-test-key', {},
                                       io.BytesIO(b'{"usage":{}}'))
        fake_opener = unittest.mock.Mock()
        fake_opener.open.side_effect = error
        with patch.object(urllib.request, 'build_opener', return_value=fake_opener) as build:
            status, body = module._transport('https://test', {}, b'{}', 45)
        self.assertEqual((status, body), (429, b'{"usage":{}}'))
        self.assertEqual(fake_opener.open.call_count, 1)
        self.assertIsInstance(build.call_args.args[0], module._NoRedirect)
        self.assertEqual(fake_opener.open.call_args.kwargs['timeout'], 45)


if __name__ == '__main__':
    unittest.main()
