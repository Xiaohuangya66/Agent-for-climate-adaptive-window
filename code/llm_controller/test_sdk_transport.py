"""Mocked SDK contract tests; no live network requests."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import openai
from llm_controller.api import ChatClient, APIError


class SDKTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'DMX_API_KEY': 'sk-'+'x'*24,
            'DMX_MODEL': 'unit-test-model', 'DMX_BASE_URL': 'https://example.test',
            'LLM_HTTP_RETRIES': '2'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.fake = Mock()
        self.fake.chat.completions.create.return_value.model_dump.return_value = {
            'id': 'test-response', 'model': 'unit-test-model',
            'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None,
                'reasoning_content': 'provider field retained in memory', 'annotations': None,
                'tool_calls': [{'id':'call-test','type':'function',
                    'function':{'name':'ping','arguments':'{"value":1}'}}]}}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}

    def test_sdk_entry_base_normalization_and_reasoning_roundtrip(self):
        with patch('openai.OpenAI', return_value=self.fake) as constructor:
            client=ChatClient(); message, metadata=client.complete([{'role':'user','content':'test'}], [])
        self.assertEqual(constructor.call_args.kwargs['base_url'], 'https://example.test/v1')
        self.assertEqual(constructor.call_args.kwargs['max_retries'], 2)
        self.assertEqual(self.fake.chat.completions.create.call_args.kwargs['model'], 'unit-test-model')
        self.assertEqual(metadata['sdk_version'], '2.53.0')
        self.assertIn('reasoning_content', message); self.assertNotIn('annotations', message)
        self.assertEqual(client.usage['total_tokens'], 5)

    def test_existing_v1_prefix_is_not_duplicated(self):
        with patch.dict(os.environ, {'DMX_BASE_URL':'https://example.test/v1'}), patch('openai.OpenAI', return_value=self.fake) as constructor:
            ChatClient()
        self.assertEqual(constructor.call_args.kwargs['base_url'], 'https://example.test/v1')

    def test_request_cap_prevents_further_sdk_call(self):
        with patch('openai.OpenAI', return_value=self.fake):
            client=ChatClient(); client.max_calls=1; client.complete([],[])
            with self.assertRaises(APIError):client.complete([],[])
        self.assertEqual(self.fake.chat.completions.create.call_count,1)

    def test_error_body_never_exposes_key(self):
        secret=os.environ['DMX_API_KEY']
        response=httpx.Response(401, request=httpx.Request('POST','https://example.test/v1/chat/completions'),
                                text='denied '+secret)
        self.fake.chat.completions.create.side_effect=openai.APIStatusError('denied',response=response,body=None)
        with patch('openai.OpenAI', return_value=self.fake):
            with self.assertRaises(APIError) as caught:ChatClient().complete([],[])
        self.assertNotIn(secret,str(caught.exception)); self.assertIn('401',str(caught.exception))


if __name__ == '__main__':unittest.main()
