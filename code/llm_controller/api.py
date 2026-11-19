"""Small Chat Completions transport. All provider settings are environment-only."""
import json
import os
import re
import time
import urllib.parse
import logging


def redact(value):
    """Defense in depth for provider messages; never serialize credentials."""
    text = str(value)
    secret = os.environ.get('DMX_API_KEY')
    if secret:
        text = text.replace(secret, '[REDACTED]')
    return re.sub(r'sk-[A-Za-z0-9_-]{12,}', '[REDACTED]', text)


class APIError(RuntimeError):
    pass


class ChatClient:
    def __init__(self):
        missing = [k for k in ('DMX_API_KEY', 'DMX_BASE_URL', 'DMX_MODEL') if not os.environ.get(k)]
        if missing:
            raise APIError('Missing environment variables: ' + ', '.join(missing))
        self._key = os.environ['DMX_API_KEY']
        self.model = os.environ['DMX_MODEL']
        base = os.environ['DMX_BASE_URL'].rstrip('/')
        url = urllib.parse.urlparse(base)
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise APIError('DMX_BASE_URL must be an HTTPS API root without credentials/query/fragment')
        if url.path in ('', '/'):base += '/v1'
        self.endpoint = base + '/chat/completions'
        self.timeout = float(os.environ.get('LLM_TIMEOUT_SECONDS', '120'))
        self.max_tokens = int(os.environ.get('LLM_MAX_OUTPUT_TOKENS', '1600'))
        self.max_calls = int(os.environ.get('LLM_MAX_CALLS', '32'))
        if not 1 <= self.max_calls <= 100 or not 1 <= self.timeout <= 300 or not 128 <= self.max_tokens <= 8192:
            raise APIError('Invalid API resource limits')
        import openai
        if openai.__version__ != '2.53.0':
            raise APIError('This formal adapter requires openai==2.53.0')
        self.sdk_version = openai.__version__
        self.http_retries = int(os.environ.get('LLM_HTTP_RETRIES', '2'))
        if not 0 <= self.http_retries <= 2:raise APIError('LLM_HTTP_RETRIES must be 0..2')
        self.client = openai.OpenAI(api_key=self._key, base_url=base,
                                   timeout=self.timeout, max_retries=self.http_retries)
        for logger in ('openai', 'httpx', 'httpcore'):
            logging.getLogger(logger).setLevel(logging.WARNING)
        self.calls = 0
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def complete(self, messages, schemas):
        if self.calls >= self.max_calls:
            raise APIError('LLM request limit reached; no rule-policy fallback')
        self.calls += 1
        payload = {'model': self.model, 'messages': messages, 'tools': schemas,
                   'tool_choice': 'auto', 'parallel_tool_calls': False, 'max_tokens': self.max_tokens}
        start = time.monotonic()
        # HTTP retries finish before a tool is executed, so cannot duplicate a
        # physics dispatch. Tool idempotency remains independently enforced.
        import openai
        try:
            response = self.client.chat.completions.create(**payload)
            data = response.model_dump(exclude_none=False)
        except openai.APIStatusError as exc:
            body = redact(exc.response.text)[:1200]
            raise APIError(f'Provider HTTP {exc.status_code}: {body}') from None
        except Exception as exc:
            raise APIError(f'Provider transport failed: {redact(exc)}') from None
        if not isinstance(data.get('choices'), list) or not data['choices']:
            raise APIError('Provider returned no choice')
        message = data['choices'][0].get('message', {})
        if message.get('role') != 'assistant':
            raise APIError('Invalid assistant message')
        # Preserve the provider's reasoning field for round-trip compatibility,
        # while excluding SDK-only nullable fields from subsequent messages.
        message = {k: v for k, v in message.items() if k in
                   {'role', 'content', 'tool_calls', 'reasoning_content'} and (v is not None or k == 'content')}
        usage = data.get('usage') or {}
        for key in self.usage:
            self.usage[key] += int(usage.get(key, 0) or 0)
        return message, {'request_number': self.calls, 'latency_seconds': time.monotonic()-start,
                         'usage': usage, 'finish_reason': data['choices'][0].get('finish_reason'),
                         'response_id': data.get('id'), 'returned_model': data.get('model'),
                         'sdk': 'openai', 'sdk_version': self.sdk_version,
                         'sdk_entry': 'client.chat.completions.create',
                         'configured_http_retries': self.http_retries}
