"""One inexpensive native function-call check; no numerical dependencies or runs."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from llm_controller.api import ChatClient


def main():
    client = ChatClient()
    message, metadata = client.complete([
        {'role': 'user', 'content': 'Call ping with value=1 to test native function calling.'}],
        [{'type': 'function', 'function': {'name': 'ping', 'description': 'Connection test only.',
          'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}},
                         'required': ['value'], 'additionalProperties': False}}}])
    calls = message.get('tool_calls') or []
    good = (len(calls) == 1 and calls[0]['function']['name'] == 'ping' and
            json.loads(calls[0]['function']['arguments']) == {'value': 1})
    print(json.dumps({'status': 'PASS' if good else 'FAIL', 'native_function_call': good,
                      'usage': metadata['usage'], 'latency_seconds': metadata['latency_seconds']}, indent=2))
    if not good:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
