"""The same LLM first resolves assets; numerical Agent is instantiated afterward."""
import copy
from datetime import datetime, timezone
import json

from llm_controller.api import redact
from llm_controller.city_assets import write_json, design_spec
from llm_controller.controller import ToolRejected, schemas, validate
from llm_controller.prompts import SYSTEM_PROMPT, PROMPT_VERSION


class CityBootstrapSession:
    def __init__(self, service, request, city_hint=None, run_options=None):
        self.service, self.request = service, request
        self.city_hint = service.canonical_city(city_hint) if city_hint else None
        self.phase, self.version = 'city_request', 0
        self.cache = {}
        self.clarification = None
        self.run_options = run_options or {}
        self.design_spec = None

    def allowed(self):
        return {'city_request': ['resolve_city', 'clarify_city'],
                'prepare': ['prepare_city_assets'], 'baseline': ['run_clear_glass_baseline'],
                'ready': [], 'blocked': []}[self.phase]

    def state(self):
        return {'state_version': self.version, 'phase': self.phase, 'allowed_tools': self.allowed(),
            'budget_used': 0,
            'reference_baseline_attempts': (self.service.context or {}).get('reference_baseline_attempts', 0),
            'city_hint': self.city_hint, 'source_cities': self.service.sources,
            'city_resolution': self.service.context, 'clarification': self.clarification,
            'design_spec': self.design_spec,
            'numerical_agent_initialized': False, 'prompt_version': PROMPT_VERSION,
            'baseline_accounting': 'reference simulation separate from design budget; not a fit label'}

    def call(self, name, arguments, call_id):
        canonical = json.dumps([name, arguments], sort_keys=True)
        if call_id in self.cache:
            old, result = self.cache[call_id]
            if old != canonical:raise ToolRejected('Reused call ID with different arguments')
            return copy.deepcopy(result)
        if name not in self.allowed():raise ToolRejected('Tool unavailable before city readiness')
        validate(arguments, schemas([name])[0]['function']['parameters'])
        if arguments['state_version'] != self.version:raise ToolRejected('Stale state_version')
        if name == 'clarify_city':
            self.clarification = arguments['question']; self.phase = 'blocked'
            result = {'status': 'needs_city_clarification', 'message': self.clarification}
        else:
            if name == 'resolve_city':
                if self.city_hint and self.service.canonical_city(arguments['city']) != self.city_hint:
                    raise ToolRejected('Requested city differs from the fixed --city argument')
                try:
                    result = self.service.inspect(arguments['city'])
                    if result['status'] in {'ready', 'needs_preparation'}:
                        self.design_spec = design_spec(result, **self.run_options)
                except (ValueError, OSError) as exc:
                    result = {'status': 'invalid_city_assets_or_configuration',
                        'city': arguments['city'], 'message': redact(exc), 'missing_assets': []}
                    self.service.context = result
            elif name == 'prepare_city_assets':result = self.service.prepare()
            else:result = self.service.run_baseline()
            self.phase = {'ready': 'ready', 'needs_preparation': 'prepare', 'needs_baseline': 'baseline'}.get(result['status'], 'blocked')
        self.version += 1
        response = {'ok': True, 'result': result, 'state': self.state()}
        self.cache[call_id] = (canonical, copy.deepcopy(response))
        write_json(self.service.output/'city_support/city_bootstrap_state.json', self.state())
        return response


def resolve_city_request(service, client, request, city_hint=None, run_options=None):
    session = CityBootstrapSession(service, request, city_hint, run_options)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': json.dumps({'request': request, 'state': session.state()}, ensure_ascii=False)}]
    trace = service.output/'city_support/city_bootstrap_trace.jsonl'; trace.parent.mkdir(parents=True, exist_ok=True)
    rejected = 0

    def log(payload):
        with trace.open('a', encoding='utf-8') as stream:
            stream.write(redact(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **payload},
                                          ensure_ascii=False, allow_nan=False))+'\n')
    try:
        while session.phase not in {'ready', 'blocked'}:
            before = session.state()
            message, metadata = client.complete(messages, schemas(session.allowed()))
            calls = message.get('tool_calls') or []
            log({'event': 'api_response', 'state_before': before, 'metadata': metadata,
                 'tool_calls': calls, 'content': message.get('content'),
                 'reasoning_present_not_logged': bool(message.get('reasoning_content'))})
            messages.append(message)
            if len(calls) != 1:
                raise RuntimeError('City bootstrap requires exactly one native tool call per response')
            call = calls[0]
            try:
                arguments = json.loads(call['function']['arguments'])
                result = session.call(call['function']['name'], arguments, call['id'])
            except (ToolRejected, json.JSONDecodeError, KeyError) as exc:
                rejected += 1
                result = {'ok': False, 'error': redact(exc), 'state': session.state()}
            log({'event': 'tool_result', 'tool_call_id': call['id'], 'name': call['function']['name'], 'payload': result})
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)})
            print(f'LLM city tool {call["function"]["name"]} ok={result["ok"]} phase={session.phase}', flush=True)
            if rejected >= 5:raise RuntimeError('Too many invalid city bootstrap calls')
    except Exception as exc:
        failure = {'status': 'city_setup_failed', 'error': redact(f'{type(exc).__name__}: {exc}'), 'state': session.state()}
        write_json(service.output/'city_support/city_bootstrap_result.json', failure)
        raise
    outcome = {'status': 'ready' if session.phase == 'ready' else 'needs_input',
               'city_resolution': service.context, 'clarification': session.clarification,
               'design_spec': session.design_spec,
               'version': session.version, 'rejected_calls': rejected, 'trace': str(trace)}
    write_json(service.output/'city_support/city_bootstrap_result.json', outcome)
    return outcome, messages
