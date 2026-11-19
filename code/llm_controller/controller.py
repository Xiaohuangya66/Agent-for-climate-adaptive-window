"""Sequential, versioned function tools. Scientific constraints are code-owned."""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from llm_controller.api import redact


from llm_controller.prompts import SYSTEM_PROMPT, PROMPT_VERSION, CONTROLLER_VERSION

EVIDENCE_FIELDS = ['clusters', 'residuals', 'best_fit_feedback', 'next_round', 'budget_used',
                   'can_refine', 'switchable_clusters', 'stagnant_rounds', 'current_cluster',
                   'surrogate_version', 'target_version', 'candidates_sampled', 'candidate_archive_size',
                   'city_role', 'city_resolution']


def decision_residuals(agent):
    result = agent.bias_corrector.get_statistics()
    for key in ['energy_mae', 'energy_bias', 'energy_std']:
        value = result.get(key)
        result[key+'_fraction_of_baseline'] = value/agent.formal.energy if value is not None else None
    result.update(energy_baseline_J=agent.formal.energy, raw_energy_unit='J',
                  udi_error_unit='absolute fraction of daylight hours',
                  normalized_energy_unit='fraction of same-city annual baseline; 0.001 = 0.1%')
    return result


class ToolRejected(ValueError):
    pass


TOOLS = {
    'resolve_city': ('Resolve requested city and source/target role from the registry; inspect weather/assets/baseline and return readiness or missing items.',
                     {'city': {'type': 'string'}}),
    'prepare_city_assets': ('Prepare matched design/clear IDFs with the target EPW location and DDY sizing.', {}),
    'run_clear_glass_baseline': ('Run ONE separate real clear-glass reference; never borrow another city baseline.', {}),
    'clarify_city': ('Return a clarification when the requested city is absent or ambiguous; execute no simulation.',
                     {'question': {'type': 'string', 'minLength': 1, 'maxLength': 500}}),
    'plan_support_probes': ('Freeze a deterministic support plan and immutable fit/validation roles.', {}),
    'run_support_probes': ('Execute the frozen support plan; all attempts consume design budget.',
                           {'plan_id': {'type': 'string'}}),
    'fit_transfer_model': ('Attempt tree continuation ONCE; code gate controls publication.', {}),
    'set_round_strategy': ('Choose bounded target action, UDI risk profile, and information count.',
        {'target_action': {'type': 'string', 'enum': ['refine', 'switch', 'keep']},
         'cluster_id': {'type': 'integer', 'minimum': 0},
         'margin_profile': {'type': 'string', 'enum': ['boundary', 'balanced', 'cautious']},
         'n_information': {'type': 'integer', 'minimum': 0},
         'evidence': {'type': 'array', 'items': {'type': 'string', 'enum': EVIDENCE_FIELDS},
                      'minItems': 1, 'maxItems': 6,
                      'description': 'EXACT top-level state field names only; no prose or nested paths.'}}),
    'generate_round': ('Generate this complete round; TTT, TMM and proxy predictions are internal.',
                       {'strategy_id': {'type': 'string'}, 'n': {'type': 'integer', 'minimum': 1,
                        'description': 'EXACT strategy.candidates, not strategy.queries.'}}),
    'evaluate_round': ('Sequential reselect -> one EnergyPlus -> residual update for each query.',
                       {'strategy_id': {'type': 'string'}, 'candidate_set_id': {'type': 'string'},
                        'k': {'type': 'integer', 'minimum': 1}}),
    'summarize': ('Return formal.summarize verified feasible best; final selection is immutable.', {}),
}


def schemas(names):
    result = []
    for name in names:
        description, specific = TOOLS[name]
        props = {'state_version': {'type': 'integer', 'minimum': 0},
                 'rationale': {'type': 'string', 'minLength': 1, 'maxLength': 1000}, **specific}
        result.append({'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': props, 'required': list(props),
                           'additionalProperties': False}}})
    return result


def validate(arguments, schema):
    """Strict subset validator for the schemas actually exposed; no silent clamps."""
    if not isinstance(arguments, dict):
        raise ToolRejected('Tool arguments must be an object')
    props = schema['properties']
    if set(arguments) != set(schema['required']):
        raise ToolRejected('Missing or additional fields: expected ' + ', '.join(props))
    for key, value in arguments.items():
        spec = props[key]
        if spec['type'] == 'integer' and (type(value) is not int or value < spec.get('minimum', -10**9)):
            raise ToolRejected(f'{key} must be a bounded integer')
        if spec['type'] == 'string' and (not isinstance(value, str) or
                not spec.get('minLength', 0) <= len(value) <= spec.get('maxLength', 10000)):
            raise ToolRejected(f'{key} must be a bounded string')
        if 'enum' in spec and value not in spec['enum']:
            raise ToolRejected(f'{key} outside allowed choices')
        if spec['type'] == 'array' and (not isinstance(value, list) or
                not spec.get('minItems', 0) <= len(value) <= spec.get('maxItems', 100) or
                not all(isinstance(v, str) and len(v) <= 150 and
                        ('enum' not in spec['items'] or v in spec['items']['enum']) for v in value)):
            raise ToolRejected(f'{key} must be a bounded string array')


class ToolSession:
    def __init__(self, agent):
        self.a = agent
        self.a.started = time.time()
        self.phase = 'resolve'
        self.version = 0
        self.cache = {}
        self.plan = None
        self.strategy = None
        self.rounds = []
        self.result = None
        self.pending_failure = None

    def allowed(self):
        return {'resolve': ['resolve_city'], 'plan': ['plan_support_probes'],
                'probes': ['run_support_probes'], 'fit': ['fit_transfer_model'],
                'strategy': ['set_round_strategy'], 'generate': ['generate_round'],
                'evaluate': ['evaluate_round'], 'summary': ['summarize'], 'done': []}[self.phase]

    def round_plan(self):
        a = self.a
        remaining = a.budget_total-a.budget_used
        rounds_left = math.ceil((a.candidate_budget-a.sampled)/a.batch_size)
        if not rounds_left:
            return {'queries': remaining, 'candidates': 0, 'rounds_left': 0,
                    'information_min': 0, 'information_max': 0}
        k = math.ceil(remaining/rounds_left)
        used = sum(r.get('selection', {}).get('mode') == 'information' for r in a.query_history)
        information_left = a.information_quota-used
        future_capacity = max(0, remaining-k-(min(rounds_left-1, remaining-k)))
        return {'queries': k, 'candidates': min(a.batch_size, a.candidate_budget-a.sampled),
                'rounds_left': rounds_left, 'information_remaining': information_left,
                'information_min': max(0, information_left-future_capacity),
                'information_max': min(max(0, k-1), information_left)}

    def state(self):
        a = self.a
        # All detailed decision feedback excludes the four publication-validation rows.
        learning = [r for r in a.query_history if r.get('adaptation_role', 'fit') == 'fit']
        measured = [r for r in learning if r.get('energyplus_verified')]
        feasible = [r for r in measured if r.get('direct_feasible')]
        best = min(feasible, key=lambda r: r['energy']) if feasible else None
        clusters = []
        for i, cluster in enumerate(a.clusters):
            rows = [r for r in measured if r.get('cluster_id_generated') == i]
            good = [r for r in rows if r.get('direct_feasible')]
            clusters.append({'id': i, 'visits': a.cluster_visits[i], 'source_city': cluster.get('source_city'),
                'predicted_energy_saving': 1-cluster['avg_energy']/a.formal.energy,
                'predicted_udi': cluster['avg_udi'], 'fit_queries': len(rows),
                'best_fit_saving': max((r['energy_saving'] for r in good), default=None)})
        switchable = [c['id'] for c in clusters if c['visits'] == 0 and
                      c['id'] != a.current_cluster_id and a.iteration >= 1]
        return {'state_version': self.version, 'phase': self.phase, 'allowed_tools': self.allowed(),
            'city': a.city, 'seed': a.seed, 'budget_total': a.budget_total, 'budget_used': a.budget_used,
            'city_role': 'target' if a.city_support else 'source',
            'city_resolution': getattr(a, 'city_resolution', None),
            'candidate_budget': a.candidate_budget, 'candidates_sampled': a.sampled,
            'candidate_archive_size': len(a.candidate_archive),
            'real_udi_threshold': a.formal.threshold, 'clear_udi': a.formal.clear_udi,
            'udi_ratio': a.formal.udi_ratio, 'current_cluster': a.current_cluster_id,
            'target_version': len(a.target_history)-1, 'surrogate_version': getattr(a.surrogate, 'VERSION', 'native'),
            'n_fit_labels': len(measured), 'n_validation_labels': sum(r.get('energyplus_verified') and
                r.get('adaptation_role') == 'validation' for r in a.query_history),
            'validation_scope': 'publication gate only; excluded from all online learning and LLM feedback',
            'best_fit_feedback': ({'saving_fraction': best['energy_saving'], 'udi': best['udi'],
                'udi_margin': best['udi']-a.formal.threshold, 'query_index': best['query_index']} if best else None),
            'state_contract': 'normalized_energy_feedback_v2',
            'residuals': decision_residuals(a), 'stagnant_rounds': a.stagnant_rounds,
            'clusters': clusters, 'can_refine': bool(a._refinement_pool()), 'switchable_clusters': switchable,
            'next_round': self.round_plan() if self.phase == 'strategy' else None,
            'strategy': self.strategy, 'completed_rounds': self.rounds,
            'probe_plan_id': 'support-plan-1' if self.plan is not None else None,
            'uncertainty': 'heuristic, no target coverage guarantee'}

    def call(self, name, arguments, call_id):
        canonical = json.dumps([name, arguments], sort_keys=True)
        if call_id in self.cache:
            old, response = self.cache[call_id]
            if old != canonical:
                raise ToolRejected('Reused tool_call_id with different arguments')
            return response
        if name not in self.allowed():
            raise ToolRejected('Tool unavailable in phase ' + self.phase)
        schema = schemas([name])[0]['function']['parameters']
        validate(arguments, schema)
        if arguments['state_version'] != self.version:
            raise ToolRejected('Stale state_version; use the returned latest version')
        # Validation of all strategy/ticket/capacity inputs occurs before mutation.
        payload = getattr(self, 'do_'+name)(arguments)
        self.version += 1
        response = {'ok': True, 'result': payload, 'state': self.state()}
        self.cache[call_id] = (canonical, copy.deepcopy(response))
        return response

    def do_resolve_city(self, x):
        from src.tools.city_support import climate_features
        a = self.a
        if x['city'] != a.city:
            raise ToolRejected('Run city is immutable')
        for key in ['epw', 'idf', 'baseline_idf', 'baseline_table']:
            if not Path(a.runtime_assets[key]).is_file():
                raise ToolRejected('Missing asset: ' + key)
        if not math.isclose(a.formal.udi_ratio, .9):
            raise ToolRejected('This protocol requires clear-glass UDI90')
        a._phase1_initialize()
        self.phase = 'plan'
        return {'assets_ready': True, 'climate': climate_features(a.runtime_assets['epw']),
                'support_mode': 'transfer' if a.city_support else 'source', 'missing_assets': []}

    def do_plan_support_probes(self, x):
        from src.runtime_profile import registry
        a = self.a
        self.plan = a.city_support.propose_probes(a, a.probe_count) if a.city_support else {'samples': []}
        self.phase = 'probes'
        roles = [r.get('adaptation_role', 'fit') for r in self.plan['samples']]
        return {'plan_id': 'support-plan-1', 'n_probes': len(roles),
                'roles': {r: roles.count(r) for r in set(roles)},
                'sources': list(a.city_support.bank.cities) if a.city_support else list(registry()['source_cities'])}

    def do_run_support_probes(self, x):
        if x['plan_id'] != 'support-plan-1':
            raise ToolRejected('Unknown frozen plan')
        self.a._run_support_probes(self.plan)
        self.phase = 'fit'
        return {'probe_attempts': self.a.budget_used, 'publication_attempted': False}

    def do_fit_transfer_model(self, x):
        a = self.a
        if a.city_support:
            a.fit_transfer_explicit()
            decision = copy.deepcopy(a.city_support.tree_decision)
        else:
            decision = {'decision': 'native_source_model_no_transfer_needed'}
        a._start_online_phase()
        self.phase = 'strategy'
        return {'gate': decision, 'publication_owner': 'numeric_tool_only'}

    def do_set_round_strategy(self, x):
        a = self.a
        plan = self.round_plan()
        if not plan['queries'] or not plan['candidates']:
            raise ToolRejected('No capacity for a new round')
        if not plan['information_min'] <= x['n_information'] <= plan['information_max']:
            raise ToolRejected('Information count cannot satisfy fixed quota/capacity')
        if any(field not in self.state() for field in x['evidence']):
            raise ToolRejected('Evidence must cite fields present in current state')
        action, index = x['target_action'], x['cluster_id']
        if action == 'switch' and index not in self.state()['switchable_clusters']:
            raise ToolRejected('Cluster switch requires a legal unvisited cluster after a completed round')
        if action != 'switch' and index != a.current_cluster_id:
            raise ToolRejected('keep/refine must name current cluster')
        if action == 'refine' and not a._should_refine_target():
            raise ToolRejected('No fit-only feasible target feedback')
        a.iteration += 1
        changed = False
        if action == 'switch':
            a._set_cluster(index, 'cluster_switch_llm')
            a.stagnant_rounds = 0
            changed = True
        elif action == 'refine':
            changed = a._refine_target()
        a.bias_corrector.margin_profile = x['margin_profile']
        a.cluster_visits[a.current_cluster_id] += 1
        self.strategy = {'strategy_id': f'round-{a.iteration}', 'round': a.iteration,
            'target_action': action, 'cluster_id': a.current_cluster_id, 'target_changed': bool(changed),
            'margin_profile': x['margin_profile'], 'margin_at_start': a.bias_corrector.get_decision_margin(),
            'n_information': x['n_information'], 'queries': plan['queries'], 'candidates': plan['candidates'],
            'rationale': x['rationale'], 'evidence': x['evidence']}
        a.target_decisions.append({'iteration': a.iteration, 'budget_used': a.budget_used, **self.strategy})
        a._json_line('llm_strategies.jsonl', {'state_version_before': self.version, **self.strategy})
        self.phase = 'generate'
        return self.strategy

    def do_generate_round(self, x):
        a = self.a
        if x['strategy_id'] != self.strategy['strategy_id'] or x['n'] != self.strategy['candidates']:
            raise ToolRejected('Wrong strategy ticket or generation count')
        a._generate_candidates()
        self.strategy['candidate_set_id'] = f'candidates-{a.iteration}'
        pool = [r for r in a.candidate_archive if r['structure'] not in a.queried]
        for row in pool:
            a._refresh_prediction(row)
        self.phase = 'evaluate'
        return {'candidate_set_id': self.strategy['candidate_set_id'], 'strategy_id': self.strategy['strategy_id'],
                'n_sampled_total': a.sampled, 'n_available': len(pool),
                'n_predicted_center_feasible': sum(r['predicted_udi'] >= a.formal.threshold for r in pool),
                'best_predicted_saving': max((1-r['predicted_energy']/a.formal.energy for r in pool), default=None)}

    def do_evaluate_round(self, x):
        a = self.a
        if (x['strategy_id'] != self.strategy['strategy_id'] or
            x['candidate_set_id'] != self.strategy['candidate_set_id'] or x['k'] != self.strategy['queries']):
            raise ToolRejected('Wrong candidate/strategy ticket or query count')
        if x['k'] > a.budget_total-a.budget_used:
            raise ToolRejected('Query budget exceeded')
        start = a.budget_used
        n = self.strategy['n_information']
        # Even spacing within this round, first slot always a performance anchor.
        slots = {math.ceil((j+1)*(x['k']-1)/n) for j in range(n)} if n else set()
        for offset in range(x['k']):
            a.information_selector.forced_due = offset in slots
            a._query_and_evaluate(1)  # Reselect and update residuals after EVERY true query.
        a.information_selector.forced_due = None
        # _query_and_evaluate(1) updates stagnation per query; restore round-level meaning.
        old_best = min((r['energy'] for r in a.query_history if r.get('adaptation_role', 'fit') == 'fit'
            and r.get('direct_feasible') and r['query_index'] <= start), default=None)
        pool = a._refinement_pool()
        new_best = pool[0]['energy'] if pool else None
        improved = new_best is not None and (old_best is None or new_best < old_best*(1-1e-6))
        prior_stagnation = self.rounds[-1]['stagnant_rounds_after'] if self.rounds else 0
        a.stagnant_rounds = 0 if improved else prior_stagnation+1
        rows = [r for r in a.query_history if r['query_index'] > start]
        fit = [r for r in rows if r.get('adaptation_role', 'fit') == 'fit']
        round_result = {'round': a.iteration, 'query_start': start+1, 'query_end': a.budget_used,
            'attempts': len(rows), 'success': sum(r['energyplus_verified'] for r in rows),
            'n_information': sum(r.get('selection', {}).get('mode') == 'information' for r in rows),
            'information_requested': n, 'information_fallbacks': sum(bool(r.get('selection', {}).get('fallback_reason')) for r in rows),
            'n_fit_feasible': sum(r['direct_feasible'] for r in fit),
            'best_fit_saving': max((r['energy_saving'] for r in fit if r['direct_feasible']), default=None),
            'stagnant_rounds_after': a.stagnant_rounds,
            'strategy': copy.deepcopy(self.strategy)}
        self.rounds.append(round_result)
        a._json_line('llm_rounds.jsonl', round_result)
        if len(rows) != x['k']:
            self.pending_failure = 'candidate_pool_exhausted_before_requested_budget'
            self.phase = 'summary'
        else:
            self.phase = 'summary' if a.budget_used >= a.budget_total or a.sampled >= a.candidate_budget else 'strategy'
        return round_result

    def finalize(self, interrupted=False):
        a = self.a
        for row in a.candidate_archive:
            a._refresh_prediction(row)
            a._compute_information_value(row)
        result = a._finalize()
        result.update(target_history=a.target_history, target_decisions=a.target_decisions,
                      adaptation_history=a.adaptation_history, selection_history=a.selection_history)
        result['controller'] = {'type': 'single_llm_tool_controller', 'version': CONTROLLER_VERSION,
            'prompt_version': PROMPT_VERSION, 'city_resolution': getattr(a, 'city_resolution', None),
            'state_contract': 'normalized_energy_feedback_v2',
            'phase': self.phase, 'rounds': self.rounds, 'rule_policy_fallbacks': 0,
            'interrupted': interrupted, 'failure': self.pending_failure,
            'validation_scope': 'publication gate; excluded from target, acquisition and LLM decision feedback',
            'full_process_resume_supported': False}
        result['method_specific_data'].update(
            controller='single_llm_tool_controller',
            target_decision_owner='API LLM; numeric target updates',
            information_timing_owner='API LLM; fixed total quota and per-round performance anchor',
            online_learning_validation_filter='fit-only target, acquisition references and bias')
        if interrupted or self.pending_failure:
            result['status'] = 'incomplete'
        for name, value in [('summary.json', result), ('candidates.json', a.candidate_archive),
                            ('rejected.json', a.generation_failures)]:
            (a.run_directory/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        self.result = result
        return result

    def do_summarize(self, x):
        result = self.finalize()
        self.phase = 'done'
        return {k: result.get(k) for k in ['status', 'n_energyplus_queries', 'n_successful_queries',
            'n_failed_queries', 'best_is_feasible', 'best_energy_saving', 'best_udi', 'best_structure']}


def run_controller(agent, client, *, initial_messages=None, initial_version=0, prior_rejections=0):
    session = ToolSession(agent)
    if initial_messages is not None:
        session.phase = 'plan'
        session.version = initial_version
        messages = initial_messages + [{'role': 'user', 'content': json.dumps({
            'task': '城市及基线已核验，继续使用当前状态完成同一个 Agent 的优化流程。',
            'state': session.state()}, ensure_ascii=False)}]
    else:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': json.dumps({'task': 'Execute one complete constrained city experiment.',
                                                      'state': session.state()}, ensure_ascii=False)}]
    rejects = prior_rejections
    trace = agent.run_directory/'llm_trace.jsonl'

    def log(payload):
        entry = {'time_utc': datetime.now(timezone.utc).isoformat(), **payload}
        with trace.open('a', encoding='utf-8') as stream:
            stream.write(redact(json.dumps(entry, ensure_ascii=False, allow_nan=False))+'\n')

    try:
        while session.phase != 'done':
            before = session.state()
            message, metadata = client.complete(messages, schemas(session.allowed()))
            calls = message.get('tool_calls') or []
            log({'event': 'api_response', 'state_before': before, 'metadata': metadata,
                 'tool_calls': calls, 'content': message.get('content'),
                 'reasoning_present_not_logged': bool(message.get('reasoning_content'))})
            # Preserve provider reasoning_content in memory for API compatibility only.
            messages.append(message)
            if len(calls) != 1:
                rejects += 1
                for call in calls:
                    messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''),
                        'content': json.dumps({'ok': False, 'error': 'Exactly one tool per response required; no call executed',
                                              'state': session.state()})})
                if not calls:
                    messages.append({'role': 'user', 'content': 'Call exactly one allowed function with the latest state_version.'})
                if rejects >= 5:
                    raise RuntimeError('Too many invalid LLM turns')
                continue
            call = calls[0]
            try:
                arguments = json.loads(call['function']['arguments'])
                result = session.call(call['function']['name'], arguments, call['id'])
            except (ToolRejected, json.JSONDecodeError, KeyError) as exc:
                rejects += 1
                result = {'ok': False, 'error': redact(exc), 'state': session.state()}
            log({'event': 'tool_result', 'tool_call_id': call['id'], 'name': call['function']['name'], 'payload': result})
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)})
            print(f"LLM tool {call['function']['name']} ok={result['ok']} state={session.version} "
                  f"queries={agent.budget_used}/{agent.budget_total}", flush=True)
            if rejects >= 5:
                raise RuntimeError('Too many invalid LLM tool calls')
    except Exception as exc:
        session.pending_failure = redact(f'{type(exc).__name__}: {exc}')
        log({'event': 'controller_failed', 'error': session.pending_failure})
        session.finalize(interrupted=True)
        raise
    finally:
        audit = {'api_requests': client.calls, 'token_usage': client.usage, 'rejected_calls': rejects,
                 'completed': session.phase == 'done', 'credential_source': 'process_environment_only',
                 'provider_configuration': 'process_environment_only', 'rule_policy_fallbacks': 0}
        (agent.run_directory/'llm_usage.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return session.result
