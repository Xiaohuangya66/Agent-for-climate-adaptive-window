"""City-prior controller with on-policy TTT and direct residual feedback."""
import json
import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from src.agent.base_agent import BaseAgent
from src.agent.baseline_common import BaselineRecords, SearchSpace, ROOT, FEATURE_ORDER
from src.tools.city_prior_learner import CityPriorLearner
from src.tools.target_refiner import TargetRefiner
from src.components.ttt_policy import TTTPolicy
from src.tools.bias_corrector import BiasCorrector
from src.tools.information_selector import InformationValueSelector


class AgentFull(BaselineRecords, BaseAgent):
    def __init__(self, city, seed, budget=40, candidate_budget=384, batch_size=128,
                 output_root=None, max_layers=None, ttt_batch_size=16, learning_rate=1e-4,
                 temperature=1., nhead=4, checkpoint_path=None, device='cpu',
                 selection_mode='hybrid', information_fraction=.3, probe_count=None, **kwargs):
        super().__init__(city, seed, budget, **kwargs)
        self.candidate_budget, self.batch_size = int(candidate_budget), int(batch_size)
        self.ttt_batch_size = int(ttt_batch_size)
        if min(self.candidate_budget, self.batch_size, self.ttt_batch_size) < 1:
            raise ValueError('Candidate and batch budgets must be positive')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self.temperature = float(temperature)
        self.space = SearchSpace(max_layers=max_layers or 20)
        self._setup_records('agent_full', output_root)
        self.city_prior_learner = self.city_support.prior_provider if self.city_support else CityPriorLearner()
        from src.support_settings import settings, asset_path
        requested = settings().get('probe_count', 8) if probe_count is None else probe_count
        if int(requested) != requested or requested < 0:
            raise ValueError('probe_count must be a nonnegative integer')
        self.probe_count = min(int(requested), max(0, self.budget_total-1)) if self.city_support else 0
        if self.city_support:self.city_support.bind(self.run_directory)
        self.target_refiner = TargetRefiner()
        self.checkpoint_path = Path(checkpoint_path or asset_path('checkpoint', ROOT.parent/'topic3/checkpoints/last.ckpt'))
        self.ttt_policy = (TTTPolicy(self.space, checkpoint=self.checkpoint_path, device=device,
                                     nhead=nhead, learning_rate=learning_rate) if budget > 0 else None)
        self.bias_corrector = BiasCorrector(window_size=20)
        if selection_mode not in {'hybrid', 'performance'}:
            raise ValueError('selection_mode must be hybrid or performance')
        self.selection_mode = selection_mode
        self.information_fraction = float(information_fraction)
        quota = InformationValueSelector.information_quota(self.budget_total, self.information_fraction)
        self.information_quota = quota if selection_mode == 'hybrid' else 0
        self.online_query_offset=None
        self.online_query_budget=None
        scale = getattr(self.surrogate, 'feature_scale', getattr(getattr(self.surrogate, 'ensemble', None), 'x_scale', None))
        self.information_selector = InformationValueSelector(self.formal.energy, self.formal.threshold, scale)
        self.selection_history = []
        self.target_current = self.target_full = self.target_confidence = None
        self.target_history, self.target_decisions, self.adaptation_history = [], [], []
        self.sampled = self.policy_version = self.n_rejected = 0
        self.seen_candidates, self.optics, self.by_structure = set(), {}, {}
        self.generation_failures = []
        self.clusters, self.cluster_visits = [], {}
        self.current_cluster_id = None
        self.stagnant_rounds = 0
        self.best_direct_energy = None

    def _run_support_probes(self, plan=None):
        if not self.city_support or self.probe_count <= 0:return
        from src.tools.city_support import write_json
        plan = plan or self.city_support.propose_probes(self, self.probe_count)
        write_json(self.run_directory/'city_support'/'probe_plan.json',plan)
        for sample in plan['samples']:
            structure = sample['structure']
            if structure in self.queried:continue
            props = self.tmm.compute(structure)
            self.n_tmm += 1
            features = {key:float(props[key]) for key in FEATURE_ORDER}
            self.optics[structure] = props
            raw = self.surrogate.predict(features)
            corrected = self.bias_corrector.correct(raw)
            prediction = self.formal.prediction(corrected)
            row = self._direct_query(structure, phase='city_support_probe', probe_id=sample['probe_id'],
                source_reference=sample.get('source_reference'), selection_reason=sample['selection_reason'],
                raw_predicted_udi=raw['UDI'], raw_predicted_energy=raw['TotalEnergy'],
                predicted_udi=prediction['predicted_udi'], predicted_energy=prediction['predicted_energy'],
                udi_half_width_90pct=prediction['udi_half_width_90pct'],
                bias_version=corrected['bias_version'], iteration=self.iteration,
                adaptation_role=sample.get('adaptation_role','fit'))
            if row is not None:
                accepted = self.bias_corrector.update([raw], [row]) if row.get('adaptation_role')!='validation' else 0
                self._json_line('bias_history.jsonl', {'query_index':row['query_index'],
                    'phase':'city_support_probe','observation_accepted':bool(accepted),
                    **self.bias_corrector.get_statistics()})
        self._update_pareto()
        pool = self._refinement_pool()
        self.best_direct_energy = pool[0]['energy'] if pool else None
        self._update_city_support()
        write_json(self.run_directory/'city_support'/'prior_clusters.json',self.clusters)

    def _update_city_support(self):
        if not self.city_support:return
        old_version=getattr(self.surrogate,'VERSION','native_city_model')
        decision=self.city_support.fit_or_keep(self)
        if getattr(self.city_support.predictor,'VERSION','native_city_model')==old_version:return
        self.surrogate=self.city_support.predictor
        # Replace stale residuals instead of correcting the new model with the
        # old model's errors. Historical query forecasts stay immutable.
        self.bias_corrector=BiasCorrector(window_size=20)
        for observed in self.query_history:
            if observed.get('energyplus_verified') and observed.get('adaptation_role','fit')=='fit':
                self.bias_corrector.update([self.surrogate.predict(observed['optical_properties'])],[observed],prequential=False)
        for candidate in self.candidate_archive:self._refresh_prediction(candidate)
        for cluster in self.clusters:
            prediction=self.surrogate.predict(cluster['center'])
            cluster.update(target_prediction=prediction,avg_energy=prediction['TotalEnergy'],avg_udi=prediction['UDI'],
                           support_version=self.surrogate.VERSION)
        self._json_line('model_updates.jsonl',{'iteration':self.iteration,'budget_used':self.budget_used,
            'previous_version':old_version,'new_version':self.surrogate.VERSION,
            'bias_rebuilt_from_fit_queries_only':True,'n_candidates_refreshed':len(self.candidate_archive),
            'bias_statistics':self.bias_corrector.get_statistics(),'decision':decision})

    def _json_line(self, name, data):
        with (self.run_directory/name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n')

    def _record_target(self, strategy, **details):
        self.target_history.append({'iteration': self.iteration, 'strategy': strategy,
            'cluster_id': self.current_cluster_id, 'target': self.target_current.tolist(),
            'target_full_spectrum': self.target_full.tolist(), 'confidence': self.target_confidence,
            'n_direct_feedback': len(self.query_history),
            'confidence_status':'unavailable_transfer_prior' if self.city_support else 'source_prior_consistency',
            'support_version':getattr(self.surrogate,'VERSION','native_city_model'), **details})

    def _set_cluster(self, index, strategy):
        cluster = self.clusters[index]
        target = np.asarray(cluster['center'], dtype=float)
        full = np.asarray(cluster['target_full_spectrum'], dtype=float)
        if target.shape != (8,) or full.shape != (221, 2) or not np.isfinite(full).all() or not np.isfinite(target).all():
            raise ValueError('Cluster must contain matched finite 8D and 221x2 targets')
        if np.any((full < 0) | (full > 1)):
            raise ValueError('Cluster transmittance is outside [0,1]')
        self.current_cluster_id = index
        self.target_current, self.target_full = target.copy(), full.copy()
        raw=self.surrogate.predict(cluster['center'])
        corrected=self.bias_corrector.correct(raw)
        ranking=self.formal.prediction(corrected)
        self._record_target(strategy, cluster_avg_energy=cluster['avg_energy'],
                            cluster_size=cluster['size'], n_clusters=len(self.clusters),
                            cluster_prediction_raw=raw,cluster_prediction_corrected=corrected,
                            cluster_query_key=list(self.formal.query_key(ranking)))

    def _phase1_initialize(self):
        prior = self.city_prior_learner.learn_prior_clusters(self.city, n_clusters=3)
        self.clusters = prior['clusters']
        if not self.clusters:
            raise ValueError('City prior contains no valid clusters')
        self.target_confidence = float(prior['confidence'])
        self.cluster_visits = {i: 0 for i in range(len(self.clusters))}
        self._set_cluster(0, 'initial_best_energy_cluster')

    def _refinement_pool(self):
        # A small nondominated front is not evidence that feedback is insufficient.
        return sorted([r for r in self.query_history if r.get('energyplus_verified') and
                       r.get('direct_feasible')], key=lambda r: (r['energy'], -r['udi']))

    def _should_refine_target(self):
        return bool(self._refinement_pool())

    def _refine_target(self):
        refined = self.target_refiner.refine_target_energy_directed(
            self.target_current, self._refinement_pool(), self.tmm,
            strategy='best_energy', momentum=.6, current_full=self.target_full, optics=self.optics)
        full = np.asarray(refined['target_full_new'], dtype=float)
        if np.allclose(full, self.target_full, rtol=0, atol=1e-8):
            return False
        self.target_current = np.asarray(refined['target_new'], dtype=float)
        self.target_full = full
        # Prior consistency is not increased merely because an update occurred.
        self._record_target('energy_directed_best_energy', deviation=refined['deviation'],
                            reason=refined['reason'], source_structures=refined['source_structures'],
                            weights=refined['weights'], momentum=refined['momentum'])
        return True

    def _choose_target(self):
        unused = [i for i in self.cluster_visits if self.cluster_visits[i] == 0
                  and i != self.current_cluster_id]
        if self.stagnant_rounds >= 1 and unused:
            if self.city_support:
                unused.sort(key=lambda i:self.formal.query_key(self.formal.prediction(
                    self.bias_corrector.correct(self.surrogate.predict(self.clusters[i]['center'])))))
            self._set_cluster(unused[0], 'cluster_switch_after_direct_stagnation')
            self.stagnant_rounds = 0
            decision = 'switch_unvisited_cluster'
        elif self._should_refine_target():
            decision = 'refine_to_verified_best' if self._refine_target() else 'keep_matching_target'
        else:
            decision = 'keep_prior_no_feasible_direct_feedback'
        self.target_decisions.append({'iteration': self.iteration, 'decision': decision,
            'cluster_id': self.current_cluster_id, 'n_feasible_feedback': len(self._refinement_pool()),
            'budget_used': self.budget_used, 'policy_version': self.policy_version})

    def _refresh_prediction(self, row):
        if 'raw_prediction' not in row:
            return
        version=getattr(self.surrogate,'VERSION','native_city_model')
        if row.get('surrogate_version')!=version:
            row['raw_prediction']=self.surrogate.predict(row['optical_properties'],strict=True)
            row['surrogate_version']=version
        corrected = self.bias_corrector.correct(row['raw_prediction'])
        row.update(self.formal.prediction(corrected))
        row.update(bias_version=corrected['bias_version'],
                   energy_bias_applied=corrected['energy_bias_applied'],
                   udi_bias_applied=corrected['udi_bias_applied'])

    def _score_structure(self, structure):
        if structure in self.by_structure:
            row = self.by_structure[structure]
            self._refresh_prediction(row)
            return row['reward']
        if structure in self.seen_candidates:
            return -10.
        self.seen_candidates.add(structure)
        row = {'structure': structure, 'iteration_generated': self.iteration,
               'cluster_id_generated': self.current_cluster_id, 'policy_version_generated': self.policy_version,
               'target_version_generated': len(self.target_history)-1}
        try:
            if not self.space.valid(structure):
                raise ValueError('Structure outside configured search space')
            self.n_tmm += 1
            props = self.tmm.compute(structure)
            features = {key: float(props[key]) for key in FEATURE_ORDER}
            raw = self.surrogate.predict(features, strict=True)
            row.update(optical_properties=features, raw_prediction=raw)
            self._refresh_prediction(row)
            self.optics[structure] = props
            self.by_structure[structure] = row
            self.candidate_archive.append(row)
            return row['reward']
        except Exception as exc:
            self.n_rejected += 1
            # Invalid numeric payloads must not poison strict JSON audit output.
            failure = {key: row[key] for key in ['structure', 'iteration_generated',
                'cluster_id_generated', 'policy_version_generated', 'target_version_generated']}
            failure['error'] = f'{type(exc).__name__}: {exc}'
            self.generation_failures.append(failure)
            return -10.

    def _generate_candidates(self):
        remaining = min(self.batch_size, self.candidate_budget-self.sampled)
        # Explicit transpose preserves cold/hot order; reshape would interleave them.
        spectrum = torch.from_numpy(self.target_full.T.copy()).float()
        while remaining:
            count = min(self.ttt_batch_size, remaining)
            rollout = self.ttt_policy.sample(spectrum, count, self.temperature)
            structures = rollout['structures']
            if len(structures) != count or rollout['token_ids'].shape[0] != count:
                raise ValueError('Policy must return exactly one trajectory per requested sample')
            self.sampled += count
            self.n_generation_attempts += count
            rewards = [self._score_structure(structure) for structure in structures]
            # Invalid and duplicate trajectories retain their positions and rewards.
            update = self.ttt_policy.adapt(spectrum, rollout, rewards)
            self.policy_version += int(update['parameters_changed'])
            update.update(iteration=self.iteration, sampled=self.sampled, count=count,
                          policy_version=self.policy_version, cluster_id=self.current_cluster_id,
                          target_version=len(self.target_history)-1,
                          support_version=getattr(self.surrogate,'VERSION','native_city_model'),
                          udi_prequential_observations=self.bias_corrector.n_udi_observations,
                          decision_udi_margin=self.bias_corrector.get_decision_margin(),
                          bias_version=self.bias_corrector.n_observations)
            self.adaptation_history.append(update)
            self._json_line('adaptation.jsonl', update)
            remaining -= count

    def _selection_references(self):
        return [row for row in self.query_history if row.get('energyplus_verified')]

    def _selection_pareto(self):
        # query_history is current after EACH query; pareto_front is updated only
        # at the end of a round and would miss within-round information.
        feasible = [row for row in self._selection_references() if row.get('direct_feasible')]
        return [row for row in feasible if not any(other['energy'] <= row['energy'] and
            other['udi'] >= row['udi'] and (other['energy'] < row['energy'] or other['udi'] > row['udi'])
            for other in feasible)]

    def _compute_information_value(self, candidate, references=None, pareto=None):
        references = self._selection_references() if references is None else references
        pareto = self._selection_pareto() if pareto is None else pareto
        details = self.information_selector.score(candidate, references, pareto)
        candidate['information_value'] = details['score']
        candidate['information_value_details'] = details
        candidate['information_value_context'] = {
            'bias_version': candidate.get('bias_version'), 'budget_used': self.budget_used,
            'n_reference_rows': len(references), 'n_direct_front_rows': len(pareto)}
        return details['score']

    def _select_for_energyplus(self, candidates, count=None):
        count = max(0, min(count if count is not None else 4, self.budget_total-self.budget_used))
        if not count:
            return []
        if self.online_query_offset is None:self._start_online_phase()
        remaining = {r['structure']: r for r in candidates if r['structure'] not in self.queried}
        for row in remaining.values():
            self._refresh_prediction(row)
        references, pareto = self._selection_references(), self._selection_pareto()
        information_used = sum(row.get('selection', {}).get('mode') == 'information' for row in self.query_history)
        selected = []
        for offset in range(min(count, len(remaining))):
            query_index = self.budget_used+offset+1
            online_index=query_index-self.online_query_offset
            due = self.selection_mode == 'hybrid' and self.information_selector.information_due_with_quota(
                online_index,self.online_query_budget,self.information_quota,information_used)
            for row in remaining.values():
                self._compute_information_value(row, references, pareto)
            # Uncertainty alone cannot make an obviously dark design eligible.
            eligible = [row for row in remaining.values() if row['predicted_udi'] >= self.formal.threshold]
            mode = 'information' if due and eligible else 'performance'
            if mode == 'information':
                ranked = sorted(eligible, key=lambda row: (-row['information_value'],
                    self.formal.query_key(row), row['structure']))
                information_used += 1
            else:
                ranked = sorted(remaining.values(), key=self.formal.query_key)
            chosen = ranked[0]
            performance_leader = min(remaining.values(), key=self.formal.query_key)
            chosen['query_selection'] = {
                'version': self.information_selector.VERSION, 'mode': mode,
                'planned_query_index': query_index, 'scheduled_information': bool(due),
                'online_query_index':online_index,'online_query_budget':self.online_query_budget,
                'online_query_offset':self.online_query_offset,
                'schedule_basis':'online phase; quota counted against total design budget',
                'fallback_reason': 'no_predicted_center_feasible_candidate' if due and not eligible else None,
                'information_quota': self.information_quota, 'information_fraction': self.information_fraction,
                'n_unqueried_candidates': len(remaining), 'n_information_eligible': len(eligible),
                'bias_version': chosen.get('bias_version'),
                'performance_udi_margin':chosen['performance_udi_margin'],
                'prediction_decision_threshold':chosen['prediction_decision_threshold'],
                'value': copy.deepcopy(chosen['information_value_details']),
                'reference_context': copy.deepcopy(chosen['information_value_context']),
                'differs_from_performance_leader': chosen['structure'] != performance_leader['structure'],
                'performance_leader': {key: performance_leader[key] for key in
                    ['structure', 'predicted_energy', 'predicted_udi', 'information_value']},
                'top_alternatives': [{'structure': row['structure'], 'score': row['information_value']}
                                     for row in ranked[:3]]}
            selected.append(chosen)
            references = references+[chosen]  # Greedy diversity for multi-pick previews.
            del remaining[chosen['structure']]
        # Previewing a batch never consumes query budget or exploration quota.
        return selected

    def _query_and_evaluate(self, count=4):
        previous = self.best_direct_energy
        for _ in range(min(count, self.budget_total-self.budget_used)):
            selected = self._select_for_energyplus(self.candidate_archive, 1)
            if not selected:
                break
            candidate = selected[0]
            raw = candidate['raw_prediction']
            row = self._direct_query(candidate['structure'], phase='online_city_agent',
                adaptation_role=self.city_support.next_observation_role() if self.city_support else 'fit',
                selection=copy.deepcopy(candidate['query_selection']),
                iteration=self.iteration, cluster_id_generated=candidate['cluster_id_generated'],
                target_version_generated=candidate['target_version_generated'],
                policy_version_generated=candidate['policy_version_generated'],
                predicted_udi=candidate['predicted_udi'], predicted_energy=candidate['predicted_energy'],
                performance_udi_margin=candidate['performance_udi_margin'],
                prediction_decision_threshold=candidate['prediction_decision_threshold'],
                raw_predicted_udi=raw['UDI'], raw_predicted_energy=raw['TotalEnergy'],
                bias_version=candidate['bias_version'], energy_bias_applied=candidate['energy_bias_applied'],
                udi_bias_applied=candidate['udi_bias_applied'],
                udi_half_width_90pct=candidate['udi_half_width_90pct'],
                ood_distance=candidate['ood_distance'], optical_properties=candidate['optical_properties'])
            if row is None:
                continue
            decision = {'query_index': row['query_index'], 'structure': row['structure'],
                        'energyplus_verified': row['energyplus_verified'],
                        'direct_feasible': row['direct_feasible'], **copy.deepcopy(row['selection'])}
            self.selection_history.append(decision)
            self._json_line('selection.jsonl', decision)
            self.logger.info('Query %s selection=%s value=%.4f components=%s',
                row['query_index'], decision['mode'], decision['value']['score'], decision['value']['components'])
            candidate['energyplus_attempted'] = True
            accepted = self.bias_corrector.update([raw], [row]) if row.get('adaptation_role')!='validation' else 0
            self._json_line('bias_history.jsonl', {'query_index': row['query_index'],
                'structure': row['structure'], 'observation_accepted': bool(accepted),
                **self.bias_corrector.get_statistics()})
        self._update_pareto()
        pool = self._refinement_pool()
        self.best_direct_energy = pool[0]['energy'] if pool else None
        improved = self.best_direct_energy is not None and (previous is None or
                    self.best_direct_energy < previous*(1-1e-6))
        self.stagnant_rounds = 0 if improved else self.stagnant_rounds+1
        self._update_city_support()

    def _start_online_phase(self):
        if self.online_query_offset is not None:return
        self.online_query_offset=self.budget_used
        self.online_query_budget=self.budget_total-self.budget_used
        self.information_quota=min(self.information_quota,max(0,self.online_query_budget-1))
        # Broad bootstrap and in-sample tree-fit errors do not initialize the
        # UDI correction used on newly generated designs. Energy history stays.
        self.bias_corrector.reset_udi()
        self._json_line('online_schedule.jsonl',{'probe_attempts':self.online_query_offset,
            'online_budget':self.online_query_budget,'information_quota':self.information_quota,
            'quota_basis':'total design budget','spacing_basis':'online query index',
            'UDI_bias_reset_at_phase_boundary':True,
            'bias_statistics':self.bias_corrector.get_statistics()})

    def _should_stop(self):
        return self.budget_used >= self.budget_total or self.sampled >= self.candidate_budget

    def _finalize(self):
        return self._finish_records(self.started, {
            'city_support':self.city_support.describe() if self.city_support else None,
            'implementation': 'cluster targets + on-policy decoder TTT + raw-residual direct feedback',
            'n_target_refinements': sum(h['strategy'].startswith('energy_directed') for h in self.target_history),
            'n_cluster_switches': sum(h['strategy'].startswith('cluster_switch') for h in self.target_history),
            'n_clusters': len(self.clusters), 'cluster_visits': self.cluster_visits,
            'final_confidence': self.target_confidence, 'confidence_definition': 'prior consistency, not optimality',
            'n_candidates_generated': self.sampled, 'n_unique_valid_candidates': len(self.candidate_archive),
            'n_rejected': self.n_rejected, 'candidate_budget': self.candidate_budget,
            'batch_size': self.batch_size, 'ttt_batch_size': self.ttt_batch_size,
            'optimizer_steps': len(self.adaptation_history),
            'nonzero_parameter_updates': sum(u['parameters_changed'] for u in self.adaptation_history),
            'checkpoint': str(self.checkpoint_path), 'temperature': self.temperature,
            'bias_statistics': self.bias_corrector.get_statistics(),
            'online_interval_caveat': 'Heuristic widening after adaptive residual correction; nominal coverage not guaranteed',
            'termination_reason': 'direct_budget_exhausted' if self.budget_used >= self.budget_total else 'candidate_budget_exhausted',
            'target_source': 'cluster-matched measured spectra of formally feasible historical rows',
            'selection': 'performance anchors plus budgeted information-value queries; direct final acceptance',
            'selection_mode': self.selection_mode, 'information_fraction': self.information_fraction,
            'information_query_quota': self.information_quota if self.selection_mode == 'hybrid' else 0,
            'online_query_offset':self.online_query_offset,'online_query_budget':self.online_query_budget,
            'UDI_bias_scope':'fresh prequential fit labels; bootstrap and tree-fit rebuild excluded',
            'n_information_queries': sum(r.get('selection', {}).get('mode') == 'information' for r in self.query_history),
            'n_performance_queries': sum(r.get('selection', {}).get('mode') == 'performance' for r in self.query_history),
            'n_information_fallbacks': sum(bool(r.get('selection', {}).get('fallback_reason')) for r in self.query_history),
            'information_selector': self.information_selector.describe()})

    def run(self):
        self.started = time.time()
        if self.budget_total > 0:
            self._phase1_initialize()
            self._run_support_probes()
        self._start_online_phase()
        while not self._should_stop():
            self.iteration += 1
            self._choose_target()
            self.cluster_visits[self.current_cluster_id] += 1
            self._generate_candidates()
            batches_left = math.ceil((self.candidate_budget-self.sampled)/self.batch_size)
            queries = math.ceil((self.budget_total-self.budget_used)/(batches_left+1))
            self._query_and_evaluate(queries)
        if self.budget_used < self.budget_total:
            self._query_and_evaluate(self.budget_total-self.budget_used)
        # All remaining archive rows must describe the same final bias version.
        for row in self.candidate_archive:
            self._refresh_prediction(row)
            self._compute_information_value(row)
        result = self._finalize()
        result.update(target_history=self.target_history, target_decisions=self.target_decisions,
                      adaptation_history=self.adaptation_history, selection_history=self.selection_history)
        result['protocol']['query_selection'] = {
            'mode': self.selection_mode, 'information_fraction': self.information_fraction,
            **self.information_selector.describe()}
        for name, value in [('summary.json', result), ('candidates.json', self.candidate_archive),
                            ('rejected.json', self.generation_failures)]:
            (self.run_directory/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        return result
