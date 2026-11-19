"""Behavioral guards for real dispatch boundaries; no external API/EnergyPlus."""
import copy
import json
import math
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm_controller.api import ChatClient, APIError, redact
from llm_controller.controller import ToolSession, ToolRejected, schemas, validate, decision_residuals
from llm_controller.numerical import LLMNumericalAgent, ProfileBiasCorrector


def fake_agent():
    return SimpleNamespace(budget_total=40, budget_used=16, candidate_budget=384, sampled=0,
                           batch_size=128, information_quota=12, query_history=[])


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = schemas(['evaluate_round'])[0]['function']['parameters']
        self.good = dict(state_version=3, rationale='next evaluation', strategy_id='r1', candidate_set_id='c1', k=8)

    def test_integer_boolean_rejected(self):
        with self.assertRaises(ToolRejected):
            validate({**self.good, 'k': True}, self.schema)

    def test_extra_role_and_budget_rejected(self):
        for key in ['role', 'budget', 'udi_ratio']:
            with self.assertRaises(ToolRejected):
                validate({**self.good, key: 'validation'}, self.schema)

    def test_nonfinite_not_integer(self):
        with self.assertRaises(ToolRejected):
            validate({**self.good, 'k': float('nan')}, self.schema)

    def test_valid_schema(self):
        validate(self.good, self.schema)

    def test_evidence_prose_rejected_at_schema(self):
        schema = schemas(['set_round_strategy'])[0]['function']['parameters']
        good = dict(state_version=4, rationale='next', target_action='keep', cluster_id=0,
                    margin_profile='balanced', n_information=1, evidence=['residuals'])
        validate(good, schema)
        with self.assertRaises(ToolRejected):
            validate({**good, 'evidence': ['residuals.energy_mae is large']}, schema)


class StateTests(unittest.TestCase):
    def session(self):
        return ToolSession(fake_agent())

    def test_round_capacity_fixed_information(self):
        s = self.session()
        self.assertEqual(s.round_plan(), dict(queries=8, candidates=128, rounds_left=3,
            information_remaining=12, information_min=0, information_max=7))
        s.a.sampled = 128; s.a.budget_used = 24
        s.a.query_history = [{'selection': {'mode': 'information'}}]*2
        self.assertEqual(s.round_plan()['information_min'], 3)
        self.assertEqual(s.round_plan()['information_max'], 7)
        s.a.sampled = 256; s.a.budget_used = 32
        s.a.query_history = [{'selection': {'mode': 'information'}}]*7
        self.assertEqual(s.round_plan()['information_min'], 5)
        self.assertEqual(s.round_plan()['information_max'], 5)

    def test_stale_version_no_dispatch(self):
        s = self.session(); s.phase = 'probes'; s.version = 4
        s.do_run_support_probes = Mock()
        with self.assertRaises(ToolRejected):
            s.call('run_support_probes', dict(state_version=3, rationale='run', plan_id='support-plan-1'), 'id1')
        s.do_run_support_probes.assert_not_called()
        self.assertEqual(s.a.budget_used, 16)

    def test_wrong_phase_no_dispatch(self):
        s = self.session()
        with self.assertRaises(ToolRejected):
            s.call('evaluate_round', dict(state_version=0, rationale='bad', strategy_id='r', candidate_set_id='c', k=8), 'id1')

    def test_idempotent_id_cannot_double_dispatch(self):
        s = self.session(); s.phase = 'probes'; s.state = lambda: {'state_version': s.version}
        s.do_run_support_probes = Mock(return_value={'executed': 16})
        args = dict(state_version=0, rationale='run', plan_id='support-plan-1')
        first = s.call('run_support_probes', args, 'id1')
        second = s.call('run_support_probes', args, 'id1')
        self.assertEqual(first, second)
        self.assertEqual(s.version, 1)
        s.do_run_support_probes.assert_called_once()
        with self.assertRaises(ToolRejected):
            s.call('run_support_probes', {**args, 'rationale': 'changed'}, 'id1')

    def test_bad_ticket_rejected_before_generation(self):
        s = self.session(); s.a._generate_candidates = Mock()
        s.strategy = {'strategy_id': 'r1', 'candidates': 128}
        with self.assertRaises(ToolRejected):
            s.do_generate_round(dict(strategy_id='r2', n=128))
        s.a._generate_candidates.assert_not_called()

    def test_over_budget_rejected_before_energyplus(self):
        s = self.session(); s.a.budget_used = 39
        s.strategy = dict(strategy_id='r1', candidate_set_id='c1', queries=8)
        s.a._query_and_evaluate = Mock()
        with self.assertRaises(ToolRejected):
            s.do_evaluate_round(dict(strategy_id='r1', candidate_set_id='c1', k=8))
        s.a._query_and_evaluate.assert_not_called()

    def test_evaluation_queries_sequential_and_first_performance(self):
        s = self.session(); a = s.a
        a.iteration = 1; a.stagnant_rounds = 0
        a.information_selector = SimpleNamespace(forced_due=None)
        a._refinement_pool = lambda: sorted(a.query_history, key=lambda r: r['energy'])
        a._json_line = Mock()
        observed = []
        def query_one(count):
            observed.append((count, len(a.query_history), a.information_selector.forced_due))
            a.budget_used += 1
            a.query_history.append({'query_index': a.budget_used, 'adaptation_role': 'fit',
                'energyplus_verified': True, 'direct_feasible': True, 'energy': 1000-a.budget_used,
                'energy_saving': .1, 'selection': {'mode': 'information' if a.information_selector.forced_due else 'performance'}})
        a._query_and_evaluate = query_one
        s.strategy = dict(strategy_id='r1', candidate_set_id='c1', queries=8, n_information=3)
        result = s.do_evaluate_round(dict(strategy_id='r1', candidate_set_id='c1', k=8))
        self.assertEqual([n for n,_,_ in observed], [1]*8)
        self.assertEqual([n for _,n,_ in observed], list(range(8)))
        self.assertFalse(observed[0][2])
        self.assertEqual(sum(v for _,_,v in observed), 3)
        self.assertEqual(result['n_information'], 3)
        self.assertEqual(a.stagnant_rounds, 0)


class IsolationTests(unittest.TestCase):
    def test_small_transfer_budget_leaves_per_round_performance_anchors(self):
        for budget, probes, candidates, batch, expected in [(20,16,384,128,1),
                (40,16,384,128,12), (20,0,384,64,6), (6,0,192,64,2)]:
            with self.subTest(budget=budget, probes=probes):
                a = object.__new__(LLMNumericalAgent)
                a.budget_total=budget; a.budget_used=probes; a.candidate_budget=candidates; a.batch_size=batch
                a.online_query_offset=None; a.online_query_budget=None
                a.information_quota=math.ceil(budget*.3)
                a.bias_corrector=Mock(); a.bias_corrector.get_statistics.return_value={}
                a._json_line=Mock()
                a._start_online_phase()
                self.assertEqual(a.information_quota,expected)

    def test_energy_residual_scale_is_explicit_and_missing_stays_unknown(self):
        a = SimpleNamespace(formal=SimpleNamespace(energy=1.6e12),
            bias_corrector=SimpleNamespace(get_statistics=lambda: dict(energy_mae=2e9, energy_bias=-1e9, energy_std=None)))
        values = decision_residuals(a)
        self.assertEqual(values['energy_mae_fraction_of_baseline'], .00125)
        self.assertEqual(values['energy_bias_fraction_of_baseline'], -.000625)
        self.assertIsNone(values['energy_std_fraction_of_baseline'])

    def test_validation_cannot_enter_targets_or_acquisition(self):
        a = object.__new__(LLMNumericalAgent)
        a.query_history = [dict(structure='fit', energyplus_verified=True, direct_feasible=True,
                               energy=100, udi=.85, adaptation_role='fit'),
                           dict(structure='validation', energyplus_verified=True, direct_feasible=True,
                               energy=50, udi=.99, adaptation_role='validation')]
        self.assertEqual([r['structure'] for r in a._refinement_pool()], ['fit'])
        self.assertEqual([r['structure'] for r in a._selection_references()], ['fit'])

    def test_margin_profiles_bounded(self):
        b = ProfileBiasCorrector()
        with patch('src.tools.optimization_policy.policy_configuration', return_value={
            'margin_default': .003, 'margin_error_multiplier': .5, 'margin_maximum': .01}):
            b.margin_profile = 'boundary'; self.assertAlmostEqual(b.get_decision_margin(), .0015)
            b.margin_profile = 'cautious'; self.assertAlmostEqual(b.get_decision_margin(), .0045)
            b.udi_observations.append({'error_udi': .1}); b.n_udi_observations = 1
            self.assertEqual(b.get_decision_margin(), .01)

    def test_automatic_publication_disabled(self):
        a = object.__new__(LLMNumericalAgent)
        a.city_support = Mock()
        a._update_city_support()
        a.city_support.fit_or_keep.assert_not_called()


class CredentialTests(unittest.TestCase):
    def test_missing_environment_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(APIError):
            ChatClient()

    def test_credential_redaction(self):
        sample = 'sk-'+'x'*24
        with patch.dict(os.environ, {'DMX_API_KEY': sample}):
            self.assertNotIn(sample, redact('error '+sample))


if __name__ == '__main__':
    unittest.main()
