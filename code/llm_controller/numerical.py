"""Reuse numeric tools; validation labels never drive online target/acquisition."""
import math
from src.agent.agent_full import AgentFull
from src.tools.bias_corrector import BiasCorrector
from src.tools.information_selector import InformationValueSelector


class ProfileBiasCorrector(BiasCorrector):
    margin_profile = 'balanced'

    def get_decision_margin(self):
        base = super().get_decision_margin()
        from src.tools.optimization_policy import policy_configuration
        cap = policy_configuration()['margin_maximum']
        return min(cap, base * {'boundary': .5, 'balanced': 1., 'cautious': 1.5}[self.margin_profile])


class RoundInformationSelector(InformationValueSelector):
    forced_due = None

    def information_due_with_quota(self, query_index, budget, quota, information_used):
        if self.forced_due is None:
            return super().information_due_with_quota(query_index, budget, quota, information_used)
        return bool(self.forced_due and query_index > 1 and information_used < quota)


class LLMNumericalAgent(AgentFull):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_profile_bias()
        old = self.information_selector
        self.information_selector = RoundInformationSelector(old.baseline_energy, old.udi_threshold,
                                                            old.feature_scale, old.weights)
        self.method_name = 'llm_agent'

    def _install_profile_bias(self):
        previous = self.bias_corrector
        replacement = ProfileBiasCorrector(window_size=previous.window_size,
                                           prior_strength=previous.prior_strength)
        replacement.__dict__.update(previous.__dict__)
        self.bias_corrector = replacement

    def _update_city_support(self):
        # An explicit fit_transfer_model tool owns the single publication attempt.
        return None

    def fit_transfer_explicit(self):
        AgentFull._update_city_support(self)
        self._install_profile_bias()

    def _start_online_phase(self):
        if self.online_query_offset is not None:return
        super()._start_online_phase()
        # Small arbitrary-city test budgets must still leave one performance
        # anchor in each round. Standard source20/target40 quotas are unchanged.
        rounds = min(self.online_query_budget, math.ceil(self.candidate_budget/self.batch_size))
        cap = max(0, self.online_query_budget-rounds)
        previous = self.information_quota
        self.information_quota = min(previous, cap)
        if previous != self.information_quota:
            self._json_line('online_schedule.jsonl', {'reason': 'reserve_per_round_performance_anchors',
                'previous_information_quota': previous, 'applied_information_quota': self.information_quota,
                'online_budget': self.online_query_budget, 'generation_rounds': rounds})

    def _refinement_pool(self):
        return [row for row in super()._refinement_pool() if row.get('adaptation_role', 'fit') == 'fit']

    def _selection_references(self):
        return [row for row in super()._selection_references() if row.get('adaptation_role', 'fit') == 'fit']
