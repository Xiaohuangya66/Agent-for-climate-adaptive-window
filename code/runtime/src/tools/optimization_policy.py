"""Explicit prediction-risk policy; final acceptance always uses real UDI."""
import math

DEFAULT_POLICY={
    'version':'boundary_search_prequential_v3_20261029',
    'margin_default':.003,
    'margin_error_multiplier':.5,
    'margin_maximum':.01,
    'reward_udi_penalty':5.,
    'reward_energy_uncertainty_penalty':.05,
    'reward_feasibility':'corrected_center_plus_adaptive_margin',
    'performance_feasibility':'corrected_center_plus_adaptive_margin',
    'information_feasibility':'corrected_center_at_real_threshold',
    'information_quota_basis':'total_design_budget',
    'information_schedule_basis':'online_queries_after_probes',
    'udi_bias_scope':'post_publication_prequential_fit_observations',
    'energy_bias_scope':'all_valid_fit_observations_current_model',
    'uncertainty_status':'heuristic_not_target_calibrated',
}


def policy_configuration():
    from src.support_settings import settings
    result={**DEFAULT_POLICY,**settings().get('optimization_policy',{})}
    for key in ['margin_default','margin_error_multiplier','margin_maximum']:
        if not math.isfinite(result[key]) or result[key]<0:raise ValueError('Invalid margin policy')
    if result['margin_default']>result['margin_maximum']:raise ValueError('Default margin exceeds cap')
    return result


def decision_margin(mae, count, policy):
    if count<1 or mae is None:return float(policy['margin_default'])
    if not math.isfinite(mae) or mae<0:raise ValueError('Invalid prequential UDI error')
    return min(float(policy['margin_maximum']),float(policy['margin_error_multiplier'])*mae)
