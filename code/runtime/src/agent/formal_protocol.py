"""One daylight-UDI/site-energy protocol for all three server methods."""
from pathlib import Path
import csv
import math

FEATURE_ORDER=('T_lum_cold','T_lum_hot','T_sol_cold','T_sol_hot',
               'Rf_sol_cold','Rb_sol_cold','Rf_sol_hot','Rb_sol_hot')


def configure_udi_ratio(value):
    """Set one process-wide ratio before constructing agents or city priors."""
    if isinstance(value,bool):
        raise ValueError('UDI ratio must be a finite number in (0,1]')
    ratio=float(value)
    if not math.isfinite(ratio) or not 0<ratio<=1:
        raise ValueError('UDI ratio must be a finite number in (0,1]')
    from src import config
    config.UDI_THRESHOLD_RATIO=ratio
    return ratio


class FormalProtocol:
    def __init__(self,city,root=None,udi_ratio=None):
        from src import config
        self.udi_ratio=float(config.UDI_THRESHOLD_RATIO if udi_ratio is None else udi_ratio)
        if not math.isfinite(self.udi_ratio) or not 0<self.udi_ratio<=1:
            raise ValueError('UDI ratio must be a finite number in (0,1]')
        if root is None:
            from src.runtime_profile import BASELINE_TABLE
            self.table=BASELINE_TABLE
        else:
            self.table=Path(root)/'protocol/city_baselines_goal2.csv'
        with self.table.open(encoding='utf-8-sig') as stream:
            self.row=next((r for r in csv.DictReader(stream) if r['city']==city),None)
        if not self.row or self.row['formal_udi_definition']!='daylight_hours':
            raise ValueError(f'Formal daylight-hours baseline missing for {city}')
        if 'NaturalGas:Facility' not in self.row['energy_definition']:
            raise ValueError('Formal baseline must include natural gas')
        self.clear_udi=float(self.row['UDI'])
        baseline_threshold_95pct=float(self.row['UDI_threshold_95pct'])
        self.energy=float(self.row['TotalEnergy_J'])
        if not (0<self.clear_udi<=1 and self.energy>0 and
                math.isclose(baseline_threshold_95pct,.95*self.clear_udi,rel_tol=1e-10)):
            raise ValueError('Invalid clear-glass 95% baseline')
        # The archived CSV's 95% column validates its generation, not this run's gate.
        self.threshold=self.udi_ratio*self.clear_udi
        from src.tools.optimization_policy import policy_configuration
        self.policy=policy_configuration()

    def describe(self):
        return {'version':'formal_tmyx_daylight_site_v6_four_source_20261106',
                'baseline_file':str(self.table),'clear_glass_udi':self.clear_udi,
                'formal_direct_udi_threshold':self.threshold,'udi_ratio_required':self.udi_ratio,
                'baseline_reference_threshold_95pct':float(self.row['UDI_threshold_95pct']),
                'energy_baseline_J':self.energy,'UDI':'daylight_hours; 100-2000 lux',
                'energy':'Electricity:Facility + NaturalGas:Facility (J)',
                'weather_dataset':self.row.get('weather_dataset'),
                'weather_period':self.row.get('weather_period'),
                'prediction_and_query_policy':self.policy,
                'formal_acceptance':'real UDI >= ratio * same-city clear UDI; prediction margins never change this gate'}

    def prediction(self,values):
        udi=float(values['UDI']);energy=float(values['TotalEnergy'])
        half=float(values.get('UDI_half_width_90pct',0))
        energy_half=float(values.get('Energy_half_width_90pct',0))
        if not all(math.isfinite(v) for v in [udi,energy,half,energy_half]) or energy<=0 or half<0 or energy_half<0:
            raise ValueError('Invalid formal surrogate prediction')
        udi_unbounded=udi
        udi=min(1.,max(0.,udi))
        lower=max(0.,udi-half);upper=min(1.,udi+half)
        saving=(self.energy-energy)/self.energy
        from src.tools.optimization_policy import DEFAULT_POLICY
        policy=getattr(self,'policy',DEFAULT_POLICY)
        margin=float(values.get('decision_udi_margin',policy['margin_default']))
        if not math.isfinite(margin) or not 0<=margin<=policy['margin_maximum']:
            raise ValueError('Invalid prediction margin')
        decision_threshold=self.threshold+margin
        return {'predicted_udi':udi,'predicted_energy':energy,
                'predicted_udi_unbounded':udi_unbounded,
                'predicted_udi_lower_90pct':lower,'predicted_udi_upper_90pct':upper,
                'udi_half_width_90pct':half,'energy_half_width_90pct':energy_half,
                'conservative_feasible':lower>=self.threshold,'optimistic_feasible':upper>=self.threshold,
                'predicted_center_feasible':udi>=self.threshold,
                'performance_udi_margin':margin,'prediction_decision_threshold':decision_threshold,
                'decision_policy_version':policy['version'],
                'reward':saving-policy['reward_udi_penalty']*max(0,decision_threshold-udi)/decision_threshold
                         -policy['reward_energy_uncertainty_penalty']*energy_half/self.energy,
                'ood_distance':float(values.get('ood_distance',0))}

    def query_key(self,row):
        from src.tools.optimization_policy import DEFAULT_POLICY
        policy=getattr(self,'policy',DEFAULT_POLICY)
        predicted_udi=row['predicted_udi']
        margin=row.get('performance_udi_margin',policy['margin_default'])
        if predicted_udi>=self.threshold+margin:
            return (0, row['predicted_energy'], -row['predicted_udi'])
        elif predicted_udi>=self.threshold:
            return (1, row['predicted_energy'], -row['predicted_udi'])
        elif row['optimistic_feasible']:
            return (2, row['predicted_energy'], -row['predicted_udi'])
        else:
            upper = predicted_udi + row['udi_half_width_90pct']
            return (3, -upper, row['predicted_energy'])

    def summarize(self,records):
        success=[r for r in records if r.get('energyplus_verified') and r.get('energy') is not None and r.get('udi') is not None]
        feasible=[r for r in success if r['udi']>=self.threshold]
        best=min(feasible,key=lambda r:(r['energy'],-r['udi'])) if feasible else None
        fallback=max(success,key=lambda r:(r['udi'],-r['energy'])) if success else None
        selected=best or fallback
        unconstrained=min(success,key=lambda r:(r['energy'],-r['udi'])) if success else None
        return {'best_structure':selected['structure'] if selected else '',
                'best_udi':selected['udi'] if selected else None,
                'best_energy':selected['energy'] if selected else None,
                'best_energy_saving':(self.energy-best['energy'])/self.energy if best else None,
                'best_is_feasible':best is not None,'best_feasible':best,'n_feasible':len(feasible),
                'optimization_status':'feasible_design_found' if best else 'no_feasible_design',
                'selection_basis':'minimum_direct_energy_subject_to_direct_udi' if best else 'maximum_direct_udi_then_minimum_energy_no_feasible',
                'diagnostic_selected_energy_saving':(self.energy-selected['energy'])/self.energy if selected else None,
                'unconstrained_min_energy':unconstrained}

    def summarize_checkpoints(self,records,budget_used,checkpoints=(20,40)):
        results=[]
        for checkpoint in checkpoints:
            prefix=[r for r in records if r['query_index']<=checkpoint]
            reached=budget_used>=checkpoint and len(prefix)==checkpoint
            best=self.summarize(prefix)['best_feasible'] if reached else None
            probes=sum(r.get('phase')=='city_support_probe' for r in prefix)
            meters=best.get('energyplus_result',{}) if best else {}
            results.append({'budget_checkpoint':checkpoint,'reached':reached,
                'attempts_recorded':len(prefix),'probe_attempts':probes,
                'online_attempts':len(prefix)-probes,'best_is_feasible':best is not None,
                'best_energy_saving':best['energy_saving'] if best else None,
                'best_energy_J':best['energy'] if best else None,'best_udi':best['udi'] if best else None,
                'UDI_retention_fraction':best['udi']/self.clear_udi if best else None,
                'best_structure':best['structure'] if best else None,
                'best_query_index':best['query_index'] if best else None,
                'ElectricityFacility_J':meters.get('ElectricityEnergy'),
                'NaturalGasFacility_J':meters.get('NaturalGasEnergy'),
                'n_WVO2':sum(t.startswith('WVO2_') for t in best['structure'].split()) if best else None,
                'n_layers':len(best['structure'].split()) if best else None,
                'dT_sol':best['optical_properties']['T_sol_cold']-best['optical_properties']['T_sol_hot'] if best and 'optical_properties' in best else None,
                'dT_lum':best['optical_properties']['T_lum_cold']-best['optical_properties']['T_lum_hot'] if best and 'optical_properties' in best else None,
                **{k:best.get('optical_properties',{}).get(k) if best else None for k in FEATURE_ORDER}})
        return results
