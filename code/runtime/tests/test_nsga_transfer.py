import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class NSGATransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=None
        if not os.environ.get('CITY_SUPPORT_SETTINGS'):
            cls.profile=tempfile.TemporaryDirectory();d=Path(cls.profile.name)
            (d/'weather_registry.json').write_text(json.dumps({'source_cities':[],'cities':{},'dataset':'TMYx','period':'2011-2025'}))
            (d/'baseline.csv').write_text('city,UDI,TotalEnergy_J\n')
            (d/'settings.json').write_text(json.dumps({'schema_version':1,'project_root':str(d),'campaign_root':str(d),'baseline_table':str(d/'baseline.csv')}))
            os.environ['CITY_SUPPORT_SETTINGS']=str(d/'settings.json')
            from src.support_settings import settings
            settings.cache_clear()
        from src.agent.nsga2_method import NSGA2Method
        cls.method=NSGA2Method

    @classmethod
    def tearDownClass(cls):
        if cls.profile:
            os.environ.pop('CITY_SUPPORT_SETTINGS',None)
            from src.support_settings import settings
            settings.cache_clear();cls.profile.cleanup()

    def test_transfer_constraint_uses_margin_not_heuristic_interval(self):
        from src.vendor.thermo_agent.nsga2 import constraint_violation
        row={'predicted_udi':.82,'predicted_energy':900.,'udi_half_width_90pct':.04,'constraint_udi_margin':.003}
        self.assertEqual(constraint_violation(self.method._proxy(row),.815),0.)
        row.pop('constraint_udi_margin')
        self.assertGreater(constraint_violation(self.method._proxy(row),.815),0.)

    def test_bootstrap_roles_and_phase_reset(self):
        from src.tools.bias_corrector import BiasCorrector
        a=self.method.__new__(self.method);a.probe_count=16;a.budget_total=40;a.budget_used=0
        samples=[{'structure':f'S{i}','probe_id':str(i),'selection_reason':'fixed',
                  'adaptation_role':'validation' if i%4==0 else 'fit'} for i in range(1,17)]
        a.city_support=SimpleNamespace(propose_probes=lambda owner,n:{'samples':samples})
        a.tmm=SimpleNamespace(compute=lambda s:{k:.3 for k in ['T_lum_cold','T_lum_hot','T_sol_cold','T_sol_hot','Rf_sol_cold','Rb_sol_cold','Rf_sol_hot','Rb_sol_hot']})
        a.queried=set();a.optics={};a.n_tmm=0;a.bias_corrector=BiasCorrector();a._json_line=Mock()
        calls=[]
        def query(candidate,**extra):
            a.budget_used+=1;a.queried.add(candidate['structure']);calls.append(extra)
        a._query_candidate=query;a._update_city_support=Mock()
        self.method._run_support_probes(a)
        self.assertEqual(a.budget_used,16);self.assertEqual(a.budget_total-a.budget_used,24)
        self.assertEqual(sum(c['adaptation_role']=='fit' for c in calls),12)
        self.assertEqual(sum(c['adaptation_role']=='validation' for c in calls),4)
        a._update_city_support.assert_called_once()
        self.assertEqual(a.bias_corrector.n_udi_observations,0)

    def test_model_publication_rebuild_excludes_validation_and_UDI(self):
        from src.tools.bias_corrector import BiasCorrector
        a=self.method.__new__(self.method)
        predictor=SimpleNamespace(VERSION='new',predict=lambda x:{'UDI':.84,'TotalEnergy':900.})
        a.surrogate=SimpleNamespace(VERSION='old')
        a.city_support=SimpleNamespace(predictor=predictor,fit_or_keep=lambda owner:{'decision':'publish'})
        a.query_history=[{'structure':'A','energyplus_verified':True,'adaptation_role':'fit','optical_properties':{},'udi':.82,'energy':1000.},
                         {'structure':'B','energyplus_verified':True,'adaptation_role':'validation','optical_properties':{},'udi':.2,'energy':2000.}]
        a.archive=[];a.budget_used=16;a._json_line=Mock();a.bias_corrector=BiasCorrector()
        self.method._update_city_support(a)
        self.assertEqual(a.surrogate.VERSION,'new');self.assertEqual(a.bias_corrector.n_observations,1)
        self.assertEqual(a.bias_corrector.n_udi_observations,0);self.assertEqual(a.bias_corrector.udi_bias,0.)

    def test_validation_query_cannot_fit_residual(self):
        a=self.method.__new__(self.method);a.city_support=object();a._refresh_prediction=lambda r:r
        a.bias_corrector=SimpleNamespace(update=Mock(),get_statistics=lambda:{})
        a._json_line=Mock();a._direct_query=lambda s,**extra:{**extra,'query_index':1,'energyplus_verified':True}
        row={'structure':'S','raw_prediction':{'UDI':.8,'TotalEnergy':900.},'predicted_energy':900.,'predicted_udi':.8,
             'performance_udi_margin':.003,'prediction_decision_threshold':.818,'udi_half_width_90pct':.02,'ood_distance':.3}
        self.method._query_candidate(a,row,phase='city_support_probe',adaptation_role='validation')
        a.bias_corrector.update.assert_not_called()

    def test_direct_search_only_uses_remaining_budget(self):
        a=self.method.__new__(self.method);a.budget_total=40;a.budget_used=0;a.fitness_source='direct'
        a.candidate_budget=384;a.population_size=20;a.requested_population_size=20
        a.evaluated=set();a.queried=set();a.population=[];a.archive=[];a.generation=0;a.generation_log=[];a.query_history=[]
        a.n_generation_attempts=0;a.formal_udi_threshold=.815;a.formal=SimpleNamespace(udi_ratio=.9)
        a.city_support=None;a.probe_count=16;a.logger=SimpleNamespace(info=lambda *x:None)
        a._run_support_probes=lambda:setattr(a,'budget_used',16)
        a._next_unique=lambda pending:f'S{len(a.evaluated)}'
        def query(s,**extra):
            a.budget_used+=1
            return {'structure':s,'energyplus_verified':True,'energy':900.,'udi':.85}
        a._direct_query=query;a._finish_records=lambda start,data:data
        self.method.run(a)
        self.assertEqual(a.budget_used,40);self.assertEqual(len(a.evaluated),24)


if __name__=='__main__':unittest.main()
