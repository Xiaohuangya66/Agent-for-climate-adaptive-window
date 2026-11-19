import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from src.tools.bias_corrector import BiasCorrector
from src.tools.information_selector import InformationValueSelector
from src.tools.optimization_policy import DEFAULT_POLICY
from src.agent.formal_protocol import FormalProtocol


def protocol():
    p=FormalProtocol.__new__(FormalProtocol)
    p.energy=1000.;p.threshold=.8150170648464163;p.clear_udi=.9055745164960182
    p.policy=dict(DEFAULT_POLICY)
    return p


class BoundaryPolicyTests(unittest.TestCase):
    def test_fit_rebuild_preserves_energy_correction_without_UDI_bias(self):
        b=BiasCorrector()
        b.update([{'UDI':.8,'TotalEnergy':900.}],[{'structure':'A','udi':.75,'energy':1000.}],prequential=False)
        self.assertEqual(b.n_observations,1);self.assertEqual(b.n_udi_observations,0)
        self.assertEqual(b.udi_bias,0);self.assertGreater(b.energy_bias,0)

    def test_new_query_updates_UDI_prequentially(self):
        b=BiasCorrector()
        b.update([{'UDI':.8,'TotalEnergy':900.}],[{'structure':'A','udi':.75,'energy':1000.}],prequential=False)
        b.update([{'UDI':.83,'TotalEnergy':900.}],[{'structure':'B','udi':.826,'energy':1000.}])
        self.assertEqual(b.n_udi_observations,1)
        self.assertAlmostEqual(b.udi_bias,-.001)

    def test_reset_UDI_keeps_energy_history(self):
        b=BiasCorrector();b.update([{'UDI':.8,'TotalEnergy':900.}],[{'structure':'A','udi':.75,'energy':1000.}])
        energy=b.energy_bias;b.reset_udi()
        self.assertEqual(b.energy_bias,energy);self.assertEqual(b.n_observations,1)
        self.assertEqual(b.udi_bias,0);self.assertEqual(b.n_udi_observations,0)

    def test_corrected_interval_tracks_center_and_width(self):
        b=BiasCorrector();b.update([{'UDI':.84,'TotalEnergy':900.}],[{'structure':'A','udi':.8,'energy':1000.}])
        p=b.correct({'UDI':.84,'TotalEnergy':900.,'UDI_half_width_90pct':.02,
                     'UDI_lower_90pct':.82,'UDI_upper_90pct':.86})
        self.assertAlmostEqual(p['UDI_lower_90pct'],max(0,p['UDI']-p['UDI_half_width_90pct']))
        self.assertAlmostEqual(p['UDI_upper_90pct'],min(1,p['UDI']+p['UDI_half_width_90pct']))

    def test_margin_fallback_and_cap(self):
        b=BiasCorrector();self.assertEqual(b.get_decision_margin(),.003)
        b.update([{'UDI':.8,'TotalEnergy':900.}],[{'structure':'A','udi':.6,'energy':1000.}])
        self.assertEqual(b.get_decision_margin(),.01)

    def test_information_schedule_is_spread_after_probe_offset(self):
        count=0;indices=[]
        for q in range(1,25):
            if InformationValueSelector.information_due_with_quota(q,24,12,count):
                indices.append(q);count+=1
        self.assertEqual(indices,list(range(2,25,2)))
        self.assertEqual(count,12)

    def test_boundary_good_design_can_be_a_performance_anchor(self):
        p=protocol()
        boundary=p.prediction({'UDI':.8184897207,'TotalEnergy':900.,'UDI_half_width_90pct':.027})
        safe=p.prediction({'UDI':.86,'TotalEnergy':930.,'UDI_half_width_90pct':.02})
        self.assertEqual(p.query_key(boundary)[0],0)
        self.assertLess(p.query_key(boundary),p.query_key(safe))
        self.assertGreater(boundary['reward'],safe['reward'])
        self.assertFalse(boundary['conservative_feasible'])

    def test_real_acceptance_remains_strict(self):
        p=protocol()
        result=p.summarize([{'structure':'dark','energyplus_verified':True,'energy':800.,'udi':.81},
                            {'structure':'valid','energyplus_verified':True,'energy':950.,'udi':.82}])
        self.assertEqual(result['best_structure'],'valid');self.assertTrue(result['best_is_feasible'])


class NonemptyPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=None
        if not os.environ.get('CITY_SUPPORT_SETTINGS'):
            cls.profile=tempfile.TemporaryDirectory();d=Path(cls.profile.name)
            (d/'weather_registry.json').write_text(json.dumps({'source_cities':[],'cities':{},'dataset':'TMYx','period':'2011-2025'}))
            (d/'baseline.csv').write_text('city,UDI,TotalEnergy_J\n')
            (d/'settings.json').write_text(json.dumps({'schema_version':1,'project_root':str(d),
                'campaign_root':str(d),'baseline_table':str(d/'baseline.csv')}))
            os.environ['CITY_SUPPORT_SETTINGS']=str(d/'settings.json')
            from src.support_settings import settings
            settings.cache_clear()
        from src.agent.agent_full import AgentFull
        cls.AgentFull=AgentFull

    @classmethod
    def tearDownClass(cls):
        if cls.profile:
            os.environ.pop('CITY_SUPPORT_SETTINGS',None)
            from src.support_settings import settings
            settings.cache_clear();cls.profile.cleanup()

    def test_nonempty_candidates_refresh_without_changing_query_snapshots(self):
        a=self.AgentFull.__new__(self.AgentFull);a.formal=protocol()
        features={k:.4 for k in ('T_lum_cold','T_lum_hot','T_sol_cold','T_sol_hot','Rf_sol_cold','Rb_sol_cold','Rf_sol_hot','Rb_sol_hot')}
        old={'UDI':.84,'TotalEnergy':850.,'UDI_half_width_90pct':.02}
        new={'UDI':.82,'TotalEnergy':900.,'UDI_half_width_90pct':.02}
        model=SimpleNamespace(VERSION='new',predict=lambda *args,**kwargs:dict(new))
        a.surrogate=SimpleNamespace(VERSION='old')
        a.city_support=SimpleNamespace(predictor=model,fit_or_keep=lambda owner:{'decision':'publish'})
        a.bias_corrector=BiasCorrector();a.iteration=1;a.budget_used=16
        a.query_history=[{'structure':f'S{i}','query_index':i+1,'energyplus_verified':True,
            'energy':920.,'udi':.81,'adaptation_role':'fit' if i<12 else 'validation',
            'optical_properties':features,'prediction_snapshot':{'raw':dict(old)}} for i in range(16)]
        before=copy.deepcopy(a.query_history)
        a.candidate_archive=[{'structure':'candidate','optical_properties':features,
            'raw_prediction':dict(old),'surrogate_version':'old'}]
        a.clusters=[{'center':np.full(8,.4)}]
        with tempfile.TemporaryDirectory() as directory:
            a.run_directory=Path(directory)
            text=''.join(json.dumps(r)+'\n' for r in a.query_history)
            (a.run_directory/'queries.jsonl').write_text(text)
            a._update_city_support()
            row=a.candidate_archive[0]
            self.assertEqual(row['surrogate_version'],'new')
            self.assertEqual(row['raw_prediction']['TotalEnergy'],900.)
            self.assertEqual(row['bias_version'],12)
            self.assertEqual(a.bias_corrector.n_udi_observations,0)
            self.assertEqual(a.query_history,before)
            self.assertEqual((a.run_directory/'queries.jsonl').read_text(),text)
            self.assertTrue((a.run_directory/'model_updates.jsonl').is_file())


if __name__=='__main__':unittest.main()
