import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from src.tools.city_support import FEATURES,TransferSurrogate,climate_weights
from src.tools.query_service import QueryService,read_facts,append_fact,writer_lock


def properties():
    return {**{k:.4 for k in FEATURES},'wavelengths_nm':np.arange(300,2501,10),
            **{'spectrum_'+s:{k:np.full(221,.4 if k=='T' else .3) for k in ['T','Rf','Rb']} for s in ['cold','hot']}}


def owner(directory,budget=3):
    return SimpleNamespace(city='Target',seed=5,budget_total=budget,budget_used=0,queried=set(),
        query_history=[],run_directory=Path(directory),n_tmm=0,n_processes=0,
        formal_udi_threshold=.45,baseline_energy=1000.,space=SimpleNamespace(valid=lambda s:s!='invalid'),
        tmm=SimpleNamespace(compute=Mock(return_value=properties())),
        energyplus=SimpleNamespace(run=Mock(return_value={'status':'success','TotalEnergy':900.,'UDI':.46,
            'process_started':True,'ElectricityEnergy':800.,'NaturalGasEnergy':100.})),
        surrogate=SimpleNamespace(),city_support=None,method_name='test',logger=logging.getLogger('test'))


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.a=owner(self.temp.name);self.q=QueryService(self.a)
    def tearDown(self):self.temp.cleanup()

    def test_probes_and_search_share_budget_and_best_pool(self):
        first=self.q.query('Ag_10',phase='city_support_probe')
        self.assertTrue(first['direct_feasible']);self.q.query('Ag_15',phase='online_search')
        self.q.query('Ag_20');self.assertIsNone(self.q.query('Ag_25'))
        self.assertEqual(self.a.budget_used,3);self.assertEqual(len(self.a.query_history),3)
        self.assertEqual(self.a.energyplus.run.call_count,3)

    def test_duplicate_and_preflight_do_not_spend_budget(self):
        self.q.query(' Ag_10 ');self.assertIsNone(self.q.query('Ag_10'))
        with self.assertRaises(ValueError):self.q.query('invalid')
        self.assertEqual(self.a.budget_used,1)

    def test_failure_spends_attempt_and_has_no_labels(self):
        self.a.energyplus.run.return_value={'status':'failed','UDI':None,'TotalEnergy':None,'process_started':True}
        r=self.q.query('Ag_10');self.assertFalse(r['energyplus_verified']);self.assertIsNone(r['energy'])
        self.assertEqual(self.a.budget_used,1);self.assertEqual(self.a.n_processes,1)

    def test_false_success_is_rejected(self):
        self.a.energyplus.run.return_value={'status':'success','UDI':float('nan'),'TotalEnergy':900.}
        # Invalid metrics are quarantined before writing strict JSON.
        row=self.q.query('Ag_10')
        self.assertFalse(row['energyplus_verified']);self.assertIsNone(row['udi'])
        self.assertEqual(row['energyplus_result']['error_code'],'INVALID_METRICS')
        self.assertEqual(self.a.budget_used,1)

    def test_completed_query_resume_does_not_resimulate(self):
        self.q.query('Ag_10',phase='city_support_probe')
        restored=owner(self.temp.name);q=QueryService(restored);q.reconcile()
        self.assertEqual(restored.budget_used,1);self.assertIsNone(q.query('Ag_10'))
        restored.energyplus.run.assert_not_called()

    def test_orphan_result_reconciles_without_resimulation(self):
        output=Path(self.temp.name)/'energyplus/query_001';output.mkdir(parents=True)
        result={'status':'success','TotalEnergy':900.,'UDI':.46,'process_started':True}
        (output/'result.json').write_text(json.dumps(result))
        append_fact(Path(self.temp.name)/'query_dispatch.jsonl',{'query_index':1,'structure':'Ag_10',
            'extra':{'phase':'city_support_probe'},'output_directory':str(output)})
        self.q.reconcile();self.assertEqual(len(self.a.query_history),1)
        self.assertTrue(self.a.query_history[0]['recovered_from_result']);self.a.energyplus.run.assert_not_called()

    def test_unresolved_dispatch_blocks_retry(self):
        append_fact(Path(self.temp.name)/'query_dispatch.jsonl',{'query_index':1,'structure':'Ag_10',
            'extra':{},'output_directory':str(Path(self.temp.name)/'absent')})
        with self.assertRaises(RuntimeError):self.q.reconcile()
        self.a.energyplus.run.assert_not_called()

    def test_infeasible_valid_data_stays_in_fit(self):
        self.a.energyplus.run.return_value={'status':'success','TotalEnergy':800.,'UDI':.1}
        r=self.q.query('Ag_10');self.assertFalse(r['direct_feasible']);self.assertTrue(r['energyplus_verified'])
        import pandas as pd
        self.assertEqual(len(pd.read_csv(Path(self.temp.name)/'city_support/support_fit.csv')),1)

    def test_prediction_snapshot_keeps_prequery_bias_version(self):
        from src.tools.bias_corrector import BiasCorrector
        self.a.bias_corrector=BiasCorrector()
        raw={'TotalEnergy':1000.,'UDI':.4}
        self.a.surrogate.predict=lambda features:dict(raw)
        first=self.q.query('Ag_10')
        self.a.bias_corrector.update([raw],[first])
        second=self.q.query('Ag_15')
        self.assertEqual(first['prediction_snapshot']['bias_version'],0)
        self.assertEqual(second['prediction_snapshot']['bias_version'],1)
        self.assertEqual(first['prediction_snapshot']['corrected']['UDI'],.4)
        self.assertGreater(second['prediction_snapshot']['corrected']['UDI'],.4)

    def test_concurrent_writer_is_rejected(self):
        with writer_lock(self.temp.name):
            with self.assertRaises(OSError):self.q.query('Ag_10')
        self.assertEqual(self.a.budget_used,0)

    def test_resume_rejects_different_seed(self):
        self.q.query('Ag_10')
        restored=owner(self.temp.name);restored.seed=11
        with self.assertRaises(ValueError):QueryService(restored).reconcile()
        restored.energyplus.run.assert_not_called()

    def test_resume_rejects_budget_increase(self):
        self.q.query('Ag_10')
        restored=owner(self.temp.name,budget=30)
        with self.assertRaises(ValueError):QueryService(restored).reconcile()
        restored.energyplus.run.assert_not_called()


class TransferTests(unittest.TestCase):
    def test_zero_degree_days_are_valid(self):
        tropical=dict(CDD18=4000,HDD18=0,GHI_kWh_m2=1800,diffuse_fraction=.5,RH=80,solar_seasonality=.1,latitude=1)
        weights,d=climate_weights(tropical,{'A':tropical,'B':{**tropical,'CDD18':1000}},1.)
        self.assertAlmostEqual(sum(weights['energy'].values()),1.)
        self.assertGreater(weights['energy']['A'],weights['energy']['B'])
        self.assertEqual(d['energy']['A'],0)

    def test_target_baseline_normalization_and_pure_predict(self):
        climate=dict(CDD18=1000,HDD18=0,GHI_kWh_m2=1000,diffuse_fraction=.5,RH=50,solar_seasonality=.1,latitude=1)
        def model(e,u):return SimpleNamespace(predict=Mock(return_value={'TotalEnergy':e,'UDI':u,
            'Energy_std':1.,'UDI_std':.01,'Energy_half_width_90pct':2.,'UDI_half_width_90pct':.02,'ood_distance':0.}))
        bank=SimpleNamespace(cities=['A','B'],climates={'A':climate,'B':climate},feature_scale=np.ones(8),
            protocols={'A':SimpleNamespace(energy=100.,clear_udi=.5),'B':SimpleNamespace(energy=200.,clear_udi=.4)},
            models={'A':model(80.,.4),'B':model(160.,.32)})
        target=SimpleNamespace(energy=1000.,clear_udi=.6)
        p=TransferSurrogate(bank,target,climate)
        r=p.predict(np.full(8,.4));self.assertAlmostEqual(r['TotalEnergy'],800.)
        self.assertAlmostEqual(r['UDI'],.48);self.assertEqual(r['uncertainty_status'],'target_uncalibrated_heuristic')
        p.predict(np.full(8,.4));self.assertEqual(bank.models['A'].predict.call_count,1)


if __name__=='__main__':unittest.main()
