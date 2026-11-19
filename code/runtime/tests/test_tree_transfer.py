import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from src.vendor.thermo_agent.surrogate import SurrogateEnsemble
from src.tools.city_support import FEATURES,TransferSurrogate
from src.tools.tree_transfer import TreeTransferSurrogate,virtual_source_labels,observed_arrays,compare_candidate


class Native:
    def __init__(self,ensemble):self.ensemble=ensemble;self._cache={}
    def predict(self,x,strict=True):
        p=self.ensemble.predict(np.asarray(x))
        raw=float(p.mean[0,0])
        return {'UDI':float(np.clip(raw,0,1)),'UDI_unbounded':raw,'TotalEnergy':float(p.mean[0,1]),
            'UDI_std':float(p.std[0,0]),'Energy_std':float(p.std[0,1]),
            'UDI_half_width_90pct':.02,'Energy_half_width_90pct':2.,'ood_distance':float(p.ood_distance[0])}


def fixture():
    rng=np.random.default_rng(5);x=rng.random((64,8));y=np.c_[.2+.2*x[:,0],100-20*x[:,1]]
    e=SurrogateEnsemble.__new__(SurrogateEnsemble)
    e.features=FEATURES;e.targets=('UDI','TotalEnergy')
    e.x_mean=np.full(8,.5);e.x_scale=np.full(8,.3);e.y_mean=np.array([.3,90.]);e.y_scale=np.array([.1,10.])
    e.train_features=(x-e.x_mean)/e.x_scale;e.conformal_multiplier=np.array([1.6,1.6]);e.models=[]
    models=[]
    for j in range(2):
        d=xgb.DMatrix(e.train_features,label=(y[:,j]-e.y_mean[j])/e.y_scale[j])
        models.append(xgb.train({'objective':'reg:squarederror','nthread':1,'max_depth':2,'eta':.2,'base_score':0},d,num_boost_round=8))
    e.models.append(models)
    climate=dict(CDD18=1000,HDD18=0,GHI_kWh_m2=1000,diffuse_fraction=.5,RH=50,solar_seasonality=.1,latitude=1)
    bank=SimpleNamespace(cities=('A',),models={'A':Native(e)},feature_scale=np.ones(8),climates={'A':climate},
        protocols={'A':SimpleNamespace(energy=100.,clear_udi=.5)})
    target=SimpleNamespace(energy=1000.,clear_udi=.8,threshold=.72)
    base=TransferSurrogate(bank,target,climate)
    replay=pd.DataFrame(x,columns=FEATURES);replay['UDI']=y[:,0];replay['TotalEnergy']=y[:,1]
    replay['id']=range(len(x));replay['split']='train_fit'
    records=[]
    for i in range(16):
        records.append({'query_index':i+1,'structure':f'Ag_{i+1}','energyplus_verified':True,
            'adaptation_role':'fit' if i<12 else 'validation',
            'optical_properties':dict(zip(FEATURES,x[i].tolist())),
            'udi':float(y[i,0]/.5*.8+.05*(1 if x[i,0]>.5 else -1)),
            'energy':float(y[i,1]*10+50*x[i,1])})
    return base,records,{'A':replay}


class TreeTests(unittest.TestCase):
    def test_frozen_and_tree_predictors_use_same_response_mapping(self):
        base,records,_=fixture()
        same=TreeTransferSurrogate(base.bank,base.protocol,base.target_climate)
        for r in records:
            left=base.predict(r['optical_properties']);right=same.predict(r['optical_properties'])
            self.assertEqual(left['UDI'],right['UDI']);self.assertEqual(left['TotalEnergy'],right['TotalEnergy'])

    def test_label_coordinates(self):
        base,records,_=fixture();x,y=observed_arrays(records[:12]);e=base.bank.models['A'].ensemble
        z=virtual_source_labels(y,base.bank.protocols['A'],base.protocol,e)
        reconstructed=(z*e.y_scale+e.y_mean)/np.array([.5,100.])*np.array([.8,1000.])
        np.testing.assert_allclose(reconstructed,y)

    def test_real_training_prefix_and_source_immutability_and_roundtrip(self):
        base,records,replay=fixture();originals=base.bank.models['A'].ensemble.models[0]
        old=[b.save_raw() for b in originals]
        candidate=TreeTransferSurrogate.fit(base,records[:12],replay,parameters={'rounds':4,'reg_lambda':1.,'min_child_weight':1,'replay_rows':16})
        for b in candidate.bank.models['A'].ensemble.models[0]:self.assertEqual(b.num_boosted_rounds(),12)
        self.assertEqual([b.save_raw() for b in originals],old)
        self.assertTrue(any(d['new_tree_prediction_delta_max']>0 for d in candidate.training_metadata['source_details']['A'][0]['models']))
        with tempfile.TemporaryDirectory() as directory:
            candidate.save(directory,[r['query_index'] for r in records[12:]])
            restored=TreeTransferSurrogate.load(directory,base)
            for row in records:
                a=candidate.predict(row['optical_properties']);b=restored.predict(row['optical_properties'])
                self.assertAlmostEqual(a['UDI'],b['UDI']);self.assertAlmostEqual(a['TotalEnergy'],b['TotalEnergy'])

    def test_target_validation_cannot_be_fitted(self):
        base,records,replay=fixture()
        with self.assertRaises(ValueError):TreeTransferSurrogate.fit(base,records,replay)

    def test_only_source_training_partition_can_be_replayed(self):
        base,records,replay=fixture();replay['A']['split']='test_family_ood'
        with self.assertRaises(ValueError):TreeTransferSurrogate.fit(base,records[:12],replay)

    def test_gate_rejects_fit_validation_structure_leakage(self):
        base,records,_=fixture()
        with self.assertRaises(ValueError):compare_candidate(base,base,records[:12],records[:4])

    def test_failed_target_is_not_a_label(self):
        base,records,replay=fixture();records[0]['energyplus_verified']=False
        with self.assertRaises(ValueError):TreeTransferSurrogate.fit(base,records[:12],replay)


if __name__=='__main__':unittest.main()
