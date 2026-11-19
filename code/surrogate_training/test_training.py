"""Focused tests for leakage, calibration ranks, units and input guards."""
import unittest
import numpy as np
import pandas as pd
from training_common import make_split,validate_split,conformal_quantile,calibrated_multipliers,FEATURES,GATES
from train_seven_city_surrogates import check_frame
from accept_models import metrics


class ProtocolTests(unittest.TestCase):
    def structures(self):
        return [f'TiO2_{20+j} '+('Ag_10 '*i)+'SiO2_10000' for i in range(1,31) for j in range(4)]
    def test_thickness_variants_never_cross_splits(self):
        frame=make_split(self.structures());validate_split(frame)
        self.assertEqual(frame.groupby('material_family')['split'].nunique().max(),1)
    def test_split_does_not_depend_on_city_or_csv_order(self):
        a=make_split(self.structures());b=make_split(self.structures()[::-1])
        pd.testing.assert_frame_equal(a,b)
    def test_corrupted_family_split_rejected(self):
        frame=make_split(self.structures());frame.loc[0,'split']='test_family_ood' if frame.loc[0,'split']!='test_family_ood' else 'train_fit'
        with self.assertRaises(ValueError):validate_split(frame)
    def test_finite_sample_rank_not_interpolated_percentile(self):
        q,rank=conformal_quantile(np.arange(1,11),.1)
        self.assertEqual(rank,10);self.assertEqual(q,10)
    def test_nonfinite_calibration_is_rejected(self):
        with self.assertRaises(ValueError):conformal_quantile([1,np.nan])
    def test_calibration_respects_target_units(self):
        y=np.array([[1.,1.e9],[2.,2.e9],[3.,3.e9]])
        members=np.stack([y*.8,y*.9,y*1.1,y*1.2,y*1.3])
        q,_=calibrated_multipliers(y,members)
        self.assertAlmostEqual(q[0],q[1])
    def test_udi_and_energy_reporting_units(self):
        y=np.array([.2,.4,.8]);mean=y+.01;std=np.full(3,.01);half=np.full(3,.02)
        item=metrics(y,mean,std,half,np.array(['a','b','c']),'UDI',1.e12)
        self.assertAlmostEqual(item['mae_pp'],1)
        item=metrics(y*1.e12,mean*1.e12,std*1.e12,half*1.e12,np.array(['a','b','c']),'TotalEnergy',1.e12)
        self.assertAlmostEqual(item['mae_pct_baseline'],1)
    def frame(self):
        return pd.DataFrame([dict(structure='WVO2_30 SiO2_10000',UDI=.5,TotalEnergy=15.,ElectricityFacility_J=10.,
            NaturalGasFacility_J=5.,UDI_qualifying_hours=2,UDI_denominator_hours=4,**{f:.3 for f in FEATURES})])
    def test_missing_gas_not_silently_treated_as_zero(self):
        frame=self.frame();frame.loc[0,'NaturalGasFacility_J']=np.nan
        with self.assertRaises(ValueError):check_frame(frame,'fixture')
    def test_wrong_udi_denominator_is_rejected(self):
        frame=self.frame();frame.loc[0,'UDI_denominator_hours']=5
        with self.assertRaises(ValueError):check_frame(frame,'fixture')
    def test_released_wvo2_constraint_is_not_reintroduced(self):
        frame=self.frame();frame.loc[0,'structure']='TiO2_30 SiO2_10000'
        check_frame(frame,'fixture')


if __name__=='__main__':unittest.main(verbosity=2)
