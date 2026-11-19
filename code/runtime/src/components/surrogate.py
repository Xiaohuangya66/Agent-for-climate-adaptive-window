"""Server dictionary interface over the verified local site-energy ensemble."""
from pathlib import Path
import json
import numpy as np
from src.vendor.thermo_agent.surrogate import SurrogateEnsemble


class Surrogate:
    def __init__(self,model_path):
        self.model_path=Path(model_path)
        # Preserve old caller paths while routing them explicitly to formal assets.
        if self.model_path.suffix=='.pkl':
            city=self.model_path.stem.removeprefix('surrogate_model_')
            self.model_path=self.model_path.parent/'formal'/city
        self.ensemble=SurrogateEnsemble(self.model_path)
        compatibility=json.loads((self.model_path/'inference_compatibility.json').read_text())
        if compatibility.get('objective')!='reg:squarederror':
            raise ValueError('Intercept compatibility is only audited for single-target squared-error trees')
        # XGBoost 3.0 can load newer trees while silently resetting a vector intercept.
        # Restore the ORIGINAL scalar intercept, exported under the training runtime.
        for description,boosters in zip(self.ensemble.metadata['members'],self.ensemble.models):
            for filename,booster in zip(description['models'],boosters):
                score=float(compatibility['scalar_base_scores'][filename])
                if not np.isfinite(score):raise ValueError('Nonfinite original model intercept')
                booster.set_param({'base_score':score})
        probes=json.loads((self.model_path/'inference_probes.json').read_text())
        predicted=self.ensemble.predict(np.asarray(probes['features']))
        for name,actual in [('mean',predicted.mean),('std',predicted.std),
                            ('half_width',predicted.conformal_half_width_90pct)]:
            expected=np.asarray(probes[name])
            if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=2e-7,atol=np.array([2e-6,2e5])):
                raise RuntimeError(f'Local/server surrogate numerical check failed: {name}')
        self.numerical_validation={'source_xgboost':compatibility['source_xgboost'],
            'n_probes':len(probes['features']),'max_abs_udi_difference':float(np.max(np.abs(predicted.mean[:,0]-np.asarray(probes['mean'])[:,0]))),
            'max_abs_energy_difference_J':float(np.max(np.abs(predicted.mean[:,1]-np.asarray(probes['mean'])[:,1])))}
        definition=self.ensemble.metadata.get('target_definition',{})
        if 'daylight' not in definition.get('UDI','').lower() or 'NaturalGas:Facility' not in definition.get('TotalEnergy',''):
            raise ValueError('Formal surrogate requires daylight-hours UDI and total site-energy labels')
        if tuple(self.ensemble.targets)!=('UDI','TotalEnergy'):
            raise ValueError('Unexpected surrogate target order')
        self.feature_names=list(self.ensemble.features)
        self.metadata=self.ensemble.metadata
        self.is_formal=True
        self._cache={}

    def predict(self,optical_features,*,strict=True):
        values=np.asarray([optical_features[k] for k in self.feature_names] if isinstance(optical_features,dict) else optical_features,dtype=float).reshape(-1)
        if values.shape!=(len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError('Surrogate expects one finite optical feature vector')
        key=tuple(values)
        if key not in self._cache:
            pred=self.ensemble.predict(values)
            raw_udi=float(pred.mean[0,0])
            bounded_udi=float(np.clip(raw_udi,0,1))
            half=float(pred.conformal_half_width_90pct[0,0])
            result={'UDI':bounded_udi,'UDI_unbounded':raw_udi,'UDI_was_clipped':bounded_udi!=raw_udi,
                    'UDI_lower_90pct':max(0.,bounded_udi-half),'UDI_upper_90pct':min(1.,bounded_udi+half),
                    'TotalEnergy':float(pred.mean[0,1]),
                    'UDI_std':float(pred.std[0,0]),'Energy_std':float(pred.std[0,1]),
                    'UDI_half_width_90pct':float(pred.conformal_half_width_90pct[0,0]),
                    'Energy_half_width_90pct':float(pred.conformal_half_width_90pct[0,1]),
                    'ood_distance':float(pred.ood_distance[0]),'udi_definition':'daylight_hours',
                    'energy_definition':'Electricity:Facility + NaturalGas:Facility'}
            if not all(np.isfinite(v) for v in result.values() if isinstance(v,float)) or result['TotalEnergy']<=0:
                raise ValueError('Invalid formal surrogate output')
            self._cache[key]=result
        return dict(self._cache[key])

    def predict_uncertainty(self,optical_features):
        return self.predict(optical_features)['UDI_half_width_90pct']
