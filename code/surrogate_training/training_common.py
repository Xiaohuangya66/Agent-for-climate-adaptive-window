"""Frozen protocol shared by preparation, fitting and independent evaluation."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ['T_lum_cold','T_lum_hot','T_sol_cold','T_sol_hot',
            'Rf_sol_cold','Rb_sol_cold','Rf_sol_hot','Rb_sol_hot']
TARGETS = ['UDI','TotalEnergy']
SEEDS = [11,17,23,29,37]
ENERGY_DEFINITION = 'Electricity:Facility + NaturalGas:Facility'
PROTOCOL = 'tmyx-four-source-grouped-ensemble-v2'
LEGACY_PROTOCOL = 'tmyx-seven-city-grouped-ensemble-v1'
SOURCE_CITIES = ('Singapore','LosAngeles','Nanjing','Beijing')
TARGET_CITIES = ('Shenzhen','Phoenix')
SPLITS = ['train_fit','calibration','test_family_ood']
MODEL_PARAMETERS = dict(n_estimators=300,learning_rate=.05,max_depth=6,
                        subsample=.9,colsample_bytree=.9,objective='reg:squarederror',
                        eval_metric='rmse',tree_method='hist',device='cpu',n_jobs=1)
# Operational review thresholds, fixed before any test predictions exist.
GATES = dict(udi_min_r2=.95,udi_max_mae_pp=2.,energy_min_r2=.90,
             energy_max_mae_pct_baseline=.5,coverage_min=.85,coverage_max=.95,
             udi_max_mean_interval_width_pp=10.,energy_max_mean_interval_width_pct_baseline=3.)


def family(structure: str) -> str:
    tokens=str(structure).split()
    if not tokens or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*_\d+(?:\.\d+)?',t) for t in tokens):
        raise ValueError(f'Malformed structure: {structure!r}')
    return ' '.join(token.rsplit('_',1)[0] for token in tokens)


def make_split(structures):
    import pandas as pd
    frame=pd.DataFrame({'structure':sorted(set(structures))})
    frame['material_family']=frame.structure.map(family)
    if frame.material_family.nunique()<10:raise ValueError('Too few independent material families')
    outer,test=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42).split(frame,groups=frame.material_family))
    remaining=frame.iloc[outer]
    fit,cal=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=31415).split(remaining,groups=remaining.material_family))
    frame['split']=''
    frame.loc[remaining.iloc[fit].index,'split']='train_fit'
    frame.loc[remaining.iloc[cal].index,'split']='calibration'
    frame.loc[frame.iloc[test].index,'split']='test_family_ood'
    validate_split(frame)
    return frame.sort_values('structure').reset_index(drop=True)


def validate_split(frame):
    if frame.structure.duplicated().any() or set(frame['split'])!=set(SPLITS):
        raise ValueError('Duplicate structures or missing split')
    if not frame.material_family.equals(frame.structure.map(family)):
        raise ValueError('Material family assignment changed')
    groups=[set(frame.loc[frame['split']==s,'material_family']) for s in SPLITS]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Material family leakage across splits')


def conformal_quantile(scores,alpha=.1):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Calibration scores must be nonempty and finite')
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    rank=min(int(np.ceil((len(values)+1)*(1-alpha))),len(values))
    return float(np.partition(values,rank-1)[rank-1]),rank


def calibrated_multipliers(y,members,alpha=.1):
    mean=members.mean(axis=0);std=members.std(axis=0,ddof=1)
    multipliers=[];descriptions={}
    for index,target in enumerate(TARGETS):
        scores=np.abs(y[:,index]-mean[:,index])/np.maximum(std[:,index],np.finfo(float).eps)
        multiplier,rank=conformal_quantile(scores,alpha)
        multipliers.append(multiplier)
        descriptions[target]=dict(alpha=alpha,nominal_coverage=1-alpha,calibration_rows=len(scores),
                                  finite_sample_rank_1_based=rank,multiplier=multiplier,
                                  zero_spread_rows=int((std[:,index]<=np.finfo(float).eps).sum()),
                                  score_definition='abs(y-ensemble_mean)/max(ensemble_std,eps)')
    return np.asarray(multipliers),descriptions


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,Path):return str(value)
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


@contextmanager
def exclusive_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        if os.name!='nt':
            import fcntl
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError(f'Another process owns {path}')
        yield


def read_config(run):
    config=json.loads((Path(run)/'run_config.json').read_text(encoding='utf-8'))
    if config['protocol'] not in {PROTOCOL,LEGACY_PROTOCOL}:raise ValueError('Unexpected training protocol')
    if config['protocol']==PROTOCOL and (set(config['cities'])!=set(SOURCE_CITIES) or
            set(config['held_out_climate_cities'])!=set(TARGET_CITIES)):
        raise ValueError('Wrong source/target scope for this release')
    return config


def predict_members(models,x,y_mean,y_scale):
    import xgboost as xgb
    matrix=xgb.DMatrix(np.asarray(x),nthread=1)
    predictions=[]
    for target_models in models:
        predictions.append(np.column_stack([booster.predict(matrix) for booster in target_models])*y_scale+y_mean)
    return np.stack(predictions)
