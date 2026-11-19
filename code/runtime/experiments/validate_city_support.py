"""Source-only LOCO selection, held-out checks, and no-simulation target preflight."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agent.formal_protocol import configure_udi_ratio,FormalProtocol
from src.runtime_profile import MODEL_RUN,city_assets
from src.support_settings import settings
from src.tools.city_support import SourceCityBank,CitySupportTool,FEATURES,climate_features,climate_weights,write_json
from src.components.tmm import TMM
from src.agent.baseline_common import SearchSpace


def diversified_indices(x,count):
    scale=np.maximum(x.std(axis=0),1e-6);z=(x-x.mean(axis=0))/scale
    chosen=[int(np.argmin(np.linalg.norm(z,axis=1)))]
    distance=np.full(len(x),np.inf)
    while len(chosen)<min(count,len(x)):
        distance=np.minimum(distance,np.linalg.norm(z-z[chosen[-1]],axis=1));distance[chosen]=-1
        chosen.append(int(np.argmax(distance)))
    return chosen


def evaluate(bank,target,frame,temperature,equal=False):
    experts=[c for c in bank.cities if c!=target]
    weights,distances=climate_weights(bank.climates[target],{c:bank.climates[c] for c in experts},temperature,equal)
    x=frame[list(FEATURES)].to_numpy(float)
    responses=[];widths=[]
    for c in experts:
        predicted=bank.models[c].ensemble.predict(x)
        baseline=np.array([bank.protocols[c].clear_udi,bank.protocols[c].energy])
        mean=predicted.mean.copy();mean[:,0]=np.clip(mean[:,0],0,1)
        responses.append(mean/baseline);widths.append(predicted.conformal_half_width_90pct/baseline)
    responses=np.stack(responses,axis=1);widths=np.stack(widths,axis=1)
    w=np.array([[weights['udi'][c],weights['energy'][c]] for c in experts])
    mean=(responses*w).sum(axis=1)
    half=np.sqrt((w*(widths**2+(responses-mean[:,None,:])**2)).sum(axis=1))
    half*=1+.25*max(min(d.values()) for d in distances.values())
    baseline=np.array([bank.protocols[target].clear_udi,bank.protocols[target].energy])
    mean[:,0]=np.clip(mean[:,0],0,1/baseline[0])
    truth=frame[['UDI','TotalEnergy']].to_numpy(float)/baseline
    return mean,half,truth,weights


def measures(predicted,truth,baseline):
    error=predicted-truth
    return {'UDI_MAE':float(np.abs(error[:,0]).mean()*baseline[0]),
            'energy_MAE_kWh':float(np.abs(error[:,1]).mean()*baseline[1]/3.6e6),
            'saving_MAE_percentage_points':float(np.abs(error[:,1]).mean()*100),
            'score_normalized_MAE':float(np.abs(error).mean(axis=0).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    configure_udi_ratio(.9);bank=SourceCityBank()
    options=[('equal',1.,True)]+[(f'climate_T{t}',t,False) for t in [.5,1.,2.]]
    calibration=[];frames={};predictions={}
    for city in bank.cities:
        frames[city]={split:pd.read_csv(MODEL_RUN/'datasets'/f'{city}_{split}.csv.gz')
                      for split in ['train_fit','calibration','test_family_ood']}
        groups=[set(df['structure']) for df in frames[city].values()]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Source fit/calibration/test structure leakage')
        baseline=np.array([bank.protocols[city].clear_udi,bank.protocols[city].energy])
        for name,t,equal in options:
            m,h,y,w=evaluate(bank,city,frames[city]['calibration'],t,equal)
            calibration.append({'held_out_city':city,'configuration':name,**measures(m,y,baseline)})
    selection=pd.DataFrame(calibration).groupby('configuration')['score_normalized_MAE'].mean().sort_values()
    chosen=str(selection.index[0]);_,temperature,equal=next(o for o in options if o[0]==chosen)
    pd.DataFrame(calibration).to_csv(args.output/'loco_calibration_selection.csv',index=False)
    # The held-out test labels never participate in configuration or probe selection.
    evaluation=[]
    for city in bank.cities:
        baseline=np.array([bank.protocols[city].clear_udi,bank.protocols[city].energy])
        train=frames[city]['train_fit'];test=frames[city]['test_family_ood']
        m,h,y,w=evaluate(bank,city,test,temperature,equal)
        tm,_,ty,_=evaluate(bank,city,train,temperature,equal)
        selected=diversified_indices(train[list(FEATURES)].to_numpy(float),16)
        for count in [0,4,8,16]:
            subset=selected[:count]
            bias=np.mean(ty[subset]-tm[subset],axis=0)*count/(count+3) if count else np.zeros(2)
            corrected=m+bias;corrected[:,0]=np.clip(corrected[:,0],0,1/baseline[0])
            evaluation.append({'held_out_city':city,'configuration':chosen,'n_probe_labels':count,
                'test_rows':len(test),'excluded_target_model':True,
                **measures(corrected,y,baseline),
                'raw_heuristic_UDI_coverage':float((np.abs(m[:,0]-y[:,0])<=h[:,0]).mean()),
                'coverage_caveat':'diagnostic only; target intervals uncalibrated',
                'probe_selection':'feature-only farthest point from train_fit; no test labels'})
    pd.DataFrame(evaluation).to_csv(args.output/'loco_test_results.csv',index=False)
    configuration=dict(settings());configuration['transfer']={'temperature':temperature,'equal_weights':equal,
        'status':'selected_on_source_LOCO_calibration_only','configuration':chosen,
        'selection_file':str(args.output/'loco_calibration_selection.csv')}
    import os
    write_json(Path(os.environ['CITY_SUPPORT_SETTINGS']),configuration)
    settings.cache_clear()
    support=CitySupportTool('Shenzhen',city_assets('Shenzhen'),FormalProtocol('Shenzhen'),TMM(Path('.'))).prepare()
    support.bind(args.output)
    from types import SimpleNamespace
    state=SimpleNamespace(space=SearchSpace(max_layers=20),budget_total=40,budget_used=0,
        queried=set(),seed=5,formal=support.formal)
    plan=support.propose_probes(state,8)
    if len(plan['samples'])!=8 or plan['n_preflight_rejected']:
        raise ValueError('Source anchors/probes did not pass unchanged structure/TMM preflight')
    if any(s['target_observation'] is not None for s in plan['samples']):raise ValueError('Source labels leaked into target probes')
    # Source paths keep their native model and exact measured prior semantics.
    from src.agent.base_agent import BaseAgent
    class ReadOnlySource(BaseAgent):
        def run(self):raise NotImplementedError
    native=ReadOnlySource('Singapore',seed=5,budget=0)
    if native.city_support is not None:raise ValueError('Source model accidentally replaced by transfer')
    x=np.array([plan['samples'][0]['optical_properties'][k] for k in FEATURES])
    left=native.surrogate.predict(x);right=bank.models['Singapore'].predict(x)
    if left!=right:raise ValueError('Native source prediction regression')
    write_json(args.output/'acceptance.json',{'status':'PASS','selected_configuration':chosen,
        'selection_scores':selection.to_dict(),'target_preflight_probes':len(plan['samples']),
        'n_source_clusters':len(support.prior_provider.clusters),'n_target_EnergyPlus_queries':0,
        'source_native_model_regression':'PASS','source_model_numerical_checks':{
            c:bank.models[c].numerical_validation for c in bank.cities},
        'UDI_ratio':.9,'require_WVO2':False,'append_missing_SiO2_10000':False,
        'residual_adapter':'keep fixed experts + existing online BiasCorrector',
        'intervals':'heuristic; coverage is not guaranteed for Shenzhen'})
    print('ACCEPTANCE_PASS',chosen,flush=True)


if __name__=='__main__':main()
