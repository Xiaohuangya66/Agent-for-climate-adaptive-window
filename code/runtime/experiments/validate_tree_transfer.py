"""Source LOCO: tune on calibration, then evaluate once on untouched OOD test."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agent.formal_protocol import configure_udi_ratio,FormalProtocol
from src.runtime_profile import MODEL_RUN,city_assets
from src.support_settings import settings
from src.tools.city_support import FEATURES,SourceCityBank,TransferSurrogate,CitySupportTool,TransferPriorProvider,write_json,climate_features
from src.runtime_profile import registry
SOURCES=tuple(registry()['source_cities'])
from src.tools.tree_transfer import TreeTransferSurrogate,compare_candidate,corrected_predictions,replay_frames
from src.tools.bias_corrector import BiasCorrector
from src.components.tmm import TMM
from src.agent.baseline_common import SearchSpace


CONFIGURATIONS={'shallow8':{'rounds':8,'max_depth':1,'reg_lambda':10.},
                'shallow24':{'rounds':24,'max_depth':2,'reg_lambda':10.},
                'shallow48':{'rounds':48,'max_depth':2,'reg_lambda':3.}}


def diverse(frame,count):
    x=frame[list(FEATURES)].to_numpy(float);scale=np.maximum(x.std(axis=0),1e-6)
    z=(x-x.mean(axis=0))/scale;chosen=[int(np.argmin(np.linalg.norm(z,axis=1)))];d=np.full(len(x),np.inf)
    while len(chosen)<min(count,len(x)):
        d=np.minimum(d,np.linalg.norm(z-z[chosen[-1]],axis=1));d[chosen]=-1
        chosen.append(int(np.argmax(d)))
    return chosen


def records(frame,offset,role):
    return [{'query_index':offset+i,'structure':r['structure'],'source_dataset_id':int(r['id']),
             'energyplus_verified':True,'adaptation_role':role,'udi':float(r['UDI']),'energy':float(r['TotalEnergy']),
             'optical_properties':{k:float(r[k]) for k in FEATURES}} for i,(_,r) in enumerate(frame.iterrows())]


def batch(model,frame,fit):
    x=frame[list(FEATURES)].to_numpy(float);p=np.zeros((len(x),2))
    for c in model.bank.cities:
        v=model.bank.models[c].ensemble.predict(x).mean
        if not getattr(model,'allow_unbounded_source_udi',False):v[:,0]=np.clip(v[:,0],0,1)
        p[:,0]+=model.weights['udi'][c]*v[:,0]/model.bank.protocols[c].clear_udi*model.protocol.clear_udi
        p[:,1]+=model.weights['energy'][c]*v[:,1]/model.bank.protocols[c].energy*model.protocol.energy
    p[:,0]=np.clip(p[:,0],0,1)
    _,bias=corrected_predictions(model,fit,fit)
    p+=np.array([bias.udi_bias,bias.energy_bias]);p[:,0]=np.clip(p[:,0],0,1)
    return p


def metrics(pred,frame,protocol):
    truth=frame[['UDI','TotalEnergy']].to_numpy(float);error=np.abs(pred-truth).mean(axis=0)
    normalized=error/np.array([protocol.clear_udi,protocol.energy])
    band=truth[:,0]>=.75*protocol.clear_udi
    band_error=np.abs(pred[band]-truth[band]).mean(axis=0) if band.any() else np.array([np.nan,np.nan])
    eligible=np.flatnonzero(pred[:,0]>=protocol.threshold)
    top=eligible[np.argsort(pred[eligible,1],kind='stable')[:12]]
    feasible_top=top[truth[top,0]>=protocol.threshold]
    best_saving=float((1-np.min(truth[feasible_top,1])/protocol.energy)*100) if len(feasible_top) else None
    return {'UDI_MAE':float(error[0]),'energy_MAE_kWh':float(error[1]/3.6e6),
            'saving_MAE_percentage_points':float(normalized[1]*100),'score':float(normalized.sum()),
            'active_band_rows':int(band.sum()),'active_band_UDI_MAE':float(band_error[0]),
            'active_band_saving_MAE_percentage_points':float(band_error[1]/protocol.energy*100),
            'false_feasible':int(((pred[:,0]>=protocol.threshold)&(truth[:,0]<protocol.threshold)).sum()),
            'top12_true_feasible_count':len(feasible_top),'top12_best_true_saving_pct':best_saving}


def check_agent_publication(base,support,fit,gate_rows):
    from src.agent.agent_full import AgentFull
    history=copy.deepcopy(fit+gate_rows)
    for row in history:row['prediction_snapshot']={'raw':base.predict(row['optical_properties'])}
    before=copy.deepcopy(history);logs=[]
    candidate=support.predictor
    proxy=SimpleNamespace(city_support=support,
        surrogate=base,bias_corrector=BiasCorrector(),formal=base.protocol,query_history=history,
        candidate_archive=[{'optical_properties':fit[0]['optical_properties'],'raw_prediction':base.predict(fit[0]['optical_properties']),
                            'surrogate_version':base.VERSION}],
        clusters=[{'center':np.array([fit[0]['optical_properties'][k] for k in FEATURES])}],iteration=1,budget_used=16,
        _json_line=lambda name,payload:logs.append(payload))
    proxy._refresh_prediction=lambda row:AgentFull._refresh_prediction(proxy,row)
    AgentFull._update_city_support(proxy)
    if candidate.VERSION==base.VERSION:
        if logs or proxy.surrogate is not base:raise ValueError('Rejected adapter changed the live predictor')
        return {'gate_rejection_preserves_live_model':True,'extra_EnergyPlus_processes':0}
    if proxy.candidate_archive[0]['surrogate_version']!=candidate.VERSION:raise ValueError('Stale candidate cache')
    if proxy.bias_corrector.n_observations!=len(fit):raise ValueError('Old or validation residuals entered rebuilt bias')
    if proxy.bias_corrector.n_udi_observations!=0:raise ValueError('Tree-fit residuals leaked into prequential UDI state')
    if before!=history:raise ValueError('Historical query predictions were modified')
    if not logs or proxy.clusters[0]['support_version']!=candidate.VERSION:raise ValueError('Publication not propagated')
    return {'candidate_refresh':True,'bias_rebuilt_fit_only':True,'historical_snapshots_immutable':True,
            'cluster_response_refreshed':True,'extra_EnergyPlus_processes':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-candidates',action='store_true',help='Re-evaluate saved fits without retraining')
    parser.add_argument('--fixed-tree-recipe',action='store_true',help='Validate the frozen 24-round tree recipe without retuning')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    configure_udi_ratio(.9);calibration=[];cache={}
    for target in SOURCES:
        print('LOCO_START',target,flush=True)
        bank=SourceCityBank([c for c in SOURCES if c!=target]);protocol=FormalProtocol(target)
        base=TransferSurrogate(bank,protocol,climate_features(city_assets(target)['epw']),temperature=.5)
        frames={s:pd.read_csv(MODEL_RUN/'datasets'/f'{target}_{s}.csv.gz') for s in ['train_fit','calibration','test_family_ood']}
        groups=[set(f['structure']) for f in frames.values()]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Partition leakage')
        fi=diverse(frames['train_fit'],12);vi=diverse(frames['calibration'],4)
        fit=records(frames['train_fit'].iloc[fi],1,'fit');gate_rows=records(frames['calibration'].iloc[vi],13,'validation')
        score_frame=frames['calibration'].drop(frames['calibration'].index[vi])
        baseline_metrics=metrics(batch(base,score_frame,fit),score_frame,protocol)
        cache[target]={'base':base,'fit':fit,'gate':gate_rows,'frames':frames,'candidates':{},'decisions':{}}
        replay=replay_frames(bank)
        configurations={'shallow24':CONFIGURATIONS['shallow24']} if args.fixed_tree_recipe else CONFIGURATIONS
        for name,parameters in configurations.items():
            print('FIT',target,name,flush=True)
            folder=args.output/target/name
            if args.reuse_candidates and (folder/'metadata.json').is_file():
                candidate=TreeTransferSurrogate.load(folder,base)
                if candidate.training_metadata['fit_structures']!=[r['structure'] for r in fit]:
                    raise ValueError('Saved candidate fit identity changed')
                saved=pd.read_csv(folder/'target_fit_rows.csv')
                actual=np.array([[r['udi'],r['energy']] for r in fit])
                if not np.allclose(saved[['UDI','TotalEnergy']].to_numpy(),actual,rtol=1e-12):
                    raise ValueError('Saved target fit labels changed')
            else:
                candidate=TreeTransferSurrogate.fit(base,fit,replay,parameters=parameters,seed=5)
                candidate.save(folder,[r['query_index'] for r in gate_rows])
            gate=compare_candidate(base,candidate,fit,gate_rows);write_json(folder/'validation_gate.json',gate)
            candidate_metrics=metrics(batch(candidate,score_frame,fit),score_frame,protocol)
            effective=candidate_metrics if gate['accepted'] else baseline_metrics
            calibration.append({'held_out_city':target,'configuration':name,'gate_accepted':gate['accepted'],
                **candidate_metrics,'baseline_score':baseline_metrics['score'],'effective_score':effective['score']})
            cache[target]['candidates'][name]=candidate;cache[target]['decisions'][name]=gate
    frame=pd.DataFrame(calibration);frame.to_csv(args.output/'source_calibration_selection.csv',index=False)
    ranking=frame.groupby('configuration')['effective_score'].mean().sort_values();chosen=str(ranking.index[0])
    configuration=dict(settings());configuration['tree_adaptation']={
        'enabled':True,'min_fit':12,'min_validation':4,'parameters':CONFIGURATIONS[chosen],
        'source_selection_file':str(args.output/'source_calibration_selection.csv'),
        'configuration':chosen,'maximum_base_model_updates_per_run':1}
    configuration['probe_count']=16
    import os
    write_json(Path(os.environ['CITY_SUPPORT_SETTINGS']),configuration);settings.cache_clear()
    results=[];publication_checks={}
    for target,entry in cache.items():
        base,fit,test=entry['base'],entry['fit'],entry['frames']['test_family_ood']
        candidate=entry['candidates'][chosen];gate=entry['decisions'][chosen]
        for mode,model in [('fixed_experts_plus_bias',base),('continued_trees_plus_bias',candidate),
                           ('published_or_fallback',candidate if gate['accepted'] else base)]:
            results.append({'held_out_city':target,'method':mode,'configuration':chosen,
                'target_fit_labels':12,'target_validation_labels':4,'test_rows':len(test),
                'gate_accepted':gate['accepted'],**metrics(batch(model,test,fit),test,base.protocol)})
        # Exercise the actual tool fit/publish path, not a mocked publication.
        support=CitySupportTool(target,city_assets(target),base.protocol,TMM(Path('.')))
        support.bank=base.bank;support.predictor=base
        support.prior_provider=TransferPriorProvider(base.bank,base,support.tmm)
        support.bind(args.output/'real_tool_integration'/target)
        for row in fit+entry['gate']:support.observe(row)
        decision=support.fit_or_keep(SimpleNamespace(seed=5))
        if decision.get('accepted')!=gate['accepted']:raise ValueError('Tool gate and offline gate disagree')
        publication_checks[target]=check_agent_publication(base,support,fit,entry['gate'])
        restored=TreeTransferSurrogate.load(args.output/target/chosen,base)
        for row in fit+entry['gate']:
            left=candidate.predict(row['optical_properties']);right=restored.predict(row['optical_properties'])
            if not np.allclose([left['UDI'],left['TotalEnergy']],[right['UDI'],right['TotalEnergy']],rtol=1e-7):
                raise ValueError('Real-source model roundtrip changed predictions')
    pd.DataFrame(results).to_csv(args.output/'independent_test_results.csv',index=False)
    support=CitySupportTool('Shenzhen',city_assets('Shenzhen'),FormalProtocol('Shenzhen'),TMM(Path('.'))).prepare()
    support.bind(args.output/'shenzhen_preflight')
    state=SimpleNamespace(space=SearchSpace(max_layers=20),budget_total=40,budget_used=0,queried=set(),seed=5,formal=support.formal)
    plan=support.propose_probes(state,16)
    if len(plan['samples'])!=16:raise ValueError('Not enough distinct valid target probes')
    if sum(r['adaptation_role']=='validation' for r in plan['samples'])!=4:raise ValueError('Probe split incorrect')
    # Existing five Shenzhen records are insufficient for 12+4; do not claim fit.
    v1=Path(configuration['v1_workspace']);old=json.loads(next((v1/'smoke/method_runs').glob('*/summary.json')).read_text())
    for row in old['query_history']:support.observe(row)
    decision=support.fit_or_keep(SimpleNamespace(seed=5))
    if decision['decision']!='keep_insufficient_target_labels':raise ValueError('Tiny smoke was misrepresented as target retraining')
    write_json(args.output/'acceptance.json',{'status':'PASS','selected_configuration':chosen,
        'source_LOCO_configuration_scores':ranking.to_dict(),'target_fit_labels':12,'target_validation_labels':4,
        'source_models_retained_and_not_overwritten':True,'genuine_added_trees':True,
        'agent_publication_checks':publication_checks,'shenzhen_probe_count':16,'shenzhen_validation_count':4,
        'Shenzhen_5_sample_decision':decision,'new_target_EnergyPlus_queries':0,
        'uncertainty':'heuristic, target coverage not calibrated','maximum_base_model_updates_per_run':1})
    print('TREE_TRANSFER_ACCEPTANCE_PASS',chosen,flush=True)


if __name__=='__main__':main()
