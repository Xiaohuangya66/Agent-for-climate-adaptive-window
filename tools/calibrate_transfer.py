"""Select climate temperature and continuation strength using source LOCO only."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from bundle_env import runtime_environment

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(runtime_environment(ROOT));sys.path.insert(0,str(ROOT/'code/runtime'))
from src.runtime_profile import registry,MODEL_RUN,city_assets
from src.agent.formal_protocol import FormalProtocol,configure_udi_ratio
from src.tools.city_support import SourceCityBank,TransferSurrogate,FEATURES,climate_features,write_json
from src.tools.tree_transfer import TreeTransferSurrogate,replay_frames,compare_candidate,corrected_predictions


def diverse_panel(frame,n):
    x=frame[list(FEATURES)].to_numpy(float)
    x=(x-x.mean(axis=0))/np.maximum(x.std(axis=0),1e-6)
    indices=[int(np.argmax(np.linalg.norm(x,axis=1)))];distance=np.full(len(x),np.inf)
    while len(indices)<min(n,len(x)):
        distance=np.minimum(distance,np.linalg.norm(x-x[indices[-1]],axis=1));distance[indices]=-1
        indices.append(int(np.argmax(distance)))
    return frame.iloc[indices].copy()


def observations(frame):
    return [{'query_index':i+1,'structure':row.structure,
        'optical_properties':{k:float(row[k]) for k in FEATURES},'energyplus_verified':True,
        'udi':float(row.UDI),'energy':float(row.TotalEnergy),
        'adaptation_role':'validation' if (i+1)%4==0 else 'fit'}
        for i,(_,row) in enumerate(frame.iterrows())]


def score(model,records,fit=()):
    if fit:prediction,_=corrected_predictions(model,fit,records)
    else:prediction=np.array([[p['UDI'],p['TotalEnergy']] for p in
        [model.predict(r['optical_properties']) for r in records]])
    truth=np.array([[r['udi'],r['energy']] for r in records])
    error=np.abs(prediction-truth).mean(axis=0)/np.array([model.protocol.clear_udi,model.protocol.energy])
    return {'UDI_normalized_MAE':float(error[0]),'energy_normalized_MAE':float(error[1]),'score':float(error.sum())}


def main():
    configure_udi_ratio(.9)
    output=ROOT/'validation/source_LOCO';output.mkdir(parents=True,exist_ok=False)
    sources=list(registry()['source_cities']);folds={};temperature_rows=[]
    for heldout in sources:
        bank=SourceCityBank([c for c in sources if c!=heldout]);formal=FormalProtocol(heldout)
        climate=climate_features(city_assets(heldout)['epw'])
        # Shared material-family partitions ensure these target structures were
        # not used to train any source expert. Test labels are never opened here.
        frame=pd.read_csv(MODEL_RUN/'datasets'/f'{heldout}_calibration.csv.gz',
            usecols=['id','structure','split',*FEATURES,'UDI','TotalEnergy'])
        assert frame['split'].eq('calibration').all()
        panel=diverse_panel(frame,80)
        probes=observations(panel.iloc[:16]);evaluation=observations(panel.iloc[16:])
        assert len(probes)==16 and len(evaluation)==64
        folds[heldout]=(bank,formal,climate,probes,evaluation)
        for temperature in [.25,.5,1.,2.]:
            base=TransferSurrogate(bank,formal,climate,temperature)
            temperature_rows.append({'heldout_city':heldout,'temperature':temperature,
                'source_cities':'|'.join(bank.cities),'evaluation_partition':'calibration',
                'evaluation_rows':len(evaluation),**score(base,evaluation)})
        print('LOCO_CLIMATE_DONE',heldout,flush=True)
    temperatures=pd.DataFrame(temperature_rows)
    temperatures.to_csv(output/'climate_selection.csv',index=False)
    best_temperature=float(temperatures.groupby('temperature').score.mean().idxmin())
    configurations={'shallow12':{'rounds':12,'max_depth':2,'reg_lambda':10.},
                    'shallow24':{'rounds':24,'max_depth':2,'reg_lambda':10.}}
    tree_rows=[];details=[]
    for heldout,(bank,formal,climate,probes,evaluation) in folds.items():
        base=TransferSurrogate(bank,formal,climate,best_temperature)
        fit=[r for r in probes if r['adaptation_role']=='fit']
        validation=[r for r in probes if r['adaptation_role']=='validation']
        replay=replay_frames(bank)
        for name,parameters in configurations.items():
            candidate=TreeTransferSurrogate.fit(base,fit,replay,parameters=parameters,seed=5)
            gate=compare_candidate(base,candidate,fit,validation)
            chosen=candidate if gate['accepted'] else base
            tree_rows.append({'heldout_city':heldout,'configuration':name,'temperature':best_temperature,
                'source_cities':'|'.join(bank.cities),'fit_rows':len(fit),'gate_rows':len(validation),
                'evaluation_rows':len(evaluation),'candidate_published':gate['accepted'],**score(chosen,evaluation,fit)})
            details.append({'heldout_city':heldout,'configuration':name,'gate':gate,
                'probe_structures':[r['structure'] for r in probes],
                'evaluation_structures':[r['structure'] for r in evaluation]})
            print('LOCO_TREES_DONE',heldout,name,gate['accepted'],flush=True)
    trees=pd.DataFrame(tree_rows);trees.to_csv(output/'tree_selection.csv',index=False)
    selected=str(trees.groupby('configuration').score.mean().idxmin())
    write_json(output/'fold_details.json',details)
    selection={'status':'PASS','source_cities':sources,'heldout_folds':sources,
        'temperature':best_temperature,'tree_configuration':selected,'parameters':configurations[selected],
        'selection_partition':'source calibration only','test_labels_read':False,
        'target_city_labels_read':False,'target_cities_excluded':registry()['transfer_target_cities'],
        'scoring':'mean over four source LOCO folds of UDI MAE/clear UDI + energy MAE/clear energy',
        'probe_role_counts':{'fit':12,'validation':4},'evaluation_rows_per_fold':64,
        'status_meaning':'selection workflow passed; not a guarantee of target-city accuracy'}
    write_json(output/'selection.json',selection)
    cfg=json.loads((ROOT/'configs/runtime.json').read_text())
    cfg['transfer'].update(temperature=best_temperature,status='selected_on_four_source_LOCO_calibration_only')
    cfg['tree_adaptation'].update(parameters=configurations[selected],configuration=selected,
        status='selected_on_four_source_LOCO_calibration_only')
    write_json(ROOT/'configs/runtime.json',cfg)
    print('FOUR_SOURCE_LOCO_COMPLETE',json.dumps(selection),flush=True)


if __name__=='__main__':main()
