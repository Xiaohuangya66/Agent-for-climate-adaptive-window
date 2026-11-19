"""Save train-only anchors/priors, real EPW descriptors and transfer cluster ranks."""
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from bundle_env import runtime_environment

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(runtime_environment(ROOT));sys.path.insert(0,str(ROOT/'code/runtime'))
from src.agent.baseline_common import SearchSpace
from src.agent.formal_protocol import FormalProtocol,configure_udi_ratio
from src.runtime_profile import registry,MODEL_RUN,city_assets
from src.support_settings import settings
from src.tools.city_support import FEATURES,SourceCityBank,TransferSurrogate,TransferPriorProvider,climate_features,write_json
from src.tools.city_prior_learner import CityPriorLearner


def main():
    configure_udi_ratio(.9);space=SearchSpace(max_layers=20);samples=[];clusters=[];climates=[]
    sources=registry()['source_cities'];targets=registry()['transfer_target_cities']
    for city in sources:
        formal=FormalProtocol(city)
        frame=pd.read_csv(MODEL_RUN/'datasets'/f'{city}_train_fit.csv.gz')
        eligible=frame[(frame.UDI>=formal.threshold)&frame.structure.map(space.valid)].sort_values('TotalEnergy')
        if eligible.empty:raise ValueError(f'No feasible source training anchor for {city}')
        row=eligible.iloc[0]
        samples.append({'probe_id':f'anchor_{city}_train_fit','structure':row.structure,
            'source_reference':{'city':city,'panel':'train_fit','id':int(row.id)},
            'source_optical_features':{k:float(row[k]) for k in FEATURES}})
        prior=CityPriorLearner().learn_prior_clusters(city,n_clusters=3)
        if len(prior['clusters'])!=3:raise ValueError(f'{city}: insufficient support for three clusters')
        write_json(ROOT/f'protocol/priors/{city}.json',prior)
        clusters.extend({'source_city':city,**c} for c in prior['clusters'])
    write_json(ROOT/'protocol/source_anchors.json',{'schema_version':2,'source_cities':sources,
        'selection_partition':'train_fit only','samples':samples,'target_labels_used':False})
    write_json(ROOT/'protocol/source_prior_bank.json',{'source_cities':sources,'clusters':clusters,
        'n_clusters':len(clusters),'selection_partition':'train_fit only'})
    bank=SourceCityBank()
    for city in sources+targets:
        assets=city_assets(city);climate=climate_features(assets['epw'])
        climates.append({'city':city,'role':'source' if city in sources else 'target',**climate,
            'HDD18_CDD18_ratio':climate['HDD18']/climate['CDD18'] if climate['CDD18']>0 else None})
        if city in targets:
            base=TransferSurrogate(bank,FormalProtocol(city),climate,settings()['transfer']['temperature'])
            prior=TransferPriorProvider(bank,base,None).learn_prior_clusters(city)
            assert len(prior['clusters'])==12
            write_json(ROOT/f'protocol/priors/{city}_transferred.json',prior)
    pd.DataFrame(climates).to_csv(ROOT/'protocol/climate_features.csv',index=False,encoding='utf-8-sig')
    write_json(ROOT/'validation/protocol_ready.json',{'status':'PASS','source_cities':sources,'target_cities':targets,
        'source_cluster_count':len(clusters),'target_cluster_count':12,'source_models':40,
        'source_priors_and_anchors_train_only':True,'new_EnergyPlus_queries':0})
    print('SIX_CITY_PROTOCOL_READY',flush=True)


if __name__=='__main__':main()
