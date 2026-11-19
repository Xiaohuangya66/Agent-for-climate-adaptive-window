"""Portable input/scaler crosscheck; never refit, recalibrate or activate models."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
from contextlib import closing
import tarfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from training_common import FEATURES,TARGETS,SPLITS,read_config,write_json
from source_prepare import audit_database,BUNDLE

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True,type=Path);args=p.parse_args()
run=args.run_dir.resolve();config=read_config(run);manifest=pd.read_csv(run/'grouped_split.csv');checks=[]
for city in config['cities']:
    path=Path(config['source_audit'][city]['source_csv'])
    source=pd.read_csv(path,usecols=['structure','structure_hash',*FEATURES,*TARGETS])
    audit=audit_database(city,path,len(source));sql_checked=False
    if audit['mode']=='read_only_sqlite':
        with closing(sqlite3.connect(Path(audit['path']).as_uri()+'?mode=ro',uri=True)) as connection:
            labels=pd.read_sql_query("SELECT structure_hash,UDI,TotalEnergy FROM site_energy_labels WHERE status='success'",connection)
        merged=source.merge(labels,on='structure_hash',suffixes=('_csv','_db'),validate='one_to_one')
        if len(merged)!=len(source):raise ValueError('Database/CSV identities differ')
        for target in TARGETS:
            if not np.allclose(merged[target+'_csv'],merged[target+'_db'],rtol=1e-12,atol=1e-12):raise ValueError('Database/CSV labels differ')
        sql_checked=True
    partitions={s:pd.read_csv(run/'datasets'/f'{city}_{s}.csv.gz') for s in SPLITS}
    combined=pd.concat(partitions.values()).set_index('structure').loc[manifest.structure]
    selected=source.set_index('structure').loc[manifest.structure]
    if not np.allclose(combined[[*FEATURES,*TARGETS]],selected[[*FEATURES,*TARGETS]],rtol=1e-12,atol=1e-12):raise ValueError('Frozen labels differ from raw CSV')
    metadata=json.loads((run/'models'/city/'metadata.json').read_text())
    for key,columns in [('x_scaler',FEATURES),('y_scaler',TARGETS)]:
        scaler=StandardScaler().fit(partitions['train_fit'][columns].to_numpy(float))
        if not np.allclose(metadata[key]['mean'],scaler.mean_,rtol=1e-12,atol=1e-12) or not np.allclose(metadata[key]['scale'],scaler.scale_,rtol=1e-12,atol=1e-12):raise ValueError('Scaler leakage or corruption')
    campaign=json.loads((path.parent/'campaign_config.json').read_text())
    identity=BUNDLE/f'protocol/baseline_identity/{city}.json'
    if identity.exists():baseline_inputs=json.loads(identity.read_text(encoding='utf-8'))
    else:
        with tarfile.open(BUNDLE/f'raw/baselines/{city}/annual_outputs.tar.gz') as archive:
            members=[m for m in archive.getmembers() if m.name.endswith('baseline_inputs.json')]
            if len(members)!=1:raise FileNotFoundError(f'Baseline identity metadata missing: {identity}')
            baseline_inputs=json.load(archive.extractfile(members[0]))
    if campaign['epw_sha256'][city]!=baseline_inputs['weather']:raise ValueError('Baseline/dataset weather identities differ')
    checks.append({'city':city,'frozen_inputs_match_source':True,'both_scalers_train_only':True,
        'baseline_and_dataset_weather_identical':True,'database_audit':audit,'fresh_SQLite_labels_compared':sql_checked})
write_json(run/'evaluation/input_crosscheck.json',{'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),
    'checks':checks,'new_fits':0,'new_calibration_parameter_updates':0,'new_energyplus_queries':0})
print(json.dumps({'input_checks':'PASS','cities':len(checks),'new_energyplus_queries':0}))
