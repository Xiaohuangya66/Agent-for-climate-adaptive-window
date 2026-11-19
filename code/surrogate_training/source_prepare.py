"""Four-source preparation; Shenzhen and Phoenix remain transfer targets."""
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from training_common import (FEATURES,TARGETS,SPLITS,SEEDS,MODEL_PARAMETERS,GATES,ENERGY_DEFINITION,
    PROTOCOL,LEGACY_PROTOCOL,SOURCE_CITIES,TARGET_CITIES,make_split,validate_split,write_json)

BUNDLE=Path(__file__).resolve().parents[2]


def audit_database(city,path,rows,bundle=BUNDLE):
    receipt_path=bundle/'provenance/source_label_audit.json'
    if receipt_path.exists() and Path(path).resolve()==(bundle/'data/source_campaign/results'/city/f'ML_Dataset_SiteEnergy_{city}.csv').resolve():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))[city]
        labels=pd.read_csv(bundle/receipt['verified_labels']).set_index('structure_hash').sort_index()
        current=pd.read_csv(path,usecols=['structure_hash','UDI','TotalEnergy']).set_index('structure_hash').sort_index()
        if (not receipt['SQLite_labels_compared'] or rows!=receipt['rows'] or
            not current.index.equals(labels.index) or not np.allclose(current,labels,rtol=1e-12,atol=1e-12)):
            raise ValueError(f'{city}: labels differ from read-only SQLite audit')
        return {'mode':'verified_SQLite_label_snapshot','status_counts':receipt['status_counts'],
                'original_database':receipt['database'],'snapshot_labels_rechecked':True}
    database=path.parent/'site_energy_labels.sqlite3'
    if database.exists():
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as connection:
            counts=dict(connection.execute('SELECT status,COUNT(*) FROM site_energy_labels GROUP BY status'))
            cities={r[0] for r in connection.execute('SELECT DISTINCT city FROM site_energy_labels')}
        if cities!={city} or counts.get('success',0)!=rows:raise ValueError('CSV/database city or count mismatch')
        return {'mode':'read_only_sqlite','path':str(database),'status_counts':counts}
    catalog_path=bundle/'provenance/archive_catalog.json'
    if not catalog_path.is_file():raise FileNotFoundError(f'No database or archived backup receipt for {city}')
    receipt=json.loads(catalog_path.read_text(encoding='utf-8'))[f'database_{city}']
    archive=bundle/receipt['path']
    if (receipt['city_counts']!={city:rows} or receipt['status_counts'].get('success')!=rows or
        receipt['sqlite_quick_check']!='ok' or not archive.is_file() or archive.stat().st_size!=receipt['bytes']):
        raise ValueError(f'{city}: archived database receipt mismatch')
    return {'mode':'archived_sqlite_backup_receipt','archive':str(archive),
            'status_counts':receipt['status_counts'],'fresh_SQLite_label_comparison':False}


def prepare_source(campaign,run,*,check_frame,baseline_table=None,split_mode='frozen',split_manifest=None):
    campaign=Path(campaign).resolve();run=Path(run).resolve()
    if run.exists():raise FileExistsError(f'Use a new output directory: {run}')
    weather=json.loads((campaign/'weather_registry.json').read_text(encoding='utf-8'))
    cities=weather['source_cities'];targets=weather.get('transfer_target_cities',[])
    if len(cities)!=len(set(cities)) or set(cities)!=set(SOURCE_CITIES) or set(targets)!=set(TARGET_CITIES):
        raise ValueError(f'Require sources {SOURCE_CITIES} and held-out targets {TARGET_CITIES}')
    if set(cities)&set(targets):raise ValueError('Target city cannot enter source training')
    if baseline_table is None:
        local=campaign/'protocol/city_baselines_goal2.csv'
        baseline_table=local if local.exists() else BUNDLE/'protocol/city_baselines_goal2.csv'
    baselines=pd.read_csv(baseline_table)
    if baselines.city.duplicated().any() or not set(cities)<=set(baselines.city) or not set(baselines.city)<=set(cities+targets):
        raise ValueError('Require one baseline per source and only declared campaign cities')
    baselines=baselines[baselines.city.isin(cities)].copy()
    if set(baselines.energy_definition)!={ENERGY_DEFINITION} or set(baselines.formal_udi_definition)!={'daylight_hours'}:
        raise ValueError('Baseline objective definition mismatch')
    columns=['id','structure','T_cold','T_hot',*FEATURES,*TARGETS,'ElectricityFacility_J','NaturalGasFacility_J','UDI_qualifying_hours','UDI_denominator_hours']
    frames={};audits={};common=None
    for city in cities:
        baseline=baselines.set_index('city').loc[city]
        if baseline.weather_dataset!='TMYx' or baseline.weather_period!=weather['period'] or baseline.epw!=weather['cities'][city]['epw']:
            raise ValueError(f'{city}: baseline weather mismatch')
        if not np.isfinite([baseline.UDI,baseline.TotalEnergy_J]).all() or not 0<baseline.UDI<=1 or baseline.TotalEnergy_J<=0:
            raise ValueError('Invalid baseline metrics')
        if not np.isclose(baseline.UDI_threshold_95pct,.95*baseline.UDI,atol=1e-12):raise ValueError('Corrupt historical baseline check column')
        path=campaign/'results'/city/f'ML_Dataset_SiteEnergy_{city}.csv'
        frame=pd.read_csv(path,usecols=columns);check_frame(frame,city)
        database_audit=audit_database(city,path,len(frame))
        frames[city]=frame.set_index('structure',drop=False)
        common=set(frame.structure) if common is None else common&set(frame.structure)
        audits[city]={'source_csv':str(path),'source_rows':len(frame),'database_audit':database_audit,
                      'failed_label_rows':database_audit['status_counts'].get('failed',0)}
    if len(common)<1000:raise ValueError('Insufficient common source structures')
    if split_mode=='frozen':
        split_manifest=Path(split_manifest or BUNDLE/'protocol/grouped_split_reference.csv').resolve()
        manifest=pd.read_csv(split_manifest);validate_split(manifest)
        if manifest.structure.duplicated().any() or not set(manifest.structure)<=common:
            raise ValueError('Frozen manifest contains missing or duplicate structures')
    elif split_mode=='recompute':
        manifest=make_split(common);split_manifest=None
    else:raise ValueError('split_mode must be frozen or recompute')
    selected_set=set(manifest.structure);ordered=manifest.structure.tolist()
    reference=frames[cities[0]].loc[ordered,FEATURES].to_numpy(float)
    for city in cities:
        if not np.allclose(reference,frames[city].loc[ordered,FEATURES].to_numpy(float),rtol=1e-10,atol=1e-12):
            raise ValueError(f'{city}: inconsistent optical features')
        if split_mode=='frozen' and split_manifest.parent.name=='source_surrogates':
            saved=pd.concat([pd.read_csv(split_manifest.parent/'datasets'/f'{city}_{split}.csv.gz') for split in SPLITS])
            saved=saved.set_index('structure').loc[ordered,[*FEATURES,*TARGETS]]
            if not np.allclose(saved,frames[city].loc[ordered,[*FEATURES,*TARGETS]],rtol=1e-12,atol=1e-12):
                raise ValueError(f'{city}: raw CSV differs from frozen labeled partitions')
            audits[city]['frozen_labels_crosschecked']=True
    run.mkdir(parents=True)
    for name in ['datasets','models','evaluation']:(run/name).mkdir()
    manifest.to_csv(run/'grouped_split.csv',index=False,encoding='utf-8-sig')
    baselines.to_csv(run/'city_baselines.csv',index=False,encoding='utf-8-sig')
    exclusions=[]
    for city in cities:
        frame=frames[city];selected=frame.loc[ordered].copy()
        selected['material_family']=manifest.material_family.to_numpy();selected['split']=manifest['split'].to_numpy()
        for split in SPLITS:selected[selected['split']==split].to_csv(run/'datasets'/f'{city}_{split}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        for structure in sorted(set(frame.index)-selected_set):
            exclusions.append({'city':city,'structure':structure,'reason':'outside_selected_frozen_panel' if split_mode=='frozen' else 'not_common_to_all_sources'})
        audits[city].update(common_rows=len(selected_set),excluded_successful_rows=len(frame)-len(selected_set))
    pd.DataFrame(exclusions,columns=['city','structure','reason']).to_csv(run/'excluded_successful_rows.csv',index=False)
    counts={split:{'rows':int((manifest['split']==split).sum()),'families':int(manifest[manifest['split']==split].material_family.nunique())} for split in SPLITS}
    if any(v['rows']<100 or v['families']<2 for v in counts.values()):raise ValueError('Insufficient partition support')
    config={'protocol':PROTOCOL,'origin_training_protocol':LEGACY_PROTOCOL if split_mode=='frozen' else None,
        'release_scope':'four-source-two-target','created_utc':datetime.now(timezone.utc).isoformat(),
        'run_directory':str(run),'campaign_root':str(campaign),'cities':cities,'held_out_climate_cities':targets,
        'weather_dataset':weather['dataset'],'weather_period':weather['period'],'features':FEATURES,'targets':TARGETS,
        'member_seeds':SEEDS,'model_parameters':MODEL_PARAMETERS,'conformal_alpha':.1,'operating_udi_ratio':.90,
        'outer_test_group_fraction':.20,'calibration_fraction_of_remaining_groups':.15,
        'split_seeds':{'test':42,'calibration':31415},'split_counts':counts,'split_mode':split_mode,
        'split_reference':str(split_manifest) if split_manifest else None,'available_common_structures':len(common),
        'common_structures':len(manifest),'material_families':int(manifest.material_family.nunique()),
        'acceptance_gates':GATES,'source_audit':audits,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgb.__version__}}
    write_json(run/'run_config.json',config);write_json(run/'weather_registry.json',weather)
    print('PREPARED',json.dumps({'protocol':PROTOCOL,'cities':cities,'target':targets,'panel':len(manifest),'split_counts':counts}),flush=True)
    return config
