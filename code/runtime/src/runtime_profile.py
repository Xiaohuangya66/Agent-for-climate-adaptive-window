"""Bind optimization to one audited weather/data/IDF/model/baseline generation."""
from functools import lru_cache
import csv
import json
from pathlib import Path
from src.support_settings import settings, asset_path

PROJECT_ROOT=asset_path('project_root','/public/home/nju_zhanghuajie/topic3_agent_v2')
CAMPAIGN_ROOT=asset_path('campaign_root',PROJECT_ROOT/'climate_campaigns/tmyx_2011_2025_20261021')
MODEL_RUN=asset_path('model_run',CAMPAIGN_ROOT/'surrogate_runs/grouped_20261026_1040')
BASELINE_TABLE=asset_path('baseline_table',CAMPAIGN_ROOT/'protocol/city_baselines_goal2.csv')
PROFILE_VERSION='four_source_two_target_v2_20261106'


@lru_cache(maxsize=1)
def registry():
    data=json.loads((CAMPAIGN_ROOT/'weather_registry.json').read_text(encoding='utf-8'))
    for city, record in settings().get('targets',{}).items():
        if city in data['source_cities']:
            raise ValueError('Target registration cannot override a trained source city')
        if record['weather_dataset']!=data['dataset'] or record['weather_period']!=data['period']:
            raise ValueError('Target weather generation mismatch')
        data['cities'][city]={**record,'role':'target'}
    return data


@lru_cache(maxsize=1)
def baseline_rows():
    with BASELINE_TABLE.open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=len({row['city'] for row in rows}):raise ValueError('Duplicate city baselines')
    if settings().get('bundle_relative_paths'):
        for row in rows:
            if row['city'] in settings().get('targets',{}) and not Path(row['epw']).is_absolute():
                row['epw']=str((PROJECT_ROOT/row['epw']).resolve())
    return {row['city']:row for row in rows}


@lru_cache(maxsize=7)
def city_assets(city):
    data=registry()
    if city in settings().get('targets',{}):
        record=settings()['targets'][city]
        if city not in baseline_rows():raise ValueError(f'Target clear-glass baseline missing: {city}')
        baseline=baseline_rows()[city]
        if any(baseline[k]!=record[k] for k in ['epw','weather_dataset','weather_period']):
            raise ValueError('Target baseline/weather mismatch')
        assets=dict(record,city=city,profile_version=PROFILE_VERSION,
                    support_mode='transfer',baseline_table=str(BASELINE_TABLE),
                    reference_baseline_processes=1,model_run=str(MODEL_RUN))
        for key in ['epw','idf','baseline_idf']:
            if not Path(assets[key]).is_file():raise FileNotFoundError(assets[key])
        return assets
    if city not in data['source_cities']:raise ValueError(f'No trained source-city runtime bundle: {city}')
    configuration=json.loads((CAMPAIGN_ROOT/'configs/project.json').read_text())
    assets=dict(profile_version=PROFILE_VERSION,city=city,weather_dataset=data['dataset'],weather_period=data['period'],
        optical_coherence_rule='SiO2_10000 phase-averaged; other material layers coherent',
        model_directory=str(MODEL_RUN/'models'/city),
        dataset=str(CAMPAIGN_ROOT/'results'/city/f'ML_Dataset_SiteEnergy_{city}.csv'),
        epw=str(CAMPAIGN_ROOT/data['cities'][city]['epw']),
        idf=str(CAMPAIGN_ROOT/configuration['cities'][city]['base_idf']),
        baseline_idf=str(CAMPAIGN_ROOT/configuration['cities'][city]['baseline_idf']),
        baseline_table=str(BASELINE_TABLE),model_run=str(MODEL_RUN))
    for key in ['dataset','epw','idf','baseline_idf','baseline_table']:
        if not Path(assets[key]).is_file():raise FileNotFoundError(assets[key])
    metadata=json.loads((Path(assets['model_directory'])/'metadata.json').read_text())
    accepted_protocols={'tmyx-four-source-grouped-ensemble-v2'}
    if metadata.get('city')!=city or metadata.get('training_protocol') not in accepted_protocols:
        raise ValueError('Wrong city or unreviewed model generation')
    if metadata.get('weather_period')!=data['period'] or metadata.get('weather_dataset')!=data['dataset']:
        raise ValueError('Model/weather generation mismatch')
    dataset_identity=(PROJECT_ROOT/metadata['source_dataset_relative']
                      if 'source_dataset_relative' in metadata else Path(metadata['source_dataset']))
    if dataset_identity.resolve()!=Path(assets['dataset']).resolve():
        raise ValueError('Model/prior data generation mismatch')
    baseline=baseline_rows()[city]
    if (baseline['epw']!=data['cities'][city]['epw'] or baseline['weather_period']!=data['period']
            or baseline['weather_dataset']!=data['dataset']):
        raise ValueError('Baseline/weather generation mismatch')
    acceptance=json.loads((MODEL_RUN/'acceptance_complete.json').read_text())
    if acceptance['cities'].get(city)!='PASS' or not acceptance['actual_server_adapter_verified']:
        raise ValueError('Model has not passed the recorded runtime acceptance')
    return assets
