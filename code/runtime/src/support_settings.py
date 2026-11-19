"""Explicit, opt-in runtime bindings for an isolated CitySupport deployment."""
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def settings():
    filename = os.environ.get('CITY_SUPPORT_SETTINGS')
    if not filename:
        return {}
    path = Path(filename).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported CitySupport settings schema')
    if data.get('bundle_relative_paths'):
        root=path.parents[1]
        def resolve(value):
            candidate=Path(value).expanduser()
            return str((candidate if candidate.is_absolute() else root/candidate).resolve())
        for key in ['project_root','campaign_root','model_run','baseline_table','reference_assets',
                    'model_file','checkpoint','vocab_file','source_anchors','energyplus_exe']:
            if key in data:data[key]=resolve(data[key])
        for record in data.get('targets',{}).values():
            for key in ['epw','idf','baseline_idf']:record[key]=resolve(record[key])
        for section in ['transfer','tree_adaptation']:
            for key in ['selection_file','source_selection_file']:
                if key in data.get(section,{}):data[section][key]=resolve(data[section][key])
        if os.environ.get('ENERGYPLUS_EXE'):
            data['energyplus_exe']=str(Path(os.environ['ENERGYPLUS_EXE']).expanduser().resolve())
    return data


def asset_path(key, default):
    return Path(settings().get(key, default)).expanduser().resolve()
