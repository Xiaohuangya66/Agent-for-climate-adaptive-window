"""Create six-city inputs from existing audited TMYx assets; never simulate labels."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--reference-bundle',type=Path,required=True)
    args=parser.parse_args();old=args.campaign_root.resolve();bundle=args.reference_bundle.resolve()
    if (ROOT/'provenance/assets_ready.json').exists():raise FileExistsError('Assets already prepared')
    plan=load(ROOT/'configs/campaign.json');sources=plan['source_cities'];targets=plan['target_cities']
    weather=load(old/'weather_registry.json');project=load(old/'configs/project.json')
    old_cfg=load(bundle/'configs/runtime.json')
    rows={r['city']:r for r in csv.DictReader((old/'protocol/city_baselines_goal2.csv').open(encoding='utf-8-sig'))}
    bundle_rows={r['city']:r for r in csv.DictReader((bundle/'protocol/city_baselines_goal2.csv').open(encoding='utf-8-sig'))}
    campaign=ROOT/'data/source_campaign'
    registry={'schema_version':1,'dataset':'TMYx','period':'2011-2025',
              'source_cities':sources,'transfer_target_cities':targets,'cities':{}}
    new_project={'schema_version':1,'cities':{}}
    baseline_rows=[];target_assets={};audits={}
    for city in sources+targets:
        role='source' if city in sources else 'target'
        if city=='Shenzhen':
            record=old_cfg['targets'][city]
            epw=bundle/record['epw'];idf=bundle/record['idf'];clear=bundle/record['baseline_idf']
            meta={'station':'594930','dataset':'TMYx','period':'2011-2025'}
            baseline=dict(bundle_rows[city])
        else:
            meta=dict(weather['cities'][city]);entry=project['cities'][city]
            epw=old/meta['epw'];idf=old/entry['base_idf'];clear=old/entry['baseline_idf']
            baseline=dict(rows[city])
        epw_relative=f'inputs/weather/{epw.name}'
        idf_relative=f'inputs/buildings/{city}/Midrise_ReadyForLoop.idf'
        clear_relative=f'inputs/buildings/{city}/Midrise_ClearGlass3mm_Baseline.idf'
        copy(epw,ROOT/epw_relative);copy(idf,ROOT/idf_relative);copy(clear,ROOT/clear_relative)
        if epw.with_suffix('.ddy').is_file():copy(epw.with_suffix('.ddy'),(ROOT/epw_relative).with_suffix('.ddy'))
        registry['cities'][city]={**meta,'role':role,'epw':'../../'+epw_relative}
        new_project['cities'][city]={'role':role,'epw':'../../'+epw_relative,
            'base_idf':'../../'+idf_relative,'baseline_idf':'../../'+clear_relative}
        baseline['epw']='../../'+epw_relative if role=='source' else epw_relative
        baseline_rows.append(baseline)
        if role=='target':
            target_assets[city]={'epw':epw_relative,'idf':idf_relative,'baseline_idf':clear_relative,
                'weather_dataset':'TMYx','weather_period':'2011-2025'}
            if city=='Shenzhen' and 'sizing' in old_cfg['targets'][city]:
                target_assets[city]['sizing']=old_cfg['targets'][city]['sizing']
        original_run=Path(baseline['baseline_run_directory'])
        for filename in ['baseline_inputs.json','baseline_row.json','eplusout.end','eplusmtr.csv','eplustbl.csv']:
            destination=ROOT/'protocol/baseline_evidence'/city/filename
            if (original_run/filename).is_file():copy(original_run/filename,destination)
            else:
                archived=bundle/f'raw/baselines/{city}/annual_outputs.tar.gz'
                if archived.exists():
                    with tarfile.open(archived) as archive:
                        members=[m for m in archive if m.isfile() and Path(m.name).name==filename]
                        if len(members)==1:
                            destination.parent.mkdir(parents=True,exist_ok=True)
                            destination.write_bytes(archive.extractfile(members[0]).read())
        if role=='source':
            source=old/'results'/city/f'ML_Dataset_SiteEnergy_{city}.csv'
            destination=campaign/'results'/city/source.name
            copy(source,destination)
            frame=pd.read_csv(source,usecols=['structure_hash','UDI','TotalEnergy'])
            database=source.parent/'site_energy_labels.sqlite3'
            with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as conn:
                counts=dict(conn.execute('SELECT status,COUNT(*) FROM site_energy_labels GROUP BY status'))
                labels=pd.read_sql_query("SELECT structure_hash,UDI,TotalEnergy FROM site_energy_labels WHERE status='success'",conn)
            merged=frame.merge(labels,on='structure_hash',suffixes=('_csv','_db'),validate='one_to_one')
            if len(merged)!=len(frame) or len(labels)!=len(frame):raise ValueError(f'{city}: CSV/SQLite rows differ')
            for name in ['UDI','TotalEnergy']:
                if not np.allclose(merged[name+'_csv'],merged[name+'_db'],rtol=1e-12,atol=1e-12):
                    raise ValueError(f'{city}: CSV/SQLite labels differ')
            verified=f'provenance/verified_labels/{city}.csv.gz'
            (ROOT/verified).parent.mkdir(parents=True,exist_ok=True)
            labels.to_csv(ROOT/verified,index=False,compression={'method':'gzip','mtime':0})
            audits[city]={'source_csv':str(source),'database':str(database),'rows':len(frame),
                'status_counts':counts,'SQLite_labels_compared':True,'verified_labels':verified}
            print('SOURCE_LABELS_VERIFIED',city,len(frame),flush=True)
    write(campaign/'weather_registry.json',registry);write(campaign/'configs/project.json',new_project)
    table=ROOT/'protocol/city_baselines_goal2.csv';table.parent.mkdir(exist_ok=True)
    fields=list(dict.fromkeys(k for row in baseline_rows for k in row))
    with table.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(baseline_rows)
    write(ROOT/'provenance/source_label_audit.json',audits)
    for name in ['last.ckpt','vocab.txt']:copy(bundle/'models/transformer'/name,ROOT/'models/transformer'/name)
    copy(bundle/'code/transformer/model.py',ROOT/'code/transformer/model.py')
    shutil.copytree(bundle/'inputs/reference_assets',ROOT/'inputs/reference_assets',dirs_exist_ok=True)
    cfg=load(ROOT/'configs/runtime.json')
    cfg.update(project_root='.',campaign_root='data/source_campaign',model_run='models/source_surrogates',
        targets=target_assets,source_anchors='protocol/source_anchors.json',
        structure_constraints={'wvo2_count':0,'wvo2_max_count':2,'terminal_layer':None,'max_layers':20},
        require_transfer_selection=True)
    cfg['transfer'].update(status='pending_four_source_LOCO',selection_file='validation/source_LOCO/selection.json')
    cfg['tree_adaptation'].update(status='pending_four_source_LOCO',source_selection_file='validation/source_LOCO/selection.json')
    if Path('/public/home/nju_zhanghuajie/EnergyPlus23.1.0/energyplus').exists():
        cfg['energyplus_exe']='/public/home/nju_zhanghuajie/EnergyPlus23.1.0/energyplus'
    write(ROOT/'configs/runtime.json',cfg)
    write(ROOT/'provenance/assets_ready.json',{'status':'PASS','source_rows':{c:v['rows'] for c,v in audits.items()},
        'source_cities':sources,'target_cities':targets,'target_full_label_tables_copied':False,
        'new_EnergyPlus_calls':0,'reference_bundle':str(bundle),'original_campaign':str(old)})


if __name__=='__main__':main()
