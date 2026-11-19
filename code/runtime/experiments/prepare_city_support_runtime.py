"""Register target EPW/DDY/IDFs and explicitly run its clear-glass reference."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.components.idf_builder import idf_objects
from src.vendor.thermo_agent.energyplus import GOAL2_AUDIT_OUTPUT_REQUESTS,parse_energyplus_outputs
from src.tools.city_support import write_json,climate_features


def localize(base, epw):
    header=next(csv.reader([epw.read_text(encoding='utf-8-sig').splitlines()[0]]))
    candidates=list(idf_objects(epw.with_suffix('.ddy').read_text(encoding='utf-8-sig')))
    selections=[]
    for condition in ['ann htg 99.6% condns db','ann clg .4% condns db=>mwb']:
        matches=[(raw,f) for raw,f in candidates if f[0].casefold()=='sizingperiod:designday' and condition in f[1].casefold()]
        if len(matches)!=1:raise ValueError(f'Expected exactly one same-station design day: {condition}')
        selections.append(matches[0])
    output=[raw for raw,f in idf_objects(base) if f[0].casefold()!='site:location' and not f[0].casefold().startswith('sizingperiod:')]
    output+=['\nSite:Location,'+','.join([header[1],*header[6:10]])+';\n']
    output+=['\n'+raw for raw,_ in selections]
    keys={tuple(x.casefold() for x in f) for _,f in idf_objects(''.join(output))}
    requests=GOAL2_AUDIT_OUTPUT_REQUESTS+'\nOutput:Variable,*,Daylighting Reference Point 1 Illuminance,Hourly;'
    for raw,f in idf_objects(requests):
        if tuple(x.casefold() for x in f) not in keys:output.append('\n'+raw)
    return ''.join(output)+'\n',{'station':header[5],'location':header[1],
        'latitude':float(header[6]),'longitude':float(header[7]),
        'heating_design_day':selections[0][1][1],'cooling_design_day':selections[1][1][1]}


def prepare(args):
    project=Path(args.project_root).resolve();workspace=Path(args.workspace).resolve()
    campaign=project/'climate_campaigns/tmyx_2011_2025_20261021'
    assets=workspace/'assets';assets.mkdir(parents=True,exist_ok=True)
    weather=assets/'weather'
    epws=list(weather.glob('*Shenzhen*TMYx.2011-2025.epw'))
    if len(epws)!=1:raise ValueError('One explicit Shenzhen TMYx 2011-2025 EPW required')
    epw=epws[0].resolve();climate=climate_features(str(epw))
    inputs=assets/'Shenzhen';inputs.mkdir(exist_ok=True)
    details=None
    for name in ['Midrise_ReadyForLoop.idf','Midrise_ClearGlass3mm_Baseline.idf']:
        text,details=localize((campaign/'inputs'/name).read_text(encoding='utf-8'),epw)
        destination=inputs/name
        if destination.exists() and destination.read_text(encoding='utf-8')!=text:
            raise ValueError('Prepared IDF changed; use a fresh isolated workspace')
        destination.write_text(text,encoding='utf-8')
    exe=Path(args.energyplus or project.parent/'EnergyPlus23.1.0/energyplus').resolve()
    version=subprocess.check_output([str(exe),'--version'],text=True).strip()
    if '23.1' not in version:raise ValueError('Expected EnergyPlus 23.1')
    row=None;run=assets/'baseline_Shenzhen'
    marker=run/'baseline_row.json'
    if marker.exists():
        row=json.loads(marker.read_text())
        metrics=parse_energyplus_outputs(run,return_code=0,udi_denominator_mode='daylight_hours')
        if metrics.status!='success' or metrics.udi!=row['UDI'] or metrics.total_energy_J!=row['TotalEnergy_J']:
            raise ValueError('Existing baseline evidence does not match its row')
    elif args.run_baseline:
        run.mkdir(exist_ok=False)
        command=[str(exe),'-w',str(epw),'-d',str(run),'-r',str(inputs/'Midrise_ClearGlass3mm_Baseline.idf')]
        write_json(run/'input.json',{'command':command,'epw':str(epw),'city':'Shenzhen',
            'energyplus_version':version,'reference_process_count':1,'sizing':details})
        print('BASELINE_STARTED',str(run),flush=True)
        with (run/'stdout.log').open('wb') as out,(run/'stderr.log').open('wb') as err:
            completed=subprocess.run(command,cwd=run,stdout=out,stderr=err,timeout=3600,check=False)
        metrics=parse_energyplus_outputs(run,return_code=completed.returncode,udi_denominator_mode='daylight_hours')
        write_json(run/'metrics.json',asdict(metrics))
        if metrics.status!='success':raise RuntimeError(f'Baseline failed: {asdict(metrics)}')
        # Reuse the production guard, without loading target runtime config.
        import pandas as pd
        frame=pd.read_csv(run/'eplusout.csv')
        for meter in ['Electricity:Facility','NaturalGas:Facility']:
            columns=[c for c in frame if meter in c and 'RunPeriod' in c]
            if len(columns)!=1:raise ValueError('Missing/ambiguous baseline meter '+meter)
            v=pd.to_numeric(frame[columns[0]],errors='coerce').dropna()
            if len(v)!=1 or v.iloc[0]<0:raise ValueError('Invalid baseline annual meter')
        row={'city':'Shenzhen','UDI':metrics.udi,'UDI_threshold_95pct':.95*metrics.udi,
             'TotalEnergy_J':metrics.total_energy_J,'TotalEnergy_kWh':metrics.total_energy_kWh,
             'formal_udi_definition':'daylight_hours','energy_definition':'Electricity:Facility + NaturalGas:Facility',
             'epw':str(epw),'weather_dataset':'TMYx','weather_period':'2011-2025',
             'baseline_run_directory':str(run),'completed_utc':datetime.now(timezone.utc).isoformat()}
        write_json(marker,row)
    table=assets/'city_baselines_goal2.csv'
    with (campaign/'protocol/city_baselines_goal2.csv').open(encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    if row:rows.append(row)
    with table.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    target={'epw':str(epw),'idf':str(inputs/'Midrise_ReadyForLoop.idf'),
            'baseline_idf':str(inputs/'Midrise_ClearGlass3mm_Baseline.idf'),
            'weather_dataset':'TMYx','weather_period':'2011-2025','sizing':details}
    settings={'schema_version':1,'project_root':str(project),'campaign_root':str(campaign),
        'baseline_table':str(table),'reference_assets':str(project/'reference_assets'),
        'energyplus_exe':str(exe),'model_file':str(project.parent/'topic3/model.py'),
        'checkpoint':str(project.parent/'topic3/checkpoints/last.ckpt'),
        'vocab_file':str(project.parent/'topic3/dataset/vocab.txt'),
        'source_anchors':str(assets/'source_anchors.json'),'probe_count':8,
        'transfer':{'temperature':1.,'equal_weights':False,'status':'initial_heuristic_pending_LOCO'},
        'targets':{'Shenzhen':target}}
    write_json(workspace/'settings.json',settings)
    write_json(assets/'preparation.json',{'climate':climate,'sizing':details,'baseline':row,
        'ready_for_agent':row is not None,'energyplus_version':version,
        'source_protocol':'UDI90, no WVO2 hard constraint, no SiO2 append'})
    print('PREPARED',json.dumps({'ready':row is not None,'settings':str(workspace/'settings.json'),'baseline':row}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--workspace',required=True)
    parser.add_argument('--energyplus');parser.add_argument('--run-baseline',action='store_true')
    prepare(parser.parse_args())
