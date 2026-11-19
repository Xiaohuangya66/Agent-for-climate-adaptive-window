"""IDF preparation, strict metric checks, and compact evidence storage."""
from dataclasses import asdict
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import io
import json
import math
import csv
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile

import os
_CODE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FOUR_CITY_CAMPAIGN_ROOT", _CODE_DIR)).resolve()
sys.path.insert(0,str(_CODE_DIR/'vendor'))
from thermo_agent.energyplus import GOAL2_AUDIT_OUTPUT_REQUESTS, glazing_spectral_block


def registry():
    return json.loads((ROOT/'weather_registry.json').read_text(encoding='utf-8'))


@contextmanager
def export_lock(directory):
    """Serialize CSV snapshots independently of the simulation writer lock."""
    with (Path(directory)/'dataset_export.lock').open('a') as stream:
        if sys.platform!='win32':
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def atomic_frame_csv(frame,destination):
    destination=Path(destination)
    with tempfile.NamedTemporaryFile('w',dir=destination.parent,delete=False,
                                     encoding='utf-8-sig',newline='') as stream:
        temporary=Path(stream.name)
        try:frame.to_csv(stream,index=False)
        except BaseException:
            stream.close();temporary.unlink(missing_ok=True)
            raise
    try:temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)


def idf_objects(text):
    end = 0
    for match in re.finditer(r'(?:[^!;]|![^\n]*(?:\n|$))*;',text):
        raw = match.group()
        fields = [x.strip() for x in re.sub(r'![^\n]*','',raw[:-1]).split(',')]
        end = match.end()
        if fields and fields[0]:yield raw,fields
    if re.sub(r'![^\n]*','',text[end:]).strip():
        raise ValueError('Unterminated IDF object')


def build_idf_text(base,cold=None,hot=None):
    output=[];seen=set();variable_requests=set()
    for raw,fields in idf_objects(base):
        key=tuple(x.casefold() for x in fields)
        if key[0]=='materialproperty:glazingspectraldata' and key[1] in ['vo2_cold_spec','vo2_hot_spec']:
            continue
        if key[0] in ['output:meter','output:variable']:
            if key in seen:continue
            seen.add(key)
            if key[0]=='output:variable' and len(key)>=4:
                variable_requests.add((key[2],key[3]))
        output.append(raw)
    required=list(idf_objects(GOAL2_AUDIT_OUTPUT_REQUESTS))
    required+=list(idf_objects('Output:Variable,*,Daylighting Reference Point 1 Illuminance,Hourly;'))
    for raw,fields in required:
        key=tuple(x.casefold() for x in fields)
        if key in seen:continue
        if key[0]=='output:variable' and (key[2],key[3]) in variable_requests:continue
        output.append('\n'+raw);seen.add(key)
        if key[0]=='output:variable':variable_requests.add((key[2],key[3]))
    if cold is not None or hot is not None:
        if cold is None or hot is None:raise ValueError('Both spectral states are required')
        output += ['\n'+glazing_spectral_block('VO2_Cold_Spec',cold),
                   '\n'+glazing_spectral_block('VO2_Hot_Spec',hot)]
    return ''.join(output)+'\n'


def localize_idf(base,city):
    """Use the same station's EPW location and two DDY sizing conditions."""
    weather=registry()['cities'][city]
    epw=ROOT/weather['epw']
    header=next(csv.reader([epw.read_text(encoding='utf-8-sig').splitlines()[0]]))
    ddy=epw.with_suffix('.ddy')
    candidates=list(idf_objects(ddy.read_text(encoding='utf-8-sig')))
    selections=[]
    for condition in ['ann htg 99.6% condns db','ann clg .4% condns db=>mwb']:
        matches=[(raw,fields) for raw,fields in candidates
                 if fields[0].casefold()=='sizingperiod:designday'
                 and condition in fields[1].casefold()]
        if len(matches)!=1:raise ValueError(f'{city}: expected one DDY condition {condition}')
        selections.append(matches[0])
    location='\nSite:Location,'+','.join([header[1],*header[6:10]])+';\n'
    output=[raw for raw,fields in idf_objects(base)
            if fields[0].casefold()!='site:location'
            and not fields[0].casefold().startswith('sizingperiod:')]
    output += [location,*['\n'+raw for raw,_ in selections]]
    details={'station':header[5],'latitude':float(header[6]),'longitude':float(header[7]),
             'ddy':str(ddy.relative_to(ROOT)),
             'heating_design_day':selections[0][1][1],
             'cooling_design_day':selections[1][1][1],
             'heating_drybulb_C':float(selections[0][1][5]),
             'cooling_drybulb_C':float(selections[1][1][5])}
    return ''.join(output)+'\n',details


def validate_metrics(directory,metrics):
    """Additional server-audited guards over the unchanged reference parser."""
    if metrics.status!='success':return
    if metrics.return_code!=0 or not metrics.success_marker or metrics.fatal_detected:
        raise ValueError('Missing clean EnergyPlus completion')
    if metrics.udi is None or not math.isfinite(metrics.udi) or not 0<=metrics.udi<=1:
        raise ValueError('Invalid real UDI')
    if metrics.udi_denominator_hours is None or metrics.udi_denominator_hours<=0:
        raise ValueError('No valid daylight-hour denominator')
    if not math.isclose(metrics.udi,metrics.udi_qualifying_hours/metrics.udi_denominator_hours,abs_tol=1e-12):
        raise ValueError('UDI/hour mismatch')
    import pandas as pd
    frame=pd.read_csv(Path(directory)/'eplusout.csv')
    for meter in ['Electricity:Facility','NaturalGas:Facility']:
        cols=[c for c in frame if meter in c and 'RunPeriod' in c]
        if len(cols)!=1:raise ValueError(f'Ambiguous/missing annual {meter}')
        values=pd.to_numeric(frame[cols[0]],errors='coerce').dropna().to_numpy()
        if len(values)!=1 or not all(math.isfinite(v) and v>=0 for v in values):
            raise ValueError(f'Invalid annual {meter}')
    if not math.isclose(metrics.total_energy_J,metrics.electricity_J+metrics.gas_J,rel_tol=1e-12):
        raise ValueError('Electricity/gas sum mismatch')


def store_evidence(database,result,work_root,keep_success_runs=False):
    """Commit one compressed archive in SQLite before removing owned scratch data."""
    if result.get('status')!='success':return
    directory=Path(result['run_directory']).resolve()
    root=Path(work_root).resolve()
    if directory==root or not directory.is_relative_to(root):
        raise ValueError('Refusing to clean a directory outside this campaign scratch root')
    buffer=io.BytesIO()
    required=['input.idf','campaign_result.json','eplusout.csv','eplusout.err','eplusout.end',
              'eplusout.eio','process.stdout.log','process.stderr.log']
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in required:
            path=directory/name
            if path.is_file():archive.write(path,arcname=name)
    with closing(sqlite3.connect(database,timeout=60)) as connection, connection:
        connection.execute('CREATE TABLE IF NOT EXISTS run_evidence '
                           '(task_key TEXT PRIMARY KEY, format TEXT NOT NULL, archive BLOB NOT NULL)')
        connection.execute('INSERT OR REPLACE INTO run_evidence VALUES (?,?,?)',
                           (result['task_key'],'zip',sqlite3.Binary(buffer.getvalue())))
    if not keep_success_runs:shutil.rmtree(directory)


def extract_evidence(database,task_key,destination):
    with closing(sqlite3.connect(database)) as connection:
        row=connection.execute('SELECT archive FROM run_evidence WHERE task_key=?',(task_key,)).fetchone()
    if not row:raise KeyError(task_key)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(row[0])) as archive:
        if any(not (destination/name).resolve().is_relative_to(destination) for name in archive.namelist()):
            raise ValueError('Invalid archive path')
        archive.extractall(destination)


def save_baseline_row(city,metrics,directory,run_directory=None):
    data=registry();weather=data['cities'][city]
    row={'city':city,'UDI':metrics.udi,'UDI_threshold_95pct':.95*metrics.udi,
         'TotalEnergy_J':metrics.total_energy_J,'TotalEnergy_kWh':metrics.total_energy_kWh,
         'formal_udi_definition':'daylight_hours',
         'energy_definition':'Electricity:Facility + NaturalGas:Facility',
         'epw':weather['epw'],'weather_dataset':data['dataset'],'weather_period':data['period'],
         'baseline_run_directory':str(Path(run_directory or directory).resolve()),
         'completed_utc':datetime.now(timezone.utc).isoformat()}
    destination=Path(directory)/'baseline_row.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
    temporary.replace(destination)
    return row
