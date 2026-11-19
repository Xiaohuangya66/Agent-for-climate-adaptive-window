"""Prepare or run ideal low-band-transmittance scans with the real E+ parser."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from bundle_env import runtime_environment

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(runtime_environment(ROOT));sys.path[:0]=[str(ROOT/'code/runtime'),str(ROOT/'code/ideal_reference')]
from ideal_reference import states
from src.agent.formal_protocol import FormalProtocol,configure_udi_ratio
from src.runtime_profile import city_assets
from src.support_settings import settings
from src.components.idf_builder import build_idf_text
from src.components.energyplus_runner import EnergyPlusRunner
from src.vendor.thermo_agent.energyplus import parse_energyplus_outputs
from src.vendor.thermo_agent.optical import OpticalFeatureCalculator


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def meters(directory):
    path=Path(directory)/'eplusmtr.csv'
    with path.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1:raise ValueError(f'Expected annual RunPeriod meter row: {path}')
    result={}
    for key,value in rows[0].items():
        match=re.fullmatch(r'(.+) \[J\]\(RunPeriod\)',key.strip())
        if match:result[match[1]]=float(value)
    if not {'Electricity:Facility','NaturalGas:Facility'}<=set(result):raise ValueError(f'Missing facility meters: {path}')
    return result


def run_case(city,low,closure,root,execute):
    folder=root/city/f'Tlow_{low:.2f}'/closure;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result.get('status')=='success':return result
        raise RuntimeError(f'Inspect failed case before a new attempt: {folder}')
    protocol=FormalProtocol(city);assets=city_assets(city);spectra=states(closure,low)
    optical=OpticalFeatureCalculator(settings()['reference_assets']).calculate(SimpleNamespace(**spectra))
    text=build_idf_text(Path(assets['idf']).read_text(encoding='utf-8'),spectra['cold'],spectra['hot'])
    idf=folder/'modified.idf';idf.write_text(text,encoding='utf-8')
    curves=[{'wavelength_nm':wave,**{f'{name}_{field}':float(getattr(spectra[name],attribute)[i])
        for name in ['cold','hot'] for field,attribute in [('T','transmittance'),('Rf','front_reflectance'),('Rb','back_reflectance')]}}
        for i,wave in enumerate(spectra['cold'].wavelengths_nm)]
    pd.DataFrame(curves).to_csv(folder/'input_spectra.csv',index=False)
    result={'city':city,'low_band_transmittance':low,'closure':closure,
        'high_band_T_cold':1.,'high_band_T_hot':0.,'status':'prepared','directory':str(folder),
        'UDI_threshold':protocol.threshold,'clear_udi':protocol.clear_udi,
        'baseline_energy_J':protocol.energy,**optical,
        'dT_sol':optical['T_sol_cold']-optical['T_sol_hot']}
    command=[settings()['energyplus_exe'],'-w',assets['epw'],'-d',str(folder),'-r',str(idf)]
    write(folder/'input.json',{**result,'command':command,'protocol':protocol.describe(),
        'band_note':'Low band 300-800nm; high band 810-2500nm. Integrated T_sol is computed, not fixed at 1/0.'})
    if not execute:return result
    if any(folder.glob('eplusout.*')):raise RuntimeError(f'Partial simulation outputs require review: {folder}')
    with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
        environment=os.environ.copy();environment.pop('DMX_API_KEY',None)
        complete=subprocess.run(command,cwd=folder,stdout=stdout,stderr=stderr,timeout=1200,check=False,env=environment)
    parsed=parse_energyplus_outputs(folder,return_code=complete.returncode,udi_denominator_mode='daylight_hours')
    result.update(asdict(parsed))
    if parsed.status=='success':
        error=EnergyPlusRunner._validate_parsed_metrics(folder,parsed)
        if error:result.update(status='failed',error_message=str(error))
        else:
            baseline=meters(ROOT/'protocol/baseline_evidence'/city);actual=meters(folder)
            if not np.isclose(sum(baseline[k] for k in ['Electricity:Facility','NaturalGas:Facility']),protocol.energy,rtol=1e-9):
                raise ValueError('Baseline facility energy mismatch')
            breakdown=[{'meter':key,'baseline_J':baseline.get(key),'ideal_J':value,
                'saving_J':baseline[key]-value if key in baseline else None} for key,value in actual.items()]
            pd.DataFrame(breakdown).to_csv(folder/'annual_energy_breakdown.csv',index=False)
            result.update(energy_saving=(protocol.energy-parsed.total_energy_J)/protocol.energy,
                UDI_retention_fraction=parsed.udi/protocol.clear_udi,feasible_at_UDI90=parsed.udi>=protocol.threshold,
                electricity_saving_J=baseline['Electricity:Facility']-parsed.electricity_J,
                gas_saving_J=baseline['NaturalGas:Facility']-parsed.gas_J)
    write(folder/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--city',action='append')
    parser.add_argument('--low-transmittance',type=float,nargs='+')
    parser.add_argument('--closure',choices=['absorptive_remainder','near_reflective_remainder'],default='near_reflective_remainder')
    parser.add_argument('--output',type=Path,default=ROOT/'new_runs/ideal_scan_udi90')
    parser.add_argument('--run',action='store_true',help='Actually run EnergyPlus; default prepares inputs only')
    args=parser.parse_args();configure_udi_ratio(.9)
    plan=json.loads((ROOT/'configs/campaign.json').read_text())
    cities=args.city or plan['source_cities']+plan['target_cities']
    values=args.low_transmittance or plan['ideal_scan']['low_band_transmittances']
    if any(not np.isfinite(v) or not 0<=v<=1 for v in values):parser.error('Transmittances must be finite in [0,1]')
    if len({f'{v:.2f}' for v in values})!=len(values):parser.error('Values must have distinct two-decimal case labels')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);rows=[]
    for city in cities:
        for low in values:
            row=run_case(city,low,args.closure,root,args.run);rows.append(row)
            pd.DataFrame(rows).to_csv(root/'scan_results.csv',index=False,encoding='utf-8-sig')
            print('IDEAL_CASE',city,low,row['status'],flush=True)
    complete=pd.DataFrame(rows)
    if args.run:
        feasible=complete[(complete.status=='success')&complete.get('feasible_at_UDI90',pd.Series(False,index=complete.index)).fillna(False)]
        best=feasible.sort_values('energy_saving',ascending=False).groupby('city',sort=False).head(1)
        best.to_csv(root/'best_feasible_ideal_per_city.csv',index=False,encoding='utf-8-sig')
        if (complete.status!='success').any():raise RuntimeError('Some ideal cases failed; inspect scan_results.csv')


if __name__=='__main__':main()
