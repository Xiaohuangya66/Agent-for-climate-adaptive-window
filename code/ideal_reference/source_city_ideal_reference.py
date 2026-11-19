"""Isolated ideal-spectrum comparison; only spectral objects differ from Agent IDFs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def setup():
    manifest = read(ROOT/'manifest.json')
    sys.path.insert(0, str(Path(manifest['agent_campaign'])/'source'))
    return manifest


def states(closure):
    import numpy as np
    from src.vendor.thermo_agent.spectra import SpectralState, validate_spectral_state
    wavelength = np.arange(300., 2501., 10.)
    cold = np.where(wavelength<=800, .5, 1.)
    hot = np.where(wavelength<=800, .5, 0.)
    assert closure in ['absorptive_remainder', 'near_reflective_remainder']
    factor = 0. if closure=='absorptive_remainder' else .999
    result = {}
    for name, transmittance in [('cold',cold),('hot',hot)]:
        reflectance = factor*(1-transmittance)
        state = SpectralState(wavelength,transmittance,reflectance,reflectance.copy())
        validation = validate_spectral_state(state)
        assert validation.valid, validation.errors
        assert np.all(state.transmittance+state.front_reflectance<=1)
        result[name] = state
    return result


def annual_meters(directory):
    with (Path(directory)/'eplusmtr.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['Date/Time'].strip()=='simdays=365'
    values = {}
    for header, value in rows[0].items():
        if header=='Date/Time':
            continue
        match = re.fullmatch(r'(.+) \[J\]\(RunPeriod\)',header.strip())
        assert match and value.strip()
        values[match[1]] = float(value)
    native, mapping, in_data = {}, {}, False
    for line in (Path(directory)/'eplusout.mtr').read_text().splitlines():
        if line=='End of Data Dictionary':
            in_data=True
            continue
        if not in_data:
            match=re.match(r'(\d+),\d+,([^\[]+) \[J\] !RunPeriod',line)
            if match:
                mapping[match[1]]=match[2]
        else:
            fields=line.split(',')
            if fields[0] in mapping:
                assert mapping[fields[0]] not in native
                native[mapping[fields[0]]]=float(fields[1])
    assert set(values)==set(native)
    assert all(math.isclose(v,native[k],rel_tol=1e-12,abs_tol=.001) for k,v in values.items())
    total=values['Electricity:Facility']+values['NaturalGas:Facility']
    end_uses={k:v for k,v in values.items() if not k.endswith(':Facility')}
    residual=total-sum(end_uses.values())
    assert abs(residual)<total*1e-9
    return dict(total_J=total,total_kWh=total/3.6e6,
        electricity_kWh=values['Electricity:Facility']/3.6e6,
        gas_kWh=values['NaturalGas:Facility']/3.6e6,
        end_use_kWh={k:v/3.6e6 for k,v in end_uses.items()},
        residual_kWh=residual/3.6e6,source_meter=str(Path(directory)/'eplusmtr.csv'))


def non_spectral(text):
    from src.components.idf_builder import idf_objects
    return [fields for _,fields in idf_objects(text)
            if not (fields[0].casefold()=='materialproperty:glazingspectraldata'
                    and fields[1].casefold() in ['vo2_cold_spec','vo2_hot_spec'])]


def parse_spectra(text):
    import numpy as np
    from src.components.idf_builder import idf_objects
    from src.vendor.thermo_agent.spectra import SpectralState,validate_spectral_state
    spectra={}
    for _,fields in idf_objects(text):
        if fields[0].casefold()!='materialproperty:glazingspectraldata' or fields[1].casefold() not in ['vo2_cold_spec','vo2_hot_spec']:
            continue
        key='cold' if 'cold' in fields[1].casefold() else 'hot'
        assert key not in spectra
        matrix=np.asarray([float(v) for v in fields[2:]]).reshape((-1,4))
        state=SpectralState(matrix[:,0]*1000,matrix[:,1],matrix[:,2],matrix[:,3])
        assert validate_spectral_state(state,tolerance=1e-5).valid
        assert len(matrix)==221 and np.allclose(matrix[:,0]*1000,np.arange(300,2501,10))
        spectra[key]=state
    assert set(spectra)=={'cold','hot'}
    return spectra


def prepare():
    manifest=setup()
    from src.components.idf_builder import build_idf_text,idf_objects
    from src.vendor.thermo_agent.energyplus import parse_energyplus_outputs
    from src.vendor.thermo_agent.optical import OpticalFeatureCalculator
    calculator=OpticalFeatureCalculator(Path(manifest['agent_campaign'])/'source/reference_assets')
    with Path(manifest['baseline_table']).open(encoding='utf-8-sig') as stream:
        baseline_rows={r['city']:r for r in csv.DictReader(stream)}
    baselines,agents,tasks={},[],[]
    for agent in manifest['agent_peaks']:
        city=agent['city']
        bundle=manifest['runtime_assets'][city]
        directory=Path(agent['source_result']).parent
        inputs=read(directory/'input.json')
        assert inputs['epw']==bundle['epw'] and inputs['base_idf']==bundle['idf']
        assert Path(inputs['epw']).is_file() and Path(inputs['energyplus_executable']).is_file()
        assert inputs['structure']==agent['structure'] and agent['structure'].split()[-1]=='SiO2_10000'
        actual=parse_energyplus_outputs(directory,return_code=0,udi_denominator_mode='daylight_hours')
        assert actual.status=='success' and actual.severe_errors==0
        assert math.isclose(actual.udi,agent['udi'],rel_tol=1e-12)
        assert math.isclose(actual.total_energy_J,agent['energy_J'],rel_tol=1e-12)
        meters=annual_meters(directory)
        assert math.isclose(meters['total_J'],actual.total_energy_J,rel_tol=1e-12)
        row=baseline_rows[city]
        baseline=annual_meters(Path(row['baseline_run_directory']))
        assert math.isclose(baseline['total_J'],float(row['TotalEnergy_J']),rel_tol=1e-12)
        base_udi=float(row['UDI'])
        threshold=float(row['UDI_threshold_95pct'])
        assert math.isclose(threshold,.95*base_udi,rel_tol=1e-12)
        assert row['epw']==str(Path(bundle['epw']).relative_to(Path(manifest['baseline_table']).parents[1]))
        baseline.update(udi=base_udi,udi_threshold=threshold)
        baselines[city]=baseline
        text=Path(agent['source_idf']).read_text()
        properties=calculator.calculate(SimpleNamespace(**parse_spectra(text)))
        assert max(abs(v-agent['optical_features'][k]) for k,v in properties.items())<1e-10
        saving=100*(baseline['total_J']-meters['total_J'])/baseline['total_J']
        assert math.isclose(saving,agent['saving_pct'],rel_tol=1e-10)
        assert actual.udi>=threshold
        agents.append(dict(id=agent['id'],city=city,seed=agent['seed'],structure=agent['structure'],
            source_idf=agent['source_idf'],source_result=agent['source_result'],udi=actual.udi,
            udi_retained_pct=100*actual.udi/base_udi,saving_pct=saving,**meters,optical_features=properties,
            electricity_saving_kWh=baseline['electricity_kWh']-meters['electricity_kWh'],
            gas_saving_kWh=baseline['gas_kWh']-meters['gas_kWh'],
            total_saving_kWh=baseline['total_kWh']-meters['total_kWh']))
        for closure in manifest['closures']:
            output=ROOT/'runs'/city/closure
            output.mkdir(parents=True,exist_ok=False)
            spec=states(closure)
            modified=build_idf_text(text,spec['cold'],spec['hot'])
            assert non_spectral(text)==non_spectral(modified),city
            parsed=parse_spectra(modified)
            import numpy as np
            for name in spec:
                for attr in ['transmittance','front_reflectance','back_reflectance']:
                    assert np.array_equal(getattr(spec[name],attr),getattr(parsed[name],attr))
            (output/'modified.idf').write_text(modified,encoding='utf-8')
            task=dict(index=len(tasks),city=city,closure=closure,run_directory=str(output),
                source_agent_id=agent['id'],reference_agent_idf=agent['source_idf'],
                epw=inputs['epw'],energyplus_executable=inputs['energyplus_executable'],
                identical_non_spectral_idf=True,optical_features=calculator.calculate(SimpleNamespace(**parsed)),
                optical_closure='Rf=Rb=0' if closure=='absorptive_remainder' else 'Rf=Rb=0.999*(1-T)',
                thermochromic_materials=[fields for _,fields in idf_objects(modified)
                    if 'thermochromic' in fields[0].casefold()],
                runperiod=[fields for _,fields in idf_objects(modified) if fields[0].casefold()=='runperiod'])
            write(output/'input.json',task)
            tasks.append(task)
    manifest.update(tasks=tasks,verified_agent_peaks=agents,verified_baselines=baselines,
        prepared_utc=datetime.now(timezone.utc).isoformat(),non_spectral_idf_equality=True)
    write(ROOT/'manifest.json',manifest)
    print(json.dumps(dict(tasks=len(tasks),cities=[a['city'] for a in agents],non_spectral_idf_equality=True,
        ideal_optical_features={t['closure']:t['optical_features'] for t in tasks[:2]}),ensure_ascii=False,indent=2))


def worker(index):
    manifest=setup()
    task=manifest['tasks'][index]
    directory=Path(task['run_directory'])
    if (directory/'process.json').exists():
        raise RuntimeError(f'Already attempted, refuse overwrite: {directory}')
    from src.vendor.thermo_agent.energyplus import parse_energyplus_outputs
    command=[task['energyplus_executable'],'-w',task['epw'],'-d',str(directory),'-r',str(directory/'modified.idf')]
    process=dict(task_index=index,city=task['city'],closure=task['closure'],command=command,
        started_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),return_code=None)
    write(directory/'process.json',process)
    print('ENERGYPLUS_STARTED',task['city'],task['closure'],flush=True)
    return_code=None
    try:
        environment=os.environ.copy()
        environment['OMP_NUM_THREADS']='1'
        with (directory/'stdout.log').open('wb') as stdout,(directory/'stderr.log').open('wb') as stderr:
            completed=subprocess.run(command,cwd=directory,stdout=stdout,stderr=stderr,env=environment,timeout=1200,check=False)
        return_code=completed.returncode
        metrics=parse_energyplus_outputs(directory,return_code=return_code,udi_denominator_mode='daylight_hours')
        result=asdict(metrics)
        if metrics.status=='success':
            assert metrics.severe_errors==0 and metrics.success_marker and not metrics.fatal_detected
            assert metrics.udi is not None and metrics.udi_denominator_hours>0
            meters=annual_meters(directory)
            assert math.isclose(meters['total_J'],metrics.total_energy_J,rel_tol=1e-12)
            baseline=manifest['verified_baselines'][task['city']]
            result.update(city=task['city'],closure=task['closure'],**meters,
                saving_pct=100*(baseline['total_J']-meters['total_J'])/baseline['total_J'],
                udi_retained_pct=100*metrics.udi/baseline['udi'],udi_feasible=metrics.udi>=baseline['udi_threshold'],
                electricity_saving_kWh=baseline['electricity_kWh']-meters['electricity_kWh'],
                gas_saving_kWh=baseline['gas_kWh']-meters['gas_kWh'],
                total_saving_kWh=baseline['total_kWh']-meters['total_kWh'],
                optical_features=task['optical_features'],run_directory=str(directory))
        write(directory/'result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if metrics.status!='success':
            raise RuntimeError(f"{metrics.error_code}: {metrics.error_message}")
    except Exception as exc:
        if not (directory/'result.json').exists():
            write(directory/'result.json',dict(status='failed',error=str(exc),city=task['city'],closure=task['closure']))
        raise
    finally:
        process.update(return_code=return_code,completed_utc=datetime.now(timezone.utc).isoformat())
        write(directory/'process.json',process)


def status():
    manifest=read(ROOT/'manifest.json')
    records=[]
    for task in manifest['tasks']:
        directory=Path(task['run_directory'])
        path=directory/'result.json'
        result=read(path) if path.exists() else {}
        records.append(dict(city=task['city'],closure=task['closure'],status=result.get('status','running' if (directory/'process.json').exists() else 'pending'),
            saving_pct=result.get('saving_pct'),udi_retained_pct=result.get('udi_retained_pct'),udi_feasible=result.get('udi_feasible')))
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        worker(args.worker)
    elif args.status:
        status()
