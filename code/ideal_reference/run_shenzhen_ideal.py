"""Rerun the established two ideal-spectrum closures on frozen Shenzhen assets."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
PROJECT=Path('/public/home/nju_zhanghuajie/topic3_agent_v2')
AGENT=PROJECT/'reviews/city_support_optimized_20261029'
NSGA=PROJECT/'reviews/shenzhen_nsga_control_20261029'
SOURCE=AGENT/'submission_batch/source'
os.environ['CITY_SUPPORT_SETTINGS']=str(AGENT/'submission_batch/settings.json')
sys.path.insert(0,str(SOURCE))
import ideal_reference as reference
from src.components.idf_builder import build_idf_text,idf_objects
from src.components.energyplus_runner import EnergyPlusRunner
from src.vendor.thermo_agent.energyplus import parse_energyplus_outputs
from src.vendor.thermo_agent.optical import OpticalFeatureCalculator


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def prepare():
    if (ROOT/'manifest.json').exists():raise RuntimeError('Manifest already exists; refuse overwrite')
    settings=read(AGENT/'submission_batch/settings.json')
    with Path(settings['baseline_table']).open(encoding='utf-8-sig') as stream:
        baseline_row=next(r for r in csv.DictReader(stream) if r['city']=='Shenzhen')
    baseline_dir=Path(baseline_row['baseline_run_directory'])
    baseline_metrics=parse_energyplus_outputs(baseline_dir,return_code=0,udi_denominator_mode='daylight_hours')
    assert baseline_metrics.status=='success'
    baseline=reference.annual_meters(baseline_dir)
    baseline.update(udi=float(baseline_row['UDI']),udi_threshold=.9*float(baseline_row['UDI']),
        udi_ratio=.9,baseline_run_directory=str(baseline_dir),baseline_table=settings['baseline_table'])
    assert math.isclose(baseline_metrics.total_energy_J,baseline['total_J'],rel_tol=1e-12)
    assert math.isclose(baseline_metrics.udi,baseline['udi'],rel_tol=1e-12)
    assert math.isclose(float(baseline_row['TotalEnergy_J']),baseline['total_J'],rel_tol=1e-12)
    calculator=OpticalFeatureCalculator(Path(settings['reference_assets']))
    designs=[];canonical=None;canonical_idf=None
    for method,campaign in [('Agent',AGENT),('NSGA-II',NSGA)]:
        for seed in [5,11,17,23]:
            summary_path=next(campaign.glob(f'formal/seed{seed}/method_runs/*/summary.json'))
            s=read(summary_path);best=s['best_feasible'];directory=Path(best['energyplus_result']['run_directory'])
            assert s['n_energyplus_queries']==s['n_successful_queries']==40
            assert s['protocol']['search_space']['max_layers']==20
            assert s['protocol']['search_space']['required_wvo2_count']==0
            assert s['protocol']['search_space']['terminal_layer'] is None
            assert s['protocol']['udi_ratio_required']==.9
            inputs=read(directory/'input.json')
            assert inputs['epw']==settings['targets']['Shenzhen']['epw']
            assert inputs['base_idf']==settings['targets']['Shenzhen']['idf']
            assert inputs['energyplus_executable']==settings['energyplus_exe']
            metrics=parse_energyplus_outputs(directory,return_code=0,udi_denominator_mode='daylight_hours')
            assert metrics.status=='success' and metrics.severe_errors==0 and metrics.success_marker
            assert math.isclose(metrics.total_energy_J,best['energy'],rel_tol=1e-12)
            assert math.isclose(metrics.udi,best['udi'],rel_tol=1e-12)
            assert metrics.udi>=baseline['udi_threshold']
            meters=reference.annual_meters(directory)
            text=(directory/'modified.idf').read_text()
            if canonical is None:canonical=reference.non_spectral(text)
            assert canonical==reference.non_spectral(text),f'Physical IDF differs: {method} seed {seed}'
            if method=='Agent' and seed==23:canonical_idf=directory/'modified.idf'
            optics=calculator.calculate(SimpleNamespace(**reference.parse_spectra(text)))
            assert max(abs(optics[k]-best['optical_properties'][k]) for k in optics)<5e-5
            direct_online=[r for r in s['query_history'][16:] if r['direct_feasible']]
            online_best=min(direct_online,key=lambda r:r['energy']) if direct_online else None
            designs.append(dict(method=method,seed=seed,summary_file=str(summary_path),
                query_index=best['query_index'],structure=best['structure'],UDI=metrics.udi,
                qualifying_hours=metrics.udi_qualifying_hours,daylight_hours=metrics.udi_denominator_hours,
                udi_retention_pct=100*metrics.udi/baseline['udi'],n_feasible=s['n_feasible'],
                saving_pct=100*(baseline['total_J']-meters['total_J'])/baseline['total_J'],
                **meters,optical_features=best['optical_properties'],
                source_idf=str(directory/'modified.idf'),source_result=str(directory/'result.json'),
                online_best_saving_pct=100*online_best['energy_saving'] if online_best else None,
                online_best_query=online_best['query_index'] if online_best else None,
                n_fitness=s['method_specific_data'].get('n_fitness_evaluations'),
                n_generations=s['method_specific_data'].get('n_generations'),
                tree_decision=s['method_specific_data']['city_support']['adapter_decision']['decision']))
    assert canonical_idf is not None
    text=canonical_idf.read_text();tasks=[]
    for closure in ['absorptive_remainder','near_reflective_remainder']:
        directory=ROOT/'runs'/closure;directory.mkdir(parents=True,exist_ok=False)
        spectra=reference.states(closure)
        modified=build_idf_text(text,spectra['cold'],spectra['hot'])
        assert reference.non_spectral(modified)==canonical
        parsed=reference.parse_spectra(modified)
        import numpy as np
        for state in ['cold','hot']:
            for field in ['transmittance','front_reflectance','back_reflectance']:
                assert np.array_equal(getattr(parsed[state],field),getattr(spectra[state],field))
        (directory/'modified.idf').write_text(modified,encoding='utf-8')
        task=dict(index=len(tasks),city='Shenzhen',closure=closure,run_directory=str(directory),
            epw=settings['targets']['Shenzhen']['epw'],energyplus_executable=settings['energyplus_exe'],
            reference_idf=str(canonical_idf),base_idf=settings['targets']['Shenzhen']['idf'],
            non_spectral_IDF_identical_to_all_eight_designs=True,
            optical_features=calculator.calculate(SimpleNamespace(**parsed)))
        write(directory/'input.json',task);tasks.append(task)
    historical=read(PROJECT/'results/ideal_spectrum_vs_agent_20261026/manifest.json')
    manifest=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),city='Shenzhen',
        sources=dict(agent=str(AGENT),nsga2=str(NSGA),frozen_code=str(SOURCE)),
        definition_source=str(PROJECT/'results/ideal_spectrum_vs_agent_20261026'),
        interpretation='Established project ideal reference; no new claim that source-paper reflection or switching convention was verified',
        spectral_protocol=historical['spectral_protocol'],baseline=baseline,designs=designs,tasks=tasks,
        weather=settings['targets']['Shenzhen'],
        thermochromic_objects=[fields for _,fields in idf_objects(text) if 'thermochromic' in fields[0].casefold()],
        runperiod=[fields for _,fields in idf_objects(text) if fields[0].casefold()=='runperiod'])
    write(ROOT/'manifest.json',manifest)
    print(json.dumps({'tasks':tasks,'baseline':baseline,'thermochromic_objects':manifest['thermochromic_objects']},indent=2))


def worker(index):
    manifest=read(ROOT/'manifest.json');task=manifest['tasks'][index];directory=Path(task['run_directory'])
    if (directory/'process.json').exists():raise RuntimeError('Refuse to rerun a dispatched annual simulation')
    command=[task['energyplus_executable'],'-w',task['epw'],'-d',str(directory),'-r',str(directory/'modified.idf')]
    process=dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_index=index,return_code=None)
    write(directory/'process.json',process)
    print('ENERGYPLUS_STARTED',task['closure'],flush=True)
    try:
        environment=os.environ.copy();environment['OMP_NUM_THREADS']='1'
        with (directory/'stdout.log').open('wb') as out,(directory/'stderr.log').open('wb') as err:
            completed=subprocess.run(command,cwd=directory,stdout=out,stderr=err,env=environment,timeout=1200,check=False)
        process['return_code']=completed.returncode
        metrics=parse_energyplus_outputs(directory,return_code=completed.returncode,udi_denominator_mode='daylight_hours')
        write(directory/'metrics.json',asdict(metrics))
        assert metrics.status=='success',asdict(metrics)
        assert metrics.severe_errors==0 and metrics.success_marker and not metrics.fatal_detected
        error=EnergyPlusRunner._validate_parsed_metrics(directory,metrics)
        assert error is None,error
        meters=reference.annual_meters(directory);baseline=manifest['baseline']
        assert math.isclose(meters['total_J'],metrics.total_energy_J,rel_tol=1e-12)
        result=dict(status='success',city='Shenzhen',closure=task['closure'],**meters,
            UDI=metrics.udi,qualifying_hours=metrics.udi_qualifying_hours,daylight_hours=metrics.udi_denominator_hours,
            udi_retention_pct=100*metrics.udi/baseline['udi'],udi_feasible=metrics.udi>=baseline['udi_threshold'],
            saving_pct=100*(baseline['total_J']-meters['total_J'])/baseline['total_J'],
            optical_features=task['optical_features'],run_directory=str(directory),
            success_marker=metrics.success_marker,severe_errors=metrics.severe_errors,
            return_code=completed.returncode,energy_definition='Electricity:Facility + NaturalGas:Facility (J)',
            udi_definition='daylight_hours; 100-2000 lux')
        write(directory/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write(directory/'failure.json',{'status':'failed','error':str(exc)});raise
    finally:
        process['completed_utc']=datetime.now(timezone.utc).isoformat();write(directory/'process.json',process)


p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int)
args=p.parse_args()
if args.prepare:prepare()
elif args.worker is not None:worker(args.worker)
