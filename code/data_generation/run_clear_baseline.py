"""Run one city-specific clear-glass baseline; invoked only in a future simulation job."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from campaign_helpers import build_idf_text,registry,validate_metrics,save_baseline_row
from thermo_agent.energyplus import parse_energyplus_outputs,file_sha256


def main():
    data=registry()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city',required=True,choices=list(data['cities']))
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/project.json').read_text())
    executable=config['energyplus']['executable']
    base=ROOT/config['cities'][args.city]['baseline_idf']
    epw=ROOT/data['cities'][args.city]['epw']
    directory=ROOT/'baselines'/args.city
    directory.mkdir(parents=True,exist_ok=True)
    run_lock=(directory/'baseline_run.lock').open('a')
    import os
    if os.name!='nt':
        import fcntl
        fcntl.flock(run_lock,fcntl.LOCK_EX)
    identity={'city':args.city,'weather':file_sha256(epw),'idf':file_sha256(base),
              'protocol':'daylight-hours-site-energy-city-sizing-v2'}
    marker=directory/'baseline_inputs.json'
    if (directory/'baseline_row.json').exists():
        if json.loads(marker.read_text())!=identity:raise RuntimeError('Baseline inputs changed; use a new campaign')
        print('BASELINE_REUSED',args.city,flush=True)
        return
    marker.write_text(json.dumps(identity,indent=2)+'\n')
    run=Path(tempfile.mkdtemp(prefix='run_',dir=directory))
    idf=run/'input.idf';idf.write_text(build_idf_text(base.read_text()),encoding='utf-8')
    command=[executable,'-w',str(epw),'-d',str(run),'-r',str(idf)]
    (run/'input.json').write_text(json.dumps({'city':args.city,'epw':str(epw),'command':command},indent=2)+'\n')
    print('BASELINE_STARTED',args.city,str(run),flush=True)
    with (run/'process.stdout.log').open('wb') as stdout,(run/'process.stderr.log').open('wb') as stderr:
        completed=subprocess.run(command,cwd=run,stdout=stdout,stderr=stderr,timeout=1800,check=False)
    metrics=parse_energyplus_outputs(run,return_code=completed.returncode,udi_denominator_mode='daylight_hours')
    (run/'metrics.json').write_text(json.dumps(asdict(metrics),indent=2)+'\n')
    if metrics.status!='success':raise RuntimeError(f'{metrics.error_code}: {metrics.error_message}; {run}')
    validate_metrics(run,metrics)
    row=save_baseline_row(args.city,metrics,directory,run_directory=run)
    print('BASELINE_SUCCESS',json.dumps(row),flush=True)


if __name__=='__main__':main()
