"""Batch the six-city UDI90 protocol with equal 40-query budgets."""
import argparse
from datetime import datetime,timezone
from itertools import product
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

RUNTIME=Path(__file__).resolve().parents[1]
ROOT=RUNTIME.parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from bundle_env import runtime_environment
os.environ.update(runtime_environment(ROOT))
sys.path.insert(0,str(RUNTIME))
from src.runtime_profile import registry

AVAILABLE_CITIES=list(registry()['source_cities'])+[c for c,r in registry()['cities'].items() if r.get('role')=='target']
AVAILABLE_METHODS=['llm_agent','agent_full','nsga2']

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=AVAILABLE_METHODS,required=True)
    p.add_argument('--cities',nargs='+',choices=AVAILABLE_CITIES,required=True)
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--budget',type=int)
    p.add_argument('--output_dir',type=Path)
    p.add_argument('--parallel',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    if args.budget is not None and args.budget<0:p.error('--budget must be nonnegative')
    return args

def generate_tasks(args):
    version=json.loads((ROOT/'configs/campaign.json').read_text(encoding='utf-8'))
    output=args.output_dir or ROOT/'new_runs'/('batch_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    tasks=[]
    for city in args.cities:
        for method,seed in product(args.methods,args.seeds or version['seeds']):
            tasks.append({'method':method,'city':city,'seed':seed,
                'budget':args.budget if args.budget is not None else version['query_budget'],
                'output_dir':str(output/city/method/f'seed{seed}')})
    return tasks

def task_command(task):
    if task['method']=='llm_agent':
        return [sys.executable,str(ROOT/'tools/run_llm_agent.py'),'--city',task['city'],
            '--seed',str(task['seed']),'--budget',str(task['budget']),'--output-dir',task['output_dir']]
    return [sys.executable,str(ROOT/'tools/run_experiment.py'),'--method',task['method'],
            '--city',task['city'],'--seed',str(task['seed']),'--budget',str(task['budget']),
            '--output-dir',task['output_dir']]

def run_task(task,dry_run=False):
    command=task_command(task)
    if dry_run:print(shlex.join(command));return True
    return subprocess.run(command,cwd=ROOT,env=runtime_environment(ROOT),check=False).returncode==0

def submit_slurm_job(task,args):
    if task['method']=='llm_agent':raise ValueError('Run LLM Agent locally with environment credentials; this server is offline')
    command=task_command(task)
    if args.dry_run:print('SLURM',shlex.join(command));return
    if os.name=='nt':raise RuntimeError('Generate Slurm submissions on the target Linux host')
    directory=Path(task['output_dir']).parent/'slurm';directory.mkdir(parents=True,exist_ok=True)
    script=directory/f"{task['method']}_{task['city']}_seed{task['seed']}.sbatch"
    if script.exists():raise FileExistsError(f'Refusing duplicate submission: {script}')
    script.write_text('#!/bin/bash\n#SBATCH --partition=normal\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=12G\n#SBATCH --time=08:00:00\nset -euo pipefail\n'+
        'cd '+shlex.quote(str(ROOT))+'\n'+shlex.join(command)+'\n',encoding='utf-8')
    subprocess.run(['sbatch','--parsable',f'--output={directory}/%j.out',f'--error={directory}/%j.err',str(script)],check=True)

def main():
    args=parse_args();tasks=generate_tasks(args)
    print(json.dumps({'cities':args.cities,'tasks':len(tasks),'query_budget':sum(t['budget'] for t in tasks),
        'udi_ratio':.9,'settings':str(ROOT/'configs/runtime.json')},indent=2))
    if args.parallel:
        for task in tasks:submit_slurm_job(task,args)
    else:
        results=[run_task(task,args.dry_run) for task in tasks]
        if not all(results):raise SystemExit(1)

if __name__=='__main__':main()
