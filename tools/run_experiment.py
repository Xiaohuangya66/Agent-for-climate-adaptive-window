"""Run the current collected Agent/NSGA interface using bundle-relative assets."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from bundle_env import runtime_environment

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--method',choices=['agent_full','nsga2'],required=True)
p.add_argument('--city',required=True)
p.add_argument('--seed',type=int,default=5);p.add_argument('--budget',type=int)
p.add_argument('--output-dir',type=Path);args,extra=p.parse_known_args()
environment=runtime_environment(ROOT)
budget=args.budget if args.budget is not None else 40
output=args.output_dir or ROOT/'new_runs'/f"{args.city}_{args.method}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if output.exists():raise ValueError(f'Use a new output directory: {output}')
cmd=[sys.executable,'-u',str(ROOT/'code/runtime/experiments/run_single.py'),
     '--method',args.method,'--city',args.city,'--seed',str(args.seed),'--budget',str(budget),
     '--udi-ratio','.90','--max-layers','20','--candidate-budget','384','--output_dir',str(output)]
if args.method=='agent_full':cmd+=['--probe-count','16']
if args.method=='agent_full':cmd+=['--agent-batch-size','128' if args.city=='Shenzhen' else '64','--batch-size','16']
else:cmd+=['--population-size','20','--fitness-source','direct','--probe-count','0']
print('Current collected runtime; archived historical source-city trajectories retain their original snapshot.',flush=True)
completed=subprocess.run(cmd+extra,env=environment,cwd=ROOT,check=True)
import shutil
summaries=list((output/'method_runs').glob('*/summary.json'))
if len(summaries)!=1:raise RuntimeError('Expected one numerical-method result')
shutil.copy2(summaries[0],output/'result.json')
