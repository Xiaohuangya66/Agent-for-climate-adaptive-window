"""Train four source ensembles with frozen family splits and strict acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from bundle_env import runtime_environment

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();code=ROOT/'code/surrogate_training';run=ROOT/'models/source_surrogates'
    env=runtime_environment(ROOT)
    def execute(script,*options):
        subprocess.run([sys.executable,str(code/script),*map(str,options)],env=env,cwd=ROOT,check=True)
    if not args.resume:
        execute('train_source_surrogates.py','prepare','--campaign-root',ROOT/'data/source_campaign',
            '--run-dir',run,'--baseline-table',ROOT/'protocol/city_baselines_goal2.csv',
            '--split-mode','frozen','--split-manifest',ROOT/'protocol/grouped_split_reference.csv')
    if args.prepare_only:return
    execute('train_source_surrogates.py','train','--run-dir',run,'--workers',args.workers,
            *(['--resume'] if args.resume else []))
    execute('accept_models.py','--run-dir',run,'--server-project-root',ROOT/'code/runtime')
    acceptance=json.loads((run/'acceptance_complete.json').read_text())
    if set(acceptance['cities'].values())!={'PASS'}:
        raise RuntimeError('Some models require review; runtime publication is blocked. See evaluation/acceptance_report.md')
    print('FOUR_SOURCE_TRAINING_ACCEPTED',acceptance['cities'],flush=True)


if __name__=='__main__':main()
