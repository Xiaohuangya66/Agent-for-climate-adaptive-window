"""Preflight an explicit plan. No EnergyPlus dispatch in this entry point."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agent.formal_protocol import configure_udi_ratio,FormalProtocol
from src.runtime_profile import city_assets
from src.components.tmm import TMM
from src.tools.city_support import CitySupportTool,write_json
from src.agent.baseline_common import SearchSpace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city',required=True);parser.add_argument('--seed',type=int,default=5)
    parser.add_argument('--budget',type=int,default=40);parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--max-layers',type=int,default=20);parser.add_argument('--udi-ratio',type=float,default=.9)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    configure_udi_ratio(args.udi_ratio)
    support=CitySupportTool(args.city,city_assets(args.city),FormalProtocol(args.city),TMM(Path('.'))).prepare()
    from types import SimpleNamespace
    state=SimpleNamespace(space=SearchSpace(max_layers=args.max_layers),budget_total=args.budget,budget_used=0,
        queried=set(),seed=args.seed,formal=support.formal)
    write_json(args.output,support.propose_probes(state,args.count))
    print('PROBE_PLAN_READY',args.output)


if __name__=='__main__':main()
