"""Explicit debug probe session using the same Agent budget and query backend."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-run',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    if not plan.get('execution_ready'):raise ValueError('Plan is not executable')
    identity={k:plan[k] for k in ['city','seed','total_design_budget','udi_ratio','assets']}
    identity['structures']=[s['structure'] for s in plan['samples']]
    if args.resume_run:
        marker=args.resume_run/'probe_session.json'
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8'))!=identity:
            raise ValueError('Resume is restricted to this exact probes-only session; full Agent runs cannot be resumed here')
    from src.agent.formal_protocol import configure_udi_ratio
    configure_udi_ratio(plan['udi_ratio'])
    from src.agent.agent_full import AgentFull
    from src.tools.query_service import QueryService
    # budget=0 avoids loading/training the policy for a probes-only diagnostic.
    agent=AgentFull(plan['city'],plan['seed'],budget=0,probe_count=0,output_root=args.output,
                    max_layers=plan['search_space']['max_layers'])
    if agent.runtime_assets!=plan['assets']:raise ValueError('Plan/runtime assets differ; prepare a fresh plan')
    if agent.space.description()!=plan['search_space']:raise ValueError('Plan/search space differs')
    agent.budget_total=plan['total_design_budget'];agent.probe_count=len(plan['samples']);agent.started=time.time()
    if args.resume_run:
        # The temporary empty session above contains no physics or labels.
        agent.run_directory=args.resume_run.resolve();agent.city_support.bind(agent.run_directory)
        protocol=json.loads((agent.run_directory/'formal_protocol.json').read_text())
        if protocol['udi_ratio_required']!=plan['udi_ratio'] or protocol['runtime_assets']!=plan['assets']:
            raise ValueError('Resume protocol/asset mismatch')
        agent.query_service=QueryService(agent);agent.query_service.reconcile()
        for row in agent.query_history:
            if row.get('adaptation_role')=='validation':continue
            raw=agent.surrogate.predict(row['optical_properties'])
            agent.bias_corrector.update([raw],[row])
    else:
        from src.tools.city_support import write_json
        write_json(agent.run_directory/'probe_session.json',identity)
    agent._phase1_initialize();agent._run_support_probes(plan)
    result=agent._finish_records(agent.started,{'mode':'probe_debug_only','city_support':agent.city_support.describe(),
        'full_agent_policy_resume':False,'unspent_budget':agent.budget_total-agent.budget_used})
    print('PROBES_COMPLETE',json.dumps({'run_directory':str(agent.run_directory),'n_queries':result['n_energyplus_queries']}))


if __name__=='__main__':main()
