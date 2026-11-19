"""Replay 16 historical labels, then run ONE new validation-only annual query."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.support_settings import settings
from src.agent.formal_protocol import configure_udi_ratio
from src.agent.agent_full import AgentFull
from src.components.ttt_policy import TTTPolicy
from src.tools.query_service import QueryService,append_fact
from src.tools.city_support import write_json
from src.tools.information_selector import InformationValueSelector


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args();configure_udi_ratio(.9)
    parent=Path(settings()['parent_workspace'])
    old=parent/'formal/seed5/method_runs/agent_full_Shenzhen_seed5_setgjslf'
    labels=[json.loads(s) for s in (old/'queries.jsonl').read_text().splitlines()][:16]
    dispatch=[json.loads(s) for s in (old/'query_dispatch.jsonl').read_text().splitlines()][:16]
    a=AgentFull('Shenzhen',5,budget=0,probe_count=0,candidate_budget=16,batch_size=16,
                ttt_batch_size=4,output_root=args.workspace/'validation/replay_runs')
    a.budget_total=40;a.started=time.time()
    # Construction at budget=0 skips the generator; restore the corresponding
    # budget-derived quota when assembling this validation-only replay session.
    a.information_quota=InformationValueSelector.information_quota(40,a.information_fraction)
    write_json(a.run_directory/'VALIDATION_REPLAY_ONLY.json',{
        'source_job':1587,'historical_labels_reused':16,'new_annual_queries_planned':1,
        'purpose':'engineering validation, excluded from all formal seed runs',
        'not_a_cold_start_performance_result':True})
    for row in labels:append_fact(a.run_directory/'queries.jsonl',copy.deepcopy(row))
    for row in dispatch:append_fact(a.run_directory/'query_dispatch.jsonl',copy.deepcopy(row))
    a.query_service=QueryService(a);a.query_service.reconcile()
    assert a.budget_used==16
    a._phase1_initialize()
    for row in labels:
        if row['adaptation_role']=='fit':
            a.bias_corrector.update([a.surrogate.predict(row['optical_properties'])],[row],prequential=False)
    a._update_city_support()
    a.best_direct_energy=a._refinement_pool()[0]['energy']
    a._start_online_phase()
    assert a.online_query_offset==16 and a.online_query_budget==24 and a.information_quota==12
    assert a.bias_corrector.n_udi_observations==0
    a.ttt_policy=TTTPolicy(a.space,checkpoint=a.checkpoint_path,device='cpu',nhead=4,learning_rate=1e-4)
    a.iteration=1;a._choose_target();a._generate_candidates()
    preview=a._select_for_energyplus(a.candidate_archive,1)
    assert preview and preview[0]['query_selection']['mode']=='performance'
    assert preview[0]['query_selection']['online_query_index']==1
    assert a.budget_used==16
    print('NEW_VALIDATION_QUERY_STARTED',a.run_directory,flush=True)
    a._query_and_evaluate(1)
    final=a.query_history[-1]
    assert a.budget_used==17 and final['query_index']==17 and final['energyplus_verified']
    assert final['selection']['mode']=='performance' and a.bias_corrector.n_udi_observations==1
    snapshot=final['prediction_snapshot']['corrected']
    assert np.isclose(snapshot['UDI_lower_90pct'],max(0,snapshot['UDI']-snapshot['UDI_half_width_90pct']))
    record={'status':'PASS','validation_only':True,'historical_labels_replayed':16,
        'new_EnergyPlus_annual_queries':1,'formal_knowledge_reused_from_validation':False,
        'run_directory':str(a.run_directory),'tree_decision':a.city_support.tree_decision,
        'n_TTT_updates':sum(x['parameters_changed'] for x in a.adaptation_history),
        'first_online_mode':final['selection']['mode'],'online_budget':24,'information_quota':12,
        'UDI_bias_observations_before_first_online':0,'UDI_bias_observations_after':1,
        'new_query_real_UDI':final['udi'],'new_query_real_saving_pct':final['energy_saving']*100}
    write_json(args.workspace/'validation/optimized_acceptance.json',record)
    print('OPTIMIZED_ACCEPTANCE_PASS',json.dumps(record),flush=True)


if __name__=='__main__':main()
