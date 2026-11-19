"""Validation-only replay of bootstrap labels plus one new annual query."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agent.formal_protocol import configure_udi_ratio
from src.agent.nsga2_method import NSGA2Method

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True);args=p.parse_args()
configure_udi_ratio(.9)
reference=json.loads(next(args.reference.glob('formal/seed5/method_runs/*/summary.json')).read_text())
agent=NSGA2Method('Shenzhen',5,17,population_size=20,candidate_budget=384,
    max_layers=20,probe_count=16,output_root=args.workspace/'validation'/'method_runs')
real_run=agent.energyplus.run;replayed=[];new_queries=[]

def run(structure,city,output_dir=None):
    if len(replayed)<16:
        expected=reference['query_history'][len(replayed)]
        assert structure==expected['structure'], 'Bootstrap plan differs from paired Agent'
        result=copy.deepcopy(expected['energyplus_result'])
        result.update(process_started=False,validation_only_replay=True,
            replayed_query_id=expected['query_index'])
        replayed.append(expected['query_index'])
        return result
    result=real_run(structure,city,output_dir=output_dir)
    new_queries.append(result)
    return result

agent.energyplus.run=run
result=agent.run()
rows=result['query_history']
assert len(rows)==17 and result['n_energyplus_processes']==1
assert all(r['energyplus_verified'] for r in rows)
assert sum(r['adaptation_role']=='validation' for r in rows)==4
assert result['method_specific_data']['n_fitness_evaluations']==384
assert result['method_specific_data']['n_generations']==20
assert all(agent.space.valid(r['structure']) for r in agent.archive)
assert len({r['structure'] for r in agent.archive})==384
assert not ({r['structure'] for r in agent.archive}&{r['structure'] for r in rows[:16]})
assert agent.city_support.tree_decision['decision']=='publish_continued_trees'
assert agent.bias_corrector.n_udi_observations==1
assert agent.bias_corrector.n_observations==13
for expected,actual in zip(reference['query_history'][:16],rows[:16]):
    assert actual['adaptation_role']==expected['adaptation_role']
assert agent.city_support.tree_decision['fit_query_ids']==[1,2,3,5,6,7,9,10,11,13,14,15]
assert agent.city_support.tree_decision['validation_query_ids']==[4,8,12,16]
differences=[]
for row in reference['query_history'][16:]:
    pred=agent.surrogate.predict(row['optical_properties'])
    old=row['prediction_snapshot']['raw']
    differences.append((abs(pred['UDI']-old['UDI']),abs(pred['TotalEnergy']-old['TotalEnergy'])/old['TotalEnergy']))
assert max(d[0] for d in differences)<1e-7
assert max(d[1] for d in differences)<1e-7
acceptance={'status':'PASS','purpose':'validation only; replayed labels are never formal NSGA labels',
    'n_replayed_bootstrap_labels':16,'n_new_annual_queries':1,'total_validation_queries':17,
    'n_candidates':384,'n_generations':20,'tree_publication':agent.city_support.tree_decision,
    'max_abs_udi_difference_against_paired_Agent_model':max(d[0] for d in differences),
    'max_relative_energy_difference_against_paired_Agent_model':max(d[1] for d in differences),
    'actual_new_query':new_queries[0],'run_directory':str(agent.run_directory),
    'fair_comparison':'Both formal methods run 16 fresh probes plus 24 search queries per seed',
    'uncertainty':'heuristic, not target calibrated; no nominal target coverage claim'}
target=args.workspace/'validation/acceptance.json';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(acceptance,indent=2,allow_nan=False))
print(json.dumps(acceptance,indent=2))
