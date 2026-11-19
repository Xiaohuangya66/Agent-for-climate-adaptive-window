"""Constraint-aware NSGA-II: surrogate evolution, then independent direct verification."""
import json
import time
from types import SimpleNamespace
from src.agent.base_agent import BaseAgent
from src.agent.baseline_common import BaselineRecords,SearchSpace,FEATURE_ORDER
from src.vendor.thermo_agent.nsga2 import rank_and_crowding,select_survivors
from src.tools.bias_corrector import BiasCorrector


class NSGA2Method(BaselineRecords,BaseAgent):
    def __init__(self,city,seed,budget,population_size=20,data_dir='data',
                 runtime_dir='runtime_assets',output_root=None,max_layers=None,
                 candidate_budget=384,fitness_source='direct',probe_count=None):
        super().__init__(city,seed,budget,data_dir,runtime_dir)
        self.requested_population_size=int(population_size)
        self.candidate_budget=int(candidate_budget)
        if self.requested_population_size<2 or self.candidate_budget<1:
            raise ValueError('population_size >= 2 and candidate_budget >= 1 required')
        if fitness_source not in ['surrogate','direct']:
            raise ValueError('fitness_source must be surrogate or direct')
        self.fitness_source=fitness_source
        limit=self.candidate_budget if fitness_source=='surrogate' else max(1,budget//2)
        self.population_size=min(self.requested_population_size,limit)
        self.space=SearchSpace(max_layers=max_layers)
        self._setup_records('nsga2',output_root)
        self.population=[];self.archive=[];self.evaluated=set();self.generation=0
        from src.support_settings import settings
        requested=(0 if fitness_source=='direct' else settings().get('probe_count',16)) if probe_count is None else probe_count
        if isinstance(requested,bool) or int(requested)!=requested or requested<0:
            raise ValueError('probe_count must be a nonnegative integer')
        self.probe_count=min(int(requested),max(0,self.budget_total-1)) if self.city_support else 0
        self.optics={}
        if self.city_support:self.bias_corrector=BiasCorrector(window_size=20)

    def _json_line(self,name,data):
        with (self.run_directory/name).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(data,allow_nan=False)+'\n')

    def _refresh_prediction(self,row):
        raw=self.surrogate.predict(row['optical_properties'],strict=True)
        corrected=self.bias_corrector.correct(raw) if self.city_support else raw
        row.update(self.formal.prediction(corrected),raw_prediction=raw,
                   support_version=getattr(self.surrogate,'VERSION','native_city_model'))
        if self.city_support:
            # The target intervals are heuristic, not calibrated safety bounds.
            # Use the same center + margin gate as the paired Agent run.
            row['constraint_udi_margin']=row['performance_udi_margin']
        return row

    def _update_city_support(self):
        if not self.city_support:return
        old_version=getattr(self.surrogate,'VERSION','native_city_model')
        decision=self.city_support.fit_or_keep(self)
        if getattr(self.city_support.predictor,'VERSION','native_city_model')==old_version:return
        self.surrogate=self.city_support.predictor
        self.bias_corrector=BiasCorrector(window_size=20)
        for observed in self.query_history:
            if observed.get('energyplus_verified') and observed.get('adaptation_role','fit')=='fit':
                self.bias_corrector.update([self.surrogate.predict(observed['optical_properties'])],
                                           [observed],prequential=False)
        for candidate in self.archive:
            if candidate.get('evaluation_status')=='success':self._refresh_prediction(candidate)
        self._json_line('model_updates.jsonl',{'budget_used':self.budget_used,
            'previous_version':old_version,'new_version':self.surrogate.VERSION,
            'bias_rebuilt_from_fit_queries_only':True,
            'bias_statistics':self.bias_corrector.get_statistics(),'decision':decision})

    def _query_candidate(self,candidate,**extra):
        self._refresh_prediction(candidate)
        raw=candidate['raw_prediction']
        role=extra.pop('adaptation_role',None)
        if role is None:role=self.city_support.next_observation_role() if self.city_support else 'fit'
        row=self._direct_query(candidate['structure'],**extra,adaptation_role=role,
            predicted_energy=candidate['predicted_energy'],predicted_udi=candidate['predicted_udi'],
            raw_predicted_energy=raw['TotalEnergy'],raw_predicted_udi=raw['UDI'],
            performance_udi_margin=candidate['performance_udi_margin'],
            prediction_decision_threshold=candidate['prediction_decision_threshold'],
            udi_half_width_90pct=candidate['udi_half_width_90pct'],ood_distance=candidate['ood_distance'])
        if self.city_support and row is not None:
            accepted=self.bias_corrector.update([raw],[row]) if role!='validation' else 0
            self._json_line('bias_history.jsonl',{'query_index':row['query_index'],
                'phase':row['phase'],'observation_accepted':bool(accepted),
                **self.bias_corrector.get_statistics()})
        return row

    def _run_support_probes(self):
        if not self.city_support or self.probe_count<=0:return
        plan=self.city_support.propose_probes(self,self.probe_count)
        for sample in plan['samples']:
            structure=sample['structure']
            if structure in self.queried:continue
            self.n_tmm+=1;props=self.tmm.compute(structure);self.optics[structure]=props
            candidate={'structure':structure,'optical_properties':{k:float(props[k]) for k in FEATURE_ORDER}}
            self._query_candidate(candidate,phase='city_support_probe',probe_id=sample['probe_id'],
                source_reference=sample.get('source_reference'),selection_reason=sample['selection_reason'],
                adaptation_role=sample.get('adaptation_role','fit'))
        self._update_city_support()
        # Do not carry pre-publication UDI residuals into the search phase.
        self.bias_corrector.reset_udi()
        self._json_line('online_schedule.jsonl',{'probe_attempts':self.budget_used,
            'online_query_budget':self.budget_total-self.budget_used,
            'bias_statistics':self.bias_corrector.get_statistics()})

    @staticmethod
    def _proxy(row):
        return SimpleNamespace(predicted_energy_J=row.get('predicted_energy',row.get('energy')),
            predicted_udi=row.get('predicted_udi',row.get('udi')),
            udi_half_width_90pct=row.get('constraint_udi_margin',row.get('udi_half_width_90pct',0.)),row=row)

    def _tournament(self):
        proxies=[self._proxy(r) for r in self.population]
        ranks,distances,_=rank_and_crowding(proxies,self.formal_udi_threshold)
        options=self.rng.sample(range(len(proxies)),min(2,len(proxies)))
        return self.population[min(options,key=lambda i:(ranks[i],-distances[i]))]

    def _next_unique(self,pending):
        for attempt in range(200):
            self.n_generation_attempts+=1
            if self.population and attempt<100:
                structure=self.space.child(self._tournament()['structure'],self._tournament()['structure'],self.rng)
            else:
                structure=self.space.random(self.rng)
            if structure not in self.evaluated and structure not in pending and structure not in self.queried:
                return structure
        raise RuntimeError('NSGA-II exhausted unique candidate retries')

    def run(self):
        started=time.time()
        self._run_support_probes()
        # A zero direct allowance must not unexpectedly start a costly search.
        remaining=self.budget_total-self.budget_used
        limit=(self.candidate_budget if self.fitness_source=='surrogate' else remaining) if remaining else 0
        while len(self.evaluated)<limit:
            requested=min(self.population_size,limit-len(self.evaluated))
            children=[];pending=set()
            for _ in range(requested):
                structure=self._next_unique(pending)
                pending.add(structure);self.evaluated.add(structure)
                if self.fitness_source=='direct':
                    row=self._direct_query(structure,generation=self.generation,
                        phase='initial' if self.generation==0 else 'offspring')
                    if row and row['energyplus_verified']:children.append(row)
                else:
                    row={'structure':structure,'generation':self.generation}
                    try:
                        self.n_tmm+=1
                        props=self.tmm.compute(structure)
                        features={k:props[k] for k in FEATURE_ORDER}
                        self.optics[structure]=props
                        row['optical_properties']={k:float(v) for k,v in features.items()}
                        self._refresh_prediction(row)
                        row['evaluation_status']='success';children.append(row)
                    except Exception as exc:
                        row.update(evaluation_status='failed',error=f'{type(exc).__name__}: {exc}')
                    self.archive.append(row)
                    with (self.run_directory/'fitness.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
            combined=self.population+children
            if combined:
                self.population=[p.row for p in select_survivors([self._proxy(r) for r in combined],
                    self.population_size,self.formal_udi_threshold)]
            self.generation_log.append({'generation':self.generation,'new_attempts':requested,
                'new_successes':len(children),'survivors':len(self.population)})
            self.logger.info('NSGA-II generation %s: evaluated=%s/%s, new_successes=%s, survivors=%s',
                self.generation,len(self.evaluated),limit,len(children),len(self.population))
            self.generation+=1
        if self.fitness_source=='surrogate':
            while self.budget_used<self.budget_total:
                available=[r for r in self.archive if r['evaluation_status']=='success' and r['structure'] not in self.queried]
                if not available:break
                # Residual correction changes after each true observation; stale
                # predictions must not determine the next expensive query.
                if self.city_support:
                    for row in available:self._refresh_prediction(row)
                row=min(available,key=self.formal.query_key)
                self._query_candidate(row,phase='post_evolution_validation')
                self._update_city_support()
            (self.run_directory/'candidates.json').write_text(json.dumps(self.archive,indent=2,allow_nan=False))
        return self._finish_records(started,{'algorithm':'NSGA-II',
            'objectives':['min_energy','max_udi'],'constraint':f'Deb center plus adaptive margin UDI >= {self.formal.udi_ratio:.0%} clear glass' if self.city_support else f'Deb conservative UDI >= {self.formal.udi_ratio:.0%} clear glass',
            'selection':'Pareto rank and crowding binary tournament','survival':'elitist parent-plus-offspring',
            'fitness_source':self.fitness_source,'candidate_budget':self.candidate_budget if self.fitness_source=='surrogate' else self.budget_total-self.probe_count,
            'n_fitness_evaluations':len(self.evaluated),'n_generations':self.generation,
            'generation_history':self.generation_log,'n_generated_attempts':self.n_generation_attempts,
            'city_support':self.city_support.describe() if self.city_support else None,
            'probe_count':self.probe_count,'n_probe_attempts':sum(r.get('phase')=='city_support_probe' for r in self.query_history),
            'bias_statistics':self.bias_corrector.get_statistics() if self.city_support else None,
            'query_selection':'post-evolution shared performance query_key, refreshed after direct fit observations',
            'excluded_agent_components':['Transformer generation','TTT policy updates','online target refinement','information-value selector'],
            'requested_population_size':self.requested_population_size,'effective_population_size':self.population_size})
