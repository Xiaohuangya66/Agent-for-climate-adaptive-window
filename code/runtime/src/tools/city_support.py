"""City support is a prediction/initialization tool; Agent owns every query."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from functools import lru_cache
import numpy as np

FEATURES = ('T_lum_cold','T_lum_hot','T_sol_cold','T_sol_hot',
            'Rf_sol_cold','Rb_sol_cold','Rf_sol_hot','Rb_sol_hot')
CLIMATE_SCALES={'energy':{'CDD18':1000,'HDD18':1000,'GHI_kWh_m2':500,
                         'diffuse_fraction':.25,'RH':20,'solar_seasonality':.25},
                'udi':{'GHI_kWh_m2':500,'diffuse_fraction':.25,'solar_seasonality':.25,'latitude':30}}


def serial(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,default=serial,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


@lru_cache(maxsize=16)
def climate_features(epw):
    """TMY hourly descriptors. Zero HDD/CDD is valid, missing hours are not."""
    with Path(epw).open(encoding='utf-8-sig') as stream:
        header=next(csv.reader([stream.readline()]))
        for _ in range(7):next(stream)
        rows=list(csv.reader(stream))
    if len(rows)!=8760:raise ValueError('Climate support expects a non-leap 8760-hour EPW')
    values=np.array([[float(r[i]) for i in [1,6,8,13,15]] for r in rows])
    month,temp,rh,ghi,dhi=values.T
    if (not np.isfinite(values).all() or np.any(np.abs(temp)>80) or
        np.any((rh<0)|(rh>100)) or np.any((ghi<0)|(ghi>=9999)) or
        np.any((dhi<0)|(dhi>=9999)) or ghi.sum()<=0):
        raise ValueError('EPW has missing/invalid climate data')
    daily=temp.reshape(365,24).mean(axis=1)
    monthly=np.array([ghi[month==m].mean() for m in range(1,13)])
    return dict(CDD18=float(np.maximum(daily-18,0).sum()),
                HDD18=float(np.maximum(18-daily,0).sum()),
                GHI_kWh_m2=float(ghi.sum()/1000),diffuse_fraction=float(dhi.sum()/ghi.sum()),
                RH=float(rh.mean()),solar_seasonality=float(monthly.std()/monthly.mean()),
                latitude=float(header[6]),hours=len(rows))


def climate_weights(target, sources, temperature=1., equal=False):
    if temperature<=0 or not np.isfinite(temperature):raise ValueError('Invalid climate temperature')
    # Explicit physical scales avoid division by a tropical city's zero HDD.
    result={};distances={}
    for response,scales in CLIMATE_SCALES.items():
        d=np.array([np.sqrt(np.mean([((target[k]-v[k])/s)**2 for k,s in scales.items()]))
                    for v in sources.values()])
        logits=-d/temperature;weights=np.exp(logits-logits.max())
        weights=np.ones_like(weights) if equal else weights
        result[response]=dict(zip(sources,(weights/weights.sum()).tolist()))
        distances[response]=dict(zip(sources,d.tolist()))
    return result,distances


class SourceCityBank:
    def __init__(self, cities=None):
        from src.runtime_profile import city_assets, registry
        from src.agent.formal_protocol import FormalProtocol
        from src.components.surrogate import Surrogate
        allowed=tuple(registry()['source_cities'])
        self.cities=tuple(allowed if cities is None else cities)
        if not self.cities or len(set(self.cities))!=len(self.cities) or not set(self.cities)<=set(allowed):
            raise ValueError(f'Transfer sources must be distinct accepted source cities: {allowed}')
        self.assets={c:city_assets(c) for c in self.cities}
        self.protocols={c:FormalProtocol(c) for c in self.cities}
        self.models={c:Surrogate(self.assets[c]['model_directory']) for c in self.cities}
        self.climates={c:climate_features(self.assets[c]['epw']) for c in self.cities}
        scales=[self.models[c].ensemble.x_scale for c in self.cities]
        self.feature_scale=np.maximum(np.mean(scales,axis=0),1e-6)


class TransferSurrogate:
    VERSION='normalized_source_experts_target_clip_v001b'
    # Virtual source responses can exceed a source city's native UDI range.
    # Both frozen and continued experts use the same mapping and clip only
    # AFTER conversion to the target city, so their comparison is not confounded.
    allow_unbounded_source_udi=True
    def __init__(self, bank, target_protocol, target_climate, temperature=1., equal=False):
        self.bank,self.protocol=bank,target_protocol
        self.target_climate=dict(target_climate)
        self.temperature=float(temperature);self.equal_weights=bool(equal)
        self.weights,self.distances=climate_weights(target_climate,bank.climates,temperature,equal)
        self.climate_ood=max(min(v.values()) for v in self.distances.values())
        self.feature_names=list(FEATURES)
        self.feature_scale=bank.feature_scale
        self._cache={}
        self.metadata={'version':self.VERSION,'uncertainty_status':'target_uncalibrated_heuristic',
                       'source_cities':list(bank.cities),'weights':self.weights}

    def predict(self, optical_features, *, strict=True):
        x=np.asarray([optical_features[k] for k in FEATURES] if isinstance(optical_features,dict)
                     else optical_features,dtype=float)
        if x.shape!=(8,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
            raise ValueError('Transfer predictor requires eight finite optical features in [0,1]')
        key=tuple(x)
        if key in self._cache:return dict(self._cache[key])
        predictions={c:self.bank.models[c].predict(x,strict=strict) for c in self.bank.cities}
        means={};sigmas={};halves={}
        for response,field,baseline,std,half in [
                ('energy','TotalEnergy','energy','Energy_std','Energy_half_width_90pct'),
                ('udi','UDI','clear_udi','UDI_std','UDI_half_width_90pct')]:
            w=np.array([self.weights[response][c] for c in self.bank.cities])
            b=np.array([getattr(self.bank.protocols[c],baseline) for c in self.bank.cities])
            m=np.array([predictions[c].get('UDI_unbounded',predictions[c]['UDI'])
                if field=='UDI' and getattr(self,'allow_unbounded_source_udi',False)
                else predictions[c][field] for c in self.bank.cities])/b
            s=np.array([predictions[c][std] for c in self.bank.cities])/b
            h=np.array([predictions[c][half] for c in self.bank.cities])/b
            mean=float(w@m);between=float(w@((m-mean)**2))
            target_baseline=getattr(self.protocol,baseline)
            means[response]=mean*target_baseline
            sigmas[response]=np.sqrt(float(w@(s*s))+between)*target_baseline
            # Disagreement plus source widths, inflated with climate distance.
            # This is NOT a conformal interval for the target city.
            halves[response]=np.sqrt(float(w@(h*h))+between)*target_baseline*(1+.25*self.climate_ood)
        if means['energy']<=0 or not np.isfinite(list(means.values())).all():
            raise ValueError('Invalid normalized transfer response')
        raw=means['udi'];udi=float(np.clip(raw,0,1));uh=float(halves['udi'])
        result={'UDI':udi,'UDI_unbounded':raw,'UDI_was_clipped':raw!=udi,
                'TotalEnergy':means['energy'],'UDI_std':float(sigmas['udi']),
                'Energy_std':float(sigmas['energy']),'UDI_half_width_90pct':uh,
                'Energy_half_width_90pct':float(halves['energy']),
                'UDI_lower_90pct':max(0.,udi-uh),'UDI_upper_90pct':min(1.,udi+uh),
                'ood_distance':max(p['ood_distance'] for p in predictions.values())+self.climate_ood,
                'climate_ood_distance':self.climate_ood,'support_version':self.VERSION,
                'uncertainty_status':'target_uncalibrated_heuristic',
                'udi_definition':'daylight_hours','energy_definition':'Electricity:Facility + NaturalGas:Facility'}
        self._cache[key]=result
        return dict(result)

    def predict_uncertainty(self, features):return self.predict(features)['UDI_half_width_90pct']


class TransferPriorProvider:
    def __init__(self, bank, predictor, tmm):
        self.bank,self.predictor,self.tmm=bank,predictor,tmm
        self.clusters=[]

    def learn_prior_clusters(self, city, n_clusters=3, **kwargs):
        from src.tools.city_prior_learner import CityPriorLearner
        from src.agent.formal_protocol import FormalProtocol
        if self.clusters:return self._result(city)
        for source in self.bank.cities:
            prior=CityPriorLearner(protocol_factory=lambda c:FormalProtocol(c,udi_ratio=.9)).learn_prior_clusters(
                source,n_clusters=n_clusters,**kwargs)
            for original in prior['clusters']:
                cluster=dict(original)
                prediction=self.predictor.predict(cluster['center'])
                cluster.update(source_city=source,source_cluster_id=original['cluster_id'],
                    source_reference={'avg_energy':original['avg_energy'],'avg_udi':original['avg_udi'],
                                      'source_consistency':prior['confidence'],
                                      'source_dataset':self.bank.assets[source]['dataset']},
                    target_prediction=prediction,avg_energy=prediction['TotalEnergy'],avg_udi=prediction['UDI'])
                self.clusters.append(cluster)
        # Rank by target response; never mix unnormalized source-city energies.
        self.clusters.sort(key=lambda c:(c['avg_udi']<self.predictor.protocol.threshold,c['avg_energy'],
            -self.predictor.weights['energy'][c['source_city']]))
        for i,c in enumerate(self.clusters):c['cluster_id']=i
        return self._result(city)

    def _result(self,city):
        return {'clusters':self.clusters,'confidence':0.,'city':city,
                'confidence_definition':'target transfer reliability unknown; source consistency is recorded separately',
                'selection':'source UDI90 elites, target predicted feasibility and normalized energy ranking'}


class ProbePlanner:
    def __init__(self, support):self.support=support

    def plan(self, space, limit, existing=()):
        from src.support_settings import settings
        if int(limit)!=limit or limit<0:raise ValueError('Probe count must be a nonnegative integer')
        candidates=[]
        anchor_file=settings().get('source_anchors')
        if anchor_file:
            for sample in json.loads(Path(anchor_file).read_text(encoding='utf-8'))['samples']:
                if sample['source_reference']['city'] in self.support.bank.cities:
                    candidates.append(dict(structure=sample['structure'],role='source_city_champion_anchor',
                        source_reference=sample['source_reference'],source_optical_features=sample.get('source_optical_features')))
        prior=self.support.prior_provider.learn_prior_clusters(self.support.city)
        candidates += [dict(structure=c['representative_structure'],role='source_cluster_representative',
                            source_reference={'city':c['source_city'],'cluster_id':c['source_cluster_id']})
                       for c in prior['clusters']]
        if settings().get('tree_adaptation',{}).get('enabled',False):
            # Expand beyond champions: actual source training structures span
            # different optical regimes, including the source UDI90 boundary.
            from src.runtime_profile import MODEL_RUN
            import pandas as pd
            for source in self.support.bank.cities:
                frame=pd.read_csv(MODEL_RUN/'datasets'/f'{source}_train_fit.csv.gz')
                protocol=self.support.bank.protocols[source]
                eligible=frame[frame['UDI'].between(.75*protocol.clear_udi,1)].copy()
                if eligible.empty:raise ValueError('No source boundary/diversity candidates')
                x=eligible[list(FEATURES)].to_numpy(float)/self.support.bank.feature_scale
                first=int(np.argmin(np.abs(eligible['UDI'].to_numpy()-protocol.threshold)))
                indices=[first];distance=np.full(len(x),np.inf)
                while len(indices)<min(10,len(x)):
                    distance=np.minimum(distance,np.linalg.norm(x-x[indices[-1]],axis=1));distance[indices]=-1
                    indices.append(int(np.argmax(distance)))
                for i in indices:
                    row=eligible.iloc[i]
                    candidates.append(dict(structure=row['structure'],role='source_boundary_or_diverse_train_structure',
                        source_reference={'city':source,'panel':'train_fit','id':int(row['id'])},
                        source_optical_features={k:float(row[k]) for k in FEATURES}))
        seen=set(existing);accepted=[];rejected=[]
        for sample in candidates:
            structure=' '.join(sample['structure'].split())
            if structure in seen:continue
            seen.add(structure)
            if not space.valid(structure):
                rejected.append(dict(sample,reason='outside_current_search_space'));continue
            try:
                props=self.support.tmm.compute(structure)
                features={k:float(props[k]) for k in FEATURES}
                prediction=self.support.predictor.predict(features)
                if sample.get('source_optical_features'):
                    difference=max(abs(features[k]-sample['source_optical_features'][k]) for k in FEATURES)
                    if difference>1e-6:raise ValueError(f'Source anchor/TMM mismatch: {difference}')
                accepted.append(dict(sample,structure=structure,probe_id=f'probe_{len(accepted)+1:03d}',
                    optical_properties=features,prediction=prediction,target_observation=None,
                    selection_reason=sample['role'],preflight='passed_without_structure_repair'))
            except Exception as exc:
                rejected.append(dict(sample,reason=f'{type(exc).__name__}: {exc}'))
        # Start with one anchor per source; later probes cover new modes by
        # standardized optical distance, UDI boundary proximity and disagreement.
        selected=accepted[:min(len(self.support.bank.cities),limit)]
        remaining=accepted[len(selected):]
        while remaining and len(selected)<limit:
            def value(row):
                x=np.array([row['optical_properties'][k] for k in FEATURES])
                novelty=min((np.linalg.norm((x-np.array([s['optical_properties'][k] for k in FEATURES]))/
                    self.support.bank.feature_scale) for s in selected),default=1.)
                p=row['prediction'];threshold=self.support.predictor.protocol.threshold
                return novelty+.25/(1+abs(p['UDI']-threshold)/.02)+p['Energy_std']/self.support.predictor.protocol.energy
            chosen=max(remaining,key=value);selected.append(chosen);remaining.remove(chosen)
        for i,row in enumerate(selected,1):
            row['adaptation_role']='validation' if i%4==0 else 'fit'
        return {'schema_version':1,'city':self.support.city,'execution_ready':True,
                'support_version':self.support.predictor.VERSION,'samples':selected,
                'n_requested':limit,'n_preflight_rejected':len(rejected),'rejected':rejected,
                'budget_policy':'probe and search attempts share the Agent budget; dispatched failures count',
                'assets':self.support.context,'no_target_labels_used':True}


class CitySupportTool:
    def __init__(self, city, context, formal, tmm):
        from src.support_settings import settings
        cfg=settings()
        if cfg.get('require_transfer_selection'):
            path=Path(cfg['transfer']['selection_file'])
            if not path.is_file():raise ValueError('Four-source LOCO calibration must finish before target-city optimization')
            selected=json.loads(path.read_text(encoding='utf-8'))
            from src.runtime_profile import registry
            if selected.get('status')!='PASS' or set(selected['source_cities'])!=set(registry()['source_cities']):
                raise ValueError('Transfer calibration/source-city generation mismatch')
        self.city,self.context,self.formal,self.tmm=city,context,formal,tmm
        self.observations=[];self.directory=None
        self.tree_fit_attempted=False
        self.tree_decision={'decision':'not_attempted','reason':'waiting_for_target_fit_and_validation_labels'}

    def prepare(self):
        from src.support_settings import settings
        self.bank=SourceCityBank()
        configuration=settings().get('transfer',{})
        self.predictor=TransferSurrogate(self.bank,self.formal,climate_features(self.context['epw']),
            temperature=float(configuration.get('temperature',1.)),equal=configuration.get('equal_weights',False))
        self.prior_provider=TransferPriorProvider(self.bank,self.predictor,self.tmm)
        self.planner=ProbePlanner(self)
        return self

    def bind(self, run_directory):
        self.directory=Path(run_directory)/'city_support';self.directory.mkdir(parents=True,exist_ok=True)
        adapter=self.directory/'adapter_v001'
        write_json(adapter/'metadata.json',{'version':self.predictor.VERSION,'kind':'fixed_normalized_source_experts',
            'source_cities':list(self.bank.cities),'feature_order':list(FEATURES),
            'target_training_observation_ids':[],
            'configuration_selection':'source-only LOCO calibration; see settings transfer.selection_file',
            'online_bias_owner':'Agent BiasCorrector, tracked separately in bias_history.jsonl',
            'uncertainty_status':'target_uncalibrated_heuristic'})
        write_json(adapter/'model_parameters.json',{'weights':self.predictor.weights,
            'climate_scales':CLIMATE_SCALES,'feature_scale':self.bank.feature_scale,
            'target_baseline':{'energy_J':self.formal.energy,'clear_glass_UDI':self.formal.clear_udi},
            'source_baselines':{c:{'energy_J':p.energy,'clear_glass_UDI':p.clear_udi} for c,p in self.bank.protocols.items()}})
        self._save()

    def propose_probes(self, agent_state, limit):
        plan=self.planner.plan(agent_state.space,min(limit,agent_state.budget_total-agent_state.budget_used),agent_state.queried)
        plan.update(seed=agent_state.seed,total_design_budget=agent_state.budget_total,
                    udi_ratio=agent_state.formal.udi_ratio,search_space=agent_state.space.description())
        if self.directory:write_json(self.directory/'probe_plan.json',plan)
        return plan

    def observe(self, observation):
        if any(r['query_index']==observation['query_index'] for r in self.observations):return
        self.observations.append(observation)
        self._save()

    def next_observation_role(self):
        from src.support_settings import settings
        config=settings().get('tree_adaptation',{})
        if config.get('enabled') and not self.tree_fit_attempted:
            count=sum(r.get('energyplus_verified') and r.get('adaptation_role')=='validation' for r in self.observations)
            if count<config.get('min_validation',4):return 'validation'
        return 'fit'

    def fit_or_keep(self, agent_state=None):
        from src.support_settings import settings
        config=settings().get('tree_adaptation',{})
        if not config.get('enabled',False):
            return {'decision':'keep_base_experts','version':self.predictor.VERSION,'reason':'tree adaptation disabled'}
        if agent_state is None or self.tree_fit_attempted:return self.tree_decision
        fit=[r for r in self.observations if r.get('energyplus_verified') and r.get('adaptation_role','fit')=='fit']
        validation=[r for r in self.observations if r.get('energyplus_verified') and r.get('adaptation_role')=='validation']
        if len(fit)<config.get('min_fit',12) or len(validation)<config.get('min_validation',4):
            self.tree_decision={'decision':'keep_insufficient_target_labels','n_fit':len(fit),'n_validation':len(validation)}
            self._save();return self.tree_decision
        from src.tools.tree_transfer import TreeTransferSurrogate,compare_candidate,replay_frames
        self.tree_fit_attempted=True
        candidate=TreeTransferSurrogate.fit(self.predictor,fit,replay_frames(self.bank),
            parameters=config.get('parameters'),seed=agent_state.seed)
        gate=compare_candidate(self.predictor,candidate,fit,validation)
        folder=self.directory/'tree_candidate_v002'
        candidate.save(folder,[r['query_index'] for r in validation])
        write_json(folder/'validation_gate.json',gate)
        self.tree_decision={**gate,'decision':'publish_continued_trees' if gate['accepted'] else 'keep_gate_rejected',
            'candidate_directory':str(folder),'n_fit':len(fit),'n_validation':len(validation),
            'version':candidate.VERSION if gate['accepted'] else self.predictor.VERSION}
        if gate['accepted']:
            # Verify roundtrip before changing the live Agent predictor.
            restored=TreeTransferSurrogate.load(folder,self.predictor)
            for row in fit+validation:
                left=candidate.predict(row['optical_properties']);right=restored.predict(row['optical_properties'])
                if not np.allclose([left['UDI'],left['TotalEnergy']],[right['UDI'],right['TotalEnergy']],rtol=1e-7):
                    raise RuntimeError('Tree adapter roundtrip failed before publication')
            self.predictor=restored;self.prior_provider.predictor=restored
        self._save()
        return self.tree_decision

    def describe(self):
        return {'schema_version':1,'city':self.city,'mode':'transfer','version':self.predictor.VERSION,
                'sources':{c:{**self.bank.assets[c],'numerical_validation':self.bank.models[c].numerical_validation}
                           for c in self.bank.cities},
                'weights':self.predictor.weights,'climate_distances':self.predictor.distances,
                'climates':{self.city:climate_features(self.context['epw']),**self.bank.climates},
                'uncertainty_status':'target_uncalibrated_heuristic',
                'n_attempts':len(self.observations),'n_valid_labels':sum(r['energyplus_verified'] for r in self.observations),
                'n_probe_attempts':sum(r.get('phase')=='city_support_probe' for r in self.observations),
                'valid_observation_ids':[r['query_index'] for r in self.observations if r['energyplus_verified']],
                'failed_observation_ids':[r['query_index'] for r in self.observations if not r['energyplus_verified']],
                'adapter_decision':self.fit_or_keep(),'target_training_database_required':False,
                'online_residual_owner':'Agent BiasCorrector; observations are fitted once against raw predictions',
                'reference_baseline_cost':'one separately recorded clear-glass annual simulation',
                'query_resume_scope':'query/probe ledger only; complete Agent policy resume is not implemented'}

    def _save(self):
        if self.directory:write_json(self.directory/'support_manifest.json',self.describe())
