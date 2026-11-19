"""Continue existing source trees with target labels and source replay.

The old trees are retained, new shallow trees are genuinely trained. Target
responses are first expressed in each source's baseline/scaler coordinates.
No target validation labels enter tree training or the comparison bias fit.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from src.tools.city_support import TransferSurrogate,FEATURES,write_json


DEFAULT_PARAMETERS={'rounds':24,'eta':.05,'max_depth':2,'min_child_weight':2.,
                    'reg_lambda':10.,'source_replay_ratio':.25,'replay_rows':256}


def observed_arrays(records):
    if not records:raise ValueError('Target fit requires real observations')
    if len({r['structure'] for r in records})!=len(records):raise ValueError('Duplicate target fit structures')
    if any(not r.get('energyplus_verified') for r in records):raise ValueError('Unverified target label')
    x=np.asarray([[r['optical_properties'][k] for k in FEATURES] for r in records],float)
    y=np.asarray([[r['udi'],r['energy']] for r in records],float)
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or
        np.any((x<0)|(x>1)) or np.any((y[:,0]<0)|(y[:,0]>1)) or np.any(y[:,1]<=0)):
        raise ValueError('Invalid target fit features/labels')
    return x,y


def virtual_source_labels(target_y, source_protocol, target_protocol, ensemble):
    """Target native labels -> baseline ratio -> source units -> source z-score."""
    target_baseline=np.array([target_protocol.clear_udi,target_protocol.energy])
    source_baseline=np.array([source_protocol.clear_udi,source_protocol.energy])
    return (np.asarray(target_y)/target_baseline*source_baseline-ensemble.y_mean)/ensemble.y_scale


class TreeTransferSurrogate(TransferSurrogate):
    VERSION='normalized_source_trees_continued_v002'
    allow_unbounded_source_udi=True

    @classmethod
    def fit(cls, base, records, replay_frames, parameters=None, seed=5):
        if any(r.get('adaptation_role')=='validation' for r in records):
            raise ValueError('Target validation labels cannot enter tree training')
        parameters={**DEFAULT_PARAMETERS,**(parameters or {})}
        if (int(parameters['rounds'])<1 or int(parameters['max_depth'])<1 or
            parameters['eta']<=0 or parameters['reg_lambda']<0 or parameters['source_replay_ratio']<=0):
            raise ValueError('Invalid tree continuation parameters')
        x,y=observed_arrays(records)
        bank=copy.copy(base.bank);bank.models={};details={}
        for source in base.bank.cities:
            native=base.bank.models[source];ensemble=native.ensemble
            if tuple(ensemble.targets)!=('UDI','TotalEnergy'):raise ValueError('Wrong response/scaler order')
            frame=replay_frames[source]
            if not frame['split'].eq('train_fit').all():raise ValueError('Source replay must use train_fit only')
            rx=frame[list(FEATURES)].to_numpy(float)
            ry=frame[['UDI','TotalEnergy']].to_numpy(float)
            if not np.isfinite(rx).all() or not np.isfinite(ry).all():raise ValueError('Invalid source replay')
            virtual=virtual_source_labels(y,base.bank.protocols[source],base.protocol,ensemble)
            clone=copy.copy(native);clone.ensemble=copy.copy(ensemble);clone.ensemble.models=[];clone._cache={}
            details[source]=[]
            for member,originals in enumerate(ensemble.models):
                rng=np.random.default_rng(seed+member*1009+list(base.bank.cities).index(source)*101)
                indices=rng.choice(len(frame),size=min(int(parameters['replay_rows']),len(frame)),replace=False)
                train_x=np.vstack([x,rx[indices]])
                train_y=np.vstack([virtual,(ry[indices]-ensemble.y_mean)/ensemble.y_scale])
                # All target rows have weight 1; replay TOTAL mass is a fraction
                # of target mass, independent of how many replay rows are stored.
                weights=np.r_[np.ones(len(x)),np.full(len(indices),len(x)*parameters['source_replay_ratio']/len(indices))]
                fitted=[];member_details=[]
                for response,original in enumerate(originals):
                    matrix=xgb.DMatrix((train_x-ensemble.x_mean)/ensemble.x_scale,
                                       label=train_y[:,response],weight=weights)
                    config=json.loads(original.save_config())
                    score=config['learner']['learner_model_param']['base_score']
                    score=float(str(score).strip('[]'))
                    params=dict(objective='reg:squarederror',tree_method='hist',nthread=1,
                        eta=float(parameters['eta']),max_depth=int(parameters['max_depth']),
                        min_child_weight=float(parameters['min_child_weight']),
                        reg_lambda=float(parameters['reg_lambda']),seed=seed+member,
                        base_score=score,boost_from_average=0,subsample=1.,colsample_bytree=1.)
                    old_rounds=original.num_boosted_rounds()
                    trained=xgb.train(params,matrix,num_boost_round=int(parameters['rounds']),
                        xgb_model=original.save_raw(raw_format='ubj'),verbose_eval=False)
                    probe=xgb.DMatrix((x-ensemble.x_mean)/ensemble.x_scale)
                    before=original.predict(probe)
                    prefix=trained.predict(probe,iteration_range=(0,old_rounds))
                    if not np.allclose(before,prefix,rtol=1e-7,atol=1e-7):
                        raise RuntimeError('Training altered the original model/intercept')
                    if trained.num_boosted_rounds()!=old_rounds+parameters['rounds']:
                        raise RuntimeError('Tree continuation did not append the requested rounds')
                    member_details.append({'response':ensemble.targets[response],
                        'source_rounds':old_rounds,'adapted_rounds':trained.num_boosted_rounds(),
                        'prefix_prediction_max_difference':float(np.max(np.abs(prefix-before))),
                        'new_tree_prediction_delta_max':float(np.max(np.abs(trained.predict(probe)-before)))})
                    fitted.append(trained)
                clone.ensemble.models.append(fitted)
                details[source].append({'member':member,'source_replay_ids':frame.iloc[indices]['id'].tolist(),
                    'replay_rows':len(indices),'target_weight_total':len(x),
                    'source_weight_total':len(x)*parameters['source_replay_ratio'],'models':member_details})
            bank.models[source]=clone
        # Reuse exactly the base climate weights; adaptation learns from target
        # labels instead of silently changing the source mixture.
        model=cls(bank,base.protocol,base.target_climate,base.temperature,base.equal_weights)
        model.training_metadata={'version':cls.VERSION,'xgboost':xgb.__version__,
            'method':'source checkpoint continuation with shallow added trees and source replay',
            'parameters':parameters,'fit_query_ids':[r['query_index'] for r in records],
            'fit_structures':[r['structure'] for r in records],
            'includes_valid_UDI_infeasible_rows':True,'source_models_unchanged':True,
            'source_details':details,'target_training_rows':len(records),
            'target_clear_UDI':base.protocol.clear_udi,'target_clear_energy_J':base.protocol.energy,
            'target_validation_labels_used_for_tree_fit':False}
        model.training_metadata['source_coordinates']={c:{
            'x_mean':base.bank.models[c].ensemble.x_mean.tolist(),
            'x_scale':base.bank.models[c].ensemble.x_scale.tolist(),
            'y_mean':base.bank.models[c].ensemble.y_mean.tolist(),
            'y_scale':base.bank.models[c].ensemble.y_scale.tolist(),
            'clear_UDI':base.bank.protocols[c].clear_udi,'clear_energy_J':base.bank.protocols[c].energy}
            for c in base.bank.cities}
        model.training_records=[{'query_index':r['query_index'],'structure':r['structure'],
            'UDI':r['udi'],'TotalEnergy':r['energy'],**r['optical_properties']} for r in records]
        return model

    def save(self,directory,validation_ids=()):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        metadata=copy.deepcopy(self.training_metadata)
        metadata.update(validation_query_ids=list(validation_ids),weights=self.weights,
                        uncertainty_status='target_uncalibrated_heuristic')
        for source in self.bank.cities:
            folder=directory/'models'/source;folder.mkdir(parents=True,exist_ok=True)
            for i,models in enumerate(self.bank.models[source].ensemble.models):
                for target,booster in zip(['UDI','TotalEnergy'],models):
                    booster.save_model(folder/f'member_{i:02d}_{target}.ubj')
        write_json(directory/'metadata.json',metadata)
        if hasattr(self,'training_records'):
            pd.DataFrame(self.training_records).to_csv(directory/'target_fit_rows.csv',index=False)

    @classmethod
    def load(cls,directory,base):
        directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text())
        if metadata['version']!=cls.VERSION:raise ValueError('Unsupported tree adapter version')
        if (metadata['target_clear_UDI']!=base.protocol.clear_udi or
            metadata['target_clear_energy_J']!=base.protocol.energy or metadata['weights']!=base.weights):
            raise ValueError('Tree adapter target baseline or source mixture changed')
        if set(metadata['source_details'])!=set(base.bank.cities):raise ValueError('Tree adapter sources changed')
        for c in base.bank.cities:
            coordinates=metadata['source_coordinates'][c];ensemble=base.bank.models[c].ensemble
            if any(not np.array_equal(getattr(ensemble,k),coordinates[k]) for k in ['x_mean','x_scale','y_mean','y_scale']):
                raise ValueError('Source tree scaler generation changed')
            if coordinates['clear_UDI']!=base.bank.protocols[c].clear_udi or coordinates['clear_energy_J']!=base.bank.protocols[c].energy:
                raise ValueError('Source clear baseline changed')
        bank=copy.copy(base.bank);bank.models={}
        for source in bank.cities:
            clone=copy.copy(base.bank.models[source]);clone.ensemble=copy.copy(clone.ensemble)
            clone.ensemble.models=[];clone._cache={}
            for i in range(len(base.bank.models[source].ensemble.models)):
                models=[]
                for response in ['UDI','TotalEnergy']:
                    booster=xgb.Booster();booster.load_model(directory/'models'/source/f'member_{i:02d}_{response}.ubj')
                    models.append(booster)
                clone.ensemble.models.append(models)
            bank.models[source]=clone
        result=cls(bank,base.protocol,base.target_climate,base.temperature,base.equal_weights)
        result.training_metadata=metadata
        return result


def corrected_predictions(model,fit_records,evaluation_records):
    from src.tools.bias_corrector import BiasCorrector
    bias=BiasCorrector(window_size=20)
    for row in fit_records:bias.update([model.predict(row['optical_properties'])],[row],prequential=False)
    predictions=np.array([[p['UDI'],p['TotalEnergy']] for p in
        (bias.correct(model.predict(row['optical_properties'])) for row in evaluation_records)])
    return predictions,bias


def compare_candidate(base,candidate,fit_records,validation_records):
    if {r['structure'] for r in fit_records}&{r['structure'] for r in validation_records}:
        raise ValueError('Target fit/validation structure leakage')
    _,truth=observed_arrays(validation_records)
    before,_=corrected_predictions(base,fit_records,validation_records)
    after,_=corrected_predictions(candidate,fit_records,validation_records)
    baseline=np.array([base.protocol.clear_udi,base.protocol.energy])
    mae_before=np.abs(before-truth).mean(axis=0)/baseline
    mae_after=np.abs(after-truth).mean(axis=0)/baseline
    # One precommitted small-sample deployment guard, not a significance test.
    improved=float(mae_after.sum())<float(mae_before.sum())*.98
    noninferior=bool(np.all(mae_after<=mae_before*1.10+np.array([.003,.001])))
    threshold=base.protocol.threshold
    false_feasible_before=int(((before[:,0]>=threshold)&(truth[:,0]<threshold)).sum())
    false_feasible_after=int(((after[:,0]>=threshold)&(truth[:,0]<threshold)).sum())
    accepted=bool(improved and noninferior and false_feasible_after<=false_feasible_before)
    return {'accepted':accepted,'n_validation':len(validation_records),
            'fit_query_ids':[r['query_index'] for r in fit_records],
            'validation_query_ids':[r['query_index'] for r in validation_records],
            'baseline_normalized_MAE':mae_before.tolist(),'candidate_normalized_MAE':mae_after.tolist(),
            'baseline_score':float(mae_before.sum()),'candidate_score':float(mae_after.sum()),
            'false_feasible_before':false_feasible_before,'false_feasible_after':false_feasible_after,
            'criterion':'sum normalized MAE improves >=2%; neither response degrades >10% + tolerance; no extra false feasible',
            'comparator':'same source mixture plus BiasCorrector fitted on the SAME target fit IDs',
            'caveat':'small online model-validation gate; not independent final optimization evidence'}


def replay_frames(bank):
    from src.runtime_profile import MODEL_RUN
    columns=['id','structure','split',*FEATURES,'UDI','TotalEnergy']
    return {c:pd.read_csv(MODEL_RUN/'datasets'/f'{c}_train_fit.csv.gz',usecols=columns) for c in bank.cities}
