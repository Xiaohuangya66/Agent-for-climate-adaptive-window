"""Prepare shared grouped splits and fit four source-city calibrated ensembles.

The fit command reads only the fit and calibration partitions. Test labels are
opened exclusively by the separate acceptance program, after fitting finishes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

from training_common import (FEATURES,TARGETS,SEEDS,SPLITS,ENERGY_DEFINITION,PROTOCOL,
    MODEL_PARAMETERS,GATES,family,make_split,validate_split,calibrated_multipliers,
    exclusive_lock,read_config,write_json,predict_members)


def check_frame(frame,city):
    if frame.structure.isna().any() or frame.structure.duplicated().any():raise ValueError(f'{city}: invalid/duplicate structures')
    if not frame.structure.equals(frame.structure.str.strip().str.replace(r'\s+',' ',regex=True)):
        raise ValueError(f'{city}: noncanonical structure strings')
    frame.structure.map(family)
    numeric=[*FEATURES,*TARGETS,'ElectricityFacility_J','NaturalGasFacility_J','UDI_qualifying_hours','UDI_denominator_hours']
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():raise ValueError(f'{city}: missing/nonfinite values')
    if not frame.UDI.between(0,1).all() or not (frame.TotalEnergy>0).all():raise ValueError(f'{city}: invalid targets')
    if (frame[['ElectricityFacility_J','NaturalGasFacility_J']]<0).any().any():raise ValueError(f'{city}: negative meters')
    if not np.allclose(frame.TotalEnergy,frame.ElectricityFacility_J+frame.NaturalGasFacility_J,rtol=1e-10,atol=.01):
        raise ValueError(f'{city}: energy definition mismatch')
    if not (frame.UDI_denominator_hours>0).all() or not np.allclose(frame.UDI,frame.UDI_qualifying_hours/frame.UDI_denominator_hours,atol=1e-12):
        raise ValueError(f'{city}: UDI denominator mismatch')
    if (frame[FEATURES]<-1e-5).any().any() or (frame[FEATURES]>1+1e-5).any().any():
        raise ValueError(f'{city}: invalid optical features')


def prepare(campaign, run, **options):
    from source_prepare import prepare_source
    return prepare_source(campaign, run, check_frame=check_frame, **options)


def scalar_base_score(booster):
    raw=json.loads(booster.save_config())['learner']['learner_model_param']['base_score']
    values=np.asarray(json.loads(raw),dtype=float).reshape(-1)
    if values.shape!=(1,) or not np.isfinite(values).all():raise ValueError('Expected one finite scalar intercept per regressor')
    return float(values[0])


def train_city(run,city,resume=False):
    run=Path(run).resolve();config=read_config(run)
    if city not in config['cities']:raise ValueError(city)
    directory=run/'models'/city;directory.mkdir(exist_ok=True)
    with exclusive_lock(directory/'training.lock'):
        if (directory/'complete.json').exists():
            if resume:return city
            raise FileExistsError(f'Already trained: {city}')
        train=pd.read_csv(run/'datasets'/f'{city}_train_fit.csv.gz')
        calibration=pd.read_csv(run/'datasets'/f'{city}_calibration.csv.gz')
        # No test partition is opened in this function.
        x_scaler=StandardScaler().fit(train[FEATURES].to_numpy(dtype=float))
        y_scaler=StandardScaler().fit(train[TARGETS].to_numpy(dtype=float))
        x_train=x_scaler.transform(train[FEATURES].to_numpy(dtype=float))
        y_train=y_scaler.transform(train[TARGETS].to_numpy(dtype=float))
        x_cal=x_scaler.transform(calibration[FEATURES].to_numpy(dtype=float))
        np.save(directory/'train_features_scaled.npy',x_train.astype(np.float32))
        descriptions=[];boosters=[];scores={}
        for member,seed in enumerate(config['member_seeds']):
            member_models=[];filenames=[]
            for index,target in enumerate(TARGETS):
                estimator=xgb.XGBRegressor(**config['model_parameters'],random_state=seed)
                estimator.fit(x_train,y_train[:,index],verbose=False)
                booster=estimator.get_booster()
                filename=f'member_{member:02d}_{target}.ubj'
                booster.save_model(directory/filename)
                scores[filename]=scalar_base_score(booster)
                filenames.append(filename);member_models.append(booster)
            descriptions.append(dict(seed=seed,models=filenames));boosters.append(member_models)
            print('MEMBER_DONE',city,member,flush=True)
        cal_members=predict_members(boosters,x_cal,y_scaler.mean_,y_scaler.scale_)
        multipliers,calibration_details=calibrated_multipliers(calibration[TARGETS].to_numpy(dtype=float),cal_members,config['conformal_alpha'])
        nearest=NearestNeighbors(n_neighbors=2,n_jobs=1).fit(x_train)
        train_distances=nearest.kneighbors(x_train)[0][:,1]
        metadata=dict(schema_version=1,city=city,training_protocol=PROTOCOL,feature_order=FEATURES,target_order=TARGETS,
            source_dataset=config['source_audit'][city]['source_csv'],
            source_dataset_relative=str(Path(config['source_audit'][city]['source_csv']).relative_to(Path(__file__).resolve().parents[2])),
            target_definition=dict(UDI='daylight-hours fraction (solar altitude > 0) with 100-2000 lux',
                                   TotalEnergy='annual Electricity:Facility + NaturalGas:Facility in joules'),
            weather_dataset=config['weather_dataset'],weather_period=config['weather_period'],
            x_scaler=dict(mean=x_scaler.mean_,scale=x_scaler.scale_),
            y_scaler=dict(mean=y_scaler.mean_,scale=y_scaler.scale_),
            members=descriptions,train_features_file='train_features_scaled.npy',
            conformal_90pct_multiplier=multipliers,conformal_calibration=calibration_details,
            ood_distance_threshold_95pct_train=float(np.quantile(train_distances,.95)),
            split_manifest=str(run/'grouped_split.csv'),split_counts=config['split_counts'],runtime=config['runtime'])
        write_json(directory/'metadata.json',metadata)
        write_json(directory/'inference_compatibility.json',dict(objective='reg:squarederror',
                   source_xgboost=xgb.__version__,scalar_base_scores=scores))
        probe_count=min(12,len(calibration))
        cal_mean=cal_members.mean(axis=0);cal_std=cal_members.std(axis=0,ddof=1)
        write_json(directory/'inference_probes.json',dict(features=calibration[FEATURES].to_numpy()[:probe_count],
            mean=cal_mean[:probe_count],std=cal_std[:probe_count],half_width=(cal_std*multipliers)[:probe_count],
            source_partition='calibration'))
        prediction=calibration[['structure','material_family']].copy()
        for index,target in enumerate(TARGETS):
            prediction[f'{target}_true']=calibration[target]
            prediction[f'{target}_mean']=cal_mean[:,index]
            prediction[f'{target}_std']=cal_std[:,index]
            prediction[f'{target}_half_width_90']=cal_std[:,index]*multipliers[index]
        prediction.to_csv(directory/'calibration_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        write_json(directory/'complete.json',dict(city=city,status='trained',completed_utc=datetime.now(timezone.utc).isoformat(),
                                                 train_rows=len(train),calibration_rows=len(calibration)))
        print('CITY_TRAINED',city,flush=True)
    return city


def train_all(run,workers,resume):
    run=Path(run).resolve();config=read_config(run)
    if workers<1 or workers>int(os.environ.get('SLURM_CPUS_PER_TASK',workers)):
        raise ValueError('Worker count exceeds allocated CPUs')
    with exclusive_lock(run/'training_run.lock'):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures=[executor.submit(train_city,str(run),city,resume) for city in config['cities']]
            for future in as_completed(futures):print('TRAINING_PROGRESS',future.result(),flush=True)
    write_json(run/'training_complete.json',dict(cities=config['cities'],completed_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare_parser=sub.add_parser('prepare');prepare_parser.add_argument('--campaign-root',type=Path,required=True);prepare_parser.add_argument('--run-dir',type=Path,required=True)
    prepare_parser.add_argument('--baseline-table',type=Path)
    prepare_parser.add_argument('--split-mode',choices=['frozen','recompute'],default='frozen')
    prepare_parser.add_argument('--split-manifest',type=Path)
    train_parser=sub.add_parser('train');train_parser.add_argument('--run-dir',type=Path,required=True);train_parser.add_argument('--workers',type=int,default=3);train_parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.campaign_root,args.run_dir,baseline_table=args.baseline_table,
        split_mode=args.split_mode,split_manifest=args.split_manifest)
    else:train_all(args.run_dir,args.workers,args.resume)


if __name__=='__main__':main()
