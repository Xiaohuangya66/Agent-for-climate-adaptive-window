"""Reload the server's actual adapter and evaluate untouched family-held-out labels."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.neighbors import NearestNeighbors

from training_common import (FEATURES,TARGETS,SPLITS,read_config,validate_split,write_json,
                             calibrated_multipliers)


def correlation(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None


def family_bootstrap_coverage(covered,families,repetitions=1000):
    frame=pd.DataFrame({'covered':np.asarray(covered,dtype=float),'family':np.asarray(families)})
    grouped=frame.groupby('family').covered.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(20261026)
    indexes=rng.integers(0,len(grouped),size=(repetitions,len(grouped)))
    values=grouped[indexes].sum(axis=1)
    coverage=values[:,0]/values[:,1]
    return np.quantile(coverage,[.025,.975]).tolist()


def metrics(y,mean,std,half,families,target,baseline):
    if not len(y):return dict(rows=0)
    error=np.abs(y-mean);covered=error<=half
    result=dict(rows=len(y),families=len(set(families)),
        r2=float(r2_score(y,mean)) if len(y)>1 and np.ptp(y)>0 else None,
        mae=float(mean_absolute_error(y,mean)),rmse=float(np.sqrt(mean_squared_error(y,mean))),
        bias=float(np.mean(mean-y)),uncalibrated_90pct_coverage=float(np.mean(error<=1.6448536269514722*std)),
        calibrated_90pct_coverage=float(covered.mean()),
        coverage_family_bootstrap_95pct_ci=family_bootstrap_coverage(covered,families),
        mean_90pct_interval_width=float((2*half).mean()),median_90pct_interval_width=float(np.median(2*half)),
        uncertainty_error_spearman=correlation(std,error),
        zero_ensemble_spread_rows=int((std<=np.finfo(float).eps).sum()))
    if target=='UDI':
        result.update(mae_pp=100*result['mae'],rmse_pp=100*result['rmse'],
                      mean_90pct_interval_width_pp=100*result['mean_90pct_interval_width'])
    else:
        result.update(mae_kWh=result['mae']/3.6e6,rmse_kWh=result['rmse']/3.6e6,
                      mae_pct_baseline=100*result['mae']/baseline,
                      rmse_pct_baseline=100*result['rmse']/baseline,
                      mean_90pct_interval_width_pct_baseline=100*result['mean_90pct_interval_width']/baseline)
    return result


def gates_for(result,gates):
    udi=result['targets']['UDI']['all_test'];energy=result['targets']['TotalEnergy']['all_test']
    checks=dict(udi_accuracy=udi['r2'] is not None and udi['r2']>=gates['udi_min_r2'] and udi['mae_pp']<=gates['udi_max_mae_pp'],
                energy_accuracy=energy['r2'] is not None and energy['r2']>=gates['energy_min_r2'] and energy['mae_pct_baseline']<=gates['energy_max_mae_pct_baseline'],
                udi_coverage=gates['coverage_min']<=udi['calibrated_90pct_coverage']<=gates['coverage_max'],
                energy_coverage=gates['coverage_min']<=energy['calibrated_90pct_coverage']<=gates['coverage_max'],
                udi_interval_width=udi['mean_90pct_interval_width_pp']<=gates['udi_max_mean_interval_width_pp'],
                energy_interval_width=energy['mean_90pct_interval_width_pct_baseline']<=gates['energy_max_mean_interval_width_pct_baseline'])
    return dict(checks=checks,status='PASS' if all(checks.values()) else 'REVIEW',
                note='Predeclared operational screening, not proof of optimization or transfer performance')


def evaluate_city(run,city,Surrogate):
    config=read_config(run);directory=run/'models'/city
    adapter=Surrogate(directory)
    metadata=adapter.metadata
    if metadata['training_protocol']!=config['protocol']:raise ValueError('Model protocol mismatch')
    train=pd.read_csv(run/'datasets'/f'{city}_train_fit.csv.gz')
    calibration=pd.read_csv(run/'datasets'/f'{city}_calibration.csv.gz')
    test=pd.read_csv(run/'datasets'/f'{city}_test_family_ood.csv.gz')
    scaler=metadata['x_scaler'];yscaler=metadata['y_scaler']
    for key,expected in [('mean',train[FEATURES].to_numpy().mean(axis=0)),('scale',train[FEATURES].to_numpy().std(axis=0))]:
        expected=np.where(expected==0,1,expected) if key=='scale' else expected
        if not np.allclose(scaler[key],expected,rtol=1e-12,atol=1e-12):raise ValueError('Feature scaler was not fit on train only')
    if not np.allclose(yscaler['mean'],train[TARGETS].to_numpy().mean(axis=0),rtol=1e-12,atol=1e-12):
        raise ValueError('Target scaler was not fit on train only')
    # Exercise the actual dictionary interface, including its inference probes.
    probe=adapter.predict(dict(zip(FEATURES,calibration.iloc[0][FEATURES].to_numpy(dtype=float))))
    cal_prediction=adapter.ensemble.predict(calibration[FEATURES].to_numpy(dtype=float))
    y_cal=calibration[TARGETS].to_numpy(dtype=float)
    recalculated=[]
    for index,target in enumerate(TARGETS):
        from training_common import conformal_quantile
        scores=np.abs(y_cal[:,index]-cal_prediction.mean[:,index])/np.maximum(cal_prediction.std[:,index],np.finfo(float).eps)
        multiplier,rank=conformal_quantile(scores,config['conformal_alpha'])
        if not np.isclose(multiplier,metadata['conformal_90pct_multiplier'][index],rtol=1e-10,atol=1e-12):
            raise ValueError(f'{city}/{target}: calibration-only multiplier cannot be reproduced')
        if rank!=metadata['conformal_calibration'][target]['finite_sample_rank_1_based']:
            raise ValueError('Conformal rank mismatch')
        recalculated.append(multiplier)
    prediction=adapter.ensemble.predict(test[FEATURES].to_numpy(dtype=float))
    if not np.isfinite(prediction.mean).all() or not np.isfinite(prediction.std).all():raise ValueError('Nonfinite inference')
    if not (prediction.mean[:,1]>0).all():raise ValueError('Nonpositive predicted energy')
    baseline=pd.read_csv(run/'city_baselines.csv').set_index('city').loc[city]
    operating_ratio=float(config.get('operating_udi_ratio',.95))
    threshold=operating_ratio*float(baseline.UDI)
    true_feasible=test.UDI.to_numpy()>=threshold
    near_threshold=np.abs(test.UDI.to_numpy()-threshold)<=.05
    selected=prediction.mean[:,0]>=threshold
    conservative=prediction.mean[:,0]-prediction.conformal_half_width_90pct[:,0]>=threshold
    count_selected=int(selected.sum());count_conservative=int(conservative.sum())
    masks=dict(all_test=np.ones(len(test),dtype=bool),true_udi_feasible=true_feasible,near_udi_threshold=near_threshold)
    result=dict(city=city,baseline_energy_J=float(baseline.TotalEnergy_J),clear_glass_udi=float(baseline.UDI),
        operating_udi_ratio=operating_ratio,udi_threshold=threshold,
        baseline_reference_threshold_95pct=float(baseline.UDI_threshold_95pct),split_counts=config['split_counts'],
        actual_server_adapter='src.components.surrogate.Surrogate',
        inference_validation=adapter.numerical_validation,calibration_multiplier_reproduced=True,
        udi_predictions_outside_physical_bounds=int(((prediction.mean[:,0]<0)|(prediction.mean[:,0]>1)).sum()),
        ood_distance_rate=float((prediction.ood_distance>metadata['ood_distance_threshold_95pct_train']).mean()),
        feasibility=dict(actual_feasible_rows=int(true_feasible.sum()),mean_selected_rows=count_selected,
            mean_false_feasible_rows=int((selected&~true_feasible).sum()),
            mean_false_feasible_rate=float((selected&~true_feasible).sum()/count_selected) if count_selected else None,
            conservative_selected_rows=count_conservative,conservative_false_feasible_rows=int((conservative&~true_feasible).sum()),
            conservative_false_feasible_rate=float((conservative&~true_feasible).sum()/count_conservative) if count_conservative else None),
        targets={})
    for index,target in enumerate(TARGETS):
        result['targets'][target]={}
        for subset,mask in masks.items():
            result['targets'][target][subset]=metrics(test[target].to_numpy()[mask],prediction.mean[mask,index],
                prediction.std[mask,index],prediction.conformal_half_width_90pct[mask,index],
                test.material_family.to_numpy()[mask],target,float(baseline.TotalEnergy_J))
    result['energy_rank_spearman_on_true_feasible']=correlation(test.TotalEnergy.to_numpy()[true_feasible],prediction.mean[true_feasible,1])
    result['acceptance']=gates_for(result,config['acceptance_gates'])
    frame=test[['id','structure','material_family']].copy()
    frame['true_udi_feasible']=true_feasible
    for index,target in enumerate(TARGETS):
        frame[f'{target}_true']=test[target]
        frame[f'{target}_mean']=prediction.mean[:,index]
        frame[f'{target}_std']=prediction.std[:,index]
        frame[f'{target}_half_width_90']=prediction.conformal_half_width_90pct[:,index]
    frame.to_csv(run/'evaluation'/f'{city}_test_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(run/'evaluation'/f'{city}_metrics.json',result)
    print('CITY_ACCEPTED',city,result['acceptance']['status'],flush=True)
    return result


def write_report(run,config,results):
    rows=[]
    for city in results:
        for target,subsets in city['targets'].items():
            for subset,item in subsets.items():
                rows.append(dict(city=city['city'],target=target,subset=subset,acceptance=city['acceptance']['status'],**item))
    frame=pd.DataFrame(rows)
    frame.to_csv(run/'evaluation/metrics.csv',index=False,encoding='utf-8-sig')
    write_json(run/'evaluation/metrics.json',results)
    udi=frame[(frame.target=='UDI')&(frame.subset=='all_test')].set_index('city')
    energy=frame[(frame.target=='TotalEnergy')&(frame.subset=='all_test')].set_index('city')
    lines=[f"# {len(config['cities'])}个源城市代理模型验收（迁移目标：{'、'.join(config['held_out_climate_cities'])}）", '',
           f"天气：{config['weather_dataset']} {config['weather_period']}；共同结构 {config['common_structures']}；材料序列分组 {config['material_families']}。",'',
           '## 数据划分','', '| 划分 | 结构数/每城市 | 材料序列组数 |','|---|---:|---:|']
    lines += [f"| {split} | {value['rows']} | {value['families']} |" for split,value in config['split_counts'].items()]
    lines += ['', f"{len(config['cities'])}个源城市共享结构划分；材料序列在训练、校准、测试间无交集。迁移城市标签不进入源模型训练。训练程序不读取测试分区。", '',
              '## 独立测试集结果','',
              '| 城市 | UDI R² | UDI MAE/百分点 | 能耗 R² | 能耗 MAE/kWh | 能耗 MAE/清玻璃能耗% | 工程门槛 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for item in results:
        city=item['city'];u=udi.loc[city];e=energy.loc[city]
        lines.append(f"| {city} | {u.r2:.5f} | {u.mae_pp:.4f} | {e.r2:.5f} | {e.mae_kWh:.2f} | {e.mae_pct_baseline:.4f} | {item['acceptance']['status']} |")
    lines += ['', '## 不确定性', '',
              '| 城市 | UDI 校准前覆盖率% | 校准后覆盖率% | UDI 平均区间全宽/百分点 | 能耗校准前覆盖率% | 校准后覆盖率% | 能耗平均区间全宽/清玻璃能耗% |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in results:
        city=item['city'];u=udi.loc[city];e=energy.loc[city]
        lines.append(f"| {city} | {100*u.uncalibrated_90pct_coverage:.2f} | {100*u.calibrated_90pct_coverage:.2f} | {u.mean_90pct_interval_width_pp:.4f} | {100*e.uncalibrated_90pct_coverage:.2f} | {100*e.calibrated_90pct_coverage:.2f} | {e.mean_90pct_interval_width_pct_baseline:.4f} |")
    lines += ['', '## 可行域与不确定性排序', '',
              '| 城市 | 实际满足 UDI 阈值的测试样本 | 可行子集能耗 MAE/基线% | 仅按预测均值筛选的误可行数 | 按 UDI 90% 区间下界筛选的误可行数 | UDI 不确定性–误差相关 | 能耗不确定性–误差相关 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in results:
        city=item['city'];u=udi.loc[city];e=energy.loc[city]
        feasible=item['feasibility'];subset=item['targets']['TotalEnergy']['true_udi_feasible']
        lines.append(f"| {city} | {feasible['actual_feasible_rows']} | {subset['mae_pct_baseline']:.4f} | {feasible['mean_false_feasible_rows']} | {feasible['conservative_false_feasible_rows']} | {u.uncertainty_error_spearman:.3f} | {e.uncertainty_error_spearman:.3f} |")
    undercovered=[item['city'] for item in results if item['targets']['TotalEnergy']['all_test']['coverage_family_bootstrap_95pct_ci'][1]<.9]
    correlations=[item['targets']['TotalEnergy']['all_test']['uncertainty_error_spearman'] for item in results]
    correlations=[value for value in correlations if value is not None]
    lines += ['', '## 验收发现与接入注意', '',
              '- 能耗覆盖率的材料组 bootstrap 区间上限仍低于名义 90% 的城市：'+('、'.join(undercovered) if undercovered else '无')+'。工程筛查通过不等于精确达到 90% 覆盖。',
              f'- 能耗集成标准差与实际误差的相关性范围为 {min(correlations):.3f}–{max(correlations):.3f}。校准扩大区间改善覆盖率，没有改变标准差对候选的排序；不应把它单独作为信息价值或查询优先级依据。',
              '- 原始回归 UDI 在接近 0 处有少量越界预测。主表保持原始输出便于复核；正式接入时应对 UDI 点预测和区间端点应用 [0,1] 物理边界，并记录原值。不能把原始模型称为严格物理有界。', '',
              f"| 城市 | 原始 UDI 越界样本数/{config['split_counts']['test_family_ood']['rows']} |", '|---|---:|']
    lines += [f"| {item['city']} | {item['udi_predictions_outside_physical_bounds']} |" for item in results]
    lines += ['', '区间目标覆盖率为 90%。这里只报告材料序列留出测试中的实际表现，不宣称对新结构分布或新气候存在无条件覆盖保证。指标 JSON 另含以材料序列组为重采样单位的覆盖率区间、误差与集成标准差的 Spearman 相关性、可行子集与阈值附近子集的误差，以及基于 UDI 下界筛选的误判数。', '',
              '## 验收边界','',
              '- 精度门槛：UDI R²≥0.95 且 MAE≤2 个百分点；能耗 R²≥0.90 且 MAE≤清玻璃能耗 0.5%。',
              '- 区间筛查：实测覆盖率 85%–95%；平均全宽 UDI≤10 个百分点、能耗≤清玻璃能耗 3%。',
              '- 以上是本次预先固定的工程筛查值，不是通用科研标准。超过上界表示区间偏保守，也会标为 REVIEW。',
              f"- 模型来自5个固定种子；每成员训练两个目标，本次共{10*len(config['cities'])}个回归器。",
              '- X/y 标准化仅使用训练集；分位数仅使用校准集；验收使用服务器真实 Surrogate 加载接口，并检查原始标量截距与数值探针。',
              '- 本轮模型保存在独立目录，没有替换现有 Agent 的默认模型、天气或基线。',
              '- 本轮不包含真实优化性能、新气候迁移性能或新增 EnergyPlus 验证。', '']
    (run/'evaluation/acceptance_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--server-project-root',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve();config=read_config(run)
    if not (run/'training_complete.json').exists():raise RuntimeError('Training incomplete')
    manifest=pd.read_csv(run/'grouped_split.csv');validate_split(manifest)
    for city in config['cities']:
        for split in SPLITS:
            partition=pd.read_csv(run/'datasets'/f'{city}_{split}.csv.gz',usecols=['structure','material_family','split'])
            expected=manifest.loc[manifest['split']==split,['structure','material_family','split']].reset_index(drop=True)
            if not partition.equals(expected):raise ValueError('City-specific split differs from shared manifest')
    sys.path.insert(0,str(args.server_project_root.resolve()))
    from src.components.surrogate import Surrogate
    results=[evaluate_city(run,city,Surrogate) for city in config['cities']]
    write_report(run,config,results)
    write_json(run/'acceptance_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        cities={r['city']:r['acceptance']['status'] for r in results},actual_server_adapter_verified=True,
        passed_cities=sum(r['acceptance']['status']=='PASS' for r in results),new_energyplus_queries=0))


if __name__=='__main__':main()
