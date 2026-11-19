"""Result metrics with explicit feasibility and missing-value semantics."""
import math
import numpy as np


def _finite(value):
    return isinstance(value,(int,float,np.number)) and math.isfinite(float(value))


def compute_energy_saving(energy,baseline):
    if baseline<=0:return 0.
    return float(np.clip((baseline-energy)/baseline,-1.,1.))


def compute_pareto_dominance(results):
    rows=[r for r in results if _finite(r.get('energy')) and _finite(r.get('udi'))]
    return [r for r in rows if not any(o['energy']<=r['energy'] and o['udi']>=r['udi'] and
        (o['energy']<r['energy'] or o['udi']>r['udi']) for o in rows)]


def compute_hypervolume(pareto_front,reference_point):
    rows=compute_pareto_dominance(pareto_front)
    if not rows:return 0.
    ref_energy=reference_point.get('energy',max(r['energy'] for r in rows))
    ref_udi=reference_point.get('udi',min(r['udi'] for r in rows))
    area=0.;highest=ref_udi
    for row in sorted(rows,key=lambda r:r['energy']):
        if row['energy']<ref_energy and row['udi']>highest:
            area+=(ref_energy-row['energy'])*(row['udi']-highest)
            highest=row['udi']
    return float(area)


def compute_convergence(query_history,udi_threshold=None):
    formal=udi_threshold is not None or any('direct_feasible' in r for r in query_history)
    best_saving=None;best_udi=None;savings=[];udis=[]
    for row in query_history:
        success=row.get('energyplus_verified',True)
        udi=row.get('udi');saving=row.get('energy_saving')
        if success and _finite(udi):best_udi=max(best_udi,udi) if best_udi is not None else udi
        feasible=(udi>=udi_threshold if success and _finite(udi) and udi_threshold is not None
                  else row.get('direct_feasible',False)) if formal else True
        if success and feasible and _finite(saving):
            best_saving=max(best_saving,saving) if best_saving is not None else saving
        savings.append(best_saving);udis.append(best_udi)
    return {'iterations':list(range(1,len(query_history)+1)),
            'best_energy_saving':savings,'best_udi':udis,
            'energy_scope':'direct_feasible_only' if formal else 'legacy_unconstrained',
            'udi_scope':'maximum_successful_udi'}


def compute_diversity(structures):
    if len(structures)<=1:return 0.
    materials=[set(t.split('_')[0] for t in s.split()) for s in structures]
    unique=set().union(*materials)
    counts=np.asarray([sum(m in row for row in materials) for m in unique],dtype=float)
    if len(counts)<=1 or counts.sum()==0:return 0.
    p=counts/counts.sum()
    return float(-(p*np.log(p)).sum()/np.log(len(counts)))


def summarize_results(results,method_name):
    if not results:return {'method':method_name,'n_runs':0}
    savings=[r['best_energy_saving'] for r in results if r.get('best_is_feasible') is not False and _finite(r.get('best_energy_saving'))]
    udis=[r['best_udi'] for r in results if _finite(r.get('best_udi'))]
    queries=[r['n_energyplus_queries'] for r in results if _finite(r.get('n_energyplus_queries'))]
    def stat(values,fn):return float(fn(values)) if values else None
    return {'method':method_name,'n_runs':len(results),
            'n_feasible_runs':sum(r.get('best_is_feasible') is True for r in results),
            'n_no_feasible_runs':sum(r.get('best_is_feasible') is False for r in results),
            'n_legacy_unverified_runs':sum('best_is_feasible' not in r for r in results),
            'energy_saving_mean':stat(savings,np.mean),'energy_saving_std':stat(savings,np.std),
            'energy_saving_max':stat(savings,np.max),'udi_mean':stat(udis,np.mean),
            'udi_std':stat(udis,np.std),'n_queries_mean':stat(queries,np.mean),
            'udi_scope':'selected_or_diagnostic_structure; check best_is_feasible'}
