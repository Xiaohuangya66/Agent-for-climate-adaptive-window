"""One direct-query budget and fact ledger for probes and optimization."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import numpy as np
from src.tools.city_support import FEATURES, write_json


@contextmanager
def writer_lock(directory):
    """Kernel lock releases on process death; never delete someone else's lock."""
    with (Path(directory)/'query_writer.lock').open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0,2)
            if stream.tell()==0:stream.write(b'0');stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def append_fact(path, payload):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(payload,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def read_facts(path):
    if not Path(path).exists():return []
    # A truncated final line is a reconciliation error, never silently dropped.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


class QueryService:
    def __init__(self, owner):
        self.owner=owner
        context={'city':owner.city,'seed':owner.seed,'total_design_budget':owner.budget_total,
                 'baseline_energy_J':owner.baseline_energy,'UDI_threshold':owner.formal_udi_threshold,
                 'runtime_assets':getattr(owner,'runtime_assets',None)}
        path=owner.run_directory/'query_context.json'
        with writer_lock(owner.run_directory):
            if path.exists():
                if json.loads(path.read_text(encoding='utf-8'))!=context:
                    raise ValueError('Query context changed: city, seed, budget, baseline or physical assets differ')
            else:write_json(path,context)

    def _complete(self, dispatch, result, recovered=False):
        a=self.owner;extra=dispatch['extra'];energy,udi=result.get('TotalEnergy'),result.get('UDI')
        valid=(result.get('status')=='success' and not isinstance(energy,bool) and not isinstance(udi,bool) and isinstance(energy,(int,float)) and
               math.isfinite(energy) and energy>0 and isinstance(udi,(int,float)) and
               math.isfinite(udi) and 0<=udi<=1)
        if not valid and result.get('status')=='success':
            result={**result,'status':'failed','error_code':'INVALID_METRICS',
                    'error_message':'Nonfinite or out-of-range direct metrics',
                    'reported_metrics':{'TotalEnergy':str(energy),'UDI':str(udi)},
                    'TotalEnergy':None,'UDI':None}
        row={**extra,'city':a.city,'seed':a.seed,'query_index':dispatch['query_index'],
             'structure':dispatch['structure'],'energyplus_attempted':True,'energyplus_verified':bool(valid),
             'energyplus_result':result,'energy':energy if valid else None,'udi':udi if valid else None,
             'energy_saving':(a.baseline_energy-energy)/a.baseline_energy if valid else None,
             'direct_feasible':bool(valid and udi>=a.formal_udi_threshold),'recovered_from_result':recovered}
        if valid and row.get('predicted_udi') is not None:
            row['surrogate_udi_error']=float(udi-row['predicted_udi']);half=row.get('udi_half_width_90pct')
            row['surrogate_interval_covers_direct_udi']=abs(row['surrogate_udi_error'])<=half if half is not None else None
        append_fact(a.run_directory/'queries.jsonl',row)
        a.query_history.append(row)
        a.n_processes+=int(result.get('process_started',result.get('return_code') is not None))
        if getattr(a,'city_support',None):a.city_support.observe(row)
        self.export()
        return row

    def query(self, structure, **extra):
        a=self.owner;structure=' '.join(structure.split())
        with writer_lock(a.run_directory):
            # A second process must not allocate from stale in-memory counters.
            if len(read_facts(a.run_directory/'query_dispatch.jsonl'))!=a.budget_used:
                raise RuntimeError('Ledger changed: explicitly reconcile the query session before continuing')
            if a.budget_used>=a.budget_total or structure in a.queried:return None
            if not a.space.valid(structure):raise ValueError('Preflight failed: structure outside search space')
            optics=getattr(a,'optics',{}).get(structure)
            if optics is None:
                optics=a.tmm.compute(structure);a.n_tmm+=1
            features={k:float(optics[k]) for k in FEATURES}
            vector=np.array(list(features.values()))
            if not np.isfinite(vector).all() or np.any((vector<0)|(vector>1)):
                raise ValueError('Invalid preflight optical features')
            if not hasattr(a,'query_optics'):a.query_optics={}
            a.query_optics[structure]=optics
            extra.setdefault('phase','online_search');extra['optical_properties']=features
            extra.setdefault('support_version',getattr(a.surrogate,'VERSION','native_city_model'))
            extra.setdefault('uncertainty_status', 'target_uncalibrated_heuristic' if getattr(a,'city_support',None) else 'source_calibrated_before_adaptive_correction')
            if hasattr(a.surrogate,'predict'):
                raw=a.surrogate.predict(features)
                corrected=a.bias_corrector.correct(raw) if hasattr(a,'bias_corrector') else raw
                extra['prediction_snapshot']={'raw':raw,'corrected':corrected,
                    'support_version':extra['support_version'],
                    'bias_version':corrected.get('bias_version',0)}
            output=a.run_directory/'energyplus'/f'query_{a.budget_used+1:03d}'
            dispatch={'query_index':a.budget_used+1,'structure':structure,'extra':extra,
                      'output_directory':str(output),'state':'dispatched'}
            append_fact(a.run_directory/'query_dispatch.jsonl',dispatch)
            a.queried.add(structure);a.budget_used+=1
            try:result=a.energyplus.run(structure,a.city,output_dir=output)
            except Exception as exc:
                result={'status':'failed','error_message':f'{type(exc).__name__}: {exc}',
                        'UDI':None,'TotalEnergy':None,'process_started':False}
            row=self._complete(dispatch,result)
            a.logger.info('%s query %s/%s phase=%s: %s energy=%s UDI=%s',
                a.method_name,a.budget_used,a.budget_total,row['phase'],result['status'],row['energy'],row['udi'])
            return row

    def reconcile(self):
        """Restore probes/query accounting only, never imply restoration of TTT/RNG."""
        a=self.owner
        with writer_lock(a.run_directory):
            dispatches=read_facts(a.run_directory/'query_dispatch.jsonl')
            rows=read_facts(a.run_directory/'queries.jsonl')
            if [d['query_index'] for d in dispatches]!=list(range(1,len(dispatches)+1)):
                raise ValueError('Noncontiguous dispatch IDs')
            if len({d['structure'] for d in dispatches})!=len(dispatches):raise ValueError('Duplicate dispatches')
            if len(dispatches)>a.budget_total:raise ValueError('Restored attempts exceed total budget')
            ids={r['query_index'] for r in rows}
            if len(ids)!=len(rows) or not ids<=set(range(1,len(dispatches)+1)):
                raise ValueError('Invalid completed query IDs')
            for row in rows:
                d=dispatches[row['query_index']-1]
                if row['structure']!=d['structure'] or row.get('city')!=a.city or row.get('seed')!=a.seed:
                    raise ValueError('Restored context mismatch')
            a.query_history=rows;a.queried={d['structure'] for d in dispatches};a.budget_used=len(dispatches)
            a.n_processes=sum(int(r['energyplus_result'].get('process_started',False)) for r in rows)
            for d in dispatches:
                if d['query_index'] in ids:continue
                result_path=Path(d['output_directory'])/'result.json'
                if not result_path.is_file():
                    raise RuntimeError(f'Unresolved dispatch {d["query_index"]}; inspect partial output before any retry')
                self._complete(d,json.loads(result_path.read_text()),recovered=True)
            if getattr(a,'city_support',None):
                for row in a.query_history:a.city_support.observe(row)
            self.export()

    def export(self):
        export_support_data(self.owner.run_directory,self.owner.query_history,
                            getattr(self.owner,'query_optics',{}),self.owner.tmm)
        import csv
        rows=self.owner.formal.summarize_checkpoints(self.owner.query_history,self.owner.budget_used)
        write_json(self.owner.run_directory/'budget_checkpoints.json',rows)
        temporary=self.owner.run_directory/'budget_checkpoints.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(self.owner.run_directory/'budget_checkpoints.csv')


def export_support_data(run_directory, rows=None, optics=None, tmm=None):
    """CSV/NPZ views are rebuildable; queries.jsonl is the single label source."""
    import pandas as pd
    directory=Path(run_directory);target=directory/'city_support';target.mkdir(exist_ok=True)
    rows=read_facts(directory/'queries.jsonl') if rows is None else rows
    observations=[];fit=[];spectra={};optics=optics or {}
    for r in rows:
        result=r['energyplus_result'];prefix=f'q{r["query_index"]:03d}'
        item={k:r.get(k) for k in ['city','seed','query_index','structure','phase','energyplus_verified',
            'direct_feasible','energy','udi','energy_saving','raw_predicted_udi','raw_predicted_energy',
            'predicted_udi','predicted_energy','support_version','bias_version','uncertainty_status']}
        item['adaptation_role']=r.get('adaptation_role','fit')
        item.update(r.get('optical_properties',{}))
        item.update(electricity_J=result.get('ElectricityEnergy'),gas_J=result.get('NaturalGasEnergy'),
                    qualifying_hours=result.get('udi_qualifying_hours'),daylight_hours=result.get('udi_denominator_hours'),
                    error_code=result.get('error_code'),error_message=result.get('error_message'),
                    evidence_directory=result.get('run_directory'),spectrum_key=prefix)
        observations.append(item)
        if r['energyplus_verified']:fit.append(item)
        props=optics.get(r['structure'])
        if props is None and tmm is not None:props=tmm.compute(r['structure'])
        if props is not None:
            spectra['wavelengths_nm']=np.asarray(props['wavelengths_nm'])
            for state in ['cold','hot']:
                for field in ['T','Rf','Rb']:
                    spectra[f'{prefix}_{state}_{field}']=np.asarray(props['spectrum_'+state][field])
    for name,data in [('observations.csv',observations),('support_fit.csv',fit),
                      ('failures.csv',[r for r in observations if not r['energyplus_verified']])]:
        temp=target/(name+'.tmp');pd.DataFrame(data).to_csv(temp,index=False);temp.replace(target/name)
    if spectra:
        with (target/'spectra.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**spectra)
        (target/'spectra.npz.tmp').replace(target/'spectra.npz')
    write_json(target/'probe_state.json',{'n_attempts':len(rows),
        'n_probe_attempts':sum(r.get('phase')=='city_support_probe' for r in rows),
        'n_success':len(fit),'n_failed':len(rows)-len(fit),
        'label_source':'../queries.jsonl','spectrum_grid':'300:10:2500 nm',
        'resume_scope':'query/probe accounting only; full Agent policy state not restored'})
