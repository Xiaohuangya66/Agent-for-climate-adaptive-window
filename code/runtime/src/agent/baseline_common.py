"""Shared direct-evaluation records for server baseline methods."""
from __future__ import annotations
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random
import re
import tempfile
from datetime import datetime, timezone

from src.config import BASELINE_ENERGY, BASELINE_UDI, STRUCTURE_CONSTRAINTS, UDI_THRESHOLD_RATIO, PROJECT_ROOT

ROOT = Path(PROJECT_ROOT).resolve()
FEATURE_ORDER = ('T_lum_cold', 'T_lum_hot', 'T_sol_cold', 'T_sol_hot',
                 'Rf_sol_cold', 'Rb_sol_cold', 'Rf_sol_hot', 'Rb_sol_hot')


class SearchSpace:
    """A recorded, config-aware token space shared by NSGA-II and TTT-RL.

    Does not reinstate the user's relaxed WVO2/terminal-layer constraints.
    """
    def __init__(self, vocab_file=None, max_layers=None):
        from src.support_settings import asset_path
        self.vocab_file = Path(vocab_file or asset_path('vocab_file', ROOT.parent / 'topic3/dataset/vocab.txt')).resolve()
        self.vocab = self.vocab_file.read_text().splitlines()
        self.token2id = {t.strip(): i for i, t in enumerate(self.vocab) if t.strip()}
        if len(self.token2id) != len(self.vocab):
            raise ValueError('Vocabulary contains blank or duplicate lines')
        self.max_layers = int(max_layers or STRUCTURE_CONSTRAINTS.get('max_layers', 30))
        self.required_wvo2 = int(STRUCTURE_CONSTRAINTS.get('wvo2_count', 0))
        self.maximum_wvo2 = STRUCTURE_CONSTRAINTS.get('wvo2_max_count')
        if self.maximum_wvo2 is not None:
            if type(self.maximum_wvo2) is not int or self.maximum_wvo2 < self.required_wvo2:
                raise ValueError('Invalid optional WVO2 upper bound')
        self.terminal = STRUCTURE_CONSTRAINTS.get('terminal_layer')
        allowed = set(STRUCTURE_CONSTRAINTS['allowed_materials'])
        self.tokens = []
        for token in self.token2id:
            match = re.fullmatch(r'([A-Za-z][A-Za-z0-9]*)_(\d+)', token)
            if not match or match[1] not in allowed:
                continue
            thickness = int(match[2])
            if token == 'SiO2_10000':
                self.tokens.append(token)
            elif STRUCTURE_CONSTRAINTS.get('min_thickness', 5) <= thickness <= STRUCTURE_CONSTRAINTS.get('max_thickness', 500):
                if match[1] != 'WVO2' or STRUCTURE_CONSTRAINTS['wvo2_thickness_range'][0] <= thickness <= STRUCTURE_CONSTRAINTS['wvo2_thickness_range'][1]:
                    self.tokens.append(token)
        self.wvo2 = [t for t in self.tokens if t.startswith('WVO2_')]
        self.general = [t for t in self.tokens if t != 'SiO2_10000' and not t.startswith('WVO2_')]
        if not self.tokens or self.max_layers < 1 or (self.required_wvo2 and not self.wvo2):
            raise ValueError('Empty or impossible configured search space')
        self.token_set = set(self.tokens)

    def valid(self, structure):
        tokens = structure.split()
        count = sum(t.startswith('WVO2_') for t in tokens)
        return bool(tokens) and len(tokens) <= self.max_layers and all(t in self.token_set for t in tokens) and (not self.terminal or tokens[-1] == self.terminal) and (not self.required_wvo2 or count == self.required_wvo2) and (self.maximum_wvo2 is None or count <= self.maximum_wvo2)

    def repair(self, tokens, rng):
        tokens = [t for t in tokens if t in self.token_set]
        if self.terminal:
            tokens = [t for t in tokens if t != self.terminal]
        slots = self.max_layers - bool(self.terminal)
        tokens = tokens[:slots]
        if self.required_wvo2:
            tokens = [t for t in tokens if not t.startswith('WVO2_')]
            if slots < self.required_wvo2:
                raise ValueError('No slots for required WVO2')
            tokens = tokens[:slots - self.required_wvo2]
            for _ in range(self.required_wvo2):
                tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(self.wvo2))
        elif self.maximum_wvo2 is not None:
            kept=[];count=0
            for token in tokens:
                if token.startswith('WVO2_'):
                    count+=1
                    if count>self.maximum_wvo2:continue
                kept.append(token)
            tokens=kept
        if self.terminal:
            tokens.append(self.terminal)
        if not tokens:
            choices = self.general if self.maximum_wvo2 == 0 else self.tokens
            tokens = [rng.choice(choices)]
        result = ' '.join(tokens)
        if not self.valid(result):
            raise ValueError('Search-space repair failed')
        return result

    def random(self, rng):
        length = rng.randint(max(1, self.required_wvo2 + bool(self.terminal)), self.max_layers)
        return self.repair([rng.choice(self.tokens) for _ in range(length)], rng)

    def child(self, left, right, rng, probability=0.3):
        a, b = left.split(), right.split()
        tokens = a[:rng.randrange(len(a) + 1)] + b[rng.randrange(len(b) + 1):]
        if rng.random() < probability:
            action = rng.choice(['replace', 'insert', 'delete'])
            if action == 'replace' and tokens:
                tokens[rng.randrange(len(tokens))] = rng.choice(self.tokens)
            elif action == 'insert':
                tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(self.tokens))
            elif tokens:
                tokens.pop(rng.randrange(len(tokens)))
        return self.repair(tokens, rng)

    def description(self):
        return {'vocab_file': str(self.vocab_file), 'max_layers': self.max_layers,
                'required_wvo2_count': self.required_wvo2, 'maximum_wvo2_count': self.maximum_wvo2, 'terminal_layer': self.terminal,
                'material_token_count': len(self.tokens), 'thickness_source': 'training vocabulary filtered by current config'}


class BaselineRecords:
    def _setup_records(self, method, output_root=None):
        if self.budget_total < 0:
            raise ValueError('budget must be nonnegative')
        self.method_name = method
        self.rng = random.Random(self.seed)
        output_root = Path(output_root or ROOT / 'results/baseline_runs').resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        self.run_directory = Path(tempfile.mkdtemp(prefix=f'{method}_{self.city}_seed{self.seed}_', dir=output_root))
        self.queried = set()
        self.baseline_energy = self.formal.energy
        self.formal_udi_threshold = self.formal.threshold
        self.n_generation_attempts = 0
        self.n_tmm = 0
        self.n_processes = 0
        self.generation_log = []
        if hasattr(self,'runtime_assets'):
            (self.run_directory/'runtime_assets.json').write_text(
                json.dumps(self.runtime_assets,indent=2,allow_nan=False),encoding='utf-8')
        (self.run_directory/'formal_protocol.json').write_text(json.dumps({
            **self.formal.describe(), 'search_space': self.space.description(),
            'runtime_assets': getattr(self,'runtime_assets',None)},
            indent=2,allow_nan=False),encoding='utf-8')
        if getattr(self,'city_support',None):
            self.city_support.bind(self.run_directory)

    def _direct_query(self, structure, **extra):
        from src.tools.query_service import QueryService
        if not hasattr(self, 'query_service'):
            self.query_service = QueryService(self)
        return self.query_service.query(structure, **extra)

    def _finish_records(self, start_time, method_data):
        import time
        success = [r for r in self.query_history if r['energyplus_verified']]
        pareto = [r for r in success if not any(o['energy'] <= r['energy'] and o['udi'] >= r['udi'] and (o['energy'] < r['energy'] or o['udi'] > r['udi']) for o in success)]
        result = {'status': 'success' if success else ('no_queries' if self.budget_total == 0 else 'failed'),
                  'pareto_front': pareto, 'query_history': self.query_history,
                  'n_energyplus_queries': self.budget_used,
                  'budget_exhausted': self.budget_used == self.budget_total,
                  'n_energyplus_processes': self.n_processes,
                  'n_successful_queries': len(success), 'n_failed_queries': self.budget_used-len(success),
                  'computation_time': time.time()-start_time, 'run_directory': str(self.run_directory),
                  'protocol': {**self.formal.describe(),
                               'search_space': self.space.description(),
                               'tmm_counter_scope': 'search-stage calls; EnergyPlus adapter also recomputes optics for each direct query',
                               'budget_policy': 'unique direct-query attempts; failed attempts count',
                               'caveat': f'A feasible best requires direct UDI >= {self.formal.udi_ratio:.0%} clear glass.'},
                  'method_specific_data': method_data}
        if hasattr(self,'runtime_assets'):
            result['protocol']['runtime_assets']=self.runtime_assets
        if hasattr(self.surrogate,'numerical_validation'):
            result['protocol']['surrogate_numerical_validation'] = self.surrogate.numerical_validation
        result.update(self.formal.summarize(self.query_history))
        result['budget_checkpoints']=self.formal.summarize_checkpoints(self.query_history,self.budget_used)
        checked = [r for r in success if r.get('surrogate_interval_covers_direct_udi') is not None]
        result['surrogate_validation'] = {
            'n_interval_checks': len(checked),
            'n_interval_misses': sum(not r['surrogate_interval_covers_direct_udi'] for r in checked),
            'mean_absolute_udi_error': sum(abs(r['surrogate_udi_error']) for r in checked)/len(checked) if checked else None,
            'acceptance_basis': 'direct EnergyPlus only; proxy feasibility is never final acceptance'}
        result['n_tmm_evaluations'] = self.n_tmm
        (self.run_directory / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return result
