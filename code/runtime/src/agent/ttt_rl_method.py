"""Test-time Transformer RL: generate, surrogate reward, policy update, direct validation."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.agent.base_agent import BaseAgent
from src.agent.baseline_common import BaselineRecords, SearchSpace, ROOT, FEATURE_ORDER
from src.components.ttt_policy import TTTPolicy
from src.config import FIXED_TARGET_SPECTRUM


class TTTRLMethod(BaselineRecords, BaseAgent):
    def __init__(self, city, seed, budget, data_dir='data', runtime_dir='runtime_assets',
                 ttt_rl_code_path=None, candidate_budget=384, batch_size=16,
                 learning_rate=1e-4, temperature=1.0, nhead=4, output_root=None,
                 checkpoint_path=None, max_layers=None, device='cpu'):
        super().__init__(city, seed, budget, data_dir, runtime_dir)
        if ttt_rl_code_path:
            raise ValueError('External arbitrary TTT entrypoints are unsupported; this class now runs the integrated policy')
        self.candidate_budget = int(candidate_budget)
        self.batch_size = int(batch_size)
        if self.candidate_budget < 2 or self.batch_size < 2:
            raise ValueError('TTT candidate_budget and batch_size must be at least 2')
        self.temperature = float(temperature)
        self.space = SearchSpace(max_layers=max_layers)
        self._setup_records('ttt_rl', output_root)
        self.policy = TTTPolicy(self.space, checkpoint=checkpoint_path, learning_rate=learning_rate,
                                device=device, nhead=nhead)
        self.learning_rate = learning_rate
        self.updates = []
        self.archive = {}
        self.reward_cache = {}

    def _target(self):
        # Match the trained 2 x 221 input contract; prefer actual local target data.
        import ast,csv
        path = ROOT.parent / 'topic3/dataset/ideal_target_2.csv'
        if not path.is_file():
            raise FileNotFoundError(f'TTT target spectrum missing: {path}')
        with path.open() as f:
            value = next(csv.reader(f))[0]
        try:
            array = np.asarray(ast.literal_eval(value), dtype=np.float32)
        except (SyntaxError,ValueError):
            array = np.asarray([float(v) for v in value.split(',')], dtype=np.float32)
        if array.shape not in [(442,), (2,221)]:
            raise ValueError(f'TTT target has unexpected shape {array.shape}')
        self.target_source = str(path)
        return torch.from_numpy(array.reshape(2,221))

    def run(self):
        start = time.time()
        if self.budget_total == 0:
            return self._finish_records(start, {'algorithm': 'TTT-RL', 'optimizer_steps': 0})
        target = self._target()
        sampled = 0
        rejected = 0
        while sampled < self.candidate_budget:
            count = min(self.batch_size, self.candidate_budget-sampled)
            rollout = self.policy.sample(target, count, self.temperature)
            rewards = []
            for structure in rollout['structures']:
                sampled += 1
                self.n_generation_attempts += 1
                if structure in self.reward_cache:
                    rewards.append(self.reward_cache[structure])
                    continue
                if not self.space.valid(structure):
                    self.reward_cache[structure] = -10.0
                    rejected += 1; rewards.append(-10.0); continue
                try:
                    self.n_tmm += 1
                    props = self.tmm.compute(structure)
                    prediction = self.surrogate.predict({key: props[key] for key in FEATURE_ORDER}, strict=True)
                    udi, energy = float(prediction['UDI']), float(prediction['TotalEnergy'])
                    if not np.isfinite([udi,energy]).all() or energy <= 0:
                        raise ValueError('Invalid surrogate prediction')
                    formal_prediction = self.formal.prediction(prediction)
                    reward = formal_prediction['reward']
                    self.archive[structure] = {'structure':structure, **formal_prediction}
                    self.reward_cache[structure] = reward
                    rewards.append(reward)
                except Exception as exc:
                    self.reward_cache[structure] = -10.0
                    rejected += 1; rewards.append(-10.0)
                    self.logger.warning('TTT candidate rejected: %s',exc)
            update = self.policy.adapt(target,rollout,rewards)
            update.update(sampled=sampled, unique_candidates=len(self.archive))
            self.updates.append(update)
            with (self.run_directory/'adaptation.jsonl').open('a') as f:
                f.write(json.dumps(update,allow_nan=False)+'\n')
            self.logger.info('TTT policy update %s: loss=%s delta=%s',len(self.updates),update['loss'],update['parameter_delta_l2'])
        # TTT baseline uses surrogate reward only during adaptation; direct results are held out.
        ranked = sorted(self.archive.values(),key=self.formal.query_key)
        for row in ranked:
            if self.budget_used >= self.budget_total:
                break
            self._direct_query(row['structure'],phase='post_adaptation_validation',
                               predicted_energy=row['predicted_energy'],predicted_udi=row['predicted_udi'],
                               udi_half_width_90pct=row['udi_half_width_90pct'],ood_distance=row['ood_distance'])
        if len(ranked) < self.budget_total:
            self.logger.warning('Only %s unique valid candidates for requested direct budget %s',len(ranked),self.budget_total)
        (self.run_directory/'candidates.json').write_text(json.dumps(ranked,indent=2,allow_nan=False))
        return self._finish_records(start,{'algorithm':'TTT-RL','implementation':'real decoder policy gradient with adaptive entropic advantages',
            'reward':'formal_site_energy_saving - 5 * conservative_daylight_UDI_shortfall - uncertainty_penalty',
            'reward_source':'formal local ensemble; direct EnergyPlus not used for adaptation',
            'candidate_budget':self.candidate_budget,'sampled_candidates':sampled,'unique_valid_candidates':len(self.archive),
            'rejected_candidates':rejected,'n_unique_tmm_evaluations':self.n_tmm,
            'optimizer_steps':len(self.updates),'nonzero_parameter_updates':sum(u['parameters_changed'] for u in self.updates),
            'adaptation_history':self.updates,'checkpoint':str(self.policy.checkpoint),'nhead':self.policy.nhead,
            'learning_rate':self.learning_rate,'temperature':self.temperature,'target_source':self.target_source,
            'surrogate_definition':'formal ensemble: daylight-hours UDI and Electricity:Facility + NaturalGas:Facility'})
