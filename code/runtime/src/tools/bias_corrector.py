"""Online residual correction fitted against RAW surrogate predictions only."""
from collections import deque
import numpy as np


class BiasCorrector:
    def __init__(self, window_size=20, prior_strength=3.0):
        if int(window_size) < 1 or prior_strength < 0:
            raise ValueError('Invalid bias-correction window or prior strength')
        self.observations = deque(maxlen=int(window_size))
        self.udi_observations = deque(maxlen=int(window_size))
        self.n_udi_observations = 0
        self.window_size = int(window_size)
        self.prior_strength = float(prior_strength)
        self.energy_bias = self.udi_bias = 0.0
        self.n_observations = 0
        self._seen = set()

    @staticmethod
    def _metrics(row, direct=False):
        energy = float(row['TotalEnergy'] if 'TotalEnergy' in row else row['energy'])
        udi = float(row['UDI'] if 'UDI' in row else row['udi'])
        if not np.isfinite([energy, udi]).all() or energy <= 0 or (direct and not 0 <= udi <= 1):
            raise ValueError('Require finite positive energy; direct UDI must be in [0,1]')
        return energy, udi

    def update(self, predictions, real_results, *, prequential=True):
        if len(predictions) != len(real_results):
            raise ValueError('Predictions and direct results must be paired one-to-one')
        pending = []
        seen = set(self._seen)
        for pred, real in zip(predictions, real_results):
            if real.get('energyplus_verified') is False or real.get('status') == 'failed':
                continue
            if pred.get('bias_corrected'):
                raise ValueError('Fit residuals against raw predictions, never corrected predictions')
            key = real.get('structure')
            if key is not None and key in seen:
                continue
            predicted_energy, predicted_udi = self._metrics(pred)
            real_energy, real_udi = self._metrics(real, direct=True)
            pending.append({'error_energy': real_energy-predicted_energy,
                            'error_udi': real_udi-predicted_udi, 'structure': key})
            if key is not None:
                seen.add(key)
        # Validate every pair before mutating state.
        self._seen = seen
        self.observations.extend(pending)
        if prequential:
            self.udi_observations.extend(pending)
            self.n_udi_observations += len(pending)
        self.n_observations += len(pending)
        if self.observations:
            weight = len(self.observations)/(len(self.observations)+self.prior_strength)
            self.energy_bias = float(weight*np.mean([o['error_energy'] for o in self.observations]))
        if self.udi_observations:
            weight=len(self.udi_observations)/(len(self.udi_observations)+self.prior_strength)
            self.udi_bias=float(weight*np.mean([o['error_udi'] for o in self.udi_observations]))
        return len(pending)

    def reset_udi(self):
        self.udi_observations.clear()
        self.n_udi_observations=0
        self.udi_bias=0.

    def get_decision_margin(self):
        from src.tools.optimization_policy import policy_configuration,decision_margin
        mae=float(np.mean([abs(o['error_udi']) for o in self.udi_observations])) if self.udi_observations else None
        return decision_margin(mae,self.n_udi_observations,policy_configuration())

    def correct(self, prediction):
        if prediction.get('bias_corrected'):
            raise ValueError('A prediction may only be corrected once')
        energy, udi = self._metrics(prediction)
        corrected = dict(prediction)
        upper_case = 'TotalEnergy' in prediction
        corrected['TotalEnergy' if upper_case else 'energy'] = energy+self.energy_bias
        corrected['UDI' if upper_case else 'udi'] = float(np.clip(udi+self.udi_bias, 0, 1))
        if energy+self.energy_bias <= 0:
            raise ValueError('Bias correction produced nonpositive energy')
        # Online intervals are heuristic; no new nominal coverage is claimed.
        for metric, name, bias in [('udi', 'UDI_half_width_90pct', self.udi_bias),
                                   ('energy', 'Energy_half_width_90pct', self.energy_bias)]:
            observed=self.udi_observations if metric=='udi' else self.observations
            errors = [o['error_'+metric] for o in observed]
            uncertainty = (1.64*float(np.std(errors, ddof=1))/np.sqrt(len(errors))
                           if len(errors) > 1 else abs(bias))
            corrected[name] = float(prediction.get(name, 0))+float(uncertainty)
        from src.tools.optimization_policy import policy_configuration
        policy=policy_configuration()
        center=corrected['UDI' if upper_case else 'udi']
        half=corrected['UDI_half_width_90pct']
        corrected.update(UDI_lower_90pct=max(0.,center-half),UDI_upper_90pct=min(1.,center+half),
            decision_udi_margin=self.get_decision_margin(),
            udi_bias_observation_count=self.n_udi_observations,
            decision_policy_version=policy['version'])
        corrected.update(bias_corrected=True, energy_bias_applied=self.energy_bias,
                         udi_bias_applied=self.udi_bias, bias_version=self.n_observations,
                         udi_center_clipped=not 0 <= udi+self.udi_bias <= 1,
                         interval_definition='original interval plus heuristic bias-estimation uncertainty')
        return corrected

    def get_statistics(self):
        result = {'n_observations': self.n_observations, 'window_size': len(self.observations),
                  'energy_bias': self.energy_bias, 'udi_bias': self.udi_bias,
                  'prior_strength': self.prior_strength,'n_udi_observations':self.n_udi_observations,
                  'udi_window_size':len(self.udi_observations),
                  'decision_udi_margin':self.get_decision_margin(),
                  'udi_bias_scope':'prequential_only; rebuilt tree-fit rows excluded'}
        for metric in ['energy', 'udi']:
            observations=self.udi_observations if metric=='udi' else self.observations
            errors = np.asarray([o['error_'+metric] for o in observations])
            result[metric+'_mae'] = float(np.abs(errors).mean()) if len(errors) else None
            result[metric+'_std'] = float(errors.std()) if len(errors) else None
        return result
