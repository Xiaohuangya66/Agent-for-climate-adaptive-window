"""Energy-directed updates with identical anchors for 8D and full spectra."""
import numpy as np

FEATURES = ('T_lum_cold', 'T_lum_hot', 'T_sol_cold', 'T_sol_hot',
            'Rf_sol_cold', 'Rb_sol_cold', 'Rf_sol_hot', 'Rb_sol_hot')


class TargetRefiner:
    def __init__(self, deviation_threshold=.2, confidence_threshold=.8):
        self.deviation_threshold = deviation_threshold
        self.confidence_threshold = confidence_threshold

    def should_refine(self, current_target, pareto_front, target_confidence, tmm):
        """Legacy discrepancy rule; AgentFull uses direct-feedback scheduling."""
        data = self._observations(pareto_front, tmm)
        if not data:
            return False
        deviation = np.linalg.norm(np.mean([d['features'] for d in data], axis=0)-current_target)
        return bool((deviation > self.deviation_threshold and target_confidence < self.confidence_threshold)
                    or deviation > .4)

    @staticmethod
    def _observations(rows, tmm, optics=None, require_full=False):
        observations = []
        for row in rows:
            if row.get('energyplus_verified') is False or row.get('direct_feasible') is False:
                continue
            energy = row.get('energy')
            if energy is None or not np.isfinite(energy) or energy <= 0:
                continue
            props = (optics or {}).get(row['structure'])
            if props is None:
                props = row.get('optical_properties')
            if props is None or (require_full and 'spectrum_cold' not in props):
                props = tmm.compute(row['structure'])
            features = np.asarray([props[key] for key in FEATURES], dtype=float)
            if features.shape != (8,) or not np.isfinite(features).all():
                continue
            full = None
            if require_full:
                full = np.column_stack([props['spectrum_cold']['T'], props['spectrum_hot']['T']])
                if full.shape != (221, 2) or not np.isfinite(full).all() or np.any((full < 0) | (full > 1)):
                    raise ValueError('Feedback requires finite measured 221x2 transmittance')
            observations.append({'structure': row['structure'], 'energy': float(energy),
                                 'features': features, 'full': full})
        return observations

    def refine_target_energy_directed(self, current_target, pareto_front, tmm,
                                      strategy='best_energy', momentum=.6,
                                      current_full=None, optics=None):
        if strategy not in {'best_energy', 'weighted', 'pareto_mean', 'exploratory'}:
            raise ValueError(f'Unknown refinement strategy: {strategy}')
        if not 0 <= momentum <= 1:
            raise ValueError('momentum must be in [0,1]')
        current = np.asarray(current_target, dtype=float)
        if current.shape != (8,) or not np.isfinite(current).all():
            raise ValueError('Target must contain eight finite optical features')
        if current_full is not None:
            current_full = np.asarray(current_full, dtype=float)
            if current_full.shape != (221, 2) or not np.isfinite(current_full).all():
                raise ValueError('Full target must have shape 221x2')
        data = self._observations(pareto_front, tmm, optics, current_full is not None)
        weights = np.zeros(len(data))
        reason = 'no_valid_feasible_feedback'
        if data:
            energies = np.asarray([d['energy'] for d in data])
            features = np.stack([d['features'] for d in data])
            if strategy == 'best_energy':
                index = int(np.argmin(energies))
                weights[index] = 1
                reason = f'best_energy={energies[index]:.6e}'
            elif strategy == 'exploratory':
                # Use a different measured anchor rather than independently
                # perturbing aggregates that cannot define a consistent spectrum.
                index = int(np.argmax(np.linalg.norm(features-current, axis=1)))
                weights[index] = 1
                reason = 'most_distant_verified_feasible_anchor'
            elif strategy == 'weighted':
                scale = max(float(energies.std()), float(energies.min())*1e-6)
                weights = np.exp(-(energies-energies.min())/scale)
                weights /= weights.sum()
                reason = 'energy_weighted_verified_anchors'
            else:
                weights[:] = 1/len(data)
                reason = 'mean_verified_anchors'
            direction = np.average(features, axis=0, weights=weights)
            direction_full = (np.average(np.stack([d['full'] for d in data]), axis=0, weights=weights)
                              if current_full is not None else None)
        else:
            direction, direction_full = current.copy(), current_full
        new_full = (momentum*current_full+(1-momentum)*direction_full
                    if current_full is not None else None)
        return {'target_new': momentum*current+(1-momentum)*direction,
                'target_old': current.copy(), 'target_direction': direction,
                'target_full_new': new_full, 'target_direction_full': direction_full,
                'pareto_mean': direction, 'deviation': float(np.linalg.norm(direction-current)),
                'confidence_update': 0.0, 'strategy_used': strategy if data else 'none',
                'reason': reason, 'momentum': momentum,
                'source_structures': [d['structure'] for d in data], 'weights': weights.tolist()}

    def refine_target(self, current_target, pareto_front, tmm, momentum=.7):
        return self.refine_target_energy_directed(current_target, pareto_front, tmm,
                                                 strategy='pareto_mean', momentum=momentum)

    def compute_deviation(self, target1, target2):
        return float(np.linalg.norm(np.asarray(target1)-np.asarray(target2)))

    def visualize_refinement(self, refine_result, save_path=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        for key in ['target_old', 'target_direction', 'target_new']:
            ax.plot(refine_result.get(key, refine_result.get('pareto_mean')), label=key)
        ax.legend()
        if save_path:
            fig.savefig(save_path)
        return fig
