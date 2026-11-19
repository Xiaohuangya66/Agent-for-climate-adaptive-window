"""Bounded, auditable query heuristics; no calibrated information-gain claim."""
import math
import numpy as np

FEATURE_ORDER = ('T_lum_cold', 'T_lum_hot', 'T_sol_cold', 'T_sol_hot',
                 'Rf_sol_cold', 'Rb_sol_cold', 'Rf_sol_hot', 'Rb_sol_hot')


class InformationValueSelector:
    VERSION = 'information_value_v1_20261021'

    def __init__(self, baseline_energy, udi_threshold, feature_scale=None, weights=None):
        self.baseline_energy = float(baseline_energy)
        self.udi_threshold = float(udi_threshold)
        if not math.isfinite(self.baseline_energy) or self.baseline_energy <= 0:
            raise ValueError('Selection requires a finite positive energy baseline')
        if not math.isfinite(self.udi_threshold) or not 0 < self.udi_threshold <= 1:
            raise ValueError('Selection requires a UDI threshold in (0,1]')
        self.feature_scale = np.asarray(feature_scale if feature_scale is not None else np.ones(8), dtype=float)
        if self.feature_scale.shape != (8,) or not np.isfinite(self.feature_scale).all() or (self.feature_scale <= 0).any():
            raise ValueError('Optical feature scales must contain eight finite positive values')
        self.weights = dict(weights or {'performance': .4, 'uncertainty': .3, 'novelty': .2, 'pareto': .1})
        if set(self.weights) != {'performance', 'uncertainty', 'novelty', 'pareto'}:
            raise ValueError('Expected performance, uncertainty, novelty, and pareto weights')
        values = np.asarray(list(self.weights.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError('Selection weights must be finite, nonnegative and not all zero')
        total = float(values.sum())
        self.weights = {key: float(value)/total for key, value in self.weights.items()}

    @staticmethod
    def _number(value, name, nonnegative=False):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid selection field: {name}') from exc
        if not math.isfinite(result) or (nonnegative and result < 0):
            raise ValueError(f'Invalid selection field: {name}')
        return result

    @staticmethod
    def _features(row):
        values = row.get('optical_properties')
        if not isinstance(values, dict) or any(key not in values for key in FEATURE_ORDER):
            return None
        try:
            vector = np.asarray([values[key] for key in FEATURE_ORDER], dtype=float)
        except (TypeError, ValueError):
            return None
        return vector if vector.shape == (8,) and np.isfinite(vector).all() and ((vector >= 0) & (vector <= 1)).all() else None

    @staticmethod
    def _structure_distance(left, right):
        """Normalized token edit distance, including material, thickness and order."""
        left, right = left.split(), right.split()
        if not left or not right:
            return None
        previous = list(range(len(right)+1))
        for i, token in enumerate(left, 1):
            current = [i]
            for j, other in enumerate(right, 1):
                current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(token != other)))
            previous = current
        return previous[-1]/max(len(left), len(right))

    def novelty(self, candidate, references):
        features = self._features(candidate)
        distances = []
        optical_references = 0
        for row in references:
            reference = self._features(row)
            optical = None
            if features is not None and reference is not None:
                distance = float(np.sqrt(np.mean(np.square((features-reference)/self.feature_scale))))
                optical = distance/(1+distance)
                optical_references += 1
            structural = self._structure_distance(candidate.get('structure', ''), row.get('structure', ''))
            if optical is not None and structural is not None:
                distances.append(.7*optical+.3*structural)
            elif optical is not None:
                distances.append(optical)
            elif structural is not None:
                distances.append(structural)
        return (float(min(distances)) if distances else 1., optical_references, len(distances))

    def _frontier(self, energy, udi, pareto):
        points = []
        for row in pareto:
            # Only actual, successful, feasible labels define the reference front.
            if not row.get('energyplus_verified') or not row.get('direct_feasible'):
                continue
            e, u = row.get('energy'), row.get('udi')
            if e is None or u is None or not np.isfinite([e, u]).all() or e <= 0 or u < self.udi_threshold:
                continue
            points.append((float(e), float(u)))
        if not points:
            return 0., None, False, 0
        distance = min(math.hypot((energy-e)/(.05*self.baseline_energy), (udi-u)/.05) for e, u in points)
        dominated = any(e <= energy and u >= udi and (e < energy or u > udi) for e, u in points)
        # A close but clearly dominated point should not receive the same bonus.
        return (1/(1+distance))*(.25 if dominated else 1.), distance, dominated, len(points)

    def score(self, candidate, references=(), pareto=()):
        energy = self._number(candidate['predicted_energy'], 'predicted_energy')
        udi = self._number(candidate['predicted_udi'], 'predicted_udi')
        if energy <= 0 or not 0 <= udi <= 1:
            raise ValueError('Selection requires positive energy and a bounded UDI prediction')
        udi_half = self._number(candidate.get('udi_half_width_90pct', 0), 'udi_half_width', True)
        energy_half = self._number(candidate.get('energy_half_width_90pct', 0), 'energy_half_width', True)
        ood = self._number(candidate.get('ood_distance', 0), 'ood_distance', True)
        energy_score = float(np.clip((1.1-energy/self.baseline_energy)/.2, 0, 1))
        daylight_score = float(np.clip(udi/self.udi_threshold, 0, 1))
        performance = .7*energy_score+.3*daylight_score
        relative_energy = energy_half/self.baseline_energy
        relative_udi = udi_half/self.udi_threshold
        uncertainty = .5*(relative_energy/(relative_energy+.02)+relative_udi/(relative_udi+.02))
        reliability = 1/(1+ood)
        novelty, optical_count, reference_count = self.novelty(candidate, references)
        frontier, distance, dominated, frontier_count = self._frontier(energy, udi, pareto)
        components = {'performance': performance, 'uncertainty': uncertainty*reliability,
                      'novelty': novelty*reliability, 'pareto': frontier}
        score = float(np.clip(sum(self.weights[key]*components[key] for key in self.weights), 0, 1))
        return {'score': score, 'components': {key: float(value) for key, value in components.items()},
                'raw_uncertainty_score': float(uncertainty), 'raw_novelty_score': float(novelty),
                'ood_reliability': float(reliability), 'energy_score': energy_score,
                'daylight_score': daylight_score, 'predicted_udi_margin': udi-self.udi_threshold,
                'predicted_center_feasible': udi >= self.udi_threshold,
                'normalized_frontier_distance': distance, 'predicted_dominated_by_direct_front': dominated,
                'n_novelty_references': reference_count, 'n_optical_references': optical_count,
                'n_frontier_references': frontier_count,
                'uncertainty_available': any(key in candidate for key in ['udi_half_width_90pct', 'energy_half_width_90pct'])}

    @staticmethod
    def information_quota(budget, fraction):
        if budget < 0 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Information budget requires nonnegative budget and fraction in [0,1]')
        return min(max(0, budget-1), int(math.ceil(budget*fraction)))

    @classmethod
    def information_due(cls, query_index, budget, fraction, information_used):
        quota = cls.information_quota(budget, fraction)
        return cls.information_due_with_quota(query_index,budget,quota,information_used)

    @staticmethod
    def information_due_with_quota(query_index,budget,quota,information_used):
        if not 0<=quota<=max(0,budget-1):raise ValueError('Quota must leave the first online query for performance')
        if query_index <= 1 or budget <= 1 or information_used >= quota:
            return False
        # Evenly spread the integer quota, keep query 1 as a performance anchor.
        planned = int(math.floor((query_index-1)*quota/(budget-1)+.5))
        return information_used < planned

    def describe(self):
        return {'version': self.VERSION, 'weights': self.weights,
                'feature_order': list(FEATURE_ORDER), 'feature_scale': self.feature_scale.tolist(),
                'performance_energy_band': [.9, 1.1], 'uncertainty_relative_reference': .02,
                'novelty': 'minimum joint distance: 0.7 standardized optical RMS + 0.3 token edit; bounded',
                'ood_discount': '1/(1+ood_distance) on uncertainty and novelty',
                'frontier_scales': {'energy_fraction': .05, 'udi_absolute': .05},
                'interpretation': 'bounded heuristic acquisition score, not calibrated information gain'}
