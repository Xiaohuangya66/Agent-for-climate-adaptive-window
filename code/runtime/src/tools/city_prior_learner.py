"""Measured, formally feasible city priors with matched 8D/full spectra."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.config import ENERGYPLUS_RAW_DIR


class CityPriorLearner:
    def __init__(self, data_dir=None, protocol_factory=None, seed=42):
        self.data_dir = Path(data_dir or ENERGYPLUS_RAW_DIR)
        self._use_runtime_profile = data_dir is None
        self.protocol_factory = protocol_factory
        self.seed = int(seed)
        self.optical_features = ['T_lum_cold', 'T_lum_hot', 'T_sol_cold', 'T_sol_hot',
                                 'Rf_sol_cold', 'Rb_sol_cold', 'Rf_sol_hot', 'Rb_sol_hot']

    @staticmethod
    def _full_spectrum(row):
        arrays = [np.fromstring(row[key], sep=',') if isinstance(row[key], str)
                  else np.asarray(row[key], dtype=float) for key in ['T_cold', 'T_hot']]
        if any(a.shape != (221,) for a in arrays):
            raise ValueError('Measured prior spectra must each contain 221 wavelengths')
        full = np.column_stack(arrays)
        if not np.isfinite(full).all() or np.any((full < 0) | (full > 1)):
            raise ValueError('Invalid measured prior transmittance')
        return full

    def _elite(self, city, top_percent):
        from src.agent.formal_protocol import FormalProtocol
        if not 0 < top_percent <= 100:
            raise ValueError('top_percent must be in (0,100]')
        protocol = (self.protocol_factory or FormalProtocol)(city)
        if self._use_runtime_profile:
            from src.runtime_profile import city_assets, MODEL_RUN
            city_assets(city)  # Require accepted city/weather/model identity.
            path=MODEL_RUN/'datasets'/f'{city}_train_fit.csv.gz'
        else:
            path = self.data_dir/f'ML_Dataset_SiteEnergy_{city}.csv'
        self.source_path=path
        data = pd.read_csv(path)
        columns = self.optical_features+['TotalEnergy', 'UDI']
        values = data[columns].apply(pd.to_numeric, errors='coerce')
        valid = np.isfinite(values).all(axis=1) & (values['TotalEnergy'] > 0)
        valid &= values['UDI'].between(protocol.threshold, 1)
        valid &= values[self.optical_features].ge(0).all(axis=1)
        valid &= values[self.optical_features].le(1).all(axis=1)
        valid &= data['structure'].map(lambda s: isinstance(s, str) and bool(s.strip()))
        feasible = data.loc[valid].copy()
        feasible[columns] = values.loc[valid]
        feasible = feasible.drop_duplicates('structure').sort_values('TotalEnergy', kind='stable')
        self.last_feasible_count = len(feasible)
        if feasible.empty:
            raise ValueError(f'No formally feasible prior rows for {city}')
        count = max(1, int(len(feasible)*top_percent/100))
        rows, full = [], []
        # A damaged elite row must not corrupt the mean or silently truncate a grid.
        for index, row in feasible.iterrows():
            try:
                spectrum = self._full_spectrum(row)
            except (ValueError, TypeError):
                continue
            rows.append(index)
            full.append(spectrum)
            if len(rows) == count:
                break
        if not rows:
            raise ValueError(f'No valid measured 221x2 prior spectra for {city}')
        elite = feasible.loc[rows].reset_index(drop=True)
        return protocol, elite, np.stack(full)

    def learn_prior(self, city, top_percent=2.0, n_bootstrap=30):
        protocol, elite, full = self._elite(city, top_percent)
        features = elite[self.optical_features].to_numpy(dtype=float)
        stability = self._bootstrap_stability(features, n_bootstrap)
        return {'target_mean': features.mean(axis=0), 'target_std': features.std(axis=0),
                'target_full_spectrum': full.mean(axis=0),
                'confidence': self._compute_confidence(features.std(axis=0), stability, len(elite)),
                'n_samples': len(elite), 'bootstrap_stability': stability,
                'pareto_structures': elite, 'city': city, 'udi_threshold': protocol.threshold,
                'n_feasible_source_rows': self.last_feasible_count,
                'selection': 'lowest energy among formally feasible rows; not nondominated sorting'}

    def learn_prior_clusters(self, city, n_clusters=3, top_percent=2.0, n_bootstrap=30):
        if int(n_clusters) < 1:
            raise ValueError('n_clusters must be positive')
        protocol, elite, full = self._elite(city, top_percent)
        features = elite[self.optical_features].to_numpy(dtype=float)
        scale = features.std(axis=0)
        scaled = (features-features.mean(axis=0))/np.where(scale > 1e-8, scale, 1)
        count = min(int(n_clusters), len(elite), len(np.unique(scaled, axis=0)))
        labels = (KMeans(n_clusters=count, random_state=self.seed, n_init=10).fit_predict(scaled)
                  if count > 1 else np.zeros(len(elite), dtype=int))
        clusters = []
        for label in np.unique(labels):
            mask = labels == label
            rows = elite.loc[mask]
            center = features[mask].mean(axis=0)
            indices = np.flatnonzero(mask)
            distances = np.linalg.norm(scaled[mask]-scaled[mask].mean(axis=0), axis=1)
            representative = int(indices[np.argmin(distances)])
            clusters.append({'source_label': int(label), 'center': center,
                'target_full_spectrum': full[mask].mean(axis=0),
                'size': int(mask.sum()), 'avg_energy': float(rows['TotalEnergy'].mean()),
                'avg_udi': float(rows['UDI'].mean()), 'std': features[mask].std(axis=0),
                'representative_structure': elite.iloc[representative]['structure'],
                'representative_structures': rows['structure'].head(5).tolist(),
                'energy_range': [float(rows['TotalEnergy'].min()), float(rows['TotalEnergy'].max())],
                'udi_range': [float(rows['UDI'].min()), float(rows['UDI'].max())]})
        clusters.sort(key=lambda c: (c['avg_energy'], -c['avg_udi']))
        for index, cluster in enumerate(clusters):
            cluster['cluster_id'] = index
        stability = self._bootstrap_stability(features, n_bootstrap)
        # Confidence describes prior consistency, never global optimality.
        confidence = self._compute_confidence(features.std(axis=0), stability, len(elite))
        print(f'[CityPriorLearner] {city}: {len(elite)} energy-ranked elite rows, {len(clusters)} clusters')
        return {'clusters': clusters, 'confidence': confidence, 'n_total_samples': len(elite),
                'n_feasible_source_rows': self.last_feasible_count, 'city': city,
                'udi_threshold': protocol.threshold, 'clear_glass_energy': protocol.energy,
                'bootstrap_stability': stability, 'clustering': 'standardized eight optical features',
                'selection': 'lowest energy among formally feasible rows'}

    def _extract_pareto_front(self, objectives, top_percent=2.0):
        """Legacy name: energy-ranked elite selection after caller feasibility filtering."""
        energy = np.asarray(objectives[0])
        if not 0 < top_percent <= 100:
            raise ValueError('top_percent must be in (0,100]')
        return np.argsort(energy, kind='stable')[:max(1, int(len(energy)*top_percent/100))].tolist()

    def _bootstrap_stability(self, spectra, n_bootstrap=30):
        if n_bootstrap < 1 or len(spectra) < 1:
            raise ValueError('Bootstrap requires observations and positive resample count')
        rng = np.random.default_rng(self.seed)
        means = [spectra[rng.integers(len(spectra), size=len(spectra))].mean(axis=0)
                 for _ in range(n_bootstrap)]
        return float(np.std(means, axis=0).mean())

    def _compute_confidence(self, target_std, bootstrap_stability, n_samples):
        std = float(np.mean(target_std))
        conf_std = .95 if std < .05 else .85 if std < .10 else .65 if std < .15 else .40
        conf_bootstrap = (1. if bootstrap_stability < .05 else .9 if bootstrap_stability < .10
                          else .7 if bootstrap_stability < .15 else .5)
        conf_samples = 1. if n_samples >= 200 else .9 if n_samples >= 100 else .7 if n_samples >= 50 else .5
        return float(.5*conf_std+.3*conf_bootstrap+.2*conf_samples)

    def compare_cities(self, cities):
        results = []
        for city in cities:
            prior = self.learn_prior(city)
            results.append({'city': city, 'confidence': prior['confidence'], 'n_samples': prior['n_samples'],
                'udi_threshold': prior['udi_threshold'], 'n_feasible_source_rows': prior['n_feasible_source_rows'],
                **{f'mean_{key}': float(prior['target_mean'][i]) for i, key in enumerate(self.optical_features)},
                **{f'std_{key}': float(prior['target_std'][i]) for i, key in enumerate(self.optical_features)}})
        return pd.DataFrame(results)

    def visualize_prior(self, city, save_path=None):
        import matplotlib.pyplot as plt
        prior = self.learn_prior(city)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(self.optical_features, prior['target_mean'], yerr=prior['target_std'])
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig
