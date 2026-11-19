from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .energyplus import EnergyPlusRunner
from .optical import FEATURE_ORDER, OpticalFeatureCalculator
from .policy import TTTPolicyAdapter
from .structures import load_vocab_tokens
from .surrogate import SurrogateEnsemble
from .tmm import TMMCalculator, TMMResult


@dataclass
class Candidate:
    iteration: int
    rollout_index: int
    structure: str
    structure_hash: str
    features: dict[str, float]
    predicted_udi: float
    predicted_energy_J: float
    udi_std: float
    energy_std_J: float
    udi_half_width_90pct: float
    energy_half_width_90pct: float
    ood_distance: float
    reward: float
    acquisition: float
    energy_saving_ratio: float
    conservative_feasible: bool
    optimistic_feasible: bool
    feasibility_probability: float
    query_eligible: bool
    acquisition_reason: str
    direct_reward: float | None = None
    direct_status: str = "not_queried"
    direct_udi: float | None = None
    direct_energy_J: float | None = None
    direct_run_directory: str | None = None
    direct_cache_hit: bool = False
    direct_cache_source_run_directory: str | None = None
    direct_input_signature: str | None = None
    direct_started_utc: str | None = None
    direct_completed_utc: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    variant: str = "full"
    use_active_acquisition: bool = True
    use_uncertainty_and_ood: bool = True
    use_direct_reward_replacement: bool = True
    use_residual_memory: bool = True
    use_adaptive_temperature_and_restart: bool = True
    use_early_stopping: bool = True

    @classmethod
    def for_variant(cls, variant: str) -> "AgentConfig":
        variants = {
            "full": {},
            "no_active_acquisition": {"use_active_acquisition": False},
            "no_uncertainty_ood": {"use_uncertainty_and_ood": False},
            "no_direct_reward": {"use_direct_reward_replacement": False},
            "no_residual_memory": {"use_residual_memory": False},
            "fixed_control": {"use_adaptive_temperature_and_restart": False},
            "no_early_stopping": {"use_early_stopping": False},
        }
        if variant not in variants:
            raise ValueError(f"Unknown Agent variant: {variant}")
        return cls(variant=variant, **variants[variant])


def load_target_spectrum(path: str | Path) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8-sig").strip().strip('"')
    values = [float(value) for value in text.split(",")]
    if len(values) != 442:
        raise ValueError(f"Expected 442 target values, received {len(values)}")
    return torch.tensor(values, dtype=torch.float32).view(2, 221)


class CityDesignAgent:
    def __init__(
        self,
        *,
        workspace: str | Path,
        city: str,
        seed: int,
        run_directory: str | Path,
        agent_config: AgentConfig | None = None,
        baseline_file: str | Path | None = None,
        udi_denominator_mode: str = "all_hours",
        energyplus_additional_idf_text: str = "",
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.city = city
        self.seed = seed
        self.run_directory = Path(run_directory).resolve()
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.agent_config = agent_config or AgentConfig()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.vocab = load_vocab_tokens(self.workspace / "dataset" / "vocab.txt")
        self.target = load_target_spectrum(self.workspace / "ideal_target_2.csv")
        self.tmm = TMMCalculator(self.workspace, max_layers=20)
        self.optical = OpticalFeatureCalculator(self.workspace)
        self.surrogate = SurrogateEnsemble(
            self.workspace / "agent_project" / "models" / "surrogates" / city
        )
        self.policy = TTTPolicyAdapter(
            self.workspace,
            checkpoint=self.workspace / "checkpoints" / "model_S-epoch=1198-val_loss_epoch=3.436.ckpt",
            vocab_file=self.workspace / "dataset" / "vocab.txt",
            max_layers=20,
        )
        baseline_path = (
            Path(baseline_file)
            if baseline_file is not None
            else self.workspace / "agent_project" / "results" / "city_baselines.csv"
        )
        baselines = pd.read_csv(baseline_path)
        baseline = baselines[baselines["city"] == city].iloc[0]
        self.baseline_energy = float(baseline["TotalEnergy_J"])
        self.udi_threshold = float(baseline["UDI_threshold_95pct"])
        self.udi_prediction_scale = float(
            baseline.get("legacy_to_daylight_proxy_scale", 1.0)
        )
        self.udi_denominator_mode = udi_denominator_mode
        if udi_denominator_mode == "daylight_hours":
            baseline_definition = str(baseline.get("energy_definition", ""))
            surrogate_definition = str(
                self.surrogate.metadata.get("target_definition", {}).get(
                    "TotalEnergy", ""
                )
            )
            if "NaturalGas:Facility" not in baseline_definition:
                raise RuntimeError(
                    "The formal baseline is not labelled as total site energy "
                    "(Electricity:Facility + NaturalGas:Facility)."
                )
            if "NaturalGas:Facility" not in surrogate_definition:
                raise RuntimeError(
                    "The formal surrogate is not trained on total site-energy labels."
                )
        self.baseline_file = str(baseline_path.resolve())
        config = json.loads(
            (self.workspace / "agent_project" / "configs" / "project.json").read_text(encoding="utf-8")
        )
        city_config = config["cities"][city]
        self.energyplus = EnergyPlusRunner(
            executable=config["energyplus"]["executable"],
            base_idf=self.workspace / config["energyplus"]["base_idf"],
            epw=self.workspace / city_config["epw"],
            output_root=self.run_directory / "energyplus",
            city=city,
            timeout_seconds=900,
            additional_idf_text=energyplus_additional_idf_text,
            udi_denominator_mode=udi_denominator_mode,
        )
        self.archive: dict[str, Candidate] = {}
        self.tmm_cache: dict[str, TMMResult] = {}
        self.residuals: list[tuple[np.ndarray, np.ndarray]] = []
        self.query_records: list[dict] = []

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _assign_acquisition(self, candidates: list[Candidate]) -> None:
        """Score direct queries by constrained value of information.

        A candidate whose 90% UDI upper bound cannot reach the clear-glass
        constraint is not a useful scarce EnergyPlus query.  Within the
        plausible-feasible set, prefer expected energy improvement near the
        constraint boundary, while retaining small uncertainty and OOD terms.
        The OOD term is capped and strongly out-of-domain points are discounted
        rather than rewarded without limit.
        """
        directly_feasible_archive = [
            item.direct_energy_J
            for item in self.archive.values()
            if item.direct_status == "success"
            and item.direct_udi is not None
            and item.direct_udi >= self.udi_threshold
            and item.direct_energy_J is not None
        ]
        conservatively_feasible_archive = [
            item.predicted_energy_J
            for item in self.archive.values()
            if item.conservative_feasible
        ]
        if directly_feasible_archive:
            incumbent_energy = min(directly_feasible_archive)
            incumbent_source = "best_direct_feasible"
        elif conservatively_feasible_archive:
            incumbent_energy = min(conservatively_feasible_archive)
            incumbent_source = "best_conservative_surrogate"
        else:
            incumbent_energy = self.baseline_energy
            incumbent_source = "clear_glass_baseline"
        ood_threshold = float(self.surrogate.metadata["ood_distance_threshold_95pct_train"])
        for candidate in candidates:
            if self.agent_config.use_uncertainty_and_ood:
                # Ensemble spread supplies the Gaussian approximation used only
                # for ranking by feasibility probability.  The calibrated
                # conformal half-width is a coverage set, not a normal standard
                # deviation, and is kept separate for optimistic/conservative
                # feasibility checks below.
                sigma_udi = max(candidate.udi_std, 1e-6)
                z_feasible = (candidate.predicted_udi - self.udi_threshold) / sigma_udi
                probability = self._normal_cdf(z_feasible)
                upper_udi = candidate.predicted_udi + candidate.udi_half_width_90pct
                optimistic_energy = candidate.predicted_energy_J - candidate.energy_half_width_90pct
                energy_uncertainty = min(
                    candidate.energy_half_width_90pct / self.baseline_energy, 0.05
                )
                udi_uncertainty = min(candidate.udi_half_width_90pct / 0.02, 1.0)
                ood_ratio = candidate.ood_distance / max(ood_threshold, 1e-9)
                domain_trust = math.exp(-0.5 * max(0.0, ood_ratio - 1.0) ** 2)
                ood_information = min(ood_ratio, 2.0)
            else:
                probability = float(candidate.predicted_udi >= self.udi_threshold)
                upper_udi = candidate.predicted_udi
                optimistic_energy = candidate.predicted_energy_J
                energy_uncertainty = 0.0
                udi_uncertainty = 0.0
                ood_ratio = 0.0
                domain_trust = 1.0
                ood_information = 0.0
            improvement = max(0.0, incumbent_energy - optimistic_energy) / self.baseline_energy
            udi_scale = (
                max(candidate.udi_half_width_90pct, 0.002)
                if self.agent_config.use_uncertainty_and_ood
                else 0.002
            )
            boundary_value = math.exp(
                -0.5 * ((candidate.predicted_udi - self.udi_threshold) / udi_scale) ** 2
            )
            candidate.feasibility_probability = probability
            candidate.optimistic_feasible = upper_udi >= self.udi_threshold
            candidate.query_eligible = candidate.optimistic_feasible
            if not self.agent_config.use_active_acquisition:
                candidate.acquisition = candidate.reward
                candidate.acquisition_reason = (
                    "active_acquisition_disabled;static_surrogate_reward_ranking;"
                    f"upper_udi={upper_udi:.6f};threshold={self.udi_threshold:.6f}"
                )
            else:
                if candidate.query_eligible:
                    candidate.acquisition = domain_trust * (
                        probability * improvement
                        + 0.05 * boundary_value
                        + 0.05 * energy_uncertainty
                        + 0.03 * udi_uncertainty
                    ) + 0.01 * ood_information
                else:
                    candidate.acquisition = -1.0 - (
                        self.udi_threshold - upper_udi
                    ) / self.udi_threshold
                candidate.acquisition_reason = (
                    f"p_feasible={probability:.6f};potential_energy_gain={improvement:.6f};"
                    f"boundary_value={boundary_value:.6f};ood_ratio={ood_ratio:.6f};"
                    f"domain_trust={domain_trust:.6f};upper_udi={upper_udi:.6f};"
                    f"threshold={self.udi_threshold:.6f};"
                    f"incumbent_energy_J={incumbent_energy:.6f};"
                    f"incumbent_source={incumbent_source}"
                )

    def _select_final_queries(
        self, candidates: list[Candidate], query_count: int, *, allow_early_stop: bool = True
    ) -> list[Candidate]:
        """Spend final verification budget on constraint-plausible candidates."""
        unqueried = [item for item in candidates if item.direct_status == "not_queried"]
        conservative = sorted(
            [item for item in unqueried if item.conservative_feasible],
            key=lambda item: item.predicted_energy_J,
        )
        optimistic = sorted(
            [
                item for item in unqueried
                if not item.conservative_feasible and item.optimistic_feasible
            ],
            key=lambda item: (-item.feasibility_probability, item.predicted_energy_J),
        )
        directly_feasible_exists = any(
            item.direct_status == "success"
            and item.direct_udi is not None
            and item.direct_udi >= self.udi_threshold
            for item in candidates
        )
        if allow_early_stop and directly_feasible_exists and not conservative and not optimistic:
            # The Agent has already secured a directly verified feasible design;
            # do not burn the remaining allowance on a candidate whose full UDI
            # interval is below the constraint.
            return []
        unreachable = sorted(
            [item for item in unqueried if not item.optimistic_feasible],
            key=lambda item: -(
                item.predicted_udi + item.udi_half_width_90pct
            ),
        )
        return (conservative + optimistic + unreachable)[:query_count]

    def _select_online_queries(
        self,
        new_candidates: list[Candidate],
        already_queried: set[str],
        query_count: int,
    ) -> list[Candidate]:
        unqueried_new = [
            candidate for candidate in new_candidates
            if candidate.structure_hash not in already_queried
        ]
        eligible = [candidate for candidate in unqueried_new if candidate.query_eligible]
        selected = sorted(
            eligible,
            key=lambda candidate: candidate.acquisition,
            reverse=True,
        )[:query_count]
        directly_feasible_exists = any(
            candidate.direct_status == "success"
            and candidate.direct_udi is not None
            and candidate.direct_udi >= self.udi_threshold
            for candidate in self.archive.values()
        )
        remaining_query_slots = query_count - len(selected)
        if (
            remaining_query_slots > 0
            and unqueried_new
            and (
                not directly_feasible_exists
                or not self.agent_config.use_early_stopping
            )
        ):
            # If the proxy declares the entire new batch unreachable before
            # any directly feasible design exists, query the smallest UDI
            # shortfall rather than disabling the high-fidelity loop.
            selected_hashes = {candidate.structure_hash for candidate in selected}
            fallback_pool = [
                candidate for candidate in unqueried_new
                if candidate.structure_hash not in selected_hashes
            ]
            fallback_selected = sorted(
                fallback_pool,
                key=lambda candidate: (
                    -(
                        candidate.predicted_udi
                        + (
                            candidate.udi_half_width_90pct
                            if self.agent_config.use_uncertainty_and_ood
                            else 0.0
                        )
                    ),
                    candidate.predicted_energy_J,
                ),
            )[:remaining_query_slots]
            for candidate in fallback_selected:
                candidate.acquisition_reason += (
                    ";fallback=minimum_predicted_udi_shortfall_no_direct_feasible_incumbent"
                )
            selected.extend(fallback_selected)
        return selected

    @staticmethod
    def _pareto_front(candidates: list[Candidate]) -> list[Candidate]:
        front = []
        for candidate in candidates:
            dominated = any(
                other.predicted_udi >= candidate.predicted_udi
                and other.predicted_energy_J <= candidate.predicted_energy_J
                and (
                    other.predicted_udi > candidate.predicted_udi
                    or other.predicted_energy_J < candidate.predicted_energy_J
                )
                for other in candidates
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda item: (-item.predicted_udi, item.predicted_energy_J))

    def _corrected_prediction(self, mean: np.ndarray, features: np.ndarray) -> np.ndarray:
        if not self.agent_config.use_residual_memory or not self.residuals:
            return mean
        scaled = (features - self.surrogate.x_mean) / self.surrogate.x_scale
        memory_features = np.stack([item[0] for item in self.residuals])
        memory_residuals = np.stack([item[1] for item in self.residuals])
        distances = np.sqrt(np.square(memory_features - scaled).sum(axis=1))
        weights = np.exp(-0.5 * np.square(distances / 1.0))
        # A unit prior weight shrinks a single noisy EnergyPlus residual and lets
        # several nearby verified points dominate only when evidence accumulates.
        correction = (weights[:, None] * memory_residuals).sum(axis=0) / (weights.sum() + 1.0)
        return mean + correction

    def _policy_reward_after_direct_query(self, candidate: Candidate) -> float:
        if (
            self.agent_config.use_direct_reward_replacement
            and candidate.direct_reward is not None
        ):
            return float(candidate.direct_reward)
        return float(candidate.reward)

    def _record_query(self, candidate: Candidate, *, phase: str, iteration: int) -> None:
        direct_feasible = (
            candidate.direct_status == "success"
            and candidate.direct_udi is not None
            and candidate.direct_udi >= self.udi_threshold
        )
        feasible_energies = [
            record["direct_energy_J"]
            for record in self.query_records
            if record["direct_feasible"] and record["direct_energy_J"] is not None
        ]
        if direct_feasible and candidate.direct_energy_J is not None:
            feasible_energies.append(candidate.direct_energy_J)
        best_energy = min(feasible_energies) if feasible_energies else None
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_index": len(self.query_records) + 1,
            "phase": phase,
            "iteration": iteration,
            "city": self.city,
            "seed": self.seed,
            "agent_variant": self.agent_config.variant,
            "structure": candidate.structure,
            "structure_hash": candidate.structure_hash,
            "predicted_udi": candidate.predicted_udi,
            "predicted_energy_J": candidate.predicted_energy_J,
            "udi_half_width_90pct": candidate.udi_half_width_90pct,
            "energy_half_width_90pct": candidate.energy_half_width_90pct,
            "ood_distance": candidate.ood_distance,
            "acquisition": candidate.acquisition,
            "acquisition_reason": candidate.acquisition_reason,
            "direct_status": candidate.direct_status,
            "direct_udi": candidate.direct_udi,
            "direct_energy_J": candidate.direct_energy_J,
            "direct_feasible": direct_feasible,
            "best_direct_feasible_energy_J": best_energy,
            "best_direct_feasible_energy_saving_ratio": (
                (self.baseline_energy - best_energy) / self.baseline_energy
                if best_energy is not None else None
            ),
            "direct_reward": candidate.direct_reward,
            "cache_hit": candidate.direct_cache_hit,
            "run_directory": candidate.direct_run_directory,
            "input_signature": candidate.direct_input_signature,
            "started_utc": candidate.direct_started_utc,
            "completed_utc": candidate.direct_completed_utc,
        }
        self.query_records.append(record)
        with (self.run_directory / "queries.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _evaluate_rollout_batch(
        self,
        rollout,
        *,
        iteration: int,
        batch_size: int,
    ) -> tuple[np.ndarray, list[Candidate], int]:
        """Evaluate one generated batch and add unique candidates to the archive."""
        reward_vector = np.full(batch_size, -2.0, dtype=np.float32)
        new_candidates: list[Candidate] = []
        feature_rows = []
        valid_indices = []
        for index, tokens in enumerate(rollout.material_strings):
            structure = " ".join(tokens)
            try:
                result = self.tmm.calculate(structure, allowed_tokens=self.vocab)
            except Exception:
                continue
            digest = result.structure.structure_hash
            features = self.optical.calculate(result)
            self.tmm_cache[digest] = result
            feature_rows.append([features[name] for name in FEATURE_ORDER])
            valid_indices.append((index, structure, digest, features))
        if valid_indices:
            prediction = self.surrogate.predict(np.asarray(feature_rows))
            for position, (index, structure, digest, features) in enumerate(valid_indices):
                feature_vector = np.asarray(feature_rows[position], dtype=float)
                raw_mean = np.asarray(prediction.mean[position], dtype=float).copy()
                raw_mean[0] *= self.udi_prediction_scale
                mean = self._corrected_prediction(raw_mean, feature_vector)
                udi, energy = float(mean[0]), float(mean[1])
                half = np.asarray(
                    prediction.conformal_half_width_90pct[position], dtype=float
                ).copy()
                std = np.asarray(prediction.std[position], dtype=float).copy()
                half[0] *= self.udi_prediction_scale
                std[0] *= self.udi_prediction_scale
                saving = (self.baseline_energy - energy) / self.baseline_energy
                conservative_udi = (
                    udi - float(half[0])
                    if self.agent_config.use_uncertainty_and_ood
                    else udi
                )
                shortfall = max(
                    0.0, self.udi_threshold - conservative_udi
                ) / self.udi_threshold
                uncertainty_penalty = (
                    0.05 * float(half[1]) / self.baseline_energy
                    if self.agent_config.use_uncertainty_and_ood
                    else 0.0
                )
                reward = saving - 5.0 * shortfall - uncertainty_penalty
                reward_vector[index] = reward
                candidate = Candidate(
                    iteration, index, structure, digest, features, udi, energy,
                    float(std[0]), float(std[1]),
                    float(half[0]), float(half[1]),
                    float(prediction.ood_distance[position]), reward, float("-inf"),
                    saving, conservative_udi >= self.udi_threshold,
                    False, 0.0, False, "pending_batch_acquisition",
                )
                previous = self.archive.get(digest)
                # A repeated structure is not a new TMM candidate.  Keep the
                # first archived record so a later memory-corrected duplicate
                # cannot overwrite its direct EnergyPlus provenance.
                if previous is None:
                    self.archive[digest] = candidate
                    new_candidates.append(candidate)
        return reward_vector, new_candidates, len(valid_indices)

    def run(
        self,
        *,
        iterations: int,
        batch_size: int,
        temperature: float,
        energyplus_queries_per_iteration: int,
        energyplus_total_budget: int | None = None,
        final_energyplus_queries: int = 0,
        method_name: str = "agent",
        stop_patience: int = 5,
    ) -> dict:
        decision_log = self.run_directory / "decisions.jsonl"
        best_reward = -np.inf
        stale = 0
        online_query_count = 0
        current_temperature = temperature
        policy_restarted = False
        stop_reason = "iteration_budget_exhausted"
        stop_snapshot: dict[str, object] | None = None
        for iteration in range(iterations):
            rollout = self.policy.generate(self.target, batch_size, current_temperature)
            reward_vector, new_candidates, valid_count = self._evaluate_rollout_batch(
                rollout, iteration=iteration, batch_size=batch_size
            )

            self._assign_acquisition(new_candidates)

            already_queried = {
                digest for digest, candidate in self.archive.items()
                if candidate.direct_status != "not_queried"
            }
            online_budget_limit = (
                None
                if energyplus_total_budget is None
                else max(0, energyplus_total_budget - final_energyplus_queries)
            )
            remaining_online_budget = (
                energyplus_queries_per_iteration
                if online_budget_limit is None
                else max(0, online_budget_limit - online_query_count)
            )
            query_count = min(energyplus_queries_per_iteration, remaining_online_budget)
            selected = self._select_online_queries(
                new_candidates, already_queried, query_count
            )
            for candidate in selected:
                result = self.energyplus.run(
                    candidate_id=f"iter{iteration}_{candidate.structure_hash[:10]}",
                    structure=candidate.structure,
                    cold=self.tmm_cache[candidate.structure_hash].cold,
                    hot=self.tmm_cache[candidate.structure_hash].hot,
                    allowed_tokens=self.vocab,
                )
                candidate.direct_status = result.metrics.status
                candidate.direct_udi = result.metrics.udi
                candidate.direct_energy_J = result.metrics.total_energy_J
                candidate.direct_run_directory = result.run_directory
                candidate.direct_cache_hit = result.cache_hit
                candidate.direct_cache_source_run_directory = result.cache_source_run_directory
                candidate.direct_input_signature = result.input_signature
                candidate.direct_started_utc = result.started_utc
                candidate.direct_completed_utc = result.completed_utc
                if result.metrics.status == "success":
                    residual = np.asarray(
                        [candidate.direct_udi - candidate.predicted_udi,
                         candidate.direct_energy_J - candidate.predicted_energy_J]
                    )
                    feature_vector = np.asarray(
                        [candidate.features[name] for name in FEATURE_ORDER], dtype=float
                    )
                    feature_scaled = (
                        feature_vector - self.surrogate.x_mean
                    ) / self.surrogate.x_scale
                    if self.agent_config.use_residual_memory:
                        self.residuals.append((feature_scaled, residual))
                    direct_saving = (
                        self.baseline_energy - candidate.direct_energy_J
                    ) / self.baseline_energy
                    direct_shortfall = max(
                        0.0, self.udi_threshold - candidate.direct_udi
                    ) / self.udi_threshold
                    candidate.direct_reward = direct_saving - 5.0 * direct_shortfall
                    # The single policy update for this round now sees the
                    # high-fidelity label instead of the surrogate reward for
                    # the queried rollout.  TTT and Agent still take the same
                    # number of gradient steps.
                    if self.agent_config.use_direct_reward_replacement:
                        reward_vector[candidate.rollout_index] = (
                            self._policy_reward_after_direct_query(candidate)
                        )
                self._record_query(candidate, phase="online", iteration=iteration)
                online_query_count += 1

            adaptation = self.policy.adapt(
                self.target, rollout, torch.tensor(reward_vector, dtype=torch.float32)
            )

            iteration_best = max(reward_vector)
            if iteration_best > best_reward + 1e-8:
                best_reward = float(iteration_best)
                stale = 0
            else:
                stale += 1
            control_action = "maintain"
            control_reason = "reward_progress_is_stable"
            next_temperature = current_temperature
            if (
                self.agent_config.use_adaptive_temperature_and_restart
                and stale >= 4 and not policy_restarted and iteration + 1 < iterations
            ):
                self.policy.restart_from_reference()
                policy_restarted = True
                next_temperature = min(2.0, max(temperature, current_temperature) * 1.20)
                control_action = "restart_and_explore"
                control_reason = "four_rounds_without_reward_improvement"
                stale = 0
            elif self.agent_config.use_adaptive_temperature_and_restart and stale >= 2:
                next_temperature = min(2.0, current_temperature * 1.12)
                control_action = "increase_exploration"
                control_reason = "two_rounds_without_reward_improvement"
            elif (
                self.agent_config.use_adaptive_temperature_and_restart
                and any(item.conservative_feasible for item in new_candidates)
            ):
                next_temperature = max(1.0, current_temperature * 0.96)
                control_action = "exploit_feasible_front"
                control_reason = "batch_contains_conservatively_feasible_candidates"
            if not self.agent_config.use_adaptive_temperature_and_restart:
                control_action = "fixed_temperature_no_restart"
                control_reason = "adaptive_temperature_and_restart_disabled"
            decision = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "generated": batch_size,
                "valid": valid_count,
                "unique_archive": len(self.archive),
                "selected_energyplus": [candidate.structure_hash for candidate in selected],
                "selected_acquisition_reasons": [
                    candidate.acquisition_reason for candidate in selected
                ],
                "adaptation": adaptation,
                "best_reward": best_reward,
                "stale_iterations": stale,
                "energyplus_memory_size": len(self.residuals),
                "temperature_used": current_temperature,
                "next_temperature": next_temperature,
                "control_action": control_action,
                "control_reason": control_reason,
                "agent_variant": self.agent_config.variant,
                "agent_config": asdict(self.agent_config),
                "udi_denominator_mode": self.udi_denominator_mode,
                "udi_prediction_scale": self.udi_prediction_scale,
            }
            with decision_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
            current_temperature = next_temperature
            if (
                self.agent_config.use_early_stopping
                and stale >= stop_patience
                and iteration + 1 < iterations
            ):
                stop_reason = "reward_plateau_patience_reached"
                stop_snapshot = {
                    "iteration": iteration,
                    "stale_iterations": stale,
                    "directly_feasible_incumbent": any(
                        item.direct_status == "success"
                        and item.direct_udi is not None
                        and item.direct_udi >= self.udi_threshold
                        for item in self.archive.values()
                    ),
                    "unqueried_optimistically_feasible_candidates": sum(
                        item.direct_status == "not_queried" and item.optimistic_feasible
                        for item in self.archive.values()
                    ),
                    "unqueried_conservatively_feasible_candidates": sum(
                        item.direct_status == "not_queried" and item.conservative_feasible
                        for item in self.archive.values()
                    ),
                }
                break

        target_unique_candidates = iterations * batch_size
        fill_rounds = 0
        while len(self.archive) < target_unique_candidates:
            if fill_rounds >= max(20, iterations * 10):
                raise RuntimeError(
                    f"Could not fill the frozen unique TMM budget: "
                    f"{len(self.archive)}/{target_unique_candidates}."
                )
            requested = min(batch_size, target_unique_candidates - len(self.archive))
            fill_iteration = iterations + fill_rounds
            rollout = self.policy.generate(self.target, requested, current_temperature)
            _, fill_candidates, valid_count = self._evaluate_rollout_batch(
                rollout, iteration=fill_iteration, batch_size=requested
            )
            self._assign_acquisition(fill_candidates)
            fill_decision = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "iteration": fill_iteration,
                "generated": requested,
                "valid": valid_count,
                "unique_archive": len(self.archive),
                "selected_energyplus": [],
                "selected_acquisition_reasons": [],
                "adaptation": {
                    "skipped": True,
                    "reason": "frozen_policy_unique_tmm_budget_fill",
                },
                "best_reward": best_reward,
                "stale_iterations": stale,
                "energyplus_memory_size": len(self.residuals),
                "temperature_used": current_temperature,
                "next_temperature": current_temperature,
                "control_action": "frozen_policy_budget_fill",
                "control_reason": "enforce_exact_unique_tmm_candidate_budget",
                "agent_variant": self.agent_config.variant,
                "agent_config": asdict(self.agent_config),
                "udi_denominator_mode": self.udi_denominator_mode,
                "udi_prediction_scale": self.udi_prediction_scale,
            }
            with decision_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fill_decision, ensure_ascii=False) + "\n")
            fill_rounds += 1

        candidates = sorted(self.archive.values(), key=lambda candidate: candidate.reward, reverse=True)
        remaining_total = (
            final_energyplus_queries
            if energyplus_total_budget is None
            else max(0, energyplus_total_budget - online_query_count)
        )
        final_query_allowance = min(final_energyplus_queries, remaining_total)
        final_selected = self._select_final_queries(
            candidates,
            final_query_allowance,
            allow_early_stop=self.agent_config.use_early_stopping,
        )
        final_query_count = len(final_selected)
        for candidate in final_selected:
            result = self.energyplus.run(
                candidate_id=f"final_{candidate.structure_hash[:10]}",
                structure=candidate.structure,
                cold=self.tmm_cache[candidate.structure_hash].cold,
                hot=self.tmm_cache[candidate.structure_hash].hot,
                allowed_tokens=self.vocab,
            )
            candidate.direct_status = result.metrics.status
            candidate.direct_udi = result.metrics.udi
            candidate.direct_energy_J = result.metrics.total_energy_J
            candidate.direct_run_directory = result.run_directory
            candidate.direct_cache_hit = result.cache_hit
            candidate.direct_cache_source_run_directory = result.cache_source_run_directory
            candidate.direct_input_signature = result.input_signature
            candidate.direct_started_utc = result.started_utc
            candidate.direct_completed_utc = result.completed_utc
            if result.metrics.status == "success":
                direct_saving = (
                    self.baseline_energy - candidate.direct_energy_J
                ) / self.baseline_energy
                direct_shortfall = max(
                    0.0, self.udi_threshold - candidate.direct_udi
                ) / self.udi_threshold
                candidate.direct_reward = direct_saving - 5.0 * direct_shortfall
            self._record_query(candidate, phase="final", iteration=iteration + 1)
        pareto = self._pareto_front(candidates)
        pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(
            self.run_directory / "candidates.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([asdict(candidate) for candidate in pareto]).to_csv(
            self.run_directory / "pareto.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(self.query_records).to_csv(
            self.run_directory / "query_trajectory.csv", index=False, encoding="utf-8-sig"
        )
        directly_feasible = [
            candidate for candidate in candidates
            if candidate.direct_status == "success"
            and candidate.direct_udi is not None
            and candidate.direct_udi >= self.udi_threshold
        ]
        directly_evaluated = [
            candidate for candidate in candidates
            if candidate.direct_status == "success" and candidate.direct_udi is not None
        ]
        conservatively_feasible = [candidate for candidate in candidates if candidate.conservative_feasible]
        if directly_feasible:
            best = min(directly_feasible, key=lambda item: item.direct_energy_J)
            selection_basis = "minimum_direct_energy_subject_to_direct_udi"
        elif directly_evaluated:
            best = min(
                directly_evaluated,
                key=lambda item: (-item.direct_udi, item.direct_energy_J),
            )
            selection_basis = "minimum_direct_udi_shortfall_then_energy_no_feasible"
        elif conservatively_feasible:
            best = min(conservatively_feasible, key=lambda item: item.predicted_energy_J)
            selection_basis = "minimum_predicted_energy_subject_to_conservative_udi"
        else:
            best = candidates[0] if candidates else None
            selection_basis = "no_feasible_candidate_highest_penalized_reward"
        summary = {
            "method": method_name,
            "city": self.city,
            "seed": self.seed,
            "run_directory": str(self.run_directory),
            "iterations_completed": iteration + 1,
            "iterations_planned": iterations,
            "stopped_early": iteration + 1 < iterations,
            "stop_reason": stop_reason,
            "stop_snapshot": stop_snapshot,
            "policy_restarted": policy_restarted,
            "target_unique_candidates": target_unique_candidates,
            "tmm_budget_fill_rounds": fill_rounds,
            "unique_candidates": len(candidates),
            "best": asdict(best) if best else None,
            "selection_basis": selection_basis,
            "pareto_candidates": len(pareto),
            "conservatively_feasible_candidates": len(conservatively_feasible),
            "udi_threshold": self.udi_threshold,
            "baseline_energy_J": self.baseline_energy,
            "energy_definition": "annual Electricity:Facility + NaturalGas:Facility (site energy, J)",
            "direct_queries": sum(candidate.direct_status != "not_queried" for candidate in candidates),
            "online_energyplus_queries_per_iteration": energyplus_queries_per_iteration,
            "energyplus_total_budget": energyplus_total_budget,
            "online_energyplus_queries": online_query_count,
            "final_energyplus_queries": final_query_count,
            "agent_variant": self.agent_config.variant,
            "agent_config": asdict(self.agent_config),
            "udi_denominator_mode": self.udi_denominator_mode,
            "udi_prediction_scale": self.udi_prediction_scale,
            "baseline_file": self.baseline_file,
            "query_trajectory": str(self.run_directory / "query_trajectory.csv"),
        }
        (self.run_directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
