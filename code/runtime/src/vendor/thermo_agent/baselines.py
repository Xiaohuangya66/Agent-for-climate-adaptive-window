from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import Candidate, CityDesignAgent, load_target_spectrum
from .energyplus import EnergyPlusRunner
from .optical import FEATURE_ORDER, OpticalFeatureCalculator
from .policy import TTTPolicyAdapter
from .nsga2 import binary_tournament_index, rank_and_crowding, select_survivors
from .structures import load_vocab_tokens
from .surrogate import SurrogateEnsemble
from .tmm import TMMCalculator, TMMResult


class CityCandidateEvaluator:
    def __init__(
        self,
        workspace: str | Path,
        city: str,
        run_directory: str | Path,
        *,
        baseline_file: str | Path | None = None,
        udi_denominator_mode: str = "all_hours",
        energyplus_additional_idf_text: str = "",
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.city = city
        self.run_directory = Path(run_directory).resolve()
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.vocab = load_vocab_tokens(self.workspace / "dataset" / "vocab.txt")
        self.tmm = TMMCalculator(self.workspace, max_layers=20)
        self.optical = OpticalFeatureCalculator(self.workspace)
        self.surrogate = SurrogateEnsemble(
            self.workspace / "agent_project" / "models" / "surrogates" / city
        )
        baseline_path = (
            Path(baseline_file)
            if baseline_file is not None
            else self.workspace / "agent_project" / "results" / "city_baselines.csv"
        )
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[baseline["city"] == city].iloc[0]
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
        self.tmm_cache: dict[str, TMMResult] = {}
        self.query_records: list[dict] = []

    def evaluate(self, structures: list[str], iteration: int = 0) -> list[Candidate]:
        valid = []
        rows = []
        for structure in structures:
            try:
                result = self.tmm.calculate(structure, allowed_tokens=self.vocab)
            except Exception:
                continue
            digest = result.structure.structure_hash
            if digest in self.tmm_cache:
                continue
            self.tmm_cache[digest] = result
            features = self.optical.calculate(result)
            valid.append((structure, digest, features))
            rows.append([features[name] for name in FEATURE_ORDER])
        if not valid:
            return []
        prediction = self.surrogate.predict(np.asarray(rows))
        candidates = []
        threshold = float(self.surrogate.metadata["ood_distance_threshold_95pct_train"])
        for index, (structure, digest, features) in enumerate(valid):
            udi, energy = map(float, prediction.mean[index])
            udi *= self.udi_prediction_scale
            half_udi, half_energy = map(float, prediction.conformal_half_width_90pct[index])
            half_udi *= self.udi_prediction_scale
            saving = (self.baseline_energy - energy) / self.baseline_energy
            conservative = udi - half_udi
            shortfall = max(0.0, self.udi_threshold - conservative) / self.udi_threshold
            reward = saving - 5.0 * shortfall - 0.05 * half_energy / self.baseline_energy
            acquisition = (
                reward + 0.25 * half_energy / self.baseline_energy
                + 0.10 * float(prediction.ood_distance[index]) / max(threshold, 1e-9)
            )
            candidates.append(
                Candidate(
                    iteration=iteration,
                    rollout_index=index,
                    structure=structure,
                    structure_hash=digest,
                    features=features,
                    predicted_udi=udi,
                    predicted_energy_J=energy,
                    udi_std=float(prediction.std[index, 0]) * self.udi_prediction_scale,
                    energy_std_J=float(prediction.std[index, 1]),
                    udi_half_width_90pct=half_udi,
                    energy_half_width_90pct=half_energy,
                    ood_distance=float(prediction.ood_distance[index]),
                    reward=reward,
                    acquisition=acquisition,
                    energy_saving_ratio=saving,
                    conservative_feasible=conservative >= self.udi_threshold,
                    optimistic_feasible=udi + half_udi >= self.udi_threshold,
                    feasibility_probability=CityDesignAgent._normal_cdf(
                        (udi - self.udi_threshold)
                        / max(
                            float(prediction.std[index, 0])
                            * self.udi_prediction_scale,
                            1e-6,
                        )
                    ),
                    query_eligible=udi + half_udi >= self.udi_threshold,
                    acquisition_reason="baseline_static_surrogate_ranking",
                )
            )
        return candidates

    def validate_direct(self, candidates: list[Candidate], budget: int) -> None:
        selected = CityDesignAgent._select_final_queries(self, candidates, budget)
        for candidate in selected:
            result = self.energyplus.run(
                candidate_id=candidate.structure_hash[:12],
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
            direct_feasible = (
                candidate.direct_status == "success"
                and candidate.direct_udi is not None
                and candidate.direct_udi >= self.udi_threshold
            )
            feasible_energies = [
                row["direct_energy_J"]
                for row in self.query_records
                if row["direct_feasible"] and row["direct_energy_J"] is not None
            ]
            if direct_feasible and candidate.direct_energy_J is not None:
                feasible_energies.append(candidate.direct_energy_J)
            best_energy = min(feasible_energies) if feasible_energies else None
            self.query_records.append(
                {
                    "query_index": len(self.query_records) + 1,
                    "phase": "final_nested",
                    "city": self.city,
                    "structure": candidate.structure,
                    "structure_hash": candidate.structure_hash,
                    "predicted_udi": candidate.predicted_udi,
                    "predicted_energy_J": candidate.predicted_energy_J,
                    "udi_half_width_90pct": candidate.udi_half_width_90pct,
                    "energy_half_width_90pct": candidate.energy_half_width_90pct,
                    "ood_distance": candidate.ood_distance,
                    "direct_status": candidate.direct_status,
                    "direct_udi": candidate.direct_udi,
                    "direct_energy_J": candidate.direct_energy_J,
                    "direct_feasible": direct_feasible,
                    "best_direct_feasible_energy_J": best_energy,
                    "best_direct_feasible_energy_saving_ratio": (
                        (self.baseline_energy - best_energy) / self.baseline_energy
                        if best_energy is not None else None
                    ),
                    "cache_hit": candidate.direct_cache_hit,
                    "run_directory": candidate.direct_run_directory,
                    "input_signature": candidate.direct_input_signature,
                    "started_utc": candidate.direct_started_utc,
                    "completed_utc": candidate.direct_completed_utc,
                }
            )
            record = self.query_records[-1]
            with (self.run_directory / "queries.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            pd.DataFrame(self.query_records).to_csv(
                self.run_directory / "query_trajectory.csv",
                index=False,
                encoding="utf-8-sig",
            )

    def save(self, method: str, seed: int, candidates: list[Candidate], budget: int) -> dict:
        candidates = sorted(candidates, key=lambda item: item.reward, reverse=True)
        pareto = CityDesignAgent._pareto_front(candidates)
        feasible_direct = [
            item for item in candidates
            if item.direct_status == "success" and item.direct_udi >= self.udi_threshold
        ]
        evaluated_direct = [
            item for item in candidates
            if item.direct_status == "success" and item.direct_udi is not None
        ]
        feasible_predicted = [item for item in candidates if item.conservative_feasible]
        if feasible_direct:
            best = min(feasible_direct, key=lambda item: item.direct_energy_J)
            basis = "minimum_direct_energy_subject_to_direct_udi"
        elif evaluated_direct:
            best = min(
                evaluated_direct,
                key=lambda item: (-item.direct_udi, item.direct_energy_J),
            )
            basis = "minimum_direct_udi_shortfall_then_energy_no_feasible"
        elif feasible_predicted:
            best = min(feasible_predicted, key=lambda item: item.predicted_energy_J)
            basis = "minimum_predicted_energy_subject_to_conservative_udi"
        else:
            best = candidates[0] if candidates else None
            basis = "no_feasible_candidate_highest_penalized_reward"
        pd.DataFrame([asdict(item) for item in candidates]).to_csv(
            self.run_directory / "candidates.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([asdict(item) for item in pareto]).to_csv(
            self.run_directory / "pareto.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(self.query_records).to_csv(
            self.run_directory / "query_trajectory.csv", index=False, encoding="utf-8-sig"
        )
        summary = {
            "method": method,
            "city": self.city,
            "seed": seed,
            "run_directory": str(self.run_directory),
            "tmm_budget": budget,
            "unique_candidates": len(candidates),
            "direct_queries": sum(item.direct_status != "not_queried" for item in candidates),
            "selection_basis": basis,
            "best": asdict(best) if best else None,
            "pareto_candidates": len(pareto),
            "udi_threshold": self.udi_threshold,
            "baseline_energy_J": self.baseline_energy,
            "energy_definition": "annual Electricity:Facility + NaturalGas:Facility (site energy, J)",
            "udi_denominator_mode": self.udi_denominator_mode,
            "udi_prediction_scale": self.udi_prediction_scale,
            "query_trajectory": str(self.run_directory / "query_trajectory.csv"),
        }
        (self.run_directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def random_structure(vocab: set[str], max_layers: int = 20) -> str:
    terminal = "SiO2_10000"
    valid_wvo2 = sorted(
        token for token in vocab
        if token.startswith("WVO2_") and 20 <= float(token.split("_", 1)[1]) <= 40
    )
    general = sorted(
        token for token in vocab
        if "_" in token and token != terminal and not token.startswith("WVO2_")
    )
    layer_count = random.randint(2, max_layers)
    materials = [random.choice(general) for _ in range(layer_count - 2)]
    materials.insert(random.randint(0, len(materials)), random.choice(valid_wvo2))
    materials.append(terminal)
    return " ".join(materials)


def repair_structure(tokens: list[str], vocab: set[str], max_layers: int = 20) -> str:
    terminal = "SiO2_10000"
    valid_wvo2 = sorted(
        token for token in vocab
        if token.startswith("WVO2_") and 20 <= float(token.split("_", 1)[1]) <= 40
    )
    general = [
        token for token in tokens
        if token in vocab and token != terminal and not token.startswith("WVO2_")
    ][: max_layers - 2]
    position = random.randint(0, len(general))
    general.insert(position, random.choice(valid_wvo2))
    general.append(terminal)
    return " ".join(general)


def mutate(
    structure: str,
    vocab: set[str],
    probability: float = 0.15,
    max_layers: int = 20,
) -> str:
    tokens = structure.split()[:-1]
    general_vocab = sorted(
        token for token in vocab
        if "_" in token and token != "SiO2_10000" and not token.startswith("WVO2_")
    )
    for index, token in enumerate(tokens):
        if random.random() < probability and not token.startswith("WVO2_"):
            tokens[index] = random.choice(general_vocab)
    if random.random() < probability and len(tokens) < max_layers - 1:
        tokens.insert(random.randrange(len(tokens) + 1), random.choice(general_vocab))
    if random.random() < probability and len(tokens) > 1:
        removable = [index for index, token in enumerate(tokens) if not token.startswith("WVO2_")]
        if removable:
            tokens.pop(random.choice(removable))
    return repair_structure(tokens, vocab, max_layers=max_layers)


def crossover(left: str, right: str, vocab: set[str], max_layers: int = 20) -> str:
    a, b = left.split()[:-1], right.split()[:-1]
    cut_a = random.randint(0, len(a))
    cut_b = random.randint(0, len(b))
    return repair_structure(a[:cut_a] + b[cut_b:], vocab, max_layers=max_layers)


def search_random(evaluator: CityCandidateEvaluator, budget: int) -> list[Candidate]:
    candidates = []
    while len(candidates) < budget:
        candidates.extend(evaluator.evaluate([random_structure(evaluator.vocab)]))
    return candidates[:budget]


def search_transformer(
    evaluator: CityCandidateEvaluator, budget: int, temperature: float
) -> list[Candidate]:
    policy = TTTPolicyAdapter(
        evaluator.workspace,
        checkpoint=evaluator.workspace / "checkpoints" / "model_S-epoch=1198-val_loss_epoch=3.436.ckpt",
        vocab_file=evaluator.workspace / "dataset" / "vocab.txt",
        max_layers=20,
    )
    target = load_target_spectrum(evaluator.workspace / "ideal_target_2.csv")
    candidates = []
    attempts = 0
    while len(candidates) < budget and attempts < budget * 4:
        count = min(32, budget - len(candidates))
        rollout = policy.generate(target, count, temperature)
        candidates.extend(evaluator.evaluate([" ".join(tokens) for tokens in rollout.material_strings]))
        attempts += count
    return candidates[:budget]


def search_genetic_algorithm_legacy(
    evaluator: CityCandidateEvaluator, budget: int, population_size: int = 16
) -> list[Candidate]:
    """Historical scalar-reward genetic search formerly mislabeled NSGA-II."""
    population = [random_structure(evaluator.vocab) for _ in range(min(population_size, budget))]
    archive = evaluator.evaluate(population, iteration=0)
    evaluated = len(archive)
    generation = 1
    while evaluated < budget:
        parents = sorted(archive, key=lambda item: item.reward, reverse=True)[:population_size]
        offspring = []
        while len(offspring) < min(population_size, budget - evaluated):
            left, right = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            offspring.append(mutate(crossover(left.structure, right.structure, evaluator.vocab), evaluator.vocab))
        new = evaluator.evaluate(offspring, iteration=generation)
        archive.extend(new)
        evaluated += len(new)
        if not new:
            population_size += 1
        generation += 1
    return archive[:budget]


def search_nsga2(
    evaluator: CityCandidateEvaluator,
    budget: int,
    population_size: int = 16,
    max_layers: int = 20,
) -> list[Candidate]:
    """Constraint-aware, elitist two-objective NSGA-II.

    Energy is minimized and UDI is maximized.  Deb-style constraint
    domination gives priority to candidates whose conservative 90% UDI lower
    bound reaches the city threshold.  Scalar ``Candidate.reward`` is retained
    only for the common result schema and is never used for evolutionary
    selection.
    """
    if budget <= 0:
        return []
    population_size = max(2, min(population_size, budget))
    archive: list[Candidate] = []
    generation = 0

    while len(archive) < population_size:
        requested = population_size - len(archive)
        structures = [
            random_structure(evaluator.vocab, max_layers=max_layers)
            for _ in range(max(requested, 4))
        ]
        archive.extend(evaluator.evaluate(structures, iteration=generation)[:requested])
        generation += 1
    population = list(archive)

    stalled_rounds = 0
    while len(archive) < budget:
        ranks, distances, _ = rank_and_crowding(population, evaluator.udi_threshold)
        requested = min(population_size, budget - len(archive))
        offspring: list[str] = []
        seen = {candidate.structure for candidate in archive}
        attempts = 0
        while len(offspring) < requested and attempts < requested * 50:
            left = population[binary_tournament_index(population, ranks, distances)]
            right = population[binary_tournament_index(population, ranks, distances)]
            child = mutate(
                crossover(
                    left.structure, right.structure, evaluator.vocab,
                    max_layers=max_layers,
                ),
                evaluator.vocab,
                max_layers=max_layers,
            )
            attempts += 1
            if child not in seen and child not in offspring:
                offspring.append(child)
        if len(offspring) < requested:
            while len(offspring) < requested:
                child = random_structure(evaluator.vocab, max_layers=max_layers)
                if child not in seen and child not in offspring:
                    offspring.append(child)

        new = evaluator.evaluate(offspring, iteration=generation)
        if not new:
            stalled_rounds += 1
            if stalled_rounds >= 10:
                raise RuntimeError("NSGA-II could not produce a new unique TMM candidate.")
            generation += 1
            continue
        stalled_rounds = 0
        archive.extend(new[: budget - len(archive)])
        population = select_survivors(
            [*population, *new], population_size, evaluator.udi_threshold
        )
        generation += 1

    final_ranks, final_distances, _ = rank_and_crowding(archive, evaluator.udi_threshold)
    for index, candidate in enumerate(archive):
        distance = final_distances[index]
        crowding = "inf" if np.isinf(distance) else f"{distance:.8g}"
        candidate.acquisition_reason = (
            f"standard_nsga2;pareto_rank={final_ranks[index]};crowding_distance={crowding};"
            "objectives=min_energy,max_udi;constraint=conservative_udi"
        )
    return archive
