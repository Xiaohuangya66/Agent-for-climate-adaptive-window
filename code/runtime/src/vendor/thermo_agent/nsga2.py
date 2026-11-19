from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def constraint_violation(candidate: Any, udi_threshold: float) -> float:
    conservative_udi = candidate.predicted_udi - candidate.udi_half_width_90pct
    return max(0.0, udi_threshold - conservative_udi) / max(udi_threshold, 1e-12)


def constrained_dominates(left: Any, right: Any, udi_threshold: float) -> bool:
    """Deb-style constraint domination followed by two-objective Pareto domination."""
    left_violation = constraint_violation(left, udi_threshold)
    right_violation = constraint_violation(right, udi_threshold)
    left_feasible = left_violation <= 1e-15
    right_feasible = right_violation <= 1e-15
    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        return left_violation < right_violation - 1e-15
    no_worse = (
        left.predicted_energy_J <= right.predicted_energy_J
        and left.predicted_udi >= right.predicted_udi
    )
    strictly_better = (
        left.predicted_energy_J < right.predicted_energy_J
        or left.predicted_udi > right.predicted_udi
    )
    return no_worse and strictly_better


def fast_non_dominated_sort(
    population: Sequence[Any], udi_threshold: float
) -> list[list[int]]:
    domination_count = [0] * len(population)
    dominates: list[list[int]] = [[] for _ in population]
    first_front: list[int] = []
    for left_index, left in enumerate(population):
        for right_index, right in enumerate(population):
            if left_index == right_index:
                continue
            if constrained_dominates(left, right, udi_threshold):
                dominates[left_index].append(right_index)
            elif constrained_dominates(right, left, udi_threshold):
                domination_count[left_index] += 1
        if domination_count[left_index] == 0:
            first_front.append(left_index)
    fronts = [first_front] if first_front else []
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for left_index in fronts[current]:
            for right_index in dominates[left_index]:
                domination_count[right_index] -= 1
                if domination_count[right_index] == 0:
                    next_front.append(right_index)
        if next_front:
            fronts.append(next_front)
        current += 1
    return fronts


def crowding_distance(population: Sequence[Any], front: Sequence[int]) -> dict[int, float]:
    distances = {index: 0.0 for index in front}
    if not front:
        return distances
    if len(front) <= 2:
        return {index: math.inf for index in front}
    objectives = (
        lambda candidate: float(candidate.predicted_energy_J),
        lambda candidate: float(candidate.predicted_udi),
    )
    for objective in objectives:
        ordered = sorted(front, key=lambda index: objective(population[index]))
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        minimum = objective(population[ordered[0]])
        maximum = objective(population[ordered[-1]])
        span = maximum - minimum
        if span <= 0:
            continue
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if math.isinf(distances[index]):
                continue
            previous_value = objective(population[ordered[position - 1]])
            next_value = objective(population[ordered[position + 1]])
            distances[index] += (next_value - previous_value) / span
    return distances


def rank_and_crowding(
    population: Sequence[Any], udi_threshold: float
) -> tuple[dict[int, int], dict[int, float], list[list[int]]]:
    fronts = fast_non_dominated_sort(population, udi_threshold)
    ranks: dict[int, int] = {}
    distances: dict[int, float] = {}
    for rank, front in enumerate(fronts):
        ranks.update({index: rank for index in front})
        distances.update(crowding_distance(population, front))
    return ranks, distances, fronts


def binary_tournament_index(
    population: Sequence[Any],
    ranks: dict[int, int],
    distances: dict[int, float],
) -> int:
    if len(population) == 1:
        return 0
    left, right = random.sample(range(len(population)), 2)
    if ranks[left] != ranks[right]:
        return left if ranks[left] < ranks[right] else right
    if distances[left] != distances[right]:
        return left if distances[left] > distances[right] else right
    return left if random.random() < 0.5 else right


def select_survivors(
    population: Sequence[Any], size: int, udi_threshold: float
) -> list[Any]:
    if size <= 0:
        return []
    _, _, fronts = rank_and_crowding(population, udi_threshold)
    selected: list[Any] = []
    for front in fronts:
        remaining = size - len(selected)
        if remaining <= 0:
            break
        distances = crowding_distance(population, front)
        ordered = sorted(front, key=lambda index: distances[index], reverse=True)
        selected.extend(population[index] for index in ordered[:remaining])
    return selected
