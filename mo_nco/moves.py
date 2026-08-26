from __future__ import annotations

import random
from typing import Tuple

from .types import Tour


def random_tour(num_cities: int, rng: random.Random) -> Tour:
    if num_cities < 3:
        raise ValueError("A tour requires at least three cities.")
    tail = list(range(1, num_cities))
    rng.shuffle(tail)
    return tuple([0] + tail)


def two_opt(tour: Tour, rng: random.Random) -> Tour:
    """Uniform symmetric 2-opt proposal with city 0 fixed."""
    if len(tour) < 4:
        return tour
    i, j = sample_two_opt_indices(len(tour), rng)
    return two_opt_at(tour, i, j)


def sample_two_opt_indices(num_cities: int, rng: random.Random) -> Tuple[int, int]:
    """Sample a valid 2-opt segment with city 0 fixed."""
    if num_cities < 4:
        raise ValueError("2-opt requires at least four cities.")
    i, j = sorted(rng.sample(range(1, num_cities), 2))
    return i, j


def city_swap(tour: Tour, rng: random.Random) -> Tour:
    """Uniform symmetric swap proposal with city 0 fixed."""
    n = len(tour)
    if n < 4:
        return tour
    i, j = rng.sample(range(1, n), 2)
    proposed = list(tour)
    proposed[i], proposed[j] = proposed[j], proposed[i]
    return tuple(proposed)


def adjacent_swap(tour: Tour, rng: random.Random) -> Tour:
    """Local symmetric adjacent swap with city 0 fixed."""
    n = len(tour)
    if n < 4:
        return tour
    i = rng.randrange(1, n - 1)
    proposed = list(tour)
    proposed[i], proposed[i + 1] = proposed[i + 1], proposed[i]
    return tuple(proposed)


def mixed_move(tour: Tour, rng: random.Random) -> Tour:
    """Symmetric proposal mixture used by the IPS sampler."""
    draw = rng.random()
    if draw < 0.55:
        return two_opt(tour, rng)
    if draw < 0.9:
        return city_swap(tour, rng)
    return adjacent_swap(tour, rng)


def order_crossover(parent_a: Tour, parent_b: Tour, rng: random.Random) -> Tour:
    """Order crossover on the tail while keeping city 0 fixed."""
    n = len(parent_a)
    if n <= 3:
        return parent_a
    i, j = sorted(rng.sample(range(1, n), 2))
    child = [-1] * n
    child[0] = 0
    child[i : j + 1] = parent_a[i : j + 1]
    used = {city for city in child if city >= 0}
    fill_values = [city for city in parent_b[1:] if city not in used]
    fill_idx = 0
    for idx in list(range(1, i)) + list(range(j + 1, n)):
        child[idx] = fill_values[fill_idx]
        fill_idx += 1
    return tuple(child)


def two_opt_at(tour: Tour, i: int, j: int) -> Tour:
    """Deterministic 2-opt move, useful for tests."""
    if i > j:
        i, j = j, i
    if i <= 0 or j >= len(tour):
        raise ValueError("2-opt indices must satisfy 1 <= i <= j < n.")
    proposed = list(tour)
    proposed[i : j + 1] = reversed(proposed[i : j + 1])
    return tuple(proposed)


def move_distance(_: Tour, __: Tour) -> int:
    """Shortest-path unit distance on the implicit feasible-move graph edge."""
    return 1
