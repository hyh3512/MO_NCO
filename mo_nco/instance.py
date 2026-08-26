from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .types import ObjectiveVector, Tour

Point = Tuple[float, float]
DistanceMatrix = Tuple[Tuple[float, ...], ...]


def instance_sha256(instance: object) -> str:
    """Fingerprint the exact objective matrices that define a TSP state function."""
    matrices = getattr(instance, "distance_matrices", None)
    num_cities = int(getattr(instance, "num_cities"))
    num_objectives = int(getattr(instance, "num_objectives"))
    if matrices is None:
        raise ValueError("Instance fingerprinting requires explicit distance matrices.")
    payload = {
        "num_cities": num_cities,
        "num_objectives": num_objectives,
        "distance_matrices": matrices,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MultiObjectiveTSPInstance:
    """A multi-objective Euclidean TSP instance.

    A feasible solution is a Hamiltonian cycle represented by a permutation with
    city 0 fixed at the first position. Each objective has its own coordinate
    system and therefore its own Euclidean distance matrix.
    """

    coords_by_objective: Tuple[Tuple[Point, ...], ...] = ()
    distance_matrices_by_objective: Tuple[DistanceMatrix, ...] = ()
    name: str = "instance"

    def __post_init__(self) -> None:
        if self.distance_matrices_by_objective:
            self._validate_distance_matrices(self.distance_matrices_by_objective)
            object.__setattr__(self, "_distance_matrices", self.distance_matrices_by_objective)
            object.__setattr__(self, "_symmetric_matrices", self._symmetric_flags(self.distance_matrices_by_objective))
            object.__setattr__(
                self,
                "_exact_two_opt_delta_flags",
                self._exact_binary64_delta_flags(
                    self.distance_matrices_by_objective
                ),
            )
            return

        if not self.coords_by_objective:
            raise ValueError("At least one objective is required.")
        n = len(self.coords_by_objective[0])
        if n < 3:
            raise ValueError("A TSP instance requires at least three cities.")
        for coords in self.coords_by_objective:
            if len(coords) != n:
                raise ValueError("All objectives must have the same number of cities.")

        matrices = tuple(self._build_distance_matrix(coords) for coords in self.coords_by_objective)
        object.__setattr__(self, "_distance_matrices", matrices)
        object.__setattr__(self, "_symmetric_matrices", tuple(True for _ in matrices))
        object.__setattr__(
            self,
            "_exact_two_opt_delta_flags",
            self._exact_binary64_delta_flags(matrices),
        )

    @property
    def num_cities(self) -> int:
        return len(self._distance_matrices[0])

    @property
    def num_objectives(self) -> int:
        return len(self._distance_matrices)

    @property
    def distance_matrices(self) -> Tuple[DistanceMatrix, ...]:
        """Read-only objective matrices for accelerators and audit code."""
        return self._distance_matrices

    @property
    def symmetric_objectives(self) -> Tuple[bool, ...]:
        return self._symmetric_matrices

    @property
    def exact_two_opt_delta_objectives(self) -> Tuple[bool, ...]:
        """Whether binary64 delta updates are exactly full-sum equivalent.

        The sufficient domain is deliberately conservative: a symmetric,
        nonnegative, integer-valued matrix whose largest possible tour sum is
        at most ``2**53``.  Every partial full-tour sum and every 2-opt
        add/subtract operation is then an exactly represented integer.
        """

        return self._exact_two_opt_delta_flags

    @property
    def exact_two_opt_delta_in_binary64(self) -> bool:
        return all(self._exact_two_opt_delta_flags)

    @property
    def objective_scale_estimates(self) -> ObjectiveVector:
        """Typical full-tour scales derived from positive off-diagonal edges."""
        estimates = []
        n = self.num_cities
        for matrix in self._distance_matrices:
            values = [
                float(matrix[i][j])
                for i in range(n)
                for j in range(n)
                if i != j and matrix[i][j] > 0.0
            ]
            edge_scale = sum(values) / len(values) if values else 1.0
            estimates.append(max(1.0, n * edge_scale))
        return tuple(estimates)

    def evaluate(self, tour: Tour) -> ObjectiveVector:
        self.validate_tour(tour)
        return self.evaluate_unchecked(tour)

    def evaluate_unchecked(self, tour: Tour) -> ObjectiveVector:
        """Evaluate a tour known to be a valid fixed-zero permutation.

        Optimizer-generated moves preserve feasibility, so hot loops can avoid
        an O(n log n) permutation sort while public entry points stay checked.
        """
        values = []
        for matrix in self._distance_matrices:
            total = 0.0
            for idx, city in enumerate(tour):
                nxt = tour[(idx + 1) % len(tour)]
                total += matrix[city][nxt]
            values.append(total)
        return tuple(values)

    def evaluate_two_opt(
        self,
        tour: Tour,
        current_objectives: ObjectiveVector,
        i: int,
        j: int,
    ) -> ObjectiveVector:
        """Exact objective vector after a 2-opt move.

        Symmetric TSP objectives admit an O(m)-objective delta because the
        reversed segment preserves all internal edge costs. For asymmetric
        matrices, the method falls back to full exact evaluation.
        """
        if i > j:
            i, j = j, i
        if i <= 0 or j >= len(tour):
            raise ValueError("2-opt indices must satisfy 1 <= i <= j < n.")
        if len(current_objectives) != self.num_objectives:
            raise ValueError("Current objective vector has the wrong dimension.")

        if not (
            all(self._symmetric_matrices)
            and self.exact_two_opt_delta_in_binary64
        ):
            proposed = list(tour)
            proposed[i : j + 1] = reversed(proposed[i : j + 1])
            return self.evaluate_unchecked(tuple(proposed))

        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % len(tour)]
        values = []
        for current, matrix in zip(current_objectives, self._distance_matrices):
            removed = matrix[a][b] + matrix[c][d]
            added = matrix[a][c] + matrix[b][d]
            values.append(current - removed + added)
        return tuple(values)

    @staticmethod
    def _exact_binary64_delta_flags(
        matrices: Sequence[DistanceMatrix],
    ) -> Tuple[bool, ...]:
        exact_integer_limit = 2**53
        flags = []
        for matrix in matrices:
            maximum_integer = 0
            safe = True
            for row in matrix:
                for raw_value in row:
                    value = float(raw_value)
                    if (
                        not math.isfinite(value)
                        or value < 0.0
                        or not value.is_integer()
                    ):
                        safe = False
                        break
                    maximum_integer = max(
                        maximum_integer,
                        int(value),
                    )
                if not safe:
                    break
            flags.append(
                safe
                and len(matrix) * maximum_integer
                <= exact_integer_limit
            )
        return tuple(flags)

    def validate_tour(self, tour: Tour) -> None:
        n = self.num_cities
        if len(tour) != n:
            raise ValueError(f"Expected tour length {n}, got {len(tour)}.")
        if tour[0] != 0:
            raise ValueError("Tours must fix city 0 at the first position.")
        if sorted(tour) != list(range(n)):
            raise ValueError("Tour must be a permutation of all cities.")

    @staticmethod
    def random_biobjective(num_cities: int, seed: int = 0, noise: float = 0.08) -> "MultiObjectiveTSPInstance":
        """Create a synthetic bi-objective Euclidean TSP with a visible tradeoff."""
        rng = random.Random(seed)
        coords_a: List[Point] = [(rng.random(), rng.random()) for _ in range(num_cities)]
        coords_b: List[Point] = []
        for x, y in coords_a:
            # Use a different geometry, not merely a rigid transform of the
            # first one. This creates nontrivial Pareto tradeoffs.
            rx = rng.random()
            ry = rng.random()
            nx = min(1.0, max(0.0, 0.35 * (1.0 - x) + 0.65 * rx + rng.uniform(-noise, noise)))
            ny = min(1.0, max(0.0, 0.35 * y + 0.65 * ry + rng.uniform(-noise, noise)))
            coords_b.append((nx, ny))
        return MultiObjectiveTSPInstance((tuple(coords_a), tuple(coords_b)), name=f"synthetic_biobj_{num_cities}_{seed}")

    @staticmethod
    def from_distance_matrices(
        matrices: Sequence[Sequence[Sequence[float]]],
        name: str = "matrix_instance",
    ) -> "MultiObjectiveTSPInstance":
        normalized = tuple(tuple(tuple(float(value) for value in row) for row in matrix) for matrix in matrices)
        return MultiObjectiveTSPInstance(distance_matrices_by_objective=normalized, name=name)

    @staticmethod
    def from_tsplib_files(paths: Sequence[str | Path]) -> "MultiObjectiveTSPInstance":
        from .tsplib import load_multiobjective_tsplib

        return load_multiobjective_tsplib(paths)

    @staticmethod
    def from_bitsp_file(path: str | Path) -> "MultiObjectiveTSPInstance":
        from .tsplib import load_bitsp

        return load_bitsp(path)

    @staticmethod
    def _build_distance_matrix(coords: Sequence[Point]) -> Tuple[Tuple[float, ...], ...]:
        matrix: List[Tuple[float, ...]] = []
        for ax, ay in coords:
            row = []
            for bx, by in coords:
                row.append(math.hypot(ax - bx, ay - by))
            matrix.append(tuple(row))
        return tuple(matrix)

    @staticmethod
    def _validate_distance_matrices(matrices: Sequence[DistanceMatrix]) -> None:
        if not matrices:
            raise ValueError("At least one distance matrix is required.")
        n = len(matrices[0])
        if n < 3:
            raise ValueError("A TSP instance requires at least three cities.")
        for matrix in matrices:
            if len(matrix) != n:
                raise ValueError("All objective matrices must have the same dimension.")
            for row in matrix:
                if len(row) != n:
                    raise ValueError("Distance matrices must be square.")
                for value in row:
                    if not math.isfinite(float(value)):
                        raise ValueError("Distance matrices must contain only finite values.")
                    if float(value) < 0.0:
                        raise ValueError("Distance matrices must be nonnegative.")

    @staticmethod
    def _symmetric_flags(matrices: Sequence[DistanceMatrix], tol: float = 1e-12) -> Tuple[bool, ...]:
        flags = []
        for matrix in matrices:
            symmetric = True
            for i, row in enumerate(matrix):
                for j in range(i + 1, len(row)):
                    if abs(row[j] - matrix[j][i]) > tol:
                        symmetric = False
                        break
                if not symmetric:
                    break
            flags.append(symmetric)
        return tuple(flags)
