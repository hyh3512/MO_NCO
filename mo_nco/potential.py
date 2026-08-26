from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .archive import ParetoArchive
from .types import ObjectiveVector


@dataclass(frozen=True)
class PotentialContext:
    """Frozen context used between two archive update stopping times."""

    ideal: ObjectiveVector
    nadir: ObjectiveVector
    reference_directions: Tuple[ObjectiveVector, ...]
    archive_objectives: Tuple[ObjectiveVector, ...] = ()


class ScalarArchivePotential:
    """Scalar mean-field potential for the IPS sampler.

    The empirical energy is

        E(mu) = average_x phi(x) + diversity_weight * average_pair_kernel(x, y).

    This is a scalar potential, so replacement deltas are exact Hamiltonian
    differences for the frozen context. The archive only changes at explicit
    stopping times handled by the sampler.
    """

    def __init__(
        self,
        reference_count: int = 11,
        chebyshev_rho: float = 0.05,
        diversity_weight: float = 0.03,
        diversity_sigma: float = 0.2,
        scale_epsilon: float = 1e-9,
    ) -> None:
        self.reference_count = reference_count
        self.chebyshev_rho = chebyshev_rho
        self.diversity_weight = diversity_weight
        self.diversity_sigma = diversity_sigma
        self.scale_epsilon = scale_epsilon

    def build_context(
        self,
        archive: ParetoArchive,
        current_objectives: Iterable[ObjectiveVector],
    ) -> PotentialContext:
        current = tuple(current_objectives)
        ideal, nadir = archive.ideal_nadir(current)
        dim = len(ideal)
        return PotentialContext(
            ideal=ideal,
            nadir=nadir,
            reference_directions=self.reference_directions(dim, self.reference_count),
            archive_objectives=archive.objectives(),
        )

    def empirical_energy(
        self,
        objectives: Sequence[ObjectiveVector],
        context: PotentialContext,
    ) -> float:
        if not objectives:
            raise ValueError("At least one objective vector is required.")
        k = len(objectives)
        single = sum(self.single_energy(obj, context) for obj in objectives) / k
        if self.diversity_weight <= 0.0 or k < 2:
            return single
        pair_sum = 0.0
        pair_count = 0
        for i in range(k):
            for j in range(i + 1, k):
                pair_sum += self.kernel(objectives[i], objectives[j], context)
                pair_count += 1
        diversity = self.diversity_weight * pair_sum / pair_count
        return single + diversity

    def delta_replace(
        self,
        objectives: Sequence[ObjectiveVector],
        index: int,
        new_objective: ObjectiveVector,
        context: PotentialContext,
    ) -> float:
        """Exact empirical-energy difference for replacing one particle.

        The previous implementation recomputed the full O(k^2) empirical
        energy twice.  Only the replaced single-site term and the k-1 pair
        terms can change, so the same exact delta is available in O(k).
        """
        if not objectives:
            raise ValueError("At least one objective vector is required.")
        if index < 0 or index >= len(objectives):
            raise IndexError("replacement index is out of range.")

        k = len(objectives)
        old_objective = objectives[index]
        delta = (
            self.single_energy(new_objective, context)
            - self.single_energy(old_objective, context)
        ) / k

        if self.diversity_weight > 0.0 and k >= 2:
            pair_count = k * (k - 1) // 2
            pair_delta = 0.0
            for other_index, other in enumerate(objectives):
                if other_index == index:
                    continue
                pair_delta += self.kernel(new_objective, other, context)
                pair_delta -= self.kernel(old_objective, other, context)
            delta += self.diversity_weight * pair_delta / pair_count
        return delta

    def single_energy(self, objective: ObjectiveVector, context: PotentialContext) -> float:
        z = self.normalize(objective, context)
        best = float("inf")
        for weight in context.reference_directions:
            cheb = max(w * value for w, value in zip(weight, z))
            aug = self.chebyshev_rho * sum(w * value for w, value in zip(weight, z))
            best = min(best, cheb + aug)
        return best

    def kernel(self, a: ObjectiveVector, b: ObjectiveVector, context: PotentialContext) -> float:
        if self.diversity_sigma <= 0.0:
            return 0.0
        za = self.normalize(a, context)
        zb = self.normalize(b, context)
        dist2 = sum((x - y) ** 2 for x, y in zip(za, zb))
        return math.exp(-dist2 / (2.0 * self.diversity_sigma**2))

    def normalize(self, objective: ObjectiveVector, context: PotentialContext) -> ObjectiveVector:
        values = []
        for value, lo, hi in zip(objective, context.ideal, context.nadir):
            scale = max(self.scale_epsilon, hi - lo)
            values.append((value - lo) / scale)
        return tuple(values)

    @staticmethod
    def reference_directions(dim: int, count: int) -> Tuple[ObjectiveVector, ...]:
        """Return exactly ``count`` deterministic simplex directions.

        The old many-objective fallback returned only ``dim + 1`` vectors,
        which made population/weight indexing fail whenever ``count`` was
        larger.  Two objectives retain the existing evenly spaced rule.  For
        more objectives, a deterministic Das--Dennis simplex lattice is built
        and evenly subsampled to the requested count.
        """
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if count <= 0:
            raise ValueError("count must be positive.")
        if dim == 1:
            return tuple((1.0,) for _ in range(count))
        if count == 1:
            return (tuple(1.0 / dim for _ in range(dim)),)
        if dim == 2:
            dirs = []
            floor = 1e-3
            for idx in range(count):
                w0 = idx / (count - 1)
                w1 = 1.0 - w0
                w0 = max(floor, w0)
                w1 = max(floor, w1)
                total = w0 + w1
                dirs.append((w0 / total, w1 / total))
            return tuple(dirs)

        def compositions(total: int, parts: int):
            if parts == 1:
                yield (total,)
                return
            for first in range(total + 1):
                for tail in compositions(total - first, parts - 1):
                    yield (first,) + tail

        divisions = 1
        while math.comb(divisions + dim - 1, dim - 1) < count:
            divisions += 1
        lattice = list(compositions(divisions, dim))
        floor = 1e-3
        directions = []
        for composition in lattice:
            raw = [max(floor, value / divisions) for value in composition]
            total = sum(raw)
            directions.append(tuple(value / total for value in raw))

        # The lattice cardinality is at least count.  Evenly spaced indices
        # preserve extremes and spread the retained directions deterministically.
        last = len(directions) - 1
        indices = [round(position * last / (count - 1)) for position in range(count)]
        return tuple(directions[index] for index in indices)


class HypervolumeArchivePotential(ScalarArchivePotential):
    """Archive-conditioned scalar potential that directly rewards coverage.

    For a frozen archive context, the empirical energy remains a scalar
    functional of the particle empirical measure:

        E(mu) = single_weight * average phi(x)
                - coverage_weight * HV_2D(archive union particles)
                + diversity_weight * pair repulsion.

    The negative hypervolume term is only used for two objectives; for higher
    dimensions the class falls back to the parent scalar energy.
    """

    def __init__(
        self,
        reference_count: int = 21,
        chebyshev_rho: float = 0.03,
        diversity_weight: float = 0.01,
        diversity_sigma: float = 0.18,
        scale_epsilon: float = 1e-9,
        single_weight: float = 0.2,
        coverage_weight: float = 2.5,
        include_archive_in_coverage: bool = False,
    ) -> None:
        super().__init__(
            reference_count=reference_count,
            chebyshev_rho=chebyshev_rho,
            diversity_weight=diversity_weight,
            diversity_sigma=diversity_sigma,
            scale_epsilon=scale_epsilon,
        )
        self.single_weight = single_weight
        self.coverage_weight = coverage_weight
        self.include_archive_in_coverage = include_archive_in_coverage

    def empirical_energy(
        self,
        objectives: Sequence[ObjectiveVector],
        context: PotentialContext,
    ) -> float:
        if not objectives:
            raise ValueError("At least one objective vector is required.")
        if len(objectives[0]) != 2:
            return super().empirical_energy(objectives, context)

        k = len(objectives)
        single = sum(self.single_energy(obj, context) for obj in objectives) / k
        energy = self.single_weight * single
        coverage_points = (
            tuple(context.archive_objectives) + tuple(objectives)
            if self.include_archive_in_coverage
            else tuple(objectives)
        )
        energy -= self.coverage_weight * self.normalized_hypervolume_2d(coverage_points, context)

        if self.diversity_weight > 0.0 and k >= 2:
            pair_sum = 0.0
            pair_count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    pair_sum += self.kernel(objectives[i], objectives[j], context)
                    pair_count += 1
            energy += self.diversity_weight * pair_sum / pair_count
        return energy

    def delta_replace(
        self,
        objectives: Sequence[ObjectiveVector],
        index: int,
        new_objective: ObjectiveVector,
        context: PotentialContext,
    ) -> float:
        if len(objectives[0]) != 2:
            return super().delta_replace(objectives, index, new_objective, context)

        k = len(objectives)
        old_objective = objectives[index]
        delta = self.single_weight * (
            self.single_energy(new_objective, context)
            -
            self.single_energy(old_objective, context)
        ) / k

        coverage_points = (
            tuple(context.archive_objectives) + tuple(objectives)
            if self.include_archive_in_coverage
            else tuple(objectives)
        )
        old_hv = self.normalized_hypervolume_2d(coverage_points, context)
        updated = list(objectives)
        updated[index] = new_objective
        new_coverage_points = (
            tuple(context.archive_objectives) + tuple(updated)
            if self.include_archive_in_coverage
            else tuple(updated)
        )
        new_hv = self.normalized_hypervolume_2d(new_coverage_points, context)
        delta -= self.coverage_weight * (new_hv - old_hv)

        if self.diversity_weight > 0.0 and k >= 2:
            pair_count = k * (k - 1) // 2
            pair_delta = 0.0
            for idx, other in enumerate(objectives):
                if idx == index:
                    continue
                pair_delta += self.kernel(new_objective, other, context)
                pair_delta -= self.kernel(old_objective, other, context)
            delta += self.diversity_weight * pair_delta / pair_count
        return delta

    def normalized_hypervolume_2d(
        self,
        objectives: Sequence[ObjectiveVector],
        context: PotentialContext,
    ) -> float:
        if not objectives:
            return 0.0
        points = [self.normalize(obj, context) for obj in objectives]
        nondominated = []
        for point in sorted(points, key=lambda z: (z[0], z[1])):
            if point[0] > 1.35 or point[1] > 1.35:
                continue
            if not nondominated or point[1] < nondominated[-1][1]:
                nondominated.append(point)
        if not nondominated:
            return 0.0

        ref_x, ref_y = 1.12, 1.12
        hv = 0.0
        prev_y = ref_y
        for x, y in nondominated:
            width = max(0.0, ref_x - x)
            height = max(0.0, prev_y - y)
            hv += width * height
            prev_y = min(prev_y, y)
        return hv
