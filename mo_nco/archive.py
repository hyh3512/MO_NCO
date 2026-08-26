from __future__ import annotations

from dataclasses import dataclass
import bisect
from typing import Iterable, List, Optional, Tuple

from .types import ObjectiveVector, Tour


@dataclass(frozen=True)
class ArchiveEntry:
    tour: Tour
    objectives: ObjectiveVector


def dominates(a: ObjectiveVector, b: ObjectiveVector, tol: float = 1e-12) -> bool:
    """Return True when objective vector a Pareto-dominates b for minimization."""
    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same dimension.")
    no_worse = all(x <= y + tol for x, y in zip(a, b))
    strictly_better = any(x < y - tol for x, y in zip(a, b))
    return no_worse and strictly_better


class ParetoArchive:
    """Maintain a nondominated archive for minimization objectives."""

    def __init__(self, max_size: Optional[int] = None, tol: float = 1e-12) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when provided.")
        if tol < 0.0:
            raise ValueError("tol must be nonnegative.")
        self.max_size = max_size
        self.tol = tol
        self._entries: List[ArchiveEntry] = []
        self._tour_index: set[Tour] = set()
        self._objective_index: set[ObjectiveVector] = set()
        self._entry_index: set[tuple[Tour, ObjectiveVector]] = set()

    @property
    def entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def objectives(self) -> Tuple[ObjectiveVector, ...]:
        return tuple(entry.objectives for entry in self._entries)

    def contains(self, entry: ArchiveEntry) -> bool:
        """Return whether the exact tour/objective pair is currently retained."""
        return (entry.tour, entry.objectives) in self._entry_index

    def update(self, candidates: Iterable[ArchiveEntry]) -> bool:
        """Insert candidates and keep only nondominated entries.

        Returns True when the archive changed.
        """
        candidates = list(candidates)
        if (
            len(candidates) == 1
            and self.tol == 0.0
            and self._can_use_2d_fast_update(candidates)
        ):
            return self._update_2d_exact_single(candidates[0])
        if candidates and self._can_use_2d_fast_update(candidates):
            return self._update_2d_fast(candidates)

        changed = False
        for candidate in candidates:
            if self._is_duplicate(candidate):
                continue
            if any(dominates(entry.objectives, candidate.objectives, self.tol) for entry in self._entries):
                continue
            self._entries = [
                entry
                for entry in self._entries
                if not dominates(candidate.objectives, entry.objectives, self.tol)
            ]
            self._entries.append(candidate)
            changed = True

        if self.max_size is not None and len(self._entries) > self.max_size:
            self._truncate_by_crowding()
            changed = True

        self._entries.sort(key=lambda item: item.objectives)
        self._rebuild_indices()
        return changed

    def _can_use_2d_fast_update(self, candidates: Iterable[ArchiveEntry]) -> bool:
        entries = list(self._entries)
        return all(len(entry.objectives) == 2 for entry in entries) and all(
            len(entry.objectives) == 2 for entry in candidates
        )

    def _rebuild_indices(self) -> None:
        self._tour_index = {entry.tour for entry in self._entries}
        self._objective_index = {entry.objectives for entry in self._entries}
        self._entry_index = {
            (entry.tour, entry.objectives) for entry in self._entries
        }

    def _update_2d_exact_single(self, candidate: ArchiveEntry) -> bool:
        """Incrementally update an exact 2D frontier without a full re-sort.

        Dominance search uses O(log n + k) comparisons, where k is the
        contiguous block removed by the candidate.  Because CPython stores the
        frontier in a list, a middle splice still has O(n) worst-case movement;
        the improvement over the previous path is that every insertion no
        longer sorts and re-reduces the complete archive.

        With zero dominance tolerance, a nondominated minimization frontier
        sorted by increasing first objective has strictly decreasing second
        objective.  The rightmost predecessor is therefore the only possible
        dominator among points to the left; points dominated by the candidate
        form one contiguous successor block.
        """

        if candidate.tour in self._tour_index:
            return False
        if candidate.objectives in self._objective_index:
            return False
        x, y = candidate.objectives
        key = (x, y, candidate.tour)
        insertion = bisect.bisect_left(
            self._entries,
            key,
            key=lambda entry: (
                entry.objectives[0],
                entry.objectives[1],
                entry.tour,
            ),
        )

        # Any exact dominator with first coordinate <= x is represented by the
        # predecessor with the smallest second coordinate.
        if insertion > 0:
            predecessor = self._entries[insertion - 1]
            px, py = predecessor.objectives
            if px <= x and py <= y and (px < x or py < y):
                return False

        # Equal-first-coordinate points may appear at insertion.  A lower y
        # dominates the candidate; a higher y is removed below.
        if insertion < len(self._entries):
            successor = self._entries[insertion]
            sx, sy = successor.objectives
            if sx <= x and sy <= y and (sx < x or sy < y):
                return False

        end = insertion
        while end < len(self._entries):
            ex, ey = self._entries[end].objectives
            if x <= ex and y <= ey and (x < ex or y < ey):
                end += 1
                continue
            break

        removed = self._entries[insertion:end]
        self._entries[insertion:end] = [candidate]
        for entry in removed:
            self._tour_index.discard(entry.tour)
            self._objective_index.discard(entry.objectives)
            self._entry_index.discard((entry.tour, entry.objectives))
        self._tour_index.add(candidate.tour)
        self._objective_index.add(candidate.objectives)
        self._entry_index.add((candidate.tour, candidate.objectives))

        if self.max_size is not None and len(self._entries) > self.max_size:
            self._truncate_by_crowding()
            self._entries.sort(key=lambda item: item.objectives)
            self._rebuild_indices()
        return True

    def _update_2d_fast(self, candidates: Iterable[ArchiveEntry]) -> bool:
        """Exact O(n log n) 2D reduction under the archive tolerance.

        A naive left-to-right sweep is incorrect when a later point lies within
        ``tol`` in the first coordinate but is substantially better in the
        second coordinate.  For each point b, the implementation checks the two
        logically exhaustive dominance cases using prefix minima:

        * some a has a_x <= b_x + tol and a_y < b_y - tol; or
        * some a has a_x <  b_x - tol and a_y <= b_y + tol.
        """

        before = tuple(self._entries)
        merged: List[ArchiveEntry] = []
        seen_tours = set()
        seen_objectives = set()
        for entry in list(self._entries) + list(candidates):
            if entry.tour in seen_tours:
                continue
            if self.tol == 0.0:
                if entry.objectives in seen_objectives:
                    continue
            elif any(
                all(
                    abs(left - right) <= self.tol
                    for left, right in zip(
                        existing.objectives,
                        entry.objectives,
                    )
                )
                for existing in merged
            ):
                continue
            seen_tours.add(entry.tour)
            seen_objectives.add(entry.objectives)
            merged.append(entry)

        merged.sort(key=lambda item: (item.objectives[0], item.objectives[1], item.tour))
        xs = [entry.objectives[0] for entry in merged]
        prefix_min_y: List[float] = []
        current = float("inf")
        for entry in merged:
            current = min(current, entry.objectives[1])
            prefix_min_y.append(current)

        nondominated: List[ArchiveEntry] = []
        for entry in merged:
            x, y = entry.objectives
            weak_x_end = bisect.bisect_right(xs, x + self.tol) - 1
            strict_x_end = bisect.bisect_left(xs, x - self.tol) - 1
            dominated_by_strict_y = (
                weak_x_end >= 0
                and prefix_min_y[weak_x_end] < y - self.tol
            )
            dominated_by_strict_x = (
                strict_x_end >= 0
                and prefix_min_y[strict_x_end] <= y + self.tol
            )
            if not dominated_by_strict_y and not dominated_by_strict_x:
                nondominated.append(entry)

        self._entries = nondominated
        if self.max_size is not None and len(self._entries) > self.max_size:
            self._truncate_by_crowding()
        self._entries.sort(key=lambda item: item.objectives)
        self._rebuild_indices()
        return tuple(self._entries) != before

    def ideal_nadir(self, fallback: Iterable[ObjectiveVector]) -> Tuple[ObjectiveVector, ObjectiveVector]:
        vectors = list(self.objectives()) + list(fallback)
        if not vectors:
            raise ValueError("Cannot compute ideal/nadir without objective vectors.")
        dim = len(vectors[0])
        ideal = tuple(min(v[i] for v in vectors) for i in range(dim))
        nadir = tuple(max(v[i] for v in vectors) for i in range(dim))
        return ideal, nadir

    def hypervolume_2d(self, reference: Optional[ObjectiveVector] = None) -> float:
        """Exact 2D hypervolume for minimization nondominated points."""
        if not self._entries:
            return 0.0
        if len(self._entries[0].objectives) != 2:
            raise ValueError("hypervolume_2d requires two objectives.")

        points = sorted((entry.objectives for entry in self._entries), key=lambda z: (z[0], z[1]))
        nondominated: List[ObjectiveVector] = []
        best_y = float("inf")
        for x, y in points:
            if y < best_y:
                nondominated.append((x, y))
                best_y = y

        if reference is None:
            max_x = max(x for x, _ in nondominated)
            max_y = max(y for _, y in nondominated)
            reference = (max_x * 1.1 + 1e-9, max_y * 1.1 + 1e-9)

        ref_x, ref_y = reference
        hv = 0.0
        prev_y = ref_y
        for x, y in nondominated:
            width = max(0.0, ref_x - x)
            height = max(0.0, prev_y - y)
            hv += width * height
            prev_y = min(prev_y, y)
        return hv

    def fixed_reference_2d(self, margin: float = 0.1) -> ObjectiveVector:
        """Build a fixed minimization reference point from the current archive."""
        if not self._entries:
            raise ValueError("Cannot build a reference point from an empty archive.")
        if len(self._entries[0].objectives) != 2:
            raise ValueError("fixed_reference_2d requires two objectives.")
        if margin < 0.0:
            raise ValueError("margin must be nonnegative.")
        xs = [entry.objectives[0] for entry in self._entries]
        ys = [entry.objectives[1] for entry in self._entries]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        return (
            max(xs) + margin * max(1.0, span_x, abs(max(xs))),
            max(ys) + margin * max(1.0, span_y, abs(max(ys))),
        )

    def _is_duplicate(self, candidate: ArchiveEntry) -> bool:
        if candidate.tour in self._tour_index:
            return True
        if self.tol == 0.0:
            return candidate.objectives in self._objective_index
        return any(
            all(
                abs(a - b) <= self.tol
                for a, b in zip(entry.objectives, candidate.objectives)
            )
            for entry in self._entries
        )

    def _truncate_by_crowding(self) -> None:
        """NSGA-II style crowding truncation in any objective dimension."""
        if self.max_size is None or len(self._entries) <= self.max_size:
            return
        entries = list(self._entries)
        dimension = len(entries[0].objectives)
        while len(entries) > self.max_size:
            distances = [0.0 for _ in entries]
            for objective_index in range(dimension):
                order = sorted(range(len(entries)), key=lambda idx: entries[idx].objectives[objective_index])
                distances[order[0]] = float("inf")
                distances[order[-1]] = float("inf")
                low = entries[order[0]].objectives[objective_index]
                high = entries[order[-1]].objectives[objective_index]
                scale = high - low
                if scale <= self.tol:
                    continue
                for position in range(1, len(order) - 1):
                    left = entries[order[position - 1]].objectives[objective_index]
                    right = entries[order[position + 1]].objectives[objective_index]
                    distances[order[position]] += (right - left) / scale
            remove_index = min(
                range(len(entries)),
                key=lambda idx: (distances[idx], entries[idx].objectives, entries[idx].tour),
            )
            entries.pop(remove_index)
        self._entries = entries
        self._rebuild_indices()

