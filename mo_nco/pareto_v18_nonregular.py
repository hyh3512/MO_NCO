"""Nonregular shared-categorical selection for Pareto-SMC v18.

Exact best-type identification is not uniformly finite at tie models.  This
module supplies a separate epsilon-optimal PAC branch that remains valid with
ties, zero categorical coordinates, and nonunique characteristic allocations.
It uses the exact-rational time-uniform Hoeffding radii already used by the v17
packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .pareto_v17_track_and_stop import (
    TrackAndStopError,
    time_uniform_hoeffding_radius,
)


class NonregularSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class EpsilonPACCellDecision:
    cell_index: int
    selected_type: int | None
    epsilon: Fraction
    lower_bounds: tuple[Fraction, ...]
    upper_bounds: tuple[Fraction, ...]
    certified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_index": self.cell_index,
            "selected_type": self.selected_type,
            "epsilon": str(self.epsilon),
            "lower_bounds": [str(value) for value in self.lower_bounds],
            "upper_bounds": [str(value) for value in self.upper_bounds],
            "certified": self.certified,
        }


@dataclass(frozen=True)
class EpsilonPACSelectionCertificate:
    decisions: tuple[EpsilonPACCellDecision, ...]
    alpha: Fraction
    all_cells_certified: bool
    exact_best_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_semantics": "simultaneous_epsilon_optimal_type_v18",
            "alpha": str(self.alpha),
            "all_cells_certified": self.all_cells_certified,
            "exact_best_claimed": False,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def epsilon_pac_selection(
    counts: Sequence[Sequence[int]],
    epsilon_by_cell: Sequence[Fraction | int | str],
    alpha: Fraction | int | str,
    *,
    denominator: int = 10**12,
) -> EpsilonPACSelectionCertificate:
    """Certify one epsilon-optimal type for every cell.

    On the simultaneous confidence event, a selected type ``r`` satisfies

        p[r,j] >= max_s p[s,j] - epsilon[j].

    The procedure does not require a unique best type or positive support.
    """

    rows = tuple(tuple(int(value) for value in row) for row in counts)
    if not rows or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise NonregularSelectionError("invalid categorical count matrix")
    if any(value < 0 for row in rows for value in row) or any(sum(row) <= 0 for row in rows):
        raise NonregularSelectionError("every type needs nonnegative counts and a positive sample size")
    r_count = len(rows)
    j_count = len(rows[0]) - 1
    epsilons = tuple(Fraction(value) for value in epsilon_by_cell)
    if len(epsilons) != j_count or any(value <= 0 or value > 1 for value in epsilons):
        raise NonregularSelectionError("one epsilon in (0,1] is required per cell")
    a = Fraction(alpha)
    if not (Fraction(0, 1) < a < Fraction(1, 1)):
        raise NonregularSelectionError("alpha must lie in (0,1)")

    radii = tuple(
        time_uniform_hoeffding_radius(
            sum(row),
            r_count,
            j_count,
            a,
            denominator=denominator,
        )
        for row in rows
    )
    decisions: list[EpsilonPACCellDecision] = []
    for j0 in range(j_count):
        j = j0 + 1
        lowers: list[Fraction] = []
        uppers: list[Fraction] = []
        for r, row in enumerate(rows):
            empirical = Fraction(row[j], sum(row))
            lowers.append(max(Fraction(0, 1), empirical - radii[r]))
            uppers.append(min(Fraction(1, 1), empirical + radii[r]))
        max_upper = max(uppers)
        eligible = [
            r
            for r in range(r_count)
            if lowers[r] >= max_upper - epsilons[j0]
        ]
        selected = min(eligible) if eligible else None
        decisions.append(
            EpsilonPACCellDecision(
                cell_index=j0,
                selected_type=selected,
                epsilon=epsilons[j0],
                lower_bounds=tuple(lowers),
                upper_bounds=tuple(uppers),
                certified=selected is not None,
            )
        )
    return EpsilonPACSelectionCertificate(
        decisions=tuple(decisions),
        alpha=a,
        all_cells_certified=all(decision.certified for decision in decisions),
    )


def exact_best_boundary_status(
    probability_matrix: Sequence[Sequence[Fraction | int | str]],
) -> dict[str, object]:
    """Detect exact ties and state the corresponding information obstruction."""

    matrix = tuple(tuple(Fraction(value) for value in row) for row in probability_matrix)
    if not matrix or len(matrix[0]) < 2 or any(len(row) != len(matrix[0]) for row in matrix):
        raise NonregularSelectionError("invalid probability matrix")
    if any(value < 0 for row in matrix for value in row):
        raise NonregularSelectionError("probabilities must be nonnegative")
    if any(sum(row, Fraction(0, 1)) != 1 for row in matrix):
        raise NonregularSelectionError("rows must be categorical distributions")
    tied_cells: list[int] = []
    for j in range(1, len(matrix[0])):
        values = [row[j] for row in matrix]
        if sum(value == max(values) for value in values) > 1:
            tied_cells.append(j - 1)
    return {
        "exact_best_regular": not tied_cells,
        "tied_cells": tied_cells,
        "characteristic_information_at_tie": "0" if tied_cells else "not_forced_to_zero",
        "finite_expected_uniform_exact_identification_available": not tied_cells,
        "recommended_branch": (
            "epsilon_pac_selection"
            if tied_cells
            else "exact_track_and_stop_or_epsilon_pac_selection"
        ),
    }


__all__ = [
    "EpsilonPACCellDecision",
    "EpsilonPACSelectionCertificate",
    "NonregularSelectionError",
    "epsilon_pac_selection",
    "exact_best_boundary_status",
]
