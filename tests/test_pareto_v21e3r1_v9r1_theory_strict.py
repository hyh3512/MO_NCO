from __future__ import annotations

import pytest

from mo_nco.pareto_v21e3r1_v9_theory import (
    DualResourceBudget,
    archive_compensated_replacement,
    composite_potential,
    operator_productivity,
    select_first_unseen,
)


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_v9r1_productivity_rejects_non_exact_integer_counts(bad: object) -> None:
    with pytest.raises(TypeError):
        operator_productivity(
            attempts=bad,  # type: ignore[arg-type]
            new_states=0,
            total_quality_gain=0.0,
        )


@pytest.mark.parametrize("bad", [True, "0.25"])
def test_v9r1_productivity_rejects_coercible_numeric_values(bad: object) -> None:
    with pytest.raises(TypeError):
        operator_productivity(
            attempts=1,
            new_states=1,
            total_quality_gain=bad,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        operator_productivity(
            attempts=1,
            new_states=1,
            total_quality_gain=0.25,
            elapsed_seconds=bad,  # type: ignore[arg-type]
        )


def test_v9r1_productivity_enforces_normalized_gain_with_boundary_tolerance() -> None:
    value = operator_productivity(
        attempts=2,
        new_states=2,
        total_quality_gain=1.0 + 5e-13,
        tolerance=1e-12,
    )
    assert value.total_quality_gain == 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        operator_productivity(
            attempts=2,
            new_states=2,
            total_quality_gain=1.0 + 2e-12,
            tolerance=1e-12,
        )


def test_v9r1_productivity_rejects_gain_without_a_new_state() -> None:
    with pytest.raises(ValueError, match="no new state"):
        operator_productivity(
            attempts=3,
            new_states=0,
            total_quality_gain=0.1,
        )
    value = operator_productivity(
        attempts=3,
        new_states=0,
        total_quality_gain=5e-13,
        tolerance=1e-12,
    )
    assert value.total_quality_gain == 0.0
    assert value.factorization_residual == 0.0


def test_v9r1_budget_rejects_permissive_elapsed_coercion() -> None:
    with pytest.raises(TypeError):
        DualResourceBudget(1, 2, 3, True)  # type: ignore[arg-type]
    budget = DualResourceBudget(1, 2, 3)
    with pytest.raises(TypeError):
        budget.permits(
            first_evaluations=1,
            attempts=2,
            screenings=3,
            elapsed_seconds="0.0",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        budget.permits(
            first_evaluations=1.0,  # type: ignore[arg-type]
            attempts=2,
            screenings=3,
        )


def test_v9r1_screen_requires_exact_bool_and_accepts_none_as_a_candidate() -> None:
    with pytest.raises(TypeError, match="exact bool"):
        select_first_unseen(["candidate"], is_seen=lambda _candidate: 0, cap=1)

    decision = select_first_unseen([None], is_seen=lambda _candidate: False, cap=1)
    assert decision.selected is None
    assert decision.selected_rank == 0
    assert not decision.exhausted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normalized_hv_gain": True, "tradeoff_lambda": 1.0},
        {"normalized_hv_gain": "0.1", "tradeoff_lambda": 1.0},
        {"normalized_hv_gain": 0.1, "tradeoff_lambda": True},
        {"normalized_hv_gain": 0.1, "tradeoff_lambda": "1.0"},
    ],
)
def test_v9r1_archive_replacement_rejects_coercible_numeric_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        archive_compensated_replacement(
            {0: 0.0},
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_delta", [True, "0.1"])
def test_v9r1_archive_replacement_rejects_coercible_deltas(
    bad_delta: object,
) -> None:
    with pytest.raises(TypeError):
        archive_compensated_replacement(
            {0: bad_delta},  # type: ignore[dict-item]
            normalized_hv_gain=0.1,
            tradeoff_lambda=1.0,
        )


def test_v9r1_archive_replacement_bounds_gain_and_records_numerical_slack() -> None:
    low = archive_compensated_replacement(
        {0: 0.0},
        normalized_hv_gain=-5e-13,
        tradeoff_lambda=1.0,
        tolerance=1e-12,
    )
    assert low.normalized_hv_gain == 0.0

    high = archive_compensated_replacement(
        {0: 0.1 + 5e-13},
        normalized_hv_gain=0.1,
        tradeoff_lambda=1.0,
        tolerance=1e-12,
    )
    assert high.selected_targets == (0,)
    assert 0.0 < high.composite_potential_change <= 1e-12
    assert high.certified_nonincrease

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        archive_compensated_replacement(
            {0: 0.0},
            normalized_hv_gain=1.0 + 2e-12,
            tradeoff_lambda=1.0,
            tolerance=1e-12,
        )


def test_v9r1_tolerance_is_small_finite_and_strictly_typed() -> None:
    with pytest.raises(TypeError):
        archive_compensated_replacement(
            {0: 0.0},
            normalized_hv_gain=0.0,
            tradeoff_lambda=0.0,
            tolerance="1e-12",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[0, 1e-6\]"):
        archive_compensated_replacement(
            {0: 0.0},
            normalized_hv_gain=0.0,
            tradeoff_lambda=0.0,
            tolerance=1e-5,
        )


@pytest.mark.parametrize(
    "values,hv,lam",
    [
        (["0.1"], 0.2, 1.0),
        ([True], 0.2, 1.0),
        ([0.1], "0.2", 1.0),
        ([0.1], 0.2, True),
    ],
)
def test_v9r1_composite_potential_rejects_coercible_numbers(
    values: list[object],
    hv: object,
    lam: object,
) -> None:
    with pytest.raises(TypeError):
        composite_potential(
            values,  # type: ignore[arg-type]
            normalized_hypervolume=hv,  # type: ignore[arg-type]
            tradeoff_lambda=lam,  # type: ignore[arg-type]
        )


def test_v9r1_composite_potential_enforces_normalized_hv() -> None:
    assert composite_potential(
        [0.2, 0.3],
        normalized_hypervolume=1.0 + 5e-13,
        tradeoff_lambda=0.5,
        tolerance=1e-12,
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        composite_potential(
            [0.2],
            normalized_hypervolume=-2e-12,
            tradeoff_lambda=0.5,
            tolerance=1e-12,
        )
