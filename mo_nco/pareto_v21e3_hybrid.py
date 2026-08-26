from __future__ import annotations

"""Prospective V21e3 matched-control hybrid search.

This module defines a new stochastic algorithm object.  It deliberately does
not modify, reinterpret, or inherit calibration evidence from V21e2.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Literal, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .moves import two_opt_at
from .pareto_v21e3r1_construction import (
    family_aware_initial_solution,
    mokp_directional_densities,
    mokp_repair,
)
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
    problem_sha256,
)
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector
from .pareto_v21e3r1_v9_theory import (
    DualResourceBudget,
    archive_compensated_replacement,
    select_first_unseen,
)


V21E3CandidateId = Literal["C0", "C1", "C2", "C3"]

_LEGACY_POST_INITIALIZATION_SEARCH_POLICY = "proposal_chain_v21e3r1_v1"
_SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY = (
    "post_commit_type_incumbent_anchor_development_v1"
)
_LEGACY_MOKP_NOVELTY_GENERATION_POLICY = (
    "legacy_retry_and_local_v21e3r1_v1"
)
_SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY = (
    "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
)
_V9_CACHE_AWARE_SCREENING_POLICY = (
    "bounded_cache_aware_structural_screen_development_v1"
)
_V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY = (
    "archive_compensated_information_lyapunov_development_v1"
)
_V9_DIAGNOSTIC_PREFIX = "V21E3R1_V9_"
# diagnostic_id -> (family, screening_policy, replacement_policy, lambda_required)
_V9_DIAGNOSTIC_POLICIES: dict[str, tuple[str, str, str, bool]] = {
    _V9_DIAGNOSTIC_PREFIX + "LEGACY_MOKP": (
        "MOKP",
        "disabled_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        False,
    ),
    _V9_DIAGNOSTIC_PREFIX + "INFORMATION_SCREEN_MOKP": (
        "MOKP",
        _V9_CACHE_AWARE_SCREENING_POLICY,
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        False,
    ),
    _V9_DIAGNOSTIC_PREFIX + "LYAPUNOV_MOKP": (
        "MOKP",
        "disabled_v1",
        _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
        True,
    ),
    _V9_DIAGNOSTIC_PREFIX + "INFORMATION_LYAPUNOV_MOKP": (
        "MOKP",
        _V9_CACHE_AWARE_SCREENING_POLICY,
        _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
        True,
    ),
    _V9_DIAGNOSTIC_PREFIX + "LEGACY_MOTSP": (
        "MOTSP",
        "disabled_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        False,
    ),
    _V9_DIAGNOSTIC_PREFIX + "INFORMATION_SCREEN_MOTSP": (
        "MOTSP",
        _V9_CACHE_AWARE_SCREENING_POLICY,
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        False,
    ),
    _V9_DIAGNOSTIC_PREFIX + "LYAPUNOV_MOTSP": (
        "MOTSP",
        "disabled_v1",
        _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
        True,
    ),
    _V9_DIAGNOSTIC_PREFIX + "INFORMATION_LYAPUNOV_MOTSP": (
        "MOTSP",
        _V9_CACHE_AWARE_SCREENING_POLICY,
        _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
        True,
    ),
}


class V21E3ResourceLimitExceeded(RuntimeError):
    """Raised after a V9 A/S/T cap writes a durable FAILURE receipt."""
_SUCCESSOR_FACTORIAL_PREFIX = "V21E3R1_SUCCESSOR_FACTORIAL_"
_SUCCESSOR_FACTORIAL_POLICIES: dict[str, tuple[str, str, str]] = {
    _SUCCESSOR_FACTORIAL_PREFIX + "MOKP_LEGACY": (
        "MOKP",
        _LEGACY_POST_INITIALIZATION_SEARCH_POLICY,
        _LEGACY_MOKP_NOVELTY_GENERATION_POLICY,
    ),
    _SUCCESSOR_FACTORIAL_PREFIX + "MOKP_ANCHOR_ONLY": (
        "MOKP",
        _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY,
        _LEGACY_MOKP_NOVELTY_GENERATION_POLICY,
    ),
    _SUCCESSOR_FACTORIAL_PREFIX + "MOKP_NOVELTY_ONLY": (
        "MOKP",
        _LEGACY_POST_INITIALIZATION_SEARCH_POLICY,
        _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY,
    ),
    _SUCCESSOR_FACTORIAL_PREFIX + "MOKP_BOTH": (
        "MOKP",
        _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY,
        _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY,
    ),
    _SUCCESSOR_FACTORIAL_PREFIX + "MOTSP_LEGACY": (
        "MOTSP",
        _LEGACY_POST_INITIALIZATION_SEARCH_POLICY,
        _LEGACY_MOKP_NOVELTY_GENERATION_POLICY,
    ),
    _SUCCESSOR_FACTORIAL_PREFIX + "MOTSP_ANCHOR": (
        "MOTSP",
        _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY,
        _LEGACY_MOKP_NOVELTY_GENERATION_POLICY,
    ),
}


@dataclass(frozen=True)
class V21E3CandidateSpec:
    candidate_id: V21E3CandidateId
    enabled_components: Tuple[str, ...]
    direction_policy: str
    construction_portfolio: Tuple[str, ...]
    native_portfolio: Tuple[str, ...]
    local_improvement_contract: str
    replacement_contract: str
    diversification_schedule: str
    exchange_schedule: str
    exchange_operator: str | None
    exchange_effort_units: int


@dataclass(frozen=True)
class V21E3ScheduleSlot:
    slot: int
    kind: Literal["native", "diversification", "exchange"]
    effort_units: int
    exchange_operator: str | None


def v21e3_schedule_slot(
    candidate_id: V21E3CandidateId,
    slot: int,
    *,
    diversification_period: int,
    exchange_period: int,
) -> V21E3ScheduleSlot:
    """Return the pre-frozen mutually-exclusive mechanism assignment."""

    v21e3_candidate_spec(candidate_id)
    if slot <= 0 or diversification_period <= 0 or exchange_period <= 0:
        raise ValueError("slot and schedule periods must be positive.")
    if candidate_id in {"C2", "C3"} and slot % exchange_period == 0:
        return V21E3ScheduleSlot(
            slot=slot,
            kind="exchange",
            effort_units=1,
            exchange_operator=(
                "matched_exchange_control_v1"
                if candidate_id == "C2"
                else "neighbor_path_relinking_v1"
            ),
        )
    if candidate_id in {"C2", "C3"} and slot % diversification_period == 0:
        return V21E3ScheduleSlot(
            slot=slot,
            kind="diversification",
            effort_units=1,
            exchange_operator=None,
        )
    return V21E3ScheduleSlot(
        slot=slot,
        kind="native",
        effort_units=1,
        exchange_operator=None,
    )


def v21e3_candidate_spec(candidate_id: V21E3CandidateId) -> V21E3CandidateSpec:
    """Return the prospective algorithm-object contract for one candidate."""

    if candidate_id not in {"C0", "C1", "C2", "C3"}:
        raise ValueError("candidate_id must be one of C0, C1, C2, C3.")
    components = ["strong_native_backbone"]
    if candidate_id != "C0":
        components.append("direction_conditioning")
    if candidate_id in {"C2", "C3"}:
        components.append("typed_diversification")
        components.append(
            "matched_exchange_control"
            if candidate_id == "C2"
            else "neighbor_path_relinking"
        )
    return V21E3CandidateSpec(
        candidate_id=candidate_id,
        enabled_components=tuple(components),
        direction_policy=(
            "central_untyped_direction_v1"
            if candidate_id == "C0"
            else "reference_type_direction_v1"
        ),
        construction_portfolio=(
            "mokp_profit_density_construction_repair_v1",
            "motsp_weighted_nearest_neighbor_construction_v1",
        ),
        native_portfolio=(
            "mokp_crossover_add_drop_swap_multibit_v1",
            "motsp_candidate_two_opt_relocate_restricted_three_opt_v1",
        ),
        local_improvement_contract=(
            "bounded_true_evaluation_scalar_improvement_v1"
        ),
        replacement_contract=(
            "bounded_reference_neighborhood_nonworse_replacement_v1"
        ),
        diversification_schedule=(
            "pre_frozen_nonexchange_diversification_slots_v1"
            if candidate_id in {"C2", "C3"}
            else "disabled"
        ),
        exchange_schedule=(
            "pre_frozen_matched_exchange_slots_v1"
            if candidate_id in {"C2", "C3"}
            else "disabled"
        ),
        exchange_operator=(
            "matched_exchange_control_v1"
            if candidate_id == "C2"
            else (
                "neighbor_path_relinking_v1"
                if candidate_id == "C3"
                else None
            )
        ),
        exchange_effort_units=(1 if candidate_id in {"C2", "C3"} else 0),
    )


@dataclass(frozen=True)
class V21E3RegionObservation:
    region: Tuple[int, ...]
    new_evaluated_cell: bool
    new_nondominated_cell: bool


class V21E3RegionOccupancy:
    """Track evaluated and current nondominated objective cells separately."""

    def __init__(
        self,
        *,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        bins: int = 20,
    ) -> None:
        self._lower = tuple(float(value) for value in lower_bounds)
        self._upper = tuple(float(value) for value in upper_bounds)
        self._bins = int(bins)
        if not self._lower or len(self._lower) != len(self._upper):
            raise ValueError("Objective bounds must have one common dimension.")
        if self._bins <= 0:
            raise ValueError("bins must be positive.")
        if any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in zip(self._lower, self._upper)
        ):
            raise ValueError("Objective bounds must be finite and nondegenerate.")
        self._evaluated: set[Tuple[int, ...]] = set()
        self._nondominated: set[Tuple[int, ...]] = set()

    @property
    def evaluated_region_count(self) -> int:
        return len(self._evaluated)

    @property
    def nondominated_region_count(self) -> int:
        return len(self._nondominated)

    def region(self, objective: Sequence[float]) -> Tuple[int, ...]:
        values = tuple(float(value) for value in objective)
        if len(values) != len(self._lower):
            raise ValueError("Objective vector has the wrong dimension.")
        output = []
        for value, lower, upper in zip(values, self._lower, self._upper):
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError("Objective vector lies outside the frozen box.")
            if value == upper:
                output.append(self._bins - 1)
            else:
                output.append(
                    int(self._bins * (value - lower) / (upper - lower))
                )
        return tuple(output)

    def observe(
        self,
        objective: Sequence[float],
        *,
        nondominated: Sequence[Sequence[float]],
    ) -> V21E3RegionObservation:
        region = self.region(objective)
        new_evaluated = region not in self._evaluated
        before = self._nondominated
        after = {self.region(item) for item in nondominated}
        new_nondominated = region in after and region not in before
        self._evaluated.add(region)
        self._nondominated = after
        return V21E3RegionObservation(
            region=region,
            new_evaluated_cell=new_evaluated,
            new_nondominated_cell=new_nondominated,
        )


V21E3EvidencePhase = Literal[
    "development",
    "calibration",
    "calibration_confirmation",
]


@dataclass(frozen=True)
class V21E3HybridConfig:
    candidate_id: V21E3CandidateId
    reference_directions: Tuple[Tuple[float, ...], ...]
    charged_evaluations: int
    checkpoint_period: int
    seed: int
    phase: V21E3EvidencePhase
    trace_database: str | None = None
    terminal_receipt: str | None = None
    receipt_database_path: str | None = None
    capture_trace: bool = True
    case_artifact_sha256: str | None = None
    source_snapshot_sha256: str | None = None
    diversification_period: int = 16
    exchange_period: int = 11
    local_improvement_steps: int = 2
    duplicate_retry_cap: int = 4
    fallback_attempt_cap: int = 16
    neighborhood_size: int = 4
    initialization_policy: str = "family_aware_matched_construction_v21e3r1_v1"
    replacement_policy: str = (
        "bounded_reference_neighborhood_nonworse_replacement_v1"
    )
    development_diagnostic_id: str | None = None
    post_initialization_search_policy: str = (
        _LEGACY_POST_INITIALIZATION_SEARCH_POLICY
    )
    mokp_novelty_generation_policy: str = (
        _LEGACY_MOKP_NOVELTY_GENERATION_POLICY
    )
    candidate_screening_policy: str = "disabled_v1"
    candidate_screening_cap: int = 1
    archive_tradeoff_lambda: float = 0.0
    attempt_cap: int | None = None
    structural_screening_cap: int | None = None
    wall_time_cap_seconds: float | None = None

    def __post_init__(self) -> None:
        v21e3_candidate_spec(self.candidate_id)
        if not self.reference_directions:
            raise ValueError("At least one reference direction is required.")
        dimension = len(self.reference_directions[0])
        if dimension <= 0:
            raise ValueError("Reference directions must be nonempty.")
        for direction in self.reference_directions:
            if len(direction) != dimension:
                raise ValueError("Reference directions must share a dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in direction):
                raise ValueError("Reference weights must be finite and positive.")
            if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Reference directions must sum to one.")
        if self.charged_evaluations < len(self.reference_directions):
            raise ValueError("The charged budget must initialize every type.")
        if self.checkpoint_period <= 0:
            raise ValueError("checkpoint_period must be positive.")
        if self.phase not in {
            "development",
            "calibration",
            "calibration_confirmation",
        }:
            raise ValueError("Unsupported evidence phase.")
        if self.diversification_period <= 0 or self.exchange_period <= 0:
            raise ValueError("Schedule periods must be positive.")
        diagnostic_mode = self.development_diagnostic_id is not None
        if self.local_improvement_steps < 0:
            raise ValueError("local_improvement_steps must be nonnegative.")
        if self.local_improvement_steps == 0 and not (
            self.phase == "development"
            and diagnostic_mode
            and self.candidate_id == "C0"
        ):
            raise ValueError(
                "Zero local-improvement depth is authorized only for a named "
                "development-only C0 diagnostic."
            )
        allowed_initialization = {
            "family_aware_matched_construction_v21e3r1_v1",
            "problem_native_exact_random_solution_development_diagnostic_v1",
        }
        if self.initialization_policy not in allowed_initialization:
            raise ValueError("Unsupported V21e3r1 initialization policy.")
        if self.initialization_policy != "family_aware_matched_construction_v21e3r1_v1":
            if not (
                self.phase == "development"
                and diagnostic_mode
                and self.candidate_id == "C0"
            ):
                raise ValueError(
                    "Alternative initialization is authorized only for a named "
                    "development-only C0 diagnostic."
                )
        allowed_replacement = {
            "bounded_reference_neighborhood_nonworse_replacement_v1",
            "self_type_nonworse_replacement_development_diagnostic_v1",
            _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
        }
        if self.replacement_policy not in allowed_replacement:
            raise ValueError("Unsupported V21e3r1 replacement policy.")
        if self.replacement_policy != (
            "bounded_reference_neighborhood_nonworse_replacement_v1"
        ):
            if not (
                self.phase == "development"
                and diagnostic_mode
                and self.candidate_id == "C0"
            ):
                raise ValueError(
                    "Alternative replacement is authorized only for a named "
                    "development-only C0 diagnostic."
                )
        allowed_screening_policies = {"disabled_v1", _V9_CACHE_AWARE_SCREENING_POLICY}
        if (
            type(self.candidate_screening_policy) is not str
            or self.candidate_screening_policy not in allowed_screening_policies
        ):
            raise ValueError("Unsupported candidate-screening policy.")
        if (
            isinstance(self.candidate_screening_cap, bool)
            or not isinstance(self.candidate_screening_cap, int)
            or self.candidate_screening_cap <= 0
        ):
            raise TypeError("candidate_screening_cap must be a positive exact integer.")
        if (
            type(self.archive_tradeoff_lambda) not in {int, float}
            or not math.isfinite(float(self.archive_tradeoff_lambda))
            or float(self.archive_tradeoff_lambda) < 0.0
        ):
            raise ValueError("archive_tradeoff_lambda must be finite and nonnegative.")
        if diagnostic_mode and self.phase != "development":
            raise ValueError("Development diagnostics cannot enter later evidence phases.")
        allowed_search_policies = {
            _LEGACY_POST_INITIALIZATION_SEARCH_POLICY,
            _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY,
        }
        if (
            type(self.post_initialization_search_policy) is not str
            or self.post_initialization_search_policy not in allowed_search_policies
        ):
            raise ValueError("Unsupported post-initialization search policy.")
        allowed_novelty_policies = {
            _LEGACY_MOKP_NOVELTY_GENERATION_POLICY,
            _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY,
        }
        if (
            type(self.mokp_novelty_generation_policy) is not str
            or self.mokp_novelty_generation_policy not in allowed_novelty_policies
        ):
            raise ValueError("Unsupported MOKP novelty-generation policy.")
        diagnostic_id = self.development_diagnostic_id
        successor_contract = _SUCCESSOR_FACTORIAL_POLICIES.get(diagnostic_id or "")
        if diagnostic_id is not None and diagnostic_id.startswith(
            _SUCCESSOR_FACTORIAL_PREFIX
        ) and successor_contract is None:
            raise ValueError("Unknown successor factorial diagnostic identity.")
        if successor_contract is None:
            if (
                self.post_initialization_search_policy
                != _LEGACY_POST_INITIALIZATION_SEARCH_POLICY
                or self.mokp_novelty_generation_policy
                != _LEGACY_MOKP_NOVELTY_GENERATION_POLICY
            ):
                raise ValueError(
                    "Successor policies are authorized only by an exact named "
                    "successor factorial diagnostic."
                )
        else:
            _expected_family, expected_search, expected_novelty = successor_contract
            if not (
                self.phase == "development"
                and self.candidate_id == "C0"
                and self.post_initialization_search_policy == expected_search
                and self.mokp_novelty_generation_policy == expected_novelty
            ):
                raise ValueError(
                    "The successor factorial diagnostic identity and policies drifted."
                )
        v9_contract = _V9_DIAGNOSTIC_POLICIES.get(diagnostic_id or "")
        if diagnostic_id is not None and diagnostic_id.startswith(
            _V9_DIAGNOSTIC_PREFIX
        ) and v9_contract is None:
            raise ValueError("Unknown V9 information-time diagnostic identity.")
        if v9_contract is None:
            if (
                self.candidate_screening_policy != "disabled_v1"
                or self.replacement_policy == _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY
                or float(self.archive_tradeoff_lambda) != 0.0
                or self.attempt_cap is not None
                or self.structural_screening_cap is not None
                or self.wall_time_cap_seconds is not None
            ):
                raise ValueError(
                    "Information-time screening and archive-Lyapunov replacement "
                    "and their A/S/T caps are authorized only by an exact V9 "
                    "development diagnostic."
                )
        else:
            (
                _v9_family,
                expected_screening,
                expected_replacement,
                lambda_required,
            ) = v9_contract
            lambda_ok = (
                float(self.archive_tradeoff_lambda) > 0.0
                if lambda_required
                else float(self.archive_tradeoff_lambda) == 0.0
            )
            if not (
                self.phase == "development"
                and self.candidate_id == "C0"
                and self.candidate_screening_policy == expected_screening
                and self.replacement_policy == expected_replacement
                and lambda_ok
            ):
                raise ValueError(
                    "The V9 diagnostic identity and policies drifted."
                )
            if (
                isinstance(self.attempt_cap, bool)
                or not isinstance(self.attempt_cap, int)
                or self.attempt_cap < self.charged_evaluations
            ):
                raise TypeError(
                    "A V9 attempt_cap must be an exact integer at least as large "
                    "as the charged-evaluation budget."
                )
            if (
                isinstance(self.structural_screening_cap, bool)
                or not isinstance(self.structural_screening_cap, int)
                or self.structural_screening_cap < 0
            ):
                raise TypeError(
                    "A V9 structural_screening_cap must be a nonnegative exact integer."
                )
            if self.candidate_screening_policy == _V9_CACHE_AWARE_SCREENING_POLICY:
                if self.structural_screening_cap < self.candidate_screening_cap:
                    raise ValueError(
                        "The run-level structural-screening cap must cover at least "
                        "one complete candidate-screen service."
                    )
            elif self.structural_screening_cap != 0:
                raise ValueError(
                    "A V9 arm with disabled screening must bind a zero structural cap."
                )
            if self.wall_time_cap_seconds is not None:
                if (
                    type(self.wall_time_cap_seconds) not in {int, float}
                    or not math.isfinite(float(self.wall_time_cap_seconds))
                    or float(self.wall_time_cap_seconds) <= 0.0
                ):
                    raise ValueError(
                        "wall_time_cap_seconds must be a finite positive exact real "
                        "when supplied."
                    )
        if self.duplicate_retry_cap < 0 or self.fallback_attempt_cap <= 0:
            raise ValueError("Duplicate retry caps must be finite and valid.")
        if self.neighborhood_size < 2:
            raise ValueError("neighborhood_size must be at least two.")
        if self.case_artifact_sha256 is not None and (
            len(self.case_artifact_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.case_artifact_sha256
            )
        ):
            raise ValueError("case_artifact_sha256 must be lowercase SHA-256.")
        if self.source_snapshot_sha256 is not None and (
            len(self.source_snapshot_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.source_snapshot_sha256
            )
        ):
            raise ValueError("source_snapshot_sha256 must be lowercase SHA-256.")
        if self.phase in {"calibration", "calibration_confirmation"} and (
            self.case_artifact_sha256 is None
            or self.source_snapshot_sha256 is None
        ):
            raise ValueError(
                "Calibration requires explicit case-artifact and source-snapshot SHA-256."
            )

    def semantic_payload(self) -> dict[str, object]:
        """Return the execution semantics, excluding output and custody paths."""

        excluded = {
            "trace_database",
            "terminal_receipt",
            "receipt_database_path",
            "capture_trace",
            "case_artifact_sha256",
            "source_snapshot_sha256",
            "post_initialization_search_policy",
            "mokp_novelty_generation_policy",
            "candidate_screening_policy",
            "candidate_screening_cap",
            "archive_tradeoff_lambda",
            "attempt_cap",
            "structural_screening_cap",
            "wall_time_cap_seconds",
        }
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in excluded
        }
        if self.development_diagnostic_id in _SUCCESSOR_FACTORIAL_POLICIES:
            payload["post_initialization_search_policy"] = (
                self.post_initialization_search_policy
            )
            payload["mokp_novelty_generation_policy"] = (
                self.mokp_novelty_generation_policy
            )
        if self.development_diagnostic_id in _V9_DIAGNOSTIC_POLICIES:
            payload["post_initialization_search_policy"] = (
                self.post_initialization_search_policy
            )
            payload["mokp_novelty_generation_policy"] = (
                self.mokp_novelty_generation_policy
            )
            payload["candidate_screening_policy"] = self.candidate_screening_policy
            payload["candidate_screening_cap"] = self.candidate_screening_cap
            payload["archive_tradeoff_lambda"] = float(self.archive_tradeoff_lambda)
            payload["attempt_cap"] = self.attempt_cap
            payload["structural_screening_cap"] = self.structural_screening_cap
            payload["wall_time_cap_seconds"] = (
                None
                if self.wall_time_cap_seconds is None
                else float(self.wall_time_cap_seconds)
            )
        return payload


@dataclass(frozen=True)
class V21E3AttemptEvent:
    attempt_index: int
    charged_evaluation_index: int | None
    status: str
    cache_hit: bool
    type_index: int
    search_slot: int
    search_phase: str
    operator: str
    proposal: Solution
    proposal_sha256: str
    retry_ordinal: int
    fallback_used: bool
    construction_variant: int | None
    generation_parent_type_ids: Tuple[int, ...]


@dataclass(frozen=True)
class V21E3EvaluationEvent:
    charged_evaluation_index: int
    attempt_index: int
    type_index: int
    search_slot: int
    search_phase: str
    operator: str
    proposal: Solution
    proposal_sha256: str
    objectives: ObjectiveVector
    effective_direction: Tuple[float, ...]
    local_search_block_id: int | None
    local_search_depth: int
    construction_variant: int | None
    generation_parent_type_ids: Tuple[int, ...]
    accepted_into_population: bool
    population_considered_type_ids: Tuple[int, ...]
    population_target_type_ids: Tuple[int, ...]
    archive_changed: bool
    retained_after_update: bool
    new_evaluated_cell: bool
    new_nondominated_cell: bool


@dataclass(frozen=True)
class V21E3RunResult:
    optimization_result: OptimizationResult
    trace: Tuple[V21E3EvaluationEvent, ...]
    attempts: Tuple[V21E3AttemptEvent, ...]


@dataclass(frozen=True)
class _GeneratedCandidate:
    solution: Solution
    operator: str
    phase: str
    generation_parents: Tuple[Solution, ...]
    generation_parent_type_ids: Tuple[int, ...]
    screening_witness: dict[str, object] | None = None


class V21E3TypedHybridParetoSearch:
    """Budget-exact V21e3 engine with duplicate-free objective charging."""

    _MOTSP_OPERATORS = (
        "motsp_candidate_list_two_opt_v21e3",
        "motsp_relocate_v21e3",
        "motsp_restricted_three_opt_v21e3",
    )
    _MOKP_OPERATORS = (
        "mokp_uniform_crossover_mutation_repair_v21e3",
        "mokp_add_drop_repair_v21e3",
        "mokp_swap_repair_v21e3",
        "mokp_multibit_repair_v21e3",
    )

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        config: V21E3HybridConfig,
    ) -> None:
        if problem.num_objectives != len(config.reference_directions[0]):
            raise ValueError("Problem and direction dimensions disagree.")
        successor_contract = _SUCCESSOR_FACTORIAL_POLICIES.get(
            config.development_diagnostic_id or ""
        )
        if successor_contract is not None:
            expected_family = successor_contract[0]
            actual_family = (
                "MOKP"
                if isinstance(problem, MultiObjectiveKnapsackInstance)
                else (
                    "MOTSP"
                    if isinstance(problem, MultiObjectiveTSPProblemAdapter)
                    else "UNSUPPORTED"
                )
            )
            if actual_family != expected_family:
                raise ValueError(
                    "The successor factorial diagnostic and problem family disagree."
                )
        v9_contract = _V9_DIAGNOSTIC_POLICIES.get(
            config.development_diagnostic_id or ""
        )
        if v9_contract is not None:
            v9_expected_family = v9_contract[0]
            actual_family = (
                "MOKP"
                if isinstance(problem, MultiObjectiveKnapsackInstance)
                else (
                    "MOTSP"
                    if isinstance(problem, MultiObjectiveTSPProblemAdapter)
                    else "UNSUPPORTED"
                )
            )
            if actual_family != v9_expected_family:
                raise ValueError(
                    "The V9 information-Lyapunov diagnostic and problem family disagree."
                )
            if problem.num_objectives != 2:
                raise ValueError(
                    "V9 normalized-HV diagnostics currently require exactly two "
                    "objectives."
                )
        from .pareto_v21e3_trace import (
            DecisionInput,
            EvaluationContext,
            V21E3SQLiteLedger,
            V21E3RunContext,
        )

        self.problem = problem
        self.config = config
        self._lower = tuple(float(value) for value in problem.objective_lower_bounds)
        self._upper = tuple(float(value) for value in problem.objective_upper_bounds)
        self._DecisionInput = DecisionInput
        self._EvaluationContext = EvaluationContext
        semantic_config = config.semantic_payload()
        config_raw = json.dumps(
            semantic_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        problem_digest = problem_sha256(problem)
        context_payload: dict[str, object] = {
            "schema": "v21e3r1_run_context_v2",
            "case_artifact_sha256": config.case_artifact_sha256 or problem_digest,
            "case_artifact_binding_kind": (
                "explicit_case_artifact_sha256_v1"
                if config.case_artifact_sha256 is not None
                else "problem_semantic_sha256_fallback_development_only_v1"
            ),
            "problem_semantic_sha256": problem_digest,
            "candidate_id": config.candidate_id,
            "algorithm_config": semantic_config,
            "candidate_config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "algorithm_source_sha256": (
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                if config.source_snapshot_sha256 is None
                else config.source_snapshot_sha256
            ),
            "algorithm_source_binding_kind": (
                "explicit_source_snapshot_or_release_manifest_sha256_v1"
                if config.source_snapshot_sha256 is not None
                else "hybrid_module_sha256_fallback_development_only_v1"
            ),
            "reference_directions": config.reference_directions,
            "seed": config.seed,
            "charged_evaluation_budget": config.charged_evaluations,
            "evidence_partition": config.phase,
        }
        if v9_contract is not None:
            context_payload.update(
                objective_lower_bounds=self._lower,
                objective_upper_bounds=self._upper,
                v9_resource_contract_schema="v21e3r1_v9_ast_resource_contract_v1",
            )
        run_context = V21E3RunContext(context_payload)
        self._ledger = V21E3SQLiteLedger.from_problem(
            problem,
            run_context=run_context,
            database_path=config.trace_database,
            receipt_path=config.terminal_receipt,
            receipt_database_path=config.receipt_database_path,
        )
        self._central_direction = tuple(
            1.0 / problem.num_objectives for _ in range(problem.num_objectives)
        )
        self._type_neighbors = self._build_type_neighbors()
        self._rng_initialization = random.Random(
            self._domain_seed(config.seed, "matched_initialization")
        )
        self._rng_native = random.Random(
            self._domain_seed(config.seed, "matched_native")
        )
        self._rng_diversification = random.Random(
            self._domain_seed(config.seed, "matched_diversification")
        )
        self._rng_exchange = random.Random(
            self._domain_seed(config.seed, "matched_exchange")
        )
        self._rng_retry = random.Random(
            self._domain_seed(config.seed, "duplicate_retry")
        )
        self._rng_fallback = random.Random(
            self._domain_seed(config.seed, "unique_fallback")
        )
        self.archive = ParetoArchive(max_size=None, tol=0.0)
        self._regions = V21E3RegionOccupancy(
            lower_bounds=self._lower,
            upper_bounds=self._upper,
            bins=20,
        )
        self._solutions: list[Solution | None] = [
            None for _ in config.reference_directions
        ]
        self._objectives: list[ObjectiveVector | None] = [
            None for _ in config.reference_directions
        ]
        self._native_calls_by_type = [0 for _ in config.reference_directions]
        # The add/drop arm has its own deterministic per-type sub-schedule.
        # Keeping this state separate from the four-arm native schedule makes
        # add and drop alternate without collapsing either into swap.
        self._mokp_add_drop_calls_by_type = [
            0 for _ in config.reference_directions
        ]
        self._mokp_successor_novelty_calls_by_origin = {
            "bounded_local_improvement": [
                0 for _ in config.reference_directions
            ],
            "post_initialization_duplicate_retry": [
                0 for _ in config.reference_directions
            ],
        }
        self._attempt_events: list[V21E3AttemptEvent] = []
        self._evaluation_events: list[V21E3EvaluationEvent] = []
        self._evaluated_entries: list[ArchiveEntry] = []
        self._diagnostics: list[Diagnostic] = []
        self._cache_hit_count = 0
        self._retry_count = 0
        self._fallback_count = 0
        self._exchange_call_count = 0
        self._diversification_call_count = 0
        self._accepted_count = 0
        self._candidate_screen_count = 0
        self._candidate_screen_cache_skip_count = 0
        self._structural_candidate_generation_count = 0
        self._archive_lyapunov_replacement_count = 0
        self._archive_lyapunov_paid_worsening_count = 0
        self._operator_call_id = 0
        self._local_block_id = 0
        self._has_run = False
        self._start = time.perf_counter()
        self._resource_start: float | None = None
        self._resource_budget = (
            None
            if v9_contract is None
            else DualResourceBudget(
                first_evaluation_cap=config.charged_evaluations,
                attempt_cap=int(config.attempt_cap),
                screening_cap=int(config.structural_screening_cap),
                wall_time_cap_seconds=(
                    None
                    if config.wall_time_cap_seconds is None
                    else float(config.wall_time_cap_seconds)
                ),
            )
        )
        self._motsp_candidate_cities = self._build_motsp_candidates()

    def _resource_elapsed_seconds(self) -> float:
        if self._resource_start is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._resource_start)

    def _resource_accounting(self) -> dict[str, object]:
        if self._resource_budget is None:
            raise RuntimeError("A non-V9 run has no A/S/T resource contract.")
        elapsed = self._resource_elapsed_seconds()
        structural_work = (
            self._structural_candidate_generation_count
            + self._candidate_screen_count
        )
        satisfied = self._resource_budget.permits(
            first_evaluations=self._ledger.evaluation_count,
            attempts=self._ledger.attempt_count,
            screenings=structural_work,
            elapsed_seconds=elapsed,
        )
        return {
            "schema": "v21e3r1_v9_ast_resource_accounting_v1",
            "first_evaluations": self._ledger.evaluation_count,
            "first_evaluation_cap": self._resource_budget.first_evaluation_cap,
            "attempts": self._ledger.attempt_count,
            "attempt_cap": self._resource_budget.attempt_cap,
            "structural_candidate_generations": (
                self._structural_candidate_generation_count
            ),
            "cache_membership_probes": self._candidate_screen_count,
            "structural_screening_work": structural_work,
            "structural_screening_cap": self._resource_budget.screening_cap,
            "elapsed_seconds": elapsed,
            "wall_time_cap_seconds": self._resource_budget.wall_time_cap_seconds,
            "all_configured_caps_satisfied": satisfied,
        }

    def _fail_resource_contract(
        self,
        *,
        resource: str,
        observed: int | float,
        cap: int | float,
        boundary: str,
    ) -> None:
        accounting = self._resource_accounting()
        accounting["all_configured_caps_satisfied"] = False
        detail = {
            "schema": "v21e3r1_v9_ast_resource_failure_v1",
            "resource": resource,
            "projected_or_observed": observed,
            "cap": cap,
            "boundary": boundary,
            "resource_accounting_at_failure": accounting,
        }
        self._ledger.finalize_failure(
            failure_code="V9_RESOURCE_CAP_EXHAUSTED",
            failure_detail=detail,
        )
        raise V21E3ResourceLimitExceeded(
            f"V9 {resource} cap exhausted at {boundary}: {observed} > {cap}."
        )

    def _check_wall_time(self, *, boundary: str) -> None:
        if (
            self._resource_budget is None
            or self._resource_budget.wall_time_cap_seconds is None
        ):
            return
        elapsed = self._resource_elapsed_seconds()
        cap = float(self._resource_budget.wall_time_cap_seconds)
        if elapsed > cap:
            self._fail_resource_contract(
                resource="wall_time_seconds",
                observed=elapsed,
                cap=cap,
                boundary=boundary,
            )

    def _before_attempt(self) -> None:
        if self._resource_budget is None:
            return
        self._check_wall_time(boundary="before_attempt")
        projected = self._ledger.attempt_count + 1
        if projected > self._resource_budget.attempt_cap:
            self._fail_resource_contract(
                resource="attempts",
                observed=projected,
                cap=self._resource_budget.attempt_cap,
                boundary="before_attempt",
            )

    def _consume_structural_work(self, *, kind: str) -> None:
        if self._resource_budget is None:
            raise RuntimeError("Structural V9 work requires an active resource contract.")
        if kind not in {"candidate_generation", "cache_membership_probe"}:
            raise ValueError("Unknown V9 structural-work kind.")
        self._check_wall_time(boundary=f"before_{kind}")
        observed = (
            self._structural_candidate_generation_count
            + self._candidate_screen_count
            + 1
        )
        if observed > self._resource_budget.screening_cap:
            self._fail_resource_contract(
                resource="structural_screening_work",
                observed=observed,
                cap=self._resource_budget.screening_cap,
                boundary=f"before_{kind}",
            )
        if kind == "candidate_generation":
            self._structural_candidate_generation_count += 1
        else:
            self._candidate_screen_count += 1

    def run(self) -> V21E3RunResult:
        if self._has_run:
            raise RuntimeError("A V21e3 optimizer instance is single-use.")
        self._has_run = True
        self._start = time.perf_counter()
        if self._resource_budget is not None:
            self._resource_start = self._start
            self._check_wall_time(boundary="run_start")
        for type_index in range(len(self.config.reference_directions)):
            proposal, operator = self._initial_solution(type_index)
            self._charge_unique(
                type_index=type_index,
                search_slot=0,
                search_phase="matched_construction",
                operator=operator,
                proposal=proposal,
                parent=None,
                local_search_block_id=None,
                local_search_depth=0,
                construction_variant=type_index,
                generation_parents=(),
                generation_parent_type_ids=(),
            )

        search_slot = 0
        while self._ledger.evaluation_count < self.config.charged_evaluations:
            search_slot += 1
            type_index = (search_slot - 1) % len(self.config.reference_directions)
            parent = self._require_solution(type_index)
            generated = self._scheduled_candidate(
                type_index,
                search_slot,
            )
            self._local_block_id += 1
            first = self._charge_unique(
                type_index=type_index,
                search_slot=search_slot,
                search_phase=generated.phase,
                operator=generated.operator,
                proposal=generated.solution,
                parent=parent,
                local_search_block_id=self._local_block_id,
                local_search_depth=0,
                construction_variant=None,
                generation_parents=generated.generation_parents,
                generation_parent_type_ids=generated.generation_parent_type_ids,
                candidate_screening_witness=generated.screening_witness,
            )
            if (
                self.config.post_initialization_search_policy
                == _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY
            ):
                anchor = self._require_solution(type_index)
                anchor_objectives = self._require_objective(type_index)
            else:
                anchor = first.proposal
                anchor_objectives = first.objectives
            for depth in range(1, self.config.local_improvement_steps + 1):
                if self._ledger.evaluation_count >= self.config.charged_evaluations:
                    break
                (
                    local_proposal,
                    local_operator,
                    local_novelty_witness,
                ) = self._local_candidate(
                    anchor,
                    type_index,
                    depth,
                )
                local_event = self._charge_unique(
                    type_index=type_index,
                    search_slot=search_slot,
                    search_phase="bounded_local_improvement",
                    operator=local_operator,
                    proposal=local_proposal,
                    parent=anchor,
                    local_search_block_id=self._local_block_id,
                    local_search_depth=depth,
                    construction_variant=None,
                    generation_parents=(anchor,),
                    generation_parent_type_ids=(type_index,),
                    successor_novelty_witness=local_novelty_witness,
                )
                if (
                    self.config.post_initialization_search_policy
                    == _SUCCESSOR_POST_INITIALIZATION_SEARCH_POLICY
                ):
                    anchor = self._require_solution(type_index)
                    anchor_objectives = self._require_objective(type_index)
                else:
                    direction = self._direction(type_index)
                    if self._scalar(local_event.objectives, direction) < self._scalar(
                        anchor_objectives,
                        direction,
                    ):
                        anchor = local_event.proposal
                        anchor_objectives = local_event.objectives

        rebuilt = ParetoArchive(max_size=None, tol=0.0)
        rebuilt.update(tuple(self._evaluated_entries))
        if rebuilt.entries != self.archive.entries:
            raise RuntimeError("V21e3 evaluated history does not reproduce archive.")
        resource_accounting = None
        if self._resource_budget is not None:
            self._check_wall_time(boundary="before_terminal_finalization")
            resource_accounting = self._resource_accounting()
            if resource_accounting["all_configured_caps_satisfied"] is not True:
                self._fail_resource_contract(
                    resource="aggregate_resource_contract",
                    observed=1,
                    cap=0,
                    boundary="before_terminal_finalization",
                )
        receipt = self._ledger.finalize(
            expected_charged_evaluations=self.config.charged_evaluations,
            expected_decisions=self.config.charged_evaluations,
            resource_accounting=resource_accounting,
        )
        particles = tuple(self._require_solution(index) for index in range(len(self._solutions)))
        objectives = tuple(self._require_objective(index) for index in range(len(self._objectives)))
        metadata = {
            "algorithm": f"v21e3-matched-hybrid-{self.config.candidate_id.lower()}",
            "candidate_id": self.config.candidate_id,
            "candidate_spec": asdict(v21e3_candidate_spec(self.config.candidate_id)),
            "direction_policy": v21e3_candidate_spec(
                self.config.candidate_id
            ).direction_policy,
            "charged_evaluation_budget": self.config.charged_evaluations,
            "charged_evaluation_count": self._ledger.evaluation_count,
            "unique_true_evaluation_count": self._ledger.evaluation_count,
            "physical_objective_call_count": self._ledger.physical_call_count,
            "attempt_count": self._ledger.attempt_count,
            "cache_hit_count": self._cache_hit_count,
            "retry_count": self._retry_count,
            "fallback_count": self._fallback_count,
            "candidate_screen_count": self._candidate_screen_count,
            "candidate_screen_cache_skip_count": self._candidate_screen_cache_skip_count,
            "structural_candidate_generation_count": (
                self._structural_candidate_generation_count
            ),
            "structural_screening_work_count": (
                self._structural_candidate_generation_count
                + self._candidate_screen_count
            ),
            "v9_resource_accounting": resource_accounting,
            "archive_lyapunov_replacement_count": self._archive_lyapunov_replacement_count,
            "archive_lyapunov_paid_worsening_count": self._archive_lyapunov_paid_worsening_count,
            "candidate_screening_policy": self.config.candidate_screening_policy,
            "candidate_screening_cap": self.config.candidate_screening_cap,
            "archive_tradeoff_lambda": float(self.config.archive_tradeoff_lambda),
            "exchange_operator_call_count": self._exchange_call_count,
            "exchange_random_draw_count": self._exchange_call_count,
            "diversification_operator_call_count": self._diversification_call_count,
            "duplicate_avoidance_rate": (
                self._cache_hit_count / self._ledger.attempt_count
                if self._ledger.attempt_count
                else 0.0
            ),
            "exact_charged_budget_gate": (
                "PASS"
                if self._ledger.evaluation_count == self.config.charged_evaluations
                else "FAIL"
            ),
            "attempt_semantics": (
                "attempts_separate_from_first_true_objective_evaluations_v1"
            ),
            "replacement_contract": self.config.replacement_policy,
            "evaluated_region_count": self._regions.evaluated_region_count,
            "nondominated_region_count": self._regions.nondominated_region_count,
            "checkpoint_hypervolume_metric": (
                "exact_2d"
                if self.problem.num_objectives == 2
                else "UNAVAILABLE_NON_2D"
            ),
            "trace_receipt": receipt,
            "v21e2_evidence_transfer": "PROHIBITED",
            "formal_evidence": "NOT_ESTABLISHED",
        }
        result = OptimizationResult(
            particles=particles,
            objectives=objectives,
            archive=self.archive,
            diagnostics=tuple(self._diagnostics),
            metadata=metadata,
        )
        return V21E3RunResult(
            optimization_result=result,
            trace=(tuple(self._evaluation_events) if self.config.capture_trace else ()),
            attempts=(tuple(self._attempt_events) if self.config.capture_trace else ()),
        )

    def _charge_unique(
        self,
        *,
        type_index: int,
        search_slot: int,
        search_phase: str,
        operator: str,
        proposal: Solution,
        parent: Solution | None,
        local_search_block_id: int | None,
        local_search_depth: int,
        construction_variant: int | None,
        generation_parents: Tuple[Solution, ...],
        generation_parent_type_ids: Tuple[int, ...],
        successor_novelty_witness: dict[str, object] | None = None,
        candidate_screening_witness: dict[str, object] | None = None,
    ) -> V21E3EvaluationEvent:
        current = tuple(proposal)
        current_operator = operator
        current_parents = tuple(generation_parents)
        current_parent_types = tuple(generation_parent_type_ids)
        current_novelty_witness = successor_novelty_witness
        current_screening_witness = candidate_screening_witness
        if (
            parent is not None
            and self.config.candidate_screening_policy
            == _V9_CACHE_AWARE_SCREENING_POLICY
        ):
            (
                current,
                current_operator,
                current_screening_witness,
            ) = self._cache_aware_screen_candidate(
                initial=current,
                parent=parent,
                type_index=type_index,
                search_slot=search_slot,
                local_search_depth=local_search_depth,
                original_operator=current_operator,
            )
        total_limit = self.config.duplicate_retry_cap + self.config.fallback_attempt_cap
        for ordinal in range(total_limit + 1):
            fallback = ordinal > self.config.duplicate_retry_cap
            if ordinal > 0:
                if fallback:
                    self._fallback_count += 1
                    current = self._fallback_candidate(type_index)
                    current_operator = "frozen_unique_fallback_v21e3"
                    current_parents = ()
                    current_parent_types = ()
                    current_novelty_witness = None
                    current_screening_witness = None
                else:
                    self._retry_count += 1
                    retry_parent = current
                    if (
                        parent is not None
                        and isinstance(
                            self.problem, MultiObjectiveKnapsackInstance
                        )
                        and self.config.mokp_novelty_generation_policy
                        == _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY
                    ):
                        (
                            current,
                            current_operator,
                            current_novelty_witness,
                        ) = self._mokp_successor_novelty_candidate(
                            current,
                            type_index,
                            origin="post_initialization_duplicate_retry",
                        )
                    else:
                        current = self._retry_candidate(
                            current, type_index, ordinal
                        )
                        current_operator = "duplicate_retry_perturbation_v21e3"
                        current_novelty_witness = None
                    current_screening_witness = None
                    current_parents = (retry_parent,)
                    current_parent_types = (type_index,)
            self._operator_call_id += 1
            operator_witness: dict[str, object] = {
                "search_slot": search_slot,
                "retry_ordinal": ordinal,
                "fallback_used": fallback,
                "construction_variant": construction_variant,
            }
            if current_novelty_witness is not None:
                operator_witness["successor_mokp_novelty"] = dict(
                    current_novelty_witness
                )
            if current_screening_witness is not None:
                operator_witness["information_time_candidate_screen"] = dict(
                    current_screening_witness
                )
            self._before_attempt()
            outcome = self._ledger.attempt(
                current,
                self._EvaluationContext(
                    evidence_partition=self.config.phase,
                    search_phase_id=search_phase,
                    stage_id=(
                        "initialization_v21e3"
                        if parent is None
                        else "search_v21e3"
                    ),
                    type_id=type_index,
                    operator_id=current_operator,
                    operator_call_id=self._operator_call_id,
                    parent_solutions=current_parents,
                    parent_type_ids=current_parent_types,
                    repair_applied=("repair" in current_operator),
                    repair_operator_id=(
                        "matched_weighted_repair_v21e3"
                        if "repair" in current_operator
                        else None
                    ),
                    local_search_block_id=local_search_block_id,
                    local_search_depth=local_search_depth,
                    operator_witness=operator_witness,
                ),
            )
            self._check_wall_time(boundary="after_attempt")
            self._attempt_events.append(
                V21E3AttemptEvent(
                    attempt_index=outcome.attempt_index,
                    charged_evaluation_index=outcome.charged_evaluation_index,
                    status=outcome.status,
                    cache_hit=outcome.cache_hit,
                    type_index=type_index,
                    search_slot=search_slot,
                    search_phase=search_phase,
                    operator=current_operator,
                    proposal=outcome.proposal,
                    proposal_sha256=outcome.proposal_sha256,
                    retry_ordinal=ordinal,
                    fallback_used=fallback,
                    construction_variant=construction_variant,
                    generation_parent_type_ids=current_parent_types,
                )
            )
            if outcome.cache_hit:
                self._cache_hit_count += 1
                continue
            if outcome.charged_evaluation_index is None:
                raise RuntimeError("A non-cache attempt was not charged.")
            return self._commit_evaluated_candidate(
                outcome=outcome,
                type_index=type_index,
                search_slot=search_slot,
                search_phase=search_phase,
                operator=current_operator,
                parent=parent,
                local_search_block_id=local_search_block_id,
                local_search_depth=local_search_depth,
                construction_variant=construction_variant,
                generation_parent_type_ids=current_parent_types,
            )
        try:
            self._ledger.finalize(
                expected_charged_evaluations=self.config.charged_evaluations,
                expected_decisions=self.config.charged_evaluations,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "V21e3 unique-state retry/fallback cap exhausted before budget "
                "charge; the ledger wrote a terminal FAILURE receipt."
            ) from error
        raise RuntimeError(
            "V21e3 retry/fallback cap exhausted without a failing finalization gate."
        )

    def _cache_aware_screen_candidate(
        self,
        *,
        initial: Solution,
        parent: Solution,
        type_index: int,
        search_slot: int,
        local_search_depth: int,
        original_operator: str,
    ) -> tuple[Solution, str, dict[str, object]]:
        """Select the first unseen structural candidate under a bounded screen.

        Screening uses exact canonical identities and no objective calls.  The
        number of candidates examined is recorded separately from attempts and
        first evaluations.  This is a development-only policy.
        """

        cap = self.config.candidate_screening_cap
        generation_count_before = self._structural_candidate_generation_count
        probe_count_before = self._candidate_screen_count
        alternatives = self._structural_screen_candidates(
            parent=parent,
            type_index=type_index,
            search_slot=search_slot,
            local_search_depth=local_search_depth,
            cap=max(0, cap - 1),
        )
        candidate_items: list[tuple[Solution, str]] = [
            (tuple(initial), original_operator)
        ]
        candidate_items.extend(alternatives)
        exact_unique: list[tuple[Solution, str]] = []
        seen_in_screen: set[Solution] = set()
        for solution, label in candidate_items:
            exact = tuple(solution)
            if exact in seen_in_screen:
                continue
            seen_in_screen.add(exact)
            exact_unique.append((exact, label))
            if len(exact_unique) >= cap:
                break
        if not exact_unique:
            exact_unique = [(tuple(initial), original_operator)]

        membership_checks: list[dict[str, object]] = []

        def _accounted_is_seen(candidate: Solution) -> bool:
            self._consume_structural_work(kind="cache_membership_probe")
            seen = self._ledger.has_evaluated_solution(candidate)
            rank = len(membership_checks)
            exact = tuple(int(value) for value in candidate)
            membership_checks.append(
                {
                    "rank": rank,
                    "solution": exact,
                    "solution_sha256": hashlib.sha256(
                        json.dumps(
                            list(exact),
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "operator": exact_unique[rank][1],
                    "seen_before_attempt": seen,
                }
            )
            return seen

        decision = select_first_unseen(
            (item[0] for item in exact_unique),
            is_seen=_accounted_is_seen,
            cap=min(cap, len(exact_unique)),
        )
        selected_solution = tuple(decision.selected)
        selected_position = next(
            index
            for index, (solution, _label) in enumerate(exact_unique)
            if solution == selected_solution
        )
        selected_label = exact_unique[selected_position][1]
        self._candidate_screen_cache_skip_count += (
            decision.cached_candidates_skipped
        )
        generated_here = (
            self._structural_candidate_generation_count - generation_count_before
        )
        probes_here = self._candidate_screen_count - probe_count_before
        witness: dict[str, object] = {
            "schema": "v21e3r1_information_time_candidate_screen_v2",
            "policy": _V9_CACHE_AWARE_SCREENING_POLICY,
            "screen_cap": cap,
            "candidates_examined": decision.candidates_examined,
            "cached_candidates_skipped": decision.cached_candidates_skipped,
            "selected_rank": selected_position,
            "screen_exhausted": decision.exhausted,
            "selected_operator": selected_label,
            "selected_solution_sha256": hashlib.sha256(
                json.dumps(
                    list(selected_solution),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "candidate_membership_checks": membership_checks,
            "objective_calls_during_screen": 0,
            "structural_candidates_generated": generated_here,
            "cache_membership_probes": probes_here,
            "total_structural_screening_work": generated_here + probes_here,
        }
        return (
            selected_solution,
            (
                selected_label
                if selected_position == 0
                else f"{selected_label}__cache_screened_unseen_v1"
            ),
            witness,
        )

    def _structural_screen_candidates(
        self,
        *,
        parent: Solution,
        type_index: int,
        search_slot: int,
        local_search_depth: int,
        cap: int,
    ) -> list[tuple[Solution, str]]:
        if cap <= 0:
            return []
        output: list[tuple[Solution, str]] = []
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            problem = self.problem
            direction = self._direction(type_index)
            densities = self._mokp_densities(direction)
            source = tuple(parent)
            selected = [index for index, value in enumerate(source) if value]
            unselected = [index for index, value in enumerate(source) if not value]
            current_weight = sum(problem.item_weights[index] for index in selected)
            feasible_adds: list[int] = []
            for index in unselected:
                self._consume_structural_work(kind="candidate_generation")
                if (
                    current_weight + problem.item_weights[index]
                    <= problem.capacity
                ):
                    feasible_adds.append(index)
            feasible_adds.sort(key=lambda index: (-densities[index], index))
            drops: list[int] = []
            for index in selected:
                self._consume_structural_work(kind="candidate_generation")
                drops.append(index)
            drops.sort(key=lambda index: (densities[index], index))
            for added in feasible_adds:
                child = list(source)
                child[added] = 1
                output.append(
                    (
                        tuple(child),
                        "mokp_cache_screen_feasible_add_no_refill_v1",
                    )
                )
                if len(output) >= cap:
                    return output
            for removed in drops:
                child = list(source)
                child[removed] = 0
                output.append(
                    (
                        tuple(child),
                        "mokp_cache_screen_drop_no_refill_v1",
                    )
                )
                if len(output) >= cap:
                    return output

            for removed in drops:
                ordered_adds = sorted(
                    unselected,
                    key=lambda added: (
                        -(densities[added] - densities[removed]),
                        added,
                    ),
                )
                for added in ordered_adds:
                    self._consume_structural_work(kind="candidate_generation")
                    if (
                        current_weight
                        - problem.item_weights[removed]
                        + problem.item_weights[added]
                        > problem.capacity
                    ):
                        continue
                    child = list(source)
                    child[removed] = 0
                    child[added] = 1
                    output.append(
                        (
                            tuple(child),
                            "mokp_cache_screen_feasible_swap_no_refill_v1",
                        )
                    )
                    if len(output) >= cap:
                        return output
            return output
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            source = tuple(parent)
            n = len(source)
            if n < 4:
                return output
            offset = (
                search_slot * 17
                + type_index * 11
                + local_search_depth * 5
            )
            pair_count = (n - 2) * (n - 1) // 2
            if pair_count <= 0:
                return output

            def _pair_at(rank: int) -> tuple[int, int]:
                low = 1
                high = n - 2
                while low <= high:
                    left = (low + high) // 2
                    row_start = (left - 1) * (2 * n - left - 2) // 2
                    row_width = n - left - 1
                    if rank < row_start:
                        high = left - 1
                    elif rank >= row_start + row_width:
                        low = left + 1
                    else:
                        return left, left + 1 + (rank - row_start)
                raise RuntimeError("MOTSP structural-screen pair rank is invalid.")

            for step in range(min(cap, pair_count)):
                left, right = _pair_at((offset + step) % pair_count)
                self._consume_structural_work(kind="candidate_generation")
                output.append(
                    (
                        two_opt_at(source, left, right),
                        "motsp_cache_screen_deterministic_two_opt_v1",
                    )
                )
            return output
        return output

    def _normalized_archive_hypervolume(self) -> float:
        if self.problem.num_objectives != 2:
            raise RuntimeError(
                "Archive-compensated replacement currently requires two objectives."
            )
        scale = (self._upper[0] - self._lower[0]) * (
            self._upper[1] - self._lower[1]
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("The frozen objective box has invalid hypervolume scale.")
        raw = self.archive.hypervolume_2d(reference=self._upper)
        normalized = raw / scale
        if not math.isfinite(normalized) or normalized < -1e-12:
            raise RuntimeError("The normalized archive hypervolume is invalid.")
        return max(0.0, normalized)

    def _commit_evaluated_candidate(
        self,
        *,
        outcome: object,
        type_index: int,
        search_slot: int,
        search_phase: str,
        operator: str,
        parent: Solution | None,
        local_search_block_id: int | None,
        local_search_depth: int,
        construction_variant: int | None,
        generation_parent_type_ids: Tuple[int, ...],
    ) -> V21E3EvaluationEvent:
        objectives = tuple(float(value) for value in outcome.objectives)
        entry = ArchiveEntry(tuple(outcome.proposal), objectives)
        lyapunov_event = (
            parent is not None
            and self.config.replacement_policy
            == _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY
        )
        normalized_hv_before = (
            self._normalized_archive_hypervolume() if lyapunov_event else None
        )
        archive_changed = self.archive.update((entry,))
        normalized_hv_after = (
            self._normalized_archive_hypervolume() if lyapunov_event else None
        )
        normalized_hv_gain = (
            0.0
            if normalized_hv_before is None or normalized_hv_after is None
            else max(0.0, normalized_hv_after - normalized_hv_before)
        )
        retained = self.archive.contains(entry)
        self._evaluated_entries.append(entry)
        direction = self._direction(type_index)
        policy_witness: dict[str, object] | None = None
        if parent is None:
            considered_targets = (type_index,)
            replacement_targets = (type_index,)
        else:
            targets = (
                (type_index,)
                if self.config.replacement_policy
                == "self_type_nonworse_replacement_development_diagnostic_v1"
                else self._type_neighbors[type_index]
            )
            considered_targets = targets
            if (
                self.config.replacement_policy
                == _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY
            ):
                deltas = {
                    target: (
                        float("-inf")
                        if self._objectives[target] is None
                        else self._scalar(objectives, self._direction(target))
                        - self._scalar(
                            self._require_objective(target),
                            self._direction(target),
                        )
                    )
                    for target in targets
                }
                preselected_empty_targets = tuple(
                    target for target in targets if self._objectives[target] is None
                )
                finite_deltas = {
                    target: value
                    for target, value in deltas.items()
                    if math.isfinite(value)
                }
                selection_capacity = max(
                    0, len(targets) - len(preselected_empty_targets)
                )
                decision = archive_compensated_replacement(
                    finite_deltas,
                    normalized_hv_gain=normalized_hv_gain,
                    tradeoff_lambda=float(self.config.archive_tradeoff_lambda),
                    max_targets=selection_capacity,
                )
                replacement_targets = (
                    preselected_empty_targets + decision.selected_targets
                )
                if replacement_targets:
                    self._archive_lyapunov_replacement_count += 1
                paid_worsening = sum(
                    1
                    for target in replacement_targets
                    if self._objectives[target] is not None
                    and self._scalar(objectives, self._direction(target))
                    > self._scalar(
                        self._require_objective(target), self._direction(target)
                    )
                )
                self._archive_lyapunov_paid_worsening_count += paid_worsening
                positive_worsening_sum = sum(
                    max(
                        0.0,
                        self._scalar(objectives, self._direction(target))
                        - self._scalar(
                            self._require_objective(target),
                            self._direction(target),
                        ),
                    )
                    for target in replacement_targets
                    if self._objectives[target] is not None
                )
                archive_credit = float(
                    self.config.archive_tradeoff_lambda
                ) * normalized_hv_gain
                if positive_worsening_sum > archive_credit + 1e-10:
                    raise RuntimeError(
                        "Positive scalar worsening exceeded archive credit."
                    )
                selected_delta_sum = sum(
                    self._scalar(objectives, self._direction(target))
                    - self._scalar(
                        self._require_objective(target), self._direction(target)
                    )
                    for target in replacement_targets
                    if self._objectives[target] is not None
                )
                potential_change = selected_delta_sum - float(
                    self.config.archive_tradeoff_lambda
                ) * normalized_hv_gain
                if potential_change > 1e-10:
                    raise RuntimeError(
                        "Archive-compensated replacement violated its Lyapunov bound."
                    )
                policy_witness = {
                    "schema": "v21e3r1_archive_compensated_replacement_v2",
                    "replacement_policy": _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY,
                    "normalized_hv_before": float(normalized_hv_before),
                    "normalized_hv_after": float(normalized_hv_after),
                    "normalized_hv_gain": normalized_hv_gain,
                    "tradeoff_lambda": float(self.config.archive_tradeoff_lambda),
                    "selected_scalar_delta_sum": selected_delta_sum,
                    "positive_scalar_worsening_sum": positive_worsening_sum,
                    "archive_credit": archive_credit,
                    "composite_potential_change": potential_change,
                    "paid_worsening_target_count": paid_worsening,
                    "considered_target_type_ids": tuple(targets),
                    "preselected_empty_target_type_ids": (
                        preselected_empty_targets
                    ),
                    "finite_scalar_delta_by_target": tuple(
                        {
                            "target_type_id": int(target),
                            "scalar_delta": float(delta),
                        }
                        for target, delta in sorted(finite_deltas.items())
                    ),
                    "finite_selection_capacity": selection_capacity,
                    "decision_selected_target_type_ids": (
                        decision.selected_targets
                    ),
                    "selected_target_type_ids": tuple(replacement_targets),
                }
            else:
                replacement_targets = tuple(
                    target
                    for target in targets
                    if self._objectives[target] is None
                    or self._scalar(objectives, self._direction(target))
                    <= self._scalar(
                        self._require_objective(target),
                        self._direction(target),
                    )
                )
        for target in replacement_targets:
            self._solutions[target] = tuple(outcome.proposal)
            self._objectives[target] = objectives
        accepted = bool(replacement_targets)
        if accepted:
            self._accepted_count += 1
        region = self._regions.observe(
            objectives,
            nondominated=tuple(item.objectives for item in self.archive.entries),
        )
        parent_objective = (
            None
            if parent is None
            else next(
                (
                    event.objectives
                    for event in reversed(self._evaluation_events)
                    if event.proposal == parent
                ),
                self._objectives[type_index],
            )
        )
        scalar_candidate = self._scalar(objectives, direction)
        scalar_parent = (
            None
            if parent_objective is None
            else self._scalar(parent_objective, direction)
        )
        self._ledger.commit_decision(
            outcome.charged_evaluation_index,
            self._DecisionInput(
                accepted_into_population=accepted,
                population_replacement_count=len(replacement_targets),
                population_target_type_ids=replacement_targets,
                decision_reason=(
                    "initial_population_fill"
                    if parent is None
                    else (
                        (
                            "archive_compensated_lyapunov_replacement"
                            if self.config.replacement_policy
                            == _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY
                            else (
                                "nonworse_self_type_replacement"
                                if self.config.replacement_policy
                                == "self_type_nonworse_replacement_development_diagnostic_v1"
                                else "nonworse_neighborhood_replacement"
                            )
                        )
                        if accepted
                        else (
                            "archive_credit_insufficient_rejection"
                            if self.config.replacement_policy
                            == _V9_ARCHIVE_LYAPUNOV_REPLACEMENT_POLICY
                            else "worse_neighborhood_rejection"
                        )
                    )
                ),
                archive_changed=archive_changed,
                retained_after_update=retained,
                archive_size_after=len(self.archive),
                scalarization_id=(
                    "central_untyped_normalized_tchebycheff_v1"
                    if self.config.candidate_id == "C0"
                    else "typed_normalized_tchebycheff_v1"
                ),
                scalar_parent=scalar_parent,
                scalar_candidate=scalar_candidate,
                scalar_advantage=(
                    None
                    if scalar_parent is None
                    else scalar_parent - scalar_candidate
                ),
                cell_id=":".join(str(value) for value in region.region),
                new_evaluated_cell=region.new_evaluated_cell,
                new_nondominated_cell=region.new_nondominated_cell,
                policy_witness=policy_witness,
            ),
        )
        event = V21E3EvaluationEvent(
            charged_evaluation_index=outcome.charged_evaluation_index,
            attempt_index=outcome.attempt_index,
            type_index=type_index,
            search_slot=search_slot,
            search_phase=search_phase,
            operator=operator,
            proposal=tuple(outcome.proposal),
            proposal_sha256=outcome.proposal_sha256,
            objectives=objectives,
            effective_direction=direction,
            local_search_block_id=local_search_block_id,
            local_search_depth=local_search_depth,
            construction_variant=construction_variant,
            generation_parent_type_ids=generation_parent_type_ids,
            accepted_into_population=accepted,
            population_considered_type_ids=considered_targets,
            population_target_type_ids=replacement_targets,
            archive_changed=archive_changed,
            retained_after_update=retained,
            new_evaluated_cell=region.new_evaluated_cell,
            new_nondominated_cell=region.new_nondominated_cell,
        )
        self._evaluation_events.append(event)
        self._record_checkpoint_if_due(outcome.charged_evaluation_index)
        return event

    def _record_checkpoint_if_due(self, evaluation_index: int) -> None:
        if (
            evaluation_index % self.config.checkpoint_period != 0
            and evaluation_index != self.config.charged_evaluations
        ):
            return
        self._diagnostics.append(
            Diagnostic(
                iteration=evaluation_index,
                temperature=0.0,
                acceptance_rate=self._accepted_count / evaluation_index,
                archive_size=len(self.archive),
                hypervolume_2d=(
                    self.archive.hypervolume_2d(reference=self._upper)
                    if self.problem.num_objectives == 2
                    else 0.0
                ),
                empirical_energy=0.0,
                positive_archive_jump=0.0,
                front=tuple(entry.objectives for entry in self.archive.entries),
                elapsed_seconds=time.perf_counter() - self._start,
            )
        )

    def _initial_solution(self, type_index: int) -> tuple[Solution, str]:
        if (
            self.config.initialization_policy
            == "problem_native_exact_random_solution_development_diagnostic_v1"
        ):
            return (
                self.problem.random_solution(self._rng_initialization),
                "problem_native_exact_random_initialization_development_diagnostic_v1",
            )
        return family_aware_initial_solution(
            self.problem,
            self._direction(type_index),
            type_index,
        )

    def _scheduled_candidate(
        self,
        type_index: int,
        search_slot: int,
    ) -> _GeneratedCandidate:
        schedule = v21e3_schedule_slot(
            self.config.candidate_id,
            search_slot,
            diversification_period=self.config.diversification_period,
            exchange_period=self.config.exchange_period,
        )
        if schedule.kind == "exchange":
            return self._exchange_candidate(type_index)
        if schedule.kind == "diversification":
            return self._diversification_candidate(type_index)
        return self._native_candidate(type_index)

    def _native_candidate(self, type_index: int) -> _GeneratedCandidate:
        call = self._native_calls_by_type[type_index]
        self._native_calls_by_type[type_index] += 1
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            operator_index = call % len(self._MOTSP_OPERATORS)
            proposal, operator, parent, parent_type = self._motsp_native(
                type_index,
                operator_index,
            )
            return _GeneratedCandidate(
                proposal,
                operator,
                "native_backbone",
                (parent,),
                (parent_type,),
            )
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            operator_index = call % len(self._MOKP_OPERATORS)
            proposal, operator, parents, parent_types = self._mokp_native(
                type_index,
                operator_index,
                operator_call=call,
            )
            return _GeneratedCandidate(
                proposal,
                operator,
                "native_backbone",
                parents,
                parent_types,
            )
        parent = self._require_solution(type_index)
        return _GeneratedCandidate(
            self.problem.propose(parent, self._rng_native),
            "generic_native_proposal_v21e3",
            "native_backbone",
            (parent,),
            (type_index,),
        )

    def _motsp_native(
        self,
        type_index: int,
        operator_index: int,
    ) -> tuple[Solution, str, Solution, int]:
        parent_type, parent = self._neighborhood_parent_with_type(type_index)
        n = len(parent)
        if operator_index == 0:
            first = self._rng_native.randrange(1, n)
            city = parent[first]
            position = {value: index for index, value in enumerate(parent)}
            choices = [
                position[value]
                for value in self._motsp_candidate_cities[city]
                if position[value] > 0 and position[value] != first
            ]
            second = self._rng_native.choice(
                choices
                or [index for index in range(1, n) if index != first]
            )
            left, right = sorted((first, second))
            return (
                two_opt_at(parent, left, right),
                self._MOTSP_OPERATORS[0],
                parent,
                parent_type,
            )
        if operator_index == 1:
            source, target = self._rng_native.sample(range(1, n), 2)
            child = list(parent)
            city = child.pop(source)
            child.insert(target, city)
            return tuple(child), self._MOTSP_OPERATORS[1], parent, parent_type
        if n < 7:
            left, right = sorted(self._rng_native.sample(range(1, n), 2))
            return (
                two_opt_at(parent, left, right),
                self._MOTSP_OPERATORS[2],
                parent,
                parent_type,
            )
        first, second, third = sorted(self._rng_native.sample(range(1, n), 3))
        return (
            parent[:first]
            + tuple(reversed(parent[first:second]))
            + tuple(reversed(parent[second:third]))
            + parent[third:],
            self._MOTSP_OPERATORS[2],
            parent,
            parent_type,
        )

    def _mokp_native(
        self,
        type_index: int,
        operator_index: int,
        *,
        operator_call: int,
    ) -> tuple[Solution, str, Tuple[Solution, ...], Tuple[int, ...]]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP operator used for a non-MOKP problem.")
        if (
            isinstance(operator_call, bool)
            or not isinstance(operator_call, int)
            or operator_call < 0
            or operator_call % len(self._MOKP_OPERATORS) != operator_index
        ):
            raise ValueError("MOKP operator index disagrees with its scheduled call.")
        direction = self._direction(type_index)
        source_type, source = self._neighborhood_parent_with_type(type_index)
        generation_parents: Tuple[Solution, ...] = (source,)
        generation_parent_types: Tuple[int, ...] = (source_type,)
        densities = self._mokp_densities(direction)
        child = list(source)
        operator_label = self._MOKP_OPERATORS[operator_index]
        repair_refill = True
        if operator_index == 0:
            neighbors = self._type_neighbors[type_index]
            guide_type = neighbors[(self._native_calls_by_type[type_index]) % len(neighbors)]
            guide = self._require_solution(guide_type)
            generation_parents = (source, guide)
            generation_parent_types = (source_type, guide_type)
            child = [
                left if self._rng_native.random() < 0.5 else right
                for left, right in zip(source, guide)
            ]
            child[self._rng_native.randrange(problem.solution_size)] ^= 1
        elif operator_index == 1:
            # Alternate a single-bit add and a single-bit drop independently
            # for each reference type.  An unavailable scheduled sub-mode is
            # an explicitly labelled no-op; it is never replaced by the
            # opposite sub-mode or by an implicit two-bit swap.
            add_drop_call = self._mokp_add_drop_calls_by_type[type_index]
            self._mokp_add_drop_calls_by_type[type_index] += 1
            selected = [index for index, value in enumerate(child) if value]
            current_weight = sum(
                weight
                for value, weight in zip(child, problem.item_weights)
                if value
            )
            if add_drop_call % 2 == 0:
                feasible_additions = [
                    index
                    for index, value in enumerate(child)
                    if not value
                    and current_weight + problem.item_weights[index]
                    <= problem.capacity
                ]
                if feasible_additions:
                    chosen = max(
                        feasible_additions,
                        key=lambda index: (densities[index], -index),
                    )
                    child[chosen] = 1
                    operator_label = "mokp_add_repair_v21e3r1"
                else:
                    operator_label = (
                        "mokp_add_noop_no_feasible_item_v21e3r1"
                    )
            elif selected:
                chosen = min(
                    selected,
                    key=lambda index: (densities[index], index),
                )
                child[chosen] = 0
                operator_label = "mokp_drop_repair_v21e3r1"
            else:
                operator_label = "mokp_drop_noop_empty_solution_v21e3r1"
            repair_refill = False
        elif operator_index == 2:
            # A successful swap is an atomic one-out/one-in exchange.  If no
            # capacity-feasible pair exists, preserve the parent and expose an
            # auditable no-op label instead of claiming a swap occurred.
            selected = [index for index, value in enumerate(child) if value]
            unselected = [index for index, value in enumerate(child) if not value]
            current_weight = sum(
                weight
                for value, weight in zip(child, problem.item_weights)
                if value
            )
            feasible_pairs = [
                (removed, added)
                for removed in selected
                for added in unselected
                if current_weight
                - problem.item_weights[removed]
                + problem.item_weights[added]
                <= problem.capacity
            ]
            if feasible_pairs:
                removed, added = max(
                    feasible_pairs,
                    key=lambda pair: (
                        densities[pair[1]] - densities[pair[0]],
                        -pair[1],
                        pair[0],
                    ),
                )
                child[removed] = 0
                child[added] = 1
            else:
                operator_label = (
                    "mokp_swap_noop_no_feasible_exchange_v21e3r1"
                )
            repair_refill = False
        else:
            count = min(problem.solution_size, max(2, problem.solution_size // 20))
            for index in self._rng_native.sample(range(problem.solution_size), count):
                child[index] ^= 1
        return (
            self._mokp_repair(child, direction, refill=repair_refill),
            operator_label,
            generation_parents,
            generation_parent_types,
        )

    def _local_candidate(
        self,
        anchor: Solution,
        type_index: int,
        depth: int,
    ) -> tuple[Solution, str, dict[str, object] | None]:
        if (
            isinstance(self.problem, MultiObjectiveKnapsackInstance)
            and self.config.mokp_novelty_generation_policy
            == _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY
        ):
            return self._mokp_successor_novelty_candidate(
                anchor,
                type_index,
                origin="bounded_local_improvement",
            )
        proposal, operator = self._local_neighbor(anchor, type_index, depth)
        return proposal, operator, None

    def _mokp_successor_novelty_candidate(
        self,
        source: Solution,
        type_index: int,
        *,
        origin: str,
    ) -> tuple[Solution, str, dict[str, object]]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP successor novelty used for a non-MOKP problem.")
        if (
            self.config.mokp_novelty_generation_policy
            != _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY
        ):
            raise RuntimeError("MOKP successor novelty is not authorized by config.")
        if origin not in self._mokp_successor_novelty_calls_by_origin:
            raise ValueError("Unsupported MOKP successor novelty origin.")
        calls = self._mokp_successor_novelty_calls_by_origin[origin]
        call_ordinal = calls[type_index] + 1
        calls[type_index] = call_ordinal
        mode = ("add", "drop", "swap")[(call_ordinal - 1) % 3]
        direction = self._direction(type_index)
        densities = self._mokp_densities(direction)
        child = list(source)
        selected = [index for index, value in enumerate(child) if value]
        unselected = [index for index, value in enumerate(child) if not value]
        current_weight = sum(
            problem.item_weights[index] for index in selected
        )
        removed: tuple[int, ...] = ()
        added: tuple[int, ...] = ()
        noop_reason: str | None = None
        if mode == "add":
            feasible_additions = [
                index
                for index in unselected
                if current_weight + problem.item_weights[index]
                <= problem.capacity
            ]
            if feasible_additions:
                chosen = max(
                    feasible_additions,
                    key=lambda index: (densities[index], -index),
                )
                child[chosen] = 1
                added = (chosen,)
            else:
                noop_reason = "no_feasible_item"
        elif mode == "drop":
            if selected:
                chosen = min(
                    selected,
                    key=lambda index: (densities[index], index),
                )
                child[chosen] = 0
                removed = (chosen,)
            else:
                noop_reason = "empty_solution"
        else:
            feasible_pairs = [
                (removed_index, added_index)
                for removed_index in selected
                for added_index in unselected
                if current_weight
                - problem.item_weights[removed_index]
                + problem.item_weights[added_index]
                <= problem.capacity
            ]
            if feasible_pairs:
                removed_index, added_index = max(
                    feasible_pairs,
                    key=lambda pair: (
                        densities[pair[1]] - densities[pair[0]],
                        -pair[1],
                        pair[0],
                    ),
                )
                child[removed_index] = 0
                child[added_index] = 1
                removed = (removed_index,)
                added = (added_index,)
            else:
                noop_reason = "no_feasible_exchange"
        output = self._mokp_repair(child, direction, refill=False)
        problem.validate_solution(output)
        observed_removed = tuple(
            index
            for index, (left, right) in enumerate(zip(source, output))
            if left == 1 and right == 0
        )
        observed_added = tuple(
            index
            for index, (left, right) in enumerate(zip(source, output))
            if left == 0 and right == 1
        )
        if observed_removed != removed or observed_added != added:
            raise RuntimeError(
                "MOKP successor no-refill repair changed the frozen move support."
            )
        origin_label = (
            "local"
            if origin == "bounded_local_improvement"
            else "retry"
        )
        noop_label = "" if noop_reason is None else f"_noop_{noop_reason}"
        operator = (
            f"mokp_successor_{origin_label}_{mode}{noop_label}_"
            "no_refill_repair_development_v1"
        )
        witness: dict[str, object] = {
            "schema": "v21e3r1_mokp_successor_novelty_witness_v1",
            "policy": _SUCCESSOR_MOKP_NOVELTY_GENERATION_POLICY,
            "origin": origin,
            "origin_call_ordinal_by_type": call_ordinal,
            "rotation_mode": mode,
            "move_applied": noop_reason is None,
            "removed_item_indices": removed,
            "added_item_indices": added,
            "repair_refill": False,
            "rng_draws_consumed": 0,
        }
        return output, operator, witness

    def _local_neighbor(
        self,
        anchor: Solution,
        type_index: int,
        depth: int,
    ) -> tuple[Solution, str]:
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            n = len(anchor)
            first = 1 + ((type_index + depth) % (n - 1))
            choices = [index for index in range(1, n) if index != first]
            second = choices[(type_index * 7 + depth) % len(choices)]
            left, right = sorted((first, second))
            return (
                two_opt_at(anchor, left, right),
                "motsp_bounded_scalar_two_opt_improvement_v21e3",
            )
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            direction = self._direction(type_index)
            densities = self._mokp_densities(direction)
            child = list(anchor)
            selected = sorted(
                (index for index, value in enumerate(child) if value),
                key=lambda index: (densities[index], index),
            )
            unselected = sorted(
                (index for index, value in enumerate(child) if not value),
                key=lambda index: (-densities[index], index),
            )
            if selected and unselected:
                child[selected[(depth - 1) % len(selected)]] = 0
                child[unselected[(depth - 1) % len(unselected)]] = 1
            elif unselected:
                child[unselected[(depth - 1) % len(unselected)]] = 1
            return (
                self._mokp_repair(child, direction, refill=True),
                "mokp_bounded_scalar_swap_improvement_repair_v21e3",
            )
        return (
            self.problem.propose(anchor, self._rng_native),
            "generic_bounded_local_improvement_v21e3",
        )

    def _diversification_candidate(self, type_index: int) -> _GeneratedCandidate:
        self._diversification_call_count += 1
        source_type, source = self._neighborhood_parent_with_type(type_index)
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = list(source)
            count = min(len(child), max(2, len(child) // 20))
            for index in self._rng_diversification.sample(range(len(child)), count):
                child[index] ^= 1
            return _GeneratedCandidate(
                self._mokp_repair(child, self._direction(type_index), refill=True),
                "mokp_typed_multibit_diversification_repair_v21e3",
                "typed_diversification",
                (source,),
                (source_type,),
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            first, second = sorted(
                self._rng_diversification.sample(range(1, len(source)), 2)
            )
            child = list(two_opt_at(source, first, second))
            move_from, move_to = self._rng_diversification.sample(
                range(1, len(source)), 2
            )
            city = child.pop(move_from)
            child.insert(move_to, city)
            return _GeneratedCandidate(
                tuple(child),
                "motsp_typed_diversification_v21e3",
                "typed_diversification",
                (source,),
                (source_type,),
            )
        return _GeneratedCandidate(
            self.problem.propose(source, self._rng_diversification),
            "generic_typed_diversification_v21e3",
            "typed_diversification",
            (source,),
            (source_type,),
        )

    def _exchange_candidate(self, type_index: int) -> _GeneratedCandidate:
        self._exchange_call_count += 1
        neighbors = tuple(
            value for value in self._type_neighbors[type_index] if value != type_index
        )
        raw = self._rng_exchange.randrange(1 << 63)
        guide_type = neighbors[raw % len(neighbors)]
        parent = self._require_solution(type_index)
        guide = self._require_solution(guide_type)
        treatment = self.config.candidate_id == "C3"
        phase = (
            "neighbor_path_relinking"
            if treatment
            else "matched_exchange_control"
        )
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = list(parent)
            if treatment:
                differences = [
                    index
                    for index, (left, right) in enumerate(zip(parent, guide))
                    if left != right
                ]
                index = (
                    differences[(raw // max(1, len(neighbors))) % len(differences)]
                    if differences
                    else (raw // max(1, len(neighbors))) % len(child)
                )
                child[index] = guide[index] if differences else 1 - child[index]
                operator = "mokp_neighbor_path_relink_repair_v21e3"
            else:
                index = (raw // max(1, len(neighbors))) % len(child)
                child[index] ^= 1
                operator = "mokp_matched_exchange_control_repair_v21e3"
            return _GeneratedCandidate(
                self._mokp_repair(child, self._direction(type_index), refill=True),
                operator,
                phase,
                (parent, guide),
                (type_index, guide_type),
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            n = len(parent)
            rank = (raw // max(1, len(neighbors))) % (n - 1)
            if treatment:
                differences = [
                    index for index in range(1, n) if parent[index] != guide[index]
                ]
                position = differences[rank % len(differences)] if differences else 1 + rank
                source = parent.index(guide[position])
                child = list(parent)
                city = child.pop(source)
                child.insert(position, city)
                return _GeneratedCandidate(
                    tuple(child),
                    "motsp_neighbor_path_relink_v21e3",
                    phase,
                    (parent, guide),
                    (type_index, guide_type),
                )
            source = 1 + rank
            target = 1 + ((rank + max(1, n // 3)) % (n - 1))
            if source == target:
                target = 1 + (target % (n - 1))
            child = list(parent)
            city = child.pop(source)
            child.insert(target, city)
            return _GeneratedCandidate(
                tuple(child),
                "motsp_matched_exchange_control_v21e3",
                phase,
                (parent, guide),
                (type_index, guide_type),
            )
        return _GeneratedCandidate(
            self.problem.propose(parent, self._rng_exchange),
            (
                "generic_neighbor_path_relink_v21e3"
                if treatment
                else "generic_matched_exchange_control_v21e3"
            ),
            phase,
            (parent, guide),
            (type_index, guide_type),
        )

    def _retry_candidate(
        self,
        proposal: Solution,
        type_index: int,
        ordinal: int,
    ) -> Solution:
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = list(proposal)
            child[self._rng_retry.randrange(len(child))] ^= 1
            return self._mokp_repair(
                child,
                self._direction(type_index),
                refill=False,
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            left, right = sorted(self._rng_retry.sample(range(1, len(proposal)), 2))
            return two_opt_at(proposal, left, right)
        return self.problem.propose(proposal, self._rng_retry)

    def _fallback_candidate(self, type_index: int) -> Solution:
        candidate = self.problem.random_solution(self._rng_fallback)
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            return self._mokp_repair(
                candidate,
                self._direction(type_index),
                refill=False,
            )
        return candidate

    def _motsp_nearest_neighbor(
        self,
        direction: Sequence[float],
        variant: int,
    ) -> Solution:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveTSPProblemAdapter):
            raise TypeError("MOTSP construction used for a non-MOTSP problem.")
        remaining = set(range(1, problem.instance.num_cities))
        tour = [0]
        while remaining:
            current = tour[-1]
            following = min(
                remaining,
                key=lambda city: (
                    sum(
                        weight * matrix[current][city]
                        for weight, matrix in zip(
                            direction,
                            problem.instance.distance_matrices,
                        )
                    ),
                    city,
                ),
            )
            tour.append(following)
            remaining.remove(following)
        output = tuple(tour)
        if variant > 0:
            pairs = tuple(
                (left, right)
                for left in range(1, len(output) - 1)
                for right in range(left + 1, len(output))
            )
            left, right = pairs[(variant - 1) % len(pairs)]
            output = two_opt_at(output, left, right)
        problem.validate_solution(output)
        return output

    def _mokp_construction(
        self,
        direction: Sequence[float],
        variant: int,
    ) -> Solution:
        """Strong density construction plus a frozen, matched portfolio variant."""

        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP construction used for a non-MOKP problem.")
        base = self._mokp_repair(
            (0,) * problem.solution_size,
            direction,
            refill=True,
        )
        if variant == 0:
            return base
        densities = self._mokp_densities(direction)
        selected = sorted(
            (index for index, value in enumerate(base) if value),
            key=lambda index: (densities[index], index),
        )
        unselected = sorted(
            (index for index, value in enumerate(base) if not value),
            key=lambda index: (-densities[index], index),
        )
        child = list(base)
        if selected:
            child[selected[(variant - 1) % len(selected)]] = 0
        current_weight = sum(
            weight
            for value, weight in zip(child, problem.item_weights)
            if value
        )
        if unselected:
            start = (variant - 1) % len(unselected)
            for offset in range(len(unselected)):
                candidate = unselected[(start + offset) % len(unselected)]
                if current_weight + problem.item_weights[candidate] <= problem.capacity:
                    child[candidate] = 1
                    break
        output = self._mokp_repair(child, direction, refill=False)
        problem.validate_solution(output)
        return output

    def _mokp_densities(self, direction: Sequence[float]) -> tuple[float, ...]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP scoring used for a non-MOKP problem.")
        return mokp_directional_densities(problem, direction)

    def _mokp_repair(
        self,
        solution: Sequence[int],
        direction: Sequence[float],
        *,
        refill: bool,
    ) -> Solution:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP repair used for a non-MOKP problem.")
        return mokp_repair(problem, solution, direction, refill=refill)

    def _neighborhood_parent_with_type(
        self,
        type_index: int,
    ) -> tuple[int, Solution]:
        direction = self._direction(type_index)
        candidates = [
            (
                self._scalar(self._require_objective(index), direction),
                self._require_objective(index),
                self._require_solution(index),
                index,
            )
            for index in self._type_neighbors[type_index]
        ]
        chosen = min(candidates)
        return chosen[3], chosen[2]

    def _direction(self, type_index: int) -> Tuple[float, ...]:
        return (
            self._central_direction
            if self.config.candidate_id == "C0"
            else tuple(self.config.reference_directions[type_index])
        )

    def _scalar(
        self,
        objective: Sequence[float],
        direction: Sequence[float],
    ) -> float:
        normalized = tuple(
            (float(value) - lower) / (upper - lower)
            for value, lower, upper in zip(objective, self._lower, self._upper)
        )
        return max(weight * value for weight, value in zip(direction, normalized))

    def _build_type_neighbors(self) -> Tuple[Tuple[int, ...], ...]:
        directions = self.config.reference_directions
        return tuple(
            tuple(
                sorted(
                    range(len(directions)),
                    key=lambda other: (
                        sum(
                            (left - right) ** 2
                            for left, right in zip(directions[index], directions[other])
                        ),
                        other,
                    ),
                )[: min(self.config.neighborhood_size, len(directions))]
            )
            for index in range(len(directions))
        )

    def _build_motsp_candidates(self) -> Tuple[Tuple[int, ...], ...]:
        if not isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            return ()
        matrices = self.problem.instance.distance_matrices
        count = self.problem.instance.num_cities
        return tuple(
            tuple(
                sorted(
                    (other for other in range(count) if other != city),
                    key=lambda other: (
                        sum(matrix[city][other] for matrix in matrices),
                        other,
                    ),
                )[: min(8, count - 1)]
            )
            for city in range(count)
        )

    def _require_solution(self, type_index: int) -> Solution:
        value = self._solutions[type_index]
        if value is None:
            raise RuntimeError("Population type is not initialized.")
        return value

    def _require_objective(self, type_index: int) -> ObjectiveVector:
        value = self._objectives[type_index]
        if value is None:
            raise RuntimeError("Population objective is not initialized.")
        return value

    @staticmethod
    def _domain_seed(seed: int, domain: str) -> int:
        payload = f"v21e3|{int(seed)}|{domain}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "V21E3AttemptEvent",
    "V21E3CandidateId",
    "V21E3CandidateSpec",
    "V21E3EvaluationEvent",
    "V21E3HybridConfig",
    "V21E3RegionObservation",
    "V21E3RegionOccupancy",
    "V21E3RunResult",
    "V21E3ScheduleSlot",
    "V21E3TypedHybridParetoSearch",
    "v21e3_candidate_spec",
    "v21e3_schedule_slot",
]
