from __future__ import annotations

"""Independent typed annealed-MH replicas for direct endpoint certificates.

This is intentionally *not* an interacting particle system.  Each replica has
its own replayable PRNG and local state, performs no resampling, and exposes an
exact endpoint-cell observation against a hash-frozen rational manifest.  The
MH energy and log-acceptance calculation remain binary64 and are labelled as
such; exact dyadic edge sums do not imply an exact-real MH kernel.
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_dyadic_objective import (
    EXACT_EDGE_SUM_CONTRACT,
    DyadicObjectiveEncoding,
)
from .pareto_frozen_cells import (
    FrozenCellManifest,
    FrozenCellManifestError,
    OBJECTIVE_ARITHMETIC_V15,
    canonical_fraction_text,
    load_frozen_cell_manifest,
)
from .pareto_v15_context import V15CertificateContext
from .types import Tour


INDEPENDENT_REPLICA_ALGORITHM_ID_V15 = (
    "typed_annealed_independent_MH_replicas_v15"
)
INDEPENDENT_REPLICA_RESULT_SCHEMA_V15 = (
    "typed_annealed_independent_MH_replica_result_v15"
)
ACCEPTANCE_SEMANTICS_V15 = "binary64_log_mh_not_machine_exact_v1"
ENDPOINT_SUM_SEMANTICS_V15 = EXACT_EDGE_SUM_CONTRACT


class IndependentReplicaError(ValueError):
    """Raised when the independent-replica contract cannot be enforced."""


@dataclass(frozen=True)
class ReplicaTypeConfiguration:
    type_id: str
    reference_direction: tuple[float, ...]
    beta_schedule: tuple[float, ...]
    mutation_steps_by_stage: tuple[int, ...]
    replica_count: int
    chebyshev_rho: float = 0.05
    global_refresh_probability: float = 0.0

    def validate(self, *, dimension: int) -> None:
        if not self.type_id:
            raise IndependentReplicaError("type_id must be nonempty.")
        if len(self.reference_direction) != dimension:
            raise IndependentReplicaError(
                f"type {self.type_id!r} has the wrong reference dimension."
            )
        if (
            any(
                not math.isfinite(value) or value < 0.0
                for value in self.reference_direction
            )
            or not math.isclose(
                sum(self.reference_direction),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise IndependentReplicaError(
                "Reference directions must be finite simplex vectors."
            )
        if (
            len(self.beta_schedule) < 2
            or self.beta_schedule[0] != 0.0
            or any(
                not math.isfinite(value)
                for value in self.beta_schedule
            )
            or any(
                right <= left
                for left, right in zip(
                    self.beta_schedule,
                    self.beta_schedule[1:],
                )
            )
        ):
            raise IndependentReplicaError(
                "beta_schedule must start at zero and increase strictly."
            )
        if len(self.mutation_steps_by_stage) != len(self.beta_schedule) - 1:
            raise IndependentReplicaError(
                "mutation_steps_by_stage must have one entry per positive stage."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.mutation_steps_by_stage
        ):
            raise IndependentReplicaError(
                "Mutation counts must be nonnegative integers."
            )
        if (
            isinstance(self.replica_count, bool)
            or not isinstance(self.replica_count, int)
            or self.replica_count <= 0
        ):
            raise IndependentReplicaError("replica_count must be positive.")
        if not math.isfinite(self.chebyshev_rho) or self.chebyshev_rho <= 0.0:
            raise IndependentReplicaError("chebyshev_rho must be positive.")
        if (
            not math.isfinite(self.global_refresh_probability)
            or not 0.0 <= self.global_refresh_probability <= 1.0
        ):
            raise IndependentReplicaError(
                "global_refresh_probability must lie in [0, 1]."
            )


@dataclass(frozen=True)
class ReplicaEndpoint:
    type_id: str
    replica_index: int
    derived_seed: int
    tour: Tour
    exact_objective: tuple[str, ...]
    binary64_objective: tuple[float, ...]
    frozen_cell: tuple[int, ...]
    observable_cell_hit: bool
    evaluations: int
    accepted_mutations: int
    mutation_attempts: int


@dataclass(frozen=True)
class IndependentReplicaBatchResult:
    schema: str
    algorithm_id: str
    instance_sha256: str
    cell_manifest_sha256: str
    configuration_sha256: str
    context_sha256: str
    stream_role: str
    endpoints: tuple[ReplicaEndpoint, ...]
    hit_counts: tuple[tuple[tuple[int, ...], int], ...]
    exact_total_evaluations: int
    population_interaction_present: bool
    resampling_performed: bool
    probability_semantics: str
    acceptance_semantics: str
    endpoint_sum_semantics: str
    endpoint_classification_semantics: str

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hit_counts"] = [
            {"cell": list(cell), "count": count}
            for cell, count in self.hit_counts
        ]
        return payload


def _configuration_hash(
    configurations: Sequence[ReplicaTypeConfiguration],
) -> str:
    payload = [
        {
            "type_id": config.type_id,
            "reference_direction": [
                value.hex() for value in config.reference_direction
            ],
            "beta_schedule": [value.hex() for value in config.beta_schedule],
            "mutation_steps_by_stage": list(config.mutation_steps_by_stage),
            "replica_count": config.replica_count,
            "chebyshev_rho": config.chebyshev_rho.hex(),
            "global_refresh_probability": (
                config.global_refresh_probability.hex()
            ),
        }
        for config in configurations
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replica_configuration_sha256(
    configurations: Sequence[ReplicaTypeConfiguration],
) -> str:
    """Return the canonical v15 configuration digest."""

    resolved = tuple(configurations)
    if not resolved:
        raise IndependentReplicaError(
            "At least one type configuration is required."
        )
    return _configuration_hash(resolved)


def replica_stream_plan_sha256(
    configurations: Sequence[ReplicaTypeConfiguration],
    *,
    master_seed: int,
    stream_role: str,
    cell_manifest_sha256: str,
) -> str:
    """Bind the replay seed domain and frozen configuration for one stream."""

    if stream_role not in {"pilot", "confirm"}:
        raise IndependentReplicaError("stream_role must be 'pilot' or 'confirm'.")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise IndependentReplicaError("master_seed must be an integer.")
    if (
        not isinstance(cell_manifest_sha256, str)
        or len(cell_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cell_manifest_sha256)
    ):
        raise IndependentReplicaError(
            "cell_manifest_sha256 must be canonical lowercase SHA-256 text."
        )
    payload = {
        "schema": "typed_annealed_independent_mh_stream_plan_v15",
        "algorithm_id": INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
        "configuration_sha256": replica_configuration_sha256(configurations),
        "cell_manifest_sha256": cell_manifest_sha256,
        "master_seed": master_seed,
        "stream_role": stream_role,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _derive_seed(
    *,
    master_seed: int,
    stream_role: str,
    type_id: str,
    replica_index: int,
    manifest_sha256: str,
) -> int:
    payload = (
        f"pareto-independent-replica-v15\0{master_seed}\0{stream_role}\0"
        f"{type_id}\0{replica_index}\0{manifest_sha256}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _dyadic_encoding(
    instance: MultiObjectiveTSPInstance,
) -> DyadicObjectiveEncoding:
    try:
        return DyadicObjectiveEncoding.from_binary64_matrices(
            instance.distance_matrices
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise IndependentReplicaError(
            "Instance matrices cannot be bound to the dyadic edge-sum contract."
        ) from error


def _energy(
    objective: tuple[Fraction, ...],
    *,
    manifest: FrozenCellManifest,
    configuration: ReplicaTypeConfiguration,
) -> float:
    # Classification performs the exact fail-closed box check.  It is called
    # for every proposal as well as for the endpoint; no normalization clamp
    # exists in this path.
    manifest.classify(objective)
    normalized = tuple(
        (value - low) / (high - low)
        for value, low, high in zip(
            objective,
            manifest.lower,
            manifest.upper,
        )
    )
    weighted = tuple(
        weight * float(value)
        for weight, value in zip(
            configuration.reference_direction,
            normalized,
        )
    )
    return max(weighted) + configuration.chebyshev_rho * sum(weighted)


def _random_tour(rng: random.Random, city_count: int) -> Tour:
    suffix = list(range(1, city_count))
    rng.shuffle(suffix)
    return tuple([0, *suffix])


def _local_two_opt(
    rng: random.Random,
    tour: Tour,
) -> tuple[Tour, int, int]:
    left, right = sorted(rng.sample(range(1, len(tour)), 2))
    proposed = list(tour)
    proposed[left : right + 1] = reversed(proposed[left : right + 1])
    return tuple(proposed), left, right


def _run_one_replica(
    *,
    instance: MultiObjectiveTSPInstance,
    encoding: DyadicObjectiveEncoding,
    manifest: FrozenCellManifest,
    configuration: ReplicaTypeConfiguration,
    replica_index: int,
    seed: int,
) -> ReplicaEndpoint:
    rng = random.Random(seed)
    current_tour = _random_tour(rng, instance.num_cities)
    current_scaled = encoding.exact_tour_scaled_sums(current_tour)
    current_objective = encoding.scaled_as_fraction(current_scaled)
    evaluations = 1
    current_energy = _energy(
        current_objective,
        manifest=manifest,
        configuration=configuration,
    )
    accepted = 0
    attempts = 0
    for beta, steps in zip(
        configuration.beta_schedule[1:],
        configuration.mutation_steps_by_stage,
    ):
        for _ in range(steps):
            if rng.random() < configuration.global_refresh_probability:
                proposed_tour = _random_tour(rng, instance.num_cities)
                proposed_scaled = encoding.exact_tour_scaled_sums(
                    proposed_tour
                )
            else:
                proposed_tour, left, right = _local_two_opt(
                    rng,
                    current_tour,
                )
                proposed_scaled = encoding.update_two_opt_scaled(
                    current_tour,
                    current_scaled,
                    left,
                    right,
                )
            proposed_objective = encoding.scaled_as_fraction(proposed_scaled)
            evaluations += 1
            proposed_energy = _energy(
                proposed_objective,
                manifest=manifest,
                configuration=configuration,
            )
            log_alpha = min(0.0, -beta * (proposed_energy - current_energy))
            draw = rng.random()
            log_draw = -math.inf if draw == 0.0 else math.log(draw)
            attempts += 1
            if log_draw < log_alpha:
                current_tour = proposed_tour
                current_scaled = proposed_scaled
                current_objective = proposed_objective
                current_energy = proposed_energy
                accepted += 1
    cell = manifest.classify(current_objective)
    expected = 1 + sum(configuration.mutation_steps_by_stage)
    if evaluations != expected:
        raise RuntimeError("Independent replica evaluation ledger is inconsistent.")
    return ReplicaEndpoint(
        type_id=configuration.type_id,
        replica_index=replica_index,
        derived_seed=seed,
        tour=current_tour,
        exact_objective=tuple(
            canonical_fraction_text(value) for value in current_objective
        ),
        binary64_objective=tuple(float(value) for value in current_objective),
        frozen_cell=cell,
        observable_cell_hit=manifest.is_observable(cell),
        evaluations=evaluations,
        accepted_mutations=accepted,
        mutation_attempts=attempts,
    )


def run_independent_replica_batch(
    instance: MultiObjectiveTSPInstance,
    *,
    cell_manifest_path: str | Path,
    certificate_context: V15CertificateContext,
    configurations: Sequence[ReplicaTypeConfiguration],
    master_seed: int,
    stream_role: str,
) -> IndependentReplicaBatchResult:
    """Run mechanically isolated typed annealed-MH endpoint replicas.

    The function accepts no free ``declared_cells`` argument: the full
    partition and observable family come exclusively from the hash-verified
    manifest.
    """

    if stream_role not in {"pilot", "confirm"}:
        raise IndependentReplicaError("stream_role must be 'pilot' or 'confirm'.")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise IndependentReplicaError("master_seed must be an integer.")
    manifest = load_frozen_cell_manifest(
        cell_manifest_path,
        expected_sha256=certificate_context.cell_manifest_sha256,
    )
    if manifest.dimension != instance.num_objectives:
        raise IndependentReplicaError(
            "Cell manifest and instance objective dimensions differ."
        )
    resolved = tuple(configurations)
    if not resolved:
        raise IndependentReplicaError("At least one type configuration is required.")
    if len({config.type_id for config in resolved}) != len(resolved):
        raise IndependentReplicaError("Replica type IDs must be unique.")
    for config in resolved:
        config.validate(dimension=instance.num_objectives)
    current_instance_sha256 = instance_sha256(instance)
    if certificate_context.instance_sha256 != current_instance_sha256:
        raise IndependentReplicaError(
            "Certificate context instance hash does not match the runner input."
        )
    configuration_sha256 = replica_configuration_sha256(resolved)
    if certificate_context.configuration_sha256 != configuration_sha256:
        raise IndependentReplicaError(
            "Certificate context configuration hash does not match."
        )
    stream_plan_sha256 = replica_stream_plan_sha256(
        resolved,
        master_seed=master_seed,
        stream_role=stream_role,
        cell_manifest_sha256=manifest.raw_sha256,
    )
    expected_plan_sha256 = (
        certificate_context.pilot_plan_sha256
        if stream_role == "pilot"
        else certificate_context.confirm_plan_sha256
    )
    if stream_plan_sha256 != expected_plan_sha256:
        raise IndependentReplicaError(
            "Certificate context stream-plan hash does not match."
        )
    encoding = _dyadic_encoding(instance)
    endpoints = []
    for config in resolved:
        for replica_index in range(config.replica_count):
            seed = _derive_seed(
                master_seed=master_seed,
                stream_role=stream_role,
                type_id=config.type_id,
                replica_index=replica_index,
                manifest_sha256=manifest.raw_sha256,
            )
            endpoints.append(
                _run_one_replica(
                    instance=instance,
                    encoding=encoding,
                    manifest=manifest,
                    configuration=config,
                    replica_index=replica_index,
                    seed=seed,
                )
            )
    hit_counts: dict[tuple[int, ...], int] = {
        cell: 0 for cell in manifest.observable_cells
    }
    for endpoint in endpoints:
        if endpoint.frozen_cell in hit_counts:
            hit_counts[endpoint.frozen_cell] += 1
    expected_total = sum(
        config.replica_count
        * (1 + sum(config.mutation_steps_by_stage))
        for config in resolved
    )
    observed_total = sum(endpoint.evaluations for endpoint in endpoints)
    if observed_total != expected_total:
        raise RuntimeError("Batch evaluation ledger is inconsistent.")
    return IndependentReplicaBatchResult(
        schema=INDEPENDENT_REPLICA_RESULT_SCHEMA_V15,
        algorithm_id=INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
        instance_sha256=current_instance_sha256,
        cell_manifest_sha256=manifest.raw_sha256,
        configuration_sha256=configuration_sha256,
        context_sha256=certificate_context.context_sha256,
        stream_role=stream_role,
        endpoints=tuple(endpoints),
        hit_counts=tuple(sorted(hit_counts.items())),
        exact_total_evaluations=observed_total,
        population_interaction_present=False,
        resampling_performed=False,
        probability_semantics=manifest.probability_semantics,
        acceptance_semantics=ACCEPTANCE_SEMANTICS_V15,
        endpoint_sum_semantics=ENDPOINT_SUM_SEMANTICS_V15,
        endpoint_classification_semantics=OBJECTIVE_ARITHMETIC_V15,
    )


__all__ = [
    "ACCEPTANCE_SEMANTICS_V15",
    "ENDPOINT_SUM_SEMANTICS_V15",
    "INDEPENDENT_REPLICA_ALGORITHM_ID_V15",
    "INDEPENDENT_REPLICA_RESULT_SCHEMA_V15",
    "IndependentReplicaBatchResult",
    "IndependentReplicaError",
    "ReplicaEndpoint",
    "ReplicaTypeConfiguration",
    "replica_configuration_sha256",
    "replica_stream_plan_sha256",
    "run_independent_replica_batch",
]
