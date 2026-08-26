from __future__ import annotations

"""Execution-only contracts for Pareto-SMC seeds and checkpoint boundaries.

This module deliberately does not define a metric certificate or a
fixed-reference specification.  It binds random streams to a predeclared
execution domain and verifies, from the executor ledger, whether a diagnostic
was emitted after a complete sweep over every reference type.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple


DOMAIN_SEPARATED_SEED_SCHEMA_V1 = "pareto_smc_domain_separated_seed_v1"
FULL_TYPE_SWEEP_CHECKPOINT_SCHEMA_V1 = (
    "pareto_smc_full_type_sweep_checkpoint_v1"
)
PARETO_SMC_V13_ALGORITHM_ROLE = "pareto-smc-pilot-confirm"
PARETO_SMC_V13_ALGORITHM_VERSION = "v13"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CASE_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DOMAIN_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")
_STREAM_ROLES = frozenset({"search", "pilot", "confirm", "memory-replay"})
_MAX_PAIRED_SEED = (1 << 64) - 1


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_canonical_token(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str],
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} is not a canonical execution-domain token.")
    return value


@dataclass(frozen=True)
class DomainSeparatedSeed:
    schema: str
    case_identity: str
    instance_sha256: str
    paired_seed: int
    algorithm_role: str
    algorithm_version: str
    stream_role: str
    derivation_sha256: str
    seed: int

    def payload(self) -> dict[str, object]:
        """Return the canonical preimage fields, excluding digest and seed."""

        return {
            "schema": self.schema,
            "case_identity": self.case_identity,
            "instance_sha256": self.instance_sha256,
            "paired_seed": self.paired_seed,
            "algorithm_role": self.algorithm_role,
            "algorithm_version": self.algorithm_version,
            "stream_role": self.stream_role,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self.payload(),
            "derivation_sha256": self.derivation_sha256,
            "derived_seed": self.seed,
            "seed_integer_contract": "unsigned_big_endian_sha256_digest",
        }


def derive_domain_separated_seed(
    *,
    case_identity: str,
    instance_sha256: str,
    paired_seed: int,
    algorithm_role: str,
    algorithm_version: str,
    stream_role: str,
    schema: str = DOMAIN_SEPARATED_SEED_SCHEMA_V1,
) -> DomainSeparatedSeed:
    """Derive one deterministic SHA-256 random-stream seed.

    ``paired_seed`` remains the matched experimental grouping label.  The
    actual PRNG seed changes with case identity, instance content, algorithm
    role/version, or stream role.  The full 256-bit digest is used as the
    Python seed; no truncation-induced collision is introduced here.
    """

    if schema != DOMAIN_SEPARATED_SEED_SCHEMA_V1:
        raise ValueError(
            "schema must be the supported domain-separated seed schema."
        )
    canonical_case = _require_canonical_token(
        case_identity,
        field="case_identity",
        pattern=_CASE_IDENTITY_PATTERN,
    )
    if (
        not isinstance(instance_sha256, str)
        or _SHA256_PATTERN.fullmatch(instance_sha256) is None
    ):
        raise ValueError("instance_sha256 must be a lowercase SHA-256 digest.")
    if (
        isinstance(paired_seed, bool)
        or not isinstance(paired_seed, int)
        or paired_seed < 0
        or paired_seed > _MAX_PAIRED_SEED
    ):
        raise ValueError("paired_seed must be an unsigned 64-bit integer.")
    canonical_algorithm = _require_canonical_token(
        algorithm_role,
        field="algorithm_role",
        pattern=_DOMAIN_TOKEN_PATTERN,
    )
    canonical_version = _require_canonical_token(
        algorithm_version,
        field="algorithm_version",
        pattern=_VERSION_PATTERN,
    )
    canonical_stream = _require_canonical_token(
        stream_role,
        field="stream_role",
        pattern=_DOMAIN_TOKEN_PATTERN,
    )
    if canonical_stream not in _STREAM_ROLES:
        raise ValueError(
            "stream_role must be one of: "
            + ", ".join(sorted(_STREAM_ROLES))
            + "."
        )
    payload = {
        "schema": schema,
        "case_identity": canonical_case,
        "instance_sha256": instance_sha256,
        "paired_seed": paired_seed,
        "algorithm_role": canonical_algorithm,
        "algorithm_version": canonical_version,
        "stream_role": canonical_stream,
    }
    digest = _canonical_sha256(payload)
    return DomainSeparatedSeed(
        **payload,
        derivation_sha256=digest,
        seed=int.from_bytes(bytes.fromhex(digest), "big", signed=False),
    )


def verify_domain_separated_seed(seed: DomainSeparatedSeed) -> None:
    """Fail closed if a seed object was forged or its preimage was mutated."""

    if not isinstance(seed, DomainSeparatedSeed):
        raise TypeError("seed must be a DomainSeparatedSeed.")
    expected = derive_domain_separated_seed(**seed.payload())
    if (
        expected.derivation_sha256 != seed.derivation_sha256
        or expected.seed != seed.seed
    ):
        raise ValueError(
            "The domain-separated seed does not match its canonical preimage."
        )


@dataclass(frozen=True)
class FullTypeSweepCheckpointVerification:
    schema: str
    gate: str
    ledger_gate: str
    verified_boundaries: Tuple[int, ...]
    requested_checkpoints: Tuple[int, ...]
    non_boundary_checkpoints: Tuple[int, ...]
    missing_diagnostic_checkpoints: Tuple[int, ...]
    reasons: Tuple[str, ...]
    stage_ledger_sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "formal_full_type_sweep_checkpoint_schema": self.schema,
            "formal_full_type_sweep_checkpoint_gate": self.gate,
            "formal_full_type_sweep_ledger_gate": self.ledger_gate,
            "verified_full_type_sweep_boundaries": self.verified_boundaries,
            "requested_full_type_sweep_checkpoints": self.requested_checkpoints,
            "non_boundary_full_type_sweep_checkpoints": (
                self.non_boundary_checkpoints
            ),
            "missing_full_type_sweep_diagnostic_checkpoints": (
                self.missing_diagnostic_checkpoints
            ),
            "formal_full_type_sweep_checkpoint_reasons": self.reasons,
            "formal_full_type_sweep_stage_ledger_sha256": (
                self.stage_ledger_sha256
            ),
            "formal_full_type_sweep_checkpoint_contract": (
                "stage_ledger_complete_reference_type_coverage_and_exact_"
                "evaluation_accounting_v1"
            ),
        }


def _strict_nonnegative_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def verify_full_type_sweep_checkpoints(
    *,
    stage_ledger: Sequence[Mapping[str, object]],
    num_reference_types: int,
    particles_per_reference: int,
    total_evaluations: int,
    checkpoint_period: Optional[int],
    diagnostic_iterations: Sequence[int],
) -> FullTypeSweepCheckpointVerification:
    """Verify exact stage-end full-type-sweep checkpoint semantics.

    The current executor processes all mutations of one reference type before
    advancing to the next type.  Consequently, without changing execution
    order, only an initial-population endpoint or a completed stage endpoint
    can be certified as a full-type-sweep boundary.  A requested checkpoint
    inside a type or stage is reported as ``FAIL`` rather than silently
    discarded.
    """

    for field, value in (
        ("num_reference_types", num_reference_types),
        ("particles_per_reference", particles_per_reference),
        ("total_evaluations", total_evaluations),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{field} must be a positive integer.")
    if checkpoint_period is not None and (
        isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
        or checkpoint_period > total_evaluations
    ):
        raise ValueError(
            "checkpoint_period must be a positive integer no larger than "
            "total_evaluations."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in diagnostic_iterations
    ):
        raise ValueError(
            "diagnostic_iterations must contain nonnegative integers."
        )

    reasons = []
    boundaries = []
    previous_end = 0
    ledger_evaluation_sum = 0
    expected_reference_indices = tuple(range(num_reference_types))
    if not stage_ledger:
        reasons.append("EMPTY_STAGE_LEDGER")

    for position, raw_stage in enumerate(stage_ledger):
        if not isinstance(raw_stage, Mapping):
            reasons.append(f"STAGE_{position}_NOT_MAPPING")
            continue
        stage_index = _strict_nonnegative_int(raw_stage.get("stage_index"))
        evaluation_start = _strict_nonnegative_int(
            raw_stage.get("evaluation_start")
        )
        evaluation_end = _strict_nonnegative_int(
            raw_stage.get("evaluation_end")
        )
        stage_evaluations = _strict_nonnegative_int(
            raw_stage.get("evaluations")
        )
        if stage_index != position:
            reasons.append(f"STAGE_{position}_INDEX_MISMATCH")
        if evaluation_start is None or evaluation_end is None:
            reasons.append(f"STAGE_{position}_INVALID_EVALUATION_RANGE")
        elif evaluation_start != previous_end:
            reasons.append(f"STAGE_{position}_NONCONTIGUOUS_EVALUATION_START")
        elif evaluation_end < evaluation_start:
            reasons.append(f"STAGE_{position}_NEGATIVE_EVALUATION_RANGE")
        if (
            evaluation_start is not None
            and evaluation_end is not None
            and stage_evaluations != evaluation_end - evaluation_start
        ):
            reasons.append(f"STAGE_{position}_EVALUATION_COUNT_MISMATCH")

        references = raw_stage.get("references")
        reference_attempt_sum = 0
        if not isinstance(references, (list, tuple)):
            reasons.append(f"STAGE_{position}_REFERENCES_NOT_SEQUENCE")
        else:
            reference_indices = []
            for reference_position, reference in enumerate(references):
                if not isinstance(reference, Mapping):
                    reasons.append(
                        f"STAGE_{position}_REFERENCE_{reference_position}_NOT_MAPPING"
                    )
                    continue
                reference_index = _strict_nonnegative_int(
                    reference.get("reference_index")
                )
                if reference_index is None:
                    reasons.append(
                        f"STAGE_{position}_REFERENCE_{reference_position}_INVALID_INDEX"
                    )
                else:
                    reference_indices.append(reference_index)
                if position > 0:
                    attempts = _strict_nonnegative_int(
                        reference.get("mutation_attempts")
                    )
                    if attempts is None:
                        reasons.append(
                            f"STAGE_{position}_REFERENCE_"
                            f"{reference_position}_INVALID_MUTATION_ATTEMPTS"
                        )
                    else:
                        reference_attempt_sum += attempts
            if tuple(reference_indices) != expected_reference_indices:
                reasons.append(
                    f"STAGE_{position}_REFERENCE_TYPE_COVERAGE_MISMATCH"
                )
        if position == 0:
            expected_initial = (
                num_reference_types * particles_per_reference
            )
            if stage_evaluations != expected_initial:
                reasons.append("INITIAL_STAGE_PARTICLE_EVALUATION_MISMATCH")
        elif stage_evaluations != reference_attempt_sum:
            reasons.append(f"STAGE_{position}_MUTATION_EVALUATION_SUM_MISMATCH")

        if stage_evaluations is not None:
            ledger_evaluation_sum += stage_evaluations
        if evaluation_end is not None:
            boundaries.append(evaluation_end)
            previous_end = evaluation_end

    if previous_end != total_evaluations:
        reasons.append("FINAL_STAGE_ENDPOINT_MISMATCH")
    if ledger_evaluation_sum != total_evaluations:
        reasons.append("STAGE_LEDGER_TOTAL_EVALUATION_MISMATCH")
    ledger_gate = "PASS" if not reasons else "FAIL"

    requested = (
        ()
        if checkpoint_period is None
        else tuple(
            range(
                checkpoint_period,
                total_evaluations,
                checkpoint_period,
            )
        )
        + (total_evaluations,)
    )
    boundary_set = set(boundaries) if ledger_gate == "PASS" else set()
    diagnostic_set = set(diagnostic_iterations)
    non_boundary = tuple(
        checkpoint for checkpoint in requested if checkpoint not in boundary_set
    )
    missing_diagnostics = tuple(
        checkpoint
        for checkpoint in requested
        if checkpoint not in diagnostic_set
    )
    checkpoint_reasons = list(reasons)
    if non_boundary:
        checkpoint_reasons.append("REQUESTED_CHECKPOINT_NOT_FULL_TYPE_SWEEP_BOUNDARY")
    if missing_diagnostics:
        checkpoint_reasons.append("REQUESTED_CHECKPOINT_DIAGNOSTIC_MISSING")
    if checkpoint_period is None:
        gate = "NOT_RUN" if ledger_gate == "PASS" else "FAIL"
    else:
        gate = "PASS" if not checkpoint_reasons else "FAIL"

    try:
        ledger_hash = _canonical_sha256(stage_ledger)
    except (TypeError, ValueError):
        ledger_hash = ""
        if "STAGE_LEDGER_NOT_CANONICAL_JSON" not in checkpoint_reasons:
            checkpoint_reasons.append("STAGE_LEDGER_NOT_CANONICAL_JSON")
        ledger_gate = "FAIL"
        gate = "FAIL"

    return FullTypeSweepCheckpointVerification(
        schema=FULL_TYPE_SWEEP_CHECKPOINT_SCHEMA_V1,
        gate=gate,
        ledger_gate=ledger_gate,
        verified_boundaries=tuple(boundaries) if ledger_gate == "PASS" else (),
        requested_checkpoints=requested,
        non_boundary_checkpoints=non_boundary,
        missing_diagnostic_checkpoints=missing_diagnostics,
        reasons=tuple(checkpoint_reasons),
        stage_ledger_sha256=ledger_hash,
    )
