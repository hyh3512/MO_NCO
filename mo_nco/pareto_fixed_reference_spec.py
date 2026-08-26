from __future__ import annotations

"""Fail-closed specification for fixed-reference Pareto-SMC certificates."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_bounds import nondominated_points
from .types import ObjectiveVector, Tour


FIXED_REFERENCE_SPEC_SCHEMA_V1 = (
    "pareto_smc_fixed_reference_certificate_spec_v1"
)
FIXED_REFERENCE_SPEC_SCHEMA_V2 = (
    "pareto_smc_fixed_reference_certificate_spec_v2"
)
FIXED_REFERENCE_SPEC_SCHEMA = FIXED_REFERENCE_SPEC_SCHEMA_V1


@dataclass(frozen=True)
class FixedReferenceCertificateSpecification:
    path: Path
    sha256: str
    instance_sha256: str
    pareto_smc_specification_sha256: str
    reference_source: str
    reference_artifact_sha256: str
    reference_objectives: Tuple[ObjectiveVector, ...]
    pilot_seed: int
    confirm_seed: int
    pilot_failure_budget: float
    confirm_failure_budget: float
    igd_p: float
    hv_reference: ObjectiveVector
    max_igd_bound: float
    max_hv_deficit_bound: float
    schema: str = FIXED_REFERENCE_SPEC_SCHEMA_V1
    reference_witnesses: Tuple[
        Tuple[Tour, ObjectiveVector],
        ...,
    ] = ()
    reference_witness_payload_sha256: str | None = None
    reference_feasibility_verified_by_runtime: bool = False
    reference_witness_max_abs_error: float = math.inf
    reference_witness_equivalence_contract: str = (
        "not_runtime_verified"
    )
    certified_archive_policy: str | None = None
    certified_archive_max_size: int | None = None
    seed_pairs: Tuple[Tuple[int, int, int], ...] = ()

    def stream_seeds(self, run_seed: int) -> Tuple[int, int]:
        if self.seed_pairs:
            matches = [
                (pilot, confirm)
                for declared, pilot, confirm in self.seed_pairs
                if declared == run_seed
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"No unique predeclared pilot-confirm pair for run seed "
                    f"{run_seed}."
                )
            return matches[0]
        if run_seed != 0:
            raise ValueError(
                "A single-pair certificate specification is bound to run "
                "seed 0; use streams.seed_pairs for a multi-seed study."
            )
        return self.pilot_seed, self.confirm_seed


@dataclass(frozen=True)
class ResolvedFixedReferenceSpecification:
    manifest_path: Path
    manifest_sha256: str
    specification_path: Path
    specification_sha256: str


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON field is forbidden: {key!r}."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number.")
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result < 1.0):
        raise ValueError(f"{label} must lie in (0, 1).")
    return result


def _finite_vector(
    value: object,
    *,
    dimension: int,
    label: str,
) -> ObjectiveVector:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{label} must be an array of length {dimension}.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError(f"{label} must contain JSON numbers.")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values.")
    return result


def resolve_fixed_reference_certificate_specification(
    path: str | Path,
    *,
    expected_instance_sha256: str,
    expected_pareto_smc_specification_sha256: str,
) -> ResolvedFixedReferenceSpecification:
    """Resolve one case-specific v2 specification from a frozen manifest."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"Fixed-reference manifest is missing: {resolved}"
        )
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "Fixed-reference manifest is not valid strict UTF-8 JSON: "
            f"{error}"
        ) from error
    root = _mapping(payload, "fixed-reference manifest")
    _exact_keys(
        root,
        {
            "schema",
            "pareto_smc_specification_sha256",
            "cases",
        },
        "fixed-reference manifest",
    )
    if root.get("schema") != "pareto_smc_fixed_reference_manifest_v1":
        raise ValueError("Unexpected fixed-reference manifest schema.")
    declared_smc_hash = _sha256(
        root.get("pareto_smc_specification_sha256"),
        "pareto_smc_specification_sha256",
    )
    if declared_smc_hash != expected_pareto_smc_specification_sha256:
        raise ValueError(
            "Fixed-reference manifest is bound to a different Pareto-SMC "
            "specification."
        )
    expected_instance_hash = _sha256(
        expected_instance_sha256,
        "expected_instance_sha256",
    )
    cases = _mapping(root.get("cases"), "fixed-reference manifest cases")
    for instance_hash, raw_entry in cases.items():
        _sha256(instance_hash, "fixed-reference manifest case key")
        entry = _mapping(
            raw_entry,
            f"fixed-reference manifest case {instance_hash}",
        )
        _exact_keys(
            entry,
            {"path", "sha256"},
            f"fixed-reference manifest case {instance_hash}",
        )
        _sha256(
            entry.get("sha256"),
            f"fixed-reference manifest case {instance_hash} SHA-256",
        )
        if not isinstance(entry.get("path"), str) or not entry.get("path"):
            raise ValueError(
                "Fixed-reference manifest paths must be nonempty strings."
            )
    if expected_instance_hash not in cases:
        raise ValueError(
            "Fixed-reference manifest has no certificate specification for "
            f"instance {expected_instance_hash}."
        )
    selected = cases[expected_instance_hash]
    specification_path = Path(str(selected["path"])).expanduser()
    if not specification_path.is_absolute():
        specification_path = resolved.parent / specification_path
    specification_path = specification_path.resolve()
    if not specification_path.is_file():
        raise ValueError(
            "Resolved fixed-reference certificate specification is missing: "
            f"{specification_path}"
        )
    expected_specification_hash = str(selected["sha256"])
    observed_specification_hash = hashlib.sha256(
        specification_path.read_bytes()
    ).hexdigest()
    if observed_specification_hash != expected_specification_hash:
        raise ValueError(
            "Resolved fixed-reference certificate specification hash "
            "mismatch."
        )
    return ResolvedFixedReferenceSpecification(
        manifest_path=resolved,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        specification_path=specification_path,
        specification_sha256=expected_specification_hash,
    )


def load_fixed_reference_certificate_specification(
    path: str | Path,
    *,
    objective_dimension: int,
    instance: MultiObjectiveTSPInstance | None = None,
) -> FixedReferenceCertificateSpecification:
    """Load the artifact that must be frozen before pilot and confirm."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"Fixed-reference certificate specification is missing: {resolved}"
        )
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "Fixed-reference certificate specification is not valid strict "
            f"UTF-8 JSON: {error}"
        ) from error
    root = _mapping(payload, "fixed-reference certificate specification")
    schema = root.get("schema")
    expected_root_keys = {
        "schema",
        "instance_sha256",
        "pareto_smc_specification_sha256",
        "reference_front",
        "streams",
        "failure_budgets",
        "metrics",
    }
    if schema == FIXED_REFERENCE_SPEC_SCHEMA_V2:
        expected_root_keys.add("certified_archive")
    _exact_keys(
        root,
        expected_root_keys,
        "fixed-reference certificate specification",
    )
    if schema not in {
        FIXED_REFERENCE_SPEC_SCHEMA_V1,
        FIXED_REFERENCE_SPEC_SCHEMA_V2,
    }:
        raise ValueError(
            "Unexpected fixed-reference certificate specification schema."
        )
    instance_hash = _sha256(
        root.get("instance_sha256"),
        "instance_sha256",
    )
    smc_hash = _sha256(
        root.get("pareto_smc_specification_sha256"),
        "pareto_smc_specification_sha256",
    )

    front = _mapping(root.get("reference_front"), "reference_front")
    front_keys = {"source", "artifact_sha256", "objectives"}
    if schema == FIXED_REFERENCE_SPEC_SCHEMA_V2:
        front_keys.update({"witnesses", "witness_payload_sha256"})
    _exact_keys(front, front_keys, "reference_front")
    source = front.get("source")
    if source not in {
        "independent_exact_solver",
        "frozen_external_archive",
        "public_benchmark_reference",
    }:
        raise ValueError(
            "reference_front.source is not an allowed independent source."
        )
    artifact_hash = _sha256(
        front.get("artifact_sha256"),
        "reference_front.artifact_sha256",
    )
    objectives_raw = front.get("objectives")
    if not isinstance(objectives_raw, list) or not objectives_raw:
        raise ValueError(
            "reference_front.objectives must be a nonempty array."
        )
    objectives = tuple(
        _finite_vector(
            point,
            dimension=objective_dimension,
            label=f"reference_front.objectives[{index}]",
        )
        for index, point in enumerate(objectives_raw)
    )
    canonical = tuple(sorted(objectives))
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            "reference_front.objectives must contain unique points."
        )
    if set(nondominated_points(canonical)) != set(canonical):
        raise ValueError(
            "reference_front.objectives must be mutually nondominated."
        )
    canonical_bytes = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical_bytes).hexdigest() != artifact_hash:
        raise ValueError(
            "reference_front.artifact_sha256 does not hash the canonical "
            "sorted objective payload."
        )

    reference_witnesses: Tuple[
        Tuple[Tour, ObjectiveVector],
        ...,
    ] = ()
    witness_payload_hash: str | None = None
    feasibility_verified = False
    witness_max_abs_error = math.inf
    witness_equivalence_contract = "not_runtime_verified"
    certified_archive_policy: str | None = None
    certified_archive_max_size: int | None = None
    if schema == FIXED_REFERENCE_SPEC_SCHEMA_V2:
        if instance is None:
            raise ValueError(
                "The v2 fixed-reference specification requires the bound "
                "instance for witness verification."
            )
        if instance.num_objectives != objective_dimension:
            raise ValueError(
                "The witness instance objective dimension does not match."
            )
        if instance_sha256(instance) != instance_hash:
            raise ValueError(
                "The witness instance hash does not match instance_sha256."
            )
        witnesses_raw = front.get("witnesses")
        if not isinstance(witnesses_raw, list) or not witnesses_raw:
            raise ValueError(
                "reference_front.witnesses must be a nonempty array."
            )
        parsed_witnesses = []
        maximum_error = 0.0
        exact_witness_equality = bool(
            getattr(
                instance,
                "exact_two_opt_delta_in_binary64",
                False,
            )
        )
        witness_equivalence_contract = (
            "exact_binary64_integer_objective_equality_v1"
            if exact_witness_equality
            else "floating_objective_rel1e-12_abs1e-12_v1"
        )
        for index, raw_witness in enumerate(witnesses_raw):
            witness = _mapping(
                raw_witness,
                f"reference_front.witnesses[{index}]",
            )
            _exact_keys(
                witness,
                {"tour", "objectives"},
                f"reference_front.witnesses[{index}]",
            )
            tour_raw = witness.get("tour")
            if not isinstance(tour_raw, list):
                raise ValueError("Reference witness tour must be an array.")
            tour = tuple(tour_raw)
            if any(
                isinstance(city, bool) or not isinstance(city, int)
                for city in tour
            ):
                raise ValueError(
                    "Reference witness tours must contain integer cities."
                )
            instance.validate_tour(tour)
            witness_objectives = _finite_vector(
                witness.get("objectives"),
                dimension=objective_dimension,
                label=f"reference_front.witnesses[{index}].objectives",
            )
            locally_evaluated = instance.evaluate(tour)
            errors = tuple(
                abs(observed - expected)
                for observed, expected in zip(
                    witness_objectives,
                    locally_evaluated,
                )
            )
            maximum_error = max(maximum_error, *errors)
            if any(
                (
                    observed != expected
                    if exact_witness_equality
                    else not math.isclose(
                        observed,
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                for observed, expected in zip(
                    witness_objectives,
                    locally_evaluated,
                )
            ):
                raise ValueError(
                    "Reference witness objective does not match local "
                    "full-tour evaluation."
                )
            parsed_witnesses.append((tour, witness_objectives))
        canonical_witnesses = tuple(
            sorted(
                parsed_witnesses,
                key=lambda record: (record[1], record[0]),
            )
        )
        if tuple(record[1] for record in canonical_witnesses) != canonical:
            raise ValueError(
                "Reference witnesses must bind exactly one feasible tour to "
                "each canonical reference objective."
            )
        canonical_witness_payload = tuple(
            {
                "tour": record[0],
                "objectives": record[1],
            }
            for record in canonical_witnesses
        )
        witness_payload_hash = _sha256(
            front.get("witness_payload_sha256"),
            "reference_front.witness_payload_sha256",
        )
        witness_bytes = json.dumps(
            canonical_witness_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(witness_bytes).hexdigest() != witness_payload_hash:
            raise ValueError(
                "reference_front.witness_payload_sha256 does not hash the "
                "canonical witness payload."
            )
        reference_witnesses = canonical_witnesses
        # The v2 metric theorem treats every frozen reference objective as an
        # actually feasible objective vector.  A tolerance-only match is useful
        # for diagnostics, but v2 does not inflate its IGD/HV bounds by that
        # tolerance.  Consequently only bit-exact local recomputation may open
        # the formal feasibility gate.  A future schema may admit approximate
        # witnesses only if it binds and propagates an explicit error vector.
        feasibility_verified = maximum_error == 0.0
        if not feasibility_verified:
            witness_equivalence_contract = (
                "floating_objective_rel1e-12_abs1e-12_diagnostic_only_"
                "exact_zero_error_required_for_formal_v2"
            )
        witness_max_abs_error = maximum_error
        certified_archive = _mapping(
            root.get("certified_archive"),
            "certified_archive",
        )
        _exact_keys(
            certified_archive,
            {"policy", "max_size"},
            "certified_archive",
        )
        certified_archive_policy_raw = certified_archive.get("policy")
        if (
            certified_archive_policy_raw
            != "deterministic_reference_coverage_v1"
        ):
            raise ValueError(
                "certified_archive.policy must be "
                "deterministic_reference_coverage_v1."
            )
        max_size_raw = certified_archive.get("max_size")
        if (
            isinstance(max_size_raw, bool)
            or not isinstance(max_size_raw, int)
            or max_size_raw <= 0
        ):
            raise ValueError(
                "certified_archive.max_size must be a positive integer."
            )
        certified_archive_policy = certified_archive_policy_raw
        certified_archive_max_size = max_size_raw

    streams = _mapping(root.get("streams"), "streams")
    stream_keys = {"pilot_seed", "confirm_seed", "independence_model"}
    if schema == FIXED_REFERENCE_SPEC_SCHEMA_V2 and "seed_pairs" in streams:
        stream_keys.add("seed_pairs")
    _exact_keys(streams, stream_keys, "streams")
    pilot_seed = streams.get("pilot_seed")
    confirm_seed = streams.get("confirm_seed")
    if (
        isinstance(pilot_seed, bool)
        or not isinstance(pilot_seed, int)
        or isinstance(confirm_seed, bool)
        or not isinstance(confirm_seed, int)
        or pilot_seed == confirm_seed
    ):
        raise ValueError(
            "Pilot and confirm seeds must be distinct integers."
        )
    if streams.get("independence_model") != "ideal_product_random_streams":
        raise ValueError(
            "streams.independence_model must be ideal_product_random_streams."
        )
    seed_pairs: Tuple[Tuple[int, int, int], ...] = ()
    if "seed_pairs" in streams:
        pairs_raw = streams.get("seed_pairs")
        if not isinstance(pairs_raw, list) or not pairs_raw:
            raise ValueError("streams.seed_pairs must be a nonempty array.")
        parsed_pairs = []
        used_run_seeds = set()
        used_stream_seeds = set()
        for index, raw_pair in enumerate(pairs_raw):
            pair = _mapping(raw_pair, f"streams.seed_pairs[{index}]")
            _exact_keys(
                pair,
                {"run_seed", "pilot_seed", "confirm_seed"},
                f"streams.seed_pairs[{index}]",
            )
            values = tuple(
                pair.get(key)
                for key in ("run_seed", "pilot_seed", "confirm_seed")
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise ValueError(
                    "Every streams.seed_pairs value must be an integer."
                )
            run_value, pilot_value, confirm_value = values
            if (
                run_value in used_run_seeds
                or pilot_value == confirm_value
                or pilot_value in used_stream_seeds
                or confirm_value in used_stream_seeds
            ):
                raise ValueError(
                    "streams.seed_pairs must use unique run and stream seeds."
                )
            used_run_seeds.add(run_value)
            used_stream_seeds.update((pilot_value, confirm_value))
            parsed_pairs.append(
                (run_value, pilot_value, confirm_value)
            )
        seed_pairs = tuple(sorted(parsed_pairs))

    budgets = _mapping(root.get("failure_budgets"), "failure_budgets")
    _exact_keys(
        budgets,
        {"pilot", "confirm"},
        "failure_budgets",
    )
    pilot_delta = _probability(budgets.get("pilot"), "failure_budgets.pilot")
    confirm_delta = _probability(
        budgets.get("confirm"),
        "failure_budgets.confirm",
    )
    if pilot_delta + confirm_delta >= 1.0:
        raise ValueError("Failure budgets must sum to less than one.")

    metrics = _mapping(root.get("metrics"), "metrics")
    _exact_keys(
        metrics,
        {
            "igd_p",
            "hv_reference",
            "max_igd_bound",
            "max_hv_deficit_bound",
        },
        "metrics",
    )
    raw_igd_p = metrics.get("igd_p")
    if isinstance(raw_igd_p, bool) or not isinstance(
        raw_igd_p,
        (int, float),
    ):
        raise ValueError("metrics.igd_p must be a JSON number.")
    igd_p = float(raw_igd_p)
    if not math.isfinite(igd_p) or igd_p < 1.0:
        raise ValueError("metrics.igd_p must be finite and at least one.")
    hv_reference = _finite_vector(
        metrics.get("hv_reference"),
        dimension=objective_dimension,
        label="metrics.hv_reference",
    )
    raw_max_igd = metrics.get("max_igd_bound")
    raw_max_hv = metrics.get("max_hv_deficit_bound")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (raw_max_igd, raw_max_hv)
    ):
        raise ValueError("Metric tolerances must be JSON numbers.")
    max_igd = float(raw_max_igd)
    max_hv = float(raw_max_hv)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (max_igd, max_hv)
    ):
        raise ValueError(
            "Metric tolerances must be finite and nonnegative."
        )

    return FixedReferenceCertificateSpecification(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        instance_sha256=instance_hash,
        pareto_smc_specification_sha256=smc_hash,
        reference_source=str(source),
        reference_artifact_sha256=artifact_hash,
        reference_objectives=canonical,
        pilot_seed=pilot_seed,
        confirm_seed=confirm_seed,
        pilot_failure_budget=pilot_delta,
        confirm_failure_budget=confirm_delta,
        igd_p=igd_p,
        hv_reference=hv_reference,
        max_igd_bound=max_igd,
        max_hv_deficit_bound=max_hv,
        schema=str(schema),
        reference_witnesses=reference_witnesses,
        reference_witness_payload_sha256=witness_payload_hash,
        reference_feasibility_verified_by_runtime=feasibility_verified,
        reference_witness_max_abs_error=witness_max_abs_error,
        reference_witness_equivalence_contract=(
            witness_equivalence_contract
        ),
        certified_archive_policy=certified_archive_policy,
        certified_archive_max_size=certified_archive_max_size,
        seed_pairs=seed_pairs,
    )
