from __future__ import annotations

"""Canonical end-to-end raw-artifact verifier for the v16 certificate branch.

The verifier accepts one strict canonical JSON packet. It reconstructs the
TSP instance, frozen cells, typed annealed-MH replica configurations, pilot
and confirm streams, exact Clopper--Pearson lower bounds, false-PASS risk, and
the bounded metric archive. No caller-supplied child certificate or Boolean
is accepted.

The probability statements still assume ideal product random streams. The
replay performed here establishes deterministic source alignment, not that a
finite PRNG implements independent mathematical random variables.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_archive_cap_certificate import ArchiveCapMetricCertificate, certify_archive_cap
from .pareto_frozen_cells import (
    FrozenCellManifest,
    canonical_fraction_text,
    load_frozen_cell_manifest,
    parse_canonical_fraction,
)
from .pareto_independent_replica_certificate import (
    ClopperPearsonLowerBracket,
    FalsePassCertificate,
    PilotPowerCertificate,
    build_false_pass_certificate,
    certify_pilot_power,
    clopper_pearson_lower_bracket,
    independent_replica_miss_probability,
    parse_canonical_probability,
)
from .pareto_independent_replica_runner import (
    IndependentReplicaBatchResult,
    ReplicaTypeConfiguration,
    replica_configuration_sha256,
    replica_stream_plan_sha256,
    run_independent_replica_batch,
)
from .pareto_v15_context import V15CertificateContext

V16_COMPOSED_BUNDLE_SCHEMA = "pareto_composed_raw_artifact_bundle_v16"
V16_COMPOSED_CERTIFICATE_SCHEMA = "pareto_composed_raw_artifact_certificate_v16"
V16_INSTANCE_ARTIFACT_SCHEMA = "pareto_binary64_tsp_instance_artifact_v16"
V16_REFERENCE_PLAN_SCHEMA = "pareto_finite_reference_metric_plan_v16"
V16_TYPE_CELL_PLAN_SCHEMA = "pareto_type_cell_probability_plan_v16"
V16_STREAM_PLAN_SCHEMA = "pareto_independent_replica_stream_seed_v16"
V16_REPLICA_CONFIGURATION_SCHEMA = "pareto_replica_configuration_v16"

class V16ArtifactError(ValueError):
    """Raised when a canonical packet cannot be reconstructed fail-closed."""

def canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V16ArtifactError("Artifact is not canonical-JSON serializable.") from error

def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V16ArtifactError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result

def _keys(payload: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise V16ArtifactError(
            f"{label} keys differ: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}."
        )

def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V16ArtifactError(f"{label} must be a JSON object.")
    return value

def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise V16ArtifactError(f"{label} must be a JSON array.")
    return value

def _int(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V16ArtifactError(f"{label} must be an integer.")
    if positive and value <= 0:
        raise V16ArtifactError(f"{label} must be positive.")
    return value

def _float_hex(value: object, *, label: str, nonnegative: bool = True) -> float:
    if not isinstance(value, str):
        raise V16ArtifactError(f"{label} must be a canonical float.hex string.")
    try:
        resolved = float.fromhex(value)
    except ValueError as error:
        raise V16ArtifactError(f"{label} is not a float.hex value.") from error
    if not math.isfinite(resolved) or (nonnegative and resolved < 0.0):
        raise V16ArtifactError(f"{label} has an invalid finite-value contract.")
    if resolved.hex() != value:
        raise V16ArtifactError(f"{label} is not canonical float.hex text.")
    return resolved

def _context(payload: object) -> V15CertificateContext:
    item = _mapping(payload, label="context")
    _keys(item, {"schema", "case_id", "instance_sha256", "configuration_sha256",
                 "cell_manifest_sha256", "reference_sha256", "type_cell_plan_sha256",
                 "pilot_plan_sha256", "confirm_plan_sha256", "context_sha256"},
          label="context")
    context = V15CertificateContext(
        case_id=str(item["case_id"]),
        instance_sha256=str(item["instance_sha256"]),
        configuration_sha256=str(item["configuration_sha256"]),
        cell_manifest_sha256=str(item["cell_manifest_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
        type_cell_plan_sha256=str(item["type_cell_plan_sha256"]),
        pilot_plan_sha256=str(item["pilot_plan_sha256"]),
        confirm_plan_sha256=str(item["confirm_plan_sha256"]),
    )
    if item["schema"] != context.canonical_payload()["schema"]:
        raise V16ArtifactError("Context schema mismatch.")
    if item["context_sha256"] != context.context_sha256:
        raise V16ArtifactError("Context digest mismatch.")
    return context

def _instance(payload: object) -> MultiObjectiveTSPInstance:
    item = _mapping(payload, label="instance")
    _keys(item, {"schema", "name", "distance_matrices_hex"}, label="instance")
    if item["schema"] != V16_INSTANCE_ARTIFACT_SCHEMA:
        raise V16ArtifactError("Unexpected instance artifact schema.")
    if not isinstance(item["name"], str) or not item["name"]:
        raise V16ArtifactError("Instance name must be nonempty text.")
    raw_matrices = _sequence(item["distance_matrices_hex"], label="distance_matrices_hex")
    if not raw_matrices:
        raise V16ArtifactError("At least one objective matrix is required.")
    matrices: list[list[list[float]]] = []
    dimension: int | None = None
    for objective_index, raw_matrix in enumerate(raw_matrices):
        rows = _sequence(raw_matrix, label=f"matrix[{objective_index}]")
        if dimension is None:
            dimension = len(rows)
        if dimension is None or dimension < 3 or len(rows) != dimension:
            raise V16ArtifactError("Objective matrices must share dimension >= 3.")
        matrix: list[list[float]] = []
        for row_index, raw_row in enumerate(rows):
            row_values = _sequence(raw_row, label=f"matrix[{objective_index}][{row_index}]")
            if len(row_values) != dimension:
                raise V16ArtifactError("Objective matrices must be square.")
            matrix.append([
                _float_hex(value, label=f"matrix[{objective_index}][{row_index}][{column}]")
                for column, value in enumerate(row_values)
            ])
        matrices.append(matrix)
    return MultiObjectiveTSPInstance.from_distance_matrices(matrices, name=str(item["name"]))

def _configurations(payload: object, *, dimension: int) -> tuple[ReplicaTypeConfiguration, ...]:
    raw = _sequence(payload, label="replica_configurations")
    if not raw:
        raise V16ArtifactError("At least one replica configuration is required.")
    result: list[ReplicaTypeConfiguration] = []
    for index, value in enumerate(raw):
        item = _mapping(value, label=f"replica_configurations[{index}]")
        _keys(item, {"schema", "type_id", "reference_direction_hex", "beta_schedule_hex",
                     "mutation_steps_by_stage", "replica_count", "chebyshev_rho_hex",
                     "global_refresh_probability_hex"}, label=f"replica_configurations[{index}]")
        if item["schema"] != V16_REPLICA_CONFIGURATION_SCHEMA:
            raise V16ArtifactError("Unexpected replica-configuration schema.")
        config = ReplicaTypeConfiguration(
            type_id=str(item["type_id"]),
            reference_direction=tuple(_float_hex(v, label="reference_direction")
                                      for v in _sequence(item["reference_direction_hex"], label="reference_direction_hex")),
            beta_schedule=tuple(_float_hex(v, label="beta_schedule")
                                for v in _sequence(item["beta_schedule_hex"], label="beta_schedule_hex")),
            mutation_steps_by_stage=tuple(_int(v, label="mutation_steps")
                                          for v in _sequence(item["mutation_steps_by_stage"], label="mutation_steps_by_stage")),
            replica_count=_int(item["replica_count"], label="replica_count", positive=True),
            chebyshev_rho=_float_hex(item["chebyshev_rho_hex"], label="chebyshev_rho_hex"),
            global_refresh_probability=_float_hex(item["global_refresh_probability_hex"], label="global_refresh_probability_hex"),
        )
        config.validate(dimension=dimension)
        result.append(config)
    if len({config.type_id for config in result}) != len(result):
        raise V16ArtifactError("Replica type IDs must be unique.")
    return tuple(result)

def _stream_seed(payload: object, *, role: str) -> int:
    item = _mapping(payload, label=f"{role}_stream")
    _keys(item, {"schema", "stream_role", "master_seed"}, label=f"{role}_stream")
    if item["schema"] != V16_STREAM_PLAN_SCHEMA or item["stream_role"] != role:
        raise V16ArtifactError(f"{role} stream schema/role mismatch.")
    return _int(item["master_seed"], label=f"{role}.master_seed")

def _fraction_point(value: object, *, label: str) -> tuple[Fraction, ...]:
    raw = _sequence(value, label=label)
    if not raw:
        raise V16ArtifactError(f"{label} must be nonempty.")
    return tuple(parse_canonical_fraction(v, label=f"{label}[{i}]") for i, v in enumerate(raw))

def _sqrt_upper(value: Fraction, *, bits: int = 256) -> Fraction:
    if value < 0:
        raise V16ArtifactError("Negative squared distance.")
    if value == 0:
        return Fraction(0)
    scaled = -(-(value.numerator << (2 * bits)) // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << bits)

def _lp_upper(left: Sequence[Fraction], right: Sequence[Fraction], p: str) -> Fraction:
    differences = tuple(abs(a - b) for a, b in zip(left, right))
    if p == "1":
        return sum(differences, Fraction(0))
    if p == "2":
        return _sqrt_upper(sum((value * value for value in differences), Fraction(0)))
    if p == "infinity":
        return max(differences, default=Fraction(0))
    raise V16ArtifactError("local_norm_p must be '1', '2', or 'infinity'.")

@dataclass(frozen=True)
class SelectedCellCertificate:
    cell: tuple[int, ...]
    selected_type_id: str
    pilot_successes: int
    pilot_trials: int
    cp_lower: str
    confirm_replicas: int
    confirm_miss_upper: str
    confirm_failure_budget: str
    confirm_hit_observed: bool

@dataclass(frozen=True)
class V16ComposedArtifactCertificate:
    schema: str
    packet_sha256: str
    context_sha256: str
    p0_correctness_gate: bool
    canonical_raw_artifacts_recomputed: bool
    pilot_result: IndependentReplicaBatchResult
    confirm_result: IndependentReplicaBatchResult
    selected_cells: tuple[SelectedCellCertificate, ...]
    pilot_power_certificates: tuple[PilotPowerCertificate, ...]
    pilot_power_design_gates_all_pass: bool
    pilot_power_assumptions_verified_by_raw_packet: bool
    false_pass_certificate: FalsePassCertificate
    archive_cap_certificate: ArchiveCapMetricCertificate
    certificate_scope: str
    exactness_scope: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "packet_sha256": self.packet_sha256,
            "context_sha256": self.context_sha256,
            "p0_correctness_gate": self.p0_correctness_gate,
            "canonical_raw_artifacts_recomputed": self.canonical_raw_artifacts_recomputed,
            "pilot_result": self.pilot_result.to_jsonable(),
            "confirm_result": self.confirm_result.to_jsonable(),
            "selected_cells": [asdict(item) for item in self.selected_cells],
            "pilot_power_certificates": [item.to_jsonable() for item in self.pilot_power_certificates],
            "pilot_power_design_gates_all_pass": self.pilot_power_design_gates_all_pass,
            "pilot_power_assumptions_verified_by_raw_packet": self.pilot_power_assumptions_verified_by_raw_packet,
            "false_pass_certificate": self.false_pass_certificate.to_jsonable(),
            "archive_cap_certificate": self.archive_cap_certificate.to_jsonable(),
            "certificate_scope": self.certificate_scope,
            "exactness_scope": self.exactness_scope,
        }

def _write_temp_json(directory: str, name: str, payload: object) -> Path:
    path = Path(directory) / name
    path.write_bytes(canonical_json_bytes(payload))
    return path

def verify_v16_composed_bundle(path: str | Path) -> V16ComposedArtifactCertificate:
    """Reload and recompute every child certificate from one canonical packet."""
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V16ArtifactError("Bundle is not strict UTF-8 JSON.") from error
    if raw != canonical_json_bytes(payload):
        raise V16ArtifactError("Bundle bytes are not the canonical JSON encoding.")
    root = _mapping(payload, label="bundle")
    _keys(root, {"schema", "context", "instance", "cell_manifest",
                 "replica_configurations", "pilot_stream", "confirm_stream",
                 "reference_plan", "type_cell_plan"}, label="bundle")
    if root["schema"] != V16_COMPOSED_BUNDLE_SCHEMA:
        raise V16ArtifactError("Unexpected bundle schema.")
    packet_sha256 = hashlib.sha256(raw).hexdigest()
    context = _context(root["context"])
    instance = _instance(root["instance"])
    if instance_sha256(instance) != context.instance_sha256:
        raise V16ArtifactError("Raw instance does not reconstruct the bound instance hash.")
    configurations = _configurations(root["replica_configurations"], dimension=instance.num_objectives)
    if replica_configuration_sha256(configurations) != context.configuration_sha256:
        raise V16ArtifactError("Raw configurations do not reconstruct the bound hash.")
    cell_payload = root["cell_manifest"]
    cell_bytes = canonical_json_bytes(cell_payload)
    if hashlib.sha256(cell_bytes).hexdigest() != context.cell_manifest_sha256:
        raise V16ArtifactError("Raw frozen-cell artifact hash mismatch.")
    reference_plan = _mapping(root["reference_plan"], label="reference_plan")
    type_cell_plan = _mapping(root["type_cell_plan"], label="type_cell_plan")
    if canonical_sha256(reference_plan) != context.reference_sha256:
        raise V16ArtifactError("Raw reference plan hash mismatch.")
    if canonical_sha256(type_cell_plan) != context.type_cell_plan_sha256:
        raise V16ArtifactError("Raw type-cell plan hash mismatch.")
    pilot_seed = _stream_seed(root["pilot_stream"], role="pilot")
    confirm_seed = _stream_seed(root["confirm_stream"], role="confirm")
    if replica_stream_plan_sha256(configurations, master_seed=pilot_seed, stream_role="pilot",
                                  cell_manifest_sha256=context.cell_manifest_sha256) != context.pilot_plan_sha256:
        raise V16ArtifactError("Pilot plan hash mismatch.")
    if replica_stream_plan_sha256(configurations, master_seed=confirm_seed, stream_role="confirm",
                                  cell_manifest_sha256=context.cell_manifest_sha256) != context.confirm_plan_sha256:
        raise V16ArtifactError("Confirm plan hash mismatch.")
    with tempfile.TemporaryDirectory() as directory:
        cell_path = _write_temp_json(directory, "cells.json", cell_payload)
        manifest = load_frozen_cell_manifest(cell_path, expected_sha256=context.cell_manifest_sha256)
        pilot = run_independent_replica_batch(instance, cell_manifest_path=cell_path,
                                              certificate_context=context, configurations=configurations,
                                              master_seed=pilot_seed, stream_role="pilot")
        confirm = run_independent_replica_batch(instance, cell_manifest_path=cell_path,
                                                certificate_context=context, configurations=configurations,
                                                master_seed=confirm_seed, stream_role="confirm")
    config_by_type = {config.type_id: config for config in configurations}
    _keys(type_cell_plan, {"schema", "selection_rule", "cells"}, label="type_cell_plan")
    if type_cell_plan["schema"] != V16_TYPE_CELL_PLAN_SCHEMA:
        raise V16ArtifactError("Unexpected type-cell plan schema.")
    if type_cell_plan["selection_rule"] != "max_cp_lower_then_lexicographic_type_id":
        raise V16ArtifactError("Unexpected type-cell selection rule.")
    raw_cells = _sequence(type_cell_plan["cells"], label="type_cell_plan.cells")
    if not raw_cells:
        raise V16ArtifactError("At least one certified cell is required.")
    pilot_by_type = {type_id: tuple(e for e in pilot.endpoints if e.type_id == type_id)
                     for type_id in config_by_type}
    confirm_by_type = {type_id: tuple(e for e in confirm.endpoints if e.type_id == type_id)
                       for type_id in config_by_type}
    selected: list[SelectedCellCertificate] = []
    power_certificates: list[PilotPowerCertificate] = []
    pilot_alpha_total = Fraction(0)
    confirm_budget_total = Fraction(0)
    selected_type_by_cell: dict[tuple[int, ...], str] = {}
    for cell_index, raw_cell in enumerate(raw_cells):
        item = _mapping(raw_cell, label=f"type_cell_plan.cells[{cell_index}]")
        _keys(item, {"cell", "pilot_alpha_by_type", "target_probability",
                     "true_probability_lower_bound", "minimum_pilot_power",
                     "confirm_failure_budget"}, label=f"type_cell_plan.cells[{cell_index}]")
        cell = tuple(_int(v, label="cell coordinate") for v in _sequence(item["cell"], label="cell"))
        if cell not in manifest.observable_cells:
            raise V16ArtifactError(f"Certified cell {cell!r} is not observable in the frozen manifest.")
        alpha_map = _mapping(item["pilot_alpha_by_type"], label="pilot_alpha_by_type")
        if set(alpha_map) != set(config_by_type):
            raise V16ArtifactError("pilot_alpha_by_type must cover every configured type exactly.")
        p0 = parse_canonical_probability(item["target_probability"], label="target_probability")
        p1 = parse_canonical_probability(item["true_probability_lower_bound"], label="true_probability_lower_bound")
        minimum_power = parse_canonical_probability(item["minimum_pilot_power"], label="minimum_pilot_power")
        confirm_budget = parse_canonical_probability(item["confirm_failure_budget"], label="confirm_failure_budget")
        if confirm_budget <= 0:
            raise V16ArtifactError("Every confirm_failure_budget must be positive.")
        confirm_budget_total += confirm_budget
        candidates: list[tuple[Fraction, str, int, int, ClopperPearsonLowerBracket]] = []
        for type_id in sorted(config_by_type):
            endpoints = pilot_by_type[type_id]
            trials = len(endpoints)
            successes = sum(endpoint.frozen_cell == cell for endpoint in endpoints)
            alpha = parse_canonical_probability(alpha_map[type_id], label=f"alpha[{type_id}]")
            if alpha <= 0 or alpha >= 1:
                raise V16ArtifactError("Pilot alpha allocations must lie in (0,1).")
            pilot_alpha_total += alpha
            bracket = clopper_pearson_lower_bracket(successes, trials, alpha)
            candidates.append((bracket.lower, type_id, successes, trials, bracket))
            power_certificates.append(certify_pilot_power(
                trials, p0, p1, alpha,
                minimum_acceptable_pass_probability=minimum_power,
            ))
        best_lower = max(value[0] for value in candidates)
        selected_type = min(value[1] for value in candidates if value[0] == best_lower)
        lower, _, successes, trials, _ = next(value for value in candidates if value[1] == selected_type)
        if lower <= 0:
            raise V16ArtifactError(f"Pilot produced no positive lower mass for cell {cell!r}.")
        confirm_endpoints = confirm_by_type[selected_type]
        replicas = len(confirm_endpoints)
        miss_upper = independent_replica_miss_probability(lower, replicas)
        hit_observed = any(endpoint.frozen_cell == cell for endpoint in confirm_endpoints)
        if miss_upper > confirm_budget:
            raise V16ArtifactError(f"Confirm risk budget failed for cell {cell!r}.")
        if not hit_observed:
            raise V16ArtifactError(f"Confirm stream did not hit certified cell {cell!r}.")
        selected_type_by_cell[cell] = selected_type
        selected.append(SelectedCellCertificate(
            cell=cell, selected_type_id=selected_type, pilot_successes=successes,
            pilot_trials=trials, cp_lower=canonical_fraction_text(lower),
            confirm_replicas=replicas, confirm_miss_upper=canonical_fraction_text(miss_upper),
            confirm_failure_budget=canonical_fraction_text(confirm_budget),
            confirm_hit_observed=True,
        ))
    if pilot_alpha_total > 1 or confirm_budget_total > 1:
        raise V16ArtifactError("Familywise probability allocations exceed one.")
    power_gates_all_pass = all(certificate.power_gate for certificate in power_certificates)
    false_pass = build_false_pass_certificate(pilot_alpha_total, confirm_budget_total)
    _keys(reference_plan, {"schema", "reference_points", "local_norm_p", "archive_cap",
                           "hv_reference", "max_ordinary_igd", "max_igd_plus",
                           "max_hv_deficit"}, label="reference_plan")
    if reference_plan["schema"] != V16_REFERENCE_PLAN_SCHEMA:
        raise V16ArtifactError("Unexpected reference plan schema.")
    reference_points = tuple(_fraction_point(value, label=f"reference_points[{index}]")
                             for index, value in enumerate(_sequence(reference_plan["reference_points"], label="reference_points")))
    if not reference_points or any(len(point) != manifest.dimension for point in reference_points):
        raise V16ArtifactError("Reference points are empty or dimensionally inconsistent.")
    p = str(reference_plan["local_norm_p"])
    if p != manifest.local_norm_p:
        raise V16ArtifactError("Reference norm differs from the frozen metric manifest.")
    witness_by_cell: dict[tuple[int, ...], tuple[Fraction, ...]] = {}
    for cell, type_id in selected_type_by_cell.items():
        endpoint = min((e for e in confirm_by_type[type_id] if e.frozen_cell == cell),
                       key=lambda e: e.replica_index)
        witness_by_cell[cell] = tuple(parse_canonical_fraction(v, label="endpoint.exact_objective")
                                      for v in endpoint.exact_objective)
    witness_cells = sorted(witness_by_cell)
    witnesses = tuple(witness_by_cell[cell] for cell in witness_cells)
    witness_index = {cell: index for index, cell in enumerate(witness_cells)}
    mapping: list[int] = []
    per_reference_distance: list[Fraction] = []
    additive = [Fraction(0) for _ in range(manifest.dimension)]
    for reference in reference_points:
        cell = manifest.classify(reference)
        if cell not in witness_index:
            raise V16ArtifactError("A frozen reference cell lacks a selected confirm witness.")
        index = witness_index[cell]
        witness = witnesses[index]
        mapping.append(index)
        per_reference_distance.append(_lp_upper(reference, witness, p))
        for coordinate, (w_value, r_value) in enumerate(zip(witness, reference)):
            additive[coordinate] = max(additive[coordinate], w_value - r_value, Fraction(0))
    ordinary_base = sum(per_reference_distance, Fraction(0)) / len(reference_points)
    cap_certificate = certify_archive_cap(
        reference_points=reference_points,
        witnesses=witnesses,
        reference_to_witness=tuple(mapping),
        cap=_int(reference_plan["archive_cap"], label="archive_cap", positive=True),
        p=p,
        ordinary_igd_base_upper=ordinary_base,
        additive_base_vector=tuple(additive),
        hv_reference=_fraction_point(reference_plan["hv_reference"], label="hv_reference"),
        max_ordinary_igd=parse_canonical_fraction(reference_plan["max_ordinary_igd"], label="max_ordinary_igd"),
        max_igd_plus=parse_canonical_fraction(reference_plan["max_igd_plus"], label="max_igd_plus"),
        max_hv_deficit=parse_canonical_fraction(reference_plan["max_hv_deficit"], label="max_hv_deficit"),
    )
    if not cap_certificate.passed:
        raise V16ArtifactError("Recomputed archive-cap metric certificate failed.")
    return V16ComposedArtifactCertificate(
        schema=V16_COMPOSED_CERTIFICATE_SCHEMA,
        packet_sha256=packet_sha256,
        context_sha256=context.context_sha256,
        p0_correctness_gate=True,
        canonical_raw_artifacts_recomputed=True,
        pilot_result=pilot,
        confirm_result=confirm,
        selected_cells=tuple(selected),
        pilot_power_certificates=tuple(power_certificates),
        pilot_power_design_gates_all_pass=power_gates_all_pass,
        pilot_power_assumptions_verified_by_raw_packet=False,
        false_pass_certificate=false_pass,
        archive_cap_certificate=cap_certificate,
        certificate_scope="frozen_supplied_reference_relative_only",
        exactness_scope=("exact_binary64_matrix_reconstruction_exact_dyadic_endpoint_sums_"
                         "exact_rational_cells_and_probability_bounds;binary64_mh_energy"),
    )

def write_canonical_v16_bundle(path: str | Path, payload: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(dict(payload))
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()

__all__ = [
    "V16_COMPOSED_BUNDLE_SCHEMA", "V16_COMPOSED_CERTIFICATE_SCHEMA",
    "V16_INSTANCE_ARTIFACT_SCHEMA", "V16_REFERENCE_PLAN_SCHEMA",
    "V16_REPLICA_CONFIGURATION_SCHEMA", "V16_STREAM_PLAN_SCHEMA",
    "V16_TYPE_CELL_PLAN_SCHEMA", "SelectedCellCertificate", "V16ArtifactError",
    "V16ComposedArtifactCertificate", "canonical_json_bytes", "canonical_sha256",
    "verify_v16_composed_bundle", "write_canonical_v16_bundle",
]
