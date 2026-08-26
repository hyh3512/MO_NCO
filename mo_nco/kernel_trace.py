from __future__ import annotations

"""Independent replay checks for certified single-site transition traces."""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .moves import random_tour, sample_two_opt_indices, two_opt_at
from .potential import ScalarArchivePotential
from .types import ObjectiveVector, Tour


@dataclass(frozen=True)
class TraceVerificationResult:
    passed: bool
    records: int
    # Total post-header transitions on the evaluation clock.
    transitions: int
    errors: Tuple[str, ...]
    final_chain_hash: str
    active_transitions: int = 0
    identity_transitions: int = 0


def _canonical_record_hash(record: Dict[str, Any]) -> str:
    payload = dict(record)
    claimed = str(payload.pop("record_hash", ""))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return actual if actual == claimed else ""


def _population_hash(population: Sequence[Tour]) -> str:
    encoded = json.dumps(population, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_fixed_zero_tour(tour: Tour, num_cities: int) -> bool:
    return (
        len(tour) == num_cities
        and bool(tour)
        and tour[0] == 0
        and sorted(tour) == list(range(num_cities))
    )


def _context_hash_from_header(header: Dict[str, Any]) -> str:
    payload = {
        "ideal": header["ideal"],
        "nadir": header["nadir"],
        "scales": header["scales"],
        "weights": header["weights"],
        "chebyshev_rho": header["chebyshev_rho"],
        "temperature": header["temperature"],
        "lazy_probability": header["lazy_probability"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _typed_energy(
    objective: ObjectiveVector,
    coordinate: int,
    ideal: ObjectiveVector,
    scales: ObjectiveVector,
    weights: Sequence[ObjectiveVector],
    chebyshev_rho: float,
) -> float:
    normalized = tuple((value - lo) / scale for value, lo, scale in zip(objective, ideal, scales))
    weighted = tuple(max(1e-3, weight) * value for weight, value in zip(weights[coordinate], normalized))
    return max(weighted) + chebyshev_rho * sum(weighted)


def verify_certified_trace(
    trace_path: str | Path,
    *,
    instance: Optional[MultiObjectiveTSPInstance] = None,
    tolerance: float = 1e-10,
    expected_context_hash: Optional[str] = None,
    expected_final_chain_hash: Optional[str] = None,
    expected_records: Optional[int] = None,
    expected_transition_attempts: Optional[int] = None,
    expected_proposal_evaluations: Optional[int] = None,
    expected_seed: Optional[int] = None,
    expected_num_particles: Optional[int] = None,
    expected_instance_sha256: Optional[str] = None,
) -> TraceVerificationResult:
    """Replay a trace and independently recompute all recorded MH decisions.

    When ``instance`` is supplied, the verifier also recomputes every proposed
    objective vector from the source problem rather than trusting the trace.
    """

    path = Path(trace_path)
    errors: List[str] = []
    if not path.exists():
        return TraceVerificationResult(False, 0, 0, (f"trace not found: {path}",), "")

    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return TraceVerificationResult(False, 0, 0, (f"trace parse failure: {exc}",), "")
    if not records:
        return TraceVerificationResult(False, 0, 0, ("trace is empty",), "")
    if any(not isinstance(record, dict) for record in records):
        return TraceVerificationResult(
            False,
            len(records),
            0,
            ("trace records must be JSON objects",),
            "",
        )

    previous_hash = "0" * 64
    for position, record in enumerate(records):
        if record.get("previous_record_hash") != previous_hash:
            errors.append(f"record {position}: broken previous_record_hash")
        try:
            actual_hash = _canonical_record_hash(record)
        except (TypeError, ValueError) as exc:
            errors.append(f"record {position}: non-canonical payload: {exc}")
            actual_hash = ""
        if not actual_hash:
            errors.append(f"record {position}: record hash mismatch")
            actual_hash = str(record.get("record_hash", ""))
        previous_hash = actual_hash

    header = records[0]
    if header.get("record_type") != "header":
        errors.append("first record is not a header")
        return TraceVerificationResult(False, len(records), 0, tuple(errors), previous_hash)

    try:
        temperature = float(header["temperature"])
        lazy_probability = float(header["lazy_probability"])
        chebyshev_rho = float(header["chebyshev_rho"])
        minimum_scale_fraction = float(header["minimum_scale_fraction"])
        absolute_scale_floor = float(header["absolute_scale_floor"])
        ideal = tuple(float(value) for value in header["ideal"])
        nadir = tuple(float(value) for value in header["nadir"])
        scales = tuple(float(value) for value in header["scales"])
        weights = tuple(tuple(float(value) for value in row) for row in header["weights"])
        population: List[Tour] = [tuple(int(city) for city in row) for row in header["initial_population"]]
        objectives: List[ObjectiveVector] = [
            tuple(float(value) for value in row) for row in header["initial_objectives"]
        ]
        seed = int(header["seed"])
        num_cities = int(header["num_cities"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid header: {exc}")
        return TraceVerificationResult(False, len(records), 0, tuple(errors), previous_hash)

    if _population_hash(population) != header.get("initial_population_hash"):
        errors.append("header: initial population hash mismatch")
    if header.get("algorithm_contract") != "theory_certified_single_site_v4":
        errors.append("header: algorithm contract mismatch")
    if header.get("implementation_version") != "0.8.0":
        errors.append("header: implementation version mismatch")
    if header.get("rng_contract") != "python_random_mt19937_seed_replay_v1":
        errors.append("header: unsupported RNG replay contract")
    replay_rng = random.Random(seed)
    replay_initial_population = [
        random_tour(num_cities, replay_rng)
        for _ in range(len(population))
    ]
    if replay_initial_population != population:
        errors.append("header: initial population RNG replay mismatch")
    if any(not _is_fixed_zero_tour(tour, num_cities) for tour in population):
        errors.append("header: invalid fixed-zero tour state")
    if expected_seed is not None and seed != expected_seed:
        errors.append("header: metadata seed mismatch")
    if expected_num_particles is not None and len(population) != expected_num_particles:
        errors.append("header: metadata particle count mismatch")
    if (
        expected_instance_sha256 is not None
        and header.get("instance_sha256") != expected_instance_sha256
    ):
        errors.append("header: metadata instance fingerprint mismatch")
    try:
        recomputed_context_hash = _context_hash_from_header(header)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"header: cannot recompute context hash: {exc}")
        recomputed_context_hash = ""
    if recomputed_context_hash != header.get("context_hash"):
        errors.append("header: context hash mismatch")
    if expected_context_hash is not None and header.get("context_hash") != expected_context_hash:
        errors.append("header: metadata context hash mismatch")
    if temperature <= 0.0:
        errors.append("header: nonpositive temperature")
    if not 0.0 < lazy_probability < 1.0:
        errors.append("header: lazy_probability must lie strictly between zero and one")
    if len(population) != len(weights) or len(population) != len(objectives):
        errors.append("header: population/weight/objective cardinality mismatch")
    if not math.isfinite(minimum_scale_fraction) or minimum_scale_fraction < 0.0:
        errors.append("header: invalid minimum_scale_fraction")
    if not math.isfinite(absolute_scale_floor) or absolute_scale_floor <= 0.0:
        errors.append("header: invalid absolute_scale_floor")
    objective_dimension = len(ideal)
    if (
        objective_dimension == 0
        or len(nadir) != objective_dimension
        or len(scales) != objective_dimension
        or any(len(weight) != objective_dimension for weight in weights)
        or any(len(objective) != objective_dimension for objective in objectives)
    ):
        errors.append("header: objective/context dimension mismatch")
    context_values = [
        temperature,
        lazy_probability,
        chebyshev_rho,
        minimum_scale_fraction,
        absolute_scale_floor,
        *ideal,
        *nadir,
        *scales,
        *(value for weight in weights for value in weight),
        *(value for objective in objectives for value in objective),
    ]
    if any(not math.isfinite(value) for value in context_values):
        errors.append("header: non-finite context/objective value")
    if any(scale <= 0.0 for scale in scales):
        errors.append("header: scales must be strictly positive")
    if instance is not None:
        if header.get("instance_sha256") != instance_sha256(instance):
            errors.append("header: source instance fingerprint mismatch")
        if instance.num_cities != int(header.get("num_cities", -1)):
            errors.append("header: source instance city count mismatch")
        if instance.num_objectives != int(header.get("num_objectives", -1)):
            errors.append("header: source instance objective count mismatch")
        source_initial_objectives: list[ObjectiveVector] = []
        for coordinate, (tour, objective) in enumerate(zip(population, objectives)):
            try:
                instance.validate_tour(tour)
                recomputed_objective = instance.evaluate_unchecked(tour)
            except (TypeError, ValueError) as exc:
                errors.append(f"header: invalid source state at coordinate {coordinate}: {exc}")
                continue
            source_initial_objectives.append(recomputed_objective)
            if recomputed_objective != objective:
                errors.append(f"header: initial objective mismatch at coordinate {coordinate}")
        if len(source_initial_objectives) == len(population):
            expected_ideal = tuple(
                min(objective[d] for objective in source_initial_objectives)
                for d in range(instance.num_objectives)
            )
            expected_nadir = tuple(
                max(objective[d] for objective in source_initial_objectives)
                for d in range(instance.num_objectives)
            )
            expected_scales = tuple(
                max(
                    absolute_scale_floor,
                    hi - lo,
                    minimum_scale_fraction * estimate,
                )
                for lo, hi, estimate in zip(
                    expected_ideal,
                    expected_nadir,
                    instance.objective_scale_estimates,
                )
            )
            expected_weights = ScalarArchivePotential.reference_directions(
                instance.num_objectives,
                len(population),
            )
            if ideal != expected_ideal or nadir != expected_nadir:
                errors.append("header: source ideal/nadir context mismatch")
            if scales != expected_scales:
                errors.append("header: source scale context mismatch")
            if weights != expected_weights:
                errors.append("header: source reference-direction context mismatch")

    active_transitions = 0
    identity_transitions = 0
    expected_attempt = 1
    expected_proposal_index = 1
    for position, record in enumerate(records[1:], start=1):
        record_type = record.get("record_type")
        if record.get("transition_attempt") != expected_attempt:
            errors.append(f"record {position}: transition_attempt sequence mismatch")
        expected_attempt += 1
        if record_type == "lazy_transition":
            identity_transitions += 1
            try:
                lazy_uniform = float(record["lazy_uniform"])
                identity_tour = tuple(int(city) for city in record["identity_tour"])
                identity_objective = tuple(float(value) for value in record["identity_objective"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"record {position}: invalid lazy transition payload: {exc}")
                continue
            if not 0.0 <= lazy_uniform < lazy_probability:
                errors.append(f"record {position}: lazy decision mismatch")
            if any(not math.isfinite(value) for value in identity_objective):
                errors.append(f"record {position}: non-finite lazy objective")
            replay_lazy_uniform = replay_rng.random()
            if replay_lazy_uniform != lazy_uniform:
                errors.append(f"record {position}: lazy RNG replay mismatch")
            population_hash = _population_hash(population)
            if record.get("population_hash_before") != population_hash:
                errors.append(f"record {position}: lazy population_hash_before mismatch")
            if record.get("population_hash_after") != population_hash:
                errors.append(f"record {position}: lazy population_hash_after mismatch")
            if not population or identity_tour != population[0]:
                errors.append(f"record {position}: lazy identity tour mismatch")
            if not objectives or identity_objective != objectives[0]:
                errors.append(f"record {position}: lazy identity objective mismatch")
            if instance is not None:
                try:
                    recomputed_objective = instance.evaluate_unchecked(identity_tour)
                except (IndexError, TypeError, ValueError) as exc:
                    errors.append(f"record {position}: invalid lazy source state: {exc}")
                else:
                    if recomputed_objective != identity_objective:
                        errors.append(f"record {position}: lazy source objective mismatch")
            continue
        if record_type != "transition":
            errors.append(f"record {position}: unexpected record_type")
            continue
        active_transitions += 1
        if record.get("proposal_index") != expected_proposal_index:
            errors.append(f"record {position}: proposal_index sequence mismatch")
        expected_proposal_index += 1
        try:
            coordinate = int(record["coordinate"])
            lazy_uniform = float(record["lazy_uniform"])
            i = int(record["two_opt_i"])
            j = int(record["two_opt_j"])
            current_tour = tuple(int(city) for city in record["current_tour"])
            proposed_tour = tuple(int(city) for city in record["proposed_tour"])
            current_objective = tuple(float(value) for value in record["current_objective"])
            proposed_objective = tuple(float(value) for value in record["proposed_objective"])
            claimed_delta_h = float(record["delta_h"])
            claimed_delta_over_t = float(record["delta_over_temperature"])
            claimed_log_alpha = float(record["log_alpha"])
            raw_log_uniform = record["log_uniform"]
            log_uniform = -math.inf if raw_log_uniform == "-inf" else float(raw_log_uniform)
            raw_accepted = record["accepted"]
            if not isinstance(raw_accepted, bool):
                raise TypeError("accepted must be a JSON boolean")
            claimed_accepted = raw_accepted
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"record {position}: invalid transition payload: {exc}")
            continue

        if coordinate < 0 or coordinate >= len(population):
            errors.append(f"record {position}: coordinate out of range")
            continue
        if not lazy_probability <= lazy_uniform < 1.0:
            errors.append(f"record {position}: active transition laziness decision mismatch")
        valid_index_pair = 1 <= i < j < num_cities
        if not valid_index_pair:
            errors.append(f"record {position}: invalid strict 2-opt index pair")
        numeric_values = [
            lazy_uniform,
            *current_objective,
            *proposed_objective,
            claimed_delta_h,
            claimed_delta_over_t,
            claimed_log_alpha,
        ]
        if any(not math.isfinite(value) for value in numeric_values):
            errors.append(f"record {position}: non-finite transition value")
        if not (math.isfinite(log_uniform) or log_uniform == -math.inf) or log_uniform > 0.0:
            errors.append(f"record {position}: invalid log_uniform")
        invalid_objective_dimensions = (
            len(current_objective) != objective_dimension
            or len(proposed_objective) != objective_dimension
        )
        if invalid_objective_dimensions:
            errors.append(f"record {position}: transition objective dimension mismatch")
        valid_tours = _is_fixed_zero_tour(current_tour, num_cities) and _is_fixed_zero_tour(
            proposed_tour,
            num_cities,
        )
        if not valid_tours:
            errors.append(f"record {position}: invalid fixed-zero tour payload")
        replay_lazy_uniform = replay_rng.random()
        replay_coordinate = replay_rng.randrange(len(population))
        replay_i, replay_j = sample_two_opt_indices(num_cities, replay_rng)
        replay_uniform_draw = replay_rng.random()
        replay_log_uniform = (
            -math.inf if replay_uniform_draw == 0.0 else math.log(replay_uniform_draw)
        )
        if replay_lazy_uniform != lazy_uniform:
            errors.append(f"record {position}: active lazy RNG replay mismatch")
        if replay_coordinate != coordinate:
            errors.append(f"record {position}: coordinate RNG replay mismatch")
        if (replay_i, replay_j) != (i, j):
            errors.append(f"record {position}: 2-opt RNG replay mismatch")
        if replay_log_uniform != log_uniform:
            errors.append(f"record {position}: acceptance RNG replay mismatch")
        if not valid_index_pair or invalid_objective_dimensions or not valid_tours:
            continue
        if population[coordinate] != current_tour:
            errors.append(f"record {position}: current tour does not match replay state")
        if objectives[coordinate] != current_objective:
            errors.append(f"record {position}: current objective does not match replay state")
        if _population_hash(population) != record.get("population_hash_before"):
            errors.append(f"record {position}: population_hash_before mismatch")

        recomputed_tour = two_opt_at(current_tour, i, j)
        if recomputed_tour != proposed_tour:
            errors.append(f"record {position}: proposed tour is not the recorded 2-opt move")

        if instance is not None:
            try:
                recomputed_objective = instance.evaluate_unchecked(proposed_tour)
            except (IndexError, TypeError, ValueError) as exc:
                errors.append(f"record {position}: invalid proposed source state: {exc}")
            else:
                if recomputed_objective != proposed_objective:
                    errors.append(f"record {position}: proposed objective mismatch")

        recomputed_delta_h = _typed_energy(
            proposed_objective,
            coordinate,
            ideal,
            scales,
            weights,
            chebyshev_rho,
        ) - _typed_energy(
            current_objective,
            coordinate,
            ideal,
            scales,
            weights,
            chebyshev_rho,
        )
        recomputed_delta_over_t = recomputed_delta_h / temperature
        recomputed_log_alpha = min(0.0, -recomputed_delta_over_t)
        recomputed_accepted = log_uniform < recomputed_log_alpha

        if abs(recomputed_delta_h - claimed_delta_h) > tolerance:
            errors.append(f"record {position}: delta_h mismatch")
        if abs(recomputed_delta_over_t - claimed_delta_over_t) > tolerance:
            errors.append(f"record {position}: delta_over_temperature mismatch")
        if abs(recomputed_log_alpha - claimed_log_alpha) > tolerance:
            errors.append(f"record {position}: log_alpha mismatch")
        if recomputed_accepted != claimed_accepted:
            errors.append(f"record {position}: acceptance decision mismatch")

        if claimed_accepted:
            population[coordinate] = proposed_tour
            objectives[coordinate] = proposed_objective
        if _population_hash(population) != record.get("population_hash_after"):
            errors.append(f"record {position}: population_hash_after mismatch")

    if expected_final_chain_hash is not None and previous_hash != expected_final_chain_hash:
        errors.append("trace: final chain hash mismatch")
    if expected_records is not None and len(records) != expected_records:
        errors.append("trace: record count mismatch")
    if expected_transition_attempts is not None and len(records) - 1 != expected_transition_attempts:
        errors.append("trace: transition attempt count mismatch")
    if (
        expected_proposal_evaluations is not None
        and active_transitions != expected_proposal_evaluations
    ):
        errors.append("trace: proposal evaluation count mismatch")

    return TraceVerificationResult(
        passed=not errors,
        records=len(records),
        transitions=active_transitions + identity_transitions,
        errors=tuple(errors),
        final_chain_hash=previous_hash,
        active_transitions=active_transitions,
        identity_transitions=identity_transitions,
    )
