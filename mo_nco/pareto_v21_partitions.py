from __future__ import annotations

"""Prospective V21 partition generation and overlap certification.

The public functions in this module deliberately separate name-bearing audit
metadata from name-free mathematical-instance fingerprints.  This allows a
renamed or repackaged instance to remain detectable as prior exposure.
"""

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


_SHAKE256_COUNTER_PREFIX = b"pareto-v21-shake256-counter-v1\0"
_V21_REGIMES = (
    "independent",
    "objective_correlated",
    "objective_conflicting",
    "structured",
    "heterogeneous",
)


class Shake256CounterRNG:
    """Portable byte-stream RNG with an explicit SHAKE256/counter contract."""

    def __init__(self, *, seed: bytes, domain: str) -> None:
        if not isinstance(seed, bytes) or not seed:
            raise ValueError("seed must be nonempty bytes.")
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain must be a nonempty string.")
        domain_bytes = domain.encode("utf-8")
        self._prefix = (
            _SHAKE256_COUNTER_PREFIX
            + len(domain_bytes).to_bytes(4, "big")
            + domain_bytes
            + len(seed).to_bytes(4, "big")
            + seed
        )
        self._counter = 0
        self._buffer = b""

    def read(self, length: int) -> bytes:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("length must be a nonnegative integer.")
        while len(self._buffer) < length:
            if self._counter >= 2**64:
                raise OverflowError("SHAKE256 counter exhausted.")
            block_input = self._prefix + self._counter.to_bytes(8, "big")
            self._buffer += hashlib.shake_256(block_input).digest(64)
            self._counter += 1
        result, self._buffer = self._buffer[:length], self._buffer[length:]
        return result

    def randbelow(self, upper_bound: int) -> int:
        """Uniform integer in ``range(upper_bound)`` via rejection sampling."""

        if (
            isinstance(upper_bound, bool)
            or not isinstance(upper_bound, int)
            or upper_bound <= 0
            or upper_bound > 2**64
        ):
            raise ValueError("upper_bound must be an integer in [1, 2**64].")
        rejection_limit = 2**64 - (2**64 % upper_bound)
        while True:
            value = int.from_bytes(self.read(8), "big")
            if value < rejection_limit:
                return value % upper_bound

    def randint(self, lower: int, upper: int) -> int:
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise ValueError("integer bounds cannot be booleans.")
        if not isinstance(lower, int) or not isinstance(upper, int) or lower > upper:
            raise ValueError("randint bounds must be ordered integers.")
        return lower + self.randbelow(upper - lower + 1)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class InstanceFingerprint:
    family: str
    problem_sha256: str
    semantic_sha256: str
    component_sha256: Mapping[str, str]


def fingerprint_instance(instance: Mapping[str, Any]) -> InstanceFingerprint:
    """Return fingerprints derived only from the optimization instance."""

    family = str(instance.get("family", "")).upper()
    if family == "MOTSP":
        return _fingerprint_motsp(instance)
    if family != "MOKP":
        raise ValueError(f"Unsupported instance family: {family or '<missing>'}.")
    weights = tuple(_positive_ints(instance.get("item_weights"), "item_weights"))
    profits = tuple(
        tuple(_nonnegative_ints(row, f"profits_by_objective[{index}]"))
        for index, row in enumerate(instance.get("profits_by_objective", ()))
    )
    capacity = _positive_int(instance.get("capacity"), "capacity")
    if not profits or any(len(row) != len(weights) for row in profits):
        raise ValueError("Each MOKP objective must have one profit per item.")
    declared_items = instance.get("num_items", len(weights))
    declared_objectives = instance.get("num_objectives", len(profits))
    if declared_items != len(weights) or declared_objectives != len(profits):
        raise ValueError("MOKP declared dimensions do not match coefficient arrays.")
    ordered_problem = {
        "family": "MOKP",
        "item_weights": weights,
        "profits_by_objective": profits,
        "capacity": capacity,
    }
    components = {
        "constraint": _digest(
            {
                "family": "MOKP",
                "item_weight_multiset": sorted(weights),
                "capacity": capacity,
            }
        )
    }
    components.update(
        {
            f"objective_{index}": _digest(
                {"family": "MOKP", "profit_multiset": sorted(row)}
            )
            for index, row in enumerate(profits)
        }
    )
    if len(profits) > 6:
        raise ValueError("Permutation-normalized MOKP hashing supports at most six objectives.")
    canonical_item_tables = []
    for objective_order in itertools.permutations(range(len(profits))):
        canonical_item_tables.append(
            tuple(
                sorted(
                    (
                        weights[item],
                        *(profits[objective][item] for objective in objective_order),
                    )
                    for item in range(len(weights))
                )
            )
        )
    semantic = {
        "family": "MOKP",
        "capacity": capacity,
        "canonical_items": min(canonical_item_tables),
    }
    return InstanceFingerprint(
        family=family,
        problem_sha256=_digest(ordered_problem),
        semantic_sha256=_digest(semantic),
        component_sha256=components,
    )


def raw_child_sha256(instance: Mapping[str, Any]) -> dict[str, str]:
    """Hash exact name-free child objects inside one released JSON packet."""

    family = str(instance.get("family", "")).upper()
    if family == "MOKP":
        weights = tuple(_positive_ints(instance.get("item_weights"), "item_weights"))
        capacity = _positive_int(instance.get("capacity"), "capacity")
        profits = tuple(
            tuple(_nonnegative_ints(row, f"profits_by_objective[{index}]"))
            for index, row in enumerate(instance.get("profits_by_objective", ()))
        )
        result = {
            "constraint": _digest(
                {"item_weights": weights, "capacity": capacity}
            )
        }
        result.update(
            {
                f"objective_{index}": _digest({"profits": row})
                for index, row in enumerate(profits)
            }
        )
        return result
    if family == "MOTSP":
        objectives = instance.get("coordinates_by_objective")
        if not isinstance(objectives, (list, tuple)) or not objectives:
            raise ValueError("MOTSP coordinates_by_objective must be nonempty.")
        return {
            f"objective_{index}": _digest({"coordinates": rows})
            for index, rows in enumerate(objectives)
        }
    raise ValueError(f"Unsupported instance family: {family or '<missing>'}.")


def _fingerprint_motsp(instance: Mapping[str, Any]) -> InstanceFingerprint:
    raw_objectives = instance.get("coordinates_by_objective")
    if not isinstance(raw_objectives, (list, tuple)) or not raw_objectives:
        raise ValueError("MOTSP coordinates_by_objective must be nonempty.")
    coordinates = tuple(
        tuple(_point(point, f"coordinates_by_objective[{objective}]") for point in rows)
        for objective, rows in enumerate(raw_objectives)
    )
    n = len(coordinates[0])
    if n < 3 or any(len(rows) != n for rows in coordinates):
        raise ValueError("MOTSP objectives must share at least three cities.")
    if instance.get("num_cities", n) != n:
        raise ValueError("MOTSP declared city count does not match coordinates.")
    if instance.get("num_objectives", len(coordinates)) != len(coordinates):
        raise ValueError("MOTSP declared objective count does not match coordinates.")
    matrices = tuple(_euclidean_matrix(rows) for rows in coordinates)
    components = {
        f"objective_{index}": _digest(
            {
                "family": "MOTSP",
                "num_cities": n,
                "undirected_edge_weight_multiset": _undirected_edge_multiset(matrix),
            }
        )
        for index, matrix in enumerate(matrices)
    }
    ordered_problem = {
        "family": "MOTSP",
        "num_cities": n,
        "distance_matrices": matrices,
    }
    semantic = {
        "family": "MOTSP",
        "num_cities": n,
        "objectives": sorted(components.values()),
    }
    return InstanceFingerprint(
        family="MOTSP",
        problem_sha256=_digest(ordered_problem),
        semantic_sha256=_digest(semantic),
        component_sha256=components,
    )


def _point(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two-coordinate points.")
    x, y = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (x, y)):
        raise ValueError(f"{label} coordinates must be integers.")
    return x, y


def _euclidean_matrix(points: tuple[tuple[int, int], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(math.hypot(ax - bx, ay - by) for bx, by in points)
        for ax, ay in points
    )


def _undirected_edge_multiset(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    return tuple(
        sorted(
            float(matrix[left][right])
            for left in range(len(matrix))
            for right in range(left + 1, len(matrix))
        )
    )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _positive_ints(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty integer sequence.")
    return tuple(_positive_int(item, label) for item in value)


def _nonnegative_ints(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty integer sequence.")
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain nonnegative integers.")
        parsed.append(item)
    return tuple(parsed)


def materialize_v21_partitions(
    output_root: str | Path,
    *,
    master_seed: bytes,
    sizes: Sequence[int] = (100, 200, 500),
    development_cases_per_size: int = 2,
    calibration_cases_per_size: int = 5,
    calibration_epoch: str = "v21v2",
) -> dict[str, Path]:
    """Exclusively create prospective development and calibration packets.

    No formal-study path or bytes are created by this API.  Formal material
    requires a separate, post-gate entropy protocol outside this module.
    """

    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Prospective output already exists: {root}")
    if not isinstance(master_seed, bytes) or not master_seed:
        raise ValueError("master_seed must be nonempty bytes.")
    if (
        not isinstance(calibration_epoch, str)
        or not calibration_epoch
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in calibration_epoch
        )
    ):
        raise ValueError(
            "calibration_epoch must contain only lowercase ASCII letters, digits, or '-'."
        )
    parsed_sizes = tuple(_positive_int(size, "size") for size in sizes)
    if not parsed_sizes or len(set(parsed_sizes)) != len(parsed_sizes):
        raise ValueError("sizes must be a nonempty sequence of unique positive integers.")
    if any(size < 3 for size in parsed_sizes):
        raise ValueError("All shared MOTSP/MOKP sizes must be at least three.")
    development_count = _positive_int(
        development_cases_per_size, "development_cases_per_size"
    )
    calibration_count = _positive_int(
        calibration_cases_per_size, "calibration_cases_per_size"
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        layout = {
            "development": staging / "development",
            "selection": staging / "calibration" / "selection",
            "confirmation": staging / "calibration" / "confirmation",
        }
        counts = {
            "development": development_count,
            "selection": calibration_count,
            "confirmation": calibration_count,
        }
        seed_commitment = hashlib.sha256(master_seed).hexdigest()
        for split, partition_root in layout.items():
            _materialize_partition(
                partition_root=partition_root,
                split=split,
                master_seed=master_seed,
                seed_commitment=seed_commitment,
                sizes=parsed_sizes,
                cases_per_size=counts[split],
                calibration_epoch=calibration_epoch,
            )
        os.rename(staging, root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "development": root / "development" / "case_manifest.json",
        "selection": root / "calibration" / "selection" / "case_manifest.json",
        "confirmation": root / "calibration" / "confirmation" / "case_manifest.json",
    }


def _materialize_partition(
    *,
    partition_root: Path,
    split: str,
    master_seed: bytes,
    seed_commitment: str,
    sizes: tuple[int, ...],
    cases_per_size: int,
    calibration_epoch: str,
) -> None:
    legacy_v2 = calibration_epoch == "v21v2"
    instance_generator_version = (
        "pareto-v21-instance-generator-v2"
        if legacy_v2
        else "pareto-v21-instance-generator-v3"
    )
    cases: list[dict[str, object]] = []
    for family in ("MOTSP", "MOKP"):
        for size in sizes:
            for case_index in range(cases_per_size):
                regime = _V21_REGIMES[case_index % len(_V21_REGIMES)]
                stream_id = (
                    f"pareto-v21/{split}/{family.lower()}/n{size}/case{case_index:02d}"
                    if legacy_v2
                    else f"pareto-v21/{calibration_epoch}/{split}/"
                    f"{family.lower()}/n{size}/case{case_index:02d}"
                )
                generator = _generator_binding(
                    family=family,
                    stream_id=stream_id,
                    seed_commitment=seed_commitment,
                    regime=regime,
                    calibration_epoch=calibration_epoch,
                )
                rng = Shake256CounterRNG(seed=master_seed, domain=stream_id)
                case_id = (
                    f"{calibration_epoch}-{family.lower()}-{split}-"
                    f"n{size}-s{case_index:02d}"
                )
                if family == "MOTSP":
                    instance = _generate_motsp_instance(
                        case_id=case_id,
                        size=size,
                        rng=rng,
                        generator=generator,
                        regime=regime,
                    )
                else:
                    instance = _generate_mokp_instance(
                        case_id=case_id,
                        size=size,
                        rng=rng,
                        generator=generator,
                        regime=regime,
                    )
                fingerprint = fingerprint_instance(instance)
                artifact_path = partition_root / "instances" / f"{case_id}.json"
                _write_json_exclusive(artifact_path, instance)
                relative_path = artifact_path.relative_to(partition_root).as_posix()
                cases.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "split": split,
                        "size": size,
                        "num_objectives": 2,
                        "regime": regime,
                        "artifact": {
                            "path": relative_path,
                            "sha256": _file_sha256(artifact_path),
                            "bytes": artifact_path.stat().st_size,
                        },
                        "fingerprints": {
                            "problem_sha256": fingerprint.problem_sha256,
                            "semantic_sha256": fingerprint.semantic_sha256,
                            "component_sha256": dict(fingerprint.component_sha256),
                            "raw_child_sha256": raw_child_sha256(instance),
                        },
                        "generator": generator,
                    }
                )
    generator_contract = {
        "algorithm": "SHAKE256_counter_rejection_sampling",
        "version": "pareto-v21-shake256-counter-v1",
        "instance_generator_version": instance_generator_version,
        "counter_endianness": "big",
        "integer_sampling": "uint64_rejection_then_modulo",
        "master_seed_encoding": "hex",
        "master_seed_hex": master_seed.hex(),
        "master_seed_sha256": seed_commitment,
        "regime_schedule": list(_V21_REGIMES),
    }
    if not legacy_v2:
        generator_contract["calibration_epoch"] = calibration_epoch
    manifest = {
        "schema": "pareto_v21_partition_manifest_v1",
        "suite_id": (
            f"pareto-v21-{split}-authors-generated-"
            f"{'v2' if legacy_v2 else 'v3'}"
        ),
        "split": split,
        "role": "prospective_algorithm_development"
        if split == "development"
        else f"prospective_calibration_{split}",
        "authors_generated": True,
        "external_independence_status": "NOT_ESTABLISHED",
        "formal_confirmatory_eligibility": False,
        "generator_contract": generator_contract,
        "cases": sorted(cases, key=lambda case: str(case["case_id"])),
    }
    if not legacy_v2:
        manifest["calibration_epoch"] = calibration_epoch
    _write_json_exclusive(partition_root / "case_manifest.json", manifest)


def _generator_binding(
    *,
    family: str,
    stream_id: str,
    seed_commitment: str,
    regime: str,
    calibration_epoch: str,
) -> dict[str, object]:
    instance_contract: dict[str, object]
    if family == "MOTSP":
        instance_contract = {
            "schema": "pareto_v21_motsp_integer_coordinates_v1",
            "coordinate_support": [0, 999_999],
            "objectives": 2,
        }
    else:
        instance_contract = {
            "schema": "pareto_v21_mokp_integer_instance_v1",
            "weight_support": [1, 100],
            "profit_support": [1, 100],
            "capacity_fraction_numerator": 2,
            "capacity_fraction_denominator": 5,
            "objectives": 2,
        }
    legacy_v2 = calibration_epoch == "v21v2"
    lineage_payload = {
        "algorithm": "SHAKE256_counter_rejection_sampling",
        "version": (
            "pareto-v21-instance-generator-v2"
            if legacy_v2
            else "pareto-v21-instance-generator-v3"
        ),
        "family": family,
        "instance_contract": {**instance_contract, "regimes": list(_V21_REGIMES)},
    }
    if not legacy_v2:
        lineage_payload["calibration_epoch"] = calibration_epoch
    lineage_sha256 = _digest(lineage_payload)
    invocation_payload = {
        "lineage_sha256": lineage_sha256,
        "master_seed_sha256": seed_commitment,
        "stream_id": stream_id,
        "regime": regime,
    }
    if not legacy_v2:
        invocation_payload["calibration_epoch"] = calibration_epoch
    result = {
        **lineage_payload,
        "stream_id": stream_id,
        "regime": regime,
        "master_seed_commitment": seed_commitment,
        "lineage_sha256": lineage_sha256,
        "invocation_sha256": _digest(invocation_payload),
    }
    return result


def _generate_motsp_instance(
    *,
    case_id: str,
    size: int,
    rng: Shake256CounterRNG,
    generator: Mapping[str, object],
    regime: str,
) -> dict[str, object]:
    first = _unique_integer_coordinates(size=size, rng=rng)
    if regime == "independent":
        second = _unique_integer_coordinates(size=size, rng=rng)
    elif regime == "objective_correlated":
        second = _jitter_coordinates(first, rng=rng, radius=50_000)
    elif regime == "objective_conflicting":
        second = _jitter_coordinates(
            _permuted_points(first, rng=rng), rng=rng, radius=25_000
        )
    elif regime == "structured":
        first = _clustered_coordinates(size=size, rng=rng)
        second = _clustered_coordinates(size=size, rng=rng)
    elif regime == "heterogeneous":
        second = _unique_integer_coordinates(size=size, rng=rng, support=100_000)
    else:
        raise ValueError(f"Unsupported MOTSP regime: {regime}")
    objectives = [first, second]
    return {
        "schema": "pareto_v21_motsp_integer_coordinates_v1",
        "family": "MOTSP",
        "case_id": case_id,
        "num_cities": size,
        "num_objectives": 2,
        "distance_contract": "binary64_euclidean_hypot_v1",
        "regime": regime,
        "coordinates_by_objective": objectives,
        "generator": dict(generator),
    }


def _unique_integer_coordinates(
    *, size: int, rng: Shake256CounterRNG, support: int = 1_000_000
) -> list[list[int]]:
    points: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    while len(points) < size:
        point = (rng.randbelow(support), rng.randbelow(support))
        if point in seen:
            continue
        seen.add(point)
        points.append([point[0], point[1]])
    return points


def _jitter_coordinates(
    points: Sequence[Sequence[int]],
    *,
    rng: Shake256CounterRNG,
    radius: int,
) -> list[list[int]]:
    result: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        while True:
            candidate = (
                min(999_999, max(0, int(x) + rng.randint(-radius, radius))),
                min(999_999, max(0, int(y) + rng.randint(-radius, radius))),
            )
            if candidate not in seen:
                seen.add(candidate)
                result.append([candidate[0], candidate[1]])
                break
    return result


def _permuted_points(
    points: Sequence[Sequence[int]], *, rng: Shake256CounterRNG
) -> list[list[int]]:
    result = [list(point) for point in points]
    for index in range(len(result) - 1, 0, -1):
        other = rng.randbelow(index + 1)
        result[index], result[other] = result[other], result[index]
    return result


def _clustered_coordinates(
    *, size: int, rng: Shake256CounterRNG
) -> list[list[int]]:
    centers = ((200_000, 200_000), (800_000, 200_000), (200_000, 800_000), (800_000, 800_000))
    result: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(size):
        center_x, center_y = centers[index % len(centers)]
        while True:
            point = (
                min(999_999, max(0, center_x + rng.randint(-100_000, 100_000))),
                min(999_999, max(0, center_y + rng.randint(-100_000, 100_000))),
            )
            if point not in seen:
                seen.add(point)
                result.append([point[0], point[1]])
                break
    return result


def _generate_mokp_instance(
    *,
    case_id: str,
    size: int,
    rng: Shake256CounterRNG,
    generator: Mapping[str, object],
    regime: str,
) -> dict[str, object]:
    weights = [rng.randint(1, 100) for _ in range(size)]
    first = [rng.randint(1, 100) for _ in range(size)]
    if regime == "independent":
        second = [rng.randint(1, 100) for _ in range(size)]
    elif regime == "objective_correlated":
        second = [_bounded_profit(value + rng.randint(-10, 10)) for value in first]
    elif regime == "objective_conflicting":
        second = [_bounded_profit(101 - value + rng.randint(-10, 10)) for value in first]
    elif regime == "structured":
        first = [_bounded_profit(weight + rng.randint(-12, 12)) for weight in weights]
        second = [
            _bounded_profit(101 - weight + rng.randint(-12, 12)) for weight in weights
        ]
    elif regime == "heterogeneous":
        first = [rng.randint(1, 25) for _ in range(size)]
        second = [rng.randint(25, 100) for _ in range(size)]
    else:
        raise ValueError(f"Unsupported MOKP regime: {regime}")
    profits = [first, second]
    capacity = max(1, (2 * sum(weights)) // 5)
    return {
        "schema": "pareto_v21_mokp_integer_instance_v1",
        "family": "MOKP",
        "case_id": case_id,
        "num_items": size,
        "num_objectives": 2,
        "item_weights": weights,
        "profits_by_objective": profits,
        "capacity": capacity,
        "regime": regime,
        "generator": dict(generator),
    }


def _bounded_profit(value: int) -> int:
    return min(100, max(1, value))


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(payload) + b"\n")


def load_partition_case(
    manifest_path: str | Path, case_id: str
) -> dict[str, object]:
    """Load one bound case after verifying bytes and name-free fingerprints."""

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pareto_v21_partition_manifest_v1":
        raise ValueError("Unsupported V21 partition manifest schema.")
    matches = [case for case in manifest.get("cases", ()) if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest case named {case_id!r}.")
    case = matches[0]
    relative_path = Path(str(case["artifact"]["path"]))
    artifact = (manifest_file.parent / relative_path).resolve()
    try:
        artifact.relative_to(manifest_file.parent)
    except ValueError as exc:
        raise ValueError("Case artifact escapes its manifest directory.") from exc
    if _file_sha256(artifact) != case["artifact"]["sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch for case {case_id}.")
    if artifact.stat().st_size != int(case["artifact"].get("bytes", -1)):
        raise ValueError(f"Artifact byte-count mismatch for case {case_id}.")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("case_id") != case_id or payload.get("family") != case.get("family"):
        raise ValueError(f"Case identity binding mismatch for case {case_id}.")
    if "regime" in case and payload.get("regime") != case.get("regime"):
        raise ValueError(f"Case regime binding mismatch for case {case_id}.")
    payload_generator = payload.get("generator")
    case_generator = case.get("generator")
    if payload_generator != case_generator or not isinstance(case_generator, Mapping):
        raise ValueError(f"Case generator binding mismatch for case {case_id}.")
    _validate_generator_binding(
        case_generator,
        family=str(case.get("family", "")),
        manifest_contract=manifest.get("generator_contract"),
    )
    fingerprint = fingerprint_instance(payload)
    expected = case["fingerprints"]
    observed = {
        "problem_sha256": fingerprint.problem_sha256,
        "semantic_sha256": fingerprint.semantic_sha256,
        "component_sha256": dict(fingerprint.component_sha256),
    }
    if "raw_child_sha256" in expected:
        observed["raw_child_sha256"] = raw_child_sha256(payload)
    if observed != expected:
        raise ValueError(f"Name-free fingerprint mismatch for case {case_id}.")
    return payload


def _validate_generator_binding(
    generator: Mapping[str, object],
    *,
    family: str,
    manifest_contract: object,
) -> None:
    if not isinstance(manifest_contract, Mapping):
        raise ValueError("Case generator binding has no manifest contract.")
    try:
        released_seed = bytes.fromhex(str(manifest_contract["master_seed_hex"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Case generator binding has invalid released seed bytes.") from exc
    seed_commitment = hashlib.sha256(released_seed).hexdigest()
    if seed_commitment != manifest_contract.get("master_seed_sha256"):
        raise ValueError("Case generator binding seed commitment is invalid.")
    if generator.get("version") != manifest_contract.get(
        "instance_generator_version"
    ):
        raise ValueError("Case generator binding version is invalid.")
    calibration_epoch = generator.get("calibration_epoch")
    if calibration_epoch != manifest_contract.get("calibration_epoch"):
        raise ValueError("Case generator binding calibration epoch is invalid.")
    if calibration_epoch is not None and not str(generator.get("stream_id", "")).startswith(
        f"pareto-v21/{calibration_epoch}/"
    ):
        raise ValueError("Case generator binding stream namespace is invalid.")
    lineage_payload = {
        "algorithm": generator.get("algorithm"),
        "version": generator.get("version"),
        "family": family,
        "instance_contract": generator.get("instance_contract"),
    }
    if calibration_epoch is not None:
        lineage_payload["calibration_epoch"] = calibration_epoch
    lineage_sha256 = _digest(lineage_payload)
    if lineage_sha256 != generator.get("lineage_sha256"):
        raise ValueError("Case generator binding lineage digest is invalid.")
    if generator.get("master_seed_commitment") != seed_commitment:
        raise ValueError("Case generator binding master-seed commitment is invalid.")
    invocation_payload = {
        "lineage_sha256": lineage_sha256,
        "master_seed_sha256": seed_commitment,
        "stream_id": generator.get("stream_id"),
    }
    if "regime" in generator:
        invocation_payload["regime"] = generator.get("regime")
    if calibration_epoch is not None:
        invocation_payload["calibration_epoch"] = calibration_epoch
    if _digest(invocation_payload) != generator.get("invocation_sha256"):
        raise ValueError("Case generator binding invocation digest is invalid.")


def build_prior_exposure_registry(
    v20_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Ingest the V20 calibration and formal manifests into a deny registry."""

    root = Path(v20_root).resolve()
    manifest_bindings = (
        ("v20_calibration", root / "calibration" / "case_manifest.json"),
        ("v20_formal", root / "formal_study" / "case_manifest.json"),
    )
    entries: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for role, manifest_path in manifest_bindings:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required V20 manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"V20 manifest contains no cases: {manifest_path}")
        sources.append(
            {
                "role": role,
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": _file_sha256(manifest_path),
                "case_count": len(raw_cases),
            }
        )
        for raw_case in raw_cases:
            entries.append(
                _ingest_prior_case(
                    raw_case,
                    manifest_path=manifest_path,
                    source_role=role,
                )
            )
    indexes = _exposure_indexes(entries)
    registry: dict[str, object] = {
        "schema": "pareto_v21_prior_exposure_registry_v1",
        "source_release": "V20",
        "scope": "all_cases_in_v20_calibration_and_formal_manifests",
        "case_count": len(entries),
        "sources": sources,
        "entries": sorted(
            entries,
            key=lambda entry: (str(entry["source_role"]), str(entry["case_id"])),
        ),
        "indexes": {key: sorted(values) for key, values in indexes.items()},
    }
    if output_path is not None:
        _write_json_exclusive(Path(output_path).resolve(), registry)
    return registry


def extend_prior_exposure_registry(
    prior_registry: Mapping[str, Any] | str | Path,
    manifest_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Add superseded V21 packets to the immutable exposure deny set."""

    base = _load_prior_registry(prior_registry)
    assert base is not None
    entries = [dict(entry) for entry in base.get("entries", ())]
    sources = [dict(source) for source in base.get("sources", ())]
    for value in manifest_paths:
        manifest_path = Path(value).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "pareto_v21_partition_manifest_v1":
            raise ValueError(f"Unsupported superseded V21 manifest: {manifest_path}")
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"Superseded V21 manifest has no cases: {manifest_path}")
        split = str(manifest.get("split", "unknown"))
        sources.append(
            {
                "role": f"superseded_v21_{split}",
                "path": _portable_path(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "case_count": len(raw_cases),
            }
        )
        for case in raw_cases:
            entries.append(
                _ingest_superseded_v21_case(
                    case,
                    manifest_path=manifest_path,
                    source_role=f"superseded_v21_{split}",
                )
            )
    registry: dict[str, object] = {
        "schema": "pareto_v21_prior_exposure_registry_v1",
        "source_release": f"{base.get('source_release', 'prior')}+superseded_V21",
        "scope": "prior_registry_plus_all_superseded_v21_packets",
        "case_count": len(entries),
        "sources": sources,
        "entries": sorted(
            entries,
            key=lambda entry: (str(entry["source_role"]), str(entry["case_id"])),
        ),
        "indexes": _exposure_indexes(entries),
    }
    if output_path is not None:
        _write_json_exclusive(Path(output_path).resolve(), registry)
    return registry


def _ingest_superseded_v21_case(
    case: Mapping[str, Any],
    *,
    manifest_path: Path,
    source_role: str,
) -> dict[str, object]:
    case_id = str(case.get("case_id", ""))
    family = str(case.get("family", "")).upper()
    artifact_binding = case.get("artifact")
    if not case_id or family not in {"MOTSP", "MOKP"} or not isinstance(
        artifact_binding, Mapping
    ):
        raise ValueError(f"Malformed superseded V21 case {case_id!r}.")
    artifact = (manifest_path.parent / str(artifact_binding["path"])).resolve()
    try:
        artifact.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"Superseded V21 case {case_id} escapes its manifest root.") from exc
    raw_hash = _file_sha256(artifact)
    if raw_hash != artifact_binding.get("sha256"):
        raise ValueError(f"Superseded V21 artifact SHA-256 mismatch for {case_id}.")
    if artifact.stat().st_size != int(artifact_binding.get("bytes", -1)):
        raise ValueError(f"Superseded V21 artifact byte-count mismatch for {case_id}.")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("case_id") != case_id or str(payload.get("family", "")).upper() != family:
        raise ValueError(f"Superseded V21 identity mismatch for {case_id}.")
    fingerprint = fingerprint_instance(payload)
    generator = payload.get("generator", {})
    if not isinstance(generator, Mapping):
        generator = {}
    return {
        "case_id": case_id,
        "family": family,
        "source_role": source_role,
        "source_manifest": manifest_path.name,
        "raw_artifact_sha256": [raw_hash],
        "raw_child_sha256": raw_child_sha256(payload),
        "problem_sha256": str(
            case.get("fingerprints", {}).get(
                "problem_sha256", fingerprint.problem_sha256
            )
        ),
        "name_free_problem_sha256": fingerprint.problem_sha256,
        "semantic_sha256": fingerprint.semantic_sha256,
        "component_sha256": dict(fingerprint.component_sha256),
        "generator_lineage_sha256": str(
            generator.get("lineage_sha256", _digest({"family": family, "generator": generator}))
        ),
        "generator_invocation_sha256": str(
            generator.get(
                "invocation_sha256",
                _digest({"family": family, "generator": generator, "raw_sha256": raw_hash}),
            )
        ),
    }


def _exposure_indexes(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    indexes: dict[str, set[str]] = {
        "case_id": set(),
        "case_id_sha256": set(),
        "raw_artifact_sha256": set(),
        "raw_child_sha256": set(),
        "problem_sha256": set(),
        "name_free_problem_sha256": set(),
        "semantic_sha256": set(),
        "component_sha256": set(),
        "generator_lineage_sha256": set(),
        "generator_invocation_sha256": set(),
    }
    for entry in entries:
        case_id = str(entry["case_id"])
        indexes["case_id"].add(case_id)
        indexes["case_id_sha256"].add(
            hashlib.sha256(case_id.encode("utf-8")).hexdigest()
        )
        indexes["raw_artifact_sha256"].update(entry.get("raw_artifact_sha256", ()))
        indexes["raw_child_sha256"].update(
            entry.get("raw_child_sha256", {}).values()
        )
        indexes["problem_sha256"].add(str(entry["problem_sha256"]))
        indexes["name_free_problem_sha256"].add(
            str(entry["name_free_problem_sha256"])
        )
        indexes["semantic_sha256"].add(str(entry["semantic_sha256"]))
        indexes["component_sha256"].update(entry["component_sha256"].values())
        indexes["generator_lineage_sha256"].add(
            str(entry["generator_lineage_sha256"])
        )
        indexes["generator_invocation_sha256"].add(
            str(entry["generator_invocation_sha256"])
        )
    return {key: sorted(values) for key, values in indexes.items()}


def _ingest_prior_case(
    raw_case: Mapping[str, Any],
    *,
    manifest_path: Path,
    source_role: str,
) -> dict[str, object]:
    case_id = str(raw_case.get("case_id", ""))
    family = str(raw_case.get("family", "")).upper()
    if not case_id or family not in {"MOTSP", "MOKP"}:
        raise ValueError(f"Malformed V20 case in {manifest_path}: {case_id!r}/{family!r}")
    artifact_bindings = raw_case.get("artifacts")
    if not isinstance(artifact_bindings, list) or not artifact_bindings:
        raise ValueError(f"V20 case {case_id} has no artifact bindings.")
    artifact_paths: list[Path] = []
    raw_hashes: list[str] = []
    for binding in artifact_bindings:
        path = (manifest_path.parent / str(binding["path"])).resolve()
        try:
            path.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"V20 case {case_id} artifact escapes the manifest root.") from exc
        observed_hash = _file_sha256(path)
        if observed_hash != binding.get("sha256"):
            raise ValueError(f"V20 artifact SHA-256 mismatch for {case_id}: {path}")
        if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
            raise ValueError(f"V20 artifact byte-count mismatch for {case_id}: {path}")
        artifact_paths.append(path)
        raw_hashes.append(observed_hash)
    artifact_generator: Mapping[str, object] = {}
    prior_raw_children: dict[str, str]
    if family == "MOKP":
        if len(artifact_paths) != 1:
            raise ValueError(f"V20 MOKP case {case_id} must bind one JSON artifact.")
        payload = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
        fingerprint = fingerprint_instance(payload)
        prior_raw_children = raw_child_sha256(payload)
        generator_value = payload.get("generator", {})
        if isinstance(generator_value, Mapping):
            artifact_generator = generator_value
    else:
        fingerprint = _fingerprint_tsplib_objectives(artifact_paths)
        prior_raw_children = {
            f"objective_{index}": digest
            for index, digest in enumerate(raw_hashes)
        }
    provenance = raw_case.get("source_provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    lineage_payload = {
        "family": family,
        "suite": provenance.get("suite"),
        "generator_name": artifact_generator.get("name"),
        "instance_schema": (
            payload.get("schema") if family == "MOKP" else "TSPLIB_objective_pair"
        ),
    }
    lineage_sha256 = _digest(lineage_payload)
    invocation_payload = {
        "lineage_sha256": lineage_sha256,
        "generator_seed": provenance.get(
            "generator_seed", artifact_generator.get("seed")
        ),
        "raw_artifact_sha256": sorted(raw_hashes),
    }
    problem_hash = str(raw_case.get("problem_sha256", ""))
    if len(problem_hash) != 64:
        raise ValueError(f"V20 case {case_id} has no valid problem SHA-256 binding.")
    return {
        "case_id": case_id,
        "family": family,
        "source_role": source_role,
        "source_manifest": manifest_path.name,
        "raw_artifact_sha256": raw_hashes,
        "raw_child_sha256": prior_raw_children,
        "problem_sha256": problem_hash,
        "name_free_problem_sha256": fingerprint.problem_sha256,
        "semantic_sha256": fingerprint.semantic_sha256,
        "component_sha256": dict(fingerprint.component_sha256),
        "generator_lineage_sha256": lineage_sha256,
        "generator_invocation_sha256": _digest(invocation_payload),
    }


def _fingerprint_tsplib_objectives(paths: Sequence[Path]) -> InstanceFingerprint:
    from .tsplib import parse_tsplib

    parsed = [parse_tsplib(path) for path in paths]
    dimension = parsed[0].dimension
    if dimension < 3 or any(problem.dimension != dimension for problem in parsed):
        raise ValueError("Prior MOTSP objective dimensions do not match.")
    matrices = tuple(
        tuple(tuple(float(value) for value in row) for row in problem.distance_matrix)
        for problem in parsed
    )
    components = {
        f"objective_{index}": _digest(
            {
                "family": "MOTSP",
                "num_cities": dimension,
                "undirected_edge_weight_multiset": _undirected_edge_multiset(matrix),
            }
        )
        for index, matrix in enumerate(matrices)
    }
    ordered_problem = {
        "family": "MOTSP",
        "num_cities": dimension,
        "distance_matrices": matrices,
    }
    semantic = {
        "family": "MOTSP",
        "num_cities": dimension,
        "objectives": sorted(components.values()),
    }
    return InstanceFingerprint(
        family="MOTSP",
        problem_sha256=_digest(ordered_problem),
        semantic_sha256=_digest(semantic),
        component_sha256=components,
    )


def audit_partition_overlap(
    manifest_paths: Sequence[str | Path],
    *,
    prior_registry: Mapping[str, Any] | str | Path | None = None,
    output_path: str | Path | None = None,
    require_pass: bool = False,
) -> dict[str, object]:
    """Fail closed on byte, problem, semantic, component, or lineage overlap."""

    if not manifest_paths:
        raise ValueError("At least one V21 partition manifest is required.")
    collisions: list[dict[str, object]] = []
    case_records: list[dict[str, object]] = []
    manifest_receipts: list[dict[str, object]] = []
    observed_splits: set[str] = set()
    for value in manifest_paths:
        manifest_path = Path(value).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "pareto_v21_partition_manifest_v1":
            raise ValueError(f"Unsupported partition manifest: {manifest_path}")
        split = str(manifest.get("split", ""))
        if split not in {"development", "selection", "confirmation"}:
            raise ValueError(
                "Overlap audit accepts only development and calibration manifests; "
                f"received split {split!r}."
            )
        if split in observed_splits:
            collisions.append(
                {"kind": "duplicate_partition_split", "value": split}
            )
        observed_splits.add(split)
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"Partition manifest has no cases: {manifest_path}")
        manifest_receipts.append(
            {
                "path": _portable_path(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "split": split,
                "case_count": len(raw_cases),
            }
        )
        for case in raw_cases:
            case_id = str(case.get("case_id", ""))
            if case.get("split") != split:
                collisions.append(
                    {
                        "kind": "case_split_binding_mismatch",
                        "case_id": case_id,
                        "manifest_split": split,
                        "case_split": case.get("split"),
                    }
                )
            load_partition_case(manifest_path, case_id)
            generator = case.get("generator", {})
            if not isinstance(generator, Mapping):
                raise ValueError(f"Case {case_id} has no generator binding.")
            component_map = case["fingerprints"]["component_sha256"]
            if not isinstance(component_map, Mapping) or not component_map:
                raise ValueError(f"Case {case_id} has no component fingerprints.")
            component_values = [str(item) for item in component_map.values()]
            raw_child_map = case["fingerprints"].get("raw_child_sha256", {})
            if raw_child_map and not isinstance(raw_child_map, Mapping):
                raise ValueError(f"Case {case_id} has invalid raw-child fingerprints.")
            raw_child_values = [str(item) for item in raw_child_map.values()]
            if len(component_values) != len(set(component_values)):
                collisions.append(
                    {
                        "kind": "within_case_component_sha256",
                        "case_id": case_id,
                    }
                )
            case_records.append(
                {
                    "case_id": case_id,
                    "case_id_sha256": hashlib.sha256(
                        case_id.encode("utf-8")
                    ).hexdigest(),
                    "split": split,
                    "family": str(case.get("family", "")),
                    "size": int(case.get("size", 0)),
                    "regime": str(case.get("regime", "UNSPECIFIED")),
                    "raw_artifact_sha256": str(case["artifact"]["sha256"]),
                    "raw_child_sha256": raw_child_values,
                    "problem_sha256": str(
                        case["fingerprints"]["problem_sha256"]
                    ),
                    "semantic_sha256": str(
                        case["fingerprints"]["semantic_sha256"]
                    ),
                    "component_sha256": component_values,
                    "generator_lineage_sha256": str(
                        generator.get("lineage_sha256", "")
                    ),
                    "generator_invocation_sha256": str(
                        generator.get("invocation_sha256", "")
                    ),
                }
            )
    for field in (
        "case_id",
        "case_id_sha256",
        "raw_artifact_sha256",
        "problem_sha256",
        "semantic_sha256",
        "generator_invocation_sha256",
    ):
        _record_cross_case_duplicates(
            case_records,
            field=field,
            kind=f"cross_case_{field}",
            collisions=collisions,
        )
    raw_child_owners: dict[str, list[dict[str, str]]] = {}
    for record in case_records:
        for child in record["raw_child_sha256"]:
            raw_child_owners.setdefault(child, []).append(
                {"case_id": str(record["case_id"]), "split": str(record["split"])}
            )
    for value, owners in sorted(raw_child_owners.items()):
        if len({owner["case_id"] for owner in owners}) > 1:
            collisions.append(
                {"kind": "cross_case_raw_child_sha256", "value": value, "owners": owners}
            )
    component_owners: dict[str, list[dict[str, str]]] = {}
    for record in case_records:
        for component in record["component_sha256"]:
            component_owners.setdefault(component, []).append(
                {
                    "case_id": str(record["case_id"]),
                    "split": str(record["split"]),
                }
            )
    for value, owners in sorted(component_owners.items()):
        if len({owner["case_id"] for owner in owners}) > 1:
            collisions.append(
                {
                    "kind": "cross_case_component_sha256",
                    "value": value,
                    "owners": owners,
                }
            )
    registry = _load_prior_registry(prior_registry)
    if registry is not None:
        prior_indexes = registry["indexes"]
        prior_bindings = (
            ("case_id", "case_id", "prior_case_id"),
            ("case_id_sha256", "case_id_sha256", "prior_case_id_sha256"),
            (
                "raw_artifact_sha256",
                "raw_artifact_sha256",
                "prior_raw_artifact_sha256",
            ),
            (
                "problem_sha256",
                "problem_sha256",
                "prior_problem_sha256",
            ),
            (
                "problem_sha256",
                "name_free_problem_sha256",
                "prior_name_free_problem_sha256",
            ),
            ("semantic_sha256", "semantic_sha256", "prior_semantic_sha256"),
            (
                "generator_lineage_sha256",
                "generator_lineage_sha256",
                "prior_generator_lineage_sha256",
            ),
            (
                "generator_invocation_sha256",
                "generator_invocation_sha256",
                "prior_generator_invocation_sha256",
            ),
        )
        for record_field, prior_field, kind in prior_bindings:
            prior_values = set(prior_indexes.get(prior_field, ()))
            for record in case_records:
                if record[record_field] in prior_values:
                    collisions.append(
                        {
                            "kind": kind,
                            "case_id": record["case_id"],
                            "value": record[record_field],
                        }
                    )
        prior_components = set(prior_indexes.get("component_sha256", ()))
        prior_raw_children = set(prior_indexes.get("raw_child_sha256", ()))
        for record in case_records:
            for child in record["raw_child_sha256"]:
                if child in prior_raw_children:
                    collisions.append(
                        {
                            "kind": "prior_raw_child_sha256",
                            "case_id": record["case_id"],
                            "value": child,
                        }
                    )
            for component in record["component_sha256"]:
                if component in prior_components:
                    collisions.append(
                        {
                            "kind": "prior_component_sha256",
                            "case_id": record["case_id"],
                            "value": component,
                        }
                    )
    calibration_regime_counts: dict[str, dict[str, int]] = {}
    calibration_balance = True
    for split in ("selection", "confirmation"):
        groups = {
            (str(record["family"]), int(record["size"]))
            for record in case_records
            if record["split"] == split
        }
        for family, size in sorted(groups):
            counts = {
                regime: sum(
                    record["split"] == split
                    and record["family"] == family
                    and record["size"] == size
                    and record["regime"] == regime
                    for record in case_records
                )
                for regime in _V21_REGIMES
            }
            calibration_regime_counts[f"{split}/{family}/n{size}"] = counts
            if set(counts.values()) != {1}:
                calibration_balance = False
    receipt: dict[str, object] = {
        "schema": "pareto_v21_partition_overlap_audit_v1",
        "status": "PASS" if not collisions else "FAIL",
        "formal_instances_materialized_by_this_tool": False,
        "external_independence_status": "NOT_ESTABLISHED",
        "checked_dimensions": [
            "case_id",
            "case_id_sha256",
            "raw_artifact_sha256",
            "raw_child_sha256",
            "problem_sha256",
            "name_free_problem_sha256",
            "semantic_sha256",
            "component_sha256",
            "generator_lineage_sha256",
            "generator_invocation_sha256",
        ],
        "cross_case_shared_components_forbidden": True,
        "manifests": manifest_receipts,
        "case_count": len(case_records),
        "case_counts_by_split": {
            split: sum(record["split"] == split for record in case_records)
            for split in sorted(observed_splits)
        },
        "case_counts_by_family": {
            family: sum(record["family"] == family for record in case_records)
            for family in sorted({str(record["family"]) for record in case_records})
        },
        "calibration_regime_balance_status": (
            "PASS" if calibration_balance else "NOT_PROTOCOL_SCALE"
        ),
        "calibration_regime_counts": calibration_regime_counts,
        "prior_registry": _prior_registry_binding(prior_registry, registry),
        "collisions": collisions,
    }
    if output_path is not None:
        _write_json_exclusive(Path(output_path).resolve(), receipt)
    if require_pass and collisions:
        raise ValueError(
            f"V21 partition overlap audit failed with {len(collisions)} collision(s)."
        )
    return receipt


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _prior_registry_binding(
    supplied: Mapping[str, Any] | str | Path | None,
    loaded: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    if loaded is None:
        return None
    binding: dict[str, object] = {
        "canonical_sha256": _digest(loaded),
        "case_count": loaded.get("case_count"),
    }
    if supplied is not None and not isinstance(supplied, Mapping):
        path = Path(supplied).resolve()
        binding.update(
            {"path": _portable_path(path), "raw_file_sha256": _file_sha256(path)}
        )
    return binding


def _record_cross_case_duplicates(
    records: Sequence[Mapping[str, object]],
    *,
    field: str,
    kind: str,
    collisions: list[dict[str, object]],
) -> None:
    owners: dict[str, list[dict[str, str]]] = {}
    for record in records:
        value = str(record[field])
        owners.setdefault(value, []).append(
            {"case_id": str(record["case_id"]), "split": str(record["split"])}
        )
    for value, bound in sorted(owners.items()):
        if len({item["case_id"] for item in bound}) > 1:
            collisions.append({"kind": kind, "value": value, "owners": bound})


def _load_prior_registry(
    value: Mapping[str, Any] | str | Path | None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    registry: Mapping[str, Any]
    if isinstance(value, Mapping):
        registry = value
    else:
        registry = json.loads(Path(value).read_text(encoding="utf-8"))
    if registry.get("schema") != "pareto_v21_prior_exposure_registry_v1":
        raise ValueError("Unsupported prior-exposure registry schema.")
    if not isinstance(registry.get("indexes"), Mapping):
        raise ValueError("Prior-exposure registry is missing indexes.")
    return registry
