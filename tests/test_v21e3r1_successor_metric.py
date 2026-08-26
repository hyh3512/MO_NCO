from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess

import pytest

from ijoc_submission_v21e3r1.scripts import (
    run_v21e3r1_successor_development_factorial as factorial_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_METRIC = (
    PROJECT_ROOT / "independent_reproduction" / "recompute_v21e3r1_metrics.py"
)
SUCCESSOR_METRIC = (
    PROJECT_ROOT
    / "independent_reproduction"
    / "recompute_v21e3r1_successor_metrics.py"
)
NONHISTORICAL_PYTHON = Path(r"C:\miniconda3\envs\ssm_env\python.exe")
HISTORICAL_PYTHON = Path(r"C:\miniconda3\python.exe")
SEALED_N500_ATTEMPT = (
    PROJECT_ROOT
    / "outputs"
    / "v21e3r1_v7_exposed_development_diagnostics_20260823"
    / "attempts"
    / "v21e3-motsp-development-n500-s00__seed-31059__arm-moead_seeded"
    / "attempt-0001"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_prefix_hypervolume_matches_frozen_reference_ieee754_bits() -> None:
    frozen = _load(FROZEN_METRIC, "v21e3r1_frozen_metric_prefix_reference")
    successor = _load(SUCCESSOR_METRIC, "v21e3r1_successor_metric_prefix")
    sequences = (
        tuple((index / 32.0, (32 - index) / 32.0) for index in range(1, 32)),
        tuple((index / 32.0, index / 32.0) for index in range(1, 32)),
        (
            (0.0, 1.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (0.5, 0.0),
            (-0.0, 0.5),
            (0.0, 0.5),
        ),
    )
    for sequence in sequences:
        observed = successor.prefix_hypervolume_2d(sequence)
        expected_after = []
        prefix = []
        for point in sequence:
            prefix.append(point)
            expected_after.append(frozen.hypervolume_2d(prefix))
        expected_before = [0.0, *expected_after[:-1]]
        assert observed["front"] == frozen.nondominated(sequence)
        assert tuple(map(_bits, observed["hv_before"])) == tuple(
            map(_bits, expected_before)
        )
        assert tuple(map(_bits, observed["hv_after"])) == tuple(
            map(_bits, expected_after)
        )


def test_fixed_seed_random_prefixes_match_frozen_reference_ieee754_bits() -> None:
    frozen = _load(FROZEN_METRIC, "v21e3r1_frozen_metric_random_reference")
    successor = _load(SUCCESSOR_METRIC, "v21e3r1_successor_metric_random")
    for seed in range(64):
        generator = random.Random(0x21E3A1 + seed)
        pool: list[tuple[float, float]] = []
        sequence = []
        for index in range(64):
            if pool and index % 7 == 0:
                point = generator.choice(pool)
            else:
                point = (
                    generator.randrange(257) / 256.0,
                    generator.randrange(257) / 256.0,
                )
                pool.append(point)
            sequence.append(point)
        observed = successor.prefix_hypervolume_2d(sequence)
        prefix = []
        expected_after = []
        for point in sequence:
            prefix.append(point)
            expected_after.append(frozen.hypervolume_2d(prefix))
        expected_before = [0.0, *expected_after[:-1]]
        assert observed["front"] == frozen.nondominated(sequence)
        assert tuple(map(_bits, observed["hv_before"])) == tuple(
            map(_bits, expected_before)
        )
        assert tuple(map(_bits, observed["hv_after"])) == tuple(
            map(_bits, expected_after)
        )


def test_nondominated_chain_exposes_quadratic_operation_bound() -> None:
    successor = _load(SUCCESSOR_METRIC, "v21e3r1_successor_metric_operations")
    count = 256
    sequence = tuple(
        (index / (count + 1.0), (count + 1 - index) / (count + 1.0))
        for index in range(1, count + 1)
    )
    observed = successor.prefix_hypervolume_2d(sequence)
    operations = observed["operation_counts"]
    assert operations == {
        "point_count": count,
        "insertion_front_probe_count": count * (count - 1),
        "hypervolume_front_scan_count": count * (count + 1) // 2,
        "max_front_size": count,
        "final_front_size": count,
    }


def test_cli_fails_closed_under_nonhistorical_interpreter(tmp_path: Path) -> None:
    assert NONHISTORICAL_PYTHON.is_file()
    assert _sha256(NONHISTORICAL_PYTHON) != (
        "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
    )
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            str(NONHISTORICAL_PYTHON),
            str(SUCCESSOR_METRIC),
            "--trace",
            str(tmp_path / "absent.sqlite3"),
            "--lower=0.0,0.0",
            "--upper=1.0,1.0",
            "--expected-evaluations",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert "historical metric interpreter identity mismatch" in completed.stderr
    assert not output.exists()


def test_real_n500_trace_matches_sealed_golden_and_exposes_distinct_identity(
    tmp_path: Path,
) -> None:
    trace = SEALED_N500_ATTEMPT / "trace.sqlite3"
    frozen_receipt_path = SEALED_N500_ATTEMPT / "independent.metric.json"
    assert trace.is_file() and frozen_receipt_path.is_file()
    assert _sha256(FROZEN_METRIC) == (
        "587d4ed4d647d8293b36449c835109ee3afa6e9899fe155f917a492fdf303ea2"
    )
    output = tmp_path / "successor.metric.json"
    completed = subprocess.run(
        [
            str(HISTORICAL_PYTHON),
            str(SUCCESSOR_METRIC),
            "--trace",
            str(trace),
            "--lower=974266.5189772252,310072.9752816262",
            "--upper=666796683.8688538,676310877.4716699",
            "--expected-evaluations",
            "2000",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    successor = json.loads(output.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_receipt_path.read_text(encoding="utf-8"))
    assert successor["schema"] == (
        "v21e3r1_successor_independent_metric_reimplementation_v3"
    )
    assert successor["status"] == "PASS_SUCCESSOR_INDEPENDENT_METRIC_IMPLEMENTATION"
    assert successor["schema"] != frozen["schema"]
    assert successor["status"] != frozen["status"]
    assert successor["metric_semantics_id"] == (
        "normalized_left_continuous_hv_auc_binary64_v21e3r1_v2"
    )
    assert successor["metric_kernel_id"] == (
        "incremental_sorted_nondominated_front_order_preserving_v1"
    )
    assert successor["legacy_reference_metric_source_sha256"] == _sha256(
        FROZEN_METRIC
    )
    assert successor["successor_metric_source_sha256"] == _sha256(SUCCESSOR_METRIC)
    assert successor["successor_metric_source_sha256"] != successor[
        "legacy_reference_metric_source_sha256"
    ]
    assert successor["historical_metric_interpreter"]["sha256"] == (
        "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
    )
    assert successor["metric_kernel_operation_counts"] == {
        "point_count": 2000,
        "insertion_front_probe_count": 71467,
        "hypervolume_front_scan_count": 126589,
        "max_front_size": 72,
        "final_front_size": 72,
    }
    for field in ("exact_left_continuous_hv_auc", "terminal_hv"):
        assert _bits(successor[field]) == _bits(frozen[field])
    for field in (
        "trace_sha256",
        "run_context_digest_sha256",
        "evaluation_count",
        "decision_count",
        "attempt_count",
    ):
        assert successor[field] == frozen[field]
    assert successor["terminal_accounting_gate"] == "PASS"
    assert successor["implementation_independence_from_project_metrics"] is True
    assert successor["algorithm_execution_independence"] is False
    assert successor["scientific_independence"] is False


def test_factorial_runner_replays_metric_with_pinned_successor_identity(
    tmp_path: Path,
) -> None:
    trace = SEALED_N500_ATTEMPT / "trace.sqlite3"
    output = tmp_path / "factorial-successor.metric.json"
    receipt = factorial_runner._successor_independent_metric_replay(
        project_root=PROJECT_ROOT,
        trace=trace,
        lower=(974266.5189772252, 310072.9752816262),
        upper=(666796683.8688538, 676310877.4716699),
        budget=2000,
        output=output,
    )
    assert output.is_file()
    assert receipt["schema"] == (
        "v21e3r1_successor_independent_metric_reimplementation_v3"
    )
    assert receipt["status"] == "PASS_SUCCESSOR_INDEPENDENT_METRIC_IMPLEMENTATION"
    assert receipt["successor_metric_source_sha256"] == _sha256(SUCCESSOR_METRIC)
    assert receipt["historical_metric_interpreter"] == {
        "path": str(HISTORICAL_PYTHON.resolve()),
        "sha256": (
            "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
        ),
        "implementation": "cpython",
        "version": receipt["historical_metric_interpreter"]["version"],
    }


def test_factorial_runner_rejects_frozen_v2_metric_drift_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factorial_runner, "FROZEN_V2_METRIC_SHA256", "0" * 64)
    output = tmp_path / "must-not-be-created.json"
    with pytest.raises(
        factorial_runner.ContractError,
        match="historical frozen V2 metric source drifted",
    ):
        factorial_runner._successor_independent_metric_replay(
            project_root=PROJECT_ROOT,
            trace=SEALED_N500_ATTEMPT / "trace.sqlite3",
            lower=(974266.5189772252, 310072.9752816262),
            upper=(666796683.8688538, 676310877.4716699),
            budget=2000,
            output=output,
        )
    assert not output.exists()

